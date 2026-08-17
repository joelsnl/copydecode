from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from copydecode.checkpoint import Checkpoint, file_fingerprint
from copydecode.changelog import ChangeLog
from copydecode.chunking import format_numbered, pack_segments
from copydecode.detect import detect_mode, sample_text
from copydecode.engine import LLMEngine
from copydecode.epub_io import Chapter, load_epub, write_epub
from copydecode.glossary import Glossary, glossary_from_data, load_glossary_file
from copydecode.hardware import DeviceProfile
from copydecode.mt_firstpass import draft_translate, nllb_available, unload_nllb
from copydecode.prompts import (
    GLOSSARY_SYSTEM,
    POLISH_SYSTEM,
    TRANSLATE_SYSTEM,
    build_span_user_prompt,
    build_user_prompt,
    job_style,
    parse_glossary_json,
    parse_numbered,
    span_prefix_text,
    span_system_prompt,
)
from copydecode.qwen_tokens import prompt_token_budget, qwen_token_count, tokenizer_label
from copydecode.router import needs_llm
from copydecode.spans import (
    EditProgram,
    SpanJob,
    pack_span_jobs,
    replacement_ok,
    span_jobs_for,
    tag_text,
    trim_echo,
)

console = Console()


@dataclass
class JobConfig:
    input_path: Path
    output_path: Path
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    mode: str = "auto"
    glossary_path: Path | None = None
    save_glossary: Path | None = None
    extract_glossary: bool = False
    style: str = ""
    max_chars: int = 0
    temperature: float = 0.25
    num_ctx: int = 0
    timeout: float = 600.0
    retries: int = 2
    from_chapter: int = 1
    to_chapter: int = 0
    state_dir: Path | None = None
    dry_run: bool = False
    clean: bool = False
    apply_glossary: bool = True
    skip_mode: str = "auto"
    workers: int = 0
    nllb_firstpass: bool = True
    allow_reasoning: bool = False
    copydecode: bool = True
    changelog: bool = False
    checkpoint: bool = False
    token_pack: bool = True
    speculate: bool = True
    learned_tagger: bool = True
    update_tagger: bool = True


def default_chunk_size(model: str, fallback: int = 900) -> int:
    name = model.lower()
    if any(tag in name for tag in (":32b", "32b", ":27b", ":22b", ":20b")):
        return min(fallback, 2800) if fallback else 2800
    if any(tag in name for tag in (":14b", ":13b", ":12b", ":9b")):
        return 1800 if not fallback else min(fallback, 2200)
    if any(tag in name for tag in (":8b", ":7b")):
        return 1400 if not fallback else min(fallback, 1600)
    return fallback or 900


def selectable_chapters(chapters: list[Chapter]) -> list[Chapter]:
    return [ch for ch in chapters if not ch.skip and ch.segments]


def length_ok(source: str, output: str, mode: str) -> bool:
    if not output.strip():
        return False
    ratio = len(output) / max(len(source), 1)
    if mode == "translate":
        return 0.35 <= ratio <= 4.5
    return 0.35 <= ratio <= 2.6


def max_tokens_for(
    segments,
    mode: str,
    num_ctx: int,
    count_tokens: Callable[[str], int] | None = None,
) -> int:
    if count_tokens:
        texts = "".join(seg.text for seg in segments)
        tokens = count_tokens(texts)
        estimate = int(tokens * (1.2 if mode == "translate" else 1.1)) + 24
    else:
        chars = sum(len(seg.text) for seg in segments)
        if mode == "translate":
            estimate = int(chars * 1.1) + 80
        else:
            estimate = int(chars * 0.55) + 80
    cap = max(128, num_ctx // 2)
    return max(96, min(estimate, cap, 3072))


def max_tokens_for_spans(
    jobs: list[SpanJob],
    num_ctx: int,
    count_tokens: Callable[[str], int] | None = None,
) -> int:
    n = max(1, len(jobs))
    if count_tokens:
        replace_tokens = sum(count_tokens(job.text) for job in jobs)
        estimate = int(replace_tokens * 1.35) + 24 * n
    else:
        chars = sum(len(job.text) for job in jobs)
        estimate = int(chars * 0.85) + 32 * n
    cap = max(128, num_ctx // 2)
    return max(48 * n, min(estimate, cap, 2048))


def sample_excerpts(chapters: list[Chapter], count: int = 10, size: int = 700) -> str:
    texts = [seg.text for ch in selectable_chapters(chapters) for seg in ch.segments]
    if not texts:
        return ""
    if len(texts) <= count:
        return "\n\n".join(texts)[: count * size]
    step = max(len(texts) // count, 1)
    excerpts = []
    for i in range(0, len(texts), step):
        excerpts.append(texts[i][:size])
        if len(excerpts) >= count:
            break
    return "\n\n".join(excerpts)


def extract_glossary(client: LLMEngine, chapters: list[Chapter]) -> Glossary:
    excerpt = sample_excerpts(chapters)
    if not excerpt:
        return Glossary()
    raw = client.generate(
        "Extract glossary terms from this document sample:\n\n" + excerpt,
        system=GLOSSARY_SYSTEM,
        max_tokens=1200,
    )
    terms = parse_glossary_json(raw)
    return glossary_from_data({"terms": terms})


def build_glossary(config: JobConfig, chapters: list[Chapter], client: LLMEngine | None) -> Glossary:
    glossary = Glossary()
    if config.glossary_path and config.glossary_path.exists():
        glossary.merge(load_glossary_file(config.glossary_path), overwrite=True)
    if config.extract_glossary:
        if client is None:
            raise RuntimeError("An LLM server is required for glossary extraction.")
        console.print("[cyan]Extracting glossary from the document…[/cyan]")
        extracted = extract_glossary(client, chapters)
        glossary.merge(extracted, overwrite=False)
        console.print(f"Glossary now has {len(glossary.terms)} terms.")
    if config.save_glossary:
        glossary.save(config.save_glossary)
        console.print(f"Wrote glossary to {config.save_glossary}")
    return glossary


def rewrite_chunk(
    client: LLMEngine,
    mode: str,
    segments,
    glossary: Glossary,
    previous: str,
    style: str,
    retries: int,
    num_ctx: int,
    count_tokens: Callable[[str], int] | None = None,
) -> list[str]:
    numbered = format_numbered(segments)
    system = POLISH_SYSTEM if mode == "polish" else TRANSLATE_SYSTEM
    leftover = glossary.as_prompt(numbered) if glossary.unapplied_hits(numbered) else ""
    user = build_user_prompt(
        numbered_text=numbered,
        glossary_block=leftover,
        previous=previous,
        extra_style=style,
        mode=mode,
    )
    last = ""
    max_tokens = max_tokens_for(segments, mode, num_ctx, count_tokens=count_tokens)
    for attempt in range(retries + 1):
        prompt = user
        if attempt:
            prompt += (
                f"\n\nReturn EXACTLY {len(segments)} numbered blocks "
                f"[1] through [{len(segments)}]. No extra commentary."
            )
        last = client.generate(prompt, system=system, max_tokens=max_tokens)
        parsed = parse_numbered(last, len(segments))
        if not parsed:
            continue
        gated = [
            out if length_ok(src.text, out, mode) else src.text
            for src, out in zip(segments, parsed)
        ]
        if any(not length_ok(src.text, out, mode) for src, out in zip(segments, parsed)):
            if all(g == src.text for g, src in zip(gated, segments)):
                continue
        return gated
    parsed = parse_numbered(last, len(segments))
    if parsed:
        return [
            out if length_ok(src.text, out, mode) else src.text
            for src, out in zip(segments, parsed)
        ]
    console.print("[yellow]Could not parse model output; keeping the original passage.[/yellow]")
    return [seg.text for seg in segments]


def _gate_span_outputs(jobs: list[SpanJob], parsed: list[str]) -> tuple[list[str], int]:
    gated: list[str] = []
    good = 0
    for job, raw in zip(jobs, parsed):
        out = trim_echo(raw, job.text, job.before, job.after)
        if replacement_ok(job.text, out):
            gated.append(out)
            good += 1
        else:
            gated.append(job.text)
    return gated, good


def rewrite_span_jobs(
    client: LLMEngine,
    jobs: list[SpanJob],
    glossary: Glossary,
    previous: str,
    style: str,
    retries: int,
    num_ctx: int,
    *,
    glossary_block: str = "",
    count_tokens: Callable[[str], int] | None = None,
    speculate: bool = True,
    n_keep: int = 0,
) -> list[str]:
    del previous
    if not glossary_block:
        numbered = " ".join(job.text for job in jobs)
        glossary_block = glossary.as_prompt(numbered) if glossary.unapplied_hits(numbered) else ""
    system = span_system_prompt(glossary_block, style)
    user = build_span_user_prompt(jobs)
    last = ""
    max_tokens = max_tokens_for_spans(jobs, num_ctx, count_tokens=count_tokens)
    stop = ["KEEP before", "KEEP after"]
    if len(jobs) == 1:
        stop.extend(["\n[1]", "\n[2]"])
    for attempt in range(retries + 1):
        prompt = user
        if attempt:
            prompt += f"\n\nOutput [{len(jobs)}] numbered REPLACE block(s) only."
        last = client.generate(
            prompt,
            system=system,
            max_tokens=max_tokens,
            stop=stop,
            speculate=speculate and attempt == 0,
            n_keep=n_keep,
        )
        parsed = parse_numbered(last, len(jobs))
        if not parsed:
            continue
        gated, good = _gate_span_outputs(jobs, parsed)
        if good:
            return gated
    parsed = parse_numbered(last, len(jobs))
    if parsed:
        gated, good = _gate_span_outputs(jobs, parsed)
        if good:
            return gated
    if len(jobs) > 1:
        mid = max(1, len(jobs) // 2)
        console.print(
            f"[dim]REPLACE pack of {len(jobs)} failed checks; splitting.[/dim]"
        )
        return rewrite_span_jobs(
            client,
            jobs[:mid],
            glossary,
            "",
            style,
            retries,
            num_ctx,
            glossary_block=glossary_block,
            count_tokens=count_tokens,
            speculate=speculate,
            n_keep=n_keep,
        ) + rewrite_span_jobs(
            client,
            jobs[mid:],
            glossary,
            "",
            style,
            retries,
            num_ctx,
            glossary_block=glossary_block,
            count_tokens=count_tokens,
            speculate=speculate,
            n_keep=n_keep,
        )
    return [job.text for job in jobs]


def pack_kwargs(
    *,
    max_chars: int,
    num_ctx: int,
    token_pack: bool,
    glossary_block: str = "",
    style: str = "",
    copydecode: bool = False,
    mode: str = "polish",
) -> dict:
    count_tokens = qwen_token_count if token_pack else None
    prefix_tokens = 0
    max_prompt_tokens = 0
    if count_tokens:
        max_prompt_tokens = prompt_token_budget(num_ctx)
        if copydecode:
            prefix_tokens = count_tokens(span_prefix_text(glossary_block, style))
        else:
            system = POLISH_SYSTEM if mode == "polish" else TRANSLATE_SYSTEM
            prefix_tokens = count_tokens(
                system + "\n\nPolish these passages. Keep the same [n] labels:\n\n"
            )
    return {
        "max_chars": max_chars,
        "max_prompt_tokens": max_prompt_tokens,
        "prefix_tokens": prefix_tokens,
        "count_tokens": count_tokens,
    }


def programs_for_chapter(
    chapter: Chapter,
    mode: str,
    skip_mode: str,
    glossary: Glossary,
    *,
    learned: bool | None = None,
) -> dict[int, EditProgram]:
    programs: dict[int, EditProgram] = {}
    for seg in chapter.segments:
        if not needs_llm(seg, mode, skip_mode, glossary):
            continue
        programs[seg.index] = tag_text(
            seg.text,
            mode,
            skip_mode,
            glossary,
            seg.tag,
            force_dirty=True,
            learned=learned,
        )
    return programs


def jobs_from_programs(programs: dict[int, EditProgram]) -> list[SpanJob]:
    jobs: list[SpanJob] = []
    for index in sorted(programs):
        jobs.extend(span_jobs_for(index, programs[index]))
    return jobs


def prepare_chapter_text(
    chapter: Chapter,
    glossary: Glossary,
    apply_glossary: bool,
    mode: str,
    use_nllb: bool,
    changes: ChangeLog | None = None,
) -> None:
    for seg in chapter.segments:
        text = seg.text
        if apply_glossary:
            if changes:
                changes.record_glossary(glossary.hit_counts(text))
            text = glossary.apply_to_text(text)
        if use_nllb and mode == "translate":
            text = draft_translate(text)
        seg.text = text


def print_plan(
    chapters: list[Chapter],
    mode: str,
    model: str,
    max_chars: int,
    skip_mode: str,
    glossary: Glossary,
    profile: DeviceProfile | None = None,
    engine_label: str = "",
    copydecode: bool = False,
    packing: dict | None = None,
    learned: bool | None = None,
    glossary_label: str = "",
) -> tuple[int, int]:
    packing = packing or {
        "max_chars": max_chars,
        "max_prompt_tokens": 0,
        "prefix_tokens": 0,
        "count_tokens": None,
    }
    pack_label = (
        f"{packing['max_prompt_tokens']} tok/pack"
        if packing.get("count_tokens")
        else f"{max_chars} chars/chunk"
    )
    table = Table(title=f"{mode} · {model} · {pack_label}")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Title")
    table.add_column("Paras", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Skip", justify="right")
    work = selectable_chapters(chapters)
    llm_total = 0
    skip_total = 0
    keep_chars = 0
    replace_chars = 0
    for i, ch in enumerate(work, start=1):
        dirty = [seg for seg in ch.segments if needs_llm(seg, mode, skip_mode, glossary)]
        skipped = len(ch.segments) - len(dirty)
        if copydecode and mode == "polish":
            programs = programs_for_chapter(ch, mode, skip_mode, glossary, learned=learned)
            jobs = jobs_from_programs(programs)
            packs = (
                pack_span_jobs(
                    jobs,
                    packing["max_chars"],
                    max_prompt_tokens=packing["max_prompt_tokens"],
                    prefix_tokens=packing["prefix_tokens"],
                    count_tokens=packing["count_tokens"],
                )
                if jobs
                else []
            )
            for prog in programs.values():
                keep_chars += prog.keep_chars
                replace_chars += prog.replace_chars
        else:
            packs = (
                pack_segments(
                    dirty,
                    packing["max_chars"],
                    max_prompt_tokens=packing["max_prompt_tokens"],
                    prefix_tokens=packing["prefix_tokens"],
                    count_tokens=packing["count_tokens"],
                )
                if dirty
                else []
            )
        llm_total += len(packs)
        skip_total += skipped
        table.add_row(
            str(i),
            ch.href,
            ch.title[:42],
            str(len(ch.segments)),
            str(len(packs)),
            str(skipped),
        )
    console.print(table)
    if profile:
        console.print(
            f"[dim]{profile.name} · {profile.backend} · "
            f"ctx {profile.num_ctx} · workers {profile.workers} · skip {skip_mode}"
            + (f" · {engine_label}" if engine_label else "")
            + (" · KEEP/REPLACE stitch" if copydecode and mode == "polish" else "")
            + (
                f" · {tokenizer_label()}"
                if packing.get("count_tokens")
                else ""
            )
            + (
                " · learned tagger"
                if copydecode and mode == "polish" and learned
                else (" · heuristic tagger" if copydecode and mode == "polish" else "")
            )
            + (f" · {glossary_label}" if glossary_label else "")
            + "[/dim]"
        )
    extras = f"LLM chunks: {llm_total}  ·  paragraphs skipped: {skip_total}"
    total_tagged = keep_chars + replace_chars
    if copydecode and mode == "polish" and total_tagged:
        pct = 100.0 * keep_chars / total_tagged
        extras += f"  ·  KEEP copy: {pct:.0f}% (AR {100.0 - pct:.0f}%)"
    console.print(extras)
    return llm_total, skip_total


def process_chapter(
    chapter: Chapter,
    *,
    client: LLMEngine,
    mode: str,
    glossary: Glossary,
    max_chars: int,
    skip_mode: str,
    style: str,
    retries: int,
    num_ctx: int,
    ckpt: Checkpoint,
    copydecode: bool = False,
    changes: ChangeLog | None = None,
    packing: dict | None = None,
    glossary_block: str = "",
    speculate: bool = True,
    learned: bool | None = None,
) -> tuple[int, int]:
    packing = packing or {
        "max_chars": max_chars,
        "max_prompt_tokens": 0,
        "prefix_tokens": 0,
        "count_tokens": None,
    }
    dirty = [seg for seg in chapter.segments if needs_llm(seg, mode, skip_mode, glossary)]
    skipped = len(chapter.segments) - len(dirty)
    if not dirty:
        return 0, skipped
    if copydecode and mode == "polish":
        return _process_chapter_copydecode(
            chapter,
            dirty=dirty,
            skipped=skipped,
            client=client,
            mode=mode,
            glossary=glossary,
            max_chars=max_chars,
            skip_mode=skip_mode,
            style=style,
            retries=retries,
            num_ctx=num_ctx,
            ckpt=ckpt,
            changes=changes,
            packing=packing,
            glossary_block=glossary_block,
            speculate=speculate,
            learned=learned,
        )
    packs = pack_segments(
        dirty,
        packing["max_chars"],
        max_prompt_tokens=packing["max_prompt_tokens"],
        prefix_tokens=packing["prefix_tokens"],
        count_tokens=packing["count_tokens"],
    )
    assembled: dict[int, list[str]] = {seg.index: [] for seg in dirty}
    originals = {seg.index: seg.text for seg in dirty}
    previous = ""
    for pack_i, pack in enumerate(packs):
        first = pack[0].index
        last = pack[-1].index
        chunk_id = f"{chapter.item_id}:p{pack_i}:{first}-{last}:{len(pack)}"
        if ckpt.done(chunk_id):
            texts = ckpt.get(chunk_id)
        else:
            texts = rewrite_chunk(
                client,
                mode,
                pack,
                glossary,
                previous,
                style,
                retries,
                num_ctx,
                count_tokens=packing["count_tokens"],
            )
            ckpt.save_chunk(chunk_id, texts)
        for seg, text in zip(pack, texts):
            assembled[seg.index].append(text)
        previous = texts[-1][-500:]
    by_index = {seg.index: seg for seg in chapter.segments}
    for index, parts in assembled.items():
        if parts and index in by_index:
            new_text = "\n".join(parts)
            if changes:
                changes.record(
                    chapter_id=chapter.item_id,
                    chapter_title=chapter.title,
                    href=chapter.href,
                    para=index,
                    before=originals.get(index, ""),
                    after=new_text,
                    kind="paragraph",
                )
            by_index[index].text = new_text
    return len(packs), skipped


def _process_chapter_copydecode(
    chapter: Chapter,
    *,
    dirty: list,
    skipped: int,
    client: LLMEngine,
    mode: str,
    glossary: Glossary,
    max_chars: int,
    skip_mode: str,
    style: str,
    retries: int,
    num_ctx: int,
    ckpt: Checkpoint,
    changes: ChangeLog | None = None,
    packing: dict | None = None,
    glossary_block: str = "",
    speculate: bool = True,
    learned: bool | None = None,
) -> tuple[int, int]:
    packing = packing or {
        "max_chars": max_chars,
        "max_prompt_tokens": 0,
        "prefix_tokens": 0,
        "count_tokens": None,
    }
    programs = programs_for_chapter(chapter, mode, skip_mode, glossary, learned=learned)
    jobs = jobs_from_programs(programs)
    if not jobs:
        return 0, skipped + len(dirty)
    packs = pack_span_jobs(
        jobs,
        packing["max_chars"],
        max_prompt_tokens=packing["max_prompt_tokens"],
        prefix_tokens=packing["prefix_tokens"],
        count_tokens=packing["count_tokens"],
    )
    replacements: dict[tuple[int, int], str] = {}
    n_keep = packing.get("prefix_tokens") or 0
    for pack_i, pack in enumerate(packs):
        first = pack[0]
        last = pack[-1]
        chunk_id = (
            f"{chapter.item_id}:cd{pack_i}:{first.seg_index}:{first.span_index}-"
            f"{last.seg_index}:{last.span_index}:{len(pack)}"
        )
        if ckpt.done(chunk_id):
            texts = ckpt.get(chunk_id)
        else:
            texts = rewrite_span_jobs(
                client,
                pack,
                glossary,
                "",
                style,
                retries,
                num_ctx,
                glossary_block=glossary_block,
                count_tokens=packing["count_tokens"],
                speculate=speculate,
                n_keep=n_keep,
            )
            ckpt.save_chunk(chunk_id, texts)
        for job, text in zip(pack, texts):
            replacements[(job.seg_index, job.span_index)] = text
            if changes:
                changes.record(
                    chapter_id=chapter.item_id,
                    chapter_title=chapter.title,
                    href=chapter.href,
                    para=job.seg_index,
                    before=job.text,
                    after=text,
                    kind="span",
                )
    by_index = {seg.index: seg for seg in chapter.segments}
    for seg_index, program in programs.items():
        if seg_index not in by_index:
            continue
        span_replacements = {
            span_i: text
            for (s_i, span_i), text in replacements.items()
            if s_i == seg_index
        }
        by_index[seg_index].text = program.stitched(span_replacements)
    return len(packs), skipped


def run_job(config: JobConfig, client: LLMEngine, profile: DeviceProfile) -> Path:
    book, chapters = load_epub(str(config.input_path))
    work = selectable_chapters(chapters)
    if not work:
        raise RuntimeError("No readable chapter text was found in this EPUB.")

    sample = sample_text(seg.text for ch in work for seg in ch.segments)
    mode = config.mode if config.mode != "auto" else detect_mode(sample)
    num_ctx = config.num_ctx or profile.num_ctx
    max_chars = config.max_chars or profile.max_chars
    skip_mode = config.skip_mode if config.skip_mode != "auto" else profile.skip_mode
    workers = config.workers or profile.workers
    start = max(config.from_chapter, 1)
    end = config.to_chapter or len(work)
    chosen = work[start - 1 : end]

    style = job_style(config.style)
    glossary = build_glossary(config, chapters, client)
    use_nllb = bool(config.nllb_firstpass and mode == "translate" and nllb_available())
    if config.nllb_firstpass and mode == "translate" and not nllb_available():
        console.print(
            "[dim]NLLB first pass off (set COPYDECODE_NLLB_PATH to a CTranslate2 model).[/dim]"
        )

    changes: ChangeLog | None = None
    if config.changelog and not config.dry_run:
        changes = ChangeLog(
            input_name=config.input_path.name,
            output_name=config.output_path.name,
            mode=mode,
            model=config.model,
            engine=client.info.label,
        )
    chosen_ids = {ch.item_id for ch in chosen}
    for chapter in work:
        prepare_chapter_text(
            chapter,
            glossary,
            config.apply_glossary,
            mode,
            use_nllb,
            changes=changes if chapter.item_id in chosen_ids else None,
        )
    if use_nllb:
        unload_nllb()

    copydecode = bool(config.copydecode and mode == "polish")
    glossary_block = glossary.as_stable_prompt() if copydecode else ""
    packing = pack_kwargs(
        max_chars=max_chars,
        num_ctx=num_ctx,
        token_pack=config.token_pack,
        glossary_block=glossary_block,
        style=style,
        copydecode=copydecode,
        mode=mode,
    )
    engine_label = f"{client.info.label} @ {client.info.host}"
    if copydecode and client.info.kind in {"vllm", "llamacpp"}:
        engine_label += " · prefix cache"
        if config.speculate:
            engine_label += " · ngram REPLACE"
    if copydecode and client.can_prefill():
        engine_label += " · parallel prefill"
    learned_on = bool(config.learned_tagger)
    if learned_on:
        from copydecode.tagger import get_tagger, tagger_id

        learned_on = get_tagger() is not None
        tagger_fp = tagger_id() if learned_on else "heur"
    else:
        tagger_fp = "off"

    print_plan(
        chapters,
        mode,
        config.model,
        max_chars,
        skip_mode,
        glossary,
        profile,
        engine_label,
        copydecode=copydecode,
        packing=packing,
        learned=learned_on,
        glossary_label="keep source register",
    )
    chosen_llm = 0
    empty_chapters = 0
    for chapter in chosen:
        dirty = [seg for seg in chapter.segments if needs_llm(seg, mode, skip_mode, glossary)]
        if not dirty:
            empty_chapters += 1
            continue
        if copydecode:
            jobs = jobs_from_programs(
                programs_for_chapter(
                    chapter,
                    mode,
                    skip_mode,
                    glossary,
                    learned=config.learned_tagger,
                )
            )
            packs = (
                pack_span_jobs(
                    jobs,
                    packing["max_chars"],
                    max_prompt_tokens=packing["max_prompt_tokens"],
                    prefix_tokens=packing["prefix_tokens"],
                    count_tokens=packing["count_tokens"],
                )
                if jobs
                else []
            )
        else:
            packs = pack_segments(
                dirty,
                packing["max_chars"],
                max_prompt_tokens=packing["max_prompt_tokens"],
                prefix_tokens=packing["prefix_tokens"],
                count_tokens=packing["count_tokens"],
            )
        if packs:
            chosen_llm += len(packs)
        else:
            empty_chapters += 1
    if config.dry_run:
        return config.output_path

    state_path = None
    if config.checkpoint:
        state_dir = config.state_dir or (
            config.input_path.parent / ".polisher" / config.input_path.stem
        )
        state_path = state_dir / "checkpoint.json"
        if config.clean and state_path.exists():
            state_path.unlink()

    fingerprint = "|".join(
        [
            file_fingerprint(config.input_path),
            config.model,
            mode,
            str(max_chars),
            str(config.glossary_path or ""),
            style,
            skip_mode,
            str(config.apply_glossary),
            str(use_nllb),
            "cd1" if copydecode else "cd0",
            "tok1" if config.token_pack else "tok0",
            "sp1" if (copydecode and config.speculate) else "sp0",
            "gate2",
            f"tg:{tagger_fp}",
        ]
    )
    ckpt = Checkpoint(
        state_path,
        {"fingerprint": fingerprint, "mode": mode, "model": config.model, "engine": client.info.kind},
    )

    progress_total = max(chosen_llm + empty_chapters, 1)
    if copydecode and client.can_prefill() and (workers <= 1 or len(chosen) == 1):
        client.prefill(span_prefix_text(glossary_block, style))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rewriting", total=progress_total)

        def run_one(chapter: Chapter) -> tuple[Chapter, int]:
            packs, _skipped = process_chapter(
                chapter,
                client=client,
                mode=mode,
                glossary=glossary,
                max_chars=max_chars,
                skip_mode=skip_mode,
                style=style,
                retries=config.retries,
                num_ctx=num_ctx,
                ckpt=ckpt,
                copydecode=copydecode,
                changes=changes,
                packing=packing,
                glossary_block=glossary_block,
                speculate=bool(config.speculate and copydecode),
                learned=config.learned_tagger,
            )
            return chapter, packs if packs else 1

        if workers <= 1 or len(chosen) == 1:
            for chapter in chosen:
                progress.update(task, description=chapter.title[:40])
                _ch, stepped = run_one(chapter)
                progress.advance(task, stepped)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, chapter): chapter for chapter in chosen}
                for fut in as_completed(futures):
                    chapter, stepped = fut.result()
                    progress.update(task, description=chapter.title[:40])
                    progress.advance(task, stepped)

    write_epub(
        book,
        chapters,
        str(config.output_path),
        rewrite_ids={ch.item_id for ch in chosen},
        source_path=str(config.input_path),
    )
    console.print(f"[green]Wrote[/green] {config.output_path}")
    if changes is not None:
        md_path = config.output_path.with_name(config.output_path.stem + ".changes.md")
        json_path = config.output_path.with_name(config.output_path.stem + ".changes.json")
        changes.write(md_path, json_path)
        console.print(
            f"[green]Change log[/green] {md_path}  ·  {len(changes.edits)} rewrite(s), "
            f"{changes.identical} unchanged"
        )
        if config.update_tagger and config.learned_tagger and (changes.edits or changes.unchanged):
            try:
                from copydecode.tagger import leaky_model_text, train_from_files

                tagger, _dest = train_from_files([json_path], merge_existing=True)
                leaky = sum(1 for edit in changes.edits if leaky_model_text(edit.after))
                console.print(
                    f"[dim]Tagger updated for the next document · {len(tagger.anchors)} gold REPLACE anchors"
                    + (f" · {leaky} leaky outputs" if leaky else "")
                    + "[/dim]"
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                console.print(f"[yellow]Tagger update skipped:[/yellow] {exc}")
    if state_path is not None:
        console.print(f"Resume state: {state_path}")
    return config.output_path
