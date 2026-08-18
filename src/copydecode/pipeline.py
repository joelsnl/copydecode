"""End-to-end job: load a document, plan the work, rewrite, write the output."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from copydecode.changelog import ChangeLog
from copydecode.checkpoint import Checkpoint, file_fingerprint
from copydecode.chunking import format_numbered, pack_segments
from copydecode.detect import detect_mode, sample_text
from copydecode.document import Chapter, Segment
from copydecode.engine import EngineError, LLMEngine
from copydecode.errors import CopydecodeError, DocumentError
from copydecode.glossary import Glossary, glossary_from_data, load_glossary_file
from copydecode.hardware import DeviceProfile
from copydecode.io import load_document, write_document
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
    output_format: str = ""


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
    segments: list[Segment],
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


def build_glossary(
    config: JobConfig,
    chapters: list[Chapter],
    client: LLMEngine | None,
    *,
    extract: bool | None = None,
) -> Glossary:
    glossary = Glossary()
    if config.glossary_path and config.glossary_path.exists():
        glossary.merge(load_glossary_file(config.glossary_path), overwrite=True)
    do_extract = config.extract_glossary if extract is None else extract
    if do_extract:
        if client is None:
            raise EngineError("An LLM server is required for glossary extraction.")
        console.print("[cyan]Extracting glossary from the document…[/cyan]")
        extracted = extract_glossary(client, chapters)
        glossary.merge(extracted, overwrite=False)
        console.print(f"Glossary now has {len(glossary.terms)} terms.")
    if config.save_glossary:
        glossary.save(config.save_glossary)
        console.print(f"Wrote glossary to {config.save_glossary}")
    return glossary


def _gate_chunk_outputs(
    segments: list[Segment], parsed: list[str], mode: str
) -> tuple[list[str], int]:
    """Keep each output only when it passes the length gate; count survivors."""
    gated: list[str] = []
    accepted = 0
    for seg, out in zip(segments, parsed, strict=True):
        if length_ok(seg.text, out, mode):
            gated.append(out)
            accepted += 1
        else:
            gated.append(seg.text)
    return gated, accepted


def rewrite_chunk(
    client: LLMEngine,
    mode: str,
    segments: list[Segment],
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
    max_tokens = max_tokens_for(segments, mode, num_ctx, count_tokens=count_tokens)
    last = ""
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
        gated, accepted = _gate_chunk_outputs(segments, parsed, mode)
        if accepted:
            return gated
    parsed = parse_numbered(last, len(segments))
    if parsed:
        gated, _accepted = _gate_chunk_outputs(segments, parsed, mode)
        return gated
    console.print("[yellow]Could not parse model output; keeping the original passage.[/yellow]")
    return [seg.text for seg in segments]


def _gate_span_outputs(jobs: list[SpanJob], parsed: list[str]) -> tuple[list[str], int]:
    gated: list[str] = []
    accepted = 0
    for job, raw in zip(jobs, parsed, strict=True):
        out = trim_echo(raw, job.text, job.before, job.after)
        if replacement_ok(job.text, out):
            gated.append(out)
            accepted += 1
        else:
            gated.append(job.text)
    return gated, accepted


def rewrite_span_jobs(
    client: LLMEngine,
    jobs: list[SpanJob],
    *,
    glossary_block: str = "",
    style: str = "",
    retries: int = 2,
    num_ctx: int = 4096,
    count_tokens: Callable[[str], int] | None = None,
    speculate: bool = True,
    n_keep: int = 0,
) -> list[str]:
    """Rewrite a pack of REPLACE spans; failed packs split in half and retry."""
    system = span_system_prompt(glossary_block, style)
    user = build_span_user_prompt(jobs)
    max_tokens = max_tokens_for_spans(jobs, num_ctx, count_tokens=count_tokens)
    stop = ["KEEP before", "KEEP after"]
    if len(jobs) == 1:
        stop.extend(["\n[1]", "\n[2]"])
    last = ""
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
        gated, accepted = _gate_span_outputs(jobs, parsed)
        if accepted:
            return gated
    parsed = parse_numbered(last, len(jobs))
    if parsed:
        gated, accepted = _gate_span_outputs(jobs, parsed)
        if accepted:
            return gated
    if len(jobs) > 1:
        mid = max(1, len(jobs) // 2)
        console.print(f"[dim]REPLACE pack of {len(jobs)} failed checks; splitting.[/dim]")
        halves = (jobs[:mid], jobs[mid:])
        return [
            text
            for half in halves
            for text in rewrite_span_jobs(
                client,
                half,
                glossary_block=glossary_block,
                style=style,
                retries=retries,
                num_ctx=num_ctx,
                count_tokens=count_tokens,
                speculate=speculate,
                n_keep=n_keep,
            )
        ]
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
        )
    return programs


def jobs_from_programs(programs: dict[int, EditProgram]) -> list[SpanJob]:
    jobs: list[SpanJob] = []
    for index in sorted(programs):
        jobs.extend(span_jobs_for(index, programs[index]))
    return jobs


@dataclass
class ChapterPlan:
    """Everything the LLM pass needs for one chapter, computed exactly once."""

    chapter: Chapter
    dirty: list[Segment]
    programs: dict[int, EditProgram] = field(default_factory=dict)
    span_packs: list[list[SpanJob]] = field(default_factory=list)
    segment_packs: list[list[Segment]] = field(default_factory=list)

    @property
    def llm_packs(self) -> int:
        return len(self.span_packs) + len(self.segment_packs)

    @property
    def skipped(self) -> int:
        return len(self.chapter.segments) - len(self.dirty)


def plan_chapter(
    chapter: Chapter,
    mode: str,
    skip_mode: str,
    glossary: Glossary,
    *,
    copydecode: bool,
    packing: dict,
) -> ChapterPlan:
    dirty = [seg for seg in chapter.segments if needs_llm(seg, mode, skip_mode, glossary)]
    programs: dict[int, EditProgram] = {}
    span_packs: list[list[SpanJob]] = []
    segment_packs: list[list[Segment]] = []
    if dirty and copydecode and mode == "polish":
        programs = programs_for_chapter(chapter, mode, skip_mode, glossary)
        jobs = jobs_from_programs(programs)
        if jobs:
            span_packs = pack_span_jobs(
                jobs,
                packing["max_chars"],
                max_prompt_tokens=packing["max_prompt_tokens"],
                prefix_tokens=packing["prefix_tokens"],
                count_tokens=packing["count_tokens"],
            )
    elif dirty:
        segment_packs = pack_segments(
            dirty,
            packing["max_chars"],
            max_prompt_tokens=packing["max_prompt_tokens"],
            prefix_tokens=packing["prefix_tokens"],
            count_tokens=packing["count_tokens"],
        )
    return ChapterPlan(
        chapter=chapter,
        dirty=dirty,
        programs=programs,
        span_packs=span_packs,
        segment_packs=segment_packs,
    )


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
    plans: list[ChapterPlan],
    mode: str,
    model: str,
    skip_mode: str,
    packing: dict,
    profile: DeviceProfile | None = None,
    engine_label: str = "",
    copydecode: bool = False,
) -> tuple[int, int]:
    pack_label = (
        f"{packing['max_prompt_tokens']} tok/pack"
        if packing.get("count_tokens")
        else f"{packing['max_chars']} chars/chunk"
    )
    table = Table(title=f"{mode} · {model} · {pack_label}")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Title")
    table.add_column("Paras", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Skip", justify="right")
    llm_total = 0
    skip_total = 0
    keep_chars = 0
    replace_chars = 0
    for i, plan in enumerate(plans, start=1):
        llm_total += plan.llm_packs
        skip_total += plan.skipped
        for program in plan.programs.values():
            keep_chars += program.keep_chars
            replace_chars += program.replace_chars
        table.add_row(
            str(i),
            plan.chapter.href,
            plan.chapter.title[:42],
            str(len(plan.chapter.segments)),
            str(plan.llm_packs),
            str(plan.skipped),
        )
    console.print(table)
    if profile:
        console.print(
            f"[dim]{profile.name} · {profile.backend} · "
            f"ctx {profile.num_ctx} · workers {profile.workers} · skip {skip_mode}"
            + (f" · {engine_label}" if engine_label else "")
            + (" · KEEP/REPLACE stitch" if copydecode and mode == "polish" else "")
            + (f" · {tokenizer_label()}" if packing.get("count_tokens") else "")
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
    plan: ChapterPlan,
    *,
    client: LLMEngine,
    mode: str,
    glossary: Glossary,
    style: str,
    retries: int,
    num_ctx: int,
    ckpt: Checkpoint,
    changes: ChangeLog | None = None,
    count_tokens: Callable[[str], int] | None = None,
    glossary_block: str = "",
    speculate: bool = True,
    n_keep: int = 0,
) -> tuple[int, int]:
    """Rewrite one planned chapter in place. Returns (packs sent, paragraphs skipped)."""
    if plan.span_packs:
        return _process_span_packs(
            plan,
            client=client,
            style=style,
            retries=retries,
            num_ctx=num_ctx,
            ckpt=ckpt,
            changes=changes,
            count_tokens=count_tokens,
            glossary_block=glossary_block,
            speculate=speculate,
            n_keep=n_keep,
        )
    if plan.segment_packs:
        return _process_segment_packs(
            plan,
            client=client,
            mode=mode,
            glossary=glossary,
            style=style,
            retries=retries,
            num_ctx=num_ctx,
            ckpt=ckpt,
            changes=changes,
            count_tokens=count_tokens,
        )
    return 0, len(plan.chapter.segments)


def _process_span_packs(
    plan: ChapterPlan,
    *,
    client: LLMEngine,
    style: str,
    retries: int,
    num_ctx: int,
    ckpt: Checkpoint,
    changes: ChangeLog | None,
    count_tokens: Callable[[str], int] | None,
    glossary_block: str,
    speculate: bool,
    n_keep: int,
) -> tuple[int, int]:
    chapter = plan.chapter
    replacements: dict[tuple[int, int], str] = {}
    for pack_i, pack in enumerate(plan.span_packs):
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
                glossary_block=glossary_block,
                style=style,
                retries=retries,
                num_ctx=num_ctx,
                count_tokens=count_tokens,
                speculate=speculate,
                n_keep=n_keep,
            )
            ckpt.save_chunk(chunk_id, texts)
        # strict: a length mismatch here means corrupt resume state, which
        # must fail loudly instead of silently dropping spans.
        for job, text in zip(pack, texts, strict=True):
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
    for seg_index, program in plan.programs.items():
        seg = by_index.get(seg_index)
        if seg is None:
            continue
        span_replacements = {
            span_i: text
            for (s_i, span_i), text in replacements.items()
            if s_i == seg_index
        }
        seg.text = program.stitched(span_replacements)
    return len(plan.span_packs), plan.skipped


def _process_segment_packs(
    plan: ChapterPlan,
    *,
    client: LLMEngine,
    mode: str,
    glossary: Glossary,
    style: str,
    retries: int,
    num_ctx: int,
    ckpt: Checkpoint,
    changes: ChangeLog | None,
    count_tokens: Callable[[str], int] | None,
) -> tuple[int, int]:
    chapter = plan.chapter
    assembled: dict[int, list[str]] = {seg.index: [] for seg in plan.dirty}
    originals = {seg.index: seg.text for seg in plan.dirty}
    previous = ""
    for pack_i, pack in enumerate(plan.segment_packs):
        chunk_id = f"{chapter.item_id}:p{pack_i}:{pack[0].index}-{pack[-1].index}:{len(pack)}"
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
                count_tokens=count_tokens,
            )
            ckpt.save_chunk(chunk_id, texts)
        for seg, text in zip(pack, texts, strict=True):
            assembled[seg.index].append(text)
        previous = texts[-1][-500:]
    by_index = {seg.index: seg for seg in chapter.segments}
    for index, parts in assembled.items():
        if not parts or index not in by_index:
            continue
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
    return len(plan.segment_packs), plan.skipped


def run_job(config: JobConfig, client: LLMEngine | None, profile: DeviceProfile) -> Path:
    """Run one document job. ``client`` may be None only for a dry run."""
    if client is None and not config.dry_run:
        raise CopydecodeError("An LLM engine is required unless dry_run is set.")
    doc = load_document(config.input_path)
    work = selectable_chapters(doc.chapters)
    if not work:
        raise DocumentError("No readable text was found in this file.")

    sample = sample_text(seg.text for ch in work for seg in ch.segments)
    mode = config.mode if config.mode != "auto" else detect_mode(sample)
    num_ctx = config.num_ctx or profile.num_ctx
    max_chars = config.max_chars or profile.max_chars
    skip_mode = config.skip_mode if config.skip_mode != "auto" else profile.skip_mode
    workers = config.workers or profile.workers
    start = max(config.from_chapter, 1)
    end = config.to_chapter or len(work)
    chosen = work[start - 1 : end]
    if not chosen:
        raise CopydecodeError(
            f"The chapter range {start}–{end or len(work)} selects no sections; "
            f"this file has {len(work)}."
        )

    style = job_style(config.style)
    if config.extract_glossary and config.dry_run:
        console.print("[dim]Glossary extraction is skipped during a dry run.[/dim]")
    glossary = build_glossary(
        config,
        doc.chapters,
        client,
        extract=config.extract_glossary and not config.dry_run,
    )

    use_nllb = bool(
        config.nllb_firstpass
        and mode == "translate"
        and not config.dry_run
        and nllb_available()
    )
    if config.nllb_firstpass and mode == "translate" and not config.dry_run and not use_nllb:
        console.print(
            "[dim]NLLB first pass off (set COPYDECODE_NLLB_PATH to a CTranslate2 model).[/dim]"
        )

    changes: ChangeLog | None = None
    if config.changelog and not config.dry_run and client is not None:
        changes = ChangeLog(
            input_name=config.input_path.name,
            output_name=config.output_path.name,
            mode=mode,
            model=config.model,
            engine=client.info.label,
        )

    # Only chapters inside the requested range are prepared and rewritten;
    # everything else is copied through byte-identical (EPUB) or verbatim.
    for chapter in chosen:
        prepare_chapter_text(chapter, glossary, config.apply_glossary, mode, use_nllb, changes=changes)
    if use_nllb:
        unload_nllb()
    if len(chosen) < len(work) and not config.dry_run:
        console.print(
            f"[dim]{len(work) - len(chosen)} section(s) outside the chapter range "
            "are copied through unchanged.[/dim]"
        )

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

    engine_label = ""
    if client is not None:
        engine_label = f"{client.info.label} @ {client.info.host}"
        if copydecode and client.info.kind in {"vllm", "llamacpp"}:
            engine_label += " · prefix cache"
            if config.speculate:
                engine_label += " · ngram REPLACE"
        if copydecode and client.can_prefill():
            engine_label += " · parallel prefill"

    plans = [
        plan_chapter(chapter, mode, skip_mode, glossary, copydecode=copydecode, packing=packing)
        for chapter in work
    ]
    chosen_ids = {ch.item_id for ch in chosen}
    chosen_plans = [plan for plan in plans if plan.chapter.item_id in chosen_ids]

    print_plan(
        plans,
        mode,
        config.model,
        skip_mode,
        packing,
        profile,
        engine_label,
        copydecode=copydecode,
    )
    if config.dry_run:
        return config.output_path
    assert client is not None

    state_path = None
    if config.checkpoint:
        state_dir = config.state_dir or (
            config.input_path.parent / ".copydecode" / config.input_path.stem
        )
        state_path = state_dir / "checkpoint.jsonl"
        if config.clean and state_path.exists():
            state_path.unlink()

    fingerprint = "|".join(
        [
            "v3",
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
        ]
    )
    ckpt = Checkpoint(
        state_path,
        {"fingerprint": fingerprint, "mode": mode, "model": config.model, "engine": client.info.kind},
    )

    progress_total = max(sum(max(plan.llm_packs, 1) for plan in chosen_plans), 1)
    n_keep = 0
    if copydecode and client.can_prefill():
        prefix = span_prefix_text(glossary_block, style)
        if workers <= 1 or len(chosen_plans) == 1:
            client.prefill(prefix)
        # n_keep must be counted by the server's own tokenizer; a client-side
        # estimate would pin the wrong prefix boundary on context shift.
        n_keep = client.count_prompt_tokens(prefix) or 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rewriting", total=progress_total)

        def run_one(plan: ChapterPlan) -> tuple[ChapterPlan, int]:
            packs, _skipped = process_chapter(
                plan,
                client=client,
                mode=mode,
                glossary=glossary,
                style=style,
                retries=config.retries,
                num_ctx=num_ctx,
                ckpt=ckpt,
                changes=changes,
                count_tokens=packing["count_tokens"],
                glossary_block=glossary_block,
                speculate=bool(config.speculate and copydecode),
                n_keep=n_keep,
            )
            return plan, packs if packs else 1

        if workers <= 1 or len(chosen_plans) == 1:
            for plan in chosen_plans:
                progress.update(task, description=plan.chapter.title[:40])
                _plan, stepped = run_one(plan)
                progress.advance(task, stepped)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, plan): plan for plan in chosen_plans}
                for fut in as_completed(futures):
                    plan, stepped = fut.result()
                    progress.update(task, description=plan.chapter.title[:40])
                    progress.advance(task, stepped)

    write_document(
        doc,
        config.output_path,
        rewrite_ids=chosen_ids,
        fmt=config.output_format or None,
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
    if state_path is not None:
        console.print(f"Resume state: {state_path}")
    return config.output_path
