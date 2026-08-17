from __future__ import annotations

from collections.abc import Callable

from copydecode.engine import (
    EngineError,
    LLMEngine,
    discover_engine,
    list_models,
    pick_model,
)
from copydecode.glossary import Glossary
from copydecode.hardware import clamp_for_model, detect_device
from copydecode.pipeline import pack_kwargs, rewrite_span_jobs
from copydecode.prompts import job_style, span_prefix_text
from copydecode.router import is_boilerplate, mtl_score, skip_threshold
from copydecode.spans import pack_span_jobs, span_jobs_for, tag_text

Cancelled = Callable[[], bool]
ProgressFn = Callable[[int, int], None]
LogFn = Callable[[str], None]


def wants_polish(text: str, skip_mode: str = "auto") -> bool:
    """True if this English paragraph still looks like MTL that needs a copy-edit."""
    raw = (text or "").strip()
    if not raw or is_boilerplate(raw):
        return False
    if " " not in raw and len(raw) < 80:
        return False
    if mtl_score(raw, "polish") >= skip_threshold(skip_mode):
        return True
    return False


def connect_engine(
    *,
    auto_serve: bool = True,
    log: LogFn | None = None,
) -> tuple[LLMEngine, object]:
    """Use vLLM or llama.cpp for polish. Installs llama.cpp if needed. Ollama is optional fallback."""
    emit = log or (lambda _msg: None)
    profile = detect_device()
    found = None
    try:
        found = discover_engine("auto", None, profile)
    except EngineError:
        found = None

    info = found if found is not None and found.kind in {"vllm", "llamacpp"} else None
    if info is None and auto_serve:
        from copydecode.serve import start_llama_server

        try:
            emit("Starting llama.cpp (downloads llama-server + Qwen GGUF if needed)…")
            handle = start_llama_server(
                profile,
                download=True,
                detach=False,
                log=emit,
            )
            info = discover_engine("llamacpp", handle.host, profile)
        except EngineError as exc:
            if found is not None and found.kind == "ollama":
                emit(f"{exc} Using Ollama for polish instead.")
                info = found
            else:
                raise
    if info is None:
        if found is not None:
            info = found
        else:
            raise EngineError(
                "No local LLM. Run `copydecode serve` to install llama.cpp, "
                "or start llama-server / vLLM."
            )
    names = list_models(info)
    model = pick_model(names, profile.max_params_b)
    if not model:
        raise EngineError("No models found on the LLM server.")
    profile = clamp_for_model(profile, model)
    client = LLMEngine(
        info,
        model=model,
        temperature=0.25,
        num_ctx=profile.num_ctx,
        timeout=600.0,
    )
    emit(f"Polish engine: {info.label} @ {info.host} · {model}")
    return client, profile


def polish_paragraphs(
    texts: list[str],
    *,
    client: LLMEngine | None = None,
    profile=None,
    progress: ProgressFn | None = None,
    cancelled: Cancelled | None = None,
    auto_serve: bool = True,
    log: LogFn | None = None,
    close_client: bool | None = None,
) -> tuple[list[str], str]:
    """
    KEEP/REPLACE polish a list of English paragraphs.
    Returns (polished_texts, model_id). Caller owns caching.
    """
    if not texts:
        return [], ""
    own_client = client is None
    if close_client is None:
        close_client = own_client
    if client is None:
        client, profile = connect_engine(auto_serve=auto_serve, log=log)
    if profile is None:
        profile = detect_device()
        profile = clamp_for_model(profile, client.model)

    glossary = Glossary()
    style = job_style("")
    skip_mode = profile.skip_mode
    packing = pack_kwargs(
        max_chars=profile.max_chars,
        num_ctx=profile.num_ctx,
        token_pack=True,
        glossary_block="",
        style=style,
        copydecode=True,
        mode="polish",
    )
    results = list(texts)
    work: list[tuple[int, object, list]] = []
    for index, text in enumerate(texts):
        if cancelled and cancelled():
            if close_client:
                client.close()
            return results, client.model
        if not (text or "").strip() or is_boilerplate(text):
            continue
        program = tag_text(
            text,
            "polish",
            skip_mode,
            glossary,
            "p",
            force_dirty=True,
            learned=True,
        )
        jobs = span_jobs_for(index, program)
        if jobs:
            work.append((index, program, jobs))

    packs_total = 0
    packed_jobs: list[tuple[int, object, list, list]] = []
    for index, program, jobs in work:
        packs = pack_span_jobs(
            jobs,
            packing["max_chars"],
            max_prompt_tokens=packing["max_prompt_tokens"],
            prefix_tokens=packing["prefix_tokens"],
            count_tokens=packing["count_tokens"],
        )
        packs_total += len(packs)
        packed_jobs.append((index, program, jobs, packs))

    if progress:
        progress(0, max(packs_total, 1))
    if not packed_jobs:
        if close_client:
            client.close()
        return results, client.model

    if client.can_prefill():
        client.prefill(span_prefix_text("", style))

    done = 0
    n_keep = packing.get("prefix_tokens") or 0
    try:
        for index, program, _jobs, packs in packed_jobs:
            if cancelled and cancelled():
                break
            replacements: dict[int, str] = {}
            for pack in packs:
                if cancelled and cancelled():
                    break
                outs = rewrite_span_jobs(
                    client,
                    pack,
                    glossary,
                    "",
                    style,
                    retries=2,
                    num_ctx=profile.num_ctx,
                    glossary_block="",
                    count_tokens=packing["count_tokens"],
                    speculate=True,
                    n_keep=n_keep,
                )
                for job, out in zip(pack, outs):
                    replacements[job.span_index] = out
                done += 1
                if progress:
                    progress(done, max(packs_total, 1))
            results[index] = program.stitched(replacements)
    finally:
        if close_client:
            client.close()
    return results, client.model
