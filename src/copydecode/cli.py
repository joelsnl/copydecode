"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from copydecode import __version__
from copydecode.document import default_output_path, detect_format, normalize_format
from copydecode.engine import (
    EngineError,
    EngineInfo,
    LLMEngine,
    discover_engine,
    list_models,
    pick_model,
)
from copydecode.errors import CopydecodeError
from copydecode.hardware import (
    DeviceProfile,
    clamp_for_model,
    detect_device,
    estimate_params_b,
    is_reasoning_model,
    recommended_serve_commands,
)
from copydecode.pipeline import JobConfig, build_glossary, run_job
from copydecode.serve import gguf_choice, plan_serve, start_llama_server, stop_server

console = Console()

COMMANDS = ("run", "glossary", "models", "devices", "serve")


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", help="Model name. Default: largest non-reasoning model that fits.")
    parser.add_argument("--host", help="LLM base URL. Default: auto-detect 8000/8080/11434")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "ollama", "llamacpp", "vllm", "openai"],
        help="Prefer a server type when several are running",
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--num-ctx", type=int, default=0, help="Context length. 0 = hardware default")
    parser.add_argument("--timeout", type=float, default=600.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copydecode",
        description="Local KEEP/REPLACE copy-edit and any-language→English translation.",
    )
    parser.add_argument("--version", action="version", version=f"copydecode {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Translate to English or polish English MTL")
    run.add_argument("input", type=Path, help="Source file: epub, pdf, txt, md, html, docx, json, jsonl")
    run.add_argument("-o", "--output", type=Path, help="Output path")
    run.add_argument(
        "-f",
        "--format",
        dest="output_format",
        default="",
        help="Output type: epub, pdf, txt, md, html, docx, json, jsonl. Default: same as the input.",
    )
    run.add_argument(
        "--mode",
        choices=["auto", "polish", "translate"],
        default="auto",
        help="auto detects non-English source scripts vs English MTL",
    )
    run.add_argument("--glossary", type=Path, help="JSON glossary to merge")
    run.add_argument("--save-glossary", type=Path, help="Write the merged glossary here")
    run.add_argument("--extract-glossary", action="store_true")
    run.add_argument("--style", default="", help="Extra style instructions")
    run.add_argument("--max-chars", type=int, default=0, help="Chunk size. 0 = hardware default")
    run.add_argument("--retries", type=int, default=2)
    run.add_argument(
        "--from-chapter",
        type=int,
        default=1,
        help="First section (EPUB chapter / PDF page with text / Markdown H1), 1-based",
    )
    run.add_argument("--to-chapter", type=int, default=0, help="Last section. 0 = last")
    run.add_argument("--state-dir", type=Path)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit. Never downloads, starts servers, or calls the LLM.",
    )
    run.add_argument("--clean", action="store_true")
    run.add_argument("--workers", type=int, default=0, help="Chapter workers. 0 = hardware default")
    run.add_argument(
        "--skip",
        dest="skip_mode",
        default="auto",
        choices=["auto", "off", "aggressive"],
        help="Skip already-clean paragraphs. auto follows VRAM/RAM",
    )
    run.add_argument("--no-glossary-apply", action="store_true", help="Do not rewrite terms before the LLM")
    run.add_argument("--no-nllb", action="store_true", help="Disable optional CTranslate2 first pass")
    run.add_argument(
        "--no-copydecode",
        action="store_true",
        help="Polish whole paragraphs instead of KEEP/REPLACE span stitch",
    )
    run.add_argument(
        "--changelog",
        action="store_true",
        help="Write before/after .changes.md and .changes.json next to the output",
    )
    run.add_argument(
        "--checkpoint",
        action="store_true",
        help="Write .copydecode/checkpoint.jsonl so a long run can resume",
    )
    run.add_argument(
        "--no-token-pack",
        action="store_true",
        help="Pack by character count instead of Qwen tokens",
    )
    run.add_argument(
        "--no-speculate",
        action="store_true",
        help="Disable ngram / prompt-lookup extras on REPLACE completions",
    )
    run.add_argument("--allow-reasoning", action="store_true", help="Allow DeepSeek-R1/QwQ anyway")
    run.add_argument(
        "--no-serve",
        action="store_true",
        help="Do not auto-download/start llama.cpp if no LLM server is running",
    )
    add_common_llm_args(run)

    gloss = sub.add_parser("glossary", help="Build a glossary JSON from a document")
    gloss.add_argument("input", type=Path)
    gloss.add_argument("-o", "--output", type=Path)
    gloss.add_argument("--glossary", type=Path)
    add_common_llm_args(gloss)

    models = sub.add_parser("models", help="List models on the detected server")
    models.add_argument("--host", help="LLM base URL")
    models.add_argument("--engine", default="auto", choices=["auto", "ollama", "llamacpp", "vllm", "openai"])

    devices = sub.add_parser("devices", help="Show CPU/GPU limits and which LLM server was found")
    devices.add_argument("--host", help="LLM base URL")
    devices.add_argument("--engine", default="auto", choices=["auto", "ollama", "llamacpp", "vllm", "openai"])

    serve = sub.add_parser(
        "serve",
        help="Download the llama.cpp build + Qwen GGUF for this machine and start :8080",
    )
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--detach", action="store_true", help="Start in the background")
    serve.add_argument("--stop", action="store_true", help="Stop a detached llama-server started by this tool")
    serve.add_argument("--no-download", action="store_true", help="Only use files already on disk")
    serve.add_argument("--dry-run", action="store_true", help="Print the detected OS/GPU and planned files")
    return parser


def connect(args: argparse.Namespace, *, auto_serve: bool = False) -> tuple[DeviceProfile, EngineInfo]:
    profile = detect_device()
    preferred = getattr(args, "engine", "auto")
    host = getattr(args, "host", None)
    try:
        info = discover_engine(preferred, host, profile)
        return profile, info
    except EngineError:
        if not auto_serve or host or getattr(args, "no_serve", False):
            raise
        console.print("[cyan]No LLM server found. Installing llama.cpp for this machine…[/cyan]")
        handle = start_llama_server(
            profile,
            download=True,
            detach=False,
            log=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
        info = discover_engine("llamacpp", handle.host, profile)
        return profile, info


def resolve_model(info: EngineInfo, profile: DeviceProfile, requested: str | None, allow_reasoning: bool) -> str:
    names = list_models(info)
    if requested:
        if requested not in names and f"{requested}:latest" not in names:
            console.print(f"[yellow]Model {requested} is not listed by the server. Trying it anyway.[/yellow]")
        if is_reasoning_model(requested) and not allow_reasoning:
            console.print(
                "[yellow]Reasoning models are a poor fit for KEEP/REPLACE copy-edit. "
                "Prefer Qwen2.5 7B/14B. Continuing because you named the model.[/yellow]"
            )
        return requested
    chosen = pick_model(names, profile.max_params_b)
    if not chosen:
        raise EngineError("No models found. Pull qwen2.5:7b or qwen2.5:14b.")
    if is_reasoning_model(chosen):
        console.print("[yellow]Only reasoning models are installed. Quality will lag and speed will tank.[/yellow]")
    return chosen


def print_profile(profile: DeviceProfile, info: EngineInfo | None = None) -> None:
    table = Table(title="This machine")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Device", profile.name)
    table.add_row("Vendor", profile.vendor)
    table.add_row("Backend", profile.backend)
    table.add_row("VRAM / unified", f"{profile.vram_mb} MB")
    table.add_row("RAM", f"{profile.ram_mb} MB")
    table.add_row("Max model", f"{profile.max_params_b:g}B (Q4-class)")
    table.add_row("Context", str(profile.num_ctx))
    table.add_row("Chunk chars", str(profile.max_chars))
    table.add_row("Workers", str(profile.workers))
    table.add_row("Skip", profile.skip_mode)
    table.add_row("Prompt pack", f"{max(256, profile.num_ctx // 2)} tokens")
    if info:
        table.add_row("Server", f"{info.label} {info.host}")
    console.print(table)
    for note in profile.notes:
        console.print(f"[dim]{note}[/dim]")
    if profile.backend in {"cuda", "vulkan", "metal"}:
        if info is not None and info.kind == "ollama":
            console.print(
                "[yellow]A faster path is `copydecode serve` (llama.cpp prefix cache + ngram). "
                "Stop Ollama first so the GPU is free, or keep using Ollama.[/yellow]"
            )
        for label, cmd in recommended_serve_commands(profile):
            console.print(f"[dim]{label}: {cmd}[/dim]")


def _connect_for_dry_run(args: argparse.Namespace) -> tuple[DeviceProfile, EngineInfo | None]:
    """Use a running server when there is one, but never install or start anything."""
    profile = detect_device()
    try:
        return profile, discover_engine(args.engine, args.host, profile)
    except EngineError:
        return profile, None


def _job_config(args: argparse.Namespace, output: Path, info: EngineInfo | None, model: str, out_fmt: str | None) -> JobConfig:
    return JobConfig(
        input_path=args.input,
        output_path=output,
        host=info.host if info else "",
        model=model,
        mode=args.mode,
        glossary_path=args.glossary,
        save_glossary=args.save_glossary,
        extract_glossary=args.extract_glossary,
        style=args.style,
        max_chars=args.max_chars,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
        retries=args.retries,
        from_chapter=args.from_chapter,
        to_chapter=args.to_chapter,
        state_dir=args.state_dir,
        dry_run=args.dry_run,
        clean=args.clean,
        apply_glossary=not args.no_glossary_apply,
        skip_mode=args.skip_mode,
        workers=args.workers,
        nllb_firstpass=not args.no_nllb,
        allow_reasoning=args.allow_reasoning,
        copydecode=not args.no_copydecode,
        changelog=bool(args.changelog),
        checkpoint=bool(args.checkpoint),
        token_pack=not args.no_token_pack,
        speculate=not args.no_speculate,
        output_format=out_fmt or "",
    )


def cmd_run(args: argparse.Namespace) -> int:
    if not args.input.exists():
        console.print(f"[red]File not found:[/red] {args.input}")
        return 1
    detect_format(args.input)
    out_fmt = normalize_format(args.output_format) if args.output_format else None

    if args.dry_run:
        profile, info = _connect_for_dry_run(args)
    else:
        profile, info = connect(args, auto_serve=True)

    if info is not None:
        model = resolve_model(info, profile, args.model, args.allow_reasoning)
    else:
        model = args.model or gguf_choice(profile)[0]
        console.print(f"[dim]No LLM server is running; planning with {model}.[/dim]")
    profile = clamp_for_model(profile, model)
    if args.workers:
        profile.workers = max(1, args.workers)
    if args.num_ctx:
        if profile.backend == "cuda" and profile.vram_mb < 16000 and args.num_ctx > profile.num_ctx:
            console.print(
                f"[yellow]--num-ctx {args.num_ctx} is above the {profile.num_ctx} "
                "hardware cap. KV cache will squeeze VRAM; leaving your override.[/yellow]"
            )
        profile.num_ctx = args.num_ctx
    if args.max_chars:
        profile.max_chars = args.max_chars
    if args.skip_mode != "auto":
        profile.skip_mode = args.skip_mode
    print_profile(profile, info)

    client: LLMEngine | None = None
    if info is not None:
        client = LLMEngine(
            info,
            model=model,
            temperature=args.temperature,
            num_ctx=profile.num_ctx,
            timeout=args.timeout,
        )
    output = args.output or default_output_path(
        args.input,
        "translate" if args.mode == "translate" else "polish",
        out_fmt,
    )
    config = _job_config(args, output, info, model, out_fmt)
    try:
        run_job(config, client, profile)
    finally:
        if client is not None:
            client.close()
    return 0


def cmd_glossary(args: argparse.Namespace) -> int:
    if not args.input.exists():
        console.print(f"[red]File not found:[/red] {args.input}")
        return 1
    from copydecode.io import load_document

    profile, info = connect(args, auto_serve=True)
    model = resolve_model(info, profile, args.model, allow_reasoning=False)
    profile = clamp_for_model(profile, model)
    doc = load_document(args.input)
    config = JobConfig(
        input_path=args.input,
        output_path=args.input,
        host=info.host,
        model=model,
        glossary_path=args.glossary,
        save_glossary=args.output or args.input.with_suffix(".glossary.json"),
        extract_glossary=True,
        temperature=args.temperature,
        num_ctx=profile.num_ctx,
        timeout=args.timeout,
    )
    with LLMEngine(
        info, model=model, temperature=args.temperature, num_ctx=profile.num_ctx, timeout=args.timeout
    ) as client:
        build_glossary(config, doc.chapters, client)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    profile, info = connect(args)
    names = list_models(info)
    if not names:
        console.print("No models listed by the server.")
        return 1
    table = Table(title=f"Models on {info.label}")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Fits")
    table.add_column("Default")
    chosen = pick_model(names, profile.max_params_b)
    for name in names:
        params = estimate_params_b(name)
        fits = (
            "yes"
            if params is not None and params <= profile.max_params_b + 0.2 and not is_reasoning_model(name)
            else "no"
        )
        size = f"{params:g}B" if params is not None else "?"
        table.add_row(name, size, fits, "yes" if name == chosen else "")
    console.print(table)
    print_profile(profile, info)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    profile = detect_device()
    print_profile(profile)
    if args.stop:
        if stop_server(log=lambda msg: console.print(msg)):
            return 0
        console.print("No llama-server pid file was found.")
        return 1
    if args.dry_run:
        table = Table(title="llama.cpp plan for this machine")
        table.add_column("Field")
        table.add_column("Value")
        for key, value in plan_serve(profile).items():
            table.add_row(key, value)
        console.print(table)
        return 0
    handle = start_llama_server(
        profile,
        port=args.port,
        download=not args.no_download,
        detach=args.detach,
        log=lambda msg: console.print(msg),
    )
    console.print(f"[green]llama.cpp[/green] {handle.host}  ·  {handle.alias}")
    console.print(f"Then: copydecode FILE --engine llamacpp --host {handle.host}")
    if args.detach or handle.proc is None:
        return 0
    console.print("[dim]Leave this window open. Ctrl+C stops the server.[/dim]")
    try:
        return int(handle.proc.wait() or 0)
    except KeyboardInterrupt:
        stop_server(log=lambda msg: console.print(msg))
        return 0


def cmd_devices(args: argparse.Namespace) -> int:
    profile = detect_device()
    try:
        info = discover_engine(args.engine, args.host, profile)
    except EngineError as exc:
        print_profile(profile)
        console.print(f"[yellow]{exc}[/yellow]")
        console.print("[dim]Tip: copydecode serve --dry-run[/dim]")
        return 0
    print_profile(profile, info)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    top_level = {"-h", "--help", "--version"}
    if raw and raw[0] not in COMMANDS and raw[0] not in top_level:
        # `copydecode book.epub` and `copydecode -o out.txt book.epub` both
        # mean `copydecode run …`.
        raw = ["run", *raw]
    parser = build_parser()
    args = parser.parse_args(raw)
    if not args.command:
        parser.print_help()
        return 0
    handlers = {
        "run": cmd_run,
        "glossary": cmd_glossary,
        "models": cmd_models,
        "devices": cmd_devices,
        "serve": cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except (CopydecodeError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
