# Author: joelsnl
"""Format-agnostic chapter/paragraph model used by every reader and writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKIP_LLM_TAGS = frozenset({"pre", "code", "samp"})

FORMAT_ALIASES = {
    "epub": "epub",
    "pdf": "pdf",
    "txt": "txt",
    "text": "txt",
    "md": "md",
    "markdown": "md",
    "html": "html",
    "htm": "html",
    "xhtml": "html",
    "docx": "docx",
    "jsonl": "jsonl",
    "json": "json",
}

OUTPUT_EXTENSIONS = {
    "epub": ".epub",
    "pdf": ".pdf",
    "txt": ".txt",
    "md": ".md",
    "html": ".html",
    "docx": ".docx",
    "jsonl": ".jsonl",
    "json": ".json",
}


@dataclass
class Segment:
    item_id: str
    index: int
    text: str
    tag: str

    @property
    def sid(self) -> str:
        return f"{self.item_id}:{self.index}"


@dataclass
class Chapter:
    item_id: str
    href: str
    title: str
    soup: Any = None
    segments: list[Segment] = field(default_factory=list)
    skip: bool = False


@dataclass
class Document:
    path: Path
    fmt: str
    chapters: list[Chapter]
    payload: Any = None
    title: str = ""


def normalize_format(value: str) -> str:
    key = (value or "").strip().lower().lstrip(".")
    if key in FORMAT_ALIASES:
        return FORMAT_ALIASES[key]
    raise ValueError(
        f"Unsupported format {value!r}. Use epub, pdf, txt, md, html, docx, json, or jsonl."
    )


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        try:
            return normalize_format(suffix)
        except ValueError:
            pass
    try:
        head = path.read_bytes()[:8]
    except OSError:
        head = b""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        name = path.name.lower()
        if name.endswith(".docx"):
            return "docx"
        return "epub"
    raise ValueError(
        f"Unsupported file type: {path.name}. Use epub, pdf, txt, md, html, docx, json, or jsonl."
    )


def default_output_path(input_path: Path, mode: str, fmt: str | None = None) -> Path:
    suffix = "en" if mode == "translate" else "polished"
    out_fmt = fmt or detect_format(input_path)
    ext = OUTPUT_EXTENSIONS.get(out_fmt, f".{out_fmt}")
    return input_path.with_name(f"{input_path.stem}.{suffix}{ext}")


def chapter_from_blocks(
    item_id: str,
    href: str,
    title: str,
    blocks: list[tuple[str, str]],
    *,
    soup: Any = None,
    skip: bool = False,
) -> Chapter:
    segments = [
        Segment(item_id=item_id, index=i, text=text, tag=tag)
        for i, (text, tag) in enumerate(blocks)
        if (text or "").strip()
    ]
    return Chapter(
        item_id=item_id,
        href=href,
        title=title,
        soup=soup,
        segments=segments,
        skip=skip,
    )
