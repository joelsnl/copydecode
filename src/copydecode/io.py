"""Load and write documents by suffix.

``READERS`` and ``WRITERS`` are plain registries keyed by format name, so a
new file type only needs two callables:

    READERS["fb2"] = load_fb2                       # (Path) -> Document
    WRITERS["fb2"] = write_fb2                      # (Document, Path, rewrite_ids) -> None
    FORMAT_ALIASES["fb2"] = "fb2"                   # accepted -f / suffix values
    OUTPUT_EXTENSIONS["fb2"] = ".fb2"

The KEEP/REPLACE pipeline itself never touches file formats.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from copydecode.document import (
    FORMAT_ALIASES,
    OUTPUT_EXTENSIONS,
    Document,
    detect_format,
    normalize_format,
)
from copydecode.docx_io import load_docx, write_docx
from copydecode.epub_io import load_epub_document, write_epub, write_fresh_epub
from copydecode.pdf_io import load_pdf, write_pdf
from copydecode.text_io import (
    load_html,
    load_json_records,
    load_markdown,
    load_txt,
    write_html,
    write_json_records,
    write_markdown,
    write_txt,
)

__all__ = [
    "FORMAT_ALIASES",
    "OUTPUT_EXTENSIONS",
    "READERS",
    "WRITERS",
    "load_document",
    "write_document",
]

Reader = Callable[[Path], Document]
Writer = Callable[[Document, Path, "set[str] | None"], None]

READERS: dict[str, Reader] = {
    "epub": load_epub_document,
    "pdf": load_pdf,
    "txt": load_txt,
    "md": load_markdown,
    "html": load_html,
    "docx": load_docx,
    "jsonl": lambda path: load_json_records(path, "jsonl"),
    "json": lambda path: load_json_records(path, "json"),
}


def _write_epub_out(doc: Document, dest: Path, rewrite_ids: set[str] | None) -> None:
    if doc.fmt == "epub" and doc.payload is not None:
        write_epub(doc.payload, doc.chapters, str(dest), rewrite_ids=rewrite_ids, source_path=doc.path)
    else:
        write_fresh_epub(doc.chapters, dest, doc.title or dest.stem)


WRITERS: dict[str, Writer] = {
    "epub": _write_epub_out,
    "pdf": lambda doc, dest, _ids: write_pdf(doc.chapters, dest, doc.title or dest.stem),
    "txt": lambda doc, dest, ids: write_txt(doc.chapters, dest, rewrite_ids=ids),
    "md": lambda doc, dest, ids: write_markdown(doc.chapters, dest, rewrite_ids=ids),
    "html": lambda doc, dest, _ids: write_html(doc, dest, doc.chapters),
    "docx": lambda doc, dest, _ids: write_docx(doc, dest, doc.chapters),
    "jsonl": lambda doc, dest, _ids: write_json_records(doc.chapters, dest, "jsonl"),
    "json": lambda doc, dest, _ids: write_json_records(doc.chapters, dest, "json"),
}


def load_document(path: str | Path) -> Document:
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(src)
    fmt = detect_format(src)
    reader = READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unsupported file type: {src.suffix}")
    return reader(src)


def write_document(
    doc: Document,
    output: str | Path,
    *,
    rewrite_ids: set[str] | None = None,
    fmt: str | None = None,
) -> None:
    dest = Path(output)
    out_fmt = normalize_format(fmt) if fmt else detect_format(dest) if dest.suffix else doc.fmt
    if not dest.suffix:
        dest = dest.with_suffix(OUTPUT_EXTENSIONS.get(out_fmt, f".{out_fmt}"))
    writer = WRITERS.get(out_fmt)
    if writer is None:
        raise ValueError(f"Unsupported output format: {out_fmt}")
    writer(doc, dest, rewrite_ids)
