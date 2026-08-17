# Author: joelsnl
"""Load and write documents by suffix. KEEP/REPLACE itself does not care about file type."""

from __future__ import annotations

from pathlib import Path

from copydecode.document import OUTPUT_EXTENSIONS, Document, detect_format, normalize_format
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

READERS = {
    "epub": load_epub_document,
    "pdf": load_pdf,
    "txt": load_txt,
    "md": load_markdown,
    "html": load_html,
    "docx": load_docx,
    "jsonl": lambda path: load_json_records(path, "jsonl"),
    "json": lambda path: load_json_records(path, "json"),
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
    chapters = doc.chapters
    if out_fmt == "epub":
        if doc.fmt == "epub" and doc.payload is not None:
            write_epub(
                doc.payload,
                chapters,
                str(dest),
                rewrite_ids=rewrite_ids,
                source_path=doc.path,
            )
            return
        write_fresh_epub(chapters, dest, doc.title or dest.stem)
        return
    if out_fmt == "pdf":
        write_pdf(chapters, dest, doc.title or dest.stem)
        return
    if out_fmt == "txt":
        write_txt(chapters, dest, rewrite_ids=rewrite_ids)
        return
    if out_fmt == "md":
        write_markdown(chapters, dest, rewrite_ids=rewrite_ids)
        return
    if out_fmt == "html":
        write_html(doc, dest, chapters)
        return
    if out_fmt == "docx":
        write_docx(doc, dest, chapters)
        return
    if out_fmt in {"jsonl", "json"}:
        write_json_records(chapters, dest, out_fmt)
        return
    raise ValueError(f"Unsupported output format: {out_fmt}")
