# Author: joelsnl
"""PDF extract + reflowed write. Layout, images, and original fonts are not preserved."""

from __future__ import annotations

import os
import re
from pathlib import Path

from copydecode.document import Chapter, Document, chapter_from_blocks

HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _paragraphs_from_page(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    chunks = re.split(r"\n\s*\n", text.strip())
    paras: list[str] = []
    for chunk in chunks:
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue
        buf = lines[0]
        for line in lines[1:]:
            if buf.endswith((".", "!", "?", "。", "！", "？", ":", ";", '"', "”", "’")):
                paras.append(buf)
                buf = line
            else:
                buf = f"{buf} {line}"
        if buf:
            paras.append(buf)
    return paras


def load_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdf. Install with: pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError("This PDF is encrypted. Decrypt it before running copydecode.") from exc
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()
    chapters: list[Chapter] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        blocks = [(para, "p") for para in _paragraphs_from_page(raw)]
        if not blocks:
            continue
        item_id = f"page{i + 1}"
        title = f"Page {i + 1}"
        chapters.append(
            chapter_from_blocks(item_id, f"{path.name}#page-{i + 1}", title, blocks)
        )
    if not chapters:
        raise RuntimeError("No extractable text in this PDF (scanned image-only PDFs are not supported).")
    return Document(
        path=path,
        fmt="pdf",
        chapters=chapters,
        title=meta_title or path.stem,
    )


def _system_font() -> Path | None:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        Path(windir) / "Fonts" / "arial.ttf",
        Path(windir) / "Fonts" / "segoeui.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def write_pdf(chapters: list[Chapter], output: Path, title: str) -> None:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError("PDF output needs fpdf2. Install with: pip install fpdf2") from exc

    font_path = _system_font()
    pdf = FPDF()
    pdf.set_title(title or "copydecode")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    if font_path is not None:
        pdf.add_font("Body", "", str(font_path))
        pdf.set_font("Body", size=11)
        body_font = "Body"
    else:
        pdf.set_font("Helvetica", size=11)
        body_font = "Helvetica"

    def write_line(text: str, size: int = 11, gap: float = 8) -> None:
        pdf.set_font(body_font, size=size)
        safe = text if font_path is not None else text.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, gap, safe)
        pdf.ln(2)

    if title:
        write_line(title, size=16, gap=10)
        pdf.ln(2)
    for chapter in chapters:
        if chapter.skip:
            continue
        if chapter.title and not chapter.title.lower().startswith("page "):
            write_line(chapter.title, size=13, gap=8)
        for seg in chapter.segments:
            if not seg.text.strip():
                continue
            size = 13 if seg.tag.startswith("h") else 11
            write_line(seg.text, size=size, gap=8 if size == 13 else 6)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    pdf.output(str(tmp))
    tmp.replace(output)
