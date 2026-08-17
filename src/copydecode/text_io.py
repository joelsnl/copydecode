# Author: joelsnl
"""Plain text, Markdown, HTML, and JSON readers/writers."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from copydecode.document import Chapter, Document, Segment, chapter_from_blocks
from copydecode.epub_io import apply_segments, extract_soup_segments, parse_item_html

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FORMFEED = "\f"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def load_txt(path: Path) -> Document:
    raw = _read_text(path)
    parts = raw.split(FORMFEED) if FORMFEED in raw else [raw]
    chapters: list[Chapter] = []
    for i, part in enumerate(parts):
        blocks = [(para, "p") for para in _paragraphs(part)]
        if not blocks:
            continue
        item_id = f"p{i}"
        title = path.stem if i == 0 else f"Section {i + 1}"
        chapters.append(chapter_from_blocks(item_id, f"{path.name}#{i}", title, blocks))
    if not chapters:
        chapters.append(chapter_from_blocks("p0", path.name, path.stem, []))
    return Document(path=path, fmt="txt", chapters=chapters, title=path.stem)


def _paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n\s*\n", text.strip())
    out: list[str] = []
    for chunk in chunks:
        para = " ".join(line.strip() for line in chunk.split("\n") if line.strip())
        if para:
            out.append(para)
    return out


def write_txt(chapters: list[Chapter], output: Path, *, rewrite_ids: set[str] | None = None) -> None:
    del rewrite_ids
    parts: list[str] = []
    for chapter in chapters:
        if chapter.skip:
            continue
        body = "\n\n".join(seg.text for seg in chapter.segments if seg.text.strip())
        if body:
            parts.append(body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(parts).rstrip() + ("\n" if parts else ""), encoding="utf-8")


def load_markdown(path: Path) -> Document:
    lines = _read_text(path).splitlines()
    chapters: list[Chapter] = []
    item_id = "md0"
    title = path.stem
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0

    def flush_para() -> None:
        if not buf:
            return
        text = " ".join(buf).strip()
        buf.clear()
        if text:
            blocks.append((text, "p"))

    def flush_chapter(next_title: str) -> None:
        nonlocal item_id, title, blocks
        flush_para()
        if blocks:
            chapters.append(chapter_from_blocks(item_id, f"{path.name}#{item_id}", title, blocks))
        item_id = f"md{len(chapters)}"
        title = next_title
        blocks = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            blocks.append(("\n".join(code), "pre"))
            i += 1
            continue
        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            if level == 1:
                if blocks or buf:
                    flush_chapter(heading_text)
                else:
                    title = heading_text
            else:
                flush_para()
            blocks.append((heading_text, f"h{level}"))
            i += 1
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        stripped = line.strip()
        if stripped[:2] in {"- ", "* ", "+ "} or re.match(r"^\d+\.\s+", stripped):
            flush_para()
            item = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", stripped)
            blocks.append((item, "li"))
            i += 1
            continue
        buf.append(stripped)
        i += 1
    flush_para()
    if blocks:
        chapters.append(chapter_from_blocks(item_id, path.name, title, blocks))
    if not chapters:
        chapters.append(chapter_from_blocks("md0", path.name, path.stem, []))
    return Document(path=path, fmt="md", chapters=chapters, title=path.stem)


def write_markdown(chapters: list[Chapter], output: Path, *, rewrite_ids: set[str] | None = None) -> None:
    del rewrite_ids
    lines: list[str] = []
    for chapter in chapters:
        if chapter.skip:
            continue
        for seg in chapter.segments:
            tag = seg.tag or "p"
            text = seg.text.rstrip()
            if tag == "pre":
                lines.append("```")
                lines.append(text)
                lines.append("```")
                lines.append("")
            elif tag.startswith("h") and tag[1:].isdigit():
                lines.append("#" * int(tag[1]) + " " + text)
                lines.append("")
            elif tag == "li":
                lines.append("- " + text)
            else:
                lines.append(text)
                lines.append("")
        if lines and lines[-1] != "":
            lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + ("\n" if lines else ""), encoding="utf-8")


def load_html(path: Path) -> Document:
    soup = parse_item_html(path.read_bytes())
    title_el = soup.find("title") or soup.find(["h1", "h2"])
    title = title_el.get_text(" ", strip=True) if title_el else path.stem
    segments = extract_soup_segments(soup, "html")
    if not segments:
        body = soup.body or soup
        text = " ".join(body.get_text(" ", strip=True).split())
        if text:
            segments = [Segment("html", 0, text, "p")]
    chapter = Chapter("html", path.name, title, soup, segments)
    return Document(path=path, fmt="html", chapters=[chapter], payload=soup, title=title)


def write_html(doc: Document, output: Path, chapters: list[Chapter] | None = None) -> None:
    work = chapters if chapters is not None else doc.chapters
    if doc.payload is not None and work and work[0].soup is not None:
        mapping = {seg.index: seg.text for seg in work[0].segments}
        content = apply_segments(work[0].soup, mapping)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        return
    title = escape(doc.title or output.stem)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{title}</title></head><body>",
    ]
    for chapter in work:
        if chapter.skip:
            continue
        for seg in chapter.segments:
            tag = seg.tag if seg.tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"} else "p"
            if tag == "pre":
                parts.append(f"<pre>{escape(seg.text)}</pre>")
            elif tag == "li":
                parts.append(f"<li>{escape(seg.text)}</li>")
            else:
                inner = "<br/>".join(escape(bit) for bit in seg.text.split("\n"))
                parts.append(f"<{tag}>{inner}</{tag}>")
    parts.append("</body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _text_from_json_obj(obj: object) -> tuple[str, str, str]:
    if isinstance(obj, str):
        return obj, "p", ""
    if not isinstance(obj, dict):
        return str(obj), "p", ""
    text = obj.get("text") or obj.get("paragraph") or obj.get("content") or ""
    tag = str(obj.get("tag") or "p")
    chapter = str(obj.get("chapter") or obj.get("title") or "")
    return str(text), tag, chapter


def load_json_records(path: Path, fmt: str) -> Document:
    raw = _read_text(path).strip()
    records: list[object] = []
    if fmt == "json":
        data = json.loads(raw or "[]")
        if isinstance(data, dict) and "paragraphs" in data:
            data = data["paragraphs"]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of strings or {text: ...} objects.")
        records = data
    else:
        for line in raw.splitlines():
            if line.strip():
                records.append(json.loads(line))
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for rec in records:
        text, tag, chapter = _text_from_json_obj(rec)
        if not str(text).strip():
            continue
        key = chapter or path.stem
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((str(text), tag))
    chapters = [
        chapter_from_blocks(f"j{i}", f"{path.name}#{i}", title, groups[title])
        for i, title in enumerate(order)
    ]
    if not chapters:
        chapters.append(chapter_from_blocks("j0", path.name, path.stem, []))
    return Document(path=path, fmt=fmt, chapters=chapters, title=path.stem)


def write_json_records(chapters: list[Chapter], output: Path, fmt: str) -> None:
    rows: list[dict[str, str]] = []
    for chapter in chapters:
        if chapter.skip:
            continue
        for seg in chapter.segments:
            rows.append({"chapter": chapter.title, "tag": seg.tag, "text": seg.text})
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
