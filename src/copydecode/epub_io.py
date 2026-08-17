from __future__ import annotations

import html
import uuid
import zipfile
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import ITEM_DOCUMENT, epub

from copydecode.document import Chapter, Document, Segment

BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "td",
    "th",
    "dt",
    "dd",
    "div",
}

SKIP_NAME_RE = ("nav", "toc", "ncx", "cover", "contents")


def _looks_like_nav(item: epub.EpubItem) -> bool:
    name = (item.get_name() or item.get_id() or "").lower()
    return any(token in name for token in SKIP_NAME_RE)


def _direct_block(el: Tag) -> bool:
    if el.name not in BLOCK_TAGS:
        return False
    for child in el.find_all(BLOCK_TAGS):
        if child is not el:
            return False
    return True


def parse_item_html(raw: bytes) -> BeautifulSoup:
    # html.parser is more forgiving than XML for messy Calibre/MTL EPUBs.
    return BeautifulSoup(raw, "html.parser")


def extract_soup_segments(soup: BeautifulSoup, item_id: str) -> list[Segment]:
    body = soup.body or soup
    segments: list[Segment] = []
    index = 0
    for el in body.find_all(BLOCK_TAGS):
        if not isinstance(el, Tag) or not _direct_block(el):
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) < 2:
            continue
        segments.append(
            Segment(item_id=item_id, index=index, text=text, tag=el.name)
        )
        el["data-np-id"] = str(index)
        index += 1
    return segments


def extract_segments(item: epub.EpubItem) -> tuple[BeautifulSoup, list[Segment]]:
    soup = parse_item_html(item.get_content())
    return soup, extract_soup_segments(soup, item.get_id())


def apply_segments(soup: BeautifulSoup, rewritten: dict[int, str]) -> bytes:
    body = soup.body or soup
    for el in body.find_all(attrs={"data-np-id": True}):
        if not isinstance(el, Tag):
            continue
        try:
            idx = int(el["data-np-id"])
        except (TypeError, ValueError):
            continue
        if idx not in rewritten:
            if "data-np-id" in el.attrs:
                del el.attrs["data-np-id"]
            continue
        text = rewritten[idx]
        el.clear()
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i:
                el.append(soup.new_tag("br"))
            el.append(NavigableString(part))
        del el.attrs["data-np-id"]

    html = str(soup)
    if "<?xml" not in html[:80]:
        html = '<?xml version="1.0" encoding="utf-8"?>\n' + html
    return html.encode("utf-8")


def load_epub(path: str) -> tuple[epub.EpubBook, list[Chapter]]:
    try:
        book = epub.read_epub(path, options={"ignore_ncx": True})
    except TypeError:
        book = epub.read_epub(path)
    chapters: list[Chapter] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup, segments = extract_segments(item)
        title = ""
        heading = soup.find(["h1", "h2", "h3", "title"])
        if heading:
            title = heading.get_text(" ", strip=True)
        if not title:
            title = item.get_name() or item.get_id()
        chapters.append(
            Chapter(
                item_id=item.get_id(),
                href=item.get_name(),
                title=title,
                soup=soup,
                segments=segments,
                skip=_looks_like_nav(item) and len(segments) < 8,
            )
        )
    return book, chapters


def match_zip_member(names: list[str], file_name: str) -> str | None:
    key = (file_name or "").replace("\\", "/").lstrip("./")
    if not key:
        return None
    if key in names:
        return key
    suffix = "/" + key
    hits = [name for name in names if name.replace("\\", "/") == key or name.replace("\\", "/").endswith(suffix)]
    if len(hits) == 1:
        return hits[0]
    base = key.rsplit("/", 1)[-1]
    hits = [name for name in names if name.replace("\\", "/").rsplit("/", 1)[-1] == base]
    if len(hits) == 1:
        return hits[0]
    return None


def _fix_toc_node(item: Any, counter: list[int]) -> None:
    if isinstance(item, epub.Link):
        if not item.uid:
            counter[0] += 1
            item.uid = f"np-toc-{counter[0]}"
        if item.title is None:
            item.title = ""
        if item.href is None:
            item.href = ""
    elif isinstance(item, epub.Section):
        if item.title is None:
            item.title = ""
        if item.href is None:
            item.href = ""


def sanitize_toc(book: epub.EpubBook) -> None:
    """ebooklib crashes NCX output when nav-derived Link.uid is None."""
    if not getattr(book, "uid", None):
        book.set_identifier(str(uuid.uuid4()))

    counter = [0]

    def walk(items: Any) -> list[Any]:
        out: list[Any] = []
        for item in items or []:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                _fix_toc_node(item[0], counter)
                out.append((item[0], walk(item[1])))
            else:
                _fix_toc_node(item, counter)
                out.append(item)
        return out

    book.toc = walk(book.toc)


def write_zip_roundtrip(source: str | Path, dest: str | Path, updates: dict[str, bytes]) -> None:
    """Copy the original EPUB zip and replace only rewritten documents."""
    source = Path(source)
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(source, "r") as zin:
        names = zin.namelist()
        resolved: dict[str, bytes] = {}
        missing: list[str] = []
        for file_name, content in updates.items():
            member = match_zip_member(names, file_name)
            if member is None:
                missing.append(file_name)
            else:
                resolved[member] = content
        if missing:
            raise FileNotFoundError(
                "Could not map rewritten files back into the EPUB zip: " + ", ".join(missing[:8])
            )
        with zipfile.ZipFile(tmp, "w") as zout:
            if "mimetype" in names:
                info = zin.getinfo("mimetype")
                out = zipfile.ZipInfo(filename="mimetype", date_time=info.date_time)
                out.compress_type = zipfile.ZIP_STORED
                zout.writestr(out, zin.read("mimetype"))
            for info in zin.infolist():
                if info.filename == "mimetype":
                    continue
                data = resolved.get(info.filename, zin.read(info.filename))
                out = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                out.compress_type = info.compress_type
                out.external_attr = info.external_attr
                zout.writestr(out, data)
    tmp.replace(dest)


def write_epub(
    book: epub.EpubBook,
    chapters: list[Chapter],
    output: str,
    rewrite_ids: set[str] | None = None,
    source_path: str | Path | None = None,
) -> None:
    rewritten_by_item = {ch.item_id: ch for ch in chapters}
    updates: dict[str, bytes] = {}
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        chapter = rewritten_by_item.get(item.get_id())
        if chapter is None or chapter.skip:
            continue
        if rewrite_ids is not None and chapter.item_id not in rewrite_ids:
            continue
        mapping = {seg.index: seg.text for seg in chapter.segments}
        content = apply_segments(chapter.soup, mapping)
        item.set_content(content)
        name = item.get_name() or getattr(item, "file_name", "") or ""
        if name:
            updates[name] = content

    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_path and Path(source_path).exists() and updates:
        try:
            write_zip_roundtrip(source_path, dest, updates)
            return
        except (OSError, zipfile.BadZipFile, FileNotFoundError, KeyError):
            pass

    sanitize_toc(book)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    epub.write_epub(str(tmp), book)
    tmp.replace(dest)


def load_epub_document(path: Path) -> Document:
    book, chapters = load_epub(str(path))
    title = ""
    try:
        titles = book.get_metadata("DC", "title")
        if titles:
            title = str(titles[0][0])
    except (KeyError, IndexError, TypeError):
        title = ""
    return Document(
        path=path,
        fmt="epub",
        chapters=chapters,
        payload=book,
        title=title or path.stem,
    )


def _segment_html(seg: Segment) -> str:
    tag = seg.tag if seg.tag in BLOCK_TAGS else "p"
    if tag == "div":
        tag = "p"
    body = "<br/>".join(html.escape(part) for part in seg.text.split("\n"))
    return f"<{tag}>{body}</{tag}>"


def write_fresh_epub(chapters: list[Chapter], output: Path, title: str) -> None:
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title or "copydecode")
    book.set_language("en")
    spine: list[Any] = ["nav"]
    toc: list[Any] = []
    for i, chapter in enumerate(chapters):
        if chapter.skip or not chapter.segments:
            continue
        name = f"chap_{i:04d}.xhtml"
        item = epub.EpubHtml(
            title=chapter.title or f"Section {i + 1}",
            file_name=name,
            lang="en",
            uid=chapter.item_id or f"ch{i}",
        )
        inner = "\n".join(_segment_html(seg) for seg in chapter.segments)
        item.content = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>'
            f"{html.escape(chapter.title or '')}</title></head><body>{inner}</body></html>"
        )
        book.add_item(item)
        spine.append(item)
        toc.append(item)
    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    epub.write_epub(str(tmp), book)
    tmp.replace(dest)
