from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from ebooklib import epub

from copydecode.epub_io import load_epub, match_zip_member, sanitize_toc, write_epub


def _write_sample_epub(path: Path) -> None:
    html = (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch</title></head>'
        b"<body><p>The mountain wind was cold.</p></body></html>"
    )
    container = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles><rootfile full-path=\"EPUB/content.opf\" "
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="id">test-1</dc:identifier>
<dc:title>Test</dc:title>
<dc:language>en</dc:language>
</metadata>
<manifest>
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
<item id="c0" href="chapter_0000.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine toc="ncx"><itemref idref="nav"/><itemref idref="c0"/></spine>
</package>
"""
    ncx = """<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="test-1"/></head>
<docTitle><text>Test</text></docTitle>
<navMap><navPoint id="n1" playOrder="1"><navLabel><text>Ch</text></navLabel>
<content src="chapter_0000.xhtml"/></navPoint></navMap>
</ncx>
"""
    nav = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        b'<body><nav epub:type="toc"><ol><li><a href="chapter_0000.xhtml">Ch</a></li></ol></nav>'
        b"</body></html>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("EPUB/content.opf", opf)
        zf.writestr("EPUB/toc.ncx", ncx)
        zf.writestr("EPUB/nav.xhtml", nav)
        zf.writestr("EPUB/chapter_0000.xhtml", html)


class ZipMatchTests(unittest.TestCase):
    def test_maps_opf_relative_name(self) -> None:
        names = ["mimetype", "EPUB/chapter_0000.xhtml", "EPUB/nav.xhtml"]
        self.assertEqual(match_zip_member(names, "chapter_0000.xhtml"), "EPUB/chapter_0000.xhtml")


class EpubWriteTests(unittest.TestCase):
    def test_zip_roundtrip_keeps_ncx_and_rewrites_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.epub"
            out = Path(tmp) / "out.epub"
            _write_sample_epub(src)
            book, chapters = load_epub(str(src))
            body = [ch for ch in chapters if "chapter_0000" in ch.href][0]
            body.segments[0].text = "The mountain wind cut like a blade."
            write_epub(book, chapters, str(out), source_path=src)
            with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out) as zout:
                self.assertEqual(zin.read("EPUB/toc.ncx"), zout.read("EPUB/toc.ncx"))
                self.assertIn(b"cut like a blade", zout.read("EPUB/chapter_0000.xhtml"))
                self.assertNotIn(b"data-np-id", zout.read("EPUB/chapter_0000.xhtml"))

    def test_none_uid_toc_writes_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.epub"
            _write_sample_epub(src)
            loaded, chapters = load_epub(str(src))
            loaded.toc = [epub.Link("chapter_0000.xhtml", "Ch", None)]
            out = Path(tmp) / "out.epub"
            write_epub(loaded, chapters, str(out), source_path=None)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 100)
            reloaded = epub.read_epub(str(out), options={"ignore_ncx": True})
            self.assertTrue(list(reloaded.get_items()))

    def test_sanitize_assigns_uids(self) -> None:
        book = epub.EpubBook()
        book.set_identifier("id-1")
        book.toc = [epub.Link("a.xhtml", "A", None), epub.Link("b.xhtml", None, None)]
        sanitize_toc(book)
        self.assertEqual(book.toc[0].uid, "np-toc-1")
        self.assertEqual(book.toc[1].uid, "np-toc-2")
        self.assertEqual(book.toc[1].title, "")


if __name__ == "__main__":
    unittest.main()
