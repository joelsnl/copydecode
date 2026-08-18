from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from copydecode.changelog import ChangeLog, same_text
from copydecode.checkpoint import Checkpoint
from copydecode.epub_io import Chapter, Segment, parse_item_html
from copydecode.glossary import Glossary
from copydecode.pipeline import pack_kwargs, plan_chapter, process_chapter


class FakeEngine:
    def __init__(self) -> None:
        self.info = type("I", (), {"kind": "ollama", "host": "http://x", "label": "Ollama"})()

    def can_prefill(self) -> bool:
        return False

    def generate(self, user, *, system, max_tokens, on_token=None, stop=None, **kwargs) -> str:
        parts = []
        for chunk in user.split("[")[1:]:
            body = chunk.split("]", 1)[-1]
            if "REPLACE:" not in body:
                continue
            dirty = body.split("REPLACE:", 1)[1]
            if "KEEP after" in dirty:
                dirty = dirty.split("KEEP after", 1)[0]
            parts.append(f"[{len(parts)+1}]\nEDIT::{dirty.strip()}")
        return "\n\n".join(parts) or "[1]\nEDIT::fallback"


class ChangeLogTests(unittest.TestCase):
    def test_skips_identical_rewrites(self) -> None:
        log = ChangeLog(mode="polish", model="qwen")
        log.record(
            chapter_id="c1",
            chapter_title="Ch 1",
            href="chapter_0000.xhtml",
            para=3,
            before="He could not help but smile.",
            after="He could not help but smile.",
        )
        self.assertEqual(log.sent, 1)
        self.assertEqual(log.identical, 1)
        self.assertEqual(log.edits, [])
        self.assertEqual(len(log.unchanged), 1)

    def test_markdown_contains_before_after(self) -> None:
        log = ChangeLog(
            input_name="book.epub",
            output_name="book.polished.epub",
            mode="polish",
            model="qwen2.5:14b",
        )
        log.record(
            chapter_id="c1",
            chapter_title="Chapter 1",
            href="chapter_0001.xhtml",
            para=4,
            before="He could not help but smile.",
            after="He couldn't help smiling.",
        )
        log.record_glossary({"jindan → Golden Core": 3})
        md = log.to_markdown()
        self.assertIn("He could not help but smile.", md)
        self.assertIn("He couldn't help smiling.", md)
        self.assertIn("Chapter 1", md)
        self.assertIn("Golden Core", md)
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "book.changes.md"
            log.write(md_path)
            self.assertTrue(md_path.exists())
            self.assertTrue(md_path.with_suffix(".json").exists())

    def test_process_chapter_records_span_edits(self) -> None:
        html = """<html><body>
        <p>The mountain wind was cold. He could not help but smile. Then he walked toward the sect gate.</p>
        <p>The sun rose over the quiet valley and birds sang.</p>
        </body></html>"""
        soup = parse_item_html(html.encode("utf-8"))
        segs = [
            Segment(
                "c1",
                0,
                "The mountain wind was cold. He could not help but smile. Then he walked toward the sect gate.",
                "p",
            ),
            Segment("c1", 1, "The sun rose over the quiet valley and birds sang.", "p"),
        ]
        chapter = Chapter("c1", "chapter_0000.xhtml", "Ch 1", soup, segs)
        log = ChangeLog(mode="polish", model="fake")
        engine = FakeEngine()
        packing = pack_kwargs(max_chars=1800, num_ctx=2048, token_pack=False)
        plan = plan_chapter(chapter, "polish", "auto", Glossary(), copydecode=True, packing=packing)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Checkpoint(Path(tmp) / "checkpoint.jsonl", {"fingerprint": "t", "mode": "polish"})
            process_chapter(
                plan,
                client=engine,  # type: ignore[arg-type]
                mode="polish",
                glossary=Glossary(),
                style="",
                retries=0,
                num_ctx=2048,
                ckpt=ckpt,
                changes=log,
            )
        self.assertGreaterEqual(log.sent, 1)
        self.assertTrue(any("could not help but" in edit.before for edit in log.edits))
        self.assertTrue(any(edit.after.startswith("EDIT::") for edit in log.edits))

    def test_same_text_ignores_whitespace(self) -> None:
        self.assertTrue(same_text("A  b\n c", "A b c"))
        self.assertFalse(same_text("A b", "A c"))

    def test_memory_checkpoint_does_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = Checkpoint(None, {"fingerprint": "t", "mode": "polish"})
            ckpt.save_chunk("c1", ["hello"])
            self.assertTrue(ckpt.done("c1"))
            self.assertEqual(ckpt.get("c1"), ["hello"])
            self.assertEqual(list(root.iterdir()), [])

    def test_job_config_skips_sidecars_by_default(self) -> None:
        from copydecode.pipeline import JobConfig

        config = JobConfig(input_path=Path("in.epub"), output_path=Path("out.epub"))
        self.assertFalse(config.changelog)
        self.assertFalse(config.checkpoint)


if __name__ == "__main__":
    unittest.main()
