from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from copydecode.checkpoint import Checkpoint
from copydecode.engine import EngineInfo
from copydecode.epub_io import Chapter, Segment, apply_segments, parse_item_html
from copydecode.glossary import Glossary
from copydecode.pipeline import pack_kwargs, plan_chapter, process_chapter, rewrite_span_jobs
from copydecode.spans import span_jobs_for, tag_text


class FakeEngine:
    def __init__(self) -> None:
        self.info = EngineInfo("ollama", "http://127.0.0.1:9", "Ollama")
        self.calls = 0
        self.last_user = ""
        self.last_system = ""
        self.last_max_tokens = 0
        self.last_speculate = False

    def can_prefill(self) -> bool:
        return False

    def prefill(self, text: str) -> bool:
        return False

    def count_prompt_tokens(self, text: str) -> int | None:
        return None

    def generate(self, user, *, system, max_tokens, on_token=None, stop=None, **kwargs) -> str:
        self.calls += 1
        self.last_user = user
        self.last_system = system
        self.last_max_tokens = max_tokens
        self.last_speculate = bool(kwargs.get("speculate"))
        parts = []
        chunks = user.split("[")
        for chunk in chunks[1:]:
            body = chunk.split("]", 1)[-1]
            if "REPLACE:" not in body:
                continue
            dirty = body.split("REPLACE:", 1)[1]
            if "KEEP after" in dirty:
                dirty = dirty.split("KEEP after", 1)[0]
            dirty = dirty.strip()
            parts.append(f"[{len(parts)+1}]\nEDIT::{dirty}")
        if not parts:
            return "[1]\nEDIT::fallback"
        return "\n\n".join(parts)


def _packing(max_chars: int = 1800, num_ctx: int = 2048) -> dict:
    return pack_kwargs(max_chars=max_chars, num_ctx=num_ctx, token_pack=False)


class CopyDecodePipelineTests(unittest.TestCase):
    def test_rewrite_span_jobs_only_returns_replace_text(self) -> None:
        text = (
            "The mountain wind was cold. "
            "He could not help but smile. "
            "Then he walked toward the sect gate."
        )
        program = tag_text(text, "polish", "auto")
        jobs = span_jobs_for(0, program)
        self.assertTrue(jobs)
        engine = FakeEngine()
        out = rewrite_span_jobs(engine, jobs, retries=0, num_ctx=2048)
        self.assertTrue(engine.last_speculate)
        stitched = program.stitched({job.span_index: repl for job, repl in zip(jobs, out, strict=True)})
        self.assertTrue(stitched.startswith("The mountain wind was cold."))
        self.assertIn("EDIT::", stitched)
        keep_bits = [span.text for span in program.spans if span.kind == "KEEP"]
        for bit in keep_bits:
            self.assertIn(bit.strip()[:20], stitched)

    def test_glossary_stays_in_system_prefix(self) -> None:
        text = "The mountain wind was cold. He could not help but smile. Then he walked on."
        program = tag_text(text, "polish", "auto")
        jobs = span_jobs_for(0, program)
        engine = FakeEngine()
        rewrite_span_jobs(
            engine,
            jobs,
            retries=0,
            num_ctx=2048,
            glossary_block="Glossary (use these exact renderings):\n- Jindan → Golden Core",
        )
        self.assertIn("Jindan → Golden Core", engine.last_system)
        self.assertNotIn("Jindan → Golden Core", engine.last_user)

    def test_process_chapter_copies_keep(self) -> None:
        html = """<html><body>
        <p>The mountain wind was cold. He could not help but smile. Then he walked toward the sect gate.</p>
        <p>The sun rose over the quiet valley and birds sang.</p>
        </body></html>"""
        soup = parse_item_html(html.encode("utf-8"))
        segs = [
            Segment("c1", 0, "The mountain wind was cold. He could not help but smile. Then he walked toward the sect gate.", "p"),
            Segment("c1", 1, "The sun rose over the quiet valley and birds sang.", "p"),
        ]
        for i, el in enumerate(soup.body.find_all("p")):
            el["data-np-id"] = str(i)
        chapter = Chapter("c1", "c1.xhtml", "Ch 1", soup, segs)
        engine = FakeEngine()
        plan = plan_chapter(chapter, "polish", "auto", Glossary(), copydecode=True, packing=_packing())
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Checkpoint(Path(tmp) / "checkpoint.jsonl", {"fingerprint": "t", "mode": "polish"})
            packs, skipped = process_chapter(
                plan,
                client=engine,  # type: ignore[arg-type]
                mode="polish",
                glossary=Glossary(),
                style="",
                retries=0,
                num_ctx=2048,
                ckpt=ckpt,
            )
        self.assertGreaterEqual(packs, 1)
        self.assertGreaterEqual(skipped, 1)
        self.assertIn("mountain wind was cold", chapter.segments[0].text)
        self.assertIn("EDIT::", chapter.segments[0].text)
        self.assertEqual(
            chapter.segments[1].text,
            "The sun rose over the quiet valley and birds sang.",
        )
        mapping = {seg.index: seg.text for seg in chapter.segments}
        encoded = apply_segments(chapter.soup, mapping)
        self.assertIn(b"mountain wind was cold", encoded)
        self.assertNotIn(b"data-np-id", encoded)

    def test_max_tokens_tracks_dirty_span_not_paragraph(self) -> None:
        text = (
            "The mountain wind was cold and the pines did not move. "
            "He could not help but smile. "
            "Then he walked toward the sect gate with a steady step."
        )
        program = tag_text(text, "polish", "auto")
        jobs = span_jobs_for(0, program)
        engine = FakeEngine()
        rewrite_span_jobs(engine, jobs, retries=0, num_ctx=4096)
        dirty_chars = sum(len(job.text) for job in jobs)
        self.assertLess(engine.last_max_tokens, max(200, int(len(text) * 0.55) + 80))
        self.assertLessEqual(
            engine.last_max_tokens,
            max(48, int(dirty_chars * 0.85) + 32 + 5),
        )

    def test_rewrite_span_jobs_keeps_original_on_hallucination(self) -> None:
        source = 'Jiang Kai\'s eyes narrowed: "Here we come."'
        program = tag_text(source, "polish", "off")
        jobs = span_jobs_for(0, program)
        self.assertTrue(jobs)

        class HallucinationEngine(FakeEngine):
            def generate(self, user, *, system, max_tokens, on_token=None, stop=None, **kwargs) -> str:
                self.calls += 1
                self.last_user = user
                self.last_system = system
                self.last_max_tokens = max_tokens
                return (
                    "[1]\nThe news of Jiang Kai's arrival spread like wildfire, "
                    "and within a short time, the entire city was filled with the news."
                )

        engine = HallucinationEngine()
        out = rewrite_span_jobs(engine, jobs, retries=0, num_ctx=2048)
        self.assertEqual(out, [job.text for job in jobs])

    def test_token_max_tokens_uses_replace_length(self) -> None:
        from copydecode.pipeline import max_tokens_for_spans
        from copydecode.spans import SpanJob

        jobs = [SpanJob(0, 0, "word " * 200, "", "")]

        def count(text: str) -> int:
            return max(1, len(text.split()))

        got = max_tokens_for_spans(jobs, 4096, count_tokens=count)
        self.assertEqual(got, int(200 * 1.35) + 24)

    def test_polish_paragraphs_uses_keep_replace(self) -> None:
        from copydecode.api import polish_paragraphs
        from copydecode.hardware import DeviceProfile

        engine = FakeEngine()
        engine.model = "fake"
        profile = DeviceProfile(
            name="t",
            backend="cpu",
            vram_mb=0,
            ram_mb=8000,
            max_params_b=3,
            num_ctx=2048,
            max_chars=1800,
            workers=1,
            skip_mode="auto",
            notes=[],
        )
        text = (
            "The mountain wind was cold. "
            "He could not help but smile. "
            "Then he walked toward the sect gate."
        )
        out, model = polish_paragraphs(
            [text],
            client=engine,
            profile=profile,
            auto_serve=False,
            close_client=False,
        )
        self.assertEqual(model, "fake")
        self.assertTrue(out[0].startswith("The mountain wind was cold."))
        self.assertIn("EDIT::", out[0])


if __name__ == "__main__":
    unittest.main()
