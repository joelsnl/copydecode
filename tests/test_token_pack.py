from __future__ import annotations

import unittest

from copydecode.glossary import Glossary, Term
from copydecode.hardware import DeviceProfile, recommended_serve_commands
from copydecode.prompts import (
    SPAN_POLISH_INSTRUCTIONS,
    SPAN_POLISH_SYSTEM,
    build_span_user_prompt,
    span_prefix_text,
    span_system_prompt,
)
from copydecode.qwen_tokens import estimate_qwen_tokens, prompt_token_budget
from copydecode.spans import SpanJob, format_span_job, pack_span_jobs, tag_text, span_jobs_for


class QwenTokenTests(unittest.TestCase):
    def test_estimate_counts_cjk_heavier(self) -> None:
        english = estimate_qwen_tokens("The mountain wind was cold. " * 4)
        cjk = estimate_qwen_tokens("山风很冷。" * 8)
        self.assertGreater(english, 8)
        self.assertGreater(cjk, 8)

    def test_qwen_token_count_matches_estimate_without_vocab(self) -> None:
        import os

        from copydecode import qwen_tokens

        previous = os.environ.get("COPYDECODE_TOKENIZER")
        os.environ["COPYDECODE_TOKENIZER"] = "off"
        qwen_tokens._hf_tokenizer.cache_clear()
        try:
            text = "He could not help but smile."
            self.assertEqual(qwen_tokens.qwen_token_count(text), estimate_qwen_tokens(text))
        finally:
            if previous is None:
                os.environ.pop("COPYDECODE_TOKENIZER", None)
            else:
                os.environ["COPYDECODE_TOKENIZER"] = previous
            qwen_tokens._hf_tokenizer.cache_clear()

    def test_prompt_budget_is_half_ctx(self) -> None:
        self.assertEqual(prompt_token_budget(4096), 2048)
        self.assertGreaterEqual(prompt_token_budget(128), 256)


class TokenPackTests(unittest.TestCase):
    def test_token_pack_fills_until_budget(self) -> None:
        jobs = [
            SpanJob(i, 0, f"REPLACE span number {i} with MTL artifacts.", "", "")
            for i in range(12)
        ]

        def count(text: str) -> int:
            return max(1, len(text.split()))

        packed = pack_span_jobs(
            jobs,
            max_chars=10**9,
            max_prompt_tokens=40,
            prefix_tokens=10,
            count_tokens=count,
        )
        self.assertGreater(len(packed), 1)
        self.assertEqual(sum(len(p) for p in packed), len(jobs))
        for pack in packed:
            used = sum(count(format_span_job(job, 1)) + 2 for job in pack)
            if len(pack) > 1:
                self.assertLessEqual(used, 30)

    def test_char_pack_still_works(self) -> None:
        program = tag_text(
            "He could not help but smile. The path was quiet. He sucked in a cold air.",
            "polish",
            "auto",
            learned=False,
        )
        jobs = span_jobs_for(0, program)
        packed = pack_span_jobs(jobs, max_chars=40)
        self.assertEqual(sum(len(p) for p in packed), len(jobs))


class PrefixCacheTests(unittest.TestCase):
    def test_user_prompt_is_spans_only(self) -> None:
        jobs = [SpanJob(0, 1, "He could not help but smile.", "cold.", "Then")]
        user = build_span_user_prompt(jobs, glossary_block="Glossary: x", previous="old", extra_style="noir")
        self.assertIn(SPAN_POLISH_INSTRUCTIONS, user)
        self.assertIn("REPLACE:", user)
        self.assertNotIn("Glossary: x", user)
        self.assertNotIn("Previously rewritten", user)
        self.assertNotIn("noir", user)

    def test_system_prefix_is_byte_stable(self) -> None:
        gloss = "Glossary (use these exact renderings):\n- Jindan → Golden Core"
        a = span_prefix_text(gloss, "keep names")
        b = span_prefix_text(gloss, "keep names")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith(SPAN_POLISH_SYSTEM))
        self.assertIn(gloss, a)
        self.assertTrue(a.endswith("Spans:\n\n"))

    def test_stable_glossary_ignores_identity_maps(self) -> None:
        glossary = Glossary(
            terms=[
                Term("Jindan", "Golden Core"),
                Term("Golden Core", "Golden Core"),
                Term("师兄", "Senior Brother"),
            ]
        )
        block = glossary.as_stable_prompt()
        self.assertIn("Jindan → Golden Core", block)
        self.assertIn("师兄 → Senior Brother", block)
        self.assertNotIn("Golden Core → Golden Core", block)
        self.assertEqual(block, glossary.as_stable_prompt())

    def test_span_system_puts_glossary_in_system(self) -> None:
        block = "Glossary (use these exact renderings):\n- Jindan → Golden Core"
        system = span_system_prompt(block, "")
        self.assertTrue(system.startswith(SPAN_POLISH_SYSTEM))
        self.assertIn("Jindan", system)


class ServeHintTests(unittest.TestCase):
    def test_cuda_hints_include_prefix_cache_and_ngram(self) -> None:
        profile = DeviceProfile(
            name="RTX 4070",
            backend="cuda",
            vram_mb=12282,
            ram_mb=32000,
            max_params_b=14.0,
            num_ctx=4096,
            max_chars=1800,
            workers=1,
            skip_mode="auto",
            notes=[],
        )
        hints = dict(recommended_serve_commands(profile))
        self.assertIn("vLLM", hints)
        self.assertIn("llama.cpp", hints)
        self.assertIn("--enable-prefix-caching", hints["vLLM"])
        self.assertIn("ngram", hints["vLLM"])
        self.assertIn("--max-model-len 4096", hints["vLLM"])
        self.assertIn("0.90", hints["vLLM"])
        self.assertIn("--cache-prompt", hints["llama.cpp"])
        self.assertIn("--spec-type ngram-simple", hints["llama.cpp"])
        self.assertIn("--parallel 1", hints["llama.cpp"])


class LlamaCppPayloadTests(unittest.TestCase):
    def test_replace_completion_sends_cache_and_ngram(self) -> None:
        from copydecode.engine import EngineInfo, LLMEngine

        captured: dict = {}
        engine = LLMEngine(
            EngineInfo("llamacpp", "http://127.0.0.1:8080", "llama.cpp"),
            model="qwen",
        )

        def fake_stream(url, payload, on_token):
            captured.clear()
            captured.update(payload)
            return "[1]\nok"

        engine._stream_llamacpp = fake_stream  # type: ignore[method-assign]
        engine.generate("REPLACE: hi", system="sys", max_tokens=32, speculate=True, n_keep=40)
        self.assertTrue(captured["cache_prompt"])
        self.assertEqual(captured["speculative.n_max"], 5)
        self.assertEqual(captured["n_keep"], 40)
        self.assertEqual(captured["repeat_penalty"], 1.0)
        captured.clear()
        engine.generate("glossary json", system="sys", max_tokens=32, speculate=False)
        self.assertTrue(captured["cache_prompt"])
        self.assertNotIn("speculative.n_max", captured)
        self.assertEqual(captured["repeat_penalty"], 1.08)
        engine.close()


if __name__ == "__main__":
    unittest.main()
