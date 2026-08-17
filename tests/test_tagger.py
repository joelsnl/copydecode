from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from copydecode.prompts import parse_numbered
from copydecode.tagger import (
    LabeledSpan,
    examples_from_changelog,
    export_kd_pairs,
    leaky_model_text,
    reset_tagger_cache,
    synthetic_examples,
    train_tagger,
)
from copydecode.spans import tag_text


class ParseNumberedTests(unittest.TestCase):
    def test_strips_replace_labels(self) -> None:
        raw = "[1]\nREPLACE:\nHe smiled.\n\n[2]\nREPLACE:\nShe left."
        parsed = parse_numbered(raw, 2)
        self.assertEqual(parsed, ["He smiled.", "She left."])
        self.assertEqual(parse_numbered("[1]\nHe smiled. [1]", 1), ["He smiled."])
        self.assertEqual(
            parse_numbered(
                "[1]\nThis child is like a phantom in this world.\n[1]\nThis child is like a specter in this world.",
                1,
            ),
            ["This child is like a phantom in this world."],
        )

    def test_single_block_without_label(self) -> None:
        self.assertEqual(parse_numbered("He smiled despite himself.", 1), ["He smiled despite himself."])

    def test_rejects_wrong_count(self) -> None:
        self.assertIsNone(parse_numbered("[1]\nOnly one", 2))


class TaggerTrainTests(unittest.TestCase):
    def test_synthetic_model_keeps_gold_replace(self) -> None:
        tagger = train_tagger([], include_synthetic=True)
        self.assertGreaterEqual(tagger.replace_recall, 0.99)
        self.assertTrue(tagger.is_replace("He could not help but smile at the news."))
        self.assertFalse(tagger.is_replace("The mountain wind was cold against his face."))
        self.assertGreaterEqual(tagger.keep_rate, 0.8)

    def test_learned_tag_text_copies_clean_clause(self) -> None:
        tagger = train_tagger([], include_synthetic=True)
        reset_tagger_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "span_tagger.json"
            path.write_text(json.dumps(tagger.to_json()), encoding="utf-8")
            import os

            old = os.environ.get("COPYDECODE_TAGGER")
            os.environ["COPYDECODE_TAGGER"] = str(path)
            reset_tagger_cache()
            try:
                text = (
                    "The mountain wind was cold. "
                    "He could not help but smile. "
                    "Then he walked toward the sect gate."
                )
                program = tag_text(text, "polish", "auto", learned=True)
                self.assertEqual(program.stitched({}), text)
                self.assertIn("REPLACE", [span.kind for span in program.spans])
                self.assertIn("KEEP", [span.kind for span in program.spans])
            finally:
                if old is None:
                    os.environ.pop("COPYDECODE_TAGGER", None)
                else:
                    os.environ["COPYDECODE_TAGGER"] = old
                reset_tagger_cache()

    def test_changelog_examples_and_kd_export(self) -> None:
        payload = {
            "edits": [
                {
                    "before": "He could not help but smile.",
                    "after": "He couldn't help smiling.",
                },
                {
                    "before": "In the next moment he left.",
                    "after": "[1]\nREPLACE:\nHe left.",
                },
            ],
            "unchanged": [{"before": "The valley was quiet."}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.changes.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            examples = examples_from_changelog(path)
            self.assertTrue(any(ex.replace and "could not help" in ex.text for ex in examples))
            self.assertTrue(any(not ex.replace and "quiet" in ex.text for ex in examples))
            pairs = export_kd_pairs(path)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["target"], "He couldn't help smiling.")
        self.assertTrue(leaky_model_text("[1]\nREPLACE:\nHe left."))
    def test_anchors_keep_gold_replace_without_eating_clean(self) -> None:
        examples = [
            LabeledSpan("Luo Feng boarded the spaceship without another word.", True, "log"),
            LabeledSpan("The valley was quiet after the storm.", False, "log"),
        ]
        tagger = train_tagger(examples, include_synthetic=True)
        self.assertTrue(tagger.is_replace("Luo Feng boarded the spaceship without another word."))
        self.assertFalse(tagger.is_replace("The mountain wind was cold against his face."))
        self.assertTrue(tagger.is_replace("He could not help but smile at the news."))

    def test_merges_previous_anchors(self) -> None:
        previous = train_tagger(
            [LabeledSpan("Old gold rewrite that must stay replaceable.", True, "old")],
            include_synthetic=True,
        )
        payload = {
            "edits": [
                {
                    "before": "He could not help but smile.",
                    "after": "He couldn't help smiling.",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "book.changes.json"
            dest = Path(tmp) / "span_tagger.json"
            log.write_text(json.dumps(payload), encoding="utf-8")
            from unittest.mock import patch

            from copydecode.tagger import load_tagger_file, train_from_files

            with patch("copydecode.tagger.get_tagger", return_value=previous):
                tagger, path = train_from_files([log], dest=dest, merge_existing=True)
            loaded = load_tagger_file(path)
            self.assertTrue(tagger.anchor_hit("Old gold rewrite that must stay replaceable."))
            self.assertTrue(loaded.anchor_hit("He could not help but smile."))


if __name__ == "__main__":
    unittest.main()
