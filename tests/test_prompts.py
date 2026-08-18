from __future__ import annotations

import unittest

from copydecode.prompts import parse_numbered


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

    def test_out_of_order_labels_map_by_index(self) -> None:
        raw = "[2]\nSecond sentence rewritten.\n\n[1]\nFirst sentence rewritten."
        self.assertEqual(
            parse_numbered(raw, 2),
            ["First sentence rewritten.", "Second sentence rewritten."],
        )

    def test_duplicate_labels_do_not_fill_missing_slots(self) -> None:
        raw = "[1]\nFirst try.\n\n[1]\nSecond try."
        self.assertIsNone(parse_numbered(raw, 2))

    def test_labels_outside_range_are_ignored(self) -> None:
        raw = "[1]\nGood block.\n\n[7]\nStray block."
        self.assertIsNone(parse_numbered(raw, 2))

    def test_unlabeled_blocks_fall_back_to_position(self) -> None:
        raw = "First block text.\n\nSecond block text."
        self.assertEqual(parse_numbered(raw, 2), ["First block text.", "Second block text."])

    def test_partial_labels_never_guess_positionally(self) -> None:
        # One labeled and one bare block: guessing positions here is how text
        # ends up in the wrong paragraph, so the parse must fail (and retry).
        raw = "[2]\nLabeled block.\n\nBare block."
        self.assertIsNone(parse_numbered(raw, 2))


if __name__ == "__main__":
    unittest.main()
