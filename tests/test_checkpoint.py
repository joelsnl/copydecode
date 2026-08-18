from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from copydecode.checkpoint import Checkpoint


class CheckpointTests(unittest.TestCase):
    def test_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            ckpt = Checkpoint(path, {"fingerprint": "fp1", "mode": "polish"})
            ckpt.save_chunk("c1", ["hello"])
            ckpt.save_chunk("c2", ["a", "b"])
            again = Checkpoint(path, {"fingerprint": "fp1", "mode": "polish"})
            self.assertTrue(again.done("c1"))
            self.assertEqual(again.get("c2"), ["a", "b"])

    def test_fingerprint_mismatch_discards_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            Checkpoint(path, {"fingerprint": "fp1"}).save_chunk("c1", ["hello"])
            other = Checkpoint(path, {"fingerprint": "fp2"})
            self.assertFalse(other.done("c1"))
            # first save under the new fingerprint replaces the old file
            other.save_chunk("c9", ["fresh"])
            reloaded = Checkpoint(path, {"fingerprint": "fp2"})
            self.assertTrue(reloaded.done("c9"))
            self.assertFalse(reloaded.done("c1"))

    def test_save_appends_one_line_per_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            ckpt = Checkpoint(path, {"fingerprint": "fp1"})
            for i in range(5):
                ckpt.save_chunk(f"c{i}", [str(i)])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1 + 5)

    def test_truncated_final_line_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            ckpt = Checkpoint(path, {"fingerprint": "fp1"})
            ckpt.save_chunk("c1", ["hello"])
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"id": "c2", "tex')  # crash mid-append
            recovered = Checkpoint(path, {"fingerprint": "fp1"})
            self.assertTrue(recovered.done("c1"))
            self.assertFalse(recovered.done("c2"))

    def test_memory_only_when_path_is_none(self) -> None:
        ckpt = Checkpoint(None, {"fingerprint": "fp1"})
        ckpt.save_chunk("c1", ["hello"])
        self.assertTrue(ckpt.done("c1"))
        self.assertEqual(ckpt.get("c1"), ["hello"])


if __name__ == "__main__":
    unittest.main()
