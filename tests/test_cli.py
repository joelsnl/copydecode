from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from copydecode.cli import main
from copydecode.engine import EngineError
from copydecode.hardware import DeviceProfile


def _cpu_profile() -> DeviceProfile:
    return DeviceProfile(
        name="test-cpu",
        backend="cpu",
        vram_mb=0,
        ram_mb=16000,
        max_params_b=7.0,
        num_ctx=4096,
        max_chars=1800,
        workers=1,
        skip_mode="auto",
        notes=[],
    )


class ArgRoutingTests(unittest.TestCase):
    def test_bare_file_routes_to_run(self) -> None:
        # Fails on the missing file, not on argument parsing.
        self.assertEqual(main(["definitely-missing-file.txt"]), 1)

    def test_leading_flag_still_routes_to_run(self) -> None:
        self.assertEqual(main(["-o", "out.txt", "definitely-missing-file.txt"]), 1)

    def test_no_args_prints_help(self) -> None:
        self.assertEqual(main([]), 0)


class DryRunTests(unittest.TestCase):
    def test_dry_run_is_offline_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sample.txt"
            src.write_text(
                "He could not help but smile.\n\nThe valley was quiet after the storm.\n",
                encoding="utf-8",
            )
            with (
                patch("copydecode.cli.detect_device", return_value=_cpu_profile()),
                patch("copydecode.cli.discover_engine", side_effect=EngineError("no server")),
                patch("copydecode.cli.start_llama_server") as start,
            ):
                code = main(["run", str(src), "--dry-run", "--no-token-pack"])
            self.assertEqual(code, 0)
            start.assert_not_called()
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["sample.txt"])

    def test_dry_run_with_flag_first_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sample.txt"
            src.write_text("He could not help but smile.\n", encoding="utf-8")
            out = Path(tmp) / "out.txt"
            with (
                patch("copydecode.cli.detect_device", return_value=_cpu_profile()),
                patch("copydecode.cli.discover_engine", side_effect=EngineError("no server")),
            ):
                code = main(["--dry-run", "-o", str(out), str(src), "--no-token-pack"])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
