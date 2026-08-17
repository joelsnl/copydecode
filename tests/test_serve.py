from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from copydecode.engine import EngineError
from copydecode.hardware import DeviceProfile
from copydecode.serve import (
    binary_preferences,
    build_server_args,
    cudart_asset_for,
    find_ollama_blob,
    gguf_choice,
    ollama_blob_from_manifest,
    parse_ollama_from,
    pick_release_asset,
    start_llama_server,
    wait_healthy,
    ServerLog,
)


def _profile(**kwargs) -> DeviceProfile:
    data = dict(
        name="GPU",
        backend="cuda",
        vram_mb=12282,
        ram_mb=32000,
        max_params_b=14.0,
        num_ctx=4096,
        max_chars=1800,
        workers=1,
        skip_mode="auto",
        notes=[],
        vendor="nvidia",
    )
    data.update(kwargs)
    return DeviceProfile(**data)


class AssetPickTests(unittest.TestCase):
    def test_picks_first_matching_suffix(self) -> None:
        names = [
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "llama-b1-bin-win-cpu-x64.zip",
            "llama-b1-bin-win-cuda-12.4-x64.zip",
            "llama-b1-bin-win-vulkan-x64.zip",
        ]
        self.assertEqual(
            pick_release_asset(names, ["bin-win-cuda-12.4-x64.zip", "bin-win-vulkan-x64.zip"]),
            "llama-b1-bin-win-cuda-12.4-x64.zip",
        )

    def test_skips_cudart_named_like_cuda(self) -> None:
        names = ["cudart-llama-bin-win-cuda-12.4-x64.zip", "llama-b1-bin-win-vulkan-x64.zip"]
        self.assertEqual(
            pick_release_asset(names, ["bin-win-cuda-12.4-x64.zip", "bin-win-vulkan-x64.zip"]),
            "llama-b1-bin-win-vulkan-x64.zip",
        )

    def test_cudart_pair(self) -> None:
        self.assertEqual(
            cudart_asset_for("llama-b1-bin-win-cuda-12.4-x64.zip"),
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        )
        self.assertIsNone(cudart_asset_for("llama-b1-bin-win-vulkan-x64.zip"))


class BinaryPreferenceTests(unittest.TestCase):
    @patch("copydecode.serve.cuda_driver_major", return_value=12)
    @patch("copydecode.serve.platform.machine", return_value="AMD64")
    @patch("copydecode.serve.platform.system", return_value="Windows")
    def test_windows_nvidia_cuda(self, *_args) -> None:
        prefs = binary_preferences(_profile(backend="cuda", vendor="nvidia"))
        self.assertEqual(prefs[0], "bin-win-cuda-12.4-x64.zip")

    @patch("copydecode.serve.platform.machine", return_value="AMD64")
    @patch("copydecode.serve.platform.system", return_value="Windows")
    def test_windows_amd_vulkan(self, *_args) -> None:
        prefs = binary_preferences(_profile(backend="vulkan", vendor="amd", name="RX 7800 XT"))
        self.assertEqual(prefs[0], "bin-win-vulkan-x64.zip")

    @patch("copydecode.serve.platform.machine", return_value="arm64")
    @patch("copydecode.serve.platform.system", return_value="Darwin")
    def test_macos_metal(self, *_args) -> None:
        prefs = binary_preferences(_profile(backend="metal", vendor="apple"))
        self.assertEqual(prefs[0], "bin-macos-arm64.tar.gz")

    @patch("copydecode.serve.platform.machine", return_value="x86_64")
    @patch("copydecode.serve.platform.system", return_value="Linux")
    def test_linux_nvidia_falls_back_to_vulkan(self, *_args) -> None:
        prefs = binary_preferences(_profile(backend="cuda", vendor="nvidia"))
        self.assertIn("bin-ubuntu-vulkan-x64.tar.gz", prefs)
        self.assertIn("bin-ubuntu-cuda-x64.tar.gz", prefs)


class GuffAndOllamaTests(unittest.TestCase):
    def test_gguf_follows_vram_cap(self) -> None:
        alias, filename, url = gguf_choice(_profile(max_params_b=14.0))
        self.assertEqual(alias, "qwen2.5:14b")
        self.assertIn("14b", filename)
        self.assertTrue(url.startswith("https://huggingface.co/"))
        alias7, _, _ = gguf_choice(_profile(max_params_b=7.0))
        self.assertEqual(alias7, "qwen2.5:7b")

    def test_parse_ollama_from(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = Path(tmp.name)
        try:
            text = f'FROM {path}\nTEMPLATE """hi"""\n'
            self.assertEqual(parse_ollama_from(text), path)
        finally:
            path.unlink(missing_ok=True)

    def test_manifest_blob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / "blobs" / "sha256-abc"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"gguf")
            manifest = root / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "14b"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:abc",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch("copydecode.serve.ollama_models_root", return_value=root):
                self.assertEqual(ollama_blob_from_manifest("qwen2.5:14b"), blob)
                self.assertEqual(find_ollama_blob("qwen2.5:14b"), blob)

    @patch("copydecode.serve.subprocess.check_output")
    def test_find_ollama_blob_does_not_run_cli(self, mock_co) -> None:
        find_ollama_blob("qwen2.5:14b")
        mock_co.assert_not_called()


class ServerArgTests(unittest.TestCase):
    def test_gpu_gets_ngl_and_flash(self) -> None:
        args = build_server_args(
            Path("llama-server"),
            Path("m.gguf"),
            _profile(),
            alias="qwen2.5:14b",
            help_text="--alias --cache-prompt --flash-attn --cont-batching --spec-type",
        )
        self.assertIn("--n-gpu-layers", args)
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "99")
        self.assertIn("--cache-prompt", args)
        self.assertIn("--flash-attn", args)
        self.assertIn("ngram-simple", args)
        self.assertEqual(args[args.index("--parallel") + 1], "1")

    def test_cpu_offloads_nothing(self) -> None:
        args = build_server_args(
            Path("llama-server"),
            Path("m.gguf"),
            _profile(backend="cpu", vendor="none", max_params_b=3.0),
            alias="qwen2.5:3b",
            help_text="--cache-prompt",
        )
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "0")
        self.assertNotIn("--flash-attn", args)


class WaitAndStartTests(unittest.TestCase):
    def test_wait_healthy_fails_immediately_if_process_exits(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.txt"
            log.write_text("failed to allocate CUDA buffer", encoding="utf-8")
            with self.assertRaises(EngineError) as ctx:
                wait_healthy("http://127.0.0.1:9", timeout=30, proc=proc, log_file=log)
        self.assertIn("exited", str(ctx.exception))
        self.assertIn("CUDA buffer", str(ctx.exception))

    def test_wait_healthy_uses_in_memory_capture(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        notes: list[str] = []
        capture = ServerLog(notes.append)
        capture.consume("ggml_cuda_init: failed to allocate CUDA buffer")
        with self.assertRaises(EngineError) as ctx:
            wait_healthy("http://127.0.0.1:9", timeout=30, proc=proc, capture=capture)
        self.assertIn("CUDA buffer", str(ctx.exception))
        self.assertTrue(any("llama.cpp:" in line for line in notes))

    def test_server_log_ignores_noise(self) -> None:
        notes: list[str] = []
        capture = ServerLog(notes.append)
        capture.consume("slot 0 prompt eval time = 12.3 ms")
        self.assertEqual(notes, [])
        self.assertIn("prompt eval", capture.tail())

    @patch("copydecode.serve.server_running", return_value=False)
    @patch("copydecode.serve.ollama_is_up", return_value=True)
    def test_start_refuses_when_ollama_holds_gpu(self, *_args) -> None:
        with self.assertRaises(EngineError) as ctx:
            start_llama_server(_profile(), download=False)
        self.assertIn("11434", str(ctx.exception))
        self.assertIn("ollama.exe", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
