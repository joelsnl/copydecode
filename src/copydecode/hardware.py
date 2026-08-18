from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class DeviceProfile:
    name: str
    backend: str  # cuda | vulkan | metal | cpu
    vram_mb: int
    ram_mb: int
    max_params_b: float
    num_ctx: int
    max_chars: int
    workers: int
    skip_mode: str  # auto | aggressive
    notes: list[str]
    vendor: str = "none"  # nvidia | amd | apple | none


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    try:
        return subprocess.check_output(cmd, timeout=timeout, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def ram_mb() -> int:
    system = platform.system()
    if system == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys // (1024 * 1024))
        return 8192
    if system == "Darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        if raw:
            return int(raw.strip()) // (1024 * 1024)
        return 8192
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 8192


def nvidia_gpus() -> list[tuple[str, int, int]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    raw = _run(
        [
            exe,
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return []
    gpus: list[tuple[str, int, int]] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append((parts[0], int(float(parts[1])), int(float(parts[2]))))
        except ValueError:
            continue
    return gpus


def cuda_driver_major() -> int:
    raw = _run(["nvidia-smi"])
    if not raw:
        return 12
    match = re.search(r"CUDA Version:\s*(\d+)", raw)
    if match:
        return int(match.group(1))
    return 12


def amd_gpus() -> list[tuple[str, int]]:
    """Return (name, vram_mb) for AMD GPUs. VRAM is best-effort; WMI often lies."""
    found: list[tuple[str, int]] = []
    rocm = shutil.which("rocm-smi")
    if rocm:
        raw = _run([rocm, "--showmeminfo", "vram"])
        if raw:
            name = "AMD GPU"
            vram = 0
            for line in raw.splitlines():
                if "Card series" in line or "GPU[" in line and "Name" in line:
                    bits = line.split(":")
                    if len(bits) > 1 and bits[-1].strip():
                        name = bits[-1].strip()
                nums = re.findall(r"(\d+)\s*MiB", line.replace("MB", "MiB"))
                if nums:
                    vram = max(vram, int(nums[-1]))
            if vram:
                found.append((name, vram))
                return found

    if platform.system() == "Windows":
        raw = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
            ]
        )
        if raw:
            for line in raw.splitlines():
                if "|" not in line:
                    continue
                name, ram_s = line.split("|", 1)
                name = name.strip()
                if not re.search(r"amd|radeon|rx\s*\d", name, re.I):
                    continue
                try:
                    vram = int(ram_s.strip()) // (1024 * 1024)
                except ValueError:
                    vram = 0
                found.append((name, vram))
            if found:
                return found

    if platform.system() == "Linux":
        raw = _run(["lspci"])
        if raw:
            for line in raw.splitlines():
                if re.search(r"VGA|3D|Display", line) and re.search(r"AMD|ATI|Radeon", line, re.I):
                    name = line.split(":", 2)[-1].strip() if ":" in line else line.strip()
                    found.append((name, 0))
    return found


def _gpu_size_caps(vram_mb: int, notes: list[str]) -> tuple[float, int, int, int, str]:
    usable = max(vram_mb - 900, 2048)
    if usable >= 20000:
        return 32.0, 8192, 2400, 1, "auto"
    if usable >= 10500:
        notes.append(
            f"{vram_mb} MB VRAM. 14B Q4 fits with a 4k context; "
            "7B can run 2 chapter workers. 32B will offload to RAM and crawl."
        )
        return 14.0, 4096, 1800, 2, "auto"
    if usable >= 7000:
        return 8.0, 4096, 1400, 1, "auto"
    if usable >= 4500:
        return 7.0, 2048, 1000, 1, "aggressive"
    notes.append("Low VRAM: 3B only, short chunks, skip clean sentences.")
    return 3.0, 2048, 800, 1, "aggressive"


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _amd_is_igpu(name: str) -> bool:
    return bool(re.search(r"radeon graphics|vega \d|graphics\s*$", name, re.I)) and not re.search(
        r"\b(RX|XT|Pro|Instinct|VII|W\d)\b", name, re.I
    )


def detect_device() -> DeviceProfile:
    notes: list[str] = []
    ram = ram_mb()
    gpus = nvidia_gpus()
    if gpus:
        name, vram, _free = max(gpus, key=lambda g: g[1])
        max_b, ctx, chars, workers, skip = _gpu_size_caps(vram, notes)
        if workers > 1:
            notes.append("Chapter-level parallelism is on; glossary is applied before any GPU work.")
        notes.append(
            "CUDA polish path: `copydecode serve` starts llama.cpp with prefix cache + ngram. "
            "Do not raise --num-ctx above this cap — KV has to share the remaining VRAM."
        )
        return DeviceProfile(
            name=name,
            backend="cuda",
            vram_mb=vram,
            ram_mb=ram,
            max_params_b=max_b,
            num_ctx=ctx,
            max_chars=chars,
            workers=workers,
            skip_mode=skip,
            notes=notes or [f"{name} · {vram} MB VRAM · {ram} MB RAM"],
            vendor="nvidia",
        )

    if is_apple_silicon():
        usable = int(ram * 0.48)
        if ram >= 24000:
            max_b, ctx, chars, workers, skip = 14.0, 4096, 1600, 1, "auto"
        elif ram >= 15000:
            max_b, ctx, chars, workers, skip = 7.0, 4096, 1200, 1, "auto"
            notes.append(
                "M-series 16 GB is shared with macOS. 7B Q4 is the ceiling for a comfortable "
                "full-document run; 14B will page. One worker only."
            )
        else:
            max_b, ctx, chars, workers, skip = 3.0, 2048, 800, 1, "aggressive"
            notes.append("8 GB Apple Silicon: 3B model, aggressive skip router.")
        return DeviceProfile(
            name=f"Apple Silicon ({platform.processor() or 'arm64'})",
            backend="metal",
            vram_mb=usable,
            ram_mb=ram,
            max_params_b=max_b,
            num_ctx=ctx,
            max_chars=chars,
            workers=workers,
            skip_mode=skip,
            notes=notes,
            vendor="apple",
        )

    amd = [item for item in amd_gpus() if not _amd_is_igpu(item[0])]
    if amd:
        name, vram = max(amd, key=lambda g: g[1])
        if vram < 3000:
            vram = 8192
            notes.append(
                f"{name}: VRAM was not reported accurately. Assuming 8 GB. "
                "If the card is larger, a 14B Q4 may still fit."
            )
        max_b, ctx, chars, workers, skip = _gpu_size_caps(vram, notes)
        notes.append(
            "AMD GPU: `copydecode serve` fetches the Vulkan llama.cpp build "
            "(HIP/ROCm zip is optional and heavier)."
        )
        return DeviceProfile(
            name=name,
            backend="vulkan",
            vram_mb=vram,
            ram_mb=ram,
            max_params_b=max_b,
            num_ctx=ctx,
            max_chars=chars,
            workers=1,
            skip_mode=skip,
            notes=notes,
            vendor="amd",
        )

    if ram >= 32000:
        max_b, ctx, chars, skip = 7.0, 2048, 1000, "aggressive"
        notes.append("CPU-only with lots of RAM. 7B will work and be slow; 3B is kinder.")
    else:
        max_b, ctx, chars, skip = 3.0, 2048, 700, "aggressive"
        notes.append("CPU-only. Cap at 3B, skip clean sentences, short context.")
    return DeviceProfile(
        name=f"{platform.system()} CPU",
        backend="cpu",
        vram_mb=0,
        ram_mb=ram,
        max_params_b=max_b,
        num_ctx=ctx,
        max_chars=chars,
        workers=1,
        skip_mode=skip,
        notes=notes,
        vendor="none",
    )


def estimate_params_b(model: str) -> float | None:
    """Parameter count in billions parsed from the model name, or None when unknown."""
    name = model.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
    if match:
        return float(match.group(1))
    if "e4b" in name or "e2b" in name:
        return 4.0
    if "e1b" in name:
        return 1.0
    return None


def is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return any(token in name for token in ("deepseek-r1", "qwq", "thinking", ":r1", "reasoner"))


def clamp_for_model(profile: DeviceProfile, model: str) -> DeviceProfile:
    params = estimate_params_b(model)
    notes = list(profile.notes)
    workers = profile.workers
    ctx = profile.num_ctx
    chars = profile.max_chars
    skip = profile.skip_mode
    gpu = profile.backend in {"cuda", "vulkan"}
    if params is None:
        notes.append(
            f"Could not infer the size of {model} from its name; keeping one worker "
            "and the hardware defaults."
        )
        workers = 1
    elif params > profile.max_params_b + 0.1:
        notes.append(
            f"{model} is about {params:g}B; this machine is sized for {profile.max_params_b:g}B. "
            "Expect RAM offload. Forcing 1 worker, shorter context, aggressive skip."
        )
        workers = 1
        ctx = min(ctx, 2048)
        chars = min(chars, 900)
        skip = "aggressive"
    elif params >= 12 and gpu and profile.vram_mb < 16000:
        workers = 1
        ctx = min(ctx, 4096)
        chars = min(chars, 1800)
    elif params <= 8 and gpu and profile.vram_mb >= 10500:
        workers = max(workers, 2)
    if is_reasoning_model(model):
        notes.append(
            "Reasoning models spend most tokens thinking. The polisher strips that, "
            "but the GPU still pays for it. Prefer Qwen2.5 7B/14B for this task."
        )
        workers = 1
        skip = "aggressive"
    return replace(
        profile,
        num_ctx=ctx,
        max_chars=chars,
        workers=workers,
        skip_mode=skip,
        notes=notes,
    )


def recommended_serve_commands(profile: DeviceProfile) -> list[tuple[str, str]]:
    """Exact vLLM / llama.cpp flags for this machine (throughput blueprint)."""
    ctx = profile.num_ctx
    if profile.max_params_b >= 12:
        gguf = "qwen2.5-14b-instruct-q4_k_m.gguf"
        vllm_model = "Qwen/Qwen2.5-14B-Instruct-AWQ"
        parallel = 1
        mem = "0.90"
    elif profile.max_params_b >= 7:
        gguf = "qwen2.5-7b-instruct-q4_k_m.gguf"
        vllm_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"
        parallel = 1 if profile.backend != "cuda" else 2
        mem = "0.88"
    else:
        gguf = "qwen2.5-3b-instruct-q4_k_m.gguf"
        vllm_model = "Qwen/Qwen2.5-3B-Instruct-AWQ"
        parallel = 1
        mem = "0.85"
    spec = '{"method":"ngram","num_speculative_tokens":5}'
    ngl = 0 if profile.backend == "cpu" else 99
    llama = (
        f"llama-server -m {gguf} --port 8080 --ctx-size {ctx} "
        f"--n-gpu-layers {ngl} --cache-prompt --parallel {parallel} "
        f"--batch-size 512 --ubatch-size 256 --cont-batching "
        f"--spec-type ngram-simple"
    )
    vllm = (
        f"vllm serve {vllm_model} --port 8000 --max-model-len {ctx} "
        f"--gpu-memory-utilization {mem} --enable-prefix-caching "
        f"--speculative-config '{spec}'"
    )
    auto = ("auto", "copydecode serve")
    if profile.backend == "cuda":
        return [auto, ("vLLM", vllm), ("llama.cpp", llama)]
    if profile.backend == "vulkan":
        return [auto, ("llama.cpp (Vulkan)", llama)]
    if profile.backend == "metal":
        metal = (
            f"llama-server -m {gguf} --port 8080 --ctx-size {ctx} "
            f"--n-gpu-layers 99 --flash-attn --cache-prompt --parallel 1 --spec-type ngram-simple"
        )
        return [auto, ("llama.cpp (Metal)", metal)]
    return [
        auto,
        (
            "llama.cpp (CPU)",
            f"llama-server -m {gguf} --port 8080 --ctx-size {ctx} --n-gpu-layers 0 --cache-prompt --parallel 1",
        ),
    ]
