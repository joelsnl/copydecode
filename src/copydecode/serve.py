from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from copydecode.engine import EngineError, classify_host
from copydecode.hardware import DeviceProfile, cuda_driver_major, estimate_params_b
from copydecode.paths import cache_dir, env_value

GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
USER_AGENT = "copydecode"
DEFAULT_PORT = 8080
DEFAULT_HOST = "http://127.0.0.1:8080"

# Official Qwen2.5 Instruct Q4_K_M GGUFs. Size picks follow hardware caps.
HF_GGUF = {
    14: (
        "qwen2.5:14b",
        "qwen2.5-14b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf",
    ),
    7: (
        "qwen2.5:7b",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
    ),
    3: (
        "qwen2.5:3b",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
}

Log = Callable[[str], None]


def _noop_log(message: str) -> None:
    del message


class ServerLog:
    """Keep llama-server stdout in memory and forward errors to `log` (no log file)."""

    _IMPORTANT = re.compile(
        r"\b(error|fatal|oom|out of memory|failed|unable to|exception)\b",
        re.I,
    )

    def __init__(self, log: Log, maxlen: int = 80) -> None:
        self._log = log
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def consume(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text:
            return
        with self._lock:
            self._lines.append(text)
        if self._IMPORTANT.search(text):
            self._log(f"llama.cpp: {text}")

    def tail(self, limit: int = 4000) -> str:
        with self._lock:
            text = "\n".join(self._lines)
        return text[-limit:].strip()

    def attach(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        threading.Thread(
            target=self._pump,
            args=(stream,),
            name="llama-server-log",
            daemon=True,
        ).start()

    def _pump(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self.consume(line)
        except (OSError, ValueError):
            return


def pid_path() -> Path:
    return cache_dir() / "llama-server.pid"


def gguf_choice(profile: DeviceProfile) -> tuple[str, str, str]:
    if profile.max_params_b >= 12:
        return HF_GGUF[14]
    if profile.max_params_b >= 7:
        return HF_GGUF[7]
    return HF_GGUF[3]


def ollama_models_root() -> Path:
    env = os.environ.get("OLLAMA_MODELS", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def parse_ollama_from(modelfile: str) -> Path | None:
    match = re.search(r"^FROM\s+(\S+)", modelfile, re.I | re.M)
    if not match:
        return None
    path = Path(match.group(1).strip().strip('"'))
    return path if path.is_file() else None


def ollama_blob_from_manifest(tag: str) -> Path | None:
    if ":" not in tag:
        tag = f"{tag}:latest"
    name, version = tag.split(":", 1)
    manifest = ollama_models_root() / "manifests" / "registry.ollama.ai" / "library" / name / version
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    digest = ""
    for layer in data.get("layers") or []:
        media = str(layer.get("mediaType") or "")
        if "model" in media or not digest:
            digest = str(layer.get("digest") or "")
            if "model" in media:
                break
    if not digest.startswith("sha256:"):
        return None
    blob = ollama_models_root() / "blobs" / digest.replace(":", "-")
    return blob if blob.is_file() else None


def find_ollama_blob(tag: str) -> Path | None:
    """Read the GGUF from Ollama's disk cache. Never call `ollama.exe` — that relaunches the app."""
    return ollama_blob_from_manifest(tag)


def find_local_gguf(filename: str) -> Path | None:
    env = env_value("COPYDECODE_GGUF", "NOVELPOLISHER_GGUF")
    if env:
        path = Path(env)
        if path.is_file():
            return path
    cached = cache_dir() / "models" / filename
    if cached.is_file() and cached.stat().st_size > 1_000_000:
        return cached
    cwd = Path.cwd() / filename
    if cwd.is_file():
        return cwd
    return None


def resolve_gguf(profile: DeviceProfile, *, download: bool, log: Log = _noop_log) -> tuple[Path, str]:
    alias, filename, url = gguf_choice(profile)
    log(f"Looking for {alias} GGUF (disk cache, then Ollama blobs — no ollama.exe)…")
    local = find_local_gguf(filename)
    if local:
        log(f"Using GGUF {local}")
        return local, alias
    blob = find_ollama_blob(alias)
    if blob and estimate_params_b(alias) <= profile.max_params_b + 0.2:
        log(f"Reusing Ollama blob for {alias}: {blob}")
        return blob, alias
    if not download:
        raise EngineError(
            f"No GGUF for {alias}. Place {filename} in {cache_dir() / 'models'} "
            "or set COPYDECODE_GGUF, or run without --no-download."
        )
    dest = cache_dir() / "models" / filename
    log(f"Downloading {filename} from Hugging Face…")
    download_file(url, dest, log=log)
    return dest, alias


def binary_preferences(profile: DeviceProfile) -> list[str]:
    """Preferred GitHub asset suffixes, first match wins."""
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in {"arm64", "aarch64"}
    if system == "Darwin":
        return ["bin-macos-arm64.tar.gz"] if arm else ["bin-macos-x64.tar.gz"]
    if system == "Windows":
        if profile.backend == "cuda" or profile.vendor == "nvidia":
            if cuda_driver_major() >= 13:
                return [
                    "bin-win-cuda-13.3-x64.zip",
                    "bin-win-cuda-12.4-x64.zip",
                    "bin-win-vulkan-x64.zip",
                ]
            return [
                "bin-win-cuda-12.4-x64.zip",
                "bin-win-cuda-13.3-x64.zip",
                "bin-win-vulkan-x64.zip",
            ]
        if profile.backend == "vulkan" or profile.vendor == "amd":
            return ["bin-win-vulkan-x64.zip", "bin-win-hip-radeon-x64.zip", "bin-win-cpu-x64.zip"]
        return ["bin-win-cpu-x64.zip"]
    arch = "arm64" if arm else "x64"
    if profile.vendor == "amd" or profile.backend == "vulkan":
        prefs = [f"bin-ubuntu-vulkan-{arch}.tar.gz"]
        if not arm:
            prefs.append("bin-ubuntu-rocm-7.2-x64.tar.gz")
        prefs.append(f"bin-ubuntu-{arch}.tar.gz")
        return prefs
    if profile.backend == "cuda":
        return [
            f"bin-ubuntu-cuda-{arch}.tar.gz",
            f"bin-ubuntu-vulkan-{arch}.tar.gz",
            f"bin-ubuntu-{arch}.tar.gz",
        ]
    return [f"bin-ubuntu-{arch}.tar.gz"]


def pick_release_asset(asset_names: list[str], preferences: list[str]) -> str | None:
    usable = [name for name in asset_names if not name.startswith("cudart-")]
    for pref in preferences:
        for name in usable:
            if name.endswith(pref):
                return name
    return None


def cudart_asset_for(binary_name: str) -> str | None:
    match = re.search(r"cuda-(\d+\.\d+)-x64\.zip$", binary_name)
    if not match:
        return None
    return f"cudart-llama-bin-win-cuda-{match.group(1)}-x64.zip"


def find_llama_server(root: Path | None = None) -> Path | None:
    env = env_value("COPYDECODE_LLAMA_SERVER", "NOVELPOLISHER_LLAMA_SERVER")
    if env:
        path = Path(env)
        if path.is_file():
            return path
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    names = ("llama-server.exe", "llama-server")
    search_roots = [root] if root else [cache_dir() / "llama-server"]
    for base in search_roots:
        if base is None or not base.exists():
            continue
        for name in names:
            hits = sorted(base.rglob(name), key=lambda p: len(p.parts))
            if hits:
                return hits[0]
    return None


def _http_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/octet-stream"}


def github_latest_release(log: Log = _noop_log) -> dict:
    cache = cache_dir() / "llama.cpp-release.json"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(GITHUB_RELEASES)
            response.raise_for_status()
            data = response.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except httpx.HTTPError as exc:
        if cache.is_file():
            log(f"GitHub unreachable ({exc}); using cached release metadata.")
            return json.loads(cache.read_text(encoding="utf-8"))
        raise EngineError(f"Could not list llama.cpp releases: {exc}") from exc


def download_file(url: str, dest: Path, log: Log = _noop_log) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    with httpx.Client(timeout=None, follow_redirects=True, headers=_http_headers()) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            console = Console()
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(dest.name, total=total or None)
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 256):
                        handle.write(chunk)
                        progress.advance(task, len(chunk))
    tmp.replace(dest)
    log(f"Saved {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(archive) as tf:
        tf.extractall(dest)


def _make_executable(path: Path) -> None:
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_llama_server(profile: DeviceProfile, *, download: bool, log: Log = _noop_log) -> Path:
    existing = find_llama_server()
    if existing:
        log(f"Using llama-server at {existing}")
        return existing
    if not download:
        raise EngineError(
            "llama-server not found. Install llama.cpp or run without --no-download "
            "so copydecode can fetch the matching GitHub build."
        )
    release = github_latest_release(log=log)
    assets = {item["name"]: item["browser_download_url"] for item in release.get("assets") or [] if "name" in item}
    chosen = pick_release_asset(list(assets), binary_preferences(profile))
    if not chosen:
        raise EngineError(
            "No llama.cpp build matches this machine. Preferences: "
            + ", ".join(binary_preferences(profile))
        )
    tag = str(release.get("tag_name") or "latest")
    out_dir = cache_dir() / "llama-server" / f"{tag}-{Path(chosen).stem}"
    exe = find_llama_server(out_dir)
    if exe:
        log(f"Using cached {chosen} ({tag})")
        return exe
    log(f"Downloading llama.cpp {tag} / {chosen}")
    archive = cache_dir() / "downloads" / chosen
    download_file(assets[chosen], archive, log=log)
    _extract(archive, out_dir)
    extra = cudart_asset_for(chosen)
    if extra and extra in assets:
        log(f"Downloading CUDA runtime {extra}")
        rt = cache_dir() / "downloads" / extra
        download_file(assets[extra], rt, log=log)
        _extract(rt, out_dir)
    exe = find_llama_server(out_dir)
    if not exe:
        raise EngineError(f"Unpacked {chosen} but llama-server was not inside it.")
    _make_executable(exe)
    return exe


# Do not run `llama-server -h`: CUDA builds often initialize the GPU just to print help.
DEFAULT_SERVER_HELP = "--alias --cache-prompt --flash-attn --cont-batching --spec-type"


def probe_help(exe: Path) -> str:
    del exe
    return DEFAULT_SERVER_HELP


def build_server_args(
    exe: Path,
    gguf: Path,
    profile: DeviceProfile,
    *,
    alias: str,
    port: int = DEFAULT_PORT,
    help_text: str = DEFAULT_SERVER_HELP,
) -> list[str]:
    help_text = help_text or DEFAULT_SERVER_HELP
    ngl = 0 if profile.backend == "cpu" else 99
    parallel = 1 if profile.max_params_b >= 12 or profile.backend != "cuda" else max(1, profile.workers)
    args = [
        str(exe),
        "-m",
        str(gguf),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--ctx-size",
        str(profile.num_ctx),
        "--n-gpu-layers",
        str(ngl),
        "--parallel",
        str(parallel),
        "--batch-size",
        "512",
        "--ubatch-size",
        "256",
    ]
    if "--alias" in help_text or "-a" in help_text:
        args.extend(["--alias", alias])
    if "--cache-prompt" in help_text:
        args.append("--cache-prompt")
    if profile.backend in {"cuda", "metal"} and "--flash-attn" in help_text:
        args.extend(["--flash-attn", "on"])
    if "--cont-batching" in help_text:
        args.append("--cont-batching")
    if "--spec-type" in help_text:
        args.extend(["--spec-type", "ngram-simple"])
    return args


def ollama_is_up() -> bool:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=0.5)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _tail_text(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return data[-limit:].strip()


def _llamacpp_ready(host: str) -> bool:
    base = host.rstrip("/")
    for path in ("/health", "/v1/models"):
        try:
            response = httpx.get(base + path, timeout=0.4)
        except httpx.HTTPError:
            continue
        if response.status_code == 200:
            return True
    return False


def _crash_tail(
    capture: ServerLog | None = None,
    log_file: Path | None = None,
) -> str:
    if capture is not None:
        return capture.tail()
    if log_file is not None:
        return _tail_text(log_file)
    return ""


def wait_healthy(
    host: str = DEFAULT_HOST,
    timeout: float = 180.0,
    *,
    proc: subprocess.Popen | None = None,
    capture: ServerLog | None = None,
    log_file: Path | None = None,
    log: Log = _noop_log,
) -> None:
    deadline = time.time() + timeout
    last_print = -10.0
    started = time.time()
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _crash_tail(capture, log_file)
            hint = f" See {log_file}" if log_file and not tail else ""
            raise EngineError(
                f"llama-server exited with code {proc.returncode} before becoming ready."
                + (f"\n{tail}" if tail else hint)
            )
        if _llamacpp_ready(host):
            return
        elapsed = time.time() - started
        if elapsed - last_print >= 5:
            log(
                f"Waiting for llama.cpp at {host} ({elapsed:.0f}s). "
                "First load of a 14B onto the GPU often takes 20–60s."
            )
            last_print = elapsed
        time.sleep(1.0)
    tail = _crash_tail(capture, log_file)
    raise EngineError(
        f"llama-server did not become ready at {host} after {timeout:.0f}s."
        + (f"\n{tail}" if tail else "")
    )


def server_running(host: str = DEFAULT_HOST) -> bool:
    try:
        return classify_host(host).kind == "llamacpp"
    except EngineError:
        return False


def stop_server(log: Log = _noop_log) -> bool:
    path = pid_path()
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        path.unlink(missing_ok=True)
        return False
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
        else:
            os.kill(pid, 15)
        log(f"Stopped llama-server (pid {pid})")
    except OSError as exc:
        log(f"Could not stop pid {pid}: {exc}")
        return False
    path.unlink(missing_ok=True)
    return True


@dataclass
class LlamaHandle:
    host: str
    alias: str
    gguf: Path
    exe: Path
    proc: subprocess.Popen | None
    args: list[str]


def _windows_creationflags(detach: bool) -> int:
    # CREATE_NO_WINDOW keeps CUDA init working. DETACHED_PROCESS often never
    # creates a GPU context, so the server looks "stuck" with no console output.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    if detach:
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return flags


def _popen_server(args: list[str], cwd: Path, *, detach: bool, stdout) -> subprocess.Popen:
    creationflags = _windows_creationflags(detach) if platform.system() == "Windows" else 0
    kwargs: dict = {
        "args": args,
        "cwd": str(cwd),
        "stdout": stdout,
        "stderr": subprocess.STDOUT,
        "creationflags": creationflags,
        "start_new_session": detach and platform.system() != "Windows",
    }
    if stdout is subprocess.PIPE:
        kwargs["text"] = True
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
        kwargs["bufsize"] = 1
    return subprocess.Popen(**kwargs)


def start_llama_server(
    profile: DeviceProfile,
    *,
    port: int = DEFAULT_PORT,
    download: bool = True,
    detach: bool = False,
    log: Log = _noop_log,
) -> LlamaHandle:
    host = f"http://127.0.0.1:{port}"
    if server_running(host):
        log(f"llama.cpp already listening on {host}")
        alias, filename, _url = gguf_choice(profile)
        gguf = find_local_gguf(filename) or Path(filename)
        exe = find_llama_server() or Path("llama-server")
        return LlamaHandle(host=host, alias=alias, gguf=gguf, exe=exe, proc=None, args=[])
    if ollama_is_up():
        raise EngineError(
            "Ollama is still answering on http://127.0.0.1:11434 and will keep the GPU. "
            "Quit it from the tray icon (Quit Ollama), not by closing the window, then retry. "
            "This tool reads the GGUF from disk and never runs ollama.exe."
        )

    exe = install_llama_server(profile, download=download, log=log)
    gguf, alias = resolve_gguf(profile, download=download, log=log)
    args = build_server_args(exe, gguf, profile, alias=alias, port=port)
    stale_log = cache_dir() / "llama-server.log"
    stale_log.unlink(missing_ok=True)
    log("Starting llama-server")
    log("Starting: " + " ".join(args))
    if detach:
        stdout: object = subprocess.DEVNULL
        capture = None
    else:
        stdout = subprocess.PIPE
        capture = ServerLog(log)
    proc = _popen_server(args, exe.parent, detach=detach, stdout=stdout)
    if capture is not None:
        capture.attach(proc)
    pid_path().write_text(str(proc.pid), encoding="utf-8")
    try:
        wait_healthy(host, proc=proc, capture=capture, log=log)
    except EngineError:
        if proc.poll() is None:
            raise
        stripped = [
            a
            for a in args
            if a not in {"--flash-attn", "on", "--spec-type", "ngram-simple", "--cont-batching"}
        ]
        if stripped == args:
            raise
        log("Retrying llama-server without optional flags…")
        proc = _popen_server(stripped, exe.parent, detach=detach, stdout=stdout)
        if capture is not None:
            capture.attach(proc)
        pid_path().write_text(str(proc.pid), encoding="utf-8")
        args = stripped
        wait_healthy(host, proc=proc, capture=capture, log=log)
    log(f"llama.cpp ready at {host} ({alias})")
    return LlamaHandle(host=host, alias=alias, gguf=gguf, exe=exe, proc=proc, args=args)


def plan_serve(profile: DeviceProfile) -> dict[str, str]:
    alias, filename, url = gguf_choice(profile)
    blob = find_ollama_blob(alias)
    local = find_local_gguf(filename)
    exe = find_llama_server()
    return {
        "os": f"{platform.system()} {platform.machine()}",
        "device": profile.name,
        "backend": profile.backend,
        "vendor": profile.vendor,
        "binary": ", ".join(binary_preferences(profile)[:3]),
        "llama-server": str(exe) if exe else "(will download from GitHub)",
        "gguf": str(local or blob or f"(will download {filename})"),
        "hf": url,
        "alias": alias,
        "cache": str(cache_dir()),
    }
