"""HTTP client for local LLM servers (llama.cpp, vLLM, Ollama, OpenAI-compatible)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from copydecode.errors import CopydecodeError
from copydecode.hardware import (
    DeviceProfile,
    estimate_params_b,
    is_reasoning_model,
    recommended_serve_commands,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)

PROBE_CANDIDATES = (
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:11434",
)


class EngineError(CopydecodeError):
    """An LLM server could not be found, started, or spoken to."""


def strip_model_noise(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = THINKING_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    return text.strip()


@dataclass
class EngineInfo:
    kind: str  # ollama | llamacpp | vllm | openai
    host: str
    label: str


def _ok(url: str, timeout: float = 1.2) -> httpx.Response | None:
    try:
        return httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return None


def classify_host(host: str) -> EngineInfo:
    """Identify which server kind answers at ``host``.

    Probe order matters: vLLM also serves ``/health``, so ``/health`` alone
    cannot distinguish vLLM from llama.cpp. ``/api/tags`` is Ollama-only and
    ``/props`` is llama.cpp-only; anything else that speaks ``/v1/models`` is
    treated as an OpenAI-compatible server (vLLM when it says so).
    """
    base = host.rstrip("/")
    tags = _ok(base + "/api/tags")
    if tags is not None and tags.status_code == 200:
        return EngineInfo("ollama", base, "Ollama")
    props = _ok(base + "/props")
    if props is not None and props.status_code == 200:
        return EngineInfo("llamacpp", base, "llama.cpp")
    models = _ok(base + "/v1/models")
    if models is not None and models.status_code == 200:
        kind = "vllm" if "vllm" in models.text.lower() else "openai"
        return EngineInfo(kind, base, "vLLM" if kind == "vllm" else "OpenAI-compatible")
    raise EngineError(
        f"No LLM server at {base}. Start Ollama (`ollama serve`), llama-server, or vLLM."
    )


def discover_engine(preferred: str | None, host: str | None, profile: DeviceProfile) -> EngineInfo:
    """Find a running server, honoring an explicit host or a preferred kind."""
    if host:
        return classify_host(host)

    found: list[EngineInfo] = []
    for candidate in PROBE_CANDIDATES:
        try:
            found.append(classify_host(candidate))
        except EngineError:
            continue
    if not found:
        lines = [
            "No local LLM server found on ports 8000, 8080, or 11434.",
            "Run `copydecode serve` to fetch llama.cpp + GGUF for this GPU, or start Ollama/vLLM.",
        ]
        for label, cmd in recommended_serve_commands(profile):
            lines.append(f"{label}: {cmd}")
        raise EngineError("\n".join(lines))

    if preferred and preferred != "auto":
        for info in found:
            if info.kind == preferred or (preferred == "openai" and info.kind in {"vllm", "openai"}):
                return info
        raise EngineError(
            f"No {preferred} server found on ports 8000, 8080, or 11434. "
            "Run `copydecode serve` to download and start llama.cpp for this machine."
        )

    order = ["vllm", "llamacpp", "openai", "ollama"]
    if profile.backend != "cuda":
        order = ["llamacpp", "ollama", "openai", "vllm"]
    rank = {kind: i for i, kind in enumerate(order)}
    found.sort(key=lambda info: rank.get(info.kind, len(order)))
    return found[0]


def list_models(info: EngineInfo) -> list[str]:
    try:
        if info.kind == "ollama":
            data = httpx.get(info.host + "/api/tags", timeout=10.0).json()
            return [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        data = httpx.get(info.host + "/v1/models", timeout=10.0).json()
        names = []
        for item in data.get("data", []):
            name = item.get("id") or item.get("name")
            if name:
                names.append(str(name))
        return names
    except httpx.HTTPError as exc:
        raise EngineError(f"Could not list models at {info.host}: {exc}") from exc


def pick_model(names: list[str], max_params_b: float) -> str | None:
    """Largest non-reasoning model that fits the hardware cap.

    Models whose size cannot be inferred from their name are never auto-picked
    while a sized, fitting model exists.
    """
    fitting: list[tuple[float, str]] = []
    others: list[str] = []
    reasoning: list[str] = []
    for name in names:
        if is_reasoning_model(name):
            reasoning.append(name)
            continue
        params = estimate_params_b(name)
        if params is not None and params <= max_params_b + 0.2:
            fitting.append((params, name))
        else:
            others.append(name)
    if fitting:
        fitting.sort(reverse=True)
        return fitting[0][1]
    if others:
        return others[0]
    return reasoning[0] if reasoning else None


class LLMEngine:
    """Streaming text generation against one classified server."""

    def __init__(
        self,
        info: EngineInfo,
        model: str,
        temperature: float = 0.25,
        num_ctx: int = 4096,
        timeout: float = 600.0,
    ) -> None:
        self.info = info
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def kind(self) -> str:
        return self.info.kind

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def can_prefill(self) -> bool:
        """llama.cpp can copy a stable prefix with n_predict=0. vLLM APC is automatic."""
        return self.info.kind == "llamacpp"

    def prefill(self, text: str) -> bool:
        """Warm the KV cache with a decode-free parallel prefill when the engine allows it."""
        if not text.strip() or self.info.kind != "llamacpp":
            return False
        payload = {
            "prompt": text,
            "n_predict": 0,
            "temperature": 0,
            "cache_prompt": True,
            "stream": False,
        }
        try:
            response = self._client.post(self.info.host + "/completion", json=payload)
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    def count_prompt_tokens(self, text: str) -> int | None:
        """Token count of ``text`` under the server's own tokenizer, or None.

        Only llama.cpp exposes ``/tokenize``. Use this for values the server
        interprets in its own token space (``n_keep``); client-side estimates
        do not line up with the server's tokenization.
        """
        if not text or self.info.kind != "llamacpp":
            return None
        try:
            response = self._client.post(self.info.host + "/tokenize", json={"content": text})
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            tokens = response.json().get("tokens")
        except ValueError:
            return None
        return len(tokens) if isinstance(tokens, list) else None

    def generate(
        self,
        user: str,
        *,
        system: str,
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
        stop: list[str] | None = None,
        speculate: bool = False,
        n_keep: int = 0,
    ) -> str:
        if self.info.kind == "ollama":
            return self._ollama(
                user, system=system, max_tokens=max_tokens, on_token=on_token, stop=stop
            )
        if self.info.kind == "llamacpp":
            return self._llamacpp(
                user,
                system=system,
                max_tokens=max_tokens,
                on_token=on_token,
                stop=stop,
                speculate=speculate,
                n_keep=n_keep,
            )
        return self._openai(
            user,
            system=system,
            max_tokens=max_tokens,
            on_token=on_token,
            stop=stop,
            speculate=speculate,
        )

    def _ollama(
        self,
        user: str,
        *,
        system: str,
        max_tokens: int,
        on_token: Callable[[str], None] | None,
        stop: list[str] | None,
    ) -> str:
        options: dict = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens,
            "repeat_penalty": 1.08,
        }
        if stop:
            options["stop"] = stop
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": True,
            "keep_alive": "60m",
            "options": options,
        }
        return self._stream_ndjson(
            self.info.host + "/api/generate",
            payload,
            content_key="response",
            on_token=on_token,
        )

    def _llamacpp(
        self,
        user: str,
        *,
        system: str,
        max_tokens: int,
        on_token: Callable[[str], None] | None,
        stop: list[str] | None,
        speculate: bool = False,
        n_keep: int = 0,
    ) -> str:
        prompt = f"{system.rstrip()}\n\n{user}"
        stops = ["<|im_start|>", "<|endoftext|>"]
        if stop:
            stops = stops + [s for s in stop if s not in stops]
        payload: dict = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            # Repetition penalty breaks n-gram draft acceptance, so it is
            # disabled whenever speculative extras are requested.
            "repeat_penalty": 1.0 if speculate else 1.08,
            "cache_prompt": True,
            "stream": True,
            "stop": stops,
        }
        if n_keep > 0:
            payload["n_keep"] = n_keep
        if speculate:
            payload["speculative.n_max"] = 5
            payload["speculative.n_min"] = 0
        url = self.info.host + "/completion"
        try:
            return self._stream_llamacpp(url, payload, on_token)
        except EngineError as exc:
            message = str(exc)
            if speculate and "HTTP 400" in message:
                payload.pop("speculative.n_max", None)
                payload.pop("speculative.n_min", None)
                try:
                    return self._stream_llamacpp(url, payload, on_token)
                except EngineError as inner:
                    message = str(inner)
            if "HTTP 404" not in message:
                raise
            # Builds without /completion still speak the OpenAI route.
            return self._openai(
                user,
                system=system,
                max_tokens=max_tokens,
                on_token=on_token,
                stop=stop,
                speculate=speculate,
            )

    def _openai(
        self,
        user: str,
        *,
        system: str,
        max_tokens: int,
        on_token: Callable[[str], None] | None,
        stop: list[str] | None,
        speculate: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if stop:
            payload["stop"] = stop
        url = self.info.host + "/v1/chat/completions"
        candidates = [payload]
        if speculate and self.info.kind == "vllm":
            # Try vLLM's n-gram-friendly sampling first; retry plain on a 400
            # from servers that reject the extra fields.
            candidates.insert(0, {**payload, "repetition_penalty": 1.0, "min_tokens": 0})
        last_error: EngineError | None = None
        for i, candidate in enumerate(candidates):
            try:
                return self._stream_openai(url, candidate, on_token)
            except EngineError as exc:
                if i + 1 < len(candidates) and "HTTP 400" in str(exc):
                    last_error = exc
                    continue
                raise
        raise last_error if last_error else EngineError(f"LLM request to {url} failed.")

    def _stream_openai(self, url: str, payload: dict, on_token: Callable[[str], None] | None) -> str:
        chunks: list[str] = []
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise EngineError(f"LLM HTTP {response.status_code} from {url}: {body[:400]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if err := data.get("error"):
                        raise EngineError(str(err))
                    choice = (data.get("choices") or [{}])[0]
                    delta = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
                    if delta:
                        chunks.append(delta)
                        if on_token:
                            on_token(delta)
        except httpx.ConnectError as exc:
            raise EngineError(f"Cannot reach LLM server at {self.info.host}.") from exc
        except httpx.ReadTimeout as exc:
            raise EngineError(f"LLM timed out after {self.timeout:.0f}s ({self.model}).") from exc
        return strip_model_noise("".join(chunks))

    def _stream_ndjson(
        self,
        url: str,
        payload: dict,
        content_key: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        chunks: list[str] = []
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise EngineError(f"LLM HTTP {response.status_code} from {url}: {body[:400]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if err := data.get("error"):
                        raise EngineError(str(err))
                    token = data.get(content_key) or ""
                    if not token:
                        message = data.get("message") or {}
                        token = message.get("content") or ""
                    if token:
                        chunks.append(token)
                        if on_token:
                            on_token(token)
        except httpx.ConnectError as exc:
            raise EngineError(f"Cannot reach LLM server at {self.info.host}.") from exc
        except httpx.ReadTimeout as exc:
            raise EngineError(f"LLM timed out after {self.timeout:.0f}s ({self.model}).") from exc
        return strip_model_noise("".join(chunks))

    def _stream_llamacpp(self, url: str, payload: dict, on_token: Callable[[str], None] | None) -> str:
        chunks: list[str] = []
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise EngineError(f"LLM HTTP {response.status_code} from {url}: {body[:400]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("content") or ""
                    if token:
                        chunks.append(token)
                        if on_token:
                            on_token(token)
        except httpx.ConnectError as exc:
            raise EngineError(f"Cannot reach llama.cpp at {self.info.host}.") from exc
        except httpx.ReadTimeout as exc:
            raise EngineError(f"llama.cpp timed out after {self.timeout:.0f}s.") from exc
        return strip_model_noise("".join(chunks))
