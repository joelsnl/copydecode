from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from copydecode.hardware import DeviceProfile, estimate_params_b, is_reasoning_model, recommended_serve_commands

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)

PROBE_CANDIDATES = (
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:11434",
)


class EngineError(RuntimeError):
    pass


OllamaError = EngineError


def strip_model_noise(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = THINKING_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    return text.strip()


@dataclass
class EngineInfo:
    kind: str
    host: str
    label: str


def _ok(url: str, timeout: float = 1.2) -> httpx.Response | None:
    try:
        return httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return None


def classify_host(host: str) -> EngineInfo:
    base = host.rstrip("/")
    tags = _ok(base + "/api/tags")
    if tags is not None and tags.status_code == 200:
        return EngineInfo("ollama", base, "Ollama")
    health = _ok(base + "/health")
    if health is not None and health.status_code == 200:
        return EngineInfo("llamacpp", base, "llama.cpp")
    models = _ok(base + "/v1/models")
    if models is not None and models.status_code == 200:
        body = models.text.lower()
        kind = "vllm" if "vllm" in body or "model_name" in body else "openai"
        label = "vLLM" if kind == "vllm" else "OpenAI-compatible"
        return EngineInfo(kind, base, label)
    completion = _ok(base + "/completion")
    if completion is not None and completion.status_code in {200, 400, 405}:
        return EngineInfo("llamacpp", base, "llama.cpp")
    raise EngineError(
        f"No LLM server at {base}. Start Ollama (`ollama serve`), llama-server, or vLLM."
    )


def discover_engine(preferred: str | None, host: str | None, profile: DeviceProfile) -> EngineInfo:
    if host:
        info = classify_host(host)
        if preferred and preferred not in {"auto", info.kind}:
            if preferred == "openai" and info.kind in {"vllm", "openai", "llamacpp"}:
                return info
            if preferred == "llamacpp" and info.kind == "llamacpp":
                return info
            if preferred == "ollama" and info.kind == "ollama":
                return info
        return info

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
            if preferred == "openai" and info.kind in {"vllm", "openai"}:
                return info
            if info.kind == preferred or (preferred == "llamacpp" and info.kind == "llamacpp"):
                return info
        raise EngineError(
            f"No {preferred} server found on ports 8000, 8080, or 11434. "
            "Run `copydecode serve` to download and start llama.cpp for this machine."
        )

    order = ["vllm", "llamacpp", "openai", "ollama"]
    if profile.backend != "cuda":
        order = ["llamacpp", "ollama", "openai", "vllm"]
    rank = {kind: i for i, kind in enumerate(order)}
    found.sort(key=lambda info: rank.get(info.kind, 9))
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
    fitting: list[tuple[float, str]] = []
    others: list[str] = []
    for name in names:
        if is_reasoning_model(name):
            others.append(name)
            continue
        params = estimate_params_b(name)
        if params <= max_params_b + 0.2:
            fitting.append((params, name))
        else:
            others.append(name)
    fitting.sort(reverse=True)
    if fitting:
        return fitting[0][1]
    non_reason = [name for name in others if not is_reasoning_model(name)]
    if non_reason:
        return non_reason[0]
    return names[0] if names else None


class LLMEngine:
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

    def can_prefill(self) -> bool:
        """llama.cpp can copy the stable prefix with n_predict=0. vLLM APC is automatic."""
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

    def complete(self, system: str, user: str, on_token: Callable[[str], None] | None = None) -> str:
        return self.generate(user, system=system, max_tokens=min(2048, self.num_ctx // 2), on_token=on_token)

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
            msg = str(exc)
            if speculate and "HTTP 400" in msg:
                payload.pop("speculative.n_max", None)
                payload.pop("speculative.n_min", None)
                try:
                    return self._stream_llamacpp(url, payload, on_token)
                except EngineError as inner:
                    msg = str(inner)
            if "HTTP 404" not in msg:
                raise
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
        if speculate and self.info.kind == "vllm":
            payload["repetition_penalty"] = 1.0
            payload["min_tokens"] = 0
        url = self.info.host + "/v1/chat/completions"
        chunks: list[str] = []
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    if speculate and response.status_code == 400 and (
                        "repetition_penalty" in payload or "min_tokens" in payload
                    ):
                        payload.pop("repetition_penalty", None)
                        payload.pop("min_tokens", None)
                        return self._openai(
                            user,
                            system=system,
                            max_tokens=max_tokens,
                            on_token=on_token,
                            stop=stop,
                            speculate=False,
                        )
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
                    delta = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    if not delta:
                        delta = ((data.get("choices") or [{}])[0].get("text") or "")
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


class OllamaClient(LLMEngine):
    """Backward-compatible wrapper used by older call sites."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.25,
        num_ctx: int = 8192,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(
            EngineInfo("ollama", host.rstrip("/"), "Ollama"),
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
        )
