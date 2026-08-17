from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from copydecode.detect import FOREIGN_SCRIPT_RE
from copydecode.paths import cache_dir, env_value

# Qwen2.5 BPE on English is typically ~2.6–3.4 chars/token; CJK/kana/Hangul ~1 token/char.
# This estimate slightly over-counts so packs stay under num_ctx/2 without the HF vocab.
_OTHER_CHARS_PER_TOKEN = 2.6


def estimate_qwen_tokens(text: str) -> int:
    if not text:
        return 0
    foreign = len(FOREIGN_SCRIPT_RE.findall(text))
    other = max(0, len(text) - foreign)
    return max(1, int(foreign * 1.08 + other / _OTHER_CHARS_PER_TOKEN) + 1)


def _tokenizer_from_file(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


@lru_cache(maxsize=1)
def _hf_tokenizer():
    """Load Qwen2.5 tokenizer.json from env, cache, or a one-time Hub fetch."""
    raw = env_value("COPYDECODE_TOKENIZER", "NOVELPOLISHER_TOKENIZER")
    if raw.lower() in {"0", "off", "none", "estimate"}:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        Tokenizer = None  # type: ignore[assignment]
    if raw:
        path = Path(raw)
        if path.is_file() and Tokenizer is not None:
            return _tokenizer_from_file(str(path))
        return None
    if Tokenizer is None:
        return None

    cached_copy = cache_dir() / "tokenizers" / "qwen2.5" / "tokenizer.json"
    if cached_copy.is_file():
        return _tokenizer_from_file(str(cached_copy))
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache("Qwen/Qwen2.5-0.5B", "tokenizer.json")
        if cached and cached != ".no_exist" and Path(str(cached)).is_file():
            return _tokenizer_from_file(str(cached))
    except Exception:
        pass
    try:
        from huggingface_hub import hf_hub_download

        tok_file = hf_hub_download(
            "Qwen/Qwen2.5-0.5B",
            "tokenizer.json",
            local_files_only=False,
        )
        path = Path(tok_file)
        if path.is_file():
            cached_copy.parent.mkdir(parents=True, exist_ok=True)
            if not cached_copy.exists():
                cached_copy.write_bytes(path.read_bytes())
            return _tokenizer_from_file(str(path))
    except Exception:
        return None
    return None


def qwen_token_count(text: str) -> int:
    tokenizer = _hf_tokenizer()
    if tokenizer is None:
        return estimate_qwen_tokens(text)
    encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return len(ids)


def tokenizer_label() -> str:
    return "Qwen2.5 tokenizer.json" if _hf_tokenizer() is not None else "Qwen2.5 estimate"


def prompt_token_budget(num_ctx: int) -> int:
    """Pack REPLACE jobs until the prompt hits about half the context window."""
    return max(256, num_ctx // 2)
