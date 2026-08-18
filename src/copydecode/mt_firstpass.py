"""Optional NLLB draft pass (CTranslate2). Loaded only when COPYDECODE_NLLB_PATH is set."""

from __future__ import annotations

import gc
from functools import lru_cache

from copydecode.detect import foreign_script_ratio, guess_nllb_src_lang
from copydecode.paths import env_value


@lru_cache(maxsize=1)
def _load_translator():
    path = env_value("COPYDECODE_NLLB_PATH")
    if not path:
        return None
    try:
        import ctranslate2
        from transformers import AutoTokenizer
    except ImportError:
        return None
    # Default CPU: the polish 14B is already resident on the GPU via vLLM/llama.cpp.
    device = env_value("COPYDECODE_NLLB_DEVICE", default="cpu").lower()
    if device not in {"cpu", "cuda"}:
        device = "cpu"
    translator = ctranslate2.Translator(path, device=device, compute_type="int8")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="zho_Hans")
    return translator, tokenizer


def unload_nllb() -> None:
    """Drop the NLLB translator so the polish model can own the GPU."""
    _load_translator.cache_clear()
    gc.collect()


def nllb_available() -> bool:
    return _load_translator() is not None


def draft_translate(text: str) -> str:
    loaded = _load_translator()
    if loaded is None or foreign_script_ratio(text) < 0.18:
        return text
    translator, tokenizer = loaded
    tokenizer.src_lang = guess_nllb_src_lang(text)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))
    results = translator.translate_batch(
        [tokens],
        target_prefix=[["eng_Latn"]],
        beam_size=1,
        max_decoding_length=min(512, max(32, len(tokens) * 2)),
    )
    out_tokens = results[0].hypotheses[0]
    if out_tokens and out_tokens[0] == "eng_Latn":
        out_tokens = out_tokens[1:]
    ids = tokenizer.convert_tokens_to_ids(out_tokens)
    return tokenizer.decode(ids, skip_special_tokens=True).strip() or text
