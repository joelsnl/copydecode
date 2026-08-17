from __future__ import annotations

import re
from typing import Iterable

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
HANGUL_RE = re.compile(r"[\u1100-\u11ff\uac00-\ud7af]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
LATIN_RE = re.compile(r"[A-Za-z]")
FOREIGN_SCRIPT_RE = re.compile(
    r"["
    r"\u0400-\u04ff"
    r"\u0600-\u06ff"
    r"\u0900-\u097f"
    r"\u0e00-\u0e7f"
    r"\u1100-\u11ff"
    r"\u3040-\u30ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uac00-\ud7af"
    r"]"
)

_NLLB_SCRIPTS = (
    ("jpn_Jpan", KANA_RE),
    ("kor_Hang", HANGUL_RE),
    ("zho_Hans", CJK_RE),
    ("arb_Arab", ARABIC_RE),
    ("rus_Cyrl", CYRILLIC_RE),
    ("hin_Deva", DEVANAGARI_RE),
    ("tha_Thai", THAI_RE),
)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = cjk + latin
    if total == 0:
        return 0.0
    return cjk / total


def foreign_script_ratio(text: str) -> float:
    """Share of letters that are not Latin (CJK, kana, Hangul, Arabic, …)."""
    if not text:
        return 0.0
    foreign = len(FOREIGN_SCRIPT_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = foreign + latin
    if total == 0:
        return 0.0
    return foreign / total


def sample_text(parts: Iterable[str], limit: int = 8000) -> str:
    buf: list[str] = []
    size = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        buf.append(stripped)
        size += len(stripped)
        if size >= limit:
            break
    return "\n".join(buf)


def detect_mode(text: str) -> str:
    """Return 'translate' for non-English source scripts, 'polish' for English MTL."""
    return "translate" if foreign_script_ratio(text) >= 0.28 else "polish"


def guess_nllb_src_lang(text: str) -> str:
    """Best-effort NLLB source tag from the dominant non-Latin script."""
    best_lang = "zho_Hans"
    best_n = 0
    for lang, pattern in _NLLB_SCRIPTS:
        n = len(pattern.findall(text))
        if n > best_n:
            best_lang = lang
            best_n = n
    return best_lang
