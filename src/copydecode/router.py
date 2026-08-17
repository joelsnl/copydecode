from __future__ import annotations

import re

from copydecode.detect import CJK_RE, foreign_script_ratio
from copydecode.document import SKIP_LLM_TAGS, Segment
from copydecode.glossary import Glossary

BOILERPLATE_RE = re.compile(
    r"^(?:page\s*)?\d+(?:\s*/\s*\d+)?$|^(?:copyright|contents|cover|title page)$",
    re.IGNORECASE,
)

ARTIFACTS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"eyes (?:flashed|gleamed|narrowed|shone)", re.I), 2),
    (re.compile(r"the corners of (?:his|her|their) mouth", re.I), 2),
    (re.compile(r"could not help but", re.I), 2),
    (re.compile(r"sucked in a (?:breath of )?cold air", re.I), 2),
    (re.compile(r"in the next moment", re.I), 1),
    (re.compile(r"did not expect that", re.I), 1),
    (re.compile(r"heaven-defying", re.I), 1),
    (re.compile(r"said (?:in a |with a )?(?:deep|cold|heavy) voice", re.I), 1),
    (re.compile(r"\bthe youth\b", re.I), 1),
    (re.compile(r"\b(?:jindan|yuan ?ying|yuanying|zhuji|huashen|lianqi)\b", re.I), 2),
    (re.compile(r"suddenly discovered", re.I), 1),
    (re.compile(r"a look of .{3,24}(?:flashed|appeared|crossed)", re.I), 1),
    (re.compile(r"\bthis (?:one|daddy) will\b", re.I), 1),
    (re.compile(r"\bvery much so\b", re.I), 1),
    (re.compile(r"\bincontinently\b", re.I), 3),
    (re.compile(r"\bthe the\b", re.I), 2),
    (re.compile(r"\balready was\b", re.I), 1),
    (re.compile(r"\bwas very much\b", re.I), 1),
]


def is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    if BOILERPLATE_RE.match(stripped):
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    return letters == 0


def mtl_score(text: str, mode: str, glossary: Glossary | None = None, tag: str = "p") -> int:
    if is_boilerplate(text):
        return 0
    score = 0
    if mode == "translate" and foreign_script_ratio(text) >= 0.18:
        return 10
    if CJK_RE.search(text):
        score += 3
    if glossary and glossary.unapplied_hits(text):
        score += 2
    for pattern, weight in ARTIFACTS:
        if pattern.search(text):
            score += weight
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        score = max(0, score - 1)
    return score


def skip_threshold(skip_mode: str) -> int:
    if skip_mode in {"off", "none"}:
        return 0
    if skip_mode == "aggressive":
        return 2
    return 1


def needs_llm(
    segment: Segment,
    mode: str,
    skip_mode: str,
    glossary: Glossary | None = None,
) -> bool:
    if is_boilerplate(segment.text):
        return False
    if segment.tag in SKIP_LLM_TAGS:
        return False
    threshold = skip_threshold(skip_mode)
    if threshold <= 0:
        return True
    if mode == "translate":
        return foreign_script_ratio(segment.text) >= 0.08 or mtl_score(segment.text, mode, glossary, segment.tag) >= 1
    return mtl_score(segment.text, mode, glossary, segment.tag) >= threshold
