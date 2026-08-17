from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from copydecode.detect import CJK_RE, cjk_ratio
from copydecode.glossary import Glossary
from copydecode.paths import cache_dir, env_value, package_data_dir
from copydecode.router import ARTIFACTS, mtl_score

# Seq2Edits-lite: sentence KEEP/REPLACE. CPU logistic model so it never
# sits on the GPU next to the 14B. Extra patterns are features only.
_EXTRA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bat this moment\b", re.I),
    re.compile(r"\bin the next second\b", re.I),
    re.compile(r"\bit was only then\b", re.I),
    re.compile(r"\ba trace of\b", re.I),
    re.compile(r"\bhis heart (?:was|skipped|trembled|stirred)\b", re.I),
    re.compile(r"\bcouldn't help but\b", re.I),
    re.compile(r"\bas if he was\b", re.I),
    re.compile(r"\bwithout (?:him|her|them) noticing\b", re.I),
    re.compile(r"\bthe next moment\b", re.I),
    re.compile(r"\beyes (?:were|was) (?:full of|filled with)\b", re.I),
    re.compile(r"\bthat kind of\b", re.I),
    re.compile(r"\bone after another\b", re.I),
]

_LEAKY_RE = re.compile(
    r"\[(?:\d+|n)\]|^\s*REPLACE\s*:|KEEP (?:before|after)",
    re.I | re.M,
)
_FEATURE_VERSION = 2

_UNSET = object()
_LOADED: Any = _UNSET


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def feature_vector(
    text: str,
    mode: str = "polish",
    glossary: Glossary | None = None,
    tag: str = "p",
) -> list[float]:
    score = float(mtl_score(text, mode, glossary, tag))
    stripped = text.strip()
    letters = sum(ch.isalpha() for ch in stripped) or 1
    features = [
        1.0,
        score / 10.0,
        cjk_ratio(text),
        min(len(stripped), 400) / 400.0,
        1.0 if CJK_RE.search(text) else 0.0,
        stripped.count("  ") / max(len(stripped), 1),
        sum(ch in ",;" for ch in stripped) / max(len(stripped), 1) * 20.0,
        sum(1 for w in stripped.split() if w[:1].isupper()) / max(len(stripped.split()), 1),
        len(stripped.split()) / 40.0,
        1.0 if letters / max(len(stripped), 1) < 0.55 else 0.0,
    ]
    for pattern, _weight in ARTIFACTS:
        features.append(1.0 if pattern.search(text) else 0.0)
    for pattern in _EXTRA_PATTERNS:
        features.append(1.0 if pattern.search(text) else 0.0)
    return features


def leaky_model_text(text: str) -> bool:
    return bool(_LEAKY_RE.search(text or ""))


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _has_mtl_cues(text: str) -> bool:
    if mtl_score(text, "polish") >= 1:
        return True
    return any(pattern.search(text) for pattern in _EXTRA_PATTERNS)


@dataclass
class LabeledSpan:
    text: str
    replace: bool
    source: str = "synthetic"


@dataclass
class SpanTagger:
    weights: list[float]
    threshold: float
    feature_version: int = _FEATURE_VERSION
    replace_recall: float = 0.0
    keep_rate: float = 0.0
    keep_precision: float = 0.0
    n_replace: int = 0
    n_keep: int = 0
    fingerprint: str = ""
    anchors: list[str] = field(default_factory=list)

    def _anchor_set(self) -> set[str]:
        cached = getattr(self, "_anchors_cached", None)
        if cached is None:
            cached = {a for a in self.anchors if a}
            self._anchors_cached = cached
        return cached

    def anchor_hit(self, text: str) -> bool:
        folded = _normalize(text)
        if not folded:
            return False
        anchors = self._anchor_set()
        if folded in anchors:
            return True
        if len(folded) < 48:
            return False
        for anchor in anchors:
            if len(anchor) >= 48 and (anchor in folded or folded in anchor):
                return True
        return False

    def logit(self, text: str, mode: str = "polish", glossary: Glossary | None = None, tag: str = "p") -> float:
        vec = feature_vector(text, mode, glossary, tag)
        if len(vec) != len(self.weights):
            return float(mtl_score(text, mode, glossary, tag))
        return sum(weight * value for weight, value in zip(self.weights, vec))

    def probability(self, text: str, mode: str = "polish", glossary: Glossary | None = None, tag: str = "p") -> float:
        return _sigmoid(self.logit(text, mode, glossary, tag))

    def is_replace(self, text: str, mode: str = "polish", glossary: Glossary | None = None, tag: str = "p") -> bool:
        if self.anchor_hit(text):
            return True
        return self.probability(text, mode, glossary, tag) >= self.threshold

    def strength(self, text: str, mode: str = "polish", glossary: Glossary | None = None, tag: str = "p") -> float:
        if self.anchor_hit(text):
            return 1.0
        return self.probability(text, mode, glossary, tag)

    def to_json(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "threshold": self.threshold,
            "feature_version": self.feature_version,
            "replace_recall": self.replace_recall,
            "keep_rate": self.keep_rate,
            "keep_precision": self.keep_precision,
            "n_replace": self.n_replace,
            "n_keep": self.n_keep,
            "fingerprint": self.fingerprint,
            "anchors": self.anchors,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SpanTagger:
        weights = [float(x) for x in data["weights"]]
        anchors = [str(a) for a in data.get("anchors") or [] if str(a).strip()]
        digest = hashlib.sha256(
            json.dumps({"w": weights, "a": anchors}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return cls(
            weights=weights,
            threshold=float(data.get("threshold", 0.5)),
            feature_version=int(data.get("feature_version", _FEATURE_VERSION)),
            replace_recall=float(data.get("replace_recall", 0.0)),
            keep_rate=float(data.get("keep_rate", 0.0)),
            keep_precision=float(data.get("keep_precision", 0.0)),
            n_replace=int(data.get("n_replace", 0)),
            n_keep=int(data.get("n_keep", 0)),
            fingerprint=str(data.get("fingerprint") or digest),
            anchors=anchors,
        )


def synthetic_examples() -> list[LabeledSpan]:
    replace_seeds = [
        "He could not help but smile at the news.",
        "The corners of his mouth curled up slightly.",
        "His eyes flashed with a cold light.",
        "She sucked in a breath of cold air.",
        "In the next moment, the youth vanished.",
        "He did not expect that the elder would agree.",
        "This was a heaven-defying treasure.",
        "He said in a deep voice, \"Leave.\"",
        "The youth clenched his fists tightly.",
        "He suddenly discovered that the formation had changed.",
        "A look of shock flashed across his face.",
        "This daddy will not spare you.",
        "That is very much so the case.",
        "Incontinently he rushed forward.",
        "He saw the the mountain peak ahead.",
        "The man already was standing there.",
        "She was very much unwilling.",
        "His jindan trembled as qi surged.",
        "At this moment, his heart was filled with killing intent.",
        "In the next second, it was only then that he understood.",
        "A trace of fear appeared in her eyes.",
        "He couldn't help but take a step back.",
        "It was as if he was looking at an ant.",
        "Without him noticing, night had fallen.",
        "His eyes were full of unwillingness.",
        "One after another, experts arrived.",
        "He already was very much so in the next moment.",
        "The corners of her mouth rose as her eyes shone.",
        "Sucked in a cold air, the youth said in a heavy voice.",
    ]
    keep_seeds = [
        "The mountain wind was cold against his face.",
        "Then he walked toward the sect gate.",
        "The valley was quiet after the storm.",
        "She set the cup down and waited.",
        "Clouds hung low over the river.",
        "Luo Feng did not answer at once.",
        "The hall doors opened onto torchlight.",
        "Rain ticked against the tiled roof.",
        "He counted the remaining arrows twice.",
        "No one spoke for a long time.",
        "The map showed three unmarked passes.",
        "Her voice was steady, almost bored.",
        "They crossed the bridge before dawn.",
        "A single lantern marked the warehouse.",
        "He remembered the lesson and slowed his breath.",
        "The elder nodded once and left it at that.",
        "Snow had drifted against the north wall.",
        "It was a small room with a clean floor.",
        "The letter had no seal and no name.",
        "She tied the horse and checked the cinch.",
        "Morning found the camp already struck.",
        "He kept the token inside his sleeve.",
        "The road east was empty for miles.",
        "Nothing in the ledger matched the story.",
        "They ate in silence and slept in shifts.",
        "The sword stayed in its sheath.",
        "A dog barked once, then thought better of it.",
        "He closed the book without marking the page.",
        "The tide had pulled the boats sideways.",
        "She smiled, and this time it reached her eyes.",
        "The formation lines were still intact.",
        "Golden Core was a distant problem.",
        "He bowed, received the token, and left.",
        "The night watch changed at the third drum.",
        "None of the names on the list were his.",
        "The tea had gone cold beside the window.",
        "He knew the path well enough to walk it dark.",
        "A child waved from the upper window.",
        "The contract was short and badly copied.",
        "They would not reach the city before snow.",
    ]
    out = [LabeledSpan(text, True, "synthetic") for text in replace_seeds]
    out.extend(LabeledSpan(text, False, "synthetic") for text in keep_seeds)
    return out


def examples_from_changelog(path: Path) -> list[LabeledSpan]:
    from copydecode.spans import replacement_ok, split_units

    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[LabeledSpan] = []
    for edit in data.get("edits") or []:
        before = str(edit.get("before") or "").strip()
        after = str(edit.get("after") or "").strip()
        if not before:
            continue
        for unit in split_units(before) or [before]:
            stripped = unit.strip()
            if stripped and _has_mtl_cues(stripped):
                out.append(LabeledSpan(stripped, True, str(path)))
        if after and not leaky_model_text(after) and replacement_ok(before, after):
            for unit in split_units(after) or [after]:
                if unit.strip() and not _has_mtl_cues(unit):
                    out.append(LabeledSpan(unit, False, str(path)))
    for item in data.get("unchanged") or []:
        before = str(item.get("before") or "").strip()
        if not before:
            continue
        for unit in split_units(before) or [before]:
            stripped = unit.strip()
            if not stripped:
                continue
            if _has_mtl_cues(stripped):
                out.append(LabeledSpan(stripped, True, str(path)))
            else:
                out.append(LabeledSpan(stripped, False, str(path)))
    return out


def load_changelog_paths(paths: Iterable[Path]) -> list[LabeledSpan]:
    examples: list[LabeledSpan] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".json":
            examples.extend(examples_from_changelog(path))
    return examples


def _fit_weights(examples: list[LabeledSpan], epochs: int = 450, lr: float = 0.25, l2: float = 0.02) -> list[float]:
    dim = len(feature_vector("x"))
    weights = [0.0] * dim
    n_replace = sum(1 for ex in examples if ex.replace) or 1
    n_keep = sum(1 for ex in examples if not ex.replace) or 1
    w_pos = n_keep / n_replace
    w_neg = n_replace / n_keep
    scale = 2.0 / (w_pos + w_neg)
    w_pos *= scale
    w_neg *= scale
    vectors = [(feature_vector(ex.text), 1.0 if ex.replace else 0.0, w_pos if ex.replace else w_neg) for ex in examples]
    n = len(vectors)
    for _ in range(epochs):
        grad = [l2 * w for w in weights]
        grad[0] = 0.0
        for vec, y, weight in vectors:
            z = sum(a * b for a, b in zip(weights, vec))
            err = _sigmoid(z) - y
            coef = weight * err / n
            for i, value in enumerate(vec):
                grad[i] += coef * value
        for i, g in enumerate(grad):
            weights[i] -= lr * g
    return weights


def _metrics(weights: list[float], examples: list[LabeledSpan], threshold: float) -> tuple[float, float, float]:
    replace = [ex for ex in examples if ex.replace]
    keep = [ex for ex in examples if not ex.replace]
    def pred(ex: LabeledSpan) -> bool:
        z = sum(w * v for w, v in zip(weights, feature_vector(ex.text)))
        return _sigmoid(z) >= threshold

    tp = sum(1 for ex in replace if pred(ex))
    tn = sum(1 for ex in keep if not pred(ex))
    recall = tp / max(len(replace), 1)
    keep_rate = tn / max(len(keep), 1)
    predicted_keep = sum(1 for ex in examples if not pred(ex))
    keep_precision = tn / max(predicted_keep, 1)
    return recall, keep_rate, keep_precision


def _choose_threshold(weights: list[float], examples: list[LabeledSpan], min_recall: float) -> float:
    replace_scores = sorted(
        _sigmoid(sum(w * v for w, v in zip(weights, feature_vector(ex.text))))
        for ex in examples
        if ex.replace
    )
    if not replace_scores:
        return 0.5
    candidates = sorted({round(s - 0.002, 3) for s in replace_scores} | {0.15, 0.25, 0.35, 0.45, 0.5})
    best = min(replace_scores)
    best_keep = -1.0
    for threshold in candidates:
        recall, keep_rate, _prec = _metrics(weights, examples, threshold)
        if recall + 1e-9 < min_recall:
            continue
        if keep_rate > best_keep:
            best_keep = keep_rate
            best = threshold
    return best


def train_tagger(
    examples: list[LabeledSpan],
    *,
    min_recall: float = 0.99,
    include_synthetic: bool = True,
) -> SpanTagger:
    pooled = list(examples)
    if include_synthetic:
        pooled.extend(synthetic_examples())
    if not pooled:
        raise ValueError("No tagger examples to train on.")
    anchors = sorted(
        {
            _normalize(ex.text)
            for ex in examples
            if ex.replace and len(_normalize(ex.text)) >= 12
        }
    )
    logistic: list[LabeledSpan] = []
    for ex in pooled:
        if ex.replace and ex.source != "synthetic":
            continue
        logistic.append(ex)
    if not logistic:
        logistic = synthetic_examples() if include_synthetic else pooled
    weights = _fit_weights(logistic)
    threshold = _choose_threshold(weights, logistic, min_recall)
    recall, keep_rate, keep_precision = _metrics(weights, logistic, threshold)
    n_replace = sum(1 for ex in logistic if ex.replace)
    n_keep = sum(1 for ex in logistic if not ex.replace)
    digest = hashlib.sha256(
        json.dumps({"w": weights, "a": anchors}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return SpanTagger(
        weights=weights,
        threshold=threshold,
        replace_recall=recall,
        keep_rate=keep_rate,
        keep_precision=keep_precision,
        n_replace=n_replace,
        n_keep=n_keep,
        fingerprint=digest,
        anchors=anchors,
    )


def default_tagger_path() -> Path:
    return cache_dir() / "span_tagger.json"


def bundled_tagger_path() -> Path:
    return package_data_dir() / "span_tagger.json"


def save_tagger(tagger: SpanTagger, path: Path | None = None) -> Path:
    dest = path or default_tagger_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(tagger.to_json(), indent=2), encoding="utf-8")
    reset_tagger_cache()
    return dest


def load_tagger_file(path: Path) -> SpanTagger:
    return SpanTagger.from_json(json.loads(path.read_text(encoding="utf-8")))


def reset_tagger_cache() -> None:
    global _LOADED
    _LOADED = _UNSET


def get_tagger() -> SpanTagger | None:
    global _LOADED
    if _LOADED is not _UNSET:
        return _LOADED
    raw = env_value("COPYDECODE_TAGGER", "NOVELPOLISHER_TAGGER")
    if raw.lower() in {"0", "off", "none", "heuristic"}:
        _LOADED = None
        return None
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    candidates.append(default_tagger_path())
    candidates.append(bundled_tagger_path())
    for path in candidates:
        if path.is_file():
            try:
                tagger = load_tagger_file(path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if tagger.feature_version != _FEATURE_VERSION:
                continue
            if len(tagger.weights) != len(feature_vector("x")):
                continue
            _LOADED = tagger
            return tagger
    _LOADED = None
    return None


def tagger_id() -> str:
    tagger = get_tagger()
    if tagger is None:
        return "heur"
    return tagger.fingerprint or "learned"


def evaluate_against_changelog(path: Path, tagger: SpanTagger | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold_replace = []
    gold_keep = []
    leaky = 0
    for edit in data.get("edits") or []:
        before = str(edit.get("before") or "").strip()
        after = str(edit.get("after") or "").strip()
        if before:
            gold_replace.append(before)
        if after and leaky_model_text(after):
            leaky += 1
        elif after:
            gold_keep.append(after)
    for item in data.get("unchanged") or []:
        before = str(item.get("before") or "").strip()
        if before:
            gold_keep.append(before)

    def rate(texts: list[str], want_replace: bool, learned: bool) -> float:
        if not texts:
            return 1.0
        hits = 0
        for text in texts:
            if learned:
                pred = bool(tagger and tagger.is_replace(text))
            else:
                pred = mtl_score(text, "polish") >= 1
            hits += int(pred == want_replace)
        return hits / len(texts)

    learned = tagger or get_tagger()
    return {
        "file": str(path),
        "llm_spans_sent": data.get("llm_spans_sent"),
        "llm_edits": data.get("llm_edits") or len(data.get("edits") or []),
        "llm_unchanged": data.get("llm_unchanged"),
        "leaky_after": leaky,
        "gold_replace": len(gold_replace),
        "gold_keep": len(gold_keep),
        "heuristic_replace_recall": rate(gold_replace, True, False),
        "learned_replace_recall": rate(gold_replace, True, True) if learned else None,
        "heuristic_keep_rate": rate(gold_keep, False, False),
        "learned_keep_rate": rate(gold_keep, False, True) if learned else None,
        "tagger": learned.fingerprint if learned else None,
    }


def export_kd_pairs(path: Path) -> list[dict[str, str]]:
    from copydecode.spans import replacement_ok

    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for edit in data.get("edits") or []:
        before = str(edit.get("before") or "").strip()
        after = str(edit.get("after") or "").strip()
        if not before or not after or leaky_model_text(after):
            continue
        if not replacement_ok(before, after):
            continue
        pairs.append({"source": before, "target": after})
    return pairs


def _tagger_fingerprint(weights: list[float], anchors: list[str]) -> str:
    return hashlib.sha256(
        json.dumps({"w": weights, "a": anchors}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def merge_anchors(tagger: SpanTagger, extra: Iterable[str]) -> SpanTagger:
    merged = sorted({_normalize(item) for item in [*tagger.anchors, *extra] if _normalize(item)})
    tagger.anchors = merged
    tagger._anchors_cached = None  # type: ignore[attr-defined]
    tagger.fingerprint = _tagger_fingerprint(tagger.weights, merged)
    return tagger


def train_from_files(
    paths: list[Path],
    *,
    dest: Path | None = None,
    min_recall: float = 0.99,
    include_synthetic: bool = True,
    merge_existing: bool = True,
) -> tuple[SpanTagger, Path]:
    previous = get_tagger() if merge_existing else None
    examples = load_changelog_paths(paths)
    tagger = train_tagger(examples, min_recall=min_recall, include_synthetic=include_synthetic)
    if previous and previous.anchors:
        merge_anchors(tagger, previous.anchors)
    return tagger, save_tagger(tagger, dest)
