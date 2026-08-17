from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from copydecode.glossary import Glossary
from copydecode.router import mtl_score, skip_threshold

# Copy-as-Decode / Seq2Edits: tiny KEEP islands between REPLACE spans are
# usually segmentation noise, not real copy opportunities.
MIN_KEEP_BRIDGE = 48
MIN_KEEP_RATIO = 0.22
CONTEXT_CHARS = 280

_SENT_END = re.compile(r"(?:\.{3}|…|(?<!\d)[.!?]|[。！？])[\"'”’)]*(?:\s+|$)")


@dataclass(frozen=True)
class Span:
    kind: str  # KEEP | REPLACE
    text: str


@dataclass(frozen=True)
class SpanJob:
    seg_index: int
    span_index: int
    text: str
    before: str
    after: str


@dataclass(frozen=True)
class EditProgram:
    spans: tuple[Span, ...]

    @property
    def keep_chars(self) -> int:
        return sum(len(span.text) for span in self.spans if span.kind == "KEEP")

    @property
    def replace_chars(self) -> int:
        return sum(len(span.text) for span in self.spans if span.kind == "REPLACE")

    @property
    def keep_ratio(self) -> float:
        total = self.keep_chars + self.replace_chars
        if total <= 0:
            return 1.0
        return self.keep_chars / total

    def stitched(self, replacements: dict[int, str]) -> str:
        parts: list[str] = []
        for index, span in enumerate(self.spans):
            if span.kind == "KEEP":
                parts.append(span.text)
            else:
                parts.append(replacements.get(index, span.text))
        return "".join(parts)


def split_units(text: str) -> list[str]:
    """Sentence-ish split that round-trips: ``''.join(units) == text``."""
    if not text:
        return []
    units: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        end = match.end()
        if end > start:
            units.append(text[start:end])
            start = end
    if start < len(text):
        units.append(text[start:])
    return _split_long_units(units or [text])


def _split_long_units(units: list[str], max_len: int = 240) -> list[str]:
    out: list[str] = []
    for unit in units:
        if len(unit) <= max_len:
            out.append(unit)
            continue
        out.extend(_split_on_delims(unit, ("; ", " — ", " - ", ", "), max_len))
    return _attach_short(out)


def _split_on_delims(text: str, delims: tuple[str, ...], max_len: int) -> list[str]:
    pieces = [text]
    for delim in delims:
        nxt: list[str] = []
        for piece in pieces:
            if len(piece) <= max_len or delim not in piece:
                nxt.append(piece)
                continue
            parts = piece.split(delim)
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    nxt.append(part + delim)
                elif part or not nxt:
                    nxt.append(part)
        pieces = nxt
        if all(len(p) <= max_len for p in pieces):
            break
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_len:
            final.append(piece)
            continue
        for i in range(0, len(piece), max_len):
            final.append(piece[i : i + max_len])
    return [p for p in final if p]


def _attach_short(units: list[str], min_len: int = 12) -> list[str]:
    if not units:
        return []
    sent_end = re.compile(r"[.!?。！？][\"'”’)]*$")
    out = [units[0]]
    for unit in units[1:]:
        stripped = unit.strip()
        if len(stripped) < min_len and not sent_end.search(stripped):
            out[-1] += unit
        else:
            out.append(unit)
    return out


def coalesce(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in spans:
        if merged and merged[-1].kind == span.kind:
            merged[-1] = Span(span.kind, merged[-1].text + span.text)
        else:
            merged.append(span)
    return merged


def absorb_bridges(spans: list[Span], min_keep: int = MIN_KEEP_BRIDGE) -> list[Span]:
    spans = coalesce(spans)
    changed = True
    while changed:
        changed = False
        out: list[Span] = []
        i = 0
        while i < len(spans):
            span = spans[i]
            if (
                0 < i < len(spans) - 1
                and span.kind == "KEEP"
                and len(span.text.strip()) < min_keep
                and spans[i - 1].kind == "REPLACE"
                and spans[i + 1].kind == "REPLACE"
            ):
                prev = out[-1]
                nxt = spans[i + 1]
                out[-1] = Span("REPLACE", prev.text + span.text + nxt.text)
                i += 2
                changed = True
                continue
            out.append(span)
            i += 1
        spans = coalesce(out)
    return spans


def clip_context(text: str, *, tail: bool, limit: int = CONTEXT_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    if tail:
        return "…" + stripped[-limit:]
    return stripped[:limit] + "…"


def tag_text(
    text: str,
    mode: str,
    skip_mode: str,
    glossary: Glossary | None = None,
    tag: str = "p",
    *,
    force_dirty: bool = False,
    learned: bool | None = None,
) -> EditProgram:
    """Seq2Edits program: KEEP clean clauses, REPLACE MTL. Optional CPU tagger."""
    if not text:
        return EditProgram((Span("KEEP", ""),))

    threshold = skip_threshold(skip_mode)
    replace_at = 1 if threshold <= 0 else threshold
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        units = [text]
    else:
        units = split_units(text)

    use_learned = bool(learned) if learned is not None else True
    tagger = None
    if use_learned:
        from copydecode.tagger import get_tagger

        tagger = get_tagger()

    labeled: list[tuple[str, str, float]] = []
    for unit in units:
        score = mtl_score(unit, mode, glossary, tag)
        if tagger is not None:
            replace = tagger.is_replace(unit, mode, glossary, tag)
            strength = tagger.strength(unit, mode, glossary, tag)
        else:
            replace = score >= replace_at
            strength = float(score)
        labeled.append((unit, "REPLACE" if replace else "KEEP", strength))

    if skip_mode in {"off", "none"} and not any(kind == "REPLACE" for _u, kind, _s in labeled):
        return EditProgram((Span("REPLACE", text),))
    if force_dirty and not any(kind == "REPLACE" for _u, kind, _s in labeled):
        if len(labeled) <= 1:
            return EditProgram((Span("REPLACE", text),))
        best = max(range(len(labeled)), key=lambda i: labeled[i][2])
        unit, _kind, strength = labeled[best]
        labeled[best] = (unit, "REPLACE", strength)

    spans = absorb_bridges([Span(kind, unit) for unit, kind, _s in labeled])
    program = EditProgram(tuple(spans))
    if program.replace_chars and program.keep_ratio < MIN_KEEP_RATIO:
        return EditProgram((Span("REPLACE", text),))
    return program


def span_jobs_for(seg_index: int, program: EditProgram) -> list[SpanJob]:
    jobs: list[SpanJob] = []
    spans = program.spans
    for index, span in enumerate(spans):
        if span.kind != "REPLACE":
            continue
        before = spans[index - 1].text if index > 0 and spans[index - 1].kind == "KEEP" else ""
        after = (
            spans[index + 1].text
            if index + 1 < len(spans) and spans[index + 1].kind == "KEEP"
            else ""
        )
        jobs.append(
            SpanJob(
                seg_index=seg_index,
                span_index=index,
                text=span.text,
                before=clip_context(before, tail=True) if before else "",
                after=clip_context(after, tail=False) if after else "",
            )
        )
    return jobs


def format_span_job(job: SpanJob, index: int) -> str:
    lines = [f"[{index}]"]
    if job.before.strip():
        lines.append("KEEP before (do not output):")
        lines.append(job.before.strip())
    lines.append("REPLACE:")
    lines.append(job.text)
    if job.after.strip():
        lines.append("KEEP after (do not output):")
        lines.append(job.after.strip())
    return "\n".join(lines)


def job_char_cost(job: SpanJob) -> int:
    return len(job.text) + len(job.before) + len(job.after) + 32


def pack_span_jobs(
    jobs: list[SpanJob],
    max_chars: int,
    *,
    max_prompt_tokens: int = 0,
    prefix_tokens: int = 0,
    count_tokens: Callable[[str], int] | None = None,
) -> list[list[SpanJob]]:
    """Pack REPLACE jobs. Token budget wins when a Qwen counter is provided."""
    if count_tokens and max_prompt_tokens > 0:
        return _pack_span_jobs_tokens(
            jobs,
            max_prompt_tokens=max_prompt_tokens,
            prefix_tokens=prefix_tokens,
            count_tokens=count_tokens,
        )
    packed: list[list[SpanJob]] = []
    current: list[SpanJob] = []
    size = 0
    for job in jobs:
        extra = job_char_cost(job)
        if extra > max_chars:
            if current:
                packed.append(current)
                current, size = [], 0
            packed.append([job])
            continue
        if current and size + extra > max_chars:
            packed.append(current)
            current, size = [], 0
        current.append(job)
        size += extra
    if current:
        packed.append(current)
    return packed


def _pack_span_jobs_tokens(
    jobs: list[SpanJob],
    *,
    max_prompt_tokens: int,
    prefix_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[list[SpanJob]]:
    budget = max(32, max_prompt_tokens - max(0, prefix_tokens))
    packed: list[list[SpanJob]] = []
    current: list[SpanJob] = []
    tokens = 0
    for job in jobs:
        extra_tokens = count_tokens(format_span_job(job, 1)) + 2
        if extra_tokens > budget:
            if current:
                packed.append(current)
                current, tokens = [], 0
            packed.append([job])
            continue
        if current and tokens + extra_tokens > budget:
            packed.append(current)
            current, tokens = [], 0
        current.append(job)
        tokens += extra_tokens
    if current:
        packed.append(current)
    return packed


def trim_echo(output: str, source: str, before: str, after: str) -> str:
    """Drop KEEP context the model echoed around a REPLACE span."""
    text = output.strip()
    before_s = before.strip()
    after_s = after.strip()
    if before_s and text.startswith(before_s):
        text = text[len(before_s) :].lstrip()
    if after_s and text.endswith(after_s):
        text = text[: -len(after_s)].rstrip()
    if source.strip() and source.strip() in text and len(text) > len(source) * 2.4:
        # Model returned the whole passage; keep the original span rather than
        # splicing a rewrite we cannot locate.
        return source
    return text or output.strip()


_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_QUOTED_CJK_RE = re.compile(r'[“”"\'「」『』]([\u3400-\u9fff]+)[“”"\'「」『』]')
_LEAK_RE = re.compile(
    r"\[(?:\d+|n)\]|"
    r"KEEP (?:before|after)|"
    r"^\s*REPLACE\s*:|"
    r"\bdo not (?:add|remove|change|repeat|invent)\b|"
    r"\bthis is the correct format\b|"
    r"\bonly the replace text\b|"
    r"\bkeep the same \[n\]",
    re.I | re.M,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "out",
    "said",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
}


def span_length_ok(source: str, output: str) -> bool:
    if not output.strip():
        return False
    ratio = len(output) / max(len(source), 1)
    return 0.45 <= ratio <= 2.2


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u3400-\u9fff]", text)
    return [w.casefold() for w in words if w.casefold() not in _STOPWORDS and len(w) > 1]


def _stems_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 4 and len(right) >= 4 and (left.startswith(right) or right.startswith(left))


def _overlap_ok(source: str, output: str) -> bool:
    src = _content_words(source)
    out = _content_words(output)
    if len(src) < 4:
        return True
    hits = sum(1 for word in src if any(_stems_match(word, other) for other in out))
    return hits / len(src) >= 0.35


def _preserves_anchors(source: str, output: str) -> bool:
    haystack = re.sub(r"'s\b", "", output)
    for name in _NAME_RE.findall(re.sub(r"'s\b", "", source)):
        if name.casefold() not in haystack.casefold():
            return False
    for glyph in _QUOTED_CJK_RE.findall(source):
        if glyph not in output:
            return False
    return True


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [p for p in re.split(r'(?<=[.!?])(?:["\'”’)]+)?\s+', stripped) if p.strip()]
    return max(1, len(parts))


def _echo_wrap(source: str, output: str) -> bool:
    src = " ".join(source.split())
    out = " ".join(output.split())
    return bool(src) and src in out and len(out) >= len(src) + 24


def _looks_truncated(text: str, source: str = "") -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped.count('"') % 2 == 1:
        return True
    if stripped.count("“") != stripped.count("”"):
        return True
    if re.search(r"[,:;]\s*$", stripped):
        return True
    last = re.findall(r"[A-Za-z']+$", stripped)
    if last and last[0].casefold() in _STOPWORDS | {"i'll", "we'll", "he'd", "she'd", "it's"}:
        return True
    src_end = (source or "").rstrip()[-1:]
    if src_end in '.!?。"”’' and re.search(r"[A-Za-z\u3400-\u9fff]$", stripped):
        if not re.search(r'[.!?。…]["\'”’)]*$', stripped):
            return True
    return False


def _duplicate_takes(text: str) -> bool:
    parts = [p.strip() for p in re.split(r"(?:\s*\[\d+\]\s*|\n\n+)", text) if p.strip()]
    if len(parts) >= 2:
        head = parts[0][:48]
        if len(head) >= 20 and any(part.startswith(head[:24]) for part in parts[1:]):
            return True
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) >= 2 and len(sentences[0]) >= 20:
        return any(s.startswith(sentences[0][:24]) for s in sentences[1:])
    return False


def replacement_ok(source: str, output: str) -> bool:
    """Keep the original span unless the model returned a faithful, complete polish."""
    text = (output or "").strip()
    if not text or not span_length_ok(source, text):
        return False
    if _LEAK_RE.search(text):
        return False
    if _echo_wrap(source, text):
        return False
    if _sentence_count(text) > _sentence_count(source) + 1:
        return False
    if _looks_truncated(text, source):
        return False
    if _duplicate_takes(text):
        return False
    if not _preserves_anchors(source, text):
        return False
    if not _overlap_ok(source, text):
        return False
    return True
