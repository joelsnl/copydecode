from __future__ import annotations

import json
import re

from copydecode.spans import SpanJob, format_span_job

NUMBERED_RE = re.compile(
    r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]\s*|\Z)",
    re.DOTALL,
)

POLISH_SYSTEM = """Edit machine-translated English. Keep meaning, names, register, and distinctive voice. Copy phrases that already read well. Do not recast the passage into another genre or domain. Return the SAME numbered [n] blocks and nothing else."""

SPAN_POLISH_SYSTEM = """Edit only REPLACE spans of machine-translated English. Keep meaning, names, register, and voice. KEEP lines are context — copy them in your head, do not output them. Do not recast the passage into a different genre or domain. Return the SAME numbered [n] blocks containing ONLY the edited REPLACE text."""

SPAN_POLISH_INSTRUCTIONS = (
    "Polish each REPLACE span. Output only the replacement for that span, "
    "with the same [n] labels. Do not repeat KEEP before/after text. "
    "Keep the source register and its own terminology; do not invent another domain's vocabulary."
)

TRANSLATE_SYSTEM = """Translate into fluent English. Keep meaning, names, and the source's register. Do not recast the text into a different genre or domain. Return the SAME numbered [n] blocks and nothing else."""

GLOSSARY_SYSTEM = """You extract a terminology glossary from source-language text or English machine translation.
Return ONLY JSON, no markdown:
{"terms":[{"source":"...","target":"...","type":"character|place|organization|technique|item|title|other","notes":""}]}
Rules:
- source is the form that appears in the input (original language or MTL English).
- target is the canonical English rendering to use everywhere.
- Person names: a conventional Latin spelling of the original name, not a calque (e.g. "Wang Lin" not "King Forest").
- Keep the text's own register and domain. Do not rewrite terms into another field's jargon.
- Skip ordinary words. Keep 15-60 high-value recurring terms.
"""

SOURCE_STYLE = (
    "Keep this text's own register, titles, names, and domain vocabulary. "
    "Do not recast it as a different genre or swap in another field's jargon."
)


def job_style(extra_style: str = "") -> str:
    extra = extra_style.strip()
    if extra:
        return f"{SOURCE_STYLE}\n{extra}"
    return SOURCE_STYLE


def build_user_prompt(
    numbered_text: str,
    glossary_block: str,
    previous: str,
    extra_style: str,
    mode: str,
) -> str:
    parts = []
    if glossary_block:
        parts.append(glossary_block)
    if extra_style:
        parts.append(f"Style notes:\n{extra_style}")
    if previous:
        parts.append(
            "Previously rewritten text (context only, do not repeat):\n" + previous
        )
    task = "Polish" if mode == "polish" else "Translate"
    parts.append(f"{task} these passages. Keep the same [n] labels:\n\n{numbered_text}")
    return "\n\n".join(parts)


def format_span_jobs(jobs: list[SpanJob]) -> str:
    return "\n\n".join(format_span_job(job, i) for i, job in enumerate(jobs, start=1))


def span_system_prompt(glossary_block: str = "", extra_style: str = "") -> str:
    """Byte-stable system prefix so vLLM/llama.cpp can hit the prefix cache."""
    parts = [SPAN_POLISH_SYSTEM]
    if extra_style:
        parts.append(f"Style notes:\n{extra_style}")
    if glossary_block:
        parts.append(glossary_block)
    return "\n\n".join(parts)


def span_prefix_text(glossary_block: str = "", extra_style: str = "") -> str:
    """llama.cpp prompt prefix (system + fixed user instructions, no spans)."""
    return (
        span_system_prompt(glossary_block, extra_style).rstrip()
        + "\n\n"
        + SPAN_POLISH_INSTRUCTIONS
        + "\n\nSpans:\n\n"
    )


def build_span_user_prompt(
    jobs: list[SpanJob],
    glossary_block: str = "",
    previous: str = "",
    extra_style: str = "",
) -> str:
    del glossary_block, previous, extra_style
    return SPAN_POLISH_INSTRUCTIONS + "\n\nSpans:\n\n" + format_span_jobs(jobs)


def parse_numbered(text: str, expected: int) -> list[str] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"\*+\[(\d+)\]\*+", r"[\1]", cleaned)
    matches = NUMBERED_RE.findall(cleaned)
    values = [_clean_span_block(m[1]) for m in matches]
    values = [v for v in values if v]
    if expected == 1:
        if values:
            return [values[0]]
        stripped = _clean_span_block(re.sub(r"^\[1\]\s*", "", cleaned))
        return [stripped] if stripped else None

    if len(values) == expected:
        return values

    blanks = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    blanks = [_clean_span_block(re.sub(r"^\[\d+\]\s*", "", p)) for p in blanks]
    blanks = [p for p in blanks if p]
    if len(blanks) == expected:
        return blanks
    return None


def _clean_span_block(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if re.fullmatch(r"(?:REPLACE|KEEP(?: before| after)?)\s*:?", line, re.I):
            continue
        line = re.sub(r"^(?:REPLACE|KEEP(?: before| after)?)\s*:\s*", "", line, flags=re.I)
        lines.append(line)
    text = "\n".join(lines).strip()
    text = re.split(
        r"(?:\nDo not (?:add|remove|change|repeat)\b|\(This is the correct format\b)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    text = re.sub(r"\s*\[\d+\]\s*", " ", text).strip()
    text = re.sub(r" +", " ", text)
    return text


def parse_glossary_json(text: str) -> list[dict]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    terms = data.get("terms", [])
    if not isinstance(terms, list):
        return []
    return [t for t in terms if isinstance(t, dict) and t.get("source") and t.get("target")]
