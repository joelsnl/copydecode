"""User-supplied terminology applied as deterministic string substitution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from copydecode.detect import FOREIGN_SCRIPT_RE


@dataclass
class Term:
    source: str
    target: str
    kind: str = "term"
    notes: str = ""


@dataclass
class Glossary:
    terms: list[Term] = field(default_factory=list)
    # (pattern, matched-text -> Term); rebuilt lazily after mutation.
    _index: tuple[re.Pattern[str], dict[str, Term]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def add(self, term: Term, overwrite: bool = False) -> None:
        key = term.source.casefold()
        for existing in self.terms:
            if existing.source.casefold() == key:
                if overwrite:
                    existing.target = term.target
                    existing.kind = term.kind or existing.kind
                    existing.notes = term.notes or existing.notes
                    self._index = None
                return
        self.terms.append(term)
        self._index = None

    def merge(self, other: Glossary, overwrite: bool = False) -> None:
        for term in other.terms:
            self.add(term, overwrite=overwrite)

    def relevant(self, text: str) -> list[Term]:
        haystack = text.casefold()
        hits = [t for t in self.terms if t.source and t.source.casefold() in haystack]
        hits.sort(key=lambda t: len(t.source), reverse=True)
        return hits[:80]

    def as_prompt(self, text: str) -> str:
        hits = self.relevant(text)
        if not hits:
            return ""
        lines = []
        for term in hits:
            extra = f" ({term.notes})" if term.notes else ""
            lines.append(f"- {term.source} → {term.target}{extra}")
        return "Glossary (use these exact renderings):\n" + "\n".join(lines)

    def as_stable_prompt(self, limit: int = 80) -> str:
        """Fixed glossary block for prefix-cache hits across every REPLACE pack."""
        seen: set[tuple[str, str]] = set()
        lines: list[str] = []
        terms = self._substitutions()
        terms.sort(key=lambda term: (-len(term.source), term.source.casefold()))
        for term in terms:
            key = (term.source.casefold(), term.target)
            if key in seen:
                continue
            seen.add(key)
            extra = f" ({term.notes})" if term.notes else ""
            lines.append(f"- {term.source} → {term.target}{extra}")
            if len(lines) >= limit:
                break
        if not lines:
            return ""
        return "Glossary (use these exact renderings):\n" + "\n".join(lines)

    def unapplied_hits(self, text: str) -> list[Term]:
        return [term for term in self.relevant(text) if term.source != term.target]

    def _substitutions(self) -> list[Term]:
        return [
            term
            for term in self.terms
            if term.source and term.target and term.source != term.target
        ]

    def _compiled(self) -> tuple[re.Pattern[str], dict[str, Term]] | None:
        """One alternation over all sources, longest first.

        A single pass means a term's *target* can never be re-matched by a
        later term ({"Wang": "Wang Lin", "Lin": "Forest"} must not turn "Wang"
        into "Wang Forest"). Replacement goes through a callable, so targets
        are never interpreted as regex templates.
        """
        if self._index is not None:
            return self._index
        terms = sorted(self._substitutions(), key=lambda t: len(t.source), reverse=True)
        if not terms:
            return None
        alternatives: list[str] = []
        lookup: dict[str, Term] = {}
        for term in terms:
            if FOREIGN_SCRIPT_RE.search(term.source):
                alternatives.append(re.escape(term.source))
            else:
                alternatives.append(rf"\b{re.escape(term.source)}\b")
            lookup.setdefault(term.source, term)
            lookup.setdefault(term.source.casefold(), term)
        pattern = re.compile("|".join(alternatives), re.IGNORECASE)
        self._index = (pattern, lookup)
        return self._index

    def _term_for(self, matched: str) -> Term | None:
        compiled = self._compiled()
        if compiled is None:
            return None
        _pattern, lookup = compiled
        return lookup.get(matched) or lookup.get(matched.casefold())

    def apply_to_text(self, text: str) -> str:
        compiled = self._compiled()
        if compiled is None:
            return text
        pattern, _lookup = compiled

        def substitute(match: re.Match[str]) -> str:
            term = self._term_for(match.group(0))
            return term.target if term else match.group(0)

        return pattern.sub(substitute, text)

    def hit_counts(self, text: str) -> dict[str, int]:
        compiled = self._compiled()
        if compiled is None:
            return {}
        pattern, _lookup = compiled
        counts: dict[str, int] = {}
        for match in pattern.finditer(text):
            term = self._term_for(match.group(0))
            if term is None:
                continue
            key = f"{term.source} → {term.target}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [
                {
                    "source": t.source,
                    "target": t.target,
                    "type": t.kind,
                    "notes": t.notes,
                }
                for t in self.terms
            ]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_glossary_file(path: str | Path) -> Glossary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return glossary_from_data(data)


def glossary_from_data(data: Any) -> Glossary:
    raw_terms = data.get("terms", data) if isinstance(data, dict) else data
    glossary = Glossary()
    if isinstance(raw_terms, dict):
        for source, target in raw_terms.items():
            glossary.add(Term(source=str(source), target=str(target)))
        return glossary
    for item in raw_terms:
        if isinstance(item, dict):
            source = str(item.get("source") or item.get("from") or "")
            target = str(item.get("target") or item.get("to") or "")
            if source and target:
                glossary.add(
                    Term(
                        source=source,
                        target=target,
                        kind=str(item.get("type") or item.get("kind") or "term"),
                        notes=str(item.get("notes") or ""),
                    )
                )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            glossary.add(Term(source=str(item[0]), target=str(item[1])))
    return glossary
