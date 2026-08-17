from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def same_text(left: str, right: str) -> bool:
    return " ".join(left.split()) == " ".join(right.split())


def _fence(text: str) -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}text\n{text.rstrip()}\n{ticks}"


@dataclass
class Edit:
    chapter_id: str
    chapter_title: str
    href: str
    para: int
    kind: str  # span | paragraph
    before: str
    after: str


@dataclass
class ChangeLog:
    input_name: str = ""
    output_name: str = ""
    mode: str = ""
    model: str = ""
    engine: str = ""
    edits: list[Edit] = field(default_factory=list)
    identical: int = 0
    sent: int = 0
    glossary: dict[str, int] = field(default_factory=dict)
    unchanged: list[Edit] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_glossary(self, counts: dict[str, int]) -> None:
        if not counts:
            return
        with self._lock:
            for key, n in counts.items():
                self.glossary[key] = self.glossary.get(key, 0) + n

    def record(
        self,
        *,
        chapter_id: str,
        chapter_title: str,
        href: str,
        para: int,
        before: str,
        after: str,
        kind: str = "span",
    ) -> None:
        with self._lock:
            self.sent += 1
            if same_text(before, after):
                self.identical += 1
                self.unchanged.append(
                    Edit(
                        chapter_id=chapter_id,
                        chapter_title=chapter_title or href,
                        href=href,
                        para=para,
                        kind=kind,
                        before=before,
                        after=after,
                    )
                )
                return
            self.edits.append(
                Edit(
                    chapter_id=chapter_id,
                    chapter_title=chapter_title or href,
                    href=href,
                    para=para,
                    kind=kind,
                    before=before,
                    after=after,
                )
            )

    def summary(self) -> dict[str, Any]:
        chapters = sorted({edit.chapter_id for edit in self.edits})
        return {
            "input": self.input_name,
            "output": self.output_name,
            "mode": self.mode,
            "model": self.model,
            "engine": self.engine,
            "llm_spans_sent": self.sent,
            "llm_unchanged": self.identical,
            "llm_edits": len(self.edits),
            "chapters_edited": len(chapters),
            "glossary": dict(sorted(self.glossary.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.summary()
        payload["edits"] = [
            {
                "chapter_id": edit.chapter_id,
                "chapter": edit.chapter_title,
                "href": edit.href,
                "para": edit.para,
                "kind": edit.kind,
                "before": edit.before,
                "after": edit.after,
            }
            for edit in self._sorted_edits()
        ]
        payload["unchanged"] = [
            {
                "chapter_id": edit.chapter_id,
                "chapter": edit.chapter_title,
                "href": edit.href,
                "para": edit.para,
                "kind": edit.kind,
                "before": edit.before,
            }
            for edit in self.unchanged
        ]
        return payload

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            f"# {summary['mode'].title() or 'Polish'} log",
            "",
            f"- Source: `{summary['input']}`",
            f"- Output: `{summary['output']}`",
            f"- Model: `{summary['model']}`",
        ]
        if summary.get("engine"):
            lines.append(f"- Engine: `{summary['engine']}`")
        lines.extend(
            [
                f"- LLM spans/paragraphs sent: {summary['llm_spans_sent']}",
                f"- Actually rewritten: **{summary['llm_edits']}** "
                f"(model left {summary['llm_unchanged']} unchanged)",
                f"- Chapters with edits: {summary['chapters_edited']}",
            ]
        )
        if summary["glossary"]:
            lines.append("- Glossary substitutions:")
            for term, count in list(summary["glossary"].items())[:40]:
                lines.append(f"  - {term}: {count}")
            extra = len(summary["glossary"]) - 40
            if extra > 0:
                lines.append(f"  - … {extra} more terms")
        lines.append("")
        if not self.edits:
            lines.append("No LLM rewrites were applied.")
            return "\n".join(lines) + "\n"

        grouped: dict[tuple[str, str, str], list[Edit]] = defaultdict(list)
        for edit in self._sorted_edits():
            grouped[(edit.href, edit.chapter_title, edit.chapter_id)].append(edit)
        for (href, title, _cid), items in grouped.items():
            lines.append(f"## {title}")
            lines.append(f"*{href} · {len(items)} edit(s)*")
            lines.append("")
            for edit in items:
                lines.append(f"### Paragraph {edit.para + 1} ({edit.kind})")
                lines.append("")
                lines.append("Before")
                lines.append(_fence(edit.before))
                lines.append("After")
                lines.append(_fence(edit.after))
                lines.append("")
        return "\n".join(lines)

    def write(self, markdown_path: Path, json_path: Path | None = None) -> None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        target = json_path or markdown_path.with_suffix(".json")
        target.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _sorted_edits(self) -> list[Edit]:
        def key(edit: Edit) -> tuple[int, str, int, str]:
            nums = re.findall(r"\d+", edit.href)
            order = int(nums[-1]) if nums else 0
            return (order, edit.href, edit.para, edit.kind)

        return sorted(self.edits, key=key)
