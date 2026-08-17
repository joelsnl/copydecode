from __future__ import annotations

from collections.abc import Callable

from copydecode.epub_io import Segment


def split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    seps = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "]
    parts = [text]
    for sep in seps:
        next_parts: list[str] = []
        for part in parts:
            if len(part) <= max_chars:
                next_parts.append(part)
                continue
            pieces = part.split(sep)
            buf = ""
            for i, piece in enumerate(pieces):
                glue = sep if i < len(pieces) - 1 else ""
                candidate = buf + piece + glue
                if buf and len(candidate) > max_chars:
                    next_parts.append(buf)
                    buf = piece + glue
                else:
                    buf = candidate
            if buf:
                next_parts.append(buf)
        parts = next_parts
        if all(len(p) <= max_chars for p in parts):
            break
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        for i in range(0, len(part), max_chars):
            final.append(part[i : i + max_chars])
    return [p.strip() for p in final if p.strip()]


def pack_segments(
    segments: list[Segment],
    max_chars: int,
    *,
    max_prompt_tokens: int = 0,
    prefix_tokens: int = 0,
    count_tokens: Callable[[str], int] | None = None,
) -> list[list[Segment]]:
    packed: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    tokens = 0
    token_budget = (
        max(32, max_prompt_tokens - max(0, prefix_tokens))
        if count_tokens and max_prompt_tokens
        else 0
    )
    split_at = max_chars if max_chars > 0 else 8000
    for seg in segments:
        pieces = split_oversized(seg.text, split_at)
        if len(pieces) > 1:
            if current:
                packed.append(current)
                current, size, tokens = [], 0, 0
            for piece in pieces:
                packed.append(
                    [
                        Segment(
                            item_id=seg.item_id,
                            index=seg.index,
                            text=piece,
                            tag=seg.tag,
                        )
                    ]
                )
            continue
        extra_chars = len(seg.text) + 8
        extra_tokens = (count_tokens(f"[1]\n{seg.text}") + 2) if count_tokens else 0
        overflow = current and (
            (token_budget and tokens + extra_tokens > token_budget)
            or (not token_budget and size + extra_chars > split_at)
        )
        if overflow:
            packed.append(current)
            current, size, tokens = [], 0, 0
        current.append(seg)
        size += extra_chars
        tokens += extra_tokens
    if current:
        packed.append(current)
    return packed


def format_numbered(segments: list[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(f"[{i}]\n{seg.text}")
    return "\n\n".join(blocks)
