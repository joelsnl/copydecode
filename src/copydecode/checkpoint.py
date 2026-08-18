"""Resume state for long runs.

Stored as JSON Lines: one header line with the run fingerprint, then one line
per completed chunk. Saving a chunk appends a single line, so cost per chunk
is constant instead of rewriting the whole file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class Checkpoint:
    def __init__(self, path: Path | None, meta: dict[str, Any]) -> None:
        self.path = path
        self.meta = meta
        self.chunks: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._header_written = False
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            return
        if header.get("fingerprint") != self.meta.get("fingerprint"):
            return
        chunks: dict[str, list[str]] = {}
        for line in lines[1:]:
            # A crash mid-append can truncate the final line; skip it and
            # anything else that does not parse.
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = record.get("id")
            texts = record.get("texts")
            if isinstance(chunk_id, str) and isinstance(texts, list):
                chunks[chunk_id] = [str(t) for t in texts]
        self.chunks = chunks
        self._header_written = True

    def done(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self.chunks

    def get(self, chunk_id: str) -> list[str]:
        with self._lock:
            return self.chunks[chunk_id]

    def save_chunk(self, chunk_id: str, texts: list[str]) -> None:
        with self._lock:
            self.chunks[chunk_id] = texts
            if self.path is None:
                return
            if not self._header_written:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                header = {"fingerprint": self.meta.get("fingerprint"), "meta": self.meta}
                self.path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
                self._header_written = True
            line = json.dumps({"id": chunk_id, "texts": texts}, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(path.stat().st_size).encode("utf-8"))
    digest.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]
