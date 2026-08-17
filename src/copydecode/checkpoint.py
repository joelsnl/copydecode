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
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == meta.get("fingerprint"):
                self.chunks = saved.get("chunks", {})

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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": self.meta.get("fingerprint"),
                "meta": self.meta,
                "chunks": self.chunks,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(path.stat().st_size).encode("utf-8"))
    digest.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]
