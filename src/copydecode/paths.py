from __future__ import annotations

import os
import platform
from pathlib import Path


def env_value(*names: str, default: str = "") -> str:
    """First non-empty environment value. COPYDECODE_* wins over legacy NOVELPOLISHER_*."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return default


def cache_dir() -> Path:
    env = env_value("COPYDECODE_CACHE", "NOVELPOLISHER_CACHE")
    if env:
        path = Path(env)
    else:
        if platform.system() == "Windows":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        preferred = base / "copydecode"
        legacy = base / "novelpolisher"
        path = preferred if preferred.exists() or not legacy.exists() else legacy
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"
