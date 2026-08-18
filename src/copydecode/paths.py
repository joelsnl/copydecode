from __future__ import annotations

import os
import platform
from pathlib import Path


def env_value(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names``."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return default


def cache_dir() -> Path:
    env = env_value("COPYDECODE_CACHE")
    if env:
        path = Path(env)
    elif platform.system() == "Windows":
        path = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "copydecode"
    else:
        path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "copydecode"
    path.mkdir(parents=True, exist_ok=True)
    return path
