"""Small project-local .env loader."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(*, override: bool = False) -> Path | None:
    """Load the nearest project .env without requiring python-dotenv."""

    env_path = _find_env_file()
    if env_path is None:
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _unquote(value.strip())
    return env_path


def _find_env_file() -> Path | None:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    seen: set[Path] = set()
    for start in candidates:
        for path in [start, *start.parents]:
            if path in seen:
                continue
            seen.add(path)
            env_path = path / ".env"
            if env_path.is_file():
                return env_path
    return None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
