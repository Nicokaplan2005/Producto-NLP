"""Shared file-access helpers used by agent tool loops."""

from __future__ import annotations

import json
from pathlib import Path


def _safe_repo_path(repo_dir: Path, path_str: str) -> Path:
    """Resolve path_str relative to repo_dir; absolute paths are returned as-is."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (repo_dir / p).resolve()


def _read_lines(
    file_path: Path,
    *,
    start_line: int = 1,
    max_lines: int = 100,
) -> dict[str, object]:
    """Read up to max_lines lines starting from start_line (1-indexed).

    Returns {"text": str, "start_line": int, "lines_returned": int, "total_lines": int}.
    """
    try:
        all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {
            "text": f"Error reading file: {exc}",
            "start_line": start_line,
            "lines_returned": 0,
            "total_lines": 0,
        }
    total = len(all_lines)
    idx = max(0, start_line - 1)
    selected = all_lines[idx: idx + max_lines]
    return {
        "text": "\n".join(selected),
        "start_line": start_line,
        "lines_returned": len(selected),
        "total_lines": total,
    }


def _truncate_obj(obj: dict, max_chars: int) -> dict:
    """Return obj unchanged if its JSON fits in max_chars, else a truncated placeholder."""
    serialized = json.dumps(obj, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return obj
    return {"_truncated": True, "_preview": serialized[:max_chars]}
