from __future__ import annotations

import os
from threading import RLock
from typing import Any

DEFAULT_THRESHOLD = 0.5
SUGGESTED_THRESHOLD = 0.5
VALID_MODES = {"manual", "automatic"}

_LOCK = RLock()
_MODE = os.getenv("DECISION_MODE", "manual").strip().lower()
_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", str(DEFAULT_THRESHOLD)))

if _MODE not in VALID_MODES:
    _MODE = "manual"
if not 0 <= _THRESHOLD <= 1:
    _THRESHOLD = DEFAULT_THRESHOLD


def get_settings() -> dict[str, Any]:
    with _LOCK:
        return {
            "mode": _MODE,
            "threshold": _THRESHOLD,
            "suggested_threshold": SUGGESTED_THRESHOLD,
            "modes": [
                {"id": "manual", "name": "No automatico"},
                {"id": "automatic", "name": "Automatico"},
            ],
        }


def update_settings(mode: str | None = None, threshold: float | None = None) -> dict[str, Any]:
    global _MODE, _THRESHOLD
    with _LOCK:
        if mode is not None:
            clean_mode = mode.strip().lower()
            if clean_mode not in VALID_MODES:
                raise ValueError("Modo invalido")
            _MODE = clean_mode
        if threshold is not None:
            clean_threshold = float(threshold)
            if not 0 <= clean_threshold <= 1:
                raise ValueError("El threshold debe estar entre 0 y 1")
            _THRESHOLD = clean_threshold
        return get_settings()


def automatic_decision(merge_probability: float, threshold: float) -> str:
    return "merge" if merge_probability >= threshold else "no_merge"
