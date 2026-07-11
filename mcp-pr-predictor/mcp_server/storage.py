"""
Persistencia SQLite de predicciones del MCP server.

- features_snapshot: JSON del DataFrame de features, usado para calcular SHAP on-demand.
- shap_cache: JSON de top_factors, se llena la primera vez que el usuario pide SHAP.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
# DB_PATH puede sobreescribirse con la env var PREDICTIONS_DB_PATH
# (útil en Render con disco persistente montado en /var/data)
DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", str(ROOT / "data" / "predictions.db")))


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pr_predictions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            pr_url                TEXT    NOT NULL,
            repo                  TEXT,
            pr_number             TEXT,
            processed_at          TEXT    NOT NULL,
            merge_probability     REAL,
            not_merge_probability REAL,
            label                 TEXT,
            confidence            TEXT,
            features_snapshot     TEXT,
            shap_cache            TEXT,
            semantic_features     TEXT
        )
    """)
    # Migración segura: agrega columnas nuevas si ya existía la tabla sin ellas
    for col_def in ("features_snapshot TEXT", "shap_cache TEXT", "semantic_features TEXT"):
        try:
            conn.execute(f"ALTER TABLE pr_predictions ADD COLUMN {col_def}")
        except Exception:
            pass
    conn.commit()
    conn.close()


def _to_json_safe(d: dict) -> str:
    """Serializa un dict a JSON convirtiendo NaN→null y tipos numpy."""
    safe: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            safe[k] = None
        elif hasattr(v, "item"):  # numpy scalar
            safe[k] = v.item()
        else:
            safe[k] = v
    return json.dumps(safe, ensure_ascii=False)


def save_prediction(
    pr_url: str,
    repo: str,
    pr_number: str,
    result: dict,
    semantic_dict: dict,
    features_snapshot: dict,
) -> int:
    """Guarda la predicción y devuelve el id insertado."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.execute(
        """
        INSERT INTO pr_predictions
            (pr_url, repo, pr_number, processed_at,
             merge_probability, not_merge_probability, label, confidence,
             features_snapshot, shap_cache, semantic_features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        """,
        (
            pr_url,
            repo,
            str(pr_number),
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            result["merge_probability"],
            result["not_merge_probability"],
            result["label"],
            result["confidence"],
            _to_json_safe(features_snapshot),
            json.dumps(semantic_dict, ensure_ascii=False),
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_prediction_features(prediction_id: int) -> dict | None:
    """Devuelve el dict de features guardado para una predicción, o None si no existe."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT features_snapshot FROM pr_predictions WHERE id = ?",
        (prediction_id,),
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def get_cached_shap(prediction_id: int) -> list | None:
    """Devuelve SHAP cacheado o None si todavía no fue calculado."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT shap_cache FROM pr_predictions WHERE id = ?",
        (prediction_id,),
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def save_shap_cache(prediction_id: int, top_factors: list) -> None:
    """Persiste el resultado de SHAP para no recalcular la próxima vez."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE pr_predictions SET shap_cache = ? WHERE id = ?",
        (json.dumps(top_factors, ensure_ascii=False), prediction_id),
    )
    conn.commit()
    conn.close()


def get_predictions(limit: int = 500) -> list[dict]:
    """Devuelve las predicciones ordenadas por fecha (más recientes primero)."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, pr_url, repo, pr_number, processed_at,
               merge_probability, not_merge_probability, label, confidence,
               CASE WHEN shap_cache IS NOT NULL THEN 1 ELSE 0 END AS shap_ready,
               CASE WHEN features_snapshot IS NOT NULL THEN 1 ELSE 0 END AS has_features
        FROM pr_predictions
        ORDER BY processed_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
