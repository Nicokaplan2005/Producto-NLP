"""
Carga el modelo final y produce predicciones + explicaciones SHAP.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from mcp_server.transforms import to_str_array  # noqa: F401 — necesario para deserializar el pickle

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
DEFAULT_MODEL_ID = "combined_temporal"

# Cache global para no recargar en cada llamada
_REGISTRY: dict[str, Any] | None = None
_ACTIVE_MODEL_ID: str | None = None
_PIPELINES: dict[str, dict] = {}
_EXPLAINERS: dict[str, shap.TreeExplainer] = {}
_MODEL_LOCK = RLock()


def _load_registry() -> dict[str, Any]:
    global _REGISTRY
    if _REGISTRY is None:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            _REGISTRY = json.load(f)
    return _REGISTRY


def _registered_models() -> list[dict[str, Any]]:
    return list(_load_registry().get("models", []))


def _model_by_id(model_id: str) -> dict[str, Any] | None:
    for model in _registered_models():
        if model.get("id") == model_id:
            return model
    return None


def _model_path(model: dict[str, Any]) -> Path:
    path = Path(model["path"])
    if not path.is_absolute():
        path = MODELS_DIR / path
    return path


def _resolve_default_model_id() -> str:
    env_model_id = os.getenv("ACTIVE_MODEL_ID") or os.getenv("MODEL_ID")
    if env_model_id and _model_by_id(env_model_id):
        return env_model_id
    registry_default = _load_registry().get("default_model_id")
    if registry_default and _model_by_id(registry_default):
        return registry_default
    return DEFAULT_MODEL_ID


def get_active_model_id() -> str:
    global _ACTIVE_MODEL_ID
    with _MODEL_LOCK:
        if _ACTIVE_MODEL_ID is None:
            _ACTIVE_MODEL_ID = _resolve_default_model_id()
        return _ACTIVE_MODEL_ID


def _public_model_info(
    model: dict[str, Any],
    *,
    active_model_id: str,
    pipeline: dict | None = None,
) -> dict[str, Any]:
    path = _model_path(model)
    metrics = model.get("metrics") or {}
    if pipeline and pipeline.get("test_metrics"):
        metrics = pipeline["test_metrics"]
    feature_columns = pipeline.get("feature_columns", []) if pipeline else []
    return {
        "id": model["id"],
        "name": model.get("name", model["id"]),
        "description": model.get("description", ""),
        "path": str(path.relative_to(ROOT)),
        "available": path.exists(),
        "active": model["id"] == active_model_id,
        "metrics": metrics,
        "feature_count": len(feature_columns) if feature_columns else None,
    }


def list_models() -> list[dict[str, Any]]:
    active_model_id = get_active_model_id()
    return [
        _public_model_info(model, active_model_id=active_model_id)
        for model in _registered_models()
    ]


def get_model_status() -> dict[str, Any]:
    active_model_id = get_active_model_id()
    pipeline = _load(active_model_id)
    active_model = _model_by_id(active_model_id)
    if active_model is None:
        raise ValueError(f"Modelo no registrado: {active_model_id}")
    return {
        "active_model": _public_model_info(
            active_model,
            active_model_id=active_model_id,
            pipeline=pipeline,
        ),
        "models": list_models(),
    }


def set_active_model(model_id: str) -> dict[str, Any]:
    global _ACTIVE_MODEL_ID
    model = _model_by_id(model_id)
    if model is None:
        raise ValueError(f"Modelo no registrado: {model_id}")
    path = _model_path(model)
    if not path.exists():
        raise ValueError(f"Artefacto de modelo no encontrado: {path.name}")
    with _MODEL_LOCK:
        _load(model_id)
        _ACTIVE_MODEL_ID = model_id
    return get_model_status()


def _load(model_id: str | None = None) -> dict:
    resolved_model_id = model_id or get_active_model_id()
    model = _model_by_id(resolved_model_id)
    if model is None:
        raise ValueError(f"Modelo no registrado: {resolved_model_id}")
    path = _model_path(model)
    with _MODEL_LOCK:
        if resolved_model_id not in _PIPELINES:
            _PIPELINES[resolved_model_id] = joblib.load(path)
        return _PIPELINES[resolved_model_id]


def _get_explainer(clf, model_id: str) -> shap.TreeExplainer:
    with _MODEL_LOCK:
        if model_id not in _EXPLAINERS:
            _EXPLAINERS[model_id] = shap.TreeExplainer(clf)
        return _EXPLAINERS[model_id]


# ── Ensamblado de features ────────────────────────────────────────────────────

PIPE_COLS = ["semantic__risk_domains", "semantic__likely_missing_cases"]


def assemble_features(
    base_features: dict[str, Any],
    semantic_features: dict[str, Any],
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Combina base__ y semantic__ features en un DataFrame de una fila,
    con los mismos nombres de columna que espera el modelo.
    """
    pipeline = _load(model_id)
    expected_cols = pipeline["feature_columns"]

    row: dict[str, Any] = {}

    # base__ features — el extractor ya las devuelve con prefijo base__
    for col in expected_cols:
        if col.startswith("base__"):
            raw_key = col[len("base__"):]
            row[col] = base_features.get(raw_key, base_features.get(col, np.nan))

    # semantic__ features
    for col in expected_cols:
        if not col.startswith("semantic__"):
            continue
        raw_key = col[len("semantic__"):]
        val = semantic_features.get(raw_key, semantic_features.get(col))

        if col in PIPE_COLS:
            # list → "a|b|c"
            if isinstance(val, list):
                row[col] = "|".join(str(v) for v in val)
            else:
                row[col] = val or ""
        else:
            row[col] = val

    return pd.DataFrame([row], columns=expected_cols)


# ── Predicción ────────────────────────────────────────────────────────────────

def predict(features_df: pd.DataFrame, model_id: str | None = None) -> dict[str, Any]:
    """
    Devuelve probabilidad de merge, label y confianza. Sin SHAP.
    Llamar explain() por separado solo si se necesita la explicación.
    """
    resolved_model_id = model_id or get_active_model_id()
    pipeline   = _load(resolved_model_id)
    pre        = pipeline["preprocessor"]
    clf        = pipeline["classifier"]
    use_log    = pipeline.get("use_log", False)
    use_ratios = pipeline.get("use_ratios", False)

    X   = _apply_fe(features_df.copy(), use_log, use_ratios)
    X_t = pre.transform(X)

    prob_merged     = float(clf.predict_proba(X_t)[0, 1])
    prob_not_merged = 1.0 - prob_merged

    label = "likely_merged" if prob_merged >= 0.5 else "likely_rejected"
    confidence = (
        "high"   if abs(prob_merged - 0.5) > 0.25 else
        "medium" if abs(prob_merged - 0.5) > 0.10 else
        "low"
    )

    model = _model_by_id(resolved_model_id) or {"id": resolved_model_id}
    return {
        "merge_probability":     round(prob_merged, 4),
        "not_merge_probability": round(prob_not_merged, 4),
        "label":                 label,
        "confidence":            confidence,
        "model_id":              resolved_model_id,
        "model_name":            model.get("name", resolved_model_id),
    }


def explain(features_df: pd.DataFrame, model_id: str | None = None) -> list[dict]:
    """
    Calcula SHAP para un DataFrame de una fila.
    Solo se llama on-demand (desde el dashboard o si el usuario lo pide).
    Devuelve lista de top_factors ordenada por impacto absoluto.
    """
    resolved_model_id = model_id or get_active_model_id()
    pipeline   = _load(resolved_model_id)
    pre        = pipeline["preprocessor"]
    clf        = pipeline["classifier"]
    use_log    = pipeline.get("use_log", False)
    use_ratios = pipeline.get("use_ratios", False)

    X   = _apply_fe(features_df.copy(), use_log, use_ratios)
    X_t = pre.transform(X)

    explainer   = _get_explainer(clf, resolved_model_id)
    shap_values = explainer.shap_values(X_t)

    feature_names = _get_transformed_names(pre, X)
    shap_row      = shap_values[0]

    top_idx = np.argsort(np.abs(shap_row))[::-1][:7]
    top_factors = []
    for i in top_idx:
        name   = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        impact = float(shap_row[i])
        top_factors.append({
            "feature":   _humanize(name),
            "impact":    round(impact, 4),
            "direction": "hacia_merge" if impact > 0 else "contra_merge",
        })
    return top_factors


# ── Helpers internos ──────────────────────────────────────────────────────────

LOG_CANDIDATES = [
    "base__commit_add_line_sum", "base__commit_delete_line_sum",
    "base__commit_total_line_sum", "base__commit_file_change",
    "base__commit_add_line_max", "base__commit_delete_line_max",
    "base__before_pr_project_commits", "base__before_pr_project_prs",
    "base__before_pr_project_issues", "base__before_pr_user_commits",
    "base__before_pr_user_pulls", "base__before_pr_user_issues",
    "base__before_pr_project_issues_comment",
    "base__before_pr_project_comments_in_prs",
    "base__before_pr_user_followers",
    "base__everyday_pr_comment_count_in_lifetime__sum",
    "base__everyday_pr_commit_count_in_lifetime__sum",
]


def _apply_fe(X: pd.DataFrame, use_log: bool, use_ratios: bool) -> pd.DataFrame:
    for col in PIPE_COLS:
        if col in X.columns:
            X[col] = X[col].fillna("").apply(
                lambda v: len([x for x in str(v).split("|") if x.strip()])
            )
    if use_log:
        for col in LOG_CANDIDATES:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
    if use_ratios:
        if {"base__before_pr_user_commits", "base__before_pr_project_commits"} <= set(X.columns):
            X["ratio_user_proj_commits"] = (
                X["base__before_pr_user_commits"] /
                (X["base__before_pr_project_commits"] + 1)
            )
        if {"base__commit_add_line_sum", "base__commit_delete_line_sum"} <= set(X.columns):
            total = X["base__commit_add_line_sum"] + X["base__commit_delete_line_sum"]
            X["ratio_add_delete"] = X["base__commit_add_line_sum"] / (total + 1)
    return X


def _get_transformed_names(pre, X: pd.DataFrame) -> list[str]:
    names = []
    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "steps"):
            last = transformer.steps[-1][1]
            if hasattr(last, "get_feature_names_out"):
                raw = last.get_feature_names_out()
                # OHE produce "x0_val","x1_val" — reemplazar xi por nombre real de columna
                fixed = []
                for feat in raw:
                    parts = feat.split("_", 1)
                    if parts[0].startswith("x") and parts[0][1:].isdigit():
                        idx = int(parts[0][1:])
                        col_name = cols[idx] if idx < len(cols) else parts[0]
                        fixed.append(f"{col_name}_{parts[1]}" if len(parts) > 1 else col_name)
                    else:
                        fixed.append(feat)
                names += fixed
            else:
                names += list(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names += list(transformer.get_feature_names_out())
        else:
            names += list(cols)
    if len(names) == 0:
        names = [f"f{i}" for i in range(pre.transform(X).shape[1])]
    return names


def _humanize(name: str) -> str:
    """Convierte 'num__base__commit_add_line_sum' → 'commit_add_line_sum'."""
    # Quitar prefijos del ColumnTransformer y del dataset
    for prefix in ("num__base__", "cat__base__", "num__semantic__", "cat__semantic__",
                   "num__", "cat__", "base__", "semantic__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name
