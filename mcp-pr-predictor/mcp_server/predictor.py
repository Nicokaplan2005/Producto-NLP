"""
Carga el modelo final y produce predicciones + explicaciones SHAP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from mcp_server.transforms import to_str_array  # noqa: F401 — necesario para deserializar el pickle

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "xgb_final.pkl"

# Cache global para no recargar en cada llamada
_PIPELINE: dict | None = None
_EXPLAINER: shap.TreeExplainer | None = None


def _load() -> dict:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = joblib.load(MODEL_PATH)
    return _PIPELINE


def _get_explainer(clf) -> shap.TreeExplainer:
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = shap.TreeExplainer(clf)
    return _EXPLAINER


# ── Ensamblado de features ────────────────────────────────────────────────────

PIPE_COLS = ["semantic__risk_domains", "semantic__likely_missing_cases"]


def assemble_features(
    base_features: dict[str, Any],
    semantic_features: dict[str, Any],
) -> pd.DataFrame:
    """
    Combina base__ y semantic__ features en un DataFrame de una fila,
    con los mismos nombres de columna que espera el modelo.
    """
    pipeline = _load()
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

def predict(features_df: pd.DataFrame) -> dict[str, Any]:
    """
    Devuelve probabilidad de merge, label y confianza. Sin SHAP.
    Llamar explain() por separado solo si se necesita la explicación.
    """
    pipeline   = _load()
    pre        = pipeline["preprocessor"]
    clf        = pipeline["classifier"]
    use_log    = pipeline["use_log"]
    use_ratios = pipeline["use_ratios"]

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

    return {
        "merge_probability":     round(prob_merged, 4),
        "not_merge_probability": round(prob_not_merged, 4),
        "label":                 label,
        "confidence":            confidence,
    }


def explain(features_df: pd.DataFrame) -> list[dict]:
    """
    Calcula SHAP para un DataFrame de una fila.
    Solo se llama on-demand (desde el dashboard o si el usuario lo pide).
    Devuelve lista de top_factors ordenada por impacto absoluto.
    """
    pipeline   = _load()
    pre        = pipeline["preprocessor"]
    clf        = pipeline["classifier"]
    use_log    = pipeline["use_log"]
    use_ratios = pipeline["use_ratios"]

    X   = _apply_fe(features_df.copy(), use_log, use_ratios)
    X_t = pre.transform(X)

    explainer   = _get_explainer(clf)
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
