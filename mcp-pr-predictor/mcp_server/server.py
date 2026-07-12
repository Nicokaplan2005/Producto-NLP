"""
MCP Server — PR Merge Predictor
================================
Expone un único tool: predict_pr_merge

Uso (stdio, para Claude Desktop / Claude Code):
    python mcp_server/server.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

# Asegurar que el root del proyecto esté en el path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from extract_pr_features import PRFeatureExtractor
import pandas as pd
from mcp_server.predictor import (
    assemble_features,
    explain,
    get_active_model_id,
    get_model_status,
    predict,
    set_active_model,
)
from mcp_server import storage
from mcp_server.dashboard import generate_html
from mcp_server.runtime_settings import automatic_decision, get_settings, update_settings
from pipeline.schemas import EnhancedPRFeatures

# Inicializar DB al arrancar el server
storage.init_db()

# ── Instrucciones que Claude lee al conectar el server ────────────────────────
INSTRUCTIONS = """
# PR Merge Predictor

Predice la probabilidad de que un Pull Request sea mergeado, combinando
métricas de código (GitHub API) con features semánticas que TÚ extraés
del contenido del PR.

## Cuándo usar este tool
Cuando el usuario te pida analizar o evaluar una Pull Request y quiera
saber si es probable que sea aceptada/mergeada.

## Tu responsabilidad antes de llamar el tool
Antes de invocar `predict_pr_merge` DEBES:
1. Leer el título, body y diff del PR (necesitás acceso al repo).
2. Extraer las 26 features semánticas del PR siguiendo las definiciones
   detalladas al final de estas instrucciones.
3. Si no tenés acceso al repo, pedile al usuario un GitHub token con
   scope `repo` antes de continuar.

## Formato de llamada
Llamá `predict_pr_merge` con:
- `pr_url`: URL completa de la PR (ej: https://github.com/owner/repo/pull/123)
- `github_token`: token de acceso a GitHub (scope `repo` para repos privados)
- `semantic_features`: objeto JSON con las 26 features que extraíste

## Si el tool devuelve errores de validación
El server valida tu JSON contra el schema. Si hay errores, corrígelos y
volvé a llamar. Los errores indican exactamente qué campo falló y por qué.

## Modo automatico y explicaciones negativas
Si `predict_pr_merge` devuelve `auto_decision: "no_merge"` junto con
`negative_explanation_prompt` y `prediction_id`, usa ese prompt para redactar
una explicacion clara para el usuario. Despues llama el tool
`save_negative_explanation` con ese `prediction_id` y tu explicacion final.

---

## Definición de las 26 features semánticas

Usá SÓLO los vocabularios cerrados indicados. Para listas (multi-hot),
devolvé un array JSON con cero o más valores permitidos.

1. **inferred_change_intent** — intención principal del PR
   Valores: `bug_fix` | `hotfix` | `feature` | `refactor` | `test` |
   `docs` | `migration` | `config` | `cleanup`
   Hotfix = fix urgente en producción. Bug_fix = corrección ordinaria.

2. **stated_vs_actual_intent_match** — el título/body coincide con el diff?
   Valores: `match` | `partial` | `mismatch` | `unclear`

3. **mixed_concerns** — el PR mezcla objetivos no relacionados? (bool)

4. **diff_addresses_stated_problem** — el código realmente resuelve el problema?
   Valores: `yes` | `no` | `unknown`

5. **unexplained_changes_present** — hay cambios sin justificación? (bool)

6. **semantic_risk_level** — riesgo global de mergear
   Valores: `low` | `medium` | `high` | `critical`

7. **risk_domains** — dominios sensibles afectados (lista, puede estar vacía)
   Valores permitidos: `security` | `auth` | `payments` | `data_integrity` |
   `performance` | `concurrency` | `api_contract` | `privacy` |
   `observability` | `configuration`

8. **affects_api_contract** — cambia interfaces públicas, rutas, schemas? (bool)

9. **backward_compatibility_risk** — riesgo de romper comportamiento existente
   Valores: `none` | `low` | `medium` | `high`

10. **breaks_existing_assumption** — viola invariantes implícitos?
    Valores: `yes` | `no` | `unknown`

11. **security_sensitive_change** — toca authn/authz, permisos, secrets? (bool)

12. **implementation_completeness** — qué tan completa es la implementación?
    Valores: `complete` | `partial` | `superficial` | `unknown`

13. **error_path_handling** — calidad del manejo de errores en código de producción
    Valores: `none` | `weak` | `adequate` | `thorough`

14. **incomplete_integration** — integración parcial con servicio/componente externo? (bool)

15. **likely_missing_cases** — escenarios relevantes que parece faltar (lista)
    Valores: `null_handling` | `empty_input` | `permissions_check` |
    `concurrency` | `timeout` | `retry_logic` | `partial_failure` |
    `data_validation` | `rollback` | `migration_edge_case`

16. **test_semantic_relevance** — los tests verifican el comportamiento cambiado?
    Valores: `none` | `weak` | `partial` | `strong`

17. **missing_regression_test** — falta test de regresión? (solo para bug_fix/hotfix)
    Valores: `true` | `false` | `not_applicable`
    Usá `not_applicable` cuando inferred_change_intent NO es bug_fix ni hotfix.

18. **missing_edge_case_tests** — faltan tests para casos borde? (bool)

19. **coupling_risk_semantic** — riesgo de aumento de acoplamiento
    Valores: `low` | `medium` | `high`

20. **abstraction_level_fit** — el cambio está en el nivel correcto del stack?
    Valores: `too_low` | `appropriate` | `too_high`

21. **follows_existing_repo_patterns** — sigue los patrones del repo?
    Valores: `yes` | `no` | `unknown`

22. **reinvents_existing_functionality** — recrea algo que ya existe en el repo?
    Valores: `yes` | `no` | `unknown`

23. **missing_update_to_related_files** — faltan actualizaciones a archivos relacionados?
    Valores: `yes` | `no` | `unknown`

24. **lack_of_contextual_adaptation** — ignora convenciones específicas del repo?
    Valores: `low` | `medium` | `high`

25. **change_scope** — amplitud del cambio
    Valores: `focused` | `cross_module` | `broad`
    focused = un módulo; cross_module = 2-3 módulos; broad = 4+ módulos o
    infraestructura central.

26. **touches_high_risk_area** — toca paths de alta criticidad del repo? (bool)
    Marcá `true` si algún archivo modificado está en directorios de
    autenticación, pagos, datos críticos, infraestructura, o similares.
""".strip()

_port = int(os.getenv("PORT", "8000"))
mcp = FastMCP("PR Merge Predictor", instructions=INSTRUCTIONS, host="0.0.0.0", port=_port)


def _build_negative_prompt(
    pr_url: str,
    result: dict[str, Any],
    threshold: float,
    top_factors: list[dict],
) -> str:
    factors_text = "\n".join(
        f"- {f['feature']}: impacto {f['impact']} ({f['direction']})"
        for f in top_factors
    )
    return (
        "El MCP PR Merge Predictor decidio NO MERGE para esta PR.\n\n"
        f"PR: {pr_url}\n"
        f"Modelo: {result.get('model_name')} ({result.get('model_id')})\n"
        f"Probabilidad de merge: {result['merge_probability']:.3f}\n"
        f"Threshold automatico: {threshold:.3f}\n\n"
        "Factores SHAP principales:\n"
        f"{factors_text}\n\n"
        "Explicale al usuario por que el resultado es negativo y que cambios concretos "
        "podrian mejorar la PR. No repitas solo los numeros: conectalos con riesgos "
        "accionables, tests faltantes, alcance del cambio y posibles mejoras."
    )


# ── Tool principal ────────────────────────────────────────────────────────────

@mcp.tool()
def predict_pr_merge(
    pr_url: str,
    github_token: str,
    semantic_features: dict[str, Any],
) -> str:
    """
    Predice la probabilidad de merge de una PR.

    Requiere que hayas leído el PR y extraído las 26 features semánticas
    siguiendo las instrucciones del server. El server calcula las métricas
    de código vía GitHub API y corre el modelo XGBoost.

    Args:
        pr_url: URL completa de la PR. Ej: https://github.com/owner/repo/pull/42
        github_token: Token de GitHub. Necesario para repos privados (scope: repo).
        semantic_features: Dict con las 26 features semánticas que extraíste del PR.

    Returns:
        JSON con merge_probability, label, confidence y top_factors (SHAP).
    """
    # ── 1. Validar semantic_features contra el schema ─────────────────────────
    try:
        validated = EnhancedPRFeatures(**semantic_features)
    except Exception as e:
        return json.dumps({
            "error": "validation_failed",
            "message": (
                "Las semantic_features no pasaron la validación. "
                "Corregí los campos indicados y volvé a llamar el tool."
            ),
            "details": str(e),
        }, ensure_ascii=False)

    # ── 2. Extraer base__ features vía GitHub API ─────────────────────────────
    try:
        # Parsear URL: https://github.com/owner/repo/pull/123
        import re
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url.strip())
        if not m:
            return json.dumps({"error": "invalid_url",
                               "message": "La pr_url debe tener el formato https://github.com/owner/repo/pull/123"},
                              ensure_ascii=False)
        owner, repo_name, pr_number = m.group(1), m.group(2), int(m.group(3))

        extractor = PRFeatureExtractor(token=github_token)
        base_raw  = extractor.extract(owner, repo_name, pr_number)
    except Exception as e:
        msg = str(e)
        if "404" in msg or "Not Found" in msg:
            return json.dumps({
                "error": "repo_access_denied",
                "message": (
                    "No pude acceder al repo. Si es privado, pedile al usuario "
                    "un GitHub token con scope 'repo' y volvé a intentar."
                ),
                "details": msg,
            }, ensure_ascii=False)
        return json.dumps({
            "error": "github_api_error",
            "message": "Error al llamar la GitHub API.",
            "details": msg,
        }, ensure_ascii=False)

    # ── 3. Ensamblar features y predecir ──────────────────────────────────────
    try:
        semantic_dict = validated.model_dump()
        # Convertir listas a pipe-separated para los campos multi-hot
        for field in ("risk_domains", "likely_missing_cases"):
            if isinstance(semantic_dict.get(field), list):
                semantic_dict[field] = "|".join(semantic_dict[field])

        model_id = get_active_model_id()
        features_df = assemble_features(base_raw, semantic_dict, model_id=model_id)
        result      = predict(features_df, model_id=model_id)
        decision_settings = get_settings()
        result["decision_mode"] = decision_settings["mode"]
        result["decision_threshold"] = decision_settings["threshold"]
        if decision_settings["mode"] == "automatic":
            decision = automatic_decision(
                result["merge_probability"],
                decision_settings["threshold"],
            )
            result["auto_decision"] = decision
            if decision == "no_merge":
                top_factors = explain(features_df, model_id=model_id)
                result["top_factors"] = top_factors
                result["negative_explanation_prompt"] = _build_negative_prompt(
                    pr_url,
                    result,
                    decision_settings["threshold"],
                    top_factors,
                )
        else:
            result["auto_decision"] = None
    except Exception as e:
        return json.dumps({
            "error": "prediction_failed",
            "message": "Error interno al predecir.",
            "details": str(e),
        }, ensure_ascii=False)

    # ── 3b. Persistir en el dashboard (sin SHAP — se calcula on-demand) ──────────
    prediction_id = None
    try:
        features_snapshot = features_df.iloc[0].to_dict()
        prediction_id = storage.save_prediction(
            pr_url=pr_url,
            repo=f"{owner}/{repo_name}",
            pr_number=str(pr_number),
            result=result,
            semantic_dict=semantic_dict,
            features_snapshot=features_snapshot,
        )
    except Exception:
        pass  # no fallar si la DB tiene problema

    # ── 4. Formatear respuesta ────────────────────────────────────────────────
    pct = round(result["merge_probability"] * 100, 1)
    label_es = "PROBABLE MERGE" if result["label"] == "likely_merged" else "PROBABLE RECHAZO"

    response = {
        "pr_url":                pr_url,
        "prediction_id":         prediction_id,
        "label":                 label_es,
        "merge_probability":     result["merge_probability"],
        "not_merge_probability": result["not_merge_probability"],
        "confidence":            result["confidence"],
        "model_id":              result["model_id"],
        "model_name":            result["model_name"],
        "decision_mode":         result["decision_mode"],
        "decision_threshold":    result["decision_threshold"],
        "auto_decision":         result.get("auto_decision"),
        "summary": (
            f"El modelo predice {label_es} con {pct}% de probabilidad de merge "
            f"(confianza: {result['confidence']})."
        ),
        "note": "El desglose SHAP está disponible en el dashboard: http://localhost:8000/dashboard",
    }

    if result.get("negative_explanation_prompt"):
        response["negative_explanation_prompt"] = result["negative_explanation_prompt"]

    return json.dumps(response, ensure_ascii=False, indent=2)


# ── Dashboard routes ──────────────────────────────────────────────────────────

@mcp.tool()
def save_negative_explanation(prediction_id: int, explanation: str) -> str:
    """
    Guarda en el dashboard la explicacion final que Claude redacto para una
    prediccion automatica negativa.
    """
    ok = storage.save_negative_explanation(prediction_id, explanation.strip())
    if not ok:
        return json.dumps({
            "error": "prediction_not_found",
            "message": "No encontre una prediccion con ese prediction_id.",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "prediction_id": prediction_id,
        "message": "Explicacion guardada en el dashboard.",
    }, ensure_ascii=False)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard_handler(request: Request) -> HTMLResponse:
    return HTMLResponse(generate_html())


@mcp.custom_route("/api/predictions", methods=["GET"])
async def api_predictions(request: Request) -> JSONResponse:
    data = storage.get_predictions()
    return JSONResponse(data)


@mcp.custom_route("/api/models", methods=["GET"])
async def api_models(request: Request) -> JSONResponse:
    return JSONResponse(get_model_status())


@mcp.custom_route("/api/models/active", methods=["POST"])
async def api_set_active_model(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    model_id = str(payload.get("model_id", "")).strip()
    if not model_id:
        return JSONResponse({"error": "model_id_required"}, status_code=400)

    try:
        return JSONResponse(set_active_model(model_id))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@mcp.custom_route("/api/settings", methods=["GET"])
async def api_settings(request: Request) -> JSONResponse:
    return JSONResponse(get_settings())


@mcp.custom_route("/api/settings", methods=["POST"])
async def api_update_settings(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        threshold = payload.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        return JSONResponse(update_settings(payload.get("mode"), threshold))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        return JSONResponse({"error": "invalid_settings"}, status_code=400)


@mcp.custom_route("/api/shap/{id}", methods=["GET"])
async def api_shap(request: Request) -> JSONResponse:
    try:
        pred_id = int(request.path_params["id"])
    except (KeyError, ValueError):
        return JSONResponse({"error": "id inválido"}, status_code=400)

    # Si ya fue calculado antes, devolver cache
    cached = storage.get_cached_shap(pred_id)
    if cached is not None:
        return JSONResponse({"top_factors": cached, "cached": True})

    # Cargar features y calcular SHAP
    feat_dict = storage.get_prediction_features(pred_id)
    if feat_dict is None:
        return JSONResponse({"error": "predicción no encontrada o sin features"}, status_code=404)

    try:
        features_df  = pd.DataFrame([feat_dict])
        top_factors  = explain(features_df, model_id=storage.get_prediction_model_id(pred_id))
        storage.save_shap_cache(pred_id, top_factors)
        return JSONResponse({"top_factors": top_factors, "cached": False})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    mcp.run(transport=args.transport)
