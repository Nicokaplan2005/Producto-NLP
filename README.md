# PR Feature Extraction Pipeline — Checkpoint

## Qué es esto

Pipeline que extrae 26 features semánticas de PRs de GitHub usando un LLM (Gemma 4 26B via Vertex AI). Para cada PR procesa el diff junto con una "carta" de repo (contexto estable del repositorio) y devuelve un JSON con las features.

Para PRs mergeados también actualiza la carta incrementalmente (card_patch), de forma que la carta refleja el estado del repo en cada momento de la historia.

## Estado de este checkpoint

- **Dataset**: `data/pr_features.csv` — 19,174 filas, 35 repos, bloque 1 (parcial)
- **Schema**: v1.3 — 26 features definidas en `pipeline/schemas.py`
- **Pendiente para completar bloque 1**: nodejs/node (4,650 PRs) + scrapy/scrapy (539 PRs)
- **Modelo**: `google/gemma-4-26b-a4b-it-maas` via Vertex AI MaaS
- Ver `run_info.json` para hashes, hiperparámetros y stats completos

## Cómo reproducir

### Requisitos

```bash
pip install -r requirements.txt  # pydantic, pandas, requests, truststore
```

Variables de entorno necesarias:
```
GEMINI_API_KEY=<tu api key de Vertex>
VERTEX_ENDPOINT=<endpoint del modelo>
LLM_BACKEND=vertex_gemma
```

### Correr un repo solo

```bash
python -m scripts.run_block \
  --csv pr_index.csv \
  --cards-dir data/cards \
  --output-dir data/output \
  --repo scrapy/scrapy \
  --bloque 1
```

### Correr todos los repos en paralelo (un proceso por repo)

```bash
python -m scripts.launch_parallel --bloque 1 --budget-usd 50
```

El script lanza un subprocess por repo que tenga carta en `data/cards/`. Cada proceso escribe en `data/output/pr_features.csv` y actualiza su carta. Si se interrumpe, al relanzar retoma desde donde estaba (chequea filas ya escritas).

### Hiperparámetros configurables por env var

| Variable | Default | Descripción |
|---|---|---|
| `PR_PIPELINE_MAX_DIFF_CHARS` | 40000 | Truncación del diff |
| `PR_PIPELINE_MAX_CARD_CHARS` | 30000 | Truncación de la carta |
| `PR_PIPELINE_MAX_OUTPUT_TOKENS` | 14000 | Tokens de salida del LLM |
| `GEMINI_RETRIES` | 3 | Reintentos ante 429/503 |
| `LLM_USAGE_LOG_PATH` | data/output/llm_usage.jsonl | Log de tokens y costo |

## Estructura del dataset

`data/pr_features.csv` — join key: `repo` + `pr_number`

| Columna | Tipo | Descripción |
|---|---|---|
| `pr_url` | string | URL canónica de la PR |
| `repo` | string | owner/repo |
| `pr_number` | int | Número de PR |
| `bloque` | int | Bloque de procesamiento |
| `merge` | 0/1 | Si el PR fue mergeado |
| `elapsed_seconds` | float | Tiempo de la llamada LLM |
| `inferred_change_intent` | categorical | bug_fix / hotfix / feature / refactor / test / docs / migration / config / cleanup |
| `semantic_risk_level` | categorical | low / medium / high / critical |
| `change_scope` | categorical | focused / cross_module / broad |
| `error_path_handling` | categorical | none / weak / adequate / thorough |
| `backward_compatibility_risk` | categorical | none / low / medium / high |
| `stated_vs_actual_intent_match` | categorical | match / partial / mismatch / unclear |
| `diff_addresses_stated_problem` | categorical | yes / no / unknown |
| `implementation_completeness` | categorical | complete / partial / superficial / unknown |
| `test_semantic_relevance` | categorical | none / weak / partial / strong |
| `missing_regression_test` | categorical | true / false / not_applicable |
| `coupling_risk_semantic` | categorical | low / medium / high |
| `abstraction_level_fit` | categorical | too_low / appropriate / too_high |
| `follows_existing_repo_patterns` | categorical | yes / no / unknown |
| `reinvents_existing_functionality` | categorical | yes / no / unknown |
| `missing_update_to_related_files` | categorical | yes / no / unknown |
| `lack_of_contextual_adaptation` | categorical | low / medium / high |
| `mixed_concerns` | bool | |
| `unexplained_changes_present` | bool | |
| `affects_api_contract` | bool | |
| `security_sensitive_change` | bool | |
| `incomplete_integration` | bool | |
| `missing_edge_case_tests` | bool | |
| `touches_high_risk_area` | bool | |
| `risk_domains` | list | security / auth / payments / data_integrity / performance / concurrency / api_contract / privacy / observability / configuration |
| `likely_missing_cases` | list | null_handling / empty_input / permissions_check / concurrency / timeout / retry_logic / partial_failure / data_validation / rollback / migration_edge_case |

## Notas de diseño

- **Sin leakage de merge status**: los prompts no le dicen al modelo si el PR fue mergeado o no. Esto se validó con un A/B test de 50 PRs (acuerdo >88% en todos los features entre prompts con y sin señal de merge).
- **Cartas incrementales**: cada PR mergeado actualiza la carta del repo, capturando la evolución del codebase. Las cartas NO están en este checkpoint (son grandes y derivables reprocesando desde el principio con los diffs).
- **Recuperación ante fallos**: el CSV se escribe fila a fila; si el proceso se interrumpe, al relanzar retoma desde la última fila exitosa.
