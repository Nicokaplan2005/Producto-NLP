# Ejecutar el experimento con Poetry

```powershell
poetry install
poetry run python scripts/run_notebook.py --mlflow
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

El runner ejecuta `experiments/pr_feature_ablation.ipynb` por default, guarda
los outputs en ese mismo notebook, usa `hyperparams.yml` y registra metricas/artifacts
en `mlflow.db` + `mlruns/`.

Para comparar escenarios en MLflow, abrir el experimento `pr_feature_ablation` y mirar
las metricas con prefijo:

- `pr_dataset_only.*`
- `pr_features_only.*`
- `combined.*`

Los modelos se guardan manualmente con nombres descriptivos:

- `model_pr_dataset_only`
- `model_pr_features_only`
- `model_combined`
