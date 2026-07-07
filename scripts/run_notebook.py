import argparse
import asyncio
import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "experiments" / "pr_feature_ablation.ipynb"


def resolve_path(path: str | None, default: Path | None = None) -> Path | None:
    if path is None:
        return default
    value = Path(path)
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def inject_mlflow_cells(nb, notebook_path: Path, hyperparams_path: Path, tracking_uri: str, experiment: str, run_name: str):
    setup = f"""
    import json
    import platform
    from pathlib import Path

    import mlflow
    import yaml

    mlflow.set_tracking_uri({tracking_uri!r})
    mlflow.set_experiment({experiment!r})
    mlflow.start_run(run_name={run_name!r})
    mlflow.set_tags({{"notebook": {str(notebook_path)!r}, "python": platform.python_version()}})

    hyperparams_path = Path({str(hyperparams_path)!r})
    if hyperparams_path.exists():
        params = yaml.safe_load(hyperparams_path.read_text(encoding="utf-8")) or {{}}

        def flatten(prefix, value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield from flatten(f"{{prefix}}.{{key}}" if prefix else str(key), nested)
            else:
                yield prefix, value

        for key, value in flatten("", params):
            mlflow.log_param(key, json.dumps(value) if isinstance(value, list) else value)
        mlflow.log_artifact(str(hyperparams_path), artifact_path="inputs")
    """
    finish = """
    import hashlib
    import tempfile
    from pathlib import Path

    import mlflow
    import mlflow.xgboost
    import numpy as np

    def metric_name(value):
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/")
        return "".join(char if char in allowed else "_" for char in str(value)).strip("_")

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if "results" in globals():
        for scenario, row in results.iterrows():
            for metric, value in row.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    mlflow.log_metric(metric_name(f"{scenario}.{metric}"), float(value))

    if "BASE_CSV" in globals() and "SEMANTIC_CSV" in globals():
        for label, path in {"base_csv": BASE_CSV, "semantic_csv": SEMANTIC_CSV}.items():
            path = Path(path)
            mlflow.log_param(f"{label}.path", str(path))
            mlflow.log_param(f"{label}.sha256", sha256(path))
            mlflow.log_param(f"{label}.bytes", path.stat().st_size)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if "results" in globals():
            results.reset_index().to_csv(tmp / "metrics.csv", index=False)
            mlflow.log_artifact(str(tmp / "metrics.csv"), artifact_path="results")
        if "deltas" in globals():
            deltas.reset_index().to_csv(tmp / "metric_deltas.csv", index=False)
            mlflow.log_artifact(str(tmp / "metric_deltas.csv"), artifact_path="results")
        if "source_contribution" in globals():
            source_contribution.to_csv(tmp / "source_contribution.csv", index=False)
            mlflow.log_artifact(str(tmp / "source_contribution.csv"), artifact_path="results")
        if "artifacts" in globals():
            for scenario_name, scenario_artifacts in artifacts.items():
                model = scenario_artifacts.get("model")
                if model is not None:
                    model_dir = tmp / f"model_{scenario_name}"
                    mlflow.xgboost.save_model(model, path=str(model_dir))
                    mlflow.log_artifacts(
                        str(model_dir),
                        artifact_path=f"models/model_{scenario_name}",
                    )
        if Path("poetry.lock").exists():
            mlflow.log_artifact("poetry.lock", artifact_path="inputs")
        if Path("pyproject.toml").exists():
            mlflow.log_artifact("pyproject.toml", artifact_path="inputs")

    mlflow.end_run(status="FINISHED")
    """
    setup_cell = nbformat.v4.new_code_cell(dedent(setup).strip())
    finish_cell = nbformat.v4.new_code_cell(dedent(finish).strip())
    setup_cell.metadata["tags"] = ["runner-injected-mlflow"]
    finish_cell.metadata["tags"] = ["runner-injected-mlflow"]
    nb.cells.insert(0, setup_cell)
    nb.cells.append(finish_cell)


def execute_notebook(
    nb_path: Path,
    out_path: Path | None,
    timeout: int,
    use_mlflow: bool,
    hyperparams_path: Path,
    tracking_uri: str,
    experiment: str,
    run_name: str,
) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    if use_mlflow:
        inject_mlflow_cells(nb, nb_path, hyperparams_path, tracking_uri, experiment, run_name)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nb.cells = [
        cell
        for cell in nb.cells
        if "runner-injected-mlflow" not in cell.get("metadata", {}).get("tags", [])
    ]
    out_path = out_path or nb_path
    nbformat.write(nb, out_path)
    print(f"Notebook ejecutado y guardado en {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--hyperparams", default="hyperparams.yml")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    parser.add_argument("--experiment", default="pr_feature_ablation")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    nb_path = resolve_path(args.notebook)
    out_path = resolve_path(args.out)
    hyperparams_path = resolve_path(args.hyperparams)
    if not nb_path.exists():
        raise SystemExit(f"No se encontro {nb_path}")

    execute_notebook(
        nb_path=nb_path,
        out_path=out_path,
        timeout=args.timeout,
        use_mlflow=args.mlflow,
        hyperparams_path=hyperparams_path,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        run_name=args.run_name or nb_path.stem,
    )


if __name__ == "__main__":
    main()
