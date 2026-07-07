"""Launch one run_block subprocess per repo that has a card file.

Usage:
    python -m scripts.launch_parallel --bloque 1
    python -m scripts.launch_parallel --bloque 1 --repos apple/swift pallets/flask
    python -m scripts.launch_parallel --bloque 1 --budget-usd 60
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    args = _parse_args()

    cards_dir = Path(args.cards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    repos = _find_repos(args, cards_dir)
    if not repos:
        print("[launch] No repos to launch — no card files found in", cards_dir)
        return 1

    # Ensure shared CSV header exists before any subprocesses touch it
    _ensure_features_csv(output_dir)

    print(f"[launch] Repos to launch ({len(repos)}): {repos}")
    print(f"[launch] Budget: ${args.budget_usd:.2f} USD total  |  bloque={args.bloque}")
    print(f"[launch] Logs:   {logs_dir}")
    print()

    procs: list[tuple[str, subprocess.Popen, object]] = []
    for repo in repos:
        slug = repo.replace("/", "__")
        log_path = logs_dir / f"{slug}.log"
        usage_log = output_dir / f"llm_usage_{slug}.jsonl"

        env = {**os.environ, "LLM_USAGE_LOG_PATH": str(usage_log)}

        cmd = [
            sys.executable, "-m", "scripts.run_block",
            "--csv", args.csv,
            "--cards-dir", args.cards_dir,
            "--output-dir", args.output_dir,
            "--repo", repo,
            "--bloque", args.bloque,
            "--budget-usd", str(args.budget_usd),
        ]

        log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        procs.append((repo, proc, log_fh))
        print(f"[launch] Started {repo}  PID={proc.pid}  log={log_path.name}")

    print()
    _monitor(procs, output_dir, check_interval=60)
    return 0


def _monitor(
    procs: list[tuple[str, subprocess.Popen, object]],
    output_dir: Path,
    check_interval: int,
) -> None:
    running = list(procs)
    while running:
        time.sleep(check_interval)
        still_running = []
        for repo, proc, log_fh in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((repo, proc, log_fh))
            else:
                print(f"[launch] {repo} FINISHED (exit={rc})")
                log_fh.close()
        running = still_running

        cost = _compute_total_cost_usd(output_dir)
        total_done = _count_processed(output_dir)
        active = [repo for repo, proc, _ in running]
        print(
            f"[launch] cost=${cost:.4f}  PRs_done={total_done}  "
            f"active={len(running)} ({', '.join(active)})"
        )

    print("[launch] All processes finished.")


# ---------------------------------------------------------------------------
# Helpers (duplicated from run_block to keep scripts self-contained)
# ---------------------------------------------------------------------------

def _compute_total_cost_usd(output_dir: Path) -> float:
    import json
    total = 0.0
    for log_file in output_dir.glob("llm_usage*.jsonl"):
        try:
            with log_file.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        meta = entry.get("usage_metadata") or entry.get("usage") or {}
                        if isinstance(meta, dict):
                            inp = meta.get("prompt_token_count") or 0
                            out = meta.get("candidates_token_count") or 0
                            total += (inp * 0.15 + out * 0.40) / 1_000_000
                    except Exception:
                        pass
        except OSError:
            pass
    return total


def _count_processed(output_dir: Path) -> int:
    total = 0
    for csv_file in output_dir.glob("pr_features.csv"):
        try:
            with csv_file.open(encoding="utf-8") as f:
                total += max(0, sum(1 for _ in f) - 1)
        except OSError:
            pass
    return total


def _ensure_features_csv(output_dir: Path) -> None:
    from pipeline.schemas import EnhancedPRFeatures
    features_csv = output_dir / "pr_features.csv"
    if features_csv.exists():
        return
    header = ["pr_url", "repo", "pr_number", "bloque", "merge", "elapsed_seconds"]
    header += list(EnhancedPRFeatures.model_fields)
    with features_csv.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(header)


def _find_repos(args: argparse.Namespace, cards_dir: Path) -> list[str]:
    if args.repos:
        return args.repos
    repos = []
    for card_file in sorted(cards_dir.glob("*.json")):
        slug = card_file.stem
        repo = slug.replace("__", "/", 1)
        repos.append(repo)
    return repos


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch parallel run_block processes")
    p.add_argument("--csv", default="pr_index.csv")
    p.add_argument("--cards-dir", default="data/cards")
    p.add_argument("--output-dir", default="data/output")
    p.add_argument("--bloque", default="1")
    p.add_argument("--budget-usd", type=float, default=60.0,
                   help="Global budget cap in USD across all repos")
    p.add_argument("--repos", nargs="*",
                   help="Repos to launch (e.g. apple/swift pallets/flask). Default: all with cards")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
