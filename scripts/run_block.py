# ruff: noqa: E402
"""Process pr_index.csv in block order, writing features to pr_features.csv.

Usage:
    python -m scripts.run_block --csv pr_index.csv --cards-dir data/cards --output-dir data/output
    python -m scripts.run_block --csv pr_index.csv --repo Genymobile/scrcpy --bloque extra
    python -m scripts.run_block --csv pr_index.csv --limit 1   # smoke-test one PR

The script maintains the carta (RepoCard) state per repo in memory and on disk.
For each PR in pr_index order (sorted by repo, bloque, pr_number):
  - merge=0 → extract features, carta unchanged.
  - merge=1 → extract features + update carta, save carta to cards_dir.

After each LLM call the row is appended immediately to pr_features.csv so that
a crash mid-run doesn't lose already-processed work.

Join key with pr_datasets.csv: repo + pr_number.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.agent_update_card import read_card, write_card  # noqa: E402
from pipeline.process_block import apply_card_patch, process_pr_from_index  # noqa: E402
from pipeline.schemas import EnhancedPRFeatures, RepoCard     # noqa: E402

FEATURE_FIELDS = list(EnhancedPRFeatures.model_fields)
CSV_HEADER = ["pr_url", "repo", "pr_number", "bloque", "merge", "elapsed_seconds"] + FEATURE_FIELDS


def _build_pr_url(repo: str, pr_number: int) -> str:
    return f"https://api.github.com/repos/{repo}/pulls/{pr_number}"


def main() -> int:
    args = _parse_args()

    index_path = Path(args.csv)
    cards_dir = Path(args.cards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set usage log default before any LLM calls (per-repo file to avoid conflicts)
    if not os.getenv("LLM_USAGE_LOG_PATH"):
        slug = args.repo.replace("/", "__") if args.repo else "all"
        os.environ["LLM_USAGE_LOG_PATH"] = str(output_dir / f"llm_usage_{slug}.jsonl")

    rows = _load_index(
        index_path,
        repo_filter=args.repo,
        bloque_filter=args.bloque,
    )
    if args.limit:
        rows = rows[: args.limit]

    features_csv = output_dir / "pr_features.csv"
    already_done = _load_already_processed(features_csv)
    _ensure_header(features_csv)

    print(f"[run_block] {len(rows)} rows to process | output={features_csv}")
    if already_done:
        print(f"[run_block] Skipping {len(already_done)} already-processed rows")

    cards: dict[str, RepoCard] = {}
    processed = errors = skipped = 0

    for row in rows:
        repo = row["repo"]
        pr_number = int(row["pr_number"])
        diff_path = Path(row["path"])
        is_merged = str(row.get("merge", "0")).strip() == "1"
        bloque = row.get("bloque", "")

        key = (repo, pr_number)
        if key in already_done:
            skipped += 1
            continue

        if repo not in cards:
            card = _load_card(repo, cards_dir)
            if card is None:
                print(f"[run_block] SKIP {repo}#{pr_number}: carta not found in {cards_dir}")
                errors += 1
                continue
            cards[repo] = card
            print(f"[run_block] Loaded carta for {repo}")

        print(f"[run_block] Processing {repo}#{pr_number}  merge={int(is_merged)}", end="", flush=True)
        t0 = time.perf_counter()

        try:
            result = process_pr_from_index(
                repo=repo,
                pr_number=pr_number,
                diff_path=diff_path,
                card=cards[repo],
                is_merged=is_merged,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR ({elapsed:.1f}s): {exc}")
            traceback.print_exc()
            errors += 1
            continue

        elapsed = time.perf_counter() - t0
        print(f"  OK ({elapsed:.1f}s)  intent={result.features.inferred_change_intent}")

        if is_merged:
            if result.card_patch is not None:
                cards[repo] = apply_card_patch(cards[repo], result.card_patch)
            else:
                print(f"[run_block]   WARNING: merged PR but LLM returned no card_patch")
            _stamp_maintenance(cards[repo], pr_number)
            _save_card(repo, cards[repo], cards_dir)
            print(f"[run_block]   carta patched for {repo}")

        _append_row(
            features_csv,
            pr_url=_build_pr_url(repo, pr_number),
            repo=repo,
            pr_number=pr_number,
            bloque=bloque,
            merge=int(is_merged),
            elapsed=round(elapsed, 2),
            features=result.features,
        )
        processed += 1

        # Budget guard: aggregate cost across ALL per-repo usage logs
        if args.budget_usd > 0:
            cost = _compute_total_cost_usd(output_dir)
            if cost >= args.budget_usd:
                print(f"\n[run_block] BUDGET LIMIT ${cost:.4f} >= ${args.budget_usd:.2f} — stopping")
                break

    print(
        f"\n[run_block] Done. processed={processed}  skipped={skipped}  errors={errors}"
    )
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _load_index(
    csv_path: Path,
    *,
    repo_filter: str | None,
    bloque_filter: str | None,
) -> list[dict]:
    _raise_csv_field_limit()
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if repo_filter and row.get("repo") != repo_filter:
                continue
            if bloque_filter and str(row.get("bloque", "")) != bloque_filter:
                continue
            rows.append(dict(row))
    rows.sort(
        key=lambda r: (r.get("repo", ""), r.get("bloque", ""), int(r.get("pr_number", 0)))
    )
    return rows


def _load_already_processed(features_csv: Path) -> set[tuple[str, int]]:
    if not features_csv.exists():
        return set()
    done: set[tuple[str, int]] = set()
    _raise_csv_field_limit()
    with features_csv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            repo = row.get("repo", "")
            try:
                pr_number = int(row.get("pr_number", ""))
            except (ValueError, TypeError):
                continue
            if repo and pr_number:
                done.add((repo, pr_number))
    return done


def _ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(CSV_HEADER)


def _append_row(
    path: Path,
    *,
    pr_url: str,
    repo: str,
    pr_number: int,
    bloque: str,
    merge: int,
    elapsed: float,
    features: EnhancedPRFeatures,
) -> None:
    data = features.model_dump(mode="json")
    row = [pr_url, repo, pr_number, bloque, merge, elapsed]
    for field in FEATURE_FIELDS:
        value = data.get(field, "")
        if isinstance(value, list):
            row.append("|".join(str(v) for v in value))
        else:
            row.append("" if value is None else value)
    lock = path.with_suffix(".lock")
    _acquire_lock(lock)
    try:
        with path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(row)
    finally:
        _release_lock(lock)


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def _repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def _load_card(repo: str, cards_dir: Path) -> RepoCard | None:
    path = cards_dir / f"{_repo_slug(repo)}.json"
    if not path.exists():
        return None
    return read_card(path)


def _save_card(repo: str, card: RepoCard, cards_dir: Path) -> None:
    path = cards_dir / f"{_repo_slug(repo)}.json"
    write_card(path, card)


def _acquire_lock(lock_path: Path) -> None:
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(0.05)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def _compute_total_cost_usd(output_dir: Path) -> float:
    """Aggregate cost from all per-repo llm_usage*.jsonl files."""
    total = 0.0
    for log_file in output_dir.glob("llm_usage*.jsonl"):
        try:
            with log_file.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        meta = entry.get("usage_metadata") or entry.get("usage") or {}
                        if isinstance(meta, dict):
                            inp = (meta.get("prompt_token_count") or 0)
                            out = (meta.get("candidates_token_count") or 0)
                            total += (inp * 0.15 + out * 0.40) / 1_000_000
                    except Exception:
                        pass
        except OSError:
            pass
    return total


def _stamp_maintenance(card: RepoCard, pr_number: int) -> None:
    """Always record the last merged PR in maintenance, regardless of LLM patch."""
    m = card.maintenance
    m["last_updated_from_pr"] = pr_number
    card.maintenance = m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process pr_index.csv and write features to pr_features.csv"
    )
    p.add_argument("--csv", default="pr_index.csv", help="Path to pr_index.csv")
    p.add_argument(
        "--cards-dir", default="data/cards",
        help="Directory with carta JSON files (owner__repo.json per repo)"
    )
    p.add_argument(
        "--output-dir", default="data/output",
        help="Output directory; pr_features.csv is written here"
    )
    p.add_argument("--repo", default=None, help="Filter: only this repo (owner/repo)")
    p.add_argument("--bloque", default=None, help="Filter: only this bloque value")
    p.add_argument("--limit", type=int, default=None, help="Max rows to process")
    p.add_argument(
        "--budget-usd", type=float, default=60.0,
        help="Stop when total LLM cost (all repos) exceeds this USD amount (0 = disabled)"
    )
    return p.parse_args()


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


if __name__ == "__main__":
    raise SystemExit(main())
