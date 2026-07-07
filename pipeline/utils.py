"""Pipeline utilities: diff materialization and repo state management."""

from __future__ import annotations

from pathlib import Path

from .diff_utils import clean_diff_text, is_empty_diff_value
from .git_ops import checkout_detached, ensure_commit_available, fetch_repo, reset_hard
from .github_api import fetch_raw_pr_diff, parse_pr_url, safe_pr_file_stem


def materialize_diff(
    pr_url: str,
    csv_diff_value: object,
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Save PR diff to a .diff file.

    Uses csv_diff_value if it contains a valid diff; otherwise fetches from
    GitHub using the raw patch URL.
    """
    ref = parse_pr_url(pr_url)
    path = output_dir / f"{safe_pr_file_stem(ref)}.diff"
    if path.exists() and not force:
        return path

    if not is_empty_diff_value(csv_diff_value):
        raw = str(csv_diff_value)
    else:
        raw = fetch_raw_pr_diff(pr_url)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_diff_text(raw), encoding="utf-8", newline="\n")
    return path


def advance_repo_to_sha(repo_dir: Path, target_sha: str) -> None:
    """Fetch and checkout an existing local clone to target_sha.

    Does not re-clone. Requires repo_dir to already be a git repo.
    """
    fetch_repo(repo_dir)
    ensure_commit_available(repo_dir, target_sha)
    checkout_detached(repo_dir, target_sha)
    reset_hard(repo_dir, target_sha)
