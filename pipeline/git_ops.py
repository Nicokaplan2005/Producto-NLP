"""Small wrappers around git commands used by the pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitCommandError(RuntimeError):
    """Raised when a git subprocess fails."""


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        command = "git " + " ".join(args)
        raise GitCommandError(f"{command} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def clone_or_fetch_repo(clone_url: str, repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        fetch_repo(repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["clone", "--no-checkout", clone_url, str(repo_dir)])
    fetch_repo(repo_dir)


def fetch_repo(repo_dir: Path) -> None:
    run_git(["fetch", "--all", "--tags", "--prune"], cwd=repo_dir)


def ensure_commit_available(repo_dir: Path, sha: str) -> None:
    try:
        run_git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo_dir)
        return
    except GitCommandError:
        run_git(["fetch", "origin", sha], cwd=repo_dir)
        run_git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo_dir)


def checkout_detached(repo_dir: Path, sha: str) -> None:
    ensure_commit_available(repo_dir, sha)
    run_git(["checkout", "--detach", sha], cwd=repo_dir)


def reset_hard(repo_dir: Path, sha: str) -> None:
    ensure_commit_available(repo_dir, sha)
    run_git(["reset", "--hard", sha], cwd=repo_dir)


def ensure_repo_at_sha(clone_url: str, repo_dir: Path, sha: str) -> None:
    clone_or_fetch_repo(clone_url, repo_dir)
    checkout_detached(repo_dir, sha)
    reset_hard(repo_dir, sha)


def current_sha(repo_dir: Path) -> str:
    return run_git(["rev-parse", "HEAD"], cwd=repo_dir)


def diff_between(repo_dir: Path, from_ref: str, to_ref: str) -> str:
    ensure_commit_available(repo_dir, from_ref)
    ensure_commit_available(repo_dir, to_ref)
    return run_git(["diff", "--binary", from_ref, to_ref], cwd=repo_dir)


def list_tracked_files(repo_dir: Path, limit: int = 5000) -> list[str]:
    output = run_git(["ls-files"], cwd=repo_dir)
    return output.splitlines()[:limit]
