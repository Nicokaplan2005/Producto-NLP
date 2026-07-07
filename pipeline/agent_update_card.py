"""Card agents: generate an initial repo card and update it from a PR diff."""

from __future__ import annotations

import json
from pathlib import Path

from .agents import default_repo_card
from .git_ops import current_sha, list_tracked_files, run_git
from .github_api import parse_pr_url
from .llm_client import predict_model
from .prompts import REPO_CARD_GENERATION_SYSTEM_PROMPT, REPO_CARD_UPDATE_SYSTEM_PROMPT
from .schemas import RepoCard


def generate_initial_card(repo_dir: Path, *, pr_url: str | None = None) -> RepoCard:
    """Generate a RepoCard from the current state of repo_dir."""
    files = list_tracked_files(repo_dir)
    sha = current_sha(repo_dir)
    remote_url = _maybe_git(repo_dir, ["remote", "get-url", "origin"])
    ref = parse_pr_url(pr_url) if pr_url else None
    languages = sorted(
        {_language_from_path(p) for p in files if _language_from_path(p)}
    )

    card = default_repo_card()
    card.repo.update(
        {
            "name": ref.repo if ref else repo_dir.name,
            "url": remote_url,
            "last_verified_commit": sha,
            "primary_languages": languages,
        }
    )
    card.special_files.update(_special_files(files))
    card.maintenance.update(
        {
            "created_from_commit": sha,
            "last_updated_commit": sha,
            "known_stale_sections": ["architecture", "modules", "risk_model", "navigation"],
        }
    )

    prompt = "\n\n".join(
        [
            "Create a complete RepoCard JSON object for this repository.",
            (
                "Use the compact v1.2 shape. Keep unknown sections as empty strings, "
                "empty arrays, or empty objects. No comments, ellipses, or prose outside JSON."
            ),
            "Starter card:",
            card.model_dump_json(indent=2),
            "Tracked files:",
            "\n".join(files[:3000]),
            "Relevant file excerpts:",
            _priority_excerpts(repo_dir, files),
        ]
    )
    return predict_model(
        RepoCard,
        prompt,
        system_prompt=REPO_CARD_GENERATION_SYSTEM_PROMPT,
        context=f"generate_card:{repo_dir.name}",
    )


def update_card(card: RepoCard, diff: str, *, pr_url: str | None = None, target_sha: str | None = None) -> RepoCard:
    """Return a new RepoCard updated with the changes introduced by diff."""
    prompt = "\n\n".join(
        [
            "Update this RepoCard using the supplied repo-state diff.",
            (
                "Return one valid compact v1.2 JSON object. No comments, ellipses, "
                "trailing commas, or prose outside JSON."
            ),
            f"PR URL: {pr_url or 'unknown'}",
            f"Target SHA: {target_sha or 'unknown'}",
            "Existing card:",
            _truncate(card.model_dump_json(indent=2), 60_000),
            "Repo-state diff:",
            _truncate(diff or "(empty diff)", 50_000),
        ]
    )
    return predict_model(
        RepoCard,
        prompt,
        system_prompt=REPO_CARD_UPDATE_SYSTEM_PROMPT,
        context=f"update_card:{pr_url or 'unknown'}",
    )


def read_card(path: Path) -> RepoCard:
    return RepoCard.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_card(path: Path, card: RepoCard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _priority_excerpts(repo_dir: Path, files: list[str]) -> str:
    priority = {"readme.md", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "go.mod", "cargo.toml", "gemfile"}
    selected = [p for p in files if Path(p).name.lower() in priority or p.startswith((".github/workflows/", "docs/"))][:40]
    chunks = []
    for rel in selected:
        path = repo_dir / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"--- {rel} ---\n{_truncate(text, 4000)}")
    return "\n\n".join(chunks) or "(no excerpts)"


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."


def _maybe_git(repo_dir: Path, args: list[str]) -> str:
    try:
        return run_git(args, cwd=repo_dir)
    except Exception:
        return ""


def _language_from_path(path: str) -> str:
    return {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++", ".c": "C"}.get(Path(path).suffix.lower(), "")


def _special_files(files: list[str]) -> dict[str, list[str]]:
    lower_map = {p.lower(): p for p in files}
    return {
        "dependency_manifests": [v for k, v in lower_map.items() if k.endswith(("requirements.txt", "pyproject.toml", "package.json", "go.mod", "cargo.toml"))],
        "lockfiles": [v for k, v in lower_map.items() if k.endswith(("package-lock.json", "yarn.lock", "poetry.lock", "uv.lock", "cargo.lock"))],
        "ci_files": [v for k, v in lower_map.items() if k.startswith(".github/workflows/")],
        "docker_files": [v for k, v in lower_map.items() if "dockerfile" in k or k.endswith("docker-compose.yml")],
        "documentation_files": [v for k, v in lower_map.items() if k.endswith((".md", ".rst"))],
        "license_files": [v for k, v in lower_map.items() if Path(k).name.startswith("license")],
    }
