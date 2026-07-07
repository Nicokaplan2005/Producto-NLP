"""GitHub URL parsing and REST helpers for pull request metadata."""

from __future__ import annotations

import json
import os
import platform
import re
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime hardening.
    certifi = None

from .schemas import PullRequestMetadata, PullRequestRefModel

GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")
USER_AGENT = "repo-state-feature-pipeline"


class GitHubAPIError(RuntimeError):
    """Raised when GitHub data cannot be fetched or parsed."""


@dataclass(frozen=True)
class PullRequestRef:
    owner: str
    repo: str
    number: int
    original_url: str
    extra_path: tuple[str, ...] = ()

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    def as_model(self) -> PullRequestRefModel:
        return PullRequestRefModel(
            owner=self.owner,
            repo=self.repo,
            number=self.number,
            original_url=self.original_url,
            extra_path=self.extra_path,
        )


_API_PR_RE = re.compile(
    r"https?://api\.github\.com/repos/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pulls/(?P<number>\d+)(?P<extra>/.*)?$",
    re.IGNORECASE,
)
_WEB_PR_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?P<extra>/.*)?$",
    re.IGNORECASE,
)
_RAW_DIFF_RE = re.compile(
    r"https?://patch-diff\.githubusercontent\.com/raw/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)\.diff(?P<extra>/.*)?$",
    re.IGNORECASE,
)


def parse_pr_url(pr_url: str) -> PullRequestRef:
    """Parse common GitHub PR URL forms used by the dataset."""

    value = pr_url.strip()
    for pattern in (_API_PR_RE, _WEB_PR_RE, _RAW_DIFF_RE):
        match = pattern.match(value)
        if match:
            extra = match.groupdict().get("extra") or ""
            extra_parts = tuple(part for part in extra.split("/") if part)
            return PullRequestRef(
                owner=match.group("owner"),
                repo=match.group("repo"),
                number=int(match.group("number")),
                original_url=value,
                extra_path=extra_parts,
            )
    raise ValueError(f"Unsupported GitHub pull request URL: {pr_url!r}")


def api_pr_url(ref: PullRequestRef | PullRequestRefModel) -> str:
    return f"https://api.github.com/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}"


def raw_diff_url(ref: PullRequestRef | PullRequestRefModel) -> str:
    return f"https://patch-diff.githubusercontent.com/raw/{ref.owner}/{ref.repo}/pull/{ref.number}.diff"


def safe_repo_dir_name(ref: PullRequestRef | PullRequestRefModel) -> str:
    return f"{ref.owner}__{ref.repo}".replace(":", "_")


def safe_pr_file_stem(
    ref: PullRequestRef | PullRequestRefModel, row_number: int | None = None
) -> str:
    parts = [safe_repo_dir_name(ref), f"pr_{ref.number}"]
    extra_path = getattr(ref, "extra_path", ()) or ()
    if extra_path:
        parts.append("_".join(_safe_path_part(part) for part in extra_path))
    if row_number is not None:
        parts.append(f"row_{row_number:07d}")
    return "__".join(parts)


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "part"


def github_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get_text(
    url: str, headers: dict[str, str] | None = None, timeout: int = 60
) -> str:
    headers = headers or {}
    request = Request(url, headers=headers)
    context = _ssl_context()
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (ssl.SSLError, URLError) as exc:
        if platform.system().lower() == "windows":
            return _http_get_text_with_powershell(url, headers, timeout)
        raise GitHubAPIError(f"GET failed for {url}: {exc}") from exc


def _ssl_context() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _http_get_text_with_powershell(
    url: str, headers: dict[str, str], timeout: int
) -> str:
    header_lines = []
    for key, value in headers.items():
        escaped_key = key.replace("'", "''")
        escaped_value = value.replace("'", "''")
        header_lines.append(f"$headers['{escaped_key}'] = '{escaped_value}'")
    script = "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            "$headers = @{}",
            *header_lines,
            f"$response = Invoke-WebRequest -UseBasicParsing -Uri '{url}' -Headers $headers -TimeoutSec {timeout}",
            "$response.Content",
        ]
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise GitHubAPIError(
            f"PowerShell GET failed for {url}: {result.stderr.strip()}"
        )
    return result.stdout


def fetch_json(url: str) -> dict[str, Any]:
    text = http_get_text(url, github_headers())
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:500].replace("\n", " ")
        raise GitHubAPIError(
            f"GitHub response was not JSON for {url}: {snippet}"
        ) from exc


def fetch_pr_metadata(pr_url: str) -> PullRequestMetadata:
    ref = parse_pr_url(pr_url)
    api_url = api_pr_url(ref)
    data = fetch_json(api_url)
    base = data.get("base") or {}
    head = data.get("head") or {}
    base_repo = base.get("repo") or {}
    clone_url = (
        base_repo.get("clone_url") or f"https://github.com/{ref.owner}/{ref.repo}.git"
    )
    state_sha, state_source = determine_repo_state_before_pr_sha(ref, data)
    return PullRequestMetadata(
        ref=ref.as_model(),
        api_url=api_url,
        html_url=data.get("html_url")
        or f"https://github.com/{ref.owner}/{ref.repo}/pull/{ref.number}",
        diff_url=data.get("diff_url")
        or f"https://github.com/{ref.owner}/{ref.repo}/pull/{ref.number}.diff",
        patch_url=data.get("patch_url")
        or f"https://github.com/{ref.owner}/{ref.repo}/pull/{ref.number}.patch",
        raw_diff_url=raw_diff_url(ref),
        base_sha=_required_str(base, "sha"),
        head_sha=_required_str(head, "sha"),
        merge_commit_sha=data.get("merge_commit_sha"),
        merged_at=data.get("merged_at"),
        base_ref=base.get("ref"),
        head_ref=head.get("ref"),
        clone_url=clone_url,
        default_branch=base_repo.get("default_branch"),
        repo_state_before_pr_sha=state_sha,
        repo_state_sha_source=state_source,
        raw=data,
    )


def determine_repo_state_before_pr_sha(
    ref: PullRequestRef, pr_data: dict[str, Any]
) -> tuple[str, str]:
    """Return the best available SHA for the base repo state before the PR lands.

    For merged PRs, the first parent of the merge/squash/rebase result is the
    closest base-branch state before the PR was integrated. For open or
    unmerged PRs, the PR payload's base SHA is the best cheap fallback.
    """

    merge_commit_sha = pr_data.get("merge_commit_sha")
    if pr_data.get("merged_at") and merge_commit_sha:
        commit = fetch_json(
            f"https://api.github.com/repos/{ref.owner}/{ref.repo}/commits/{merge_commit_sha}"
        )
        parents = commit.get("parents") or []
        if parents and parents[0].get("sha"):
            return parents[0]["sha"], "merge_commit_first_parent"
    base_sha = (pr_data.get("base") or {}).get("sha")
    if base_sha:
        return base_sha, "pull_request_base_sha"
    raise GitHubAPIError(
        f"Could not determine base state SHA for {ref.repo_full_name}#{ref.number}"
    )


def fetch_raw_pr_diff(pr_url: str) -> str:
    ref = parse_pr_url(pr_url)
    return http_get_text(raw_diff_url(ref), github_headers("text/plain"))


def write_json(
    path: Path, data: BaseException | dict[str, Any] | PullRequestMetadata
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, PullRequestMetadata):
        payload = data.model_dump(mode="json")
    elif isinstance(data, dict):
        payload = data
    else:
        payload = {"error": str(data)}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise GitHubAPIError(f"Missing required GitHub field: {key}")
    return value
