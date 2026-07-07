"""Utilities for cleaning, writing, and summarizing unified diffs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .github_api import PullRequestRef, parse_pr_url, safe_pr_file_stem
from .schemas import ChangedFileFeature

EMPTY_DIFF_MARKERS = {"", "none", "null", "nan", "na", "n/a", "0"}


@dataclass(frozen=True)
class MaterializedDiff:
    pr_ref: PullRequestRef
    path: Path
    source: str
    was_empty_input: bool
    bytes_written: int


def is_empty_diff_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value)
    return text.strip().lower() in EMPTY_DIFF_MARKERS


def clean_diff_text(diff_text: str) -> str:
    """Keep patch semantics while normalizing transport/CSV artifacts."""

    text = str(diff_text).replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text if text.endswith("\n") else text + "\n"


def materialize_diff_text(
    pr_url: str,
    diff_text: str,
    output_dir: Path,
    *,
    source: str,
    row_number: int | None = None,
    force: bool = False,
) -> MaterializedDiff:
    ref = parse_pr_url(pr_url)
    cleaned = clean_diff_text(diff_text)
    path = output_dir / f"{safe_pr_file_stem(ref, row_number=row_number)}.diff"
    if path.exists() and not force:
        return MaterializedDiff(
            pr_ref=ref,
            path=path,
            source=f"{source}:existing",
            was_empty_input=False,
            bytes_written=path.stat().st_size,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned, encoding="utf-8", newline="\n")
    return MaterializedDiff(
        pr_ref=ref,
        path=path,
        source=source,
        was_empty_input=False,
        bytes_written=path.stat().st_size,
    )


def parse_changed_files(diff_text: str) -> list[ChangedFileFeature]:
    changed: list[ChangedFileFeature] = []
    current: dict[str, object] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            changed.append(ChangedFileFeature.model_validate(current))
            current = None

    for line in clean_diff_text(diff_text).splitlines():
        if line.startswith("diff --git "):
            flush()
            old_path, new_path = _parse_diff_git_paths(line)
            current = {
                "path": new_path,
                "old_path": old_path if old_path != new_path else None,
                "status": "modified",
                "additions": 0,
                "deletions": 0,
            }
            continue
        if current is None:
            continue
        if line.startswith("new file mode"):
            current["status"] = "added"
        elif line.startswith("deleted file mode"):
            current["status"] = "deleted"
        elif line.startswith("rename from "):
            current["old_path"] = line.removeprefix("rename from ").strip()
            current["status"] = "renamed"
        elif line.startswith("rename to "):
            current["path"] = line.removeprefix("rename to ").strip()
            current["status"] = "renamed"
        elif line.startswith("+") and not line.startswith("+++"):
            current["additions"] = int(current["additions"]) + 1
        elif line.startswith("-") and not line.startswith("---"):
            current["deletions"] = int(current["deletions"]) + 1
    flush()
    return changed


def diff_summary_features(diff_text: str) -> dict[str, int]:
    files = parse_changed_files(diff_text)
    return {
        "changed_file_count": len(files),
        "added_file_count": sum(1 for item in files if item.status == "added"),
        "deleted_file_count": sum(1 for item in files if item.status == "deleted"),
        "renamed_file_count": sum(1 for item in files if item.status == "renamed"),
        "total_additions": sum(item.additions for item in files),
        "total_deletions": sum(item.deletions for item in files),
        "diff_word_count": len(clean_diff_text(diff_text).split()),
    }


def _parse_diff_git_paths(line: str) -> tuple[str, str]:
    parts = line.split()
    if len(parts) >= 4:
        old_path = parts[2].removeprefix("a/")
        new_path = parts[3].removeprefix("b/")
        return old_path, new_path
    return "", ""
