"""Single-call PR processing: routes by merge flag, returns features + optional card update.

Each PR is processed with ONE LLM call (no tool loop):
  - merge=0 → UNMERGED_PR_SYSTEM_PROMPT → {"features": EnhancedPRFeatures}
  - merge=1 → MERGED_PR_SYSTEM_PROMPT   → {"features": EnhancedPRFeatures,
                                            "updated_card": RepoCard}

The caller is responsible for persisting the updated card and appending features
to the output CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

from .llm_client import predict_model
from .prompts import MERGED_PR_SYSTEM_PROMPT, UNMERGED_PR_SYSTEM_PROMPT
from .schemas import CardPatch, PRProcessingOutput, RepoCard

MAX_DIFF_CHARS = int(os.getenv("PR_PIPELINE_MAX_DIFF_CHARS", "40000"))
MAX_CARD_CHARS = int(os.getenv("PR_PIPELINE_MAX_CARD_CHARS", "30000"))
MAX_OUTPUT_TOKENS = int(os.getenv("PR_PIPELINE_MAX_OUTPUT_TOKENS", "14000"))


def process_pr_from_index(
    *,
    repo: str,
    pr_number: int,
    diff_path: Path,
    card: RepoCard,
    is_merged: bool,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> PRProcessingOutput:
    diff_text = _read_diff(diff_path)
    card_json = _truncate(card.model_dump_json(indent=2), MAX_CARD_CHARS)
    system_prompt = MERGED_PR_SYSTEM_PROMPT if is_merged else UNMERGED_PR_SYSTEM_PROMPT

    parts = [
        f"Repo: {repo}",
        f"PR: #{pr_number}",
    ]
    parts += [
        "--- REPOSITORY CARD ---",
        card_json,
        "--- PR DIFF ---",
        _truncate(diff_text, MAX_DIFF_CHARS),
    ]
    prompt = "\n\n".join(parts)

    return predict_model(
        PRProcessingOutput,
        prompt,
        system_prompt=system_prompt,
        context=f"process_block:{repo}#{pr_number}",
        max_output_tokens=max_output_tokens,
    )


def apply_card_patch(card: RepoCard, patch: CardPatch) -> RepoCard:
    """Merge patch onto card and return a validated RepoCard.

    Dict sections are deep-merged (nested keys preserved unless overwritten).
    List sections and scalars are replaced wholesale by the patch value.
    """
    base = card.model_dump(mode="json")
    changes = patch.model_dump(exclude_none=True)
    return RepoCard.model_validate(_deep_merge(base, changes))


def _deep_merge(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list) and len(value) == 0 and len(result[key]) > 0:
            pass  # never wipe a non-empty list with an empty patch
        else:
            result[key] = value
    return result


def _read_diff(diff_path: Path) -> str:
    try:
        return diff_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(diff unavailable: {exc})"


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."
