"""Router principal del pipeline: carta + diff → features, con actualización de carta para PRs mergeadas.

Uso desde línea de comandos:
    python -m pipeline.process_pr <pr_url> [opciones]

O como librería:
    from pipeline.process_pr import process_pr
    result = process_pr(pr_url, repos_dir=..., cards_dir=..., output_dir=...)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .agent_extract_features import extract_features
from .agent_update_card import generate_initial_card, read_card, update_card, write_card
from .diff_utils import materialize_diff_text
from .env import load_project_env
from .git_ops import diff_between, ensure_repo_at_sha
from .github_api import (
    fetch_pr_metadata,
    fetch_raw_pr_diff,
    parse_pr_url,
    safe_repo_dir_name,
)
from .schemas import FeatureExtractionResult

load_project_env()


def process_pr(
    pr_url: str,
    *,
    repos_dir: Path,
    cards_dir: Path,
    output_dir: Path,
    diffs_dir: Path | None = None,
    diff_text: str | None = None,
    skip_repo_clone: bool = False,
    max_iterations: int = 30,
) -> FeatureExtractionResult:
    """Procesa un PR completo: extrae features y actualiza la carta del repo si fue mergeada.

    Args:
        pr_url: URL del PR en GitHub (API, web, o raw diff).
        repos_dir: Directorio donde se clonan los repos localmente.
        cards_dir: Directorio donde se guardan las cartas por repo (un JSON por repo).
        output_dir: Directorio donde se guarda el FeatureExtractionResult JSON.
        diffs_dir: Directorio donde buscar diffs ya descargados antes de ir a la API.
        diff_text: Si ya tenés el texto del diff en memoria, pasalo acá para no buscarlo.
        skip_repo_clone: Si True, asume que el repo ya está clonado (útil para tests).
        max_iterations: Máximo de iteraciones para el agente de extracción.

    Returns:
        FeatureExtractionResult con features semánticas y metadata.
    """
    ref = parse_pr_url(pr_url)
    repo_slug = safe_repo_dir_name(ref)

    repos_dir = repos_dir.resolve()
    cards_dir = cards_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Metadata del PR ───────────────────────────────────────────────────
    print(f"[router] Fetching metadata: {ref.owner}/{ref.repo}#{ref.number}")
    metadata = fetch_pr_metadata(pr_url)
    is_merged = bool(metadata.merged_at)
    print(f"[router] merged={is_merged} | base_sha={metadata.repo_state_before_pr_sha[:8]} | source={metadata.repo_state_sha_source}")

    # ── 2. Diff ──────────────────────────────────────────────────────────────
    diff_path = _resolve_diff(
        ref=ref,
        metadata_diff_text=diff_text,
        diffs_dir=diffs_dir,
        output_dir=output_dir,
    )
    print(f"[router] diff → {diff_path} ({diff_path.stat().st_size:,} bytes)")

    # ── 3. Repo local ────────────────────────────────────────────────────────
    repo_dir = repos_dir / repo_slug
    if not skip_repo_clone:
        print(f"[router] Ensuring repo at SHA {metadata.repo_state_before_pr_sha[:8]}")
        ensure_repo_at_sha(metadata.clone_url, repo_dir, metadata.repo_state_before_pr_sha)

    # ── 4. Carta del repo ────────────────────────────────────────────────────
    card_path = cards_dir / f"{repo_slug}.json"
    if card_path.exists():
        print(f"[router] Loading existing card: {card_path.name}")
        card = read_card(card_path)
    else:
        print(f"[router] Generating initial card for {repo_slug}")
        card = generate_initial_card(repo_dir, pr_url=pr_url)
        write_card(card_path, card)
        print(f"[router] Initial card saved → {card_path.name}")

    # ── 5. Extracción de features ────────────────────────────────────────────
    print(f"[router] Extracting features (max_iterations={max_iterations})")
    semantic_features = extract_features(
        diff_path=diff_path,
        card_path=card_path,
        repo_dir=repo_dir,
        pr_url=pr_url,
        max_iterations=max_iterations,
    )
    print(f"[router] Features extracted: intent={semantic_features.inferred_change_intent} risk={semantic_features.semantic_risk_level}")

    # ── 6. Actualizar carta (solo PRs mergeadas) ─────────────────────────────
    if is_merged:
        print(f"[router] PR merged — updating card from {metadata.repo_state_before_pr_sha[:8]}..{metadata.head_sha[:8]}")
        try:
            repo_state_diff = diff_between(repo_dir, metadata.repo_state_before_pr_sha, metadata.head_sha)
            updated_card = update_card(
                card,
                repo_state_diff,
                pr_url=pr_url,
                target_sha=metadata.head_sha,
            )
            write_card(card_path, updated_card)
            print(f"[router] Card updated → {card_path.name}")
        except Exception as exc:
            print(f"[router] WARNING: card update failed ({exc}), keeping previous card")
    else:
        print("[router] PR not merged — skipping card update")

    # ── 7. Guardar resultado ─────────────────────────────────────────────────
    result = FeatureExtractionResult(
        pr_url=pr_url,
        repo=ref.repo_full_name,
        pr_number=ref.number,
        repo_state_sha=metadata.repo_state_before_pr_sha,
        diff_path=str(diff_path),
        card_path=str(card_path),
        semantic_features=semantic_features,
    )

    out_file = output_dir / f"{repo_slug}__pr_{ref.number}.json"
    out_file.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"[router] Result saved → {out_file.name}")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_diff(
    *,
    ref,
    metadata_diff_text: str | None,
    diffs_dir: Path | None,
    output_dir: Path,
) -> Path:
    """Obtiene el diff en el orden: texto en memoria → archivo local → GitHub API."""
    diffs_out = output_dir / "diffs"

    if metadata_diff_text:
        result = materialize_diff_text(ref.original_url, metadata_diff_text, diffs_out, source="memory")
        return result.path

    if diffs_dir is not None:
        candidates = [
            diffs_dir / f"{ref.owner}__{ref.repo}__pr_{ref.number}.diff",
            diffs_dir / ref.repo / f"{ref.number}.diff",
            diffs_dir / "pr_diffs" / ref.repo_full_name / f"{ref.number}.diff",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                print(f"[router] Using cached diff: {candidate}")
                return candidate

    print(f"[router] Fetching diff from GitHub for {ref.owner}/{ref.repo}#{ref.number}")
    raw = fetch_raw_pr_diff(ref.original_url)
    result = materialize_diff_text(ref.original_url, raw, diffs_out, source="github_api")
    return result.path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Procesa un PR: carta + diff → features + actualización de carta")
    p.add_argument("pr_url", help="URL del PR en GitHub")
    p.add_argument("--repos-dir", default="data/repos", help="Directorio de repos clonados (default: data/repos)")
    p.add_argument("--cards-dir", default="data/cards", help="Directorio de cartas por repo (default: data/cards)")
    p.add_argument("--output-dir", default="data/output", help="Directorio de resultados (default: data/output)")
    p.add_argument("--diffs-dir", default=None, help="Directorio con diffs pre-descargados (opcional)")
    p.add_argument("--max-iterations", type=int, default=30, help="Máximo de iteraciones del agente (default: 30)")
    p.add_argument("--skip-clone", action="store_true", help="Asumir repo ya clonado en repos-dir")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = process_pr(
            args.pr_url,
            repos_dir=Path(args.repos_dir),
            cards_dir=Path(args.cards_dir),
            output_dir=Path(args.output_dir),
            diffs_dir=Path(args.diffs_dir) if args.diffs_dir else None,
            skip_repo_clone=args.skip_clone,
            max_iterations=args.max_iterations,
        )
        print(f"\n[router] Done. intent={result.semantic_features.inferred_change_intent} risk={result.semantic_features.semantic_risk_level}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
