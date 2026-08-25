#!/usr/bin/env python3
"""Statistical summary and quality analysis for P2 counterfactual targets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p2_counterfactual_target_quality_contract_2026_09 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    P2_PANEL_DIR,
    P2_ROOT,
    P2_SUMMARY_DIR,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_END_EXCLUSIVE,
    SEED_START,
    SUMMARY_SCHEMA,
    ContractError,
    check_directory_boundary,
    paired_bootstrap_ci,
    resolve_k0_checkpoint,
    sha256_file,
)

logger = logging.getLogger("p2_summary")


def summarize_diff_series(diffs: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Compute distribution counts, non-zero rate, mean, and bootstrap CI for a difference series."""
    n = len(diffs)
    if n == 0:
        raise ValueError("Cannot summarize empty array")

    pos_count = int(np.sum(diffs > 0))
    neg_count = int(np.sum(diffs < 0))
    tie_count = int(np.sum(diffs == 0))
    nonzero_count = pos_count + neg_count

    mean_val, ci = paired_bootstrap_ci(diffs, reps=reps, seed=seed, ci=BOOTSTRAP_CI)

    return {
        "total_pairs": n,
        "nonzero_count": nonzero_count,
        "nonzero_rate": float(nonzero_count / n),
        "positive_count": pos_count,
        "positive_rate": float(pos_count / n),
        "negative_count": neg_count,
        "negative_rate": float(neg_count / n),
        "tie_count": tie_count,
        "tie_rate": float(tie_count / n),
        "mean_diff": float(mean_val),
        "ci95": ci,
    }


def adjudicate_p2_counterfactual_panel(
    panel_dir: Path = P2_PANEL_DIR,
    summary_dir: Path = P2_SUMMARY_DIR,
    allowed_root: Path = P2_ROOT,
) -> dict[str, Any]:
    """Load counterfactual_panel_manifest.json, compute quality statistics, and produce formal summary."""
    check_directory_boundary(summary_dir, allowed_root)
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = panel_dir / "counterfactual_panel_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    actual_manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema") != PANEL_MANIFEST_SCHEMA:
        raise ContractError(f"Manifest schema mismatch: got {manifest.get('schema')}, expected {PANEL_MANIFEST_SCHEMA}")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"Manifest experiment_id mismatch: got {manifest.get('experiment_id')}, expected {EXPERIMENT_ID}")
    if manifest.get("verdict") != "panel_generation_completed":
        raise ContractError(f"Manifest verdict is not panel_generation_completed: {manifest.get('verdict')}")

    _, k0_sha = resolve_k0_checkpoint()
    logged_k0 = manifest.get("parent_model", {})
    if logged_k0.get("sha256") != k0_sha:
        raise ContractError(f"Manifest parent K0 SHA mismatch: got {logged_k0.get('sha256')}, expected {k0_sha}")

    pairs = manifest.get("pairs", [])
    if len(pairs) != PANEL_GAMES:
        raise ContractError(f"Pairs count mismatch: got {len(pairs)}, expected {PANEL_GAMES}")

    seeds = [p["seed"] for p in pairs]
    if seeds != list(range(SEED_START, SEED_END_EXCLUSIVE)):
        raise ContractError(f"Seeds range mismatch: got {seeds[0]}..{seeds[-1]}, expected {SEED_START}..{SEED_END_EXCLUSIVE-1}")

    delta_rank_pts = np.array([float(p["delta_rank_point"]) for p in pairs], dtype=np.float64)
    delta_scores = np.array([float(p["delta_final_score"]) for p in pairs], dtype=np.float64)
    margins = np.array([float(p["margin"]) for p in pairs], dtype=np.float64)

    rank_summary = summarize_diff_series(delta_rank_pts, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    score_summary = summarize_diff_series(delta_scores, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)

    # Sub-bucket by margin: tight (<= 0.5) vs wide (> 0.5)
    tight_mask = (margins <= 0.5)
    tight_rank_summary = summarize_diff_series(delta_rank_pts[tight_mask]) if np.any(tight_mask) else None
    wide_rank_summary = summarize_diff_series(delta_rank_pts[~tight_mask]) if np.any(~tight_mask) else None

    hard_gates = {
        "manifest_verified": True,
        "k0_parent_verified": True,
        "exact_128_pairs_analyzed": (len(pairs) == PANEL_GAMES),
        "seeds_contiguous": True,
        "bootstrap_computed": True,
    }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "panel_manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_sha,
        },
        "parent_model": {
            "name": "K0_70k",
            "sha256": k0_sha,
        },
        "hard_gates": hard_gates,
        "metrics": {
            "delta_rank_point": rank_summary,
            "delta_final_score": score_summary,
            "margin_subgroups": {
                "tight_margin_le_0_5": {
                    "count": int(np.sum(tight_mask)),
                    "stats": tight_rank_summary,
                },
                "wide_margin_gt_0_5": {
                    "count": int(np.sum(~tight_mask)),
                    "stats": wide_rank_summary,
                },
            },
        },
        "verdict": "counterfactual_target_quality_evaluated",
        "promotion": {
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "k1": None,
        },
    }

    summary_path = summary_dir / "p2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=P2_PANEL_DIR, help="Directory containing panel manifest")
    parser.add_argument("--summary-dir", type=Path, default=P2_SUMMARY_DIR, help="Output directory for summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = adjudicate_p2_counterfactual_panel(
        panel_dir=args.panel_dir,
        summary_dir=args.summary_dir,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
