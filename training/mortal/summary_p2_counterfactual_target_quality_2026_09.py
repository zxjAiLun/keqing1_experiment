#!/usr/bin/env python3
"""Statistical summary and quality analysis for P2 counterfactual targets with strict raw log verification."""

from __future__ import annotations

import argparse
import gzip
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
    EXPECTED_PANEL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    FOCAL_SEAT,
    P2_PANEL_DIR,
    P2_ROOT,
    P2_SUMMARY_DIR,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    SPLIT_NAME,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    ContractError,
    action_matches_pai,
    canonical_log_content_sha256,
    check_directory_boundary,
    compute_final_ranks,
    final_scores_with_reach_accepted,
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


def verify_and_recalculate_pair(pair: dict[str, Any], allowed_root: Path) -> dict[str, Any]:
    """Strictly verify a single counterfactual pair against its raw gzipped log files on disk."""
    seed = pair["seed"]
    seed_key = pair["seed_key"]
    if seed_key != SEED_KEY:
        raise ContractError(f"Seed key mismatch in pair record {seed}: {seed_key} vs {SEED_KEY}")

    log_a_path = Path(pair["branch_a_log_path"])
    log_b_path = Path(pair["branch_b_log_path"])

    check_directory_boundary(log_a_path, allowed_root)
    check_directory_boundary(log_b_path, allowed_root)

    if not log_a_path.exists():
        raise FileNotFoundError(f"Branch A raw log not found at {log_a_path}")
    if not log_b_path.exists():
        raise FileNotFoundError(f"Branch B raw log not found at {log_b_path}")

    # Verify canonical content SHAs
    sha_a = canonical_log_content_sha256(log_a_path)
    sha_b = canonical_log_content_sha256(log_b_path)
    if sha_a != pair.get("branch_a_canonical_content_sha256"):
        raise ContractError(f"Branch A canonical content SHA mismatch for seed {seed}: disk={sha_a} vs manifest={pair.get('branch_a_canonical_content_sha256')}")
    if sha_b != pair.get("branch_b_canonical_content_sha256"):
        raise ContractError(f"Branch B canonical content SHA mismatch for seed {seed}: disk={sha_b} vs manifest={pair.get('branch_b_canonical_content_sha256')}")

    with gzip.open(log_a_path, "rt", encoding="utf-8") as f:
        events_a = [json.loads(line) for line in f]
    with gzip.open(log_b_path, "rt", encoding="utf-8") as f:
        events_b = [json.loads(line) for line in f]

    if not events_a or events_a[0].get("type") != "start_game" or events_a[-1].get("type") != "end_game":
        raise ContractError(f"Branch A log {log_a_path.name} is incomplete")
    if not events_b or events_b[0].get("type") != "start_game" or events_b[-1].get("type") != "end_game":
        raise ContractError(f"Branch B log {log_b_path.name} is incomplete")

    # Check seed tuple in logs
    if events_a[0].get("seed") != [seed, seed_key] or events_b[0].get("seed") != [seed, seed_key]:
        raise ContractError(f"Seed tuple in log header mismatch for seed {seed}")

    div_idx = pair["divergence_event_index"]
    if div_idx <= 0 or div_idx >= len(events_a) or div_idx >= len(events_b):
        raise ContractError(f"Invalid divergence index {div_idx} for seed {seed}")

    # Prefix match check
    for idx in range(div_idx):
        if events_a[idx] != events_b[idx]:
            raise ContractError(f"Pre-intervention prefix mismatch at event {idx} for seed {seed}")

    ev_a_div = events_a[div_idx]
    ev_b_div = events_b[div_idx]
    if ev_a_div.get("type") != "dahai" or ev_b_div.get("type") != "dahai":
        raise ContractError(f"Event at divergence index {div_idx} is not dahai for seed {seed}")
    if ev_a_div.get("actor") != FOCAL_SEAT or ev_b_div.get("actor") != FOCAL_SEAT:
        raise ContractError(f"Divergence actor is not focal seat {FOCAL_SEAT} for seed {seed}")

    top1_action = pair["top1_action"]
    top2_action = pair["top2_action"]
    if not action_matches_pai(top1_action, ev_a_div.get("pai", "")):
        raise ContractError(f"Branch A dahai '{ev_a_div.get('pai')}' does not match top1 action {top1_action}")
    if not action_matches_pai(top2_action, ev_b_div.get("pai", "")):
        raise ContractError(f"Branch B dahai '{ev_b_div.get('pai')}' does not match top2 action {top2_action}")

    # Recalculate scores and ranks independently
    scores_a = final_scores_with_reach_accepted(events_a)
    scores_b = final_scores_with_reach_accepted(events_b)
    if scores_a is None or scores_b is None:
        raise ContractError(f"Scores reconstruction failed for seed {seed}")

    ranks_a = compute_final_ranks(scores_a)
    ranks_b = compute_final_ranks(scores_b)

    score_top1 = float(scores_a[FOCAL_SEAT])
    score_top2 = float(scores_b[FOCAL_SEAT])
    rank_top1 = int(ranks_a[FOCAL_SEAT])
    rank_top2 = int(ranks_b[FOCAL_SEAT])

    pt_top1 = float(TENHOU_RANK_POINTS[rank_top1])
    pt_top2 = float(TENHOU_RANK_POINTS[rank_top2])

    delta_rank_point = pt_top2 - pt_top1
    delta_final_score = score_top2 - score_top1

    # Verify parity with manifest numbers
    if (
        score_top1 != pair["score_top1"]
        or score_top2 != pair["score_top2"]
        or rank_top1 != pair["rank_top1"]
        or rank_top2 != pair["rank_top2"]
        or pt_top1 != pair["pt_top1"]
        or pt_top2 != pair["pt_top2"]
        or delta_rank_point != pair["delta_rank_point"]
        or delta_final_score != pair["delta_final_score"]
    ):
        raise ContractError(f"Independent recalculated metrics mismatch for seed {seed}")

    return {
        "seed": seed,
        "delta_rank_point": delta_rank_point,
        "delta_final_score": delta_final_score,
        "margin": float(pair["margin"]),
    }


def adjudicate_p2_counterfactual_panel(
    panel_dir: Path = P2_PANEL_DIR,
    summary_dir: Path = P2_SUMMARY_DIR,
    allowed_root: Path = P2_ROOT,
) -> dict[str, Any]:
    """Load counterfactual_panel_manifest.json, strictly verify all raw branch logs, and produce summary."""
    check_directory_boundary(summary_dir, allowed_root)
    check_directory_boundary(panel_dir, allowed_root)
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

    # Validate exact hard gates in manifest
    panel_hard_gates = manifest.get("hard_gates", {})
    if set(panel_hard_gates.keys()) != set(EXPECTED_PANEL_HARD_GATES):
        raise ContractError(f"Manifest hard gates mismatch: {set(panel_hard_gates.keys())} vs {set(EXPECTED_PANEL_HARD_GATES)}")
    if not all(panel_hard_gates.values()):
        raise ContractError(f"Manifest contains failing hard gate: {panel_hard_gates}")

    # Validate parent K0 SHA
    _, k0_sha = resolve_k0_checkpoint()
    logged_k0 = manifest.get("parent_model", {})
    if logged_k0.get("sha256") != k0_sha:
        raise ContractError(f"Manifest parent K0 SHA mismatch: got {logged_k0.get('sha256')}, expected {k0_sha}")

    # Validate config
    cfg = manifest.get("panel_config", {})
    if cfg.get("total_pairs") != PANEL_GAMES:
        raise ContractError(f"Config total_pairs mismatch: got {cfg.get('total_pairs')}, expected {PANEL_GAMES}")
    if cfg.get("seed_start") != SEED_START or cfg.get("seed_end_exclusive") != SEED_END_EXCLUSIVE:
        raise ContractError(f"Config seed range mismatch: got [{cfg.get('seed_start')}, {cfg.get('seed_end_exclusive')})")
    if cfg.get("seed_key") != SEED_KEY or cfg.get("focal_seat") != FOCAL_SEAT or cfg.get("split_name") != SPLIT_NAME:
        raise ContractError(f"Config seating/key mismatch: {cfg}")

    pairs = manifest.get("pairs", [])
    if len(pairs) != PANEL_GAMES:
        raise ContractError(f"Pairs count mismatch: got {len(pairs)}, expected {PANEL_GAMES}")

    seeds = [p["seed"] for p in pairs]
    if seeds != list(range(SEED_START, SEED_END_EXCLUSIVE)):
        raise ContractError(f"Seeds range mismatch: got {seeds[0]}..{seeds[-1]}, expected {SEED_START}..{SEED_END_EXCLUSIVE-1}")

    # Strictly verify each raw log file and recalculate metrics independently
    recalculated_pairs = []
    for pair in pairs:
        rec = verify_and_recalculate_pair(pair, allowed_root=allowed_root)
        recalculated_pairs.append(rec)

    delta_rank_pts = np.array([p["delta_rank_point"] for p in recalculated_pairs], dtype=np.float64)
    delta_scores = np.array([p["delta_final_score"] for p in recalculated_pairs], dtype=np.float64)
    margins = np.array([p["margin"] for p in recalculated_pairs], dtype=np.float64)

    rank_summary = summarize_diff_series(delta_rank_pts, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    score_summary = summarize_diff_series(delta_scores, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)

    tight_mask = (margins <= 0.5)
    tight_rank_summary = summarize_diff_series(delta_rank_pts[tight_mask]) if np.any(tight_mask) else None
    wide_rank_summary = summarize_diff_series(delta_rank_pts[~tight_mask]) if np.any(~tight_mask) else None

    hard_gates: dict[str, bool] = {
        "manifest_verified": True,
        "k0_parent_verified": True,
        "exact_128_pairs_analyzed": (len(pairs) == PANEL_GAMES),
        "seeds_contiguous": True,
        "all_branch_logs_verified": True,
        "canonical_content_hashes_verified": True,
        "independent_metrics_recalculated_match": True,
        "bootstrap_computed": True,
    }

    if set(hard_gates.keys()) != set(EXPECTED_SUMMARY_HARD_GATES):
        raise ContractError(f"Summary hard gates key mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_SUMMARY_HARD_GATES)}")

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
