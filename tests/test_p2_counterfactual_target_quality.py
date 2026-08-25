"""Targeted unit tests for P2 counterfactual target quality generation and summary contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.generate_p2_counterfactual_panel_2026_09 import (
    generate_single_counterfactual_pair,
)
from training.mortal.p2_counterfactual_target_quality_contract_2026_09 import (
    EXPERIMENT_ID,
    FOCAL_SEAT,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    SPLIT_NAME,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    resolve_k0_checkpoint,
)
from training.mortal.summary_p2_counterfactual_target_quality_2026_09 import (
    adjudicate_p2_counterfactual_panel,
    summarize_diff_series,
)


def test_1_p2_contract_invariants_and_seed_disjointness() -> None:
    """Test 1: P2 contract invariants: exactly 128 games, seeds 3000000..3000128, seat 0, split 'a'."""
    assert PANEL_GAMES == 128
    assert SEED_START == 3000000
    assert SEED_END_EXCLUSIVE == 3000128
    assert SEED_KEY == 8192
    assert FOCAL_SEAT == 0
    assert SPLIT_NAME == "a"
    assert len(TENHOU_RANK_POINTS) == 4

    # Seed disjointness against O2 (2000000..2000640 for training, 2100000..2101000 for eval)
    p2_seeds = set(range(SEED_START, SEED_END_EXCLUSIVE))
    o2_train_seeds = set(range(2000000, 2000641))
    o2_eval_seeds = set(range(2100000, 2101000))
    assert len(p2_seeds.intersection(o2_train_seeds)) == 0
    assert len(p2_seeds.intersection(o2_eval_seeds)) == 0


def test_2_single_pair_generation_and_hard_divergence_check(tmp_path: Path) -> None:
    """Test 2: Single pair generation strictly enforces bit-exact prefix, single intervention, and dahai divergence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_logs = tmp_path / "raw_logs"
    raw_logs.mkdir()

    pair = generate_single_counterfactual_pair(
        seed=3000000,
        seed_key=8192,
        raw_logs_dir=raw_logs,
        device=device,
    )

    assert pair["seed"] == 3000000
    assert pair["seed_key"] == 8192
    assert pair["focal_seat"] == 0
    assert pair["divergence_event_index"] > 0

    # Divergence must be distinct dahai actions
    assert pair["branch_a_dahai_pai"] != pair["branch_b_dahai_pai"]
    assert pair["top1_action"] != pair["top2_action"]

    # Scores and ranks must be bounded
    assert 0 <= pair["rank_top1"] <= 3
    assert 0 <= pair["rank_top2"] <= 3
    assert pair["pt_top1"] in TENHOU_RANK_POINTS
    assert pair["pt_top2"] in TENHOU_RANK_POINTS
    assert pair["delta_rank_point"] == pair["pt_top2"] - pair["pt_top1"]
    assert pair["delta_final_score"] == pair["score_top2"] - pair["score_top1"]


def test_3_fixed_seed_rerun_bit_exact_reproducibility(tmp_path: Path) -> None:
    """Test 3: Generating the same seed twice in independent directories produces bit-exact identical metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_1 = tmp_path / "run_1"
    raw_2 = tmp_path / "run_2"
    raw_1.mkdir()
    raw_2.mkdir()

    p1 = generate_single_counterfactual_pair(seed=3000001, seed_key=8192, raw_logs_dir=raw_1, device=device)
    p2 = generate_single_counterfactual_pair(seed=3000001, seed_key=8192, raw_logs_dir=raw_2, device=device)

    assert p1["target_context"] == p2["target_context"]
    assert p1["top1_action"] == p2["top1_action"]
    assert p1["top2_action"] == p2["top2_action"]
    assert p1["divergence_event_index"] == p2["divergence_event_index"]
    assert p1["branch_a_dahai_pai"] == p2["branch_a_dahai_pai"]
    assert p1["branch_b_dahai_pai"] == p2["branch_b_dahai_pai"]
    assert p1["rank_top1"] == p2["rank_top1"]
    assert p1["rank_top2"] == p2["rank_top2"]
    assert p1["score_top1"] == p2["score_top1"]
    assert p1["score_top2"] == p2["score_top2"]
    assert p1["delta_rank_point"] == p2["delta_rank_point"]
    assert p1["delta_final_score"] == p2["delta_final_score"]


def test_4_diff_series_summary_and_bootstrap_ci() -> None:
    """Test 4: summarize_diff_series accurately calculates rates, counts, mean, and deterministic CI."""
    diffs = np.array([45.0, -90.0, 0.0, 90.0, -45.0, 0.0, 45.0, 0.0], dtype=np.float64)
    summary = summarize_diff_series(diffs, reps=1000, seed=12345)

    assert summary["total_pairs"] == 8
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == 2
    assert summary["tie_count"] == 3
    assert summary["nonzero_count"] == 5
    assert summary["nonzero_rate"] == 5.0 / 8.0
    assert summary["positive_rate"] == 3.0 / 8.0
    assert summary["negative_rate"] == 2.0 / 8.0
    assert summary["tie_rate"] == 3.0 / 8.0
    assert summary["mean_diff"] == float(np.mean(diffs))
    assert len(summary["ci95"]) == 2
    assert summary["ci95"][0] <= summary["mean_diff"] <= summary["ci95"][1]


def test_5_adjudicate_p2_counterfactual_panel_mock(tmp_path: Path) -> None:
    """Test 5: Summarizer validates manifest schema, parent K0, pairs count, and writes formal summary."""
    panel_dir = tmp_path / "counterfactual_panel"
    summary_dir = tmp_path / "summary"
    panel_dir.mkdir()
    summary_dir.mkdir()

    _, k0_sha = resolve_k0_checkpoint()

    # Generate 128 synthetic pairs
    pairs = []
    for s in range(SEED_START, SEED_END_EXCLUSIVE):
        pairs.append({
            "seed": s,
            "seed_key": SEED_KEY,
            "focal_seat": FOCAL_SEAT,
            "target_context": [s, SEED_KEY, FOCAL_SEAT, 0, 1],
            "top1_action": 10,
            "top2_action": 11,
            "top1_q": 0.5,
            "top2_q": 0.2,
            "margin": 0.3,
            "divergence_event_index": 5,
            "branch_a_dahai_pai": "2p",
            "branch_b_dahai_pai": "3p",
            "branch_a_total_events": 100,
            "branch_b_total_events": 100,
            "branch_a_log_path": f"/dummy/{s}_a.json.gz",
            "branch_b_log_path": f"/dummy/{s}_b.json.gz",
            "rank_top1": 1,
            "rank_top2": 2,
            "score_top1": 30000.0,
            "score_top2": 20000.0,
            "pt_top1": 45.0,
            "pt_top2": 0.0,
            "delta_rank_point": -45.0,
            "delta_final_score": -10000.0,
        })

    manifest = {
        "schema": PANEL_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "panel_config": {
            "total_pairs": 128,
            "seed_start": SEED_START,
            "seed_end_exclusive": SEED_END_EXCLUSIVE,
            "seed_key": SEED_KEY,
            "focal_seat": FOCAL_SEAT,
            "split_name": SPLIT_NAME,
            "device": "cpu",
        },
        "hard_gates": {"all_pass": True},
        "pairs": pairs,
        "verdict": "panel_generation_completed",
    }
    manifest_path = panel_dir / "counterfactual_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    summary = adjudicate_p2_counterfactual_panel(
        panel_dir=panel_dir,
        summary_dir=summary_dir,
        allowed_root=tmp_path,
    )

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["experiment_id"] == EXPERIMENT_ID
    assert summary["hard_gates"]["manifest_verified"] is True
    assert summary["hard_gates"]["exact_128_pairs_analyzed"] is True
    assert summary["promotion"]["recipe_promotion"] is False
    assert summary["promotion"]["checkpoint_promotion"] is False
    assert summary["promotion"]["k1"] is None
    assert (summary_dir / "p2_summary.json").exists()
