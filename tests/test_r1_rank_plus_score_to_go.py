"""Targeted unit tests for R1 rank_plus_score_to_go pilot contracts, math, and runner stubs."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    BATCH_SIZE,
    BOOTSTRAP_SEED,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    LEARNING_RATE,
    OPTIMIZER_STEPS,
    SCORE_TO_GO_SCALE,
    SCORE_TO_GO_WEIGHT,
    STEPS_START,
    STEPS_TARGET,
    TRAINING_SEED,
    compute_r1_target,
    compute_r1_target_batch,
    paired_bootstrap_ci,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
)
from training.mortal.summary_r1_rank_plus_score_to_go_2026_09 import (
    adjudicate_r1_pilot,
)


def test_1_r1_contract_invariants() -> None:
    """Test 1: Check R1 frozen constants and hard gate sets."""
    assert EXPERIMENT_ID == "R1_rank_plus_score_to_go_pilot_2026_09"
    assert TRAINING_SEED == 20260807
    assert STEPS_START == 70000
    assert STEPS_TARGET == 70400
    assert OPTIMIZER_STEPS == 400
    assert BATCH_SIZE == 512
    assert LEARNING_RATE == 1e-4
    assert SCORE_TO_GO_WEIGHT == 0.25
    assert SCORE_TO_GO_SCALE == 10000.0
    assert BOOTSTRAP_SEED == 20260906

    assert len(EXPECTED_TRAINING_HARD_GATES) == 8
    assert len(EXPECTED_EVAL_HARD_GATES) == 6
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    # Verify K0 checkpoint and M0 index paths exist
    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists()
    assert len(k0_sha) == 64

    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists()
    assert len(m0_sha) == 64


def test_2_r1_target_mathematics_and_clipping() -> None:
    """Test 2: Test scalar and vectorized R1 target calculation and clipping."""
    # Case 1: 1st place (+3.0), +15000 score gained -> diff=+15000 -> 15000/10000 = +1.5 -> +3.0 + 0.25*1.5 = +3.375
    t1 = compute_r1_target(final_rank=0, final_score=40000.0, score_at_current_kyoku_start=25000.0)
    assert t1 == 3.0 + 0.25 * 1.5
    assert t1 == 3.375

    # Case 2: 1st place (+3.0), +50000 score gained -> diff=+50000 -> 50000/10000 = +5.0, clipped to +3.0 -> +3.0 + 0.25*3.0 = +3.75
    t2 = compute_r1_target(final_rank=0, final_score=75000.0, score_at_current_kyoku_start=25000.0)
    assert t2 == 3.0 + 0.25 * 3.0
    assert t2 == 3.75

    # Case 3: 4th place (-3.0), -50000 score lost -> diff=-50000 -> -5.0, clipped to -3.0 -> -3.0 + 0.25*(-3.0) = -3.75
    t3 = compute_r1_target(final_rank=3, final_score=-25000.0, score_at_current_kyoku_start=25000.0)
    assert t3 == -3.75

    # Case 4: 2nd place (+1.0), +5000 score gained -> diff=+5000 -> score_to_go = +0.5 -> target = +1.0 + 0.25*0.5 = +1.125
    t4 = compute_r1_target(final_rank=1, final_score=30000.0, score_at_current_kyoku_start=25000.0)
    assert t4 == 1.0 + 0.25 * 0.5
    assert t4 == 1.125

    # Vectorized batch test
    ranks_t = torch.tensor([0, 0, 3, 1], dtype=torch.long)
    scores_t = torch.tensor([40000.0, 75000.0, -25000.0, 30000.0], dtype=torch.float32)
    start_t = torch.tensor([25000.0, 25000.0, 25000.0, 25000.0], dtype=torch.float32)

    batch_res = compute_r1_target_batch(ranks_t, scores_t, start_t)
    assert torch.allclose(batch_res, torch.tensor([3.375, 3.75, -3.75, 1.125], dtype=torch.float32))


def test_3_paired_bootstrap_ci() -> None:
    """Test 3: Paired bootstrap CI helper works deterministically."""
    arr = np.array([45.0, -90.0, 45.0, 0.0, 90.0, -45.0], dtype=np.float64)
    mean_val, ci = paired_bootstrap_ci(arr, reps=1000, seed=20260906)
    assert mean_val == float(np.mean(arr))
    assert ci[0] <= mean_val <= ci[1]


def test_4_adjudicate_r1_pilot_mock(tmp_path: Path) -> None:
    """Test 4: Adjudication pipeline loads manifests, parses logs, and creates valid r1_summary."""
    tr_dir = tmp_path / "training"
    ev_dir = tmp_path / "evaluation"
    sm_dir = tmp_path / "summary"
    raw_logs = ev_dir / "raw_logs"

    tr_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)
    sm_dir.mkdir(parents=True)
    raw_logs.mkdir(parents=True)

    # 1. Stubs for checkpoints and training manifest
    ctrl_ckpt = tr_dir / "mortal_control_70400.pth"
    var_ckpt = tr_dir / "mortal_variant_70400.pth"
    ctrl_ckpt.write_text("ctrl_mock")
    var_ckpt.write_text("var_mock")

    _, k0_sha = resolve_k0_checkpoint()
    tr_manifest = {
        "schema": "keqing.mortal.r1_training_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "verdict": "training_completed",
        "hard_gates": {g: True for g in EXPECTED_TRAINING_HARD_GATES},
    }
    (tr_dir / "r1_training_manifest.json").write_text(json.dumps(tr_manifest))

    # 2. Synthetic raw logs for 1000 games (4 shards x 250 seeds x 4 splits)
    # To keep unit test fast, write synthetic logs
    for shard_idx in range(4):
        shard_dir = raw_logs / f"shard_{shard_idx:03d}"
        shard_dir.mkdir(parents=True)
        s_start = 2200000 + shard_idx * 250
        for s in range(s_start, s_start + 250):
            for split in ["a", "b", "c", "d"]:
                log_path = shard_dir / f"{s}_8192_{split}.json.gz"
                events = [
                    {"type": "start_game", "names": ["K0_70k", "ext_mortal", "Control_70400", "Variant_70400"]},
                    {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
                    {"type": "hora", "deltas": [0, 0, -8000, 8000]},  # Variant wins over Control
                    {"type": "end_game"},
                ]
                with gzip.open(log_path, "wt", encoding="utf-8") as f:
                    for ev in events:
                        f.write(json.dumps(ev) + "\n")

    ev_manifest = {
        "schema": "keqing.mortal.r1_eval_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "verdict": "evaluation_completed",
        "hard_gates": {g: True for g in EXPECTED_EVAL_HARD_GATES},
    }
    (ev_dir / "r1_eval_manifest.json").write_text(json.dumps(ev_manifest))

    summary = adjudicate_r1_pilot(training_dir=tr_dir, eval_dir=ev_dir, summary_dir=sm_dir)
    assert summary["schema"] == "keqing.mortal.r1_summary.v1"
    assert summary["experiment_id"] == EXPERIMENT_ID
    assert summary["hard_gates"]["all_logs_verified"] is True
    assert summary["metrics"]["primary_contrast_variant_minus_control"]["mean_pt"] > 0
    assert summary["verdict"] == "r1_pilot_promising"
