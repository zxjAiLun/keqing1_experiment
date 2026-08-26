"""Targeted unit tests for R1 rank_plus_score_to_go pilot contracts, math, and runner stubs."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

from training.mortal.objective import compute_objective_losses
from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    ADAM_BETAS,
    ADAM_EPS,
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
    WEIGHT_DECAY,
    adjudicate_r1_verdict,
    compute_r1_target,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
)
from training.mortal.summary_r1_rank_plus_score_to_go_2026_09 import (
    adjudicate_r1_pilot,
)
from training.mortal.train_r1_rank_plus_score_to_go_2026_09 import (
    _build_dataloader,
    _build_optimizer_and_models,
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
    assert WEIGHT_DECAY == 0.1
    assert ADAM_BETAS == (0.9, 0.999)
    assert ADAM_EPS == 1e-8
    assert SCORE_TO_GO_WEIGHT == 0.25
    assert SCORE_TO_GO_SCALE == 10000.0
    assert BOOTSTRAP_SEED == 20260906

    assert len(EXPECTED_TRAINING_HARD_GATES) == 9
    assert len(EXPECTED_EVAL_HARD_GATES) == 7
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    # Verify K0, ext_mortal checkpoint, and M0 index paths exist
    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists()
    assert len(k0_sha) == 64

    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    assert ext_path.exists()
    assert len(ext_sha) == 64

    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists()
    assert len(m0_sha) == 64


def test_2_r1_target_mathematics_and_clipping() -> None:
    """Test 2: Test scalar R1 target calculation and clipping."""
    # Centered pts: [3.0, 1.0, -1.0, -3.0]
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


def test_3_real_k0_and_m0_one_step_zero_update_smoke() -> None:
    """Test 3: Verify real K0 parent + real M0 batch one-step zero-update forward & loss computation."""
    k0_path, _ = resolve_k0_checkpoint()
    m0_path, _ = resolve_m0_dataset_index()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    brain, dqn, aux_net, _optimizer = _build_optimizer_and_models(k0_path, device)
    dl_ctrl = _build_dataloader(m0_path, seed=20260807, reward_mode="final_rank_mc", batch_size=32)
    batch_ctrl = next(iter(dl_ctrl))

    dl_var = _build_dataloader(m0_path, seed=20260807, reward_mode="rank_plus_score_to_go_mc", batch_size=32)
    batch_var = next(iter(dl_var))

    # 1. Verify identical row identity (obs, actions, masks, player_ranks) between control and variant
    assert torch.allclose(batch_ctrl[0], batch_var[0])  # obs
    assert torch.equal(batch_ctrl[1], batch_var[1])    # actions
    assert torch.equal(batch_ctrl[2], batch_var[2])    # masks
    assert torch.equal(batch_ctrl[3], batch_var[3])    # steps_to_done
    assert torch.equal(batch_ctrl[5], batch_var[5])    # player_ranks

    # 2. Verify kyoku_rewards are different due to score_to_go
    assert not torch.allclose(batch_ctrl[4], batch_var[4])

    # 3. Test forward and objective losses on control
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = [t.to(device) for t in batch_ctrl]
    q_target_mc = (1.0 ** steps_to_done * kyoku_rewards).to(torch.float32)

    phi = brain(obs)
    q_out = dqn(phi, masks)
    (next_rank_logits,) = aux_net(phi)

    losses = compute_objective_losses(
        q_out=q_out,
        masks=masks,
        actions=actions,
        q_target_mc=q_target_mc,
        next_rank_logits=next_rank_logits,
        player_ranks=player_ranks,
        mode="legal_mean_mc",
        cql_weight=5.0,
        aux_weight=0.2,
    )
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["value_loss"])
    assert torch.isfinite(losses["cql_loss"])
    assert torch.isfinite(losses["next_rank_loss"])


def test_4_adjudicate_r1_pilot_verdicts_and_mock(tmp_path: Path) -> None:
    """Test 4: Adjudication pipeline loads manifests, parses logs, tests 4-level verdicts, and creates valid r1_summary."""
    # Test 4-level verdict logic
    assert adjudicate_r1_verdict(primary_mean=2.5, primary_ci_lower=0.5) == "strong_positive"
    assert adjudicate_r1_verdict(primary_mean=2.5, primary_ci_lower=-0.5) == "weak_positive"
    assert adjudicate_r1_verdict(primary_mean=-1.0, primary_ci_lower=-3.0) == "not_promising"
    assert adjudicate_r1_verdict(primary_mean=0.0, primary_ci_lower=-2.0) == "not_promising"

    tr_dir = tmp_path / "training"
    ev_dir = tmp_path / "evaluation"
    sm_dir = tmp_path / "summary"

    tr_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)
    sm_dir.mkdir(parents=True)

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

    # 2. Synthetic raw logs for exact 1000 games across 4 shards (250 logs per shard)
    for shard_idx in range(4):
        shard_dir = ev_dir / f"shard_{shard_idx:03d}"
        logs_dir = shard_dir / "logs"
        logs_dir.mkdir(parents=True)
        s_start = 2200000 + shard_idx * 250
        for s in range(s_start, s_start + 250):
            log_path = logs_dir / f"{s}_8192_0.json.gz"
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
    assert summary["metrics"]["total_games"] == 1000
    assert summary["metrics"]["primary_contrast_variant_minus_control"]["mean_pt"] > 0
    assert summary["verdict"] == "strong_positive"
