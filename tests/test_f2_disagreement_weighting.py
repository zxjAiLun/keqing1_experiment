"""Targeted unit tests for F2 disagreement-weighting-without-resampling pilot contracts and runners."""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

from training.mortal.f2_disagreement_weighting_contract_2026_09 import (
    AGREEMENT_WEIGHT,
    BOOTSTRAP_SEED,
    CANONICAL_K1_SEED,
    DISAGREEMENT_WEIGHT,
    EVAL_GAMES_PER_PANEL,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_TOTAL_GAMES,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FROZEN_R2_CONTROL,
    MAIN_REWARD_MODE,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    ROW_IDENTITY_FIELDS,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_SEEDS,
    ContractError,
    adjudicate_f2_verdict,
    compute_f2_row_weights,
    crossed_bootstrap_ci,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    sha256_file,
    verify_training_manifest,
)
from training.mortal.summary_f2_disagreement_weighting_2026_09 import adjudicate_f2_weighting
from training.mortal.train_f2_disagreement_weighting_2026_09 import (
    _build_dataloader,
    _build_frozen_scorer,
    _build_models,
    _build_optimizer,
    _verify_optimizer_structure,
    _verify_q_target_values,
    compute_high_signal_mask,
    compute_weighted_f2_losses,
    _scorer_parameter_digest,
)
import model


def test_1_f2_contract_invariants() -> None:
    """Test 1: Frozen constants, frozen R2 Control binding, and gate sets."""
    assert EXPERIMENT_ID == "F2_disagreement_weighting_without_resampling_pilot_2026_09"
    assert TRAINING_SEEDS == [20260910, 20260911, 20260912]
    assert CANONICAL_K1_SEED == 20260911
    assert OPTIMIZER_STEPS == 400
    assert BOOTSTRAP_SEED == 20261001
    assert EVAL_GAMES_PER_PANEL == 1000
    assert EVAL_TOTAL_GAMES == 3000
    assert EVAL_SEED_START == 2600000

    assert MAIN_REWARD_MODE == "final_rank_mc"
    assert OBJECTIVE_MODE == "behavior_action_mc"
    assert OBJECTIVE_VALUE_STATISTIC == "behavior_action_q"
    assert TRAINABLE_PLAYER_NAMES == ("ext_mortal",)

    assert DISAGREEMENT_WEIGHT == 2.0
    assert AGREEMENT_WEIGHT == 1.0

    assert len(EXPECTED_TRAINING_HARD_GATES) == 12
    assert len(EXPECTED_EVAL_HARD_GATES) == 7
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    for s in TRAINING_SEEDS:
        path, ckpt_sha, row_sha = resolve_r2_control_checkpoint(s)
        assert path.exists()
        assert ckpt_sha == FROZEN_R2_CONTROL[s]["checkpoint_sha256"]
        assert row_sha == FROZEN_R2_CONTROL[s]["row_identity_sha256"]

    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists() and len(k0_sha) == 64
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    assert ext_path.exists() and len(ext_sha) == 64
    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists() and len(m0_sha) == 64


def test_2_row_weights_and_no_resampling_semantics() -> None:
    """Test 2: weights 2.0/1.0, length-preserving, no resampling/dropping/duplication."""
    # Mixed batch.
    disagreement = np.zeros(512, dtype=bool)
    disagreement[[0, 5, 100, 511]] = True
    w = compute_f2_row_weights(disagreement)
    assert w.shape == (512,)
    assert w.dtype == np.float64
    assert set(np.unique(w).tolist()) == {AGREEMENT_WEIGHT, DISAGREEMENT_WEIGHT}
    assert float(w.sum()) == 4 * DISAGREEMENT_WEIGHT + 508 * AGREEMENT_WEIGHT
    # Weight positions track the disagreement mask exactly.
    assert np.array_equal(w == DISAGREEMENT_WEIGHT, disagreement)
    assert np.array_equal(w == AGREEMENT_WEIGHT, ~disagreement)

    # All-agreement batch: only weight 1.0.
    w0 = compute_f2_row_weights(np.zeros(512, dtype=bool))
    assert set(np.unique(w0).tolist()) == {AGREEMENT_WEIGHT}
    # All-disagreement batch: only weight 2.0.
    w1 = compute_f2_row_weights(np.ones(512, dtype=bool))
    assert set(np.unique(w1).tolist()) == {DISAGREEMENT_WEIGHT}

    # Length preservation means each row used exactly once (no resampling).
    assert w0.size == 512 and w1.size == 512


def test_3_weighted_loss_matches_unweighted_when_weights_equal() -> None:
    """Test 3: Uniform weights reproduce the legacy unweighted objective exactly."""
    torch.manual_seed(0)
    B, A = 6, 46
    masks = torch.zeros(B, A, dtype=torch.bool)
    masks[:, :8] = True
    q_raw = torch.randn(B, A, dtype=torch.float64, requires_grad=True)
    q_out = q_raw.masked_fill(~masks, -torch.inf)
    actions = torch.tensor([0, 1, 2, 3, 4, 5])
    targets = torch.tensor([3.0, 1.0, -1.0, -3.0, 1.0, -1.0], dtype=torch.float64)
    logits = torch.randn(B, 4, dtype=torch.float64, requires_grad=True)
    ranks = torch.tensor([0, 1, 2, 3, 0, 1])

    uniform = torch.ones(B, dtype=torch.float64)
    weighted = compute_weighted_f2_losses(
        q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
        next_rank_logits=logits, player_ranks=ranks, row_weights=uniform,
        cql_weight=5.0, aux_weight=0.2,
    )

    from training.mortal.objective import compute_objective_losses
    legacy = compute_objective_losses(
        q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
        next_rank_logits=logits, player_ranks=ranks,
        mode="behavior_action_mc", cql_weight=5.0, aux_weight=0.2,
    )
    for key in ("value_loss", "preference_loss", "next_rank_loss", "total_loss"):
        assert torch.allclose(weighted[key], legacy[key], atol=1e-12, rtol=1e-12), key

    # Gradients also match under uniform weights.
    g1 = torch.autograd.grad(weighted["total_loss"], q_raw, retain_graph=True)[0]
    g2 = torch.autograd.grad(legacy["total_loss"], q_raw, retain_graph=True)[0]
    assert torch.allclose(g1, g2, atol=1e-12, rtol=1e-12)


def test_4_weighted_loss_formula_exactness() -> None:
    """Test 4: weighted_mean formula, gradient flow, and fail-closed weight validation."""
    torch.manual_seed(1)
    B = 5
    masks = torch.ones(B, 4, dtype=torch.bool)
    q_out = torch.randn(B, 4, dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([0, 1, 2, 3, 0])
    targets = torch.tensor([3.0, 1.0, -1.0, -3.0, 1.0], dtype=torch.float64)
    logits = torch.randn(B, 4, dtype=torch.float64, requires_grad=True)
    ranks = torch.tensor([0, 1, 2, 3, 0])
    weights = torch.tensor([2.0, 1.0, 2.0, 1.0, 2.0], dtype=torch.float64)

    out = compute_weighted_f2_losses(
        q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
        next_rank_logits=logits, player_ranks=ranks, row_weights=weights,
        cql_weight=5.0, aux_weight=0.2,
    )

    behavior_q = q_out[torch.arange(B), actions]
    value_row = 0.5 * (behavior_q - targets) ** 2
    cql_row = q_out.logsumexp(-1) - behavior_q
    ce_row = torch.nn.functional.cross_entropy(logits, ranks, reduction="none")
    wsum = weights.sum()
    expected_value = (weights * value_row).sum() / wsum
    expected_cql = (weights * cql_row).sum() / wsum
    expected_ce = (weights * ce_row).sum() / wsum
    assert torch.allclose(out["value_loss"], expected_value, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["preference_loss"], expected_cql, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["next_rank_loss"], expected_ce, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["total_loss"], expected_value + 5.0 * expected_cql + 0.2 * expected_ce, atol=1e-12, rtol=1e-12)

    # Weighted loss differentiates from unweighted when weights differ.
    uniform = compute_weighted_f2_losses(
        q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
        next_rank_logits=logits, player_ranks=ranks, row_weights=torch.ones(B, dtype=torch.float64),
        cql_weight=5.0, aux_weight=0.2,
    )
    assert not torch.allclose(out["total_loss"], uniform["total_loss"])

    # Gradient flows (weights are constants; grads exist on q/logits).
    grads = torch.autograd.grad(out["total_loss"], [q_out, logits], retain_graph=False)
    assert bool(torch.isfinite(grads[0]).all()) and bool(torch.isfinite(grads[1]).all())
    assert float(grads[0].abs().sum()) > 0 and float(grads[1].abs().sum()) > 0

    # Nonpositive / nonfinite / wrong-length weights fail closed.
    with pytest.raises(ContractError):
        compute_weighted_f2_losses(
            q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
            next_rank_logits=logits, player_ranks=ranks,
            row_weights=torch.zeros(B, dtype=torch.float64), cql_weight=5.0, aux_weight=0.2,
        )
    with pytest.raises(ContractError):
        compute_weighted_f2_losses(
            q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
            next_rank_logits=logits, player_ranks=ranks,
            row_weights=torch.full((B,), float("nan"), dtype=torch.float64), cql_weight=5.0, aux_weight=0.2,
        )
    with pytest.raises(ContractError):
        compute_weighted_f2_losses(
            q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
            next_rank_logits=logits, player_ranks=ranks,
            row_weights=torch.ones(B + 1, dtype=torch.float64), cql_weight=5.0, aux_weight=0.2,
        )


def test_5_frozen_scorer_and_disagreement_parity_with_f1() -> None:
    """Test 5: Frozen scorer bit-exact; disagreement identical to F1's determination."""
    k0_path, _ = resolve_k0_checkpoint()
    scorer_brain, scorer_dqn = _build_frozen_scorer(k0_path, "cpu")
    digest_before = _scorer_parameter_digest(scorer_brain, scorer_dqn)

    obs = torch.randn(12, 1012, 34)
    masks = torch.zeros(12, 46, dtype=torch.bool)
    masks[:, :10] = True
    actions = torch.zeros(12, dtype=torch.int64)

    disagreement, greedy = compute_high_signal_mask(scorer_brain, scorer_dqn, obs, actions, masks)
    # F1's determination is the same function; compare to its module directly.
    from training.mortal.train_f1_disagreement_prioritized_replay_2026_09 import (
        compute_high_signal_mask as f1_mask,
    )
    d_f1, g_f1 = f1_mask(scorer_brain, scorer_dqn, obs, actions, masks)
    assert np.array_equal(disagreement, d_f1)
    assert np.array_equal(greedy, g_f1)

    digest_after = _scorer_parameter_digest(scorer_brain, scorer_dqn)
    assert digest_before == digest_after

    # Scorer stays out of the training optimizer.
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)
    scorer_ids = {id(p) for p in list(scorer_brain.parameters()) + list(scorer_dqn.parameters())}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert scorer_ids.isdisjoint(opt_ids)

    # Illegal behavior action fails closed.
    bad_masks = torch.zeros(12, 46, dtype=torch.bool)
    bad_masks[:, 0] = True
    with pytest.raises(ContractError):
        compute_high_signal_mask(scorer_brain, scorer_dqn, obs, torch.full((12,), 3, dtype=torch.int64), bad_masks)


def test_6_optimizer_preserved_k0_moments_410() -> None:
    """Test 6: Optimizer restores exactly 410 K0 moments in two groups [165, 245]."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)

    assert len(optimizer.param_groups) == 2
    assert [len(g["params"]) for g in optimizer.param_groups] == [165, 245]
    assert len(optimizer.state) == 410
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0
    _verify_optimizer_structure(optimizer)


def test_7_q_target_domain_and_evaluator_lineup() -> None:
    """Test 7: Main Q target restricted to final_rank_mc domain; evaluator lineup frozen."""
    _verify_q_target_values(torch.tensor([3.0, 1.0, -1.0, -3.0]))
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([3.0, 0.25]))
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([2.0]))

    from training.mortal.eval_f2_disagreement_weighting_2026_09 import _lineup_for_seed
    for s in TRAINING_SEEDS:
        assert _lineup_for_seed(s) == (
            "K0_70k",
            "ext_mortal",
            f"R2_Control_seed_{s}",
            f"F2_WeightedVariant_seed_{s}",
        )


def _write_game_log(log_path: Path, seed: int, names: list[str], *, variant_wins: bool = True) -> None:
    if variant_wins:
        deltas = [-8000, -6000, -4000, 30000]
    else:
        deltas = [30000, -8000, -6000, -4000]
    events = [
        {"type": "start_game", "seed": [seed, EVAL_SEED_KEY], "names": names},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": deltas},
        {"type": "end_game"},
    ]
    with gzip.open(log_path, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _build_strict_fixture(root: Path) -> dict:
    """Build a complete, contract-conformant F2 artifact fixture with real a/b/c/d logs and real SHAs."""
    tr_dir = root / "training"
    ev_dir = root / "evaluation"
    sm_dir = root / "summary"
    tr_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)
    sm_dir.mkdir(parents=True)

    _, k0_sha = resolve_k0_checkpoint()
    _, ext_sha = resolve_ext_mortal_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    checkpoints_manifest: dict = {}
    row_by_seed: dict = {}
    for s in TRAINING_SEEDS:
        r2c_path, r2c_sha, r2c_row = resolve_r2_control_checkpoint(s)
        v_path = tr_dir / f"mortal_weighted_variant_70400_seed_{s}.pth"
        v_path.write_bytes(f"wt_variant_{s}".encode())
        v_sha = sha256_file(v_path)
        checkpoints_manifest[f"seed_{s}"] = {
            "r2_control": {"name": r2c_path.name, "path": str(r2c_path), "sha256": r2c_sha},
            "weighted_variant": {"name": v_path.name, "path": str(v_path), "sha256": v_sha},
        }
        per_step = [
            {
                "step": i + 1,
                "rows_used": 512,
                "disagreement_count": 56,
                "disagreement_rate": 56 / 512,
                "weights_unique": [1.0, 2.0],
                "weight_sum": 56 * 2.0 + (512 - 56) * 1.0,
            }
            for i in range(OPTIMIZER_STEPS)
        ]
        row_by_seed[f"seed_{s}"] = {
            "r2_control_sha256": r2c_row,
            "base_row_sha256": r2c_row,  # must equal frozen R2 control digest
            "matches_r2_control": True,
            "weighting_stats": {
                "batches": OPTIMIZER_STEPS,
                "rows_per_batch": 512,
                "disagreement_count_min": 56,
                "disagreement_count_max": 60,
                "disagreement_rate_mean": 0.11,
                "per_step": per_step,
            },
        }

    tr_man = {
        "schema": "keqing.mortal.f2_training_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "frozen_scorer": {
            "name": "K0_70k",
            "sha256": k0_sha,
            "mode": "eval",
            "amp": False,
            "dtype": "float32",
            "parameters_bit_exact": True,
        },
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "objective": {"mode": OBJECTIVE_MODE, "value_statistic": OBJECTIVE_VALUE_STATISTIC, "preference_loss": "existing_cql"},
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [6.0, 4.0, 2.0, 0.0]},
        "disagreement_weighting": {
            "disagreement_weight": DISAGREEMENT_WEIGHT,
            "agreement_weight": AGREEMENT_WEIGHT,
            "resampling": False,
            "row_usage": "each_row_exactly_once",
            "normalization": "batch_weight_sum",
            "applied_to": ["value_loss_row", "cql_loss_row", "next_rank_loss_row"],
            "q_margin_threshold": None,
            "uses_reward_or_eval_signal": False,
        },
        "training_config": {"training_seeds": TRAINING_SEEDS, "device": "cpu"},
        "checkpoints": checkpoints_manifest,
        "row_identity": {"fields": [name for name, _ in ROW_IDENTITY_FIELDS], "excluded_field": "kyoku_rewards", "by_seed": row_by_seed},
        "hard_gates": {g: True for g in EXPECTED_TRAINING_HARD_GATES},
        "verdict": "training_completed",
    }
    tr_path = tr_dir / "f2_training_manifest.json"
    tr_path.write_text(json.dumps(tr_man), encoding="utf-8")
    tr_sha = sha256_file(tr_path)

    for s in TRAINING_SEEDS:
        panel_dir = ev_dir / f"panel_seed_{s}"
        lineup = ["K0_70k", "ext_mortal", f"R2_Control_seed_{s}", f"F2_WeightedVariant_seed_{s}"]
        for shard_idx in range(4):
            logs_dir = panel_dir / f"shard_{shard_idx:03d}" / "logs"
            logs_dir.mkdir(parents=True)
            for g_id in range(2600000 + shard_idx * 250, 2600000 + (shard_idx + 1) * 250):
                suffix = ["a", "b", "c", "d"][g_id % 4]
                _write_game_log(logs_dir / f"{g_id}_{EVAL_SEED_KEY}_{suffix}.json.gz", g_id, lineup, variant_wins=True)

    ev_man = {
        "schema": "keqing.mortal.f2_eval_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_path), "sha256": tr_sha},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "ext_mortal_model": {"name": "ext_mortal", "path": "mock", "sha256": ext_sha},
        "eval_config": {"panels_count": 3, "games_per_panel": 1000, "total_games": 3000},
        "panels": {
            f"seed_{s}": {
                "training_seed": s,
                "panel_dir": str(ev_dir / f"panel_seed_{s}"),
                "games_count": 1000,
                "models": {
                    "r2_control": {"sha256": checkpoints_manifest[f"seed_{s}"]["r2_control"]["sha256"]},
                    "weighted_variant": {"sha256": checkpoints_manifest[f"seed_{s}"]["weighted_variant"]["sha256"]},
                },
            }
            for s in TRAINING_SEEDS
        },
        "hard_gates": {g: True for g in EXPECTED_EVAL_HARD_GATES},
        "total_games_evaluated": 3000,
        "verdict": "evaluation_completed",
    }
    ev_path = ev_dir / "f2_eval_manifest.json"
    ev_path.write_text(json.dumps(ev_man), encoding="utf-8")

    return {"tr_dir": tr_dir, "ev_dir": ev_dir, "sm_dir": sm_dir, "tr_path": tr_path, "ev_path": ev_path}


def test_8_f2_summary_pipeline_and_fail_closed(tmp_path: Path) -> None:
    """Test 8: Full summary pipeline plus fail-closed rejections for all tamper classes."""
    fx = _build_strict_fixture(tmp_path / "valid")
    summary = adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])
    assert summary["schema"] == "keqing.mortal.f2_summary.v1"
    assert summary["hard_gates"]["all_3000_logs_verified"] is True
    assert summary["metrics"]["total_games"] == 3000
    assert summary["training_manifest"]["sha256"] == sha256_file(fx["tr_path"])
    assert summary["eval_manifest"]["sha256"] == sha256_file(fx["ev_path"])
    # Variant wins over both Control and K0 in the mock logs -> reweighting_promising.
    assert summary["verdict"] == "reweighting_promising"
    # Adaptive pilot: never promotes K1 even on a fully passing verdict.
    assert summary["promotion"]["recipe_promotion"] is False
    assert summary["promotion"]["checkpoint_promotion"] is False
    assert summary["promotion"]["k1"] is None
    assert summary["pilot_policy"] == "adaptive pilot: no verdict directly promotes K1"
    assert summary["weighting_stats"]["seed_20260910"]["disagreement_count_min"] == 56

    # (a) Base row digest mismatch with frozen R2 Control must be rejected.
    fx = _build_strict_fixture(tmp_path / "row_mismatch")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260910"]["base_row_sha256"] = "0" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (b) Weight tamper (3.0) must be rejected.
    fx = _build_strict_fixture(tmp_path / "weight_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["disagreement_weighting"]["disagreement_weight"] = 3.0
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (c) Resampling introduction must be rejected.
    fx = _build_strict_fixture(tmp_path / "resampling_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["disagreement_weighting"]["resampling"] = True
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (d) Q-margin threshold introduction must be rejected.
    fx = _build_strict_fixture(tmp_path / "margin_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["disagreement_weighting"]["q_margin_threshold"] = 0.5
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (e) Reward shaping leakage into main_reward must be rejected.
    fx = _build_strict_fixture(tmp_path / "main_reward_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["main_reward"]["mode"] = "rank_plus_score_to_go_mc"
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (f) Frozen R2 Control checkpoint SHA tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "r2c_sha_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["checkpoints"]["seed_20260911"]["r2_control"]["sha256"] = "f" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (g) Checkpoint disk tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "ckpt_tamper")
    (fx["tr_dir"] / "mortal_weighted_variant_70400_seed_20260910.pth").write_bytes(b"tampered")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (h) Duplicate game ID must be rejected.
    fx = _build_strict_fixture(tmp_path / "dup_id")
    logs = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs"
    (logs / f"{2600001}_{EVAL_SEED_KEY}_b.json.gz").unlink()
    shutil.copy(logs / f"{2600000}_{EVAL_SEED_KEY}_a.json.gz", logs / f"{2600000}_{EVAL_SEED_KEY}_c.json.gz")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (i) Wrong lineup in one log must be rejected.
    fx = _build_strict_fixture(tmp_path / "lineup_tamper")
    bad = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs" / f"{2600000}_{EVAL_SEED_KEY}_a.json.gz"
    _write_game_log(bad, 2600000, ["K0_70k", "ext_mortal", "R2_Control_seed_20260910", "Intruder"])
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (j) Training-manifest SHA chain rewrite must be rejected.
    fx = _build_strict_fixture(tmp_path / "sha_chain")
    with open(fx["tr_path"], "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (k) Missing eval hard gate must be rejected.
    fx = _build_strict_fixture(tmp_path / "gate_tamper")
    data = json.loads(fx["ev_path"].read_text(encoding="utf-8"))
    data["hard_gates"].pop("zero_missing_games")
    fx["ev_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (l) Per-step rows_used tamper (dropped rows) must be rejected.
    fx = _build_strict_fixture(tmp_path / "rows_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260912"]["weighting_stats"]["per_step"][0]["rows_used"] = 256
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (m) Per-step weight_sum inconsistency must be rejected.
    fx = _build_strict_fixture(tmp_path / "wsum_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260910"]["weighting_stats"]["per_step"][0]["weight_sum"] = 999.0
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (n) Scorer bit-exact flag false must be rejected.
    fx = _build_strict_fixture(tmp_path / "scorer_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["frozen_scorer"]["parameters_bit_exact"] = False
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])


def test_9_crossed_bootstrap_and_three_state_verdict() -> None:
    """Test 9: Crossed bootstrap shared resampling and three-state F2 pilot verdict logic."""
    rng = np.random.default_rng(42)
    mat1 = rng.normal(loc=5.0, scale=20.0, size=(3, 1000))
    mat2 = mat1 + 2.0

    m1, ci1, idx = crossed_bootstrap_ci(mat1, reps=500, seed=20261001, ci=95.0, return_sampled_indices=True)
    assert idx is not None
    m2, ci2, _ = crossed_bootstrap_ci(mat2, reps=500, seed=20261001, ci=95.0, shared_indices=idx)
    assert abs(m2 - (m1 + 2.0)) < 1e-6
    assert abs(ci2[0] - (ci1[0] + 2.0)) < 1e-6
    assert abs(ci2[1] - (ci1[1] + 2.0)) < 1e-6

    # reweighting_promising: both primary and absolute fully pass; never promotes K1.
    v, r, c, k = adjudicate_f2_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "reweighting_promising" and r is False and c is False and k is None

    # control_recovery_only: primary passes, absolute fails.
    v, r, c, k = adjudicate_f2_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, -1.0, 10.0],
        absolute_ci_lower=-0.5,
    )
    assert v == "control_recovery_only" and r is False and c is False and k is None

    # not_supported: primary seed mean negative.
    v, r, c, k = adjudicate_f2_verdict(
        primary_seed_means=[5.0, -2.0, 7.0],
        primary_ci_lower=-1.0,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None

    # not_supported: primary CI crosses zero.
    v, r, c, k = adjudicate_f2_verdict(
        primary_seed_means=[2.0, 1.0, 3.0],
        primary_ci_lower=-0.5,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None


def test_10_exact_seed_set_enforcement() -> None:
    """Test 10: All F2 runners reject any seed set other than the frozen three."""
    from training.mortal.f2_disagreement_weighting_contract_2026_09 import validate_f2_seed_set

    assert validate_f2_seed_set(None) == [20260910, 20260911, 20260912]
    assert validate_f2_seed_set([20260910, 20260911, 20260912]) == [20260910, 20260911, 20260912]

    for bad in (
        [20260910],
        [20260910, 20260911],
        [20260910, 20260911, 20260912, 20260913],
        [20260910, 20260912, 20260911],
        [20260911, 20260912, 20260913],
    ):
        with pytest.raises(ContractError):
            validate_f2_seed_set(bad)

    from training.mortal.eval_f2_disagreement_weighting_2026_09 import run_f2_evaluation
    from training.mortal.summary_f2_disagreement_weighting_2026_09 import adjudicate_f2_weighting
    from training.mortal.train_f2_disagreement_weighting_2026_09 import run_f2_training

    with pytest.raises(ContractError):
        run_f2_training(seeds=[20260910])
    with pytest.raises(ContractError):
        run_f2_evaluation(seeds=[20260910, 20260911])
    with pytest.raises(ContractError):
        adjudicate_f2_weighting(seeds=[20260913])


def test_11_checkpoint_evaluator_compatibility() -> None:
    """Test 11: F2 checkpoint keeps the standard four_player_native-loadable structure."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "mortal_weighted_variant_70400_seed_20260910.pth"
        save_state = {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "aux_net": aux_net.state_dict(),
            "steps": 70400,
            "condition": "disagreement_weighting_without_resampling",
            "config": {
                "control": {"version": 4, "online": False, "batch_size": 512},
                "resnet": {"conv_channels": 192, "num_blocks": 40},
            },
        }
        torch.save(save_state, ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        m = model.Brain(version=4, conv_channels=192, num_blocks=40)
        q = model.DQN(version=4)
        a = model.AuxNet((4,))
        m.load_state_dict(state["mortal"])
        q.load_state_dict(state["current_dqn"])
        a.load_state_dict(state["aux_net"])
        assert "score_aux_head" not in state
        assert state["condition"] == "disagreement_weighting_without_resampling"


def test_12_dataloader_interface_unchanged() -> None:
    """Test 12: F2 dataloader consumes the standard 6-field interface (mainline_dataloader untouched)."""
    m0_path, _ = resolve_m0_dataset_index()
    seed = 20260910
    dl = _build_dataloader(m0_path, seed=seed, batch_size=16)
    batch = next(iter(dl))
    assert len(batch) == 6
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    assert np.asarray(kyoku_rewards).dtype == np.float64
    dl2 = _build_dataloader(m0_path, seed=seed, batch_size=16)
    batch2 = next(iter(dl2))
    for a, b in zip(batch, batch2, strict=True):
        assert torch.equal(a, b)


def test_13_objective_default_path_regression() -> None:
    """Test 13: objective.py default path unchanged (guard against accidental edits)."""
    import inspect
    from training.mortal import objective as obj_mod
    source = inspect.getsource(obj_mod.compute_objective_losses)
    assert "preference_loss = q_out.logsumexp(-1).mean() - behavior_q.mean()" in source
    assert "value_loss = 0.5 * F.mse_loss(value_prediction, q_target_mc)" in source
    assert "row_weights" not in source  # weighting lives in F2 only

    # Legacy formula equivalence via the existing objective test fixture shape.
    torch.manual_seed(1234)
    raw_q = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    masks = torch.tensor([
        [True, True, False, True, False, False],
        [True, False, True, True, True, False],
        [False, True, True, False, True, True],
    ])
    q_out = raw_q.masked_fill(~masks, -torch.inf)
    actions = torch.tensor([0, 3, 5])
    targets = torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64)
    logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ranks = torch.tensor([0, 2, 1])
    result = obj_mod.compute_objective_losses(
        q_out=q_out, masks=masks, actions=actions, q_target_mc=targets,
        next_rank_logits=logits, player_ranks=ranks,
        mode="behavior_action_mc", cql_weight=5.0, aux_weight=0.2,
    )
    behavior_q = q_out[torch.arange(3), actions]
    legacy_value = 0.5 * torch.mean((behavior_q - targets) ** 2)
    legacy_cql = q_out.logsumexp(-1).mean() - behavior_q.mean()
    legacy_aux = torch.nn.functional.cross_entropy(logits, ranks)
    assert torch.allclose(result["value_loss"], legacy_value)
    assert torch.allclose(result["preference_loss"], legacy_cql)
    assert torch.allclose(result["next_rank_loss"], legacy_aux)
    assert torch.allclose(result["total_loss"], legacy_value + 5.0 * legacy_cql + 0.2 * legacy_aux)
