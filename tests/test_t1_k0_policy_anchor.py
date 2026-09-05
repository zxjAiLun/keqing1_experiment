from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.mortal.eval_t1_k0_policy_anchor_2026_09 import _lineup_for_seed, _verify_resume_prefix
from training.mortal.summary_t1_k0_policy_anchor_2026_09 import _scores_and_ranks_from_events
from training.mortal.t1_k0_policy_anchor_contract_2026_09 import (
    ANCHOR_DIRECTION,
    ANCHOR_LOSS,
    ANCHOR_TEMPERATURE,
    CALIBRATION_SCHEMA,
    EXPERIMENT_ID,
    FROZEN_R2_CONTROL,
    TARGET_GRADIENT_RATIO,
    TRAINING_SEEDS,
    ContractError,
    adjudicate_t1_verdict,
    crossed_bootstrap_ci,
    legal_centered_q,
    legal_policy_kl_rows,
    validate_t1_seed_set,
    verify_calibration,
    verify_recorded_training_evidence,
)
from training.mortal.train_t1_k0_policy_anchor_2026_09 import compute_t1_losses


def _valid_calibration() -> dict:
    return {
        "schema": CALIBRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol": {
            "anchor_loss": ANCHOR_LOSS,
            "direction": ANCHOR_DIRECTION,
            "temperature": ANCHOR_TEMPERATURE,
            "target_gradient_ratio": TARGET_GRADIENT_RATIO,
            "uses_reward_or_evaluation_signal": False,
        },
        "by_seed": {
            f"seed_{seed}": {
                "r2_control_sha256": FROZEN_R2_CONTROL[seed]["checkpoint_sha256"],
                "base_gradient_norm": 2.0 + idx,
                "anchor_gradient_norm": 0.5 + idx,
                "lambda_for_target_ratio": 0.4 + idx,
            }
            for idx, seed in enumerate(TRAINING_SEEDS)
        },
        "selected_lambda": 1.4,
        "hard_gates": {"all_pass": True},
        "verdict": "calibration_completed",
    }


def test_legal_policy_kl_is_zero_for_identical_q_and_shift_invariant() -> None:
    masks = torch.tensor([[True, True, False], [True, False, True]])
    parent = torch.tensor([[2.0, 0.0, -99.0], [1.0, -99.0, -1.0]])
    current = parent + torch.tensor([[10.0, 10.0, 1000.0], [-7.0, 500.0, -7.0]])
    rows = legal_policy_kl_rows(current, parent, masks)
    assert torch.allclose(rows, torch.zeros(2), atol=1e-6)


def test_legal_policy_kl_ignores_illegal_q_and_has_finite_gradient() -> None:
    masks = torch.tensor([[True, True, False]])
    parent = torch.tensor([[2.0, 0.0, -1.0]])
    current = torch.tensor([[0.0, 2.0, 1e30]], requires_grad=True)
    loss = legal_policy_kl_rows(current, parent, masks).mean()
    assert float(loss.detach()) > 0
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 2].item() == 0.0


def test_legal_centered_q_rejects_bad_masks_and_nonfinite_legal_values() -> None:
    with pytest.raises(ContractError, match="at least one legal"):
        legal_centered_q(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ContractError, match="finite"):
        legal_centered_q(torch.tensor([[float("nan"), 0.0]]), torch.tensor([[True, False]]))


def test_compute_t1_losses_adds_only_lambda_times_anchor() -> None:
    masks = torch.tensor([[True, True, False]])
    current = torch.tensor([[0.0, 1.0, -torch.inf]], requires_grad=True)
    parent = torch.tensor([[1.0, 0.0, -torch.inf]])
    base = {"total_loss": torch.tensor(3.0, requires_grad=True), "value_loss": torch.tensor(1.0)}
    out = compute_t1_losses(
        base_losses=base,
        q_current_anchor=current,
        q_parent=parent,
        masks=masks,
        anchor_lambda=0.25,
    )
    expected = base["total_loss"] + 0.25 * out["anchor_loss"]
    assert torch.allclose(out["total_loss_with_anchor"], expected)
    with pytest.raises(ContractError, match="positive"):
        compute_t1_losses(base_losses=base, q_current_anchor=current, q_parent=parent, masks=masks, anchor_lambda=0.0)


def test_verify_calibration_accepts_frozen_contract() -> None:
    assert verify_calibration(_valid_calibration()) == pytest.approx(1.4)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda obj: obj["protocol"].__setitem__("uses_reward_or_evaluation_signal", True), "must not use"),
        (lambda obj: obj.__setitem__("selected_lambda", 0.0), "Invalid selected lambda"),
        (lambda obj: obj["by_seed"].pop("seed_20260912"), "seed set"),
        (lambda obj: obj["by_seed"]["seed_20260910"].__setitem__("anchor_gradient_norm", 0.0), "invalid"),
    ],
)
def test_verify_calibration_fails_closed(mutator, message: str) -> None:
    obj = copy.deepcopy(_valid_calibration())
    mutator(obj)
    with pytest.raises(ContractError, match=message):
        verify_calibration(obj)


def test_t1_seed_set_is_exact() -> None:
    assert validate_t1_seed_set(None) == TRAINING_SEEDS
    assert validate_t1_seed_set(TRAINING_SEEDS) == TRAINING_SEEDS
    with pytest.raises(ContractError, match="exact seeds"):
        validate_t1_seed_set(TRAINING_SEEDS[:2])


def test_lineup_is_exact_and_uses_anchor_variant() -> None:
    assert _lineup_for_seed(20260910) == (
        "K0_70k",
        "ext_mortal",
        "R2_Control_seed_20260910",
        "T1_AnchorVariant_seed_20260910",
    )


def test_crossed_bootstrap_is_deterministic_and_shares_indices() -> None:
    matrix = np.arange(3000, dtype=np.float64).reshape(3, 1000)
    mean1, ci1, indices = crossed_bootstrap_ci(matrix, reps=50, seed=7, return_sampled_indices=True)
    mean2, ci2, _ = crossed_bootstrap_ci(matrix, reps=50, seed=999, shared_indices=indices)
    assert mean1 == pytest.approx(matrix.mean())
    assert mean2 == pytest.approx(matrix.mean())
    assert ci1 == pytest.approx(ci2)


@pytest.mark.parametrize(
    ("mechanism", "primary", "p_ci", "absolute", "a_ci", "expected"),
    [
        (True, [1, 2, 3], 0.1, [1, 1, 1], 0.1, "anchor_promising"),
        (True, [1, 2, 3], 0.1, [1, -1, 1], -0.1, "stability_only"),
        (True, [1, -1, 3], -0.1, [1, 1, 1], 0.1, "not_supported"),
        (False, [1, 2, 3], 0.1, [1, 1, 1], 0.1, "mechanism_not_supported"),
    ],
)
def test_adjudication_is_frozen(mechanism, primary, p_ci, absolute, a_ci, expected) -> None:
    verdict, recipe, checkpoint, k1 = adjudicate_t1_verdict(
        mechanism_pass=mechanism,
        primary_seed_means=primary,
        primary_ci_lower=p_ci,
        absolute_seed_means=absolute,
        absolute_ci_lower=a_ci,
    )
    assert verdict == expected
    assert recipe is False and checkpoint is False and k1 is None


def test_score_reconstruction_applies_reach_accepted() -> None:
    events = [
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    scores, ranks = _scores_and_ranks_from_events(events, "fixture")
    assert scores == [24000.0, 25000.0, 25000.0, 25000.0]
    assert ranks[0] == 3


def _resume_log(directory: Path, game_id: int, *, complete: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    events = [{"type": "start_game", "seed": [game_id, 8192], "names": list(_lineup_for_seed(20260910))}]
    if complete:
        events.append({"type": "end_game"})
    with gzip.open(directory / f"{game_id}_8192_a.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(event) for event in events))


def test_resume_preserves_exact_complete_batch_hashes(tmp_path: Path) -> None:
    for game_id in range(2800000, 2800050):
        _resume_log(tmp_path / "logs", game_id)
    hashes = _verify_resume_prefix(tmp_path, 20260910, 2800000, 250)
    assert len(hashes) == 50
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize("case", ["gap", "incomplete", "partial_batch", "wrong_lineup"])
def test_resume_rejects_unsafe_existing_logs(tmp_path: Path, case: str) -> None:
    count = 49 if case == "partial_batch" else 50
    for offset in range(count):
        game_id = 2800000 + offset + (1 if case == "gap" else 0)
        _resume_log(tmp_path / "logs", game_id, complete=not (case == "incomplete" and offset == 0))
    seed = 20260911 if case == "wrong_lineup" else 20260910
    expected = {"gap": "contiguous prefix", "incomplete": "Incomplete resume log", "partial_batch": "50-game batch", "wrong_lineup": "Lineup mismatch"}
    with pytest.raises(ContractError, match=expected[case]):
        _verify_resume_prefix(tmp_path, seed, 2800000, 250)


def _recorded_evidence() -> dict:
    metrics = {"mean_kl_to_k0": 0.2, "greedy_disagreement_rate_to_k0": 0.1, "centered_advantage_rmse_to_k0": 0.8}
    return {
        "training_config": {
            "training_seeds": [20260910, 20260911, 20260912], "steps_start": 70000, "steps_target": 70400,
            "optimizer_steps": 400, "batch_size": 512, "learning_rate": 1e-4, "weight_decay": 0.1,
            "cql_min_q_weight": 5.0, "aux_weight": 0.2, "gamma": 1.0, "device": "cuda",
        },
        "policy_anchor": {"lambda": 0.5},
        "row_identity": {"by_seed": {
            f"seed_{seed}": {"anchor_stats": {"per_step": [
                {"step": step, "base_total_loss": 2.0, "anchor_kl": 0.2, "weighted_anchor_loss": 0.1, "total_loss": 2.1}
                for step in range(1, 401)
            ]}} for seed in TRAINING_SEEDS
        }},
        "mechanism_audit": {
            "panel": "held_out_batches_401_to_416", "all_seed_directions_pass": True,
            "by_seed": {f"seed_{seed}": {
                "rows": 8192, "batches": 16, "skip_batches": 400, "row_sha256": "a" * 64,
                "control": dict(metrics), "variant": {key: value / 2 for key, value in metrics.items()},
                "directions": {"kl_lower": True, "greedy_disagreement_lower": True, "centered_advantage_rmse_lower": True},
                "all_directions_pass": True,
            } for seed in TRAINING_SEEDS},
        },
    }


def test_recorded_evidence_accepts_both_supported_and_honest_failed_mechanism() -> None:
    data = _recorded_evidence()
    verify_recorded_training_evidence(data)
    audit = data["mechanism_audit"]["by_seed"]["seed_20260910"]
    audit["variant"]["mean_kl_to_k0"] = audit["control"]["mean_kl_to_k0"]
    audit["directions"]["kl_lower"] = False
    audit["all_directions_pass"] = False
    data["mechanism_audit"]["all_seed_directions_pass"] = False
    verify_recorded_training_evidence(data)


@pytest.mark.parametrize("case, message", [
    ("config", "configuration"), ("step", "step sequence"), ("nan_loss", "Nonfinite"),
    ("weighted", "arithmetic"), ("total", "arithmetic"), ("held_out", "held-out range"),
    ("metric", "directions mismatch"), ("nan_metric", "mechanism metric"), ("aggregate", "aggregate"),
])
def test_recorded_evidence_rejects_false_positive_gates(case: str, message: str) -> None:
    data = _recorded_evidence()
    step = data["row_identity"]["by_seed"]["seed_20260910"]["anchor_stats"]["per_step"][0]
    audit = data["mechanism_audit"]["by_seed"]["seed_20260910"]
    if case == "config":
        data["training_config"]["learning_rate"] = 0.01
    elif case == "step":
        step["step"] = 2
    elif case == "nan_loss":
        step["base_total_loss"] = float("nan")
    elif case == "weighted":
        step["weighted_anchor_loss"] = 0.2
    elif case == "total":
        step["total_loss"] = 3.0
    elif case == "held_out":
        audit["skip_batches"] = 399
    elif case == "metric":
        audit["variant"]["mean_kl_to_k0"] = 0.3
    elif case == "nan_metric":
        audit["variant"]["mean_kl_to_k0"] = float("nan")
    else:
        data["mechanism_audit"]["all_seed_directions_pass"] = False
    with pytest.raises(ContractError, match=message):
        verify_recorded_training_evidence(data)
