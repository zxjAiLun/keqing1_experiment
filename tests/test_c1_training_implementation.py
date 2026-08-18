from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from training.mortal import prepare_c1_training_2026_08 as prepare
from training.mortal import run_c1_training_2026_08 as launcher
from training.mortal.objective import compute_objective_losses
from training.mortal.preflight_c1_training_2026_08 import (
    assert_no_training_outputs,
    assert_run_matrix,
)


def _source_config() -> dict:
    return {
        "control": {
            "state_file": "old/mortal.pth",
            "best_state_file": "old/mortal_best.pth",
            "tensorboard_dir": "old/tb",
            "batch_size": 512,
            "enable_amp": False,
        },
        "dataset": {"num_workers": 0},
        "cql": {"min_q_weight": 5.0},
        "experiment": {
            "route": "M0_control",
            "trainable_label": "ext_mortal",
            "training_seed": 20260806,
            "parent_steps": 70000,
            "reward_mode": "final_rank_mc",
        },
        "objective": {"mode": "behavior_action_mc"},
        "reward": {"mode": "final_rank_mc"},
    }


def _runs() -> list[dict[str, int | str]]:
    return [
        {"route": route, "seed": seed}
        for route in prepare.ROUTES
        for seed in prepare.SEEDS
    ]


def _gradient_inputs() -> tuple[torch.Tensor, ...]:
    q_out = torch.tensor(
        [[0.4, -0.2, float("-inf")], [0.1, 0.7, -0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    masks = torch.tensor([[True, True, False], [True, True, True]])
    actions = torch.tensor([0, 1])
    q_target = torch.tensor([0.2, 0.5])
    next_rank_logits = torch.tensor(
        [[0.2, -0.1, 0.4, 0.0], [0.4, 0.2, -0.3, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    player_ranks = torch.tensor([0, 2])
    return q_out, masks, actions, q_target, next_rank_logits, player_ranks


def test_exact_six_run_matrix_and_wrong_shapes_fail_closed() -> None:
    assert_run_matrix(_runs())
    with pytest.raises(prepare.ContractError):
        assert_run_matrix(_runs()[:-1])
    with pytest.raises(prepare.ContractError):
        assert_run_matrix(_runs() + [{"route": "M0_CQL_OFF", "seed": 999}])
    with pytest.raises(prepare.ContractError):
        assert_run_matrix(_runs() + [{"route": "EXTRA", "seed": 20260806}])


def test_semantic_diff_allows_only_cql_output_and_provenance() -> None:
    source = _source_config()
    generated = prepare.build_c1_config(
        source,
        route="M0_CQL_OFF",
        seed=20260806,
        run_dir=Path("/tmp/c1-test-run"),
        source_sha256="a" * 64,
    )
    gate = prepare.validate_generated_config(
        source,
        generated,
        route="M0_CQL_OFF",
        seed=20260806,
        run_dir=Path("/tmp/c1-test-run"),
        source_sha256="a" * 64,
    )
    assert gate["passed"] is True
    assert gate["unexpected_differences"] == []

    invalid = copy.deepcopy(generated)
    invalid["objective"]["mode"] = "legal_mean_mc"
    with pytest.raises(prepare.ContractError, match="semantic diff gate"):
        prepare.validate_generated_config(
            source,
            invalid,
            route="M0_CQL_OFF",
            seed=20260806,
            run_dir=Path("/tmp/c1-test-run"),
            source_sha256="a" * 64,
        )


def test_generated_cql_weight_must_be_zero() -> None:
    source = _source_config()
    generated = prepare.build_c1_config(
        source,
        route="M0_CQL_OFF",
        seed=20260806,
        run_dir=Path("/tmp/c1-test-run"),
        source_sha256="a" * 64,
    )
    generated["cql"]["min_q_weight"] = 1.0
    with pytest.raises(prepare.ContractError):
        prepare.validate_generated_config(
            source,
            generated,
            route="M0_CQL_OFF",
            seed=20260806,
            run_dir=Path("/tmp/c1-test-run"),
            source_sha256="a" * 64,
        )


def test_source_config_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[cql]\nmin_q_weight = 5.0\n", encoding="utf-8")
    with pytest.raises(prepare.ContractError):
        if prepare.sha256_file(path) != "0" * 64:
            raise prepare.ContractError("source config SHA mismatch")


def test_governance_status_and_loader_stream_mismatch_fail_closed() -> None:
    registry = json.loads(
        (prepare.REPO_ROOT / "training/docs/mortal/research_registry.json").read_text(encoding="utf-8")
    )
    loader = json.loads(
        (
            prepare.REPO_ROOT
            / "artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/loader_compatibility.json"
        ).read_text(encoding="utf-8")
    )
    invalid_registry = copy.deepcopy(registry)
    invalid_registry["records"][-1]["status"] = "not_authorized"
    with pytest.raises(prepare.ContractError):
        prepare.validate_governance_payload(invalid_registry, loader)

    invalid_loader = copy.deepcopy(loader)
    invalid_loader["runs"][0]["stream"]["current_stream_sha256"] = "0" * 64
    with pytest.raises(prepare.ContractError):
        prepare.validate_governance_payload(registry, invalid_loader)


def test_k0_parent_mismatch_and_output_checkpoint_fail_closed(tmp_path: Path) -> None:
    bad_parent = tmp_path / "bad_parent.pth"
    bad_parent.write_bytes(b"not a checkpoint")
    with pytest.raises(prepare.ContractError):
        prepare.inspect_parent(bad_parent)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "mortal.pth").write_bytes(b"checkpoint")
    with pytest.raises(prepare.ContractError):
        assert_no_training_outputs(run_dir)


def test_cql_off_gradient_removes_only_cql_contribution() -> None:
    current_inputs = _gradient_inputs()
    off_inputs = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in current_inputs)
    current = compute_objective_losses(
        q_out=current_inputs[0],
        masks=current_inputs[1],
        actions=current_inputs[2],
        q_target_mc=current_inputs[3],
        next_rank_logits=current_inputs[4],
        player_ranks=current_inputs[5],
        mode="behavior_action_mc",
        cql_weight=5.0,
        aux_weight=0.2,
    )
    off = compute_objective_losses(
        q_out=off_inputs[0],
        masks=off_inputs[1],
        actions=off_inputs[2],
        q_target_mc=off_inputs[3],
        next_rank_logits=off_inputs[4],
        player_ranks=off_inputs[5],
        mode="behavior_action_mc",
        cql_weight=0.0,
        aux_weight=0.2,
    )
    assert torch.equal(current["cql_loss"].detach(), off["cql_loss"].detach())
    assert torch.equal(current["value_loss"].detach(), off["value_loss"].detach())
    assert torch.equal(current["next_rank_loss"].detach(), off["next_rank_loss"].detach())
    expected_off = off["value_loss"] + 0.2 * off["next_rank_loss"]
    grad_off = torch.autograd.grad(off["total_loss"], off_inputs[0], retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected_off, off_inputs[0])[0]
    assert torch.equal(grad_off, grad_expected)
    assert torch.equal(
        off["total_loss"].detach(),
        (off["value_loss"] + 0.2 * off["next_rank_loss"]).detach(),
    )


def test_launcher_execute_fails_before_subprocess_and_has_no_scientific_overrides(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not be called while unauthorized")

    monkeypatch.setattr(launcher, "subprocess", type("ForbiddenSubprocess", (), {"run": forbidden}), raising=False)
    with pytest.raises(SystemExit, match="not authorized"):
        launcher.main(["--route", "M0_CQL_OFF", "--seed", "20260806", "--execute"])
    assert called is False

    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(
            ["--route", "M0_CQL_OFF", "--seed", "20260806", "--config", "override.toml"]
        )
