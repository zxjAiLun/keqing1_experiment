from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import toml
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


def test_runtime_index_preserves_order_and_payload_identity(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_data"
    source_dir.mkdir()
    source_paths = []
    for index in range(6000):
        path = source_dir / f"sample_{index:04d}.json.gz"
        path.touch()
        source_paths.append(str(path.resolve()))
    source_index = tmp_path / "source_index.pth"
    runtime_index = tmp_path / "runtime_inputs" / "M0_file_index_linux.pth"
    torch.save({"file_list": source_paths}, source_index)

    record = prepare.build_runtime_file_index(source_path=source_index, runtime_path=runtime_index)
    validated = prepare.validate_runtime_file_index(
        record,
        expected_source_path=source_index,
        expected_source_sha256=prepare.sha256_file(source_index),
    )
    assert validated == record
    assert record["file_count"] == 6000
    assert record["ordered_path_mapping_sha256"]
    assert record["source_payload_without_file_list_sha256"] == record["runtime_payload_without_file_list_sha256"]
    runtime_payload, runtime_paths = prepare.load_file_index(runtime_index)
    assert runtime_payload.keys() == {"file_list"}
    assert runtime_paths == source_paths


def test_git_scope_allows_absent_or_tolerated_1md_only() -> None:
    clean = {"branch": "main", "tracked_changes": [], "untracked": []}
    with_1md = {"branch": "main", "tracked_changes": [], "untracked": ["1.md"]}
    prepare.validate_git_scope(clean)
    prepare.validate_git_scope(with_1md)
    with pytest.raises(prepare.ContractError):
        prepare.validate_git_scope({"branch": "main", "tracked_changes": [], "untracked": ["other.txt"]})


def _authorized_launcher_fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    route = "M0_CQL_OFF"
    seed = 20260806
    implementation_commit = "a" * 40
    parent_path = tmp_path / "K0_parent.pth"
    parent_path.write_bytes(b"K0 parent fixture")
    parent_digest = {"steps": 70000, "fixture": True}
    parent_record = {
        "path": str(parent_path.resolve()),
        "sha256": prepare.K0_PARENT_SHA256,
        "digest": parent_digest,
        "optimizer_moments_covered": True,
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_paths = []
    for index in range(6000):
        path = data_dir / f"sample_{index:04d}.json.gz"
        path.touch()
        source_paths.append(str(path.resolve()))
    source_index = tmp_path / "source_index.pth"
    runtime_index = tmp_path / "runtime_inputs" / "M0_file_index_linux.pth"
    torch.save({"file_list": source_paths}, source_index)
    label_path = tmp_path / "player_names.txt"
    label_path.write_text("fixture-player\n", encoding="utf-8")
    source_config_path = tmp_path / "source_config.toml"
    source = {
        "control": {
            "state_file": str((tmp_path / "historical_mortal.pth").resolve()),
            "best_state_file": str((tmp_path / "historical_best.pth").resolve()),
            "tensorboard_dir": str((tmp_path / "historical_tb").resolve()),
            "batch_size": 512,
            "enable_amp": False,
        },
        "dataset": {
            "globs": [str((data_dir / "*.json.gz").resolve())],
            "file_index": str(source_index.resolve()),
            "file_batch_size": 512,
            "reserve_ratio": 0.0,
            "num_workers": 0,
            "player_names_files": [str(label_path.resolve())],
            "num_epochs": 1,
            "enable_augmentation": False,
            "augmented_first": False,
        },
        "cql": {"min_q_weight": 0.0},
        "experiment": {
            "route": "M0_control",
            "trainable_label": "ext_mortal",
            "training_seed": seed,
            "parent_steps": 70000,
            "reward_mode": "final_rank_mc",
        },
        "objective": {"mode": "behavior_action_mc"},
        "reward": {"mode": "final_rank_mc"},
    }
    source["cql"]["min_q_weight"] = 5.0
    source_config_path.write_text(toml.dumps(source), encoding="utf-8")
    source_sha256 = prepare.sha256_file(source_config_path)
    runtime_record = prepare.build_runtime_file_index(source_path=source_index, runtime_path=runtime_index)
    runtime_dataset = prepare.map_dataset_paths(source)
    runtime_dataset["file_index"] = runtime_record["runtime_file_index_path"]
    run_dir = tmp_path / "run"
    config = prepare.build_c1_config(
        source,
        route=route,
        seed=seed,
        run_dir=run_dir,
        source_sha256=source_sha256,
        runtime_dataset=runtime_dataset,
    )
    config_path = tmp_path / "formal_config.toml"
    config_path.write_text(toml.dumps(config), encoding="utf-8")
    config_sha256 = prepare.sha256_file(config_path)
    semantic = prepare.validate_generated_config(
        source,
        config,
        route=route,
        seed=seed,
        run_dir=run_dir,
        source_sha256=source_sha256,
        runtime_dataset=runtime_dataset,
    )
    label_binding = prepare.build_label_binding(source)
    runtime_provenance = prepare.runtime_provenance()
    mortal_provenance = {
        "repo": str((prepare.REPO_ROOT / "third_party/Mortal").resolve()),
        "current_mortal_revision": "fixture-current-mortal",
        "historical_mortal_revision": prepare.HISTORICAL_MORTAL_REVISION,
        "content_matches_historical": True,
        "sources": [
            {
                "path": "mortal/model.py",
                "current_sha256": "model-fixture",
                "current_git_blob_oid": "model-blob",
                "historical_sha256": "model-fixture",
                "historical_git_blob_oid": "model-blob",
                "content_matches_historical": True,
            },
            {
                "path": "mortal/lr_scheduler.py",
                "current_sha256": "lr-fixture",
                "current_git_blob_oid": "lr-blob",
                "historical_sha256": "lr-fixture",
                "historical_git_blob_oid": "lr-blob",
                "content_matches_historical": True,
            },
            {
                "path": "mortal/config.py",
                "current_sha256": "config-fixture",
                "current_git_blob_oid": "config-blob",
                "historical_sha256": "config-fixture",
                "historical_git_blob_oid": "config-blob",
                "content_matches_historical": True,
            },
        ],
    }
    historical_records = [
        {
            "route": candidate_route,
            "seed": candidate_seed,
            "path": str((tmp_path / f"historical_{candidate_route}_{candidate_seed}.pth").resolve()),
            "sha256": f"historical-{candidate_route}-{candidate_seed}",
            "steps": 72000,
            "mortal_revision": prepare.HISTORICAL_MORTAL_REVISION,
        }
        for candidate_route in prepare.ROUTES
        for candidate_seed in prepare.SEEDS
    ]
    command_argv = prepare.future_training_argv(
        config_path=config_path,
        parent_path=parent_path,
        seed=seed,
        run_dir=run_dir,
        executable=runtime_provenance["sys_executable"],
    )
    selected = {
        "route": route,
        "seed": seed,
        "source_current_config": str(source_config_path.resolve()),
        "source_current_config_sha256": source_sha256,
        "cql_off_config": str(config_path.resolve()),
        "cql_off_config_sha256": config_sha256,
        "formal_training_config_sha256": config_sha256,
        "smoke_config_sha256": config_sha256,
        "exact_same_config": True,
        "semantic_diff": semantic,
        "source_file_index_path": runtime_record["source_file_index_path"],
        "source_file_index_sha256": runtime_record["source_file_index_sha256"],
        "runtime_file_index_path": runtime_record["runtime_file_index_path"],
        "runtime_file_index_sha256": runtime_record["runtime_file_index_sha256"],
        "file_count": runtime_record["file_count"],
        "ordered_path_mapping_sha256": runtime_record["ordered_path_mapping_sha256"],
        "source_payload_without_file_list_sha256": runtime_record[
            "source_payload_without_file_list_sha256"
        ],
        "runtime_payload_without_file_list_sha256": runtime_record[
            "runtime_payload_without_file_list_sha256"
        ],
        "file_index": runtime_record["source_file_index_path"],
        "file_index_sha256": runtime_record["source_file_index_sha256"],
        "label_files": label_binding["player_names_files"],
        "label_binding": label_binding,
        "historical_mortal_revision": prepare.HISTORICAL_MORTAL_REVISION,
        "historical_checkpoint_sha256": historical_records[0]["sha256"],
        "runtime_dataset": runtime_dataset,
        "run_output_dir": str(run_dir.resolve()),
        "future_training_argv": command_argv,
        "future_training_command": " ".join(command_argv),
    }
    runs = [
        {"route": candidate_route, "seed": candidate_seed}
        for candidate_route in prepare.ROUTES
        for candidate_seed in prepare.SEEDS
    ]
    runs[0] = selected
    sources = launcher.current_implementation_sources()
    manifest = {
        "experiment_id": prepare.C1_ID,
        "status": "prepared_not_authorized",
        "training_authorized": False,
        "evaluation_authorized": False,
        "new_training_runs": 6,
        "implementation_commit": implementation_commit,
        "implementation_sources": sources,
        "parent": parent_record,
        "runtime_provenance": runtime_provenance,
        "mortal_provenance": mortal_provenance,
        "historical_mortal_checkpoints": historical_records,
        "runtime_inputs": {
            "M0": dict(runtime_record, route="M0"),
            "D1": dict(runtime_record, route="D1"),
        },
        "training_command_policy": {
            "authoritative_field": "future_training_argv",
            "frozen_executable": runtime_provenance["sys_executable"],
            "shell": False,
        },
        "runs": runs,
    }
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = prepare.sha256_file(manifest_path)
    preflight = {
        "passed": True,
        "manifest_sha256": manifest_sha256,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "git": {"commit": implementation_commit},
        "checks": {"formal_config_equals_smoke_config": True},
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    preflight_sha256 = prepare.sha256_file(preflight_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_inspect(path: Path) -> dict[str, object]:
        if path.read_bytes() != b"K0 parent fixture":
            raise launcher.AuthorizationError("tampered parent fixture")
        return {"sha256": prepare.K0_PARENT_SHA256, "digest": parent_digest}

    class CaptureSubprocess:
        @staticmethod
        def run(argv, **kwargs):
            calls.append((list(argv), kwargs))

    monkeypatch.setattr(launcher, "DEFAULT_MANIFEST", manifest_path)
    monkeypatch.setattr(launcher, "DEFAULT_PREFLIGHT", preflight_path)
    monkeypatch.setattr(launcher, "TRAINING_AUTHORIZED", True)
    monkeypatch.setattr(launcher, "APPROVED_IMPLEMENTATION_COMMIT", implementation_commit)
    monkeypatch.setattr(launcher, "AUTHORIZED_PREFLIGHT_SHA256", preflight_sha256)
    monkeypatch.setattr(launcher, "AUTHORIZED_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(launcher, "inspect_parent", fake_inspect)
    monkeypatch.setattr(launcher, "SOURCE_CONFIG_SHA256", {(route, seed): source_sha256})
    monkeypatch.setattr(
        launcher,
        "validate_source_inputs",
        lambda config, *, route, seed: {
            "file_index_path": str(source_index.resolve()),
            "file_index_sha256": prepare.sha256_file(source_index),
            "file_count": 6000,
            "label_files": label_binding["player_names_files"],
            "label_binding": label_binding,
            "mapped_file_count": 6000,
        },
    )
    monkeypatch.setattr(launcher, "mortal_source_provenance", lambda: copy.deepcopy(mortal_provenance))
    monkeypatch.setattr(
        launcher,
        "validate_historical_mortal_checkpoint",
        lambda expected, source_config_path_value, *, route, seed: copy.deepcopy(expected),
    )
    monkeypatch.setattr(launcher, "subprocess", CaptureSubprocess)
    return {
        "route": route,
        "seed": seed,
        "token": launcher.confirmation_token(
            route=route,
            seed=seed,
            implementation_commit=implementation_commit,
            preflight_sha256=preflight_sha256,
        ),
        "calls": calls,
        "argv": command_argv,
        "manifest_path": manifest_path,
        "preflight_path": preflight_path,
        "config_path": config_path,
        "source_index": source_index,
        "runtime_index": runtime_index,
        "label_path": label_path,
        "parent_path": parent_path,
        "sources": sources,
        "mortal_provenance": mortal_provenance,
        "runtime_provenance": runtime_provenance,
    }


def test_launcher_simulated_authorization_calls_exact_argv(monkeypatch, tmp_path: Path) -> None:
    fixture = _authorized_launcher_fixture(tmp_path, monkeypatch)
    result = launcher.main(
        [
            "--route",
            fixture["route"],
            "--seed",
            str(fixture["seed"]),
            "--execute",
            "--confirmation-token",
            fixture["token"],
        ]
    )
    assert result == 0
    calls = fixture["calls"]
    assert len(calls) == 1
    actual_argv, kwargs = calls[0]
    assert actual_argv == fixture["argv"]
    assert kwargs["cwd"] == launcher.REPO_ROOT
    assert kwargs["check"] is True
    assert kwargs["shell"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "preflight",
        "manifest",
        "config",
        "parent",
        "source",
        "token",
        "runtime_index",
        "source_index",
        "label",
        "mortal_model",
        "mortal_lr_scheduler",
        "native",
        "python",
    ],
)
def test_launcher_tamper_guards_never_call_subprocess(monkeypatch, tmp_path: Path, tamper: str) -> None:
    fixture = _authorized_launcher_fixture(tmp_path, monkeypatch)
    if tamper == "preflight":
        path = fixture["preflight_path"]
        path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    elif tamper == "manifest":
        path = fixture["manifest_path"]
        path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    elif tamper == "config":
        path = fixture["config_path"]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "parent":
        fixture["parent_path"].write_bytes(b"tampered parent")
    elif tamper == "source":
        monkeypatch.setattr(launcher, "current_implementation_sources", lambda: fixture["sources"] + [{"tampered": "1"}])
    elif tamper == "runtime_index":
        payload, _ = prepare.load_file_index(fixture["runtime_index"])
        payload["file_list"][0] = str(tmp_path / "tampered-runtime-file")
        torch.save(payload, fixture["runtime_index"])
    elif tamper == "source_index":
        payload, _ = prepare.load_file_index(fixture["source_index"])
        payload["file_list"][0] = str(tmp_path / "tampered-source-file")
        torch.save(payload, fixture["source_index"])
    elif tamper == "label":
        fixture["label_path"].write_text("tampered-player\n", encoding="utf-8")
    elif tamper in {"mortal_model", "mortal_lr_scheduler"}:
        tampered_mortal = copy.deepcopy(fixture["mortal_provenance"])
        target_path = "mortal/model.py" if tamper == "mortal_model" else "mortal/lr_scheduler.py"
        for record in tampered_mortal["sources"]:
            if record["path"] == target_path:
                record["current_sha256"] = "tampered-content"
        monkeypatch.setattr(launcher, "mortal_source_provenance", lambda: tampered_mortal)
    elif tamper in {"native", "python"}:
        def runtime_tamper(_expected, *, tamper=tamper):
            raise launcher.AuthorizationError(f"tampered {tamper} runtime provenance")

        monkeypatch.setattr(launcher, "validate_runtime_provenance", runtime_tamper)
    else:
        fixture["token"] = "wrong-token"
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--route",
                fixture["route"],
                "--seed",
                str(fixture["seed"]),
                "--execute",
                "--confirmation-token",
                fixture["token"],
            ]
        )
    assert fixture["calls"] == []


def test_launcher_rejects_non_frozen_route_and_seed_before_subprocess(monkeypatch, tmp_path: Path) -> None:
    fixture = _authorized_launcher_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--route",
                "BAD_ROUTE",
                "--seed",
                str(fixture["seed"]),
                "--execute",
                "--confirmation-token",
                fixture["token"],
            ]
        )
    assert fixture["calls"] == []
