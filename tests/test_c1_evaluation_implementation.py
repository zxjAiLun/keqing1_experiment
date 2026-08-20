from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.mortal import c1_evaluation_contract_2026_08 as contract
from training.mortal import run_c1_evaluation_2026_08 as launcher
from training.mortal import summarize_c1_interaction_2026_08 as summary


def _model_records() -> dict[str, dict[str, str]]:
    labels = ["70k", "ext_mortal"]
    labels += [f"{route}_{condition}_{seed}" for condition in ("CURRENT", "CQL_OFF") for route in ("M0", "D1") for seed in contract.TRAINING_SEEDS]
    return {label: {"path": f"/tmp/c1/{label}.pth", "sha256": "synthetic"} for label in labels}


def test_exact_24_shard_matrix_starts_model_order_and_argv() -> None:
    models = _model_records()
    runs = contract.build_run_matrix(models, executable="/frozen/.venv/bin/python3")
    assert len(runs) == 24
    assert {(row["condition"], row["training_seed"], row["shard"]) for row in runs} == {
        (condition, seed, shard)
        for condition in contract.CONDITIONS
        for seed in contract.TRAINING_SEEDS
        for shard in contract.SHARDS
    }
    for row in runs:
        seed = row["training_seed"]
        shard = row["shard"]
        assert row["hanchan_seed_start"] == contract.SHARD_STARTS[seed][shard]
        assert row["hanchan_seed_end_exclusive"] == contract.SHARD_STARTS[seed][shard] + 250
        assert tuple(row["model_order"]) == contract.model_order(row["condition"], seed)
        assert row["games"] == row["native_batch_games"] == 250
        assert row["seed_key"] == 8192
        assert row["seat_mode"] == "random"
        assert row["device"] == "cuda"
        assert row["require_cuda"] is True
        assert row["amp"] is False
        assert row["resume"] is False
        assert row["future_argv"][0] == "/frozen/.venv/bin/python3"
        assert row["future_argv"][1] == "training/mortal/four_player_native.py"
        assert "--enable-amp" not in row["future_argv"]
        assert "--resume" not in row["future_argv"]
        assert row["future_argv"][-2:] == ["--rank-points-profile", "tenhou_reference"]


def test_unauthorized_execute_reaches_no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[object, ...]] = []

    def forbidden(*args: object, **kwargs: object) -> SimpleNamespace:
        called.append(args)
        raise AssertionError("arena subprocess must not be reached")

    monkeypatch.setattr(launcher, "EVALUATION_AUTHORIZED", False)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(SystemExit, match="not authorized"):
        launcher.main(["--condition", "CURRENT", "--seed", "20260806", "--shard", "0", "--execute"])
    assert called == []


class _FakeStat:
    @classmethod
    def from_log(cls, raw_text: str, player_id: int) -> _FakeStat:
        events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        ranks = summary.ranks_from_events(events)
        target = ranks[str(events[0]["names"][player_id])]
        value = cls()
        for rank in (1, 2, 3, 4):
            setattr(value, f"rank_{rank}", int(rank == target))
        return value


def _write_synthetic_log(path: Path, names: list[str] | None = None) -> None:
    names = names or ["D1_CURRENT_20260806", "70k", "ext_mortal", "M0_CURRENT_20260806"]
    events = [
        {"type": "start_game", "names": names, "seed": [1700000, 8192]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 0},
        {"type": "hora", "actor": 1, "target": 0, "deltas": [-15000, 15000, 0, 0]},
        {"type": "end_kyoku"},
    ]
    payload = "\n".join(json.dumps(event) for event in events).encode("utf-8")
    path.write_bytes(gzip.compress(payload))


def test_complete_hanchan_parser_reachaccepted_and_native_rank_equivalence(tmp_path: Path) -> None:
    path = tmp_path / "1700000_8192_a.json.gz"
    _write_synthetic_log(path)
    row = summary.parse_raw_log(
        path,
        condition="CURRENT",
        training_seed=20260806,
        expected_seed_start=1700000,
        expected_seed_end=1700001,
        stat_cls=_FakeStat,
    )
    # D1 paid the 15000 delta and the 1000 reach stick: it is rank 4.
    assert row["ranks_by_role"]["D1"] == 4
    assert row["ranks_by_role"]["70k"] == 1
    assert row["role_to_seat"] == {"D1": 0, "70k": 1, "ext_mortal": 2, "M0": 3}
    assert row["seed_key"] == 8192
    assert row["gap_pt_d1_minus_m0"] == -135.0
    assert row["source_log_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("condition", "names"),
    [
        ("CURRENT", ["D1_CQL_OFF_20260806", "70k", "ext_mortal", "M0_CQL_OFF_20260806"]),
        ("CQL_OFF", ["D1_CURRENT_20260806", "70k", "ext_mortal", "M0_CURRENT_20260806"]),
        ("CURRENT", ["D1_CURRENT_20260807", "70k", "ext_mortal", "M0_CURRENT_20260807"]),
    ],
)
def test_raw_log_rejects_cross_condition_or_wrong_seed_identity(
    tmp_path: Path, condition: str, names: list[str]
) -> None:
    path = tmp_path / "1700000_8192_identity.json.gz"
    _write_synthetic_log(path, names)
    with pytest.raises(contract.ContractError, match="model identity"):
        summary.parse_raw_log(
            path,
            condition=condition,
            training_seed=20260806,
            expected_seed_start=1700000,
            expected_seed_end=1700001,
            stat_cls=_FakeStat,
        )


def test_raw_log_accepts_exact_cql_off_identity_and_preserves_seat_mapping(tmp_path: Path) -> None:
    path = tmp_path / "1700000_8192_off.json.gz"
    names = ["D1_CQL_OFF_20260806", "70k", "ext_mortal", "M0_CQL_OFF_20260806"]
    _write_synthetic_log(path, names)
    row = summary.parse_raw_log(
        path,
        condition="CQL_OFF",
        training_seed=20260806,
        expected_seed_start=1700000,
        expected_seed_end=1700001,
        stat_cls=_FakeStat,
    )
    assert row["seat_order"] == names
    assert row["role_to_seat"] == {"D1": 0, "70k": 1, "ext_mortal": 2, "M0": 3}


def _valid_row(seed: int, hanchan_seed: int, condition: str, *, role_to_seat: dict[str, int] | None = None) -> dict[str, object]:
    ranks = {"70k": 1, "ext_mortal": 2, "M0": 3, "D1": 4}
    points = summary.points_for_ranks(ranks)
    return {
        "condition": condition,
        "training_seed": seed,
        "hanchan_seed": hanchan_seed,
        "seed_key": 8192,
        "role_order": list(summary.ROLE_ORDER),
        "role_to_seat": role_to_seat or {"70k": 0, "ext_mortal": 1, "M0": 2, "D1": 3},
        "ranks_by_role": ranks,
        "pts": points,
        "gap_pt_d1_minus_m0": points["D1"] - points["M0"],
    }


def test_pairing_duplicate_missing_role_and_rank_point_gates() -> None:
    current = [_valid_row(20260806, 1700000, "CURRENT")]
    off = [_valid_row(20260806, 1700000, "CQL_OFF")]
    assert summary.pair_current_off(current, off)[0]["interaction_row"] == 0.0
    with pytest.raises(contract.ContractError, match="duplicate"):
        summary.pair_current_off(current + current, off)
    with pytest.raises(contract.ContractError, match="identities"):
        summary.pair_current_off(current, [_valid_row(20260806, 1700001, "CQL_OFF")])
    mismatch = _valid_row(20260806, 1700000, "CQL_OFF", role_to_seat={"70k": 1, "ext_mortal": 0, "M0": 2, "D1": 3})
    with pytest.raises(contract.ContractError, match="role_to_seat"):
        summary.pair_current_off(current, [mismatch])
    bad_points = _valid_row(20260806, 1700000, "CURRENT")
    bad_points["pts"] = dict(bad_points["pts"]) | {"D1": 90.0}
    with pytest.raises(contract.ContractError, match="rank-point"):
        summary.validate_row_integrity(bad_points)


def test_pairing_preserves_both_source_paths_and_hashes() -> None:
    current = _valid_row(20260806, 1700000, "CURRENT")
    off = _valid_row(20260806, 1700000, "CQL_OFF")
    current.update({"source_log": "/tmp/current.json.gz", "source_log_sha256": "current-sha"})
    off.update({"source_log": "/tmp/off.json.gz", "source_log_sha256": "off-sha"})
    paired = summary.pair_current_off([current], [off])[0]
    assert paired["current_source_log"] == "/tmp/current.json.gz"
    assert paired["current_source_log_sha256"] == "current-sha"
    assert paired["off_source_log"] == "/tmp/off.json.gz"
    assert paired["off_source_log_sha256"] == "off-sha"


def test_duplicate_and_missing_seed_set_is_fail_closed() -> None:
    rows = [{"hanchan_seed": seed} for seed in range(1700000, 1701000)]
    summary.validate_hanchan_seed_set(rows, start=1700000)
    with pytest.raises(contract.ContractError, match="duplicate"):
        summary.validate_hanchan_seed_set(rows[:-1] + [{"hanchan_seed": 1700000}], start=1700000)
    with pytest.raises(contract.ContractError, match="range mismatch"):
        summary.validate_hanchan_seed_set(rows[:-1] + [{"hanchan_seed": 1701001}], start=1700000)


def _paired_rows(seed: int, interaction: float) -> list[dict[str, object]]:
    return [
        {
            "training_seed": seed,
            "hanchan_seed": contract.SHARD_STARTS[seed][0] + index,
            "interaction_row": interaction,
        }
        for index in range(1000)
    ]


def test_zero_and_known_positive_interaction_adjudication() -> None:
    positive = summary.summarize_interaction_rows(
        {seed: _paired_rows(seed, 1.0) for seed in contract.TRAINING_SEEDS},
        gates={"training": True, "provenance": True, "runtime": True, "pairing": True},
    )
    assert positive["primary_interaction_mean"] == 1.0
    assert positive["adjudication"]["verdict"] == "interaction_supported"
    zero = summary.summarize_interaction_rows(
        {seed: _paired_rows(seed, 0.0) for seed in contract.TRAINING_SEEDS},
        gates={"training": True, "provenance": True, "runtime": True, "pairing": True},
    )
    assert zero["primary_interaction_mean"] == 0.0
    assert zero["adjudication"]["verdict"] == "interaction_not_confirmed"


def test_direction_ci_and_provenance_gates_are_separate() -> None:
    one_seed_nonpositive = summary.machine_adjudication(
        {20260806: 1.0, 20260807: 0.0, 20260808: 1.0},
        [0.5, 1.5],
        {"training": True, "provenance": True},
    )
    assert one_seed_nonpositive["verdict"] == "interaction_not_confirmed"
    ci_not_positive = summary.machine_adjudication(
        {seed: 1.0 for seed in contract.TRAINING_SEEDS},
        [-0.1, 1.5],
        {"training": True, "provenance": True},
    )
    assert ci_not_positive["verdict"] == "interaction_not_confirmed"
    gate_failed = summary.machine_adjudication(
        {seed: 1.0 for seed in contract.TRAINING_SEEDS},
        [0.5, 1.5],
        {"training": True, "provenance": False},
    )
    assert gate_failed["verdict"] == "no_verdict_gates_failed"


def test_hierarchical_bootstrap_is_exact_and_deterministic() -> None:
    values = {seed: np.arange(1000, dtype=np.float64) + seed / 100000.0 for seed in contract.TRAINING_SEEDS}
    first = summary.hierarchical_bootstrap(values)
    second = summary.hierarchical_bootstrap(values)
    assert first == second
    assert first["bootstrap_reps"] == 5000
    assert first["bootstrap_seed"] == 20260818


def test_cli_rejects_bootstrap_override() -> None:
    with pytest.raises(SystemExit):
        summary.build_parser().parse_args(["--output-dir", "/tmp/c1", "--bootstrap-reps", "1"])


@pytest.mark.parametrize("option", ["--plan", "--eval-root", "--preflight", "--execution-manifest"])
def test_formal_cli_rejects_noncanonical_input_override(option: str) -> None:
    with pytest.raises(SystemExit):
        summary.build_parser().parse_args(["--output-dir", "/tmp/c1", option, "/tmp/tampered.json"])


def test_formal_summary_fails_closed_before_auth_and_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    monkeypatch.setattr(launcher, "EVALUATION_AUTHORIZED", False)
    assert summary.main(["--output-dir", str(output)]) == 2
    assert output.exists() is False


def test_git_scope_allows_absent_or_only_1md_untracked() -> None:
    base = {"branch": "main", "tracked_clean": True, "tracked_changes": []}
    contract.validate_git_scope(base | {"untracked": []})
    contract.validate_git_scope(base | {"untracked": ["1.md"]})
    with pytest.raises(contract.ContractError, match="unexpected untracked"):
        contract.validate_git_scope(base | {"untracked": ["1.md", "other.txt"]})


def test_frozen_evaluator_requires_exact_commit_path_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[:2] == ["git", "cat-file"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="wrong-blob\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(contract.subprocess, "run", fake_run)
    with pytest.raises(contract.ContractError, match="wrong blob"):
        contract.validate_frozen_evaluator_object()


def _synthetic_shard_artifacts(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, str]], Path, Path]:
    output_dir = tmp_path / "shard"
    output_dir.mkdir()
    log_path = output_dir / "1700000_8192_synthetic.json.gz"
    _write_synthetic_log(log_path)
    row = summary.parse_raw_log(
        log_path,
        condition="CURRENT",
        training_seed=20260806,
        expected_seed_start=1700000,
        expected_seed_end=1700001,
        stat_cls=_FakeStat,
    )
    rows = [row for _ in range(250)]
    labels = summary.model_order("CURRENT", 20260806)
    effective_models: dict[str, dict[str, str]] = {}
    for label in labels:
        checkpoint = tmp_path / f"{label}.pth"
        checkpoint.write_bytes(label.encode("utf-8"))
        effective_models[label] = {"path": str(checkpoint), "sha256": "synthetic"}
    counts = {
        label: [
            sum(int(item["ranks_by_role"][summary.normalize_role(label)]) == rank for item in rows)
            for rank in (1, 2, 3, 4)
        ]
        for label in labels
    }
    run = {
        "condition": "CURRENT",
        "training_seed": 20260806,
        "shard": 0,
        "hanchan_seed_start": 1700000,
        "hanchan_seed_end_exclusive": 1700250,
        "output_dir": str(output_dir),
        "rows": rows,
    }
    metrics = {
        "run": {
            "kind": "four_player_native",
            "backend": "libriichi.arena.FourPlayer",
            "models": {label: str(Path(effective_models[label]["path"]).resolve()) for label in labels},
            "seed_start": 1700000,
            "seed_key": 8192,
            "games": 250,
            "seat_mode": "random",
            "native_batch_games": 250,
            "device": "cuda",
            "rank_points_values": [90.0, 45.0, 0.0, -135.0],
        },
        "metrics": {label: {"games": 250, "rank_counts": values} for label, values in counts.items()},
    }
    detailed = {
        "players": {
            label: {
                "raw": {
                    "game": 250,
                    **{f"rank_{rank}": values[rank - 1] for rank in (1, 2, 3, 4)},
                }
            }
            for label, values in counts.items()
        }
    }
    metrics_path = output_dir / "metrics.json"
    detailed_path = output_dir / "detailed_stats.json"
    contract.dump_json(metrics_path, metrics)
    contract.dump_json(detailed_path, detailed)
    return run, effective_models, metrics_path, detailed_path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed_start", 1700001),
        ("seed_key", 1),
        ("games", 249),
        ("native_batch_games", 249),
        ("seat_mode", "rotation"),
        ("rank_points_values", [90.0, 45.0, 0.0, -1.0]),
    ],
)
def test_metrics_shard_contract_rejects_frozen_run_field_tamper(
    tmp_path: Path, field: str, value: object
) -> None:
    run, effective_models, metrics_path, _detailed_path = _synthetic_shard_artifacts(tmp_path)
    metrics = json.loads(metrics_path.read_text())
    metrics["run"][field] = value
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(contract.ContractError):
        summary._validate_shard_artifacts(
            run=run,
            output_dir=metrics_path.parent,
            rows=list(run["rows"]),
            effective_models=effective_models,
        )


def test_metrics_rejects_checkpoint_path_and_rank_count_tamper(tmp_path: Path) -> None:
    run, effective_models, metrics_path, _detailed_path = _synthetic_shard_artifacts(tmp_path)
    metrics = json.loads(metrics_path.read_text())
    metrics["run"]["models"]["70k"] = "/tmp/wrong-checkpoint.pth"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="checkpoint path"):
        summary._validate_shard_artifacts(
            run=run,
            output_dir=metrics_path.parent,
            rows=list(run["rows"]),
            effective_models=effective_models,
        )

    metrics = json.loads(metrics_path.read_text())
    metrics["run"]["models"]["70k"] = str(Path(effective_models["70k"]["path"]).resolve())
    metrics["metrics"]["70k"]["rank_counts"][0] += 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="rank mismatch"):
        summary._validate_shard_artifacts(
            run=run,
            output_dir=metrics_path.parent,
            rows=list(run["rows"]),
            effective_models=effective_models,
        )


def test_detailed_stats_rejects_raw_game_or_rank_mismatch(tmp_path: Path) -> None:
    run, effective_models, _metrics_path, detailed_path = _synthetic_shard_artifacts(tmp_path)
    detailed = json.loads(detailed_path.read_text())
    detailed["players"]["70k"]["raw"]["game"] = 249
    detailed_path.write_text(json.dumps(detailed), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="game count"):
        summary._validate_shard_artifacts(
            run=run,
            output_dir=detailed_path.parent,
            rows=list(run["rows"]),
            effective_models=effective_models,
        )

    detailed = json.loads(detailed_path.read_text())
    detailed["players"]["70k"]["raw"]["game"] = 250
    detailed["players"]["70k"]["raw"]["rank_1"] += 1
    detailed_path.write_text(json.dumps(detailed), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="detailed rank mismatch"):
        summary._validate_shard_artifacts(
            run=run,
            output_dir=detailed_path.parent,
            rows=list(run["rows"]),
            effective_models=effective_models,
        )


def _authorized_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    executable = str(Path(sys.executable).resolve())
    model_files: dict[str, Path] = {}
    for label in _model_records():
        path = tmp_path / f"{label}.pth"
        path.write_bytes(label.encode("utf-8"))
        model_files[label] = path
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    current_records: dict[str, dict[str, object]] = {}
    for label in ("70k", "ext_mortal", *[f"{route}_CURRENT_{seed}" for route in ("M0", "D1") for seed in contract.TRAINING_SEEDS]):
        current_records[label] = {"path": str(model_files[label]), "sha256": digest(model_files[label]), "state": "available"}
        if label not in {"70k", "ext_mortal"}:
            current_records[label]["steps"] = 72000
    model_map = {
        label: {"label": label, "path": str(path), "sha256": current_records.get(label, {}).get("sha256")}
        for label, path in model_files.items()
    }
    for route in ("M0", "D1"):
        for seed in contract.TRAINING_SEEDS:
            label = f"{route}_CQL_OFF_{seed}"
            model_map[label]["sha256"] = None
    runs = contract.build_run_matrix(model_map, executable=executable)
    for row in runs:
        output_dir = tmp_path / "outputs" / str(row["condition"]) / str(row["training_seed"]) / f"shard_{row['shard']}"
        row["output_dir"] = str(output_dir)
        row["future_argv"] = contract.build_evaluator_argv(
            condition=str(row["condition"]),
            seed=int(row["training_seed"]),
            shard=int(row["shard"]),
            models=model_map,
            output_dir=output_dir,
            executable=executable,
        )
    plan = {
        "schema": "keqing.mortal.c1_evaluation_plan.v1",
        "experiment_id": contract.C1_ID,
        "status": "prepared_not_authorized",
        "evaluation_authorized": False,
        "evaluation_games_run": 0,
        "git_scope": {"commit": "authorized-test-commit"},
        "runtime_provenance": {"sys_executable": executable},
        "evaluator_provenance": {},
        "evaluation_dependency_sources": [],
        "mortal_revision": contract.MORTAL_REVISION,
        "models": list(model_map.values()),
        "runs": runs,
    }
    plan_path = tmp_path / "evaluation_plan.json"
    contract.dump_json(plan_path, plan)
    preflight_path = tmp_path / "implementation_preflight.json"
    contract.dump_json(
        preflight_path,
        {
            "implementation_preflight_passed": True,
            "passed": True,
            "plan_sha256": digest(plan_path),
            "git": {"commit": "authorized-test-commit"},
            "evaluation_games_run": 0,
            "new_checkpoints": 0,
        },
    )
    closure_rows = []
    for route in ("M0", "D1"):
        for seed in contract.TRAINING_SEEDS:
            label = f"{route}_CQL_OFF_{seed}"
            closure_rows.append(
                {
                    "route": f"{route}_CQL_OFF",
                    "training_seed": seed,
                    "final_checkpoint_path": str(model_files[label]),
                    "final_checkpoint_sha256": digest(model_files[label]),
                    "steps": 72000,
                    "trained_optimizer_steps": 2000,
                    "parent_checkpoint_sha256": contract.K0_SHA256,
                    "cql_min_q_weight": 0.0,
                    "objective": "behavior_action_mc",
                    "reward": "final_rank_mc",
                    "initialization": {"optimizer": "preserved", "scheduler": "fresh", "scaler": "fresh", "data_stream": "fresh"},
                    "data_seed": seed,
                }
            )
    closure_path = tmp_path / "training_completion_closure.json"
    contract.dump_json(closure_path, {"experiment_id": contract.C1_ID, "runs": closure_rows})
    execution = contract.resolve_execution_manifest(plan, {"experiment_id": contract.C1_ID, "runs": closure_rows})
    execution_path = tmp_path / "execution_manifest.json"
    contract.dump_json(execution_path, execution)

    monkeypatch.setattr(launcher, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(launcher, "DEFAULT_PREFLIGHT", preflight_path)
    monkeypatch.setattr(launcher, "DEFAULT_COMPLETION_CLOSURE", closure_path)
    monkeypatch.setattr(launcher, "DEFAULT_EXECUTION_MANIFEST", execution_path)
    monkeypatch.setattr(launcher, "EVALUATION_AUTHORIZED", True)
    monkeypatch.setattr(launcher, "APPROVED_EVALUATION_IMPLEMENTATION_COMMIT", "authorized-test-commit")
    monkeypatch.setattr(launcher, "AUTHORIZED_EVALUATION_PLAN_SHA256", digest(plan_path))
    monkeypatch.setattr(launcher, "AUTHORIZED_EVALUATION_PREFLIGHT_SHA256", digest(preflight_path))
    monkeypatch.setattr(launcher, "AUTHORIZED_TRAINING_COMPLETION_SHA256", digest(closure_path))
    monkeypatch.setattr(launcher, "AUTHORIZED_EXECUTION_MANIFEST_SHA256", digest(execution_path))
    monkeypatch.setattr(launcher, "current_checkpoint_records", lambda: current_records)
    monkeypatch.setattr(launcher, "validate_runtime_provenance", lambda expected: expected)
    monkeypatch.setattr(launcher, "validate_source_provenance", lambda expected: expected)
    monkeypatch.setattr(launcher, "validate_frozen_evaluator_object", lambda: None)
    return {"plan": plan_path, "preflight": preflight_path, "closure": closure_path, "execution": execution_path, "off": model_files["M0_CQL_OFF_20260806"], "output": Path(runs[0]["output_dir"]), "digest": digest}


def _bind_summary_fixture(fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summary, "EVALUATION_PLAN_PATH", Path(fixture["plan"]))
    monkeypatch.setattr(summary, "IMPLEMENTATION_PREFLIGHT_PATH", Path(fixture["preflight"]))
    monkeypatch.setattr(summary, "TRAINING_COMPLETION_CLOSURE_PATH", Path(fixture["closure"]))
    monkeypatch.setattr(summary, "EXECUTION_MANIFEST_PATH", Path(fixture["execution"]))


@pytest.mark.parametrize("tamper", ["plan_sha", "preflight_semantics", "completion_semantics", "execution_semantics"])
def test_formal_artifact_chain_rejects_authorization_or_semantic_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    _bind_summary_fixture(fixture, monkeypatch)
    summary._validate_formal_artifact_chain()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if tamper == "plan_sha":
        path = Path(fixture["plan"])
        value = json.loads(path.read_text())
        value["status"] = "tampered"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "preflight_semantics":
        path = Path(fixture["preflight"])
        value = json.loads(path.read_text())
        value["passed"] = False
        path.write_text(json.dumps(value), encoding="utf-8")
        monkeypatch.setattr(launcher, "AUTHORIZED_EVALUATION_PREFLIGHT_SHA256", digest(path))
    elif tamper == "completion_semantics":
        path = Path(fixture["closure"])
        value = json.loads(path.read_text())
        value["runs"][0]["data_seed"] = 1
        path.write_text(json.dumps(value), encoding="utf-8")
        monkeypatch.setattr(launcher, "AUTHORIZED_TRAINING_COMPLETION_SHA256", digest(path))
    else:
        path = Path(fixture["execution"])
        value = json.loads(path.read_text())
        value["status"] = "tampered"
        path.write_text(json.dumps(value), encoding="utf-8")
        monkeypatch.setattr(launcher, "AUTHORIZED_EXECUTION_MANIFEST_SHA256", digest(path))

    with pytest.raises(contract.ContractError):
        summary._validate_formal_artifact_chain()


def test_authorized_simulation_executes_exactly_once_shell_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    token = launcher.confirmation_token(
        condition="CQL_OFF",
        seed=20260806,
        shard=0,
        implementation_commit=str(launcher.APPROVED_EVALUATION_IMPLEMENTATION_COMMIT),
        preflight_sha256=str(launcher.AUTHORIZED_EVALUATION_PREFLIGHT_SHA256),
    )
    assert launcher.main(["--condition", "CQL_OFF", "--seed", "20260806", "--shard", "0", "--execute", "--confirmation-token", token]) == 0
    assert len(calls) == 1
    assert calls[0]["shell"] is False
    assert calls[0]["check"] is True
    assert calls[0]["cwd"] == contract.REPO_ROOT
    assert calls[0]["command"][1] == "training/mortal/four_player_native.py"
    assert fixture["output"].exists() is False


@pytest.mark.parametrize("tamper", ["evaluator", "native", "checkpoint", "plan", "completion", "execution", "seed", "model_order", "output"])
def test_authorized_tamper_guards_never_reach_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    if tamper == "evaluator":
        monkeypatch.setattr(launcher, "validate_frozen_evaluator_object", lambda: (_ for _ in ()).throw(launcher.AuthorizationError("evaluator tamper")))
    elif tamper == "native":
        monkeypatch.setattr(launcher, "validate_runtime_provenance", lambda expected: (_ for _ in ()).throw(launcher.AuthorizationError("native tamper")))
    elif tamper == "checkpoint":
        Path(fixture["off"]).write_bytes(b"tampered checkpoint")
    elif tamper in {"plan", "seed", "model_order"}:
        path = Path(fixture["plan"])
        value = json.loads(path.read_text())
        if tamper == "seed":
            value["runs"][0]["training_seed"] = 999
        elif tamper == "model_order":
            value["runs"][0]["model_order"] = list(reversed(value["runs"][0]["model_order"]))
        else:
            value["status"] = "tampered"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "completion":
        path = Path(fixture["closure"])
        value = json.loads(path.read_text())
        value["runs"][0]["data_seed"] = 1
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "execution":
        path = Path(fixture["execution"])
        value = json.loads(path.read_text())
        value["status"] = "tampered"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "output":
        output = Path(fixture["output"])
        output.mkdir(parents=True)
        (output / "existing.log").write_text("no resume", encoding="utf-8")

    called: list[object] = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: called.append(args))
    with pytest.raises(SystemExit):
        launcher.main(["--condition", "CQL_OFF", "--seed", "20260806", "--shard", "0", "--execute"])
    assert called == []
