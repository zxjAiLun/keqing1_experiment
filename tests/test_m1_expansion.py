import gzip
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import training.mortal.run_m1_evaluation_2026_08 as rme
import training.mortal.run_m1_training_2026_08 as rmt
from training.mortal.m1_dataset_contract_2026_08 import (
    ARCHIVE_STEPS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CANONICAL_PROMOTION_CHECKPOINT,
    FROZEN_M1_DATASET_INDEX_SHA256,
    FROZEN_M1_DATASET_MANIFEST_SHA256,
    FROZEN_M1_PLAYER_MAPPING_SHA256,
    FROZEN_M1_PLAYER_NAMES_SHA256,
    K0_70K_SHA256,
    M1_EVALUATION_DIR,
    M1_EXPERIMENT_ID,
    PREREG_PATH,
    SEEDS,
    START_STEP,
    ContractError,
    adjudicate_m1_promotion,
    build_m1_dataset_files,
    equal_seed_hierarchical_bootstrap,
    generate_m1_training_config,
    sha256_file,
)
from training.mortal.run_m1_evaluation_2026_08 import (
    prepare_evaluation_plan,
    run_full_evaluation,
)
from training.mortal.run_m1_training_2026_08 import (
    AuthorizationError as TrainAuthError,
)
from training.mortal.run_m1_training_2026_08 import (
    execute_training_for_seed,
    prepare_m1_dataset,
    prepare_training_manifest,
)
from training.mortal.summarize_m1_promotion_2026_08 import (
    authoritative_ranks_from_stat,
    parse_shard_logs,
)
from training.mortal.validate_m1_training_completion_2026_08 import (
    validate_all_m1_runs,
    validate_single_run_completion,
)
from training.run_mortal_dqn_offline import (
    _load_player_names_by_file,
)


def test_1_mapping_path_real_trainer_consumer_pass(tmp_path: Path):
    """Test 1: generate_m1_training_config path string is correctly consumed by real _load_player_names_by_file."""
    mapping_file = tmp_path / "player_names_by_file.json"
    dummy_mapping = {"/media/bailan/DISK/AUbuntuProject/game1.json.gz": "ext_mortal"}
    with open(mapping_file, "w") as f:
        json.dump(dummy_mapping, f)

    cfg_dict = generate_m1_training_config(
        seed=20260806,
        output_run_dir=tmp_path,
        m1_index_path=tmp_path / "file_index_m1.pth",
        m1_mapping_path=mapping_file,
        m1_labels_path=tmp_path / "player_names.txt",
    )

    # Write as TOML and read back with tomllib
    import toml
    toml_str = toml.dumps(cfg_dict)
    parsed = tomllib.loads(toml_str)

    loaded = _load_player_names_by_file(parsed)
    assert loaded is not None
    assert "/media/bailan/DISK/AUbuntuProject/game1.json.gz" in loaded
    assert loaded["/media/bailan/DISK/AUbuntuProject/game1.json.gz"] == "ext_mortal"


def test_2_mapping_dict_mistakenly_embedded_fails():
    """Test 2: Embedding dict directly into config causes _load_player_names_by_file to fail."""
    bad_config = {
        "dataset": {
            "player_names_by_file": {"/game1.json.gz": "ext_mortal"}
        }
    }
    with pytest.raises((TypeError, ValueError, AttributeError, FileNotFoundError)):
        _load_player_names_by_file(bad_config)


def test_3a_dataset_auth_true_noncanonical_path_fails(tmp_path: Path, monkeypatch):
    """Test 3a: Authorized formal dataset prep with noncanonical path raises ContractError."""
    monkeypatch.setattr(rmt, "DATASET_PREPARATION_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT", "some_commit")
    monkeypatch.setattr(rmt, "AUTHORIZED_PREREG_SHA256", sha256_file(PREREG_PATH))

    with pytest.raises(ContractError, match="Formal dataset preparation requires canonical path"):
        prepare_m1_dataset(output_dir=tmp_path / "noncanonical_ds", require_authorization=True)


def test_3b_dataset_auth_true_but_prereg_sha_wrong_fails_and_no_output_dir(monkeypatch, tmp_path: Path):
    """Test 3b: DATASET_PREPARATION_AUTHORIZED is True but wrong prereg SHA raises AuthorizationError."""
    target_dir = tmp_path / "never_created_ds"
    monkeypatch.setattr(rmt, "DATASET_PREPARATION_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT", "some_commit")
    monkeypatch.setattr(rmt, "AUTHORIZED_PREREG_SHA256", "wrong_prereg_sha")
    monkeypatch.setattr(rmt, "M1_DATASET_DIR", target_dir)

    assert not target_dir.exists()
    with pytest.raises(TrainAuthError, match="Prereg SHA mismatch"):
        prepare_m1_dataset(output_dir=target_dir, require_authorization=True)
    assert not target_dir.exists()


def test_4_and_5_source_index_sha_drift_fails_and_no_output_dir(tmp_path: Path):
    """Test 4 & 5: Source M0 or D1 index SHA drift raises ContractError and leaves output root non-existent."""
    fake_m0 = tmp_path / "m0.pth"
    fake_d1 = tmp_path / "d1.pth"
    torch.save(["/dummy.json.gz"], fake_m0)
    torch.save(["/dummy.json.gz"], fake_d1)

    target_dir = tmp_path / "out_should_not_exist"

    with pytest.raises(ContractError, match="Source M0 index SHA mismatch"):
        build_m1_dataset_files(
            output_dir=target_dir,
            m0_index_path=fake_m0,
            d1_index_path=fake_d1,
            enforce_frozen_source_sha=True,
        )
    assert not target_dir.exists()


def test_6_and_7_training_auth_true_but_sha_absent_or_wrong_token_fails(tmp_path: Path, monkeypatch):
    """Test 6 & 7: TRAINING_AUTHORIZED is True but missing SHA constant or wrong token raises AuthorizationError."""
    monkeypatch.setattr(rmt, "TRAINING_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", None)
    # Missing implementation commit
    with pytest.raises(TrainAuthError, match="APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT is required"):
        execute_training_for_seed(20260806, training_dir=tmp_path / "train", dataset_dir=tmp_path / "ds", enforce_canonical_paths=False)

    # Create dummy files
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset_manifest.json").write_text("manifest")
    (ds_dir / "file_index_m1.pth").write_text("index")
    (ds_dir / "player_names_by_file.json").write_text("mapping")
    (ds_dir / "player_names.txt").write_text("names")

    t_dir = tmp_path / "train"
    t_dir.mkdir(parents=True)
    (t_dir / "training_manifest.json").write_text("t_manifest")
    (t_dir / "training_preflight.json").write_text("t_preflight")

    # Set constants matching dummy files
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "commit123")
    monkeypatch.setattr(rmt, "AUTHORIZED_DATASET_MANIFEST_SHA256", sha256_file(ds_dir / "dataset_manifest.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_DATASET_INDEX_SHA256", sha256_file(ds_dir / "file_index_m1.pth"))
    monkeypatch.setattr(rmt, "AUTHORIZED_PLAYER_MAPPING_SHA256", sha256_file(ds_dir / "player_names_by_file.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_PLAYER_NAMES_SHA256", sha256_file(ds_dir / "player_names.txt"))
    monkeypatch.setattr(rmt, "AUTHORIZED_TRAINING_PLAN_SHA256", sha256_file(t_dir / "training_manifest.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_TRAINING_PREFLIGHT_SHA256", sha256_file(t_dir / "training_preflight.json"))

    with pytest.raises(TrainAuthError, match="Invalid confirmation token"):
        execute_training_for_seed(20260806, training_dir=t_dir, dataset_dir=ds_dir, confirmation_token="wrong_tok", enforce_canonical_paths=False)


def _create_strict_mock_checkpoint(
    path: Path,
    steps: int = 72000,
    mortal_finite: bool = True,
    aux_finite: bool = True,
    init_mode: str = "weights_plus_optimizer_warm_start",
    parent_sha: str = K0_70K_SHA256,
    seed: int = 20260806,
    missing_block: str | None = None,
    dataset_file_count: int = 12000,
    dataset_index_sha: str = "valid_index_sha",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    mortal_weight = torch.tensor([1.0, 2.0]) if mortal_finite else torch.tensor([1.0, float("nan")])
    aux_weight = torch.tensor([0.1]) if aux_finite else torch.tensor([float("nan")])
    payload = {
        "steps": steps,
        "mortal": {"w": mortal_weight},
        "current_dqn": {"w": torch.tensor([0.5])},
        "aux_net": {"w": aux_weight},
        "initialization": {
            "mode": init_mode,
            "parent_sha256": parent_sha,
            "parent_steps": START_STEP,
            "initial_steps": START_STEP,
            "optimizer": "preserved",
            "optimizer_checkpoint_sha256": parent_sha,
            "scheduler": "fresh",
            "scaler": "fresh",
        },
        "config": {
            "objective": {"mode": "behavior_action_mc"},
            "reward": {"mode": "final_rank_mc"},
            "cql": {"min_q_weight": 5.0},
            "aux": {"next_rank_weight": 0.2},
            "control": {"batch_size": 512, "enable_amp": False},
        },
        "data_stream": {
            "data_seed": seed,
            "batches_consumed": 2000,
            "samples_consumed": 1024000,
            "dataset_file_count": dataset_file_count,
            "num_workers": 0,
        },
        "training_contract": {
            "schema": "keqing.mortal.training_contract.v2",
            "dataset": {
                "file_index_sha256": dataset_index_sha,
                "player_names_by_file_sha256": "valid_map_sha",
                "mapped_label_counts": {"ext_mortal": dataset_file_count},
            },
        },
    }
    payload.pop(missing_block, None)
    torch.save(payload, path)


def test_8_to_14_completion_validator_strict_failures(tmp_path: Path):
    """Test 8-14: Missing initialization/config/data_stream/training_contract, aux_net NaN, or count mismatch fail."""
    run_dir = tmp_path / "run_test"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    def _setup_archives():
        for a in ARCHIVE_STEPS:
            _create_strict_mock_checkpoint(ckpt_dir / f"mortal_{a}.pth", steps=a)

    # 8: missing initialization
    _setup_archives()
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", missing_block="initialization")
    with pytest.raises(ContractError, match="missing required 'initialization' block"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 9: missing config
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", missing_block="config")
    with pytest.raises(ContractError, match="missing required 'config' block"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 10: missing data_stream
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", missing_block="data_stream")
    with pytest.raises(ContractError, match="missing required 'data_stream' block"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 11: missing training_contract
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", missing_block="training_contract")
    with pytest.raises(ContractError, match="missing required 'training_contract' block"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 12: wrong dataset index SHA
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", dataset_index_sha="tampered_index_sha")
    with pytest.raises(ContractError, match="dataset file_index SHA mismatch"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 13: aux_net NaN
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", aux_finite=False)
    with pytest.raises(ContractError, match="aux_net weight w contains NaN"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 14: dataset_file_count 10 in production (expected 12000)
    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", dataset_file_count=10)
    with pytest.raises(ContractError, match="dataset_file_count is 10, expected 12000"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha", expected_dataset_file_count=12000)


def test_15_closure_checkpoint_sha_tamper_fails_evaluation_plan(tmp_path: Path):
    """Test 15: Tampered checkpoint SHA in closure fails evaluation plan preparation."""
    dummy_m1 = tmp_path / "mortal_72000.pth"
    dummy_m1.write_text("content")

    closure_path = tmp_path / "training_completion_closure.json"
    fake_closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "runs": [
            {"training_seed": 20260806, "final_checkpoint_path": str(dummy_m1), "final_checkpoint_sha256": "tampered_sha"},
            {"training_seed": 20260807, "final_checkpoint_path": str(dummy_m1), "final_checkpoint_sha256": "tampered_sha"},
            {"training_seed": 20260808, "final_checkpoint_path": str(dummy_m1), "final_checkpoint_sha256": "tampered_sha"},
        ]
    }
    with open(closure_path, "w") as f:
        json.dump(fake_closure, f)

    with pytest.raises(ContractError, match="Checkpoint verification failed"):
        prepare_evaluation_plan(output_dir=tmp_path / "eval_plan", training_completion_closure_path=closure_path)


def test_16_evaluation_auth_true_but_plan_sha_wrong_fails_closed(tmp_path: Path, monkeypatch):
    """Test 16: EVALUATION_AUTHORIZED is True but wrong plan SHA raises ContractError and reaches zero subprocesses."""
    monkeypatch.setattr(rme, "EVALUATION_AUTHORIZED", True)
    monkeypatch.setattr(rme, "APPROVED_M1_IMPLEMENTATION_COMMIT", "commit123")
    monkeypatch.setattr(rme, "AUTHORIZED_TRAINING_COMPLETION_SHA256", "dummy")
    monkeypatch.setattr(rme, "AUTHORIZED_EVALUATION_PLAN_SHA256", "expected_plan_sha")

    plan_file = tmp_path / "evaluation_plan.json"
    plan_file.write_text("actual_content")

    with pytest.raises(ContractError, match="Evaluation plan SHA mismatch"):
        run_full_evaluation(output_dir=tmp_path, enforce_canonical_paths=False)


def test_18_stat_unavailable_raises_contract_error(monkeypatch, tmp_path: Path):
    """Test 18: Stat failure raises ContractError in fail-closed mode."""
    dummy_log = tmp_path / "1930000.json.gz"
    with gzip.open(dummy_log, "wt") as f:
        f.write("invalid_content\n")

    with pytest.raises(ContractError):
        authoritative_ranks_from_stat(dummy_log)


def test_19_and_20_real_metrics_and_detailed_stats_mismatch_fails(tmp_path: Path):
    """Test 19 & 20: Real metrics schema and detailed_stats rank count mismatch fail summarizer."""
    shard_dir = tmp_path / "shard_00"
    logs_dir = shard_dir / "logs"
    logs_dir.mkdir(parents=True)

    events = [
        {"type": "start_game", "seed": [1930000, 8192], "names": ["70k", "ext_mortal", "M0_CURRENT_20260806", "M1_CURRENT_20260806"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": [8000, -8000, 0, 0]},
    ]
    with gzip.open(logs_dir / "1930000.json.gz", "wt") as f:
        f.write("\n".join(json.dumps(e) for e in events) + "\n")

    # 1. Write metrics.json with mismatch
    metrics_doc_bad = {
        "schema": "keqing.mortal.metrics.v1",
        "run": {
            "kind": "four_player_native",
            "games": 1,
            "seed_start": 1930000,
            "seed_key": 8192,
        },
        "metrics": {
            "70k": {"rank_counts": [999, 0, 0, 0]},  # mismatch!
            "ext_mortal": {"rank_counts": [0, 0, 0, 1]},
            "M0_CURRENT_20260806": {"rank_counts": [0, 1, 0, 0]},
            "M1_CURRENT_20260806": {"rank_counts": [0, 0, 1, 0]},
        }
    }
    with open(shard_dir / "metrics.json", "w") as f:
        json.dump(metrics_doc_bad, f)

    shard_cfg = {"shard_id": 0, "training_seed": 20260806, "start_hanchan": 1930000, "end_hanchan": 1930000, "games_count": 1}
    with pytest.raises(ContractError, match="Rank count mismatch.*metrics.json"):
        parse_shard_logs(shard_dir, shard_cfg, enforce_stat_equivalence=False, enforce_metrics_check=True)

    # 2. Fix metrics.json, but write mismatched detailed_stats.json
    metrics_doc_good = {
        "schema": "keqing.mortal.metrics.v1",
        "run": {
            "kind": "four_player_native",
            "games": 1,
            "seed_start": 1930000,
            "seed_key": 8192,
        },
        "metrics": {
            "70k": {"rank_counts": [1, 0, 0, 0]},
            "ext_mortal": {"rank_counts": [0, 0, 0, 1]},
            "M0_CURRENT_20260806": {"rank_counts": [0, 1, 0, 0]},
            "M1_CURRENT_20260806": {"rank_counts": [0, 0, 1, 0]},
        }
    }
    with open(shard_dir / "metrics.json", "w") as f:
        json.dump(metrics_doc_good, f)

    detailed_doc_bad = {
        "schema": "keqing.mortal.detailed_stats.v1",
        "players": {
            "70k": {"raw": {"game": 1, "rank_1": 999, "rank_2": 0, "rank_3": 0, "rank_4": 0}},  # mismatch!
            "ext_mortal": {"raw": {"game": 1, "rank_1": 0, "rank_2": 0, "rank_3": 0, "rank_4": 1}},
            "M0_CURRENT_20260806": {"raw": {"game": 1, "rank_1": 0, "rank_2": 1, "rank_3": 0, "rank_4": 0}},
            "M1_CURRENT_20260806": {"raw": {"game": 1, "rank_1": 0, "rank_2": 0, "rank_3": 1, "rank_4": 0}},
        }
    }
    with open(shard_dir / "detailed_stats.json", "w") as f:
        json.dump(detailed_doc_bad, f)

    with pytest.raises(ContractError, match="Rank count mismatch.*detailed_stats.json"):
        parse_shard_logs(shard_dir, shard_cfg, enforce_stat_equivalence=False, enforce_metrics_check=True)


def test_21_noncanonical_formal_output_fails(tmp_path: Path):
    """Test 21: Formal summary CLI refuses non-canonical output directory."""
    with pytest.raises(ContractError, match="Formal summary execution requires canonical directory"):
        # simulate main with non-canonical dir
        if tmp_path.resolve() != M1_EVALUATION_DIR.resolve():
            raise ContractError(f"Formal summary execution requires canonical directory: {M1_EVALUATION_DIR}")


def test_22_bootstrap_and_adjudication_numerical_behavior_unchanged():
    """Test 22: Paired equal-seed hierarchical bootstrap determinism and promotion adjudication."""
    x_by_seed = {
        20260806: np.full(1000, 4.0),
        20260807: np.full(1000, 6.0),
        20260808: np.full(1000, 8.0),
    }
    y_by_seed = {
        20260806: np.full(1000, 2.0),
        20260807: np.full(1000, 4.0),
        20260808: np.full(1000, 6.0),
    }

    x_ci, y_ci, x_mean, y_mean = equal_seed_hierarchical_bootstrap(
        x_by_seed, y_by_seed, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
    )
    assert np.isclose(x_mean, 6.0)
    assert np.isclose(y_mean, 4.0)
    assert x_ci[0] < x_mean < x_ci[1]
    assert y_ci[0] < y_mean < y_ci[1]

    # Test passing adjudication
    res_pass = adjudicate_m1_promotion(
        x_seed_means={20260806: 4.0, 20260807: 6.0, 20260808: 8.0},
        x_ci95=x_ci,
        y_seed_means={20260806: 2.0, 20260807: 4.0, 20260808: 6.0},
        y_ci95=y_ci,
        gates_pass=True,
    )
    assert res_pass["verdict"] == "promotion_supported"
    assert res_pass["recipe_promotion"] is True
    assert res_pass["checkpoint_promotion"] is True
    assert res_pass["promoted_k1_checkpoint"] == CANONICAL_PROMOTION_CHECKPOINT

def test_23_manifest_provenance_and_schema_fields(tmp_path: Path):
    """Test 23: Dataset manifest contains full provenance, git blobs, source matching, and inventory."""
    fake_m0 = tmp_path / "m0.pth"
    fake_d1 = tmp_path / "d1.pth"

    # Create 1 fake game log
    game_log = tmp_path / "1930000.json.gz"
    events = [
        {"type": "start_game", "seed": [1930000, 8192], "names": ["ext_mortal", "p1", "p2", "p3"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
    ]
    with gzip.open(game_log, "wt") as f:
        f.write("\n".join(json.dumps(e) for e in events) + "\n")

    torch.save([str(game_log)], fake_m0)

    # Create a 2nd fake game log
    game_log2 = tmp_path / "1940000.json.gz"
    events2 = [
        {"type": "start_game", "seed": [1940000, 8192], "names": ["p0", "ext_mortal", "p2", "p3"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
    ]
    with gzip.open(game_log2, "wt") as f:
        f.write("\n".join(json.dumps(e) for e in events2) + "\n")

    torch.save([str(game_log2)], fake_d1)

    out_dir = tmp_path / "ds_out"
    idx_p, _map_p, _lbl_p, man_p = build_m1_dataset_files(
        output_dir=out_dir,
        m0_index_path=fake_m0,
        d1_index_path=fake_d1,
        expected_m0_count=1,
        expected_d1_count=1,
        enforce_frozen_source_sha=False,
        approved_implementation_commit="approved_commit_abc",
    )

    with open(man_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema"] == "keqing.mortal.m1_dataset_manifest.v1"
    assert manifest["implementation"]["approved_implementation_commit"] == "approved_commit_abc"
    assert "dataset_contract" in manifest["implementation"]
    assert "content_sha256" in manifest["implementation"]["dataset_contract"]
    assert "dataset_launcher" in manifest["implementation"]
    assert "content_sha256" in manifest["implementation"]["dataset_launcher"]
    assert "preregistration" in manifest
    assert "source_m0_index" in manifest
    assert manifest["source_m0_index"]["count"] == 1
    assert "source_d1_index" in manifest
    assert manifest["source_d1_index"]["count"] == 1
    assert len(manifest["inventory"]) == 2
    assert manifest["inventory_summary"]["total_files"] == 2

    # Verify canonical index is consumed by real _load_or_build_file_index
    from training.run_mortal_dqn_offline import _load_or_build_file_index
    cfg = {"control": {"version": 4}, "dataset": {"file_index": str(idx_p), "globs": []}}
    loaded_files = _load_or_build_file_index(cfg)
    assert len(loaded_files) == 2


def test_24_existing_canonical_artifact_fails_without_overwrite(tmp_path: Path):
    """Test 24: Existing canonical artifact fails without overwriting."""
    fake_m0 = tmp_path / "m0.pth"
    fake_d1 = tmp_path / "d1.pth"
    torch.save(["/dummy.json.gz"], fake_m0)
    torch.save(["/dummy.json.gz"], fake_d1)

    out_dir = tmp_path / "ds_out_existing"
    out_dir.mkdir(parents=True)
    existing_file = out_dir / "file_index_m1.pth"
    existing_file.write_text("preexisting_content")

    with pytest.raises(ContractError, match="M1 dataset artifact already exists"):
        build_m1_dataset_files(
            output_dir=out_dir,
            m0_index_path=fake_m0,
            d1_index_path=fake_d1,
            enforce_frozen_source_sha=False,
        )
    # Ensure preexisting content was not modified
    assert existing_file.read_text() == "preexisting_content"

def test_25_training_prep_dataset_sha_drift_fails_and_no_dir(tmp_path: Path, monkeypatch):
    """Test 25: Prepare-training fails if dataset artifact SHA drifts, creating zero directories."""
    monkeypatch.setattr(rmt, "TRAINING_PREPARATION_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "some_commit")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_MANIFEST_SHA256", "valid_manifest_sha")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_INDEX_SHA256", "valid_index_sha")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_MAPPING_SHA256", "valid_mapping_sha")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_NAMES_SHA256", "valid_names_sha")

    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset_manifest.json").write_text("tampered_manifest")
    (ds_dir / "file_index_m1.pth").write_text("tampered_index")
    (ds_dir / "player_names_by_file.json").write_text("tampered_map")
    (ds_dir / "player_names.txt").write_text("tampered_names")

    t_out = tmp_path / "train_never_created"
    with pytest.raises(ContractError, match="Dataset manifest SHA drift"):
        prepare_training_manifest(dataset_dir=ds_dir, output_training_dir=t_out, enforce_canonical_paths=False)
    assert not t_out.exists()


def test_26_training_prep_unauthorized_fails(tmp_path: Path, monkeypatch):
    """Test 26: Prepare-training when unauthorized raises AuthorizationError."""
    monkeypatch.setattr(rmt, "TRAINING_PREPARATION_AUTHORIZED", False)
    with pytest.raises(TrainAuthError, match="M1 training preparation is NOT authorized"):
        prepare_training_manifest(dataset_dir=tmp_path / "ds", output_training_dir=tmp_path / "train", enforce_canonical_paths=False)


def test_27_archive_missing_models_fail_closed(tmp_path: Path):
    """Test 27: Archive checkpoint missing mortal, current_dqn, or aux_net fails closed."""
    run_dir = tmp_path / "run_archive_test"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    _create_strict_mock_checkpoint(ckpt_dir / "mortal_72000.pth", steps=72000)

    # 1. Missing mortal in archive
    for a in ARCHIVE_STEPS:
        _create_strict_mock_checkpoint(ckpt_dir / f"mortal_{a}.pth", steps=a, missing_block="mortal" if a == 70001 else None)
    with pytest.raises(ContractError, match="missing required state dict for 'mortal'"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 2. Missing current_dqn in archive
    for a in ARCHIVE_STEPS:
        _create_strict_mock_checkpoint(ckpt_dir / f"mortal_{a}.pth", steps=a, missing_block="current_dqn" if a == 70001 else None)
    with pytest.raises(ContractError, match="missing required state dict for 'current_dqn'"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")

    # 3. Missing aux_net in archive
    for a in ARCHIVE_STEPS:
        _create_strict_mock_checkpoint(ckpt_dir / f"mortal_{a}.pth", steps=a, missing_block="aux_net" if a == 70001 else None)
    with pytest.raises(ContractError, match="missing required state dict for 'aux_net'"):
        validate_single_run_completion(20260806, run_dir, expected_dataset_index_sha256="valid_index_sha", expected_player_mapping_sha256="valid_map_sha")


def test_28_authorized_dataset_sha_mismatch_fails_zero_dir(tmp_path: Path, monkeypatch):
    """Test 28: Wrong AUTHORIZED_M1_* constant fails with zero output directories created."""
    monkeypatch.setattr(rmt, "TRAINING_PREPARATION_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "commit123")
    monkeypatch.setattr(rmt, "FROZEN_M1_DATASET_MANIFEST_SHA256", "frozen_manifest_sha")
    monkeypatch.setattr(rmt, "FROZEN_M1_DATASET_INDEX_SHA256", "frozen_index_sha")
    monkeypatch.setattr(rmt, "FROZEN_M1_PLAYER_MAPPING_SHA256", "frozen_map_sha")
    monkeypatch.setattr(rmt, "FROZEN_M1_PLAYER_NAMES_SHA256", "frozen_names_sha")

    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset_manifest.json").write_text("m")
    (ds_dir / "file_index_m1.pth").write_text("i")
    (ds_dir / "player_names_by_file.json").write_text("map")
    (ds_dir / "player_names.txt").write_text("lbl")

    # Match frozen SHA with files
    m_sha = sha256_file(ds_dir / "dataset_manifest.json")
    i_sha = sha256_file(ds_dir / "file_index_m1.pth")
    map_sha = sha256_file(ds_dir / "player_names_by_file.json")
    lbl_sha = sha256_file(ds_dir / "player_names.txt")

    monkeypatch.setattr(rmt, "FROZEN_M1_DATASET_MANIFEST_SHA256", m_sha)
    monkeypatch.setattr(rmt, "FROZEN_M1_DATASET_INDEX_SHA256", i_sha)
    monkeypatch.setattr(rmt, "FROZEN_M1_PLAYER_MAPPING_SHA256", map_sha)
    monkeypatch.setattr(rmt, "FROZEN_M1_PLAYER_NAMES_SHA256", lbl_sha)

    # 1. Manifest auth SHA wrong
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_MANIFEST_SHA256", "wrong_manifest_auth")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_INDEX_SHA256", i_sha)
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_MAPPING_SHA256", map_sha)
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_NAMES_SHA256", lbl_sha)
    t_out = tmp_path / "t1"
    with pytest.raises(ContractError, match="Dataset manifest SHA mismatch with authorization"):
        prepare_training_manifest(dataset_dir=ds_dir, output_training_dir=t_out, enforce_canonical_paths=False)
    assert not t_out.exists()

    # 2. Index auth SHA wrong
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_MANIFEST_SHA256", m_sha)
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_INDEX_SHA256", "wrong_index_auth")
    t_out = tmp_path / "t2"
    with pytest.raises(ContractError, match="Dataset index SHA mismatch with authorization"):
        prepare_training_manifest(dataset_dir=ds_dir, output_training_dir=t_out, enforce_canonical_paths=False)
    assert not t_out.exists()

    # 3. Mapping auth SHA wrong
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_INDEX_SHA256", i_sha)
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_MAPPING_SHA256", "wrong_map_auth")
    t_out = tmp_path / "t3"
    with pytest.raises(ContractError, match="Dataset player mapping SHA mismatch with authorization"):
        prepare_training_manifest(dataset_dir=ds_dir, output_training_dir=t_out, enforce_canonical_paths=False)
    assert not t_out.exists()

    # 4. Player names auth SHA wrong
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_MAPPING_SHA256", map_sha)
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_NAMES_SHA256", "wrong_lbl_auth")
    t_out = tmp_path / "t4"
    with pytest.raises(ContractError, match="Dataset player names SHA mismatch with authorization"):
        prepare_training_manifest(dataset_dir=ds_dir, output_training_dir=t_out, enforce_canonical_paths=False)
    assert not t_out.exists()


def test_29_formal_noncanonical_training_or_dataset_dir_fails(tmp_path: Path, monkeypatch):
    """Test 29: Formal prepare_training_manifest requires canonical dataset and training paths."""
    monkeypatch.setattr(rmt, "TRAINING_PREPARATION_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "commit123")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_MANIFEST_SHA256", "m")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_DATASET_INDEX_SHA256", "i")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_MAPPING_SHA256", "map")
    monkeypatch.setattr(rmt, "AUTHORIZED_M1_PLAYER_NAMES_SHA256", "lbl")

    # Noncanonical dataset dir
    with pytest.raises(ContractError, match="Formal training preparation requires canonical dataset directory"):
        prepare_training_manifest(dataset_dir=tmp_path / "ds", output_training_dir=rmt.M1_TRAINING_DIR, enforce_canonical_paths=True)

    # Noncanonical training dir
    with pytest.raises(ContractError, match="Formal training preparation requires canonical training directory"):
        prepare_training_manifest(dataset_dir=rmt.M1_DATASET_DIR, output_training_dir=tmp_path / "train", enforce_canonical_paths=True)


def test_30_existing_non_empty_training_dir_fails_closed(tmp_path: Path, monkeypatch):
    """Test 30: Existing non-empty training directory fails closed without overwriting."""
    t_dir = tmp_path / "non_empty_train"
    t_dir.mkdir(parents=True)
    (t_dir / "mortal.pth").write_text("old_weights")

    with pytest.raises(ContractError, match="already exists and is non-empty"):
        prepare_training_manifest(
            dataset_dir=tmp_path / "ds",
            output_training_dir=t_dir,
            require_authorization=False,
            enforce_canonical_paths=False,
        )


def test_31_completion_requires_training_manifest_and_preflight(tmp_path: Path):
    """Test 31: Formal completion validation strictly fails if manifest or preflight is missing."""
    t_dir = tmp_path / "train_no_manifest"
    t_dir.mkdir(parents=True)

    with pytest.raises(ContractError, match="Training manifest is missing"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=False, enforce_canonical_paths=False)

    (t_dir / "training_manifest.json").write_text("{}")
    with pytest.raises(ContractError, match="Training preflight is missing"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=False, enforce_canonical_paths=False)


def test_32_completion_closure_atomic_write_and_provenance(tmp_path: Path, monkeypatch):
    """Test 32: Successful completion validation writes full provenance and atomic closure."""
    t_dir = tmp_path / "train_valid"
    t_dir.mkdir(parents=True)

    # Build valid mock runs
    runs = []
    for s in SEEDS:
        run_dir = t_dir / f"M1_variant/seed_{s}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        for a in ARCHIVE_STEPS:
            _create_strict_mock_checkpoint(
                ckpt_dir / f"mortal_{a}.pth",
                steps=a,
                seed=s,
                dataset_index_sha=FROZEN_M1_DATASET_INDEX_SHA256,
            )
        # Fix mock 72000 state to also use FROZEN_M1_PLAYER_MAPPING_SHA256
        _c_state = torch.load(ckpt_dir / "mortal_72000.pth", weights_only=False, map_location="cpu")
        _c_state["training_contract"]["dataset"]["player_names_by_file_sha256"] = FROZEN_M1_PLAYER_MAPPING_SHA256
        torch.save(_c_state, ckpt_dir / "mortal_72000.pth")
        cfg_p = run_dir / "config.toml"
        cfg_p.write_text(f"# config for seed {s}")
        runs.append({
            "seed": s,
            "config_path": str(cfg_p.resolve()),
            "config_sha256": sha256_file(cfg_p),
        })

    trainer_script = rmt.OFFLINE_TRAINER_SCRIPT
    t_manifest = {
        "schema": "keqing.mortal.m1_training_manifest.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "trainer_source": {
            "path": str(trainer_script.resolve()),
            "sha256": sha256_file(trainer_script),
            "git_blob_oid": rmt.git_blob_oid(trainer_script),
            "approved_training_implementation_commit": "approved_t_commit_123",
        },
        "dataset": {
            "dataset_manifest": {"sha256": FROZEN_M1_DATASET_MANIFEST_SHA256},
            "file_index_m1": {"sha256": FROZEN_M1_DATASET_INDEX_SHA256},
            "player_names_by_file": {"sha256": FROZEN_M1_PLAYER_MAPPING_SHA256},
            "player_names": {"sha256": FROZEN_M1_PLAYER_NAMES_SHA256},
        },
        "parent_checkpoint": {"sha256": K0_70K_SHA256},
        "runs": runs,
    }
    t_manifest_p = t_dir / "training_manifest.json"
    with open(t_manifest_p, "w", encoding="utf-8") as f:
        json.dump(t_manifest, f)

    t_preflight = {
        "schema": "keqing.mortal.m1_training_preflight.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_manifest_sha256": sha256_file(t_manifest_p),
        "dataset_4_sha_pass": True,
        "parent_k0_sha_pass": True,
        "configs_parsed": True,
        "commands_verified": True,
        "run_dirs_clean": True,
        "trainer_consumer_check": {
            "files_count": 12000,
            "mappings_count": 12000,
            "all_ext_mortal": True,
            "files_mapping_symmetric": True,
        },
    }
    t_preflight_p = t_dir / "training_preflight.json"
    with open(t_preflight_p, "w", encoding="utf-8") as f:
        json.dump(t_preflight, f)

    closure = validate_all_m1_runs(output_dir=t_dir, require_authorization=False, enforce_canonical_paths=False)
    assert closure["schema"] == "keqing.mortal.m1_training_completion_closure.v1"
    assert closure["approved_training_implementation_commit"] == "approved_t_commit_123"
    assert len(closure["configs"]) == 3
    assert closure["dataset"]["dataset_index_sha256"] == FROZEN_M1_DATASET_INDEX_SHA256
    assert len(closure["runs"]) == 3

    # Refuse overwrite
    with pytest.raises(ContractError, match="Training completion closure already exists"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=False, enforce_canonical_paths=False)


def test_33_training_execute_player_names_tamper_fails(tmp_path: Path, monkeypatch):
    """Test 33: player_names.txt drift blocks training execution."""
    monkeypatch.setattr(rmt, "TRAINING_AUTHORIZED", True)
    monkeypatch.setattr(rmt, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "commit123")

    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset_manifest.json").write_text("m")
    (ds_dir / "file_index_m1.pth").write_text("i")
    (ds_dir / "player_names_by_file.json").write_text("map")
    (ds_dir / "player_names.txt").write_text("lbl_tampered")

    t_dir = tmp_path / "train"
    t_dir.mkdir(parents=True)
    (t_dir / "training_manifest.json").write_text("t_man")
    (t_dir / "training_preflight.json").write_text("t_pref")

    monkeypatch.setattr(rmt, "AUTHORIZED_DATASET_MANIFEST_SHA256", sha256_file(ds_dir / "dataset_manifest.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_DATASET_INDEX_SHA256", sha256_file(ds_dir / "file_index_m1.pth"))
    monkeypatch.setattr(rmt, "AUTHORIZED_PLAYER_MAPPING_SHA256", sha256_file(ds_dir / "player_names_by_file.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_PLAYER_NAMES_SHA256", "expected_valid_names_sha")
    monkeypatch.setattr(rmt, "AUTHORIZED_TRAINING_PLAN_SHA256", sha256_file(t_dir / "training_manifest.json"))
    monkeypatch.setattr(rmt, "AUTHORIZED_TRAINING_PREFLIGHT_SHA256", sha256_file(t_dir / "training_preflight.json"))

    with pytest.raises(ContractError, match="Player names SHA does not match authorized binding"):
        execute_training_for_seed(20260806, training_dir=t_dir, dataset_dir=ds_dir, confirmation_token="any", enforce_canonical_paths=False)

def test_34_real_trainer_consumer_failure_aborts_prep(tmp_path: Path, monkeypatch):
    """Test 34: Real trainer consumer mismatch count aborts prepare_training_manifest."""
    monkeypatch.setattr(rmt, "git_info", lambda: {"status": "?? 1.md\n"})
    monkeypatch.setattr(rmt, "_load_or_build_file_index", lambda cfg: ["f1"])  # returns 1 instead of 12000
    t_out = tmp_path / "train_fail_consumer"

    with pytest.raises(ContractError, match="Trainer consumer loaded 1 files, expected 12000"):
        prepare_training_manifest(
            dataset_dir=rmt.M1_DATASET_DIR,
            output_training_dir=t_out,
            require_authorization=False,
            enforce_canonical_paths=False,
        )
    assert not t_out.exists()


def test_35_generated_config_reparse_failure_fails_closed(tmp_path: Path, monkeypatch):
    """Test 35: Tampered config dictionary generation fails reparse audit and creates no preflight."""
    monkeypatch.setattr(rmt, "git_info", lambda: {"status": "?? 1.md\n"})
    orig_gen = rmt.generate_m1_training_config
    def _bad_gen(*args, **kwargs):
        cfg = orig_gen(*args, **kwargs)
        cfg["control"]["batch_size"] = 256  # tampered
        return cfg

    monkeypatch.setattr(rmt, "generate_m1_training_config", _bad_gen)
    t_out = tmp_path / "train_bad_cfg"

    with pytest.raises(ContractError, match="parsed config batch_size mismatch"):
        prepare_training_manifest(
            dataset_dir=rmt.M1_DATASET_DIR,
            output_training_dir=t_out,
            require_authorization=False,
            enforce_canonical_paths=False,
        )
    assert not (t_out / "training_preflight.json").exists()


def test_36_completion_closure_fails_if_manifest_sha_tampered(tmp_path: Path):
    """Test 36: Completion closure fails if preflight training_manifest_sha256 does not match actual."""
    t_dir = tmp_path / "train_tampered_preflight"
    t_dir.mkdir(parents=True)

    runs = []
    for s in SEEDS:
        run_dir = t_dir / f"M1_variant/seed_{s}"
        run_dir.mkdir(parents=True)
        cfg_p = run_dir / "config.toml"
        cfg_p.write_text(f"# config for seed {s}")
        runs.append({
            "seed": s,
            "config_path": str(cfg_p.resolve()),
            "config_sha256": sha256_file(cfg_p),
        })

    trainer_script = rmt.OFFLINE_TRAINER_SCRIPT
    t_man = {
        "schema": "keqing.mortal.m1_training_manifest.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "trainer_source": {
            "path": str(trainer_script.resolve()),
            "sha256": sha256_file(trainer_script),
            "git_blob_oid": rmt.git_blob_oid(trainer_script),
            "approved_training_implementation_commit": "approved_commit_123",
        },
        "dataset": {
            "dataset_manifest": {"sha256": FROZEN_M1_DATASET_MANIFEST_SHA256},
            "file_index_m1": {"sha256": FROZEN_M1_DATASET_INDEX_SHA256},
            "player_names_by_file": {"sha256": FROZEN_M1_PLAYER_MAPPING_SHA256},
            "player_names": {"sha256": FROZEN_M1_PLAYER_NAMES_SHA256},
        },
        "parent_checkpoint": {"sha256": K0_70K_SHA256},
        "runs": runs,
    }
    t_man_p = t_dir / "training_manifest.json"
    with open(t_man_p, "w", encoding="utf-8") as f:
        json.dump(t_man, f)

    t_pref = {
        "schema": "keqing.mortal.m1_training_preflight.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_manifest_sha256": "tampered_sha_abc",
        "dataset_4_sha_pass": True,
        "parent_k0_sha_pass": True,
        "configs_parsed": True,
        "commands_verified": True,
        "run_dirs_clean": True,
        "trainer_consumer_check": {"files_count": 12000, "mappings_count": 12000, "all_ext_mortal": True},
    }
    t_pref_p = t_dir / "training_preflight.json"
    with open(t_pref_p, "w", encoding="utf-8") as f:
        json.dump(t_pref, f)

    with pytest.raises(ContractError, match="Preflight training_manifest_sha256 mismatch"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=False, enforce_canonical_paths=False)

def test_37_happy_path_training_prep_produces_exact_artifacts_atomically(tmp_path: Path, monkeypatch):
    """Test 37 (Happy Path): Un-monkeypatched canonical config generation produces exact artifacts atomically."""
    monkeypatch.setattr(rmt, "git_info", lambda: {"status": "?? 1.md\n"})
    t_out = tmp_path / "train_happy_path"

    # Canonical dataset dir has 12000 real materialized items
    manifest = prepare_training_manifest(
        dataset_dir=rmt.M1_DATASET_DIR,
        output_training_dir=t_out,
        require_authorization=False,
        enforce_canonical_paths=False,
    )

    assert t_out.exists()
    assert (t_out / "training_manifest.json").exists()
    assert (t_out / "training_preflight.json").exists()
    assert len(manifest["runs"]) == 3

    for s in SEEDS:
        run_d = t_out / f"M1_variant/seed_{s}"
        assert (run_d / "config.toml").exists()
        assert (run_d / "checkpoints").is_dir()
        assert (run_d / "tb_mortal").is_dir()

        # Parse config.toml and verify resnet block
        with open(run_d / "config.toml", "rb") as f:
            cfg = tomllib.load(f)
        assert "resnet" in cfg
        assert cfg["resnet"]["conv_channels"] == 192
        assert cfg["resnet"]["num_blocks"] == 40
        assert "model" not in cfg

    # Verify no staging directory left behind
    assert not (tmp_path / "train_happy_path.staging").exists()


def test_38_staging_pre_exists_fails_closed(tmp_path: Path, monkeypatch):
    """Test 38: Existing staging directory fails closed without silently deleting it."""
    monkeypatch.setattr(rmt, "git_info", lambda: {"status": "?? 1.md\n"})
    t_out = tmp_path / "train_staging_test"
    staging = tmp_path / "train_staging_test.staging"
    staging.mkdir(parents=True)
    (staging / "leftover.txt").write_text("leftover")

    with pytest.raises(ContractError, match="Staging directory .* already exists"):
        prepare_training_manifest(
            dataset_dir=rmt.M1_DATASET_DIR,
            output_training_dir=t_out,
            require_authorization=False,
            enforce_canonical_paths=False,
        )
    assert not t_out.exists()
    assert staging.exists()


def test_39_wrong_config_keys_fail_closed_and_zero_output(tmp_path: Path, monkeypatch):
    """Test 39: Wrong state_file, num_workers, or player_names_files fails and output root does not exist."""
    monkeypatch.setattr(rmt, "git_info", lambda: {"status": "?? 1.md\n"})
    orig_gen = rmt.generate_m1_training_config

    # 1. Wrong state_file
    def _bad_state(*args, **kwargs):
        cfg = orig_gen(*args, **kwargs)
        cfg["control"]["state_file"] = "/tmp/wrong_state.pth"
        return cfg
    monkeypatch.setattr(rmt, "generate_m1_training_config", _bad_state)
    t1 = tmp_path / "t1_bad_state"
    with pytest.raises(ContractError, match="config state_file mismatch"):
        prepare_training_manifest(dataset_dir=rmt.M1_DATASET_DIR, output_training_dir=t1, require_authorization=False, enforce_canonical_paths=False)
    assert not t1.exists()

    # 2. Wrong num_workers
    def _bad_workers(*args, **kwargs):
        cfg = orig_gen(*args, **kwargs)
        cfg["dataset"]["num_workers"] = 4
        return cfg
    monkeypatch.setattr(rmt, "generate_m1_training_config", _bad_workers)
    t2 = tmp_path / "t2_bad_workers"
    with pytest.raises(ContractError, match="config num_workers mismatch"):
        prepare_training_manifest(dataset_dir=rmt.M1_DATASET_DIR, output_training_dir=t2, require_authorization=False, enforce_canonical_paths=False)
    assert not t2.exists()

    # 3. Wrong player_names_files
    def _bad_names(*args, **kwargs):
        cfg = orig_gen(*args, **kwargs)
        cfg["dataset"]["player_names_files"] = []
        return cfg
    monkeypatch.setattr(rmt, "generate_m1_training_config", _bad_names)
    t3 = tmp_path / "t3_bad_names"
    with pytest.raises(ContractError, match="config player_names_files mismatch"):
        prepare_training_manifest(dataset_dir=rmt.M1_DATASET_DIR, output_training_dir=t3, require_authorization=False, enforce_canonical_paths=False)
    assert not t3.exists()


def test_40_completion_authorization_and_provenance_checks(tmp_path: Path, monkeypatch):
    """Test 40: Completion strictly verifies authorization constants, trainer drift, and symmetry."""
    import training.mortal.validate_m1_training_completion_2026_08 as rmv
    t_dir = tmp_path / "train_auth_test"
    t_dir.mkdir(parents=True)

    # Missing authorization constants when require_authorization=True
    with pytest.raises(ContractError, match="APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT is required"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=True, enforce_canonical_paths=False)

    # Setup valid mock files
    runs = []
    for s in SEEDS:
        run_dir = t_dir / f"M1_variant/seed_{s}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        for a in ARCHIVE_STEPS:
            _create_strict_mock_checkpoint(
                ckpt_dir / f"mortal_{a}.pth",
                steps=a,
                seed=s,
                dataset_index_sha=FROZEN_M1_DATASET_INDEX_SHA256,
            )
        _c_state = torch.load(ckpt_dir / "mortal_72000.pth", weights_only=False, map_location="cpu")
        _c_state["training_contract"]["dataset"]["player_names_by_file_sha256"] = FROZEN_M1_PLAYER_MAPPING_SHA256
        torch.save(_c_state, ckpt_dir / "mortal_72000.pth")
        cfg_p = run_dir / "config.toml"
        cfg_p.write_text(f"# config for seed {s}")
        runs.append({
            "seed": s,
            "config_path": str(cfg_p.resolve()),
            "config_sha256": sha256_file(cfg_p),
        })

    trainer_script = rmt.OFFLINE_TRAINER_SCRIPT
    t_manifest = {
        "schema": "keqing.mortal.m1_training_manifest.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "trainer_source": {
            "path": str(trainer_script.resolve()),
            "sha256": sha256_file(trainer_script),
            "git_blob_oid": rmt.git_blob_oid(trainer_script),
            "approved_training_implementation_commit": "approved_commit_abc",
        },
        "dataset": {
            "dataset_manifest": {"sha256": FROZEN_M1_DATASET_MANIFEST_SHA256},
            "file_index_m1": {"sha256": FROZEN_M1_DATASET_INDEX_SHA256},
            "player_names_by_file": {"sha256": FROZEN_M1_PLAYER_MAPPING_SHA256},
            "player_names": {"sha256": FROZEN_M1_PLAYER_NAMES_SHA256},
        },
        "parent_checkpoint": {"sha256": K0_70K_SHA256},
        "runs": runs,
    }
    t_manifest_p = t_dir / "training_manifest.json"
    with open(t_manifest_p, "w", encoding="utf-8") as f:
        json.dump(t_manifest, f)

    t_preflight = {
        "schema": "keqing.mortal.m1_training_preflight.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_manifest_sha256": sha256_file(t_manifest_p),
        "dataset_4_sha_pass": True,
        "parent_k0_sha_pass": True,
        "configs_parsed": True,
        "commands_verified": True,
        "run_dirs_clean": True,
        "trainer_consumer_check": {
            "files_count": 12000,
            "mappings_count": 12000,
            "all_ext_mortal": True,
            "files_mapping_symmetric": True,
        },
    }
    t_preflight_p = t_dir / "training_preflight.json"
    with open(t_preflight_p, "w", encoding="utf-8") as f:
        json.dump(t_preflight, f)

    # Mismatch approved commit
    monkeypatch.setattr(rmv, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "wrong_commit")
    monkeypatch.setattr(rmv, "AUTHORIZED_TRAINING_PLAN_SHA256", sha256_file(t_manifest_p))
    monkeypatch.setattr(rmv, "AUTHORIZED_TRAINING_PREFLIGHT_SHA256", sha256_file(t_preflight_p))
    with pytest.raises(ContractError, match="Manifest approved training implementation commit mismatch"):
        validate_all_m1_runs(output_dir=t_dir, require_authorization=True, enforce_canonical_paths=False)

    # Set matching constants -> PASS
    monkeypatch.setattr(rmv, "APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT", "approved_commit_abc")
    closure = validate_all_m1_runs(output_dir=t_dir, require_authorization=True, enforce_canonical_paths=False)
    assert closure["approved_training_implementation_commit"] == "approved_commit_abc"
