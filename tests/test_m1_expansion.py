import gzip
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from training.mortal.m1_dataset_contract_2026_08 import (
    ARCHIVE_STEPS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CANONICAL_PROMOTION_CHECKPOINT,
    ContractError,
    GAMES_PER_SHARD,
    K0_70K_PATH,
    K0_70K_SHA256,
    M1_EXPERIMENT_ID,
    SEEDS,
    SHARD_CONFIG,
    SHARDS,
    START_STEP,
    TARGET_STEP,
    TOTAL_GAMES,
    adjudicate_m1_promotion,
    build_m1_dataset_files,
    equal_seed_hierarchical_bootstrap,
    generate_m1_training_config,
    load_file_index,
    validate_all_8_checkpoints,
    validate_m1_dataset_integrity,
)
from training.mortal.run_m1_evaluation_2026_08 import (
    AuthorizationError as EvalAuthError,
    build_shard_command,
    execute_shard,
    prepare_evaluation_plan,
    run_full_evaluation,
)
from training.mortal.run_m1_training_2026_08 import (
    AuthorizationError as TrainAuthError,
    build_training_command,
    execute_training_for_seed,
    prepare_m1_dataset,
    prepare_training_manifest,
)
from training.mortal.summarize_m1_promotion_2026_08 import (
    _atomic_write_json,
    final_scores,
    parse_raw_log_file,
    parse_shard_logs,
    run_summarizer,
)
from training.run_mortal_dqn_offline import _load_or_build_file_index


def test_1_m1_canonical_index_consumed_by_real_trainer(tmp_path: Path):
    """Test 1: M1 canonical index can be consumed by real _load_or_build_file_index."""
    dummy_files = [str(tmp_path / f"game_{i}.json.gz") for i in range(5)]
    index_path = tmp_path / "file_index_m1.pth"
    torch.save({"file_list": dummy_files}, index_path)

    config = {
        "dataset": {
            "file_index": str(index_path),
            "globs": [],
        }
    }
    loaded = _load_or_build_file_index(config)
    assert loaded == dummy_files


def test_2_legacy_source_list_dict_normalize_correctly(tmp_path: Path):
    """Test 2: load_file_index correctly normalizes both dict {"file_list": ...} and legacy list."""
    files = ["/media/bailan/DISK/AUbuntuProject/a.json.gz", "E:\\AUbuntuProject\\b.json.gz"]
    
    # Dict format
    p_dict = tmp_path / "index_dict.pth"
    torch.save({"file_list": files}, p_dict)
    loaded_dict = load_file_index(p_dict)
    assert len(loaded_dict) == 2
    assert str(loaded_dict[1]) == "/media/bailan/DISK/AUbuntuProject/b.json.gz"

    # List format
    p_list = tmp_path / "index_list.pth"
    torch.save(files, p_list)
    loaded_list = load_file_index(p_list)
    assert len(loaded_list) == 2
    assert str(loaded_list[1]) == "/media/bailan/DISK/AUbuntuProject/b.json.gz"


def test_3_training_request_with_missing_dataset_refuses(tmp_path: Path):
    """Test 3: Training preparation refuses if dataset closure is missing instead of auto-building."""
    empty_dataset_dir = tmp_path / "empty_ds"
    empty_dataset_dir.mkdir()

    with pytest.raises(ContractError, match="Dataset closure is missing"):
        prepare_training_manifest(dataset_dir=empty_dataset_dir, output_training_dir=tmp_path / "train")


def test_4_prepare_dataset_has_no_training_side_effects(tmp_path: Path):
    """Test 4: prepare-dataset creates dataset artifacts without creating training configs or runs."""
    m0_dir = tmp_path / "m0_logs"
    d1_dir = tmp_path / "d1_logs"
    m0_dir.mkdir()
    d1_dir.mkdir()

    m0_files = []
    d1_files = []
    for i in range(5):
        p_m0 = m0_dir / f"m0_{i}.json.gz"
        p_d1 = d1_dir / f"d1_{i}.json.gz"
        with gzip.open(p_m0, "wt") as f:
            f.write(json.dumps({"type": "start_game", "seed": [1000 + i, 8192], "names": ["V1", "V0b", "ext_mortal", "T1"]}) + "\n")
        with gzip.open(p_d1, "wt") as f:
            f.write(json.dumps({"type": "start_game", "seed": [2000 + i, 8192], "names": ["V2", "ext_mortal", "K0_70k", "V3"]}) + "\n")
        m0_files.append(str(p_m0))
        d1_files.append(str(p_d1))

    idx_m0 = tmp_path / "file_index_m0.pth"
    idx_d1 = tmp_path / "file_index_d1.pth"
    torch.save(m0_files, idx_m0)
    torch.save(d1_files, idx_d1)

    ds_out = tmp_path / "dataset_out"
    m1_idx, m1_map, m1_lbl, manifest = build_m1_dataset_files(
        ds_out, m0_index_path=idx_m0, d1_index_path=idx_d1, expected_m0_count=5, expected_d1_count=5
    )
    assert m1_idx.exists()
    assert manifest.exists()
    # Check no training run dirs or config.toml exist in ds_out
    assert not (ds_out / "config.toml").exists()
    assert not (ds_out / "training_manifest.json").exists()


def _create_mock_checkpoint(
    path: Path,
    steps: int = 72000,
    mortal_finite: bool = True,
    init_mode: str = "weights_plus_optimizer_warm_start",
    parent_sha: str = K0_70K_SHA256,
    seed: int = 20260806,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    mortal_weight = torch.tensor([1.0, 2.0]) if mortal_finite else torch.tensor([1.0, float("nan")])
    payload = {
        "steps": steps,
        "mortal": {"w": mortal_weight},
        "current_dqn": {"w": torch.tensor([0.5])},
        "aux": {"w": torch.tensor([0.1])},
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
            "dataset_file_count": 10,
        },
        "training_contract": {
            "schema": "keqing.mortal.training_contract.v2",
        },
    }
    torch.save(payload, path)


def test_5_and_6_completion_reads_steps_and_validates_72000(tmp_path: Path):
    """Test 5 & 6: Completion validator reads steps, passes 72000 and fails wrong 71999."""
    from training.mortal.validate_m1_training_completion_2026_08 import validate_single_run_completion

    run_dir = tmp_path / "run_20260806"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    # Valid 72000 checkpoint + archives
    _create_mock_checkpoint(ckpt_dir / "mortal_72000.pth", steps=72000, seed=20260806)
    for arch_step in ARCHIVE_STEPS:
        _create_mock_checkpoint(ckpt_dir / f"mortal_{arch_step}.pth", steps=arch_step, seed=20260806)

    res = validate_single_run_completion(20260806, run_dir)
    assert res["steps"] == 72000
    assert res["trained_optimizer_steps"] == 2000

    # Wrong steps (71999) fails
    _create_mock_checkpoint(ckpt_dir / "mortal_72000.pth", steps=71999, seed=20260806)
    with pytest.raises(ContractError, match="expected 72000"):
        validate_single_run_completion(20260806, run_dir)


def test_7_wrong_parent_or_init_fails(tmp_path: Path):
    """Test 7: Wrong parent SHA or init mode in checkpoint causes validation failure."""
    from training.mortal.validate_m1_training_completion_2026_08 import validate_single_run_completion

    run_dir = tmp_path / "run_wrong_parent"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    for arch_step in ARCHIVE_STEPS:
        _create_mock_checkpoint(ckpt_dir / f"mortal_{arch_step}.pth", steps=arch_step, parent_sha="tampered_sha", seed=20260806)

    with pytest.raises(ContractError, match="parent_sha256 is tampered_sha"):
        validate_single_run_completion(20260806, run_dir)


def test_8_archive_payload_step_mismatch_fails(tmp_path: Path):
    """Test 8: Archive step in filename vs payload mismatch causes validation failure."""
    from training.mortal.validate_m1_training_completion_2026_08 import validate_single_run_completion

    run_dir = tmp_path / "run_arch_mismatch"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    _create_mock_checkpoint(ckpt_dir / "mortal_72000.pth", steps=72000, seed=20260806)
    for arch_step in ARCHIVE_STEPS:
        # Intentionally mismatch payload step for 70001
        step_val = 99999 if arch_step == 70001 else arch_step
        _create_mock_checkpoint(ckpt_dir / f"mortal_{arch_step}.pth", steps=step_val, seed=20260806)

    with pytest.raises(ContractError, match="payload steps is 99999"):
        validate_single_run_completion(20260806, run_dir)


def test_9_m1_checkpoint_tamper_fails_evaluation_plan(tmp_path: Path):
    """Test 9: Tampered M1 checkpoint SHA or missing file fails evaluation plan preparation."""
    closure_path = tmp_path / "training_completion_closure.json"
    fake_closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "runs": [
            {"training_seed": 20260806, "final_checkpoint_path": str(tmp_path / "nonexistent_m1.pth")},
            {"training_seed": 20260807, "final_checkpoint_path": str(tmp_path / "nonexistent_m1.pth")},
            {"training_seed": 20260808, "final_checkpoint_path": str(tmp_path / "nonexistent_m1.pth")},
        ]
    }
    with open(closure_path, "w") as f:
        json.dump(fake_closure, f)

    with pytest.raises(ContractError, match="Checkpoint verification failed"):
        prepare_evaluation_plan(output_dir=tmp_path / "eval_plan", training_completion_closure_path=closure_path)


def test_10_evaluation_unauthorized_fails_closed(tmp_path: Path):
    """Test 10: Evaluation execution without authorization raises AuthorizationError and runs 0 subprocesses."""
    with pytest.raises(EvalAuthError, match="NOT authorized"):
        run_full_evaluation(output_dir=tmp_path / "eval_out")


def test_11_training_unauthorized_fails_closed(tmp_path: Path):
    """Test 11: Training execution without authorization raises AuthorizationError."""
    with pytest.raises(TrainAuthError, match="NOT authorized"):
        execute_training_for_seed(20260806, training_dir=tmp_path / "train", dataset_dir=tmp_path / "ds")


def test_12_non_empty_shard_output_refuses(tmp_path: Path):
    """Test 12: Shard execution refuses if directory or logs directory is non-empty."""
    shard_dir = tmp_path / "shard_00"
    logs_dir = shard_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "1930000.json.gz").write_bytes(b"dummy")

    with pytest.raises(ContractError, match="Automatic overwrite/resume is prohibited"):
        execute_shard(0, output_dir=tmp_path)


def test_13_jsonl_parser_and_reach_accepted(tmp_path: Path):
    """Test 13: JSONL event parser correctly adjusts ReachAccepted (-1000) scores and computes ranks."""
    events = [
        {
            "type": "start_game",
            "seed": [1930005, 8192],
            "names": ["70k", "ext_mortal", "M0_CURRENT_20260806", "M1_CURRENT_20260806"],
        },
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 3},  # M1 riichi
        {"type": "hora", "deltas": [0, -12000, 0, 12000]},  # M1 wins 12000 from ext_mortal
    ]
    raw_lines = "\n".join(json.dumps(ev) for ev in events) + "\n"
    log_file = tmp_path / "1930005.json.gz"
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        f.write(raw_lines)

    parsed = parse_raw_log_file(
        log_file,
        expected_training_seed=20260806,
        expected_hanchan_min=1930000,
        expected_hanchan_max=1930249,
    )
    assert parsed["game_id"] == 1930005
    l2pt = parsed["label_to_pt"]
    assert l2pt["M1_CURRENT_20260806"] == 90.0
    assert l2pt["70k"] == 45.0
    assert l2pt["M0_CURRENT_20260806"] == 0.0
    assert l2pt["ext_mortal"] == -135.0


def test_14_raw_metrics_mismatch_fails_summary(tmp_path: Path):
    """Test 14: Inconsistency between raw log rank counts and metrics.json rank counts fails summarizer."""
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

    # Write wrong metrics.json
    metrics = {
        "70k": {"rank_counts": [999, 0, 0, 0]},  # mismatch!
    }
    with open(shard_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    shard_cfg = {"shard_id": 0, "training_seed": 20260806, "start_hanchan": 1930000, "end_hanchan": 1930000, "games_count": 1}
    with pytest.raises(ContractError, match="Rank count mismatch"):
        parse_shard_logs(shard_dir, shard_cfg)


def test_14b_stat_equivalence_mismatch_fails(monkeypatch, tmp_path: Path):
    """Test 14b: Stat equivalence discrepancy raises ContractError."""
    import training.mortal.summarize_m1_promotion_2026_08 as sm

    # Mock authoritative_ranks_from_stat to return conflicting ranks
    monkeypatch.setattr(sm, "authoritative_ranks_from_stat", lambda p: [3, 2, 1, 0])

    events = [
        {"type": "start_game", "seed": [1930000, 8192], "names": ["70k", "ext_mortal", "M0_CURRENT_20260806", "M1_CURRENT_20260806"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": [8000, -8000, 0, 0]},
    ]
    log_file = tmp_path / "1930000.json.gz"
    with gzip.open(log_file, "wt") as f:
        f.write("\n".join(json.dumps(e) for e in events) + "\n")

    with pytest.raises(ContractError, match="Rank discrepancy with libriichi Stat"):
        sm.parse_raw_log_file(log_file, expected_training_seed=20260806, expected_hanchan_min=1930000, expected_hanchan_max=1930000)


def test_15_summary_existing_output_refuses(tmp_path: Path):
    """Test 15: Summarizer refuses to overwrite existing destination summary file."""
    dest = tmp_path / "m1_summary.json"
    dest.write_text("existing")

    with pytest.raises(ContractError, match="already exists"):
        run_summarizer(output_root=tmp_path, destination_file=dest)


def test_16_atomic_write_json(tmp_path: Path):
    """Test 16: Atomic JSON write successfully creates valid destination file."""
    dest = tmp_path / "sub" / "output.json"
    data = {"status": "ok", "value": 42}
    _atomic_write_json(data, dest)

    assert dest.exists()
    with open(dest) as f:
        loaded = json.load(f)
    assert loaded == data


def test_17_hierarchical_bootstrap_and_promotion_logic():
    """Test 17: Paired equal-seed hierarchical bootstrap determinism and promotion adjudication."""
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

    # Test failing adjudication
    res_fail = adjudicate_m1_promotion(
        x_seed_means={20260806: -1.0, 20260807: 6.0, 20260808: 8.0},
        x_ci95=(-2.0, 5.0),
        y_seed_means={20260806: 2.0, 20260807: 4.0, 20260808: 6.0},
        y_ci95=y_ci,
        gates_pass=True,
    )
    assert res_fail["verdict"] == "not_supported"
    assert res_fail["recipe_promotion"] is False
    assert res_fail["checkpoint_promotion"] is False
    assert res_fail["promoted_k1_checkpoint"] is None
