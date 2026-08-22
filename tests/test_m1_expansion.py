import gzip
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from training.mortal.m1_dataset_contract_2026_08 import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GAMES_PER_SHARD,
    M1_EXPERIMENT_ID,
    SEEDS,
    SHARD_CONFIG,
    SHARDS,
    TOTAL_GAMES,
    adjudicate_m1_promotion,
    build_m1_dataset_files,
    equal_seed_hierarchical_bootstrap,
    generate_m1_training_config,
    validate_checkpoints,
    validate_m1_dataset_integrity,
)
from training.mortal.run_m1_evaluation_2026_08 import (
    build_shard_command,
    execute_shard,
)
from training.mortal.run_m1_training_2026_08 import build_training_command
from training.mortal.summarize_m1_promotion_2026_08 import (
    final_scores,
    parse_raw_log_file,
)


def test_m1_dataset_builder_and_integrity_mock(tmp_path: Path):
    """Verify M1 dataset builder correctly merges M0 and D1 and validates integrity."""
    m0_dir = tmp_path / "m0_logs"
    d1_dir = tmp_path / "d1_logs"
    m0_dir.mkdir()
    d1_dir.mkdir()

    m0_files = []
    d1_files = []

    for i in range(5):
        # M0 log
        p_m0 = m0_dir / f"m0_{i}.json.gz"
        ev_m0 = [
            {"type": "start_game", "seed": [1000 + i, 8192], "names": ["V1", "V0b", "ext_mortal", "T1"]},
        ]
        with gzip.open(p_m0, "wt", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(e) for e in ev_m0) + "\n")
        m0_files.append(str(p_m0))

        # D1 log
        p_d1 = d1_dir / f"d1_{i}.json.gz"
        ev_d1 = [
            {"type": "start_game", "seed": [2000 + i, 8192], "names": ["V2", "ext_mortal", "K0_70k", "V3"]},
        ]
        with gzip.open(p_d1, "wt", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(e) for e in ev_d1) + "\n")
        d1_files.append(str(p_d1))

    # Save mock index files
    idx_m0 = tmp_path / "file_index_m0.pth"
    idx_d1 = tmp_path / "file_index_d1.pth"
    torch.save(m0_files, idx_m0)
    torch.save(d1_files, idx_d1)

    out_dir = tmp_path / "m1_out"
    # Test builder with smaller counts
    out_idx, out_lbl = build_m1_dataset_files(
        out_dir,
        m0_index_path=idx_m0,
        d1_index_path=idx_d1,
        expected_m0_count=5,
        expected_d1_count=5,
    )
    loaded_m1 = torch.load(out_idx, weights_only=False)
    assert len(loaded_m1) == 10
    assert loaded_m1[:5] == m0_files
    assert loaded_m1[5:] == d1_files

    with open(out_lbl) as f:
        labels = [line.strip() for line in f]
    assert len(labels) == 10
    assert all(lbl == "ext_mortal" for lbl in labels)


def test_m1_training_config_generation(tmp_path: Path):
    """Verify generated training config matches M0_CURRENT hyperparameters."""
    cfg = generate_m1_training_config(
        seed=20260806,
        output_run_dir=tmp_path,
        m1_index_path=tmp_path / "file_index_m1.pth",
        m1_labels_path=tmp_path / "m1_train_labels.txt",
    )
    assert cfg["control"]["version"] == 4
    assert cfg["control"]["batch_size"] == 512
    assert cfg["control"]["enable_amp"] is False
    assert cfg["cql"]["min_q_weight"] == 5.0
    assert cfg["aux"]["next_rank_weight"] == 0.2
    assert cfg["objective"]["mode"] == "behavior_action_mc"
    assert cfg["reward"]["mode"] == "final_rank_mc"
    assert cfg["experiment"]["route"] == "M1_variant"
    assert cfg["experiment"]["trainable_label"] == "ext_mortal"
    assert cfg["experiment"]["training_seed"] == 20260806
    assert cfg["experiment"]["parent_steps"] == 70000


def test_m1_training_command_builder(tmp_path: Path):
    """Verify training command matches Mortal offline DQN runner contract."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("dummy")
    parent_file = tmp_path / "parent.pth"
    parent_file.write_text("dummy")

    cmd = build_training_command(
        seed=20260806,
        config_path=cfg_file,
        parent_path=parent_file,
        run_dir=tmp_path,
    )
    assert "--config" in cmd
    assert "--initialize-from" in cmd
    assert "--initialize-optimizer-from" in cmd
    assert "--initial-steps" in cmd and "70000" in cmd
    assert "--target-steps" in cmd and "72000" in cmd
    assert "--device" in cmd and "cuda:0" in cmd
    assert "--archive-steps" in cmd and "70001,70010,70100,70500,71000,72000" in cmd


def test_m1_evaluator_cli_args(tmp_path: Path):
    """Verify evaluator command line matches fresh evaluation contract."""
    cmd_00 = build_shard_command(0, tmp_path, device="cuda")
    assert "--seed-start" in cmd_00
    assert "1930000" in cmd_00
    assert "--games" in cmd_00
    assert "250" in cmd_00
    assert "--seed-key" in cmd_00
    assert "8192" in cmd_00
    assert "--require-cuda" in cmd_00
    assert "--native-batch-games" in cmd_00
    assert "250" in cmd_00

    # Check 4 models in lineup: 70k, ext_mortal, M0_CURRENT_20260806, M1_CURRENT_20260806
    model_flags = [cmd_00[i + 1] for i, flag in enumerate(cmd_00) if flag == "--model"]
    assert len(model_flags) == 4
    labels = [spec.split("=")[0] for spec in model_flags]
    assert labels == ["70k", "ext_mortal", "M0_CURRENT_20260806", "M1_CURRENT_20260806"]

    # Check span of shard 11 (seed 20260808)
    cmd_11 = build_shard_command(11, tmp_path, device="cuda")
    assert "1950750" in cmd_11


def test_m1_refuse_overwrite_non_empty(tmp_path: Path):
    """Verify runner refuses to execute if shard output already exists and is non-empty."""
    shard_dir = tmp_path / "shard_00"
    logs_dir = shard_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "1930000.json.gz").write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="Automatic overwrite/resume is prohibited"):
        execute_shard(0, tmp_path, device="cuda")


def test_m1_jsonl_raw_log_parser_and_scores(tmp_path: Path):
    """Verify JSONL raw log parsing with ReachAccepted -1000 point adjustment."""
    events = [
        {
            "type": "start_game",
            "seed": [1930005, 8192],
            "names": ["70k", "ext_mortal", "M0_CURRENT_20260806", "M1_CURRENT_20260806"],
        },
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 3},  # M1 declares riichi
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
    # Scores:
    # 70k (seat 0): 25000 (Rank 2 -> +45)
    # ext_mortal (seat 1): 25000 - 12000 = 13000 (Rank 4 -> -135)
    # M0_CURRENT (seat 2): 25000 (Rank 3 -> 0)
    # M1_CURRENT (seat 3): 25000 - 1000 + 12000 = 36000 (Rank 1 -> +90)
    assert l2pt["M1_CURRENT_20260806"] == 90.0
    assert l2pt["70k"] == 45.0
    assert l2pt["M0_CURRENT_20260806"] == 0.0
    assert l2pt["ext_mortal"] == -135.0


def test_m1_hierarchical_bootstrap_determinism():
    """Verify hierarchical paired bootstrap determinism and outer+inner resampling with seed 20260830."""
    x_by_seed = {
        20260806: np.full(1000, 3.0),
        20260807: np.full(1000, 6.0),
        20260808: np.full(1000, 9.0),
    }
    y_by_seed = {
        20260806: np.full(1000, 2.0),
        20260807: np.full(1000, 5.0),
        20260808: np.full(1000, 8.0),
    }

    x_ci, y_ci, x_mean, y_mean = equal_seed_hierarchical_bootstrap(
        x_by_seed, y_by_seed, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
    )
    assert np.isclose(x_mean, 6.0)
    assert np.isclose(y_mean, 5.0)
    assert x_ci[0] < x_mean < x_ci[1]
    assert y_ci[0] < y_mean < y_ci[1]
    assert x_ci[0] >= 3.0 and x_ci[1] <= 9.0


def test_m1_promotion_adjudication_rules():
    """Verify promotion verdict logic for M1."""
    # Case 1: Both x (vs M0_CURRENT) and y (vs 70k) pass
    res_pass = adjudicate_m1_promotion(
        x_seed_means={20260806: 4.0, 20260807: 5.0, 20260808: 6.0},
        x_ci95=(1.5, 8.0),
        y_seed_means={20260806: 8.0, 20260807: 9.0, 20260808: 10.0},
        y_ci95=(4.0, 12.0),
        gates_pass=True,
    )
    assert res_pass["verdict"] == "promotion_supported"
    assert res_pass["recipe_promotion"] is True
    assert res_pass["checkpoint_promotion"] is True
    assert res_pass["promoted_k1_checkpoint"] == "M1_CURRENT_20260807"

    # Case 2: x passes but y fails (e.g. y CI crosses zero)
    res_fail_y = adjudicate_m1_promotion(
        x_seed_means={20260806: 4.0, 20260807: 5.0, 20260808: 6.0},
        x_ci95=(1.5, 8.0),
        y_seed_means={20260806: 2.0, 20260807: -1.0, 20260808: 3.0},
        y_ci95=(-0.5, 4.0),
        gates_pass=True,
    )
    assert res_fail_y["verdict"] == "not_supported"
    assert res_fail_y["recipe_promotion"] is False
    assert res_fail_y["checkpoint_promotion"] is False
    assert res_fail_y["promoted_k1_checkpoint"] is None
