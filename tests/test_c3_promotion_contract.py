import gzip
import json
from pathlib import Path
import numpy as np
import pytest

from training.mortal.c3_evaluation_contract_2026_08 import (
    C3_EXPERIMENT_ID,
    SEEDS,
    SHARDS,
    GAMES_PER_SHARD,
    TOTAL_GAMES,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    SHARD_CONFIG,
    adjudicate_c3_promotion,
    equal_seed_hierarchical_bootstrap,
    validate_checkpoints,
)
from training.mortal.run_c3_evaluation_2026_08 import (
    build_shard_command,
    execute_shard,
)
from training.mortal.summarize_c3_promotion_2026_08 import (
    final_scores,
    parse_raw_log_file,
    parse_shard_logs,
)


def test_c3_reach_accepted_regression():
    """Verify ReachAccepted -1000 score adjustment regression test."""
    events = [
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 0},
        {"type": "hora", "deltas": [12000, -12000, 0, 0]},
    ]
    scores = final_scores(events)
    assert scores is not None
    # Player 0: 25000 - 1000 (riichi) + 12000 = 36000
    # Player 1: 25000 - 12000 = 13000
    assert scores[0] == 36000.0
    assert scores[1] == 13000.0
    assert scores[2] == 25000.0
    assert scores[3] == 25000.0


def test_c3_evaluator_cli_args(tmp_path: Path):
    """Verify evaluator command line arguments match four_player_native contract."""
    cmd = build_shard_command(0, tmp_path, device="cuda")
    assert "--seed-start" in cmd
    assert "1900000" in cmd
    assert "--games" in cmd
    assert "250" in cmd
    assert "--require-cuda" in cmd
    assert "--native-batch-games" in cmd
    assert "250" in cmd
    assert "--rank-points-profile" in cmd
    assert "tenhou_reference" in cmd
    # Check 4 models present
    model_flags = [cmd[i+1] for i, flag in enumerate(cmd) if flag == "--model"]
    assert len(model_flags) == 4
    labels = [spec.split("=")[0] for spec in model_flags]
    assert labels == ["70k", "ext_mortal", "M0_CURRENT_20260806", "D1_CQL_OFF_20260806"]


def test_c3_refuse_overwrite_non_empty(tmp_path: Path):
    """Verify runner refuses to execute if shard directory or logs already exist and are non-empty."""
    shard_dir = tmp_path / "shard_00"
    logs_dir = shard_dir / "logs"
    logs_dir.mkdir(parents=True)
    dummy_log = logs_dir / "1900000.json.gz"
    dummy_log.write_bytes(b"dummy")

    with pytest.raises(RuntimeError, match="Automatic overwrite/resume is prohibited"):
        execute_shard(0, tmp_path, device="cuda")


def test_c3_jsonl_raw_log_parser(tmp_path: Path):
    """Verify JSONL event stream parsing with seed_key, lineup labels and ReachAccepted."""
    events = [
        {
            "type": "start_game",
            "seed": [1900005, 8192],
            "names": ["70k", "ext_mortal", "M0_CURRENT_20260806", "D1_CQL_OFF_20260806"],
        },
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 3},  # D1_CQL_OFF declares riichi
        {"type": "hora", "deltas": [-8000, 0, 0, 8000]},  # D1_CQL_OFF wins 8000 from 70k
    ]
    # Write as JSONL gzip
    raw_lines = "\n".join(json.dumps(ev) for ev in events) + "\n"
    log_file = tmp_path / "1900005.json.gz"
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        f.write(raw_lines)

    parsed = parse_raw_log_file(
        log_file,
        expected_training_seed=20260806,
        expected_hanchan_min=1900000,
        expected_hanchan_max=1900249,
    )
    assert parsed["game_id"] == 1900005
    l2pt = parsed["label_to_pt"]
    # Final scores:
    # 70k (seat 0): 25000 - 8000 = 17000 (Rank 4 -> -135)
    # ext_mortal (seat 1): 25000 (Rank 2 -> +45)
    # M0_CURRENT (seat 2): 25000 (Rank 3 -> 0)
    # D1_CQL_OFF (seat 3): 25000 - 1000 + 8000 = 32000 (Rank 1 -> +90)
    assert l2pt["D1_CQL_OFF_20260806"] == 90.0
    assert l2pt["70k"] == -135.0
    assert l2pt["ext_mortal"] == 45.0
    assert l2pt["M0_CURRENT_20260806"] == 0.0


def test_c3_shard_configuration():
    """Verify shard index and range bounds for 3000 total games across 12 shards."""
    assert len(SHARDS) == 12
    assert TOTAL_GAMES == 3000
    assert GAMES_PER_SHARD == 250

    for s in SEEDS:
        seed_shards = [cfg for cfg in SHARD_CONFIG if cfg["training_seed"] == s]
        assert len(seed_shards) == 4
        starts = [cfg["start_hanchan"] for cfg in seed_shards]
        ends = [cfg["end_hanchan"] for cfg in seed_shards]
        assert starts == sorted(starts)
        assert ends[-1] - starts[0] + 1 == 1000


def test_c3_adjudication_rules():
    """Verify promotion verdict logic on positive and failed stats."""
    # Case 1: Both x and y pass all 3 seeds and CI lower > 0
    res_pass = adjudicate_c3_promotion(
        x_seed_means={20260806: 5.0, 20260807: 4.0, 20260808: 6.0},
        x_ci95=(1.2, 8.5),
        y_seed_means={20260806: 3.0, 20260807: 2.0, 20260808: 4.0},
        y_ci95=(0.5, 6.0),
        gates_pass=True,
    )
    assert res_pass["verdict"] == "promotion_supported"
    assert res_pass["recipe_promotion"] is True
    assert res_pass["checkpoint_promotion"] is True
    assert res_pass["promoted_k1_checkpoint"] == "D1_CQL_OFF_20260807"

    # Case 2: One seed mean negative on x
    res_fail_x_seed = adjudicate_c3_promotion(
        x_seed_means={20260806: -0.5, 20260807: 4.0, 20260808: 6.0},
        x_ci95=(0.2, 8.5),
        y_seed_means={20260806: 3.0, 20260807: 2.0, 20260808: 4.0},
        y_ci95=(0.5, 6.0),
        gates_pass=True,
    )
    assert res_fail_x_seed["verdict"] == "not_supported"
    assert res_fail_x_seed["recipe_promotion"] is False
    assert res_fail_x_seed["checkpoint_promotion"] is False
    assert res_fail_x_seed["promoted_k1_checkpoint"] is None

    # Case 3: y CI lower <= 0
    res_fail_y_ci = adjudicate_c3_promotion(
        x_seed_means={20260806: 5.0, 20260807: 4.0, 20260808: 6.0},
        x_ci95=(1.2, 8.5),
        y_seed_means={20260806: 3.0, 20260807: 2.0, 20260808: 4.0},
        y_ci95=(-0.1, 6.0),
        gates_pass=True,
    )
    assert res_fail_y_ci["verdict"] == "not_supported"
    assert res_fail_y_ci["recipe_promotion"] is False

    # Case 4: Gates fail
    res_invalid = adjudicate_c3_promotion(
        x_seed_means={20260806: 5.0, 20260807: 4.0, 20260808: 6.0},
        x_ci95=(1.2, 8.5),
        y_seed_means={20260806: 3.0, 20260807: 2.0, 20260808: 4.0},
        y_ci95=(0.5, 6.0),
        gates_pass=False,
    )
    assert res_invalid["verdict"] == "invalid"
    assert res_invalid["recipe_promotion"] is False
    assert res_invalid["promoted_k1_checkpoint"] is None


def test_outer_and_inner_hierarchical_bootstrap():
    """Verify outer seed resampling and inner paired game resampling."""
    # Generate distinct distributions for 3 seeds
    x_by_seed = {
        20260806: np.full(1000, 2.0),
        20260807: np.full(1000, 4.0),
        20260808: np.full(1000, 6.0),
    }
    y_by_seed = {
        20260806: np.full(1000, 1.0),
        20260807: np.full(1000, 3.0),
        20260808: np.full(1000, 5.0),
    }

    x_ci, y_ci, x_mean, y_mean = equal_seed_hierarchical_bootstrap(
        x_by_seed, y_by_seed, reps=5000, seed=BOOTSTRAP_SEED
    )
    # Mean of means is exactly (2+4+6)/3 = 4.0 for x, (1+3+5)/3 = 3.0 for y
    assert np.isclose(x_mean, 4.0)
    assert np.isclose(y_mean, 3.0)
    # Because outer seed resampling picks seeds with replacement {2,4,6}, bootstrap draws vary between 2.0 and 6.0
    assert x_ci[0] < x_mean < x_ci[1]
    assert y_ci[0] < y_mean < y_ci[1]
    # For x: min possible outer mean is 2.0, max is 6.0
    assert x_ci[0] >= 2.0 and x_ci[1] <= 6.0
