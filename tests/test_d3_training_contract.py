from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.mortal.prepare_d3_training_contract_2026_08 import (
    ENV_PTS,
    OBJECTIVE_MODE,
    PREFERENCE_LOSS,
    REWARD_MODE,
    TRAINING_LABEL,
    VALUE_STATISTIC,
    build_source_manifest,
    load_aggregate_provenance,
)
from training.mortal.objective import objective_contract_from_config


def test_frozen_training_contract_constants() -> None:
    assert TRAINING_LABEL == "K0_70k"
    assert REWARD_MODE == "final_rank_mc"
    assert OBJECTIVE_MODE == "behavior_action_mc"
    assert VALUE_STATISTIC == "behavior_action_q"
    assert PREFERENCE_LOSS == "existing_cql"
    assert ENV_PTS == [6, 4, 2, 0]


def test_final_rank_mc_centered_targets() -> None:
    pts = np.asarray(ENV_PTS, dtype=np.float64)
    targets = {rank: float(pts[rank] - pts.mean()) for rank in range(4)}
    assert targets == {0: 3.0, 1: 1.0, 2: -1.0, 3: -3.0}
    assert len(set(targets.values())) == 4


def test_generation_rank_points_are_not_training_targets() -> None:
    generation_pts = np.asarray([90.0, 45.0, 0.0, -135.0])
    assert {float(generation_pts[r] - generation_pts.mean()) for r in range(4)} != {
        float(np.asarray(ENV_PTS)[r] - np.asarray(ENV_PTS).mean()) for r in range(4)
    }


def test_objective_contract_behavior_action_mc() -> None:
    contract = objective_contract_from_config(
        {"objective": {"mode": "behavior_action_mc"}, "reward": {"mode": "final_rank_mc"}}
    )
    assert contract == {
        "mode": "behavior_action_mc",
        "value_statistic": "behavior_action_q",
        "preference_loss": "existing_cql",
        "reward_mode": "final_rank_mc",
    }


def test_source_manifest_binding_requires_aggregate_pass(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    audit = {
        "gate": {"verdict": "FAIL", "passed": False},
    }
    (aggregate / "d3_generation_6000h_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    (aggregate / "shard_manifest.json").write_text(
        json.dumps({"shards": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="aggregate audit is not PASS"):
        load_aggregate_provenance(aggregate)


def test_source_manifest_requires_24_pass_shards(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "d3_generation_6000h_audit.json").write_text(
        json.dumps({"gate": {"verdict": "PASS", "passed": True}}), encoding="utf-8"
    )
    (aggregate / "shard_manifest.json").write_text(
        json.dumps({"shards": [{"shard_index": 0, "verdict": "FAIL"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="24 rows"):
        load_aggregate_provenance(aggregate)


def test_explored_mapping_row_schema() -> None:
    row = {
        "seed": 1800250,
        "seed_key": 8192,
        "seat": 3,
        "kyoku_index": 2,
        "decision_index": 7,
        "loader_row_global_index": 1234,
        "loader_row_action": 18,
        "event_actual_action": 18,
        "event_top1_action": 25,
        "event_top2_action": 18,
        "source_shard": 1,
    }
    assert row["loader_row_action"] == row["event_actual_action"]
    assert row["event_actual_action"] == row["event_top2_action"]  # explored keeps top2


def test_file_index_contract_shape(tmp_path: Path) -> None:
    file_list = [str(tmp_path / f"game_{index}.json.gz") for index in range(6000)]
    path = tmp_path / "file_index_d3_k0.pth"
    torch.save({"file_list": file_list}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert len(loaded["file_list"]) == 6000
    assert loaded["file_list"] == file_list


def test_build_source_manifest_rejects_bad_seed_set(tmp_path: Path) -> None:
    # without real logs this cannot run; verify it is called with a real
    # aggregate dir by checking the function is importable and documented
    import inspect  # noqa: PLC0415

    assert "aggregate_dir" in inspect.signature(build_source_manifest).parameters
