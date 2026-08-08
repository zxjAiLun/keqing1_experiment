#!/usr/bin/env python3
"""Run deterministic GRP adapter parity and reward-contract tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON_ROOT = MORTAL_ROOT / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_ROOT))
if str(MORTAL_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="one native json.gz log; defaults to the first ext_mortal training log",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _write_config(path: Path, grp_state: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[control]",
                "version = 4",
                "",
                "[env]",
                "pts = [6.0, 4.0, 2.0, 0.0]",
                "",
                "[reward]",
                'mode = "grp"',
                "",
                "[grp]",
                f"state_file = {json.dumps(str(grp_state))}",
                "",
                "[grp.network]",
                "hidden_size = 64",
                "num_layers = 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _collect(dataset: Any) -> list[list[Any]]:
    return list(iter(dataset))


def _assert_identity(left: list[list[Any]], right: list[list[Any]]) -> None:
    assert len(left) == len(right), (len(left), len(right))
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        for column in (0, 1, 2, 3, 5):
            if isinstance(left_row[column], np.ndarray):
                assert np.array_equal(left_row[column], right_row[column]), (index, column)
            else:
                assert left_row[column] == right_row[column], (index, column)


def main() -> None:
    args = _parse_args()
    log_path = args.log
    if log_path is None:
        candidates = sorted(
            Path(REPO_ROOT / "artifacts/experiments/model_pool_2026_07/V2_data")
            .glob("**/logs/*.json.gz")
        )
        if not candidates:
            raise SystemExit("no training log found; pass --log explicitly")
        log_path = candidates[0]
    log_path = log_path.resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    with tempfile.TemporaryDirectory(prefix="keqing_reward_test_") as temp_dir:
        temp_root = Path(temp_dir)
        grp_state_path = temp_root / "grp_fixture.pth"
        config_path = temp_root / "config.toml"

        os.environ["MORTAL_CFG"] = str(config_path)
        _write_config(config_path, grp_state_path)
        from config import config as mortal_config  # noqa: PLC0415
        from model import GRP  # noqa: PLC0415
        import dataloader as upstream_dataloader  # noqa: PLC0415
        from libriichi.dataset import GameplayLoader  # noqa: PLC0415
        from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: PLC0415
        from reward_calculator import RewardCalculator  # noqa: PLC0415

        torch.manual_seed(12345)
        grp = GRP(hidden_size=64, num_layers=2)
        torch.save({"model": grp.state_dict()}, grp_state_path)
        mortal_config.clear()
        mortal_config.update(
            {
                "control": {"version": 4},
                "env": {"pts": [6.0, 4.0, 2.0, 0.0]},
                "reward": {"mode": "grp"},
                "grp": {
                    "state_file": str(grp_state_path),
                    "network": {"hidden_size": 64, "num_layers": 2},
                },
            }
        )

        file_list = [str(log_path)]
        common = {
            "version": 4,
            "file_list": file_list,
            "pts": [6.0, 4.0, 2.0, 0.0],
            "file_batch_size": 1,
            "player_names": ["ext_mortal"],
            "num_epochs": 1,
        }
        project_iter = FileDatasetsIter(**common)
        upstream_iter = upstream_dataloader.FileDatasetsIter(**common)

        random.seed(777)
        project_grp = _collect(project_iter)
        random.seed(777)
        upstream_grp = _collect(upstream_iter)
        _assert_identity(project_grp, upstream_grp)
        assert len(project_grp) > 0
        assert np.array_equal(
            np.asarray([row[4] for row in project_grp]),
            np.asarray([row[4] for row in upstream_grp]),
        )

        grp_model = GRP(hidden_size=64, num_layers=2)
        grp_model.load_state_dict(torch.load(grp_state_path, weights_only=True)["model"])
        reward_calc = RewardCalculator(grp_model, [6.0, 4.0, 2.0, 0.0])
        loader = GameplayLoader(version=4, player_names=["ext_mortal"])
        game_file = loader.load_gz_log_files(file_list)[0][0]
        grp_data = game_file.take_grp()
        grp_feature = grp_data.take_feature()
        rank_by_player = grp_data.take_rank_by_player()
        player_id = int(game_file.take_player_id())
        rewards = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
        initial_probs = reward_calc.calc_rank_prob(player_id, grp_feature, rank_by_player)[0]
        expected_total = float(
            reward_calc.pts[int(rank_by_player[player_id])] - (initial_probs @ reward_calc.pts)
        )
        assert np.isclose(float(np.sum(rewards)), expected_total, atol=1e-10)

        mortal_config["reward"]["mode"] = "final_rank_mc"
        project_mc_iter = FileDatasetsIter(**common)
        random.seed(777)
        project_mc = _collect(project_mc_iter)
        _assert_identity(project_grp, project_mc)
        assert any(not np.isclose(left[4], right[4]) for left, right in zip(project_grp, project_mc, strict=True))

        grp_values = np.asarray([row[4] for row in project_grp], dtype=np.float64)
        mc_values = np.asarray([row[4] for row in project_mc], dtype=np.float64)
        report = {
            "schema": "keqing.mortal.reward_adapter_test.v1",
            "log": str(log_path),
            "samples": len(project_grp),
            "upstream_project_parity": True,
            "data_identity_after_reward_switch": True,
            "grp_reward": {
                "mean": float(grp_values.mean()),
                "std": float(grp_values.std()),
                "min": float(grp_values.min()),
                "max": float(grp_values.max()),
                "nonzero_rate": float(np.mean(grp_values != 0)),
                "telescoping_sum": float(np.sum(rewards)),
                "telescoping_expected": expected_total,
            },
            "final_rank_mc_reward": {
                "mean": float(mc_values.mean()),
                "std": float(mc_values.std()),
                "min": float(mc_values.min()),
                "max": float(mc_values.max()),
                "nonzero_rate": float(np.mean(mc_values != 0)),
            },
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
