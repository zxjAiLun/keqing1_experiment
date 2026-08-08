#!/usr/bin/env python3
"""Scan a Mortal dataset and report the configured reward distribution."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
import random
import sys
import tomllib

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON_ROOT = MORTAL_ROOT / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_ROOT))
if str(MORTAL_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260718)
    parser.add_argument("--max-files", type=int, default=0, help="optional bounded smoke limit")
    return parser.parse_args()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    quantile_values = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "q01": float(quantile_values[0]),
        "q05": float(quantile_values[1]),
        "q50": float(quantile_values[2]),
        "q95": float(quantile_values[3]),
        "q99": float(quantile_values[4]),
    }


def _grp_rewards_by_game(config_data: dict, file_list: list[str], player_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from model import GRP  # noqa: PLC0415
    from libriichi.dataset import Grp  # noqa: PLC0415
    from reward_calculator import RewardCalculator  # noqa: PLC0415
    import torch  # noqa: PLC0415

    grp_config = config_data["grp"]
    grp = GRP(**grp_config["network"])
    state = torch.load(grp_config["state_file"], weights_only=False, map_location=torch.device("cpu"))
    grp.load_state_dict(state["model"])
    reward_calc = RewardCalculator(
        grp,
        [float(value) for value in config_data["env"]["pts"]],
        uniform_init=bool(grp_config.get("uniform_init", False)),
    )
    all_rewards: list[float] = []
    game_abs_sums: list[float] = []
    for index, file_path in enumerate(file_list, start=1):
        with gzip.open(file_path, "rt", encoding="utf-8") as handle:
            start_game = json.loads(next(handle))
        names = start_game.get("names", [])
        target_ids = [idx for idx, name in enumerate(names) if name in player_names]
        if len(target_ids) != 1:
            raise ValueError(f"expected one target player in {file_path}: names={names!r}")
        game = Grp.load_gz_log_files([file_path])[0]
        feature = game.take_feature()
        rewards = np.asarray(
            reward_calc.calc_delta_pt(target_ids[0], feature, game.take_rank_by_player()),
            dtype=np.float64,
        )
        all_rewards.extend(float(value) for value in rewards)
        game_abs_sums.append(float(np.abs(rewards).sum()))
        if index % 100 == 0:
            logging.info("scanned games=%s/%s", index, len(file_list))
    return np.asarray(all_rewards, dtype=np.float64), np.asarray(game_abs_sums, dtype=np.float64)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    import os

    os.environ["MORTAL_CFG"] = str(config_path)
    with config_path.open("rb") as handle:
        config_data = tomllib.load(handle)

    from training.mortal.mainline_dataloader import (  # noqa: PLC0415
        FileDatasetsIter,
        reward_contract_from_config,
    )
    from training.run_mortal_dqn_offline import (  # noqa: PLC0415
        _load_or_build_file_index,
        _load_player_names,
    )

    file_list = _load_or_build_file_index(config_data)
    if args.max_files > 0:
        file_list = file_list[: args.max_files]
    player_names = _load_player_names(config_data)
    random.seed(int(args.data_seed))
    dataset_config = config_data["dataset"]
    dataset = FileDatasetsIter(
        version=int(config_data["control"]["version"]),
        file_list=list(file_list),
        pts=config_data["env"]["pts"],
        file_batch_size=int(dataset_config["file_batch_size"]),
        reserve_ratio=float(dataset_config["reserve_ratio"]),
        player_names=player_names,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
    )

    reward_mode = str(config_data.get("reward", {}).get("mode", "final_rank_mc"))
    if reward_mode in {"grp", "mortal_grp_delta_pt"}:
        values, game_abs_sums = _grp_rewards_by_game(config_data, file_list, player_names)
    else:
        rewards: list[float] = []
        for entry in dataset:
            rewards.append(float(entry[4]))
        values = np.asarray(rewards, dtype=np.float64)
        game_abs_sums = np.asarray([], dtype=np.float64)
    if not len(values):
        raise RuntimeError("reward preflight produced zero samples")
    samples = len(values)
    contract = reward_contract_from_config(config_data)
    report = {
        "schema": "keqing.mortal.reward_distribution_preflight.v2",
        "config": str(config_path),
        "reward_contract": contract,
        "data_seed": int(args.data_seed),
        "file_count": len(file_list),
        "player_names": sorted(player_names),
        "decision_samples": int(samples),
        "reward": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "nonzero_rate": float(np.mean(values != 0)),
            "quantiles": _quantiles(values),
            "abs_delta_pt": {
                "quantiles": _quantiles(np.abs(values)),
                "max": float(np.abs(values).max()),
            },
        },
        "per_hanchan_abs_delta_pt": {
            "hanchans": int(len(game_abs_sums)),
            "quantiles": _quantiles(game_abs_sums),
            "max": float(game_abs_sums.max()) if len(game_abs_sums) else None,
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
