#!/usr/bin/env python3
"""Project-owned Mortal dataset iterator and reward adapters."""

from __future__ import annotations

import random
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import IterableDataset

from config import config
from libriichi.dataset import GameplayLoader


SUPPORTED_REWARD_MODES = {"final_rank_mc", "terminal_rank", "mortal_grp_delta_pt"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reward_contract_from_config(config_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a reproducible description of the configured reward source."""

    reward_mode = str(config_data.get("reward", {}).get("mode", "final_rank_mc"))
    if reward_mode == "grp":
        reward_mode = "mortal_grp_delta_pt"
    if reward_mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"unsupported project reward mode: {reward_mode}")

    contract: dict[str, Any] = {
        "mode": reward_mode,
        "rank_pts": [float(value) for value in config_data["env"]["pts"]],
    }
    if reward_mode == "mortal_grp_delta_pt":
        grp_config = config_data.get("grp")
        if not isinstance(grp_config, Mapping):
            raise ValueError("mortal_grp_delta_pt requires a [grp] config table")
        state_file = Path(str(grp_config["state_file"])).resolve()
        if not state_file.exists():
            raise FileNotFoundError(f"GRP checkpoint does not exist: {state_file}")
        contract["grp"] = {
            "state_file": str(state_file),
            "sha256": _sha256_file(state_file),
            "network": {
                "hidden_size": int(grp_config["network"]["hidden_size"]),
                "num_layers": int(grp_config["network"]["num_layers"]),
            },
            "uniform_init": bool(grp_config.get("uniform_init", False)),
        }
    return contract


def _build_reward_adapter(reward_mode: str, pts: np.ndarray, config_data: Mapping[str, Any]):
    if reward_mode != "mortal_grp_delta_pt":
        return None

    # These imports must stay inside build_iter: on Windows, DataLoader worker
    # initialization happens in a fresh process and Mortal's config module must
    # already be on sys.path before GRP is imported.
    from model import GRP  # noqa: PLC0415
    from reward_calculator import RewardCalculator  # noqa: PLC0415

    grp_config = config_data["grp"]
    grp = GRP(**grp_config["network"])
    state_file = Path(str(grp_config["state_file"])).resolve()
    try:
        state = torch.load(state_file, weights_only=True, map_location=torch.device("cpu"))
    except (pickle.UnpicklingError, RuntimeError):
        # Project-owned GRP checkpoints also carry NumPy/Python RNG state.
        # The path is already part of the explicit training contract.
        state = torch.load(state_file, weights_only=False, map_location=torch.device("cpu"))
    grp.load_state_dict(state["model"])
    return RewardCalculator(
        grp,
        [float(value) for value in pts],
        uniform_init=bool(grp_config.get("uniform_init", False)),
    )


class FileDatasetsIter(IterableDataset):
    """Load Mortal gameplay samples with an explicit project reward contract."""

    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle=False,
        file_batch_size=20,
        reserve_ratio=0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = np.asarray(pts, dtype=np.float64)
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        reward_mode = str(config.get("reward", {}).get("mode", "final_rank_mc"))
        if reward_mode == "grp":
            reward_mode = "mortal_grp_delta_pt"
        if reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(f"unsupported project reward mode: {reward_mode}")
        self.reward_contract = reward_contract_from_config(config)
        self.reward_calc = _build_reward_adapter(reward_mode, self.pts, config)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first, reward_mode)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first, reward_mode)

    def load_files(self, augmented, reward_mode):
        random.shuffle(self.file_list)
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size], reward_mode)
            buffer_size = len(self.buffer)
            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue
            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]

        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list, reward_mode):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                grp = game.take_grp()
                player_id = int(game.take_player_id())
                game_size = len(obs)
                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                final_rank = int(rank_by_player[player_id])

                if reward_mode == "mortal_grp_delta_pt":
                    kyoku_rewards = self.reward_calc.calc_delta_pt(
                        player_id,
                        grp_feature,
                        rank_by_player,
                    )
                elif reward_mode == "final_rank_mc":
                    # Centering preserves the official [6,4,2,0] ordering while
                    # matching the expected initial uniform rank value of 3.
                    terminal_return = float(self.pts[final_rank] - self.pts.mean())
                    kyoku_rewards = np.full(len(grp_feature), terminal_return, dtype=np.float64)
                else:
                    # Kept only to make old checkpoints diagnosable; new runs must
                    # use final_rank_mc because terminal_rank was too sparse.
                    kyoku_rewards = np.zeros(len(grp_feature), dtype=np.float64)
                    kyoku_rewards[min(len(kyoku_rewards) - 1, int(at_kyoku[-1]))] = float(self.pts[final_rank])

                assert len(kyoku_rewards) >= at_kyoku[-1] + 1
                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
