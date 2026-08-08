#!/usr/bin/env python3
"""Preflight a fresh-Adam versus preserved-Adam continuation pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.amp import GradScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))

from training.run_mortal_dqn_offline import _optimizer_param_groups  # noqa: E402
from lr_scheduler import LinearWarmUpCosineAnnealingLR  # noqa: E402
from model import AuxNet, Brain, DQN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-config", type=Path, required=True)
    parser.add_argument("--preserved-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--optimizer-parent", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=True, map_location="cpu")


def model_bundle(config: dict[str, Any], state: dict[str, Any]):
    version = int(config["control"]["version"])
    brain = Brain(version=version, **config["resnet"])
    dqn = DQN(version=version)
    aux = AuxNet((4,))
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    aux.load_state_dict(state["aux_net"])
    return brain, dqn, aux


def make_optimizer(config: dict[str, Any], modules: tuple[torch.nn.Module, ...]):
    return optim.AdamW(
        _optimizer_param_groups(modules, weight_decay=float(config["optim"]["weight_decay"])),
        lr=1,
        weight_decay=0,
        betas=tuple(float(value) for value in config["optim"]["betas"]),
        eps=float(config["optim"]["eps"]),
    )


def make_scheduler(config: dict[str, Any], optimizer):
    return LinearWarmUpCosineAnnealingLR(optimizer, **config["optim"]["scheduler"])


def normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    return value


def group_metadata(optimizer) -> list[dict[str, Any]]:
    return [
        {key: normalize(value) for key, value in group.items() if key != "params"}
        for group in optimizer.state_dict()["param_groups"]
    ]


def model_weights_equal(left: tuple[torch.nn.Module, ...], right: tuple[torch.nn.Module, ...]) -> bool:
    for left_module, right_module in zip(left, right):
        left_state = left_module.state_dict()
        right_state = right_module.state_dict()
        if left_state.keys() != right_state.keys():
            return False
        if any(not torch.equal(left_state[key], right_state[key]) for key in left_state):
            return False
    return True


def module_parameter_tensor_count(modules: tuple[torch.nn.Module, ...]) -> int:
    return sum(1 for module in modules for _ in module.parameters())


def stream_preview(config: dict[str, Any], data_seed: int) -> dict[str, Any]:
    file_index = Path(str(config["dataset"]["file_index"])).resolve()
    payload = torch.load(file_index, weights_only=True, map_location="cpu")
    file_list = list(payload["file_list"])
    shuffled = list(file_list)
    random.Random(int(data_seed)).shuffle(shuffled)
    file_batch_size = int(config["dataset"]["file_batch_size"])
    prefix = shuffled[: file_batch_size * 3]
    digest = hashlib.sha256()
    for filename in prefix:
        digest.update(str(filename).encode("utf-8"))
        digest.update(b"\0")
    return {
        "file_index": str(file_index),
        "file_index_sha256": sha256_file(file_index),
        "file_count": len(file_list),
        "data_seed": int(data_seed),
        "first_file_batches": [prefix[i : i + file_batch_size] for i in range(0, len(prefix), file_batch_size)],
        "first_file_batches_sha256": digest.hexdigest(),
    }


def comparable_recipe(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": config["control"]["version"],
        "batch_size": config["control"]["batch_size"],
        "opt_step_every": config["control"]["opt_step_every"],
        "dataset": {
            key: config["dataset"].get(key)
            for key in (
                "globs",
                "file_index",
                "file_batch_size",
                "reserve_ratio",
                "num_workers",
                "num_epochs",
                "enable_augmentation",
                "augmented_first",
            )
        },
        "env": config["env"],
        "reward": config.get("reward", {}),
        "resnet": config["resnet"],
        "cql": config["cql"],
        "aux": config["aux"],
        "freeze_bn": config["freeze_bn"],
        "optim": config["optim"],
    }


def parent_objective(state: dict[str, Any]) -> dict[str, Any]:
    contract = state.get("training_contract") or {}
    config = state.get("config") or {}
    contract_mode = contract.get("reward_mode") or contract.get("reward", {}).get("mode")
    config_mode = config.get("reward", {}).get("mode")
    if contract_mode or config_mode:
        inferred = contract_mode or config_mode
    else:
        inferred = "unknown_legacy"
    return {
        "training_contract_reward_mode": contract_mode,
        "config_reward_mode": config_mode,
        "legacy_grp_field_present": "grp" in config,
        "inferred_parent_optimizer_objective": inferred,
        "contract_present": bool(contract),
    }


def main() -> None:
    args = parse_args()
    fresh_config_path = args.fresh_config.resolve()
    preserved_config_path = args.preserved_config.resolve()
    parent_path = args.parent.resolve()
    optimizer_parent_path = (args.optimizer_parent or args.parent).resolve()
    fresh_config = load_toml(fresh_config_path)
    preserved_config = load_toml(preserved_config_path)
    parent = load_checkpoint(parent_path)
    optimizer_parent = load_checkpoint(optimizer_parent_path)
    if fresh_config["reward"]["mode"] != "final_rank_mc" or preserved_config["reward"]["mode"] != "final_rank_mc":
        raise SystemExit("optimizer A/B requires final_rank_mc in both configs")

    fresh_modules = model_bundle(fresh_config, parent)
    preserved_modules = model_bundle(preserved_config, parent)
    fresh_optimizer = make_optimizer(fresh_config, fresh_modules)
    preserved_optimizer = make_optimizer(preserved_config, preserved_modules)
    fresh_scheduler = make_scheduler(fresh_config, fresh_optimizer)
    preserved_scheduler = make_scheduler(preserved_config, preserved_optimizer)
    fresh_scaler = GradScaler("cuda", enabled=bool(fresh_config["control"]["enable_amp"]))
    preserved_scaler = GradScaler("cuda", enabled=bool(preserved_config["control"]["enable_amp"]))

    parent_optimizer = optimizer_parent.get("optimizer")
    if not isinstance(parent_optimizer, dict):
        raise SystemExit("parent checkpoint has no optimizer state")
    preserved_optimizer.load_state_dict(parent_optimizer)

    fresh_groups = group_metadata(fresh_optimizer)
    preserved_groups = group_metadata(preserved_optimizer)
    parent_groups = [
        {key: normalize(value) for key, value in group.items() if key != "params"}
        for group in parent_optimizer["param_groups"]
    ]
    preserved_states = preserved_optimizer.state_dict()["state"]
    required_fields = {"step", "exp_avg", "exp_avg_sq"}
    state_field_ok = bool(preserved_states) and all(required_fields.issubset(entry) for entry in preserved_states.values())
    expected_state_count = module_parameter_tensor_count(preserved_modules)
    recipe_equal = comparable_recipe(fresh_config) == comparable_recipe(preserved_config)
    stream_fresh = stream_preview(fresh_config, int(args.data_seed))
    stream_preserved = stream_preview(preserved_config, int(args.data_seed))
    checks = {
        "parent_steps_70000": int(parent.get("steps", -1)) == 70000,
        "parent_sha256_matches_optimizer_source": sha256_file(parent_path) == sha256_file(optimizer_parent_path),
        "model_weights_identical": model_weights_equal(fresh_modules, preserved_modules),
        "recipe_equal": recipe_equal,
        "scheduler_fresh_equal": normalize(fresh_scheduler.state_dict()) == normalize(preserved_scheduler.state_dict()),
        "scaler_fresh_equal": normalize(fresh_scaler.state_dict()) == normalize(preserved_scaler.state_dict()),
        "fresh_optimizer_state_empty": len(fresh_optimizer.state_dict()["state"]) == 0,
        "optimizer_group_metadata_equal": fresh_groups == preserved_groups,
        "optimizer_group_metadata_matches_parent": preserved_groups == parent_groups,
        "preserved_state_has_moments": state_field_ok,
        "preserved_state_count_matches_parameters": len(preserved_states) == expected_state_count,
        "data_stream_preview_equal": stream_fresh == stream_preserved,
    }
    report = {
        "schema": "keqing.mortal.optimizer_ab_preflight.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "fresh_config": str(fresh_config_path),
        "preserved_config": str(preserved_config_path),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "steps": int(parent.get("steps", -1)),
            "optimizer_source_path": str(optimizer_parent_path),
            "optimizer_source_sha256": sha256_file(optimizer_parent_path),
            "optimizer_state_count": len(parent_optimizer["state"]),
            "objective": parent_objective(parent),
        },
        "optimizer": {
            "fresh_groups": fresh_groups,
            "preserved_groups": preserved_groups,
            "parent_groups": parent_groups,
            "preserved_state_count": len(preserved_states),
            "expected_parameter_count": expected_state_count,
            "required_state_fields": sorted(required_fields),
        },
        "data_stream": {"fresh": stream_fresh, "preserved": stream_preserved},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output)}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit("optimizer preflight failed")


if __name__ == "__main__":
    main()
