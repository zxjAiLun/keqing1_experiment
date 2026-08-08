#!/usr/bin/env python3
"""Validate the control/variant contract before legal-mean training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tomllib
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON_ROOT = REPO_ROOT / "third_party" / "Mortal" / "mortal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.setdefault("control", {}).pop("state_file", None)
    value.setdefault("control", {}).pop("best_state_file", None)
    value.setdefault("control", {}).pop("tensorboard_dir", None)
    value.setdefault("dataset", {}).pop("player_names_files", None)
    value.pop("objective", None)
    return value


def checkpoint_tensor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for section in ("mortal", "current_dqn", "aux_net"):
        payload = state.get(section)
        if not isinstance(payload, dict):
            raise ValueError(f"parent checkpoint is missing {section}")
        for name in sorted(payload):
            tensor = payload[name]
            if not isinstance(tensor, torch.Tensor):
                continue
            digest.update(section.encode())
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_hash_child(config_path: Path, data_seed: int, batch_count: int) -> None:
    os.environ["MORTAL_CFG"] = str(config_path.resolve())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(MORTAL_PYTHON_ROOT) not in sys.path:
        sys.path.insert(0, str(MORTAL_PYTHON_ROOT))
    from config import config  # noqa: PLC0415
    from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: PLC0415

    file_index = Path(str(config["dataset"]["file_index"]))
    file_list = list(torch.load(file_index, weights_only=True)["file_list"])
    player_names: list[str] = []
    for label_path in config["dataset"]["player_names_files"]:
        player_names.extend(
            line.strip()
            for line in Path(str(label_path)).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    random.seed(data_seed)
    torch.manual_seed(data_seed)
    dataset = FileDatasetsIter(
        version=int(config["control"]["version"]),
        file_list=file_list,
        pts=config["env"]["pts"],
        file_batch_size=int(config["dataset"]["file_batch_size"]),
        reserve_ratio=float(config["dataset"]["reserve_ratio"]),
        player_names=player_names,
        num_epochs=int(config["dataset"]["num_epochs"]),
        enable_augmentation=bool(config["dataset"]["enable_augmentation"]),
        augmented_first=bool(config["dataset"]["augmented_first"]),
    )
    loader = iter(DataLoader(dataset, batch_size=int(config["control"]["batch_size"]), drop_last=True, num_workers=0))
    digest = hashlib.sha256()
    batches: list[str] = []
    for _ in range(batch_count):
        batch = next(loader)
        batch_digest = hashlib.sha256()
        for tensor in batch:
            batch_digest.update(str(tensor.dtype).encode())
            batch_digest.update(str(tuple(tensor.shape)).encode())
            batch_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        value = batch_digest.hexdigest()
        batches.append(value)
        digest.update(value.encode())
    print(json.dumps({"batches": batches, "digest": digest.hexdigest()}), flush=True)


def batch_hash(config_path: Path, data_seed: int, batch_count: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--batch-hash-child",
            "--config",
            str(config_path.resolve()),
            "--data-seed",
            str(data_seed),
            "--batch-count",
            str(batch_count),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    if "--batch-hash-child" in sys.argv:
        child = argparse.ArgumentParser()
        child.add_argument("--batch-hash-child", action="store_true")
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--data-seed", type=int, required=True)
        child.add_argument("--batch-count", type=int, required=True)
        child_args = child.parse_args()
        _batch_hash_child(child_args.config, child_args.data_seed, child_args.batch_count)
        return
    args = parse_args()
    control_config = load_config(args.control_config.resolve())
    variant_config = load_config(args.variant_config.resolve())
    control_mode = str(control_config.get("objective", {}).get("mode", "behavior_action_mc"))
    variant_mode = str(variant_config.get("objective", {}).get("mode", "behavior_action_mc"))
    if control_mode != "behavior_action_mc":
        raise ValueError(f"control objective must be behavior_action_mc, got {control_mode}")
    if variant_mode != "legal_mean_mc":
        raise ValueError(f"variant objective must be legal_mean_mc, got {variant_mode}")
    for name, config in (("control", control_config), ("variant", variant_config)):
        if str(config.get("reward", {}).get("mode")) != "final_rank_mc":
            raise ValueError(f"{name} reward must be final_rank_mc")
    if normalized_config(control_config) != normalized_config(variant_config):
        raise ValueError("control and variant configs differ outside objective/run-path fields")
    parent = args.parent.resolve()
    state = torch.load(parent, weights_only=True, map_location="cpu")
    steps = int(state.get("steps", -1))
    if steps != 70000:
        raise ValueError(f"parent checkpoint must be at step 70000, got {steps}")
    if "optimizer" not in state:
        raise ValueError("parent checkpoint has no Adam optimizer state")
    current_git_commit = git_value("rev-parse", "HEAD")
    git_dirty = bool(git_value("status", "--porcelain", "--untracked-files=all"))
    if git_dirty:
        raise ValueError("working tree is dirty; commit before running objective preflight")
    control_config_path = args.control_config.resolve()
    variant_config_path = args.variant_config.resolve()
    control_file_index = Path(str(control_config["dataset"]["file_index"])).resolve()
    variant_file_index = Path(str(variant_config["dataset"]["file_index"])).resolve()
    if control_file_index != variant_file_index:
        raise ValueError("control and variant use different file indexes")
    if not control_file_index.exists():
        raise FileNotFoundError(control_file_index)
    control_labels = [Path(str(value)).resolve() for value in control_config["dataset"]["player_names_files"]]
    variant_labels = [Path(str(value)).resolve() for value in variant_config["dataset"]["player_names_files"]]
    if len(control_labels) != len(variant_labels):
        raise ValueError("control and variant label-file counts differ")
    if any(path.read_bytes() != other.read_bytes() for path, other in zip(control_labels, variant_labels, strict=True)):
        raise ValueError("control and variant label-file contents differ")
    control_batches = batch_hash(control_config_path, args.data_seed, 2)
    variant_batches = batch_hash(variant_config_path, args.data_seed, 2)
    if control_batches != variant_batches:
        raise ValueError("control and variant first data batches differ")
    report = {
        "schema": "keqing.mortal.legal_mean_value_preflight.v1",
        "passed": True,
        "control_config": str(control_config_path),
        "variant_config": str(variant_config_path),
        "data_seed": args.data_seed,
        "parent": str(parent),
        "parent_sha256": sha256_file(parent),
        "parent_tensor_digest": checkpoint_tensor_digest(state),
        "parent_steps": steps,
        "optimizer_state_present": True,
        "reward_mode": "final_rank_mc",
        "control_objective": control_mode,
        "variant_objective": variant_mode,
        "config_equal_except_objective_and_run_paths": True,
        "fingerprints": {
            "control_config_sha256": sha256_file(control_config_path),
            "variant_config_sha256": sha256_file(variant_config_path),
            "parent_sha256": sha256_file(parent),
            "file_index": str(control_file_index),
            "file_index_sha256": sha256_file(control_file_index),
            "control_label_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in control_labels
            ],
            "variant_label_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in variant_labels
            ],
            "git_commit": current_git_commit,
            "git_dirty": git_dirty,
        },
        "first_data_batches": {
            "data_seed": args.data_seed,
            "batch_count": 2,
            "control": control_batches,
            "variant": variant_batches,
            "identical": True,
        },
        "required_initialization": {
            "weights": "parent",
            "optimizer": "same_parent",
            "initial_steps": 70000,
            "scheduler": "fresh",
            "scaler": "fresh",
            "data_stream": "fresh",
            "data_seed_must_match_model_seed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
