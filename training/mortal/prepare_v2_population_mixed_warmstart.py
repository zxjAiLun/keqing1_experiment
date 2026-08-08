#!/usr/bin/env python3
"""Prepare the archived no-teacher-CE V2 mixed-ecology experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPERIMENT_ID = "V2_population_mixed_v4_warmstart_2026_07"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V2_data"
PARENT_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
POOL_SPECS = (
    ("v4_70k_t1_v0b_2000h", 960000),
    ("v4_70k_v1_80k_2000h", 962000),
    ("v4_v0b_v1_t1_2000h", 964000),
)


def read_checkpoint_steps(checkpoint: Path) -> int:
    import torch  # noqa: PLC0415

    return int(torch.load(checkpoint, weights_only=True, map_location="cpu")["steps"])


def dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def render_table(prefix: list[str], table: Mapping[str, Any]) -> None:
        scalars = [(str(key), value) for key, value in table.items() if not isinstance(value, Mapping)]
        children = [(str(key), value) for key, value in table.items() if isinstance(value, Mapping)]
        if prefix:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        for key, value in scalars:
            lines.append(f"{key} = {toml_value(value)}")
        for key, value in children:
            render_table([*prefix, key], value)

    def toml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return "[" + ", ".join(toml_value(item) for item in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value).__name__}")

    render_table([], data)
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("configs/mortal_offline_mainline.toml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument("--initial-steps", type=int, default=70000)
    parser.add_argument("--stage1-steps", type=int, default=72000)
    parser.add_argument("--final-steps", type=int, default=74000)
    parser.add_argument("--model-seed", type=int, default=20260712)
    parser.add_argument("--data-seed", type=int, default=20260712)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_host_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_host_paths(item) for item in value]
    if not isinstance(value, str) or os.name != "nt":
        return value
    if not value.startswith("/mnt/") or len(value) < 7 or value[6] != "/":
        return value
    return str(Path(f"{value[5].upper()}:/{value[7:]}").resolve())


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(base_config: dict[str, Any], *, exp_dir: Path, data_root: Path) -> dict[str, Any]:
    config = _normalize_host_paths(copy.deepcopy(base_config))
    control = config.setdefault("control", {})
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())

    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [str((data_root / pool_id / "logs" / "**" / "*.json.gz").resolve()) for pool_id, _ in POOL_SPECS]
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    dataset["num_workers"] = 0
    dataset["player_names_files"] = [str((exp_dir / "ext_mortal_train_labels.txt").resolve())]
    dataset["num_epochs"] = 2
    dataset["enable_augmentation"] = False

    return config


def main() -> None:
    args = parse_args()
    if not 0 <= int(args.initial_steps) < int(args.stage1_steps) < int(args.final_steps):
        raise ValueError("steps must satisfy 0 <= initial < stage1 < final")
    exp_dir = args.output_root / EXPERIMENT_ID
    config = prepare_config(_load_toml(args.base_config), exp_dir=exp_dir, data_root=args.data_root)
    config_path = exp_dir / "config.toml"
    checkpoints_dir = exp_dir / "checkpoints"
    parent_steps = read_checkpoint_steps(args.parent_checkpoint)
    pools = [{"pool_id": pool_id, "seed_start": seed_start, "games": 2000, "train_label": "ext_mortal"} for pool_id, seed_start in POOL_SPECS]
    train_base = [
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", str(config_path), "--device", "cuda", "--num-workers", "0",
        "--seed", str(args.model_seed), "--data-seed", str(args.data_seed),
        "--initialize-from", str(args.parent_checkpoint), "--initial-steps", str(args.initial_steps),
        "--initialize-optimizer-from", str(args.parent_checkpoint),
        "--archive-steps", f"{args.stage1_steps},{args.final_steps}", "--archive-dir", str(checkpoints_dir), "--log-every", "50",
    ]
    manifest = {
        "schema": "keqing.mortal.v2_population_mixed_warmstart.v1",
        "experiment_id": EXPERIMENT_ID,
        "initialization": {
            "parent_checkpoint": str(args.parent_checkpoint),
            "optimizer_checkpoint": str(args.parent_checkpoint),
            "optimizer_checkpoint_must_match_parent": True,
            "parent_steps": parent_steps,
            "initial_steps": args.initial_steps,
            "optimizer": "preserved",
            "scheduler": "fresh",
            "scaler": "fresh",
            "data_stream": "fresh",
        },
        "objective": "offline DQN + CQL + next-rank auxiliary",
        "train_labels": ["ext_mortal"],
        "pools": pools,
        "expected_unique_hanchans": 6000,
        "expected_trainable_v4_seat_hanchans": 6000,
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "dataset_audit_command": ["uv", "run", "--no-sync", "python", "training/mortal/audit_population_synthetic_dataset.py", "--data-root", str(args.data_root), "--output", str(exp_dir / "dataset_audit.json")],
        "training_command": [*train_base, "--target-steps", str(args.final_steps)],
        "stage1_archive": str(checkpoints_dir / f"mortal_{args.stage1_steps}.pth"),
        "final_archive": str(checkpoints_dir / f"mortal_{args.final_steps}.pth"),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "ext_mortal_train_labels.txt").write_text("ext_mortal\n", encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
