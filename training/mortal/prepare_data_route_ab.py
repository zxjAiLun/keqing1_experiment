#!/usr/bin/env python3
"""Prepare the fixed-corpus mixed-vs-pure selfplay screening runs."""

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


EXPERIMENT_ID = "data_route_ab_2026_07"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
DEFAULT_BASE_CONFIG = Path("configs/mortal_offline_mainline.toml")
PARENT_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
MIXED_FILE_INDEX = Path(
    "artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/file_index.pth"
)
PURE_FILE_INDEX = Path(
    "artifacts/experiments/model_pool_2026_07/S0_pure_ext_selfplay_6000h/file_index.pth"
)
TRAINING_SEEDS = (20260731, 20260801, 20260802)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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


def dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def render(prefix: list[str], table: Mapping[str, Any]) -> None:
        scalars = [(str(key), value) for key, value in table.items() if not isinstance(value, Mapping)]
        children = [(str(key), value) for key, value in table.items() if isinstance(value, Mapping)]
        if prefix:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        for key, value in scalars:
            lines.append(f"{key} = {toml_value(value)}")
        for key, value in children:
            render([*prefix, key], value)

    render([], data)
    return "\n".join(lines).rstrip() + "\n"


def normalize_host_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_host_paths(item) for item in value]
    if not isinstance(value, str) or not value.startswith("/mnt/") or len(value) < 7 or value[6] != "/":
        return value
    return str(Path(f"{value[5].upper()}:/{value[7:]}").resolve())


def prepare_config(
    base: dict[str, Any],
    *,
    run_dir: Path,
    file_index: Path,
    globs: list[Path],
    labels_file: Path,
) -> dict[str, Any]:
    config = normalize_host_paths(copy.deepcopy(base)) if os.name == "nt" else copy.deepcopy(base)
    control = config.setdefault("control", {})
    control["state_file"] = str((run_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((run_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((run_dir / "tb_mortal").resolve())
    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [str(path.resolve()) for path in globs]
    dataset["file_index"] = str(file_index.resolve())
    dataset["file_batch_size"] = 15
    dataset["num_workers"] = 0
    dataset["player_names_files"] = [str(labels_file.resolve())]
    dataset["num_epochs"] = 3
    dataset["enable_augmentation"] = False
    dataset["augmented_first"] = False
    config.setdefault("reward", {})["mode"] = "final_rank_mc"
    config.setdefault("env", {})["pts"] = [6.0, 4.0, 2.0, 0.0]
    return config


def main() -> None:
    args = parse_args()
    if not args.parent_checkpoint.exists():
        raise FileNotFoundError(args.parent_checkpoint)
    if not args.base_config.exists():
        raise FileNotFoundError(args.base_config)
    if not MIXED_FILE_INDEX.exists() or not PURE_FILE_INDEX.exists():
        raise FileNotFoundError("M0/S0 file index is missing; finish data audits first")

    output_dir = args.output_root / EXPERIMENT_ID
    base = load_toml(args.base_config)
    routes = {
        "M0_mixed": {
            "file_index": MIXED_FILE_INDEX,
            "globs": [args.output_root / "V2_data" / "*" / "logs" / "**" / "*.json.gz"],
            "label": "ext_mortal",
            "description": "frozen mixed-opponent corpus from V3",
        },
        "S0_pure": {
            "file_index": PURE_FILE_INDEX,
            "globs": [args.output_root / "S0_pure_ext_selfplay_6000h" / "logs" / "**" / "*.json.gz"],
            "label": "train_ext",
            "description": "frozen pure ext_mortal selfplay corpus",
        },
    }
    manifest: dict[str, Any] = {
        "schema": "keqing.mortal.data_route_ab.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "optimizer_initialization": "preserved 70k Adam; scheduler/scaler/data stream fresh",
        "reward_mode": "final_rank_mc",
        "target_steps": 72000,
        "initial_steps": 70000,
        "training_seeds": list(TRAINING_SEEDS),
        "routes": {},
    }
    for route_id, route in routes.items():
        route_dir = output_dir / route_id
        labels_file = route_dir / "train_labels.txt"
        config_path = route_dir / "config.toml"
        route_manifest = {
            "route_id": route_id,
            "description": route["description"],
            "label": route["label"],
            "file_index": str(Path(route["file_index"]).resolve()),
            "expected_hanchans": 6000,
            "config": str(config_path.resolve()),
            "runs": [],
        }
        config = prepare_config(
            base,
            run_dir=route_dir,
            file_index=Path(route["file_index"]),
            globs=[Path(value) for value in route["globs"]],
            labels_file=labels_file,
        )
        for seed in TRAINING_SEEDS:
            run_dir = route_dir / f"seed_{seed}"
            run_config = copy.deepcopy(config)
            run_config["control"]["state_file"] = str((run_dir / "mortal.pth").resolve())
            run_config["control"]["best_state_file"] = str((run_dir / "mortal_best.pth").resolve())
            run_config["control"]["tensorboard_dir"] = str((run_dir / "tb_mortal").resolve())
            route_manifest["runs"].append(
                {
                    "seed": seed,
                    "data_seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "config": str((run_dir / "config.toml").resolve()),
                    "target_steps": 72000,
                }
            )
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "config.toml").write_text(dump_toml(run_config), encoding="utf-8")
                (run_dir / "train_labels.txt").write_text(str(route["label"]) + "\n", encoding="utf-8")
        if not args.dry_run:
            route_dir.mkdir(parents=True, exist_ok=True)
            labels_file.write_text(str(route["label"]) + "\n", encoding="utf-8")
            config_path.write_text(dump_toml(config), encoding="utf-8")
            (route_dir / "manifest.json").write_text(
                json.dumps(route_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        manifest["routes"][route_id] = route_manifest
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
