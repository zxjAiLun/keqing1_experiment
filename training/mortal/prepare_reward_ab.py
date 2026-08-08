#!/usr/bin/env python3
"""Prepare matched-seed final-rank-MC versus Mortal-GRP reward experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.prepare_v2_population_mixed_warmstart import POOL_SPECS
from training.mortal.prepare_v3_final_rank_mc_warmstart import _normalize_host_paths


DEFAULT_EXPERIMENT_ID = "reward_ab_2026_07_epoch2"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V2_data"
PARENT_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("configs/mortal_offline_mainline.toml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument("--grp-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-steps", type=int, default=72000)
    parser.add_argument("--initial-steps", type=int, default=70000)
    parser.add_argument("--seeds", default="20260718,20260719,20260720")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def render(prefix: list[str], table: Mapping[str, Any]) -> None:
        scalars = [(str(key), value) for key, value in table.items() if not isinstance(value, Mapping)]
        children = [(str(key), value) for key, value in table.items() if isinstance(value, Mapping)]
        if prefix:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in children:
            render([*prefix, key], value)

    render([], data)
    return "\n".join(lines).rstrip() + "\n"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_config(
    base_config: dict[str, Any],
    *,
    run_dir: Path,
    shared_file_index: Path,
    data_root: Path,
    reward_mode: str,
    grp_checkpoint: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    control = config.setdefault("control", {})
    control["state_file"] = str((run_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((run_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((run_dir / "tb_mortal").resolve())
    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [
        str((data_root / pool_id / "logs" / "**" / "*.json.gz").resolve())
        for pool_id, _ in POOL_SPECS
    ]
    dataset["file_index"] = str(shared_file_index.resolve())
    dataset["player_names_files"] = [str((run_dir / "ext_mortal_train_labels.txt").resolve())]
    dataset["num_workers"] = 0
    dataset["num_epochs"] = 2
    dataset["enable_augmentation"] = False
    dataset["augmented_first"] = False
    config.setdefault("reward", {})["mode"] = reward_mode
    if reward_mode == "mortal_grp_delta_pt":
        config["grp"] = {
            "state_file": str(grp_checkpoint.resolve()),
            "network": {"hidden_size": 64, "num_layers": 2},
            "uniform_init": False,
        }
    return config


def main() -> None:
    args = parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if args.initial_steps >= args.stage_steps:
        raise ValueError("--initial-steps must be lower than --stage-steps")
    parent = args.parent_checkpoint.resolve()
    grp_checkpoint = args.grp_checkpoint.resolve()
    if not parent.exists():
        raise FileNotFoundError(parent)
    if not grp_checkpoint.exists():
        raise FileNotFoundError(
            f"GRP checkpoint is required before preparing the A/B run: {grp_checkpoint}"
        )

    output_root = args.output_root / args.experiment_id
    shared_dir = output_root / "shared"
    shared_file_index = shared_dir / "file_index.pth"
    base_config = _load_toml(args.base_config)
    if sys.platform == "win32":
        base_config = _normalize_host_paths(base_config)

    groups = (
        ("F_final_rank_mc_weights_only", "final_rank_mc"),
        ("G_mortal_grp_delta_pt_weights_only", "mortal_grp_delta_pt"),
    )
    runs: list[dict[str, Any]] = []
    for group_id, reward_mode in groups:
        for seed in seeds:
            run_dir = output_root / group_id / f"seed_{seed}"
            config = _prepare_config(
                base_config,
                run_dir=run_dir,
                shared_file_index=shared_file_index,
                data_root=args.data_root,
                reward_mode=reward_mode,
                grp_checkpoint=grp_checkpoint,
            )
            config_path = run_dir / "config.toml"
            command = [
                "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
                "--config", str(config_path), "--target-steps", str(args.stage_steps),
                "--device", "cuda", "--num-workers", "0", "--seed", str(seed),
                "--data-seed", str(seed), "--initialize-from", str(parent),
                "--initial-steps", str(args.initial_steps), "--archive-steps", str(args.stage_steps),
                "--archive-dir", str(run_dir / "checkpoints"), "--log-every", "50",
            ]
            runs.append(
                {
                    "group": group_id,
                    "seed": seed,
                    "reward_mode": reward_mode,
                    "config": str(config_path),
                    "command": command,
                    "state_file": str(Path(config["control"]["state_file"])),
                }
            )
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(_dump_toml(config), encoding="utf-8")
                (run_dir / "ext_mortal_train_labels.txt").write_text("ext_mortal\n", encoding="utf-8")

    manifest = {
        "schema": "keqing.mortal.reward_ab.v1",
        "experiment_id": args.experiment_id,
        "parent_checkpoint": str(parent),
        "parent_init_mode": "weights_only_fresh_adam_fresh_stream",
        "grp_checkpoint": str(grp_checkpoint),
        "grp_checkpoint_sha256": _sha256_file(grp_checkpoint),
        "matched_seeds": list(seeds),
        "fixed_recipe": {
            "gamma": 1.0,
            "cql_min_q_weight": 5.0,
            "next_rank_weight": 0.2,
            "lr": 1e-4,
            "dataset": "shared 6000-hanchan file index, num_epochs=2",
        },
        "runs": runs,
    }
    manifest_path = output_root / "manifest.json"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
