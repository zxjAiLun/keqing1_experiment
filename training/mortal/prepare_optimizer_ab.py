#!/usr/bin/env python3
"""Prepare matched final-rank-MC fresh-versus-preserved Adam continuations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
DEFAULT_SOURCE_CONFIG = Path(
    "artifacts/experiments/model_pool_2026_07/reward_ab_2026_07_epoch3/"
    "F_final_rank_mc_weights_only/seed_20260721/config.toml"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default="optimizer_ab_2026_07_epoch1")
    parser.add_argument("--seeds", default="20260724,20260725,20260726")
    parser.add_argument("--initial-steps", type=int, default=70000)
    parser.add_argument("--target-steps", type=int, default=72000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def render(prefix: list[str], table: dict[str, Any]) -> None:
        scalars = [(str(key), value) for key, value in table.items() if not isinstance(value, dict)]
        children = [(str(key), value) for key, value in table.items() if isinstance(value, dict)]
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(base: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["control"]["state_file"] = str((run_dir / "mortal.pth").resolve())
    config["control"]["best_state_file"] = str((run_dir / "mortal_best.pth").resolve())
    config["control"]["tensorboard_dir"] = str((run_dir / "tb_mortal").resolve())
    config["dataset"]["player_names_files"] = [str((run_dir / "ext_mortal_train_labels.txt").resolve())]
    config["reward"] = {"mode": "final_rank_mc"}
    return config


def main() -> None:
    args = parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer seeds")
    if args.initial_steps >= args.target_steps:
        raise ValueError("--initial-steps must be lower than --target-steps")
    parent = args.parent.resolve()
    source_config = args.source_config.resolve()
    if not parent.exists():
        raise FileNotFoundError(parent)
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    output_root = (args.output_root / args.experiment_id).resolve()
    base = load_config(source_config)
    parent_sha = sha256_file(parent)
    groups = (
        ("A_final_rank_mc_fresh_adam", "fresh"),
        ("B_final_rank_mc_preserved_adam", "preserved"),
    )
    runs: list[dict[str, Any]] = []
    for group_id, optimizer_mode in groups:
        for seed in seeds:
            run_dir = output_root / group_id / f"seed_{seed}"
            config = prepare_config(base, run_dir)
            config_path = run_dir / "config.toml"
            labels_path = run_dir / "ext_mortal_train_labels.txt"
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(dump_toml(config), encoding="utf-8")
                labels_path.write_text("ext_mortal\n", encoding="utf-8")
            runs.append(
                {
                    "group": group_id,
                    "optimizer_mode": optimizer_mode,
                    "seed": seed,
                    "data_seed": seed,
                    "config": str(config_path),
                    "state_file": str((run_dir / "mortal.pth").resolve()),
                    "archive_dir": str((run_dir / "checkpoints").resolve()),
                    "parent_checkpoint": str(parent),
                    "parent_sha256": parent_sha,
                    "training_command": [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        "scripts/run_mortal_dqn_offline.py",
                        "--config",
                        str(config_path),
                        "--target-steps",
                        str(args.target_steps),
                        "--device",
                        "cuda",
                        "--num-workers",
                        "0",
                        "--seed",
                        str(seed),
                        "--data-seed",
                        str(seed),
                        "--initialize-from",
                        str(parent),
                        "--initial-steps",
                        str(args.initial_steps),
                        "--archive-steps",
                        "70001,70010,70100,70500,71000,72000",
                        "--archive-dir",
                        str((run_dir / "checkpoints").resolve()),
                        "--log-every",
                        "50",
                    ]
                    + (["--initialize-optimizer-from", str(parent)] if optimizer_mode == "preserved" else []),
                }
            )
    manifest = {
        "schema": "keqing.mortal.optimizer_ab.v1",
        "experiment_id": args.experiment_id,
        "objective": "final_rank_mc continuation: 70k legacy Adam-state carryover versus Adam reset",
        "parent_checkpoint": str(parent),
        "parent_sha256": parent_sha,
        "parent_steps": int(args.initial_steps),
        "parent_optimizer_objective": "unknown_legacy",
        "source_config": str(source_config),
        "seeds": list(seeds),
        "target_steps": int(args.target_steps),
        "archive_steps": [70001, 70010, 70100, 70500, 71000, 72000],
        "fixed_recipe": {
            "reward_mode": "final_rank_mc",
            "data_seed_equals_model_seed": True,
            "fresh_scheduler": True,
            "fresh_scaler": True,
            "fresh_data_stream": True,
            "dataset_file_index": str(Path(base["dataset"]["file_index"]).resolve()),
            "num_epochs": int(base["dataset"]["num_epochs"]),
            "batch_size": int(base["control"]["batch_size"]),
            "cql_min_q_weight": float(base["cql"]["min_q_weight"]),
            "next_rank_weight": float(base["aux"]["next_rank_weight"]),
            "lr": float(base["optim"]["scheduler"]["peak"]),
        },
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain", "--untracked-files=all")),
        "runs": runs,
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if manifest["git_dirty"]:
        raise SystemExit("working tree is dirty; commit the experiment implementation before training")


if __name__ == "__main__":
    main()
