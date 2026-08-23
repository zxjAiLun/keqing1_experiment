#!/usr/bin/env python3
"""Fail-closed launcher and dataset preparer for M1 ext_mortal 12,000-hanchan expansion."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.m1_dataset_contract_2026_08 import (
    ARCHIVE_STEPS,
    K0_70K_PATH,
    K0_70K_SHA256,
    M1_DATASET_DIR,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    PREREG_COMMIT,
    PREREG_PATH,
    REPO_ROOT,
    SEEDS,
    START_STEP,
    TARGET_STEP,
    ContractError,
    build_m1_dataset_files,
    generate_m1_training_config,
    git_blob_oid,
    git_info,
    sha256_file,
    validate_all_8_checkpoints,
    validate_m1_dataset_integrity,
)

OFFLINE_TRAINER_SCRIPT = REPO_ROOT / "training/run_mortal_dqn_offline.py"

# Authorization flags (fail closed by default)
DATASET_PREPARATION_AUTHORIZED = False
APPROVED_M1_IMPLEMENTATION_COMMIT = None
AUTHORIZED_PREREG_SHA256 = None

TRAINING_AUTHORIZED = False
AUTHORIZED_DATASET_MANIFEST_SHA256 = None
AUTHORIZED_DATASET_INDEX_SHA256 = None
AUTHORIZED_PLAYER_MAPPING_SHA256 = None
AUTHORIZED_TRAINING_PLAN_SHA256 = None
AUTHORIZED_TRAINING_PREFLIGHT_SHA256 = None


class AuthorizationError(RuntimeError):
    """Raised when an action is requested without required authorization."""


def build_training_command(
    seed: int,
    config_path: Path,
    parent_path: Path = K0_70K_PATH,
    run_dir: Path | None = None,
) -> list[str]:
    """Construct command-line arguments for training/run_mortal_dqn_offline.py."""
    if run_dir is None:
        run_dir = M1_TRAINING_DIR / f"M1_variant/seed_{seed}"
    archive_dir = run_dir / "checkpoints"
    archive_steps = ",".join(str(s) for s in ARCHIVE_STEPS)

    return [
        sys.executable,
        "-u",
        str(OFFLINE_TRAINER_SCRIPT),
        "--config", str(config_path.resolve()),
        "--initialize-from", str(parent_path.resolve()),
        "--initialize-optimizer-from", str(parent_path.resolve()),
        "--initial-steps", str(START_STEP),
        "--target-steps", str(TARGET_STEP),
        "--device", "cuda:0",
        "--seed", str(seed),
        "--data-seed", str(seed),
        "--num-workers", "0",
        "--archive-steps", archive_steps,
        "--archive-dir", str(archive_dir.resolve()),
    ]


def prepare_m1_dataset(
    output_dir: Path = M1_DATASET_DIR,
    require_authorization: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """Build and validate M1 concatenated 12000 file index and manifest."""
    if require_authorization and not DATASET_PREPARATION_AUTHORIZED:
        raise AuthorizationError(
            "M1 dataset preparation is NOT authorized. "
            "Formal dataset creation requires review and an authorization-only commit."
        )

    print(f"Building M1 dataset in {output_dir}...")
    m1_idx, m1_map, m1_lbl, manifest = build_m1_dataset_files(output_dir)
    print(f"Dataset manifest generated at {manifest} (SHA256: {sha256_file(manifest)})")
    return m1_idx, m1_map, m1_lbl, manifest


def prepare_training_manifest(
    dataset_dir: Path = M1_DATASET_DIR,
    output_training_dir: Path = M1_TRAINING_DIR,
) -> dict[str, Any]:
    """Prepare and freeze training configs and execution manifest for all 3 seeds."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    m1_idx_path = dataset_dir / "file_index_m1.pth"
    m1_map_path = dataset_dir / "player_names_by_file.json"
    m1_lbl_path = dataset_dir / "player_names.txt"

    if not manifest_path.exists() or not m1_idx_path.exists():
        raise ContractError(
            f"Dataset closure is missing at {dataset_dir}. "
            "Run --prepare-dataset first before preparing training configs."
        )

    output_training_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for s in SEEDS:
        run_dir = output_training_dir / f"M1_variant/seed_{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "tb_mortal").mkdir(parents=True, exist_ok=True)

        config_dict = generate_m1_training_config(
            seed=s,
            output_run_dir=run_dir,
            m1_index_path=m1_idx_path,
            m1_mapping_path=m1_map_path,
            m1_labels_path=m1_lbl_path,
        )

        config_path = run_dir / "config.toml"
        # Dump to TOML
        try:
            import tomli_w
            with open(config_path, "wb") as f:
                tomli_w.dump(config_dict, f)
        except ImportError:
            import toml
            with open(config_path, "w", encoding="utf-8") as f:
                toml.dump(config_dict, f)

        cmd = build_training_command(
            seed=s,
            config_path=config_path,
            parent_path=K0_70K_PATH,
            run_dir=run_dir,
        )

        runs.append({
            "seed": s,
            "route": "M1_variant",
            "run_dir": str(run_dir.resolve()),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "command": cmd,
            "command_str": shlex.join(cmd),
        })

    training_manifest = {
        "schema": "keqing.mortal.m1_training_manifest.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_index_sha256": sha256_file(m1_idx_path),
        "parent_checkpoint": {
            "path": str(K0_70K_PATH.resolve()),
            "sha256": sha256_file(K0_70K_PATH),
        },
        "runs": runs,
    }

    t_manifest_path = output_training_dir / "training_manifest.json"
    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Training manifest created at {t_manifest_path} (SHA256: {sha256_file(t_manifest_path)})")
    return training_manifest


def execute_training_for_seed(
    seed: int,
    training_dir: Path = M1_TRAINING_DIR,
    dataset_dir: Path = M1_DATASET_DIR,
    confirmation_token: str | None = None,
) -> None:
    """Execute training for a single seed, strictly checking authorization and non-empty output."""
    if not TRAINING_AUTHORIZED:
        raise AuthorizationError(
            f"M1 training is NOT authorized. Cannot execute seed {seed}."
        )

    t_manifest_path = training_dir / "training_manifest.json"
    if not t_manifest_path.exists():
        raise ContractError(f"Training manifest not found: {t_manifest_path}")

    with open(t_manifest_path, "r", encoding="utf-8") as f:
        t_manifest = json.load(f)

    run_info = next((r for r in t_manifest.get("runs", []) if r["seed"] == seed), None)
    if not run_info:
        raise ContractError(f"Seed {seed} not found in training manifest {t_manifest_path}")

    run_dir = Path(run_info["run_dir"])
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        raise ContractError(
            f"Run checkpoints directory is not empty: {ckpt_dir}. Automatic resume/overwrite is prohibited."
        )

    # Verify config SHA hasn't drifted
    config_path = Path(run_info["config_path"])
    current_config_sha = sha256_file(config_path)
    if current_config_sha != run_info["config_sha256"]:
        raise ContractError(f"Config SHA mismatch for seed {seed}: {current_config_sha} vs expected {run_info['config_sha256']}")

    # Verify parent checkpoint SHA
    parent_sha = sha256_file(K0_70K_PATH)
    if parent_sha != K0_70K_SHA256:
        raise ContractError(f"Parent K0_70k SHA mismatch: {parent_sha} vs {K0_70K_SHA256}")

    cmd = run_info["command"]
    print(f"Launching training for seed {seed}...")
    print(f"Command: {shlex.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed} with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prepare-dataset", action="store_true", help="Prepare 12000 M1 dataset and manifest")
    mode_group.add_argument("--prepare-training", action="store_true", help="Prepare configs and training manifest")
    mode_group.add_argument("--execute", action="store_true", help="Execute training for given seed")
    mode_group.add_argument("--status", action="store_true", help="Print M1 training pipeline status")

    parser.add_argument("--seed", type=int, choices=list(SEEDS), default=None, help="Training seed (required with --execute)")
    parser.add_argument("--confirmation-token", type=str, default=None, help="Execution confirmation token")
    parser.add_argument("--dataset-dir", type=Path, default=M1_DATASET_DIR, help="Dataset directory")
    parser.add_argument("--training-dir", type=Path, default=M1_TRAINING_DIR, help="Training root directory")
    args = parser.parse_args()

    if args.prepare_dataset:
        prepare_m1_dataset(output_dir=args.dataset_dir)
    elif args.prepare_training:
        prepare_training_manifest(dataset_dir=args.dataset_dir, output_training_dir=args.training_dir)
    elif args.execute:
        if args.seed is None:
            raise ValueError("--seed is required when --execute is specified")
        execute_training_for_seed(
            seed=args.seed,
            training_dir=args.training_dir,
            dataset_dir=args.dataset_dir,
            confirmation_token=args.confirmation_token,
        )
    elif args.status:
        print(f"M1 Experiment ID: {M1_EXPERIMENT_ID}")
        print(f"Dataset preparation authorized: {DATASET_PREPARATION_AUTHORIZED}")
        print(f"Training authorized: {TRAINING_AUTHORIZED}")


if __name__ == "__main__":
    main()
