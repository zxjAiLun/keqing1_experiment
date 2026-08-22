#!/usr/bin/env python3
"""Launcher for M1 ext_mortal 12,000-hanchan mixed expansion training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.m1_dataset_contract_2026_08 import (
    ARCHIVE_STEPS,
    K0_70K_PATH,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    REPO_ROOT,
    SEEDS,
    START_STEP,
    TARGET_STEP,
    build_m1_dataset_files,
    generate_m1_training_config,
    sha256_file,
    validate_checkpoints,
    validate_m1_dataset_integrity,
)

OFFLINE_TRAINER_SCRIPT = REPO_ROOT / "training/run_mortal_dqn_offline.py"
DATASET_PREP_DIR = REPO_ROOT / "artifacts/experiments/M1_ext_mixed_expansion_2026_08/dataset_prep_2026_08"


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


def prepare_m1_dataset() -> tuple[Path, Path]:
    """Build and validate M1 concatenated 12000 file index and labels."""
    print("Building M1 dataset (6000 M0 + 6000 D1 ext_mortal perspectives)...")
    m1_idx_path, m1_lbl_path = build_m1_dataset_files(DATASET_PREP_DIR)
    
    print(f"Validating dataset integrity of {m1_idx_path}...")
    integrity = validate_m1_dataset_integrity(m1_idx_path)
    print(f"Integrity check PASS: {json.dumps(integrity, indent=2)}")

    return m1_idx_path, m1_lbl_path


def run_training_for_seed(
    seed: int,
    m1_idx_path: Path,
    m1_lbl_path: Path,
    execute: bool = False,
) -> list[str]:
    """Prepare config and optionally execute training for a single seed."""
    run_dir = M1_TRAINING_DIR / f"M1_variant/seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb_mortal").mkdir(parents=True, exist_ok=True)

    config_dict = generate_m1_training_config(
        seed=seed,
        output_run_dir=run_dir,
        m1_index_path=m1_idx_path,
        m1_labels_path=m1_lbl_path,
    )
    
    config_path = run_dir / "config.toml"
    # Dump to TOML
    # Custom simple TOML dumper or python toml
    try:
        import tomli_w
        with open(config_path, "wb") as f:
            tomli_w.dump(config_dict, f)
    except ImportError:
        # Fallback python manual TOML serializer
        import toml
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(config_dict, f)

    cmd = build_training_command(
        seed=seed,
        config_path=config_path,
        parent_path=K0_70K_PATH,
        run_dir=run_dir,
    )

    print(f"\nPrepared training for seed {seed}:")
    print(f"Config: {config_path}")
    print(f"Command: {shlex.join(cmd)}")

    if execute:
        print(f"Executing training for seed {seed}...")
        res = subprocess.run(cmd, check=True)
        if res.returncode != 0:
            raise RuntimeError(f"Training failed for seed {seed} with return code {res.returncode}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dataset", action="store_true", help="Build M1 file index and train labels")
    parser.add_argument("--seed", type=int, choices=list(SEEDS), default=None, help="Train specific seed")
    parser.add_argument("--execute", action="store_true", help="Actually launch subprocess training execution")
    args = parser.parse_args()

    ckpt_ok, ckpt_records = validate_checkpoints()
    if not ckpt_ok:
        raise RuntimeError(f"Parent / baseline checkpoints missing or corrupt: {json.dumps(ckpt_records, indent=2)}")

    m1_idx_path = DATASET_PREP_DIR / "file_index_m1.pth"
    m1_lbl_path = DATASET_PREP_DIR / "m1_train_labels.txt"

    if args.build_dataset or not m1_idx_path.exists() or not m1_lbl_path.exists():
        m1_idx_path, m1_lbl_path = prepare_m1_dataset()

    target_seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for s in target_seeds:
        run_training_for_seed(s, m1_idx_path, m1_lbl_path, execute=args.execute)


if __name__ == "__main__":
    main()
