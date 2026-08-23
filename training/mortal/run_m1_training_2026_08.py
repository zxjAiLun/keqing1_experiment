#!/usr/bin/env python3
"""Fail-closed launcher and dataset preparer for M1 ext_mortal 12,000-hanchan expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
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
)

OFFLINE_TRAINER_SCRIPT = REPO_ROOT / "training/run_mortal_dqn_offline.py"

# Authorization flags (fail closed by default)
DATASET_PREPARATION_AUTHORIZED = True
APPROVED_M1_IMPLEMENTATION_COMMIT = "7055675a866df4cae1d0c61217840bd084a4375a"
AUTHORIZED_PREREG_SHA256 = "aa150aa1c6660428a32ad906bfaf24c05ce644dd28cdc47138c6a7d2944f4e08"

TRAINING_AUTHORIZED = False
AUTHORIZED_DATASET_MANIFEST_SHA256 = None
AUTHORIZED_DATASET_INDEX_SHA256 = None
AUTHORIZED_PLAYER_MAPPING_SHA256 = None
AUTHORIZED_TRAINING_PLAN_SHA256 = None
AUTHORIZED_TRAINING_PREFLIGHT_SHA256 = None


class AuthorizationError(RuntimeError):
    """Raised when an action is requested without required authorization."""


def compute_confirmation_token(*, seed: int, approved_commit: str, preflight_sha256: str) -> str:
    """Compute deterministic confirmation token for training execution."""
    payload = f"{seed}:{approved_commit}:{preflight_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


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
    enforce_frozen_source_sha: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """Build and validate M1 concatenated 12000 file index and manifest."""
    if require_authorization:
        if output_dir.resolve() != M1_DATASET_DIR.resolve():
            raise ContractError(
                f"Formal dataset preparation requires canonical path: {M1_DATASET_DIR}, got {output_dir}"
            )
        if not DATASET_PREPARATION_AUTHORIZED:
            raise AuthorizationError(
                "M1 dataset preparation is NOT authorized. "
                "Formal dataset creation requires review and an authorization-only commit."
            )
        if not APPROVED_M1_IMPLEMENTATION_COMMIT:
            raise AuthorizationError("APPROVED_M1_IMPLEMENTATION_COMMIT must be set when dataset prep is authorized.")
        if not AUTHORIZED_PREREG_SHA256:
            raise AuthorizationError("AUTHORIZED_PREREG_SHA256 must be set when dataset prep is authorized.")
        current_prereg_sha = sha256_file(PREREG_PATH)
        if current_prereg_sha != AUTHORIZED_PREREG_SHA256:
            raise AuthorizationError(f"Prereg SHA mismatch: {current_prereg_sha} vs expected {AUTHORIZED_PREREG_SHA256}")

    print(f"Building M1 dataset in {output_dir}...")
    m1_idx, m1_map, m1_lbl, manifest = build_m1_dataset_files(
        output_dir,
        enforce_frozen_source_sha=enforce_frozen_source_sha,
        approved_implementation_commit=APPROVED_M1_IMPLEMENTATION_COMMIT,
    )
    print(f"Dataset manifest generated at {manifest} (SHA256: {sha256_file(manifest)})")
    return m1_idx, m1_map, m1_lbl, manifest


def prepare_training_manifest(
    dataset_dir: Path = M1_DATASET_DIR,
    output_training_dir: Path = M1_TRAINING_DIR,
) -> dict[str, Any]:
    """Prepare and freeze training configs and execution manifest for all 3 seeds."""
    t_manifest_path = output_training_dir / "training_manifest.json"
    t_preflight_path = output_training_dir / "training_preflight.json"

    if t_manifest_path.exists() or t_preflight_path.exists():
        raise ContractError(
            f"Training manifest or preflight already exists in {output_training_dir}. Refusing to overwrite."
        )

    manifest_path = dataset_dir / "dataset_manifest.json"
    m1_idx_path = dataset_dir / "file_index_m1.pth"
    m1_map_path = dataset_dir / "player_names_by_file.json"
    m1_lbl_path = dataset_dir / "player_names.txt"

    if not manifest_path.exists() or not m1_idx_path.exists() or not m1_map_path.exists() or not m1_lbl_path.exists():
        raise ContractError(
            f"Dataset closure is missing at {dataset_dir}. "
            "Run --prepare-dataset first before preparing training configs."
        )

    # Re-verify K0 parent SHA
    k0_sha = sha256_file(K0_70K_PATH)
    if k0_sha != K0_70K_SHA256:
        raise ContractError(f"Parent K0_70k SHA mismatch: {k0_sha} vs {K0_70K_SHA256}")

    output_training_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for s in SEEDS:
        run_dir = output_training_dir / f"M1_variant/seed_{s}"
        config_path = run_dir / "config.toml"
        if config_path.exists():
            raise ContractError(f"Config already exists at {config_path}. Refusing to overwrite.")

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
        "player_mapping_sha256": sha256_file(m1_map_path),
        "player_names_sha256": sha256_file(m1_lbl_path),
        "parent_checkpoint": {
            "path": str(K0_70K_PATH.resolve()),
            "sha256": k0_sha,
        },
        "trainer_source": {
            "path": str(OFFLINE_TRAINER_SCRIPT.resolve()),
            "sha256": sha256_file(OFFLINE_TRAINER_SCRIPT),
            "blob_oid": git_blob_oid(OFFLINE_TRAINER_SCRIPT),
        },
        "runs": runs,
    }

    with open(t_manifest_path, "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    preflight = {
        "schema": "keqing.mortal.m1_training_preflight.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_manifest_sha256": sha256_file(t_manifest_path),
        "git": git_info(),
    }
    with open(t_preflight_path, "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Training manifest created at {t_manifest_path} (SHA256: {sha256_file(t_manifest_path)})")
    print(f"Training preflight created at {t_preflight_path} (SHA256: {sha256_file(t_preflight_path)})")
    return training_manifest


def execute_training_for_seed(
    seed: int,
    training_dir: Path = M1_TRAINING_DIR,
    dataset_dir: Path = M1_DATASET_DIR,
    confirmation_token: str | None = None,
    enforce_canonical_paths: bool = True,
) -> None:
    """Execute training for a single seed, strictly checking authorization, SHA bindings, and non-empty output."""
    if not TRAINING_AUTHORIZED:
        raise AuthorizationError(
            f"M1 training is NOT authorized. Cannot execute seed {seed}."
        )

    if enforce_canonical_paths:
        if training_dir.resolve() != M1_TRAINING_DIR.resolve():
            raise ContractError(f"Non-canonical training directory in formal execute: {training_dir}")
        if dataset_dir.resolve() != M1_DATASET_DIR.resolve():
            raise ContractError(f"Non-canonical dataset directory in formal execute: {dataset_dir}")

    # Check all authorized SHA constants are non-empty
    if not APPROVED_M1_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("APPROVED_M1_IMPLEMENTATION_COMMIT is required for training execution")
    if not AUTHORIZED_DATASET_MANIFEST_SHA256:
        raise AuthorizationError("AUTHORIZED_DATASET_MANIFEST_SHA256 is required for training execution")
    if not AUTHORIZED_DATASET_INDEX_SHA256:
        raise AuthorizationError("AUTHORIZED_DATASET_INDEX_SHA256 is required for training execution")
    if not AUTHORIZED_PLAYER_MAPPING_SHA256:
        raise AuthorizationError("AUTHORIZED_PLAYER_MAPPING_SHA256 is required for training execution")
    if not AUTHORIZED_TRAINING_PLAN_SHA256:
        raise AuthorizationError("AUTHORIZED_TRAINING_PLAN_SHA256 is required for training execution")
    if not AUTHORIZED_TRAINING_PREFLIGHT_SHA256:
        raise AuthorizationError("AUTHORIZED_TRAINING_PREFLIGHT_SHA256 is required for training execution")

    # Verify actual file SHAs match authorized bindings
    ds_manifest = dataset_dir / "dataset_manifest.json"
    ds_index = dataset_dir / "file_index_m1.pth"
    ds_map = dataset_dir / "player_names_by_file.json"
    t_manifest_path = training_dir / "training_manifest.json"
    t_preflight_path = training_dir / "training_preflight.json"

    if sha256_file(ds_manifest) != AUTHORIZED_DATASET_MANIFEST_SHA256:
        raise ContractError("Dataset manifest SHA does not match authorized binding")
    if sha256_file(ds_index) != AUTHORIZED_DATASET_INDEX_SHA256:
        raise ContractError("Dataset index SHA does not match authorized binding")
    if sha256_file(ds_map) != AUTHORIZED_PLAYER_MAPPING_SHA256:
        raise ContractError("Player mapping SHA does not match authorized binding")
    if sha256_file(t_manifest_path) != AUTHORIZED_TRAINING_PLAN_SHA256:
        raise ContractError("Training manifest SHA does not match authorized binding")
    if sha256_file(t_preflight_path) != AUTHORIZED_TRAINING_PREFLIGHT_SHA256:
        raise ContractError("Training preflight SHA does not match authorized binding")

    # Verify confirmation token
    expected_token = compute_confirmation_token(
        seed=seed,
        approved_commit=APPROVED_M1_IMPLEMENTATION_COMMIT,
        preflight_sha256=AUTHORIZED_TRAINING_PREFLIGHT_SHA256,
    )
    if confirmation_token != expected_token:
        raise AuthorizationError(f"Invalid confirmation token: {confirmation_token}, expected {expected_token}")

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

    # Reconstruct expected command and verify equality
    reconstructed_cmd = build_training_command(
        seed=seed,
        config_path=config_path,
        parent_path=K0_70K_PATH,
        run_dir=run_dir,
    )
    if reconstructed_cmd != run_info["command"]:
        raise ContractError(f"Command mismatch for seed {seed}: {reconstructed_cmd} vs manifest {run_info['command']}")

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
        if args.dataset_dir.resolve() != M1_DATASET_DIR.resolve():
            raise ContractError(
                f"Formal --prepare-dataset requires canonical directory: {M1_DATASET_DIR}, got {args.dataset_dir}"
            )
        prepare_m1_dataset(output_dir=M1_DATASET_DIR)
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
