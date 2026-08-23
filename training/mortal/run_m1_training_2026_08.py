#!/usr/bin/env python3
"""Fail-closed launcher and dataset preparer for M1 ext_mortal 12,000-hanchan expansion."""

from __future__ import annotations

import argparse
import hashlib
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
    FROZEN_M1_DATASET_INDEX_SHA256,
    FROZEN_M1_DATASET_MANIFEST_SHA256,
    FROZEN_M1_PLAYER_MAPPING_SHA256,
    FROZEN_M1_PLAYER_NAMES_SHA256,
    K0_70K_PATH,
    K0_70K_SHA256,
    M1_DATASET_DIR,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    PREREG_PATH,
    REPO_ROOT,
    SEEDS,
    SOURCE_D1_INDEX_SHA256,
    SOURCE_M0_INDEX_SHA256,
    START_STEP,
    TARGET_STEP,
    ContractError,
    build_m1_dataset_files,
    generate_m1_training_config,
    git_blob_oid,
    git_info,
    sha256_file,
)
from training.run_mortal_dqn_offline import (
    _load_or_build_file_index,
    _load_player_names_by_file,
)

OFFLINE_TRAINER_SCRIPT = REPO_ROOT / "training/run_mortal_dqn_offline.py"

# Authorization flags (fail closed by default)
DATASET_PREPARATION_AUTHORIZED = False
APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT = "7055675a866df4cae1d0c61217840bd084a4375a"
AUTHORIZED_PREREG_SHA256 = "aa150aa1c6660428a32ad906bfaf24c05ce644dd28cdc47138c6a7d2944f4e08"

TRAINING_PREPARATION_AUTHORIZED = True
APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT = "ac3eb0a63d4951fa5144afb57e85d3f716ab35a4"
AUTHORIZED_M1_DATASET_MANIFEST_SHA256 = "206f5445544c55aaa88d909253ef5eb422274998c7e78c8b2d569d57b3c2dde4"
AUTHORIZED_M1_DATASET_INDEX_SHA256 = "3d190247fb6e16b423d786ec07bd3b0ff3cd8903306de70ba57955e45226c07f"
AUTHORIZED_M1_PLAYER_MAPPING_SHA256 = "7c1b0433a207ce1c941ff42c0d7dfbaa53087fd3968a9228927f214357164469"
AUTHORIZED_M1_PLAYER_NAMES_SHA256 = "29f5f7c619c5481352e6fe29d4c5feb9442b6d1f1cec1ea7f4f405b330ce58d0"

TRAINING_AUTHORIZED = True
AUTHORIZED_DATASET_MANIFEST_SHA256 = "206f5445544c55aaa88d909253ef5eb422274998c7e78c8b2d569d57b3c2dde4"
AUTHORIZED_DATASET_INDEX_SHA256 = "3d190247fb6e16b423d786ec07bd3b0ff3cd8903306de70ba57955e45226c07f"
AUTHORIZED_PLAYER_MAPPING_SHA256 = "7c1b0433a207ce1c941ff42c0d7dfbaa53087fd3968a9228927f214357164469"
AUTHORIZED_PLAYER_NAMES_SHA256 = "29f5f7c619c5481352e6fe29d4c5feb9442b6d1f1cec1ea7f4f405b330ce58d0"
AUTHORIZED_TRAINING_PLAN_SHA256 = "2e0fc94eb9febf3e65f8081549738a198c9e0dbe7a7f0c4bef382ed3a87a84e6"
AUTHORIZED_TRAINING_PREFLIGHT_SHA256 = "bbddbf6e603d835326f64340efa0e1b996f17bf48bd61c96adda49bf151534fa"


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
        if not APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT:
            raise AuthorizationError("APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT must be set when dataset prep is authorized.")
        if not AUTHORIZED_PREREG_SHA256:
            raise AuthorizationError("AUTHORIZED_PREREG_SHA256 must be set when dataset prep is authorized.")
        current_prereg_sha = sha256_file(PREREG_PATH)
        if current_prereg_sha != AUTHORIZED_PREREG_SHA256:
            raise AuthorizationError(f"Prereg SHA mismatch: {current_prereg_sha} vs expected {AUTHORIZED_PREREG_SHA256}")

    print(f"Building M1 dataset in {output_dir}...")
    m1_idx, m1_map, m1_lbl, manifest = build_m1_dataset_files(
        output_dir,
        enforce_frozen_source_sha=enforce_frozen_source_sha,
        approved_implementation_commit=APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT,
    )
    print(f"Dataset manifest generated at {manifest} (SHA256: {sha256_file(manifest)})")
    return m1_idx, m1_map, m1_lbl, manifest


def prepare_training_manifest(
    dataset_dir: Path = M1_DATASET_DIR,
    output_training_dir: Path = M1_TRAINING_DIR,
    require_authorization: bool = True,
    enforce_canonical_paths: bool = True,
) -> dict[str, Any]:
    """Prepare and freeze training configs and execution manifest for all 3 seeds with atomic publication (fail closed)."""
    if require_authorization:
        if not TRAINING_PREPARATION_AUTHORIZED:
            raise AuthorizationError(
                "M1 training preparation is NOT authorized. "
                "Formal training preparation requires dataset review and an authorization-only commit."
            )
        if not APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT:
            raise AuthorizationError("APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT is required for training preparation")
        if not AUTHORIZED_M1_DATASET_MANIFEST_SHA256:
            raise AuthorizationError("AUTHORIZED_M1_DATASET_MANIFEST_SHA256 is required for training preparation")
        if not AUTHORIZED_M1_DATASET_INDEX_SHA256:
            raise AuthorizationError("AUTHORIZED_M1_DATASET_INDEX_SHA256 is required for training preparation")
        if not AUTHORIZED_M1_PLAYER_MAPPING_SHA256:
            raise AuthorizationError("AUTHORIZED_M1_PLAYER_MAPPING_SHA256 is required for training preparation")
        if not AUTHORIZED_M1_PLAYER_NAMES_SHA256:
            raise AuthorizationError("AUTHORIZED_M1_PLAYER_NAMES_SHA256 is required for training preparation")

    if enforce_canonical_paths:
        if dataset_dir.resolve() != M1_DATASET_DIR.resolve():
            raise ContractError(f"Formal training preparation requires canonical dataset directory: {M1_DATASET_DIR}")
        if output_training_dir.resolve() != M1_TRAINING_DIR.resolve():
            raise ContractError(f"Formal training preparation requires canonical training directory: {M1_TRAINING_DIR}")

    # Check output training directory is pristine (must not exist, or must be completely empty)
    if output_training_dir.exists():
        entries = list(output_training_dir.iterdir())
        if entries:
            raise ContractError(
                f"Training directory {output_training_dir} already exists and is non-empty ({len(entries)} entries). "
                "Formal preparation requires pristine state without prior artifacts."
            )

    staging_root = output_training_dir.parent / f"{output_training_dir.name}.staging"
    if staging_root.exists():
        raise ContractError(
            f"Staging directory {staging_root} already exists. Refusing to overwrite or silently delete. Pristine state required."
        )

    manifest_path = dataset_dir / "dataset_manifest.json"
    m1_idx_path = dataset_dir / "file_index_m1.pth"
    m1_map_path = dataset_dir / "player_names_by_file.json"
    m1_lbl_path = dataset_dir / "player_names.txt"

    # Preflight 1: Check 4 dataset artifact files exist
    if not manifest_path.exists() or not m1_idx_path.exists() or not m1_map_path.exists() or not m1_lbl_path.exists():
        raise ContractError(
            f"Dataset closure is missing at {dataset_dir}. "
            "Run --prepare-dataset first before preparing training configs."
        )

    # Preflight 2: Verify dataset artifact SHAs against frozen constants AND authorized bindings
    actual_manifest_sha = sha256_file(manifest_path)
    actual_index_sha = sha256_file(m1_idx_path)
    actual_map_sha = sha256_file(m1_map_path)
    actual_lbl_sha = sha256_file(m1_lbl_path)

    if actual_manifest_sha != FROZEN_M1_DATASET_MANIFEST_SHA256:
        raise ContractError(f"Dataset manifest SHA drift: {actual_manifest_sha} vs expected {FROZEN_M1_DATASET_MANIFEST_SHA256}")
    if actual_index_sha != FROZEN_M1_DATASET_INDEX_SHA256:
        raise ContractError(f"Dataset index SHA drift: {actual_index_sha} vs expected {FROZEN_M1_DATASET_INDEX_SHA256}")
    if actual_map_sha != FROZEN_M1_PLAYER_MAPPING_SHA256:
        raise ContractError(f"Dataset player mapping SHA drift: {actual_map_sha} vs expected {FROZEN_M1_PLAYER_MAPPING_SHA256}")
    if actual_lbl_sha != FROZEN_M1_PLAYER_NAMES_SHA256:
        raise ContractError(f"Dataset player names SHA drift: {actual_lbl_sha} vs expected {FROZEN_M1_PLAYER_NAMES_SHA256}")

    if require_authorization:
        if actual_manifest_sha != AUTHORIZED_M1_DATASET_MANIFEST_SHA256:
            raise ContractError(f"Dataset manifest SHA mismatch with authorization: {actual_manifest_sha} vs {AUTHORIZED_M1_DATASET_MANIFEST_SHA256}")
        if actual_index_sha != AUTHORIZED_M1_DATASET_INDEX_SHA256:
            raise ContractError(f"Dataset index SHA mismatch with authorization: {actual_index_sha} vs {AUTHORIZED_M1_DATASET_INDEX_SHA256}")
        if actual_map_sha != AUTHORIZED_M1_PLAYER_MAPPING_SHA256:
            raise ContractError(f"Dataset player mapping SHA mismatch with authorization: {actual_map_sha} vs {AUTHORIZED_M1_PLAYER_MAPPING_SHA256}")
        if actual_lbl_sha != AUTHORIZED_M1_PLAYER_NAMES_SHA256:
            raise ContractError(f"Dataset player names SHA mismatch with authorization: {actual_lbl_sha} vs {AUTHORIZED_M1_PLAYER_NAMES_SHA256}")

    # Preflight 3: Validate dataset manifest contents
    with open(manifest_path, "r", encoding="utf-8") as f:
        ds_manifest_data = json.load(f)

    if ds_manifest_data.get("schema") != "keqing.mortal.m1_dataset_manifest.v1":
        raise ContractError(f"Dataset manifest schema mismatch: {ds_manifest_data.get('schema')}")
    if ds_manifest_data.get("experiment_id") != M1_EXPERIMENT_ID:
        raise ContractError(f"Dataset manifest experiment_id mismatch: {ds_manifest_data.get('experiment_id')}")
    if ds_manifest_data.get("implementation", {}).get("approved_implementation_commit") != APPROVED_M1_DATASET_IMPLEMENTATION_COMMIT:
        raise ContractError("Dataset manifest approved_implementation_commit mismatch")
    if ds_manifest_data.get("preregistration", {}).get("content_sha256") != AUTHORIZED_PREREG_SHA256:
        raise ContractError("Dataset manifest preregistration SHA mismatch")

    s_m0 = ds_manifest_data.get("source_m0_index", {})
    if not s_m0.get("match") or s_m0.get("count") != 6000 or s_m0.get("actual_sha256") != SOURCE_M0_INDEX_SHA256:
        raise ContractError("Source M0 index verification failed in manifest")
    s_d1 = ds_manifest_data.get("source_d1_index", {})
    if not s_d1.get("match") or s_d1.get("count") != 6000 or s_d1.get("actual_sha256") != SOURCE_D1_INDEX_SHA256:
        raise ContractError("Source D1 index verification failed in manifest")

    inv_sum = ds_manifest_data.get("inventory_summary", {})
    if (
        inv_sum.get("total_files") != 12000
        or inv_sum.get("m0_files") != 6000
        or inv_sum.get("d1_files") != 6000
        or inv_sum.get("seed_overlap") != 0
        or inv_sum.get("all_single_ext_mortal") is not True
    ):
        raise ContractError(f"Dataset inventory summary validation failed: {inv_sum}")

    ds_artifacts = ds_manifest_data.get("dataset_artifacts", {})
    if ds_artifacts.get("file_index_m1", {}).get("sha256") != actual_index_sha:
        raise ContractError("Manifest file_index_m1 SHA does not match disk actual")
    if ds_artifacts.get("player_names_by_file", {}).get("sha256") != actual_map_sha:
        raise ContractError("Manifest player_names_by_file SHA does not match disk actual")
    if ds_artifacts.get("player_names", {}).get("sha256") != actual_lbl_sha:
        raise ContractError("Manifest player_names SHA does not match disk actual")

    # Preflight 4: Re-verify K0 parent SHA
    k0_sha = sha256_file(K0_70K_PATH)
    if k0_sha != K0_70K_SHA256:
        raise ContractError(f"Parent K0_70k SHA mismatch: {k0_sha} vs {K0_70K_SHA256}")

    # Preflight 5: Real trainer consumer verification
    smoke_files = _load_or_build_file_index({"control": {"version": 4}, "dataset": {"file_index": str(m1_idx_path), "globs": []}})
    if len(smoke_files) != 12000:
        raise ContractError(f"Trainer consumer loaded {len(smoke_files)} files, expected 12000")
    smoke_mapping = _load_player_names_by_file({"dataset": {"player_names_by_file": str(m1_map_path)}})
    if not smoke_mapping or len(smoke_mapping) != 12000:
        raise ContractError(f"Trainer consumer loaded {len(smoke_mapping) if smoke_mapping else 0} mappings, expected 12000")
    if not all(v == "ext_mortal" for v in smoke_mapping.values()):
        raise ContractError("Trainer consumer mappings contain labels other than 'ext_mortal'")
    if set(smoke_files) != set(smoke_mapping.keys()):
        raise ContractError("Trainer consumer loaded files set does not equal mapping keys set")

    # Preflight 6: Git working tree cleanliness check
    g_info = git_info()
    git_status = g_info.get("status", "")
    for line in git_status.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line != "?? 1.md":
            raise ContractError(f"Git working tree is dirty during training prep: {line}")

    # Preflight 7: In-memory generation, serialization, re-parsing, and contract validation
    runs_in_memory: list[dict[str, Any]] = []
    serialized_configs: dict[int, bytes] = {}

    for s in SEEDS:
        canonical_run_dir = output_training_dir / f"M1_variant/seed_{s}"
        canonical_config_path = canonical_run_dir / "config.toml"

        config_dict = generate_m1_training_config(
            seed=s,
            output_run_dir=canonical_run_dir,
            m1_index_path=m1_idx_path,
            m1_mapping_path=m1_map_path,
            m1_labels_path=m1_lbl_path,
        )

        try:
            import tomli_w
            cfg_bytes = tomli_w.dumps(config_dict).encode("utf-8")
        except ImportError:
            import toml
            cfg_bytes = toml.dumps(config_dict).encode("utf-8")

        serialized_configs[s] = cfg_bytes

        # Re-parse serialized config bytes
        try:
            import tomllib
            parsed_cfg = tomllib.loads(cfg_bytes.decode("utf-8"))
        except ImportError:
            import tomli
            parsed_cfg = tomli.loads(cfg_bytes.decode("utf-8"))

        # Verify exact config contract keys
        ctrl = parsed_cfg.get("control", {})
        if ctrl.get("version") != 4:
            raise ContractError(f"Seed {s} config control.version mismatch: {ctrl.get('version')}")
        if ctrl.get("batch_size") != 512:
            raise ContractError(f"Seed {s} parsed config batch_size mismatch: {ctrl.get('batch_size')}")
        if ctrl.get("opt_step_every") != 1:
            raise ContractError(f"Seed {s} config opt_step_every mismatch: {ctrl.get('opt_step_every')}")
        if ctrl.get("device") != "cuda:0":
            raise ContractError(f"Seed {s} config device mismatch: {ctrl.get('device')}")
        if ctrl.get("enable_amp") is not False:
            raise ContractError(f"Seed {s} parsed config enable_amp is not False")
        if ctrl.get("state_file") != str((canonical_run_dir / "mortal.pth").resolve()):
            raise ContractError(f"Seed {s} config state_file mismatch: {ctrl.get('state_file')}")
        if ctrl.get("best_state_file") != str((canonical_run_dir / "mortal_best.pth").resolve()):
            raise ContractError(f"Seed {s} config best_state_file mismatch: {ctrl.get('best_state_file')}")
        if ctrl.get("tensorboard_dir") != str((canonical_run_dir / "tb_mortal").resolve()):
            raise ContractError(f"Seed {s} config tensorboard_dir mismatch: {ctrl.get('tensorboard_dir')}")

        ds_sec = parsed_cfg.get("dataset", {})
        if ds_sec.get("file_index") != str(m1_idx_path.resolve()):
            raise ContractError(f"Seed {s} parsed config file_index mismatch: {ds_sec.get('file_index')}")
        if ds_sec.get("player_names_by_file") != str(m1_map_path.resolve()):
            raise ContractError(f"Seed {s} parsed config player_names_by_file mismatch: {ds_sec.get('player_names_by_file')}")
        if ds_sec.get("player_names_files") != [str(m1_lbl_path.resolve())]:
            raise ContractError(f"Seed {s} parsed config player_names_files mismatch: {ds_sec.get('player_names_files')}")
        if ds_sec.get("num_workers") != 0:
            raise ContractError(f"Seed {s} parsed config num_workers mismatch: {ds_sec.get('num_workers')}")
        if ds_sec.get("reserve_ratio") != 0.0:
            raise ContractError(f"Seed {s} parsed config reserve_ratio mismatch: {ds_sec.get('reserve_ratio')}")
        if ds_sec.get("enable_augmentation") is not False or ds_sec.get("augmented_first") is not False:
            raise ContractError(f"Seed {s} parsed config data augmentation must be False")

        if parsed_cfg.get("objective", {}).get("mode") != "behavior_action_mc":
            raise ContractError(f"Seed {s} parsed config objective mismatch")
        if parsed_cfg.get("reward", {}).get("mode") != "final_rank_mc":
            raise ContractError(f"Seed {s} parsed config reward mismatch")
        if parsed_cfg.get("cql", {}).get("min_q_weight") != 5.0:
            raise ContractError(f"Seed {s} parsed config CQL min_q_weight mismatch")
        if parsed_cfg.get("aux", {}).get("next_rank_weight") != 0.2:
            raise ContractError(f"Seed {s} parsed config aux weight mismatch")

        resnet_sec = parsed_cfg.get("resnet", {})
        if resnet_sec.get("conv_channels") != 192 or resnet_sec.get("num_blocks") != 40:
            raise ContractError(f"Seed {s} parsed config resnet architecture mismatch: {resnet_sec}")

        if parsed_cfg.get("env", {}).get("gamma") != 1.0:
            raise ContractError(f"Seed {s} parsed config env.gamma mismatch: {parsed_cfg.get('env', {}).get('gamma')}")

        cmd = build_training_command(
            seed=s,
            config_path=canonical_config_path,
            parent_path=K0_70K_PATH,
            run_dir=canonical_run_dir,
        )

        cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
        runs_in_memory.append({
            "seed": s,
            "data_seed": s,
            "route": "M1_variant",
            "run_dir": str(canonical_run_dir.resolve()),
            "archive_dir": str((canonical_run_dir / "checkpoints").resolve()),
            "archive_steps": list(ARCHIVE_STEPS),
            "config_path": str(canonical_config_path.resolve()),
            "config_sha256": cfg_sha,
            "command": cmd,
            "command_str": shlex.join(cmd),
        })

    # Preflight 8: Audit commands equality
    commands_verified = True
    for r in runs_in_memory:
        reconstructed = build_training_command(
            seed=r["seed"],
            config_path=Path(r["config_path"]),
            parent_path=K0_70K_PATH,
            run_dir=Path(r["run_dir"]),
        )
        if reconstructed != r["command"]:
            commands_verified = False
            raise ContractError(f"Command verification failed for seed {r['seed']}")

    trainer_blob = git_blob_oid(OFFLINE_TRAINER_SCRIPT)
    trainer_sha = sha256_file(OFFLINE_TRAINER_SCRIPT)

    training_manifest = {
        "schema": "keqing.mortal.m1_training_manifest.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "dataset": {
            "dataset_manifest": {"path": str(manifest_path.resolve()), "sha256": actual_manifest_sha},
            "file_index_m1": {"path": str(m1_idx_path.resolve()), "sha256": actual_index_sha},
            "player_names_by_file": {"path": str(m1_map_path.resolve()), "sha256": actual_map_sha},
            "player_names": {"path": str(m1_lbl_path.resolve()), "sha256": actual_lbl_sha},
        },
        "parent_checkpoint": {
            "path": str(K0_70K_PATH.resolve()),
            "sha256": k0_sha,
        },
        "trainer_source": {
            "path": str(OFFLINE_TRAINER_SCRIPT.resolve()),
            "sha256": trainer_sha,
            "git_blob_oid": trainer_blob,
            "approved_training_implementation_commit": APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT,
        },
        "frozen_parameters": {
            "start_step": START_STEP,
            "target_step": TARGET_STEP,
            "batch_size": 512,
            "num_workers": 0,
            "enable_amp": False,
            "objective_mode": "behavior_action_mc",
            "reward_mode": "final_rank_mc",
            "cql_min_q_weight": 5.0,
            "aux_next_rank_weight": 0.2,
            "conv_channels": 192,
            "num_blocks": 40,
            "optimizer": "preserved",
            "scheduler": "fresh",
            "scaler": "fresh",
        },
        "runs": runs_in_memory,
    }

    manifest_bytes = (json.dumps(training_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    preflight = {
        "schema": "keqing.mortal.m1_training_preflight.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_manifest_sha256": manifest_sha,
        "dataset_4_sha_pass": True,
        "parent_k0_sha_pass": True,
        "configs_parsed": True,
        "trainer_consumer_check": {
            "files_count": len(smoke_files),
            "mappings_count": len(smoke_mapping),
            "all_ext_mortal": True,
            "files_mapping_symmetric": True,
        },
        "commands_verified": commands_verified,
        "run_dirs_clean": True,
        "git": g_info,
    }
    preflight_bytes = (json.dumps(preflight, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    # All validations passed: write to staging root and fsync
    staging_root.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        s_run_dir = staging_root / f"M1_variant/seed_{s}"
        s_run_dir.mkdir(parents=True, exist_ok=True)
        (s_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (s_run_dir / "tb_mortal").mkdir(parents=True, exist_ok=True)
        cfg_out_p = s_run_dir / "config.toml"
        with open(cfg_out_p, "wb") as f:
            f.write(serialized_configs[s])
            f.flush()
            os.fsync(f.fileno())

    s_manifest_p = staging_root / "training_manifest.json"
    with open(s_manifest_p, "wb") as f:
        f.write(manifest_bytes)
        f.flush()
        os.fsync(f.fileno())

    s_preflight_p = staging_root / "training_preflight.json"
    with open(s_preflight_p, "wb") as f:
        f.write(preflight_bytes)
        f.flush()
        os.fsync(f.fileno())

    # Ensure canonical output directory still does not exist before atomic replace
    if output_training_dir.exists():
        entries = list(output_training_dir.iterdir())
        if entries:
            raise ContractError(f"Target directory {output_training_dir} became non-empty before atomic rename.")
        output_training_dir.rmdir()

    os.replace(staging_root, output_training_dir)

    final_manifest_path = output_training_dir / "training_manifest.json"
    final_preflight_path = output_training_dir / "training_preflight.json"

    print(f"Training manifest created at {final_manifest_path} (SHA256: {sha256_file(final_manifest_path)})")
    print(f"Training preflight created at {final_preflight_path} (SHA256: {sha256_file(final_preflight_path)})")
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
    if not APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT is required for training execution")
    if not AUTHORIZED_DATASET_MANIFEST_SHA256:
        raise AuthorizationError("AUTHORIZED_DATASET_MANIFEST_SHA256 is required for training execution")
    if not AUTHORIZED_DATASET_INDEX_SHA256:
        raise AuthorizationError("AUTHORIZED_DATASET_INDEX_SHA256 is required for training execution")
    if not AUTHORIZED_PLAYER_MAPPING_SHA256:
        raise AuthorizationError("AUTHORIZED_PLAYER_MAPPING_SHA256 is required for training execution")
    if not AUTHORIZED_PLAYER_NAMES_SHA256:
        raise AuthorizationError("AUTHORIZED_PLAYER_NAMES_SHA256 is required for training execution")
    if not AUTHORIZED_TRAINING_PLAN_SHA256:
        raise AuthorizationError("AUTHORIZED_TRAINING_PLAN_SHA256 is required for training execution")
    if not AUTHORIZED_TRAINING_PREFLIGHT_SHA256:
        raise AuthorizationError("AUTHORIZED_TRAINING_PREFLIGHT_SHA256 is required for training execution")

    # Verify actual file SHAs match authorized bindings
    ds_manifest = dataset_dir / "dataset_manifest.json"
    ds_index = dataset_dir / "file_index_m1.pth"
    ds_map = dataset_dir / "player_names_by_file.json"
    ds_lbl = dataset_dir / "player_names.txt"
    t_manifest_path = training_dir / "training_manifest.json"
    t_preflight_path = training_dir / "training_preflight.json"

    if sha256_file(ds_manifest) != AUTHORIZED_DATASET_MANIFEST_SHA256:
        raise ContractError("Dataset manifest SHA does not match authorized binding")
    if sha256_file(ds_index) != AUTHORIZED_DATASET_INDEX_SHA256:
        raise ContractError("Dataset index SHA does not match authorized binding")
    if sha256_file(ds_map) != AUTHORIZED_PLAYER_MAPPING_SHA256:
        raise ContractError("Player mapping SHA does not match authorized binding")
    if sha256_file(ds_lbl) != AUTHORIZED_PLAYER_NAMES_SHA256:
        raise ContractError("Player names SHA does not match authorized binding")
    if sha256_file(t_manifest_path) != AUTHORIZED_TRAINING_PLAN_SHA256:
        raise ContractError("Training manifest SHA does not match authorized binding")
    if sha256_file(t_preflight_path) != AUTHORIZED_TRAINING_PREFLIGHT_SHA256:
        raise ContractError("Training preflight SHA does not match authorized binding")

    # Verify confirmation token
    expected_token = compute_confirmation_token(
        seed=seed,
        approved_commit=APPROVED_M1_TRAINING_IMPLEMENTATION_COMMIT,
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

    # Verify trainer source SHA and blob
    trainer_sha = sha256_file(OFFLINE_TRAINER_SCRIPT)
    trainer_blob = git_blob_oid(OFFLINE_TRAINER_SCRIPT)
    manifest_trainer = t_manifest.get("trainer_source", {})
    if trainer_sha != manifest_trainer.get("sha256"):
        raise ContractError(f"Trainer source SHA mismatch: {trainer_sha} vs manifest {manifest_trainer.get('sha256')}")
    if trainer_blob != manifest_trainer.get("git_blob_oid"):
        raise ContractError(f"Trainer source blob OID mismatch: {trainer_blob} vs manifest {manifest_trainer.get('git_blob_oid')}")

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
    args = parser.parse_args()

    if args.prepare_dataset:
        prepare_m1_dataset(output_dir=M1_DATASET_DIR)
    elif args.prepare_training:
        prepare_training_manifest(dataset_dir=M1_DATASET_DIR, output_training_dir=M1_TRAINING_DIR)
    elif args.execute:
        if args.seed is None:
            raise ValueError("--seed is required when --execute is specified")
        execute_training_for_seed(
            seed=args.seed,
            training_dir=M1_TRAINING_DIR,
            dataset_dir=M1_DATASET_DIR,
            confirmation_token=args.confirmation_token,
        )
    elif args.status:
        print(f"M1 Experiment ID: {M1_EXPERIMENT_ID}")
        print(f"Dataset preparation authorized: {DATASET_PREPARATION_AUTHORIZED}")
        print(f"Training preparation authorized: {TRAINING_PREPARATION_AUTHORIZED}")
        print(f"Training execution authorized: {TRAINING_AUTHORIZED}")


if __name__ == "__main__":
    main()
