#!/usr/bin/env python3
"""Validate that M1 training runs reached 72,000 steps and write formal completion closure."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

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
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    SEEDS,
    START_STEP,
    TARGET_STEP,
    ContractError,
    git_blob_oid,
    sha256_file,
)


def validate_single_run_completion(
    seed: int,
    run_dir: Path,
    expected_dataset_index_sha256: str,
    expected_player_mapping_sha256: str,
    expected_dataset_file_count: int = 12000,
) -> dict[str, Any]:
    """Verify final 72000 checkpoint and all intermediate step archives for a single seed (fail-closed)."""
    final_path = run_dir / "checkpoints" / f"mortal_{TARGET_STEP}.pth"
    if not final_path.exists():
        raise FileNotFoundError(f"Final checkpoint missing for seed {seed}: {final_path}")

    state = torch.load(final_path, weights_only=False, map_location="cpu")

    # 1. Steps check
    if "steps" not in state:
        raise ContractError(f"Seed {seed} checkpoint missing 'steps' field")
    steps = state["steps"]
    if steps != TARGET_STEP:
        raise ContractError(f"Seed {seed} final checkpoint steps is {steps}, expected {TARGET_STEP}")

    # 2. Check initialization contract (REQUIRED)
    if "initialization" not in state or not isinstance(state["initialization"], dict):
        raise ContractError(f"Seed {seed} checkpoint missing required 'initialization' block")
    init_contract = state["initialization"]

    mode = init_contract.get("mode")
    if mode != "weights_plus_optimizer_warm_start":
        raise ContractError(f"Seed {seed} initialization mode is {mode}, expected weights_plus_optimizer_warm_start")
    p_sha = init_contract.get("parent_sha256")
    if p_sha != K0_70K_SHA256:
        raise ContractError(f"Seed {seed} parent_sha256 is {p_sha}, expected {K0_70K_SHA256}")
    if init_contract.get("parent_steps") != START_STEP or init_contract.get("initial_steps") != START_STEP:
        raise ContractError(f"Seed {seed} initial steps mismatch in initialization block")
    if init_contract.get("optimizer") != "preserved":
        raise ContractError(f"Seed {seed} optimizer is {init_contract.get('optimizer')}, expected preserved")
    if init_contract.get("optimizer_checkpoint_sha256") != K0_70K_SHA256:
        raise ContractError(f"Seed {seed} optimizer_checkpoint_sha256 mismatch")
    if init_contract.get("scheduler") != "fresh" or init_contract.get("scaler") != "fresh":
        raise ContractError(f"Seed {seed} scheduler or scaler not fresh")

    trained_steps = steps - init_contract["initial_steps"]
    if trained_steps != 2000:
        raise ContractError(f"Seed {seed} trained optimizer steps is {trained_steps}, expected 2000")

    # 3. Check finiteness of required network weights (mortal, current_dqn, aux_net)
    for net_key in ("mortal", "current_dqn", "aux_net"):
        if net_key not in state or not isinstance(state[net_key], dict):
            raise ContractError(f"Seed {seed} checkpoint missing required state dict for '{net_key}'")
        for k, v in state[net_key].items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise ContractError(f"Seed {seed} {net_key} weight {k} contains NaN/Inf values")

    # 4. Check training config in state (REQUIRED)
    if "config" not in state or not isinstance(state["config"], dict):
        raise ContractError(f"Seed {seed} checkpoint missing required 'config' block")
    cfg = state["config"]
    if cfg.get("objective", {}).get("mode") != "behavior_action_mc":
        raise ContractError(f"Seed {seed} objective mode mismatch: {cfg.get('objective')}")
    if cfg.get("reward", {}).get("mode") != "final_rank_mc":
        raise ContractError(f"Seed {seed} reward mode mismatch: {cfg.get('reward')}")
    if cfg.get("cql", {}).get("min_q_weight") != 5.0:
        raise ContractError(f"Seed {seed} CQL min_q_weight mismatch: {cfg.get('cql')}")
    if cfg.get("aux", {}).get("next_rank_weight") != 0.2:
        raise ContractError(f"Seed {seed} aux next_rank_weight mismatch: {cfg.get('aux')}")
    if cfg.get("control", {}).get("batch_size") != 512:
        raise ContractError(f"Seed {seed} batch size mismatch: {cfg.get('control')}")
    if cfg.get("control", {}).get("enable_amp") is not False:
        raise ContractError(f"Seed {seed} AMP mismatch: {cfg.get('control')}")

    # 5. Check data_stream block (REQUIRED)
    if "data_stream" not in state or not isinstance(state["data_stream"], dict):
        raise ContractError(f"Seed {seed} checkpoint missing required 'data_stream' block")
    data_stream = state["data_stream"]
    if data_stream.get("data_seed") != seed:
        raise ContractError(f"Seed {seed} data_seed mismatch: {data_stream.get('data_seed')}")
    if data_stream.get("batches_consumed") != 2000:
        raise ContractError(f"Seed {seed} batches_consumed mismatch: {data_stream.get('batches_consumed')}")
    if data_stream.get("samples_consumed") != 1024000:
        raise ContractError(f"Seed {seed} samples_consumed mismatch: {data_stream.get('samples_consumed')}")
    if data_stream.get("dataset_file_count") != expected_dataset_file_count:
        raise ContractError(f"Seed {seed} dataset_file_count is {data_stream.get('dataset_file_count')}, expected {expected_dataset_file_count}")
    if data_stream.get("num_workers") != 0:
        raise ContractError(f"Seed {seed} num_workers mismatch: {data_stream.get('num_workers')}")

    # 6. Check training_contract schema (REQUIRED)
    if "training_contract" not in state or not isinstance(state["training_contract"], dict):
        raise ContractError(f"Seed {seed} checkpoint missing required 'training_contract' block")
    t_contract = state["training_contract"]
    if t_contract.get("schema") != "keqing.mortal.training_contract.v2":
        raise ContractError(f"Seed {seed} training contract schema mismatch: {t_contract.get('schema')}")

    ds_contract = t_contract.get("dataset", {})
    if ds_contract.get("file_index_sha256") != expected_dataset_index_sha256:
        raise ContractError(f"Seed {seed} dataset file_index SHA mismatch in training_contract: {ds_contract.get('file_index_sha256')} vs expected {expected_dataset_index_sha256}")
    if ds_contract.get("player_names_by_file_sha256") != expected_player_mapping_sha256:
        raise ContractError(f"Seed {seed} player_names_by_file SHA mismatch in training_contract: {ds_contract.get('player_names_by_file_sha256')} vs expected {expected_player_mapping_sha256}")
    if ds_contract.get("mapped_label_counts") != {"ext_mortal": expected_dataset_file_count}:
        raise ContractError(f"Seed {seed} mapped_label_counts mismatch: {ds_contract.get('mapped_label_counts')}")

    # 7. Check all archive steps (FAIL CLOSED on missing state dicts)
    archives = []
    for arch_step in ARCHIVE_STEPS:
        p = run_dir / "checkpoints" / f"mortal_{arch_step}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Archive step {arch_step} missing for seed {seed}: {p}")
        arch_state = torch.load(p, weights_only=False, map_location="cpu")
        arch_steps = arch_state.get("steps")
        if arch_steps != arch_step:
            raise ContractError(f"Archive {p.name} payload steps is {arch_steps}, expected {arch_step}")

        # Check required state dicts exist and weights are finite
        for net_key in ("mortal", "current_dqn", "aux_net"):
            if net_key not in arch_state or not isinstance(arch_state[net_key], dict):
                raise ContractError(f"Archive {p.name} missing required state dict for '{net_key}'")
            for k, v in arch_state[net_key].items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                    raise ContractError(f"Archive {p.name} {net_key} weight {k} contains NaN/Inf")

        archives.append({
            "step": arch_step,
            "path": str(p.resolve()),
            "sha256": sha256_file(p),
            "size": p.stat().st_size,
        })

    return {
        "route": "M1_variant",
        "training_seed": seed,
        "label": f"M1_CURRENT_{seed}",
        "steps": TARGET_STEP,
        "trained_optimizer_steps": trained_steps,
        "data_seed": data_stream.get("data_seed"),
        "batches_consumed": data_stream.get("batches_consumed"),
        "samples_consumed": data_stream.get("samples_consumed"),
        "dataset_file_count": data_stream.get("dataset_file_count"),
        "training_contract_dataset": {
            "file_index_sha256": ds_contract.get("file_index_sha256"),
            "player_names_by_file_sha256": ds_contract.get("player_names_by_file_sha256"),
            "mapped_label_counts": ds_contract.get("mapped_label_counts"),
        },
        "final_checkpoint_path": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256_file(final_path),
        "final_checkpoint_size": final_path.stat().st_size,
        "archives": archives,
        "all_tensors_finite": True,
    }


def validate_all_m1_runs(
    output_dir: Path = M1_TRAINING_DIR,
    enforce_canonical_paths: bool = True,
    expected_dataset_file_count: int = 12000,
) -> dict[str, Any]:
    """Validate all 3 M1 training runs from frozen training manifest/preflight and generate closure summary."""
    if enforce_canonical_paths and output_dir.resolve() != M1_TRAINING_DIR.resolve():
        raise ContractError(f"Formal training completion requires canonical directory: {M1_TRAINING_DIR}")

    closure_path = output_dir / "training_completion_closure.json"
    if closure_path.exists():
        raise ContractError(f"Training completion closure already exists: {closure_path}. Refusing to overwrite.")

    t_manifest_path = output_dir / "training_manifest.json"
    t_preflight_path = output_dir / "training_preflight.json"

    if not t_manifest_path.exists():
        raise ContractError(f"Training manifest is missing at {t_manifest_path}. Formal completion requires frozen training manifest.")
    if not t_preflight_path.exists():
        raise ContractError(f"Training preflight is missing at {t_preflight_path}. Formal completion requires frozen training preflight.")

    # 1. Parse and validate training manifest
    with open(t_manifest_path, "r", encoding="utf-8") as f:
        t_manifest = json.load(f)

    if t_manifest.get("schema") != "keqing.mortal.m1_training_manifest.v1":
        raise ContractError(f"Training manifest schema mismatch: {t_manifest.get('schema')}")
    if t_manifest.get("experiment_id") != M1_EXPERIMENT_ID:
        raise ContractError(f"Training manifest experiment_id mismatch: {t_manifest.get('experiment_id')}")

    ds_block = t_manifest.get("dataset", {})
    manifest_ds_manifest_sha = ds_block.get("dataset_manifest", {}).get("sha256")
    manifest_index_sha = ds_block.get("file_index_m1", {}).get("sha256")
    manifest_mapping_sha = ds_block.get("player_names_by_file", {}).get("sha256")
    manifest_names_sha = ds_block.get("player_names", {}).get("sha256")

    if manifest_ds_manifest_sha != FROZEN_M1_DATASET_MANIFEST_SHA256:
        raise ContractError(f"Manifest dataset_manifest SHA mismatch: {manifest_ds_manifest_sha} vs frozen {FROZEN_M1_DATASET_MANIFEST_SHA256}")
    if manifest_index_sha != FROZEN_M1_DATASET_INDEX_SHA256:
        raise ContractError(f"Manifest file_index_m1 SHA mismatch: {manifest_index_sha} vs frozen {FROZEN_M1_DATASET_INDEX_SHA256}")
    if manifest_mapping_sha != FROZEN_M1_PLAYER_MAPPING_SHA256:
        raise ContractError(f"Manifest player_names_by_file SHA mismatch: {manifest_mapping_sha} vs frozen {FROZEN_M1_PLAYER_MAPPING_SHA256}")
    if manifest_names_sha != FROZEN_M1_PLAYER_NAMES_SHA256:
        raise ContractError(f"Manifest player_names SHA mismatch: {manifest_names_sha} vs frozen {FROZEN_M1_PLAYER_NAMES_SHA256}")

    if t_manifest.get("parent_checkpoint", {}).get("sha256") != K0_70K_SHA256:
        raise ContractError("Parent checkpoint SHA in training manifest does not match K0_70K_SHA256")

    approved_training_commit = t_manifest.get("trainer_source", {}).get("approved_training_implementation_commit")

    runs_in_manifest = t_manifest.get("runs", [])
    if len(runs_in_manifest) != len(SEEDS):
        raise ContractError(f"Training manifest runs count is {len(runs_in_manifest)}, expected {len(SEEDS)}")
    manifest_seeds = tuple(r["seed"] for r in runs_in_manifest)
    if manifest_seeds != SEEDS:
        raise ContractError(f"Training manifest seeds mismatch: {manifest_seeds} vs expected {SEEDS}")

    config_provenance: list[dict[str, Any]] = []
    for r in runs_in_manifest:
        cfg_p = Path(r["config_path"])
        if not cfg_p.exists():
            raise ContractError(f"Config file missing for seed {r['seed']}: {cfg_p}")
        cfg_sha = sha256_file(cfg_p)
        if cfg_sha != r["config_sha256"]:
            raise ContractError(f"Config SHA drift for seed {r['seed']}: {cfg_sha} vs manifest {r['config_sha256']}")
        config_provenance.append({
            "seed": r["seed"],
            "config_path": str(cfg_p.resolve()),
            "config_sha256": cfg_sha,
        })

    # 2. Parse and validate training preflight
    with open(t_preflight_path, "r", encoding="utf-8") as f:
        t_preflight = json.load(f)

    if t_preflight.get("schema") != "keqing.mortal.m1_training_preflight.v1":
        raise ContractError(f"Training preflight schema mismatch: {t_preflight.get('schema')}")
    if t_preflight.get("experiment_id") != M1_EXPERIMENT_ID:
        raise ContractError(f"Training preflight experiment_id mismatch: {t_preflight.get('experiment_id')}")

    actual_manifest_sha = sha256_file(t_manifest_path)
    if t_preflight.get("training_manifest_sha256") != actual_manifest_sha:
        raise ContractError(f"Preflight training_manifest_sha256 mismatch: {t_preflight.get('training_manifest_sha256')} vs actual {actual_manifest_sha}")

    if not t_preflight.get("dataset_4_sha_pass"):
        raise ContractError("Preflight dataset_4_sha_pass is not True")
    if not t_preflight.get("parent_k0_sha_pass"):
        raise ContractError("Preflight parent_k0_sha_pass is not True")
    if not t_preflight.get("configs_parsed"):
        raise ContractError("Preflight configs_parsed is not True")
    if not t_preflight.get("commands_verified"):
        raise ContractError("Preflight commands_verified is not True")
    if not t_preflight.get("run_dirs_clean"):
        raise ContractError("Preflight run_dirs_clean is not True")

    consumer_chk = t_preflight.get("trainer_consumer_check", {})
    if consumer_chk.get("files_count") != 12000 or consumer_chk.get("mappings_count") != 12000 or not consumer_chk.get("all_ext_mortal"):
        raise ContractError(f"Preflight trainer consumer check did not pass: {consumer_chk}")

    # 3. Validate all 3 runs
    runs = []
    for s in SEEDS:
        run_dir = output_dir / f"M1_variant/seed_{s}"
        run_info = validate_single_run_completion(
            s,
            run_dir,
            expected_dataset_index_sha256=manifest_index_sha,
            expected_player_mapping_sha256=manifest_mapping_sha,
            expected_dataset_file_count=expected_dataset_file_count,
        )
        runs.append(run_info)

    this_script = Path(__file__).resolve()

    closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "approved_training_implementation_commit": approved_training_commit,
        "dataset": {
            "dataset_manifest_sha256": FROZEN_M1_DATASET_MANIFEST_SHA256,
            "dataset_index_sha256": manifest_index_sha,
            "player_mapping_sha256": manifest_mapping_sha,
            "player_names_sha256": FROZEN_M1_PLAYER_NAMES_SHA256,
        },
        "training_manifest": {
            "path": str(t_manifest_path.resolve()),
            "sha256": actual_manifest_sha,
        },
        "training_preflight": {
            "path": str(t_preflight_path.resolve()),
            "sha256": sha256_file(t_preflight_path),
        },
        "configs": config_provenance,
        "parent_checkpoint": {
            "path": str(K0_70K_PATH.resolve()),
            "sha256": K0_70K_SHA256,
        },
        "validator_source": {
            "path": str(this_script),
            "content_sha256": sha256_file(this_script),
            "git_blob_oid": git_blob_oid(this_script),
        },
        "target_steps": TARGET_STEP,
        "runs": runs,
    }

    # Atomic write to avoid partial JSON
    tmp_closure_path = output_dir / "training_completion_closure.json.tmp"
    with open(tmp_closure_path, "w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_closure_path, closure_path)

    closure_sha = sha256_file(closure_path)
    print(f"Validation PASS: Training completion closure written to {closure_path} (SHA256: {closure_sha})")
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=M1_TRAINING_DIR, help="M1 training root directory")
    args = parser.parse_args()

    if args.output_dir.resolve() != M1_TRAINING_DIR.resolve():
        raise ContractError(f"Formal completion validation requires canonical directory: {M1_TRAINING_DIR}")

    validate_all_m1_runs(output_dir=M1_TRAINING_DIR)


if __name__ == "__main__":
    main()
