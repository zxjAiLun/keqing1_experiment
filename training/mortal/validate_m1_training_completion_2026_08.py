#!/usr/bin/env python3
"""Validate that M1 training runs reached 72,000 steps and write formal completion closure."""

from __future__ import annotations

import argparse
import json
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
    expected_dataset_index_sha256: str | None = None,
    expected_player_mapping_sha256: str | None = None,
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
    if not expected_dataset_index_sha256:
        raise ContractError(f"Seed {seed} expected_dataset_index_sha256 is required for validation")
    if ds_contract.get("file_index_sha256") != expected_dataset_index_sha256:
        raise ContractError(f"Seed {seed} dataset file_index SHA mismatch in training_contract: {ds_contract.get('file_index_sha256')} vs expected {expected_dataset_index_sha256}")
    if expected_player_mapping_sha256 and ds_contract.get("player_names_by_file_sha256") != expected_player_mapping_sha256:
        raise ContractError(f"Seed {seed} player_names_by_file SHA mismatch in training_contract")
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
        "final_checkpoint_path": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256_file(final_path),
        "final_checkpoint_size": final_path.stat().st_size,
        "archives": archives,
        "all_tensors_finite": True,
    }


def validate_all_m1_runs(
    output_dir: Path = M1_TRAINING_DIR,
    expected_dataset_index_sha256: str | None = None,
    expected_player_mapping_sha256: str | None = None,
    expected_dataset_file_count: int = 12000,
) -> dict[str, Any]:
    """Validate all 3 M1 training runs and generate closure summary."""
    closure_path = output_dir / "training_completion_closure.json"
    if closure_path.exists():
        raise ContractError(f"Training completion closure already exists: {closure_path}. Refusing to overwrite.")

    t_manifest_path = output_dir / "training_manifest.json"
    t_preflight_path = output_dir / "training_preflight.json"

    if expected_dataset_index_sha256 is None:
        if t_manifest_path.exists():
            with open(t_manifest_path, "r", encoding="utf-8") as f:
                t_manifest = json.load(f)
            expected_dataset_index_sha256 = t_manifest.get("dataset", {}).get("file_index_m1", {}).get("sha256")
            expected_player_mapping_sha256 = t_manifest.get("dataset", {}).get("player_names_by_file", {}).get("sha256")
        else:
            expected_dataset_index_sha256 = FROZEN_M1_DATASET_INDEX_SHA256
            expected_player_mapping_sha256 = FROZEN_M1_PLAYER_MAPPING_SHA256

    if not expected_dataset_index_sha256:
        raise ContractError("Cannot validate training completion without expected dataset index SHA")

    runs = []
    for s in SEEDS:
        run_dir = output_dir / f"M1_variant/seed_{s}"
        run_info = validate_single_run_completion(
            s,
            run_dir,
            expected_dataset_index_sha256=expected_dataset_index_sha256,
            expected_player_mapping_sha256=expected_player_mapping_sha256,
            expected_dataset_file_count=expected_dataset_file_count,
        )
        runs.append(run_info)

    this_script = Path(__file__).resolve()

    closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "dataset": {
            "dataset_manifest_sha256": FROZEN_M1_DATASET_MANIFEST_SHA256,
            "dataset_index_sha256": expected_dataset_index_sha256,
            "player_mapping_sha256": expected_player_mapping_sha256,
            "player_names_sha256": FROZEN_M1_PLAYER_NAMES_SHA256,
        },
        "training_manifest": {
            "path": str(t_manifest_path.resolve()) if t_manifest_path.exists() else None,
            "sha256": sha256_file(t_manifest_path) if t_manifest_path.exists() else None,
        },
        "training_preflight": {
            "path": str(t_preflight_path.resolve()) if t_preflight_path.exists() else None,
            "sha256": sha256_file(t_preflight_path) if t_preflight_path.exists() else None,
        },
        "parent_k0_sha256": K0_70K_SHA256,
        "validator_source": {
            "path": str(this_script),
            "content_sha256": sha256_file(this_script),
            "git_blob_oid": git_blob_oid(this_script),
        },
        "target_steps": TARGET_STEP,
        "runs": runs,
    }

    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2, ensure_ascii=False)
        f.write("\n")

    closure_sha = sha256_file(closure_path)
    print(f"Validation PASS: Training completion closure written to {closure_path} (SHA256: {closure_sha})")
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=M1_TRAINING_DIR, help="M1 training root directory")
    parser.add_argument("--expected-dataset-index-sha", type=str, default=None, help="Expected dataset index SHA256")
    parser.add_argument("--expected-player-mapping-sha", type=str, default=None, help="Expected player mapping SHA256")
    args = parser.parse_args()

    validate_all_m1_runs(
        output_dir=args.output_dir,
        expected_dataset_index_sha256=args.expected_dataset_index_sha,
        expected_player_mapping_sha256=args.expected_player_mapping_sha,
    )


if __name__ == "__main__":
    main()
