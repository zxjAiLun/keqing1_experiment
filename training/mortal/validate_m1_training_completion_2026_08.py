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
    K0_70K_SHA256,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    REPO_ROOT,
    SEEDS,
    START_STEP,
    TARGET_STEP,
    ContractError,
    sha256_file,
)


def validate_single_run_completion(
    seed: int,
    run_dir: Path,
    expected_dataset_index_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify final 72000 checkpoint and all intermediate step archives for a single seed."""
    final_path = run_dir / "checkpoints" / f"mortal_{TARGET_STEP}.pth"
    if not final_path.exists():
        raise FileNotFoundError(f"Final checkpoint missing for seed {seed}: {final_path}")

    # Inspect checkpoint
    state = torch.load(final_path, weights_only=False, map_location="cpu")
    
    # 1. Steps check
    steps = state.get("steps")
    if steps != TARGET_STEP:
        raise ContractError(f"Seed {seed} final checkpoint steps is {steps}, expected {TARGET_STEP}")

    trained_steps = steps - START_STEP
    if trained_steps != 2000:
        raise ContractError(f"Seed {seed} trained optimizer steps is {trained_steps}, expected 2000")

    # 2. Check finiteness of network weights
    for k, v in state.get("mortal", {}).items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            raise ContractError(f"Seed {seed} mortal weight {k} contains NaN/Inf values")
    for k, v in state.get("current_dqn", {}).items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            raise ContractError(f"Seed {seed} dqn weight {k} contains NaN/Inf values")
    for k, v in state.get("aux", {}).items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            raise ContractError(f"Seed {seed} aux weight {k} contains NaN/Inf values")

    # 3. Check initialization contract
    init_contract = state.get("initialization", {})
    if init_contract:
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

    # 4. Check training config in state
    cfg = state.get("config", {})
    if cfg:
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

    # 5. Check data_stream block
    data_stream = state.get("data_stream", {})
    if data_stream:
        if data_stream.get("data_seed") != seed:
            raise ContractError(f"Seed {seed} data_seed mismatch: {data_stream.get('data_seed')}")
        if data_stream.get("batches_consumed") != 2000:
            raise ContractError(f"Seed {seed} batches_consumed mismatch: {data_stream.get('batches_consumed')}")
        if data_stream.get("samples_consumed") != 1024000:
            raise ContractError(f"Seed {seed} samples_consumed mismatch: {data_stream.get('samples_consumed')}")
        if data_stream.get("dataset_file_count") not in (12000, 10):
            raise ContractError(f"Seed {seed} dataset_file_count mismatch: {data_stream.get('dataset_file_count')}")

    # 6. Check training_contract schema
    t_contract = state.get("training_contract", {})
    if t_contract:
        if t_contract.get("schema") != "keqing.mortal.training_contract.v2":
            raise ContractError(f"Seed {seed} training contract schema mismatch: {t_contract.get('schema')}")
        if expected_dataset_index_sha256 and t_contract.get("dataset_file_index_sha256") != expected_dataset_index_sha256:
            raise ContractError(f"Seed {seed} dataset file_index SHA mismatch")

    # 7. Check all archive steps
    archives = []
    for arch_step in ARCHIVE_STEPS:
        p = run_dir / "checkpoints" / f"mortal_{arch_step}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Archive step {arch_step} missing for seed {seed}: {p}")
        arch_state = torch.load(p, weights_only=False, map_location="cpu")
        arch_steps = arch_state.get("steps")
        if arch_steps != arch_step:
            raise ContractError(f"Archive {p.name} payload steps is {arch_steps}, expected {arch_step}")

        # Check finiteness of archive model
        for k, v in arch_state.get("mortal", {}).items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise ContractError(f"Archive {p.name} mortal weight {k} contains NaN/Inf")

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
) -> dict[str, Any]:
    """Validate all 3 M1 training runs and generate closure summary."""
    runs = []
    for s in SEEDS:
        run_dir = output_dir / f"M1_variant/seed_{s}"
        run_info = validate_single_run_completion(
            s, run_dir, expected_dataset_index_sha256=expected_dataset_index_sha256
        )
        runs.append(run_info)

    closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "target_steps": TARGET_STEP,
        "runs": runs,
    }

    closure_path = output_dir / "training_completion_closure.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2, ensure_ascii=False)
        f.write("\n")

    closure_sha = sha256_file(closure_path)
    print(f"Validation PASS: Training completion closure written to {closure_path} (SHA256: {closure_sha})")
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=M1_TRAINING_DIR, help="M1 training root directory")
    args = parser.parse_args()

    validate_all_m1_runs(args.output_dir)


if __name__ == "__main__":
    main()
