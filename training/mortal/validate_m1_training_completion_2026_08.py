#!/usr/bin/env python3
"""Validate that M1 training runs reached 72,000 steps and write completion closure."""

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
    M1_CHECKPOINTS,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    REPO_ROOT,
    SEEDS,
    TARGET_STEP,
    sha256_file,
)


def validate_single_run_completion(seed: int, run_dir: Path) -> dict[str, Any]:
    """Verify final 72000 checkpoint and all intermediate step archives for a single seed."""
    final_path = run_dir / "checkpoints" / f"mortal_{TARGET_STEP}.pth"
    if not final_path.exists():
        raise FileNotFoundError(f"Final checkpoint missing for seed {seed}: {final_path}")

    # Inspect checkpoint
    state = torch.load(final_path, weights_only=False, map_location="cpu")
    step = state.get("step")
    if step != TARGET_STEP:
        raise ValueError(f"Seed {seed} final checkpoint step is {step}, expected {TARGET_STEP}")

    # Check finiteness of network weights
    for k, v in state.get("mortal", {}).items():
        if not torch.isfinite(v).all():
            raise ValueError(f"Seed {seed} mortal weight {k} contains NaN/Inf values")
    for k, v in state.get("current_dqn", {}).items():
        if not torch.isfinite(v).all():
            raise ValueError(f"Seed {seed} dqn weight {k} contains NaN/Inf values")

    # Check all archive steps
    archives = []
    for arch_step in ARCHIVE_STEPS:
        p = run_dir / "checkpoints" / f"mortal_{arch_step}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Archive step {arch_step} missing for seed {seed}: {p}")
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
        "final_checkpoint_path": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256_file(final_path),
        "final_checkpoint_size": final_path.stat().st_size,
        "archives": archives,
    }


def validate_all_m1_runs(output_dir: Path = M1_TRAINING_DIR) -> dict[str, Any]:
    """Validate all 3 M1 training runs and generate closure summary."""
    runs = []
    for s in SEEDS:
        run_dir = output_dir / f"M1_variant/seed_{s}"
        run_info = validate_single_run_completion(s, run_dir)
        runs.append(run_info)

    closure = {
        "schema": "keqing.mortal.m1_training_completion_closure.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "runs": runs,
    }

    closure_path = output_dir / "training_completion_closure.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Validation PASS: Training completion closure written to {closure_path} (SHA256: {sha256_file(closure_path)})")
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=M1_TRAINING_DIR, help="M1 training root directory")
    args = parser.parse_args()

    validate_all_m1_runs(args.output_dir)


if __name__ == "__main__":
    main()
