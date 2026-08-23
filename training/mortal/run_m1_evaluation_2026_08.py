#!/usr/bin/env python3
"""Fail-closed four-player evaluation launcher for M1 ext_mortal expansion absolute promotion."""

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
    AMP,
    DEVICE,
    EXT_MORTAL_PATH,
    EXT_MORTAL_SHA256,
    GAMES_PER_SHARD,
    K0_70K_PATH,
    K0_70K_SHA256,
    M0_CURRENT_CHECKPOINTS,
    M1_EVALUATION_DIR,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    RANK_POINTS,
    REPO_ROOT,
    SEAT_MODE,
    SEED_KEY,
    SEEDS,
    SHARD_CONFIG,
    SHARDS,
    TOTAL_GAMES,
    ContractError,
    git_blob_oid,
    git_info,
    sha256_file,
    validate_all_8_checkpoints,
)

EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"

# Authorization flags (fail closed by default)
EVALUATION_AUTHORIZED = False
APPROVED_M1_IMPLEMENTATION_COMMIT = None
AUTHORIZED_TRAINING_COMPLETION_SHA256 = None
AUTHORIZED_EVALUATION_PLAN_SHA256 = None


class AuthorizationError(RuntimeError):
    """Raised when evaluation execution is attempted without required authorization."""


def build_shard_command(
    shard_id: int,
    output_dir: Path,
    m1_checkpoints: dict[int, Path | str] | None = None,
    device: str = DEVICE,
) -> list[str]:
    """Construct exact command-line arguments for four_player_native.py evaluator on a specific shard."""
    cfg = next(c for c in SHARD_CONFIG if c["shard_id"] == shard_id)
    seed = cfg["training_seed"]
    start_h = cfg["start_hanchan"]
    count = cfg["games_count"]

    if m1_checkpoints and seed in m1_checkpoints:
        m1_path = Path(m1_checkpoints[seed])
    else:
        m1_path = M1_TRAINING_DIR / f"M1_variant/seed_{seed}/checkpoints/mortal_72000.pth"

    shard_out_dir = output_dir / f"shard_{shard_id:02d}"

    cmd = [
        sys.executable,
        "-u",
        str(EVALUATOR_PATH),
        "--games", str(count),
        "--seed-start", str(start_h),
        "--seed-key", str(SEED_KEY),
        "--seat-mode", SEAT_MODE,
        "--device", device,
        "--output-dir", str(shard_out_dir),
        "--require-cuda",
        "--native-batch-games", "250",
        "--rank-points-profile", "tenhou_reference",
        "--model", f"70k={K0_70K_PATH}",
        "--model", f"ext_mortal={EXT_MORTAL_PATH}",
        "--model", f"M0_CURRENT_{seed}={M0_CURRENT_CHECKPOINTS[seed]['path']}",
        "--model", f"M1_CURRENT_{seed}={m1_path}",
    ]
    if AMP:
        cmd.append("--enable-amp")

    return cmd


def prepare_evaluation_plan(
    output_dir: Path = M1_EVALUATION_DIR,
    training_completion_closure_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize frozen evaluation plan binding all 8 checkpoints and 12 shards."""
    if training_completion_closure_path is None:
        training_completion_closure_path = M1_TRAINING_DIR / "training_completion_closure.json"

    if not training_completion_closure_path.exists():
        raise ContractError(
            f"Training completion closure missing at {training_completion_closure_path}. "
            "Cannot prepare evaluation plan before training is completed and verified."
        )

    with open(training_completion_closure_path, "r", encoding="utf-8") as f:
        closure = json.load(f)

    closure_sha = sha256_file(training_completion_closure_path)

    m1_checkpoints: dict[int, Path | str] = {}
    for run in closure.get("runs", []):
        s = int(run["training_seed"])
        m1_checkpoints[s] = run["final_checkpoint_path"]

    ckpt_ok, ckpt_records = validate_all_8_checkpoints(m1_checkpoints)
    if not ckpt_ok:
        raise ContractError(f"Checkpoint verification failed in evaluation plan preparation: {json.dumps(ckpt_records, indent=2)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "evaluation_plan.json"

    plan = {
        "schema": "keqing.mortal.m1_evaluation_plan.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_completion_closure_sha256": closure_sha,
        "evaluator": {
            "path": str(EVALUATOR_PATH.resolve()),
            "sha256": sha256_file(EVALUATOR_PATH),
            "blob_oid": git_blob_oid(EVALUATOR_PATH),
        },
        "parameters": {
            "seed_key": SEED_KEY,
            "seat_mode": SEAT_MODE,
            "rank_points": RANK_POINTS,
            "amp": AMP,
            "device": DEVICE,
            "native_batch_games": 250,
            "rank_points_profile": "tenhou_reference",
            "require_cuda": True,
        },
        "checkpoints": ckpt_records,
        "shards": SHARD_CONFIG,
    }

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        f.write("\n")

    plan_sha = sha256_file(plan_path)
    print(f"Evaluation plan created at {plan_path} (SHA256: {plan_sha})")
    return plan


def execute_shard(
    shard_id: int,
    output_dir: Path = M1_EVALUATION_DIR,
    m1_checkpoints: dict[int, Path | str] | None = None,
    device: str = DEVICE,
) -> None:
    """Run four_player_native evaluation for a single shard, refusing to overwrite non-empty output."""
    cfg = next(c for c in SHARD_CONFIG if c["shard_id"] == shard_id)
    shard_out_dir = output_dir / f"shard_{shard_id:02d}"
    
    logs_dir = shard_out_dir / "logs"
    if logs_dir.exists() and any(logs_dir.iterdir()):
        raise ContractError(
            f"Shard {shard_id:02d} logs directory already exists and is non-empty: {logs_dir}. "
            "Automatic overwrite/resume is prohibited."
        )
    if shard_out_dir.exists() and any(shard_out_dir.iterdir()):
        raise ContractError(
            f"Shard {shard_id:02d} directory already exists and is non-empty: {shard_out_dir}. "
            "Automatic overwrite/resume is prohibited."
        )

    shard_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_shard_command(shard_id, output_dir, m1_checkpoints=m1_checkpoints, device=device)
    print(f"Executing shard {shard_id:02d} (seed {cfg['training_seed']}, hanchan {cfg['start_hanchan']}..{cfg['end_hanchan']})...")
    print(f"Command: {shlex.join(cmd)}")
    
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Shard {shard_id:02d} failed with return code {res.returncode}")


def run_full_evaluation(
    output_dir: Path = M1_EVALUATION_DIR,
    shard: int | None = None,
    confirmation_token: str | None = None,
) -> None:
    """Execute evaluation for all shards, strictly requiring authorization."""
    if not EVALUATION_AUTHORIZED:
        raise AuthorizationError(
            "M1 evaluation execution is NOT authorized. "
            "Formal evaluation requires completed training, verified closure, and an authorization-only commit."
        )

    plan_path = output_dir / "evaluation_plan.json"
    if not plan_path.exists():
        raise ContractError(f"Evaluation plan missing: {plan_path}. Run --prepare-plan first.")

    target_shards = [shard] if shard is not None else list(SHARDS)
    for s_id in target_shards:
        execute_shard(s_id, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prepare-plan", action="store_true", help="Prepare evaluation plan from training closure")
    mode_group.add_argument("--execute", action="store_true", help="Execute 3000-game evaluation across 12 shards")
    mode_group.add_argument("--print-command", action="store_true", help="Print evaluator CLI command for a shard")
    mode_group.add_argument("--status", action="store_true", help="Print evaluation pipeline status")

    parser.add_argument("--shard", type=int, choices=list(SHARDS), default=None, help="Shard index (0..11)")
    parser.add_argument("--confirmation-token", type=str, default=None, help="Execution confirmation token")
    parser.add_argument("--output-dir", type=Path, default=M1_EVALUATION_DIR, help="Evaluation root directory")
    args = parser.parse_args()

    if args.prepare_plan:
        prepare_evaluation_plan(output_dir=args.output_dir)
    elif args.execute:
        run_full_evaluation(output_dir=args.output_dir, shard=args.shard, confirmation_token=args.confirmation_token)
    elif args.print_command:
        s_id = args.shard if args.shard is not None else 0
        cmd = build_shard_command(s_id, output_dir=args.output_dir)
        print(f"Shard {s_id:02d} command: {shlex.join(cmd)}")
    elif args.status:
        print(f"M1 Experiment ID: {M1_EXPERIMENT_ID}")
        print(f"Evaluation authorized: {EVALUATION_AUTHORIZED}")


if __name__ == "__main__":
    main()
