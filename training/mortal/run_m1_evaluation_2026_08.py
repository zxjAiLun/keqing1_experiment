#!/usr/bin/env python3
"""Fail-closed four-player evaluation launcher for M1 ext_mortal expansion absolute promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
    K0_70K_PATH,
    M0_CURRENT_CHECKPOINTS,
    M1_EVALUATION_DIR,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    RANK_POINTS,
    REPO_ROOT,
    SEAT_MODE,
    SEED_KEY,
    SHARD_CONFIG,
    SHARDS,
    ContractError,
    git_blob_oid,
    sha256_file,
    validate_all_8_checkpoints,
)

EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"

# Authorization flags (fail closed by default)
EVALUATION_AUTHORIZED = True
APPROVED_M1_IMPLEMENTATION_COMMIT = "ac3eb0a63d4951fa5144afb57e85d3f716ab35a4"
AUTHORIZED_TRAINING_COMPLETION_SHA256 = "d7762b699adc46acf90a84c3f11efdb7bdeaf63712038efd666803ad174d70b8"
AUTHORIZED_EVALUATION_PLAN_SHA256 = "670f7df41554e173cae9a0152b985da695efb032b85cf951232801cb11e75f1e"


class AuthorizationError(RuntimeError):
    """Raised when evaluation execution is attempted without required authorization."""


def compute_evaluation_confirmation_token(*, approved_commit: str, plan_sha256: str) -> str:
    """Compute deterministic confirmation token for evaluation execution."""
    payload = f"{approved_commit}:{plan_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_shard_command(
    shard_id: int,
    output_dir: Path,
    m1_checkpoints: dict[int, dict[str, str]] | None = None,
    device: str = DEVICE,
) -> list[str]:
    """Construct exact command-line arguments for four_player_native.py evaluator on a specific shard."""
    cfg = next(c for c in SHARD_CONFIG if c["shard_id"] == shard_id)
    seed = cfg["training_seed"]
    start_h = cfg["start_hanchan"]
    count = cfg["games_count"]

    if m1_checkpoints and seed in m1_checkpoints:
        m1_path = Path(m1_checkpoints[seed]["path"])
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
    plan_path = output_dir / "evaluation_plan.json"
    if plan_path.exists():
        raise ContractError(f"Evaluation plan already exists at {plan_path}. Refusing to overwrite.")

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

    m1_checkpoints: dict[int, dict[str, str]] = {}
    for run in closure.get("runs", []):
        s = int(run["training_seed"])
        m1_checkpoints[s] = {
            "path": run["final_checkpoint_path"],
            "sha256": run["final_checkpoint_sha256"],
        }

    ckpt_ok, ckpt_records = validate_all_8_checkpoints(m1_checkpoints)
    if not ckpt_ok:
        raise ContractError(f"Checkpoint verification failed in evaluation plan preparation: {json.dumps(ckpt_records, indent=2)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build shard future commands using the exact M1 checkpoints from closure
    shards_with_commands = []
    for cfg in SHARD_CONFIG:
        s_id = cfg["shard_id"]
        cmd = build_shard_command(s_id, output_dir, m1_checkpoints=m1_checkpoints, device=DEVICE)
        shards_with_commands.append({
            **cfg,
            "command": cmd,
            "command_str": shlex.join(cmd),
        })

    plan = {
        "schema": "keqing.mortal.m1_evaluation_plan.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "training_completion_closure_path": str(training_completion_closure_path.resolve()),
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
        "shards": shards_with_commands,
        "runtime_provenance": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
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
    m1_checkpoints: dict[int, dict[str, str]] | None = None,
    device: str = DEVICE,
    plan: dict[str, Any] | None = None,
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

    if plan:
        shard_info = next(s for s in plan["shards"] if s["shard_id"] == shard_id)
        cmd = shard_info["command"]
    else:
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
    enforce_canonical_paths: bool = True,
) -> None:
    """Execute evaluation for all shards, strictly requiring authorization and plan SHA match."""
    if not EVALUATION_AUTHORIZED:
        raise AuthorizationError(
            "M1 evaluation execution is NOT authorized. "
            "Formal evaluation requires completed training, verified closure, and an authorization-only commit."
        )

    if enforce_canonical_paths and output_dir.resolve() != M1_EVALUATION_DIR.resolve():
        raise ContractError(f"Non-canonical evaluation output directory in formal execute: {output_dir}")

    if not APPROVED_M1_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("APPROVED_M1_IMPLEMENTATION_COMMIT is required for evaluation execution")
    if not AUTHORIZED_TRAINING_COMPLETION_SHA256:
        raise AuthorizationError("AUTHORIZED_TRAINING_COMPLETION_SHA256 is required for evaluation execution")
    if not AUTHORIZED_EVALUATION_PLAN_SHA256:
        raise AuthorizationError("AUTHORIZED_EVALUATION_PLAN_SHA256 is required for evaluation execution")

    plan_path = output_dir / "evaluation_plan.json"
    if not plan_path.exists():
        raise ContractError(f"Evaluation plan missing: {plan_path}. Run --prepare-plan first.")

    current_plan_sha = sha256_file(plan_path)
    if current_plan_sha != AUTHORIZED_EVALUATION_PLAN_SHA256:
        raise ContractError(f"Evaluation plan SHA mismatch: {current_plan_sha} vs expected {AUTHORIZED_EVALUATION_PLAN_SHA256}")

    expected_token = compute_evaluation_confirmation_token(
        approved_commit=APPROVED_M1_IMPLEMENTATION_COMMIT,
        plan_sha256=AUTHORIZED_EVALUATION_PLAN_SHA256,
    )
    if confirmation_token != expected_token:
        raise AuthorizationError(f"Invalid confirmation token: {confirmation_token}, expected {expected_token}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Re-verify all 8 checkpoints from plan
    ckpt_ok, ckpt_records = validate_all_8_checkpoints({
        int(lbl.split("_")[-1]): {"path": rec["path"], "sha256": rec["expected_sha256"]}
        for lbl, rec in plan["checkpoints"].items() if lbl.startswith("M1_CURRENT")
    })
    if not ckpt_ok:
        raise ContractError(f"Pre-execution checkpoint re-verification failed: {json.dumps(ckpt_records, indent=2)}")

    target_shards = [shard] if shard is not None else list(SHARDS)
    for s_id in target_shards:
        execute_shard(s_id, output_dir=output_dir, plan=plan)


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
