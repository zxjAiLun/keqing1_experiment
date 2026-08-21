#!/usr/bin/env python3
"""Execute four-player evaluation for C3 D1_CQL_OFF Absolute Promotion."""

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

from training.mortal.c3_evaluation_contract_2026_08 import (
    AMP,
    C3_EXPERIMENT_ID,
    DEVICE,
    GAMES_PER_SHARD,
    RANK_POINTS,
    REPO_ROOT,
    SEAT_MODE,
    SEED_KEY,
    SEEDS,
    SHARD_CONFIG,
    SHARDS,
    TOTAL_GAMES,
    model_lineup_for_seed,
    sha256_file,
    validate_checkpoints,
)

EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/experiments/C3_d1_cql_off_absolute_promotion_2026_08"


def build_shard_command(
    shard_id: int,
    output_dir: Path,
    device: str = DEVICE,
) -> list[str]:
    """Construct command-line arguments for four_player_native.py evaluator on a specific shard."""
    cfg = next(c for c in SHARD_CONFIG if c["shard_id"] == shard_id)
    seed = cfg["training_seed"]
    start_h = cfg["start_hanchan"]
    count = cfg["games_count"]

    lineup = model_lineup_for_seed(seed)
    
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
    ]
    if AMP:
        cmd.append("--enable-amp")

    # Add 4 models: 70k, ext_mortal, M0_CURRENT_<seed>, D1_CQL_OFF_<seed>
    for m in lineup:
        cmd.extend(["--model", f"{m['label']}={m['path']}"])

    return cmd


def execute_shard(shard_id: int, output_dir: Path, device: str = DEVICE) -> None:
    """Run four_player_native evaluation for a single shard, refusing to overwrite non-empty output."""
    cfg = next(c for c in SHARD_CONFIG if c["shard_id"] == shard_id)
    shard_out_dir = output_dir / f"shard_{shard_id:02d}"
    
    # Refuse to overwrite non-empty shard directory
    logs_dir = shard_out_dir / "logs"
    if logs_dir.exists() and any(logs_dir.iterdir()):
        raise RuntimeError(
            f"Shard {shard_id:02d} logs directory already exists and is non-empty: {logs_dir}. "
            "Automatic overwrite/resume is prohibited."
        )
    if shard_out_dir.exists() and any(shard_out_dir.iterdir()):
        raise RuntimeError(
            f"Shard {shard_id:02d} directory already exists and is non-empty: {shard_out_dir}. "
            "Automatic overwrite/resume is prohibited."
        )

    shard_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_shard_command(shard_id, output_dir, device=device)
    print(f"Executing shard {shard_id:02d} (seed {cfg['training_seed']}, hanchan {cfg['start_hanchan']}..{cfg['end_hanchan']})...")
    print(f"Command: {shlex.join(cmd)}")
    
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Shard {shard_id:02d} failed with return code {res.returncode}")


def run_full_evaluation(output_dir: Path, device: str = DEVICE, shard: int | None = None) -> None:
    """Execute evaluation for all shards or a specific shard."""
    ckpt_ok, ckpt_records = validate_checkpoints()
    if not ckpt_ok:
        raise RuntimeError(f"Checkpoint validation failed: {json.dumps(ckpt_records, indent=2)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "c3_input_manifest.json"
    if not manifest_path.exists():
        manifest = {
            "experiment_id": C3_EXPERIMENT_ID,
            "checkpoints": ckpt_records,
            "shard_config": SHARD_CONFIG,
            "evaluator": {
                "path": str(EVALUATOR_PATH),
                "sha256": sha256_file(EVALUATOR_PATH),
            },
            "parameters": {
                "seed_key": SEED_KEY,
                "seat_mode": SEAT_MODE,
                "rank_points": RANK_POINTS,
                "amp": AMP,
                "device": device,
                "native_batch_games": 250,
                "rank_points_profile": "tenhou_reference",
                "require_cuda": True,
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Created manifest at {manifest_path}")

    target_shards = [shard] if shard is not None else list(SHARDS)
    for s_id in target_shards:
        execute_shard(s_id, output_dir, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=list(SHARDS), default=None, help="Run only specific shard (0..11)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device (default: cuda)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    run_full_evaluation(args.output_dir, device=args.device, shard=args.shard)


if __name__ == "__main__":
    main()
