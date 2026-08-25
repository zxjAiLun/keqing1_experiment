#!/usr/bin/env python3
"""Fail-closed four-player evaluation launcher for O2 online continuation pilot."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.o2_online_continuation_contract_2026_08 import (
    AMP,
    EVALUATION_GAMES,
    EVALUATION_GAMES_PER_SHARD,
    EVALUATION_LINEUP,
    EVALUATION_MANIFEST_SCHEMA,
    EVALUATION_SEED_START,
    EVALUATION_SHARDS,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    O2_CHECKPOINT_70400_SHA256,
    O2_EVALUATION_DIR,
    O2_ROOT,
    O2_TRAINING_COMPLETION_SHA256,
    O2_TRAINING_DIR,
    O2_TRAINING_RUNNER_SHA256,
    SEED_KEY,
    TRAINING_COMPLETION_SCHEMA,
    ContractError,
    check_directory_boundary,
    ensure_clean_staging_dir,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_20260807_checkpoint,
    sha256_file,
)

logger = logging.getLogger("o2_evaluation")
EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"


def build_shard_spec(
    shard_id: int,
    output_dir: Path,
    o2_checkpoint_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Construct shard configuration and command using exact four_player_native.py CLI."""
    k0_path, _ = resolve_k0_checkpoint()
    ext_path, _ = resolve_ext_mortal_checkpoint()
    m0_path, _ = resolve_m0_20260807_checkpoint()

    start_hanchan = EVALUATION_SEED_START + shard_id * EVALUATION_GAMES_PER_SHARD
    shard_dir = output_dir / f"shard_{shard_id:02d}"

    model_specs = [
        f"K0_70k={k0_path}",
        f"ext_mortal={ext_path}",
        f"M0_CURRENT_20260807={m0_path}",
        f"O2_70400={o2_checkpoint_path}",
    ]

    cmd = [
        sys.executable,
        str(EVALUATOR_PATH),
        "--output-dir",
        str(shard_dir),
        "--seed-start",
        str(start_hanchan),
        "--games",
        str(EVALUATION_GAMES_PER_SHARD),
        "--seed-key",
        str(SEED_KEY),
        "--seat-mode",
        "random",
        "--device",
        device,
        "--rank-points-profile",
        "tenhou_reference",
        "--native-batch-games",
        str(EVALUATION_GAMES_PER_SHARD),
    ]
    if device.startswith("cuda"):
        cmd.append("--require-cuda")
    if AMP:
        cmd.append("--enable-amp")
    for m in model_specs:
        cmd.extend(["--model", m])

    return {
        "shard_id": shard_id,
        "shard_dir": shard_dir,
        "start_hanchan": start_hanchan,
        "games_count": EVALUATION_GAMES_PER_SHARD,
        "command": cmd,
    }


def run_o2_evaluation(
    *,
    training_dir: Path = O2_TRAINING_DIR,
    output_dir: Path = O2_EVALUATION_DIR,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute complete 4-shard 1000-game four-player evaluation for O2."""
    check_directory_boundary(output_dir, O2_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    o2_checkpoint = training_dir / "mortal_70400.pth"
    if not o2_checkpoint.exists():
        raise FileNotFoundError(f"O2 checkpoint mortal_70400.pth not found in {training_dir}")
    actual_o2_sha = sha256_file(o2_checkpoint)

    # 1. Verify training completion existence, schema, experiment ID and SHA
    train_comp_file = training_dir / "training_completion.json"
    if not train_comp_file.exists():
        raise ContractError(f"training_completion.json not found in {training_dir}")
    actual_comp_sha = sha256_file(train_comp_file)

    train_summary = json.loads(train_comp_file.read_text(encoding="utf-8"))
    if train_summary.get("schema") != TRAINING_COMPLETION_SCHEMA:
        raise ContractError(f"Training completion schema mismatch: got {train_summary.get('schema')}, expected {TRAINING_COMPLETION_SCHEMA}")
    if train_summary.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"Training completion experiment_id mismatch: got {train_summary.get('experiment_id')}, expected {EXPERIMENT_ID}")
    if train_summary.get("verdict") != "training_completed":
        raise ContractError(f"Training verdict is not 'training_completed': {train_summary.get('verdict')}")

    # 2. Validate exact 14 hard gates in training completion
    hard_gates = train_summary.get("hard_gates", {})
    if set(hard_gates.keys()) != set(EXPECTED_TRAINING_HARD_GATES):
        raise ContractError(f"Training hard gates key set mismatch: got {set(hard_gates.keys())}, expected {set(EXPECTED_TRAINING_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Training hard gates contains False: {hard_gates}")

    # 3. Validate checkpoint step and SHA against training completion and canonical SHA
    logged_ckpt = train_summary.get("final_checkpoint", {})
    if logged_ckpt.get("step") != 70400:
        raise ContractError(f"Training checkpoint step mismatch: got {logged_ckpt.get('step')}, expected 70400")
    if logged_ckpt.get("sha256") != actual_o2_sha:
        raise ContractError(
            f"Checkpoint SHA mismatch: disk={actual_o2_sha} vs training_completion={logged_ckpt.get('sha256')}"
        )

    shards_meta = []
    for shard_id in range(EVALUATION_SHARDS):
        spec = build_shard_spec(shard_id, output_dir, o2_checkpoint, device)
        shard_dir = spec["shard_dir"]
        # Entire shard directory must be clean before running
        ensure_clean_staging_dir(shard_dir, output_dir)

        logger.info("Executing evaluation shard %d (%d games)...", shard_id, EVALUATION_GAMES_PER_SHARD)
        res = subprocess.run(spec["command"], check=False, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Shard {shard_id} evaluation failed: \n{res.stderr}")

        shards_meta.append({
            "shard_id": shard_id,
            "start_hanchan": spec["start_hanchan"],
            "games": spec["games_count"],
            "status": "completed",
        })

    eval_manifest = {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "lineup": EVALUATION_LINEUP,
        "total_games": EVALUATION_GAMES,
        "shards": shards_meta,
        "o2_checkpoint": {
            "path": str(o2_checkpoint),
            "sha256": actual_o2_sha,
        },
        "training_completion": {
            "path": str(train_comp_file),
            "sha256": actual_comp_sha,
        },
        "provenance": {
            "training_runner_sha256": O2_TRAINING_RUNNER_SHA256,
            "expected_checkpoint_sha256": O2_CHECKPOINT_70400_SHA256,
            "expected_completion_sha256": O2_TRAINING_COMPLETION_SHA256,
        },
        "verdict": "evaluation_completed",
    }

    manifest_path = output_dir / "evaluation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return eval_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=O2_TRAINING_DIR, help="Training dir containing mortal_70400.pth")
    parser.add_argument("--output-dir", type=Path, default=O2_EVALUATION_DIR, help="Evaluation output dir")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_o2_evaluation(
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
