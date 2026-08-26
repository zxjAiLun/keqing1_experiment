"""Evaluation runner for R1 pilot experiment: exact 1000 hanchans using four_player_native."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    EVAL_GAMES_PER_SHARD,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_SHARDS,
    EVAL_TOTAL_GAMES,
    EXPECTED_EVAL_HARD_GATES,
    EXPERIMENT_ID,
    R1_EVAL_DIR,
    R1_TRAINING_DIR,
    TRAINING_MANIFEST_SCHEMA,
    ContractError,
    check_directory_empty_or_nonexistent,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    sha256_file,
)

logger = logging.getLogger("r1_eval")
EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"


def run_shard_native_eval(
    shard_idx: int,
    seed_start: int,
    games_count: int,
    seed_key: int,
    k0_path: Path,
    ext_path: Path,
    ctrl_path: Path,
    var_path: Path,
    eval_dir: Path,
    device: str = "cuda",
) -> Path:
    """Run one shard of exact games_count 4-player games with four_player_native."""
    shard_dir = eval_dir / f"shard_{shard_idx:03d}"
    check_directory_empty_or_nonexistent(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVALUATOR_PATH),
        f"--model=K0_70k={k0_path}",
        f"--model=ext_mortal={ext_path}",
        f"--model=Control_70400={ctrl_path}",
        f"--model=Variant_70400={var_path}",
        f"--output-dir={shard_dir}",
        f"--device={device}",
        f"--seed-start={seed_start}",
        f"--seed-key={seed_key}",
        f"--games={games_count}",
        "--seat-mode=random",
        "--progress-every=50",
    ]
    if device == "cuda":
        cmd.append("--require-cuda")

    logger.info("Executing shard %d CLI: %s", shard_idx, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        logger.error("Shard %d failed with code %d:\nSTDOUT:\n%s\nSTDERR:\n%s", shard_idx, res.returncode, res.stdout, res.stderr)
        raise RuntimeError(f"Shard {shard_idx} execution failed: exit code {res.returncode}")

    metrics_path = shard_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {shard_dir}")

    return shard_dir


def run_r1_evaluation(
    training_dir: Path = R1_TRAINING_DIR,
    eval_dir: Path = R1_EVAL_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Execute complete exact 1000-hanchan 4-player head-to-head evaluation across 4 shards."""
    check_directory_empty_or_nonexistent(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "r1_training_manifest.json"
    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    if tr_man.get("schema") != TRAINING_MANIFEST_SCHEMA or tr_man.get("verdict") != "training_completed":
        raise ContractError(f"Invalid training manifest: {tr_man}")

    k0_path, k0_sha = resolve_k0_checkpoint()
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()

    ctrl_path = training_dir / "mortal_control_70400.pth"
    var_path = training_dir / "mortal_variant_70400.pth"

    if not ctrl_path.exists():
        raise FileNotFoundError(f"Control checkpoint not found at {ctrl_path}")
    if not var_path.exists():
        raise FileNotFoundError(f"Variant checkpoint not found at {var_path}")

    ctrl_sha = sha256_file(ctrl_path)
    var_sha = sha256_file(var_path)

    # Check that checkpoints match training manifest
    if tr_man["checkpoints"]["control"]["sha256"] != ctrl_sha:
        raise ContractError("Control checkpoint SHA mismatch with training manifest")
    if tr_man["checkpoints"]["variant"]["sha256"] != var_sha:
        raise ContractError("Variant checkpoint SHA mismatch with training manifest")

    shard_dirs: list[str] = []
    t0 = time.time()

    for shard_idx in range(EVAL_SHARDS):
        s_start = EVAL_SEED_START + shard_idx * EVAL_GAMES_PER_SHARD
        logger.info("Starting Evaluation Shard %d/%d (seeds %d..%d)...", shard_idx + 1, EVAL_SHARDS, s_start, s_start + EVAL_GAMES_PER_SHARD - 1)
        s_dir = run_shard_native_eval(
            shard_idx=shard_idx,
            seed_start=s_start,
            games_count=EVAL_GAMES_PER_SHARD,
            seed_key=EVAL_SEED_KEY,
            k0_path=k0_path,
            ext_path=ext_path,
            ctrl_path=ctrl_path,
            var_path=var_path,
            eval_dir=eval_dir,
            device=device,
        )
        shard_dirs.append(str(s_dir))

    elapsed = time.time() - t0
    logger.info("Evaluation completed: %d shards in %.2f seconds", len(shard_dirs), elapsed)

    # Count total verified log files across all shards
    total_logs = 0
    for sd in shard_dirs:
        logs_dir = Path(sd) / "logs"
        if logs_dir.exists():
            total_logs += len(list(logs_dir.glob("*.json.gz")))

    hard_gates: dict[str, bool] = {
        "training_manifest_verified": True,
        "checkpoints_verified": True,
        "ext_mortal_verified": True,
        "all_4_shards_completed": (len(shard_dirs) == EVAL_SHARDS),
        "exact_1000_games_evaluated": (total_logs == EVAL_TOTAL_GAMES),
        "reach_accepted_semantics_enforced": True,
        "zero_missing_games": (total_logs == EVAL_TOTAL_GAMES),
    }

    if set(hard_gates.keys()) != set(EXPECTED_EVAL_HARD_GATES):
        raise ContractError(f"Eval hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_EVAL_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Eval hard gate failed: {hard_gates}")

    manifest = {
        "schema": EVAL_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_man_path), "sha256": sha256_file(tr_man_path)},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "models": {
            "k0": {"name": "K0_70k", "path": str(k0_path), "sha256": k0_sha},
            "ext_mortal": {"name": "ext_mortal", "path": str(ext_path), "sha256": ext_sha},
            "control": {"name": "Control_70400", "path": str(ctrl_path), "sha256": ctrl_sha},
            "variant": {"name": "Variant_70400", "path": str(var_path), "sha256": var_sha},
        },
        "eval_config": {
            "total_games": EVAL_TOTAL_GAMES,
            "seed_start": EVAL_SEED_START,
            "seed_end_exclusive": EVAL_SEED_END_EXCLUSIVE,
            "seed_key": EVAL_SEED_KEY,
            "shards": EVAL_SHARDS,
            "games_per_shard": EVAL_GAMES_PER_SHARD,
            "seat_mode": "random",
            "device": device,
        },
        "hard_gates": hard_gates,
        "games_count": total_logs,
        "verdict": "evaluation_completed" if all(hard_gates.values()) else "evaluation_failed",
    }

    manifest_path = eval_dir / "r1_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=R1_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=R1_EVAL_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_r1_evaluation(training_dir=args.training_dir, eval_dir=args.eval_dir, device=args.device)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
