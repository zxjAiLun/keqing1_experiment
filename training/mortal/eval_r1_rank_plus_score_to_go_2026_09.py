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
    EVALUATION_LINEUP,
    EXPECTED_EVAL_HARD_GATES,
    EXPERIMENT_ID,
    EXT_MORTAL_EXPECTED_SHA256,
    K0_EXPECTED_SHA256,
    R1_EVAL_DIR,
    R1_TRAINING_DIR,
    ContractError,
    check_directory_empty_or_nonexistent,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    sha256_file,
    verify_training_manifest,
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


def _verify_eval_logs(eval_dir: Path) -> tuple[list[int], bool]:
    """Parse every game log; return (game_ids, reach_semantics_ok).

    reach_semantics_ok is the structural precondition for the summary's
    reach_accepted score adjustment: every game must expose four-player
    start_kyoku scores and every reach_accepted event must carry a valid actor.
    """
    game_ids: list[int] = []
    reach_semantics_ok = True
    for shard_idx in range(EVAL_SHARDS):
        logs_dir = eval_dir / f"shard_{shard_idx:03d}" / "logs"
        if not logs_dir.exists():
            raise FileNotFoundError(f"Missing logs directory in {eval_dir / f'shard_{shard_idx:03d}'}")
        log_files = sorted(logs_dir.glob("*.json.gz"))
        if len(log_files) != EVAL_GAMES_PER_SHARD:
            raise ContractError(
                f"Shard {shard_idx} contains {len(log_files)} logs, expected {EVAL_GAMES_PER_SHARD}"
            )
        for log_path in log_files:
            ident = parse_game_identity(log_path)
            game_ids.append(int(ident["game_id"]))

            events = ident["events"]
            if events[-1].get("type") != "end_game":
                raise ContractError(f"Incomplete log file: {log_path}")
            has_kyoku_scores = False
            for ev in events:
                ev_type = ev.get("type")
                if ev_type == "start_kyoku" and isinstance(ev.get("scores"), list) and len(ev["scores"]) == 4:
                    has_kyoku_scores = True
                elif ev_type == "reach_accepted":
                    actor = ev.get("actor")
                    if not isinstance(actor, int) or not (0 <= actor < 4):
                        reach_semantics_ok = False
            if not has_kyoku_scores:
                reach_semantics_ok = False
    return game_ids, reach_semantics_ok


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
    tr_manifest_ok = verify_training_manifest(tr_man)

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

    # Checkpoints must match the training manifest on disk.
    checkpoints_ok = (
        tr_man["checkpoints"]["control"]["sha256"] == ctrl_sha
        and tr_man["checkpoints"]["variant"]["sha256"] == var_sha
    )
    if not checkpoints_ok:
        raise ContractError(
            "Checkpoint SHA mismatch with training manifest: "
            f"control disk={ctrl_sha} manifest={tr_man['checkpoints']['control']['sha256']}, "
            f"variant disk={var_sha} manifest={tr_man['checkpoints']['variant']['sha256']}"
        )

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

    # Parse every game log: fail-closed on identity violations and collect game IDs.
    game_ids, reach_semantics_ok = _verify_eval_logs(eval_dir)
    expected_ids = list(range(EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE))
    ids_sorted = sorted(game_ids)
    games_exact = ids_sorted == expected_ids
    total_logs = len(game_ids)

    hard_gates: dict[str, bool] = {
        "training_manifest_verified": tr_manifest_ok,
        "checkpoints_verified": checkpoints_ok,
        "ext_mortal_verified": (ext_sha == EXT_MORTAL_EXPECTED_SHA256) and (k0_sha == K0_EXPECTED_SHA256),
        "all_4_shards_completed": (len(shard_dirs) == EVAL_SHARDS),
        "exact_1000_games_evaluated": games_exact,
        "reach_accepted_semantics_enforced": reach_semantics_ok,
        "zero_missing_games": games_exact and (len(set(game_ids)) == EVAL_TOTAL_GAMES),
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
        "lineup": list(EVALUATION_LINEUP),
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
        "game_id_range": [EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE],
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
