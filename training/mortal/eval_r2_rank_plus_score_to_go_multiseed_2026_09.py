"""Evaluation runner for R2 multi-seed confirmation experiment: 3 panels x 1000 hanchans."""

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

from training.mortal.r2_rank_plus_score_to_go_multiseed_contract_2026_09 import (
    EVAL_GAMES_PER_PANEL,
    EVAL_GAMES_PER_SHARD,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_SHARDS_PER_PANEL,
    EVALUATION_LINEUP,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    EXT_MORTAL_EXPECTED_SHA256,
    K0_EXPECTED_SHA256,
    R2_EVAL_DIR,
    R2_TRAINING_DIR,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEEDS,
    ContractError,
    check_directory_empty_or_nonexistent,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    sha256_file,
    verify_training_manifest,
)

logger = logging.getLogger("r2_eval")
EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"


def run_single_shard(
    panel_name: str,
    shard_idx: int,
    seed_start: int,
    games_count: int,
    seed_key: int,
    k0_path: Path,
    ext_path: Path,
    ctrl_path: Path,
    var_path: Path,
    panel_dir: Path,
    device: str = "cuda",
) -> Path:
    """Run one shard of exact games_count 4-player games with four_player_native."""
    shard_dir = panel_dir / f"shard_{shard_idx:03d}"
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

    logger.info("[%s] Executing shard %d CLI: %s", panel_name, shard_idx, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        logger.error("[%s] Shard %d failed with code %d:\nSTDOUT:\n%s\nSTDERR:\n%s", panel_name, shard_idx, res.returncode, res.stdout, res.stderr)
        raise RuntimeError(f"[{panel_name}] Shard {shard_idx} execution failed: exit code {res.returncode}")

    metrics_path = shard_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {shard_dir}")

    return shard_dir


def _verify_panel_logs(panel_dir: Path) -> tuple[list[int], bool]:
    """Parse every game log in one panel; return (game_ids, reach_semantics_ok)."""
    game_ids: list[int] = []
    reach_semantics_ok = True
    for shard_idx in range(EVAL_SHARDS_PER_PANEL):
        logs_dir = panel_dir / f"shard_{shard_idx:03d}" / "logs"
        if not logs_dir.exists():
            raise FileNotFoundError(f"Missing logs directory in {panel_dir / f'shard_{shard_idx:03d}'}")
        log_files = sorted(logs_dir.glob("*.json.gz"))
        if len(log_files) != EVAL_GAMES_PER_SHARD:
            raise ContractError(
                f"Panel {panel_dir.name} shard {shard_idx} contains {len(log_files)} logs, expected {EVAL_GAMES_PER_SHARD}"
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


def run_r2_evaluation(
    training_dir: Path = R2_TRAINING_DIR,
    eval_dir: Path = R2_EVAL_DIR,
    seeds: list[int] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Execute complete 3-panel 3000-hanchan 4-player evaluation on common-random-number seeds."""
    target_seeds = seeds or TRAINING_SEEDS
    check_directory_empty_or_nonexistent(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "r2_training_manifest.json"
    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    # Fail-closed manifest verification (exact gate sets, objective/label/reward, row-identity)
    verify_training_manifest(tr_man)
    tr_manifest_ok = True

    k0_path, k0_sha = resolve_k0_checkpoint()
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    if ext_sha != EXT_MORTAL_EXPECTED_SHA256 or k0_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"Canonical model SHA mismatch: k0={k0_sha} ext={ext_sha}")

    panels_manifest: dict[str, Any] = {}
    total_logs_across_all_panels = 0
    all_panels_reach_ok = True
    all_panels_contiguous = True
    t0 = time.time()

    for training_seed in target_seeds:
        panel_key = f"seed_{training_seed}"
        panel_dir = eval_dir / f"panel_seed_{training_seed}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        seed_data = tr_man["checkpoints"][panel_key]
        ctrl_path = Path(seed_data["control"]["path"])
        var_path = Path(seed_data["variant"]["path"])

        if not ctrl_path.exists() or not var_path.exists():
            raise FileNotFoundError(f"Checkpoints for seed {training_seed} not found")

        ctrl_sha = sha256_file(ctrl_path)
        var_sha = sha256_file(var_path)

        if seed_data["control"]["sha256"] != ctrl_sha or seed_data["variant"]["sha256"] != var_sha:
            raise ContractError(f"Checkpoint SHA mismatch for seed {training_seed}: disk vs training manifest")

        shard_dirs: list[str] = []
        for shard_idx in range(EVAL_SHARDS_PER_PANEL):
            s_start = EVAL_SEED_START + shard_idx * EVAL_GAMES_PER_SHARD
            logger.info("Starting Panel %s Shard %d/%d (seeds %d..%d)...", panel_key, shard_idx + 1, EVAL_SHARDS_PER_PANEL, s_start, s_start + EVAL_GAMES_PER_SHARD - 1)
            s_dir = run_single_shard(
                panel_name=panel_key,
                shard_idx=shard_idx,
                seed_start=s_start,
                games_count=EVAL_GAMES_PER_SHARD,
                seed_key=EVAL_SEED_KEY,
                k0_path=k0_path,
                ext_path=ext_path,
                ctrl_path=ctrl_path,
                var_path=var_path,
                panel_dir=panel_dir,
                device=device,
            )
            shard_dirs.append(str(s_dir))

        # Fail-closed log verification: parse every game, check lineage, contiguous IDs
        game_ids, reach_ok = _verify_panel_logs(panel_dir)
        expected_ids = list(range(EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE))
        panel_contiguous = (len(game_ids) == EVAL_GAMES_PER_PANEL and len(set(game_ids)) == EVAL_GAMES_PER_PANEL and sorted(game_ids) == expected_ids)
        if not panel_contiguous:
            all_panels_contiguous = False
        if not reach_ok:
            all_panels_reach_ok = False

        panel_logs = len(game_ids)
        total_logs_across_all_panels += panel_logs
        panels_manifest[panel_key] = {
            "training_seed": training_seed,
            "panel_dir": str(panel_dir),
            "games_count": panel_logs,
            "models": {
                "control": {"name": ctrl_path.name, "path": str(ctrl_path), "sha256": ctrl_sha},
                "variant": {"name": var_path.name, "path": str(var_path), "sha256": var_sha},
            },
        }

    elapsed = time.time() - t0
    logger.info("All 3 panels evaluation completed: %d total games in %.2f seconds", total_logs_across_all_panels, elapsed)

    # Additional per-panel 1000 unique contiguous check
    total_contiguous = all_panels_contiguous and (total_logs_across_all_panels == len(target_seeds) * EVAL_GAMES_PER_PANEL)
    hard_gates: dict[str, bool] = {
        "training_manifest_verified": tr_manifest_ok,
        "all_checkpoints_verified": True,  # verified per-seed above
        "ext_mortal_verified": (ext_sha == EXT_MORTAL_EXPECTED_SHA256 and k0_sha == K0_EXPECTED_SHA256),
        "all_3_panels_completed": (len(panels_manifest) == len(target_seeds)),
        "exact_3000_games_evaluated": total_contiguous,
        "reach_accepted_semantics_enforced": all_panels_reach_ok,
        "zero_missing_games": total_contiguous and all_panels_reach_ok,
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
        "ext_mortal_model": {"name": "ext_mortal", "path": str(ext_path), "sha256": ext_sha},
        "eval_config": {
            "panels_count": len(target_seeds),
            "games_per_panel": EVAL_GAMES_PER_PANEL,
            "total_games": total_logs_across_all_panels,
            "seed_start": EVAL_SEED_START,
            "seed_end_exclusive": EVAL_SEED_END_EXCLUSIVE,
            "seed_key": EVAL_SEED_KEY,
            "shards_per_panel": EVAL_SHARDS_PER_PANEL,
            "games_per_shard": EVAL_GAMES_PER_SHARD,
            "seat_mode": "random",
            "device": device,
        },
        "panels": panels_manifest,
        "hard_gates": hard_gates,
        "total_games_evaluated": total_logs_across_all_panels,
        "verdict": "evaluation_completed" if all(hard_gates.values()) else "evaluation_failed",
    }

    manifest_path = eval_dir / "r2_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=R2_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=R2_EVAL_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_r2_evaluation(training_dir=args.training_dir, eval_dir=args.eval_dir, device=args.device)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
