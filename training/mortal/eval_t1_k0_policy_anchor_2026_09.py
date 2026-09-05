"""Evaluation runner for the T1 K0 policy-anchor pilot: 3 panels x 1000 hanchans."""

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

from training.mortal.t1_k0_policy_anchor_contract_2026_09 import (  # noqa: E402
    EVAL_GAMES_PER_PANEL,
    EVAL_GAMES_PER_SHARD,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_SHARDS_PER_PANEL,
    EVALUATION_LINEUP_TEMPLATE,
    EXPECTED_EVAL_HARD_GATES,
    EXPERIMENT_ID,
    EXT_MORTAL_EXPECTED_SHA256,
    K0_EXPECTED_SHA256,
    T1_EVAL_DIR,
    T1_TRAINING_DIR,
    ContractError,
    check_directory_empty_or_nonexistent,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_r2_control_checkpoint,
    sha256_file,
    validate_t1_seed_set,
    verify_training_manifest,
)

logger = logging.getLogger("t1_eval")
EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"


def _lineup_for_seed(seed: int) -> tuple[str, ...]:
    return tuple(name.format(seed=seed) for name in EVALUATION_LINEUP_TEMPLATE)


def _verify_resume_prefix(shard_dir: Path, seed: int, seed_start: int, games_count: int) -> dict[str, str]:
    """Only reuse complete logs forming an exact prefix of the frozen shard."""
    paths = sorted((shard_dir / "logs").glob("*.json.gz"))
    ids = []
    hashes = {}
    for path in paths:
        ident = parse_game_identity(path, lineup=_lineup_for_seed(seed))
        if ident["events"][-1].get("type") != "end_game":
            raise ContractError(f"Incomplete resume log: {path}")
        ids.append(ident["game_id"])
        hashes[str(path)] = sha256_file(path)
    if len(ids) > games_count or sorted(ids) != list(range(seed_start, seed_start + len(ids))):
        raise ContractError(f"Resume logs must form a unique contiguous prefix: {shard_dir}")
    # Preserve the original 50-game inference batch boundaries.
    if len(ids) % 50:
        raise ContractError(f"Resume prefix is not a complete 50-game batch: {shard_dir}")
    return hashes


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
    resume: bool = False,
) -> Path:
    """Run one shard of exact games_count 4-player games with four_player_native."""
    shard_dir = panel_dir / f"shard_{shard_idx:03d}"
    if not resume:
        check_directory_empty_or_nonexistent(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    preserved = _verify_resume_prefix(shard_dir, seed_key, seed_start, games_count) if resume else {}
    metrics_path = shard_dir / "metrics.json"
    if resume and metrics_path.exists():
        run = json.loads(metrics_path.read_text(encoding="utf-8"))["run"]
        expected = {
            "seed_start": seed_start, "seed_key": EVAL_SEED_KEY, "games": games_count,
            "seat_mode": "random", "native_batch_games": 50, "device": device,
            "models": dict(zip(_lineup_for_seed(seed_key), map(str, (k0_path, ext_path, ctrl_path, var_path)), strict=True)),
        }
        if any(run.get(key) != value for key, value in expected.items()):
            raise ContractError(f"Resume metrics protocol mismatch: {metrics_path}")
        if len(preserved) != games_count:
            raise ContractError(f"Completed metrics with missing logs: {shard_dir}")
        logger.info("[%s] Reusing verified completed shard %d", panel_name, shard_idx)
        return shard_dir

    cmd = [
        sys.executable,
        str(EVALUATOR_PATH),
        f"--model=K0_70k={k0_path}",
        f"--model=ext_mortal={ext_path}",
        f"--model=R2_Control_seed_{seed_key}={ctrl_path}",
        f"--model=T1_AnchorVariant_seed_{seed_key}={var_path}",
        f"--output-dir={shard_dir}",
        f"--device={device}",
        f"--seed-start={seed_start}",
        f"--seed-key={EVAL_SEED_KEY}",
        f"--games={games_count}",
        "--seat-mode=random",
        "--progress-every=50",
    ]
    if device == "cuda":
        cmd.append("--require-cuda")
    if preserved:
        cmd.append("--resume")

    logger.info("[%s] Executing shard %d CLI: %s", panel_name, shard_idx, " ".join(cmd))
    # Keep child diagnostics on disk even if the calling terminal disappears.
    with (shard_dir / "execution.log").open("a", encoding="utf-8") as output:
        res = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"[{panel_name}] Shard {shard_idx} execution failed: exit code {res.returncode}")

    for path, digest in preserved.items():
        if sha256_file(Path(path)) != digest:
            raise ContractError(f"Existing resume log changed: {path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {shard_dir}")

    return shard_dir


def _verify_panel_logs(panel_dir: Path, seed: int) -> tuple[list[int], bool]:
    """Parse every game log in one panel; return (game_ids, reach_semantics_ok)."""
    lineup = _lineup_for_seed(seed)
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
            ident = parse_game_identity(log_path, lineup=lineup)
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


def run_t1_evaluation(
    training_dir: Path = T1_TRAINING_DIR,
    eval_dir: Path = T1_EVAL_DIR,
    seeds: list[int] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Execute complete 3-panel 3000-hanchan evaluation reusing frozen R2 Control checkpoints."""
    target_seeds = validate_t1_seed_set(seeds)
    if not resume:
        check_directory_empty_or_nonexistent(eval_dir)
    elif (eval_dir / "t1_eval_manifest.json").exists():
        raise ContractError("Evaluation already has a final manifest; do not resume")
    eval_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "t1_training_manifest.json"
    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    verify_training_manifest(tr_man)
    tr_manifest_ok = True

    k0_path, k0_sha = resolve_k0_checkpoint()
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    if ext_sha != EXT_MORTAL_EXPECTED_SHA256 or k0_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"Canonical model SHA mismatch: k0={k0_sha} ext={ext_sha}")

    preserved_logs: dict[str, str] = {}
    if resume:
        # Validate every existing shard before launching any further games.
        for seed in target_seeds:
            for idx in range(EVAL_SHARDS_PER_PANEL):
                directory = eval_dir / f"panel_seed_{seed}" / f"shard_{idx:03d}"
                preserved_logs.update(_verify_resume_prefix(
                    directory, seed, EVAL_SEED_START + idx * EVAL_GAMES_PER_SHARD, EVAL_GAMES_PER_SHARD,
                ))
        all_paths = {str(path) for path in eval_dir.rglob("*.json.gz")}
        if all_paths != set(preserved_logs):
            raise ContractError("Unexpected game logs outside frozen panel/shard layout")
        receipt = {
            "experiment_id": EXPERIMENT_ID, "training_manifest_sha256": sha256_file(tr_man_path),
            "existing_log_count": len(preserved_logs), "existing_log_sha256": preserved_logs,
        }
        receipt_path = eval_dir / f"resume_receipt_{time.time_ns()}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    panels_manifest: dict[str, Any] = {}
    total_logs_across_all_panels = 0
    all_panels_reach_ok = True
    all_panels_contiguous = True
    t0 = time.time()

    for training_seed in target_seeds:
        panel_key = f"seed_{training_seed}"
        panel_dir = eval_dir / f"panel_seed_{training_seed}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        # Frozen R2 Control checkpoint (SHA re-verified on disk).
        ctrl_path, ctrl_sha, _ = resolve_r2_control_checkpoint(training_seed)

        seed_data = tr_man["checkpoints"][panel_key]
        var_path = Path(seed_data["anchor_variant"]["path"])
        if not var_path.exists():
            raise FileNotFoundError(f"T1 AnchorVariant checkpoint for seed {training_seed} not found at {var_path}")
        var_sha = sha256_file(var_path)
        if seed_data["anchor_variant"]["sha256"] != var_sha:
            raise ContractError(f"AnchorVariant checkpoint SHA mismatch for seed {training_seed}")

        shard_dirs: list[str] = []
        for shard_idx in range(EVAL_SHARDS_PER_PANEL):
            s_start = EVAL_SEED_START + shard_idx * EVAL_GAMES_PER_SHARD
            logger.info("Starting Panel %s Shard %d/%d (seeds %d..%d)...", panel_key, shard_idx + 1, EVAL_SHARDS_PER_PANEL, s_start, s_start + EVAL_GAMES_PER_SHARD - 1)
            s_dir = run_single_shard(
                panel_name=panel_key,
                shard_idx=shard_idx,
                seed_start=s_start,
                games_count=EVAL_GAMES_PER_SHARD,
                seed_key=training_seed,
                k0_path=k0_path,
                ext_path=ext_path,
                ctrl_path=ctrl_path,
                var_path=var_path,
                panel_dir=panel_dir,
                device=device,
                resume=resume,
            )
            shard_dirs.append(str(s_dir))

        game_ids, reach_ok = _verify_panel_logs(panel_dir, training_seed)
        expected_ids = list(range(EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE))
        panel_contiguous = (
            len(game_ids) == EVAL_GAMES_PER_PANEL
            and len(set(game_ids)) == EVAL_GAMES_PER_PANEL
            and sorted(game_ids) == expected_ids
        )
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
                "r2_control": {"name": ctrl_path.name, "path": str(ctrl_path), "sha256": ctrl_sha},
                "anchor_variant": {"name": var_path.name, "path": str(var_path), "sha256": var_sha},
            },
        }

    elapsed = time.time() - t0
    logger.info("All 3 panels evaluation completed: %d total games in %.2f seconds", total_logs_across_all_panels, elapsed)

    total_contiguous = all_panels_contiguous and (total_logs_across_all_panels == len(target_seeds) * EVAL_GAMES_PER_PANEL)
    hard_gates: dict[str, bool] = {
        "training_manifest_verified": tr_manifest_ok,
        "all_checkpoints_verified": True,
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
    for path, digest in preserved_logs.items():
        if sha256_file(Path(path)) != digest:
            raise ContractError(f"Preserved log changed during evaluation: {path}")

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
        "resumed_existing_logs": len(preserved_logs),
        "verdict": "evaluation_completed" if all(hard_gates.values()) else "evaluation_failed",
    }

    manifest_path = eval_dir / "t1_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=T1_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=T1_EVAL_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Verify and preserve existing contiguous 50-game batches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_t1_evaluation(training_dir=args.training_dir, eval_dir=args.eval_dir, device=args.device, resume=args.resume)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
