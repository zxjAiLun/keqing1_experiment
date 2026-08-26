"""Statistical summary and adjudication for R1 pilot experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SHARDS,
    EVAL_TOTAL_GAMES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    R1_EVAL_DIR,
    R1_SUMMARY_DIR,
    R1_TRAINING_DIR,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    TRAINING_MANIFEST_SCHEMA,
    ContractError,
    adjudicate_r1_verdict,
    check_directory_empty_or_nonexistent,
    paired_bootstrap_ci,
    resolve_k0_checkpoint,
    sha256_file,
)

logger = logging.getLogger("r1_summary")


def _parse_game_log(log_path: Path) -> dict[str, Any]:
    """Parse one raw game log and extract final ranks and scores with exact reach_accepted semantics."""
    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    if not events or events[0].get("type") != "start_game" or events[-1].get("type") != "end_game":
        raise ContractError(f"Incomplete log file: {log_path}")

    names = events[0].get("names", [])
    scores: list[float] | None = None

    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "start_kyoku" and isinstance(ev.get("scores"), list):
            values = ev["scores"]
            if len(values) == 4:
                scores = [float(v) for v in values]
        elif ev_type == "reach_accepted" and scores is not None:
            actor = ev.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif ev_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = ev.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [s + float(d) for s, d in zip(scores, deltas, strict=True)]

    if scores is None:
        raise ContractError(f"Failed to reconstruct scores for {log_path}")

    # Ranks (tie-breaking: seat order)
    indexed = [(score, -seat) for seat, score in enumerate(scores)]
    sorted_seats = [-seat for _, seat in sorted(indexed, reverse=True)]
    ranks = [0] * 4
    for r, seat in enumerate(sorted_seats):
        ranks[seat] = r

    # Map names to seats
    name_to_seat: dict[str, int] = {}
    for seat, name in enumerate(names):
        name_to_seat[name] = seat

    return {
        "names": names,
        "name_to_seat": name_to_seat,
        "scores": scores,
        "ranks": ranks,
    }


def adjudicate_r1_pilot(
    training_dir: Path = R1_TRAINING_DIR,
    eval_dir: Path = R1_EVAL_DIR,
    summary_dir: Path = R1_SUMMARY_DIR,
) -> dict[str, Any]:
    """Load manifests, verify logs, compute primary and secondary contrasts, and generate summary."""
    check_directory_empty_or_nonexistent(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "r1_training_manifest.json"
    ev_man_path = eval_dir / "r1_eval_manifest.json"

    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    if not ev_man_path.exists():
        raise FileNotFoundError(f"Eval manifest not found at {ev_man_path}")

    tr_man_sha = sha256_file(tr_man_path)
    ev_man_sha = sha256_file(ev_man_path)

    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    ev_man = json.loads(ev_man_path.read_text(encoding="utf-8"))

    if tr_man.get("schema") != TRAINING_MANIFEST_SCHEMA or tr_man.get("verdict") != "training_completed":
        raise ContractError(f"Invalid training manifest: {tr_man}")
    if ev_man.get("schema") != EVAL_MANIFEST_SCHEMA or ev_man.get("verdict") != "evaluation_completed":
        raise ContractError(f"Invalid eval manifest: {ev_man}")

    _, k0_sha = resolve_k0_checkpoint()

    # Collect per-hanchan game logs from all 4 shards
    diff_var_minus_ctrl: list[float] = []
    diff_var_minus_k0: list[float] = []
    diff_ctrl_minus_k0: list[float] = []

    total_logs_verified = 0
    for shard_idx in range(EVAL_SHARDS):
        shard_dir = eval_dir / f"shard_{shard_idx:03d}"
        logs_dir = shard_dir / "logs"
        if not logs_dir.exists():
            raise FileNotFoundError(f"Missing logs directory in {shard_dir}")

        log_files = sorted(logs_dir.glob("*.json.gz"))
        for log_path in log_files:
            res = _parse_game_log(log_path)
            n2s = res["name_to_seat"]
            ranks = res["ranks"]

            seat_k0 = n2s["K0_70k"]
            seat_ctrl = n2s["Control_70400"]
            seat_var = n2s["Variant_70400"]

            pt_k0 = TENHOU_RANK_POINTS[ranks[seat_k0]]
            pt_ctrl = TENHOU_RANK_POINTS[ranks[seat_ctrl]]
            pt_var = TENHOU_RANK_POINTS[ranks[seat_var]]

            diff_var_minus_ctrl.append(float(pt_var - pt_ctrl))
            diff_var_minus_k0.append(float(pt_var - pt_k0))
            diff_ctrl_minus_k0.append(float(pt_ctrl - pt_k0))
            total_logs_verified += 1

    if total_logs_verified != EVAL_TOTAL_GAMES:
        raise ContractError(f"Total verified logs mismatch: {total_logs_verified} vs {EVAL_TOTAL_GAMES}")

    arr_var_ctrl = np.array(diff_var_minus_ctrl, dtype=np.float64)
    arr_var_k0 = np.array(diff_var_minus_k0, dtype=np.float64)
    arr_ctrl_k0 = np.array(diff_ctrl_minus_k0, dtype=np.float64)

    mean_vc, ci_vc = paired_bootstrap_ci(arr_var_ctrl, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
    mean_vk0, ci_vk0 = paired_bootstrap_ci(arr_var_k0, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 1, ci=BOOTSTRAP_CI)
    mean_ck0, ci_ck0 = paired_bootstrap_ci(arr_ctrl_k0, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 2, ci=BOOTSTRAP_CI)

    # Pilot verdict adjudication: strong_positive / weak_positive / not_promising
    verdict = adjudicate_r1_verdict(primary_mean=mean_vc, primary_ci_lower=ci_vc[0])

    hard_gates: dict[str, bool] = {
        "training_manifest_verified": True,
        "eval_manifest_verified": True,
        "all_logs_verified": True,
        "paired_metrics_recalculated": True,
        "primary_contrast_computed": True,
        "secondary_contrast_computed": True,
        "bootstrap_computed": True,
    }

    if set(hard_gates.keys()) != set(EXPECTED_SUMMARY_HARD_GATES):
        raise ContractError(f"Summary hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_SUMMARY_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Summary hard gate failed: {hard_gates}")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_man_path), "sha256": tr_man_sha},
        "eval_manifest": {"path": str(ev_man_path), "sha256": ev_man_sha},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "hard_gates": hard_gates,
        "metrics": {
            "total_games": len(arr_var_ctrl),
            "primary_contrast_variant_minus_control": {
                "mean_pt": mean_vc,
                "ci95": ci_vc,
            },
            "secondary_contrast_variant_minus_k0": {
                "mean_pt": mean_vk0,
                "ci95": ci_vk0,
            },
            "reference_contrast_control_minus_k0": {
                "mean_pt": mean_ck0,
                "ci95": ci_ck0,
            },
        },
        "verdict": verdict,
        "promotion": {
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "k1": None,
        },
    }

    summary_path = summary_dir / "r1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=R1_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=R1_EVAL_DIR)
    parser.add_argument("--summary-dir", type=Path, default=R1_SUMMARY_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = adjudicate_r1_pilot(training_dir=args.training_dir, eval_dir=args.eval_dir, summary_dir=args.summary_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
