#!/usr/bin/env python3
"""Summary and statistical adjudication for O2 online continuation pilot."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.o2_online_continuation_contract_2026_08 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    EVALUATION_GAMES,
    EVALUATION_GAMES_PER_SHARD,
    EVALUATION_LINEUP,
    EVALUATION_SEED_END_EXCLUSIVE,
    EVALUATION_SEED_START,
    EVALUATION_SHARDS,
    EXPERIMENT_ID,
    O2_EVALUATION_DIR,
    O2_ROOT,
    O2_TRAINING_DIR,
    SEED_KEY,
    TENHOU_RANK_POINTS,
    ContractError,
    adjudicate_o2_verdict,
    check_directory_boundary,
    compute_final_ranks,
    final_scores_with_reach_accepted,
    paired_bootstrap_ci,
)

logger = logging.getLogger("o2_summary")
LOG_NAME_RE = re.compile(r"^(?P<seed>d+)(?:_[^/]*)?.json.gz$")


def parse_evaluation_log(path: Path) -> dict[str, Any]:
    """Parse one gzipped JSONL log file, extract ranks, and assign Tenhou reference rank points."""
    m = LOG_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Invalid log file name: {path.name}")
    hanchan_id = int(m.group("seed"))

    raw_bytes = path.read_bytes()
    raw_text = gzip.decompress(raw_bytes).decode("utf-8")
    events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if not events or events[0].get("type") != "start_game":
        raise ValueError(f"Log {path.name} does not start with start_game")

    seed_tuple = events[0].get("seed")
    if not isinstance(seed_tuple, (list, tuple)) or len(seed_tuple) != 2:
        raise ValueError(f"Invalid start_game seed tuple in {path.name}: {seed_tuple}")
    log_hanchan, log_key = int(seed_tuple[0]), int(seed_tuple[1])
    if log_hanchan != hanchan_id or log_key != SEED_KEY:
        raise ValueError(f"Seed/key mismatch in {path.name}: log=({log_hanchan}, {log_key}) vs expected=({hanchan_id}, {SEED_KEY})")

    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ValueError(f"Invalid names array in {path.name}: {names}")

    expected_labels = set(EVALUATION_LINEUP)
    if set(names) != expected_labels:
        raise ValueError(f"Lineup labels mismatch in {path.name}: got {set(names)}, expected {expected_labels}")

    scores = final_scores_with_reach_accepted(events)
    if scores is None:
        raise ValueError(f"Could not reconstruct scores in {path.name}")

    ranks = compute_final_ranks(scores)
    pts = [float(TENHOU_RANK_POINTS[r]) for r in ranks]
    label_to_pt = {label: pts[idx] for idx, label in enumerate(names)}
    label_to_rank = {label: ranks[idx] for idx, label in enumerate(names)}

    return {
        "game_id": hanchan_id,
        "label_to_pt": label_to_pt,
        "label_to_rank": label_to_rank,
    }


def adjudicate_o2_evaluation(
    evaluation_dir: Path = O2_EVALUATION_DIR,
    training_dir: Path = O2_TRAINING_DIR,
) -> dict[str, Any]:
    """Load all 1000 logs, compute paired differences and bootstrap CIs, and produce formal adjudication summary."""
    check_directory_boundary(evaluation_dir, O2_ROOT)

    o2_checkpoint = training_dir / "mortal_70400.pth"
    if not o2_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {o2_checkpoint}")

    hard_gates: dict[str, bool] = {
        "all_shards_present": False,
        "exact_1000_games_parsed": False,
        "game_ids_contiguous": False,
        "scores_and_ranks_valid": False,
        "reach_accepted_parsed": True,
        "bootstrap_computed": False,
    }

    all_games: list[dict[str, Any]] = []
    for shard_id in range(EVALUATION_SHARDS):
        shard_raw_dir = evaluation_dir / f"shard_{shard_id:02d}" / "raw_eval"
        if not shard_raw_dir.exists():
            raise FileNotFoundError(f"Shard raw eval dir not found: {shard_raw_dir}")
        shard_logs = sorted(shard_raw_dir.glob("*.json.gz"))
        if len(shard_logs) != EVALUATION_GAMES_PER_SHARD:
            raise ContractError(f"Shard {shard_id} has {len(shard_logs)} logs, expected {EVALUATION_GAMES_PER_SHARD}")
        for log_p in shard_logs:
            parsed = parse_evaluation_log(log_p)
            all_games.append(parsed)

    hard_gates["all_shards_present"] = True
    hard_gates["exact_1000_games_parsed"] = (len(all_games) == EVALUATION_GAMES)

    all_games.sort(key=lambda g: g["game_id"])
    game_ids = [g["game_id"] for g in all_games]
    expected_ids = list(range(EVALUATION_SEED_START, EVALUATION_SEED_END_EXCLUSIVE))
    hard_gates["game_ids_contiguous"] = (game_ids == expected_ids)
    hard_gates["scores_and_ranks_valid"] = all(
        len(g["label_to_pt"]) == 4 and len(g["label_to_rank"]) == 4 for g in all_games
    )

    # Compute paired differences per hanchan
    # x = Pt(O2_70400) - Pt(K0_70k)
    # y = Pt(O2_70400) - Pt(M0_CURRENT_20260807)
    diffs_x = np.array([g["label_to_pt"]["O2_70400"] - g["label_to_pt"]["K0_70k"] for g in all_games], dtype=np.float64)
    diffs_y = np.array([g["label_to_pt"]["O2_70400"] - g["label_to_pt"]["M0_CURRENT_20260807"] for g in all_games], dtype=np.float64)

    mean_x, ci_x = paired_bootstrap_ci(diffs_x, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
    mean_y, ci_y = paired_bootstrap_ci(diffs_y, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
    hard_gates["bootstrap_computed"] = True

    all_gates_pass = all(hard_gates.values())
    verdict = adjudicate_o2_verdict(
        all_gates_pass=all_gates_pass,
        mean_x=mean_x,
        ci_x=ci_x,
        mean_y=mean_y,
        ci_y=ci_y,
    )

    summary = {
        "schema": "keqing.mortal.o2_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "evaluation_protocol": {
            "total_games": len(all_games),
            "seed_range": [EVALUATION_SEED_START, EVALUATION_SEED_END_EXCLUSIVE - 1],
            "lineup": EVALUATION_LINEUP,
            "rank_points": TENHOU_RANK_POINTS.tolist(),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci": BOOTSTRAP_CI,
        },
        "hard_gates": hard_gates,
        "results": {
            "x_vs_k0": {
                "mean_diff_pt": mean_x,
                "ci95": ci_x,
            },
            "y_vs_m0": {
                "mean_diff_pt": mean_y,
                "ci95": ci_y,
            },
        },
        "verdict": verdict,
        "promotion": {
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "k1": None,
            "eligible_for_o3_confirmation": verdict in ("strong_signal", "promising"),
        },
    }

    summary_path = evaluation_dir / "o2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=O2_EVALUATION_DIR, help="Evaluation directory")
    parser.add_argument("--training-dir", type=Path, default=O2_TRAINING_DIR, help="Training directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = adjudicate_o2_evaluation(
        evaluation_dir=args.evaluation_dir,
        training_dir=args.training_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
