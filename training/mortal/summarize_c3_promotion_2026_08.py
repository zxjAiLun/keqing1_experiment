#!/usr/bin/env python3
"""Parse raw game logs for C3 evaluation and adjudicate promotion verdict."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.c3_evaluation_contract_2026_08 import (
    C3_EXPERIMENT_ID,
    GAMES_PER_SHARD,
    RANK_POINTS,
    SEED_KEY,
    SEEDS,
    SHARD_CONFIG,
    TOTAL_GAMES,
    TOTAL_SHARDS,
    adjudicate_c3_promotion,
    equal_seed_hierarchical_bootstrap,
    model_lineup_for_seed,
    sha256_file,
    validate_checkpoints,
)

LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)(?:_[^/]*)?\.json\.gz$")


def final_scores(events: list[dict[str, Any]]) -> list[float] | None:
    """Reconstruct final scores with the corrected ReachAccepted semantics (-1000)."""
    scores: list[float] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(value) for value in values]
        elif event_type == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif event_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def compute_final_ranks(scores: list[float]) -> list[int]:
    """Rank players from 0 (1st) to 3 (4th) based on descending final scores (tie-breaking: seat order)."""
    indexed = [(score, -seat) for seat, score in enumerate(scores)]
    sorted_seats = [-seat for _, seat in sorted(indexed, reverse=True)]
    ranks = [0] * 4
    for r, seat in enumerate(sorted_seats):
        ranks[seat] = r
    return ranks


def parse_raw_log_file(
    path: Path,
    expected_training_seed: int,
    expected_hanchan_min: int,
    expected_hanchan_max: int,
) -> dict[str, Any]:
    """Parse one gzipped JSONL log file and validate all exact structural gates."""
    m = LOG_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Invalid log file name: {path.name}")
    hanchan_id = int(m.group("seed"))
    if not (expected_hanchan_min <= hanchan_id <= expected_hanchan_max):
        raise ValueError(f"Game ID {hanchan_id} outside expected range [{expected_hanchan_min}..{expected_hanchan_max}] in {path.name}")

    try:
        raw_bytes = path.read_bytes()
        raw_text = gzip.decompress(raw_bytes).decode("utf-8")
        events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    except Exception as exc:
        raise ValueError(f"Cannot parse JSONL log {path.name}: {exc}") from exc

    if not events or events[0].get("type") != "start_game":
        raise ValueError(f"Log {path.name} does not start with start_game")

    # Validate seed_key and hanchan_id from start_game
    seed_tuple = events[0].get("seed")
    if not isinstance(seed_tuple, (list, tuple)) or len(seed_tuple) != 2:
        raise ValueError(f"Invalid start_game seed tuple in {path.name}: {seed_tuple}")
    log_hanchan, log_key = int(seed_tuple[0]), int(seed_tuple[1])
    if log_hanchan != hanchan_id or log_key != SEED_KEY:
        raise ValueError(f"Seed/key mismatch in {path.name}: log=({log_hanchan}, {log_key}) vs expected=({hanchan_id}, {SEED_KEY})")

    # Validate model lineup labels
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ValueError(f"Invalid names array in {path.name}: {names}")
    
    expected_labels = {m["label"] for m in model_lineup_for_seed(expected_training_seed)}
    if set(names) != expected_labels:
        raise ValueError(f"Lineup labels mismatch in {path.name}: got {set(names)}, expected {expected_labels}")

    scores = final_scores(events)
    if scores is None:
        raise ValueError(f"Could not reconstruct scores in {path.name}")

    ranks = compute_final_ranks(scores)
    pts = [RANK_POINTS[r] for r in ranks]

    label_to_pt = {label: pts[idx] for idx, label in enumerate(names)}

    return {
        "game_id": hanchan_id,
        "path": str(path),
        "label_to_pt": label_to_pt,
    }


def parse_shard_logs(shard_dir: Path, shard_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse all game logs in shard_dir/logs/ directory."""
    logs_dir = shard_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Shard logs directory not found: {logs_dir}")

    files = sorted(logs_dir.glob("*.json.gz"))
    expected_count = shard_cfg["games_count"]
    if len(files) != expected_count:
        raise ValueError(f"Shard {shard_cfg['shard_id']} has {len(files)} logs, expected {expected_count}")

    records = []
    seen_game_ids = set()
    for p in files:
        rec = parse_raw_log_file(
            p,
            expected_training_seed=shard_cfg["training_seed"],
            expected_hanchan_min=shard_cfg["start_hanchan"],
            expected_hanchan_max=shard_cfg["end_hanchan"],
        )
        gid = rec["game_id"]
        if gid in seen_game_ids:
            raise ValueError(f"Duplicate game ID {gid} within shard {shard_cfg['shard_id']}")
        seen_game_ids.add(gid)
        records.append(rec)

    return records


def run_summarizer(output_root: Path) -> dict[str, Any]:
    """Summarize full C3 evaluation output and compute formal promotion verdict."""
    ckpt_ok, ckpt_records = validate_checkpoints()

    shards_found = {}
    records_by_seed: dict[int, list[dict[str, Any]]] = {s: [] for s in SEEDS}
    all_game_ids = set()

    for cfg in SHARD_CONFIG:
        shard_id = cfg["shard_id"]
        seed = cfg["training_seed"]
        shard_dir = output_root / f"shard_{shard_id:02d}"
        if not shard_dir.exists():
            shards_found[shard_id] = {"exists": False, "count": 0}
            continue
        try:
            parsed = parse_shard_logs(shard_dir, cfg)
            shards_found[shard_id] = {"exists": True, "count": len(parsed), "valid": True}
            records_by_seed[seed].extend(parsed)
            for r in parsed:
                all_game_ids.add(r["game_id"])
        except Exception as exc:
            shards_found[shard_id] = {"exists": True, "count": 0, "valid": False, "error": str(exc)}

    total_parsed_games = sum(len(recs) for recs in records_by_seed.values())
    unique_3000_games = len(all_game_ids) == TOTAL_GAMES
    all_shards_complete = (
        len(shards_found) == TOTAL_SHARDS
        and all(info.get("exists") and info.get("count") == GAMES_PER_SHARD and info.get("valid") for info in shards_found.values())
        and total_parsed_games == TOTAL_GAMES
        and unique_3000_games
    )

    gates = {
        "checkpoints_verified": ckpt_ok,
        "shards_complete": all_shards_complete,
        "total_3000_unique_games_verified": unique_3000_games,
    }

    gates_pass = all(gates.values())

    if not gates_pass:
        adjudication = adjudicate_c3_promotion(
            x_seed_means={s: 0.0 for s in SEEDS},
            x_ci95=(0.0, 0.0),
            y_seed_means={s: 0.0 for s in SEEDS},
            y_ci95=(0.0, 0.0),
            gates_pass=False,
        )
        return {
            "schema": "keqing.mortal.c3_promotion_summary.v1",
            "experiment_id": C3_EXPERIMENT_ID,
            "hard_gates": gates,
            "checkpoints": ckpt_records,
            "shards": shards_found,
            "adjudication": adjudication,
        }

    # Compute paired deltas x and y for each game
    x_by_seed = {}
    y_by_seed = {}
    x_seed_means = {}
    y_seed_means = {}

    for s in SEEDS:
        recs = records_by_seed[s]
        x_list = []
        y_list = []
        m0_label = f"M0_CURRENT_{s}"
        d1_off_label = f"D1_CQL_OFF_{s}"
        for r in recs:
            l2pt = r["label_to_pt"]
            pt_70k = l2pt["70k"]
            pt_m0 = l2pt[m0_label]
            pt_d1 = l2pt[d1_off_label]

            x_list.append(pt_d1 - pt_70k)
            y_list.append(pt_d1 - pt_m0)

        x_arr = np.array(x_list, dtype=np.float64)
        y_arr = np.array(y_list, dtype=np.float64)
        x_by_seed[s] = x_arr
        y_by_seed[s] = y_arr
        x_seed_means[s] = float(np.mean(x_arr))
        y_seed_means[s] = float(np.mean(y_arr))

    x_ci95, y_ci95, x_equal_mean, y_equal_mean = equal_seed_hierarchical_bootstrap(
        x_by_seed, y_by_seed
    )

    adjudication = adjudicate_c3_promotion(
        x_seed_means=x_seed_means,
        x_ci95=x_ci95,
        y_seed_means=y_seed_means,
        y_ci95=y_ci95,
        gates_pass=True,
    )

    summary = {
        "schema": "keqing.mortal.c3_promotion_summary.v1",
        "experiment_id": C3_EXPERIMENT_ID,
        "hard_gates": gates,
        "checkpoints": ckpt_records,
        "shards": shards_found,
        "statistics": {
            "x_vs_70k": {
                "description": "Pt(D1_CQL_OFF) - Pt(K0_70k)",
                "equal_seed_mean": x_equal_mean,
                "seed_means": x_seed_means,
                "hierarchical_ci95": list(x_ci95),
                "pass_all_seeds_positive": all(x_seed_means[s] > 0.0 for s in SEEDS),
                "pass_ci_lower_positive": x_ci95[0] > 0.0,
            },
            "y_vs_m0_current": {
                "description": "Pt(D1_CQL_OFF) - Pt(M0_CURRENT)",
                "equal_seed_mean": y_equal_mean,
                "seed_means": y_seed_means,
                "hierarchical_ci95": list(y_ci95),
                "pass_all_seeds_positive": all(y_seed_means[s] > 0.0 for s in SEEDS),
                "pass_ci_lower_positive": y_ci95[0] > 0.0,
            },
        },
        "adjudication": adjudication,
    }

    summary_path = output_root / "c3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary_sha = sha256_file(summary_path)
    print(f"Summary written to {summary_path} (SHA256: {summary_sha})")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_REPO_ROOT / "artifacts/experiments/C3_d1_cql_off_absolute_promotion_2026_08",
        help="Path to evaluation artifacts output directory",
    )
    args = parser.parse_args()
    run_summarizer(args.output_dir)


if __name__ == "__main__":
    main()
