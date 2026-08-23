#!/usr/bin/env python3
"""Parse raw game logs for M1 evaluation, verify hard gates, and adjudicate promotion verdict."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.m1_dataset_contract_2026_08 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CANONICAL_PROMOTION_CHECKPOINT,
    GAMES_PER_SHARD,
    K0_70K_PATH,
    K0_70K_SHA256,
    M0_CURRENT_CHECKPOINTS,
    M1_DATASET_DIR,
    M1_EVALUATION_DIR,
    M1_EXPERIMENT_ID,
    M1_TRAINING_DIR,
    PREREG_COMMIT,
    PREREG_PATH,
    RANK_POINTS,
    REPO_ROOT,
    SEED_KEY,
    SEEDS,
    SHARD_CONFIG,
    TOTAL_GAMES,
    TOTAL_SHARDS,
    ContractError,
    adjudicate_m1_promotion,
    equal_seed_hierarchical_bootstrap,
    git_blob_oid,
    git_info,
    sha256_file,
    validate_all_8_checkpoints,
)

EVALUATOR_PATH = REPO_ROOT / "training/mortal/four_player_native.py"
LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)(?:_[^/]*)?\.json\.gz$")
FORMAL_ADJUDICATION_DIR = M1_EVALUATION_DIR / "formal_adjudication"


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


def authoritative_ranks_from_stat(path: Path) -> list[int] | None:
    """Check native libriichi Stat per-seat final ranks (0-based seats -> 0..3 rank)."""
    try:
        from libriichi.stat import Stat  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw_log = handle.read()
        ranks: list[int] = []
        for player_id in range(4):
            stat = Stat.from_log(raw_log, player_id)
            rank = [rank_id - 1 for rank_id in (1, 2, 3, 4) if getattr(stat, f"rank_{rank_id}") == 1]
            if len(rank) != 1:
                return None
            ranks.append(rank[0])
        return ranks
    except Exception:
        return None


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

    seed_tuple = events[0].get("seed")
    if not isinstance(seed_tuple, (list, tuple)) or len(seed_tuple) != 2:
        raise ValueError(f"Invalid start_game seed tuple in {path.name}: {seed_tuple}")
    log_hanchan, log_key = int(seed_tuple[0]), int(seed_tuple[1])
    if log_hanchan != hanchan_id or log_key != SEED_KEY:
        raise ValueError(f"Seed/key mismatch in {path.name}: log=({log_hanchan}, {log_key}) vs expected=({hanchan_id}, {SEED_KEY})")

    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ValueError(f"Invalid names array in {path.name}: {names}")
    
    expected_labels = {"70k", "ext_mortal", f"M0_CURRENT_{expected_training_seed}", f"M1_CURRENT_{expected_training_seed}"}
    if set(names) != expected_labels:
        raise ValueError(f"Lineup labels mismatch in {path.name}: got {set(names)}, expected {expected_labels}")

    scores = final_scores(events)
    if scores is None:
        raise ValueError(f"Could not reconstruct scores in {path.name}")

    ranks = compute_final_ranks(scores)
    
    # Check Stat equivalence if available
    stat_ranks = authoritative_ranks_from_stat(path)
    if stat_ranks is not None and stat_ranks != ranks:
        raise ContractError(f"Rank discrepancy with libriichi Stat in {path.name}: event_ranks={ranks} vs stat_ranks={stat_ranks}")

    pts = [RANK_POINTS[r] for r in ranks]
    label_to_pt = {label: pts[idx] for idx, label in enumerate(names)}
    label_to_rank = {label: ranks[idx] for idx, label in enumerate(names)}

    return {
        "game_id": hanchan_id,
        "path": str(path),
        "label_to_pt": label_to_pt,
        "label_to_rank": label_to_rank,
    }


def parse_shard_logs(shard_dir: Path, shard_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse all game logs in shard_dir/logs/ directory and verify rank count consistency."""
    logs_dir = shard_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Shard logs directory not found: {logs_dir}")

    files = sorted(logs_dir.glob("*.json.gz"))
    expected_count = shard_cfg["games_count"]
    if len(files) != expected_count:
        raise ValueError(f"Shard {shard_cfg['shard_id']} has {len(files)} logs, expected {expected_count}")

    records = []
    seen_game_ids = set()
    label_rank_counts: dict[str, list[int]] = {}

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

        for lbl, rk in rec["label_to_rank"].items():
            if lbl not in label_rank_counts:
                label_rank_counts[lbl] = [0, 0, 0, 0]
            label_rank_counts[lbl][rk] += 1

    # Check metrics.json if present
    metrics_path = shard_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            for lbl, counts in label_rank_counts.items():
                if lbl in metrics:
                    m_counts = metrics[lbl].get("rank_counts")
                    if m_counts is not None and m_counts != counts:
                        raise ContractError(f"Rank count mismatch in {shard_dir.name} metrics.json for {lbl}: {m_counts} vs parsed {counts}")
        except Exception as exc:
            raise ContractError(f"Error checking metrics.json in {shard_dir.name}: {exc}") from exc

    return records


def run_summarizer(
    output_root: Path = M1_EVALUATION_DIR,
    m1_checkpoints: dict[int, Path | str] | None = None,
    destination_file: Path | None = None,
) -> dict[str, Any]:
    """Summarize full M1 evaluation output and publish atomic formal summary."""
    if destination_file is None:
        FORMAL_ADJUDICATION_DIR.mkdir(parents=True, exist_ok=True)
        destination_file = FORMAL_ADJUDICATION_DIR / "m1_summary.json"

    if destination_file.exists():
        raise ContractError(f"Destination summary file already exists: {destination_file}. Refusing to overwrite.")

    ckpt_ok, ckpt_records = validate_all_8_checkpoints(m1_checkpoints)

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
        adjudication = adjudicate_m1_promotion(
            x_seed_means={s: 0.0 for s in SEEDS},
            x_ci95=(0.0, 0.0),
            y_seed_means={s: 0.0 for s in SEEDS},
            y_ci95=(0.0, 0.0),
            gates_pass=False,
        )
        summary = {
            "schema": "keqing.mortal.m1_promotion_summary.v1",
            "experiment_id": M1_EXPERIMENT_ID,
            "hard_gates": gates,
            "checkpoints": ckpt_records,
            "shards": shards_found,
            "adjudication": adjudication,
        }
        _atomic_write_json(summary, destination_file)
        return summary

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
        m1_label = f"M1_CURRENT_{s}"
        for r in recs:
            l2pt = r["label_to_pt"]
            pt_70k = l2pt["70k"]
            pt_m0 = l2pt[m0_label]
            pt_m1 = l2pt[m1_label]

            x_list.append(pt_m1 - pt_m0)
            y_list.append(pt_m1 - pt_70k)

        x_arr = np.array(x_list, dtype=np.float64)
        y_arr = np.array(y_list, dtype=np.float64)
        x_by_seed[s] = x_arr
        y_by_seed[s] = y_arr
        x_seed_means[s] = float(np.mean(x_arr))
        y_seed_means[s] = float(np.mean(y_arr))

    x_ci95, y_ci95, x_equal_mean, y_equal_mean = equal_seed_hierarchical_bootstrap(
        x_by_seed, y_by_seed
    )

    adjudication = adjudicate_m1_promotion(
        x_seed_means=x_seed_means,
        x_ci95=x_ci95,
        y_seed_means=y_seed_means,
        y_ci95=y_ci95,
        gates_pass=True,
    )

    result_provenance = {
        "experiment_id": M1_EXPERIMENT_ID,
        "prereg_commit": PREREG_COMMIT,
        "evaluator_path": str(EVALUATOR_PATH.resolve()),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "total_games": TOTAL_GAMES,
    }

    summary = {
        "schema": "keqing.mortal.m1_promotion_summary.v1",
        "experiment_id": M1_EXPERIMENT_ID,
        "result_provenance": result_provenance,
        "hard_gates": gates,
        "checkpoints": ckpt_records,
        "shards": shards_found,
        "statistics": {
            "x_vs_m0_current": {
                "description": "Pt(M1) - Pt(M0_CURRENT)",
                "equal_seed_mean": x_equal_mean,
                "seed_means": x_seed_means,
                "hierarchical_ci95": list(x_ci95),
                "pass_all_seeds_positive": all(x_seed_means[s] > 0.0 for s in SEEDS),
                "pass_ci_lower_positive": x_ci95[0] > 0.0,
            },
            "y_vs_70k": {
                "description": "Pt(M1) - Pt(K0_70k)",
                "equal_seed_mean": y_equal_mean,
                "seed_means": y_seed_means,
                "hierarchical_ci95": list(y_ci95),
                "pass_all_seeds_positive": all(y_seed_means[s] > 0.0 for s in SEEDS),
                "pass_ci_lower_positive": y_ci95[0] > 0.0,
            },
        },
        "adjudication": adjudication,
    }

    _atomic_write_json(summary, destination_file)
    summary_sha = sha256_file(destination_file)
    print(f"Summary written to {destination_file} (SHA256: {summary_sha})")
    return summary


def _atomic_write_json(data: dict[str, Any], destination_path: Path) -> None:
    """Write JSON data to a temporary file, fsync, and atomic rename to destination."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = destination_path.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        json.dump(data, tf, indent=2, ensure_ascii=False)
        tf.write("\n")
        tf.flush()
        os.fsync(tf.fileno())

    os.replace(temp_name, destination_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=M1_EVALUATION_DIR,
        help="Path to evaluation artifacts output directory",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Destination summary JSON path",
    )
    args = parser.parse_args()
    run_summarizer(output_root=args.output_dir, destination_file=args.destination)


if __name__ == "__main__":
    main()
