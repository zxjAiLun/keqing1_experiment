#!/usr/bin/env python3
"""Compare two directories of native arena logs at the game-event level.

Why this exists
---------------
An evaluation result is only comparable to a historical baseline if the runs
that produced them are comparable.  Comparing gzip container bytes does not
answer that question (compression level, mtimes and timing metadata all vary),
and neither does comparing aggregate rank counts.  What matters is whether the
*game* unfolded identically: the events, their actors, tiles, scores and the
game-determining parts of each decision.

This tool therefore compares, per log file, the ordered sequence of game events
after projecting each record onto a declared field set, and reports the first
real divergence rather than scanning for a match.

Declared field tiers
--------------------
game (must be identical)
    Every non-``meta`` field of every record (``type``, ``actor``, ``pai``,
    ``consumed``, ``target``, ``tsumogiri``, ``deltas``, ``ura_markers``,
    ``dora_marker``, ``tehais``, ``scores``, ``bakaze``/``kyoku``/``honba``/
    ``kyotaku``/``oya``, ``names``, ``seed``), plus the game-determining
    ``meta`` entries GAME_META_KEYS (legal-action ``mask_bits``, ``is_greedy``,
    ``shanten``, ``at_furiten``).

float (reported, not decisive)
    Floating-point inference outputs (``meta.q_values``).  These are recorded as
    a numeric difference report; a last-digit difference is NOT by itself
    evidence that the rule environment differs.  Their argmax is also compared,
    because that is what actually selects the action.

excluded (never a verdict)
    Runtime/bookkeeping fields that cannot be part of a game-identity claim:
    ``meta.eval_time_ns`` (wall-clock of one inference), ``meta.batch_size``
    (runtime batching shape), and the gzip container itself.  Differences are
    counted and reported for transparency only.

Interpretation rule (do not over-read the verdict)
--------------------------------------------------
``REPRODUCED`` means the specified environment reproduced this batch of
historical logs; it does NOT license a retroactive claim that results produced
under a *different* native build are comparable.  ``DIVERGED`` means
reproduction is not established yet and the reported first divergence is where
to start looking -- it is not proof that "the environment changed".
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "keqing.mortal.arena_log_comparison.v1"

# Game-determining decision metadata: these change which action is chosen or
# record that internally, so they belong to the game identity.
GAME_META_KEYS: tuple[str, ...] = ("mask_bits", "is_greedy", "shanten", "at_furiten")

# Float inference outputs: compared numerically and reported, never decisive.
FLOAT_META_KEYS: tuple[str, ...] = ("q_values",)

# Runtime bookkeeping: cannot be part of a game-identity claim.
EXCLUDED_META_KEYS: tuple[str, ...] = ("eval_time_ns", "batch_size")

INTERPRETATION = (
    "REPRODUCED supports that the specified environment can reproduce this batch of "
    "historical logs. It does NOT retroactively certify that results produced under a "
    "different native build are comparable. DIVERGED means reproduction is not yet "
    "established; the reported first divergence is the place to start, and it is not "
    "by itself proof that the environment changed."
)


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def game_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Non-meta fields plus the game-determining meta fields."""
    projected = {key: value for key, value in record.items() if key != "meta"}
    meta = record.get("meta") or {}
    if isinstance(meta, dict):
        for key in GAME_META_KEYS:
            if key in meta:
                projected["meta." + key] = meta[key]
    return projected


def float_fields(record: dict[str, Any]) -> dict[str, list[float]]:
    meta = record.get("meta") or {}
    if not isinstance(meta, dict):
        return {}
    return {
        key: [float(x) for x in meta[key]]
        for key in FLOAT_META_KEYS
        if isinstance(meta.get(key), list)
    }


def _argmax(values: Iterable[float]) -> int:
    best_index = 0
    best_value = None
    for index, value in enumerate(values):
        if best_value is None or value > best_value:
            best_value = value
            best_index = index
    return best_index


def expected_file_names(seed_start: int, seed_count: int, seed_key: int,
                        splits: str) -> list[str]:
    return [
        f"{seed}_{seed_key}_{split}.json.gz"
        for seed in range(seed_start, seed_start + seed_count)
        for split in splits
    ]


def compare_files(baseline_path: Path, candidate_path: Path,
                  file_label: str) -> dict[str, Any]:
    """Compare one pair of log files.  Returns a divergence record or a match."""
    base = load_records(baseline_path)
    cand = load_records(candidate_path)

    result: dict[str, Any] = {
        "file": file_label,
        "records_baseline": len(base),
        "records_candidate": len(cand),
        "game_mismatches": 0,
        "first_divergence": None,
        "float_report": {},
        "excluded_diffs": {},
    }

    if len(base) != len(cand):
        result["first_divergence"] = {
            "kind": "record_count",
            "detail": f"baseline has {len(base)} records, candidate has {len(cand)}",
        }
        return result

    float_max_abs: dict[str, float] = {}
    float_values_compared: dict[str, int] = {}
    float_argmax_mismatches: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}

    for index, (a, b) in enumerate(zip(base, cand)):
        if game_projection(a) != game_projection(b):
            result["game_mismatches"] += 1
            if result["first_divergence"] is None:
                result["first_divergence"] = {
                    "kind": "game_event",
                    "record_index": index,
                    "baseline": game_projection(a),
                    "candidate": game_projection(b),
                }
                # First real divergence is the answer; do not hunt for a match.
                break

        for key, values_a in float_fields(a).items():
            values_b = float_fields(b).get(key)
            if not values_b or len(values_a) != len(values_b):
                continue
            float_values_compared[key] = (
                float_values_compared.get(key, 0) + len(values_a))
            for x, y in zip(values_a, values_b):
                delta = abs(x - y)
                if delta > float_max_abs.get(key, 0.0):
                    float_max_abs[key] = delta
            if _argmax(values_a) != _argmax(values_b):
                float_argmax_mismatches[key] = (
                    float_argmax_mismatches.get(key, 0) + 1)

        meta_a = a.get("meta") or {}
        meta_b = b.get("meta") or {}
        if isinstance(meta_a, dict) and isinstance(meta_b, dict):
            for key in EXCLUDED_META_KEYS:
                if (key in meta_a or key in meta_b) and meta_a.get(key) != meta_b.get(key):
                    excluded_counts[key] = excluded_counts.get(key, 0) + 1

    # Always report float fields that were actually compared, so "identical"
    # (max_abs_diff == 0.0 with values_compared > 0) is distinguishable from
    # "not compared at all" (field absent).
    result["float_report"] = {
        key: {
            "values_compared": float_values_compared[key],
            "max_abs_diff": float_max_abs.get(key, 0.0),
            "argmax_mismatches": float_argmax_mismatches.get(key, 0),
        }
        for key in float_values_compared
    }
    result["excluded_diffs"] = excluded_counts
    return result


def compare_run_dirs(baseline_dir: Path, candidate_dir: Path, seed_start: int,
                     seed_count: int, seed_key: int, splits: str,
                     stop_on_first_divergence: bool = True) -> dict[str, Any]:
    names = expected_file_names(seed_start, seed_count, seed_key, splits)

    missing: list[str] = []
    per_file: list[dict[str, Any]] = []
    records_compared = 0
    game_mismatches = 0
    float_report: dict[str, dict[str, float]] = {}
    excluded_totals: dict[str, int] = {}
    first_divergence: dict[str, Any] | None = None

    for name in names:
        base_path = baseline_dir / name
        cand_path = candidate_dir / name
        if not base_path.exists() or not cand_path.exists():
            missing.append(name)
            continue

        comparison = compare_files(base_path, cand_path, name)
        per_file.append(comparison)
        records_compared += min(comparison["records_baseline"],
                                comparison["records_candidate"])
        game_mismatches += comparison["game_mismatches"]

        for key, entry in comparison["float_report"].items():
            current = float_report.setdefault(
                key, {"values_compared": 0, "max_abs_diff": 0.0, "argmax_mismatches": 0})
            current["values_compared"] += entry["values_compared"]
            current["max_abs_diff"] = max(current["max_abs_diff"],
                                          entry["max_abs_diff"])
            current["argmax_mismatches"] += entry["argmax_mismatches"]
        for key, count in comparison["excluded_diffs"].items():
            excluded_totals[key] = excluded_totals.get(key, 0) + count

        if comparison["first_divergence"] is not None and first_divergence is None:
            first_divergence = {"file": name, **comparison["first_divergence"]}
            if stop_on_first_divergence:
                break

    verdict = "REPRODUCED" if (not missing and first_divergence is None) else "DIVERGED"

    return {
        "schema": SCHEMA,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "selection": {
            "seed_start": seed_start,
            "seed_count": seed_count,
            "seed_key": seed_key,
            "splits": splits,
            "expected_files": len(names),
        },
        "verdict": verdict,
        "interpretation": INTERPRETATION,
        "files_compared": len(per_file),
        "records_compared": records_compared,
        "game_field_mismatches": game_mismatches,
        "first_divergence": first_divergence,
        "missing_files": missing,
        "float_field_report": float_report,
        "excluded_field_diffs": excluded_totals,
        "declared_fields": {
            "game_meta_fields": list(GAME_META_KEYS),
            "float_meta_fields": list(FLOAT_META_KEYS),
            "excluded_meta_fields": list(EXCLUDED_META_KEYS),
            "excluded_containers": ["gzip container bytes", "file mtime/size",
                                    "log directory layout"],
        },
        "per_file": per_file,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two native arena log directories at game-event level.")
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--seed-start", type=int, default=710000)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--splits", default="abcd")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--continue-after-divergence", action="store_true",
                        help="report every file instead of stopping at the first "
                             "real divergence (default: stop at the first)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    document = compare_run_dirs(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        seed_key=args.seed_key,
        splits=args.splits,
        stop_on_first_divergence=not args.continue_after_divergence,
    )
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0 if document["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
