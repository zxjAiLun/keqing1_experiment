#!/usr/bin/env python3
"""Paired-by-seed comparison of two one-vs-three evaluation runs.

The 1v3 evaluator already reports per-seed cluster statistics for a single run.
This compares two runs that used the *same seed band* (so the same deals), which
is the right unit for asking "did the candidate change anything" without
relying on the absolute score of either run.

Pairing is by seed over that seed's four seat rotations.  The two directions of
a bidirectional evaluation are reported separately and are not independent, so
no combined p-value is produced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _load(path: Path) -> tuple[dict[str, Any], dict[int, list[int]], tuple[float, ...]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    per_seed = {int(seed): list(ranks) for seed, ranks in document["integrity"]["per_seed_ranks"].items()}
    points = tuple(float(value) for value in document["rank_points_values"])
    return document, per_seed, points


def _seed_pt(per_seed: dict[int, list[int]], points: tuple[float, ...]) -> dict[int, float]:
    return {seed: float(np.mean([points[rank - 1] for rank in ranks])) for seed, ranks in per_seed.items()}


def _seed_rank(per_seed: dict[int, list[int]]) -> dict[int, float]:
    return {seed: float(np.mean(ranks)) for seed, ranks in per_seed.items()}


def _paired(values: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = [float(values[rng.integers(0, values.size, size=values.size)].mean()) for _ in range(reps)]
    non_tie = values[np.abs(values) > 1e-12]
    positive = int((non_tie > 0).sum())
    n = int(non_tie.size)
    sign_p = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)
    return {
        "clusters": int(values.size),
        "mean_delta": float(values.mean()),
        "cluster_bootstrap_ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "seeds_positive": positive,
        "seeds_non_tie": n,
        "one_sided_sign_p": sign_p,
        "bootstrap_reps": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-by-seed comparison of two 1v3 runs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", default="candidate_minus_baseline")
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    base_doc, base_seeds, base_points = _load(args.baseline)
    cand_doc, cand_seeds, cand_points = _load(args.candidate)
    if base_points != cand_points:
        raise SystemExit("rank points differ between runs; not comparable")
    if set(base_seeds) != set(cand_seeds):
        raise SystemExit(
            f"seed sets differ (baseline {len(base_seeds)} seeds, candidate {len(cand_seeds)}); pairing requires the same band"
        )

    base_pt, cand_pt = _seed_pt(base_seeds, base_points), _seed_pt(cand_seeds, cand_points)
    base_rank, cand_rank = _seed_rank(base_seeds), _seed_rank(cand_seeds)
    seeds = sorted(base_seeds)
    delta_pt = np.asarray([cand_pt[s] - base_pt[s] for s in seeds], dtype=np.float64)
    delta_rank = np.asarray([cand_rank[s] - base_rank[s] for s in seeds], dtype=np.float64)

    still_fresh = all(not batch.get("reused") for batch in base_doc["integrity"]["native_batches"]) and all(
        not batch.get("reused") for batch in cand_doc["integrity"]["native_batches"]
    )
    report = {
        "label": args.label,
        "baseline": {
            "path": str(args.baseline),
            "solo_seat": base_doc["challenger"]["label"],
            "opponent_trio": base_doc["champion"]["label"],
            "hanchans": base_doc["hanchans"],
            "avg_rank_pt": base_doc["challenger_result"]["avg_rank_pt"],
            "avg_rank": base_doc["challenger_result"]["avg_rank"],
            "rank_counts": base_doc["challenger_result"]["rank_counts"],
        },
        "candidate": {
            "path": str(args.candidate),
            "solo_seat": cand_doc["challenger"]["label"],
            "opponent_trio": cand_doc["champion"]["label"],
            "hanchans": cand_doc["hanchans"],
            "avg_rank_pt": cand_doc["challenger_result"]["avg_rank_pt"],
            "avg_rank": cand_doc["challenger_result"]["avg_rank"],
            "rank_counts": cand_doc["challenger_result"]["rank_counts"],
        },
        "same_labels": (
            base_doc["challenger"]["label"] == cand_doc["challenger"]["label"]
            and base_doc["champion"]["label"] == cand_doc["champion"]["label"]
        ),
        "same_opponent_trio_checkpoint": (
            base_doc["champion"]["checkpoint"] == cand_doc["champion"]["checkpoint"]
        ),
        "solo_seat_checkpoint_changed": (
            base_doc["challenger"]["checkpoint"] != cand_doc["challenger"]["checkpoint"]
        ),
        "all_batches_fresh": still_fresh,
        "paired_delta_avg_rank_pt": _paired(delta_pt, int(args.reps), int(args.seed)),
        "paired_delta_avg_rank": _paired(delta_rank, int(args.reps), int(args.seed)),
        "note": (
            "paired by seed over that seed's four seat rotations; the two directions of a "
            "bidirectional evaluation are not independent and are reported separately"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
