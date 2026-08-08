#!/usr/bin/env python3
"""Summarize matched-seed native evaluations for the reward-semantics A/B."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RANK_POINTS = (90.0, 45.0, 0.0, -135.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        action="append",
        dest="eval_roots",
        required=True,
        help="evaluation root; repeat to combine matched seed pairs from multiple epochs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=250)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _raw_row(label: str, raw: dict[str, Any]) -> dict[str, Any]:
    games = int(raw["game"])
    rounds = int(raw["round"])
    rank_counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
    rank_pt = sum(count * point for count, point in zip(rank_counts, RANK_POINTS, strict=True))
    if games <= 0 or rounds <= 0:
        raise ValueError(f"invalid game/round counts for {label}: {raw}")
    return {
        "label": label,
        "games": games,
        "rounds": rounds,
        "rank_counts": rank_counts,
        "avg_rank": sum((rank + 1) * count for rank, count in enumerate(rank_counts)) / games,
        "avg_rank_pt": rank_pt / games,
        "avg_score_delta": float(raw["point"]) / games,
        "agari_rate": int(raw["agari"]) / rounds,
        "houjuu_rate": int(raw["houjuu"]) / rounds,
        "fuuro_rate": int(raw["fuuro"]) / rounds,
        "riichi_rate": int(raw["riichi"]) / rounds,
        "ryukyoku_rate": int(raw["ryukyoku"]) / rounds,
        "tobi_rate": int(raw["tobi"]) / games,
        "agari": int(raw["agari"]),
        "houjuu": int(raw["houjuu"]),
        "fuuro": int(raw["fuuro"]),
        "riichi": int(raw["riichi"]),
    }


def _aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows for {label}")
    games = sum(int(row["games"]) for row in rows)
    rounds = sum(int(row["rounds"]) for row in rows)
    rank_counts = [sum(int(row["rank_counts"][index]) for row in rows) for index in range(4)]
    agari = sum(int(row["agari"]) for row in rows)
    houjuu = sum(int(row["houjuu"]) for row in rows)
    fuuro = sum(int(row["fuuro"]) for row in rows)
    riichi = sum(int(row["riichi"]) for row in rows)
    return {
        "label": label,
        "runs": len(rows),
        "games": games,
        "rounds": rounds,
        "rank_counts": rank_counts,
        "avg_rank": sum((rank + 1) * count for rank, count in enumerate(rank_counts)) / games,
        "avg_rank_pt": sum(
            count * point for count, point in zip(rank_counts, RANK_POINTS, strict=True)
        ) / games,
        "avg_score_delta": sum(float(row["avg_score_delta"]) * int(row["games"]) for row in rows) / games,
        "agari_rate": agari / rounds,
        "houjuu_rate": houjuu / rounds,
        "fuuro_rate": fuuro / rounds,
        "riichi_rate": riichi / rounds,
        "ryukyoku_rate": sum(int(row["ryukyoku_rate"] * row["rounds"]) for row in rows) / rounds,
        "tobi_rate": sum(float(row["tobi_rate"]) * int(row["games"]) for row in rows) / games,
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _paired_rows(run_dir: Path, expected_games: int) -> list[dict[str, Any]]:
    """Read one F/G result from each hanchan, preserving the game as a cluster."""
    path = run_dir / "platform_accounts" / "per_game_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    games: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row["model_label"])
            if not (label.startswith("F_") or label.startswith("G_")):
                continue
            source_log = str(row["source_log"])
            games.setdefault(source_log, {})[label[:1]] = int(row["rank"])
    if len(games) != expected_games:
        raise ValueError(f"{run_dir}: paired CSV has {len(games)} games, expected {expected_games}")
    result = []
    for source_log, ranks in sorted(games.items()):
        if set(ranks) != {"F", "G"}:
            raise ValueError(f"{run_dir}: missing F/G rank in {source_log}: {ranks}")
        f_rank = ranks["F"]
        g_rank = ranks["G"]
        f_pt = RANK_POINTS[f_rank - 1]
        g_pt = RANK_POINTS[g_rank - 1]
        result.append(
            {
                "source_log": source_log,
                "f_rank": f_rank,
                "g_rank": g_rank,
                "f_rank_pt": f_pt,
                "g_rank_pt": g_pt,
                "delta_pt": g_pt - f_pt,
                "delta_rank": f_rank - g_rank,
                "g_ahead": int(g_rank < f_rank),
                "tie": int(g_rank == f_rank),
            }
        )
    return result


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _hierarchical_delta_pt_ci(
    per_seed_rows: list[list[dict[str, Any]]],
    rng: np.random.Generator,
    reps: int,
) -> list[float]:
    """Bootstrap seed pairs outside and hanchans inside, weighting seeds equally."""
    if not per_seed_rows or any(not rows for rows in per_seed_rows):
        raise ValueError("cannot hierarchically bootstrap an empty seed group")
    seed_count = len(per_seed_rows)
    seed_values = [
        np.asarray([float(row["delta_pt"]) for row in rows], dtype=np.float64)
        for rows in per_seed_rows
    ]
    outer_indices = rng.integers(0, seed_count, size=(reps, seed_count))
    estimates = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        selected_means = []
        for seed_index in outer_indices[rep]:
            values = seed_values[int(seed_index)]
            inner_indices = rng.integers(0, values.size, size=values.size)
            selected_means.append(float(values[inner_indices].mean()))
        estimates[rep] = float(np.mean(selected_means))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _paired_summary(
    rows: list[dict[str, Any]],
    label: str,
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no paired rows for {label}")
    delta_pt = np.asarray([float(row["delta_pt"]) for row in rows], dtype=np.float64)
    delta_rank = np.asarray([float(row["delta_rank"]) for row in rows], dtype=np.float64)
    g_ahead = np.asarray([float(row["g_ahead"]) for row in rows], dtype=np.float64)
    ties = np.asarray([float(row["tie"]) for row in rows], dtype=np.float64)
    f_counts = [sum(int(row["f_rank"] == rank) for row in rows) for rank in range(1, 5)]
    g_counts = [sum(int(row["g_rank"] == rank) for row in rows) for rank in range(1, 5)]
    return {
        "label": label,
        "games": len(rows),
        "mean_delta_pt": float(delta_pt.mean()),
        "median_delta_pt": float(np.median(delta_pt)),
        "mean_delta_rank": float(delta_rank.mean()),
        "g_ahead_rate": float(g_ahead.mean()),
        "tie_rate": float(ties.mean()),
        "delta_pt_bootstrap_95ci": _bootstrap_mean_ci(delta_pt, rng, bootstrap_reps),
        "delta_rank_bootstrap_95ci": _bootstrap_mean_ci(delta_rank, rng, bootstrap_reps),
        "g_ahead_bootstrap_95ci": _bootstrap_mean_ci(g_ahead, rng, bootstrap_reps),
        "f_rank_counts": f_counts,
        "g_rank_counts": g_counts,
        "rank_rate_diff_pp": [
            100.0 * (g_count - f_count) / len(rows)
            for f_count, g_count in zip(f_counts, g_counts, strict=True)
        ],
        "bootstrap_reps": bootstrap_reps,
        "cluster": "hanchan",
    }


def main() -> None:
    args = parse_args()
    eval_roots = [path.resolve() for path in args.eval_roots]
    run_dirs = sorted(
        path
        for eval_root in eval_roots
        for path in eval_root.glob("F_G_*")
        if path.is_dir()
    )
    if not run_dirs:
        raise SystemExit(f"no F_G_* evaluation directories under {eval_roots}")
    run_names = [path.name for path in run_dirs]
    if len(run_names) != len(set(run_names)):
        raise ValueError(f"duplicate evaluation run names across roots: {run_names}")

    per_seed: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    paired_per_seed: list[dict[str, Any]] = []
    bootstrap_rng = np.random.default_rng(20260719)
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        detailed = _read_json(run_dir / "detailed_stats.json")
        log_count = len(list((run_dir / "logs").glob("*.json.gz")))
        if log_count != args.expected_games:
            raise ValueError(f"{run_dir}: expected {args.expected_games} logs, got {log_count}")
        run = metrics.get("run", {})
        if run.get("games") != args.expected_games:
            raise ValueError(f"{run_dir}: metrics games mismatch: {run.get('games')}")
        if run.get("device") != "cuda":
            raise ValueError(f"{run_dir}: expected CUDA device, got {run.get('device')}")
        if run.get("seat_mode") != "random":
            raise ValueError(f"{run_dir}: expected random seat mode")
        labels = [label for label in detailed["players"] if label in {"70k", "ext_mortal"} or label.startswith(("F_", "G_"))]
        if len(labels) != 4:
            raise ValueError(f"{run_dir}: unexpected players: {list(detailed['players'])}")
        seed = next(label.split("_", 1)[1] for label in labels if label.startswith("F_"))
        rows = {
            label: _raw_row(label, detailed["players"][label]["raw"])
            for label in labels
        }
        per_seed.append({"run": run_dir.name, "seed": int(seed), "models": rows})
        paired_per_seed.append(
            {
                "run": run_dir.name,
                "eval_root": str(run_dir.parent),
                "seed": int(seed),
                "paired": _paired_summary(
                    _paired_rows(run_dir, args.expected_games),
                    run_dir.name,
                    bootstrap_rng,
                    bootstrap_reps=5000,
                ),
            }
        )
        source_checks.append(
            {
                "run": run_dir.name,
                "eval_root": str(run_dir.parent),
                "games": log_count,
                "metrics": str(run_dir / "metrics.json"),
                "detailed_stats": str(run_dir / "detailed_stats.json"),
                "platform_accounts": (run_dir / "platform_accounts").is_dir(),
                "seed_start": run.get("seed_start"),
                "seed_key": run.get("seed_key"),
                "device": run.get("device"),
                "seat_mode": run.get("seat_mode"),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in per_seed:
        for label, row in item["models"].items():
            grouped.setdefault("F" if label.startswith("F_") else "G" if label.startswith("G_") else label, []).append(row)
    pooled = {label: _aggregate(rows, label) for label, rows in sorted(grouped.items())}
    pairwise: list[dict[str, Any]] = []
    for item in per_seed:
        f = next(row for label, row in item["models"].items() if label.startswith("F_"))
        g = next(row for label, row in item["models"].items() if label.startswith("G_"))
        pairwise.append(
            {
                "run": item["run"],
                "seed": item["seed"],
                "F_avg_rank": f["avg_rank"],
                "G_avg_rank": g["avg_rank"],
                "F_avg_rank_pt": f["avg_rank_pt"],
                "G_avg_rank_pt": g["avg_rank_pt"],
                "G_minus_F_avg_rank_pt": g["avg_rank_pt"] - f["avg_rank_pt"],
                "G_minus_F_avg_rank": g["avg_rank"] - f["avg_rank"],
                "G_minus_F_agari_pp": (g["agari_rate"] - f["agari_rate"]) * 100.0,
                "G_minus_F_houjuu_pp": (g["houjuu_rate"] - f["houjuu_rate"]) * 100.0,
                "G_minus_F_fuuro_pp": (g["fuuro_rate"] - f["fuuro_rate"]) * 100.0,
                "G_minus_F_riichi_pp": (g["riichi_rate"] - f["riichi_rate"]) * 100.0,
            }
        )

    all_paired_rows = [
        row
        for run_dir in run_dirs
        for row in _paired_rows(run_dir, args.expected_games)
    ]
    pooled_paired = _paired_summary(
        all_paired_rows,
        "pooled_hanchans",
        bootstrap_rng,
        bootstrap_reps=5000,
    )
    hierarchical_ci = _hierarchical_delta_pt_ci(
        [[row for row in _paired_rows(run_dir, args.expected_games)] for run_dir in run_dirs],
        bootstrap_rng,
        reps=5000,
    )
    seed_means = [float(item["paired"]["mean_delta_pt"]) for item in paired_per_seed]
    eps = 1e-12
    non_tie_seed_means = [value for value in seed_means if abs(value) > eps]
    positive_seed_count = sum(value > 0 for value in non_tie_seed_means)
    seed_count = len(non_tie_seed_means)
    sign_test_p = (
        sum(math.comb(seed_count, k) for k in range(positive_seed_count, seed_count + 1)) / (2 ** seed_count)
        if seed_count
        else 1.0
    )
    recipe_summary = {
        "training_seed_count": len(seed_means),
        "seed_mean_delta_pt": seed_means,
        "non_tie_seed_mean_delta_pt": non_tie_seed_means,
        "mean_of_seed_means_delta_pt": float(np.mean(seed_means)),
        "median_of_seed_means_delta_pt": float(np.median(seed_means)),
        "positive_seed_count": positive_seed_count,
        "non_tie_seed_count": seed_count,
        "tie_seed_count": len(seed_means) - seed_count,
        "seed_direction_sign_test_one_sided_p": float(sign_test_p),
        "interpretation": "sign test excludes seed-level deltas within eps of zero; n is the number of non-tied training seeds, not hanchans",
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "keqing.mortal.reward_ab_eval_summary.v1",
        "eval_root": str(eval_roots[0]) if len(eval_roots) == 1 else None,
        "eval_roots": [str(path) for path in eval_roots],
        "expected_games_per_pair": args.expected_games,
        "run_count": len(per_seed),
        "rank_points": list(RANK_POINTS),
        "source_checks": source_checks,
        "per_seed": per_seed,
        "pairwise": pairwise,
        "paired_per_seed": paired_per_seed,
        "pooled_paired": pooled_paired,
        "hierarchical_paired": {
            "delta_pt_bootstrap_95ci": hierarchical_ci,
            "outer_cluster": "training_seed_pair",
            "inner_cluster": "hanchan",
            "seed_weighting": "equal",
            "bootstrap_reps": 5000,
        },
        "recipe_seed_summary": recipe_summary,
        "pooled": pooled,
        "interpretation": {
            "scope": f"{args.expected_games} hanchans per matched seed pair; paired screening evaluation",
            "favorable_G_pairs_by_avg_rank_pt": sum(1 for row in pairwise if row["G_minus_F_avg_rank_pt"] > 0),
            "pair_count": len(pairwise),
            "pooled_paired_delta_pt_ci": pooled_paired["delta_pt_bootstrap_95ci"],
        },
    }
    json_path = output_dir / f"reward_ab_eval_{args.expected_games}h_summary.json"
    markdown_path = output_dir / f"reward_ab_eval_{args.expected_games}h_summary.md"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Reward Semantics A/B: {args.expected_games}-Hanchan Screening",
        "",
        "This is a matched-seed native random-seat screening evaluation.",
        "All rows are reported as separate F/G results; no two-way aggregate is used.",
        "",
        "## Per Pair",
        "",
        "| Pair | F avg rank | G avg rank | F avg Pt | G avg Pt | G-F Pt | G-F agari | G-F houjuu | G-F fuuro | G-F riichi |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise:
        lines.append(
            f"| {row['run']} | {row['F_avg_rank']:.3f} | {row['G_avg_rank']:.3f} | "
            f"{row['F_avg_rank_pt']:.2f} | {row['G_avg_rank_pt']:.2f} | "
            f"{row['G_minus_F_avg_rank_pt']:+.2f} | {row['G_minus_F_agari_pp']:+.2f}pp | "
            f"{row['G_minus_F_houjuu_pp']:+.2f}pp | {row['G_minus_F_fuuro_pp']:+.2f}pp | "
            f"{row['G_minus_F_riichi_pp']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Paired Hanchan Differential",
            "",
            "`delta_pt = Pt(G) - Pt(F)` and `delta_rank = rank(F) - rank(G)`. Bootstrap resamples complete hanchans, not individual seats.",
            "",
            "| Scope | Games | Mean delta Pt | Median delta Pt | Mean delta rank | G ahead | 95% CI for mean delta Pt | Rank-rate diff (1st/2nd/3rd/4th) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in paired_per_seed + [{"run": "pooled_hanchans", "paired": pooled_paired}]:
        row = item["paired"]
        lines.append(
            f"| {item['run']} | {row['games']} | {row['mean_delta_pt']:+.2f} | {row['median_delta_pt']:+.2f} | "
            f"{row['mean_delta_rank']:+.3f} | {_fmt_pct(row['g_ahead_rate'])} | "
            f"[{row['delta_pt_bootstrap_95ci'][0]:+.2f}, {row['delta_pt_bootstrap_95ci'][1]:+.2f}] | "
            f"{[f'{value:+.2f}pp' for value in row['rank_rate_diff_pp']]} |"
        )
    lines.extend(
        [
            "",
            "## Training-Seed View",
            "",
            f"- Seed-level mean delta Pt: `{[round(value, 2) for value in seed_means]}`.",
            f"- Positive non-tie seed count: `{recipe_summary['positive_seed_count']}/{recipe_summary['non_tie_seed_count']}`; one-sided sign-test p-value under the zero-direction null: `{recipe_summary['seed_direction_sign_test_one_sided_p']:.4f}`.",
            "- The hanchan bootstrap CI measures arena uncertainty conditional on these checkpoints; it does not remove the separate training-seed uncertainty.",
            f"- Hierarchical seed-weighted bootstrap CI (outer training seed, inner hanchan): `[{hierarchical_ci[0]:+.2f}, {hierarchical_ci[1]:+.2f}]`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Pooled Auxiliary View",
            "",
            "| Model | Games | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | Rank counts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label in ("70k", "ext_mortal", "F", "G"):
        row = pooled[label]
        lines.append(
            f"| {label} | {row['games']} | {row['avg_rank']:.3f} | {row['avg_rank_pt']:.2f} | "
            f"{_fmt_pct(row['agari_rate'])} | {_fmt_pct(row['houjuu_rate'])} | "
            f"{_fmt_pct(row['fuuro_rate'])} | {_fmt_pct(row['riichi_rate'])} | {row['rank_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- G is ahead of F on average rank Pt in {document['interpretation']['favorable_G_pairs_by_avg_rank_pt']}/{len(pairwise)} matched pairs.",
            "- This is a direction check; reward promotion still requires interpreting both the paired hanchan CI and the separate training-seed uncertainty.",
            "- The 70k and ext_mortal rows are controls for this lineup, not a claim that this screen replaces the final model-pool league.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(per_seed), "pooled": pooled, "pairwise": pairwise}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
