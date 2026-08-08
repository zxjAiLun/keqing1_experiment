#!/usr/bin/env python3
"""Summarize the matched fresh-Adam/preserved-Adam native evaluation."""

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
        required=True,
        help="evaluation root; repeat to combine matched training-seed batches",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _hierarchical_ci(
    per_seed: list[np.ndarray], rng: np.random.Generator, reps: int
) -> list[float]:
    seed_count = len(per_seed)
    outer = rng.integers(0, seed_count, size=(reps, seed_count))
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        means = []
        for seed_index in outer[index]:
            values = per_seed[int(seed_index)]
            inner = rng.integers(0, values.size, size=values.size)
            means.append(float(values[inner].mean()))
        estimates[index] = float(np.mean(means))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _paired_rows(run_dir: Path, expected_games: int) -> list[dict[str, Any]]:
    path = run_dir / "platform_accounts" / "per_game_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    games: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row["model_label"])
            if label.startswith("fresh_"):
                key = "fresh"
            elif label.startswith("preserved_"):
                key = "preserved"
            elif label == "70k":
                key = "70k"
            else:
                continue
            source = str(row["source_log"])
            games.setdefault(source, {})[key] = {
                "rank": int(row["rank"]),
                "final_score": int(row["final_score"]),
                "pt": RANK_POINTS[int(row["rank"]) - 1],
            }
    if len(games) != expected_games:
        raise ValueError(f"{run_dir}: found {len(games)} paired games, expected {expected_games}")
    rows = []
    for source, values in sorted(games.items()):
        if set(values) != {"70k", "fresh", "preserved"}:
            raise ValueError(f"{run_dir}: incomplete pair in {source}: {values}")
        anchor = values["70k"]
        fresh = values["fresh"]
        preserved = values["preserved"]
        rows.append(
            {
                "source_log": source,
                "70k_rank": anchor["rank"],
                "70k_final_score": anchor["final_score"],
                "fresh_rank": fresh["rank"],
                "preserved_rank": preserved["rank"],
                "fresh_final_score": fresh["final_score"],
                "preserved_final_score": preserved["final_score"],
                "delta_pt": preserved["pt"] - fresh["pt"],
                "delta_rank": fresh["rank"] - preserved["rank"],
                "preserved_minus_70k_pt": preserved["pt"] - anchor["pt"],
                "fresh_minus_70k_pt": fresh["pt"] - anchor["pt"],
                "preserved_minus_70k_rank": preserved["rank"] - anchor["rank"],
                "fresh_minus_70k_rank": fresh["rank"] - anchor["rank"],
                "preserved_ahead": int(preserved["rank"] < fresh["rank"]),
                "tie": int(preserved["rank"] == fresh["rank"]),
            }
        )
    return rows


def _paired_summary(rows: list[dict[str, Any]], rng: np.random.Generator, reps: int) -> dict[str, Any]:
    delta_pt = np.asarray([row["delta_pt"] for row in rows], dtype=np.float64)
    delta_rank = np.asarray([row["delta_rank"] for row in rows], dtype=np.float64)
    ahead = np.asarray([row["preserved_ahead"] for row in rows], dtype=np.float64)
    ties = np.asarray([row["tie"] for row in rows], dtype=np.float64)
    fresh_counts = [sum(row["fresh_rank"] == rank for row in rows) for rank in range(1, 5)]
    preserved_counts = [sum(row["preserved_rank"] == rank for row in rows) for rank in range(1, 5)]
    return {
        "games": len(rows),
        "mean_delta_pt_preserved_minus_fresh": float(delta_pt.mean()),
        "median_delta_pt_preserved_minus_fresh": float(np.median(delta_pt)),
        "mean_delta_rank_fresh_minus_preserved": float(delta_rank.mean()),
        "preserved_ahead_rate": float(ahead.mean()),
        "tie_rate": float(ties.mean()),
        "delta_pt_bootstrap_95ci": _bootstrap_ci(delta_pt, rng, reps),
        "delta_rank_bootstrap_95ci": _bootstrap_ci(delta_rank, rng, reps),
        "preserved_ahead_bootstrap_95ci": _bootstrap_ci(ahead, rng, reps),
        "fresh_rank_counts": fresh_counts,
        "preserved_rank_counts": preserved_counts,
        "rank_rate_diff_pp_preserved_minus_fresh": [
            100.0 * (preserved - fresh) / len(rows)
            for fresh, preserved in zip(fresh_counts, preserved_counts, strict=True)
        ],
        "cluster": "complete_hanchan",
        "bootstrap_reps": reps,
    }


def _comparison_summary(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    """Summarize one candidate against the same 70k seat in each hanchan."""
    pt_key = f"{candidate}_minus_{baseline}_pt"
    rank_key = f"{candidate}_minus_{baseline}_rank"
    delta_pt = np.asarray([row[pt_key] for row in rows], dtype=np.float64)
    delta_rank = np.asarray([row[rank_key] for row in rows], dtype=np.float64)
    candidate_ranks = np.asarray([row[f"{candidate}_rank"] for row in rows], dtype=np.int64)
    baseline_ranks = np.asarray([row[f"{baseline}_rank"] for row in rows], dtype=np.int64)
    ahead = (candidate_ranks < baseline_ranks).astype(np.float64)
    ties = (candidate_ranks == baseline_ranks).astype(np.float64)
    candidate_counts = [int(np.sum(candidate_ranks == rank)) for rank in range(1, 5)]
    baseline_counts = [int(np.sum(baseline_ranks == rank)) for rank in range(1, 5)]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "games": len(rows),
        "mean_delta_pt_candidate_minus_baseline": float(delta_pt.mean()),
        "median_delta_pt_candidate_minus_baseline": float(np.median(delta_pt)),
        "mean_delta_rank_candidate_minus_baseline": float(delta_rank.mean()),
        "candidate_ahead_rate": float(ahead.mean()),
        "tie_rate": float(ties.mean()),
        "delta_pt_bootstrap_95ci": _bootstrap_ci(delta_pt, rng, reps),
        "delta_rank_bootstrap_95ci": _bootstrap_ci(delta_rank, rng, reps),
        "candidate_ahead_bootstrap_95ci": _bootstrap_ci(ahead, rng, reps),
        "candidate_rank_counts": candidate_counts,
        "baseline_rank_counts": baseline_counts,
        "rank_rate_diff_pp_candidate_minus_baseline": [
            100.0 * (candidate_count - baseline_count) / len(rows)
            for candidate_count, baseline_count in zip(candidate_counts, baseline_counts, strict=True)
        ],
        "cluster": "complete_hanchan",
        "bootstrap_reps": reps,
    }


def _exact_sign_test(seed_means: list[float], eps: float = 1e-12) -> dict[str, Any]:
    non_ties = [value for value in seed_means if abs(value) > eps]
    positive_count = sum(value > 0 for value in non_ties)
    count = len(non_ties)
    p_value = (
        sum(math.comb(count, k) for k in range(positive_count, count + 1)) / (2**count)
        if count
        else 1.0
    )
    return {
        "positive_seed_count": positive_count,
        "non_tie_seed_count": count,
        "tie_seed_count": len(seed_means) - count,
        "one_sided_p": float(p_value),
        "null": "each non-tied seed direction is independently positive with probability 0.5",
    }


def _model_snapshot(run_dir: Path, label: str) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json")
    detailed = _read_json(run_dir / "detailed_stats.json")
    metric = metrics["metrics"][label]
    player = detailed["players"][label]
    raw = player["raw"]
    derived = player["derived"]
    fields = (
        "agari_rate",
        "houjuu_rate",
        "fuuro_rate",
        "riichi_rate",
        "agari_rate_after_riichi",
        "houjuu_rate_after_riichi",
        "agari_rate_after_fuuro",
        "houjuu_rate_after_fuuro",
        "avg_point_per_agari",
        "avg_point_per_houjuu",
    )
    return {
        "games": int(metric["games"]),
        "rank_counts": metric["rank_counts"],
        "avg_rank": float(metric["avg_rank"]),
        "avg_rank_pt": float(metric["avg_rank_pt"]),
        "raw": {key: raw[key] for key in ("game", "round", "agari", "houjuu", "fuuro", "riichi")},
        "derived": {key: float(derived[key]) for key in fields if key in derived},
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    args = parse_args()
    eval_roots = [root.resolve() for root in args.eval_root]
    if len(eval_roots) != len(set(eval_roots)):
        raise ValueError(f"duplicate evaluation roots: {eval_roots}")
    run_dirs = sorted(
        run_dir
        for eval_root in eval_roots
        for run_dir in eval_root.glob("lineup_*")
    )
    if len(run_dirs) < 3:
        raise ValueError(f"expected at least three lineup directories under {eval_roots}, found {len(run_dirs)}")
    run_names = [run_dir.name for run_dir in run_dirs]
    if len(run_names) != len(set(run_names)):
        raise ValueError(f"duplicate lineup directories across evaluation roots: {run_names}")
    rng = np.random.default_rng(20260722)
    per_seed: list[dict[str, Any]] = []
    paired_arrays: list[np.ndarray] = []
    baseline_arrays: dict[str, list[np.ndarray]] = {
        "preserved_vs_70k": [],
        "fresh_vs_70k": [],
    }
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        run = metrics["run"]
        if run.get("games") != args.expected_games or run.get("device") != "cuda" or run.get("seat_mode") != "random":
            raise ValueError(f"{run_dir}: metrics contract mismatch")
        log_count = len(list((run_dir / "logs").glob("*.json.gz")))
        if log_count != args.expected_games:
            raise ValueError(f"{run_dir}: expected {args.expected_games} logs, got {log_count}")
        labels = list(metrics["metrics"])
        fresh = next(label for label in labels if label.startswith("fresh_"))
        preserved = next(label for label in labels if label.startswith("preserved_"))
        rows = _paired_rows(run_dir, args.expected_games)
        paired = _paired_summary(rows, rng, args.bootstrap_reps)
        preserved_vs_70k = _comparison_summary(rows, "preserved", "70k", rng, args.bootstrap_reps)
        fresh_vs_70k = _comparison_summary(rows, "fresh", "70k", rng, args.bootstrap_reps)
        paired_arrays.append(np.asarray([row["delta_pt"] for row in rows], dtype=np.float64))
        baseline_arrays["preserved_vs_70k"].append(
            np.asarray([row["preserved_minus_70k_pt"] for row in rows], dtype=np.float64)
        )
        baseline_arrays["fresh_vs_70k"].append(
            np.asarray([row["fresh_minus_70k_pt"] for row in rows], dtype=np.float64)
        )
        per_seed.append(
            {
                "run": run_dir.name,
                "seed": int(fresh.split("_", 1)[1]),
                "seed_start": run["seed_start"],
                "source_checks": {
                    "games": log_count,
                    "seed_key": run["seed_key"],
                    "device": run["device"],
                    "seat_mode": run["seat_mode"],
                    "metrics": str(run_dir / "metrics.json"),
                    "detailed_stats": str(run_dir / "detailed_stats.json"),
                    "platform_accounts": (run_dir / "platform_accounts").is_dir(),
                },
                "models": {
                    "70k": _model_snapshot(run_dir, "70k"),
                    "ext_mortal": _model_snapshot(run_dir, "ext_mortal"),
                    "fresh": _model_snapshot(run_dir, fresh),
                    "preserved": _model_snapshot(run_dir, preserved),
                },
                "paired": paired,
                "baseline_comparisons": {
                    "preserved_vs_70k": preserved_vs_70k,
                    "fresh_vs_70k": fresh_vs_70k,
                },
            }
        )

    all_rows = [row for run_dir in run_dirs for row in _paired_rows(run_dir, args.expected_games)]
    pooled = _paired_summary(all_rows, rng, args.bootstrap_reps)
    hierarchical = _hierarchical_ci(paired_arrays, rng, args.bootstrap_reps)
    baseline_pooled = {
        name: _comparison_summary(
            all_rows,
            "preserved" if name.startswith("preserved") else "fresh",
            "70k",
            rng,
            args.bootstrap_reps,
        )
        for name in baseline_arrays
    }
    baseline_hierarchical = {
        name: _hierarchical_ci(values, rng, args.bootstrap_reps)
        for name, values in baseline_arrays.items()
    }
    seed_means = [item["paired"]["mean_delta_pt_preserved_minus_fresh"] for item in per_seed]
    seeds = [int(item["seed"]) for item in per_seed]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate training seeds across lineup directories: {seeds}")
    sign_test = _exact_sign_test(seed_means)
    document = {
        "schema": "keqing.mortal.optimizer_ab_eval_summary.v1",
        "eval_root": str(eval_roots[0]) if len(eval_roots) == 1 else [str(root) for root in eval_roots],
        "eval_roots": [str(root) for root in eval_roots],
        "expected_games_per_seed": args.expected_games,
        "training_seed_count": len(per_seed),
        "rank_points": list(RANK_POINTS),
        "per_seed": per_seed,
        "pooled_paired": pooled,
        "hierarchical_paired": {
            "delta_pt_bootstrap_95ci": hierarchical,
            "outer_cluster": "training_seed",
            "inner_cluster": "complete_hanchan",
            "seed_weighting": "equal",
            "bootstrap_reps": args.bootstrap_reps,
        },
        "baseline_comparisons": {
            name: {
                "pooled": baseline_pooled[name],
                "hierarchical_delta_pt_bootstrap_95ci": baseline_hierarchical[name],
                "outer_cluster": "training_seed",
                "inner_cluster": "complete_hanchan",
                "seed_weighting": "equal",
            }
            for name in baseline_arrays
        },
        "recipe_summary": {
            "seed_mean_delta_pt": seed_means,
            "mean_of_seed_means_delta_pt": float(np.mean(seed_means)),
            "median_of_seed_means_delta_pt": float(np.median(seed_means)),
            "positive_seed_count": sign_test["positive_seed_count"],
            "non_tie_seed_count": sign_test["non_tie_seed_count"],
            "tie_seed_count": sign_test["tie_seed_count"],
            "seed_direction_sign_test_one_sided_p": sign_test["one_sided_p"],
            "interpretation": "delta_pt is preserved Adam minus fresh Adam; hanchan and training-seed uncertainty are reported separately",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "optimizer_ab_eval_1000h_summary.json"
    md_path = args.output_dir / "optimizer_ab_eval_1000h_summary.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fresh Adam vs Preserved Adam: 1000-Hanchan Evaluation",
        "",
        "Matched native four-model random-seat evaluation. `delta_pt = preserved - fresh`; no two-way aggregate is used.",
        "",
        "## Per Training Seed",
        "",
        "| Seed | Fresh avg rank | Preserved avg rank | Fresh Pt | Preserved Pt | Preserved-Fresh Pt | Preserved ahead | Rank-rate diff (1/2/3/4) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in per_seed:
        f = item["models"]["fresh"]
        p = item["models"]["preserved"]
        pair = item["paired"]
        lines.append(
            f"| {item['seed']} | {f['avg_rank']:.3f} | {p['avg_rank']:.3f} | {f['avg_rank_pt']:+.2f} | "
            f"{p['avg_rank_pt']:+.2f} | {pair['mean_delta_pt_preserved_minus_fresh']:+.2f} | "
            f"{_fmt_pct(pair['preserved_ahead_rate'])} | "
            f"{[f'{v:+.2f}pp' for v in pair['rank_rate_diff_pp_preserved_minus_fresh']]} |"
        )
    lines.extend(
        [
            "",
            "## Paired Differential",
            "",
            "Bootstrap resamples complete hanchans. The hierarchical bootstrap resamples training seeds outside and hanchans inside, weighting seeds equally.",
            "",
            f"- Pooled mean delta Pt: `{pooled['mean_delta_pt_preserved_minus_fresh']:+.3f}`; hanchan bootstrap 95% CI: `[{pooled['delta_pt_bootstrap_95ci'][0]:+.3f}, {pooled['delta_pt_bootstrap_95ci'][1]:+.3f}]`.",
            f"- Seed means: `{[round(value, 3) for value in seed_means]}`; mean of seed means: `{np.mean(seed_means):+.3f}`.",
            f"- Equal-seed hierarchical bootstrap 95% CI: `[{hierarchical[0]:+.3f}, {hierarchical[1]:+.3f}]`.",
            f"- Preserved is ahead by average Pt in `{sign_test['positive_seed_count']}/{sign_test['non_tie_seed_count']}` non-tied training seeds; exact one-sided sign-test `p={sign_test['one_sided_p']:.3f}`.",
            "",
            "## Against 70k Anchor",
            "",
            "These are complete-hanchan paired comparisons inside the same four-model lineup. A negative rank delta means the candidate finished better than 70k.",
            "",
            "| Seed | Preserved - 70k Pt | Fresh - 70k Pt | Preserved - 70k rank | Fresh - 70k rank |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in per_seed:
        preserved = item["baseline_comparisons"]["preserved_vs_70k"]
        fresh = item["baseline_comparisons"]["fresh_vs_70k"]
        lines.append(
            f"| {item['seed']} | {preserved['mean_delta_pt_candidate_minus_baseline']:+.3f} | "
            f"{fresh['mean_delta_pt_candidate_minus_baseline']:+.3f} | "
            f"{preserved['mean_delta_rank_candidate_minus_baseline']:+.3f} | "
            f"{fresh['mean_delta_rank_candidate_minus_baseline']:+.3f} |"
        )
    for name, label in (("preserved_vs_70k", "Pooled preserved - 70k"), ("fresh_vs_70k", "Pooled fresh - 70k")):
        summary = baseline_pooled[name]
        ci = baseline_hierarchical[name]
        lines.extend(
            [
                "",
                f"- {label}: mean Pt `{summary['mean_delta_pt_candidate_minus_baseline']:+.3f}`, hanchan CI `[{summary['delta_pt_bootstrap_95ci'][0]:+.3f}, {summary['delta_pt_bootstrap_95ci'][1]:+.3f}]`, equal-seed hierarchical CI `[{ci[0]:+.3f}, {ci[1]:+.3f}]`.",
            ]
        )
    lines.extend(["", "## Controls and Behavior", "", "| Seed | Model | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | After-riichi A/H | After-fuuro A/H |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"])
    for item in per_seed:
        for label in ("70k", "ext_mortal", "fresh", "preserved"):
            model = item["models"][label]
            derived = model["derived"]
            lines.append(
                f"| {item['seed']} | {label} | {model['avg_rank']:.3f} | {model['avg_rank_pt']:+.2f} | "
                f"{_fmt_pct(derived.get('agari_rate', 0.0))} | {_fmt_pct(derived.get('houjuu_rate', 0.0))} | "
                f"{_fmt_pct(derived.get('fuuro_rate', 0.0))} | {_fmt_pct(derived.get('riichi_rate', 0.0))} | "
                f"{_fmt_pct(derived.get('agari_rate_after_riichi', 0.0))}/{_fmt_pct(derived.get('houjuu_rate_after_riichi', 0.0))} | "
                f"{_fmt_pct(derived.get('agari_rate_after_fuuro', 0.0))}/{_fmt_pct(derived.get('houjuu_rate_after_fuuro', 0.0))} |"
            )
    lines.extend(
        [
            "",
            "## Decision Scope",
            "",
            "This evaluates the optimizer state-transfer variable only. It does not promote a checkpoint to the serving/default model and does not establish that preserved Adam is a generally better optimizer recipe.",
            f"The final decision must consider all {len(per_seed)} seed-level effects together with the hierarchical interval; the pooled hanchan interval alone does not represent training-seed uncertainty.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "seed_means": seed_means, "hierarchical_ci": hierarchical}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
