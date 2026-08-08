#!/usr/bin/env python3
"""Summarize the matched mixed-vs-pure replay route evaluation.

The unit of pairing is one complete hanchan. Training seeds are kept as a
separate outer sampling unit so arena uncertainty is not presented as recipe
uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any


DEFAULT_SEEDS = (20260731, 20260801, 20260802)
DEFAULT_MODELS = ("70k", "ext_mortal", "M_candidate", "S_candidate")
RANK_POINTS = (90.0, 45.0, 0.0, -135.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        action="append",
        required=True,
        help="Evaluation directory containing seed_<id>/platform_accounts/per_game_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the JSON and Markdown summary.",
    )
    parser.add_argument("--mixed-label", default="M_candidate")
    parser.add_argument("--pure-label", default="S_candidate")
    parser.add_argument("--anchor-label", default="70k")
    parser.add_argument("--external-label", default="ext_mortal")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _ci(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "ci95_low": _percentile(values, 0.025),
        "ci95_high": _percentile(values, 0.975),
    }


def _sign_test_one_sided(values: list[float], epsilon: float = 1e-12) -> dict[str, Any]:
    non_ties = [value for value in values if abs(value) > epsilon]
    positive = sum(value > 0 for value in non_ties)
    n = len(non_ties)
    tail = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n) if n else None
    return {
        "positive_count": positive,
        "negative_count": n - positive,
        "tie_count": len(values) - n,
        "non_tie_count": n,
        "one_sided_p": tail,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(
    eval_dir: Path,
    *,
    mixed_label: str,
    pure_label: str,
    anchor_label: str,
    external_label: str,
) -> dict[str, Any]:
    results_path = eval_dir / "platform_accounts" / "per_game_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"missing platform result table: {results_path}")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["source_log"]), int(row["game_index"]))
            grouped.setdefault(key, {})[str(row["model_label"])] = row
    required = {mixed_label, pure_label, anchor_label, external_label}
    if len(grouped) != 1000:
        raise ValueError(f"{eval_dir}: expected 1000 hanchans, got {len(grouped)}")
    pairs: list[dict[str, Any]] = []
    for (source_log, game_index), players in sorted(grouped.items(), key=lambda item: item[0]):
        missing = required - players.keys()
        if missing:
            raise ValueError(f"{eval_dir}: {source_log} missing players {sorted(missing)}")
        parsed = {
            label: {
                "rank": int(players[label]["rank"]),
                "final_score": int(players[label]["final_score"]),
                "pt": float(RANK_POINTS[int(players[label]["rank"]) - 1]),
            }
            for label in required
        }
        pairs.append(
            {
                "source_log": source_log,
                "game_index": game_index,
                "players": parsed,
                "delta_pt_pure_minus_mixed": parsed[pure_label]["pt"] - parsed[mixed_label]["pt"],
                "delta_pt_pure_minus_anchor": parsed[pure_label]["pt"] - parsed[anchor_label]["pt"],
                "delta_pt_mixed_minus_anchor": parsed[mixed_label]["pt"] - parsed[anchor_label]["pt"],
                "delta_rank_pure_minus_mixed": parsed[mixed_label]["rank"] - parsed[pure_label]["rank"],
                "pure_ahead_of_mixed": parsed[pure_label]["rank"] < parsed[mixed_label]["rank"],
            }
        )
    metrics_path = eval_dir / "metrics.json"
    detailed_path = eval_dir / "detailed_stats.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    detailed = json.loads(detailed_path.read_text(encoding="utf-8")) if detailed_path.exists() else None
    return {
        "eval_dir": str(eval_dir),
        "eval_dir_name": eval_dir.name,
        "rows": pairs,
        "metrics": metrics,
        "detailed_stats": detailed,
        "source_results_sha256": _sha256(results_path),
    }


def _bootstrap_hanchans(values: list[float], rng: random.Random, replicates: int) -> dict[str, float]:
    n = len(values)
    samples = [_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(replicates)]
    return _ci(samples)


def _hierarchical_bootstrap(
    seed_values: list[list[float]], rng: random.Random, replicates: int
) -> dict[str, float]:
    if not seed_values or any(not values for values in seed_values):
        raise ValueError("hierarchical bootstrap requires non-empty seed samples")
    samples: list[float] = []
    seed_count = len(seed_values)
    for _ in range(replicates):
        outer = [seed_values[rng.randrange(seed_count)] for _ in range(seed_count)]
        inner_means = []
        for values in outer:
            n = len(values)
            inner_means.append(_mean([values[rng.randrange(n)] for _ in range(n)]))
        samples.append(_mean(inner_means))
    return _ci(samples)


def _aggregate_metric(rows: list[dict[str, Any]], label: str) -> dict[str, float]:
    ranks = [row["players"][label]["rank"] for row in rows]
    pts = [row["players"][label]["pt"] for row in rows]
    return {
        "games": len(rows),
        "avg_rank": _mean([float(value) for value in ranks]),
        "avg_pt": _mean(pts),
        "rank_1_rate": ranks.count(1) / len(ranks),
        "rank_2_rate": ranks.count(2) / len(ranks),
        "rank_3_rate": ranks.count(3) / len(ranks),
        "rank_4_rate": ranks.count(4) / len(ranks),
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# M0 Mixed vs S0 Pure Selfplay Route A/B",
        "",
        f"- Eval seeds: `{', '.join(summary['training_seeds'])}`",
        f"- Hanchans per seed: `{summary['hanchans_per_seed']}`",
        "- Unit of pairing: one complete hanchan; no seat-level resampling.",
        "- Rank points: `[90, 45, 0, -135]`.",
        "",
        "## Decision Pairs",
        "",
        "| Training seed | S-M Pt | S-70k Pt | M-70k Pt | S ahead of M | S-M rank delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_seed"]:
        lines.append(
            "| {seed} | {sm} | {sa} | {ma} | {ahead:.3f} | {rank:.3f} |".format(
                seed=row["training_seed"],
                sm=_format(row["paired"]["delta_pt_pure_minus_mixed"]["mean"]),
                sa=_format(row["paired"]["delta_pt_pure_minus_anchor"]["mean"]),
                ma=_format(row["paired"]["delta_pt_mixed_minus_anchor"]["mean"]),
                ahead=row["paired"]["pure_ahead_rate"],
                rank=row["paired"]["delta_rank_pure_minus_mixed_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Seed-Level Summary",
            "",
            "| Pair | Mean | Median | Equal-seed hierarchical CI 95% |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, title in (
        ("delta_pt_pure_minus_mixed", "S-M Pt"),
        ("delta_pt_pure_minus_anchor", "S-70k Pt"),
        ("delta_pt_mixed_minus_anchor", "M-70k Pt"),
    ):
        item = summary["seed_level"][key]
        lines.append(
            f"| {title} | {_format(item['mean'])} | {_format(item['median'])} | "
            f"[{_format(item['hierarchical_bootstrap']['ci95_low'])}, "
            f"{_format(item['hierarchical_bootstrap']['ci95_high'])}] |"
        )
    sign = summary["seed_level"]["delta_pt_pure_minus_mixed"]["sign_test"]
    lines.extend(
        [
            "",
            f"S-M positive seeds: `{sign['positive_count']}/{sign['non_tie_count']}`; "
            f"one-sided exact sign-test p=`{_format(sign['one_sided_p'], 6)}`.",
            "",
            "## Aggregate Model Readout",
            "",
            "| Model | Avg Pt | Avg rank | 1st | 2nd | 3rd | 4th |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, metrics in summary["aggregate_models"].items():
        lines.append(
            f"| {label} | {_format(metrics['avg_pt'])} | {_format(metrics['avg_rank'])} | "
            f"{_format(metrics['rank_1_rate'] * 100, 2)}% | "
            f"{_format(metrics['rank_2_rate'] * 100, 2)}% | "
            f"{_format(metrics['rank_3_rate'] * 100, 2)}% | "
            f"{_format(metrics['rank_4_rate'] * 100, 2)}% |"
        )
    lines.extend(["", "## Behavior Readout", ""])
    behavior_keys = (
        "agari_rate",
        "houjuu_rate",
        "fuuro_rate",
        "riichi_rate",
        "agari_rate_after_fuuro",
        "houjuu_rate_after_fuuro",
        "agari_rate_after_riichi",
        "houjuu_rate_after_riichi",
    )
    lines.append("| Metric | " + " | ".join(summary["models"]) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in summary["models"]) + " |")
    for key in behavior_keys:
        values = []
        for label in summary["models"]:
            per_seed = [
                row["behavior"].get(label, {}).get("derived", {}).get(key)
                for row in summary["per_seed"]
            ]
            per_seed = [float(value) for value in per_seed if value is not None]
            values.append(_format(_mean(per_seed) * 100 if per_seed else None, 3) + "%")
        lines.append("| " + " | ".join([key, *values]) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is a fixed-corpus route comparison. It tests mixed replay ecology "
            "versus pure ext_mortal selfplay under the same preserved-Adam continuation "
            "contract; it does not establish a new default checkpoint or an oracle label.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    eval_dirs = [Path(value) for value in args.eval_root]
    names = [path.name for path in eval_dirs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate evaluation directory names")
    loaded = [
        _load_rows(
            path,
            mixed_label=args.mixed_label,
            pure_label=args.pure_label,
            anchor_label=args.anchor_label,
            external_label=args.external_label,
        )
        for path in eval_dirs
    ]
    seeds = []
    for item in loaded:
        name = item["eval_dir_name"]
        if not name.startswith("seed_"):
            raise ValueError(f"evaluation directory must be seed_<id>, got {name}")
        seeds.append(int(name.removeprefix("seed_")))
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate training seeds")
    if set(seeds) != set(DEFAULT_SEEDS):
        raise ValueError(f"expected pre-registered seeds {DEFAULT_SEEDS}, got {tuple(seeds)}")
    source_hashes = [item["source_results_sha256"] for item in loaded]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("duplicate evaluation result payloads")
    loaded.sort(key=lambda item: int(item["eval_dir_name"].removeprefix("seed_")))

    pair_keys = (
        "delta_pt_pure_minus_mixed",
        "delta_pt_pure_minus_anchor",
        "delta_pt_mixed_minus_anchor",
    )
    per_seed = []
    seed_values: dict[str, list[list[float]]] = {key: [] for key in pair_keys}
    for item in loaded:
        rows = item["rows"]
        pair_summary: dict[str, Any] = {}
        for key in pair_keys:
            values = [float(row[key]) for row in rows]
            seed_values[key].append(values)
            pair_summary[key] = {
                "mean": _mean(values),
                "median": _percentile(values, 0.5),
                "hanchan_bootstrap": _bootstrap_hanchans(
                    values, random.Random(args.seed + len(per_seed) * 100 + len(key)), args.bootstrap_replicates
                ),
            }
        pair_summary["pure_ahead_rate"] = _mean(
            [1.0 if row["pure_ahead_of_mixed"] else 0.0 for row in rows]
        )
        pair_summary["delta_rank_pure_minus_mixed_mean"] = _mean(
            [float(row["delta_rank_pure_minus_mixed"]) for row in rows]
        )
        behavior = (item.get("detailed_stats") or {}).get("players", {})
        per_seed.append(
            {
                "training_seed": int(item["eval_dir_name"].removeprefix("seed_")),
                "eval_dir": item["eval_dir"],
                "paired": pair_summary,
                "behavior": behavior,
                "source_results_sha256": item["source_results_sha256"],
            }
        )

    rng = random.Random(args.seed)
    seed_level: dict[str, Any] = {}
    for key in pair_keys:
        means = [_mean(values) for values in seed_values[key]]
        seed_level[key] = {
            "mean": _mean(means),
            "median": _percentile(means, 0.5),
            "seed_means": means,
            "sign_test": _sign_test_one_sided(means),
            "hierarchical_bootstrap": _hierarchical_bootstrap(
                seed_values[key], rng, args.bootstrap_replicates
            ),
        }

    all_rows = [row for item in loaded for row in item["rows"]]
    aggregate_models = {
        label: _aggregate_metric(all_rows, label)
        for label in (args.anchor_label, args.external_label, args.mixed_label, args.pure_label)
    }
    summary = {
        "schema": "keqing.mortal.data_route_ab_summary.v1",
        "experiment": "data_route_ab_2026_07",
        "training_seeds": [str(seed) for seed in sorted(seeds)],
        "hanchans_per_seed": len(loaded[0]["rows"]),
        "models": [args.anchor_label, args.external_label, args.mixed_label, args.pure_label],
        "rank_points": list(RANK_POINTS),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "random_seed": int(args.seed),
        "per_seed": per_seed,
        "seed_level": seed_level,
        "aggregate_models": aggregate_models,
        "source_eval_dirs": [item["eval_dir"] for item in loaded],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_route_ab_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "data_route_ab_summary.md").write_text(
        _build_markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = summarize(_parse_args())
    print(json.dumps(summary["seed_level"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
