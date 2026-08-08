#!/usr/bin/env python3
"""Summarize the three-seed D1 versus M0 B250 evaluation.

The pairing unit is one complete hanchan.  Shard-level averages are never
used as the primary comparison; every delta is reconstructed from the same
source log containing 70k, ext_mortal, M0 and D1.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SEEDS = (20260806, 20260807, 20260808)
EXPECTED_SEED_STARTS = {20260806: 1700000, 20260807: 1710000, 20260808: 1720000}
RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
MODEL_NAMES = ("70k", "ext_mortal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_scores(events: list[dict[str, Any]]) -> list[float] | None:
    scores: list[float] | None = None
    for event in events:
        if event.get("type") == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(value) for value in values]
        elif event.get("type") in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def ranks_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    if not events or events[0].get("type") != "start_game":
        raise ValueError("missing start_game")
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4 or len(set(names)) != 4:
        raise ValueError(f"invalid player names: {names!r}")
    scores = final_scores(events)
    if scores is None:
        raise ValueError("could not reconstruct final scores")
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    return {str(names[seat]): order.index(seat) + 1 for seat in range(4)}


def paired_row(seed: int, path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    ranks = ranks_from_events(events)
    required = ("70k", "ext_mortal", f"M0_{seed}", f"D1_{seed}")
    if set(ranks) != set(required):
        raise ValueError(f"{path}: expected names {required}, got {sorted(ranks)}")
    points = {label: RANK_POINTS[ranks[label] - 1] for label in required}
    return {
        "seed": seed,
        "source_log": str(path),
        "source_log_sha256": sha256(path),
        "rank_70k": ranks["70k"],
        "rank_ext_mortal": ranks["ext_mortal"],
        "rank_m0": ranks[f"M0_{seed}"],
        "rank_d1": ranks[f"D1_{seed}"],
        "pt_70k": points["70k"],
        "pt_ext_mortal": points["ext_mortal"],
        "pt_m0": points[f"M0_{seed}"],
        "pt_d1": points[f"D1_{seed}"],
        "delta_pt_d1_minus_m0": points[f"D1_{seed}"] - points[f"M0_{seed}"],
        "delta_pt_d1_minus_70k": points[f"D1_{seed}"] - points["70k"],
        "delta_pt_m0_minus_70k": points[f"M0_{seed}"] - points["70k"],
        "delta_rank_m0_minus_d1": ranks[f"M0_{seed}"] - ranks[f"D1_{seed}"],
        "delta_rank_70k_minus_d1": ranks["70k"] - ranks[f"D1_{seed}"],
        "delta_rank_70k_minus_m0": ranks["70k"] - ranks[f"M0_{seed}"],
        "d1_ahead_of_m0": ranks[f"D1_{seed}"] < ranks[f"M0_{seed}"],
        "d1_ahead_of_70k": ranks[f"D1_{seed}"] < ranks["70k"],
        "m0_ahead_of_70k": ranks[f"M0_{seed}"] < ranks["70k"],
    }


def validate_shards(root: Path, protocol: dict[str, Any]) -> tuple[dict[int, list[Path]], dict[int, dict[str, Any]]]:
    seed_dirs = sorted(root.glob("seed_*/"))
    found_seeds = tuple(sorted(int(path.name.split("_", 1)[1]) for path in seed_dirs))
    if found_seeds != EXPECTED_SEEDS:
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, found {found_seeds}")
    shard_paths: dict[int, list[Path]] = {}
    detailed: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        expected_start = EXPECTED_SEED_STARTS[seed]
        paths = []
        for shard in range(4):
            path = root / f"seed_{seed}" / f"eval_shard_{shard:02d}"
            logs = sorted((path / "logs").glob("*.json.gz"))
            if len(logs) != 250:
                raise ValueError(f"{path}: expected 250 logs, found {len(logs)}")
            for required in ("metrics.json", "detailed_stats.json"):
                if not (path / required).is_file():
                    raise ValueError(f"missing {path / required}")
            metrics = read_json(path / "metrics.json")
            run = metrics["run"]
            if (
                int(run["seed_start"]) != expected_start + shard * 250
                or int(run["games"]) != 250
                or int(run["native_batch_games"]) != 250
                or str(run["seat_mode"]) != "random"
                or str(run["device"]) != "cuda"
                or tuple(float(value) for value in run["rank_points_values"]) != RANK_POINTS
            ):
                raise ValueError(f"protocol mismatch in {path}")
            if set(metrics["metrics"]) != {"70k", "ext_mortal", f"M0_{seed}", f"D1_{seed}"}:
                raise ValueError(f"lineup mismatch in {path}")
            paths.extend(logs)
            detailed.setdefault(seed, {})[f"eval_shard_{shard:02d}"] = read_json(path / "detailed_stats.json")
        if len({path.name for path in paths}) != 1000:
            raise ValueError(f"duplicate log names in seed {seed}")
        shard_paths[seed] = paths
    if protocol.get("native_batch_games") != 250 or protocol.get("seed_key") != 8192:
        raise ValueError("root protocol does not describe B250/8192")
    if protocol.get("git_dirty") is not False:
        raise ValueError("evaluation protocol was not frozen on a clean worktree")
    return shard_paths, detailed


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def exact_sign_test(values: list[float]) -> dict[str, Any]:
    non_ties = [value for value in values if abs(value) > 1e-12]
    positive = sum(value > 0 for value in non_ties)
    n = len(non_ties)
    if n == 0:
        p_value = 1.0
    else:
        p_value = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)
    return {"positive_count": positive, "non_tie_count": n, "one_sided_p": p_value}


def bootstrap(values_by_seed: dict[int, np.ndarray], reps: int, seed: int) -> dict[str, Any]:
    arrays = [values_by_seed[key] for key in EXPECTED_SEEDS]
    pooled = np.concatenate(arrays)
    rng = np.random.default_rng(seed)
    pooled_means: list[float] = []
    hierarchical_means: list[float] = []
    for _ in range(reps):
        pooled_means.append(float(rng.choice(pooled, size=pooled.size, replace=True).mean()))
        selected = []
        for _ in EXPECTED_SEEDS:
            values = arrays[int(rng.integers(0, len(arrays)))]
            selected.append(float(rng.choice(values, size=values.size, replace=True).mean()))
        hierarchical_means.append(float(np.mean(selected)))
    return {
        "pooled_hanchan_bootstrap_ci95": [percentile(pooled_means, 0.025), percentile(pooled_means, 0.975)],
        "hierarchical_equal_seed_bootstrap_ci95": [percentile(hierarchical_means, 0.025), percentile(hierarchical_means, 0.975)],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def comparison(rows_by_seed: dict[int, list[dict[str, Any]]], field: str, reps: int, seed: int) -> dict[str, Any]:
    values_by_seed = {seed_id: np.asarray([float(row[field]) for row in rows], dtype=np.float64) for seed_id, rows in rows_by_seed.items()}
    seed_means = {str(seed_id): float(values.mean()) for seed_id, values in values_by_seed.items()}
    all_values = np.concatenate(list(values_by_seed.values()))
    result = {
        "field": field,
        "seed_means": seed_means,
        "mean": float(all_values.mean()),
        "median": float(np.median(all_values)),
        "seed_mean_mean": float(np.mean(list(seed_means.values()))),
        "seed_mean_median": float(np.median(list(seed_means.values()))),
        "seed_sign_test": exact_sign_test(list(seed_means.values())),
    }
    result.update(bootstrap(values_by_seed, reps, seed))
    return result


def behavior_summary(detailed_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "avg_rank",
        "agari_rate",
        "houjuu_rate",
        "fuuro_rate",
        "riichi_rate",
        "avg_point_per_game",
        "agari_rate_after_riichi",
        "houjuu_rate_after_riichi",
        "agari_rate_after_fuuro",
        "houjuu_rate_after_fuuro",
    )
    label_sources = {
        "70k": EXPECTED_SEEDS,
        "ext_mortal": EXPECTED_SEEDS,
    }
    label_sources.update({f"M0_{seed}": (seed,) for seed in EXPECTED_SEEDS})
    label_sources.update({f"D1_{seed}": (seed,) for seed in EXPECTED_SEEDS})
    result: dict[str, Any] = {}
    for label, source_seeds in label_sources.items():
        weighted: dict[str, list[float]] = {field: [] for field in fields}
        raw_games = 0
        raw_rank_counts = Counter()
        for seed in source_seeds:
            stats = detailed_by_seed[seed]
            for report in stats.values():
                player = report["players"][label]
                games = int(player["raw"]["game"])
                raw_games += games
                for rank in range(1, 5):
                    raw_rank_counts[str(rank)] += int(player["raw"][f"rank_{rank}"])
                for field in fields:
                    weighted[field].append(float(player["derived"][field]) * games)
        result[label] = {
            field: sum(weighted[field]) / raw_games for field in fields
        }
        result[label]["games"] = raw_games
        result[label]["rank_counts"] = dict(sorted(raw_rank_counts.items()))
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# D1 B250 Evaluation Summary",
        "",
        "Three matched 1000-hanchan evaluations comparing M0 mixed replay against D1 project-owned K0-view replay. All primary deltas are paired within complete hanchans.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{EXPECTED_SEEDS}`; 1000 hanchans per seed; 4 x 250 B250 shards.",
        f"- Seed starts: `{EXPECTED_SEED_STARTS}`; key `8192`; random seats; CUDA required; AMP disabled.",
        f"- Rank points: `{list(RANK_POINTS)}`; evaluator commit: `{summary['protocol']['git_commit']}`.",
        "",
        "## Paired Comparisons",
        "",
        "| Comparison | Seed means | Mean | Median | Pooled 95% CI | Hierarchical 95% CI | Positive seeds | Sign-test p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["comparisons"].items():
        lines.append(
            f"| `{name}` | `{value['seed_means']}` | {value['mean']:+.3f} | {value['median']:+.3f} | "
            f"`[{value['pooled_hanchan_bootstrap_ci95'][0]:+.3f}, {value['pooled_hanchan_bootstrap_ci95'][1]:+.3f}]` | "
            f"`[{value['hierarchical_equal_seed_bootstrap_ci95'][0]:+.3f}, {value['hierarchical_equal_seed_bootstrap_ci95'][1]:+.3f}]` | "
            f"{value['seed_sign_test']['positive_count']}/{value['seed_sign_test']['non_tie_count']} | "
            f"{value['seed_sign_test']['one_sided_p']:.6g} |"
        )
    lines.extend(["", "## Behavior Readout", "", "| Model | Avg rank | Agari | Houjuu | Fuuro | Riichi | Avg pt/game | After riichi A/H | After fuuro A/H |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label, values in summary["behavior"].items():
        lines.append(
            f"| `{label}` | {values['avg_rank']:.4f} | {values['agari_rate']:.4%} | {values['houjuu_rate']:.4%} | {values['fuuro_rate']:.4%} | {values['riichi_rate']:.4%} | {values['avg_point_per_game']:+.1f} | {values['agari_rate_after_riichi']:.4%} / {values['houjuu_rate_after_riichi']:.4%} | {values['agari_rate_after_fuuro']:.4%} / {values['houjuu_rate_after_fuuro']:.4%} |"
        )
    lines.extend(["", "## Boundary", "", "- `D1-M0` is the data-lineage comparison.", "- `D1-70k` determines whether D1 produced a K1-strength continuation, not merely a better training route.", "- Bootstrap intervals describe hanchan sampling and equal-seed recipe uncertainty separately; they do not turn this evaluation into a checkpoint promotion automatically.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    root = args.eval_root.resolve()
    protocol = read_json(root / "protocol.json")
    shard_paths, detailed = validate_shards(root, protocol)
    rows_by_seed = {seed: [paired_row(seed, path) for path in paths] for seed, paths in shard_paths.items()}
    all_rows = [row for seed in EXPECTED_SEEDS for row in rows_by_seed[seed]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_hanchan_paired.csv", all_rows)
    comparisons = {
        "D1-M0": comparison(rows_by_seed, "delta_pt_d1_minus_m0", args.bootstrap_reps, args.bootstrap_seed),
        "D1-70k": comparison(rows_by_seed, "delta_pt_d1_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 1),
        "M0-70k": comparison(rows_by_seed, "delta_pt_m0_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 2),
    }
    summary = {
        "schema": "keqing.mortal.d1_b250_eval_summary.v1",
        "protocol": protocol,
        "hanchans": {str(seed): len(rows) for seed, rows in rows_by_seed.items()},
        "comparisons": comparisons,
        "rank_counts": {
            label: dict(sorted(Counter(row[key] for row in all_rows).items()))
            for label, key in (("70k", "rank_70k"), ("ext_mortal", "rank_ext_mortal"), ("M0", "rank_m0"), ("D1", "rank_d1"))
        },
        "behavior": behavior_summary(detailed),
        "source_log_sha256": [row["source_log_sha256"] for row in all_rows],
    }
    (args.output_dir / "d1_b250_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "d1_b250_eval_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "d1_b250_eval_summary.json"), "hanchans": summary["hanchans"], "d1_minus_m0": comparisons["D1-M0"]["mean"], "d1_minus_70k": comparisons["D1-70k"]["mean"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
