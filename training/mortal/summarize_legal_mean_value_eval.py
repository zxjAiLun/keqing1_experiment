#!/usr/bin/env python3
"""Summarize the B250 legal-mean-value objective evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
EXPECTED_TRAINING_SEEDS = (20260803, 20260804, 20260805)
EXPECTED_EVAL_SEED_STARTS = (1500000, 1510000, 1520000)
BEHAVIOR_FIELDS = (
    "agari_rate",
    "houjuu_rate",
    "fuuro_rate",
    "riichi_rate",
    "avg_point_per_agari",
    "avg_point_per_houjuu",
    "agari_rate_after_riichi",
    "houjuu_rate_after_riichi",
    "agari_rate_after_fuuro",
    "houjuu_rate_after_fuuro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=1000)
    parser.add_argument("--expected-batch", type=int, default=250)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _hierarchical_ci(per_seed: list[np.ndarray], rng: np.random.Generator, reps: int) -> list[float]:
    outer = rng.integers(0, len(per_seed), size=(reps, len(per_seed)))
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        seed_means = []
        for seed_index in outer[index]:
            values = per_seed[int(seed_index)]
            inner = rng.integers(0, values.size, size=values.size)
            seed_means.append(float(values[inner].mean()))
        estimates[index] = float(np.mean(seed_means))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _exact_sign_test(values: list[float], eps: float = 1e-12) -> dict[str, Any]:
    non_ties = [value for value in values if abs(value) > eps]
    positive = sum(value > 0 for value in non_ties)
    count = len(non_ties)
    p_value = (
        sum(math.comb(count, k) for k in range(positive, count + 1)) / (2**count)
        if count
        else 1.0
    )
    return {
        "positive_seed_count": positive,
        "non_tie_seed_count": count,
        "tie_seed_count": len(values) - count,
        "one_sided_p": float(p_value),
    }


def _model_key(label: str) -> str | None:
    if label == "70k":
        return "70k"
    if label == "ext_mortal":
        return "ext_mortal"
    if label.startswith("C_behavior_action_mc"):
        return "C"
    if label.startswith("V_legal_mean_mc"):
        return "V"
    return None


def _read_paired_rows(run_dir: Path, expected_games: int) -> list[dict[str, Any]]:
    path = run_dir / "platform_accounts" / "per_game_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    games: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = _model_key(str(row["model_label"]))
            if key is None:
                continue
            source = str(row["source_log"])
            values = games.setdefault(source, {})
            if key in values:
                raise ValueError(f"{path}: duplicate source/model row: {source} / {key}")
            values[key] = {
                "rank": int(row["rank"]),
                "final_score": int(row["final_score"]),
                "pt": RANK_POINTS[int(row["rank"]) - 1],
            }
    if len(games) != expected_games:
        raise ValueError(f"{run_dir}: found {len(games)} games, expected {expected_games}")
    rows = []
    for source, values in sorted(games.items()):
        if set(values) != {"70k", "ext_mortal", "C", "V"}:
            raise ValueError(f"{run_dir}: incomplete game {source}: {sorted(values)}")
        rows.append({"source_log": source, **values})
    return rows


def _comparison(rows: list[dict[str, Any]], candidate: str, baseline: str, rng: np.random.Generator, reps: int) -> dict[str, Any]:
    delta_pt = np.asarray([row[candidate]["pt"] - row[baseline]["pt"] for row in rows], dtype=np.float64)
    delta_rank = np.asarray([row[baseline]["rank"] - row[candidate]["rank"] for row in rows], dtype=np.float64)
    candidate_ranks = np.asarray([row[candidate]["rank"] for row in rows], dtype=np.int64)
    baseline_ranks = np.asarray([row[baseline]["rank"] for row in rows], dtype=np.int64)
    candidate_counts = [int(np.sum(candidate_ranks == rank)) for rank in range(1, 5)]
    baseline_counts = [int(np.sum(baseline_ranks == rank)) for rank in range(1, 5)]
    ahead = (candidate_ranks < baseline_ranks).astype(np.float64)
    ties = (candidate_ranks == baseline_ranks).astype(np.float64)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "games": len(rows),
        "mean_delta_pt_candidate_minus_baseline": float(delta_pt.mean()),
        "median_delta_pt_candidate_minus_baseline": float(np.median(delta_pt)),
        "mean_delta_rank_baseline_minus_candidate": float(delta_rank.mean()),
        "candidate_ahead_rate": float(ahead.mean()),
        "tie_rate": float(ties.mean()),
        "delta_pt_bootstrap_95ci": _bootstrap_ci(delta_pt, rng, reps),
        "delta_rank_bootstrap_95ci": _bootstrap_ci(delta_rank, rng, reps),
        "candidate_rank_counts": candidate_counts,
        "baseline_rank_counts": baseline_counts,
        "rank_rate_diff_pp_candidate_minus_baseline": [
            100.0 * (candidate_count - baseline_count) / len(rows)
            for candidate_count, baseline_count in zip(candidate_counts, baseline_counts, strict=True)
        ],
        "cluster": "complete_hanchan",
        "bootstrap_reps": reps,
    }


def _snapshot(run_dir: Path, label: str) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json")
    detailed = _read_json(run_dir / "detailed_stats.json")
    metric = metrics["metrics"][label]
    player = detailed["players"][label]
    derived = player["derived"]
    return {
        "label": label,
        "games": int(metric["games"]),
        "rank_counts": metric["rank_counts"],
        "avg_rank": float(metric["avg_rank"]),
        "avg_rank_pt": float(metric["avg_rank_pt"]),
        "behavior": {field: float(derived[field]) for field in BEHAVIOR_FIELDS if field in derived},
    }


def _mean_behavior(per_seed: list[dict[str, Any]], model: str) -> dict[str, float]:
    return {
        field: float(np.mean([item["models"][model]["behavior"].get(field, 0.0) for item in per_seed]))
        for field in BEHAVIOR_FIELDS
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    run_dirs = sorted(eval_root.glob("seed_*"))
    if len(run_dirs) != 3:
        raise ValueError(f"expected three seed directories under {eval_root}, found {len(run_dirs)}")
    protocol = _read_json(eval_root / "protocol.json")
    registered_starts = tuple(
        int(protocol.get("evaluation_seed_starts", {}).get(str(seed), -1))
        for seed in EXPECTED_TRAINING_SEEDS
    )
    if (
        protocol.get("native_batch_games") != args.expected_batch
        or protocol.get("git_dirty") is not False
        or tuple(int(value) for value in protocol.get("training_seeds", [])) != EXPECTED_TRAINING_SEEDS
        or registered_starts != EXPECTED_EVAL_SEED_STARTS
        or tuple(float(value) for value in protocol.get("rank_points", [])) != RANK_POINTS
    ):
        raise ValueError("protocol metadata does not match clean B250 contract")
    rng = np.random.default_rng(20260729)
    per_seed: list[dict[str, Any]] = []
    comparisons: dict[str, list[np.ndarray]] = {"V-C": [], "V-70k": [], "C-70k": []}
    all_rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    all_sources: list[str] = []

    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        run = metrics["run"]
        if (
            run.get("games") != args.expected_games
            or run.get("native_batch_games") != args.expected_batch
            or run.get("device") != "cuda"
            or run.get("seat_mode") != "random"
            or run.get("seed_key") != 8192
            or tuple(float(value) for value in metrics.get("rank_points_values", [])) != RANK_POINTS
        ):
            raise ValueError(f"{run_dir}: evaluation contract mismatch")
        log_count = len(list((run_dir / "logs").glob("*.json.gz")))
        if log_count != args.expected_games:
            raise ValueError(f"{run_dir}: expected {args.expected_games} logs, got {log_count}")
        rows = _read_paired_rows(run_dir, args.expected_games)
        all_rows.extend(rows)
        seed = int(run_dir.name.split("_", 1)[1])
        seeds.append(seed)
        all_sources.extend(row["source_log"] for row in rows)
        if int(run["seed_start"]) != EXPECTED_EVAL_SEED_STARTS[len(seeds) - 1]:
            raise ValueError(f"{run_dir}: unexpected evaluation seed start {run['seed_start']}")
        expected_labels = {
            "70k",
            "ext_mortal",
            f"C_behavior_action_mc_{seed}",
            f"V_legal_mean_mc_{seed}",
        }
        if set(run["models"]) != expected_labels:
            raise ValueError(f"{run_dir}: model labels do not match registered lineup")
        for label, path_text in run["models"].items():
            checkpoint = Path(path_text)
            expected_hash = protocol.get("model_sha256", {}).get(label)
            if expected_hash is None or _sha256(checkpoint) != expected_hash:
                raise ValueError(f"{run_dir}: model SHA256 mismatch for {label}")
        labels = {key: next(label for label in metrics["metrics"] if _model_key(label) == key) for key in ("C", "V")}
        model_snapshots = {key: _snapshot(run_dir, label) for key, label in (("70k", "70k"), ("ext_mortal", "ext_mortal"), *labels.items())}
        seed_comparisons = {
            "V-C": _comparison(rows, "V", "C", rng, args.bootstrap_reps),
            "V-70k": _comparison(rows, "V", "70k", rng, args.bootstrap_reps),
            "C-70k": _comparison(rows, "C", "70k", rng, args.bootstrap_reps),
        }
        for key in comparisons:
            candidate, baseline = key.split("-")
            comparisons[key].append(
                np.asarray([row[candidate]["pt"] - row[baseline]["pt"] for row in rows], dtype=np.float64)
            )
        per_seed.append(
            {
                "training_seed": seed,
                "evaluation_seed_start": int(run["seed_start"]),
                "models": model_snapshots,
                "comparisons": seed_comparisons,
                "source_checks": {
                    "log_count": log_count,
                    "metrics": str(run_dir / "metrics.json"),
                    "detailed_stats": str(run_dir / "detailed_stats.json"),
                    "platform_accounts": (run_dir / "platform_accounts").is_dir(),
                    "per_game_results_sha256": _sha256(
                        run_dir / "platform_accounts" / "per_game_results.csv"
                    ),
                },
            }
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate training seeds: {seeds}")
    if tuple(sorted(seeds)) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(f"unexpected training seeds: {seeds}")
    if len(all_sources) != len(set(all_sources)):
        raise ValueError("duplicate source logs across training seeds")

    pooled = {
        key: _comparison(all_rows, *key.split("-"), rng, args.bootstrap_reps)
        for key in comparisons
    }
    hierarchical = {
        key: {
            "delta_pt_bootstrap_95ci": _hierarchical_ci(values, rng, args.bootstrap_reps),
            "outer_cluster": "training_seed",
            "inner_cluster": "complete_hanchan",
            "seed_weighting": "equal",
            "bootstrap_reps": args.bootstrap_reps,
        }
        for key, values in comparisons.items()
    }
    seed_means = {key: [item["comparisons"][key]["mean_delta_pt_candidate_minus_baseline"] for item in per_seed] for key in comparisons}
    sign_tests = {key: _exact_sign_test(values) for key, values in seed_means.items()}
    aggregate_models = {
        model: {
            "avg_rank": float(np.mean([item["models"][model]["avg_rank"] for item in per_seed])),
            "avg_rank_pt": float(np.mean([item["models"][model]["avg_rank_pt"] for item in per_seed])),
            "behavior": _mean_behavior(per_seed, model),
        }
        for model in ("70k", "ext_mortal", "C", "V")
    }
    document = {
        "schema": "keqing.mortal.legal_mean_value_b250_summary.v1",
        "eval_root": str(eval_root),
        "protocol": protocol,
        "expected_games_per_seed": args.expected_games,
        "per_seed": per_seed,
        "pooled": pooled,
        "hierarchical": hierarchical,
        "seed_means": seed_means,
        "sign_tests": sign_tests,
        "aggregate_models_equal_seed_mean": aggregate_models,
        "interpretation_scope": "B250 research arena; no B25/B250 result mixing and no checkpoint promotion from this summary alone",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "legal_mean_value_b250_summary.json"
    md_path = args.output_dir / "legal_mean_value_b250_summary.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    v_c = pooled["V-C"]
    v_70 = pooled["V-70k"]
    c_70 = pooled["C-70k"]
    lines = [
        "# Legal-Mean Value Objective: B250 Evaluation",
        "",
        "Three matched 1000-hanchan native random-seat evaluations. `B250` is the research protocol; the legacy B25 run is excluded.",
        "",
        f"- Git commit: `{protocol['git_commit']}`; GPU: `{protocol['runtime']['gpu']}`; PyTorch: `{protocol['runtime']['torch']}`.",
        f"- Native batch: `{protocol['native_batch_games']}`; seed key: `{protocol['seed_key']}`; rank points: `{protocol['rank_points']}`.",
        "",
        "## Per Training Seed",
        "",
        "| Seed | C avg Pt | V avg Pt | V-C Pt | V-70k Pt | C-70k Pt | V-C ahead | V rank / C rank |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in per_seed:
        c = item["models"]["C"]
        v = item["models"]["V"]
        vc = item["comparisons"]["V-C"]
        v70 = item["comparisons"]["V-70k"]
        c70 = item["comparisons"]["C-70k"]
        lines.append(
            f"| {item['training_seed']} | {c['avg_rank_pt']:+.2f} | {v['avg_rank_pt']:+.2f} | "
            f"{vc['mean_delta_pt_candidate_minus_baseline']:+.2f} | {v70['mean_delta_pt_candidate_minus_baseline']:+.2f} | "
            f"{c70['mean_delta_pt_candidate_minus_baseline']:+.2f} | {_fmt_pct(vc['candidate_ahead_rate'])} | "
            f"{v['avg_rank']:.3f} / {c['avg_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Differences",
            "",
            "All paired statistics resample complete hanchans. Hierarchical intervals resample training seeds outside and hanchans inside with equal seed weighting.",
            "",
            f"- `V-C`: mean `{v_c['mean_delta_pt_candidate_minus_baseline']:+.3f}` Pt; hanchan CI `[{v_c['delta_pt_bootstrap_95ci'][0]:+.3f}, {v_c['delta_pt_bootstrap_95ci'][1]:+.3f}]`; hierarchical CI `[{hierarchical['V-C']['delta_pt_bootstrap_95ci'][0]:+.3f}, {hierarchical['V-C']['delta_pt_bootstrap_95ci'][1]:+.3f}]`; seed means `{[round(v, 3) for v in seed_means['V-C']]}`; sign-test `p={sign_tests['V-C']['one_sided_p']:.3f}`.",
            f"- `V-70k`: mean `{v_70['mean_delta_pt_candidate_minus_baseline']:+.3f}` Pt; hanchan CI `[{v_70['delta_pt_bootstrap_95ci'][0]:+.3f}, {v_70['delta_pt_bootstrap_95ci'][1]:+.3f}]`; hierarchical CI `[{hierarchical['V-70k']['delta_pt_bootstrap_95ci'][0]:+.3f}, {hierarchical['V-70k']['delta_pt_bootstrap_95ci'][1]:+.3f}]`.",
            f"- `C-70k`: mean `{c_70['mean_delta_pt_candidate_minus_baseline']:+.3f}` Pt; hanchan CI `[{c_70['delta_pt_bootstrap_95ci'][0]:+.3f}, {c_70['delta_pt_bootstrap_95ci'][1]:+.3f}]`; hierarchical CI `[{hierarchical['C-70k']['delta_pt_bootstrap_95ci'][0]:+.3f}, {hierarchical['C-70k']['delta_pt_bootstrap_95ci'][1]:+.3f}]`.",
            "",
            "## Equal-Seed Behavior Mean",
            "",
            "| Model | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | Agari after riichi | Houjuu after riichi | Agari after fuuro | Houjuu after fuuro |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in ("70k", "ext_mortal", "C", "V"):
        snapshot = aggregate_models[model]
        behavior = snapshot["behavior"]
        lines.append(
            f"| {model} | {snapshot['avg_rank']:.3f} | {snapshot['avg_rank_pt']:+.2f} | "
            f"{_fmt_pct(behavior['agari_rate'])} | {_fmt_pct(behavior['houjuu_rate'])} | "
            f"{_fmt_pct(behavior['fuuro_rate'])} | {_fmt_pct(behavior['riichi_rate'])} | "
            f"{_fmt_pct(behavior['agari_rate_after_riichi'])} | {_fmt_pct(behavior['houjuu_rate_after_riichi'])} | "
            f"{_fmt_pct(behavior['agari_rate_after_fuuro'])} | {_fmt_pct(behavior['houjuu_rate_after_fuuro'])} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is an objective A/B in the fixed B250 research arena. It does not promote a checkpoint or establish that legal-mean MC is a generally better objective. The primary comparison is `V-C`; `V-70k` and `C-70k` distinguish objective effect from continuation strength.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "seed_means": seed_means}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
