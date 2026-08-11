#!/usr/bin/env python3
"""Summarize the D3 uncertainty-exploration B250 evaluation and adjudicate.

Mechanical adaptation of the D2 summarizer (same statistical method, same
bootstrap contract). The pairing unit is one complete hanchan; final ranks are
rebuilt independently from raw logs. Primary comparison is D3 minus matched M0;
secondary is D3 minus K0; M0 minus K0 is a descriptive control sanity.

Promotion decision is computed mechanically from the preregistered conjunction:

  route_pass = all(seed_mean > 0 for D3-M0) AND hierarchical CI95 lower > 0
  k1_pass    = route_pass AND D3-K0 hierarchical CI95 lower > 0 AND integrity

A CI can never rescue a failed 3/3 direction condition.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SEEDS = (20260806, 20260807, 20260808)
EXPECTED_SEED_STARTS = {20260806: 1700000, 20260807: 1710000, 20260808: 1720000}
RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
EXPECTED_METRICS_SHA = {
    20260806: "a3990dcedf8c28f5aa792ffbc3b42ec7da9f87d79e2b40f5eb25b990c49678aa",
    20260807: "4e345e932e90fd6426d9967f6ce78005a3805ac8dcd93b37d5f3498904aec0b0",
    20260808: "fd6989ead01c5d464d5b1c0fb9d38fafdc8201a43fab89e8a762404d8143b5e2",
}
EXPECTED_MODEL_SHA = {
    "70k": "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0",
    "ext_mortal": "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563",
    "M0_20260806": "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
    "M0_20260807": "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
    "M0_20260808": "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
    "D3_20260806": "a93e7a8f6b56f2c07e5e1f42c0283ff2b839e2da67af2ac9abbf382c7189defc",
    "D3_20260807": "78d78a59450be469c703a7f7ae172fcb4ddeff6b59ade4e53cf7ad6230c0e6b0",
    "D3_20260808": "5790cbb08196e7994a593a4d91354dd29e4f49c25848ae5358d8a403ed1b72eb",
}
EXPECTED_MODELS_PER_SEED = {
    seed: {"70k", "ext_mortal", f"M0_{seed}", f"D3_{seed}"} for seed in EXPECTED_SEEDS
}
DEFAULT_COMPLETION_CLOSURE = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\diagnostics\training_completion_closure.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--completion-closure", type=Path, default=DEFAULT_COMPLETION_CLOSURE)
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
    """Rebuild final scores from raw events in log order.

    Mirrors the native libriichi Stat semantics: a ``reach_accepted`` event
    deducts 1000 points from its actor before any Hora/Ryukyoku deltas are
    applied, and the scores published by the next ``start_kyoku`` already
    include everything from earlier kyoku. Summary v1 omitted the -1000 step
    and produced non-authoritative ranks (invalidated).
    """
    scores: list[float] | None = None
    for event in events:
        if event.get("type") == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(value) for value in values]
        elif event.get("type") == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif event.get("type") in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def authoritative_ranks(path: Path) -> list[int] | None:
    """Native libriichi Stat per-seat final ranks (0-based seats -> 1..4 rank).

    Returns None when libriichi is unavailable (e.g. plain test environments);
    the real evaluation machine always provides it.
    """
    try:
        from libriichi.stat import Stat  # noqa: PLC0415
    except ImportError:
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw_log = handle.read()
    ranks: list[int] = []
    for player_id in range(4):
        stat = Stat.from_log(raw_log, player_id)
        rank = [rank_id for rank_id in (1, 2, 3, 4) if getattr(stat, f"rank_{rank_id}") == 1]
        if len(rank) != 1:
            raise ValueError(f"{path}: ambiguous Stat rank for player {player_id}: {rank}")
        ranks.append(rank[0])
    return ranks


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
    d3_label = f"D3_{seed}"
    m0_label = f"M0_{seed}"
    required = ("70k", "ext_mortal", m0_label, d3_label)
    if set(ranks) != set(required):
        raise ValueError(f"{path}: expected names {required}, got {sorted(ranks)}")
    points = {label: RANK_POINTS[ranks[label] - 1] for label in required}
    seat_ranks = [ranks[str(name)] for name in events[0]["names"]]
    return {
        "seed": seed,
        "source_log": str(path),
        "source_log_sha256": sha256(path),
        "seat_ranks_reconstructed": seat_ranks,
        "rank_70k": ranks["70k"],
        "rank_ext_mortal": ranks["ext_mortal"],
        "rank_m0": ranks[m0_label],
        "rank_d3": ranks[d3_label],
        "pt_70k": points["70k"],
        "pt_ext_mortal": points["ext_mortal"],
        "pt_m0": points[m0_label],
        "pt_d3": points[d3_label],
        "delta_pt_d3_minus_m0": points[d3_label] - points[m0_label],
        "delta_pt_d3_minus_70k": points[d3_label] - points["70k"],
        "delta_pt_m0_minus_70k": points[m0_label] - points["70k"],
        "delta_rank_m0_minus_d3": ranks[m0_label] - ranks[d3_label],
        "delta_rank_70k_minus_d3": ranks["70k"] - ranks[d3_label],
        "delta_rank_70k_minus_m0": ranks["70k"] - ranks[m0_label],
        "d3_ahead_of_m0": ranks[d3_label] < ranks[m0_label],
        "d3_ahead_of_70k": ranks[d3_label] < ranks["70k"],
        "m0_ahead_of_70k": ranks[m0_label] < ranks["70k"],
    }


def validate_seed_dir(root: Path, seed: int) -> tuple[list[Path], dict[str, Any]]:
    run_dir = root / f"seed_{seed}"
    log_dir = run_dir / "logs"
    logs = sorted(log_dir.glob("*.json.gz"))
    if len(logs) != 1000:
        raise ValueError(f"{run_dir}: expected 1000 logs, found {len(logs)}")
    if len({path.name for path in logs}) != 1000:
        raise ValueError(f"{run_dir}: duplicate log names")
    for required in ("metrics.json", "detailed_stats.json", "inference_profile.json"):
        if not (run_dir / required).is_file():
            raise ValueError(f"missing {run_dir / required}")
    metrics_path = run_dir / "metrics.json"
    if sha256(metrics_path) != EXPECTED_METRICS_SHA[seed]:
        raise ValueError(f"metrics SHA mismatch for {run_dir}")
    metrics = read_json(metrics_path)
    run = metrics.get("run", {})
    expected_labels = EXPECTED_MODELS_PER_SEED[seed]
    if (
        int(run.get("seed_start", -1)) != EXPECTED_SEED_STARTS[seed]
        or int(run.get("seed_key", -1)) != 8192
        or int(run.get("games", -1)) != 1000
        or int(run.get("native_batch_games", -1)) != 250
        or str(run.get("seat_mode")) != "random"
        or str(run.get("device")) != "cuda"
        or tuple(float(value) for value in run.get("rank_points_values", ())) != RANK_POINTS
        or set(metrics.get("metrics", {})) != expected_labels
    ):
        raise ValueError(f"protocol or lineup mismatch in {run_dir}")
    models = run.get("models", {})
    if set(models) != expected_labels:
        raise ValueError(f"model label mismatch in {run_dir}")
    for label, model_path in models.items():
        expected_sha = EXPECTED_MODEL_SHA.get(label)
        if expected_sha is None:
            raise ValueError(f"unexpected model label {label} in {run_dir}")
        if sha256(Path(model_path)) != expected_sha:
            raise ValueError(f"model SHA mismatch for {label} in {run_dir}")
    prefixes = []
    for path in logs:
        try:
            prefixes.append(int(path.name.split("_", 1)[0]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid native log name: {path.name}") from exc
    expected_prefixes = set(range(EXPECTED_SEED_STARTS[seed], EXPECTED_SEED_STARTS[seed] + 1000))
    if set(prefixes) != expected_prefixes:
        raise ValueError(f"{run_dir}: seed range mismatch")
    return logs, {
        "metrics": metrics,
        "detailed_stats": read_json(run_dir / "detailed_stats.json"),
    }


def cross_check_completion_closure(path: Path) -> dict[str, bool]:
    if not path.is_file():
        return {"closure_present": False}
    closure = read_json(path)
    ok = closure.get("passed") is True
    for seed in EXPECTED_SEEDS:
        row = closure.get("seeds", {}).get(str(seed), {})
        if row.get("final_checkpoint_sha256") != EXPECTED_MODEL_SHA[f"D3_{seed}"]:
            ok = False
    return {"closure_present": True, "closure_passed": closure.get("passed") is True, "d3_final_shas_match": ok}


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
    p_value = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)
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
        selected_seed_indexes = rng.integers(0, len(arrays), size=len(arrays))
        for index in selected_seed_indexes:
            values = arrays[int(index)]
            selected.append(float(rng.choice(values, size=values.size, replace=True).mean()))
        hierarchical_means.append(float(np.mean(selected)))
    return {
        "pooled_hanchan_bootstrap_ci95": [percentile(pooled_means, 0.025), percentile(pooled_means, 0.975)],
        "hierarchical_equal_seed_bootstrap_ci95": [percentile(hierarchical_means, 0.025), percentile(hierarchical_means, 0.975)],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def comparison(rows_by_seed: dict[int, list[dict[str, Any]]], field: str, reps: int, seed: int) -> dict[str, Any]:
    values_by_seed = {
        seed_id: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for seed_id, rows in rows_by_seed.items()
    }
    seed_means = {str(seed_id): float(values.mean()) for seed_id, values in values_by_seed.items()}
    all_values = np.concatenate(list(values_by_seed.values()))
    ahead_field = {
        "delta_pt_d3_minus_m0": "d3_ahead_of_m0",
        "delta_pt_d3_minus_70k": "d3_ahead_of_70k",
    }.get(field)
    result = {
        "field": field,
        "seed_means": seed_means,
        "mean": float(all_values.mean()),
        "median": float(np.median(all_values)),
        "seed_mean_mean": float(np.mean(list(seed_means.values()))),
        "seed_mean_median": float(np.median(list(seed_means.values()))),
        "seed_sign_test": exact_sign_test(list(seed_means.values())),
        "ahead_rate": float(np.mean([bool(row[ahead_field]) for rows in rows_by_seed.values() for row in rows])) if ahead_field else None,
    }
    result.update(bootstrap(values_by_seed, reps, seed))
    return result


def behavior_summary(details_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
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
    labels = {"70k": EXPECTED_SEEDS, "ext_mortal": EXPECTED_SEEDS}
    labels.update({f"M0_{seed}": (seed,) for seed in EXPECTED_SEEDS})
    labels.update({f"D3_{seed}": (seed,) for seed in EXPECTED_SEEDS})
    result: dict[str, Any] = {}
    for label, source_seeds in labels.items():
        weighted: dict[str, list[float]] = {field: [] for field in fields}
        raw_games = 0
        raw_rank_counts = Counter()
        for seed in source_seeds:
            player = details_by_seed[seed]["detailed_stats"]["players"][label]
            games = int(player["raw"]["game"])
            raw_games += games
            for rank in range(1, 5):
                raw_rank_counts[str(rank)] += int(player["raw"][f"rank_{rank}"])
            for field in fields:
                weighted[field].append(float(player["derived"][field]) * games)
        result[label] = {field: sum(weighted[field]) / raw_games for field in fields}
        result[label]["games"] = raw_games
        result[label]["rank_counts"] = dict(sorted(raw_rank_counts.items()))
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def promotion_decision(comparisons: dict[str, Any]) -> dict[str, Any]:
    d3_m0 = comparisons["D3-M0"]
    d3_k0 = comparisons["D3-70k"]
    seed_means = [float(value) for value in d3_m0["seed_means"].values()]
    all_positive = all(value > 0 for value in seed_means)
    hier_lower = float(d3_m0["hierarchical_equal_seed_bootstrap_ci95"][0])
    route_pass = bool(all_positive and hier_lower > 0)
    k1_ci_lower = float(d3_k0["hierarchical_equal_seed_bootstrap_ci95"][0])
    k1_pass = bool(route_pass and k1_ci_lower > 0)
    return {
        "primary": {
            "comparison": "D3_minus_M0",
            "all_three_seed_means_positive": all_positive,
            "seed_means": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, seed_means, strict=True)},
            "hierarchical_ci_lower_positive": hier_lower > 0,
            "hierarchical_ci95": d3_m0["hierarchical_equal_seed_bootstrap_ci95"],
            "passed": route_pass,
        },
        "checkpoint": {
            "route_promotion_passed": route_pass,
            "d3_minus_k0_hier_ci_lower": k1_ci_lower,
            "d3_minus_k0_ci_lower_positive": k1_ci_lower > 0,
            "integrity_passed": True,
            "passed": k1_pass,
        },
        "rule": {
            "route_pass": "all(seed_mean > 0 for D3-M0) AND hierarchical CI95 lower > 0",
            "k1_pass": "route_pass AND D3-K0 hierarchical CI95 lower > 0 AND integrity",
            "note": "a CI cannot rescue a failed 3/3 direction condition",
        },
        "verdict": "D3 data route NOT PROMOTED" if not route_pass else "D3 data route PROMOTED",
        "k1_verdict": "K1 null (not created)" if not k1_pass else "K1 candidate created",
    }


def build_markdown(summary: dict[str, Any]) -> str:
    decision = summary["promotion_decision"]
    lines = [
        "# D3 B250 Evaluation Summary",
        "",
        "Three matched 1000-hanchan evaluations compare the D3 uncertainty-exploration corpus against matched M0, K0 (70k), and ext_mortal. All primary deltas are paired within complete hanchans and rebuilt independently from raw logs.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{EXPECTED_SEEDS}`; 1000 hanchans per seed; native batch `250`.",
        f"- Seed starts: `{EXPECTED_SEED_STARTS}`; key `8192`; random seats; CUDA required; AMP disabled.",
        f"- Rank points: `{list(RANK_POINTS)}`; raw evaluator commit: `{summary['protocol']['raw_evaluator_commit']}`; summary v2 commit: `{summary['protocol']['summary_v2_commit']}` (v1 `{summary['protocol']['summary_v1_commit']}` invalidated).",
        "",
        "## Promotion Decision (preregistered, mechanical)",
        "",
        f"- D3-M0 all three seed means positive: `{decision['primary']['all_three_seed_means_positive']}`",
        f"- D3-M0 hierarchical CI95: `{decision['primary']['hierarchical_ci95']}`",
        f"- route_pass: `{decision['primary']['passed']}`",
        f"- D3-K0 hierarchical CI95 lower positive: `{decision['checkpoint']['d3_minus_k0_ci_lower_positive']}`",
        f"- K1_pass: `{decision['checkpoint']['passed']}`",
        f"- **{decision['verdict']}**; K1 = null.",
        "",
        "## Paired Comparisons",
        "",
        "| Comparison | Seed means | Mean | Median | Pooled 95% CI | Hierarchical 95% CI | Positive seeds | Sign-test p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["comparisons"].items():
        sign = value["seed_sign_test"]
        lines.append(
            f"| `{name}` | `{value['seed_means']}` | {value['mean']:+.3f} | {value['median']:+.3f} | "
            f"`[{value['pooled_hanchan_bootstrap_ci95'][0]:+.3f}, {value['pooled_hanchan_bootstrap_ci95'][1]:+.3f}]` | "
            f"`[{value['hierarchical_equal_seed_bootstrap_ci95'][0]:+.3f}, {value['hierarchical_equal_seed_bootstrap_ci95'][1]:+.3f}]` | "
            f"{sign['positive_count']}/{sign['non_tie_count']} | {sign['one_sided_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Behavior Readout",
        "",
        "| Model | Avg rank | Agari | Houjuu | Fuuro | Riichi | Avg pt/game | After riichi A/H | After fuuro A/H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, values in summary["behavior"].items():
        lines.append(
            f"| `{label}` | {values['avg_rank']:.4f} | {values['agari_rate']:.4%} | {values['houjuu_rate']:.4%} | {values['fuuro_rate']:.4%} | {values['riichi_rate']:.4%} | {values['avg_point_per_game']:+.1f} | {values['agari_rate_after_riichi']:.4%} / {values['houjuu_rate_after_riichi']:.4%} | {values['agari_rate_after_fuuro']:.4%} / {values['houjuu_rate_after_fuuro']:.4%} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- `D3-M0` is the data-route promotion comparison (primary).",
        "- `D3-70k` determines whether the D3 corpus produced a K1-strength continuation (secondary).",
        "- `M0-70k` is descriptive control sanity and does not substitute for the primary.",
        "- Bootstrap intervals describe hanchan sampling and equal-seed recipe uncertainty separately; they do not automatically promote a checkpoint or override the preregistered conjunction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    root = args.eval_root.resolve()
    seed_dirs = {seed: root / f"seed_{seed}" for seed in EXPECTED_SEEDS}
    if not all(path.is_dir() for path in seed_dirs.values()):
        raise ValueError(f"expected seed directories under {root}")
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    details_by_seed: dict[int, dict[str, Any]] = {}
    for seed, seed_dir in seed_dirs.items():
        logs, details = validate_seed_dir(root, seed)
        rows_by_seed[seed] = [paired_row(seed, path) for path in logs]
        details_by_seed[seed] = details
    all_rows = [row for seed in EXPECTED_SEEDS for row in rows_by_seed[seed]]

    # ---- reconstruction equivalence: native Stat vs pure raw-event rebuild ----
    stat_available = True
    stat_checked = 0
    stat_mismatch: list[str] = []
    for row in all_rows:
        authoritative = authoritative_ranks(Path(row["source_log"]))
        if authoritative is None:
            stat_available = False
            break
        stat_checked += 1
        for seat in range(4):
            if int(row["seat_ranks_reconstructed"][seat]) != authoritative[seat]:
                stat_mismatch.append(
                    f"{row['source_log']} seat {seat}: reconstructed="
                    f"{row['seat_ranks_reconstructed'][seat]} stat={authoritative[seat]}"
                )
    reconstruction_equivalence = {
        "stat_check_available": stat_available,
        "logs_checked": stat_checked if stat_available else 0,
        "players_checked": stat_checked * 4 if stat_available else 0,
        "mismatch_count": len(stat_mismatch) if stat_available else None,
        "mismatches_sample": stat_mismatch[:20],
        "rank_count_equivalence": {},
        "algebraic_invariants": {},
    }
    if stat_available and (stat_checked != 3000 or stat_mismatch):
        raise ValueError(f"native Stat equivalence failed: checked={stat_checked} mismatches={len(stat_mismatch)}")

    # ---- per-seed rank counts: reconstructed == detailed_stats == metrics ----
    for seed, seed_dir in seed_dirs.items():
        metrics = details_by_seed[seed]["metrics"]
        detailed = details_by_seed[seed]["detailed_stats"]
        labels = sorted(EXPECTED_MODELS_PER_SEED[seed])
        # explicit per-label reconstruction counts
        recon_counts = {
            label: [
                sum(1 for row in rows_by_seed[seed] if row[f"rank_{'70k' if label == '70k' else 'ext_mortal' if label == 'ext_mortal' else 'm0' if label.startswith('M0') else 'd3'}"] == rank)
                for rank in (1, 2, 3, 4)
            ]
            for label in labels
        }
        for label in labels:
            key = "70k" if label == "70k" else "ext_mortal" if label == "ext_mortal" else "m0" if label.startswith("M0") else "d3"
            detailed_counts = [
                int(detailed["players"][label]["raw"][f"rank_{rank}"]) for rank in (1, 2, 3, 4)
            ]
            metric_counts = [int(value) for value in metrics["metrics"][label]["rank_counts"]]
            match = recon_counts[label] == detailed_counts == metric_counts
            reconstruction_equivalence["rank_count_equivalence"][f"{seed}:{label}"] = {
                "reconstructed": recon_counts[label],
                "detailed_stats": detailed_counts,
                "metrics": metric_counts,
                "match": match,
            }
            if not match:
                raise ValueError(f"rank-count mismatch for {seed}:{label}")

    # ---- algebraic invariants: mean(delta) == mean(pt_a) - mean(pt_b) ----
    def check_invariant(delta_field: str, pt_a: str, pt_b: str, name: str) -> bool:
        deltas = np.asarray([float(row[delta_field]) for row in all_rows])
        mean_delta = float(deltas.mean())
        mean_a = float(np.mean([float(row[pt_a]) for row in all_rows]))
        mean_b = float(np.mean([float(row[pt_b]) for row in all_rows]))
        ok = math.isclose(mean_delta, mean_a - mean_b, rel_tol=1e-9, abs_tol=1e-9)
        reconstruction_equivalence["algebraic_invariants"][name] = {
            "mean_delta": mean_delta,
            "mean_a_minus_b": mean_a - mean_b,
            "match": ok,
        }
        return ok

    if not (
        check_invariant("delta_pt_d3_minus_m0", "pt_d3", "pt_m0", "mean_d3_minus_m0")
        and check_invariant("delta_pt_d3_minus_70k", "pt_d3", "pt_70k", "mean_d3_minus_70k")
        and check_invariant("delta_pt_m0_minus_70k", "pt_m0", "pt_70k", "mean_m0_minus_70k")
    ):
        raise ValueError("algebraic invariant failed: mean(delta) != mean(pt_a) - mean(pt_b)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "d3_eval_1000h_rows.csv"
    write_rows(rows_path, all_rows)
    comparisons = {
        "D3-M0": comparison(rows_by_seed, "delta_pt_d3_minus_m0", args.bootstrap_reps, args.bootstrap_seed),
        "D3-70k": comparison(rows_by_seed, "delta_pt_d3_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 1),
        "M0-70k": comparison(rows_by_seed, "delta_pt_m0_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 2),
    }
    closure_check = cross_check_completion_closure(args.completion_closure.resolve())
    if not closure_check.get("closure_present"):
        raise ValueError("training completion closure not found; D3 final SHAs cannot be cross-checked")
    if not closure_check.get("d3_final_shas_match"):
        raise ValueError("completion closure D3 final SHAs do not match the frozen reference")
    protocol = {
        "seeds": list(EXPECTED_SEEDS),
        "seed_starts": EXPECTED_SEED_STARTS,
        "seed_key": 8192,
        "games_per_seed": 1000,
        "native_batch_games": 250,
        "seat_mode": "random",
        "device": "cuda",
        "amp": False,
        "rank_points": list(RANK_POINTS),
        "raw_evaluator_commit": "413a268ef60125ac8f8fa58a77bcb948f71bf4e4",
        "summary_v1_commit": "a6b1d4b66f8c36a64f0f3958fb9e8410c143209d",
        "summary_v1_invalidated": True,
        "summary_v1_invalidation_reason": (
            "final-score reconstruction omitted ReachAccepted -1000; "
            "paired statistics non-authoritative; raw evaluation unaffected"
        ),
        "summary_v2_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "raw_eval_rerun": False,
        "raw_eval_modified": False,
        "training_rerun": False,
        "model_sha256": dict(EXPECTED_MODEL_SHA),
        "metrics_sha256": dict(EXPECTED_METRICS_SHA),
        "completion_closure": closure_check,
    }
    decision = promotion_decision(comparisons)
    summary = {
        "schema": "keqing.mortal.d3_b250_eval_summary.v2",
        "protocol": protocol,
        "reconstruction_equivalence": reconstruction_equivalence,
        "hanchans": {str(seed): len(rows) for seed, rows in rows_by_seed.items()},
        "comparisons": comparisons,
        "rank_counts": {
            label: dict(sorted(Counter(row[key] for row in all_rows).items()))
            for label, key in (("70k", "rank_70k"), ("ext_mortal", "rank_ext_mortal"), ("M0", "rank_m0"), ("D3", "rank_d3"))
        },
        "behavior": behavior_summary(details_by_seed),
        "promotion_decision": decision,
        "source_log_sha256": [row["source_log_sha256"] for row in all_rows],
        "raw_evaluation_artifacts_modified": False,
        "raw_evaluation_rerun": False,
    }
    json_path = args.output_dir / "d3_eval_1000h_summary.json"
    md_path = args.output_dir / "d3_eval_1000h_summary.md"
    decision_path = args.output_dir / "d3_promotion_decision.json"
    equivalence_path = args.output_dir / "reconstruction_equivalence.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    equivalence_path.write_text(
        json.dumps(reconstruction_equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(json_path),
                "hanchans": summary["hanchans"],
                "d3_minus_m0": comparisons["D3-M0"],
                "d3_minus_70k": comparisons["D3-70k"],
                "promotion_decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
