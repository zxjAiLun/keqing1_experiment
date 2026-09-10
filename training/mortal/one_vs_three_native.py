#!/usr/bin/env python3
"""One-vs-three native arena evaluation: one solo challenger against a trio.

Design notes (why this is not `four_player_native.py` with three duplicate
seats):

- ``libriichi.arena.OneVsThree.py_vs_py`` builds exactly **two** batch inference
  agents (challenger + champion).  The three identical opponents share one
  engine, so their requests are batched together instead of issuing three
  independent inference streams per decision.
- Each seed is played as four seat rotations (log split ``a``/``b``/``c``/``d``
  with the challenger in seat 0/1/2/3), so seat effects cancel without a
  separate rotation schedule.
- The seed is the statistical cluster: its four rotations share one deal, so
  confidence intervals resample seeds, never the three opponents as if they
  were independent samples.

Inference settings (fp32, ``quick_eval``, rule-based agari guard, version, and
model construction) come from ``four_player_native._load_engine`` so the
1v3 evaluation runs under exactly the same inference contract as the four-seat
arena.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.eval_metrics import (  # noqa: E402
    add_rank_point_args,
    resolve_rank_points,
    summarize_rank_counts,
)
from training.mortal.four_player_native import _load_engine  # noqa: E402

# Log split letter -> challenger seat.  libriichi writes one file per rotation
# in the order the agent index table declares them.
_SPLIT_SEATS = (("a", 0), ("b", 1), ("c", 2), ("d", 3))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"model spec must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise ValueError(f"model spec must be LABEL=PATH, got: {value}")
    return label, Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-vs-three native arena evaluation")
    parser.add_argument("--challenger", required=True, help="LABEL=CHECKPOINT for the solo seat")
    parser.add_argument("--champion", required=True, help="LABEL=CHECKPOINT for the three-seat group")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=64, help="seeds; each seed is 4 hanchans (four seat rotations)")
    parser.add_argument("--seed-start", type=int, default=700000)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--batch-seeds", type=int, default=8, help="seeds per native batch (8 seeds = 32 hanchans)")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse hanchan logs already on disk")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    add_rank_point_args(parser)
    return parser.parse_args()


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


def _cluster_bootstrap(seed_values: list[float], reps: int, seed: int) -> dict[str, Any]:
    """Resample the seed clusters with replacement (equal weight per seed)."""
    values = np.asarray(seed_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(reps):
        picked = rng.integers(0, values.size, size=values.size)
        means.append(float(values[picked].mean()))
    return {
        "seed_cluster_bootstrap_ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "clusters": int(values.size),
    }


def _exact_sign_test(values: list[float]) -> dict[str, Any]:
    non_ties = [value for value in values if abs(value) > 1e-12]
    positive = sum(value > 0 for value in non_ties)
    n = len(non_ties)
    p_value = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)
    return {"seed_means_positive": positive, "seed_means_non_tie": n, "one_sided_sign_p": p_value}


def _challenger_rank(log_text: str, challenger_seat: int) -> int:
    from libriichi.stat import Stat  # noqa: PLC0415

    stat = Stat.from_log(log_text, challenger_seat)
    counts = [_stat_rank_count(stat, rank) for rank in range(1, 5)]
    if sum(counts) != 1:
        raise RuntimeError(f"native Stat returned ranks {counts} for seat {challenger_seat}")
    return counts.index(1) + 1


def _stat_rank_count(stat: Any, rank: int) -> int:
    """`rank_N` is an attribute getter; tolerate a method on other builds."""
    value = getattr(stat, f"rank_{rank}")
    return int(value() if callable(value) else value)


def _seed_ranks(log_dir: Path, seed: int, seed_key: int) -> list[int]:
    ranks: list[int] = []
    for split, seat in _SPLIT_SEATS:
        path = log_dir / f"{seed}_{seed_key}_{split}.json.gz"
        if not path.exists():
            raise FileNotFoundError(f"missing hanchan log: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            ranks.append(_challenger_rank(handle.read(), seat))
    return ranks


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    if int(args.seeds) <= 0:
        raise ValueError("--seeds must be positive")
    if int(args.batch_seeds) <= 0:
        raise ValueError("--batch-seeds must be positive")

    challenger_label, challenger_path = _parse_model_spec(args.challenger)
    champion_label, champion_path = _parse_model_spec(args.champion)
    if challenger_label == champion_label:
        raise ValueError("challenger and champion labels must differ")

    rank_points_profile, rank_points = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resume must be bound to the exact evaluation, not just to log filenames:
    # the seed band and the two checkpoints (by content hash) are part of the
    # identity, so a different run can never silently reuse another run's logs.
    identity = {
        "challenger": {"label": challenger_label, "path": str(challenger_path), "sha256": _sha256_file(challenger_path)},
        "champion": {"label": champion_label, "path": str(champion_path), "sha256": _sha256_file(champion_path)},
        "seeds": {"seed_start": int(args.seed_start), "seed_count": int(args.seeds), "seed_key": int(args.seed_key)},
        "batch_seeds": int(args.batch_seeds),
        "rank_points_profile": rank_points_profile,
        "rank_points_values": [float(value) for value in rank_points],
        "enable_amp": bool(args.enable_amp),
    }
    identity_path = args.output_dir / "run_identity.json"
    if args.resume and identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded != identity:
            raise RuntimeError(
                "refusing --resume: run identity differs from the recorded one "
                f"({identity_path}); use a fresh output directory"
            )
        print(f"[one_vs_three] resume identity verified against {identity_path}", flush=True)
    elif args.resume:
        raise RuntimeError(
            f"refusing --resume: {identity_path} is missing, so existing logs cannot be "
            "bound to a known evaluation"
        )
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    from libriichi.arena import OneVsThree  # noqa: PLC0415

    load_times: dict[str, float] = {}
    engines: dict[str, Any] = {}
    for label, path in ((challenger_label, challenger_path), (champion_label, champion_path)):
        started = time.perf_counter()
        engines[label] = _load_engine(
            label=label,
            state_file=path,
            mortal_root=args.mortal_root,
            device=str(args.device),
            enable_amp=bool(args.enable_amp),
            enable_profile=bool(args.profile),
        )
        load_times[label] = time.perf_counter() - started
        print(f"loaded {label:<10} in {load_times[label]:.1f}s", flush=True)

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    total_seeds = int(args.seeds)
    native_batches: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    play_started = time.perf_counter()

    for offset in range(0, total_seeds, int(args.batch_seeds)):
        count = min(int(args.batch_seeds), total_seeds - offset)
        batch_seed_start = int(args.seed_start) + offset
        seeds = [batch_seed_start + i for i in range(count)]
        if args.resume and all(
            (log_dir / f"{seed}_{int(args.seed_key)}_{split}.json.gz").exists()
            for seed in seeds
            for split, _seat in _SPLIT_SEATS
        ):
            print(f"[one_vs_three] reusing existing logs for seeds {batch_seed_start}..{seeds[-1]}", flush=True)
            native_batches.append({"seed_start": batch_seed_start, "seed_count": count, "rankings": None, "reused": True})
            completed_seeds.extend(seeds)
            continue

        print(
            f"[one_vs_three] seeds {batch_seed_start}..{seeds[-1]} "
            f"({count * 4} hanchans) challenger={challenger_label} champion={champion_label}",
            file=sys.stderr,
            flush=True,
        )
        try:
            rankings = env.py_vs_py(
                challenger=engines[challenger_label],
                champion=engines[champion_label],
                seed_start=(batch_seed_start, int(args.seed_key)),
                seed_count=count,
            )
        except TypeError:
            # Older extension builds expose positional-only parameters.
            rankings = env.py_vs_py(
                engines[challenger_label],
                engines[champion_label],
                (batch_seed_start, int(args.seed_key)),
                count,
            )
        rankings = [int(value) for value in rankings]
        if sum(rankings) != count * 4:
            raise RuntimeError(f"native arena returned {rankings} for {count} seeds (expected {count * 4} games)")
        native_batches.append(
            {"seed_start": batch_seed_start, "seed_count": count, "rankings": rankings, "reused": False}
        )
        completed_seeds.extend(seeds)

    play_seconds = time.perf_counter() - play_started

    # ---- per-game ranks from the native Stat (authoritative) ----------------
    ranks_by_seed: dict[int, list[int]] = {}
    for seed in completed_seeds:
        ranks_by_seed[seed] = _seed_ranks(log_dir, seed, int(args.seed_key))

    flat_ranks = [rank for seed in completed_seeds for rank in ranks_by_seed[seed]]
    rank_counts = [0, 0, 0, 0]
    for rank in flat_ranks:
        rank_counts[rank - 1] += 1

    native_counts = [0, 0, 0, 0]
    verified_seeds: list[int] = []
    reused_seed_count = 0
    for batch in native_batches:
        if batch["rankings"] is None:
            reused_seed_count += batch["seed_count"]
            continue
        verified_seeds.extend(batch["seed_start"] + i for i in range(batch["seed_count"]))
        for index, value in enumerate(batch["rankings"]):
            native_counts[index] += value

    # Compare only over the batches the native arena actually returned in this
    # invocation; reused logs have no native aggregate to check against.
    verified_counts = [0, 0, 0, 0]
    for seed in verified_seeds:
        for rank in ranks_by_seed[seed]:
            verified_counts[rank - 1] += 1
    if verified_seeds:
        ranks_match_native: bool | None = native_counts == verified_counts
    else:
        ranks_match_native = None

    pt_by_rank = [float(value) for value in rank_points]
    games = len(flat_ranks)
    avg_rank = sum(flat_ranks) / games
    challenger_avg_pt = sum(pt_by_rank[rank - 1] for rank in flat_ranks) / games
    # Rank points are zero-sum inside a hanchan, so the three identical
    # opponents average exactly minus one third of the challenger's score.
    trio_avg_pt = -challenger_avg_pt / 3.0
    pt_difference = challenger_avg_pt - trio_avg_pt

    seed_avg_pts = [sum(pt_by_rank[rank - 1] for rank in ranks_by_seed[seed]) / len(ranks_by_seed[seed]) for seed in completed_seeds]
    seed_avg_ranks = [sum(ranks_by_seed[seed]) / len(ranks_by_seed[seed]) for seed in completed_seeds]

    throughput = {
        "hanchans": games,
        "play_seconds": play_seconds,
        "hanchans_per_minute": (games / play_seconds * 60.0) if play_seconds > 0 else None,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None,
    }

    document: dict[str, Any] = {
        "schema": "keqing.mortal.one_vs_three.evaluation.v1",
        "kind": "one_vs_three_native",
        "backend": "libriichi.arena.OneVsThree.py_vs_py",
        "challenger": {"label": challenger_label, "checkpoint": str(challenger_path)},
        "champion": {"label": champion_label, "checkpoint": str(champion_path)},
        "seeds": {"seed_start": int(args.seed_start), "seed_count": total_seeds, "seed_key": int(args.seed_key)},
        "hanchans": games,
        "batch_seeds": int(args.batch_seeds),
        "device": str(args.device),
        "enable_amp": bool(args.enable_amp),
        "model_load_time_sec": load_times,
        "rank_points_profile": rank_points_profile,
        "rank_points_values": pt_by_rank,
        "throughput": throughput,
        "challenger_result": {
            "rank_counts": rank_counts,
            "rank_rates": [count / games for count in rank_counts],
            **summarize_rank_counts(rank_counts, rank_points=rank_points),
            "avg_rank_pt": challenger_avg_pt,
        },
        "trio_result": {
            "games": games,
            "avg_rank_pt": trio_avg_pt,
            "note": "three identical champion seats; rank points are zero-sum per hanchan",
        },
        "pt_difference": {
            "definition": "challenger avg rank pt - trio avg rank pt",
            "value": pt_difference,
            "positive_means_challenger_outscored_the_trio": True,
        },
        "cluster_statistics": {
            "cluster_unit": "seed (4 seat-rotation hanchans per seed)",
            "challenger_avg_pt": _cluster_bootstrap(seed_avg_pts, int(args.bootstrap_reps), int(args.bootstrap_seed)),
            "challenger_avg_rank": _cluster_bootstrap(seed_avg_ranks, int(args.bootstrap_reps), int(args.bootstrap_seed)),
            "sign_test": _exact_sign_test(seed_avg_pts),
        },
        "integrity": {
            "native_rank_counts": native_counts,
            "log_rank_counts": rank_counts,
            "verified_log_rank_counts": verified_counts,
            "verified_seeds": len(verified_seeds),
            "reused_seed_count": reused_seed_count,
            "ranks_match_native": ranks_match_native,
            "reused_logs_are_unverified": reused_seed_count > 0,
            "native_batches": native_batches,
            "per_seed_ranks": {str(seed): ranks_by_seed[seed] for seed in completed_seeds},
        },
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if ranks_match_native is False:
        print("[one_vs_three] WARNING: log-derived ranks disagree with native counts", flush=True)
    if reused_seed_count:
        print(
            f"[one_vs_three] NOTE: {reused_seed_count} seeded batch(es) were reused from disk "
            "and are unverified against a native return",
            flush=True,
        )
    return document


def main() -> None:
    document = run(_parse_args())
    summary = {
        "challenger": document["challenger"]["label"],
        "champion": document["champion"]["label"],
        "hanchans": document["hanchans"],
        "rank_counts": document["challenger_result"]["rank_counts"],
        "avg_rank": document["challenger_result"]["avg_rank"],
        "avg_rank_pt": document["challenger_result"]["avg_rank_pt"],
        "trio_avg_rank_pt": document["trio_result"]["avg_rank_pt"],
        "pt_difference": document["pt_difference"]["value"],
        "avg_pt_ci95": document["cluster_statistics"]["challenger_avg_pt"]["seed_cluster_bootstrap_ci95"],
        "throughput": document["throughput"],
        "ranks_match_native": document["integrity"]["ranks_match_native"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
