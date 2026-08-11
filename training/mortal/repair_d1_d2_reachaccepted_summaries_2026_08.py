#!/usr/bin/env python3
"""D1/D2 historical summary correction for the ReachAccepted -1000 defect.

Read-only repair of the D1/D2 paired statistics: the inherited summarizer's
final-score reconstruction omitted the reach_accepted -1000 deduction, so the
recorded D1/D2 effect sizes and bootstrap intervals are non-authoritative. Raw
evaluations are never rerun or modified; old summaries are never overwritten.

This tool recomputes full statistics (seed means, pooled/hierarchical bootstrap
CIs, sign test, D-K0, M0-K0) from the immutable raw logs using the fixed
reconstruction, with the same hard equivalence as the D3 v2 summary:

  * raw-event ranks == native libriichi Stat.from_log ranks (12000 player-ranks)
  * reconstructed rank counts == detailed_stats == metrics
  * mean(delta) == mean(pt_a) - mean(pt_b)
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
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.mortal.summarize_d3_b250_eval_2026_08 import (  # noqa: E402
    EXPECTED_SEED_STARTS,
    EXPECTED_SEEDS,
    RANK_POINTS,
    authoritative_ranks,
    bootstrap,
    exact_sign_test,
    final_scores,
    percentile,
    ranks_from_events,
)

D1_ROOT = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D1_project_owned_population_2026_07\eval_b250_1000h_2026_08"
)
D2_ROOT = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D2_project_owned_descendant_view_mix_2026_08\eval_b250_1000h"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\diagnostics\d1_d2_corrected_summaries"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def seed_logs_and_refs(
    root: Path, seed: int, arm: str
) -> tuple[list[Path], dict[str, Any]]:
    """Return (1000 logs, aggregate rank-count references per label)."""
    run_dir = root / f"seed_{seed}"
    refs: dict[str, dict[str, list[int]]] = {}
    if (run_dir / "logs").is_dir():
        logs = sorted((run_dir / "logs").glob("*.json.gz"))
        if len(logs) != 1000:
            raise ValueError(f"{run_dir}: expected 1000 logs, got {len(logs)}")
        metrics = read_json(run_dir / "metrics.json")
        detailed = read_json(run_dir / "detailed_stats.json")
        for label in ("70k", "ext_mortal", f"M0_{seed}", f"{arm}_{seed}"):
            refs[label] = {
                "detailed": [int(detailed["players"][label]["raw"][f"rank_{rank}"]) for rank in (1, 2, 3, 4)],
                "metrics": [int(value) for value in metrics["metrics"][label]["rank_counts"]],
            }
    else:
        logs = []
        for shard in range(4):
            shard_dir = run_dir / f"eval_shard_{shard:02d}"
            shard_logs = sorted((shard_dir / "logs").glob("*.json.gz"))
            if len(shard_logs) != 250:
                raise ValueError(f"{shard_dir}: expected 250 logs, got {len(shard_logs)}")
            logs.extend(shard_logs)
            metrics = read_json(shard_dir / "metrics.json")
            detailed = read_json(shard_dir / "detailed_stats.json")
            for label in ("70k", "ext_mortal", f"M0_{seed}", f"{arm}_{seed}"):
                refs.setdefault(label, {"detailed": [0, 0, 0, 0], "metrics": [0, 0, 0, 0]})
                for rank in (1, 2, 3, 4):
                    refs[label]["detailed"][rank - 1] += int(detailed["players"][label]["raw"][f"rank_{rank}"])
                    refs[label]["metrics"][rank - 1] += int(metrics["metrics"][label]["rank_counts"][rank - 1])
        if len({path.name for path in logs}) != 1000:
            raise ValueError(f"duplicate log names in seed {seed}")
    return logs, refs


def paired_row(seed: int, path: Path, arm_label: str) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    ranks = ranks_from_events(events)
    d_label = f"{arm_label}_{seed}"
    m0_label = f"M0_{seed}"
    required = ("70k", "ext_mortal", m0_label, d_label)
    if set(ranks) != set(required):
        raise ValueError(f"{path}: expected names {required}, got {sorted(ranks)}")
    points = {label: RANK_POINTS[ranks[label] - 1] for label in required}
    return {
        "seed": seed,
        "source_log": str(path),
        "source_log_sha256": sha256(path),
        "seat_ranks_reconstructed": [ranks[str(name)] for name in events[0]["names"]],
        "rank_70k": ranks["70k"],
        "rank_ext_mortal": ranks["ext_mortal"],
        "rank_m0": ranks[m0_label],
        "rank_d": ranks[d_label],
        "pt_70k": points["70k"],
        "pt_ext_mortal": points["ext_mortal"],
        "pt_m0": points[m0_label],
        "pt_d": points[d_label],
        "delta_pt_d_minus_m0": points[d_label] - points[m0_label],
        "delta_pt_d_minus_70k": points[d_label] - points["70k"],
        "delta_pt_m0_minus_70k": points[m0_label] - points["70k"],
        "d_ahead_of_m0": ranks[d_label] < ranks[m0_label],
        "d_ahead_of_70k": ranks[d_label] < ranks["70k"],
    }


def comparison(rows_by_seed, field, reps, seed) -> dict[str, Any]:
    values_by_seed = {
        seed_id: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for seed_id, rows in rows_by_seed.items()
    }
    seed_means = {str(seed_id): float(values.mean()) for seed_id, values in values_by_seed.items()}
    all_values = np.concatenate(list(values_by_seed.values()))
    ahead_field = {
        "delta_pt_d_minus_m0": "d_ahead_of_m0",
        "delta_pt_d_minus_70k": "d_ahead_of_70k",
    }.get(field)
    result = {
        "field": field,
        "seed_means": seed_means,
        "mean": float(all_values.mean()),
        "median": float(np.median(all_values)),
        "seed_mean_mean": float(np.mean(list(seed_means.values()))),
        "seed_sign_test": exact_sign_test(list(seed_means.values())),
        "ahead_rate": float(np.mean([bool(row[ahead_field]) for rows in rows_by_seed.values() for row in rows])) if ahead_field else None,
    }
    result.update(bootstrap(values_by_seed, reps, seed))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("D1", "D2"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    args = parser.parse_args(argv)
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    root = D1_ROOT if args.arm == "D1" else D2_ROOT
    output_dir = (args.output_dir or DEFAULT_OUTPUT_ROOT / args.arm).resolve()
    arm_label = args.arm

    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    refs_by_seed: dict[int, dict[str, dict[str, list[int]]]] = {}
    for seed in EXPECTED_SEEDS:
        logs, refs = seed_logs_and_refs(root, seed, arm_label)
        rows = [paired_row(seed, path, arm_label) for path in logs]
        rows_by_seed[seed] = rows
        refs_by_seed[seed] = refs
    all_rows = [row for seed in EXPECTED_SEEDS for row in rows_by_seed[seed]]

    # ---- hard equivalence ----
    stat_mismatch: list[str] = []
    for row in all_rows:
        authoritative = authoritative_ranks(Path(row["source_log"]))
        if authoritative is None:
            raise ValueError("libriichi stat unavailable; equivalence cannot be proven")
        for seat in range(4):
            if int(row["seat_ranks_reconstructed"][seat]) != authoritative[seat]:
                stat_mismatch.append(f"{row['source_log']} seat {seat}")
    if stat_mismatch:
        raise ValueError(f"Stat equivalence failed: {len(stat_mismatch)} mismatches")
    rank_equivalence: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        for label in ("70k", "ext_mortal", f"M0_{seed}", f"{arm_label}_{seed}"):
            rank_key = "rank_d" if label == f"{arm_label}_{seed}" else "rank_m0" if label == f"M0_{seed}" else "rank_ext_mortal" if label == "ext_mortal" else "rank_70k"
            reconstructed = [
                sum(1 for row in rows_by_seed[seed] if row[rank_key] == rank) for rank in (1, 2, 3, 4)
            ]
            refs = refs_by_seed[seed][label]
            match = reconstructed == refs["detailed"] == refs["metrics"]
            rank_equivalence[f"{seed}:{label}"] = {
                "reconstructed": reconstructed,
                "detailed_stats": refs["detailed"],
                "metrics": refs["metrics"],
                "match": match,
            }
            if not match:
                raise ValueError(f"rank-count mismatch {seed}:{label}")

    def check_invariant(field: str, pt_a: str, pt_b: str, name: str) -> bool:
        mean_delta = float(np.mean([float(row[field]) for row in all_rows]))
        mean_diff = float(
            np.mean([float(row[pt_a]) for row in all_rows])
            - np.mean([float(row[pt_b]) for row in all_rows])
        )
        ok = math.isclose(mean_delta, mean_diff, rel_tol=1e-9, abs_tol=1e-9)
        if not ok:
            raise ValueError(f"algebraic invariant failed: {name}")
        return ok

    invariants = {
        "mean_d_minus_m0": check_invariant("delta_pt_d_minus_m0", "pt_d", "pt_m0", "d-m0"),
        "mean_d_minus_70k": check_invariant("delta_pt_d_minus_70k", "pt_d", "pt_70k", "d-70k"),
        "mean_m0_minus_70k": check_invariant("delta_pt_m0_minus_70k", "pt_m0", "pt_70k", "m0-70k"),
    }
    equivalence = {
        "stat_logs_checked": len(all_rows),
        "stat_players_checked": len(all_rows) * 4,
        "stat_mismatch_count": 0,
        "rank_count_equivalence": rank_equivalence,
        "algebraic_invariants": invariants,
        "passed": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{args.arm.lower()}_b250_eval_1000h_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    comparisons = {
        f"{args.arm}-M0": comparison(rows_by_seed, "delta_pt_d_minus_m0", args.bootstrap_reps, args.bootstrap_seed),
        f"{args.arm}-70k": comparison(rows_by_seed, "delta_pt_d_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 1),
        "M0-70k": comparison(rows_by_seed, "delta_pt_m0_minus_70k", args.bootstrap_reps, args.bootstrap_seed + 2),
    }
    summary = {
        "schema": f"keqing.mortal.{args.arm.lower()}_b250_eval_summary.v2",
        "protocol": {
            "arm": args.arm,
            "seeds": list(EXPECTED_SEEDS),
            "seed_starts": EXPECTED_SEED_STARTS,
            "seed_key": 8192,
            "games_per_seed": 1000,
            "native_batch_games": 250,
            "seat_mode": "random",
            "device": "cuda",
            "amp": False,
            "rank_points": list(RANK_POINTS),
            "raw_evaluator_commit": None,
            "old_summary_invalidated": True,
            "old_summary_reason": "omitted ReachAccepted -1000",
            "raw_eval_rerun": False,
            "raw_eval_modified": False,
            "repair_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip(),
        },
        "reconstruction_equivalence": equivalence,
        "hanchans": {str(seed): len(rows) for seed, rows in rows_by_seed.items()},
        "comparisons": comparisons,
        "rank_counts": {
            label: dict(sorted(Counter(row[key] for row in all_rows).items()))
            for label, key in (("70k", "rank_70k"), ("ext_mortal", "rank_ext_mortal"), ("M0", "rank_m0"), (args.arm, "rank_d"))
        },
    }
    json_path = output_dir / f"{args.arm.lower()}_b250_eval_summary_v2.json"
    md_path = output_dir / f"{args.arm.lower()}_b250_eval_summary_v2.md"
    equiv_path = output_dir / "reconstruction_equivalence.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"# {args.arm} B250 Evaluation Summary v2 (corrected)", "",
             "- Historical v1 invalidated: omitted ReachAccepted -1000 in final-score reconstruction.",
             f"- Raw evaluation never rerun or modified; repair commit `{summary['protocol']['repair_commit']}`.",
             "", "## Paired Comparisons", "",
             "| Comparison | Seed means | Mean | Pooled 95% CI | Hierarchical 95% CI | Sign test |",
             "|---|---|---:|---:|---:|---:|"]
    for name, value in comparisons.items():
        sign = value["seed_sign_test"]
        lines.append(
            f"| `{name}` | `{value['seed_means']}` | {value['mean']:+.3f} | "
            f"`[{value['pooled_hanchan_bootstrap_ci95'][0]:+.3f}, {value['pooled_hanchan_bootstrap_ci95'][1]:+.3f}]` | "
            f"`[{value['hierarchical_equal_seed_bootstrap_ci95'][0]:+.3f}, {value['hierarchical_equal_seed_bootstrap_ci95'][1]:+.3f}]` | "
            f"{sign['positive_count']}/{sign['non_tie_count']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    equiv_path.write_text(json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "summary_json_sha256": sha256(json_path),
                "summary_md_sha256": sha256(md_path),
                "equivalence_sha256": sha256(equiv_path),
                "rows_sha256": sha256(rows_path),
                "comparisons": comparisons,
                "equivalence": {"stat_players_checked": equivalence["stat_players_checked"], "mismatches": 0, "passed": True},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
