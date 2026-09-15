#!/usr/bin/env python3
"""Adjudicate a bidirectional one-vs-three gate with the frozen section-6 metric.

Frozen formula (P4-M11 plan, section 6):

    S_i   = candidate's mean rank point in the SOLO direction, seed i
            (the solo run's *challenger*, over that seed's four seat rotations)
    O_i   = reference's mean rank point in the MIRROR direction, seed i
            (the mirror run's *challenger*, i.e. the same three-or-more-seat role)
    mirror candidate benefit = -O_i / 3
    G_i  = (S_i - O_i / 3) / 2

Pairing is by seed: the two directions must use the same seed band, and the
bootstrap resamples seed *indices* jointly, so a seed that is hard for both sides
stays hard in every draw. 5000 reps, fixed seed, two-sided 95% percentile CI.

PASS requires all three:
  1. mean(S) > 0 AND mean(-O/3) > 0
  2. G CI lower bound > 0
  3. mean(G) >= threshold (default +2.0 rank pt / candidate seat / hanchan)

The two directions are not independent; no combined p-value is produced.

Promoted from ``artifacts/eval/_p4m11_gateA_logs/adjudicate_gate.py``, which
produced ``artifacts/eval/p4m11_gate{A,B}_verdict.json`` with the P4-M11 schema
name; pass ``--schema keqing.mortal.p4m11_gate_verdict.v1`` to reproduce those
byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SCHEMA = "keqing.mortal.ovt_gate_verdict.v1"

# One seed is four seat rotations -- the gate protocol, frozen in the plan.
HANCHANS_PER_SEED = 4
VALID_RANKS = (1, 2, 3, 4)


def percentile(values: list[float], quantile: float) -> float:
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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_pt(document: dict[str, Any]) -> dict[int, float]:
    """Per-seed mean rank point for the run's *challenger* (the solo seat)."""
    per_seed = document["integrity"]["per_seed_ranks"]
    points = tuple(float(value) for value in document["rank_points_values"])
    out: dict[int, float] = {}
    for seed, ranks in per_seed.items():
        vals = [points[rank - 1] for rank in ranks]
        out[int(seed)] = float(np.mean(vals))
    return out


def band(document: dict[str, Any]) -> tuple[str, int, int]:
    payload = document.get("seeds") or {}
    return str(payload.get("seed_start")), int(payload.get("seed_count", -1)), int(payload.get("seed_key", -1))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def data_problems(
    name: str,
    document: dict[str, Any],
    *,
    expected_start: int,
    expected_count: int,
    per_seed: int = HANCHANS_PER_SEED,
) -> list[str]:
    """Problems with the records that *exist*, not with the declared metadata.

    ``seeds``/``hanchans`` are claims the run writes about itself.  A truncated
    run can keep a 256-seed / 1024-hanchan header while recording one seed, and
    adjudicating that would print a gate verdict computed on a fraction of the
    data.  So the per-seed rank lists are checked against the declared band:
    every expected seed present, no extras, exactly one rank per seat rotation,
    and every rank a real finishing place.
    """
    ranks_by_seed = (document.get("integrity") or {}).get("per_seed_ranks") or {}
    problems: list[str] = []

    expected = list(range(expected_start, expected_start + expected_count))
    actual = sorted(seed for seed in (_as_int(key) for key in ranks_by_seed) if seed is not None)
    if len(actual) != len(ranks_by_seed):
        problems.append(f"{name}: per_seed_ranks has non-integer seed keys")
    if actual != expected:
        if not actual:
            problems.append(
                f"{name}: per_seed_ranks is empty; expected {expected_count} seeds {expected_start}..{expected[-1]}"
            )
        else:
            problems.append(
                f"{name}: per_seed_ranks covers {len(actual)} seeds {actual[0]}..{actual[-1]}; "
                f"expected {expected_count} seeds {expected_start}..{expected[-1]}"
            )

    wrong_length = sorted(
        str(seed) for seed, ranks in ranks_by_seed.items() if len(ranks) != per_seed
    )
    if wrong_length:
        problems.append(
            f"{name}: {len(wrong_length)} seeds do not hold exactly {per_seed} ranks "
            f"(e.g. {wrong_length[:3]})"
        )

    invalid: list[str] = []
    for seed, ranks in ranks_by_seed.items():
        for rank in ranks:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank not in VALID_RANKS:
                invalid.append(str(seed))
                break
    if invalid:
        problems.append(
            f"{name}: {len(invalid)} seeds hold a rank outside {list(VALID_RANKS)} (e.g. {sorted(invalid)[:3]})"
        )

    recorded = sum(len(ranks) for ranks in ranks_by_seed.values())
    declared = document.get("hanchans")
    if recorded != declared:
        problems.append(f"{name}: {recorded} recorded hanchans but the metadata declares {declared}")
    if declared != expected_count * per_seed:
        problems.append(
            f"{name}: {declared} declared hanchans != {expected_count} seeds x {per_seed}"
        )
    return problems


def identity_of(metrics_path: Path, document: dict[str, Any], role: str) -> dict[str, Any]:
    """Raw path + sha256 for one seat, preferring the run_identity.json sidecar.

    metrics.json records only the label and the path it was handed on the command
    line; the sidecar the evaluator wrote next to it carries the hash that actually
    ran.  Falling back to the metrics file keeps this usable on older runs.
    """
    path = document[role].get("checkpoint")
    digest = document[role].get("sha256")
    sidecar = metrics_path.parent / "run_identity.json"
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding="utf-8")).get(role) or {}
        path = recorded.get("path", path)
        digest = recorded.get("sha256", digest)
    return {"path": path, "sha256": digest}


def artifact_id(metrics_path: Path, document: dict[str, Any], role: str) -> tuple[str | None, str | None]:
    """Comparable identity of the checkpoint in ``role`` (path normalized, plus digest)."""
    identity = identity_of(metrics_path, document, role)
    path = identity["path"]
    normalized = str(path).replace("\\", "/").lower() if path else None
    return (normalized, identity["sha256"])


def describe_side(document: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    solo = identity_of(metrics_path, document, "challenger")
    trio = identity_of(metrics_path, document, "champion")
    return {
        "metrics": str(metrics_path),
        "solo_seat": document["challenger"]["label"],
        "solo_checkpoint": solo["path"],
        "solo_checkpoint_sha256": solo["sha256"],
        "opponent_trio": document["champion"]["label"],
        "opponent_trio_checkpoint": trio["path"],
        "opponent_trio_sha256": trio["sha256"],
        "seeds": document["seeds"],
        "hanchans": document["hanchans"],
        "batch_seeds": document["batch_seeds"],
        "enable_amp": document.get("enable_amp"),
        "device": document.get("device"),
        "challenger_result": document["challenger_result"],
        "published_cluster_ci95": document.get("cluster_statistics", {})
        .get("challenger_avg_pt", {})
        .get("seed_cluster_bootstrap_ci95"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--solo-metrics", type=Path, required=True)
    parser.add_argument("--mirror-metrics", type=Path, required=True)
    parser.add_argument("--gate-name", default="practical_gate")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--expect-seeds", type=int, default=256)
    parser.add_argument("--expect-hanchans", type=int, default=1024)
    parser.add_argument("--expect-seed-start", type=int, default=None)
    parser.add_argument("--expect-seed-key", type=int, default=8192)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    solo, mirror = load(args.solo_metrics), load(args.mirror_metrics)

    # --- identity ---------------------------------------------------------
    problems: list[str] = []
    for name, document in (("solo", solo), ("mirror", mirror)):
        start, count, key = band(document)
        if document["challenger"]["label"] != (args.candidate_label if name == "solo" else args.reference_label):
            problems.append(f"{name}: challenger label is {document['challenger']['label']!r}")
        if document["champion"]["label"] != (args.reference_label if name == "solo" else args.candidate_label):
            problems.append(f"{name}: champion label is {document['champion']['label']!r}")
        if count != args.expect_seeds or document["hanchans"] != args.expect_hanchans:
            problems.append(f"{name}: {count} seeds / {document['hanchans']} hanchans (expected {args.expect_seeds}/{args.expect_hanchans})")
        if args.expect_seed_start is not None and start != str(args.expect_seed_start):
            problems.append(f"{name}: seed_start is {start}")
        if args.expect_seed_key is not None and key != args.expect_seed_key:
            problems.append(f"{name}: seed_key is {key}")
        if document.get("enable_amp"):
            problems.append(f"{name}: enable_amp is set (the gate is FP32)")
        reused = [b for b in document["integrity"].get("native_batches", []) if b.get("reused")]
        if reused:
            problems.append(f"{name}: {len(reused)} reused native batches")
        # The declared band above is a claim; these are the records that exist.
        expected_start = args.expect_seed_start if args.expect_seed_start is not None else _as_int(start)
        if expected_start is None:
            problems.append(
                f"{name}: seed_start is {start!r}, so the recorded seed range cannot be checked "
                "(pass --expect-seed-start)"
            )
        else:
            problems += data_problems(
                name, document, expected_start=expected_start, expected_count=args.expect_seeds
            )

    # One artifact as the candidate (solo seat in one direction, trio in the other),
    # one as the reference.  A differing path or digest means the two directions did
    # not test the same pair, which is not a gate.
    if artifact_id(args.solo_metrics, solo, "challenger") != artifact_id(args.mirror_metrics, mirror, "champion"):
        problems.append("candidate artifact differs between the two directions")
    if artifact_id(args.solo_metrics, solo, "champion") != artifact_id(args.mirror_metrics, mirror, "challenger"):
        problems.append("reference artifact differs between the two directions")
    if solo["rank_points_values"] != mirror["rank_points_values"]:
        problems.append("rank point tables differ between the two runs")

    if problems:
        raise SystemExit("refusing to adjudicate:\n  - " + "\n  - ".join(problems))

    # --- per-seed pairing -------------------------------------------------
    s_by_seed, o_by_seed = seed_pt(solo), seed_pt(mirror)
    if set(s_by_seed) != set(o_by_seed):
        raise SystemExit(
            f"seed sets differ (solo {len(s_by_seed)}, mirror {len(o_by_seed)}); pairing requires the same band"
        )
    seeds = sorted(s_by_seed)
    S = np.asarray([s_by_seed[s] for s in seeds], dtype=np.float64)
    # O in the frozen formula; the name is spelled out to avoid ruff E741.
    O_mirror = np.asarray([o_by_seed[s] for s in seeds], dtype=np.float64)

    neg_o3 = -O_mirror / 3.0
    G = (S - O_mirror / 3.0) / 2.0

    # --- paired cluster bootstrap ----------------------------------------
    rng = np.random.default_rng(args.bootstrap_seed)
    draws = {"mean_S": [], "mean_O": [], "mean_negO3": [], "mean_G": []}
    for _ in range(args.reps):
        idx = rng.integers(0, len(seeds), size=len(seeds))
        draws["mean_S"].append(float(S[idx].mean()))
        draws["mean_O"].append(float(O_mirror[idx].mean()))
        draws["mean_negO3"].append(float(neg_o3[idx].mean()))
        draws["mean_G"].append(float(G[idx].mean()))

    def summary(values: np.ndarray, key: str) -> dict[str, Any]:
        return {
            "mean": float(values.mean()),
            "ci95": [percentile(draws[key], 0.025), percentile(draws[key], 0.975)],
            "seeds_positive": int((values > 0).sum()),
            "seeds_non_tie": int((np.abs(values) > 1e-12).sum()),
        }

    stat_S = summary(S, "mean_S")
    stat_O = summary(O_mirror, "mean_O")
    stat_neg = summary(neg_o3, "mean_negO3")
    stat_G = summary(G, "mean_G")

    cond_both_directions = stat_S["mean"] > 0.0 and stat_neg["mean"] > 0.0
    cond_ci = stat_G["ci95"][0] > 0.0
    cond_threshold = stat_G["mean"] >= args.threshold
    passed = bool(cond_both_directions and cond_ci and cond_threshold)

    verdict = {
        "schema": args.schema,
        "gate": args.gate_name,
        "verdict": "supported" if passed else "not_supported",
        "metric": "(S - O/3) / 2, per candidate seat per hanchan (plan section 6)",
        "pairing": f"paired by seed over {len(seeds)} seeds x 4 seat rotations; both directions resampled jointly",
        "bootstrap": {"reps": args.reps, "seed": args.bootstrap_seed, "ci": "two-sided 95% percentile"},
        "candidate_label": args.candidate_label,
        "reference_label": args.reference_label,
        "rank_points_profile": solo.get("rank_points_profile"),
        "S_solo_candidate": stat_S,
        "O_mirror_reference": stat_O,
        "mirror_candidate_benefit_negO3": stat_neg,
        "G": {**stat_G, "threshold": args.threshold},
        "conditions": {
            "both_directions_positive": cond_both_directions,
            "G_ci_lower_above_zero": cond_ci,
            f"mean_G_at_least_{args.threshold}": cond_threshold,
        },
        "solo_run": describe_side(solo, args.solo_metrics),
        "mirror_run": describe_side(mirror, args.mirror_metrics),
        "note": (
            "the two directions share deals and are not independent; each direction is reported "
            "separately as required, and no combined p-value is claimed"
        ),
    }

    text = json.dumps(verdict, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")

    print(f"gate                : {verdict['gate']}  ->  {verdict['verdict'].upper()}")
    print(f"seeds / hanchans    : {len(seeds)} / {solo['hanchans']}  (band {band(solo)})")
    print(f"S  (solo candidate) : {stat_S['mean']:+.4f}  CI95 [{stat_S['ci95'][0]:+.4f}, {stat_S['ci95'][1]:+.4f}]")
    print(f"O  (mirror ref)     : {stat_O['mean']:+.4f}  CI95 [{stat_O['ci95'][0]:+.4f}, {stat_O['ci95'][1]:+.4f}]")
    print(f"-O/3 (mirror bene)  : {stat_neg['mean']:+.4f}  CI95 [{stat_neg['ci95'][0]:+.4f}, {stat_neg['ci95'][1]:+.4f}]")
    print(f"G  (frozen metric)  : {stat_G['mean']:+.4f}  CI95 [{stat_G['ci95'][0]:+.4f}, {stat_G['ci95'][1]:+.4f}]  threshold >= {args.threshold}")
    for key, value in verdict["conditions"].items():
        print(f"  {'PASS' if value else 'FAIL'}  {key}")
    if args.output:
        print(f"verdict written     : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
