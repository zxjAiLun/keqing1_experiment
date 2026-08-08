"""Shared metric export helpers for Mortal evaluation scripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

METRICS_SCHEMA = "keqing.mortal.eval.metrics.v1"
TENHOU_RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
RANK_POINT_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "tenhou_reference": TENHOU_RANK_POINTS,
    "mortal_default": (6.0, 4.0, 2.0, 0.0),
    "avoid4_norm": (15.0 / 7.0, 9.0 / 7.0, 3.0 / 7.0, -27.0 / 7.0),
    "top1_norm": (3.3, 0.3, -0.9, -2.7),
    "avoid4_raw": (4.0, 3.0, 2.0, -3.0),
    "top1_raw": (8.0, 3.0, 1.0, -2.0),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_rank_points(value: str | Sequence[int | float]) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            points = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"rank points must be comma-separated numbers, got {value!r}") from exc
    else:
        points = tuple(float(part) for part in value)
    if len(points) != 4:
        raise ValueError(f"rank points must have length 4, got {len(points)}")
    if not all(math.isfinite(point) for point in points):
        raise ValueError(f"rank points must be finite numbers, got {points!r}")
    return (points[0], points[1], points[2], points[3])


def resolve_rank_points(
    *,
    rank_points: str | Sequence[int | float] | None = None,
    profile: str = "tenhou_reference",
) -> tuple[str, tuple[float, float, float, float]]:
    profile = str(profile or "tenhou_reference")
    if rank_points is not None:
        return "custom", parse_rank_points(rank_points)
    if profile == "custom":
        raise ValueError("--rank-points-profile custom requires --rank-points")
    try:
        return profile, RANK_POINT_PROFILES[profile]
    except KeyError as exc:
        known = ", ".join(sorted((*RANK_POINT_PROFILES, "custom")))
        raise ValueError(f"unknown rank point profile {profile!r}; known profiles: {known}") from exc


def rank_points_metadata(
    *,
    rank_points: str | Sequence[int | float] | None = None,
    profile: str = "tenhou_reference",
) -> dict[str, Any]:
    resolved_profile, points = resolve_rank_points(rank_points=rank_points, profile=profile)
    return {
        "rank_points_profile": resolved_profile,
        "rank_points_values": [float(value) for value in points],
    }


def add_rank_point_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rank-points-profile",
        choices=(*sorted(RANK_POINT_PROFILES), "custom"),
        default="tenhou_reference",
        help="Rank-point profile for avg_rank_pt metrics. Defaults to Tenhou reference.",
    )
    parser.add_argument(
        "--rank-points",
        default=None,
        help="Custom comma-separated rank points, e.g. 90,45,0,-135. Implies profile=custom.",
    )


def summarize_rank_counts(
    rank_counts: Sequence[int | float],
    *,
    rank_points: Sequence[int | float] = TENHOU_RANK_POINTS,
) -> dict[str, Any]:
    counts = [int(value) for value in rank_counts]
    if len(counts) != 4:
        raise ValueError(f"rank_counts must have length 4, got {len(counts)}")
    points = [float(value) for value in rank_points]
    if len(points) != 4:
        raise ValueError(f"rank_points must have length 4, got {len(points)}")
    games = sum(counts)
    if games <= 0:
        return {
            "games": 0,
            "rank_counts": counts,
            "avg_rank": None,
            "avg_rank_pt": None,
        }
    return {
        "games": games,
        "rank_counts": counts,
        "avg_rank": sum((idx + 1) * count for idx, count in enumerate(counts)) / games,
        "avg_rank_pt": sum(points[idx] * count for idx, count in enumerate(counts)) / games,
    }


def summarize_rank_counts_with_references(
    rank_counts: Sequence[int | float],
    *,
    rank_points: Sequence[int | float],
) -> dict[str, Any]:
    summary = summarize_rank_counts(rank_counts, rank_points=rank_points)
    summary["avg_rank_pt_training_profile"] = summary["avg_rank_pt"]
    summary["avg_rank_pt_tenhou_reference"] = summarize_rank_counts(
        rank_counts,
        rank_points=TENHOU_RANK_POINTS,
    )["avg_rank_pt"]
    return summary


def build_metrics_document(
    *,
    run: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    rank_points_profile: str | None = None,
    rank_points_values: Sequence[int | float] | None = None,
) -> dict[str, Any]:
    document = {
        "schema": METRICS_SCHEMA,
        "created_at": utc_now_iso(),
        "run": dict(run),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts or {}),
    }
    if rank_points_profile is not None or rank_points_values is not None:
        if rank_points_values is None:
            _, points = resolve_rank_points(profile=rank_points_profile or "tenhou_reference")
        else:
            points = parse_rank_points(rank_points_values)
        document["rank_points_profile"] = rank_points_profile or "custom"
        document["rank_points_values"] = [float(value) for value in points]
    return document


def write_metrics(path: str | Path, document: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
