#!/usr/bin/env python3
"""Audit the complete 24-shard D1 native selfplay population."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import re
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_d1_dataset import canonical_hash, audit as audit_shard  # noqa: E402


LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-d])?\.json\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=6000)
    parser.add_argument("--shard-games", type=int, default=250)
    parser.add_argument("--expected-shards", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=1600000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_population(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_shards * args.shard_games != args.expected_games:
        raise ValueError("expected-shards * shard-games must equal expected-games")

    expected_names = {f"shard_{index:02d}" for index in range(args.expected_shards)}
    actual_names = {
        path.name
        for path in args.data_root.iterdir()
        if path.is_dir() and path.name.startswith("shard_")
    } if args.data_root.is_dir() else set()
    missing_shards = sorted(expected_names - actual_names)
    unexpected_shards = sorted(actual_names - expected_names)

    shard_reports: list[dict[str, Any]] = []
    canonical_seen: dict[str, str] = {}
    seed_seen: dict[str, str] = {}
    cross_shard_canonical_duplicates: list[dict[str, str]] = []
    cross_shard_seed_duplicates: list[dict[str, str]] = []

    for index in range(args.expected_shards):
        shard_name = f"shard_{index:02d}"
        shard_dir = args.data_root / shard_name
        shard_args = Namespace(
            data_dir=shard_dir,
            expected_games=args.shard_games,
            seed_start=args.seed_start + index * args.shard_games,
            seed_key=args.seed_key,
            output=shard_dir / "dataset_audit.json",
        )
        if shard_dir.is_dir():
            report = audit_shard(shard_args)
        else:
            report = {
                "data_dir": str(shard_dir.resolve()),
                "expected_games": args.shard_games,
                "summary": {
                    "file_count": 0,
                    "canonical_unique_count": 0,
                    "seed_key_unique_count": 0,
                    "trainable_K0_70k_perspective_count": 0,
                    "malformed_count": 1,
                    "passed": False,
                },
                "malformed": [{"path": str(shard_dir), "error": "missing shard directory"}],
            }
        report["shard"] = shard_name
        shard_reports.append(report)

        for path in sorted((shard_dir / "logs").glob("*.json.gz")):
            match = LOG_NAME_RE.fullmatch(path.name)
            if match is None:
                continue
            seed_key = f"{match['seed']}_{match['key']}"
            previous_seed_path = seed_seen.get(seed_key)
            if previous_seed_path is not None:
                cross_shard_seed_duplicates.append({"seed_key": seed_key, "first": previous_seed_path, "second": str(path)})
            seed_seen[seed_key] = str(path)
            digest = canonical_hash(path)
            previous_canonical_path = canonical_seen.get(digest)
            if previous_canonical_path is not None:
                cross_shard_canonical_duplicates.append({"canonical_sha256": digest, "first": previous_canonical_path, "second": str(path)})
            canonical_seen[digest] = str(path)

    file_count = sum(int(report["summary"].get("file_count", 0)) for report in shard_reports)
    malformed_count = sum(int(report["summary"].get("malformed_count", 0)) for report in shard_reports)
    trainable_count = sum(int(report["summary"].get("trainable_K0_70k_perspective_count", 0)) for report in shard_reports)
    shard_passed = all(bool(report["summary"].get("passed", False)) for report in shard_reports)
    passed = (
        not missing_shards
        and not unexpected_shards
        and shard_passed
        and file_count == args.expected_games
        and len(seed_seen) == args.expected_games
        and len(canonical_seen) == args.expected_games
        and not cross_shard_seed_duplicates
        and not cross_shard_canonical_duplicates
        and malformed_count == 0
        and trainable_count == args.expected_games
    )
    return {
        "schema": "keqing.mortal.d1_population_audit.v1",
        "data_root": str(args.data_root.resolve()),
        "expected_games": args.expected_games,
        "expected_shards": args.expected_shards,
        "shard_games": args.shard_games,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.expected_games - 1,
        "seed_key": args.seed_key,
        "missing_shards": missing_shards,
        "unexpected_shards": unexpected_shards,
        "summary": {
            "file_count": file_count,
            "canonical_unique_count": len(canonical_seen),
            "seed_key_unique_count": len(seed_seen),
            "trainable_K0_70k_perspective_count": trainable_count,
            "malformed_count": malformed_count,
            "cross_shard_canonical_duplicate_count": len(cross_shard_canonical_duplicates),
            "cross_shard_seed_duplicate_count": len(cross_shard_seed_duplicates),
            "passed": passed,
        },
        "cross_shard_canonical_duplicates": cross_shard_canonical_duplicates[:20],
        "cross_shard_seed_duplicates": cross_shard_seed_duplicates[:20],
        "shards": shard_reports,
    }


def main() -> None:
    args = parse_args()
    report = audit_population(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if not report["summary"]["passed"]:
        raise SystemExit("D1 population audit failed")


if __name__ == "__main__":
    main()
