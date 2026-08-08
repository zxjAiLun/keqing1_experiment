#!/usr/bin/env python3
"""Audit one D1 training-view data shard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-d])?\.json\.gz$")
EXPECTED_LABELS = {"K0_70k", "ext_mortal", "V3_74000", "V2_74000"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            event.pop("meta", None)
            if event.get("type") == "start_game":
                event.pop("names", None)
            digest.update(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    files = sorted((args.data_dir / "logs").glob("*.json.gz"))
    malformed: list[dict[str, str]] = []
    canonical: dict[str, list[str]] = {}
    seed_keys: dict[str, list[str]] = {}
    action_counts: dict[str, int] = {}
    trainable_perspectives = 0
    for path in files:
        try:
            match = LOG_NAME_RE.fullmatch(path.name)
            if match is None:
                raise ValueError("unexpected native log filename")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                start = json.loads(next(handle))
                events = [json.loads(line) for line in handle if line.strip()]
            if start.get("type") != "start_game" or start.get("seed") != [int(match["seed"]), int(match["key"])]:
                raise ValueError("start_game seed does not match filename")
            names = start.get("names")
            if not isinstance(names, list) or len(names) != 4 or set(names) != EXPECTED_LABELS:
                raise ValueError(f"unexpected model labels: {names!r}")
            if names.count("K0_70k") != 1:
                raise ValueError("expected exactly one K0_70k trainable perspective")
            trainable_perspectives += 1
            seed = int(match["seed"])
            key = int(match["key"])
            if key != args.seed_key or not args.seed_start <= seed < args.seed_start + args.expected_games:
                raise ValueError(f"seed/key outside expected range: {seed}_{key}")
            seed_keys.setdefault(f"{seed}_{key}", []).append(str(path))
            canonical.setdefault(canonical_hash(path), []).append(str(path))
            for event in events:
                event_type = str(event.get("type"))
                action_counts[event_type] = action_counts.get(event_type, 0) + 1
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(path), "error": str(exc)})
    file_count = len(files)
    canonical_unique = len(canonical)
    seed_unique = len(seed_keys)
    passed = (
        file_count == args.expected_games
        and trainable_perspectives == args.expected_games
        and len(malformed) == 0
        and canonical_unique == args.expected_games
        and seed_unique == args.expected_games
    )
    return {
        "schema": "keqing.mortal.d1_dataset_audit.v1",
        "data_dir": str(args.data_dir.resolve()),
        "expected_games": args.expected_games,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.expected_games - 1,
        "seed_key": args.seed_key,
        "expected_labels": sorted(EXPECTED_LABELS),
        "summary": {
            "file_count": file_count,
            "canonical_unique_count": canonical_unique,
            "canonical_duplicate_count": file_count - canonical_unique,
            "seed_key_unique_count": seed_unique,
            "seed_key_overlap_count": file_count - seed_unique,
            "trainable_K0_70k_perspective_count": trainable_perspectives,
            "malformed_count": len(malformed),
            "action_counts": action_counts,
            "passed": passed,
        },
        "malformed": malformed[:20],
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if not report["summary"]["passed"]:
        raise SystemExit("D1 dataset audit failed")


if __name__ == "__main__":
    main()
