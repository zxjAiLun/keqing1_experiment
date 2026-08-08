#!/usr/bin/env python3
"""Audit the retained mixed-ecology synthetic pools before offline training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.prepare_v2_population_mixed_warmstart import POOL_SPECS


LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-d])?\.json\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/experiments/model_pool_2026_07/V2_data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/model_pool_2026_07/V2_population_mixed_v4_warmstart_2026_07/dataset_audit.json"))
    parser.add_argument("--min-coverage", type=float, default=0.95)
    return parser.parse_args()


def canonical_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
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


def audit_pool(data_root: Path, pool_id: str, expected_start: int) -> dict[str, Any]:
    files = sorted((data_root / pool_id / "logs").glob("*.json.gz"))
    malformed: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = {}
    seed_keys: dict[str, list[str]] = {}
    trainable_ext_mortal_seats = 0
    for file_path in files:
        try:
            match = LOG_NAME_RE.fullmatch(file_path.name)
            if match is None:
                raise ValueError("unexpected native log filename")
            with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                start = json.loads(next(handle))
            names = start.get("names")
            if not isinstance(names, list) or len(names) != 4:
                raise ValueError(f"invalid start_game names: {names!r}")
            if names.count("ext_mortal") != 1:
                raise ValueError(f"expected exactly one ext_mortal seat, found {names!r}")
            trainable_ext_mortal_seats += 1
            seed = int(match.group("seed"))
            key = int(match.group("key"))
            if seed < expected_start or seed >= expected_start + 2000:
                raise ValueError(f"seed {seed} outside expected [{expected_start}, {expected_start + 1999}]")
            seed_keys.setdefault(f"{seed}_{key}", []).append(str(file_path))
            hashes.setdefault(canonical_hash(file_path), []).append(str(file_path))
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(file_path), "error": str(exc)})
    return {
        "pool_id": pool_id,
        "expected_games": 2000,
        "file_count": len(files),
        "trainable_ext_mortal_seat_count": trainable_ext_mortal_seats,
        "malformed": malformed[:10],
        "malformed_count": len(malformed),
        "canonical_hashes": [[digest, paths[0]] for digest, paths in hashes.items()],
        "seed_keys": [[seed_key, paths[0]] for seed_key, paths in seed_keys.items()],
        "within_pool_duplicate_count": len(files) - len(hashes),
        "within_pool_seed_duplicate_count": len(files) - len(seed_keys),
    }


def main() -> None:
    args = parse_args()
    pools = [audit_pool(args.data_root, pool_id, seed_start) for pool_id, seed_start in POOL_SPECS]
    hash_owners: dict[str, list[str]] = {}
    seed_owners: dict[str, list[str]] = {}
    for pool in pools:
        for digest, file_path in pool.pop("canonical_hashes"):
            hash_owners.setdefault(digest, []).append(file_path)
        for seed_key, file_path in pool.pop("seed_keys"):
            seed_owners.setdefault(seed_key, []).append(file_path)
    file_count = sum(int(pool["file_count"]) for pool in pools)
    ext_mortal_seats = sum(int(pool["trainable_ext_mortal_seat_count"]) for pool in pools)
    malformed = sum(int(pool["malformed_count"]) for pool in pools)
    duplicate_count = file_count - len(hash_owners)
    seed_overlap_count = file_count - len(seed_owners)
    coverage = file_count / 6000
    passed = (
        coverage >= float(args.min_coverage)
        and ext_mortal_seats / 6000 >= float(args.min_coverage)
        and malformed == 0
        and duplicate_count == 0
        and seed_overlap_count == 0
    )
    report = {
        "schema": "keqing.mortal.population_synthetic_dataset_audit.v1",
        "data_root": str(args.data_root),
        "summary": {
            "expected_games": 6000,
            "file_count": file_count,
            "coverage": coverage,
            "expected_trainable_ext_mortal_seats": 6000,
            "trainable_ext_mortal_seat_count": ext_mortal_seats,
            "malformed_count": malformed,
            "canonical_unique_count": len(hash_owners),
            "canonical_duplicate_count": duplicate_count,
            "seed_key_unique_count": len(seed_owners),
            "seed_key_overlap_count": seed_overlap_count,
            "passed": passed,
        },
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit("population synthetic dataset audit failed")


if __name__ == "__main__":
    main()
