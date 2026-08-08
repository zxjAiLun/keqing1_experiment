#!/usr/bin/env python3
"""Audit and split an independent corpus for the project-owned GRP model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON_ROOT = MORTAL_ROOT / "mortal"
for import_root in (REPO_ROOT, MORTAL_ROOT, MORTAL_PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DEFAULT_OUTPUT = Path("artifacts/experiments/model_pool_2026_07/keqing_grp_v1")
DEFAULT_SOURCES = [
    "artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/league_1000h_balanced/combined_logs/*.json.gz",
    "artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/eval_250h_native_screen/logs/*.json.gz",
    "artifacts/experiments/model_pool_2026_07/V2_population_mixed_v4_warmstart_2026_07/eval_250h_native_random/logs/*.json.gz",
    "artifacts/experiments/model_pool_2026_07/V2_population_mixed_v4_warmstart_2026_07/eval_500h_native_random/logs/*.json.gz",
]
DEFAULT_EXCLUDES = [
    "artifacts/experiments/model_pool_2026_07/V2_data/**/*.json.gz",
]
SEED_KEY_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-z])?\.json\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", action="append", dest="sources", default=None)
    parser.add_argument("--exclude", action="append", dest="excludes", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260718)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--allow-small", action="store_true")
    return parser.parse_args()


def _expand(patterns: list[str]) -> list[Path]:
    from glob import glob

    files: set[Path] = set()
    for pattern in patterns:
        expanded = glob(str((REPO_ROOT / pattern).resolve()), recursive=True)
        files.update(Path(item).resolve() for item in expanded if Path(item).is_file())
    return sorted(files)


def canonical_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            event.pop("meta", None)
            if event.get("type") == "start_game":
                event.pop("names", None)
            digest.update(
                json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _seed_key(path: Path) -> str | None:
    match = SEED_KEY_RE.fullmatch(path.name)
    if match is None:
        return None
    return f"{match.group('seed')}_{match.group('key')}"


def _validate_with_libriichi(path: Path) -> None:
    from libriichi.dataset import Grp  # noqa: PLC0415

    games = Grp.load_gz_log_files([str(path)])
    if len(games) != 1 or int(games[0].take_feature().shape[0]) == 0:
        raise ValueError("Grp parser returned no kyoku features")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def render(prefix: list[str], table: Mapping[str, Any]) -> None:
        scalars = [(str(key), value) for key, value in table.items() if not isinstance(value, Mapping)]
        children = [(str(key), value) for key, value in table.items() if isinstance(value, Mapping)]
        if prefix:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(prefix) + "]")
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in children:
            render([*prefix, key], value)

    render([], data)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must be non-negative and below 1")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train and validation fractions must leave a holdout split")

    sources = args.sources or DEFAULT_SOURCES
    excludes = args.excludes or DEFAULT_EXCLUDES
    source_files = _expand(sources)
    excluded_files = _expand(excludes)
    if args.max_files > 0:
        source_files = source_files[: args.max_files]
    if not source_files:
        raise SystemExit("no GRP source logs found")

    excluded_hashes: set[str] = set()
    exclude_errors: list[dict[str, str]] = []
    for path in excluded_files:
        try:
            excluded_hashes.add(canonical_hash(path))
        except Exception as exc:  # noqa: BLE001
            exclude_errors.append({"path": str(path), "error": str(exc)})

    entries: list[dict[str, Any]] = []
    malformed: list[dict[str, str]] = []
    duplicate_hashes = 0
    seen_hashes: set[str] = set()
    seed_keys: dict[str, str] = {}
    seed_overlaps: list[dict[str, str]] = []
    for path in source_files:
        try:
            digest = canonical_hash(path)
            if digest in excluded_hashes:
                continue
            if digest in seen_hashes:
                duplicate_hashes += 1
                continue
            seen_hashes.add(digest)
            seed_key = _seed_key(path)
            if seed_key is not None and seed_key in seed_keys:
                seed_overlaps.append({"seed_key": seed_key, "first": seed_keys[seed_key], "duplicate": str(path)})
            elif seed_key is not None:
                seed_keys[seed_key] = str(path)
            _validate_with_libriichi(path)
            entries.append({"path": str(path), "canonical_hash": digest, "seed_key": seed_key})
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(path), "error": str(exc)})

    if not args.allow_small and len(entries) < 1000:
        raise SystemExit(f"independent GRP corpus is too small: {len(entries)} logs; pass --allow-small to override")
    if malformed:
        raise SystemExit(f"GRP corpus contains malformed logs: {len(malformed)}")
    if seed_overlaps:
        raise SystemExit(f"GRP corpus contains seed/key overlaps: {len(seed_overlaps)}")

    rng = random.Random(args.split_seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * args.train_fraction)
    val_end = train_end + int(len(shuffled) * args.val_fraction)
    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "holdout": shuffled[val_end:],
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split, split_entries in splits.items():
        (output / f"{split}_files.json").write_text(
            json.dumps([entry["path"] for entry in split_entries], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    config = {
        "grp": {
            "state_file": str(output / "keqing_grp_v1.pth"),
            "best_state_file": str(output / "keqing_grp_v1_best.pth"),
            "network": {"hidden_size": 64, "num_layers": 2},
            "control": {
                "device": "cuda:0",
                "batch_size": 512,
                "archive_steps": 2000,
                "val_steps": 400,
            },
            "dataset": {
                "train_files_file": str(output / "train_files.json"),
                "validation_files_file": str(output / "validation_files.json"),
                "holdout_files_file": str(output / "holdout_files.json"),
                "manifest_file": str(output / "dataset_manifest.json"),
                "file_batch_size": 50,
            },
            "optim": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
            },
        },
    }

    manifest = {
        "schema": "keqing.mortal.grp_dataset.v1",
        "output": str(output),
        "sources": sources,
        "excludes": excludes,
        "excluded_file_count": len(excluded_files),
        "excluded_hash_count": len(excluded_hashes),
        "exclude_errors": exclude_errors,
        "split_seed": args.split_seed,
        "fractions": {"train": args.train_fraction, "validation": args.val_fraction, "holdout": 1 - args.train_fraction - args.val_fraction},
        "splits": {
            split: {
                "count": len(split_entries),
                "files": [entry["path"] for entry in split_entries],
                "file_list_sha256": _sha256_json([entry["path"] for entry in split_entries]),
                "canonical_hash_sha256": _sha256_json(sorted(entry["canonical_hash"] for entry in split_entries)),
            }
            for split, split_entries in splits.items()
        },
        "summary": {
            "source_file_count": len(source_files),
            "independent_unique_count": len(entries),
            "within_source_duplicate_count": duplicate_hashes,
            "malformed_count": len(malformed),
            "seed_key_overlap_count": len(seed_overlaps),
            "formal_ab_overlap_count": sum(1 for entry in entries if entry["canonical_hash"] in excluded_hashes),
            "passed": not malformed and not seed_overlaps and bool(entries),
        },
        "entries": entries,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "config.toml").write_text(_dump_toml(config), encoding="utf-8")
    print(json.dumps(manifest["summary"] | {"splits": {key: value["count"] for key, value in manifest["splits"].items()}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
