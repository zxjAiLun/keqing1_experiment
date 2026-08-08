#!/usr/bin/env python3
"""Prepare the D2 V2/V3 descendant-view mix without starting training.

The input is the completed D1 population.  Each source hanchan is assigned
exactly one trainable perspective by canonical-log hash parity, so the D2
dataset has 3,000 V2 views and 3,000 V3 views rather than double-counting both
views from every hanchan.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

try:
    import toml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("prepare script requires the project's toml dependency") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_D1_ROOT = REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07"
DEFAULT_D1_INDEX = DEFAULT_D1_ROOT / "training_prep_2026_07/file_index_d1.pth"
DEFAULT_BASE_CONFIG = DEFAULT_D1_ROOT / "training_prep_2026_07/M0_control/seed_20260806/config.toml"
DEFAULT_PARENT = REPO_ROOT / "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D2_project_owned_descendant_view_mix_2026_08"
SEEDS = (20260806, 20260807, 20260808)
VIEW_LABELS = ("V2_74000", "V3_74000")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_file_index(path: Path) -> list[Path]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"file index has no file_list: {path}")
    files = [resolve_path(value) for value in values]
    if len(files) != len(set(files)):
        raise ValueError(f"duplicate file path in index: {path}")
    missing = [str(value) for value in files if not value.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return files


def start_event(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        event = json.loads(next(handle))
    if event.get("type") != "start_game":
        raise ValueError(f"first event is not start_game: {path}")
    return event


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
            digest.update(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def write_index(path: Path, files: list[Path]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"file_list": [str(value) for value in files]}, path)
    return sha256_file(path)


def build_config(
    base: dict[str, Any],
    *,
    output_dir: Path,
    file_index: Path,
    mapping_path: Path,
    label_file: Path,
    data_glob: str,
    seed: int,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["control"]["state_file"] = str((output_dir / "mortal.pth").resolve())
    config["control"]["best_state_file"] = str((output_dir / "mortal_best.pth").resolve())
    config["control"]["tensorboard_dir"] = str((output_dir / "tb_mortal").resolve())
    dataset = config["dataset"]
    dataset["globs"] = [data_glob]
    dataset["file_index"] = str(file_index.resolve())
    dataset["player_names_files"] = [str(label_file.resolve())]
    dataset["player_names_by_file"] = str(mapping_path.resolve())
    dataset["num_workers"] = 0
    dataset["num_epochs"] = 3
    config["objective"] = {"mode": "behavior_action_mc"}
    config["experiment"] = {
        "route": "D2_variant",
        "trainable_label": "V2_74000+V3_74000",
        "training_seed": seed,
        "parent_steps": 70000,
        "reward_mode": "final_rank_mc",
        "view_assignment": "canonical_hash_parity_50_50",
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-index", type=Path, default=DEFAULT_D1_INDEX)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    d1_index = args.d1_index.resolve()
    d1_root = args.d1_root.resolve()
    base_config_path = args.base_config.resolve()
    parent_path = args.parent.resolve()
    output = args.output_dir.resolve()
    if not d1_index.is_file() or not base_config_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("D1 index, base config, and 70k parent are required")
    if git_info()["branch"] != "codex/mortal-training-next":
        raise SystemExit("D2 preparation must run on codex/mortal-training-next")

    files = read_file_index(d1_index)
    if len(files) != 6000:
        raise ValueError(f"expected 6000 D1 files, got {len(files)}")
    rows = []
    for index, path in enumerate(files, start=1):
        start = start_event(path)
        names = list(start.get("names", []))
        missing = [label for label in VIEW_LABELS if label not in names]
        if missing:
            raise ValueError(f"{path} is missing expected descendant labels: {missing}")
        canonical = canonical_hash(path)
        seed = start.get("seed")
        if not isinstance(seed, list) or len(seed) != 2:
            raise ValueError(f"invalid seed in {path}")
        rows.append(
            {
                "path": str(path),
                "compressed_sha256": sha256_file(path),
                "canonical_hanchan_sha256": canonical,
                "seed": [int(seed[0]), int(seed[1])],
            }
        )
        if index % 500 == 0:
            print(f"[d2-prepare] hashed {index}/6000", flush=True)
    if len({row["canonical_hanchan_sha256"] for row in rows}) != 6000:
        raise ValueError("D1 source contains duplicate canonical hanchans")
    if len({tuple(row["seed"]) for row in rows}) != 6000:
        raise ValueError("D1 source contains duplicate seed/key pairs")

    rows.sort(key=lambda row: row["canonical_hanchan_sha256"])
    assignment: dict[str, str] = {}
    for index, row in enumerate(rows):
        assignment[row["path"]] = VIEW_LABELS[index % 2]
        row["trainable_label"] = assignment[row["path"]]
    counts = {label: sum(value == label for value in assignment.values()) for label in VIEW_LABELS}
    if counts != {"V2_74000": 3000, "V3_74000": 3000}:
        raise AssertionError(f"unexpected D2 view counts: {counts}")

    dataset_dir = output / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    all_files = [Path(row["path"]) for row in rows]
    v2_files = [path for path in all_files if assignment[str(path)] == "V2_74000"]
    v3_files = [path for path in all_files if assignment[str(path)] == "V3_74000"]
    d2_index_sha = write_index(dataset_dir / "file_index_d2.pth", all_files)
    v2_index_sha = write_index(dataset_dir / "file_index_v2.pth", v2_files)
    v3_index_sha = write_index(dataset_dir / "file_index_v3.pth", v3_files)
    mapping_bytes = (json.dumps(assignment, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    mapping_path = dataset_dir / "player_names_by_file.json"
    mapping_path.write_bytes(mapping_bytes)
    rows_path = dataset_dir / "view_assignment.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    label_path = dataset_dir / "d2_train_labels.txt"
    label_path.write_text("V2_74000\nV3_74000\n", encoding="utf-8")

    base = toml.load(str(base_config_path))
    data_glob = str((d1_root / "data/shard_*/logs/*.json.gz").resolve())
    prep_dir = output / "training_prep_2026_08"
    configs = []
    for seed in SEEDS:
        run_dir = prep_dir / "D2_variant" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(
            base,
            output_dir=run_dir,
            file_index=dataset_dir / "file_index_d2.pth",
            mapping_path=mapping_path,
            label_file=label_path,
            data_glob=data_glob,
            seed=seed,
        )
        config_path = run_dir / "config.toml"
        config_path.write_text(toml.dumps(config), encoding="utf-8")
        configs.append(
            {
                "seed": seed,
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "run_dir": str(run_dir.resolve()),
            }
        )

    parent_state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if int(parent_state.get("steps", -1)) != 70000:
        raise ValueError("parent checkpoint must be step 70000")
    parent_sha = sha256_file(parent_path)
    source_manifest = d1_root / "manifest.json"
    source_audit = d1_root / "dataset_audit.json"
    contract = {
        "schema": "keqing.mortal.d2_training_prep.v1",
        "experiment_id": "D2_project_owned_descendant_view_mix_2026_08",
        "status": "prepared_not_started",
        "git": git_info(),
        "seeds": list(SEEDS),
        "source": {
            "d1_root": str(d1_root),
            "d1_file_index": str(d1_index),
            "d1_file_index_sha256": sha256_file(d1_index),
            "d1_manifest": str(source_manifest),
            "d1_audit": str(source_audit),
            "source_file_count": len(files),
            "source_canonical_set_sha256": sha256_bytes(
                "\n".join(row["canonical_hanchan_sha256"] for row in rows).encode("utf-8")
            ),
        },
        "view_assignment": {
            "method": "sort canonical_hanchan_sha256, even index V2, odd index V3",
            "outcome_independent": True,
            "counts": counts,
            "mapping": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "assignment_sha256": sha256_file(rows_path),
        },
        "dataset": {
            "file_index": str((dataset_dir / "file_index_d2.pth").resolve()),
            "file_index_sha256": d2_index_sha,
            "file_count": len(all_files),
            "v2_file_index": str((dataset_dir / "file_index_v2.pth").resolve()),
            "v2_file_index_sha256": v2_index_sha,
            "v3_file_index": str((dataset_dir / "file_index_v3.pth").resolve()),
            "v3_file_index_sha256": v3_index_sha,
            "trainable_labels": list(VIEW_LABELS),
            "single_perspective_per_file": True,
        },
        "protocol": {
            "objective": "behavior_action_mc",
            "reward": "final_rank_mc",
            "parent_checkpoint": str(parent_path),
            "parent_sha256": parent_sha,
            "parent_steps": 70000,
            "target_steps": 72000,
            "preserved_adam": True,
            "fresh_scheduler": True,
            "fresh_scaler": True,
            "fresh_data_stream": True,
            "amp": bool(base["control"]["enable_amp"]),
        },
        "configs": configs,
        "commands": {
            "training_entry": "scripts/run_mortal_dqn_offline.py",
            "required_args": [
                "--initialize-from <parent>",
                "--initialize-optimizer-from <same-parent>",
                "--initial-steps 70000",
                "--target-steps 72000",
                "--device cuda",
                "--num-workers 0",
            ],
            "archive_steps": [70001, 70010, 70100, 70500, 71000, 72000],
        },
    }
    manifest_path = prep_dir / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": contract["status"],
        "output_dir": str(output),
        "file_count": len(all_files),
        "view_counts": counts,
        "training_manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
