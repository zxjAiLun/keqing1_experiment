#!/usr/bin/env python3
"""Freeze the M0/D1 training comparison without starting training.

The preparation pass creates deterministic file indexes, content manifests,
matched configs, and a machine-readable contract for the three M0/D1 seed
pairs.  It intentionally does not load CUDA, train a model, or modify source
replay logs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch

try:
    import toml
except ImportError as exc:  # pragma: no cover - environment failure
    raise SystemExit("prepare script requires the project's toml dependency") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = REPO_ROOT / "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
DEFAULT_BASE_CONFIG = REPO_ROOT / (
    "artifacts/experiments/model_pool_2026_07/data_route_ab_2026_07/"
    "M0_mixed/seed_20260731/config.toml"
)
DEFAULT_M0_INDEX = REPO_ROOT / (
    "artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/file_index.pth"
)
DEFAULT_D1_ROOT = REPO_ROOT / (
    "artifacts/experiments/model_pool_2026_07/"
    "D1_project_owned_population_2026_07"
)
DEFAULT_OUTPUT = DEFAULT_D1_ROOT / "training_prep_2026_07"
SEED_VALUES = (20260806, 20260807, 20260808)
SEED_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-d])?\.json\.gz$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_event_hash(path: Path) -> str:
    """Hash one hanchan while ignoring generated names/meta fields."""

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


def read_start(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        line = next(handle, "")
    event = json.loads(line)
    if event.get("type") != "start_game":
        raise ValueError(f"first event is not start_game: {path}")
    return event


def resolve_file(value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_file_index(path: Path) -> list[Path]:
    payload = torch.load(path.resolve(), weights_only=False, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("file_list")
    if not isinstance(payload, list):
        raise ValueError(f"file index has no file_list: {path}")
    files = [resolve_file(value) for value in payload]
    if len(files) != len(set(files)):
        raise ValueError(f"file index contains duplicate paths: {path}")
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"file index contains missing files, first={missing[0]}")
    return files


def d1_files(data_root: Path) -> list[Path]:
    files = sorted(data_root.glob("data/shard_*/logs/*.json.gz"))
    if len(files) != 6000:
        raise ValueError(f"expected 6000 D1 logs, found {len(files)}")
    return [path.resolve() for path in files]


def write_file_index(path: Path, files: list[Path]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"file_list": [str(file) for file in files]}, path)
    return sha256_file(path)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def content_manifest(files: list[Path], output_dir: Path, name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        start = read_start(path)
        seed = start.get("seed")
        if not isinstance(seed, list) or len(seed) != 2:
            match = SEED_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"cannot determine seed/key: {path}")
            seed = [int(match["seed"]), int(match["key"])]
        if len(seed) != 2:
            raise ValueError(f"invalid seed/key: {path}")
        rows.append(
            {
                "relative_path": relative_path(path),
                "compressed_sha256": sha256_file(path),
                "canonical_hanchan_sha256": canonical_event_hash(path),
                "seed": int(seed[0]),
                "seed_key": int(seed[1]),
            }
        )
        if index % 500 == 0:
            print(f"[prepare] {name} manifest {index}/{len(files)}", flush=True)

    rows.sort(key=lambda row: row["relative_path"])
    seeds = [(row["seed"], row["seed_key"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed/key in {name} content manifest")
    canonical = [row["canonical_hanchan_sha256"] for row in rows]
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"duplicate canonical hanchan in {name} content manifest")

    payload = {
        "schema": "keqing.mortal.content_manifest.v1",
        "source": name,
        "file_count": len(rows),
        "rows": rows,
    }
    json_bytes = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    json_path = output_dir / f"content_manifest_{name}.json"
    json_path.write_bytes(json_bytes)
    tsv_path = output_dir / f"content_manifest_{name}.tsv"
    tsv_lines = ["relative_path\tcompressed_sha256\tcanonical_hanchan_sha256\tseed\tseed_key"]
    tsv_lines.extend(
        "\t".join(
            [
                row["relative_path"],
                row["compressed_sha256"],
                row["canonical_hanchan_sha256"],
                str(row["seed"]),
                str(row["seed_key"]),
            ]
        )
        for row in rows
    )
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    return {
        "path": str(json_path.resolve()),
        "tsv_path": str(tsv_path.resolve()),
        "sha256": sha256_bytes(json_bytes),
        "tsv_sha256": sha256_file(tsv_path),
        "file_count": len(rows),
        "seed_min": min(row["seed"] for row in rows),
        "seed_max": max(row["seed"] for row in rows),
        "seed_key_values": sorted({row["seed_key"] for row in rows}),
    }


def tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(prefix.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, dict):
            for key in sorted(item, key=lambda value: repr(value)):
                visit(key, prefix + ".key")
                visit(item[key], prefix + f"[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, prefix + f"[{index}]")
            return
        digest.update(prefix.encode())
        digest.update(repr(item).encode())

    visit(value, "root")
    return digest.hexdigest()


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


def base_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        config = toml.load(config_path)
    config.setdefault("objective", {})["mode"] = "behavior_action_mc"
    return config


def build_config(
    base: dict[str, Any],
    *,
    route: str,
    seed: int,
    output_dir: Path,
    file_index: Path,
    data_glob: str,
    label_file: Path,
    label: str,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    control = config["control"]
    control["state_file"] = str((output_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((output_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((output_dir / "tb_mortal").resolve())
    dataset = config["dataset"]
    dataset["globs"] = [data_glob]
    dataset["file_index"] = str(file_index.resolve())
    dataset["player_names_files"] = [str(label_file.resolve())]
    dataset["num_workers"] = 0
    dataset["num_epochs"] = 3
    config["objective"] = {"mode": "behavior_action_mc"}
    config["experiment"] = {
        "route": route,
        "trainable_label": label,
        "training_seed": seed,
        "parent_steps": 70000,
        "reward_mode": "final_rank_mc",
    }
    return config


def write_label(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{label}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--m0-index", type=Path, default=DEFAULT_M0_INDEX)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_config_path = args.base_config.resolve()
    m0_index_path = args.m0_index.resolve()
    d1_root = args.d1_root.resolve()
    parent_path = args.parent.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if git_info()["branch"] != "codex/mortal-training-next":
        raise SystemExit("preparation must run on codex/mortal-training-next")
    if not base_config_path.is_file() or not m0_index_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("base config, M0 index, and parent checkpoint are required")

    population_audit_path = d1_root / "dataset_audit.json"
    generation_manifest_path = d1_root / "manifest.json"
    if not population_audit_path.is_file() or not generation_manifest_path.is_file():
        raise FileNotFoundError("D1 population audit and generation manifest are required")
    population_audit = json.loads(population_audit_path.read_text(encoding="utf-8"))
    if not population_audit.get("summary", {}).get("passed"):
        raise SystemExit("D1 population audit is not passed")

    m0_files = read_file_index(m0_index_path)
    d1_files_list = d1_files(d1_root)
    if len(m0_files) != 6000:
        raise ValueError(f"expected 6000 M0 files, found {len(m0_files)}")

    m0_manifest = content_manifest(m0_files, output_dir, "m0")
    d1_manifest = content_manifest(d1_files_list, output_dir, "d1")
    m0_out_index = output_dir / "file_index_m0.pth"
    d1_out_index = output_dir / "file_index_d1.pth"
    m0_index_sha = write_file_index(m0_out_index, m0_files)
    d1_index_sha = write_file_index(d1_out_index, d1_files_list)

    base = base_config(base_config_path)
    labels_dir = output_dir / "labels"
    m0_label_file = labels_dir / "m0_train_labels.txt"
    d1_label_file = labels_dir / "d1_train_labels.txt"
    write_label(m0_label_file, "ext_mortal")
    write_label(d1_label_file, "K0_70k")
    m0_glob = str((REPO_ROOT / "artifacts/experiments/model_pool_2026_07/V2_data/*/logs/**/*.json.gz").resolve())
    d1_glob = str((d1_root / "data/shard_*/logs/*.json.gz").resolve())
    run_configs: list[dict[str, Any]] = []
    for seed in SEED_VALUES:
        for route, file_index, data_glob, label_file, label in (
            ("M0_control", m0_out_index, m0_glob, m0_label_file, "ext_mortal"),
            ("D1_variant", d1_out_index, d1_glob, d1_label_file, "K0_70k"),
        ):
            run_dir = output_dir / route / f"seed_{seed}"
            config = build_config(
                base,
                route=route,
                seed=seed,
                output_dir=run_dir,
                file_index=file_index,
                data_glob=data_glob,
                label_file=label_file,
                label=label,
            )
            config_path = run_dir / "config.toml"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(toml.dumps(config), encoding="utf-8")
            run_configs.append(
                {
                    "route": route,
                    "seed": seed,
                    "label": label,
                    "config": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "run_dir": str(run_dir.resolve()),
                    "file_index_sha256": m0_index_sha if route == "M0_control" else d1_index_sha,
                }
            )

    parent_state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if int(parent_state.get("steps", -1)) != 70000:
        raise ValueError(f"parent checkpoint must be at step 70000, got {parent_state.get('steps')}")
    parent_sha = sha256_file(parent_path)
    parent_digest = {
        "checkpoint_sha256": parent_sha,
        "mortal_sha256": tensor_digest(parent_state["mortal"]),
        "current_dqn_sha256": tensor_digest(parent_state["current_dqn"]),
        "aux_net_sha256": tensor_digest(parent_state["aux_net"]),
        "optimizer_sha256": tensor_digest(parent_state["optimizer"]),
        "optimizer_state_count": len(parent_state["optimizer"]["state"]),
        "steps": int(parent_state["steps"]),
    }
    del parent_state

    git = git_info()
    contract = {
        "schema": "keqing.mortal.d1_training_prep.v1",
        "experiment_id": "D1_M0_training_view_ab_2026_07",
        "status": "prepared_not_started",
        "git": git,
        "seeds": list(SEED_VALUES),
        "pairing": {
            "model_seed_equals_data_seed": True,
            "fresh_scheduler": True,
            "fresh_scaler": True,
            "fresh_data_stream": True,
            "initial_steps": 70000,
            "target_steps": 72000,
        },
        "protocol": {
            "objective": "behavior_action_mc",
            "reward": "final_rank_mc",
            "parent_checkpoint": str(parent_path),
            "parent_sha256": parent_sha,
            "parent_tensor_digest": parent_digest,
            "architecture": base["resnet"],
            "cql": base["cql"],
            "aux": base["aux"],
            "env": base["env"],
            "optim": base["optim"],
            "amp": bool(base["control"]["enable_amp"]),
            "batch_size": int(base["control"]["batch_size"]),
        },
        "datasets": {
            "M0_control": {
                "trainable_label": "ext_mortal",
                "file_index": str(m0_out_index),
                "file_index_sha256": m0_index_sha,
                "content_manifest_sha256": m0_manifest["sha256"],
                "content_manifest": m0_manifest,
            },
            "D1_variant": {
                "trainable_label": "K0_70k",
                "file_index": str(d1_out_index),
                "file_index_sha256": d1_index_sha,
                "dataset_audit": str(population_audit_path),
                "dataset_audit_sha256": sha256_file(population_audit_path),
                "generation_manifest": str(generation_manifest_path),
                "generation_manifest_sha256": sha256_file(generation_manifest_path),
                "content_manifest_sha256": d1_manifest["sha256"],
                "content_manifest": d1_manifest,
            },
        },
        "configs": run_configs,
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
    contract_path = output_dir / "training_manifest.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": contract["status"],
        "output_dir": str(output_dir),
        "m0_files": len(m0_files),
        "d1_files": len(d1_files_list),
        "m0_content_manifest_sha256": m0_manifest["sha256"],
        "d1_content_manifest_sha256": d1_manifest["sha256"],
        "training_manifest": str(contract_path),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
