#!/usr/bin/env python3
"""Audit historical/current Mortal loader equivalence without training.

The audit reconstructs the exact M0/D1 data stream used by the historical
continuation contract and compares it with the current project loader for all
2000 delivered batches per route and seed. It never creates a checkpoint,
updates an optimizer, or writes an authoritative artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_REPO_DEFAULT = REPO_ROOT.parent / "keqing1"
HISTORICAL_COMMIT = "90d148aedbcb905aa36615775462f8e2eece080b"
HISTORICAL_LOADER_SOURCE = "scripts/mortal/mainline_dataloader.py"
HISTORICAL_RUNNER_SOURCE = "scripts/run_mortal_dqn_offline.py"
CURRENT_LOADER_SOURCE = "training/mortal/mainline_dataloader.py"
CURRENT_RUNNER_SOURCE = "training/run_mortal_dqn_offline.py"
TRAINING_ROOT_DEFAULT = (
    HISTORICAL_REPO_DEFAULT
    / "artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07"
)
OUTPUT_ROOT_DEFAULT = REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility"
SEEDS = (20260806, 20260807, 20260808)
ROUTES = ("M0", "D1")
BATCHES = 2000
BATCH_SIZE = 512
EXPECTED_SAMPLES = BATCHES * BATCH_SIZE
CANONICAL_DTYPES = (
    torch.float32,
    torch.int64,
    torch.bool,
    torch.int64,
    torch.float64,
    torch.int64,
)
CANONICAL_NAMES = (
    "obs",
    "actions",
    "masks",
    "steps_to_done",
    "kyoku_rewards",
    "player_ranks",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_source(repo: Path, commit: str, source_path: str) -> tuple[bytes, str]:
    content = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{source_path}"],
        check=True,
        capture_output=True,
    ).stdout
    blob_sha = run_git(repo, "rev-parse", f"{commit}:{source_path}")
    return content, blob_sha


def native_path(raw: str | Path, *, anchor_root: Path = REPO_ROOT) -> Path:
    """Resolve frozen Windows paths against the local AUbuntuProject root."""

    text = str(raw)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        parts = text.replace("\\", "/").split("/")
        if "AUbuntuProject" in parts and "AUbuntuProject" in anchor_root.parts:
            root_index = anchor_root.parts.index("AUbuntuProject")
            path_index = parts.index("AUbuntuProject")
            return Path(*anchor_root.parts[: root_index + 1], *parts[path_index + 1 :]).resolve()
    return Path(text).expanduser().resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected TOML object: {config_path}")
    return value


def load_labels(config: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    labels: set[str] = set()
    files: list[dict[str, str]] = []
    for raw_path in config["dataset"]["player_names_files"]:
        path = native_path(raw_path)
        content = path.read_text(encoding="utf-8")
        labels.update(line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#"))
        files.append({"path": str(path), "sha256": sha256_file(path)})
    return sorted(labels), files


def load_file_index(config: dict[str, Any]) -> tuple[Path, list[str]]:
    index_path = native_path(config["dataset"]["file_index"])
    payload = torch.load(index_path, weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise TypeError(f"file index has no file_list: {index_path}")
    paths = [str(native_path(value)) for value in values]
    if len(paths) != 6000:
        raise ValueError(f"expected 6000 indexed files, found {len(paths)}: {index_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate indexed files: {index_path}")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"indexed file is missing: {missing[0]}")
    return index_path, paths


def contract_for_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    labels, label_files = load_labels(config)
    index_path, file_list = load_file_index(config)
    dataset = config["dataset"]
    experiment = config.get("experiment", {})
    contract = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "file_index_path": str(index_path),
        "file_index_sha256": sha256_file(index_path),
        "label_files": label_files,
        "labels": labels,
        "file_count": len(file_list),
        "file_batch_size": int(dataset["file_batch_size"]),
        "reserve_ratio": float(dataset["reserve_ratio"]),
        "num_epochs": int(dataset["num_epochs"]),
        "enable_augmentation": bool(dataset["enable_augmentation"]),
        "augmented_first": bool(dataset["augmented_first"]),
        "batch_size": int(config["control"]["batch_size"]),
        "num_workers": int(dataset["num_workers"]),
        "training_seed": int(experiment["training_seed"]),
    }
    if contract["batch_size"] != BATCH_SIZE or contract["num_workers"] != 0:
        raise ValueError(f"unexpected batch contract: {contract}")
    if contract["training_seed"] not in SEEDS:
        raise ValueError(f"unexpected training seed: {contract['training_seed']}")
    return config, {"metadata": contract, "file_list": file_list, "labels": labels}


def load_loader_class(source: bytes, source_name: str, config: dict[str, Any], module_name: str) -> type:
    text = source.decode("utf-8")
    import_line = "from config import config"
    if text.count(import_line) != 1:
        raise ValueError(f"unexpected config import count in {source_name}")
    text = text.replace(import_line, "config = _audit_config", 1)
    namespace: dict[str, Any] = {"__name__": module_name, "_audit_config": copy.deepcopy(config)}
    exec(compile(text, source_name, "exec"), namespace)  # noqa: S102 - execute the frozen source under test
    loader_class = namespace.get("FileDatasetsIter")
    if not isinstance(loader_class, type):
        raise TypeError(f"FileDatasetsIter missing from {source_name}")
    return loader_class


def make_loader(
    loader_class: type,
    config: dict[str, Any],
    file_list: list[str],
    labels: list[str],
    seed: int,
) -> Any:
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = loader_class(
        version=int(config["control"]["version"]),
        file_list=list(file_list),
        pts=config["env"]["pts"],
        file_batch_size=int(config["dataset"]["file_batch_size"]),
        reserve_ratio=float(config["dataset"]["reserve_ratio"]),
        player_names=list(labels),
        num_epochs=int(config["dataset"]["num_epochs"]),
        enable_augmentation=bool(config["dataset"]["enable_augmentation"]),
        augmented_first=bool(config["dataset"]["augmented_first"]),
    )
    return iter(
        DataLoader(
            dataset=dataset,
            batch_size=int(config["control"]["batch_size"]),
            drop_last=True,
            num_workers=0,
            # Pinning is a transport optimization for CUDA training. The audit
            # hashes canonical CPU bytes and disables it to avoid retaining a
            # second page-locked copy of the large replay batches.
            pin_memory=False,
        )
    )


def canonical_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to(dtype=dtype, device="cpu").contiguous()


def canonical_batch(batch: Any) -> tuple[torch.Tensor, ...]:
    if not isinstance(batch, (tuple, list)) or len(batch) != len(CANONICAL_NAMES):
        raise ValueError(f"unexpected batch structure: {type(batch).__name__} / {len(batch)}")
    values = tuple(canonical_tensor(value, dtype) for value, dtype in zip(batch, CANONICAL_DTYPES, strict=True))
    if int(values[0].shape[0]) != BATCH_SIZE:
        raise ValueError(f"unexpected delivered batch size: {values[0].shape}")
    return values


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.numpy().tobytes(order="C")


def hash_batch(batch: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(batch):
        digest.update(index.to_bytes(2, "big"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def hash_tensor(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor_bytes(value))
    return digest.hexdigest()


def update_stream_digest(digest: Any, batch_index: int, batch_hash: str) -> None:
    digest.update(batch_index.to_bytes(8, "big"))
    digest.update(bytes.fromhex(batch_hash))


def ordered_stream_sha256(batch_hashes: list[str]) -> str:
    digest = hashlib.sha256()
    for batch_index, batch_hash in enumerate(batch_hashes):
        update_stream_digest(digest, batch_index, batch_hash)
    return digest.hexdigest()


def first_batch_mismatch(
    historical: tuple[torch.Tensor, ...],
    current: tuple[torch.Tensor, ...],
) -> str | None:
    for name, left, right in zip(CANONICAL_NAMES, historical, current, strict=True):
        if left.dtype != right.dtype:
            return f"{name}.dtype"
        if left.shape != right.shape:
            return f"{name}.shape"
        if tensor_bytes(left) != tensor_bytes(right):
            return f"{name}.bytes"
    return None


def consume_stream(
    *,
    loader_class: type,
    config: dict[str, Any],
    file_list: list[str],
    labels: list[str],
    seed: int,
) -> dict[str, Any]:
    """Consume one complete 2000-batch stream from a fresh RNG state."""

    loader_iter = make_loader(loader_class, config, file_list, labels, seed)
    stream_digest = hashlib.sha256()
    batch_hashes: list[str] = []
    component_hashes: list[tuple[str, ...]] = []
    first_batch_contract: list[dict[str, Any]] | None = None
    for batch_index in range(BATCHES):
        try:
            batch = canonical_batch(next(loader_iter))
        except StopIteration as exc:
            raise RuntimeError(f"loader ended before batch {batch_index}") from exc
        if first_batch_contract is None:
            first_batch_contract = [
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in zip(CANONICAL_NAMES, batch, strict=True)
            ]
        batch_hash = hash_batch(batch)
        batch_hashes.append(batch_hash)
        component_hashes.append(tuple(hash_tensor(value) for value in batch))
        update_stream_digest(stream_digest, batch_index, batch_hash)
    return {
        "stream_sha256": stream_digest.hexdigest(),
        "batch_hashes": batch_hashes,
        "component_hashes": component_hashes,
        "first_batch_contract": first_batch_contract,
    }


def run_stream_pair(
    *,
    historical_loader: type,
    current_loader: type,
    config: dict[str, Any],
    file_list: list[str],
    labels: list[str],
    seed: int,
) -> dict[str, Any]:
    # Run each implementation from an independently reset RNG state. Running
    # the iterators alternately would let one implementation advance the global
    # random state seen by the other implementation.
    historical = consume_stream(
        loader_class=historical_loader,
        config=config,
        file_list=file_list,
        labels=labels,
        seed=seed,
    )
    current = consume_stream(
        loader_class=current_loader,
        config=config,
        file_list=file_list,
        labels=labels,
        seed=seed,
    )
    first_mismatch_batch: int | None = None
    first_mismatch_tensor: str | None = None
    for batch_index, (historical_hash, current_hash) in enumerate(
        zip(historical["batch_hashes"], current["batch_hashes"], strict=True)
    ):
        if historical_hash != current_hash:
            first_mismatch_batch = batch_index
            historical_components = historical["component_hashes"][batch_index]
            current_components = current["component_hashes"][batch_index]
            first_mismatch_tensor = next(
                (
                    name
                    for name, historical_component, current_component in zip(
                        CANONICAL_NAMES,
                        historical_components,
                        current_components,
                        strict=True,
                    )
                    if historical_component != current_component
                ),
                "unknown",
            )
            break

    return {
        "batches_compared": BATCHES,
        "samples_compared": EXPECTED_SAMPLES,
        "historical_stream_sha256": historical["stream_sha256"],
        "current_stream_sha256": current["stream_sha256"],
        "exact_match": first_mismatch_batch is None,
        "first_mismatch_batch": first_mismatch_batch,
        "first_mismatch_tensor": first_mismatch_tensor,
        "first_batch_contract": historical["first_batch_contract"],
        "historical_batch_hashes": historical["batch_hashes"],
        "current_batch_hashes": current["batch_hashes"],
    }


def compare_stream_hashes(
    *,
    historical_hashes: list[str],
    current_hashes: list[str],
    historical_components: list[list[str]] | None = None,
    current_components: list[list[str]] | None = None,
    first_batch_contract: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(historical_hashes) != BATCHES or len(current_hashes) != BATCHES:
        raise ValueError(
            f"expected {BATCHES} hashes per implementation, "
            f"got {len(historical_hashes)} and {len(current_hashes)}"
        )
    first_mismatch_batch = next(
        (
            index
            for index, (historical_hash, current_hash) in enumerate(
                zip(historical_hashes, current_hashes, strict=True)
            )
            if historical_hash != current_hash
        ),
        None,
    )
    first_mismatch_tensor: str | None = None
    if first_mismatch_batch is not None and historical_components and current_components:
        historical_batch = historical_components[first_mismatch_batch]
        current_batch = current_components[first_mismatch_batch]
        first_mismatch_tensor = next(
            (
                name
                for name, historical_component, current_component in zip(
                    CANONICAL_NAMES,
                    historical_batch,
                    current_batch,
                    strict=True,
                )
                if historical_component != current_component
            ),
            "unknown",
        )
    return {
        "batches_compared": BATCHES,
        "samples_compared": EXPECTED_SAMPLES,
        "historical_stream_sha256": ordered_stream_sha256(historical_hashes),
        "current_stream_sha256": ordered_stream_sha256(current_hashes),
        "exact_match": first_mismatch_batch is None,
        "first_mismatch_batch": first_mismatch_batch,
        "first_mismatch_tensor": first_mismatch_tensor,
        "first_batch_contract": first_batch_contract,
        "historical_batch_hashes": historical_hashes,
        "current_batch_hashes": current_hashes,
    }


def source_record(repo: Path, commit: str, source_path: str) -> dict[str, str]:
    content, blob_oid = git_source(repo, commit, source_path)
    return {
        "repository": str(repo.resolve()),
        "commit": commit,
        "source_path": source_path,
        "git_blob_oid": blob_oid,
        "content_sha256": sha256_bytes(content),
    }


def current_source_record(repo: Path, commit: str, source_path: str) -> dict[str, str]:
    record = source_record(repo, commit, source_path)
    worktree_path = repo / source_path
    worktree_sha = sha256_file(worktree_path)
    if worktree_sha != record["content_sha256"]:
        raise RuntimeError(f"current source differs from committed HEAD: {worktree_path}")
    record["worktree_path"] = str(worktree_path.resolve())
    record["worktree_sha256"] = worktree_sha
    return record


def config_path_for(training_root: Path, route: str, seed: int) -> Path:
    route_dir = "M0_control" if route == "M0" else "D1_variant"
    return training_root / "training_prep_2026_07" / route_dir / f"seed_{seed}" / "config.toml"


def load_completed_stream(path: Path, *, route: str, seed: int) -> dict[str, Any]:
    """Reuse only a complete hash file from an interrupted audit."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("route") != route or int(value.get("seed", -1)) != seed:
        raise ValueError(f"completed hash file identity mismatch: {path}")
    historical = value.get("historical")
    current = value.get("current")
    if (
        not isinstance(historical, list)
        or not isinstance(current, list)
        or len(historical) != BATCHES
        or len(current) != BATCHES
    ):
        raise ValueError(f"incomplete batch hash file: {path}")
    stream = compare_stream_hashes(
        historical_hashes=[str(value) for value in historical],
        current_hashes=[str(value) for value in current],
    )
    stream["resumed_from_hash_file"] = True
    return stream


def stream_artifact_path(output_root: Path, route: str, seed: int, implementation: str) -> Path:
    return output_root / f"{route.lower()}_{seed}_{implementation}_stream.json"


def load_stream_artifact(
    path: Path,
    *,
    route: str,
    seed: int,
    implementation: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("route") != route
        or int(value.get("seed", -1)) != seed
        or value.get("implementation") != implementation
    ):
        raise ValueError(f"stream artifact identity mismatch: {path}")
    batch_hashes = value.get("batch_hashes")
    if not isinstance(batch_hashes, list) or len(batch_hashes) != BATCHES:
        raise ValueError(f"incomplete stream artifact: {path}")
    batch_hashes = [str(batch_hash) for batch_hash in batch_hashes]
    stream_sha256 = ordered_stream_sha256(batch_hashes)
    if value.get("stream_sha256") != stream_sha256:
        raise ValueError(f"stream digest mismatch: {path}")
    components = value.get("component_hashes")
    if components is not None and (
        not isinstance(components, list)
        or len(components) != BATCHES
        or any(not isinstance(batch, list) or len(batch) != len(CANONICAL_NAMES) for batch in components)
    ):
        raise ValueError(f"invalid component hashes: {path}")
    return {
        "implementation": implementation,
        "stream_sha256": stream_sha256,
        "batch_hashes": batch_hashes,
        "component_hashes": components,
        "first_batch_contract": value.get("first_batch_contract"),
        "path": str(path.resolve()),
    }


def write_batch_hash_pair(output_root: Path, route: str, seed: int, stream: dict[str, Any]) -> Path:
    path = output_root / f"{route.lower()}_{seed}_batch_hashes.json"
    path.write_text(
        json.dumps(
            {
                "route": route,
                "seed": seed,
                "historical": stream.pop("historical_batch_hashes"),
                "current": stream.pop("current_batch_hashes"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def combine_stream_artifacts(
    *,
    output_root: Path,
    route: str,
    seed: int,
) -> tuple[dict[str, Any], Path]:
    historical_path = stream_artifact_path(output_root, route, seed, "historical")
    current_path = stream_artifact_path(output_root, route, seed, "current")
    historical = load_stream_artifact(
        historical_path,
        route=route,
        seed=seed,
        implementation="historical",
    )
    current = load_stream_artifact(
        current_path,
        route=route,
        seed=seed,
        implementation="current",
    )
    stream = compare_stream_hashes(
        historical_hashes=historical["batch_hashes"],
        current_hashes=current["batch_hashes"],
        historical_components=historical["component_hashes"],
        current_components=current["component_hashes"],
        first_batch_contract=historical["first_batch_contract"],
    )
    stream["separate_implementation_artifacts"] = {
        "historical": historical["path"],
        "current": current["path"],
    }
    pair_path = write_batch_hash_pair(output_root, route, seed, stream)
    return stream, pair_path


def run_implementation_stream(
    *,
    implementation: str,
    route: str,
    seed: int,
    historical_repo: Path,
    historical_commit: str,
    training_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if implementation not in {"historical", "current"}:
        raise ValueError(f"unsupported implementation: {implementation}")
    if route not in ROUTES or seed not in SEEDS:
        raise ValueError(f"unsupported route/seed: {route}/{seed}")

    current_commit = run_git(REPO_ROOT, "rev-parse", "HEAD")
    config_path = config_path_for(training_root, route, seed).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing frozen config: {config_path}")
    config, loaded = contract_for_config(config_path)
    if loaded["metadata"]["training_seed"] != seed:
        raise ValueError(f"config seed mismatch: {config_path}")

    if implementation == "historical":
        source, _ = git_source(historical_repo, historical_commit, HISTORICAL_LOADER_SOURCE)
        source_info = source_record(historical_repo, historical_commit, HISTORICAL_LOADER_SOURCE)
        source_name = f"{historical_repo}@{historical_commit}:{HISTORICAL_LOADER_SOURCE}"
    else:
        source, _ = git_source(REPO_ROOT, current_commit, CURRENT_LOADER_SOURCE)
        source_info = current_source_record(REPO_ROOT, current_commit, CURRENT_LOADER_SOURCE)
        source_name = f"{REPO_ROOT}@{current_commit}:{CURRENT_LOADER_SOURCE}"
    loader_class = load_loader_class(
        source,
        source_name,
        config,
        f"c1_{implementation}_loader_{route}_{seed}",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = stream_artifact_path(output_root, route, seed, implementation)
    if output_path.is_file():
        return load_stream_artifact(
            output_path,
            route=route,
            seed=seed,
            implementation=implementation,
        )

    stream = consume_stream(
        loader_class=loader_class,
        config=config,
        file_list=loaded["file_list"],
        labels=loaded["labels"],
        seed=seed,
    )
    payload = {
        "schema": "keqing.mortal.c1_loader_stream.v1",
        "route": route,
        "seed": seed,
        "implementation": implementation,
        "batches": BATCHES,
        "samples": EXPECTED_SAMPLES,
        "stream_sha256": stream["stream_sha256"],
        "batch_hashes": stream["batch_hashes"],
        "component_hashes": [list(value) for value in stream["component_hashes"]],
        "first_batch_contract": stream["first_batch_contract"],
        "contract": loaded["metadata"],
        "source": source_info,
        "training_started": False,
        "generation_started": False,
        "evaluation_started": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return load_stream_artifact(
        output_path,
        route=route,
        seed=seed,
        implementation=implementation,
    )


def run_audit(
    *,
    historical_repo: Path,
    historical_commit: str,
    training_root: Path,
    output_root: Path,
    routes: tuple[str, ...] = ROUTES,
    seeds: tuple[int, ...] = SEEDS,
    execute_missing: bool = True,
) -> dict[str, Any]:
    current_commit = run_git(REPO_ROOT, "rev-parse", "HEAD")

    sources = {
        "historical_loader": source_record(historical_repo, historical_commit, HISTORICAL_LOADER_SOURCE),
        "historical_runner": source_record(historical_repo, historical_commit, HISTORICAL_RUNNER_SOURCE),
        "current_loader": current_source_record(REPO_ROOT, current_commit, CURRENT_LOADER_SOURCE),
        "current_runner": current_source_record(REPO_ROOT, current_commit, CURRENT_RUNNER_SOURCE),
    }
    runs: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for route in routes:
        for seed in seeds:
            config_path = config_path_for(training_root, route, seed).resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"missing frozen config: {config_path}")
            config, loaded = contract_for_config(config_path)
            if loaded["metadata"]["training_seed"] != seed:
                raise ValueError(f"config seed mismatch: {config_path}")

            batch_hash_path = output_root / f"{route.lower()}_{seed}_batch_hashes.json"
            if batch_hash_path.is_file():
                stream = load_completed_stream(batch_hash_path, route=route, seed=seed)
            elif (
                stream_artifact_path(output_root, route, seed, "historical").is_file()
                and stream_artifact_path(output_root, route, seed, "current").is_file()
            ):
                stream, batch_hash_path = combine_stream_artifacts(
                    output_root=output_root,
                    route=route,
                    seed=seed,
                )
            elif not execute_missing:
                raise FileNotFoundError(
                    "missing complete pair or separate implementation streams for "
                    f"{route}/{seed} under {output_root}"
                )
            else:
                historical_loader_source, _ = git_source(
                    historical_repo,
                    historical_commit,
                    HISTORICAL_LOADER_SOURCE,
                )
                current_loader_source, _ = git_source(REPO_ROOT, current_commit, CURRENT_LOADER_SOURCE)
                historical_loader = load_loader_class(
                    historical_loader_source,
                    f"{historical_repo}@{historical_commit}:{HISTORICAL_LOADER_SOURCE}",
                    config,
                    f"c1_historical_loader_{route}_{seed}",
                )
                current_loader = load_loader_class(
                    current_loader_source,
                    f"{REPO_ROOT}@{current_commit}:{CURRENT_LOADER_SOURCE}",
                    config,
                    f"c1_current_loader_{route}_{seed}",
                )
                stream = run_stream_pair(
                    historical_loader=historical_loader,
                    current_loader=current_loader,
                    config=config,
                    file_list=loaded["file_list"],
                    labels=loaded["labels"],
                    seed=seed,
                )
                batch_hash_path = write_batch_hash_pair(
                    output_root,
                    route,
                    seed,
                    stream,
                )
            stream.pop("historical_batch_hashes", None)
            stream.pop("current_batch_hashes", None)
            runs.append(
                {
                    "route": route,
                    "seed": seed,
                    "contract": loaded["metadata"],
                    "stream": stream,
                    "batch_hashes_path": str(batch_hash_path.resolve()),
                }
            )

    expected_run_count = len(routes) * len(seeds)
    complete = len(runs) == expected_run_count
    exact = complete and all(bool(run["stream"]["exact_match"]) for run in runs)
    report = {
        "schema": "keqing.mortal.c1_loader_compatibility.v1",
        "status": "PASS" if exact else "INCOMPLETE" if not complete else "FAIL",
        "gate": {
            "historical_current_training_reuse": "PASS" if exact else "PENDING" if not complete else "FAIL",
            "resolved_training_count": 6 if exact else 12 if complete else None,
            "batches_per_stream": BATCHES,
            "samples_per_stream": EXPECTED_SAMPLES,
            "streams_compared": len(runs),
            "streams_required": expected_run_count,
        },
        "training_started": False,
        "generation_started": False,
        "evaluation_started": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "current_commit": current_commit,
        "historical_commit": historical_commit,
        "historical_repository": str(historical_repo.resolve()),
        "training_root": str(training_root.resolve()),
        "output_root": str(output_root.resolve()),
        "execution_mode": "finalize_only" if not execute_missing else "audit_or_resume",
        "sources": sources,
        "runs": runs,
    }
    report_path = output_root / "loader_compatibility.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-repo", type=Path, default=HISTORICAL_REPO_DEFAULT)
    parser.add_argument("--historical-commit", default=HISTORICAL_COMMIT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--batches", type=int, default=BATCHES, help="must remain exactly 2000")
    parser.add_argument("--route", choices=ROUTES, default=None, help="resume one route only")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None, help="resume one seed only")
    parser.add_argument(
        "--implementation",
        choices=("historical", "current"),
        default=None,
        help="run one implementation in an isolated child process",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="compare only complete pair/separate stream artifacts; never execute loaders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batches) != BATCHES:
        raise ValueError("loader compatibility audit requires exactly 2000 delivered batches")
    if args.implementation is not None:
        if args.finalize or args.route is None or args.seed is None:
            raise ValueError("--implementation requires --route and --seed and cannot be combined with --finalize")
        stream = run_implementation_stream(
            implementation=str(args.implementation),
            route=str(args.route),
            seed=int(args.seed),
            historical_repo=args.historical_repo.resolve(),
            historical_commit=str(args.historical_commit),
            training_root=args.training_root.resolve(),
            output_root=args.output_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "implementation": args.implementation,
                    "route": args.route,
                    "seed": args.seed,
                    "batches": BATCHES,
                    "samples": EXPECTED_SAMPLES,
                    "stream_sha256": stream["stream_sha256"],
                    "artifact": stream["path"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if args.finalize and (args.route is not None or args.seed is not None):
        raise ValueError("--finalize requires the complete M0/D1 x three-seed scope")
    routes = (str(args.route),) if args.route is not None else ROUTES
    seeds = (int(args.seed),) if args.seed is not None else SEEDS
    report = run_audit(
        historical_repo=args.historical_repo.resolve(),
        historical_commit=str(args.historical_commit),
        training_root=args.training_root.resolve(),
        output_root=args.output_root.resolve(),
        routes=routes,
        seeds=seeds,
        execute_missing=not args.finalize,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate": report["gate"],
                "report": str(args.output_root.resolve() / "loader_compatibility.json"),
                "runs": [
                    {
                        "route": run["route"],
                        "seed": run["seed"],
                        "exact_match": run["stream"]["exact_match"],
                        "historical_stream_sha256": run["stream"]["historical_stream_sha256"],
                        "current_stream_sha256": run["stream"]["current_stream_sha256"],
                    }
                    for run in report["runs"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
