#!/usr/bin/env python3
"""Prepare the frozen, not-authorized C1 CQL_OFF training contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
C1_ID = "C1_corpus_cql_interaction_2026_08"
PREREG_COMMIT = "65a8b220d361fc8285c3f9494349307bb2b86913"
PREREG_SHA256 = "e02ba3a667a7f2538a97e180fee8646ed819350c779db934b1852d97d5f33573"
REGISTRY_SHA256 = "65c04d0bf9617648cc3d0cff685313caca39437ae3673485133491d4e6c091c0"
LOADER_SHA256 = "aff26cee9fa32781b426b3f02302df0e0625baee9bdd5be9c000fce2acbff46c"
K0_PARENT_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"
SEEDS = (20260806, 20260807, 20260808)
ROUTES = ("M0_CQL_OFF", "D1_CQL_OFF")
START_STEP = 70000
TARGET_STEP = 72000
OPTIMIZER_STEPS = 2000
ARCHIVE_STEPS = (70001, 70010, 70100, 70500, 71000, 72000)
HISTORICAL_MORTAL_REVISION = "0cff2b52982be5b1163aa9a62fb01f03ce91e0d2"
MORTAL_REPO_ROOT = REPO_ROOT / "third_party/Mortal"
MORTAL_SOURCE_FILES = (
    "mortal/config.py",
    "mortal/model.py",
    "mortal/lr_scheduler.py",
)
NATIVE_RUNTIME_MODULE = "riichi"

SOURCE_CONFIG_SHA256 = {
    ("M0_CQL_OFF", 20260806): "89d8a9947402d12aaa879c561f1729f2c671358ac8639e723209308834f05d93",
    ("M0_CQL_OFF", 20260807): "e97b4b5dc09d32cffe8068007e0cdbe012cdb3b68cf9955bafe377f76febfe55",
    ("M0_CQL_OFF", 20260808): "0302c11ad71f19fad38e0e6e7db696898d338cfc66475e9e3a9e58a11cd4b694",
    ("D1_CQL_OFF", 20260806): "08d35c0aaf2a2db4a5039cb5242cf02a6d53a1bd5c6a8c042d2f29eda8f0a0dd",
    ("D1_CQL_OFF", 20260807): "655339eebbacf8e9fd85de820c612515c9840025b4dbc1c7dfa4bb3ec03841b3",
    ("D1_CQL_OFF", 20260808): "4baf422cad7711f85f85b456c4c4329c0fd4d874df09b51506b0e03821b12d2e",
}
SOURCE_FILE_INDEX_SHA256 = {
    "M0": "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2",
    "D1": "e357bdb00d5bf3cd7e0afa6960ee43af656421cfed381a3320f6b83ac56087f0",
}
LOADER_STREAM_SHA256 = {
    ("M0", 20260806): "c111c3b1fe223bfc42a52507226963b093c17be792e9197ef0d3686f5b794b3f",
    ("M0", 20260807): "6d418d89a23509293d69cf359de91e99f23b210dcfc6570ce2b4ac8d95ffb2a0",
    ("M0", 20260808): "6f35793b3d18f9bb5325ce57232b0ad02ee23e3b3cb68927c3aaef6737d604ed",
    ("D1", 20260806): "f7e8c46436b069206583b0c5151c3a4be7c6019ade054a100d0950990dea823f",
    ("D1", 20260807): "1b68076ec2683d60af28a1aa9b8724d049f568e0c97738ae3c47f1cfca475d35",
    ("D1", 20260808): "74bb248b0bf17d192b1ebad986e7ff7f56e387ddf3e605bd0db765cc552b05ce",
}
IMPLEMENTATION_SOURCES = (
    "training/mortal/objective.py",
    "training/run_mortal_dqn_offline.py",
    "training/mortal/mainline_dataloader.py",
)
OUTPUT_PATHS = {
    "control.state_file",
    "control.best_state_file",
    "control.tensorboard_dir",
}
SCIENTIFIC_PATHS = {"cql.min_q_weight"}
EXECUTION_PATH_ROOTS = {
    "dataset.file_index",
    "dataset.globs",
    "dataset.player_names_files",
    "dataset.player_names_by_file",
}
RUNTIME_INDEX_FILENAME = {
    "M0_CQL_OFF": "M0_file_index_linux.pth",
    "D1_CQL_OFF": "D1_file_index_linux.pth",
}
ALLOWED_UNTRACKED = {"1.md"}
SOURCE_ROUTE_DIR = {"M0_CQL_OFF": "M0_control", "D1_CQL_OFF": "D1_variant"}
WINDOWS_ROOT_MAP = (
    (
        "e:/aubuntuproject/project/keqing1",
        REPO_ROOT.parent / "keqing1",
    ),
    (
        "e:/aubuntuproject/keqing-data",
        REPO_ROOT.parents[1] / "keqing-data",
    ),
)
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/"
    "D1_project_owned_population_2026_07/training_prep_2026_07"
)
DEFAULT_PARENT = (
    REPO_ROOT.parents[1]
    / "keqing-data/mortal/authoritative/D3_top2_discard_v1_2026_08/"
    "models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/training_implementation_2026_08"


class ContractError(RuntimeError):
    """Raised when a frozen C1 implementation contract fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_blob_oid(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_info() -> dict[str, Any]:
    status_lines = run_git("status", "--porcelain", "--untracked-files=all").splitlines()
    tracked_changes = run_git("diff", "--name-only").splitlines()
    tracked_changes += run_git("diff", "--cached", "--name-only").splitlines()
    untracked = [line[3:] for line in status_lines if line.startswith("?? ")]
    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
        "tracked_changes": sorted(set(tracked_changes)),
        "untracked": sorted(untracked),
        "status": status_lines,
        "tracked_clean": not tracked_changes,
    }


def _mortal_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=MORTAL_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_blob_oid_bytes(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def runtime_provenance() -> dict[str, Any]:
    """Capture the exact Python, torch, CUDA, and loaded native runtime."""

    try:
        native_module = importlib.import_module(NATIVE_RUNTIME_MODULE)
    except Exception as exc:  # pragma: no cover - the failure is environment-specific
        raise ContractError(f"unable to import native runtime {NATIVE_RUNTIME_MODULE}: {exc}") from exc
    native_file = getattr(native_module, "__file__", None)
    if not native_file:
        raise ContractError(f"native runtime {NATIVE_RUNTIME_MODULE} has no __file__")
    native_path = Path(native_file).resolve()
    if not native_path.is_file():
        raise ContractError(f"native runtime path is missing: {native_path}")
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "native_module": NATIVE_RUNTIME_MODULE,
        "native_path": str(native_path),
        "native_sha256": sha256_file(native_path),
        "sys_executable": str(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_name": str(cuda_device_name) if cuda_device_name is not None else None,
    }


def validate_runtime_provenance(expected: dict[str, Any]) -> dict[str, Any]:
    actual = runtime_provenance()
    if actual != expected:
        raise ContractError(f"runtime provenance mismatch: expected={expected}, actual={actual}")
    if actual.get("cuda_available") is not True or not actual.get("cuda_device_name"):
        raise ContractError("C1 training runtime does not have the frozen CUDA device available")
    return actual


def mortal_source_provenance() -> dict[str, Any]:
    """Freeze current Mortal revision/blob/content provenance against the historical revision."""

    if not MORTAL_REPO_ROOT.is_dir():
        raise ContractError(f"Mortal repository is missing: {MORTAL_REPO_ROOT}")
    current_revision = _mortal_git("rev-parse", "HEAD")
    source_records: list[dict[str, Any]] = []
    for relative_path in MORTAL_SOURCE_FILES:
        current_path = MORTAL_REPO_ROOT / relative_path
        if not current_path.is_file():
            raise ContractError(f"Mortal source is missing: {current_path}")
        current_content = current_path.read_bytes()
        historical_content = subprocess.run(
            ["git", "show", f"{HISTORICAL_MORTAL_REVISION}:{relative_path}"],
            cwd=MORTAL_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        current_sha256 = sha256_bytes(current_content)
        historical_sha256 = sha256_bytes(historical_content)
        current_blob_oid = _git_blob_oid_bytes(current_content)
        historical_blob_oid = _git_blob_oid_bytes(historical_content)
        if current_sha256 != historical_sha256 or current_blob_oid != historical_blob_oid:
            raise ContractError(
                f"Mortal source content mismatch against {HISTORICAL_MORTAL_REVISION}: {relative_path}"
            )
        source_records.append(
            {
                "path": relative_path,
                "current_sha256": current_sha256,
                "current_git_blob_oid": current_blob_oid,
                "historical_sha256": historical_sha256,
                "historical_git_blob_oid": historical_blob_oid,
                "content_matches_historical": True,
            }
        )
    return {
        "repo": str(MORTAL_REPO_ROOT.resolve()),
        "current_mortal_revision": current_revision,
        "historical_mortal_revision": HISTORICAL_MORTAL_REVISION,
        "content_matches_historical": True,
        "sources": source_records,
    }


def validate_git_scope(info: dict[str, Any]) -> None:
    unexpected_untracked = sorted(set(info.get("untracked", [])) - ALLOWED_UNTRACKED)
    if info.get("branch") != "main" or info.get("tracked_changes"):
        raise ContractError(f"C1 requires main with a clean tracked tree: {info}")
    if unexpected_untracked:
        raise ContractError(f"unexpected untracked files are not allowed: {unexpected_untracked}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def map_external_value(value: str) -> str:
    """Map frozen Windows path strings to the local read-only data roots."""

    normalized = str(value).replace("\\", "/")
    lowered = normalized.lower()
    for prefix, local_root in WINDOWS_ROOT_MAP:
        if lowered == prefix or lowered.startswith(prefix + "/"):
            suffix = normalized[len(prefix) :].lstrip("/")
            return str((local_root / suffix).resolve())
    return str(Path(value).resolve())


def map_external_pattern(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    lowered = normalized.lower()
    for prefix, local_root in WINDOWS_ROOT_MAP:
        if lowered == prefix or lowered.startswith(prefix + "/"):
            suffix = normalized[len(prefix) :].lstrip("/")
            return str(local_root / suffix)
    return normalized


def map_path_structure(value: Any) -> Any:
    """Relocate Windows-rooted path strings while preserving other values."""

    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        lowered = normalized.lower()
        if any(lowered == prefix or lowered.startswith(prefix + "/") for prefix, _ in WINDOWS_ROOT_MAP):
            return map_external_value(value)
        return value
    if isinstance(value, list):
        return [map_path_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(map_path_structure(item) for item in value)
    if isinstance(value, dict):
        return {map_path_structure(key): map_path_structure(item) for key, item in value.items()}
    return value


def map_dataset_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic Linux execution-path relocation for a source config."""

    dataset = config.get("dataset", {})
    mapped: dict[str, Any] = {}
    if "file_index" in dataset:
        mapped["file_index"] = str(Path(map_external_value(str(dataset["file_index"]))).resolve())
    if "globs" in dataset:
        mapped["globs"] = [map_external_pattern(str(value)) for value in dataset["globs"]]
    if "player_names_files" in dataset:
        mapped["player_names_files"] = [
            str(Path(map_external_value(str(value))).resolve()) for value in dataset["player_names_files"]
        ]
    if "player_names_by_file" in dataset:
        mapped["player_names_by_file"] = map_path_structure(dataset["player_names_by_file"])
    return mapped


def _is_windows_rooted_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/").lower()
    return any(normalized == prefix or normalized.startswith(prefix + "/") for prefix, _ in WINDOWS_ROOT_MAP)


def _collect_external_path_bindings(
    source_value: Any,
    runtime_value: Any,
    *,
    location: str,
) -> list[dict[str, str]]:
    if isinstance(source_value, str) and _is_windows_rooted_path(source_value):
        runtime_path = Path(str(runtime_value)).resolve()
        if not runtime_path.is_file():
            raise ContractError(f"mapped label path is missing: {runtime_path}")
        return [
            {
                "location": location,
                "source_path": source_value,
                "path": str(runtime_path),
                "sha256": sha256_file(runtime_path),
            }
        ]
    if isinstance(source_value, list) and isinstance(runtime_value, list):
        records: list[dict[str, str]] = []
        for index, (source_item, runtime_item) in enumerate(zip(source_value, runtime_value, strict=True)):
            records.extend(
                _collect_external_path_bindings(
                    source_item,
                    runtime_item,
                    location=f"{location}[{index}]",
                )
            )
        return records
    if isinstance(source_value, dict) and isinstance(runtime_value, dict):
        records = []
        for key, source_item in source_value.items():
            if key not in runtime_value:
                raise ContractError(f"mapped label structure is missing {location}.{key}")
            records.extend(
                _collect_external_path_bindings(
                    source_item,
                    runtime_value[key],
                    location=f"{location}.{key}",
                )
            )
        return records
    return []


def build_label_binding(config: dict[str, Any]) -> dict[str, Any]:
    """Bind every label path and any optional player_names_by_file path to content SHA."""

    dataset = config.get("dataset", {})
    if "player_names_files" not in dataset:
        raise ContractError("source dataset.player_names_files is missing")
    mapped_dataset = map_dataset_paths(config)
    source_files = list(dataset["player_names_files"])
    runtime_files = list(mapped_dataset.get("player_names_files", []))
    if len(source_files) != len(runtime_files):
        raise ContractError("source/runtime player_names_files lengths differ")
    file_records: list[dict[str, str]] = []
    for index, (source_path, runtime_path) in enumerate(zip(source_files, runtime_files, strict=True)):
        mapped_path = Path(str(runtime_path)).resolve()
        if not mapped_path.is_file():
            raise ContractError(f"source label file missing: {mapped_path}")
        file_records.append(
            {
                "location": f"dataset.player_names_files[{index}]",
                "source_path": str(source_path),
                "path": str(mapped_path),
                "sha256": sha256_file(mapped_path),
            }
        )
    by_file_records: list[dict[str, str]] | None = None
    if "player_names_by_file" in dataset:
        by_file_records = _collect_external_path_bindings(
            dataset["player_names_by_file"],
            mapped_dataset.get("player_names_by_file"),
            location="dataset.player_names_by_file",
        )
    return {
        "player_names_files": file_records,
        "player_names_by_file": by_file_records,
    }


def validate_label_binding(
    source: dict[str, Any],
    generated: dict[str, Any],
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    actual_binding = build_label_binding(source)
    if actual_binding != expected_binding:
        raise ContractError("label path/content binding mismatch")
    source_dataset = map_dataset_paths(source)
    generated_dataset = generated.get("dataset", {})
    for key in ("player_names_files", "player_names_by_file"):
        if key in source_dataset and generated_dataset.get(key) != source_dataset[key]:
            raise ContractError(f"generated label relocation mismatch: dataset.{key}")
    return actual_binding


def ordered_path_mapping_sha256(source_paths: list[str], runtime_paths: list[str]) -> str:
    if len(source_paths) != len(runtime_paths):
        raise ContractError("ordered source/runtime file mapping lengths differ")
    pairs = [
        {"index": index, "source": source, "runtime": runtime}
        for index, (source, runtime) in enumerate(zip(source_paths, runtime_paths, strict=True))
    ]
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _payload_without_file_list(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "file_list"}


def build_runtime_file_index(*, source_path: Path, runtime_path: Path) -> dict[str, Any]:
    """Create one route-level Linux index while preserving source order and payload identity."""

    source_payload, source_paths = load_file_index(source_path)
    runtime_paths = [map_external_value(value) for value in source_paths]
    missing = next((value for value in runtime_paths if not Path(value).is_file()), None)
    if missing is not None:
        raise ContractError(f"runtime path mapping points to missing file: {missing}")
    runtime_payload = copy.deepcopy(source_payload)
    runtime_payload["file_list"] = runtime_paths
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(runtime_payload, runtime_path)
    record = {
        "source_file_index_path": str(source_path.resolve()),
        "source_file_index_sha256": sha256_file(source_path),
        "runtime_file_index_path": str(runtime_path.resolve()),
        "runtime_file_index_sha256": sha256_file(runtime_path),
        "file_count": len(source_paths),
        "ordered_path_mapping_sha256": ordered_path_mapping_sha256(source_paths, runtime_paths),
        "source_payload_without_file_list_sha256": tensor_digest(_payload_without_file_list(source_payload)),
        "runtime_payload_without_file_list_sha256": tensor_digest(_payload_without_file_list(runtime_payload)),
    }
    if record["source_payload_without_file_list_sha256"] != record["runtime_payload_without_file_list_sha256"]:
        raise ContractError("runtime file index changed payload fields outside file_list")
    return record


def validate_runtime_file_index(
    record: dict[str, Any],
    *,
    expected_source_path: Path,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Re-derive and verify a formal runtime index from its frozen source index."""

    source_path = Path(record["source_file_index_path"]).resolve()
    runtime_path = Path(record["runtime_file_index_path"]).resolve()
    if source_path != expected_source_path.resolve() or record.get("source_file_index_sha256") != expected_source_sha256:
        raise ContractError("runtime index source binding mismatch")
    if not runtime_path.is_file():
        raise ContractError(f"formal runtime file index is missing: {runtime_path}")
    if sha256_file(source_path) != expected_source_sha256:
        raise ContractError(f"source file index SHA mismatch: {source_path}")
    source_payload, source_paths = load_file_index(source_path)
    runtime_payload, runtime_paths = load_file_index(runtime_path)
    expected_runtime_paths = [map_external_value(value) for value in source_paths]
    if runtime_paths != expected_runtime_paths:
        raise ContractError("formal runtime file index is not the deterministic ordered relocation")
    if any(not Path(value).is_file() for value in runtime_paths):
        raise ContractError("formal runtime file index contains a missing mapped file")
    expected_record = build_runtime_index_record(
        source_path=source_path,
        runtime_path=runtime_path,
        source_payload=source_payload,
        source_paths=source_paths,
        runtime_payload=runtime_payload,
        runtime_paths=runtime_paths,
    )
    for key, value in expected_record.items():
        if record.get(key) != value:
            raise ContractError(f"formal runtime file index binding mismatch: {key}")
    return expected_record


def build_runtime_index_record(
    *,
    source_path: Path,
    runtime_path: Path,
    source_payload: dict[str, Any],
    source_paths: list[str],
    runtime_payload: dict[str, Any],
    runtime_paths: list[str],
) -> dict[str, Any]:
    return {
        "source_file_index_path": str(source_path.resolve()),
        "source_file_index_sha256": sha256_file(source_path),
        "runtime_file_index_path": str(runtime_path.resolve()),
        "runtime_file_index_sha256": sha256_file(runtime_path),
        "file_count": len(source_paths),
        "ordered_path_mapping_sha256": ordered_path_mapping_sha256(source_paths, runtime_paths),
        "source_payload_without_file_list_sha256": tensor_digest(_payload_without_file_list(source_payload)),
        "runtime_payload_without_file_list_sha256": tensor_digest(_payload_without_file_list(runtime_payload)),
    }


def source_config_path(source_root: Path, route: str, seed: int) -> Path:
    return source_root / SOURCE_ROUTE_DIR[route] / f"seed_{seed}" / "config.toml"


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_file_index(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("file_list"), list):
        raise ContractError(f"file index has no file_list: {path}")
    values = [str(value) for value in payload["file_list"]]
    if len(values) != 6000 or len(set(values)) != 6000:
        raise ContractError(f"file index must contain 6000 unique paths: {path}")
    return payload, values


def inspect_historical_mortal_checkpoint(
    source_config_path_value: Path,
    *,
    route: str,
    seed: int,
) -> dict[str, Any]:
    source = load_toml(source_config_path_value)
    raw_state_file = source.get("control", {}).get("state_file")
    if not raw_state_file:
        raise ContractError(f"historical source config has no control.state_file: {source_config_path_value}")
    checkpoint_path = Path(map_external_value(str(raw_state_file))).resolve()
    if not checkpoint_path.is_file():
        raise ContractError(f"historical CURRENT checkpoint is missing: {checkpoint_path}")
    state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    contract = state.get("training_contract")
    if not isinstance(contract, dict) or not contract.get("mortal_revision"):
        raise ContractError(f"historical checkpoint has no training_contract.mortal_revision: {checkpoint_path}")
    return {
        "route": route,
        "seed": seed,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "steps": int(state.get("steps", -1)),
        "mortal_revision": str(contract["mortal_revision"]),
    }


def validate_historical_mortal_checkpoint(
    expected: dict[str, Any],
    source_config_path_value: Path,
    *,
    route: str,
    seed: int,
) -> dict[str, Any]:
    actual = inspect_historical_mortal_checkpoint(source_config_path_value, route=route, seed=seed)
    if actual != expected:
        raise ContractError(f"historical checkpoint provenance mismatch: {route}/{seed}")
    if actual["steps"] != TARGET_STEP or actual["mortal_revision"] != HISTORICAL_MORTAL_REVISION:
        raise ContractError(f"historical checkpoint contract mismatch: {route}/{seed}")
    return actual


def validate_source_inputs(config: dict[str, Any], *, route: str, seed: int) -> dict[str, Any]:
    if float(config["cql"]["min_q_weight"]) != 5.0:
        raise ContractError(f"source CURRENT CQL weight is not 5.0: {route}/{seed}")
    if config["objective"]["mode"] != "behavior_action_mc":
        raise ContractError(f"source objective mismatch: {route}/{seed}")
    if config["reward"]["mode"] != "final_rank_mc":
        raise ContractError(f"source reward mismatch: {route}/{seed}")
    if int(config["control"]["batch_size"]) != 512 or int(config["dataset"]["num_workers"]) != 0:
        raise ContractError(f"source batch/worker contract mismatch: {route}/{seed}")
    if bool(config["control"]["enable_amp"]):
        raise ContractError(f"source AMP must be false: {route}/{seed}")
    experiment = config.get("experiment", {})
    if int(experiment.get("training_seed", -1)) != seed or int(experiment.get("parent_steps", -1)) != START_STEP:
        raise ContractError(f"source seed/parent step mismatch: {route}/{seed}")
    index_path = Path(map_external_value(str(config["dataset"]["file_index"])))
    if not index_path.is_file():
        raise ContractError(f"source file index missing: {index_path}")
    route_key = route.split("_", maxsplit=1)[0]
    index_sha256 = sha256_file(index_path)
    if index_sha256 != SOURCE_FILE_INDEX_SHA256[route_key]:
        raise ContractError(f"source file index SHA mismatch: {route}/{seed}")
    _, file_list = load_file_index(index_path)
    mapped_files = [map_external_value(value) for value in file_list]
    missing = next((value for value in mapped_files if not Path(value).is_file()), None)
    if missing is not None:
        raise ContractError(f"source file index points to missing data: {missing}")
    label_binding = build_label_binding(config)
    return {
        "file_index_path": str(index_path.resolve()),
        "file_index_sha256": index_sha256,
        "file_count": len(file_list),
        "label_files": label_binding["player_names_files"],
        "label_binding": label_binding,
        "mapped_file_count": len(mapped_files),
    }


def semantic_diff(source: Any, generated: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(source, dict) and isinstance(generated, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(source) | set(generated)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in source or key not in generated:
                differences.append(
                    {
                        "path": child_path,
                        "source": source.get(key),
                        "generated": generated.get(key),
                    }
                )
            else:
                differences.extend(semantic_diff(source[key], generated[key], child_path))
        return differences
    if source != generated:
        return [{"path": path, "source": source, "generated": generated}]
    return []


def diff_path_allowed(path: str) -> bool:
    if path in OUTPUT_PATHS or path in SCIENTIFIC_PATHS or path == "experiment" or path.startswith("experiment."):
        return True
    return any(
        path == root or path.startswith((root + "[", root + "."))
        for root in EXECUTION_PATH_ROOTS
    )


def _path_in_roots(path: str, roots: set[str]) -> bool:
    return any(path == root or path.startswith((root + "[", root + ".")) for root in roots)


def semantic_diff_gate(source: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    differences = semantic_diff(source, generated)
    unexpected = [item for item in differences if not diff_path_allowed(str(item["path"]))]
    scientific_differences = [item for item in differences if _path_in_roots(str(item["path"]), SCIENTIFIC_PATHS)]
    execution_path_differences = [
        item for item in differences if _path_in_roots(str(item["path"]), EXECUTION_PATH_ROOTS | OUTPUT_PATHS)
    ]
    provenance_differences = [
        item
        for item in differences
        if str(item["path"]) == "experiment" or str(item["path"]).startswith("experiment.")
    ]
    passed = not unexpected and scientific_differences == [
        {
            "path": "cql.min_q_weight",
            "source": 5.0,
            "generated": 0.0,
        }
    ]
    return {
        "passed": passed,
        "differences": differences,
        "scientific_differences": scientific_differences,
        "execution_path_differences": execution_path_differences,
        "provenance_differences": provenance_differences,
        "unexpected_differences": unexpected,
        "allowed_paths": sorted(
            OUTPUT_PATHS | SCIENTIFIC_PATHS | EXECUTION_PATH_ROOTS | {"experiment.*"}
        ),
    }


def build_c1_config(
    source: dict[str, Any],
    *,
    route: str,
    seed: int,
    run_dir: Path,
    source_sha256: str,
    runtime_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    config["control"]["state_file"] = str((run_dir / "mortal.pth").resolve())
    config["control"]["best_state_file"] = str((run_dir / "mortal_best.pth").resolve())
    config["control"]["tensorboard_dir"] = str((run_dir / "tb_mortal").resolve())
    if runtime_dataset is not None:
        config.setdefault("dataset", {}).update(copy.deepcopy(runtime_dataset))
    config["cql"]["min_q_weight"] = 0.0
    experiment = copy.deepcopy(config.get("experiment", {}))
    experiment.update(
        {
            "experiment_id": C1_ID,
            "condition": "CQL_OFF",
            "route": route,
            "training_seed": seed,
            "parent_steps": START_STEP,
            "prereg_commit": PREREG_COMMIT,
            "prereg_sha256": PREREG_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "loader_report_sha256": LOADER_SHA256,
            "source_current_config_sha256": source_sha256,
        }
    )
    config["experiment"] = experiment
    return config


def validate_generated_config(
    source: dict[str, Any],
    generated: dict[str, Any],
    *,
    route: str,
    seed: int,
    run_dir: Path,
    source_sha256: str,
    runtime_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = semantic_diff_gate(source, generated)
    if not gate["passed"]:
        raise ContractError(f"semantic diff gate failed for {route}/{seed}: {gate}")
    if float(generated["cql"]["min_q_weight"]) != 0.0:
        raise ContractError(f"generated CQL_OFF weight is not 0.0: {route}/{seed}")
    expected_paths = {
        "state_file": str((run_dir / "mortal.pth").resolve()),
        "best_state_file": str((run_dir / "mortal_best.pth").resolve()),
        "tensorboard_dir": str((run_dir / "tb_mortal").resolve()),
    }
    for key, expected in expected_paths.items():
        if generated["control"][key] != expected:
            raise ContractError(f"generated output path mismatch: {route}/{seed}/{key}")
    if runtime_dataset is not None:
        for key, expected in runtime_dataset.items():
            if generated.get("dataset", {}).get(key) != expected:
                raise ContractError(f"generated runtime dataset path mismatch: {route}/{seed}/{key}")
    experiment = generated["experiment"]
    expected_provenance = {
        "experiment_id": C1_ID,
        "condition": "CQL_OFF",
        "route": route,
        "training_seed": seed,
        "parent_steps": START_STEP,
        "prereg_commit": PREREG_COMMIT,
        "prereg_sha256": PREREG_SHA256,
        "registry_sha256": REGISTRY_SHA256,
        "loader_report_sha256": LOADER_SHA256,
        "source_current_config_sha256": source_sha256,
    }
    if any(experiment.get(key) != value for key, value in expected_provenance.items()):
        raise ContractError(f"generated provenance mismatch: {route}/{seed}")
    allowed_experiment_updates = set(expected_provenance)
    for key, value in source.get("experiment", {}).items():
        if key not in allowed_experiment_updates and experiment.get(key) != value:
            raise ContractError(f"generated non-provenance experiment field changed: {route}/{seed}/{key}")
    if "allow_legacy_data_replay" in generated:
        raise ContractError("legacy replay override must not be present in generated config")
    return gate


def validate_governance_payload(
    registry: dict[str, Any],
    loader: dict[str, Any],
) -> dict[str, Any]:
    if registry.get("schema") != "keqing.mortal.research_registry.v1":
        raise ContractError("registry schema mismatch")
    state = registry.get("current_state", {})
    if state.get("next_experiment") != C1_ID or state.get("next_experiment_status") != "preregistered_frozen":
        raise ContractError("registry current_state is not the registered C1 state")
    records = [record for record in registry.get("records", []) if record.get("experiment_id") == C1_ID]
    if len(records) != 1:
        raise ContractError(f"expected exactly one C1 registry record, got {len(records)}")
    record = records[0]
    preregistration = record.get("preregistration", {})
    if record.get("status") != "preregistered_frozen":
        raise ContractError("C1 registry status is not preregistered_frozen")
    if preregistration.get("freeze_commit") != PREREG_COMMIT or preregistration.get("document_sha256") != PREREG_SHA256:
        raise ContractError("C1 preregistration provenance mismatch")
    if preregistration.get("training_authorized") is not False or preregistration.get("evaluation_authorized") is not False:
        raise ContractError("C1 authorization flags are not false")
    if loader.get("status") != "PASS":
        raise ContractError("loader feasibility report is not PASS")
    gate = loader.get("gate", {})
    if gate.get("resolved_training_count") != 6 or gate.get("streams_compared") != 6 or gate.get("streams_required") != 6:
        raise ContractError("loader resolved count/stream count mismatch")
    if loader.get("training_started") or loader.get("evaluation_started") or loader.get("generation_started"):
        raise ContractError("loader report indicates execution has started")
    if loader.get("optimizer_steps") != 0 or loader.get("new_checkpoints") != 0:
        raise ContractError("loader report indicates steps/checkpoints")
    seen: set[tuple[str, int]] = set()
    stream_records = []
    for run in loader.get("runs", []):
        route = str(run.get("route"))
        seed = int(run.get("seed"))
        key = (route, seed)
        if key not in LOADER_STREAM_SHA256 or run.get("stream", {}).get("exact_match") is not True:
            raise ContractError(f"loader stream mismatch: {key}")
        if run["stream"].get("current_stream_sha256") != LOADER_STREAM_SHA256[key]:
            raise ContractError(f"loader stream digest mismatch: {key}")
        seen.add(key)
        stream_records.append({"route": route, "seed": seed, "stream_sha256": LOADER_STREAM_SHA256[key]})
    if seen != set(LOADER_STREAM_SHA256):
        raise ContractError(f"loader stream matrix mismatch: {seen}")
    return {"status": "PASS", "resolved_training_count": 6, "streams": stream_records}


def inspect_parent(parent_path: Path) -> dict[str, Any]:
    if not parent_path.is_file() or sha256_file(parent_path) != K0_PARENT_SHA256:
        raise ContractError(f"K0 parent SHA mismatch: {parent_path}")
    state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if int(state.get("steps", -1)) != START_STEP:
        raise ContractError("K0 parent step is not 70000")
    for key in ("mortal", "current_dqn", "aux_net", "optimizer"):
        if key not in state:
            raise ContractError(f"K0 parent missing {key}")
    optimizer = state["optimizer"]
    missing_moments = []
    for group in optimizer.get("param_groups", []):
        for parameter_index in group.get("params", []):
            entry = optimizer.get("state", {}).get(parameter_index)
            if not isinstance(entry, dict) or "exp_avg" not in entry or "exp_avg_sq" not in entry:
                missing_moments.append(str(parameter_index))
    digest = {
        "checkpoint_sha256": sha256_file(parent_path),
        "mortal_sha256": tensor_digest(state["mortal"]),
        "current_dqn_sha256": tensor_digest(state["current_dqn"]),
        "aux_net_sha256": tensor_digest(state["aux_net"]),
        "optimizer_sha256": tensor_digest(state["optimizer"]),
        "optimizer_state_count": len(optimizer.get("state", {})),
        "steps": int(state["steps"]),
    }
    return {
        "path": str(parent_path.resolve()),
        "sha256": K0_PARENT_SHA256,
        "digest": digest,
        "optimizer_moments_covered": not missing_moments,
        "optimizer_missing_moments": missing_moments,
        "optimizer_source_equals_parent": True,
    }


def implementation_sources() -> list[dict[str, str]]:
    records = []
    for relative_path in IMPLEMENTATION_SOURCES:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise ContractError(f"implementation source missing: {path}")
        records.append(
            {
                "path": relative_path,
                "content_sha256": sha256_file(path),
                "git_blob_oid": git_blob_oid(path),
            }
        )
    return records


def future_training_argv(
    *,
    config_path: Path,
    parent_path: Path,
    seed: int,
    run_dir: Path,
    executable: str | None = None,
) -> list[str]:
    archive_steps = ",".join(str(step) for step in ARCHIVE_STEPS)
    return [
        str(executable or sys.executable),
        "training/run_mortal_dqn_offline.py",
        "--config",
        str(config_path.resolve()),
        "--initialize-from",
        str(parent_path.resolve()),
        "--initialize-optimizer-from",
        str(parent_path.resolve()),
        "--initial-steps",
        str(START_STEP),
        "--target-steps",
        str(TARGET_STEP),
        "--device",
        "cuda:0",
        "--seed",
        str(seed),
        "--data-seed",
        str(seed),
        "--num-workers",
        "0",
        "--archive-steps",
        archive_steps,
        "--archive-dir",
        str((run_dir / "checkpoints").resolve()),
    ]


def future_training_command(
    *,
    config_path: Path,
    parent_path: Path,
    seed: int,
    run_dir: Path,
    executable: str | None = None,
) -> str:
    return shlex.join(
        future_training_argv(
            config_path=config_path,
            parent_path=parent_path,
            seed=seed,
            run_dir=run_dir,
            executable=executable,
        )
    )


def load_governance(registry_path: Path, prereg_path: Path, loader_path: Path) -> dict[str, Any]:
    if sha256_file(registry_path) != REGISTRY_SHA256:
        raise ContractError("registry SHA mismatch")
    if sha256_file(prereg_path) != PREREG_SHA256:
        raise ContractError("frozen prereg SHA mismatch")
    if sha256_file(loader_path) != LOADER_SHA256:
        raise ContractError("loader report SHA mismatch")
    registry = load_json(registry_path)
    loader = load_json(loader_path)
    loader_contract = validate_governance_payload(registry, loader)
    return {
        "registry_sha256": REGISTRY_SHA256,
        "prereg_sha256": PREREG_SHA256,
        "loader_report_sha256": LOADER_SHA256,
        "loader_contract": loader_contract,
    }


def prepare(
    *,
    output_dir: Path,
    source_root: Path,
    parent_path: Path,
    registry_path: Path,
    prereg_path: Path,
    loader_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    source_root = source_root.resolve()
    parent_path = parent_path.resolve()
    governance = load_governance(registry_path.resolve(), prereg_path.resolve(), loader_path.resolve())
    git = git_info()
    validate_git_scope(git)
    parent = inspect_parent(parent_path)
    frozen_runtime = runtime_provenance()
    frozen_mortal = mortal_source_provenance()
    historical_mortal_checkpoints: dict[tuple[str, int], dict[str, Any]] = {}
    for route in ROUTES:
        for seed in SEEDS:
            historical_path = source_config_path(source_root, route, seed).resolve()
            historical_mortal_checkpoints[(route, seed)] = inspect_historical_mortal_checkpoint(
                historical_path,
                route=route,
                seed=seed,
            )
    historical_revisions = {
        item["mortal_revision"] for item in historical_mortal_checkpoints.values()
    }
    if historical_revisions != {HISTORICAL_MORTAL_REVISION}:
        raise ContractError(f"six historical CURRENT Mortal revisions are not identical: {historical_revisions}")
    source_records: dict[str, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    runtime_inputs: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for route in ROUTES:
        route_key = route.split("_", maxsplit=1)[0]
        for seed in SEEDS:
            source_path = source_config_path(source_root, route, seed).resolve()
            expected_source_sha = SOURCE_CONFIG_SHA256[(route, seed)]
            if not source_path.is_file() or sha256_file(source_path) != expected_source_sha:
                raise ContractError(f"source config SHA mismatch: {route}/{seed}: {source_path}")
            source = load_toml(source_path)
            source_inputs = validate_source_inputs(source, route=route, seed=seed)
            runtime_spec = runtime_inputs.get(route_key)
            if runtime_spec is None:
                runtime_spec = build_runtime_file_index(
                    source_path=Path(source_inputs["file_index_path"]),
                    runtime_path=output_dir / "runtime_inputs" / RUNTIME_INDEX_FILENAME[route],
                )
                runtime_spec["route"] = route_key
                runtime_inputs[route_key] = runtime_spec
            else:
                validate_runtime_file_index(
                    runtime_spec,
                    expected_source_path=Path(source_inputs["file_index_path"]),
                    expected_source_sha256=source_inputs["file_index_sha256"],
                )
            if runtime_spec["source_file_index_sha256"] != source_inputs["file_index_sha256"]:
                raise ContractError(f"runtime source index SHA differs across {route} seeds")
            runtime_dataset = map_dataset_paths(source)
            runtime_dataset["file_index"] = runtime_spec["runtime_file_index_path"]
            run_dir = output_dir / route / f"seed_{seed}"
            generated = build_c1_config(
                source,
                route=route,
                seed=seed,
                run_dir=run_dir,
                source_sha256=expected_source_sha,
                runtime_dataset=runtime_dataset,
            )
            validate_label_binding(source, generated, source_inputs["label_binding"])
            gate = validate_generated_config(
                source,
                generated,
                route=route,
                seed=seed,
                run_dir=run_dir,
                source_sha256=expected_source_sha,
                runtime_dataset=runtime_dataset,
            )
            import toml

            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.toml"
            config_path.write_text(toml.dumps(generated), encoding="utf-8")
            if tomllib.loads(config_path.read_text(encoding="utf-8")) != generated:
                raise ContractError(f"generated TOML round-trip changed config: {config_path}")
            config_sha256 = sha256_file(config_path)
            argv = future_training_argv(
                config_path=config_path,
                parent_path=parent_path,
                seed=seed,
                run_dir=run_dir,
                executable=frozen_runtime["sys_executable"],
            )
            run_records.append(
                {
                    "route": route,
                    "seed": seed,
                    "source_current_config": str(source_path),
                    "source_current_config_sha256": expected_source_sha,
                    "cql_off_config": str(config_path.resolve()),
                    "cql_off_config_sha256": config_sha256,
                    "formal_training_config_sha256": config_sha256,
                    "smoke_config_sha256": config_sha256,
                    "exact_same_config": True,
                    "semantic_diff": gate,
                    "source_file_index_path": runtime_spec["source_file_index_path"],
                    "source_file_index_sha256": runtime_spec["source_file_index_sha256"],
                    "runtime_file_index_path": runtime_spec["runtime_file_index_path"],
                    "runtime_file_index_sha256": runtime_spec["runtime_file_index_sha256"],
                    "file_count": runtime_spec["file_count"],
                    "ordered_path_mapping_sha256": runtime_spec["ordered_path_mapping_sha256"],
                    "source_payload_without_file_list_sha256": runtime_spec[
                        "source_payload_without_file_list_sha256"
                    ],
                    "runtime_payload_without_file_list_sha256": runtime_spec[
                        "runtime_payload_without_file_list_sha256"
                    ],
                    "file_index": runtime_spec["source_file_index_path"],
                    "file_index_sha256": runtime_spec["source_file_index_sha256"],
                    "label_files": source_inputs["label_files"],
                    "label_binding": source_inputs["label_binding"],
                    "historical_mortal_revision": historical_mortal_checkpoints[(route, seed)][
                        "mortal_revision"
                    ],
                    "historical_checkpoint_sha256": historical_mortal_checkpoints[(route, seed)]["sha256"],
                    "runtime_dataset": runtime_dataset,
                    "loader_stream_sha256": LOADER_STREAM_SHA256[(route_key, seed)],
                    "run_output_dir": str(run_dir.resolve()),
                    "future_training_argv": argv,
                    "future_training_command": shlex.join(argv),
                }
            )
            source_records[f"{route}/{seed}"] = {"path": str(source_path), "sha256": expected_source_sha}
    if {(item["route"], int(item["seed"])) for item in run_records} != {
        (route, seed) for route in ROUTES for seed in SEEDS
    }:
        raise ContractError("generated run matrix is not exactly six C1 CQL_OFF runs")
    manifest = {
        "schema": "keqing.mortal.c1_training_manifest.v1",
        "experiment_id": C1_ID,
        "status": "prepared_not_authorized",
        "training_authorized": False,
        "evaluation_authorized": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "new_training_runs": 6,
        "implementation_commit": git["commit"],
        "implementation_sources": implementation_sources(),
        "git": git,
        "governance": {
            "prereg_commit": PREREG_COMMIT,
            "prereg_sha256": governance["prereg_sha256"],
            "registry_sha256": governance["registry_sha256"],
            "loader_report_sha256": governance["loader_report_sha256"],
            "loader_contract": governance["loader_contract"],
        },
        "parent": parent,
        "runtime_provenance": frozen_runtime,
        "mortal_provenance": frozen_mortal,
        "historical_mortal_checkpoints": [
            historical_mortal_checkpoints[(route, seed)]
            for route in ROUTES
            for seed in SEEDS
        ],
        "runtime_inputs": runtime_inputs,
        "fixed_contract": {
            "factor_a": ["M0_operational_control", "D1_project_owned_k0_view"],
            "factor_b": {"CURRENT": 5.0, "CQL_OFF": 0.0},
            "historical_current_training_reuse": "APPROVED",
            "new_training_runs": 6,
            "start_step": START_STEP,
            "target_step": TARGET_STEP,
            "optimizer_steps": OPTIMIZER_STEPS,
            "objective": "behavior_action_mc",
            "reward": "final_rank_mc",
            "batch_size": 512,
            "num_workers": 0,
            "amp": False,
            "archive_steps": list(ARCHIVE_STEPS),
            "runtime_file_count": 6000,
        },
        "source_root": str(source_root),
        "source_configs": source_records,
        "runs": run_records,
        "execution_boundary": {
            "training_started": False,
            "evaluation_started": False,
            "generation_started": False,
            "optimizer_steps": 0,
            "new_checkpoints": 0,
            "allow_legacy_data_replay": False,
            "formal_config_equals_smoke_config": True,
        },
        "training_command_policy": {
            "initialize_from": "same K0 parent for weights and preserved Adam",
            "device": "cuda:0",
            "data_seed_equals_training_seed": True,
            "user_scientific_overrides": False,
            "authoritative_field": "future_training_argv",
            "frozen_executable": frozen_runtime["sys_executable"],
            "shell": False,
        },
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": {
            f"{item['route']}/{item['seed']}": item["cql_off_config_sha256"] for item in run_records
        },
        "runtime_file_index_sha256": {
            route: record["runtime_file_index_sha256"] for route, record in runtime_inputs.items()
        },
        "implementation_commit": git["commit"],
        "training_authorized": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "training/docs/mortal/research_registry.json")
    parser.add_argument(
        "--prereg",
        type=Path,
        default=REPO_ROOT / "training/docs/mortal/experiments_zh/2026-08_C1语料_CQL交互因果实验_预注册设计.md",
    )
    parser.add_argument(
        "--loader-report",
        type=Path,
        default=REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/loader_compatibility.json",
    )
    args = parser.parse_args(argv)
    result = prepare(
        output_dir=args.output_dir,
        source_root=args.source_root,
        parent_path=args.parent,
        registry_path=args.registry,
        prereg_path=args.prereg,
        loader_path=args.loader_report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
