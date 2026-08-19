#!/usr/bin/env python3
"""Frozen C1-I2 evaluation contract and provenance helpers.

This module contains only contract construction and validation.  It never
imports the four-player arena, starts a subprocess, creates an evaluation
shard, or performs a game.  The two command-line preparation scripts and the
future fail-closed launcher share these definitions so that the plan cannot
drift from the execution guard.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
C1_ID = "C1_corpus_cql_interaction_2026_08"

PREREG_COMMIT = "65a8b220d361fc8285c3f9494349307bb2b86913"
PREREG_SHA256 = "e02ba3a667a7f2538a97e180fee8646ed819350c779db934b1852d97d5f33573"
REGISTRY_SHA256 = "65c04d0bf9617648cc3d0cff685313caca39437ae3673485133491d4e6c091c0"
I1_COMMIT = "5c2bed32e88aab9210fb1a9cd99134f82d7edd58"
I1_MANIFEST_SHA256 = "58218430f02b7b4179e29d89b21385c8fc9052fbc830f7d1c0b4c6f296f7186c"
I1_PREFLIGHT_SHA256 = "e3efb349430c45a5958ea378e3744426176dc71803e9b9035d93ba429925cb3e"
I1_MANIFEST_PATH = (
    REPO_ROOT
    / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/"
    "training_implementation_2026_08/training_manifest.json"
)
I1_PREFLIGHT_PATH = I1_MANIFEST_PATH.parent / "preflight/training_preflight.json"

PREREG_PATH = REPO_ROOT / "training/docs/mortal/experiments_zh/2026-08_C1语料_CQL交互因果实验_预注册设计.md"
REGISTRY_PATH = REPO_ROOT / "training/docs/mortal/research_registry.json"

EVALUATION_ROOT = (
    REPO_ROOT
    / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/"
    "evaluation_implementation_2026_08"
)
EVALUATION_PLAN_PATH = EVALUATION_ROOT / "evaluation_plan.json"
IMPLEMENTATION_PREFLIGHT_PATH = EVALUATION_ROOT / "implementation_preflight.json"

EVALUATOR_RELATIVE_PATH = "training/mortal/four_player_native.py"
EVALUATOR_PATH = REPO_ROOT / EVALUATOR_RELATIVE_PATH
EVALUATOR_FROZEN_COMMIT = "8c2c2eeb530bc54515e063be915292644a5163ee"
EVALUATOR_SHA256 = "6663badce02e04150f3ccc13f0ab3d4568e3743ed556ed22c9b3dd23cbf775fe"
EVALUATOR_BLOB_OID = "90487ea578850966f9c33bbcfb5e75a0fc589081"

DIRECT_EVALUATION_SOURCES = (
    "training/mortal/eval_metrics.py",
    "training/mortal/stat_report.py",
    "third_party/Mortal/mortal/engine.py",
    "third_party/Mortal/mortal/model.py",
)
DIRECT_EVALUATION_SOURCE_SHA256 = {
    "training/mortal/eval_metrics.py": "68559cadc3f185c7e89012197d1df2cd581474589c038cf9a9e10c13ffb791a3",
    "training/mortal/stat_report.py": "6cc948dd177ece97400a559c9f4401a79536d6570514c9c5b54b27d0f1078569",
    "third_party/Mortal/mortal/engine.py": "eecfeb1c337293820996a5eb4531dbc01ebee0203ed97539b74cc749369f135b",
    "third_party/Mortal/mortal/model.py": "217e2d3dc20631c769a57ea3f09ca4fac232430fc28f76b21d5f37cec7bf84f2",
}
DIRECT_EVALUATION_SOURCE_BLOB_OID = {
    "training/mortal/eval_metrics.py": "07e5002fda90905ccc93d3fdacfa086826e71a39",
    "training/mortal/stat_report.py": "73b7768b8b5a9dd7a51a3a2a41597002e7232df2",
    "third_party/Mortal/mortal/engine.py": "62bb8f168797d39619beea6d802b0b197fe4219a",
    "third_party/Mortal/mortal/model.py": "0fce8aaf19e3cbff2be3d9c241c53dad4e59ddce",
}
MORTAL_REVISION = "813859fc8110ea178f56f009994bc4f1b9fee645"

K0_PATH = (
    REPO_ROOT.parents[1]
    / "keqing-data/mortal/authoritative/D3_top2_discard_v1_2026_08/"
    "models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"
EXT_MORTAL_PATH = (
    REPO_ROOT.parents[1]
    / "keqing-data/mortal/authoritative/D3_top2_discard_v1_2026_08/"
    "models/ext_mortal/external_mortal_20240308_best_min.pth"
)
EXT_MORTAL_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

CURRENT_ROOT = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/"
    "D1_project_owned_population_2026_07/training_prep_2026_07"
)
CURRENT_CHECKPOINTS = {
    ("M0", 20260806): {
        "path": CURRENT_ROOT / "M0_control/seed_20260806/mortal.pth",
        "sha256": "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
    },
    ("M0", 20260807): {
        "path": CURRENT_ROOT / "M0_control/seed_20260807/mortal.pth",
        "sha256": "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
    },
    ("M0", 20260808): {
        "path": CURRENT_ROOT / "M0_control/seed_20260808/mortal.pth",
        "sha256": "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
    },
    ("D1", 20260806): {
        "path": CURRENT_ROOT / "D1_variant/seed_20260806/mortal.pth",
        "sha256": "9425109b2562eb48a86ca7b3a250738b5691503f9156f29bc50a2b20e7a922aa",
    },
    ("D1", 20260807): {
        "path": CURRENT_ROOT / "D1_variant/seed_20260807/mortal.pth",
        "sha256": "e2718ee8d572071b8d46d04beaf5f2aa6d90ad847762254f80648de9639a0b3d",
    },
    ("D1", 20260808): {
        "path": CURRENT_ROOT / "D1_variant/seed_20260808/mortal.pth",
        "sha256": "985a3e532ef13cd7fab945c92839a941390fd9f7cc5dc0e177d4d4182a116f41",
    },
}

CONDITIONS = ("CURRENT", "CQL_OFF")
TRAINING_SEEDS = (20260806, 20260807, 20260808)
SHARDS = (0, 1, 2, 3)
GAMES_PER_SHARD = 250
TOTAL_SHARDS = len(CONDITIONS) * len(TRAINING_SEEDS) * len(SHARDS)
TOTAL_GAMES = TOTAL_SHARDS * GAMES_PER_SHARD
SEED_KEY = 8192
SHARD_STARTS = {
    20260806: (1700000, 1700250, 1700500, 1700750),
    20260807: (1710000, 1710250, 1710500, 1710750),
    20260808: (1720000, 1720250, 1720500, 1720750),
}
RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260818

ALLOWED_UNTRACKED = {"1.md"}


class ContractError(RuntimeError):
    """Raised when a frozen C1-I2 contract cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_oid(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_info() -> dict[str, Any]:
    status = run_git("status", "--porcelain", "--untracked-files=all").splitlines()
    tracked = run_git("diff", "--name-only").splitlines()
    tracked += run_git("diff", "--cached", "--name-only").splitlines()
    untracked = [line[3:] for line in status if line.startswith("?? ")]
    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
        "tracked_changes": sorted(set(tracked)),
        "untracked": sorted(untracked),
        "status": status,
        "tracked_clean": not tracked,
    }


def validate_git_scope(info: dict[str, Any], *, require_main: bool = True) -> None:
    if require_main and info.get("branch") != "main":
        raise ContractError(f"C1-I2 must be prepared on main, got {info.get('branch')!r}")
    if info.get("tracked_clean") is not True:
        raise ContractError(f"tracked worktree is dirty: {info.get('tracked_changes')}")
    if set(info.get("untracked", ())) != ALLOWED_UNTRACKED:
        raise ContractError(f"unexpected untracked files: {info.get('untracked')}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def current_model_label(route: str, seed: int) -> str:
    if route not in {"M0", "D1"} or seed not in TRAINING_SEEDS:
        raise ContractError(f"invalid CURRENT model identity: {route}/{seed}")
    return f"{route}_CURRENT_{seed}"


def off_model_label(route: str, seed: int) -> str:
    if route not in {"M0", "D1"} or seed not in TRAINING_SEEDS:
        raise ContractError(f"invalid CQL_OFF model identity: {route}/{seed}")
    return f"{route}_CQL_OFF_{seed}"


def model_order(condition: str, seed: int) -> tuple[str, ...]:
    if condition not in CONDITIONS or seed not in TRAINING_SEEDS:
        raise ContractError(f"invalid condition/seed: {condition}/{seed}")
    suffix = "CURRENT" if condition == "CURRENT" else "CQL_OFF"
    return ("70k", "ext_mortal", f"M0_{suffix}_{seed}", f"D1_{suffix}_{seed}")


def expected_seed_start(seed: int, shard: int) -> int:
    if seed not in SHARD_STARTS or shard not in SHARDS:
        raise ContractError(f"invalid evaluation shard: {seed}/{shard}")
    return SHARD_STARTS[seed][shard]


def evaluation_shard_dir(condition: str, seed: int, shard: int) -> Path:
    if condition not in CONDITIONS or seed not in TRAINING_SEEDS or shard not in SHARDS:
        raise ContractError(f"invalid evaluation identity: {condition}/{seed}/{shard}")
    return EVALUATION_ROOT / condition / f"seed_{seed}" / f"shard_{shard:02d}"


def file_record(path: Path, *, expected_sha256: str | None = None, state: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.is_file()
    record: dict[str, Any] = {"path": str(resolved), "sha256": sha256_file(resolved) if exists else None}
    if expected_sha256 is not None and exists and record["sha256"] != expected_sha256:
        raise ContractError(f"SHA256 mismatch: {resolved}")
    if state is not None:
        record["state"] = state
    return record


def runtime_provenance() -> dict[str, Any]:
    try:
        import torch

        native_module = importlib.import_module("riichi")
    except Exception as exc:  # pragma: no cover - machine-specific failure
        raise ContractError(f"unable to load frozen runtime: {exc}") from exc
    native_path = Path(getattr(native_module, "__file__", "")).resolve()
    if not native_path.is_file():
        raise ContractError(f"native runtime path is missing: {native_path}")
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "native_module": "riichi",
        "native_path": str(native_path),
        "native_sha256": sha256_file(native_path),
        # The frozen contract binds the invocation string, not the target of
        # the venv symlink.  Future evaluation must therefore use this exact
        # ``.venv/bin/python3`` token rather than a bare ``python`` or the
        # underlying uv interpreter path.
        "sys_executable": str(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_name": str(device_name) if device_name is not None else None,
        "device": "cuda",
        "require_cuda": True,
        "amp": False,
    }


def validate_runtime_provenance(expected: dict[str, Any]) -> dict[str, Any]:
    actual = runtime_provenance()
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ContractError(f"runtime provenance mismatch at {key}: {actual.get(key)!r} != {value!r}")
    if actual.get("cuda_available") is not True or actual.get("cuda_device_name") != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise ContractError("fresh runtime is not the required CUDA device")
    return actual


def source_provenance() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in DIRECT_EVALUATION_SOURCES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ContractError(f"evaluation dependency source is missing: {path}")
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "git_blob_oid": git_blob_oid(path),
            }
        )
    return {
        "evaluator": {
            "path": EVALUATOR_RELATIVE_PATH,
            "absolute_path": str(EVALUATOR_PATH.resolve()),
            "frozen_commit": EVALUATOR_FROZEN_COMMIT,
            "sha256": sha256_file(EVALUATOR_PATH),
            "git_blob_oid": git_blob_oid(EVALUATOR_PATH),
        },
        "direct_dependencies": records,
        "mortal_revision": run_git("-C", str(REPO_ROOT / "third_party/Mortal"), "rev-parse", "HEAD"),
    }


def validate_source_provenance(expected: dict[str, Any]) -> dict[str, Any]:
    actual = source_provenance()
    validate_frozen_evaluator_object()
    if actual != expected:
        raise ContractError("evaluation evaluator/dependency source provenance changed")
    if actual["evaluator"]["sha256"] != EVALUATOR_SHA256 or actual["evaluator"]["git_blob_oid"] != EVALUATOR_BLOB_OID:
        raise ContractError("frozen evaluator source hash/blob mismatch")
    for item in actual["direct_dependencies"]:
        relative = item["path"]
        if item["sha256"] != DIRECT_EVALUATION_SOURCE_SHA256[relative] or item["git_blob_oid"] != DIRECT_EVALUATION_SOURCE_BLOB_OID[relative]:
            raise ContractError(f"frozen dependency source mismatch: {relative}")
    if actual["mortal_revision"] != MORTAL_REVISION:
        raise ContractError("Mortal revision changed")
    return actual


def validate_frozen_evaluator_object() -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{EVALUATOR_FROZEN_COMMIT}:{EVALUATOR_RELATIVE_PATH}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ContractError("frozen evaluator commit/path is not available")


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"checkpoint is missing: {path}")
    try:
        import torch

        state = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:  # pragma: no cover - artifact-specific failure
        raise ContractError(f"cannot inspect checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ContractError(f"checkpoint is not a mapping: {path}")
    steps = state.get("steps")
    if not isinstance(steps, int):
        raise ContractError(f"checkpoint has no integer steps: {path}")
    training_contract = state.get("training_contract")
    if not isinstance(training_contract, dict):
        raise ContractError(f"checkpoint has no training contract: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "steps": steps,
        "training_contract": training_contract,
        "initialization": state.get("initialization"),
        "cql_min_q_weight": state.get("config", {}).get("cql", {}).get("min_q_weight")
        if isinstance(state.get("config"), dict)
        else None,
    }


def validate_checkpoint(path: Path, expected_sha256: str, *, expected_steps: int = 72000) -> dict[str, Any]:
    record = inspect_checkpoint(path)
    if record["sha256"] != expected_sha256:
        raise ContractError(f"checkpoint SHA mismatch: {path}")
    if record["steps"] != expected_steps:
        raise ContractError(f"checkpoint step mismatch: {path}: {record['steps']}")
    return record


def current_checkpoint_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {
        "70k": file_record(K0_PATH, expected_sha256=K0_SHA256, state="available"),
        "ext_mortal": file_record(EXT_MORTAL_PATH, expected_sha256=EXT_MORTAL_SHA256, state="available"),
    }
    for (route, seed), spec in CURRENT_CHECKPOINTS.items():
        label = current_model_label(route, seed)
        inspected = validate_checkpoint(spec["path"], spec["sha256"])
        records[label] = inspected | {"state": "available"}
    return records


def cql_off_checkpoint_path(training_manifest: dict[str, Any], route: str, seed: int) -> Path:
    for run in training_manifest.get("runs", []):
        if run.get("route") == f"{route}_CQL_OFF" and int(run.get("seed", -1)) == seed:
            run_dir = Path(str(run["run_output_dir"])).resolve()
            return run_dir / "checkpoints/mortal_72000.pth"
    raise ContractError(f"I1 manifest has no CQL_OFF run: {route}/{seed}")


def pending_cql_off_records(training_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for route in ("M0", "D1"):
        for seed in TRAINING_SEEDS:
            label = off_model_label(route, seed)
            path = cql_off_checkpoint_path(training_manifest, route, seed)
            if path.exists():
                raise ContractError(f"CQL_OFF checkpoint exists before authorized training: {path}")
            records[label] = file_record(path, state="pending_training")
    return records


def build_evaluator_argv(
    *,
    condition: str,
    seed: int,
    shard: int,
    models: dict[str, dict[str, Any]],
    output_dir: Path,
    executable: str,
) -> list[str]:
    labels = model_order(condition, seed)
    try:
        specs = [f"{label}={Path(models[label]['path']).resolve()}" for label in labels]
    except KeyError as exc:
        raise ContractError(f"model record missing for {condition}/{seed}: {exc}") from exc
    start = expected_seed_start(seed, shard)
    argv = [
        executable,
        EVALUATOR_RELATIVE_PATH,
        "--mortal-root",
        str((REPO_ROOT / "third_party/Mortal").resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--device",
        "cuda",
        "--require-cuda",
        "--seed-start",
        str(start),
        "--seed-key",
        str(SEED_KEY),
        "--games",
        str(GAMES_PER_SHARD),
        "--seat-mode",
        "random",
        "--native-batch-games",
        str(GAMES_PER_SHARD),
        "--rank-points-profile",
        "tenhou_reference",
    ]
    model_args: list[str] = []
    for spec in specs:
        model_args.extend(("--model", spec))
    return argv[:2] + model_args + argv[2:]


def build_run_matrix(models: dict[str, dict[str, Any]], *, executable: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            for shard in SHARDS:
                start = expected_seed_start(seed, shard)
                output_dir = evaluation_shard_dir(condition, seed, shard)
                argv = build_evaluator_argv(
                    condition=condition,
                    seed=seed,
                    shard=shard,
                    models=models,
                    output_dir=output_dir,
                    executable=executable,
                )
                runs.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "shard": shard,
                        "hanchan_seed_start": start,
                        "hanchan_seed_end_exclusive": start + GAMES_PER_SHARD,
                        "games": GAMES_PER_SHARD,
                        "native_batch_games": GAMES_PER_SHARD,
                        "seed_key": SEED_KEY,
                        "seat_mode": "random",
                        "device": "cuda",
                        "require_cuda": True,
                        "amp": False,
                        "rank_points_profile": "tenhou_reference",
                        "model_order": list(model_order(condition, seed)),
                        "output_dir": str(output_dir.resolve()),
                        "future_argv": argv,
                        "resume": False,
                    }
                )
    if len(runs) != TOTAL_SHARDS:
        raise ContractError("internal C1 evaluation matrix cardinality mismatch")
    return runs


def assert_exact_run_matrix(runs: Iterable[dict[str, Any]], models: dict[str, dict[str, Any]] | None = None) -> None:
    values = list(runs)
    expected_keys = {(condition, seed, shard) for condition in CONDITIONS for seed in TRAINING_SEEDS for shard in SHARDS}
    actual_keys = {(str(row.get("condition")), int(row.get("training_seed", -1)), int(row.get("shard", -1))) for row in values}
    if len(values) != TOTAL_SHARDS or actual_keys != expected_keys:
        raise ContractError(f"C1 evaluation matrix mismatch: {len(values)} rows")
    for row in values:
        condition = str(row["condition"])
        seed = int(row["training_seed"])
        shard = int(row["shard"])
        start = expected_seed_start(seed, shard)
        if row.get("hanchan_seed_start") != start or row.get("hanchan_seed_end_exclusive") != start + GAMES_PER_SHARD:
            raise ContractError(f"seed range mismatch: {condition}/{seed}/{shard}")
        if row.get("games") != GAMES_PER_SHARD or row.get("native_batch_games") != GAMES_PER_SHARD:
            raise ContractError(f"B250 mismatch: {condition}/{seed}/{shard}")
        if row.get("seed_key") != SEED_KEY or row.get("seat_mode") != "random" or row.get("device") != "cuda":
            raise ContractError(f"runtime matrix mismatch: {condition}/{seed}/{shard}")
        if row.get("require_cuda") is not True or row.get("amp") is not False or row.get("resume") is not False:
            raise ContractError(f"fail-closed runtime flags mismatch: {condition}/{seed}/{shard}")
        if tuple(row.get("model_order", ())) != model_order(condition, seed):
            raise ContractError(f"model order mismatch: {condition}/{seed}/{shard}")
        if row.get("rank_points_profile") != "tenhou_reference":
            raise ContractError(f"rank point profile mismatch: {condition}/{seed}/{shard}")
        if models is not None:
            expected_argv = build_evaluator_argv(
                condition=condition,
                seed=seed,
                shard=shard,
                models=models,
                output_dir=Path(str(row["output_dir"])),
                executable=str(row["future_argv"][0]),
            )
            if row.get("future_argv") != expected_argv:
                raise ContractError(f"future evaluator argv mismatch: {condition}/{seed}/{shard}")


def validate_i1_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    if not I1_MANIFEST_PATH.is_file() or not I1_PREFLIGHT_PATH.is_file():
        raise ContractError("approved I1 manifest/preflight is missing")
    if sha256_file(I1_MANIFEST_PATH) != I1_MANIFEST_SHA256:
        raise ContractError("I1 manifest SHA mismatch")
    if sha256_file(I1_PREFLIGHT_PATH) != I1_PREFLIGHT_SHA256:
        raise ContractError("I1 preflight SHA mismatch")
    manifest = load_json(I1_MANIFEST_PATH)
    preflight = load_json(I1_PREFLIGHT_PATH)
    if manifest.get("implementation_commit") != I1_COMMIT or manifest.get("status") != "prepared_not_authorized":
        raise ContractError("I1 manifest is not the approved prepared-not-authorized baseline")
    if manifest.get("training_authorized") is not False or manifest.get("evaluation_authorized") is not False:
        raise ContractError("I1 manifest authorization flags are not false")
    if manifest.get("optimizer_steps") != 0 or manifest.get("new_checkpoints") != 0:
        raise ContractError("I1 manifest records execution")
    if preflight.get("passed") is not True or preflight.get("training_authorized") is not False:
        raise ContractError("I1 preflight is not passed and not-authorized")
    if preflight.get("evaluation_authorized") is not False or preflight.get("optimizer_steps") != 0 or preflight.get("new_checkpoints") != 0:
        raise ContractError("I1 preflight records execution or authorization")
    if preflight.get("manifest_sha256") != I1_MANIFEST_SHA256:
        raise ContractError("I1 preflight does not bind the approved manifest SHA")
    if manifest.get("governance", {}).get("prereg_commit") != PREREG_COMMIT or manifest.get("governance", {}).get("prereg_sha256") != PREREG_SHA256:
        raise ContractError("I1 governance preregistration binding changed")
    if manifest.get("governance", {}).get("registry_sha256") != REGISTRY_SHA256:
        raise ContractError("I1 registry binding changed")
    return manifest, preflight


def validate_governance_files() -> dict[str, str]:
    if not PREREG_PATH.is_file() or sha256_file(PREREG_PATH) != PREREG_SHA256:
        raise ContractError("frozen C1 preregistration SHA mismatch")
    if not REGISTRY_PATH.is_file() or sha256_file(REGISTRY_PATH) != REGISTRY_SHA256:
        raise ContractError("frozen research registry SHA mismatch")
    registry = load_json(REGISTRY_PATH)
    entries = registry.get("records", registry.get("experiments", []))
    matching = [entry for entry in entries if entry.get("experiment_id") == C1_ID]
    if len(matching) != 1 or matching[0].get("status") != "preregistered_frozen":
        raise ContractError("C1 registry registration is not the exact frozen entry")
    return {"prereg_commit": PREREG_COMMIT, "prereg_sha256": PREREG_SHA256, "registry_sha256": REGISTRY_SHA256}


def validate_no_evaluation_outputs(*, allow_plan_and_preflight: bool = True) -> None:
    if not EVALUATION_ROOT.exists():
        return
    allowed = {EVALUATION_PLAN_PATH.name, IMPLEMENTATION_PREFLIGHT_PATH.name} if allow_plan_and_preflight else set()
    unexpected = []
    for path in EVALUATION_ROOT.rglob("*"):
        if (path.is_file() and path.name not in allowed) or (path.is_dir() and path != EVALUATION_ROOT):
            unexpected.append(str(path))
    if unexpected:
        raise ContractError(f"evaluation outputs/logs already exist: {unexpected[:8]}")


def training_completion_runs(closure: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    raw_runs = closure.get("runs")
    if isinstance(raw_runs, list):
        values = raw_runs
    elif isinstance(closure.get("seeds"), dict):
        values = []
        for key, row in closure["seeds"].items():
            if isinstance(row, dict):
                values.append({**row, "seed": row.get("seed", key)})
    else:
        raise ContractError("training completion closure has no runs list")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in values:
        route = str(row.get("route", row.get("training_route", "")))
        seed = int(row.get("training_seed", row.get("seed", -1)))
        if route in {"M0", "D1"}:
            route = f"{route}_CQL_OFF"
        result[(route, seed)] = row
    return result


def resolve_execution_manifest(plan: dict[str, Any], training_completion_closure: dict[str, Any]) -> dict[str, Any]:
    if plan.get("experiment_id") != C1_ID or plan.get("evaluation_authorized") is not False:
        raise ContractError("execution manifest can only resolve from the frozen unauthorized plan")
    rows = training_completion_runs(training_completion_closure)
    expected = {(f"{route}_CQL_OFF", seed) for route in ("M0", "D1") for seed in TRAINING_SEEDS}
    if set(rows) != expected:
        raise ContractError(f"training completion closure matrix mismatch: {set(rows)}")
    plan_models = {str(row["label"]): row for row in plan.get("models", []) if isinstance(row, dict) and row.get("label")}
    resolved: list[dict[str, Any]] = []
    for route, seed in sorted(expected):
        row = rows[(route, seed)]
        required = {
            "steps": 72000,
            "trained_optimizer_steps": 2000,
            "parent_checkpoint_sha256": K0_SHA256,
            "cql_min_q_weight": 0.0,
            "objective": "behavior_action_mc",
            "reward": "final_rank_mc",
            "data_seed": seed,
        }
        for key, expected_value in required.items():
            if row.get(key) != expected_value:
                raise ContractError(f"completion closure mismatch {route}/{seed}: {key}")
        initialization = row.get("initialization", {})
        if initialization.get("optimizer") != "preserved":
            raise ContractError(f"optimizer initialization is not preserved: {route}/{seed}")
        for key in ("scheduler", "scaler", "data_stream"):
            if initialization.get(key) != "fresh":
                raise ContractError(f"completion closure {key} is not fresh: {route}/{seed}")
        label = off_model_label(route.split("_", 1)[0], seed)
        if label not in plan_models:
            raise ContractError(f"plan has no CQL_OFF model record: {label}")
        expected_path = Path(str(plan_models[label]["path"])).resolve()
        checkpoint_path = Path(str(row.get("final_checkpoint_path", ""))).resolve()
        if checkpoint_path != expected_path:
            raise ContractError(f"completion checkpoint path mismatch: {route}/{seed}")
        checkpoint_sha = str(row.get("final_checkpoint_sha256", ""))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
            raise ContractError(f"completion checkpoint SHA mismatch: {route}/{seed}")
        resolved.append(
            {
                "route": route,
                "training_seed": seed,
                "label": label,
                "final_checkpoint_path": str(checkpoint_path),
                "final_checkpoint_sha256": checkpoint_sha,
                "steps": 72000,
                "trained_optimizer_steps": 2000,
                "parent_checkpoint_sha256": K0_SHA256,
                "cql_min_q_weight": 0.0,
                "objective": "behavior_action_mc",
                "reward": "final_rank_mc",
                "initialization": {"optimizer": "preserved", "scheduler": "fresh", "scaler": "fresh", "data_stream": "fresh"},
                "data_seed": seed,
            }
        )
    return {
        "schema": "keqing.mortal.c1_evaluation_execution_manifest.v1",
        "experiment_id": C1_ID,
        "status": "resolved_not_authorized",
        "evaluation_authorized": False,
        "evaluation_games_run": 0,
        "runs": resolved,
    }
