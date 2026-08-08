"""Preflight and final-call guards for the frozen D3 production B250 gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_exploration_engine import CONTRACT_ID
from training.mortal.d3_production_contract import (
    AMP, AUTHORITATIVE_NATIVE_BINARY_SHA256, AUTHORITATIVE_NATIVE_PATCH_SHA256,
    DEVICE, GAMES, GATE_ID, NATIVE_BATCH_GAMES, RANK_POINTS, REQUIRED_LABELS,
    SEAT_MODE, SEED_END_EXCLUSIVE, SEED_KEY, SEED_START, ContractError,
    assert_empty_output, ignored_path_audit, implementation_manifest,
    load_authoritative_smoke_protocol, mortal_lineage, parse_model_specs,
    project_lineage, sha256_file,
)

def _runtime_preflight() -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from libriichi import _riichi  # noqa: PLC0415

    extension_path = Path(_riichi.__file__).resolve()
    extension_sha = sha256_file(extension_path)
    errors: list[str] = []
    if not torch.cuda.is_available():
        errors.append("CUDA required but torch.cuda.is_available() is False")
    if extension_sha != AUTHORITATIVE_NATIVE_BINARY_SHA256:
        errors.append(
            "loaded libriichi binary differs from authoritative smoke: "
            f"expected={AUTHORITATIVE_NATIVE_BINARY_SHA256} actual={extension_sha}"
        )
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": DEVICE,
        "amp": AMP,
        "loaded_libriichi_path": str(extension_path),
        "loaded_libriichi_sha256": extension_sha,
        "native_build_profile": "release",
        "errors": errors,
        "passed": not errors,
    }

def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    assert_empty_output(output_dir)
    if "generation_smoke" in output_dir.parts:
        raise ContractError(f"production output cannot reuse a smoke directory: {output_dir}")
    smoke_path = args.authoritative_smoke_protocol.resolve()
    smoke_protocol, smoke_protocol_sha = load_authoritative_smoke_protocol(smoke_path)
    models = parse_model_specs(
        args.model,
        authoritative_models=smoke_protocol["models"],
        repo_root=REPO_ROOT,
    )
    project = project_lineage(REPO_ROOT)
    native = mortal_lineage(args.mortal_root.resolve())
    runtime = _runtime_preflight()
    patch_path = REPO_ROOT / "scripts/mortal/patches/libriichi_d3_decision_context.patch"
    patch_sha = sha256_file(patch_path)
    ignored_artifacts = ignored_path_audit(
        {
            "production_output": output_dir,
            "authoritative_smoke_protocol": smoke_path,
            "nested_mortal_root": args.mortal_root.resolve(),
            **{f"model_{label}": path for label, path in models.items()},
        }
    )
    errors = [*project["errors"], *native["errors"], *runtime["errors"]]
    output_ignore = ignored_artifacts["production_output"]
    if not output_ignore["external_to_repo"] and output_ignore["ignored"] is not True:
        errors.append("production output inside the repository must be covered by .gitignore")
    if patch_sha != AUTHORITATIVE_NATIVE_PATCH_SHA256:
        errors.append(
            "D3 native patch differs from authoritative smoke: "
            f"expected={AUTHORITATIVE_NATIVE_PATCH_SHA256} actual={patch_sha}"
        )
    preflight = {
        "schema": "keqing.mortal.d3_production_gate_preflight.v1",
        "gate_id": GATE_ID,
        "contract_id": CONTRACT_ID,
        "output_dir": str(output_dir),
        "authoritative_smoke": {
            "protocol_path": str(smoke_path),
            "protocol_sha256": smoke_protocol_sha,
            "project_commit": smoke_protocol["project_git_commit"],
            "mortal_source_commit": smoke_protocol["mortal_source_commit"],
            "d3_native_patch_sha256": smoke_protocol["d3_native_patch_sha256"],
            "loaded_libriichi_sha256": smoke_protocol["loaded_libriichi_sha256"],
            "models": smoke_protocol["models"],
        },
        "project_lineage": project,
        "mortal_lineage": native,
        "runtime": runtime,
        "d3_native_patch": {"path": str(patch_path), "sha256": patch_sha},
        "production_implementation": implementation_manifest(REPO_ROOT),
        "ignored_artifacts": ignored_artifacts,
        "models": {
            label: {
                "path": str(models[label]),
                "sha256": sha256_file(models[label]),
            }
            for label in REQUIRED_LABELS
        },
        "execution_shape": {
            "seed_start": SEED_START,
            "seed_end_exclusive": SEED_END_EXCLUSIVE,
            "seed_key": SEED_KEY,
            "games": GAMES,
            "native_batch_games": NATIVE_BATCH_GAMES,
            "seat_mode": SEAT_MODE,
            "device": DEVICE,
            "amp": AMP,
            "rank_points": list(RANK_POINTS),
            "native_call_count": 1,
            "resume_supported": False,
            "auto_continue_remaining_5750": False,
        },
        "errors": errors,
        "passed": not errors,
    }
    return preflight

def final_call_guard(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    """Recheck mutable inputs immediately before the sole native B250 call."""

    project = project_lineage(REPO_ROOT)
    native = mortal_lineage(args.mortal_root.resolve())
    errors = [*project["errors"], *native["errors"]]
    if project["commit"] != preflight["project_lineage"]["commit"]:
        errors.append("project commit changed after initial preflight")
    if native["commit"] != preflight["mortal_lineage"]["commit"]:
        errors.append("nested Mortal commit changed after initial preflight")
    patch_sha = sha256_file(Path(preflight["d3_native_patch"]["path"]))
    if patch_sha != preflight["d3_native_patch"]["sha256"]:
        errors.append("D3 native patch changed after initial preflight")
    current_implementation = implementation_manifest(REPO_ROOT)
    if current_implementation != preflight["production_implementation"]:
        errors.append("production runner/auditor implementation changed after initial preflight")
    model_shas: dict[str, str] = {}
    for label in REQUIRED_LABELS:
        row = preflight["models"][label]
        actual_sha = sha256_file(Path(row["path"]))
        model_shas[label] = actual_sha
        if actual_sha != row["sha256"]:
            errors.append(f"model changed after initial preflight: {label}")
    return {
        "project_lineage": project,
        "mortal_lineage": native,
        "d3_native_patch_sha256": patch_sha,
        "production_implementation": current_implementation,
        "model_sha256": model_shas,
        "errors": errors,
        "passed": not errors,
    }
