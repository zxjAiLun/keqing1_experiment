"""Protocol and current-lineage checks for the D3 B250 production audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.mortal.d3_exploration_engine import CONTRACT_ID
from training.mortal.d3_production_contract import (
    AMP, AUTHORITATIVE_MORTAL_COMMIT, AUTHORITATIVE_NATIVE_BINARY_SHA256,
    AUTHORITATIVE_NATIVE_PATCH_SHA256, DEVICE, GAMES, GATE_ID, NATIVE_BATCH_GAMES,
    PRODUCTION_IMPLEMENTATION_PATHS, PRODUCTION_SCHEMA, RANK_POINTS, REQUIRED_LABELS, REPO_ROOT,
    SEAT_MODE, SEED_END_EXCLUSIVE, SEED_KEY, SEED_START, implementation_manifest,
    mortal_lineage, project_lineage, sha256_file,
)

def _protocol_checks(protocol: dict[str, Any], run_dir: Path) -> dict[str, bool]:
    fixed = protocol.get("fixed_protocol", {})
    project = protocol.get("project_lineage", {})
    native = protocol.get("mortal_lineage", {})
    runtime = protocol.get("runtime", {})
    models = protocol.get("models", {})
    archives = protocol.get("lineage_archives", {})
    authoritative = protocol.get("authoritative_smoke", {})
    ignored_artifacts = protocol.get("ignored_artifacts", {})
    production_output = ignored_artifacts.get("production_output", {})
    production_implementation = protocol.get("production_implementation", {})
    final_guard = protocol.get("final_call_guard", {})

    def artifact_matches(path_key: str, sha_key: str) -> bool:
        raw_path = archives.get(path_key)
        expected_sha = archives.get(sha_key)
        if not raw_path or not expected_sha:
            return False
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        return path.is_file() and sha256_file(path) == expected_sha

    return {
        "schema": protocol.get("schema") == PRODUCTION_SCHEMA,
        "gate_id": protocol.get("gate_id") == GATE_ID,
        "contract_id": protocol.get("contract_id") == CONTRACT_ID,
        "generation_completed": protocol.get("status") == "generation_completed_audit_pending",
        "final_call_guard_passed": final_guard.get("passed") is True,
        "final_guard_project_commit": final_guard.get("project_lineage", {}).get("commit")
        == project.get("commit"),
        "final_guard_mortal_commit": final_guard.get("mortal_lineage", {}).get("commit")
        == native.get("commit"),
        "final_guard_native_patch": final_guard.get("d3_native_patch_sha256")
        == protocol.get("d3_native_patch", {}).get("sha256"),
        "final_guard_implementation": final_guard.get("production_implementation")
        == production_implementation,
        "final_guard_models": all(
            final_guard.get("model_sha256", {}).get(label) == models.get(label, {}).get("sha256")
            for label in REQUIRED_LABELS
        ),
        "authoritative_project_commit": authoritative.get("project_commit")
        == "7eb48f1310e917cfb2f1f45445e546b4e92d1a89",
        "authoritative_mortal_commit": authoritative.get("mortal_source_commit")
        == AUTHORITATIVE_MORTAL_COMMIT,
        "authoritative_native_patch": authoritative.get("d3_native_patch_sha256")
        == AUTHORITATIVE_NATIVE_PATCH_SHA256,
        "authoritative_native_binary": authoritative.get("loaded_libriichi_sha256")
        == AUTHORITATIVE_NATIVE_BINARY_SHA256,
        "seed_start": fixed.get("seed_start") == SEED_START,
        "seed_end": fixed.get("seed_end_exclusive") == SEED_END_EXCLUSIVE,
        "seed_key": fixed.get("seed_key") == SEED_KEY,
        "games": fixed.get("games") == GAMES,
        "native_batch_games": fixed.get("native_batch_games") == NATIVE_BATCH_GAMES,
        "native_call_count": fixed.get("native_call_count") == 1,
        "seat_mode": fixed.get("seat_mode") == SEAT_MODE,
        "device": fixed.get("device") == DEVICE,
        "amp_false": fixed.get("amp") is AMP,
        "resume_forbidden": fixed.get("resume_supported") is False,
        "no_auto_continue": fixed.get("auto_continue_remaining_5750") is False,
        "production_output_ignored_or_external": production_output.get("external_to_repo") is True
        or production_output.get("ignored") is True,
        "rank_points": fixed.get("rank_points") == list(RANK_POINTS),
        "project_clean": project.get("dirty") is False,
        "project_branch": project.get("branch") == "codex/mortal-training-next",
        "project_semantic_unchanged": project.get("semantic_diff_paths") == [],
        "mortal_clean": native.get("dirty") is False,
        "mortal_commit": native.get("commit") == AUTHORITATIVE_MORTAL_COMMIT,
        "native_patch": protocol.get("d3_native_patch", {}).get("sha256")
        == AUTHORITATIVE_NATIVE_PATCH_SHA256,
        "native_binary": runtime.get("loaded_libriichi_sha256")
        == AUTHORITATIVE_NATIVE_BINARY_SHA256,
        "native_release": runtime.get("native_build_profile") == "release",
        "cuda_available": runtime.get("cuda_available") is True,
        "model_labels": set(models) == set(REQUIRED_LABELS),
        "engine_order": protocol.get("engine_order") == list(REQUIRED_LABELS),
        "production_implementation_manifest": set(production_implementation)
        == set(PRODUCTION_IMPLEMENTATION_PATHS),
        "model_manifest_matches_smoke": all(
            models.get(label, {}).get("sha256")
            == authoritative.get("models", {}).get(label, {}).get("sha256")
            for label in REQUIRED_LABELS
        ),
        "format_patch_saved": artifact_matches("format_patch", "format_patch_sha256"),
    }

def _current_lineage_checks(protocol: dict[str, Any], mortal_root: Path) -> dict[str, Any]:
    from libriichi import _riichi  # noqa: PLC0415

    project = project_lineage(REPO_ROOT)
    native = mortal_lineage(mortal_root)
    model_errors: list[str] = []
    for label in REQUIRED_LABELS:
        row = protocol.get("models", {}).get(label, {})
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            model_errors.append(f"model missing after generation: {label} -> {path}")
        elif sha256_file(path) != row.get("sha256"):
            model_errors.append(f"model SHA changed after generation: {label}")
    expected_project_commit = protocol.get("project_lineage", {}).get("commit")
    project_commit_exact = project["commit"] == expected_project_commit
    loaded_binary_path = Path(_riichi.__file__).resolve()
    loaded_binary_sha = sha256_file(loaded_binary_path)
    binary_exact = loaded_binary_sha == protocol.get("runtime", {}).get(
        "loaded_libriichi_sha256"
    ) == AUTHORITATIVE_NATIVE_BINARY_SHA256
    patch_path = REPO_ROOT / "scripts/mortal/patches/libriichi_d3_decision_context.patch"
    patch_sha = sha256_file(patch_path)
    patch_exact = patch_sha == protocol.get("d3_native_patch", {}).get(
        "sha256"
    ) == AUTHORITATIVE_NATIVE_PATCH_SHA256
    smoke_row = protocol.get("authoritative_smoke", {})
    smoke_protocol_path = Path(str(smoke_row.get("protocol_path", "")))
    smoke_protocol_exact = (
        smoke_protocol_path.is_file()
        and sha256_file(smoke_protocol_path) == smoke_row.get("protocol_sha256")
    )
    implementation_error: str | None = None
    try:
        current_implementation = implementation_manifest(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001
        current_implementation = {}
        implementation_error = str(exc)
    implementation_exact = current_implementation == protocol.get("production_implementation")
    extra_errors: list[str] = []
    if not project_commit_exact:
        extra_errors.append(
            f"project HEAD changed after generation: expected={expected_project_commit} "
            f"actual={project['commit']}"
        )
    if not binary_exact:
        extra_errors.append(f"loaded native binary changed after generation: {loaded_binary_sha}")
    if not patch_exact:
        extra_errors.append(f"D3 native patch changed after generation: {patch_sha}")
    if not smoke_protocol_exact:
        extra_errors.append(
            f"authoritative smoke protocol missing or changed after generation: {smoke_protocol_path}"
        )
    if not implementation_exact:
        extra_errors.append(
            "production runner/auditor implementation changed after generation"
            + (f": {implementation_error}" if implementation_error else "")
        )
    return {
        "project": project,
        "mortal": native,
        "project_commit_exact": project_commit_exact,
        "loaded_libriichi_path": str(loaded_binary_path),
        "loaded_libriichi_sha256": loaded_binary_sha,
        "native_binary_exact": binary_exact,
        "d3_native_patch_sha256": patch_sha,
        "native_patch_exact": patch_exact,
        "authoritative_smoke_protocol_path": str(smoke_protocol_path),
        "authoritative_smoke_protocol_exact": smoke_protocol_exact,
        "production_implementation": current_implementation,
        "production_implementation_exact": implementation_exact,
        "model_errors": model_errors,
        "extra_errors": extra_errors,
        "passed": (
            project["passed"]
            and native["passed"]
            and project_commit_exact
            and binary_exact
            and patch_exact
            and smoke_protocol_exact
            and implementation_exact
            and not model_errors
        ),
    }
