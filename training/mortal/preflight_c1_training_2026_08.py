#!/usr/bin/env python3
"""Fail-closed C1-I1 preflight; never performs an optimizer step."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support both ``python -m ...`` and direct execution by path.
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.prepare_c1_training_2026_08 import (
    C1_ID,
    LOADER_SHA256,
    LOADER_STREAM_SHA256,
    PREREG_COMMIT,
    PREREG_SHA256,
    REGISTRY_SHA256,
    REPO_ROOT,
    ROUTES,
    SEEDS,
    SOURCE_CONFIG_SHA256,
    ContractError,
    git_info,
    implementation_sources,
    inspect_parent,
    load_governance,
    load_json,
    load_toml,
    map_dataset_paths,
    sha256_file,
    validate_generated_config,
    validate_git_scope,
    validate_runtime_file_index,
    validate_source_inputs,
)

ZERO_STEP_SCRIPT = REPO_ROOT / "training/mortal/zero_step_mortal_smoke_2026_07.py"


def assert_run_matrix(runs: list[dict[str, Any]]) -> None:
    expected = {(route, seed) for route in ROUTES for seed in SEEDS}
    actual = {(str(run.get("route")), int(run.get("seed"))) for run in runs}
    if actual != expected or len(runs) != len(expected):
        raise ContractError(f"C1 run matrix mismatch: expected={expected}, actual={actual}")


def assert_no_training_outputs(run_dir: Path) -> None:
    forbidden = [run_dir / "mortal.pth", run_dir / "mortal_best.pth"]
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        forbidden.extend(checkpoint_dir.glob("mortal_*.pth"))
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ContractError(f"training output/checkpoint exists before preflight: {existing}")


def verify_git_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    current = git_info()
    validate_git_scope(current)
    if current["commit"] != manifest.get("implementation_commit"):
        raise ContractError(
            f"manifest implementation commit does not match HEAD: {manifest.get('implementation_commit')} != {current['commit']}"
        )
    if manifest.get("git", {}).get("commit") != current["commit"]:
        raise ContractError("manifest git commit does not match current HEAD")
    if manifest.get("git", {}).get("tracked_clean") is not True:
        raise ContractError("manifest was not prepared from a clean tracked tree")
    current_sources = {item["path"]: item for item in implementation_sources()}
    for item in manifest.get("implementation_sources", []):
        current_item = current_sources.get(item.get("path"))
        if current_item != item:
            raise ContractError(f"implementation source binding mismatch: {item.get('path')}")
    return current


def run_zero_step_smoke(
    *,
    config_path: Path,
    parent: Path,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(ZERO_STEP_SCRIPT),
            "--config",
            str(config_path),
            "--parent",
            str(parent),
            "--data-seed",
            str(seed),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return load_json(output)


def validate_manifest_contract(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "keqing.mortal.c1_training_manifest.v1":
        raise ContractError("unsupported C1 training manifest schema")
    if manifest.get("experiment_id") != C1_ID or manifest.get("status") != "prepared_not_authorized":
        raise ContractError("manifest is not the prepared-not-authorized C1 contract")
    if manifest.get("training_authorized") is not False or manifest.get("evaluation_authorized") is not False:
        raise ContractError("manifest authorization flags must be false")
    if manifest.get("optimizer_steps") != 0 or manifest.get("new_checkpoints") != 0:
        raise ContractError("manifest records non-zero execution")
    if manifest.get("governance", {}).get("prereg_commit") != PREREG_COMMIT:
        raise ContractError("manifest prereg commit mismatch")
    fixed = manifest.get("fixed_contract", {})
    if fixed.get("factor_b") != {"CURRENT": 5.0, "CQL_OFF": 0.0}:
        raise ContractError("manifest CQL weights mismatch")
    if fixed.get("new_training_runs") != 6 or fixed.get("start_step") != 70000 or fixed.get("target_step") != 72000:
        raise ContractError("manifest run/step contract mismatch")
    if manifest.get("new_training_runs") != 6:
        raise ContractError("manifest new_training_runs mismatch")
    if manifest.get("training_command_policy", {}).get("authoritative_field") != "future_training_argv":
        raise ContractError("manifest does not bind authoritative argv")
    if manifest.get("training_command_policy", {}).get("shell") is not False:
        raise ContractError("manifest shell policy is not false")
    if manifest.get("execution_boundary", {}).get("formal_config_equals_smoke_config") is not True:
        raise ContractError("manifest formal/smoke config binding is not exact")
    runtime_inputs = manifest.get("runtime_inputs", {})
    if set(runtime_inputs) != {"M0", "D1"}:
        raise ContractError("manifest runtime index route matrix mismatch")
    assert_run_matrix(manifest.get("runs", []))
    return manifest, load_json(REPO_ROOT / "training/docs/mortal/research_registry.json"), load_json(
        REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/loader_compatibility.json"
    )


def run_preflight(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = manifest_path.parent
    preflight_dir = output_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    manifest, _registry, loader = validate_manifest_contract(manifest_path)
    current_git = verify_git_binding(manifest)
    governance = load_governance(
        REPO_ROOT / "training/docs/mortal/research_registry.json",
        REPO_ROOT / "training/docs/mortal/experiments_zh/2026-08_C1语料_CQL交互因果实验_预注册设计.md",
        REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/loader_compatibility.json",
    )
    if manifest.get("governance", {}).get("registry_sha256") != REGISTRY_SHA256:
        raise ContractError("manifest registry SHA mismatch")
    if manifest.get("governance", {}).get("loader_report_sha256") != LOADER_SHA256:
        raise ContractError("manifest loader SHA mismatch")
    if governance["loader_contract"] != manifest["governance"]["loader_contract"]:
        raise ContractError("manifest loader contract differs from authoritative report")
    parent_path = Path(manifest["parent"]["path"]).resolve()
    parent = inspect_parent(parent_path)
    if parent["digest"] != manifest["parent"]["digest"] or not parent["optimizer_moments_covered"]:
        raise ContractError("K0 parent digest or preserved Adam moments mismatch")
    runs_by_key = {(str(run["route"]), int(run["seed"])): run for run in manifest["runs"]}
    zero_step_reports: list[dict[str, Any]] = []
    for route in ROUTES:
        route_key = route.split("_", maxsplit=1)[0]
        runtime_spec = manifest["runtime_inputs"][route_key]
        for seed in SEEDS:
            run = runs_by_key[(route, seed)]
            source_path = Path(run["source_current_config"]).resolve()
            generated_path = Path(run["cql_off_config"]).resolve()
            if sha256_file(source_path) != SOURCE_CONFIG_SHA256[(route, seed)]:
                raise ContractError(f"source config SHA mismatch during preflight: {route}/{seed}")
            if sha256_file(generated_path) != run["cql_off_config_sha256"]:
                raise ContractError(f"generated config SHA mismatch during preflight: {route}/{seed}")
            source = load_toml(source_path)
            generated = load_toml(generated_path)
            source_inputs = validate_source_inputs(source, route=route, seed=seed)
            validate_runtime_file_index(
                runtime_spec,
                expected_source_path=Path(source_inputs["file_index_path"]),
                expected_source_sha256=source_inputs["file_index_sha256"],
            )
            if source_inputs["file_index_sha256"] != run["source_file_index_sha256"]:
                raise ContractError(f"file index binding mismatch: {route}/{seed}")
            if source_inputs["label_files"] != run["label_files"]:
                raise ContractError(f"label binding mismatch: {route}/{seed}")
            if run.get("runtime_file_index_path") != runtime_spec["runtime_file_index_path"]:
                raise ContractError(f"runtime file index route binding mismatch: {route}/{seed}")
            if run.get("runtime_file_index_sha256") != runtime_spec["runtime_file_index_sha256"]:
                raise ContractError(f"runtime file index SHA binding mismatch: {route}/{seed}")
            if run.get("file_count") != 6000 or run.get("ordered_path_mapping_sha256") != runtime_spec[
                "ordered_path_mapping_sha256"
            ]:
                raise ContractError(f"runtime file index mapping digest mismatch: {route}/{seed}")
            runtime_dataset = map_dataset_paths(source)
            runtime_dataset["file_index"] = runtime_spec["runtime_file_index_path"]
            if run.get("runtime_dataset") != runtime_dataset:
                raise ContractError(f"runtime dataset relocation binding mismatch: {route}/{seed}")
            semantic = validate_generated_config(
                source,
                generated,
                route=route,
                seed=seed,
                run_dir=Path(run["run_output_dir"]),
                source_sha256=run["source_current_config_sha256"],
                runtime_dataset=runtime_dataset,
            )
            if semantic != run["semantic_diff"]:
                raise ContractError(f"semantic diff manifest mismatch: {route}/{seed}")
            formal_config_sha256 = sha256_file(generated_path)
            if run.get("formal_training_config_sha256") != formal_config_sha256:
                raise ContractError(f"formal training config SHA mismatch: {route}/{seed}")
            if run.get("smoke_config_sha256") != formal_config_sha256 or run.get("exact_same_config") is not True:
                raise ContractError(f"formal/smoke config SHA binding mismatch: {route}/{seed}")
            if run.get("loader_stream_sha256") != LOADER_STREAM_SHA256[(route_key, seed)]:
                raise ContractError(f"loader stream provenance mismatch: {route}/{seed}")
            assert_no_training_outputs(Path(run["run_output_dir"]))
            smoke_output = preflight_dir / f"zero_step_{route}_seed_{seed}.json"
            smoke = run_zero_step_smoke(
                config_path=generated_path,
                parent=parent_path,
                seed=seed,
                output=smoke_output,
            )
            if Path(smoke.get("config", "")).resolve() != generated_path:
                raise ContractError(f"zero-step used a non-formal config: {route}/{seed}")
            if smoke.get("samples") != 512 or smoke.get("objective", {}).get("mode") != "behavior_action_mc":
                raise ContractError(f"zero-step sample/objective mismatch: {route}/{seed}")
            if not smoke.get("finite") or not bool(smoke.get("losses", {}).get("cql_loss") is not None):
                raise ContractError(f"zero-step finite/cql loss missing: {route}/{seed}")
            if smoke.get("optimizer_step_performed") is not False or smoke.get("state_file_written") is not False:
                raise ContractError(f"zero-step execution boundary failed: {route}/{seed}")
            assert_no_training_outputs(Path(run["run_output_dir"]))
            zero_step_reports.append(
                {
                    "route": route,
                    "seed": seed,
                    "smoke_config_sha256": formal_config_sha256,
                    "formal_training_config_sha256": formal_config_sha256,
                    "exact_same_config": True,
                    "samples": smoke["samples"],
                    "finite": smoke["finite"],
                    "cql_loss": smoke["losses"].get("cql_loss"),
                    "optimizer_step_performed": smoke["optimizer_step_performed"],
                    "state_file_written": smoke["state_file_written"],
                    "report": str(smoke_output.resolve()),
                }
            )
    report = {
        "schema": "keqing.mortal.c1_training_preflight.v1",
        "passed": True,
        "experiment_id": C1_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git": current_git,
        "governance": {
            "prereg_commit": PREREG_COMMIT,
            "prereg_sha256": PREREG_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "loader_report_sha256": LOADER_SHA256,
            "loader_status": loader["status"],
            "resolved_training_count": 6,
            "streams_exact_match": True,
        },
        "parent": parent,
        "runtime_inputs": list(manifest["runtime_inputs"].values()),
        "zero_step_smokes": zero_step_reports,
        "training_authorized": False,
        "evaluation_authorized": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "checks": {
            "governance_exact": True,
            "loader_six_streams_pass": True,
            "k0_sha_and_step_exact": True,
            "preserved_adam_complete": True,
            "six_source_config_sha_exact": True,
            "six_cql_off_config_sha_exact": True,
            "semantic_diff_gate_six_runs": True,
            "runtime_index_relocation_exact": True,
            "formal_config_equals_smoke_config": True,
            "output_safety": True,
            "implementation_sources_exact": True,
            "six_zero_step_cuda_smokes": True,
            "training_not_authorized": True,
        },
    }
    report_path = preflight_dir / "training_preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["preflight_report"] = str(report_path.resolve())
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/training_implementation_2026_08/training_manifest.json",
    )
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.manifest)
    except Exception as exc:
        output_dir = args.manifest.resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "keqing.mortal.c1_training_preflight.v1",
            "passed": False,
            "experiment_id": C1_ID,
            "training_authorized": False,
            "evaluation_authorized": False,
            "optimizer_steps": 0,
            "new_checkpoints": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report_path = output_dir / "preflight/training_preflight.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
