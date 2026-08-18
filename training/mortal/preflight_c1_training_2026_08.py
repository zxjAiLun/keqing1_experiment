#!/usr/bin/env python3
"""Fail-closed C1-I1 preflight; never performs an optimizer step."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from training.mortal.prepare_c1_training_2026_08 import (
    C1_ID,
    LOADER_SHA256,
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
    load_file_index,
    load_governance,
    load_json,
    load_toml,
    map_external_pattern,
    map_external_value,
    sha256_file,
    validate_generated_config,
    validate_source_inputs,
)

ZERO_STEP_SCRIPT = REPO_ROOT / "training/mortal/zero_step_mortal_smoke_2026_07.py"
EXPECTED_UNTRACKED = ["1.md"]


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
    if current["branch"] != "main" or current["tracked_changes"] or current["untracked"] != EXPECTED_UNTRACKED:
        raise ContractError(f"preflight requires main with only 1.md untracked: {current}")
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


def materialize_runtime_inputs(
    *,
    run: dict[str, Any],
    config_path: Path,
    runtime_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create path-mapped smoke-only inputs without changing the C1 config."""

    import toml

    route = str(run["route"])
    seed = int(run["seed"])
    config = load_toml(config_path)
    source_index = Path(map_external_value(str(config["dataset"]["file_index"])))
    payload, file_list = load_file_index(source_index)
    runtime_payload = copy.deepcopy(payload)
    runtime_file_list = [map_external_value(value) for value in file_list]
    missing = next((value for value in runtime_file_list if not Path(value).is_file()), None)
    if missing is not None:
        raise ContractError(f"runtime path mapping points to missing file: {missing}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_index = runtime_root / f"{route}_seed_{seed}_file_index.pth"
    runtime_payload["file_list"] = runtime_file_list
    torch.save(runtime_payload, runtime_index)
    runtime_config = copy.deepcopy(config)
    runtime_config["dataset"]["file_index"] = str(runtime_index.resolve())
    runtime_config["dataset"]["globs"] = [
        map_external_pattern(str(value)) for value in config["dataset"]["globs"]
    ]
    runtime_config["dataset"]["player_names_files"] = [
        map_external_value(str(value)) for value in config["dataset"]["player_names_files"]
    ]
    runtime_config_path = runtime_root / f"{route}_seed_{seed}_runtime.toml"
    runtime_config_path.write_text(toml.dumps(runtime_config), encoding="utf-8")
    return runtime_config_path, {
        "runtime_config": str(runtime_config_path.resolve()),
        "runtime_file_index": str(runtime_index.resolve()),
        "runtime_file_index_sha256": sha256_file(runtime_index),
        "source_file_index": str(source_index.resolve()),
        "source_file_index_sha256": sha256_file(source_index),
        "mapped_samples": len(runtime_file_list),
    }


def run_zero_step_smoke(
    *,
    runtime_config: Path,
    parent: Path,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(ZERO_STEP_SCRIPT),
            "--config",
            str(runtime_config),
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
    runtime_records: list[dict[str, Any]] = []
    for route in ROUTES:
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
            if source_inputs["file_index_sha256"] != run["file_index_sha256"]:
                raise ContractError(f"file index binding mismatch: {route}/{seed}")
            if source_inputs["label_files"] != run["label_files"]:
                raise ContractError(f"label binding mismatch: {route}/{seed}")
            semantic = validate_generated_config(
                source,
                generated,
                route=route,
                seed=seed,
                run_dir=Path(run["run_output_dir"]),
                source_sha256=run["source_current_config_sha256"],
            )
            if semantic != run["semantic_diff"]:
                raise ContractError(f"semantic diff manifest mismatch: {route}/{seed}")
            assert_no_training_outputs(Path(run["run_output_dir"]))
            runtime_config, runtime_record = materialize_runtime_inputs(
                run=run,
                config_path=generated_path,
                runtime_root=preflight_dir / "runtime_inputs",
            )
            smoke_output = preflight_dir / f"zero_step_{route}_seed_{seed}.json"
            smoke = run_zero_step_smoke(
                runtime_config=runtime_config,
                parent=parent_path,
                seed=seed,
                output=smoke_output,
            )
            if smoke.get("samples") != 512 or smoke.get("objective", {}).get("mode") != "behavior_action_mc":
                raise ContractError(f"zero-step sample/objective mismatch: {route}/{seed}")
            if not smoke.get("finite") or not bool(smoke.get("losses", {}).get("cql_loss") is not None):
                raise ContractError(f"zero-step finite/cql loss missing: {route}/{seed}")
            if smoke.get("optimizer_step_performed") is not False or smoke.get("state_file_written") is not False:
                raise ContractError(f"zero-step execution boundary failed: {route}/{seed}")
            assert_no_training_outputs(Path(run["run_output_dir"]))
            runtime_records.append({"route": route, "seed": seed, **runtime_record})
            zero_step_reports.append(
                {
                    "route": route,
                    "seed": seed,
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
        "runtime_inputs": runtime_records,
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
