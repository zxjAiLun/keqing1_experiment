#!/usr/bin/env python3
"""Run the C1-I2 implementation preflight without executing evaluation.

This is a passing implementation preflight while training and evaluation are
still unauthorized.  It verifies the exact matrix, evaluator/runtime/source
provenance, approved I1 baseline, existing CURRENT checkpoints, and absence of
all six future CQL_OFF checkpoints and evaluation outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.c1_evaluation_contract_2026_08 import (
    C1_ID,
    EVALUATION_PLAN_PATH,
    I1_COMMIT,
    IMPLEMENTATION_PREFLIGHT_PATH,
    K0_SHA256,
    ContractError,
    assert_exact_run_matrix,
    current_checkpoint_records,
    dump_json,
    git_info,
    load_json,
    pending_cql_off_records,
    sha256_file,
    source_provenance,
    validate_frozen_evaluator_object,
    validate_git_scope,
    validate_governance_files,
    validate_i1_baseline,
    validate_no_evaluation_outputs,
    validate_runtime_provenance,
    validate_source_provenance,
)


def _models_by_label(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = plan.get("models")
    if not isinstance(models, list):
        raise ContractError("evaluation plan has no model list")
    values: dict[str, dict[str, Any]] = {}
    for item in models:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ContractError("evaluation plan contains an invalid model record")
        label = str(item["label"])
        if label in values:
            raise ContractError(f"duplicate model label in evaluation plan: {label}")
        values[label] = item
    return values


def _validate_model_bindings(plan: dict[str, Any], i1_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    planned = _models_by_label(plan)
    live_current = current_checkpoint_records()
    live_pending = pending_cql_off_records(i1_manifest)
    if set(planned) != set(live_current) | set(live_pending):
        raise ContractError("evaluation plan model label set differs from frozen 14-model set")
    for label, live in {**live_current, **live_pending}.items():
        item = planned[label]
        if Path(str(item.get("path"))).resolve() != Path(str(live["path"])).resolve():
            raise ContractError(f"evaluation plan model path mismatch: {label}")
        if item.get("sha256") != live.get("sha256"):
            raise ContractError(f"evaluation plan model SHA mismatch: {label}")
        expected_state = "available" if label in live_current else "pending_training"
        if item.get("state") != expected_state:
            raise ContractError(f"evaluation plan model state mismatch: {label}")
        if label in live_current and label not in {"70k", "ext_mortal"} and item.get("steps") != 72000:
            raise ContractError(f"CURRENT checkpoint step binding mismatch: {label}")
    if planned["70k"].get("sha256") != K0_SHA256:
        raise ContractError("K0 model binding mismatch")
    return planned


def validate_plan(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path = plan_path.resolve()
    if plan_path != EVALUATION_PLAN_PATH.resolve():
        raise ContractError("C1-I2 plan path is frozen")
    plan = load_json(plan_path)
    if plan.get("schema") != "keqing.mortal.c1_evaluation_plan.v1" or plan.get("experiment_id") != C1_ID:
        raise ContractError("unsupported C1-I2 evaluation plan")
    if plan.get("status") != "prepared_not_authorized":
        raise ContractError("evaluation plan is not prepared_not_authorized")
    for key in ("evaluation_authorized", "training_authorized", "training_started", "evaluation_started"):
        if plan.get(key) is not False:
            raise ContractError(f"evaluation plan authorization/state flag is not false: {key}")
    if plan.get("execution_ready") is not False or plan.get("evaluation_games_run") != 0:
        raise ContractError("evaluation plan records execution readiness or games")
    if plan.get("optimizer_steps") != 0 or plan.get("new_checkpoints") != 0:
        raise ContractError("evaluation plan records optimizer/checkpoint output")
    if plan.get("pending_cql_off_checkpoints") != 6:
        raise ContractError("evaluation plan pending CQL_OFF count is not six")
    i1_manifest, i1_preflight = validate_i1_baseline()
    if plan.get("i1_baseline", {}).get("commit") != I1_COMMIT:
        raise ContractError("evaluation plan I1 commit binding mismatch")
    if plan.get("i1_baseline", {}).get("manifest", {}).get("sha256") != sha256_file(
        Path(plan["i1_baseline"]["manifest"]["path"])
    ):
        raise ContractError("evaluation plan I1 manifest SHA binding mismatch")
    if plan.get("i1_baseline", {}).get("preflight", {}).get("sha256") != sha256_file(
        Path(plan["i1_baseline"]["preflight"]["path"])
    ):
        raise ContractError("evaluation plan I1 preflight SHA binding mismatch")
    planned_models = _validate_model_bindings(plan, i1_manifest)
    assert_exact_run_matrix(plan.get("runs", []), planned_models)
    return plan, i1_manifest, i1_preflight


def run_preflight(plan_path: Path = EVALUATION_PLAN_PATH) -> dict[str, Any]:
    plan, i1_manifest, i1_preflight = validate_plan(plan_path)
    validate_governance_files()
    validate_frozen_evaluator_object()
    current_sources = source_provenance()
    if current_sources != {
        "evaluator": plan["evaluator_provenance"],
        "direct_dependencies": plan["evaluation_dependency_sources"],
        "mortal_revision": plan["mortal_revision"],
    }:
        raise ContractError("evaluation source provenance differs from plan")
    validate_source_provenance(
        {
            "evaluator": plan["evaluator_provenance"],
            "direct_dependencies": plan["evaluation_dependency_sources"],
            "mortal_revision": plan["mortal_revision"],
        }
    )
    current_runtime = validate_runtime_provenance(i1_manifest["runtime_provenance"])
    if current_runtime != plan["runtime_provenance"]:
        raise ContractError("runtime provenance differs from evaluation plan")
    current_git = git_info()
    validate_git_scope(current_git)
    if current_git["commit"] != plan.get("git_scope", {}).get("commit"):
        raise ContractError("evaluation plan source commit differs from current HEAD")
    if i1_preflight.get("optimizer_steps") != 0 or i1_preflight.get("new_checkpoints") != 0:
        raise ContractError("I1 preflight baseline changed during C1-I2 preflight")
    validate_no_evaluation_outputs(allow_plan_and_preflight=True)

    report = {
        "schema": "keqing.mortal.c1_evaluation_implementation_preflight.v1",
        "implementation_preflight_passed": True,
        "passed": True,
        "experiment_id": C1_ID,
        "plan": str(EVALUATION_PLAN_PATH.resolve()),
        "plan_sha256": sha256_file(EVALUATION_PLAN_PATH),
        "git": current_git,
        "i1_baseline": {
            "commit": I1_COMMIT,
            "manifest_sha256": sha256_file(Path(plan["i1_baseline"]["manifest"]["path"])),
            "preflight_sha256": sha256_file(Path(plan["i1_baseline"]["preflight"]["path"])),
            "training_authorized": i1_manifest["training_authorized"],
            "evaluation_authorized": i1_manifest["evaluation_authorized"],
            "optimizer_steps": i1_manifest["optimizer_steps"],
            "new_checkpoints": i1_manifest["new_checkpoints"],
        },
        "evaluation_authorized": False,
        "execution_ready": False,
        "reason": "waiting_for_authorized_training_and_six_CQL_OFF_checkpoints",
        "pending_cql_off_checkpoints": 6,
        "evaluation_games_run": 0,
        "training_authorized": False,
        "training_started": False,
        "evaluation_started": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "checks": {
            "governance_exact": True,
            "i1_commit_manifest_preflight_exact": True,
            "i1_training_not_started": True,
            "i1_optimizer_steps_zero": True,
            "i1_new_checkpoints_zero": True,
            "evaluator_source_sha_and_blob_exact": True,
            "evaluation_dependency_sources_exact": True,
            "fresh_runtime_exact": True,
            "K0_and_ext_mortal_exact": True,
            "six_current_checkpoints_exact": True,
            "six_cql_off_checkpoints_absent": True,
            "exact_24_run_matrix": True,
            "no_evaluation_outputs_or_logs": True,
            "no_arena_or_real_games_run": True,
        },
        "evaluator_provenance": plan["evaluator_provenance"],
        "evaluation_dependency_sources": plan["evaluation_dependency_sources"],
        "runtime_provenance": plan["runtime_provenance"],
        "models": plan["models"],
        "runs": plan["runs"],
    }
    dump_json(IMPLEMENTATION_PREFLIGHT_PATH, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=EVALUATION_PLAN_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_preflight(args.plan)
    except ContractError as exc:
        print(f"C1-I2 implementation preflight failed: {exc}", file=sys.stderr)
        return 2
    print(f"implementation_preflight_passed={report['implementation_preflight_passed']}")
    print(f"evaluation_authorized={report['evaluation_authorized']}")
    print(f"execution_ready={report['execution_ready']}")
    print(f"reason={report['reason']}")
    print(f"pending_cql_off_checkpoints={report['pending_cql_off_checkpoints']}")
    print(f"evaluation_games_run={report['evaluation_games_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
