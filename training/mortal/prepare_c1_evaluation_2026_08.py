#!/usr/bin/env python3
"""Prepare the frozen, not-authorized C1-I2 evaluation execution plan.

Preparation validates provenance and existing checkpoints, then writes only
``evaluation_plan.json``.  It never creates shard directories and never
imports or launches the arena evaluator.
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
    CONDITIONS,
    DIRECT_EVALUATION_SOURCES,
    EVALUATION_PLAN_PATH,
    EVALUATION_ROOT,
    EVALUATOR_FROZEN_COMMIT,
    EVALUATOR_RELATIVE_PATH,
    GAMES_PER_SHARD,
    I1_COMMIT,
    I1_MANIFEST_PATH,
    I1_MANIFEST_SHA256,
    I1_PREFLIGHT_PATH,
    I1_PREFLIGHT_SHA256,
    PREREG_COMMIT,
    PREREG_SHA256,
    RANK_POINTS,
    REGISTRY_SHA256,
    SEED_KEY,
    TOTAL_GAMES,
    TOTAL_SHARDS,
    TRAINING_SEEDS,
    ContractError,
    build_run_matrix,
    current_checkpoint_records,
    git_info,
    pending_cql_off_records,
    source_provenance,
    validate_frozen_evaluator_object,
    validate_git_scope,
    validate_governance_files,
    validate_i1_baseline,
    validate_no_evaluation_outputs,
    validate_runtime_provenance,
    validate_source_provenance,
)


def _compact_model_records(
    current: dict[str, dict[str, Any]], pending: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for label, item in current.items():
        if label in {"70k", "ext_mortal"}:
            record = {
                "label": label,
                "condition": "anchor",
                "path": item["path"],
                "sha256": item["sha256"],
                "state": "available",
            }
        else:
            route = label.split("_", 1)[0]
            seed_text = label.rsplit("_", 1)[1]
            record = {
                "label": label,
                "condition": "CURRENT",
                "route": route,
                "training_seed": int(seed_text),
                "path": item["path"],
                "sha256": item["sha256"],
                "steps": item["steps"],
                "state": "available",
            }
        records.append(record)
        by_label[label] = record
    for label, item in pending.items():
        route = label.split("_", 1)[0]
        seed_text = label.rsplit("_", 1)[1]
        record = {
            "label": label,
            "condition": "CQL_OFF",
            "route": route,
            "training_seed": int(seed_text),
            "path": item["path"],
            "sha256": None,
            "steps": 72000,
            "state": "pending_training",
        }
        records.append(record)
        by_label[label] = record
    expected_count = 2 + len(CONDITIONS) * len(TRAINING_SEEDS) * 2
    if len(records) != expected_count:
        raise ContractError(f"model record cardinality mismatch: {len(records)}")
    return records, by_label


def _build_plan() -> dict[str, Any]:
    current_manifest, current_preflight = validate_i1_baseline()
    validate_governance_files()
    validate_frozen_evaluator_object()
    frozen_sources = source_provenance()
    validate_source_provenance(frozen_sources)
    runtime = validate_runtime_provenance(current_manifest["runtime_provenance"])
    current = current_checkpoint_records()
    pending = pending_cql_off_records(current_manifest)
    model_records, models_by_label = _compact_model_records(current, pending)
    executable = str(runtime["sys_executable"])
    runs = build_run_matrix(models_by_label, executable=executable)

    # No shard directory or raw log may exist at plan time.  The plan itself
    # is written below, so the root must be absent on the first preparation.
    validate_no_evaluation_outputs(allow_plan_and_preflight=False)
    if EVALUATION_ROOT.exists():
        raise ContractError(f"evaluation implementation root already exists: {EVALUATION_ROOT}")

    git = git_info()
    validate_git_scope(git)
    return {
        "schema": "keqing.mortal.c1_evaluation_plan.v1",
        "experiment_id": C1_ID,
        "status": "prepared_not_authorized",
        "evaluation_authorized": False,
        "execution_ready": False,
        "evaluation_games_run": 0,
        "training_authorized": False,
        "training_started": False,
        "evaluation_started": False,
        "optimizer_steps": 0,
        "new_checkpoints": 0,
        "pending_cql_off_checkpoints": len(pending),
        "pending_models": sorted(pending),
        "i1_baseline": {
            "commit": I1_COMMIT,
            "manifest": {"path": str(I1_MANIFEST_PATH.resolve()), "sha256": I1_MANIFEST_SHA256},
            "preflight": {"path": str(I1_PREFLIGHT_PATH.resolve()), "sha256": I1_PREFLIGHT_SHA256},
            "training_authorized": current_manifest["training_authorized"],
            "evaluation_authorized": current_manifest["evaluation_authorized"],
            "optimizer_steps": current_manifest["optimizer_steps"],
            "new_checkpoints": current_manifest["new_checkpoints"],
            "preflight_passed": current_preflight["passed"],
        },
        "governance": {
            "prereg_commit": PREREG_COMMIT,
            "prereg_sha256": PREREG_SHA256,
            "registry_sha256": REGISTRY_SHA256,
        },
        "evaluator_provenance": frozen_sources["evaluator"],
        "evaluation_dependency_sources": frozen_sources["direct_dependencies"],
        "mortal_revision": frozen_sources["mortal_revision"],
        "runtime_provenance": runtime,
        "anchors": {
            "K0_70k": {"label": "70k", "path": models_by_label["70k"]["path"], "sha256": models_by_label["70k"]["sha256"]},
            "ext_mortal": {"label": "ext_mortal", "path": models_by_label["ext_mortal"]["path"], "sha256": models_by_label["ext_mortal"]["sha256"]},
        },
        "models": sorted(model_records, key=lambda item: item["label"]),
        "evaluation_matrix": {
            "conditions": list(CONDITIONS),
            "training_seeds": list(TRAINING_SEEDS),
            "shards_per_condition_seed": 4,
            "games_per_shard": GAMES_PER_SHARD,
            "total_shards": TOTAL_SHARDS,
            "total_games": TOTAL_GAMES,
            "seed_key": SEED_KEY,
            "seat_mode": "random",
            "device": "cuda",
            "require_cuda": True,
            "amp": False,
            "native_batch_games": GAMES_PER_SHARD,
            "rank_points_profile": "tenhou_reference",
            "no_resume": True,
        },
        "runs": runs,
        "statistics": {
            "rank_points_profile": "tenhou_reference",
            "rank_points": list(RANK_POINTS),
            "paired_unit": "complete_hanchan",
            "primary_estimand": "mean_s((D1_CQL_OFF-M0_CQL_OFF)_s - (D1_CURRENT-M0_CURRENT)_s)",
            "interaction_row": "d_off - d_current",
            "training_seed_mean": "mean_1000_hanchans",
            "primary_mean": "equal_mean_of_three_training_seed_means",
            "pairing_gate": [
                "same absolute hanchan seed",
                "same training seed",
                "same seed_key",
                "same lineup role order",
                "same role_to_seat assignment",
                "no requirement for same trajectories/ranks/events",
            ],
            "bootstrap": {
                "method": "equal_seed_hierarchical_interaction_rows",
                "reps": 5000,
                "seed": 20260818,
                "outer_units": 3,
                "inner_rows_per_seed": 1000,
                "resample_current_and_off_jointly": True,
            },
        },
        "adjudication": {
            "supported_if": [
                "all three interaction_seed means > 0",
                "hierarchical_bootstrap_ci95_lower > 0",
                "all training/provenance/runtime/pairing gates pass",
            ],
            "otherwise_if_gates_pass": "interaction_not_confirmed",
            "otherwise": "no_verdict_gates_failed",
            "K1": None,
        },
        "training_completion_interface": {
            "schema": "keqing.mortal.c1_training_completion_closure.v1",
            "required_runs": 6,
            "required_fields": [
                "route",
                "training_seed",
                "final_checkpoint_path",
                "final_checkpoint_sha256",
                "steps=72000",
                "trained_optimizer_steps=2000",
                "parent_checkpoint_sha256=K0_SHA256",
                "cql_min_q_weight=0.0",
                "objective=behavior_action_mc",
                "reward=final_rank_mc",
                "initialization.optimizer=preserved",
                "initialization.scheduler=fresh",
                "initialization.scaler=fresh",
                "initialization.data_stream=fresh",
                "data_seed=training_seed",
            ],
            "resolver": "resolve_execution_manifest(plan, training_completion_closure)",
        },
        "artifact_policy": {
            "plan_path": str(EVALUATION_PLAN_PATH.resolve()),
            "preflight_path": str((EVALUATION_ROOT / "implementation_preflight.json").resolve()),
            "allowed_before_authorization": ["evaluation_plan.json", "implementation_preflight.json"],
            "shard_logs_created": False,
            "raw_evaluation_started": False,
        },
        "git_scope": {
            "branch": git["branch"],
            "commit": git["commit"],
            "tracked_clean": git["tracked_clean"],
            "untracked": git["untracked"],
            "source_commit_required_before_prepare": "feat(mortal): implement frozen C1 paired evaluation preflight",
            "direct_evaluation_sources": list(DIRECT_EVALUATION_SOURCES),
            "evaluator_source": EVALUATOR_RELATIVE_PATH,
            "frozen_evaluator_commit": EVALUATOR_FROZEN_COMMIT,
        },
    }


def prepare(output_path: Path = EVALUATION_PLAN_PATH) -> dict[str, Any]:
    plan = _build_plan()
    output_path = output_path.resolve()
    if output_path != EVALUATION_PLAN_PATH.resolve():
        raise ContractError("C1-I2 plan output path is frozen")
    from training.mortal.c1_evaluation_contract_2026_08 import dump_json, sha256_file

    dump_json(output_path, plan)
    plan["plan_sha256"] = sha256_file(output_path)
    # Bind the plan's own digest outside the file body to avoid a recursive
    # digest.  Consumers always hash the bytes on disk.
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVALUATION_PLAN_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        plan = prepare(args.output)
    except ContractError as exc:
        print(f"C1-I2 prepare failed closed: {exc}", file=sys.stderr)
        return 2
    print(f"prepared {EVALUATION_PLAN_PATH}")
    print(f"evaluation_authorized={plan['evaluation_authorized']}")
    print(f"execution_ready={plan['execution_ready']}")
    print(f"pending_cql_off_checkpoints={plan['pending_cql_off_checkpoints']}")
    print("evaluation_games_run=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
