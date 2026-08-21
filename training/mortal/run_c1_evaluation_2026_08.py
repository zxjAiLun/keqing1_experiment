#!/usr/bin/env python3
"""Fail-closed future C1 evaluation launcher.

The checked-in constants intentionally make execution impossible.  A future
authorization must bind an approved implementation commit and the exact SHA
of every governance artifact before this module can reach ``subprocess.run``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.c1_evaluation_contract_2026_08 import (
    C1_ID,
    CONDITIONS,
    EVALUATION_PLAN_PATH,
    EVALUATION_ROOT,
    EVALUATOR_PATH,
    GAMES_PER_SHARD,
    IMPLEMENTATION_PREFLIGHT_PATH,
    MORTAL_REVISION,
    REPO_ROOT,
    SEED_KEY,
    SHARDS,
    TRAINING_SEEDS,
    ContractError,
    assert_exact_run_matrix,
    build_evaluator_argv,
    current_checkpoint_records,
    load_json,
    model_order,
    off_model_label,
    resolve_execution_manifest,
    sha256_file,
    validate_frozen_evaluator_object,
    validate_runtime_provenance,
    validate_source_provenance,
)

DEFAULT_PLAN = EVALUATION_PLAN_PATH
DEFAULT_PREFLIGHT = IMPLEMENTATION_PREFLIGHT_PATH
DEFAULT_COMPLETION_CLOSURE = EVALUATION_ROOT / "training_completion_closure.json"
DEFAULT_EXECUTION_MANIFEST = EVALUATION_ROOT / "execution_manifest.json"

# These are deliberately the only authorization controls.  They remain
# false/None in the implementation commit and must not be inferred from a
# local file or command-line flag.
EVALUATION_AUTHORIZED = False
APPROVED_EVALUATION_IMPLEMENTATION_COMMIT = "a8c7ee4b5c1134794b83d47532e4356e3e365b66"
AUTHORIZED_EVALUATION_PLAN_SHA256 = "2d9f85144492cbd2f86c786ebc5d6ad10722ce8449bdcde5176bf8f6578f18f7"
AUTHORIZED_EVALUATION_PREFLIGHT_SHA256 = "9f1ecbe20473b3ceccfcdc70aa1a26b041890c532f0f15fc5a6b476abd20c0d0"
AUTHORIZED_TRAINING_COMPLETION_SHA256 = "cdaaaa8d67bc8497ad7dcf279de78db5bc0110d071882ab47daaf1b3e07b2b9f"
AUTHORIZED_EXECUTION_MANIFEST_SHA256 = "241cfcb5559598fa5c103161ceb28363c80af65e6944f894d3fc2fde7fd1a151"


class AuthorizationError(RuntimeError):
    """Raised when an evaluation launch is not fully authorized and bound."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", choices=TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--shard", choices=SHARDS, type=int, required=True)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation-token")
    return parser


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorizationError(f"C1-I2 evaluation plan is missing: {path}")
    return load_json(path)


def selected_run(plan: dict[str, Any], condition: str, seed: int, shard: int) -> dict[str, Any]:
    matches = [
        row
        for row in plan.get("runs", [])
        if row.get("condition") == condition
        and int(row.get("training_seed", -1)) == seed
        and int(row.get("shard", -1)) == shard
    ]
    if len(matches) != 1:
        raise AuthorizationError(f"C1-I2 run is not unique in the frozen plan: {condition}/{seed}/{shard}")
    return matches[0]


def confirmation_token(*, condition: str, seed: int, shard: int, implementation_commit: str, preflight_sha256: str) -> str:
    if not implementation_commit or not preflight_sha256:
        raise AuthorizationError("confirmation token cannot be derived from empty authorization bindings")
    return f"C1_EVAL_{condition}_{seed}_{shard}_{implementation_commit[:12]}_{preflight_sha256[:12]}"


def _models_by_label(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for item in plan.get("models", []):
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise AuthorizationError("evaluation plan contains an invalid model record")
        label = str(item["label"])
        if label in values:
            raise AuthorizationError(f"duplicate model label: {label}")
        values[label] = item
    return values


def _assert_empty_or_absent_output_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise AuthorizationError(f"evaluation output path is not a directory: {path}")
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if files:
        raise AuthorizationError(f"evaluation shard has existing output; resume is forbidden: {files[:4]}")


def _execution_rows(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = manifest.get("runs")
    if not isinstance(rows, list):
        raise AuthorizationError("execution manifest has no runs")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        route = str(row.get("route", ""))
        seed = int(row.get("training_seed", -1))
        key = (route, seed)
        if key in result:
            raise AuthorizationError(f"duplicate execution-manifest run: {key}")
        result[key] = row
    expected = {(f"{route}_CQL_OFF", seed) for route in ("M0", "D1") for seed in TRAINING_SEEDS}
    if set(result) != expected:
        raise AuthorizationError("execution manifest does not contain exactly six CQL_OFF completions")
    return result


def _validate_authorized_artifacts(
    *,
    plan: dict[str, Any],
    preflight: dict[str, Any],
    completion: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan.get("experiment_id") != C1_ID or plan.get("status") not in {"prepared_not_authorized", "authorized"}:
        raise AuthorizationError("evaluation plan experiment/status mismatch")
    if plan.get("evaluation_games_run") != 0:
        raise AuthorizationError("evaluation plan already records games")
    if preflight.get("implementation_preflight_passed") is not True or preflight.get("passed") is not True:
        raise AuthorizationError("implementation preflight is not passed")
    if preflight.get("plan_sha256") != sha256_file(DEFAULT_PLAN.resolve()):
        raise AuthorizationError("preflight does not bind the authorized plan SHA")
    if preflight.get("evaluation_games_run") != 0 or preflight.get("new_checkpoints") != 0:
        raise AuthorizationError("preflight records execution")
    if plan.get("git_scope", {}).get("commit") != APPROVED_EVALUATION_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("plan implementation commit mismatch")
    if preflight.get("git", {}).get("commit") != APPROVED_EVALUATION_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("preflight implementation commit mismatch")
    if completion.get("experiment_id") != C1_ID:
        raise AuthorizationError("training completion closure experiment mismatch")
    if execution.get("schema") != "keqing.mortal.c1_evaluation_execution_manifest.v1":
        raise AuthorizationError("execution manifest schema mismatch")
    if execution.get("experiment_id") != C1_ID:
        raise AuthorizationError("execution manifest experiment mismatch")
    execution_rows = _execution_rows(execution)
    if completion.get("runs") is None and completion.get("seeds") is None:
        raise AuthorizationError("training completion closure has no six-run interface")
    # Resolve again from the closure.  This checks path, SHA, 72000 steps,
    # 2000 optimizer steps, parent, CQL weight, objective/reward, and fresh
    # scheduler/scaler/data stream; it does not start training.
    try:
        resolved = resolve_execution_manifest(plan, completion)
    except ContractError as exc:
        raise AuthorizationError(str(exc)) from exc
    resolved_rows = _execution_rows(resolved)
    if set(resolved_rows) != set(execution_rows):
        raise AuthorizationError("execution manifest and closure matrices differ")
    for key, resolved_row in resolved_rows.items():
        actual_row = execution_rows[key]
        for field in ("final_checkpoint_path", "final_checkpoint_sha256", "steps", "trained_optimizer_steps"):
            if actual_row.get(field) != resolved_row.get(field):
                raise AuthorizationError(f"execution manifest completion binding mismatch: {key}/{field}")

    models = _models_by_label(plan)
    if len(models) != 14:
        raise AuthorizationError("authorized plan does not bind exactly 14 model records")
    current_models = current_checkpoint_records()
    for label, current in current_models.items():
        item = models.get(label)
        if item is None or Path(str(item.get("path"))).resolve() != Path(str(current["path"])).resolve() or item.get("sha256") != current.get("sha256"):
            raise AuthorizationError(f"CURRENT/anchor model binding mismatch: {label}")
    return models, resolved_rows


def validate_authorized_launch(
    *,
    condition: str,
    seed: int,
    shard: int,
    supplied_token: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    if EVALUATION_AUTHORIZED is not True:
        raise AuthorizationError("C1 evaluation is not authorized: EVALUATION_AUTHORIZED=false")
    if not APPROVED_EVALUATION_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("approved C1-I2 implementation commit is not bound")
    for name, value in {
        "AUTHORIZED_EVALUATION_PLAN_SHA256": AUTHORIZED_EVALUATION_PLAN_SHA256,
        "AUTHORIZED_EVALUATION_PREFLIGHT_SHA256": AUTHORIZED_EVALUATION_PREFLIGHT_SHA256,
        "AUTHORIZED_TRAINING_COMPLETION_SHA256": AUTHORIZED_TRAINING_COMPLETION_SHA256,
        "AUTHORIZED_EXECUTION_MANIFEST_SHA256": AUTHORIZED_EXECUTION_MANIFEST_SHA256,
    }.items():
        if not value:
            raise AuthorizationError(f"{name} is not bound")
    expected_token = confirmation_token(
        condition=condition,
        seed=seed,
        shard=shard,
        implementation_commit=APPROVED_EVALUATION_IMPLEMENTATION_COMMIT,
        preflight_sha256=AUTHORIZED_EVALUATION_PREFLIGHT_SHA256,
    )
    if supplied_token != expected_token:
        raise AuthorizationError("evaluation confirmation token mismatch")

    plan_path = DEFAULT_PLAN.resolve()
    preflight_path = DEFAULT_PREFLIGHT.resolve()
    completion_path = DEFAULT_COMPLETION_CLOSURE.resolve()
    execution_path = DEFAULT_EXECUTION_MANIFEST.resolve()
    if sha256_file(plan_path) != AUTHORIZED_EVALUATION_PLAN_SHA256:
        raise AuthorizationError("authorized evaluation plan SHA mismatch")
    if sha256_file(preflight_path) != AUTHORIZED_EVALUATION_PREFLIGHT_SHA256:
        raise AuthorizationError("authorized implementation preflight SHA mismatch")
    if sha256_file(completion_path) != AUTHORIZED_TRAINING_COMPLETION_SHA256:
        raise AuthorizationError("authorized training completion SHA mismatch")
    if sha256_file(execution_path) != AUTHORIZED_EXECUTION_MANIFEST_SHA256:
        raise AuthorizationError("authorized execution manifest SHA mismatch")
    plan = load_plan(plan_path)
    preflight = load_json(preflight_path)
    completion = load_json(completion_path)
    execution = load_json(execution_path)
    models, resolved_rows = _validate_authorized_artifacts(
        plan=plan,
        preflight=preflight,
        completion=completion,
        execution=execution,
    )
    assert_exact_run_matrix(plan.get("runs", []), models)
    selected = selected_run(plan, condition, seed, shard)
    if tuple(selected.get("model_order", ())) != model_order(condition, seed):
        raise AuthorizationError("selected model order is not frozen")
    if selected.get("games") != GAMES_PER_SHARD or selected.get("native_batch_games") != GAMES_PER_SHARD:
        raise AuthorizationError("selected B250 binding mismatch")
    if selected.get("seed_key") != SEED_KEY or selected.get("seat_mode") != "random":
        raise AuthorizationError("selected seed/seat binding mismatch")
    if selected.get("device") != "cuda" or selected.get("require_cuda") is not True or selected.get("amp") is not False:
        raise AuthorizationError("selected runtime binding mismatch")
    if selected.get("resume") is not False:
        raise AuthorizationError("resume is not explicitly forbidden")
    if "--resume" in selected.get("future_argv", []) or "--enable-amp" in selected.get("future_argv", []):
        raise AuthorizationError("forbidden resume/AMP argument is present")
    selected_models = {label: dict(models[label]) for label in model_order(condition, seed)}
    if condition == "CQL_OFF":
        for route in ("M0", "D1"):
            label = off_model_label(route, seed)
            completion_row = resolved_rows[(f"{route}_CQL_OFF", seed)]
            if selected_models[label].get("path") != completion_row.get("final_checkpoint_path"):
                raise AuthorizationError(f"selected CQL_OFF checkpoint binding mismatch: {label}")
            selected_models[label]["sha256"] = completion_row["final_checkpoint_sha256"]
    else:
        for label, model in selected_models.items():
            if model.get("sha256") is None:
                raise AuthorizationError(f"selected CURRENT model is pending: {label}")

    expected_runtime = plan.get("runtime_provenance")
    if not isinstance(expected_runtime, dict):
        raise AuthorizationError("plan runtime provenance is missing")
    current_runtime = validate_runtime_provenance(expected_runtime)
    if current_runtime.get("sys_executable") != expected_runtime.get("sys_executable"):
        raise AuthorizationError("current executable differs from frozen executable")
    expected_sources = {
        "evaluator": plan["evaluator_provenance"],
        "direct_dependencies": plan["evaluation_dependency_sources"],
        "mortal_revision": plan["mortal_revision"],
    }
    validate_frozen_evaluator_object()
    validate_source_provenance(expected_sources)
    if expected_sources["mortal_revision"] != MORTAL_REVISION:
        raise AuthorizationError("Mortal revision binding mismatch")

    output_dir = Path(str(selected["output_dir"])).resolve()
    _assert_empty_or_absent_output_dir(output_dir)
    command = build_evaluator_argv(
        condition=condition,
        seed=seed,
        shard=shard,
        models=selected_models,
        output_dir=output_dir,
        executable=str(expected_runtime["sys_executable"]),
    )
    if selected.get("future_argv") != command:
        raise AuthorizationError("selected authoritative evaluator argv mismatch")
    if command[0] != str(expected_runtime["sys_executable"]) or command[1] != "training/mortal/four_player_native.py":
        raise AuthorizationError("evaluator executable/script binding mismatch")
    if str(EVALUATOR_PATH.resolve()) != str((REPO_ROOT / command[1]).resolve()):
        raise AuthorizationError("evaluator script path binding mismatch")
    return plan, preflight, execution, selected, command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        plan = load_plan()
        row = selected_run(plan, args.condition, args.seed, args.shard)
        command = row.get("future_argv")
        if not isinstance(command, list):
            raise SystemExit("evaluation plan does not contain authoritative future_argv")
        if args.print_command:
            print(shlex.join(command))
        else:
            print(
                json.dumps(
                    {
                        "experiment_id": plan.get("experiment_id"),
                        "condition": args.condition,
                        "training_seed": args.seed,
                        "shard": args.shard,
                        "evaluation_authorized": EVALUATION_AUTHORIZED,
                        "command_available": True,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    try:
        _plan, _preflight, _execution, row, command = validate_authorized_launch(
            condition=args.condition,
            seed=args.seed,
            shard=args.shard,
            supplied_token=args.confirmation_token,
        )
    except (AuthorizationError, ContractError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.print_command:
        print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True, shell=False)
    print(
        json.dumps(
            {
                "condition": args.condition,
                "training_seed": args.seed,
                "shard": args.shard,
                "executed": True,
                "future_argv": row["future_argv"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
