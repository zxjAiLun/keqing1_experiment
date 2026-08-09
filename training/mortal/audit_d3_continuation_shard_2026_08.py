#!/usr/bin/env python3
"""Audit one D3 continuation shard (B250, 1800250..1805999) with audit-v2 semantics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_continuation_contract import (
    AMP,
    CONTINUATION_AUDIT_SCHEMA,
    CONTINUATION_SCHEMA,
    DEVICE,
    GAMES_PER_SHARD,
    RANK_POINTS,
    REQUIRED_LABELS,
    SEAT_MODE,
    SEED_KEY,
    continuation_lineage,
    shard_confirmation_token,
    shard_output_dir,
    shard_seed_end_exclusive,
    shard_seed_keys,
    shard_seed_start,
)
from training.mortal.d3_continuation_preflight import implementation_manifest_continuation
from training.mortal.d3_exploration_engine import (
    CONTRACT_ID,
    DISCARD_ACTION_LIMIT,
    EXPLORATION_PROBABILITY,
)
from training.mortal.d3_production_audit_core import _load_events, _load_log_manifest
from training.mortal.d3_production_contract import (
    AUTHORITATIVE_MORTAL_COMMIT,
    AUTHORITATIVE_NATIVE_BINARY_SHA256,
    AUTHORITATIVE_NATIVE_PATCH_SHA256,
    AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
    mortal_lineage,
    project_lineage,
    read_json,
    sha256_file,
    write_json,
)
from training.mortal.d3_production_event_audit import audit_event_records
from training.mortal.d3_production_replay_audit import _build_decision_snapshots
from training.mortal.d3_production_report import _exploration_distribution, _write_markdown


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args(argv)


def _protocol_checks(protocol: dict[str, Any], run_dir: Path, shard_index: int) -> dict[str, bool]:
    fixed = protocol.get("fixed_protocol", {})
    project = protocol.get("project_lineage", {})
    continuation = protocol.get("continuation_lineage", {})
    native = protocol.get("mortal_lineage", {})
    runtime = protocol.get("runtime", {})
    models = protocol.get("models", {})
    archives = protocol.get("lineage_archives", {})
    authoritative = protocol.get("authoritative_smoke", {})
    ignored_artifacts = protocol.get("ignored_artifacts", {})
    production_output = ignored_artifacts.get("production_output", {})
    production_implementation = protocol.get("production_implementation", {})
    final_guard = protocol.get("final_call_guard", {})
    seed_start = shard_seed_start(shard_index)
    seed_end = shard_seed_end_exclusive(shard_index)
    gate_id = f"D3_continuation_shard_{shard_index:03d}_gate_2026_08"

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
        "schema": protocol.get("schema") == CONTINUATION_SCHEMA,
        "gate_id": protocol.get("gate_id") == gate_id,
        "contract_id": protocol.get("contract_id") == CONTRACT_ID,
        "shard_index": protocol.get("shard_index") == shard_index,
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
        == AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
        "authoritative_mortal_commit": authoritative.get("mortal_source_commit")
        == AUTHORITATIVE_MORTAL_COMMIT,
        "authoritative_native_patch": authoritative.get("d3_native_patch_sha256")
        == AUTHORITATIVE_NATIVE_PATCH_SHA256,
        "authoritative_native_binary": authoritative.get("loaded_libriichi_sha256")
        == AUTHORITATIVE_NATIVE_BINARY_SHA256,
        "seed_start": fixed.get("seed_start") == seed_start,
        "seed_end": fixed.get("seed_end_exclusive") == seed_end,
        "seed_key": fixed.get("seed_key") == SEED_KEY,
        "games": fixed.get("games") == GAMES_PER_SHARD,
        "native_batch_games": fixed.get("native_batch_games") == GAMES_PER_SHARD,
        "native_call_count": fixed.get("native_call_count") == 1,
        "seat_mode": fixed.get("seat_mode") == SEAT_MODE,
        "device": fixed.get("device") == DEVICE,
        "amp_false": fixed.get("amp") is AMP,
        "resume_forbidden": fixed.get("resume_supported") is False,
        "no_auto_continue": fixed.get("auto_continue_enabled") is False,
        "production_output_ignored_or_external": production_output.get("external_to_repo") is True
        or production_output.get("ignored") is True,
        "rank_points": fixed.get("rank_points") == list(RANK_POINTS),
        "project_clean": project.get("dirty") is False,
        "project_branch": project.get("branch") == "main",
        "project_transfer_anchor": project.get("transfer_anchor_is_ancestor") is True,
        "project_semantic_unchanged": project.get("semantic_diff_paths") == [],
        "continuation_governance_ancestor": continuation.get("governance_is_ancestor") is True,
        "continuation_semantic_unchanged": continuation.get("semantic_diff_paths") == [],
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
        == set(implementation_manifest_continuation(REPO_ROOT)),
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
    lineage = continuation_lineage(REPO_ROOT)
    native = mortal_lineage(mortal_root)
    model_errors: list[str] = []
    for label in REQUIRED_LABELS:
        row = protocol.get("models", {}).get(label, {})
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            model_errors.append(f"model missing after generation: {label} -> {path}")
        elif sha256_file(path) != row.get("sha256"):
            model_errors.append(f"model SHA changed after generation: {label}")
    loaded_binary_path = Path(_riichi.__file__).resolve()
    loaded_binary_sha = sha256_file(loaded_binary_path)
    binary_exact = loaded_binary_sha == protocol.get("runtime", {}).get(
        "loaded_libriichi_sha256"
    ) == AUTHORITATIVE_NATIVE_BINARY_SHA256
    patch_path = REPO_ROOT / "training/mortal/patches/libriichi_d3_decision_context.patch"
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
    extra_errors: list[str] = []
    if not project["passed"]:
        extra_errors.extend(project["errors"])
    if not lineage["passed"]:
        extra_errors.extend(lineage["errors"])
    if not binary_exact:
        extra_errors.append(f"loaded native binary changed after generation: {loaded_binary_sha}")
    if not patch_exact:
        extra_errors.append(f"D3 native patch changed after generation: {patch_sha}")
    if not smoke_protocol_exact:
        extra_errors.append(
            f"authoritative smoke protocol missing or changed after generation: {smoke_protocol_path}"
        )
    return {
        "project": project,
        "continuation_lineage": lineage,
        "mortal": native,
        "loaded_libriichi_path": str(loaded_binary_path),
        "loaded_libriichi_sha256": loaded_binary_sha,
        "native_binary_exact": binary_exact,
        "d3_native_patch_sha256": patch_sha,
        "native_patch_exact": patch_exact,
        "authoritative_smoke_protocol_path": str(smoke_protocol_path),
        "authoritative_smoke_protocol_exact": smoke_protocol_exact,
        "model_errors": model_errors,
        "extra_errors": extra_errors,
        "passed": (
            project["passed"]
            and lineage["passed"]
            and native["passed"]
            and binary_exact
            and patch_exact
            and smoke_protocol_exact
            and not model_errors
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    shard_index = args.shard_index
    run_dir = (args.run_dir or shard_output_dir(shard_index)).resolve()
    protocol = read_json(run_dir / "protocol.json")
    summary = read_json(run_dir / "production_summary.json")
    protocol_checks = _protocol_checks(protocol, run_dir, shard_index)
    current_lineage = _current_lineage_checks(protocol, args.mortal_root.resolve())
    log_manifest = _load_log_manifest(
        run_dir / "logs", expected=shard_seed_keys(shard_index)
    )
    k0_path = Path(protocol["models"]["K0_70k"]["path"])
    snapshots, behavior_metrics = _build_decision_snapshots(
        log_manifest=log_manifest,
        k0_path=k0_path,
        mortal_root=args.mortal_root.resolve(),
        q_batch_size=args.q_batch_size,
    )
    events = _load_events(run_dir)
    event_audit = audit_event_records(
        events, snapshots, expected_seeds=shard_seed_keys(shard_index)
    )

    exploration_summary = read_json(run_dir / "exploration" / "exploration_summary.json")
    from training.mortal.stat_report import write_stat_report  # noqa: PLC0415

    audit_dir = run_dir / "audit_v2"
    detailed_stats = write_stat_report(
        output_dir=audit_dir / "stats",
        log_dir=run_dir / "logs",
        players={label: label for label in REQUIRED_LABELS},
        mortal_root=args.mortal_root.resolve(),
        rank_pts=RANK_POINTS,
        rank_points_profile=f"D3_continuation_shard_{shard_index:03d}_[90,45,0,-135]",
    )
    rank_counts_match = True
    for label in REQUIRED_LABELS:
        raw = detailed_stats["players"][label]["raw"]
        expected_counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
        if summary.get("rank_counts", {}).get(label) != expected_counts:
            rank_counts_match = False

    data_checks = {
        "250_log_files": log_manifest["file_count"] == GAMES_PER_SHARD,
        "250_unique_seed_keys": log_manifest["unique_seed_count"] == GAMES_PER_SHARD,
        "exact_seed_set": log_manifest["expected_seed_set"],
        "250_unique_canonical_hanchans": log_manifest["unique_canonical_hanchans"]
        == GAMES_PER_SHARD,
        "zero_malformed_logs": not log_manifest["malformed"],
        "no_smoke_seed": not log_manifest["smoke_seeds_mixed"],
        "one_k0_trainable_perspective_per_hanchan": behavior_metrics["trainable_perspectives"]
        == GAMES_PER_SHARD,
        "nonempty_trainable_decisions": behavior_metrics["total_primary_decisions"] > 0,
        "zero_duplicate_decision_context": behavior_metrics["unique_decision_contexts"]
        == behavior_metrics["total_primary_decisions"],
        "all_training_rows_behavior_actions_legal_finite": math.isclose(
            behavior_metrics["all_training_rows_behavior_action_legal_finite_rate"], 1.0
        ),
        "all_primary_behavior_actions_legal_finite": math.isclose(
            behavior_metrics["primary_behavior_action_legal_finite_rate"], 1.0
        ),
        "gameplay_loader_zero_malformed": not behavior_metrics["malformed"],
        "rank_counts_match_logs": rank_counts_match,
    }
    reconstruction_checks = {
        "zero_native_scene_label_mismatches": behavior_metrics["reconstruction_errors"] == [],
        "no_unmapped_event_context": event_audit["mapping_violations"].get(
            "unmapped_context", 0
        ) == 0,
        "behavior_action_correspondence_zero": event_audit["mapping_violations"].get(
            "behavior_mismatch", 0
        ) == 0,
        "no_duplicate_event_context": event_audit["contract_violations"].get(
            "duplicate_context", 0
        ) == 0,
        "no_extra_generation_context": event_audit["extra_event_count"] == 0,
    }
    contract_checks = {
        "contract_id": all(event.get("contract_id") == CONTRACT_ID for event in events),
        "probability_0_25": all(
            math.isclose(
                float(event.get("exploration_probability", float("nan"))),
                EXPLORATION_PROBABILITY,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for event in events
        ),
        "distinct_discards": all(
            0 <= int(event["top1_action"]) < DISCARD_ACTION_LIMIT
            and 0 <= int(event["top2_action"]) < DISCARD_ACTION_LIMIT
            and int(event["top1_action"]) != int(event["top2_action"])
            for event in events
        ),
        "event_q_finite": event_audit["contract_violations"].get("finite_q", 0) == 0,
        "margin_identity": event_audit["contract_violations"].get("margin_identity", 0) == 0,
        "margin_le_0_5": event_audit["contract_violations"].get("threshold", 0) == 0,
        "hash_exact": event_audit["contract_violations"].get("hash", 0) == 0,
        "budget_exact": event_audit["contract_violations"].get("budget", 0) == 0
        and event_audit["contract_violations"].get("budget_reason", 0) == 0,
        "reason_exact": event_audit["contract_violations"].get("budget_reason", 0) == 0,
        "explored_and_action_exact": event_audit["contract_violations"].get(
            "actual_action", 0
        ) == 0,
        "base_action_exact": event_audit["contract_violations"].get("base_action", 0) == 0,
        "seed_range_exact": event_audit["contract_violations"].get("seed_range", 0) == 0,
    }
    lineage_checks = {
        "runtime_protocol_lineage": all(protocol_checks.values()),
        "current_generation_lineage_unchanged": current_lineage["passed"],
    }
    hard_pass = (
        all(protocol_checks.values())
        and all(data_checks.values())
        and all(contract_checks.values())
        and all(reconstruction_checks.values())
        and all(lineage_checks.values())
    )
    errors = [
        *log_manifest["malformed"],
        *behavior_metrics["malformed"],
        *behavior_metrics["reconstruction_errors"],
        *event_audit["errors"],
        *current_lineage["project"]["errors"],
        *current_lineage["continuation_lineage"]["errors"],
        *current_lineage["mortal"]["errors"],
        *current_lineage["model_errors"],
        *current_lineage["extra_errors"],
    ]
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_data = [name for name, passed in data_checks.items() if not passed]
    failed_contract = [name for name, passed in contract_checks.items() if not passed]
    failed_mapping = [name for name, passed in reconstruction_checks.items() if not passed]
    if failed_protocol:
        errors.append(f"failed provenance checks: {failed_protocol}")
    if failed_data:
        errors.append(f"failed data integrity checks: {failed_data}")
    if failed_contract:
        errors.append(f"failed event contract checks: {failed_contract}")
    if failed_mapping:
        errors.append(f"failed correspondence checks: {failed_mapping}")

    exploration_distribution = _exploration_distribution(
        events, snapshots, behavior_metrics["all_kyoku_keys"]
    )
    eligible_count = event_audit["event_count"]
    explored_count = event_audit["explored_count"]
    k0_stats = detailed_stats["players"]["K0_70k"]
    report = {
        "schema": CONTINUATION_AUDIT_SCHEMA,
        "gate": {
            "shard_index": shard_index,
            "gate_id": f"D3_continuation_shard_{shard_index:03d}_gate_2026_08",
            "verdict": "PASS" if hard_pass else "FAIL",
            "passed": hard_pass,
            "checks": {
                "provenance": protocol_checks,
                "data_integrity": data_checks,
                "event_contract": contract_checks,
                "correspondence": reconstruction_checks,
                "lineage": lineage_checks,
            },
            "confirmation_token": shard_confirmation_token(shard_index),
            "generation_commit": protocol.get("project_lineage", {}).get("commit"),
            "auditor_commit": current_lineage["project"]["commit"],
            "d3_semantic_anchor": protocol.get("continuation_lineage", {}).get(
                "d3_semantic_anchor"
            ),
            "governance_commit": protocol.get("continuation_lineage", {}).get(
                "continuation_governance"
            ),
            "failure_policy": (
                "FAIL stops the whole continuation; do not modify parameters or skip shards"
                if not hard_pass
                else "PASS freezes this shard; continue with the next shard only"
            ),
        },
        "protocol": protocol,
        "data_integrity": {
            key: value for key, value in log_manifest.items() if key not in {"paths", "rows"}
        },
        "event_audit": event_audit,
        "current_lineage": current_lineage,
        "descriptive_metrics": {
            "k0_behavior": {
                key: value
                for key, value in behavior_metrics.items()
                if key != "all_kyoku_keys"
            },
            "exploration": {
                "eligible_count": eligible_count,
                "explored_count": explored_count,
                "realized_explored_over_eligible": (
                    explored_count / eligible_count if eligible_count else 0.0
                ),
                "reason_counts": event_audit["reason_counts"],
                **exploration_distribution,
            },
            "rank_and_pt": {
                "rank_points": list(RANK_POINTS),
                "rank_counts": summary["rank_counts"],
                "k0_total_rank_pt": k0_stats["derived"]["total_rank_pt"],
                "k0_avg_rank_pt": k0_stats["derived"]["avg_rank_pt"],
                "k0_avg_rank": k0_stats["derived"]["avg_rank"],
            },
            "k0_outcomes": {
                "agari": k0_stats["raw"]["agari"],
                "houjuu": k0_stats["raw"]["houjuu"],
                "fuuro": k0_stats["raw"]["fuuro"],
                "riichi": k0_stats["raw"]["riichi"],
                "agari_rate": k0_stats["derived"]["agari_rate"],
                "houjuu_rate": k0_stats["derived"]["houjuu_rate"],
                "fuuro_rate": k0_stats["derived"]["fuuro_rate"],
                "riichi_rate": k0_stats["derived"]["riichi_rate"],
            },
            "detailed_stats": detailed_stats,
        },
        "errors": errors,
        "scope_notes": [
            "rank/Pt and behavior metrics are descriptive only and cannot change the frozen D3 recipe",
            "replay-Q/ranking/eligibility are descriptive diagnostics, not hard gates",
            "this audit does not generate other shards and does not start training",
        ],
    }
    output = args.output or (
        audit_dir / f"d3_continuation_shard_{shard_index:03d}_audit_v2.json"
    )
    markdown = args.markdown_output or (
        audit_dir / f"d3_continuation_shard_{shard_index:03d}_audit_v2.md"
    )
    write_json(output.resolve(), report)
    _write_markdown(report, markdown.resolve())
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit(args)
    print(
        json.dumps(
            {
                "shard_index": report["gate"]["shard_index"],
                "verdict": report["gate"]["verdict"],
                "output": str(
                    (
                        args.output
                        or (
                            (args.run_dir or shard_output_dir(args.shard_index))
                            / "audit_v2"
                            / f"d3_continuation_shard_{args.shard_index:03d}_audit_v2.json"
                        )
                    ).resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
