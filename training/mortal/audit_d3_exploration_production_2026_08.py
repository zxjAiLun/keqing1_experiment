#!/usr/bin/env python3
"""Audit the frozen first D3 production B250 gate without changing the recipe."""
# ruff: noqa: E402

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

from training.mortal.d3_exploration_engine import (
    CONTRACT_ID, DISCARD_ACTION_LIMIT, EXPLORATION_PROBABILITY,
)
from training.mortal.d3_production_contract import (
    DEFAULT_OUTPUT_DIR, GAMES, GATE_ID, RANK_POINTS, REQUIRED_LABELS, read_json, write_json,
)
from training.mortal.d3_production_audit_core import (
    _load_events, _load_log_manifest,
)
from training.mortal.d3_production_event_audit import audit_event_records
from training.mortal.d3_production_lineage_audit import _current_lineage_checks, _protocol_checks
from training.mortal.d3_production_replay_audit import _build_decision_snapshots
from training.mortal.d3_production_report import _exploration_distribution, _write_markdown

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args(argv)

def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    protocol = read_json(run_dir / "protocol.json")
    summary = read_json(run_dir / "production_summary.json")
    protocol_checks = _protocol_checks(protocol, run_dir)
    current_lineage = _current_lineage_checks(protocol, args.mortal_root.resolve())
    log_manifest = _load_log_manifest(run_dir / "logs")
    k0_path = Path(protocol["models"]["K0_70k"]["path"])
    snapshots, behavior_metrics = _build_decision_snapshots(
        log_manifest=log_manifest,
        k0_path=k0_path,
        mortal_root=args.mortal_root.resolve(),
        q_batch_size=args.q_batch_size,
    )
    events = _load_events(run_dir)
    event_audit = audit_event_records(events, snapshots)

    exploration_summary = read_json(run_dir / "exploration" / "exploration_summary.json")
    exploration_counters = exploration_summary.get("counters", {})
    from training.mortal.stat_report import write_stat_report  # noqa: PLC0415

    audit_dir = run_dir / "audit_v2"
    detailed_stats = write_stat_report(
        output_dir=audit_dir / "stats",
        log_dir=run_dir / "logs",
        players={label: label for label in REQUIRED_LABELS},
        mortal_root=args.mortal_root.resolve(),
        rank_pts=RANK_POINTS,
        rank_points_profile="D3_B250_fixed_[90,45,0,-135]",
    )
    rank_counts_match = True
    for label in REQUIRED_LABELS:
        raw = detailed_stats["players"][label]["raw"]
        expected_counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
        if summary.get("rank_counts", {}).get(label) != expected_counts:
            rank_counts_match = False

    # ---- gate B: data integrity ----
    data_checks = {
        "250_log_files": log_manifest["file_count"] == GAMES,
        "250_unique_seed_keys": log_manifest["unique_seed_count"] == GAMES,
        "exact_seed_set": log_manifest["expected_seed_set"],
        "250_unique_canonical_hanchans": log_manifest["unique_canonical_hanchans"] == GAMES,
        "zero_malformed_logs": not log_manifest["malformed"],
        "no_smoke_seed": not log_manifest["smoke_seeds_mixed"],
        "one_k0_trainable_perspective_per_hanchan": behavior_metrics["trainable_perspectives"]
        == GAMES,
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

    # ---- gate D: reconstruction integrity ----
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

    # ---- gate C: event internal contract (frozen, no replay-Q) ----
    contract_checks = {
        "contract_id": all(
            event.get("contract_id") == CONTRACT_ID for event in events
        ),
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
    generation_commit = protocol.get("project_lineage", {}).get("commit")
    auditor_commit = current_lineage["auditor_commit"]
    report = {
        "schema": "keqing.mortal.d3_production_gate_reaudit.v2",
        "gate": {
            "gate_id": GATE_ID,
            "verdict": "PASS" if hard_pass else "FAIL",
            "passed": hard_pass,
            "checks": {
                "provenance": protocol_checks,
                "data_integrity": data_checks,
                "event_contract": contract_checks,
                "correspondence": reconstruction_checks,
                "lineage": lineage_checks,
            },
            "generation_commit": generation_commit,
            "auditor_commit": auditor_commit,
            "generation_artifacts_modified": False,
            "supersedes_verdict_from": "audit-v1",
            "supersession_reason": (
                "authoritative 25h smoke reproduced audit-v1 failure signature; "
                "audit-v1 replay-Q equality is not an interchangeable oracle for "
                "generation-time arena inference and is demoted to diagnostics"
            ),
            "failure_policy": (
                "FAIL closes D3_top2_discard_v1; do not change parameters under the same experiment ID"
                if not hard_pass
                else "PASS only authorizes later 1800250..1805999 shards after governance update"
            ),
        },
        "protocol": protocol,
        "data_integrity": {
            key: value for key, value in log_manifest.items() if key not in {"paths", "rows"}
        },
        "event_audit": event_audit,
        "current_lineage": current_lineage,
        "descriptive_metrics": {
            "k0_behavior": {key: value for key, value in behavior_metrics.items() if key != "all_kyoku_keys"},
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
            "replay-Q/ranking/eligibility are descriptive diagnostics (layer E), not hard gates",
            "this audit does not generate the remaining 5750 hanchans and does not start training",
            "a PASS still requires Chinese report, registry, overview, and consistency updates before continuation",
        ],
    }
    output = args.output or (audit_dir / "d3_production_gate_reaudit_v2.json")
    markdown = args.markdown_output or (audit_dir / "d3_production_gate_reaudit_v2.md")
    write_json(output.resolve(), report)
    _write_markdown(report, markdown.resolve())
    return report

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit(args)
    print(
        json.dumps(
            {
                "verdict": report["gate"]["verdict"],
                "output": str((args.output or (args.run_dir / "audit_v2/d3_production_gate_reaudit_v2.json")).resolve()),
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
