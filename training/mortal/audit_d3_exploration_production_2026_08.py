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
    CONTRACT_ID, EXPLORATION_PROBABILITY, HANCHAN_BUDGET, KYOKU_BUDGET, MARGIN_THRESHOLD,
)
from training.mortal.d3_production_contract import (
    AUDIT_SCHEMA, DEFAULT_OUTPUT_DIR, GAMES, GATE_ID, RANK_POINTS, REQUIRED_LABELS, read_json, write_json,
)
from training.mortal.d3_production_audit_core import (
    DecisionSnapshot, _load_events, _load_log_manifest, primary_row_flags,
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

    audit_dir = run_dir / "audit"
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
    summary_reason_counts = {
        reason: int(exploration_counters.get(f"{reason}_count", 0))
        for reason in (
            "explored",
            "hash_rejected",
            "kyoku_budget_exhausted",
            "hanchan_budget_exhausted",
        )
    }
    exploration_checks = {
        "summary_contract_id": exploration_summary.get("contract_id") == CONTRACT_ID,
        "summary_probability": exploration_summary.get("probability") == EXPLORATION_PROBABILITY,
        "summary_margin": exploration_summary.get("margin_threshold") == MARGIN_THRESHOLD,
        "summary_kyoku_budget": exploration_summary.get("kyoku_budget") == KYOKU_BUDGET,
        "summary_hanchan_budget": exploration_summary.get("hanchan_budget") == HANCHAN_BUDGET,
        "summary_primary_state_count": exploration_counters.get("states")
        == behavior_metrics["total_primary_decisions"],
        "summary_event_count": exploration_counters.get("event_count")
        == event_audit["event_count"],
        "summary_eligible_count": exploration_counters.get("eligible_count")
        == event_audit["event_count"],
        "summary_explored_count": exploration_counters.get("explored_count")
        == event_audit["explored_count"],
        "summary_reason_counts": summary_reason_counts == event_audit["reason_counts"],
        "runner_summary_counters_match": summary.get("exploration_counters")
        == exploration_counters,
        "protocol_counters_match": protocol.get("exploration_counters")
        == exploration_counters,
        "eligible_events_positive": event_audit["event_count"] > 0,
        "explored_events_positive": event_audit["explored_count"] > 0,
        "eligible_event_set_exact": event_audit["missing_event_count"] == 0
        and event_audit["extra_event_count"] == 0,
        "event_contract_recomputed": event_audit["passed"],
        "primary_context_violation_zero": event_audit["violation_counts"].get(
            "primary_context", 0
        )
        == 0,
        "own_riichi_violation_zero": event_audit["violation_counts"].get("own_riichi", 0)
        == 0,
        "semantic_violation_zero": event_audit["violation_counts"].get("semantic", 0) == 0,
        "legal_finite_violation_zero": event_audit["violation_counts"].get(
            "legal_finite", 0
        )
        == 0,
        "stable_ranking_violation_zero": event_audit["violation_counts"].get("ranking", 0)
        == 0,
        "finite_q_violation_zero": event_audit["violation_counts"].get("finite_q", 0) == 0,
        "q_recompute_violation_zero": event_audit["violation_counts"].get("q_recompute", 0)
        == 0,
        "threshold_violation_zero": event_audit["violation_counts"].get("threshold", 0) == 0,
        "hash_violation_zero": event_audit["violation_counts"].get("hash", 0) == 0,
        "budget_violation_zero": event_audit["violation_counts"].get("budget", 0) == 0
        and event_audit["violation_counts"].get("budget_reason", 0) == 0,
        "actual_action_violation_zero": event_audit["violation_counts"].get(
            "actual_action", 0
        )
        == 0,
        "base_action_violation_zero": event_audit["violation_counts"].get("base_action", 0)
        == 0,
        "auxiliary_exploration_zero": "auxiliary_exploration_count" in exploration_counters
        and exploration_counters.get("auxiliary_exploration_count") == 0,
    }
    lineage_checks = {
        "runtime_protocol_lineage": all(protocol_checks.values()),
        "current_project_native_models_unchanged": current_lineage["passed"],
    }
    hard_pass = (
        all(protocol_checks.values())
        and all(data_checks.values())
        and all(exploration_checks.values())
        and all(lineage_checks.values())
    )
    errors = [
        *log_manifest["malformed"],
        *behavior_metrics["malformed"],
        *event_audit["errors"],
        *current_lineage["project"]["errors"],
        *current_lineage["mortal"]["errors"],
        *current_lineage["model_errors"],
        *current_lineage["extra_errors"],
    ]
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_data = [name for name, passed in data_checks.items() if not passed]
    failed_exploration = [name for name, passed in exploration_checks.items() if not passed]
    if failed_protocol:
        errors.append(f"failed protocol checks: {failed_protocol}")
    if failed_data:
        errors.append(f"failed data checks: {failed_data}")
    if failed_exploration:
        errors.append(f"failed exploration checks: {failed_exploration}")

    exploration_distribution = _exploration_distribution(
        events, snapshots, behavior_metrics["all_kyoku_keys"]
    )
    eligible_count = event_audit["event_count"]
    explored_count = event_audit["explored_count"]
    k0_stats = detailed_stats["players"]["K0_70k"]
    report = {
        "schema": AUDIT_SCHEMA,
        "gate": {
            "gate_id": GATE_ID,
            "verdict": "PASS" if hard_pass else "FAIL",
            "passed": hard_pass,
            "checks": {
                "protocol": protocol_checks,
                "data_integrity": data_checks,
                "exploration_contract": exploration_checks,
                "lineage": lineage_checks,
            },
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
            "this audit does not generate the remaining 5750 hanchans and does not start training",
            "a PASS still requires Chinese report, registry, overview, and consistency updates before continuation",
        ],
    }
    output = args.output or (audit_dir / "d3_production_gate_audit.json")
    markdown = args.markdown_output or (audit_dir / "d3_production_gate_audit.md")
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
                "output": str((args.output or (args.run_dir / "audit/d3_production_gate_audit.json")).resolve()),
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
