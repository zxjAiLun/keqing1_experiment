"""Read-only guard/preemption reconciliation for the P4-M10 record.

The training run wrote ``preemption_and_guard.batches_with_agari_guard = None``
with a note claiming the P4-M9 reconciler had counted it. That count was never
computed at run time. This script replays the *existing* cycle records and arena
logs through the existing reconciler to produce the missing numbers.

It is strictly read-only with respect to the training artifacts: nothing under
``artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256/`` is
modified, and no training, evaluation or arena run is performed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p4m9_probe_onpolicy import (
    check_sampling_vs_execution,
)

RUN_DIR = (
    REPO_ROOT
    / "artifacts"
    / "experiments"
    / "student_policy_v1"
    / "P4-M10_onpolicy_pg_4x256"
)
CHALLENGER_LABEL = "p4m10_candidate"
SEED_KEY = 0x2000
SEEDS_PER_CYCLE = 64

CLASSES = (
    "policy_decisions",
    "aligned",
    "sampled_agari",
    "sampled_pass",
    "sampled_ryukyoku",
    "executed_action_differs_count",
    "agari_guard_suspected_count",
    "claim_not_executed_count",
    "unmatched_log_events",
    "kan_select_states",
    "forced_decisions_not_offered_to_model",
)


def main() -> None:
    cycles = []
    totals = {name: 0 for name in CLASSES}
    mismatches = 0
    for cycle in (1, 2, 3, 4):
        cycle_dir = RUN_DIR / f"cycle{cycle}"
        records_path = cycle_dir / "probe_records.jsonl"
        if not records_path.exists():
            raise SystemExit(f"missing records for cycle {cycle}: {records_path}")
        with records_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        seed_start = 720000 + (cycle - 1) * SEEDS_PER_CYCLE
        report = check_sampling_vs_execution(
            log_dir=cycle_dir / "logs",
            records=records,
            challenger_label=CHALLENGER_LABEL,
            seed_key=SEED_KEY,
            seed_start=seed_start,
            seed_count=SEEDS_PER_CYCLE,
        )
        row = {"cycle": cycle, "seed_range": [seed_start, seed_start + SEEDS_PER_CYCLE - 1]}
        for name in CLASSES:
            row[name] = report[name]
            totals[name] += report[name]
        row["alignment_complete"] = bool(report["alignment_complete"])
        row["decision_class_accounting_ok"] = bool(report["decision_class_accounting_ok"])
        row["mismatches"] = len(report["mismatches"])
        row["guard_instances"] = report["agari_guard_suspected"]
        row["preempted_instances"] = report["claim_not_executed"]
        mismatches += len(report["mismatches"])
        cycles.append(row)
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "policy_decisions": row["policy_decisions"],
                    "guard": row["agari_guard_suspected_count"],
                    "claim_not_executed": row["claim_not_executed_count"],
                    "executed_action_differs": row["executed_action_differs_count"],
                    "aligned": row["aligned"],
                    "sampled_agari": row["sampled_agari"],
                    "forced_no_meta": row["forced_decisions_not_offered_to_model"],
                    "alignment_complete": row["alignment_complete"],
                    "mismatches": row["mismatches"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    document = {
        "schema": "keqing.mortal.p4m10_guard_backfill.v1",
        "provenance": {
            "kind": "POST_HOC_READ_ONLY_BACKFILL",
            "created_at_unix": time.time(),
            "why": (
                "p4m10_result.json records preemption_and_guard."
                "batches_with_agari_guard = null for every cycle: the field was "
                "never computed at run time. This backfill replays the retained "
                "cycle records and arena logs through the existing P4-M9 "
                "reconciler to supply the missing count."
            ),
            "what_it_is_not": (
                "not a run-time measurement, and not a re-run of training, "
                "evaluation or any arena game; no training artifact was modified"
            ),
            "reconciler": "training/mortal/p4m9_probe_onpolicy.check_sampling_vs_execution",
            "challenger_label": CHALLENGER_LABEL,
            "seed_key": SEED_KEY,
        },
        "cycles": cycles,
        "totals": {
            **totals,
            "mismatches": mismatches,
            "alignment_complete_all_cycles": all(row["alignment_complete"] for row in cycles),
        },
        "interpretation": {
            "agari_guard": (
                "agari_guard_suspected_count is the number of sampled agari whose "
                "kyoku did not resolve for the challenger seat, i.e. the signature "
                "of the mortal.rs guard rewriting action 43. It is reported, never "
                "fail-closed."
            ),
            "claim_preemption": (
                "claim_not_executed is the number of legal proposals another "
                "player's higher-priority claim pre-empted. Those samples keep "
                "their sampling-time log-prob and stay in the loss."
            ),
        },
    }
    out = RUN_DIR / "guard_reconciliation_backfill.json"
    out.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print()
    print("TOTALS:", json.dumps(totals, ensure_ascii=False))
    print("mismatches:", mismatches, "| wrote", out)


if __name__ == "__main__":
    main()
