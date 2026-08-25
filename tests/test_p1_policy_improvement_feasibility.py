"""Targeted unit tests for P1 policy improvement feasibility audit and deterministic rerun branching."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.audit_p1_policy_improvement_target_feasibility_2026_09 import (
    audit_direct_snapshot_api,
    run_deterministic_rerun_branching_smoke,
)


def test_p1_direct_snapshot_audit_reports_exact_api_blockers() -> None:
    """Verify that P1 snapshot audit narrows verdict to direct_snapshot_api_blocked."""
    report = audit_direct_snapshot_api()
    assert report["schema"] == "keqing.mortal.p1_direct_snapshot_audit.v1"
    assert report["has_board_exposed"] is False
    assert report["can_step_arena"] is False
    assert report["can_restore_arena"] is False
    assert len(report["technical_blockers"]) == 4
    assert report["verdict"] == "direct_snapshot_api_blocked"


def test_p1_deterministic_rerun_branching_smoke_and_overall_verdict() -> None:
    """Verify that seed-replay counterfactual rollout successfully diverges at target context with exact prefix match."""
    smoke = run_deterministic_rerun_branching_smoke(seed=5000000, seed_key=8192)
    assert smoke["schema"] == "keqing.mortal.p1_smoke_result.v1"
    assert smoke["both_branches_completed"] is True
    assert smoke["exact_prefix_matched"] is True
    assert smoke["events_before_divergence"] > 0
    assert smoke["top1_action"] < 34
    assert smoke["top2_action"] < 34
    assert smoke["top1_action"] != smoke["top2_action"]
    assert smoke["verdict"] == "seed_replay_counterfactual_feasible"
