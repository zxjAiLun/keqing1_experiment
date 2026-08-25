"""Targeted unit tests for P1 policy improvement feasibility audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libriichi

from training.mortal.audit_p1_policy_improvement_target_feasibility_2026_09 import (
    audit_simulator_capabilities,
)


def test_p1_simulator_audit_reports_exact_technical_blockers() -> None:
    """Verify that P1 feasibility audit accurately identifies Python binding limitations for counterfactual rollouts."""
    report = audit_simulator_capabilities()
    assert report["schema"] == "keqing.mortal.p1_feasibility_audit.v1"
    assert report["has_player_state"] is True
    assert report["has_action_candidate"] is True
    assert report["has_four_player_arena"] is True

    # Python bindings lack step/restore/Board exposure for counterfactual branching
    assert report["has_board_exposed"] is False
    assert report["can_step_arena"] is False
    assert report["can_restore_arena"] is False
    assert report["can_perform_counterfactual_rollout"] is False

    assert len(report["technical_blockers"]) == 4
    assert report["verdict"] == "counterfactual_rollout_blocked"


def test_p1_player_state_passive_update_contract() -> None:
    """Verify that libriichi.state.PlayerState only accepts sequential events via update() and lacks state cloning."""
    ps = libriichi.state.PlayerState(0)
    assert hasattr(ps, "update")
    assert not hasattr(ps, "step")
    assert not hasattr(ps, "clone")
    assert not hasattr(ps, "restore")
