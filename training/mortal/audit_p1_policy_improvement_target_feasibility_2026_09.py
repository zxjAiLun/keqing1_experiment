"""Feasibility audit for P1 project-owned policy improvement target / counterfactual rollout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libriichi


def audit_simulator_capabilities() -> dict[str, Any]:
    """Audit libriichi Python bindings for state restoration, step execution, and counterfactual rollouts."""
    has_player_state = hasattr(libriichi, "state") and hasattr(libriichi.state, "PlayerState")
    has_action_candidate = hasattr(libriichi, "state") and hasattr(libriichi.state, "ActionCandidate")
    has_mjai_bot = hasattr(libriichi, "mjai") and hasattr(libriichi.mjai, "Bot")
    has_four_player = hasattr(libriichi, "arena") and hasattr(libriichi.arena, "FourPlayer")
    has_one_vs_three = hasattr(libriichi, "arena") and hasattr(libriichi.arena, "OneVsThree")

    # Check for state snapshot / step / restore API in Python bindings
    # 1. PlayerState: inspect methods
    ps_methods = dir(libriichi.state.PlayerState) if has_player_state else []
    can_step_player_state = "step" in ps_methods or "act" in ps_methods

    # 2. Arena: inspect methods
    arena_methods = dir(libriichi.arena.FourPlayer) if has_four_player else []
    can_step_arena = "step" in arena_methods or "step_action" in arena_methods
    can_restore_arena = "from_state" in arena_methods or "restore_state" in arena_methods

    # 3. Environment/Board exposed in Python
    has_board = hasattr(libriichi, "arena") and hasattr(libriichi.arena, "Board")

    # Evaluate whether full counterfactual simulation is currently supported in Python bindings
    can_perform_counterfactual_rollout = (
        has_board and can_step_arena and can_restore_arena
    )

    blockers = []
    if not has_board:
        blockers.append("libriichi.arena.Board struct is defined in Rust but not registered/exposed to Python in arena/mod.rs.")
    if not can_step_arena:
        blockers.append("libriichi.arena.FourPlayer only exposes monolithic batch runners ('py_vs_py', 'py_vs_py_random_seats'); no single-step action execution API.")
    if not can_restore_arena:
        blockers.append("No state snapshot serialization/deserialization or checkpointed BoardState restore API exposed to Python.")
    if not can_step_player_state:
        blockers.append("libriichi.state.PlayerState is a passive state tracker for an individual seat (via .update(event)), not a game engine that can transition the entire table on counterfactual actions.")

    return {
        "schema": "keqing.mortal.p1_feasibility_audit.v1",
        "has_player_state": has_player_state,
        "has_action_candidate": has_action_candidate,
        "has_mjai_bot": has_mjai_bot,
        "has_four_player_arena": has_four_player,
        "has_one_vs_three_arena": has_one_vs_three,
        "has_board_exposed": has_board,
        "can_step_arena": can_step_arena,
        "can_restore_arena": can_restore_arena,
        "can_perform_counterfactual_rollout": can_perform_counterfactual_rollout,
        "technical_blockers": blockers,
        "verdict": "counterfactual_rollout_blocked" if blockers else "counterfactual_rollout_feasible",
    }


if __name__ == "__main__":
    res = audit_simulator_capabilities()
    print(json.dumps(res, indent=2, ensure_ascii=False))
