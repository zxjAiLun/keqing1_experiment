"""Feasibility audit and deterministic smoke for P1 project-owned policy improvement target."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import libriichi
import libriichi.arena
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import engine
import model

from training.mortal.o2_online_continuation_contract_2026_08 import (
    resolve_k0_checkpoint,
)


def audit_direct_snapshot_api() -> dict[str, Any]:
    """Audit libriichi Python bindings for direct in-memory BoardState snapshot/restore API."""
    has_player_state = hasattr(libriichi, "state") and hasattr(libriichi.state, "PlayerState")
    hasattr(libriichi, "state") and hasattr(libriichi.state, "ActionCandidate")
    has_four_player = hasattr(libriichi, "arena") and hasattr(libriichi.arena, "FourPlayer")
    has_board = hasattr(libriichi, "arena") and hasattr(libriichi.arena, "Board")

    ps_methods = dir(libriichi.state.PlayerState) if has_player_state else []
    can_step_player_state = "step" in ps_methods or "act" in ps_methods

    arena_methods = dir(libriichi.arena.FourPlayer) if has_four_player else []
    can_step_arena = "step" in arena_methods or "step_action" in arena_methods
    can_restore_arena = "from_state" in arena_methods or "restore_state" in arena_methods

    api_blockers = []
    if not has_board:
        api_blockers.append("libriichi.arena.Board struct is defined in Rust but not registered to Python in arena/mod.rs.")
    if not can_step_arena:
        api_blockers.append("libriichi.arena.FourPlayer only exposes monolithic batch runners; no single-step API.")
    if not can_restore_arena:
        api_blockers.append("No in-memory state snapshot serialization/restore API exposed to Python.")
    if not can_step_player_state:
        api_blockers.append("libriichi.state.PlayerState is a passive event listener for one seat, not a full-table simulator.")

    return {
        "schema": "keqing.mortal.p1_direct_snapshot_audit.v1",
        "has_board_exposed": has_board,
        "can_step_arena": can_step_arena,
        "can_restore_arena": can_restore_arena,
        "technical_blockers": api_blockers,
        "verdict": "direct_snapshot_api_blocked" if api_blockers else "direct_snapshot_api_feasible",
    }


class CounterfactualBranchingEngine:
    """Wrapper that records decision contexts and forces alternative actions at a target context."""

    def __init__(
        self,
        base_engine: Any,
        target_context: tuple[int, int, int, int, int] | None = None,
        forced_choice: str | None = None,
    ) -> None:
        self.base_engine = base_engine
        self.target_context = target_context
        self.forced_choice = forced_choice
        self.recorded_contexts: list[dict[str, Any]] = []
        self.intervened = False
        self.supports_decision_context = True

        self.name = base_engine.name
        self.is_oracle = base_engine.is_oracle
        self.version = base_engine.version
        self.enable_quick_eval = base_engine.enable_quick_eval
        self.enable_rule_based_agari_guard = base_engine.enable_rule_based_agari_guard
        self.enable_amp = base_engine.enable_amp
        self.device = base_engine.device
        self.engine_type = base_engine.engine_type

    def react_batch(
        self,
        obs: Any,
        masks: Any,
        invisible_obs: Any,
        decision_contexts: Any,
    ) -> Any:
        actions, q_values, returned_masks, is_greedy = self.base_engine.react_batch(obs, masks, invisible_obs)
        out_actions = list(actions)
        for i, ctx in enumerate(decision_contexts):
            gen_seed, seed_key, seat, kyoku_idx, dec_idx, own_riichi, _ = ctx
            c_tuple = (int(gen_seed), int(seed_key), int(seat), int(kyoku_idx), int(dec_idx))

            q_row = [float(v) for v in q_values[i]]
            m_row = [bool(v) for v in returned_masks[i]]
            legal = [act for act, ok in enumerate(m_row) if ok and torch.isfinite(torch.tensor(q_row[act]))]
            ranked = sorted(legal, key=lambda a: (-q_row[a], a))

            if len(legal) >= 2 and not own_riichi and ranked[0] < 34 and ranked[1] < 34:
                self.recorded_contexts.append({
                    "ctx": c_tuple,
                    "top1": ranked[0],
                    "top2": ranked[1],
                    "margin": q_row[ranked[0]] - q_row[ranked[1]],
                })

            if self.target_context is not None and c_tuple == self.target_context:
                if self.forced_choice == "top1":
                    out_actions[i] = ranked[0]
                elif self.forced_choice == "top2":
                    out_actions[i] = ranked[1]
                self.intervened = True

        return out_actions, q_values, returned_masks, is_greedy


def _create_mortal_engine(name: str = "mortal", device: str = "cpu") -> engine.MortalEngine:
    k0_path, _ = resolve_k0_checkpoint()
    k0_state = torch.load(k0_path, map_location=device)
    m = model.Brain(version=4, conv_channels=192, num_blocks=40).eval()
    d = model.DQN(version=4).eval()
    m.load_state_dict(k0_state["mortal"])
    d.load_state_dict(k0_state["current_dqn"])
    return engine.MortalEngine(
        m,
        d,
        is_oracle=False,
        version=4,
        device=torch.device(device),
        name=name,
        enable_rule_based_agari_guard=True,
    )


def run_deterministic_rerun_branching_smoke(
    seed: int = 5000000,
    seed_key: int = 8192,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Execute deterministic rerun branching smoke and verify counterfactual feasibility."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Baseline run: discover decision context with >= 2 legal discards
        base_eng = _create_mortal_engine("E0", device=device)
        branch_eng = CounterfactualBranchingEngine(base_eng)
        e1 = _create_mortal_engine("E1", device=device)
        e2 = _create_mortal_engine("E2", device=device)
        e3 = _create_mortal_engine("E3", device=device)

        arena = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=str(tmp_path / "baseline"))
        arena.py_vs_py(branch_eng, e1, e2, e3, (seed, seed_key), 1)

        if not branch_eng.recorded_contexts:
            raise RuntimeError(f"No eligible branching context found in game seed ({seed}, {seed_key})")

        target = branch_eng.recorded_contexts[0]
        target_ctx = target["ctx"]

        # 2. Branch A: forced top1
        dir_a = tmp_path / "branch_a"
        dir_a.mkdir()
        eng_a = CounterfactualBranchingEngine(_create_mortal_engine("E0", device=device), target_context=target_ctx, forced_choice="top1")
        arena_a = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=str(dir_a))
        arena_a.py_vs_py(eng_a, _create_mortal_engine("E1", device=device), _create_mortal_engine("E2", device=device), _create_mortal_engine("E3", device=device), (seed, seed_key), 1)
        if not eng_a.intervened:
            raise RuntimeError("Branch A did not encounter target context")

        # 3. Branch B: forced top2
        dir_b = tmp_path / "branch_b"
        dir_b.mkdir()
        eng_b = CounterfactualBranchingEngine(_create_mortal_engine("E0", device=device), target_context=target_ctx, forced_choice="top2")
        arena_b = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=str(dir_b))
        arena_b.py_vs_py(eng_b, _create_mortal_engine("E1", device=device), _create_mortal_engine("E2", device=device), _create_mortal_engine("E3", device=device), (seed, seed_key), 1)
        if not eng_b.intervened:
            raise RuntimeError("Branch B did not encounter target context")

        # 4. Read logs and verify prefix parity
        log_a_path = next(iter(dir_a.glob("*.json.gz")))
        log_b_path = next(iter(dir_b.glob("*.json.gz")))
        with gzip.open(log_a_path, "rt", encoding="utf-8") as f:
            events_a = [json.loads(line) for line in f]
        with gzip.open(log_b_path, "rt", encoding="utf-8") as f:
            events_b = [json.loads(line) for line in f]

        if events_a[-1].get("type") != "end_game" or events_b[-1].get("type") != "end_game":
            raise RuntimeError("Both branches must complete a full hanchan ending in end_game")

        div_idx = -1
        for idx, (ea, eb) in enumerate(zip(events_a, events_b, strict=False)):
            if ea != eb:
                div_idx = idx
                break

        if div_idx <= 0:
            raise RuntimeError("Branches must match on prefix and diverge at target action")

        return {
            "schema": "keqing.mortal.p1_smoke_result.v1",
            "seed": seed,
            "seed_key": seed_key,
            "target_context": target_ctx,
            "top1_action": target["top1"],
            "top2_action": target["top2"],
            "margin": target["margin"],
            "events_before_divergence": div_idx,
            "branch_a_total_events": len(events_a),
            "branch_b_total_events": len(events_b),
            "both_branches_completed": True,
            "exact_prefix_matched": True,
            "verdict": "seed_replay_counterfactual_feasible",
        }


def run_p1_complete_audit() -> dict[str, Any]:
    """Run both direct snapshot API audit and deterministic rerun branching smoke."""
    snapshot_audit = audit_direct_snapshot_api()
    smoke_result = run_deterministic_rerun_branching_smoke()

    return {
        "schema": "keqing.mortal.p1_feasibility_audit.v2",
        "direct_snapshot_audit": snapshot_audit,
        "seed_replay_smoke": smoke_result,
        "verdict": smoke_result["verdict"],
    }


if __name__ == "__main__":
    res = run_p1_complete_audit()
    print(json.dumps(res, indent=2, ensure_ascii=False))
