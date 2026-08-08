from __future__ import annotations

from pathlib import Path

import pytest

from training.mortal.audit_d3_exploration_production_2026_08 import (
    DecisionSnapshot,
    audit_event_records,
    primary_row_flags,
)
from training.mortal.d3_exploration_engine import canonical_hash_u
from training.mortal.run_d3_exploration_production_2026_08 import (
    parse_args as parse_runner_args,
    run_single_native_b250,
)
from training.mortal.d3_production_contract import (
    AMP,
    GAMES,
    NATIVE_BATCH_GAMES,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    ContractError,
    assert_empty_output,
    validate_authoritative_smoke_protocol,
)


def test_first_production_gate_is_frozen_single_b250() -> None:
    assert GAMES == 250
    assert NATIVE_BATCH_GAMES == 250
    assert SEED_START == 1_800_000
    assert SEED_END_EXCLUSIVE == 1_800_250
    assert SEED_KEY == 8192
    assert AMP is False



def test_runner_exposes_no_resume_or_variable_batch_controls() -> None:
    args = parse_runner_args([])
    assert args.execute is False
    with pytest.raises(SystemExit):
        parse_runner_args(["--games", "25"])
    with pytest.raises(SystemExit):
        parse_runner_args(["--resume"])



def test_single_native_helper_makes_exact_frozen_call() -> None:
    class FakeArena:
        def __init__(self) -> None:
            self.calls = []

        def py_vs_py_random_seats(self, *args):
            self.calls.append(args)
            return "ok"

    arena = FakeArena()
    engines = [object(), object(), object(), object()]
    assert run_single_native_b250(arena, engines) == "ok"
    assert len(arena.calls) == 1
    assert arena.calls[0][:4] == tuple(engines)
    assert arena.calls[0][4] == (1_800_000, 8192)
    assert arena.calls[0][5] == 250


def test_gameplay_loader_auxiliary_kan_rows_do_not_increment_context() -> None:
    assert primary_row_flags([0, 42, 5, 1, 42, 9, 2]) == [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="missing its adjacent"):
        primary_row_flags([42])
    with pytest.raises(ValueError, match="invalid auxiliary"):
        primary_row_flags([42, 43])


def test_nonempty_output_cannot_resume(tmp_path: Path) -> None:
    (tmp_path / "partial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="delete the whole shard"):
        assert_empty_output(tmp_path)


def test_authoritative_smoke_manifest_requires_exact_lineage() -> None:
    models = {
        label: {"path": f"{label}.pth", "sha256": "a" * 64}
        for label in ("K0_70k", "V2_74000", "V3_74000", "ext_mortal")
    }
    protocol = {
        "schema": "keqing.mortal.d3_generation_smoke_protocol.v1",
        "contract_id": "D3_top2_discard_v1",
        "seed_start": 1_799_000,
        "seed_end_exclusive": 1_799_025,
        "seed_key": 8192,
        "games": 25,
        "native_batch_games": 25,
        "seat_mode": "random",
        "device": "cuda",
        "amp": False,
        "project_git_dirty": False,
        "mortal_source_dirty": False,
        "project_git_commit": "7eb48f1310e917cfb2f1f45445e546b4e92d1a89",
        "mortal_source_commit": "813859fc8110ea178f56f009994bc4f1b9fee645",
        "d3_native_patch_sha256": "fa7ff1e12687c3ba7ec4d5ea47902ded2c263f7695e0da195e0daabf437ffed1",
        "loaded_libriichi_sha256": "19bb181eaa70d0ae90417a3bd22433f6ca08d7654602f865ff3bdb102b7d9914",
        "native_build_profile": "release",
        "models": models,
        "engine_order": ["K0_70k", "V2_74000", "V3_74000", "ext_mortal"],
    }
    assert validate_authoritative_smoke_protocol(protocol) == []
    protocol["games"] = 250
    assert "authoritative smoke games mismatch" in validate_authoritative_smoke_protocol(protocol)


def test_event_audit_recomputes_hash_budget_and_loader_decision() -> None:
    context = (1_800_000, 8192, 2, 0, 0)
    canonical, digest, hash_u = canonical_hash_u(*context)
    expected_explored = hash_u < 0.25
    top1, top2 = 4, 5
    actual = top2 if expected_explored else top1
    reason = "explored" if expected_explored else "hash_rejected"
    snapshot = DecisionSnapshot(
        context=context,
        action=actual,
        own_riichi=False,
        shanten=2,
        phase="early",
        legal_action_count=3,
        finite_legal_actions=(4, 5, 6),
        top1_action=top1,
        top2_action=top2,
        top1_q=1.0,
        top2_q=0.8,
        margin=0.2,
        eligible=True,
    )
    event = {
        "contract_id": "D3_top2_discard_v1",
        "generation_seed": context[0],
        "seed_key": context[1],
        "seat": context[2],
        "kyoku_index": context[3],
        "decision_index": context[4],
        "own_riichi": False,
        "context_kind": "primary_action",
        "exploration_allowed": True,
        "exploration_probability": 0.25,
        "top1_action": top1,
        "top2_action": top2,
        "top1_q": 1.0,
        "top2_q": 0.8,
        "margin": 0.2,
        "hash_input": canonical,
        "hash_sha256": digest,
        "hash_u": hash_u,
        "kyoku_exploration_count_before": 0,
        "hanchan_exploration_count_before": 0,
        "actual_action": actual,
        "explored": expected_explored,
        "reason": reason,
        "base_action": top1,
    }
    result = audit_event_records([event], {context: snapshot})
    assert result["passed"] is True
    assert result["missing_event_count"] == 0


def test_event_audit_rejects_missing_independently_eligible_event() -> None:
    context = (1_800_000, 8192, 0, 0, 0)
    snapshot = DecisionSnapshot(
        context=context,
        action=1,
        own_riichi=False,
        shanten=1,
        phase="early",
        legal_action_count=2,
        finite_legal_actions=(1, 2),
        top1_action=1,
        top2_action=2,
        top1_q=0.9,
        top2_q=0.7,
        margin=0.2,
        eligible=True,
    )
    result = audit_event_records([], {context: snapshot})
    assert result["passed"] is False
    assert result["missing_event_count"] == 1
