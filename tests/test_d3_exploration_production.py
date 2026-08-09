from __future__ import annotations

from pathlib import Path

import pytest

from training.mortal.d3_exploration_engine import canonical_hash_u
from training.mortal.d3_native_scene import chi_label, tile_label
from training.mortal.d3_production_audit_core import (
    DecisionSnapshot,
    primary_row_flags,
)
from training.mortal.d3_production_event_audit import audit_event_records
from training.mortal.run_d3_exploration_production_2026_08 import (
    parse_args as parse_runner_args,
    run_single_native_b250,
)
from training.mortal.d3_production_contract import (
    AMP,
    AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
    D3_SEMANTIC_PATHS,
    EXPECTED_BRANCH,
    GAMES,
    LEGACY_TRAINING_SOURCE_COMMIT,
    MIGRATION_CONTENT_COMMIT,
    NATIVE_BATCH_GAMES,
    PRODUCTION_IMPLEMENTATION_PATHS,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    TRAINING_TRANSFER_ANCHOR,
    ContractError,
    _project_lineage_facts,
    assert_empty_output,
    implementation_manifest,
    project_lineage,
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


def _valid_event_and_snapshot(seed: int = 1_800_000, seat: int = 2) -> tuple[dict, DecisionSnapshot]:
    context = (seed, 8192, seat, 0, 0)
    canonical, digest, hash_u = canonical_hash_u(*context)
    explored = hash_u < 0.25
    top1, top2 = 4, 5
    actual = top2 if explored else top1
    reason = "explored" if explored else "hash_rejected"
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
        "top1_action": top1,
        "top2_action": top2,
        "top1_q": 1.0,
        "top2_q": 0.8,
        "margin": 0.2,
        "hash_input": canonical,
        "hash_sha256": digest,
        "hash_u": hash_u,
        "exploration_probability": 0.25,
        "kyoku_exploration_count_before": 0,
        "hanchan_exploration_count_before": 0,
        "actual_action": actual,
        "explored": explored,
        "reason": reason,
        "base_action": top1,
    }
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
    return event, snapshot


def test_event_contract_recomputes_hash_budget_and_action() -> None:
    event, snapshot = _valid_event_and_snapshot()
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is True
    assert result["missing_event_count"] == 0
    assert result["contract_violations"] == {}
    assert result["mapping_violations"] == {}
    assert result["explored_count"] == (1 if event["explored"] else 0)


def event_context(event: dict) -> tuple[int, int, int, int, int]:
    return (
        int(event["generation_seed"]),
        int(event["seed_key"]),
        int(event["seat"]),
        int(event["kyoku_index"]),
        int(event["decision_index"]),
    )


def test_event_contract_rejects_margin_identity_break() -> None:
    event, snapshot = _valid_event_and_snapshot()
    event["margin"] = 0.5
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is False
    assert result["contract_violations"]["margin_identity"] == 1


def test_event_contract_rejects_hash_digest_break() -> None:
    event, snapshot = _valid_event_and_snapshot()
    event["hash_sha256"] = "0" * 64
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is False
    assert result["contract_violations"]["hash"] == 1


def test_event_contract_rejects_wrong_reason_and_action() -> None:
    event, snapshot = _valid_event_and_snapshot()
    event["reason"] = "kyoku_budget_exhausted"
    event["actual_action"] = int(event["top1_action"])
    event["explored"] = False
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is False
    assert result["contract_violations"]["budget_reason"] == 1
    assert result["contract_violations"]["actual_action"] == 2


def test_event_contract_rejects_wrong_base_action() -> None:
    event, snapshot = _valid_event_and_snapshot()
    event["base_action"] = int(event["top2_action"])
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is False
    assert result["contract_violations"]["base_action"] == 1


def test_mapping_rejects_event_without_native_scene() -> None:
    event, snapshot = _valid_event_and_snapshot()
    result = audit_event_records([event], {})
    assert result["passed"] is False
    assert result["mapping_violations"]["unmapped_context"] == 1
    assert result["missing_event_count"] == 1


def test_mapping_rejects_behavior_action_mismatch() -> None:
    event, snapshot = _valid_event_and_snapshot()
    snapshot = DecisionSnapshot(
        context=snapshot.context,
        action=int(event["top1_action"]),
        own_riichi=False,
        shanten=2,
        phase="early",
        legal_action_count=3,
        finite_legal_actions=(4, 5, 6),
        top1_action=4,
        top2_action=5,
        top1_q=1.0,
        top2_q=0.8,
        margin=0.2,
        eligible=True,
    )
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is False
    assert result["mapping_violations"]["behavior_mismatch"] == 1


def test_replay_q_diagnostics_are_descriptive_not_gating() -> None:
    event, snapshot = _valid_event_and_snapshot()
    snapshot = DecisionSnapshot(
        context=snapshot.context,
        action=snapshot.action,
        own_riichi=False,
        shanten=2,
        phase="early",
        legal_action_count=3,
        finite_legal_actions=(4, 5, 6),
        top1_action=5,
        top2_action=4,
        top1_q=1.05,
        top2_q=0.75,
        margin=0.3,
        eligible=True,
    )
    result = audit_event_records([event], {event_context(event): snapshot})
    assert result["passed"] is True
    diag = result["q_diagnostics"]
    assert diag["ranking_flip_count"] == 1
    assert diag["eligibility_flip_count"] == 0
    assert diag["abs_diff_top1_q"]["max"] > 0


def test_native_scene_tile_labels() -> None:
    assert tile_label("5m") == 4
    assert tile_label("9s") == 26
    assert tile_label("1p") == 9
    assert tile_label("W") == 29
    assert tile_label("5mr") == 34
    assert tile_label("5pr") == 35
    assert tile_label("5sr") == 36


def test_native_scene_chi_labels() -> None:
    assert chi_label(["6m", "7m"], "5m") == 38
    assert chi_label(["4m", "6m"], "5m") == 39
    assert chi_label(["3m", "4m"], "5m") == 40
    assert chi_label(["5mr", "7m"], "6m") == 39
    assert chi_label(["5m", "7m"], "6m") == 39
    assert chi_label(["W", "7m"], "6m") is None


class _FakeCans:
    def __init__(self, can_ryukyoku: bool = False) -> None:
        self.can_ryukyoku = can_ryukyoku
        self.can_chi = False
        self.can_pon = False
        self.can_daiminkan = False
        self.can_ron_agari = False


class _FakeState:
    def __init__(self, can_ryukyoku: bool = False) -> None:
        self.last_cans = _FakeCans(can_ryukyoku)


def test_native_scene_ryukyoku_label_requires_can_ryukyoku() -> None:
    from training.mortal.d3_native_scene import expected_label

    # draw followed by ryukyoku with the player able to vote -> label 44
    events = [
        {"type": "tsumo", "actor": 1, "pai": "6p"},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    assert expected_label(events, 0, seat=1, state=_FakeState(can_ryukyoku=True)) == 44
    # same draw but the player cannot vote (exhaustive-draw edge): no loader row
    assert expected_label(events, 0, seat=1, state=_FakeState(can_ryukyoku=False)) is None


def test_lineage_constants_target_migrated_repo_identity() -> None:
    assert EXPECTED_BRANCH == "main"
    assert TRAINING_TRANSFER_ANCHOR == "74a3154d0c543b805a75e679ab93c74f2afbefaf"
    assert MIGRATION_CONTENT_COMMIT == "8e3b58f50c08c3c9ad795ea63c1af44e3b5ed11b"
    assert LEGACY_TRAINING_SOURCE_COMMIT == "6ff580cb"
    assert AUTHORITATIVE_SMOKE_PROJECT_COMMIT == "7eb48f1310e917cfb2f1f45445e546b4e92d1a89"
    assert TRAINING_TRANSFER_ANCHOR != AUTHORITATIVE_SMOKE_PROJECT_COMMIT


def test_lineage_paths_use_training_mortal_layout() -> None:
    assert all(p.startswith("training/mortal/") for p in D3_SEMANTIC_PATHS)
    assert all(p.startswith("training/mortal/") for p in PRODUCTION_IMPLEMENTATION_PATHS)
    assert "training/mortal/d3_exploration_engine.py" in D3_SEMANTIC_PATHS
    assert "training/mortal/patches/libriichi_d3_decision_context.patch" in D3_SEMANTIC_PATHS
    assert "scripts/mortal/" not in "".join(PRODUCTION_IMPLEMENTATION_PATHS)


def test_implementation_manifest_resolves_new_paths_without_contract_error() -> None:
    manifest = implementation_manifest()
    assert set(manifest) == set(PRODUCTION_IMPLEMENTATION_PATHS)
    for relative, row in manifest.items():
        assert relative.startswith("training/mortal/")
        assert len(row["sha256"]) == 64


def test_lineage_pass_case_requires_main_transfer_ancestor_and_clean() -> None:
    result = _project_lineage_facts(
        branch="main",
        commit="a" * 40,
        dirty_entries=[],
        transfer_anchor_is_ancestor=True,
        semantic_diff_paths=[],
    )
    assert result["passed"] is True
    assert result["errors"] == []
    assert result["transfer_anchor"] == TRAINING_TRANSFER_ANCHOR
    assert result["authoritative_smoke_commit"] == AUTHORITATIVE_SMOKE_PROJECT_COMMIT
    assert result["legacy_training_source_commit"] == LEGACY_TRAINING_SOURCE_COMMIT


def test_lineage_rejects_non_main_branch() -> None:
    result = _project_lineage_facts(
        branch="codex/mortal-training-next",
        commit="a" * 40,
        dirty_entries=[],
        transfer_anchor_is_ancestor=True,
        semantic_diff_paths=[],
    )
    assert result["passed"] is False
    assert any("branch must be main" in e for e in result["errors"])


def test_lineage_rejects_missing_transfer_anchor_ancestry() -> None:
    result = _project_lineage_facts(
        branch="main",
        commit="a" * 40,
        dirty_entries=[],
        transfer_anchor_is_ancestor=False,
        semantic_diff_paths=[],
    )
    assert result["passed"] is False
    assert any("transfer anchor" in e for e in result["errors"])


def test_lineage_rejects_semantic_path_change_since_anchor() -> None:
    result = _project_lineage_facts(
        branch="main",
        commit="a" * 40,
        dirty_entries=[],
        transfer_anchor_is_ancestor=True,
        semantic_diff_paths=["training/mortal/d3_exploration_engine.py"],
    )
    assert result["passed"] is False
    assert any("semantic paths changed" in e for e in result["errors"])


def test_lineage_rejects_dirty_worktree() -> None:
    result = _project_lineage_facts(
        branch="main",
        commit="a" * 40,
        dirty_entries=[" M training/mortal/d3_production_contract.py"],
        transfer_anchor_is_ancestor=True,
        semantic_diff_paths=[],
    )
    assert result["passed"] is False
    assert any("worktree is dirty" in e for e in result["errors"])


def test_lineage_does_not_require_legacy_smoke_commit_as_git_ancestor() -> None:
    result = _project_lineage_facts(
        branch="main",
        commit="a" * 40,
        dirty_entries=[],
        transfer_anchor_is_ancestor=True,
        semantic_diff_paths=[],
    )
    assert result["passed"] is True
    assert result["authoritative_smoke_commit"] == AUTHORITATIVE_SMOKE_PROJECT_COMMIT
    assert result["authoritative_smoke_commit"] != result["transfer_anchor"]


def test_migrated_repo_project_lineage_passes_on_main() -> None:
    lineage = project_lineage()
    assert lineage["branch"] == "main"
    assert lineage["transfer_anchor_is_ancestor"] is True
    assert lineage["semantic_diff_paths"] == []
    assert lineage["authoritative_smoke_commit"] == AUTHORITATIVE_SMOKE_PROJECT_COMMIT
    assert lineage["migration_content_commit"] == MIGRATION_CONTENT_COMMIT
    assert lineage["legacy_training_source_commit"] == LEGACY_TRAINING_SOURCE_COMMIT
