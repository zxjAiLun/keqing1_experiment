#!/usr/bin/env python3
"""Unit tests for the P4-M9 on-policy probe's pure contract logic.

The point of these tests is that the *checker* is trustworthy: an injected
sampling/execution divergence, mask mismatch or missing log event must be
reported as a failure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p4m9_probe_onpolicy import (
    ACTION_AGARI,
    ACTION_CHI_HIGH,
    ACTION_CHI_LOW,
    ACTION_CHI_MID,
    ACTION_KAN,
    ACTION_PASS,
    ACTION_PON,
    ACTION_RIICHI,
    ACTION_RYUKYOKU,
    ACTION_SPACE,
    bits_to_mask,
    chi_action_index,
    compact_legal,
    compatible_with_action,
    event_to_action,
    hanchan_return,
    legal_logits,
    mask_to_bits,
    pg_loss,
    q_values_match,
    reconcile_kyoku,
    sampling_log_prob,
    tile_index,
)


def _q_with_legal(legal: list[int], value: float = 1.0) -> tuple[list[float], list[bool]]:
    mask = [False] * ACTION_SPACE
    q = [-float("inf")] * ACTION_SPACE
    for index, raw in enumerate(legal):
        mask[index] = True
        q[index] = value + index
    return q, mask


# ---------------------------------------------------------------- tile mapping
def test_tile_index_matches_libriichi_order() -> None:
    assert tile_index("1m") == 0
    assert tile_index("9m") == 8
    assert tile_index("1p") == 9
    assert tile_index("9s") == 26
    assert tile_index("E") == 27
    assert tile_index("S") == 28
    assert tile_index("W") == 29
    assert tile_index("N") == 30
    assert tile_index("P") == 31
    assert tile_index("F") == 32
    assert tile_index("C") == 33
    assert tile_index("5mr") == 34
    assert tile_index("5pr") == 35
    assert tile_index("5sr") == 36


# ------------------------------------------------------------------- masks/ops
def test_mask_bits_roundtrip() -> None:
    mask = [False] * ACTION_SPACE
    for action in (0, 5, 37, 43, 45):
        mask[action] = True
    bits = mask_to_bits(mask)
    assert bits_to_mask(bits) == mask
    assert mask_to_bits([False] * ACTION_SPACE) == 0


def test_compact_legal_keeps_ascending_action_order() -> None:
    q, mask = _q_with_legal([3, 7, 11], value=0.5)
    assert compact_legal(q, mask) == [0.5, 1.5, 2.5]


# ------------------------------------------------------------- sampling logprob
def test_sampling_log_prob_is_log_softmax_over_legal_actions() -> None:
    q, mask = _q_with_legal([0, 1, 2], value=0.0)
    values = np.array([0.0, 1.0, 2.0])
    expected = values - (2.0 + math.log(math.exp(-2.0) + math.exp(-1.0) + 1.0))
    for action in (0, 1, 2):
        assert sampling_log_prob(q, mask, action) == pytest.approx(float(expected[action]))
    assert float(np.sum(np.exp(expected))) == pytest.approx(1.0)


def test_sampling_log_prob_rejects_illegal_action() -> None:
    q, mask = _q_with_legal([0], value=0.0)
    assert sampling_log_prob(q, mask, 5) == float("-inf")


def test_sampling_log_prob_respects_temperature() -> None:
    q, mask = _q_with_legal([0, 1], value=0.0)
    hot = sampling_log_prob(q, mask, 1, temperature=2.0)
    cold = sampling_log_prob(q, mask, 1, temperature=0.5)
    assert hot < cold  # lower temperature sharpens the distribution


def test_legal_logits_are_neg_inf_on_illegal() -> None:
    q, mask = _q_with_legal([0, 1], value=0.0)
    logits = legal_logits(q, mask)
    assert np.isneginf(logits[2])
    assert np.isfinite(logits[0]) and np.isfinite(logits[1])


# ------------------------------------------------------------------- chi mapping
def test_chi_action_index_classifies_low_mid_high() -> None:
    assert chi_action_index("3m", ["4m", "5m"]) == ACTION_CHI_LOW
    assert chi_action_index("4m", ["3m", "5m"]) == ACTION_CHI_MID
    assert chi_action_index("5m", ["3m", "4m"]) == ACTION_CHI_HIGH


def test_chi_action_index_normalises_aka() -> None:
    assert chi_action_index("3m", ["4m", "5mr"]) == ACTION_CHI_LOW
    assert chi_action_index("5p", ["3p", "4p"]) == ACTION_CHI_HIGH
    assert chi_action_index("1m", ["2m", "4m"]) is None


# ------------------------------------------------------------------ event -> action
def test_event_to_action_covers_every_decision_type() -> None:
    assert event_to_action({"type": "dahai", "pai": "1m"}) == 0
    assert event_to_action({"type": "dahai", "pai": "5sr"}) == 36
    assert event_to_action({"type": "reach"}) == ACTION_RIICHI
    assert event_to_action({"type": "pon"}) == ACTION_PON
    assert event_to_action({"type": "ankan"}) == ACTION_KAN
    assert event_to_action({"type": "kakan"}) == ACTION_KAN
    assert event_to_action({"type": "daiminkan"}) == ACTION_KAN
    assert event_to_action({"type": "hora"}) == ACTION_AGARI
    assert event_to_action({"type": "ryukyoku"}) == ACTION_RYUKYOKU
    assert event_to_action({"type": "none"}) == ACTION_PASS
    assert event_to_action({"type": "tsumo", "pai": "1m"}) is None


def test_compatible_with_action_requires_matching_class() -> None:
    assert compatible_with_action(0, "dahai")
    assert not compatible_with_action(0, "reach")
    assert compatible_with_action(ACTION_RIICHI, "reach")
    assert compatible_with_action(ACTION_CHI_MID, "chi")
    assert compatible_with_action(ACTION_KAN, "daiminkan")
    assert not compatible_with_action(ACTION_AGARI, "dahai")


def test_q_values_match_is_float32_tolerant_but_not_sloppy() -> None:
    base = [1.0, 2.0, -3.5]
    assert q_values_match(base, [np.float32(v) for v in base])
    assert not q_values_match(base, [1.0, 2.0])
    assert not q_values_match(base, [1.0, 2.0, -3.6])
    assert q_values_match([1e-8], [2e-8])


# ------------------------------------------------------------------------ loss
def test_pg_loss_sums_within_hanchan_and_averages_across_hanchans() -> None:
    log_probs = torch.tensor([-0.5, -1.0, -0.25, -2.0], dtype=torch.float32)
    hanchan_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    returns = torch.tensor([1.0, 3.0], dtype=torch.float32)
    loss = pg_loss(log_probs, hanchan_ids, returns, 2, baseline=0.0)
    expected = -((1.0 * (-1.5)) + (3.0 * (-2.25))) / 2
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_pg_loss_with_zero_baseline_scales_with_return() -> None:
    log_probs = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    hanchan_ids = torch.tensor([0, 1], dtype=torch.long)
    one = pg_loss(log_probs, hanchan_ids, torch.tensor([1.0, 1.0]), 2, baseline=0.0)
    two = pg_loss(log_probs, hanchan_ids, torch.tensor([2.0, 2.0]), 2, baseline=0.0)
    assert float(two) == pytest.approx(2 * float(one))


def test_pg_loss_baseline_is_subtracted_from_the_return() -> None:
    log_probs = torch.tensor([-1.0], dtype=torch.float32)
    hanchan_ids = torch.tensor([0], dtype=torch.long)
    returns = torch.tensor([5.0], dtype=torch.float32)
    assert float(pg_loss(log_probs, hanchan_ids, returns, 1, baseline=5.0)) == 0.0


def test_pg_loss_chunked_accumulation_equals_full_batch() -> None:
    """Gradient accumulation over micro-batches must equal the full-batch loss."""
    generator = torch.Generator().manual_seed(7)
    log_probs = torch.randn(40, generator=generator)
    hanchan_ids = torch.tensor([index // 8 for index in range(40)], dtype=torch.long)
    returns = torch.arange(5, dtype=torch.float32)
    full = pg_loss(log_probs, hanchan_ids, returns, 5, baseline=0.0)
    chunked = sum(
        pg_loss(
            log_probs[start:start + 9],
            hanchan_ids[start:start + 9],
            returns,
            5,
            baseline=0.0,
        )
        for start in range(0, 40, 9)
    )
    assert float(chunked) == pytest.approx(float(full), rel=1e-6)


def test_pg_loss_requires_positive_hanchan_count() -> None:
    with pytest.raises(ValueError):
        pg_loss(torch.zeros(1), torch.zeros(1, dtype=torch.long), torch.zeros(1), 0)


def test_hanchan_return_is_a_declared_placeholder() -> None:
    assert hanchan_return(25000) == 0.0
    assert hanchan_return(35000) == pytest.approx(10.0)
    assert hanchan_return(5000) == pytest.approx(-20.0)


# ------------------------------------------------------------------ reconciler
# A probe record and a logged decision event carry the same q/mask identity, so
# the helpers below produce matching identities by default.
IDENTITY_MASK = 0b111
IDENTITY_Q = [1.0, 2.0, 3.0]


def _probe_record(
    dec: int,
    action: int,
    *,
    explore: bool = True,
    mask_bits: int = IDENTITY_MASK,
    q_legal: list[float] | None = None,
) -> dict:
    return {
        "dec": dec,
        "explore": explore,
        "action": action,
        "mask_bits": mask_bits,
        "q_legal": list(q_legal if q_legal is not None else IDENTITY_Q),
        "logprob": -1.0,
    }


def _log_event(
    event_type: str,
    action: int | None,
    *,
    mask_bits: int = IDENTITY_MASK,
    q_values: list[float] | None = None,
    kan_select: dict | None = None,
) -> dict:
    meta: dict = {
        "mask_bits": mask_bits,
        "q_values": list(q_values if q_values is not None else IDENTITY_Q),
    }
    if kan_select is not None:
        meta["kan_select"] = kan_select
    return {"type": event_type, "meta": meta, "observed_action": action}


def test_reconcile_reports_consistency_when_sampled_action_was_executed() -> None:
    records = [_probe_record(0, 5), _probe_record(1, ACTION_RIICHI)]
    events = [_log_event("dahai", 5), _log_event("reach", ACTION_RIICHI)]
    report = reconcile_kyoku(records, events)
    assert report["aligned"] == 2
    assert not report["mismatches"]
    assert not report["executed_action_differs"]
    assert report["unmatched_log_events"] == 0
    assert report["alignment_complete"]
    assert report["sampled_action_equals_executed_action"]


def test_reconcile_detects_a_different_executed_action() -> None:
    """Negative control: same decision identity, different executed action."""
    records = [_probe_record(0, 5)]
    events = [_log_event("dahai", 9)]
    report = reconcile_kyoku(records, events)
    assert report["aligned"] == 0
    assert report["executed_action_differs"][0]["executed_action"] == 9
    assert report["alignment_complete"]
    assert not report["sampled_action_equals_executed_action"]


def test_reconcile_does_not_match_on_mask_alone() -> None:
    """Different q values mean a different decision, so a discard cannot match."""
    records = [_probe_record(0, 5)]
    events = [_log_event("dahai", 5, q_values=[1.0, 2.0, 3.5])]
    report = reconcile_kyoku(records, events)
    assert report["aligned"] == 0
    assert report["mismatches"][0]["kind"] == "discard_without_executed_event"
    assert report["unmatched_log_events"] == 1
    assert not report["alignment_complete"]


def test_reconcile_does_not_match_on_q_alone() -> None:
    records = [_probe_record(0, 5)]
    events = [_log_event("dahai", 5, mask_bits=0b101)]
    report = reconcile_kyoku(records, events)
    assert report["aligned"] == 0
    assert report["mismatches"]


def test_reconcile_accepts_an_honoured_agari() -> None:
    """Sampled agari, kyoku resolved for this seat -> honoured, no rewrite."""
    records = [_probe_record(0, ACTION_AGARI, mask_bits=0b1001, q_legal=[0.25])]
    report = reconcile_kyoku(
        records,
        [],
        terminal={"hora_actors": [2], "ended_by": "hora"},
        challenger_seat=2,
    )
    assert report["sampled_agari"] == 1
    assert not report["agari_guard_suspected"]
    assert report["sampled_action_equals_executed_action"]


def test_reconcile_accepts_a_double_ron_where_the_seat_also_won() -> None:
    """A second winner must not be mistaken for an agari-guard rewrite."""
    records = [_probe_record(0, ACTION_AGARI)]
    report = reconcile_kyoku(
        records,
        [],
        terminal={"hora_actors": [2, 3], "ended_by": "hora"},
        challenger_seat=3,
    )
    assert report["sampled_agari"] == 1
    assert not report["agari_guard_suspected"]


def test_reconcile_flags_an_agari_the_kyoku_did_not_resolve() -> None:
    """Negative control for the rust agari guard."""
    records = [_probe_record(0, ACTION_AGARI, mask_bits=0b1001, q_legal=[0.25])]
    report = reconcile_kyoku(
        records,
        [],
        terminal={"hora_actors": [3], "ended_by": "hora"},
        challenger_seat=2,
    )
    assert report["agari_guard_suspected"][0]["sampled_action"] == ACTION_AGARI
    assert report["agari_guard_suspected"][0]["kyoku_winners"] == [3]
    assert not report["sampled_action_equals_executed_action"]


def test_reconcile_flags_an_agari_rewritten_into_a_discard() -> None:
    """The exact signature seen in the wild: identity match, discard executed."""
    records = [_probe_record(0, ACTION_AGARI)]
    events = [_log_event("dahai", 30)]
    report = reconcile_kyoku(records, events)
    assert report["executed_action_differs"][0]["sampled_action"] == ACTION_AGARI
    assert report["executed_action_differs"][0]["executed_action"] == 30
    assert not report["sampled_action_equals_executed_action"]


def test_reconcile_classifies_a_superseded_claim() -> None:
    """A legal claim another player pre-empted is reported, not silently dropped."""
    records = [_probe_record(0, ACTION_PON, mask_bits=0b1000, q_legal=[0.5])]
    report = reconcile_kyoku(
        records,
        [],
        terminal={"hora_actors": [2], "ended_by": "hora", "last_discard_actor": 3},
        challenger_seat=1,
    )
    assert report["claim_not_executed"][0]["sampled_action"] == ACTION_PON
    assert report["claim_not_executed"][0]["kyoku_winners"] == [2]
    assert not report["mismatches"]
    assert report["alignment_complete"]
    assert not report["sampled_action_equals_executed_action"]


def test_reconcile_does_not_hide_a_claim_behind_the_seat_own_win() -> None:
    records = [_probe_record(0, ACTION_PON)]
    report = reconcile_kyoku(
        records,
        [],
        terminal={"hora_actors": [1], "ended_by": "hora"},
        challenger_seat=1,
    )
    assert report["claim_not_executed"]
    assert not report["mismatches"]


def test_reconcile_pass_and_ryukyoku_consume_no_log_event() -> None:
    records = [
        _probe_record(0, ACTION_PASS),
        _probe_record(1, ACTION_RYUKYOKU),
        _probe_record(2, 3),
    ]
    events = [_log_event("dahai", 3)]
    report = reconcile_kyoku(records, events)
    assert report["sampled_pass"] == 1
    assert report["sampled_ryukyoku"] == 1
    assert report["aligned"] == 1
    assert not report["mismatches"]
    assert report["alignment_complete"]


def test_reconcile_keeps_the_comparison_aligned_after_a_skipped_action() -> None:
    """A single non-executed action must not shift every later comparison."""
    records = [
        _probe_record(0, ACTION_PON, mask_bits=0b1000, q_legal=[0.5]),
        _probe_record(1, 7),
        _probe_record(2, 8),
    ]
    events = [_log_event("dahai", 7), _log_event("dahai", 8)]
    report = reconcile_kyoku(
        records, events, terminal={"hora_actors": [0]}, challenger_seat=1
    )
    assert report["claim_not_executed"]
    assert report["aligned"] == 2
    assert not report["mismatches"]
    assert report["unmatched_log_events"] == 0
    assert report["alignment_complete"]


def test_reconcile_reports_unmatched_log_events() -> None:
    records = [_probe_record(0, 5)]
    events = [_log_event("dahai", 5), _log_event("dahai", 6)]
    report = reconcile_kyoku(records, events)
    assert report["unmatched_log_events"] == 1
    assert not report["alignment_complete"]


def test_reconcile_excludes_kan_select_states_from_policy_decisions() -> None:
    kan_meta = {"mask_bits": 0b100, "q_values": [9.0]}
    records = [
        _probe_record(0, ACTION_KAN, explore=False, mask_bits=0b100, q_legal=[9.0]),
        _probe_record(0, ACTION_KAN),
    ]
    events = [_log_event("ankan", ACTION_KAN, kan_select=kan_meta)]
    report = reconcile_kyoku(records, events)
    assert report["kan_select_states"] == 1
    assert report["policy_decisions"] == 1
    assert report["aligned"] == 1
    assert report["kan_select_checks"] == 1
    assert not report["kan_select_mismatches"]


def test_reconcile_detects_a_kan_select_mismatch() -> None:
    kan_meta = {"mask_bits": 0b100, "q_values": [9.0]}
    records = [
        _probe_record(0, ACTION_KAN, explore=False, mask_bits=0b100, q_legal=[8.0]),
        _probe_record(0, ACTION_KAN),
    ]
    events = [_log_event("ankan", ACTION_KAN, kan_select=kan_meta)]
    report = reconcile_kyoku(records, events)
    assert report["kan_select_mismatches"]


def test_reconcile_counts_every_decision_in_exactly_one_class() -> None:
    records = [
        _probe_record(0, ACTION_PASS),
        _probe_record(1, 5),
        _probe_record(2, ACTION_AGARI, mask_bits=0b1001, q_legal=[0.25]),
        _probe_record(3, ACTION_PON, mask_bits=0b1000, q_legal=[0.5]),
        _probe_record(4, 9),
    ]
    events = [_log_event("dahai", 5), _log_event("dahai", 9)]
    report = reconcile_kyoku(
        records,
        events,
        terminal={"hora_actors": [0], "ended_by": "hora"},
        challenger_seat=1,
    )
    accounted = (
        report["aligned"]
        + report["sampled_agari"]
        + report["sampled_ryukyoku"]
        + report["sampled_pass"]
        + len(report["executed_action_differs"])
        + len(report["claim_not_executed"])
    )
    assert accounted == report["policy_decisions"] == 5
