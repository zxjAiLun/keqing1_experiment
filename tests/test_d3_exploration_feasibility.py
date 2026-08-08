from __future__ import annotations

import numpy as np

from training.mortal.audit_d3_exploration_feasibility_2026_08 import (
    DecisionRecord,
    MarginAccumulator,
    hard_finite_checks,
)


def _record(mask: list[bool], action: int) -> DecisionRecord:
    return DecisionRecord(
        obs=np.zeros(1, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.bool_),
        action=action,
        target=0.0,
        target_rank=1,
        phase="early",
        kyoku=0,
        current_rank=1,
        score_gap=0.0,
        own_riichi=False,
        legal_actions=sum(mask),
        shanten=2,
        action_kind="discard",
    )


def test_finite_q_partition_and_top2_margin_are_hard_checked() -> None:
    accumulator = MarginAccumulator()
    accumulator.update(_record([True, True, False], 0), np.asarray([1.0, 0.5, 99.0]))

    summary = accumulator.to_json()
    assert summary["states"] == 1
    assert summary["single_legal_action_states"] == 0
    assert summary["two_or_more_legal_action_states"] == 1
    assert summary["zero_finite_legal_action_states"] == 0
    assert summary["nonfinite_legal_q_values"] == 0
    assert all(hard_finite_checks(summary).values())
    assert summary["second_action_parent_q_regret"]["mean"] == 0.5


def test_nonfinite_legal_q_fails_hard_check() -> None:
    accumulator = MarginAccumulator()
    accumulator.update(_record([True, True, False], 0), np.asarray([1.0, np.inf, 99.0]))

    summary = accumulator.to_json()
    assert summary["nonfinite_q_states"] == 1
    assert summary["nonfinite_legal_q_values"] == 1
    assert not hard_finite_checks(summary)["nonfinite_legal_q_values_is_zero"]


def test_illegal_behavior_action_fails_hard_check() -> None:
    accumulator = MarginAccumulator()
    accumulator.update(_record([True, False, False], 1), np.asarray([1.0, 0.0, 99.0]))

    summary = accumulator.to_json()
    assert summary["behavior_action_illegal_count"] == 1
    assert not hard_finite_checks(summary)["behavior_action_legal_count_equals_states"]
