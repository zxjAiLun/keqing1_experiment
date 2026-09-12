"""Tests for the P4-M10 fixed-panel policy-displacement check (read-only).

The point of these tests is that a displacement number is only meaningful if the
measurement itself is trustworthy, so what is pinned here is:

* the metrics are *exact* on hand-computable inputs (a wrong TV or KL would
  silently misreport how far the policy moved);
* illegal actions never contribute to any metric (a mask leak would deflate TV);
* a flip is detected on argmax over legal actions only;
* the panel loader refuses a layout it cannot index safely, because a silent
  mis-indexing of a 5 GB observation file would produce a plausible wrong answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p4m10_policy_displacement import (
    FLIP_MARGIN_NOTE,
    INTERPRETATION_BOUNDARIES,
    METRIC_ROLES,
    PANEL_SUBSTITUTION,
    legal_action_count,
    legal_mask_from_records,
    load_panel,
    log_softmax64,
    pair_metrics,
    seed_clustered_ci,
    top_two_margin,
)

ACTION_SPACE = 46


def q_row(legal: dict[int, float]) -> np.ndarray:
    row = np.full(ACTION_SPACE, -np.inf, dtype=np.float64)
    for index, value in legal.items():
        row[index] = value
    return row


def test_identical_policies_have_zero_displacement():
    q = np.stack([q_row({0: 1.0, 1: 0.5, 2: -3.0})] * 4)
    legal = np.isfinite(q)  # the mask implied by the rows, as in production
    metrics = pair_metrics(q, q, legal)
    assert np.all(metrics["flip"] == 0.0)
    assert np.allclose(metrics["tv"], 0.0)
    assert np.allclose(metrics["kl_ab"], 0.0)
    assert np.allclose(metrics["kl_ba"], 0.0)
    # q_delta is nan off-mask by design, so compare only the legal entries.
    assert np.allclose(metrics["q_delta"][legal], 0.0)


def test_flip_is_argmax_over_legal_actions():
    # Same two legal actions, the preference order swapped.
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 2.0, 1: 1.0})])
    b = np.stack([q_row({0: 1.0, 1: 2.0})])
    assert pair_metrics(a, b, both)["flip"][0] == 1.0
    # An illegal action with a huge score must not become the argmax.
    a = np.stack([q_row({0: 1.0, 1: 0.0})])
    b = np.stack([q_row({0: 1.0, 1: 0.0})])
    b[0][7] = 1e6
    assert pair_metrics(a, b, both)["flip"][0] == 0.0


def test_tv_matches_hand_computed_value():
    # Two legal actions, logits (0, 0) vs (ln 3, 0):
    #   pi_a = (1/2, 1/2), pi_b = (3/4, 1/4)  ->  TV = 1/4
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 0.0, 1: 0.0})])
    b = np.stack([q_row({0: float(np.log(3.0)), 1: 0.0})])
    metrics = pair_metrics(a, b, both)
    assert metrics["tv"][0] == pytest.approx(0.25, abs=1e-12)
    assert metrics["tv"][0] <= 1.0


def test_kl_matches_hand_computed_value():
    # pi_a = (1/2, 1/2), pi_b = (3/4, 1/4)
    #   KL(a||b) = 1/2 ln(2/3) + 1/2 ln 2 = 0.5 * ln(4/3)
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 0.0, 1: 0.0})])
    b = np.stack([q_row({0: float(np.log(3.0)), 1: 0.0})])
    metrics = pair_metrics(a, b, both)
    assert metrics["kl_ab"][0] == pytest.approx(0.5 * np.log(4.0 / 3.0), abs=1e-12)
    assert metrics["kl_ba"][0] > 0.0


def test_illegal_actions_do_not_contribute_to_any_metric():
    """The legal set is a property of the state, so scores off it are ignored."""
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 0.0, 1: 0.0})])
    b = q_row({0: 0.0, 1: 0.0})
    # Pile wildly different scores onto actions the state does not allow.
    b[10] = 500.0
    b[11] = -500.0
    metrics = pair_metrics(a, np.stack([b]), both)
    assert metrics["tv"][0] == pytest.approx(0.0, abs=1e-12)
    assert metrics["kl_ab"][0] == pytest.approx(0.0, abs=1e-12)
    assert metrics["flip"][0] == 0.0
    assert np.all(np.isnan(metrics["q_delta"][0][~both[0]]))


def test_non_finite_score_on_a_legal_action_is_rejected():
    """A partially-finite row would silently misreport the displacement."""
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 0.0})])  # action 1 is legal but -inf
    b = np.stack([q_row({0: 0.0, 1: 0.0})])
    with pytest.raises(ValueError, match="not finite"):
        pair_metrics(a, b, both)


def test_legal_mask_from_records_uses_mask_bits():
    records = [{"mask_bits": (1 << 0) | (1 << 2)}, {"mask_bits": 1 << 5}]
    legal = legal_mask_from_records(records)
    assert legal.shape == (2, ACTION_SPACE)
    assert legal[0].sum() == 2 and legal[0][0] and legal[0][2]
    assert legal[1].sum() == 1 and legal[1][5]


def test_q_delta_is_nan_only_on_illegal_entries():
    both = np.zeros((1, ACTION_SPACE), dtype=bool)
    both[0, [0, 1]] = True
    a = np.stack([q_row({0: 5.0, 1: 5.0})])
    b = np.stack([q_row({0: 3.0, 1: 5.0})])
    delta = pair_metrics(a, b, both)["q_delta"][0]
    assert delta[0] == pytest.approx(2.0)
    assert delta[1] == pytest.approx(0.0)
    assert np.isnan(delta[2:]).all()


def test_top_two_margin_and_single_legal_action():
    legal = np.zeros((1, ACTION_SPACE), dtype=bool)
    legal[0, [0, 1]] = True
    margin = top_two_margin(np.stack([q_row({0: 4.0, 1: 1.0})]), legal)
    assert margin[0] == pytest.approx(3.0)

    only_one = np.zeros((1, ACTION_SPACE), dtype=bool)
    only_one[0, 0] = True
    assert top_two_margin(np.stack([q_row({0: 4.0})]), only_one)[0] == 0.0


def test_log_softmax_keeps_illegal_at_minus_inf_and_normalises():
    log_p = log_softmax64(np.stack([q_row({0: 1.0, 2: 1.0})]))
    assert np.isneginf(log_p[0][1])
    mask = np.isfinite(log_p[0])
    assert np.exp(log_p[0][mask]).sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(log_p[0][mask], np.log(0.5))


def test_seed_clustered_ci_is_seed_resampling_not_decision_resampling():
    # Two clusters with very different values: the CI must span the cluster
    # spread, which decision-level resampling would not show.
    values = np.array([0.0] * 50 + [1.0] * 50)
    seeds = np.array([1] * 50 + [2] * 50)
    low, high = seed_clustered_ci(values, seeds, reps=2000, seed=1)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)


def test_legal_action_count_matches_mask_bits():
    # bit 0 and bit 3 set -> two legal actions
    assert legal_action_count({"mask_bits": (1 << 0) | (1 << 3)}) == 2
    assert legal_action_count({"mask_bits": 0}) == 0


def test_load_panel_rejects_non_uniform_obs_bytes(tmp_path: Path):
    """A layout this code cannot index must fail loudly, not silently mis-read."""
    records = [
        {"i": 0, "explore": True, "obs_off": 0, "obs_bytes": 8, "obs_shape": [2, 1]},
        {"i": 1, "explore": True, "obs_off": 8, "obs_bytes": 16, "obs_shape": [4, 1]},
    ]
    (tmp_path / "probe_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    (tmp_path / "obs_fp32.bin").write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="obs_bytes is not uniform"):
        load_panel(tmp_path)


def test_load_panel_selects_only_explore_true_rows(tmp_path: Path):
    records = [
        {"i": 0, "explore": True, "obs_off": 0, "obs_bytes": 8, "obs_shape": [2, 1],
         "mask_bits": (1 << 0) | (1 << 1)},
        {"i": 1, "explore": False, "obs_off": 8, "obs_bytes": 8, "obs_shape": [2, 1],
         "mask_bits": 1},
        {"i": 2, "explore": True, "obs_off": 16, "obs_bytes": 8, "obs_shape": [2, 1],
         "mask_bits": 1},
    ]
    (tmp_path / "probe_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    payload = np.arange(6, dtype=np.float32).reshape(3, 2, 1)
    (tmp_path / "obs_fp32.bin").write_bytes(payload.tobytes())

    kept, panel = load_panel(tmp_path)
    assert [r["i"] for r in kept] == [0, 2]
    # Rows 0 and 2 of the raw array, i.e. the excluded row 1 is really skipped.
    assert np.array_equal(panel[0], payload[0])
    assert np.array_equal(panel[1], payload[2])
    assert panel.shape == (2, 2, 1)


# --------------------------------------------------------------------------
# Interpretation contract.  These tests exist so that a later edit cannot
# silently promote the raw action-score drift back into the headline result,
# or turn the descriptive margin split into a mechanism claim.
# --------------------------------------------------------------------------


def test_only_flip_tv_kl_may_be_used_as_the_result():
    """Raw action-score drift is telemetry: the dueling head is shift-invariant."""
    assert METRIC_ROLES["primary_result"] == ["flip_rate_argmax", "tv_mean", "kl_parent_C4"]
    assert set(METRIC_ROLES["telemetry_only"]) == {"q_delta_mean_abs", "q_delta_max_abs"}
    # The reason must stay recorded, because it is the whole justification.
    assert "mean(a_legal)" in METRIC_ROLES["rule"]
    assert "model.py:221" in METRIC_ROLES["rule"]


def test_margin_split_may_not_be_read_as_a_mechanism_claim():
    assert "descriptive" in FLIP_MARGIN_NOTE.lower()
    assert "not a per-state noise bound" in FLIP_MARGIN_NOTE
    near = INTERPRETATION_BOUNDARIES["near_ties"]
    assert "does not establish the decision importance" in near
    assert "near-ties" in near  # named as forbidden, not as a claim


def test_flip_rate_and_sampling_disagreement_are_not_to_be_divided():
    note = INTERPRETATION_BOUNDARIES["no_ratio_between_flip_and_sampling_disagreement"]
    assert "do not divide" in note
    assert "different quantities" in note.lower() or "Different quantities" in note


def test_panel_substitution_is_registered_with_its_boundaries():
    assert PANEL_SUBSTITUTION["rerun_required"] is False
    assert "P4-M4" in PANEL_SUBSTITUTION["previously_discussed"]
    boundaries = " ".join(PANEL_SUBSTITUTION["boundaries"])
    assert "NOT an independent holdout" in boundaries
    assert "NOT C4's own visitation distribution" in boundaries


def test_the_check_is_not_a_budget_rule():
    assert "does not answer" in INTERPRETATION_BOUNDARIES["not_a_budget_rule"]
