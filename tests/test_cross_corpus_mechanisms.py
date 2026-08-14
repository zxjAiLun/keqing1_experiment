from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.mortal.audit_cross_corpus_mechanisms_2026_08 import (
    CONSUMED_SAMPLES,
    action_name,
    bucket_legal,
    bucket_score_gap,
    bucket_shanten,
    decide_readout,
    exposure_weighted_distribution,
    gate_f_checks,
    gini,
    jsd,
    phase_bucket,
    simulate_exposure,
    strata_key,
    summarize_d3_diag,
    tv_distance,
    weighted_missing_mass,
)


def test_bucket_functions() -> None:
    assert bucket_score_gap(15000) == "ahead_big"
    assert bucket_score_gap(5000) == "ahead"
    assert bucket_score_gap(-5000) == "behind"
    assert bucket_score_gap(-15000) == "behind_big"
    assert bucket_legal(3) == "1_5"
    assert bucket_legal(8) == "6_10"
    assert bucket_legal(14) == "11_plus"
    assert bucket_shanten(-1) == "tenpai"
    assert bucket_shanten(1) == "1"
    assert bucket_shanten(2) == "2"
    assert bucket_shanten(5) == "3_plus"


def test_phase_bucket_boundaries() -> None:
    # trusted audit_replay_distribution.phase_bucket: <4 early, <8 middle, else late
    for kyoku in (0, 3):
        assert phase_bucket(kyoku) == "early"
    for kyoku in (4, 7):
        assert phase_bucket(kyoku) == "middle"
    for kyoku in (8, 12, 20):
        assert phase_bucket(kyoku) == "late"


def test_strata_key_uses_phase_bucket() -> None:
    key = strata_key(2, 1, 5000.0, False, 8, 1)
    assert key == ("early", "1", "ahead", "False", "6_10", "1")
    assert key == strata_key(2, 1, 5000.0, False, 8, 1)
    assert strata_key(4, 1, 5000.0, False, 8, 1)[0] == "middle"
    assert strata_key(9, 1, 5000.0, False, 8, 1)[0] == "late"


def test_action_name_family_mapping() -> None:
    # trusted audit_replay_distribution.action_name semantics: all 37 discards
    # collapse to "discard"; the rest map to their action kinds exactly.
    for action in (0, 17, 36):
        assert action_name(action) == "discard"
    expected = {
        37: "reach",
        38: "chi_low",
        39: "chi_mid",
        40: "chi_high",
        41: "pon",
        42: "kan",
        43: "agari",
        44: "ryukyoku",
        45: "pass",
    }
    for action, kind in expected.items():
        assert action_name(action) == kind


def test_jsd_and_tv_identical_zero() -> None:
    p = np.asarray([0.5, 0.5])
    assert jsd(p, p) == 0.0
    assert tv_distance(p, p) == 0.0
    q = np.asarray([0.25, 0.75])
    assert jsd(p, q) > 0.0
    assert tv_distance(p, q) == 0.25


def test_weighted_missing_mass() -> None:
    source = Counter({"a": 3, "b": 2, "c": 5})
    target = Counter({"a": 1, "c": 4})
    assert weighted_missing_mass(source, target) == 0.2
    assert weighted_missing_mass(source, source) == 0.0


def test_gini_uniform_is_zero() -> None:
    assert gini(np.full(100, 10.0)) == 0.0
    assert gini(np.asarray([0.0, 0.0, 0.0])) == 0.0
    skewed = np.asarray([90.0, 5.0, 3.0, 2.0])
    assert 0 < gini(skewed) < 1


def test_simulate_exposure_deterministic_and_bounded() -> None:
    # simulate_exposure only uses path strings, so no real files are needed
    files = [Path(f"virtual/g{index}.json.gz") for index in range(6)]
    counters = {index: Counter({"rows": 60_000 + index * 100}) for index in range(6)}
    first = simulate_exposure(files, seed=20260806, file_batch_size=3, num_epochs=3, per_file_counters=counters)
    second = simulate_exposure(files, seed=20260806, file_batch_size=3, num_epochs=3, per_file_counters=counters)
    assert first == second
    assert first["samples_consumed"] == CONSUMED_SAMPLES
    assert 1 <= first["unique_hanchans_exposed"] <= 6
    assert 0.0 <= first["repeat_rate"] < 1.0
    assert first["effective_hanchan_n"] <= 6
    different_seed = simulate_exposure(files, seed=20260807, file_batch_size=3, num_epochs=3, per_file_counters=counters)
    assert different_seed != first or different_seed["exposure_gini"] != first["exposure_gini"]


def test_simulate_exposure_respects_row_counts() -> None:
    files = [Path("virtual/a.json.gz"), Path("virtual/b.json.gz")]
    counters = {0: Counter({"rows": 1}), 1: Counter({"rows": 1})}
    result = simulate_exposure(files, seed=20260806, file_batch_size=1, num_epochs=1, per_file_counters=counters)
    # each epoch: shuffle [0,1], chunk size 1 -> two chunks; with 2 rows per
    # epoch and 1 epoch, only 2 samples are available
    assert result["samples_consumed"] == 2
    assert result["unique_hanchans_exposed"] == 2


def test_simulate_exposure_consumed_per_file_sums() -> None:
    files = [Path(f"virtual/f{index}.json.gz") for index in range(6)]
    counters = {index: Counter({"rows": 1000 + index}) for index in range(6)}
    result = simulate_exposure(files, seed=20260806, file_batch_size=2, num_epochs=2, per_file_counters=counters)
    assert len(result["consumed_per_file"]) == 6
    assert sum(result["consumed_per_file"]) == result["samples_consumed"]


def test_exposure_weighted_distribution_scales_by_weights() -> None:
    weights = np.asarray([2.0, 0.0, 1.0])
    per_file = [Counter({"a": 1}), Counter({"b": 1}), Counter({"a": 3})]
    result = exposure_weighted_distribution(weights, per_file)
    assert result["a"] == 5.0
    assert result["b"] == 0


def test_decide_readout_a_pass_fixture() -> None:
    exclusive = {"D1": 0.05, "D2": 0.0, "D3": 0.01}
    assert decide_readout(coverage_votes=3, num_families=4, m0_exclusive_mass=exclusive, gates_ok=True) == "A_coverage_priority"


def test_decide_readout_a_fail_is_inconclusive() -> None:
    assert decide_readout(coverage_votes=0, num_families=4, m0_exclusive_mass={"D1": 0.05}, gates_ok=True) == "inconclusive"
    # exclusive mass alone does not promote
    assert decide_readout(coverage_votes=1, num_families=4, m0_exclusive_mass={"D1": 0.05}, gates_ok=True) == "inconclusive"


def test_decide_readout_never_promotes_b() -> None:
    # no quantitative B promotion threshold was preregistered, so the machine
    # rule must never output B_credit_assignment_priority -- even for metrics
    # that would have satisfied the previously hard-coded 0.8/0.3 heuristic.
    outputs: set[str] = set()
    for votes in range(5):
        outputs.add(decide_readout(coverage_votes=votes, num_families=4, m0_exclusive_mass={"D1": 0.1}, gates_ok=True))
        outputs.add(decide_readout(coverage_votes=votes, num_families=4, m0_exclusive_mass={"D1": 0.0}, gates_ok=True))
    assert outputs == {"A_coverage_priority", "inconclusive"}
    assert "B_credit_assignment_priority" not in outputs


def test_decide_readout_any_gate_false_blocks_verdict() -> None:
    # even a unanimous A-support pattern yields no verdict while any A-F gate is false
    assert decide_readout(coverage_votes=4, num_families=4, m0_exclusive_mass={"D1": 0.5}, gates_ok=False) == "no_verdict_gates_failed"
    assert decide_readout(coverage_votes=0, num_families=4, m0_exclusive_mass={"D1": 0.0}, gates_ok=False) == "no_verdict_gates_failed"


def _synthetic_d3_diag() -> dict:
    """One explored event mapped to one hanchan, written as 9 histogram cells."""
    diag: dict = {
        "category_hanchan": defaultdict(lambda: defaultdict(Counter)),
        "category_total": Counter(),
        "category_mapped_events": Counter(),
        "category_unconsumed": Counter(),
        "category_hanchan_events": defaultdict(Counter),
    }
    diag["category_total"]["explored"] += 1
    diag["category_mapped_events"]["explored"] += 1
    diag["category_hanchan_events"]["explored"][7] += 1
    for field in ("target", "final_rank", "behavior_q", "q_regret", "margin", "phase", "rank", "score_gap", "shanten"):
        diag["category_hanchan"]["explored"][7][f"{field}:sample"] += 1
    return diag


def test_summarize_d3_diag_counts_events_not_histogram_cells() -> None:
    summary = summarize_d3_diag(_synthetic_d3_diag())
    assert summary["total_events"] == 1
    assert summary["total_mapped_events"] == 1
    assert summary["total_unconsumed_events"] == 0
    # 1 event + 9 histogram fields: the mapped EVENT count must still be 1
    assert summary["explored"]["events_mapped_to_rows"] == 1
    assert summary["explored"]["hanchans_with_events"] == 1
    assert summary["explored"]["per_hanchan_event_count"] == {"min": 1, "median": 1.0, "max": 1}
    assert len(summary["explored"]["histograms"]) == 9


def test_gate_f_checks_pass_exactly_once() -> None:
    checks = gate_f_checks(summarize_d3_diag(_synthetic_d3_diag()))
    assert checks == {
        "diagnostic_present": True,
        "all_events_mapped_exactly_once": True,
        "no_unconsumed_events": True,
        "category_counts_exact": True,
    }


def test_gate_f_checks_reject_histogram_cell_counts() -> None:
    # the old bug: histogram-cell increments (9) reported as mapped events (1)
    summary = summarize_d3_diag(_synthetic_d3_diag())
    inflated = dict(summary)
    inflated["mapped_event_totals"] = {"explored": 9}
    inflated["total_mapped_events"] = 9
    checks = gate_f_checks(inflated)
    assert not checks["all_events_mapped_exactly_once"]
    assert not checks["category_counts_exact"]


def test_gate_f_checks_reject_unconsumed() -> None:
    diag = _synthetic_d3_diag()
    diag["category_total"]["hash_rejected"] += 1
    diag["category_unconsumed"]["hash_rejected"] += 1
    checks = gate_f_checks(summarize_d3_diag(diag))
    assert not checks["all_events_mapped_exactly_once"]
    assert not checks["no_unconsumed_events"]
    assert not checks["category_counts_exact"]


def test_gate_f_checks_reject_empty_diagnostic() -> None:
    diag: dict = {
        "category_hanchan": defaultdict(lambda: defaultdict(Counter)),
        "category_total": Counter(),
        "category_mapped_events": Counter(),
        "category_unconsumed": Counter(),
        "category_hanchan_events": defaultdict(Counter),
    }
    checks = gate_f_checks(summarize_d3_diag(diag))
    assert not any(checks.values())
