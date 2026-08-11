from __future__ import annotations

import gzip
import json

import numpy as np

from training.mortal.summarize_d3_b250_eval_2026_08 import (
    EXPECTED_MODEL_SHA,
    EXPECTED_MODELS_PER_SEED,
    EXPECTED_SEED_STARTS,
    EXPECTED_SEEDS,
    RANK_POINTS,
    bootstrap,
    comparison,
    final_scores,
    paired_row,
    promotion_decision,
    ranks_from_events,
)


def _synthetic_log(names: list[str], final_scores: list[float]) -> bytes:
    events = [
        {"type": "start_game", "names": names, "seed": [1700000, 8192]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1m",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [[], [], [], []],
        },
        {"type": "hora", "actor": 0, "target": None, "deltas": [0, 0, 0, 0]},
    ]
    # append a final kyoku carrying the resolved final scores
    events.append(
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1m",
            "kyoku": 2,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [float(value) for value in final_scores],
            "tehais": [[], [], [], []],
        }
    )
    return "\n".join(json.dumps(event) for event in events).encode("utf-8")


def test_expected_seed_mapping() -> None:
    assert EXPECTED_SEEDS == (20260806, 20260807, 20260808)
    assert EXPECTED_SEED_STARTS == {20260806: 1700000, 20260807: 1710000, 20260808: 1720000}


def test_d3_m0_labels_per_seed() -> None:
    assert EXPECTED_MODELS_PER_SEED[20260806] == {"70k", "ext_mortal", "M0_20260806", "D3_20260806"}
    assert EXPECTED_MODEL_SHA["D3_20260806"] == "a93e7a8f6b56f2c07e5e1f42c0283ff2b839e2da67af2ac9abbf382c7189defc"
    assert EXPECTED_MODEL_SHA["M0_20260806"] == "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd"


def test_rank_point_mapping() -> None:
    assert RANK_POINTS == (90.0, 45.0, 0.0, -135.0)
    # D3 rank 1, M0 rank 4 -> delta +225
    assert RANK_POINTS[0] - RANK_POINTS[3] == 225.0
    # D3 rank 2, M0 rank 1 -> delta -45
    assert RANK_POINTS[1] - RANK_POINTS[0] == -45.0


def test_ranks_from_events_and_paired_delta_sign(tmp_path) -> None:
    names = ["70k", "ext_mortal", "M0_20260806", "D3_20260806"]
    # final scores: D3 first, M0 last
    path = tmp_path / "1700000_8192_a.json.gz"
    # seats: [70k, ext_mortal, M0, D3]; give D3 (seat 3) the top score
    path.write_bytes(gzip.compress(_synthetic_log(names, [10000, 20000, 30000, 40000])))
    row = paired_row(20260806, path)
    assert row["rank_d3"] == 1
    assert row["rank_m0"] == 2
    assert row["delta_pt_d3_minus_m0"] == 90.0 - 45.0
    assert row["delta_pt_d3_minus_70k"] == 90.0 - -135.0
    assert row["d3_ahead_of_m0"] is True
    assert row["d3_ahead_of_70k"] is True
    # reversed: D3 last
    path2 = tmp_path / "1700001_8192_b.json.gz"
    path2.write_bytes(gzip.compress(_synthetic_log(names, [40000, 30000, 20000, 10000])))
    row2 = paired_row(20260806, path2)
    assert row2["rank_d3"] == 4
    assert row2["rank_m0"] == 3
    assert row2["delta_pt_d3_minus_m0"] == -135.0 - 0.0
    assert row2["d3_ahead_of_m0"] is False


def test_bootstrap_is_deterministic() -> None:
    values = {seed: np.random.default_rng(seed).normal(size=100) for seed in EXPECTED_SEEDS}
    first = bootstrap(values, reps=500, seed=20260802)
    second = bootstrap(values, reps=500, seed=20260802)
    assert first["pooled_hanchan_bootstrap_ci95"] == second["pooled_hanchan_bootstrap_ci95"]
    assert first["hierarchical_equal_seed_bootstrap_ci95"] == second["hierarchical_equal_seed_bootstrap_ci95"]


def _fake_comparison(seed_means: list[float], hier_ci: list[float]) -> dict:
    return {
        "seed_means": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, seed_means, strict=True)},
        "hierarchical_equal_seed_bootstrap_ci95": hier_ci,
    }


def test_promotion_conjunction_requires_3_of_3_direction() -> None:
    # The preregistered fixture: seed means all negative, hypothetical positive CI.
    comparisons = {
        "D3-M0": _fake_comparison([-0.855, -11.025, -2.610], [0.5, 3.0]),
        "D3-70k": _fake_comparison([2.610, -6.840, -5.310], [-8.0, 2.0]),
    }
    decision = promotion_decision(comparisons)
    assert decision["primary"]["all_three_seed_means_positive"] is False
    assert decision["primary"]["hierarchical_ci_lower_positive"] is True
    # CI positive must NOT rescue the failed direction condition
    assert decision["primary"]["passed"] is False
    assert decision["checkpoint"]["passed"] is False
    assert decision["verdict"] == "D3 data route NOT PROMOTED"


def test_promotion_conjunction_full_pass_path() -> None:
    comparisons = {
        "D3-M0": _fake_comparison([1.0, 2.0, 3.0], [0.5, 3.0]),
        "D3-70k": _fake_comparison([1.0, 2.0, 3.0], [0.2, 2.0]),
    }
    decision = promotion_decision(comparisons)
    assert decision["primary"]["passed"] is True
    assert decision["checkpoint"]["passed"] is True


def test_promotion_k1_requires_route_pass() -> None:
    comparisons = {
        "D3-M0": _fake_comparison([1.0, 2.0, 3.0], [0.5, 3.0]),
        "D3-70k": _fake_comparison([1.0, 2.0, 3.0], [-0.1, 2.0]),
    }
    decision = promotion_decision(comparisons)
    assert decision["primary"]["passed"] is True
    assert decision["checkpoint"]["d3_minus_k0_ci_lower_positive"] is False
    assert decision["checkpoint"]["passed"] is False


def test_final_scores_applies_reach_accepted_minus_1000() -> None:
    events = [
        {"type": "start_game", "names": ["70k", "ext_mortal", "M0_20260806", "D3_20260806"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "tsumo", "actor": 3, "pai": "1m"},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "9m"},
        {"type": "hora", "actor": 0, "target": 3, "deltas": [12000, 0, 0, -12000]},
    ]
    scores = final_scores(events)
    assert scores is not None
    # seat 3 (D3) paid 12000 plus the 1000 reach stick; seat 0 gained 12000
    assert scores[0] == 25000 + 12000
    assert scores[3] == 25000 - 12000 - 1000
    assert scores[1] == 25000
    assert scores[2] == 25000


def test_final_scores_without_reach_unchanged() -> None:
    events = [
        {"type": "start_game", "names": ["70k", "ext_mortal", "M0_20260806", "D3_20260806"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "actor": 0, "target": 3, "deltas": [12000, 0, 0, -12000]},
    ]
    scores = final_scores(events)
    assert scores == [37000.0, 25000.0, 25000.0, 13000.0]


def test_reach_accepted_paired_pipeline_uses_final_state(tmp_path) -> None:
    # D3 declares riichi and loses; the second start_kyoku carries the
    # post-deduction final scores, so the reconstruction must rank D3 last.
    names = ["70k", "ext_mortal", "M0_20260806", "D3_20260806"]
    events = [
        {"type": "start_game", "names": names, "seed": [1700000, 8192]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "tsumo", "actor": 3, "pai": "1m"},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "9m"},
        {"type": "hora", "actor": 0, "target": 3, "deltas": [15000, 0, 0, -15000]},
        {
            "type": "start_kyoku",
            "scores": [40000, 25000, 10000, 9000],
        },
    ]
    path = tmp_path / "1700000_8192_a.json.gz"
    path.write_bytes(gzip.compress("\n".join(json.dumps(event) for event in events).encode("utf-8")))
    row = paired_row(20260806, path)
    assert row["rank_m0"] == 3
    assert row["rank_d3"] == 4
    assert row["delta_pt_d3_minus_m0"] == -135.0
    assert row["d3_ahead_of_m0"] is False
