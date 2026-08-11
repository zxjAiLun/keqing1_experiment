from __future__ import annotations

import gzip
import json

import numpy as np

from training.mortal.repair_d1_d2_reachaccepted_summaries_2026_08 import (
    comparison,
    paired_row,
)
from training.mortal.summarize_d3_b250_eval_2026_08 import (
    EXPECTED_SEEDS,
    RANK_POINTS,
    bootstrap,
    final_scores,
    ranks_from_events,
)


def test_d1_d2_repair_reuses_fixed_reconstruction() -> None:
    events = [
        {"type": "start_game", "names": ["70k", "ext_mortal", "M0_20260806", "D1_20260806"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "tsumo", "actor": 3, "pai": "1m"},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "9m"},
        {"type": "hora", "actor": 0, "target": 3, "deltas": [12000, 0, 0, -12000]},
    ]
    scores = final_scores(events)
    assert scores[3] == 25000 - 12000 - 1000


def test_repair_paired_row_labels_and_deltas(tmp_path) -> None:
    names = ["70k", "ext_mortal", "M0_20260806", "D2_20260806"]
    events = [
        {"type": "start_game", "names": names, "seed": [1710000, 8192]},
        {"type": "start_kyoku", "scores": [10000, 20000, 30000, 40000]},
    ]
    path = tmp_path / "1710000_8192_a.json.gz"
    path.write_bytes(gzip.compress("\n".join(json.dumps(event) for event in events).encode("utf-8")))
    row = paired_row(20260806, path, "D2")
    assert row["rank_d"] == 1
    assert row["rank_m0"] == 2
    assert row["delta_pt_d_minus_m0"] == 90.0 - 45.0
    assert row["d_ahead_of_m0"] is True


def test_repair_comparison_and_bootstrap_deterministic() -> None:
    rows = [
        {
            "seed": seed,
            "delta_pt_d_minus_m0": float(value),
            "delta_pt_d_minus_70k": float(value) - 3.0,
            "delta_pt_m0_minus_70k": -3.0,
            "d_ahead_of_m0": value > 0,
            "d_ahead_of_70k": value - 3.0 > 0,
        }
        for seed, values in zip(
            EXPECTED_SEEDS,
            (np.random.default_rng(seed).normal(size=100) for seed in EXPECTED_SEEDS),
            strict=True,
        )
        for value in values
    ]
    rows_by_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in EXPECTED_SEEDS}
    first = comparison(rows_by_seed, "delta_pt_d_minus_m0", reps=200, seed=20260802)
    second = comparison(rows_by_seed, "delta_pt_d_minus_m0", reps=200, seed=20260802)
    assert first["hierarchical_equal_seed_bootstrap_ci95"] == second["hierarchical_equal_seed_bootstrap_ci95"]
    assert first["pooled_hanchan_bootstrap_ci95"] == second["pooled_hanchan_bootstrap_ci95"]


def test_repair_rank_points_are_generation_points() -> None:
    assert RANK_POINTS == (90.0, 45.0, 0.0, -135.0)


def test_repair_seed_mapping() -> None:
    assert list(EXPECTED_SEEDS) == [20260806, 20260807, 20260808]
