from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from training.mortal.audit_k0_representation_space_2026_08 import (
    FORMAL_RUN_AUTHORIZED,
    FROZEN_INPUT_SHA256,
    PREREG_COMMIT,
    PREREG_FILE,
    PREREG_FILE_SHA256,
    _exposure_proxy_probabilities,
    _metric_bootstrap_deltas,
    _reservoir_row_indices,
    _weighted_proxy_sample,
    checkpoint_smoke,
)
from training.mortal.k0_representation_audit_core import (
    BOOTSTRAP_SEED,
    PAIR_SEED,
    PERMUTATION_SEED,
    PROJECTION_DIM,
    ROUTE_ORDER,
    bootstrap_hanchan_draws,
    build_global_hanchan_ids,
    combine_verdict,
    credit_ambiguity_vote,
    estimate_sigma_from_pairs,
    knn_neighbor_indices_blockwise,
    knn_row_stats_and_indicators,
    make_projection_matrix,
    make_rff_features,
    make_sw_directions,
    permutation_draws,
    rff_mmd2_weighted,
    sample_cross_hanchan_pairs,
    sha256_array,
    sliced_wasserstein_weighted,
    weighted_quantile_values,
)


def _normalized(seed: int, n_rows: int = 40, dim: int = PROJECTION_DIM) -> np.ndarray:
    z = np.random.default_rng(seed).normal(size=(n_rows, dim))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def test_projection_and_sw_directions_golden_hashes() -> None:
    projection = make_projection_matrix().numpy()
    directions = make_sw_directions().numpy()
    assert projection.shape == (256, 1024)
    assert directions.shape == (256, 256)
    assert np.allclose(np.linalg.norm(projection, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert sha256_array(projection) == "87cb9ce48397652da8f326b1cbbe31656576ed13d83ee4ca5d41af302bfedd21"
    assert sha256_array(directions) == "87c78f2c68880c49966f89bde3edc4a0f5dbcb50d944b8c6982aa175fc4e8a30"


def test_weighted_quantile_values_are_frozen() -> None:
    values = np.asarray([0.0, 1.0, 2.0])
    weights = np.ones(3)
    assert np.allclose(
        weighted_quantile_values(values, weights, n_quantiles=5),
        np.asarray([0.0, 0.0, 0.5, 1.1, 1.7]),
    )
    assert np.allclose(
        weighted_quantile_values(np.asarray([0.0, 1.0]), np.asarray([1.0, 3.0]), n_quantiles=4),
        np.asarray([0.0, 0.16666666666666666, 0.5, 0.8333333333333334]),
    )


def test_sliced_wasserstein_golden_value() -> None:
    directions = make_sw_directions().numpy()
    z_a = _normalized(42)
    z_b = _normalized(43)
    value = sliced_wasserstein_weighted(z_a, np.ones(40), z_b, np.ones(40), directions)
    assert value == 0.016029842429886587


def test_sigma_pair_sampling_is_cross_hanchan_unique_and_deterministic() -> None:
    hanchan_ids = np.repeat(np.arange(10), 4)
    pairs = sample_cross_hanchan_pairs(hanchan_ids, n_pairs=20, seed=PAIR_SEED, route_index=0)
    assert pairs.shape == (20, 2)
    assert (hanchan_ids[pairs[:, 0]] != hanchan_ids[pairs[:, 1]]).all()
    keys = {tuple(sorted(row)) for row in pairs.tolist()}
    assert len(keys) == 20
    assert sha256_array(pairs) == "b0fb3af823a8981a44761f0a97b0e55b340ec4a3d83c8c44a8744706f88a54b1"
    z = {route: _normalized(44) for route in ROUTE_ORDER}
    han = {route: hanchan_ids for route in ROUTE_ORDER}
    assert estimate_sigma_from_pairs(z, han, pairs_per_route=10) == 1.4152793953473308


def test_rff_features_and_mmd_are_deterministic() -> None:
    omega, bias = make_rff_features(0.7)
    assert omega.shape == (2048, 256)
    assert bias.shape == (2048,)
    assert sha256_array(omega.numpy()) == "737e07ef0b6a3f6e2c88a4ff9849a901bff3871087ded982daa11b3aceb3104b"
    assert sha256_array(bias.numpy()) == "c4ef172992fcf080db17237dfa4472a0ed9121318c272bbe5dda0d5c2ef8943d"
    z_a = _normalized(45, n_rows=8)
    z_b = _normalized(46, n_rows=8)
    first = rff_mmd2_weighted(z_a, np.ones(8), z_b, np.ones(8), omega.numpy(), bias.numpy())
    second = rff_mmd2_weighted(z_a, np.ones(8), z_b, np.ones(8), omega.numpy(), bias.numpy())
    assert first == second
    assert first > 0


def test_knn_excludes_self_and_same_hanchan() -> None:
    z = np.zeros((9, 256))
    z[0] = 1.0
    z[3] = 1.0
    z[6] = 2.0
    z[7] = 2.0
    z[8] = 2.0
    for index in range(9):
        norm = np.linalg.norm(z[index])
        z[index] = z[index] / norm if norm else z[index]
    hanchan = np.repeat(np.arange(3), 3)
    neighbors = knn_neighbor_indices_blockwise(z, z, hanchan, hanchan, k=2)
    assert neighbors.shape == (9, 2)
    for row_index, row in enumerate(neighbors):
        assert hanchan[row_index] not in set(hanchan[row].tolist())
    assert neighbors[0].tolist() == [3, 6]


def test_missing_indicator_uses_reference_self_geometry() -> None:
    z_a = _normalized(47, n_rows=12)
    z_b = _normalized(48, n_rows=12)
    hanchan_a = np.repeat(np.arange(4), 3)
    hanchan_b = np.repeat(np.arange(4), 3)
    stats = knn_row_stats_and_indicators(z_a, z_b, hanchan_a, hanchan_b, k=3)
    assert stats["a_missing_in_b"].shape == (12,)
    assert stats["b_missing_in_a"].shape == (12,)
    assert set(np.unique(stats["a_missing_in_b"])) <= {0, 1}
    assert set(np.unique(stats["b_missing_in_a"])) <= {0, 1}


def test_bootstrap_draws_are_frozen_and_d12_is_shared() -> None:
    draws = bootstrap_hanchan_draws(n_hanchans=7, reps=3, seed=BOOTSTRAP_SEED)
    assert draws["m0"].shape == (3, 7)
    assert draws["d12"].shape == (3, 7)
    assert draws["d3"].shape == (3, 7)
    assert sha256_array(draws["m0"]) == "62adcad902b959c53a05acf318dad692f8e2baf208f5fbc1ce88c80e342db1c0"
    assert sha256_array(draws["d12"]) == "bd028f0cd7cf94248ecc50377d89c6b6481aa96e9a7a3b65023a620b443c3a74"
    assert sha256_array(draws["d3"]) == "8721ef90f0f8641ba2c6ababcc5eed4f1548211b7e5c40e1d6b213f2b964dba0"


def test_permutation_draws_are_frozen_and_d12_is_shared() -> None:
    draws = permutation_draws(n_hanchans=7, reps=3, seed=PERMUTATION_SEED)
    assert draws["m0"].shape == (3, 7)
    assert draws["d12"].shape == (3, 7)
    assert draws["d3"].shape == (3, 7)
    for key in draws:
        assert np.array_equal(np.sort(draws[key], axis=1), np.tile(np.arange(7), (3, 1)))
    assert sha256_array(draws["m0"]) == "c7369b1fe4b4967af690352d6314ce6ccd8e441781b3582144cee255afed93e5"
    assert sha256_array(draws["d12"]) == "7e6fbc53fb243b270e13bdb50e1c5e9339f52fc278dae47f1c23c0f5114c6164"
    assert sha256_array(draws["d3"]) == "13d04c91660f8f712fb22c8aba45aac0a5df053586f56c70c637688246bf8442"


def test_credit_direction_is_not_inverted() -> None:
    null = np.asarray([0.10, 0.20, 0.30, 0.40])
    predictive = credit_ambiguity_vote(0.50, null)
    assert predictive["p_predict"] == 0.2
    assert predictive["vote"] is True
    ambiguous = credit_ambiguity_vote(0.01, null)
    assert ambiguous["p_predict"] == 1.0
    assert ambiguous["vote"] is True
    not_detected = credit_ambiguity_vote(0.45, np.asarray([0.10, 0.20, 0.30]))
    assert not_detected["vote"] is True


def test_combine_verdict_four_states_only() -> None:
    assert combine_verdict(False, False) == "inconclusive"
    assert combine_verdict(True, False) == "latent_coverage_signal"
    assert combine_verdict(False, True) == "latent_credit_ambiguity_signal"
    assert combine_verdict(True, True) == "both_signals"


def test_reservoir_row_indices_are_midpoints() -> None:
    assert _reservoir_row_indices(5).tolist() == [0, 2, 4]
    assert _reservoir_row_indices(10).tolist() == [1, 5, 8]


def test_preregistration_sha_is_frozen() -> None:
    from training.mortal.k0_representation_audit_core import sha256_file

    assert PREREG_COMMIT == "729741ad585eea93e8a2fa02d04020a59ae95716"
    assert PREREG_FILE_SHA256 == "3ebd88b5afc8e7fda28c1c5e61aff5cdf6babf90d082a9096a1529545088b359"
    assert sha256_file(PREREG_FILE) == PREREG_FILE_SHA256
    assert FORMAL_RUN_AUTHORIZED is False


def test_global_hanchan_identity_separates_sorted_position_from_canonical_hash() -> None:
    route_ids, _hash_to_id = build_global_hanchan_ids({"A": ["aaa", "ccc"], "B": ["bbb", "ccc"]})
    assert route_ids["A"][0] != route_ids["B"][0]  # same sorted position, different hash
    assert route_ids["A"][1] == route_ids["B"][1]  # same hash, different sorted position


def test_credit_predictive_case_p001_vote_false() -> None:
    null = np.linspace(0.0, 0.01, 999)
    result = credit_ambiguity_vote(0.02, null)
    assert result["p_predict"] == 1.0 / 1000.0
    assert result["vote"] is False


def test_exposure_proxy_probabilities_sum_to_one() -> None:
    file_index_by_row = np.asarray([0, 0, 0, 1, 1, 1])
    consumed = np.asarray([2.0, 3.0])
    probabilities = _exposure_proxy_probabilities(file_index_by_row, consumed)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(probabilities[0], 2.0 / 15.0)
    sample = _weighted_proxy_sample(file_index_by_row, consumed, n_samples=2000, seed=123, route_index=0)
    assert sample.shape == (2000,)
    assert set(sample.tolist()) <= {0, 1, 2, 3, 4, 5}


def test_d3_event_mapping_uses_arena_to_row(monkeypatch) -> None:
    import training.mortal.audit_k0_representation_space_2026_08 as runner
    import training.mortal.d3_native_scene as native_scene
    import training.mortal.d3_production_audit_core as audit_core

    monkeypatch.setattr(audit_core, "primary_row_flags", lambda actions: [True] * len(list(actions)))
    monkeypatch.setattr(
        native_scene,
        "reconstruct_native_scenes",
        lambda *args, **kwargs: {
            "scenes": [
                {"kyoku": 0, "arena_index": 2, "loader_row_index": 1, "arena_consulted": True},
            ]
        },
    )
    actions = np.asarray([0, 0, 0], dtype=np.int64)
    legal_counts = np.asarray([5, 5, 5], dtype=np.int64)
    at_kyoku = np.asarray([0, 0, 0], dtype=np.int64)
    events = [{"kyoku_index": 0, "decision_index": 2, "explored": True}]
    mapped = runner._d3_event_category_by_loader_row(__import__("pathlib").Path("virtual.json.gz"), 0, actions, legal_counts, at_kyoku, events)
    assert mapped == {1: "explored"}


def test_d3_mapping_categories_roundtrip(monkeypatch) -> None:
    import training.mortal.audit_k0_representation_space_2026_08 as runner
    import training.mortal.d3_native_scene as native_scene
    import training.mortal.d3_production_audit_core as audit_core

    monkeypatch.setattr(audit_core, "primary_row_flags", lambda actions: [True] * len(list(actions)))
    monkeypatch.setattr(
        native_scene,
        "reconstruct_native_scenes",
        lambda *args, **kwargs: {
            "scenes": [
                {"kyoku": 0, "arena_index": 0, "loader_row_index": 0, "arena_consulted": True},
                {"kyoku": 0, "arena_index": 1, "loader_row_index": 1, "arena_consulted": True},
                {"kyoku": 0, "arena_index": 2, "loader_row_index": 2, "arena_consulted": True},
            ]
        },
    )
    actions = np.asarray([0, 0, 0], dtype=np.int64)
    legal_counts = np.asarray([5, 5, 5], dtype=np.int64)
    at_kyoku = np.asarray([0, 0, 0], dtype=np.int64)
    events = [
        {"kyoku_index": 0, "decision_index": 0, "explored": True},
        {"kyoku_index": 0, "decision_index": 1, "reason": "hash_rejected"},
        {"kyoku_index": 0, "decision_index": 2, "reason": "budget"},
    ]
    mapped = runner._d3_event_category_by_loader_row(Path("virtual.json.gz"), 0, actions, legal_counts, at_kyoku, events)
    assert mapped == {0: "explored", 1: "hash_rejected", 2: "budget_exhausted"}


def test_perspective_label_roundtrip_metadata(tmp_path) -> None:
    import training.mortal.audit_k0_representation_space_2026_08 as runner

    route_dir = tmp_path / "route_artifacts" / "D2"
    route_dir.mkdir(parents=True)
    z = np.eye(2, PROJECTION_DIM, dtype=np.float32)
    np.save(route_dir / "event_z.npy", z)
    np.savez(
        route_dir / "event_metadata.npz",
        file_index=np.asarray([0, 1], dtype=np.int64),
        row_index=np.asarray([0, 1], dtype=np.int64),
        hanchan_index=np.asarray([0, 1], dtype=np.int32),
        target=np.asarray([-1.0, 1.0]),
        phi_l2_norm=np.asarray([1.0, 1.0]),
    )
    (route_dir / "event_perspective_labels.json").write_text(json.dumps(["V2_74000", "V3_74000"]), encoding="utf-8")
    (route_dir / "event_canonical_hanchan_hashes.json").write_text(json.dumps(["hash-a", "hash-b"]), encoding="utf-8")
    loaded = runner._load_extra_route_rows(tmp_path, "D2", "event")
    assert loaded["rows"] == 2
    assert loaded["perspective_labels"] == ["V2_74000", "V3_74000"]
    assert loaded["canonical_hanchan_hashes"] == ["hash-a", "hash-b"]


def test_optimized_sw_rff_matches_naive_on_small_fixture() -> None:
    rng = np.random.default_rng(123)
    z_m0 = rng.normal(size=(9, PROJECTION_DIM))
    z_m0 /= np.linalg.norm(z_m0, axis=1, keepdims=True)
    z_d1 = rng.normal(size=(9, PROJECTION_DIM))
    z_d1 /= np.linalg.norm(z_d1, axis=1, keepdims=True)
    z_d2 = rng.normal(size=(9, PROJECTION_DIM))
    z_d2 /= np.linalg.norm(z_d2, axis=1, keepdims=True)
    z_d3 = rng.normal(size=(9, PROJECTION_DIM))
    z_d3 /= np.linalg.norm(z_d3, axis=1, keepdims=True)
    routes = {
        name: {"z": z, "hanchan_index": np.repeat(np.arange(3), 3).astype(np.int32)}
        for name, z in (("M0", z_m0), ("D1", z_d1), ("D2", z_d2), ("D3", z_d3))
    }
    directions = np.random.default_rng(5).normal(size=(2, PROJECTION_DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    omega = np.random.default_rng(6).normal(size=(4, PROJECTION_DIM))
    bias = np.random.default_rng(7).uniform(0, 2 * np.pi, size=4)
    knn_stats = {
        (left, right): knn_row_stats_and_indicators(
            routes[left]["z"], routes[right]["z"], routes[left]["hanchan_index"], routes[right]["hanchan_index"], k=2
        )
        for left, right in (("M0", "D1"), ("M0", "D3"), ("D1", "D2"))
    }
    draws = {"m0": np.asarray([[0, 1, 2]]), "d12": np.asarray([[0, 1, 2]]), "d3": np.asarray([[0, 1, 2]])}
    result = _metric_bootstrap_deltas(routes, directions, omega, bias, knn_stats, draws, reps=1)
    expected_sw = sliced_wasserstein_weighted(z_m0, np.ones(9), z_d1, np.ones(9), directions)
    expected_mmd = rff_mmd2_weighted(z_m0, np.ones(9), z_d1, np.ones(9), omega, bias)
    assert np.isclose(result["families"]["sliced_wasserstein"]["point"]["d_m0_d1"], expected_sw, rtol=1e-12)
    assert np.isclose(result["families"]["rbf_mmd"]["point"]["d_m0_d1"], expected_mmd, rtol=1e-12)


def test_formal_preflight_sha_mismatch_fails_closed(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_representation_space_2026_08 as runner

    original_sha = runner.sha256
    monkeypatch.setattr(runner, "sha256", lambda path: "0" * 64)
    preflight = runner.formal_preflight(torch.device("cpu"), tmp_path / "absent")
    assert preflight["all_pass"] is False
    assert preflight["checks"]["k0_checkpoint"] is False
    assert preflight["checks"]["device_is_cuda0"] is False
    monkeypatch.setattr(runner, "sha256", original_sha)


def test_formal_preflight_rejects_dirty_worktree_and_nonempty_output(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_representation_space_2026_08 as runner

    monkeypatch.setattr(
        runner,
        "git_worktree_metadata",
        lambda: {"git_worktree_clean": False, "git_worktree_status": ["M dirty.py"]},
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("x", encoding="utf-8")
    preflight = runner.formal_preflight(torch.device("cuda"), output)
    assert preflight["checks"]["git_worktree_clean"] is False
    assert preflight["checks"]["output_dir_absent_or_empty"] is False
    assert preflight["all_pass"] is False


def test_frozen_input_gate_a_covers_all_prereg_artifacts() -> None:
    expected_keys = {
        "k0_checkpoint",
        "m0_index",
        "d1_index",
        "d2_index",
        "d3_index",
        "d2_mapping",
        "v2_report",
        "v2_route_cache_manifest",
        "v2_training_exposure",
        "riichi_extension",
    }
    assert set(FROZEN_INPUT_SHA256) == expected_keys
    assert all(len(value) == 2 for value in FROZEN_INPUT_SHA256.values())


def test_checkpoint_smoke_phi_shape() -> None:
    smoke = checkpoint_smoke(torch.device("cpu"))
    assert smoke["smoke_pass"] is True
    assert smoke["phi_dim"] == 1024
    assert smoke["phi_finite_fraction"] == 1.0
