from __future__ import annotations

import numpy as np
import torch

from training.mortal.audit_k0_representation_space_2026_08 import (
    FORMAL_RUN_AUTHORIZED,
    PREREG_COMMIT,
    PREREG_FILE,
    PREREG_FILE_SHA256,
    _reservoir_row_indices,
    checkpoint_smoke,
)
from training.mortal.k0_representation_audit_core import (
    BOOTSTRAP_SEED,
    PAIR_SEED,
    PERMUTATION_SEED,
    PROJECTION_DIM,
    ROUTE_ORDER,
    bootstrap_hanchan_draws,
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


def test_checkpoint_smoke_phi_shape() -> None:
    smoke = checkpoint_smoke(torch.device("cpu"))
    assert smoke["smoke_pass"] is True
    assert smoke["phi_dim"] == 1024
    assert smoke["phi_finite_fraction"] == 1.0
