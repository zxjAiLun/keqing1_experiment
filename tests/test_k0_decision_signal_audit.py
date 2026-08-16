from __future__ import annotations

import numpy as np
import torch

from training.mortal.audit_k0_decision_signal_2026_08 import (
    FORMAL_RUN_AUTHORIZED,
    PREREG_COMMIT,
    PREREG_FILE,
    PREREG_FILE_SHA256,
    check_preregistration,
)
from training.mortal.k0_decision_signal_audit_core import (
    ACTION_DIM,
    CQL_LAMBDA,
    K_CREDIT,
    RFF_DIM,
    ROUTE_ORDER,
    action_credit_route_stats,
    action_credit_stats,
    adam_alignment_metrics,
    build_pooled_anchor_weights,
    centered_preference_pressure,
    combine_decision_verdict,
    compute_row_gradients,
    compute_support_neighbors,
    cosine_defined,
    g1_rff_mmd2_weighted,
    gradient_family_bootstrap_deltas,
    gradient_family_vote,
    make_frozen_bootstrap_draws,
    make_g1_rff_features,
    normalize_gradient_direction,
    precompute_g1_rff_features,
    q_gradient_signal_from_family_votes,
    rff_mmd2_from_features,
    row_rehydration_matches,
    support_bootstrap_deltas,
    support_metrics_for_anchor,
    support_signal_from_bootstrap,
    weighted_wasserstein_1d,
)
from training.mortal.k0_representation_audit_core import (
    build_global_hanchan_ids,
    sha256_array,
    sha256_file,
)


def _legal_mask(actions: int = 5) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:actions] = True
    return mask


def test_preregistration_is_frozen_and_formal_run_disabled() -> None:
    assert PREREG_COMMIT == "7bee592c7c1d00614ca1f5083032dc16b1665d36"
    assert PREREG_FILE_SHA256 == "1e27e97e6efb509eba80299f644507fe025e4e66375183155f7190a76c639a9d"
    assert sha256_file(PREREG_FILE) == PREREG_FILE_SHA256
    assert FORMAL_RUN_AUTHORIZED is False
    result = check_preregistration()
    assert result["preregistration_sha_matches"] is True


def test_analytic_gradients_match_autograd() -> None:
    q_t = torch.zeros(ACTION_DIM, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        q_t[:10] = torch.tensor([0.2, 1.5, -0.7, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    legal = np.zeros(ACTION_DIM, dtype=bool)
    legal[:10] = True
    action = 1
    target = 0.8
    lambda_ = CQL_LAMBDA

    # value loss
    value_loss = 0.5 * (q_t[action] - target) ** 2
    g_value_auto = torch.autograd.grad(value_loss, q_t, retain_graph=True)[0].detach().numpy()

    # CQL loss = lambda * (logsumexp(q_legal) - q_action)
    legal_q = q_t[legal]
    cql_loss = lambda_ * (torch.logsumexp(legal_q, dim=0) - q_t[action])
    g_cql_auto = torch.autograd.grad(cql_loss, q_t, retain_graph=True)[0].detach().numpy()

    total_loss = value_loss + cql_loss
    g_total_auto = torch.autograd.grad(total_loss, q_t)[0].detach().numpy()

    analytic = compute_row_gradients(
        q_t.detach().numpy(), legal, action, target, cql_lambda=lambda_
    )
    assert np.allclose(analytic["g_value"], g_value_auto, rtol=1e-10, atol=1e-12)
    assert np.allclose(analytic["g_cql"], g_cql_auto, rtol=1e-10, atol=1e-12)
    assert np.allclose(analytic["g_total"], g_total_auto, rtol=1e-10, atol=1e-12)
    assert np.allclose(analytic["g_cql"][legal].sum(), 0.0, atol=1e-12)
    assert np.allclose(analytic["g_value"][~legal], 0.0)
    assert np.allclose(analytic["g_cql"][~legal], 0.0)


def test_single_legal_action_zero_gradient_boundaries() -> None:
    legal = np.zeros(ACTION_DIM, dtype=bool)
    legal[3] = True
    q = np.zeros(ACTION_DIM)
    q[3] = 1.0
    grad = compute_row_gradients(q, legal, 3, 0.5)
    assert np.linalg.norm(grad["g_cql"]) == 0.0
    assert np.linalg.norm(grad["g_value"]) > 0.0
    direction, zero = normalize_gradient_direction(grad["g_total"])
    assert zero is False
    assert np.isclose(np.linalg.norm(direction), 1.0)

    # pure zero total gradient
    direction, zero = normalize_gradient_direction(np.zeros(ACTION_DIM))
    assert zero is True
    assert np.all(direction == 0.0)

    cos, defined = cosine_defined(grad["g_value"], grad["g_cql"])
    assert defined is False
    assert cos is None


def test_support_neighbors_are_deterministic_and_not_reference_resampled() -> None:
    rng = np.random.default_rng(11)
    anchor_z = rng.normal(size=(3, 8))
    anchor_z /= np.linalg.norm(anchor_z, axis=1, keepdims=True)
    ref_z = rng.normal(size=(40, 8))
    ref_z /= np.linalg.norm(ref_z, axis=1, keepdims=True)
    ref_hanchan = np.repeat(np.arange(4), 10).astype(np.int64)
    neighbors_by_route = {
        route: ref_hanchan.copy()
        for route in ROUTE_ORDER
    }
    refs = {route: ref_z.copy() for route in ROUTE_ORDER}
    first = compute_support_neighbors(anchor_z, np.zeros(3, dtype=np.int64), refs, neighbors_by_route, k=16)
    second = compute_support_neighbors(anchor_z, np.zeros(3, dtype=np.int64), refs, neighbors_by_route, k=16)
    for route in ROUTE_ORDER:
        assert np.array_equal(first[route], second[route])
        assert first[route].shape == (3, 16)
        # Every neighbor belongs to a different hanchan than the anchor (anchor hanchan id 0).
        assert (ref_hanchan[first[route]] != 0).all()


def test_support_metrics_use_mutual_legality_and_fixed_denominator() -> None:
    query_legal = _legal_mask(6)
    query_action = 0
    # 16 neighbors: 8 switch to action 1 (mutually legal), 4 switch to action 2,
    # 2 are not mutually legal, 2 are same action.
    neighbor_actions = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 9, 9], dtype=np.int64)
    neighbor_legals = np.zeros((16, ACTION_DIM), dtype=bool)
    neighbor_legals[:, :6] = True
    # Make action 9 legal in the neighbor but not in query => not mutually legal.
    neighbor_legals[14, 9] = True
    neighbor_legals[15, 9] = True
    # For neighbor action 9 to be eligible, query_action=0 must be legal in neighbor (yes)
    # and neighbor_action=9 must be legal in query (no), so not eligible.
    result = support_metrics_for_anchor(query_action, query_legal, neighbor_actions, neighbor_legals, k=16)
    assert result["switchable_rate"] == 12.0 / 16.0
    assert result["distinct_alt"] == 2.0
    assert result["eligible_count"] == 14.0
    assert result["switch_count"] == 12.0


def test_support_bootstrap_conditional_on_fixed_reference_metrics() -> None:
    # Metrics are precomputed per anchor; bootstrap only changes anchor weights.
    n_anchors = 6
    metric = {
        "M0": np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "D1": np.asarray([0.2, 0.2, 0.3, 0.3, 0.4, 0.4]),
        "D2": np.asarray([0.2, 0.2, 0.3, 0.3, 0.4, 0.4]),
        "D3": np.asarray([0.3, 0.3, 0.3, 0.4, 0.4, 0.5]),
    }
    weights = np.ones((4, n_anchors))
    result = support_bootstrap_deltas(metric, weights, reps=4)
    assert result["delta1"].shape == (4,)
    assert result["delta3"].shape == (4,)
    # With equal weights and D1==D2, view_gap=0, M0-D1 positive and M0-D3 positive.
    assert np.all(result["delta1"] > 0)
    assert np.all(result["delta3"] > 0)
    # Different anchor weights should change S values.
    weights2 = np.arange(1, n_anchors + 1, dtype=np.float64)[None, :].repeat(2, axis=0)
    result2 = support_bootstrap_deltas(metric, weights2, reps=2)
    assert not np.allclose(result2["s_m0"][0], result["s_m0"][0])


def test_weighted_wasserstein_hand_fixture_and_equal_weight_reduction() -> None:
    # Hand fixture: A mass at 0,1,1; B mass at 0,2 with weights (2,1).
    a = np.asarray([0.0, 1.0, 1.0])
    wa = np.asarray([1.0, 1.0, 1.0])
    b = np.asarray([0.0, 2.0])
    wb = np.asarray([2.0, 1.0])
    assert np.isclose(weighted_wasserstein_1d(a, wa, b, wb), 2.0 / 3.0)

    # Equal weights must equal ordinary empirical W1 for sorted scalar data.
    x = np.asarray([0.0, 0.5, 2.0, 3.0])
    y = np.asarray([0.1, 0.6, 1.0, 4.0])
    ordinary = weighted_wasserstein_1d(x, np.ones(4), y, np.ones(4))
    # Brute force ordinary W1 = mean |sorted(x)-sorted(y)| for 1D equal mass.
    assert np.isclose(ordinary, np.mean(np.abs(np.sort(x) - np.sort(y))))


def test_g1_rff_features_are_frozen_shape_and_deterministic() -> None:
    omega, bias = make_g1_rff_features(sigma=1.0)
    assert omega.shape == (RFF_DIM, ACTION_DIM)
    assert bias.shape == (RFF_DIM,)
    omega2, bias2 = make_g1_rff_features(sigma=1.0)
    assert np.array_equal(omega, omega2)
    assert np.array_equal(bias, bias2)
    a = np.zeros((3, ACTION_DIM))
    a[:, 0] = 1.0
    b = np.zeros((3, ACTION_DIM))
    b[:, 1] = 1.0
    value = g1_rff_mmd2_weighted(a, np.ones(3), b, np.ones(3), omega, bias)
    assert value >= 0.0
    assert np.isfinite(value)


def test_gradient_family_bootstrap_deltas_uses_hanchan_multiplicity() -> None:
    n_hanchans = 3
    rows_per_route = 6
    rng = np.random.default_rng(0)
    values = {route: rng.normal(size=rows_per_route) for route in ROUTE_ORDER}
    row_hanchan = {route: np.repeat(np.arange(n_hanchans), 2).astype(np.int64) for route in ROUTE_ORDER}
    draws = {
        "m0": np.asarray([[0, 0, 1]]),
        "d12": np.asarray([[0, 1, 2]]),
        "d3": np.asarray([[0, 1, 1]]),
    }
    result = gradient_family_bootstrap_deltas(
        values, row_hanchan, draws,
        distance_fn=lambda va, wa, vb, wb: weighted_wasserstein_1d(va, wa, vb, wb),
        reps=1,
    )
    assert result["d_m0_d1"].shape == (1,)
    assert result["d_m0_d3"].shape == (1,)
    assert result["d_d1_d2"].shape == (1,)
    assert result["delta1"].shape == (1,)
    assert result["delta3"].shape == (1,)
    assert gradient_family_vote(result) in (True, False)


def test_gradient_signal_vote_is_at_least_two_of_three() -> None:
    assert q_gradient_signal_from_family_votes([True, True, False]) is True
    assert q_gradient_signal_from_family_votes([True, False, False]) is False
    assert q_gradient_signal_from_family_votes([False, False, False]) is False
    assert q_gradient_signal_from_family_votes([True, True, True]) is True


def test_support_signal_requires_both_metrics_dual_delta() -> None:
    ok = {
        "delta1_ci": (0.01, 0.1),
        "delta3_ci": (0.01, 0.1),
    }
    bad = {
        "delta1_ci": (-0.01, 0.1),
        "delta3_ci": (0.01, 0.1),
    }
    assert support_signal_from_bootstrap(ok, ok) is True
    assert support_signal_from_bootstrap(ok, bad) is False
    assert support_signal_from_bootstrap(bad, ok) is False


def test_centered_preference_pressure_has_zero_common_component_sanity() -> None:
    legal = _legal_mask(4)
    g = np.zeros(ACTION_DIM)
    g[:4] = [1.0, -2.0, 3.0, -4.0]
    out = centered_preference_pressure(g, legal)
    u = legal.astype(np.float64) / np.sqrt(4)
    assert np.isclose(np.dot(out["g_centered"], u), 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(out["g_common"] + out["g_centered"]), np.linalg.norm(g))


def test_action_credit_requires_sufficient_groups_and_size() -> None:
    query_legal = _legal_mask(6)
    query_action = 0
    k = K_CREDIT
    neighbor_actions = np.asarray([1] * 30 + [2] * 30 + [3] * 3 + [4] * 1, dtype=np.int64)
    neighbor_legals = np.zeros((k, ACTION_DIM), dtype=bool)
    neighbor_legals[:, :6] = True
    neighbor_targets = np.linspace(-1.0, 1.0, k)
    result = action_credit_stats(query_action, query_legal, neighbor_actions, neighbor_legals, neighbor_targets, k=k)
    assert result["eligible"] is True
    assert result["valid_group_count"] == 2  # action 1 and 2 have >=4; action 3 group has size 3 and is invalid
    assert result["action_information_gain"] is not None

    # Too-small groups make query ineligible.
    small_actions = np.asarray([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [0] * (k - 12), dtype=np.int64)
    small = action_credit_stats(query_action, query_legal, small_actions, neighbor_legals, neighbor_targets, k=k)
    assert small["eligible"] is False


def test_row_rehydration_gate_checks_all_conditions() -> None:
    z = np.asarray([0.1, 0.2, 0.3])
    assert row_rehydration_matches(z, z, True, True, True) is True
    assert row_rehydration_matches(z, z + 1e-4, True, True, True) is False
    assert row_rehydration_matches(z, z, False, True, True) is False
    assert row_rehydration_matches(z, z, True, False, True) is False
    assert row_rehydration_matches(z, z, True, True, False) is False


def test_combine_decision_verdict_four_states() -> None:
    assert combine_decision_verdict(False, False) == "inconclusive"
    assert combine_decision_verdict(True, False) == "state_action_support_signal"
    assert combine_decision_verdict(False, True) == "objective_gradient_shift_signal"
    assert combine_decision_verdict(True, True) == "both_signals"


def test_formal_preflight_rejects_when_not_authorized(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    monkeypatch.setattr(runner, "sha256", lambda path: "0" * 64)
    result = runner.formal_preflight(torch.device("cpu"), tmp_path / "absent")
    assert result["all_pass"] is False
    assert result["formal_run_authorized"] is False
    assert result["checks"]["device_is_cuda0"] is False


def test_frozen_bootstrap_draws_and_pooled_anchor_weights() -> None:
    n = 3
    draws = make_frozen_bootstrap_draws(n_hanchans=n, reps=2)
    assert draws["m0"].shape == (2, n)
    assert draws["d12"].shape == (2, n)
    assert draws["d3"].shape == (2, n)
    # Deterministic under the frozen seed.
    draws2 = make_frozen_bootstrap_draws(n_hanchans=n, reps=2)
    for key in draws:
        assert np.array_equal(draws[key], draws2[key])

    routes = np.asarray(["M0", "D1", "D2", "D3"] * 2)
    hanchan = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    weights = build_pooled_anchor_weights(routes, hanchan, draws, reps=2)
    assert weights.shape == (2, 8)
    # M0 anchors only depend on m0 draw; D1/D2 only on d12; D3 only on d3.
    for rep in range(2):
        for idx, route in enumerate(routes):
            key = "m0" if route == "M0" else "d12" if route in ("D1", "D2") else "d3"
            han = hanchan[idx]
            expected = np.bincount(draws[key][rep], minlength=n)[han]
            assert weights[rep, idx] == expected


def test_action_credit_includes_small_groups_in_entropy_decomposition() -> None:
    query_legal = _legal_mask(6)
    query_action = 0
    k = K_CREDIT
    neighbor_actions = np.asarray([1] * 30 + [2] * 30 + [3] * 3 + [4] * 1, dtype=np.int64)
    neighbor_legals = np.zeros((k, ACTION_DIM), dtype=bool)
    neighbor_legals[:, :6] = True
    neighbor_targets = np.linspace(-1.0, 1.0, k)
    result = action_credit_stats(query_action, query_legal, neighbor_actions, neighbor_legals, neighbor_targets, k=k)
    assert result["eligible"] is True
    assert result["valid_group_count"] == 2
    assert result["eligible_count"] == k
    # Because all neighbors are eligible and we include every action group,
    # the weighted group entropies must cover all 64 rows.
    # Recompute expectation from all groups.
    groups = {1: 30, 2: 30, 3: 3, 4: 1}
    total = k

    def ent(vals: np.ndarray) -> float:
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return -sum((c / len(vals)) * np.log(c / len(vals)) for c in counts.values())

    expected = 0.0
    for action, count in groups.items():
        vals = neighbor_targets[neighbor_actions == action]
        expected += (count / total) * ent(vals)
    assert np.isclose(result["h_state_action"], expected)
    assert np.isclose(result["h_state"], ent(neighbor_targets))


def test_g1_optimized_rff_matches_naive() -> None:
    omega, bias = make_g1_rff_features(sigma=0.7)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, ACTION_DIM))
    b = rng.normal(size=(5, ACTION_DIM))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    wa = np.asarray([1.0, 2.0, 1.0, 0.5, 0.5])
    wb = np.asarray([0.5, 0.5, 1.0, 2.0, 1.0])
    naive = g1_rff_mmd2_weighted(a, wa, b, wb, omega, bias)
    fa = precompute_g1_rff_features(a, omega, bias)
    fb = precompute_g1_rff_features(b, omega, bias)
    optimized = rff_mmd2_from_features(fa, wa, fb, wb)
    assert np.isclose(naive, optimized)


def test_global_hanchan_identity_avoids_route_local_collision() -> None:
    ids, _ = build_global_hanchan_ids({"M0": ["aaa", "ccc"], "D1": ["bbb", "ccc"]})
    assert ids["M0"][0] != ids["D1"][0]  # same local sorted position, different hash
    assert ids["M0"][1] == ids["D1"][1]  # same global hash


def test_formal_preflight_includes_prereg_and_array_hashes(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    monkeypatch.setattr(runner, "sha256", lambda path: "0" * 64)
    monkeypatch.setattr(runner, "sha256_array", lambda arr: "0" * 64)
    result = runner.formal_preflight(torch.device("cpu"), tmp_path / "absent")
    assert result["checks"]["preregistration_sha"] is False
    assert result["checks"]["output_dir_absent"] is True
    assert result["all_pass"] is False


def test_action_credit_route_stats_aggregates_eligible_fraction() -> None:
    z = np.random.default_rng(0).normal(size=(18, 8)).astype(np.float64)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    hanchan = np.repeat(np.arange(6), 3).astype(np.int64)
    actions = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    masks = np.zeros((18, ACTION_DIM), dtype=bool)
    masks[:, :6] = True
    targets = np.linspace(-1.0, 1.0, 18)
    stats = action_credit_route_stats(z, actions, masks, targets, hanchan, k=8)
    assert stats["rows"] == 18
    assert 0.0 <= stats["eligible_query_fraction"] <= 1.0


def test_adam_alignment_metrics_returns_cosines() -> None:
    g = np.asarray([1.0, 0.0, 0.0])
    m = np.asarray([1.0, 0.0, 0.0])
    v = np.asarray([1.0, 0.0, 0.0])
    out = adam_alignment_metrics(g, m, v)
    assert out["cos_g_m"] == 1.0
    assert out["cos_g_m_den"] == 1.0
    out2 = adam_alignment_metrics(np.asarray([0.0, 1.0, 0.0]), m, v)
    assert out2["cos_g_m"] == 0.0


def test_formal_preflight_requires_output_dir_absent_not_merely_empty(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    monkeypatch.setattr(runner, "sha256", lambda path: "0" * 64)
    monkeypatch.setattr(runner, "sha256_array", lambda arr: "0" * 64)
    output = tmp_path / "output"
    output.mkdir()
    result = runner.formal_preflight(torch.device("cpu"), output)
    assert result["checks"]["output_dir_absent"] is False
    assert result["all_pass"] is False


def test_rng_golden_hashes_for_bootstrap_and_g1() -> None:
    draws = make_frozen_bootstrap_draws(n_hanchans=7, reps=3)
    assert sha256_array(draws["m0"]) == "c4a75931942a49e248767729fc2b82c4eb0376cba9af8954517d5c95fa573108"
    assert sha256_array(draws["d12"]) == "505f4195807dfc4411a1b8cb1d41ee071a2d41c618100f9bc9bf806038e77e57"
    assert sha256_array(draws["d3"]) == "9f93ae373de132b5e18ad2a3afc02a97c87520f58e4e55856c4324906f23599c"
    omega, bias = make_g1_rff_features(sigma=0.7)
    assert sha256_array(omega) == "71e9746a49b68d8db1c17d8aaca1836934e4330b54e60e7c49478d22bc168e6d"
    assert sha256_array(bias) == "02bbda40af71bba17181728941623c1cc5a13b813e1c67cb09f6dc3bbed5ae65"
