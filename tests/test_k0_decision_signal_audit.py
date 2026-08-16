from __future__ import annotations

import numpy as np
import torch

from training.mortal.audit_k0_decision_signal_2026_08 import (
    FORMAL_RUN_AUTHORIZED,
    PREREG_COMMIT,
    PREREG_FILE,
    PREREG_FILE_SHA256,
    _adam_diagnostic_for_route,
    _adam_diagnostic_from_optimizer,
    _build_preserved_optimizer,
    _compute_support_metrics,
    _flatten_group_gradient,
    _json_safe_report,
    _rehydrate_canonical_rows,
    _run_decision_analysis,
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
    alternative_action_from_q,
    build_pooled_anchor_weights,
    centered_preference_pressure,
    combine_decision_verdict,
    compute_row_gradients,
    compute_support_neighbors,
    cosine_defined,
    g1_rff_mmd2_weighted,
    gradient_family_bootstrap_deltas,
    gradient_family_vote,
    greedy_action_from_q,
    make_frozen_bootstrap_draws,
    make_g1_rff_features,
    normalize_gradient_direction,
    precompute_g1_rff_features,
    q_gradient_signal_from_family_votes,
    rff_mmd2_from_features,
    row_rehydration_matches,
    sample_microbatch_rows,
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


def test_formal_preflight_rejects_on_sha_mismatch(monkeypatch, tmp_path) -> None:
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


def test_support_metrics_use_anchor_query_not_reference_query() -> None:
    # 4 anchors, each from M0/D1/D2/D3; reference D1 has only 2 rows here but
    # we test the query semantics directly with k=2.
    anchor_data = {
        "routes": np.asarray(["M0", "D1", "D2", "D3"]),
    }
    anchor_actions = np.asarray([0, 1, 2, 3], dtype=np.int64)
    anchor_masks = np.zeros((4, ACTION_DIM), dtype=bool)
    anchor_masks[:, :6] = True
    # Reference D1 has 2 rows.  Neighbor indices map both anchors to [0,1].
    neighbor_indices = {"D1": np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]])}
    actions_by_route = {"D1": np.asarray([4, 5], dtype=np.int64)}
    masks_by_route = {"D1": np.zeros((2, ACTION_DIM), dtype=bool)}
    masks_by_route["D1"][:, :6] = True
    out = _compute_support_metrics(
        anchor_data,
        anchor_actions,
        anchor_masks,
        neighbor_indices,
        actions_by_route,
        masks_by_route,
    )
    # Anchor 0 is M0 with query action 0; D1 neighbors are actions 4,5.
    # Both are legal in query (query_legal[:6]) and 0 is legal in D1,
    # so both are eligible and switchable => 2/16.
    assert out["D1"]["s1"][0] == 2.0 / 16.0
    # Anchor 1 is D1 with query action 1; same neighbors => also 2/16.
    assert out["D1"]["s1"][1] == 2.0 / 16.0


def test_json_safe_report_converts_ndarray_to_artifact(tmp_path) -> None:
    analysis = {
        "a": np.asarray([1.0, 2.0]),
        "nested": {"b": np.asarray([[1, 2]], dtype=np.int64), "c": 3},
        "list": [np.asarray([True, False])],
    }
    safe = _json_safe_report(analysis, tmp_path)
    import json
    text = json.dumps(safe, ensure_ascii=False, indent=2)
    assert "ndarray" not in text
    assert (tmp_path / "bootstrap" / "a.npy").is_file()
    assert (tmp_path / "bootstrap" / "nested_b.npy").is_file()
    assert (tmp_path / "bootstrap" / "list_0.npy").is_file()


def test_adam_diagnostic_uses_live_parameter_objects() -> None:
    from torch import nn, optim

    model = nn.Linear(2, 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    # Initialize optimizer state so state[param] has exp_avg/exp_avg_sq.
    dummy = torch.zeros(2, requires_grad=True)
    loss = model(dummy).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    grads = {param: torch.ones_like(param) for param in model.parameters()}
    result = _adam_diagnostic_from_optimizer(optimizer, grads)
    assert "group_0" in result
    assert result["group_0"]["param_count"] == 2
    assert result["group_0"]["cos_g_m_mean"] is not None


def test_g1_pair_and_microbatch_golden_hashes() -> None:
    from training.mortal.k0_representation_audit_core import sample_cross_hanchan_pairs

    hanchan = np.repeat(np.arange(10), 3)
    pairs = sample_cross_hanchan_pairs(hanchan, n_pairs=20, seed=20260822, route_index=0)
    assert sha256_array(pairs) == "03e2d2c0fdbca832df5b1121660beef6df82ee247a1230186f1297288f3b465a"

    row_hanchan = np.repeat(np.arange(10), 3)
    samples = sample_microbatch_rows(row_hanchan, batch_size=5, n_batches=3, seed=20260824)
    assert sha256_array(samples) == "ba5a15d5fbcff7bb440c3eb1717a2faf8f8f0ed64badaf0ece34e46ce26ba540"


def test_greedy_and_alt_are_selected_from_q_not_gradient() -> None:
    q = np.zeros(ACTION_DIM)
    q[:5] = [0.1, 0.9, 0.2, 0.8, 0.0]
    legal = np.zeros(ACTION_DIM, dtype=bool)
    legal[:5] = True
    # Gradient has largest at index 0, but Q has largest at index 1.
    g_total = np.zeros(ACTION_DIM)
    g_total[0] = 10.0
    g_total[1] = 0.0
    assert greedy_action_from_q(q, legal) == 1
    assert alternative_action_from_q(q, legal, behavior_action=1) == 3
    assert int(np.argmax(g_total[legal])) == 0


def test_tiny_analysis_orchestration_and_json_report(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    n = 12
    rng = np.random.default_rng(123)
    z = rng.normal(size=(n, 8))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    hanchan = np.repeat(np.arange(4), 3).astype(np.int64)
    hashes = [f"h-{h}" for h in hanchan]

    def fake_load_canonical_route(root, route):
        return {
            "z": z,
            "hanchan_index": hanchan,
            "sorted_hashes": sorted(set(hashes)),
            "manifest": {},
        }

    import training.mortal.audit_k0_representation_space_2026_08 as repr_runner
    monkeypatch.setattr(repr_runner, "_load_canonical_route", fake_load_canonical_route)
    monkeypatch.setattr(runner, "estimate_g1_sigma", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        runner,
        "make_g1_rff_features",
        lambda sigma: (
            np.random.default_rng(0).normal(size=(4, ACTION_DIM)),
            np.random.default_rng(1).uniform(0, 2 * np.pi, size=4),
        ),
    )
    monkeypatch.setattr(runner, "BOOTSTRAP_REPS", 3)
    monkeypatch.setattr(runner, "SUPPORT_K", 2)
    monkeypatch.setattr(runner, "K_CREDIT", 4)
    small_draws = {
        "m0": np.asarray([[0, 1, 2, 3]] * 3, dtype=np.int32),
        "d12": np.asarray([[0, 1, 2, 3]] * 3, dtype=np.int32),
        "d3": np.asarray([[0, 1, 2, 3]] * 3, dtype=np.int32),
    }
    monkeypatch.setattr(runner, "make_frozen_bootstrap_draws", lambda n, reps, seed: small_draws)

    canonical_by_route = {}
    rehydrated_by_route = {}
    q_by_route = {}
    actions = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    masks = np.zeros((n, ACTION_DIM), dtype=bool)
    masks[:, :6] = True
    targets = np.linspace(-1.0, 1.0, n)
    q = rng.normal(size=(n, ACTION_DIM))
    q[:, 6:] = -np.inf
    for route in ROUTE_ORDER:
        canonical_by_route[route] = {
            "rows": n,
            "hanchan_index": hanchan,
            "file_index": np.arange(n, dtype=np.int64),
            "row_index": np.arange(n, dtype=np.int64),
            "target": targets,
            "z": z,
            "canonical_hanchan_hashes": hashes,
            "perspective_labels": [route] * n,
        }
        rehydrated_by_route[route] = {
            "actions": actions.copy(),
            "masks": masks.copy(),
            "targets": targets.copy(),
            "z": z.copy(),
            "q": q.copy(),
        }
        q_by_route[route] = q.copy()

    analysis = _run_decision_analysis(
        tmp_path / "repr",
        canonical_by_route,
        rehydrated_by_route,
        q_by_route,
        z,
    )
    assert "support" in analysis
    assert "gradient" in analysis
    assert "action_credit" in analysis
    assert analysis["verdict"] in {
        "inconclusive",
        "state_action_support_signal",
        "objective_gradient_shift_signal",
        "both_signals",
    }
    safe = _json_safe_report(analysis, tmp_path / "report_out")
    import json
    text = json.dumps(safe, ensure_ascii=False, indent=2)
    assert "ndarray" not in text
    assert (tmp_path / "report_out" / "bootstrap").is_dir()


def test_flatten_group_gradient_uses_exact_full_tuple_slicing() -> None:
    from torch import nn

    brain = nn.Linear(2, 2)
    dqn = nn.Linear(2, 2)
    aux = nn.Linear(2, 2)
    all_params = list(brain.parameters()) + list(dqn.parameters()) + list(aux.parameters())
    assert len(all_params) == 6
    grads = tuple(torch.ones_like(p) * i for i, p in enumerate(all_params))
    grads_none = list(grads)
    grads_none[0] = None
    grads_none = tuple(grads_none)

    brain_flat = _flatten_group_gradient(all_params, grads_none, list(brain.parameters()))
    assert brain_flat.shape[0] == sum(p.numel() for p in brain.parameters())
    assert brain_flat[0] == 0.0
    assert brain_flat[-1] == 1.0

    dqn_flat = _flatten_group_gradient(all_params, grads, list(dqn.parameters()))
    assert dqn_flat.shape[0] == sum(p.numel() for p in dqn.parameters())
    assert dqn_flat[0] == 2.0

    aux_flat = _flatten_group_gradient(all_params, grads, list(aux.parameters()))
    assert aux_flat.shape[0] == sum(p.numel() for p in aux.parameters())
    assert aux_flat[0] == 4.0

    bd_flat = _flatten_group_gradient(all_params, grads, list(brain.parameters()) + list(dqn.parameters()))
    assert bd_flat.shape[0] == sum(p.numel() for p in brain.parameters()) + sum(p.numel() for p in dqn.parameters())
    assert bd_flat[0] == 0.0
    assert bd_flat[-1] == 3.0


def test_adam_diagnostic_for_route_toy_smoke() -> None:
    from torch import nn, optim

    class ToyBrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            return self.fc(x)

    class ToyDQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, ACTION_DIM)

        def forward(self, phi, mask):
            q = self.fc(phi)
            q = q.masked_fill(~mask, float("-inf"))
            return q

    class ToyAux(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, phi):
            return (self.fc(phi),)

    class Rec:
        def __init__(self):
            self.obs = np.random.default_rng(0).normal(size=4).astype(np.float32)
            self.mask = np.zeros(ACTION_DIM, dtype=bool)
            self.mask[:4] = True
            self.action = 1
            self.q_target = 0.5
            self.player_rank = 0

    brain = ToyBrain()
    dqn = ToyDQN()
    aux = ToyAux()
    optimizer = optim.AdamW(list(brain.parameters()) + list(dqn.parameters()) + list(aux.parameters()), lr=0.01)
    dummy = torch.zeros(4)
    phi = brain(dummy)
    mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
    mask[0, :4] = True
    loss = dqn(phi, mask).sum() + aux(phi)[0].sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    config = {
        "cql": {"min_q_weight": 5.0},
        "aux": {"next_rank_weight": 0.2},
        "optim": {"eps": 1e-8},
    }
    records = [Rec() for _ in range(64)]
    result = _adam_diagnostic_for_route("M0", records, brain, dqn, aux, optimizer, config, torch.device("cpu"))
    assert result["batch_count"] == 2
    assert "brain" in result["summary"]
    assert "dqn" in result["summary"]
    assert "aux" in result["summary"]
    assert "brain_dqn" in result["summary"]
    assert result["dqn_cql_parameter_cosine"]["brain"]["cos_mean"] is not None


def test_outer_run_formal_audit_synthetic_e2e(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    monkeypatch.setattr(runner, "FORMAL_RUN_AUTHORIZED", True)
    monkeypatch.setattr(runner, "formal_preflight", lambda device, out: {"all_pass": True})
    def fake_canonical(root):
        return {
            route: {
                "canonical_hanchan_hashes": [f"{route}-h"],
                "file_index": np.asarray([0], dtype=np.int64),
                "row_index": np.asarray([0], dtype=np.int64),
                "perspective_labels": [route],
            }
            for route in ROUTE_ORDER
        }
    monkeypatch.setattr(runner, "_load_decision_canonical", fake_canonical)
    monkeypatch.setattr(
        runner,
        "_rehydrate_canonical_rows",
        lambda *a, **k: {
            "actions": np.asarray([0], dtype=np.int64),
            "masks": np.eye(1, ACTION_DIM, dtype=bool),
            "targets": np.asarray([0.0]),
            "q": np.zeros((1, ACTION_DIM)),
        },
    )
    monkeypatch.setattr(runner, "_collect_microbatch_records", lambda *a, **k: [])
    monkeypatch.setattr(
        runner,
        "_adam_diagnostic_for_route",
        lambda *a, **k: {"summary": {"brain": {"cos_g_m_mean": 0.5}}, "per_batch": []},
    )
    monkeypatch.setattr(
        runner,
        "_run_decision_analysis",
        lambda *a, **k: {
            "verdict": "inconclusive",
            "support": {"x": np.asarray([1.0, 2.0])},
            "gradient": {"y": np.asarray([3.0])},
            "action_credit": {},
        },
    )
    import training.mortal.audit_k0_representation_space_2026_08 as repr_runner
    import training.mortal.audit_objective_learnability as obj_runner
    import training.mortal.audit_replay_distribution as replay_runner
    monkeypatch.setattr(obj_runner, "_build_model", lambda state, device: (None, None, None, 4))
    monkeypatch.setattr(replay_runner, "load_model", lambda state, device: (None, None, 4))
    monkeypatch.setattr(runner, "_build_preserved_optimizer", lambda *a, **k: None)
    monkeypatch.setattr(
        repr_runner,
        "_load_canonical_route",
        lambda root, route: {
            "z": np.zeros((1, 8)),
            "hanchan_index": np.asarray([0], dtype=np.int64),
            "sorted_hashes": ["h"],
        },
    )
    monkeypatch.setattr(runner, "sample_microbatch_rows", lambda *a, **k: np.zeros((1, 1), dtype=np.int64))
    out_root = tmp_path / "formal_out"
    runner.run_formal_audit(torch.device("cpu"), out_root)

    report_path = out_root / "report.json"
    assert report_path.is_file()
    import json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["verdict"]["authoritative"] is True
    assert report["verdict"]["readout"] == "inconclusive"
    assert "runtime" in report
    assert "adam_diagnostic" in report["analysis"]
    # Arrays should have been artifactized, not leaked as ndarray.
    assert (out_root / "bootstrap" / "support_x.npy").is_file()
    assert (out_root / "bootstrap" / "gradient_y.npy").is_file()


def test_outer_run_formal_audit_preflight_failure_does_not_write_scientific_report(monkeypatch, tmp_path) -> None:
    import training.mortal.audit_k0_decision_signal_2026_08 as runner

    monkeypatch.setattr(runner, "FORMAL_RUN_AUTHORIZED", True)
    monkeypatch.setattr(runner, "formal_preflight", lambda device, out: {"all_pass": False})
    out_root = tmp_path / "formal_fail"
    try:
        runner.run_formal_audit(torch.device("cpu"), out_root)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on preflight failure")
    assert not (out_root / "report.json").exists()


def _toy_optimizer_state_for_build_preserved():
    from torch import nn

    from training.mortal.preflight_optimizer_ab import make_optimizer, make_scheduler

    brain = nn.Linear(4, 4)
    dqn = nn.Linear(4, 4)
    aux = nn.Linear(4, 4)
    config = {
        "optim": {
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": {
                "peak": 1e-4,
                "final": 1e-5,
                "warm_up_steps": 100,
                "max_steps": 1000,
                "init": 1e-8,
            },
        }
    }
    optimizer = make_optimizer(config, (brain, dqn, aux))
    _ = make_scheduler(config, optimizer)
    # Create optimizer state without needing real data.
    for param in list(brain.parameters()) + list(dqn.parameters()) + list(aux.parameters()):
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    return brain, dqn, aux, config, optimizer.state_dict()


def test_build_preserved_optimizer_accepts_matching_state() -> None:
    brain, dqn, aux, config, state_dict = _toy_optimizer_state_for_build_preserved()
    state = {"config": config, "optimizer": state_dict}
    optimizer = _build_preserved_optimizer(state, brain, dqn, aux)
    assert optimizer is not None


def test_build_preserved_optimizer_rejects_tampered_group_metadata() -> None:
    import copy

    brain, dqn, aux, config, state_dict = _toy_optimizer_state_for_build_preserved()
    state = {"config": config, "optimizer": copy.deepcopy(state_dict)}
    # Tamper with eps in the preserved checkpoint param group.
    for group in state["optimizer"]["param_groups"]:
        group["eps"] = 9.9
    try:
        _build_preserved_optimizer(state, brain, dqn, aux)
    except RuntimeError:
        return
    raise AssertionError("expected preserved optimizer validation to fail")


def test_rehydration_uses_batched_inference(monkeypatch) -> None:
    from libriichi import dataset
    from torch import nn

    class FakeGRP:
        def take_rank_by_player(self):
            return [0, 1, 2, 3]

    class FakeGame:
        def __init__(self, n):
            self.n = n

        def take_obs(self):
            return [np.zeros(4, dtype=np.float32) for _ in range(self.n)]

        def take_actions(self):
            return [0, 1, 2][: self.n]

        def take_masks(self):
            masks = np.zeros((self.n, ACTION_DIM), dtype=bool)
            masks[:, :4] = True
            return list(masks)

        def take_grp(self):
            return FakeGRP()

        def take_player_id(self):
            return 0

    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def load_gz_log_files(self, paths):
            return [[FakeGame(3)]]

    monkeypatch.setattr(dataset, "GameplayLoader", FakeLoader)

    brain_batch_sizes = []
    dqn_batch_sizes = []

    class FakeBrain(nn.Module):
        def forward(self, obs):
            brain_batch_sizes.append(obs.shape[0])
            return torch.ones(obs.shape[0], 1024, dtype=torch.float64)

    class FakeDQN(nn.Module):
        def forward(self, phi, mask):
            dqn_batch_sizes.append(phi.shape[0])
            q = torch.zeros(phi.shape[0], ACTION_DIM, dtype=torch.float64)
            q[mask] = 0.5
            q[~mask] = float("-inf")
            return q

    canonical = {
        "file_index": np.asarray([0, 0, 0], dtype=np.int64),
        "row_index": np.asarray([0, 1, 2], dtype=np.int64),
        "target": np.asarray([3.0, 3.0, 3.0]),
        "perspective_labels": ["M0", "M0", "M0"],
        "z": np.tile(np.asarray([0.5, 0.5, 0.5, 0.5]), (3, 1)),
    }
    route_spec = {"index": ["dummy.json.gz"], "labels": ["M0"], "by_file": None}
    result = _rehydrate_canonical_rows(
        "M0",
        route_spec,
        canonical,
        FakeBrain(),
        FakeDQN(),
        torch.device("cpu"),
        np.eye(4, 1024, dtype=np.float64),
    )
    assert result["actions"].tolist() == [0, 1, 2]
    assert brain_batch_sizes == [3]
    assert dqn_batch_sizes == [3]
