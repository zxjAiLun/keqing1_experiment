#!/usr/bin/env python3
"""Pure frozen numeric core for K0 State-Action Training-Signal audit (v1).

This module implements only the frozen protocol from:

    training/docs/mortal/experiments_zh/
    2026-08_K0状态动作训练信号机制审计_预注册设计.md

It deliberately contains no loader, model, corpus, or report code.
All functions are deterministic for fixed seed and fixed input ordering.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from training.mortal.k0_representation_audit_core import (
    estimate_sigma_from_pairs,
    knn_neighbor_indices_blockwise,
    make_rff_features,
)

# ---------------------------------------------------------------- frozen knobs
ACTION_DIM = 46
SUPPORT_K = 16
K_CREDIT = 64
CQL_LAMBDA = 5.0
GRAD_EPS = 1e-12
RFF_DIM = 2048
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260821
PAIR_SEED_G1 = 20260822
RFF_SEED_G1 = 20260823
MICROBATCH_SEED = 20260824
FALLBACK_SIGMA = 0.1
ROUTE_ORDER = ("M0", "D1", "D2", "D3")
REHYDRATION_RTOL = 1e-5
REHYDRATION_ATOL = 1e-6


# ---------------------------------------------------------------- Q gradients
def _softmax_legal(q: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Stable softmax over legal coordinates only, zeros elsewhere."""
    q = np.asarray(q, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if q.shape != legal_mask.shape or q.ndim != 1:
        raise ValueError("q and legal_mask must be same-shape 1-D arrays")
    legal_q = q[legal_mask]
    if legal_q.size == 0:
        raise ValueError("legal_mask must contain at least one legal action")
    shifted = legal_q - float(np.max(legal_q))
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    out = np.zeros_like(q, dtype=np.float64)
    out[legal_mask] = probs
    return out


def compute_row_gradients(
    q: np.ndarray,
    legal_mask: np.ndarray,
    behavior_action: int,
    target: float,
    cql_lambda: float = CQL_LAMBDA,
) -> dict[str, np.ndarray]:
    """Analytic row-level Q-output gradients for behavior_action_mc + CQL.

    Returns ``g_value``, ``g_cql``, ``g_total`` as 46-d arrays.
    """
    q = np.asarray(q, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if q.shape != (ACTION_DIM,) or legal_mask.shape != (ACTION_DIM,):
        raise ValueError(f"q/legal_mask must be 1-D of length {ACTION_DIM}")
    if not legal_mask[behavior_action]:
        raise ValueError("behavior_action must be legal")
    one_hot = np.zeros(ACTION_DIM, dtype=np.float64)
    one_hot[behavior_action] = 1.0
    g_value = (float(q[behavior_action]) - float(target)) * one_hot
    softmax_legal = _softmax_legal(q, legal_mask)
    g_cql = cql_lambda * (softmax_legal - one_hot)
    g_total = g_value + g_cql
    return {"g_value": g_value, "g_cql": g_cql, "g_total": g_total}


def normalize_gradient_direction(
    g_total: np.ndarray,
    grad_eps: float = GRAD_EPS,
) -> tuple[np.ndarray, bool]:
    """Frozen G1 zero/normalization rule."""
    g_total = np.asarray(g_total, dtype=np.float64)
    norm = float(np.linalg.norm(g_total))
    if norm > grad_eps:
        return g_total / norm, False
    return np.zeros_like(g_total), True


def cosine_defined(
    g_value: np.ndarray,
    g_cql: np.ndarray,
    grad_eps: float = GRAD_EPS,
) -> tuple[float | None, bool]:
    """Frozen G2 defined-row cosine."""
    g_value = np.asarray(g_value, dtype=np.float64)
    g_cql = np.asarray(g_cql, dtype=np.float64)
    norm_v = float(np.linalg.norm(g_value))
    norm_c = float(np.linalg.norm(g_cql))
    if norm_v <= grad_eps or norm_c <= grad_eps:
        return None, False
    return float(np.dot(g_value, g_cql) / (norm_v * norm_c)), True


def centered_preference_pressure(
    g_total: np.ndarray,
    legal_mask: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Frozen G3 common/centered decomposition."""
    g_total = np.asarray(g_total, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    legal_count = int(legal_mask.sum())
    if legal_count <= 0:
        raise ValueError("legal_mask must contain at least one legal action")
    u = legal_mask.astype(np.float64) / float(np.sqrt(legal_count))
    g_common = float(np.dot(g_total, u)) * u
    g_centered = g_total - g_common
    return {
        "g_common": g_common,
        "g_centered": g_centered,
        "m_centered": float(np.linalg.norm(g_centered)),
        "legal_count": legal_count,
    }


# ---------------------------------------------------------------- bootstrap draws
def make_frozen_bootstrap_draws(
    n_hanchans: int,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Frozen decision-signal bootstrap draws.

    ``M0`` and ``D3`` are independent; ``D1`` and ``D2`` share ``d12``.
    """
    if n_hanchans <= 0:
        raise ValueError("n_hanchans must be positive")
    rng = np.random.default_rng(seed)
    m0 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    d12 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    d3 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    return {"m0": m0, "d12": d12, "d3": d3}


def build_pooled_anchor_weights(
    anchor_routes: np.ndarray,
    anchor_hanchan: np.ndarray,
    draws: dict[str, np.ndarray],
    reps: int,
) -> np.ndarray:
    """Build the frozen 72k pooled anchor weights.

    ``anchor_routes`` is a string array of per-anchor route labels.
    ``anchor_hanchan`` is the per-anchor route-local hanchan index.
    For each replicate:

        M0 anchors -> w_M0
        D1 anchors -> w_D12
        D2 anchors -> same w_D12
        D3 anchors -> w_D3

    Returns shape ``(reps, n_anchors)``.
    """
    anchor_routes = np.asarray(anchor_routes)
    anchor_hanchan = np.asarray(anchor_hanchan, dtype=np.int64)
    n_anchors = anchor_routes.shape[0]
    if anchor_hanchan.shape != (n_anchors,):
        raise ValueError("anchor_routes and anchor_hanchan must have equal length")
    weights = np.zeros((reps, n_anchors), dtype=np.float64)
    for route, key in (("M0", "m0"), ("D1", "d12"), ("D2", "d12"), ("D3", "d3")):
        mask = anchor_routes == route
        if not np.any(mask):
            continue
        local_hanchan = anchor_hanchan[mask]
        draw = np.asarray(draws[key], dtype=np.int64)
        if draw.shape[0] < reps:
            raise ValueError(f"not enough replicates in draw {key}")
        n_hanchans = int(draw.shape[1])
        for rep in range(reps):
            counts = np.bincount(draw[rep], minlength=n_hanchans).astype(np.float64)
            weights[rep, mask] = counts[local_hanchan]
    return weights


# ---------------------------------------------------------------- support
def compute_support_neighbors(

    anchor_z: np.ndarray,
    anchor_hanchan: np.ndarray,
    reference_z_by_route: dict[str, np.ndarray],
    reference_hanchan_by_route: dict[str, np.ndarray],
    k: int = SUPPORT_K,
) -> dict[str, np.ndarray]:
    """Frozen support neighbor lookup.

    This function computes the four fixed reference neighborhoods once.  The
    support bootstrap later changes only anchor weights and never recomputes or
    reweights these neighbor sets.
    """
    anchor_z = np.asarray(anchor_z, dtype=np.float64)
    anchor_hanchan = np.asarray(anchor_hanchan)
    if anchor_z.ndim != 2 or anchor_hanchan.shape != (anchor_z.shape[0],):
        raise ValueError("anchor_z/anchor_hanchan shape mismatch")
    out: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        ref_z = np.asarray(reference_z_by_route[route], dtype=np.float64)
        ref_hanchan = np.asarray(reference_hanchan_by_route[route])
        if ref_z.ndim != 2 or ref_hanchan.shape != (ref_z.shape[0],):
            raise ValueError(f"{route} reference shape mismatch")
        out[route] = knn_neighbor_indices_blockwise(
            anchor_z, ref_z, anchor_hanchan, ref_hanchan, k=k
        )
    return out


def support_metrics_for_anchor(

    query_action: int,
    query_legal: np.ndarray,
    neighbor_actions: np.ndarray,
    neighbor_legals: np.ndarray,
    k: int = SUPPORT_K,
) -> dict[str, float]:
    """S1/S2 for one anchor against one frozen reference neighborhood."""
    query_legal = np.asarray(query_legal, dtype=bool)
    neighbor_actions = np.asarray(neighbor_actions, dtype=np.int64)
    neighbor_legals = np.asarray(neighbor_legals, dtype=bool)
    if neighbor_actions.shape != (k,) or neighbor_legals.shape != (k, ACTION_DIM):
        raise ValueError("neighbor_actions/neighbor_legals shape mismatch")
    if not query_legal[query_action]:
        raise ValueError("anchor behavior action must be legal in query")
    eligible = np.zeros(k, dtype=bool)
    for j in range(k):
        a_j = int(neighbor_actions[j])
        eligible[j] = bool(query_legal[a_j] and neighbor_legals[j, query_action])
    switched = eligible & (neighbor_actions != query_action)
    # Denominator is fixed to SUPPORT_K per prereg, not the supplied k.
    switchable_rate = float(np.count_nonzero(switched)) / float(SUPPORT_K)
    distinct_alt = float(len({int(neighbor_actions[j]) for j in range(k) if switched[j]}))
    return {
        "switchable_rate": switchable_rate,
        "distinct_alt": distinct_alt,
        "eligible_count": float(np.count_nonzero(eligible)),
        "switch_count": float(np.count_nonzero(switched)),
    }


def support_bootstrap_deltas(
    anchor_metric_by_route: dict[str, np.ndarray],
    anchor_weights: np.ndarray,
    reps: int,
    ci: float = 2.5,
) -> dict[str, Any]:
    """Bootstrap deltas for one support metric.

    ``anchor_metric_by_route`` maps each route to per-anchor metric values.
    ``anchor_weights`` has shape ``(reps, n_anchors)``; each row is normalized
    internally.  Reference neighbor membership is fixed outside this function.
    """
    n_anchors = None
    for route, values in anchor_metric_by_route.items():
        values = np.asarray(values, dtype=np.float64)
        if n_anchors is None:
            n_anchors = values.size
        elif values.size != n_anchors:
            raise ValueError("all route metric arrays must have the same anchor count")
    if n_anchors is None:
        raise ValueError("anchor_metric_by_route must be non-empty")
    anchor_weights = np.asarray(anchor_weights, dtype=np.float64)
    if anchor_weights.ndim != 2 or anchor_weights.shape[1] != n_anchors:
        raise ValueError(f"anchor_weights must have shape (reps, {n_anchors})")
    if anchor_weights.shape[0] < reps:
        raise ValueError("anchor_weights has fewer reps than requested")

    s_by_route: dict[str, np.ndarray] = {}
    for route, values in anchor_metric_by_route.items():
        values = np.asarray(values, dtype=np.float64)
        w = anchor_weights[:reps]
        total = w.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("each replicate anchor weight sum must be positive")
        s_by_route[route] = (values[None, :] * w).sum(axis=1) / total[:, 0]

    s_m0 = s_by_route["M0"]
    s_d1 = s_by_route["D1"]
    s_d2 = s_by_route["D2"]
    s_d3 = s_by_route["D3"]
    view_gap = np.abs(s_d1 - s_d2)
    delta1 = (s_m0 - s_d1) - view_gap
    delta3 = (s_m0 - s_d3) - view_gap
    return {
        "s_m0": s_m0,
        "s_d1": s_d1,
        "s_d2": s_d2,
        "s_d3": s_d3,
        "view_gap": view_gap,
        "delta1": delta1,
        "delta3": delta3,
        "delta1_ci": (float(np.percentile(delta1, ci)), float(np.percentile(delta1, 100.0 - ci))),
        "delta3_ci": (float(np.percentile(delta3, ci)), float(np.percentile(delta3, 100.0 - ci))),
    }


def support_signal_from_bootstrap(
    s1_result: dict[str, Any],
    s2_result: dict[str, Any],
) -> bool:
    """Frozen 2/2 support rule."""
    return bool(
        s1_result["delta1_ci"][0] > 0.0
        and s1_result["delta3_ci"][0] > 0.0
        and s2_result["delta1_ci"][0] > 0.0
        and s2_result["delta3_ci"][0] > 0.0
    )


# ---------------------------------------------------------------- weighted W1
def weighted_wasserstein_1d(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    """Exact weighted empirical CDF 1-Wasserstein.

    Duplicate scalar values are aggregated; each side's weights are normalized
    to sum 1; the integral over the merged breakpoint union is exact.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    weights_a = np.asarray(weights_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    if values_a.ndim != 1 or weights_a.shape != values_a.shape:
        raise ValueError("values_a/weights_a must be 1-D equal length")
    if values_b.ndim != 1 or weights_b.shape != values_b.shape:
        raise ValueError("values_b/weights_b must be 1-D equal length")
    if values_a.size == 0 or values_b.size == 0:
        raise ValueError("weighted_wasserstein_1d requires non-empty inputs")
    if np.any(weights_a < 0) or np.any(weights_b < 0):
        raise ValueError("weights must be non-negative")
    if weights_a.sum() <= 0 or weights_b.sum() <= 0:
        raise ValueError("weights must have positive sum")

    def _cdf_at(values: np.ndarray, weights: np.ndarray, points: np.ndarray) -> np.ndarray:
        normalized = weights / weights.sum()
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_weights = normalized[order]
        cum = np.cumsum(sorted_weights)
        # CDF at each point: total mass of values <= point.
        indices = np.searchsorted(sorted_values, points, side="right") - 1
        out = np.zeros(points.shape, dtype=np.float64)
        valid = indices >= 0
        out[valid] = cum[indices[valid]]
        return out

    points = np.union1d(values_a, values_b)
    cdf_a = _cdf_at(values_a, weights_a, points)
    cdf_b = _cdf_at(values_b, weights_b, points)
    total = 0.0
    for i in range(points.size - 1):
        width = float(points[i + 1] - points[i])
        total += width * abs(float(cdf_a[i] - cdf_b[i]))
    return float(total)


# ---------------------------------------------------------------- gradient metrics
def gradient_family_bootstrap_deltas(
    values_by_route: dict[str, np.ndarray],
    row_hanchan_by_route: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    reps: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap deltas for a scalar-per-row gradient family.

    ``distance_fn`` receives ``(values_a, weights_a, values_b, weights_b)`` and
    returns a scalar distance.  For G2/G3 this is weighted Wasserstein; for G1
    the values are unit gradient directions and distance_fn is RFF-MMD.
    """
    # Build per-route multiplicity weights for each replicate.
    weights_by_route: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        row_hanchan = np.asarray(row_hanchan_by_route[route], dtype=np.int64)
        if row_hanchan.size == 0:
            raise ValueError(f"route {route} has no rows")
        if route == "M0":
            key = "m0"
        elif route in ("D1", "D2"):
            key = "d12"
        else:
            key = "d3"
        counts = np.zeros(int(row_hanchan.max()) + 1, dtype=np.float64)
        rep_weights = np.empty((reps, row_hanchan.size), dtype=np.float64)
        for rep in range(reps):
            counts.fill(0.0)
            np.add.at(counts, draws[key][rep].astype(np.int64), 1.0)
            rep_weights[rep] = counts[row_hanchan]
        weights_by_route[route] = rep_weights

    d_m0_d1 = np.empty(reps, dtype=np.float64)
    d_m0_d3 = np.empty(reps, dtype=np.float64)
    d_d1_d2 = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        d_m0_d1[rep] = distance_fn(
            values_by_route["M0"], weights_by_route["M0"][rep],
            values_by_route["D1"], weights_by_route["D1"][rep],
        )
        d_m0_d3[rep] = distance_fn(
            values_by_route["M0"], weights_by_route["M0"][rep],
            values_by_route["D3"], weights_by_route["D3"][rep],
        )
        d_d1_d2[rep] = distance_fn(
            values_by_route["D1"], weights_by_route["D1"][rep],
            values_by_route["D2"], weights_by_route["D2"][rep],
        )
    delta1 = d_m0_d1 - d_d1_d2
    delta3 = d_m0_d3 - d_d1_d2
    return {
        "d_m0_d1": d_m0_d1,
        "d_m0_d3": d_m0_d3,
        "d_d1_d2": d_d1_d2,
        "delta1": delta1,
        "delta3": delta3,
        "delta1_ci": (float(np.percentile(delta1, 2.5)), float(np.percentile(delta1, 97.5))),
        "delta3_ci": (float(np.percentile(delta3, 2.5)), float(np.percentile(delta3, 97.5))),
    }


def make_g1_rff_features(
    sigma: float,
    seed: int = RFF_SEED_G1,
    input_dim: int = ACTION_DIM,
    n_features: int = RFF_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen G1 random Fourier features on 46-d unit Q-gradients."""
    omega, bias = make_rff_features(
        sigma=sigma, seed=seed, input_dim=input_dim, n_features=n_features
    )
    return omega.numpy(), bias.numpy()


def estimate_g1_sigma(
    gdirs_by_route: dict[str, np.ndarray],
    hanchan_by_route: dict[str, np.ndarray],
    pair_seed: int = PAIR_SEED_G1,
    pairs_per_route: int = 65536,
) -> float:
    """Frozen G1 sigma from pooled four-route cross-hanchan pair distances."""
    z_by_route = {route: np.asarray(gdirs_by_route[route], dtype=np.float64) for route in ROUTE_ORDER}
    hanchan_ids = {route: np.asarray(hanchan_by_route[route]) for route in ROUTE_ORDER}
    return estimate_sigma_from_pairs(z_by_route, hanchan_ids, pair_seed=pair_seed, pairs_per_route=pairs_per_route)


def precompute_g1_rff_features(
    values: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Compute the frozen RFF feature matrix once for G1."""
    values = np.asarray(values, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    scale = float(np.sqrt(2.0 / omega.shape[0]))
    return scale * np.cos(values @ omega.T + bias[None, :])


def rff_mmd2_from_features(
    features_a: np.ndarray,
    weights_a: np.ndarray,
    features_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    """RFF-MMD using precomputed feature rows (no repeated cos)."""
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    weights_a = np.asarray(weights_a, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    if weights_a.sum() <= 0 or weights_b.sum() <= 0:
        raise ValueError("RFF-MMD requires positive weight sums")
    weights_a = weights_a / weights_a.sum()
    weights_b = weights_b / weights_b.sum()
    mean_a = weights_a @ features_a
    mean_b = weights_b @ features_b
    return float(np.sum((mean_a - mean_b) ** 2))


def g1_rff_mmd2_weighted(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
) -> float:
    """Frozen G1 RFF-MMD distance on unit gradient directions."""
    features_a = precompute_g1_rff_features(values_a, omega, bias)
    features_b = precompute_g1_rff_features(values_b, omega, bias)
    return rff_mmd2_from_features(features_a, weights_a, features_b, weights_b)


def g1_bootstrap_deltas_from_features(
    features_by_route: dict[str, np.ndarray],
    row_hanchan_by_route: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    reps: int,
) -> dict[str, Any]:
    """Optimized G1 bootstrap using precomputed RFF feature rows.

    RFF features are computed once per route; each replicate only forms
    multiplicity-weighted feature means from per-hanchan feature sums.
    """
    n_hanchans = int(max(int(np.max(row_hanchan_by_route[route])) for route in ROUTE_ORDER) + 1)
    hanchan_sums: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        features = np.asarray(features_by_route[route], dtype=np.float64)
        row_hanchan = np.asarray(row_hanchan_by_route[route], dtype=np.int64)
        sums = np.zeros((n_hanchans, features.shape[1]), dtype=np.float64)
        for hanchan in range(n_hanchans):
            rows = np.flatnonzero(row_hanchan == hanchan)
            if rows.size == 0:
                continue
            sums[hanchan] = features[rows].sum(axis=0)
        hanchan_sums[route] = sums

    def _counts(key: str, rep: int) -> np.ndarray:
        return np.bincount(draws[key][rep].astype(np.int64), minlength=n_hanchans).astype(np.float64)

    def _mmd(route_a: str, route_b: str, rep: int) -> float:
        key_a = "m0" if route_a == "M0" else "d12" if route_a in ("D1", "D2") else "d3"
        key_b = "m0" if route_b == "M0" else "d12" if route_b in ("D1", "D2") else "d3"
        counts_a = _counts(key_a, rep)
        counts_b = _counts(key_b, rep)
        mean_a = (counts_a @ hanchan_sums[route_a]) / (3.0 * counts_a.sum())
        mean_b = (counts_b @ hanchan_sums[route_b]) / (3.0 * counts_b.sum())
        return float(np.sum((mean_a - mean_b) ** 2))

    d_m0_d1 = np.asarray([_mmd("M0", "D1", rep) for rep in range(reps)], dtype=np.float64)
    d_m0_d3 = np.asarray([_mmd("M0", "D3", rep) for rep in range(reps)], dtype=np.float64)
    d_d1_d2 = np.asarray([_mmd("D1", "D2", rep) for rep in range(reps)], dtype=np.float64)
    delta1 = d_m0_d1 - d_d1_d2
    delta3 = d_m0_d3 - d_d1_d2
    return {
        "d_m0_d1": d_m0_d1,
        "d_m0_d3": d_m0_d3,
        "d_d1_d2": d_d1_d2,
        "delta1": delta1,
        "delta3": delta3,
        "delta1_ci": (float(np.percentile(delta1, 2.5)), float(np.percentile(delta1, 97.5))),
        "delta3_ci": (float(np.percentile(delta3, 2.5)), float(np.percentile(delta3, 97.5))),
    }


def gradient_family_vote(result: dict[str, Any]) -> bool:
    return bool(result["delta1_ci"][0] > 0.0 and result["delta3_ci"][0] > 0.0)


def q_gradient_signal_from_family_votes(votes: list[bool]) -> bool:
    return sum(bool(vote) for vote in votes) >= 2


def greedy_action_from_q(q: np.ndarray, legal_mask: np.ndarray) -> int:
    """Frozen descriptive helper: argmax K0 Q over legal actions."""
    q = np.asarray(q, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise ValueError("legal_mask must contain at least one action")
    return int(legal_indices[int(np.argmax(q[legal_indices]))])


def alternative_action_from_q(
    q: np.ndarray,
    legal_mask: np.ndarray,
    behavior_action: int,
) -> int:
    """Frozen descriptive helper: argmax K0 Q among legal excluding behavior."""
    q = np.asarray(q, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if not legal_mask[behavior_action]:
        raise ValueError("behavior_action must be legal")
    legal_indices = np.flatnonzero(legal_mask)
    alt_indices = legal_indices[legal_indices != behavior_action]
    if alt_indices.size == 0:
        raise ValueError("no alternative legal action")
    return int(alt_indices[int(np.argmax(q[alt_indices]))])


# ---------------------------------------------------------------- action credit
def action_credit_stats(
    query_action: int,
    query_legal: np.ndarray,
    neighbor_actions: np.ndarray,
    neighbor_legals: np.ndarray,
    neighbor_targets: np.ndarray,
    k: int = K_CREDIT,
    min_groups: int = 2,
    min_group_size: int = 4,
) -> dict[str, Any]:
    """Descriptive action-level credit on a k=64 neighborhood.

    Returns NaN/None fields when the query is ineligible.
    """
    query_legal = np.asarray(query_legal, dtype=bool)
    neighbor_actions = np.asarray(neighbor_actions, dtype=np.int64)
    neighbor_legals = np.asarray(neighbor_legals, dtype=bool)
    neighbor_targets = np.asarray(neighbor_targets, dtype=np.float64)
    if neighbor_actions.shape != (k,) or neighbor_legals.shape != (k, ACTION_DIM):
        raise ValueError("neighbor arrays shape mismatch")
    if neighbor_targets.shape != (k,):
        raise ValueError("neighbor_targets shape mismatch")
    eligible = np.zeros(k, dtype=bool)
    for j in range(k):
        a_j = int(neighbor_actions[j])
        eligible[j] = bool(query_legal[a_j] and neighbor_legals[j, query_action])
    groups: Counter[int] = Counter()
    for j in np.flatnonzero(eligible):
        groups[int(neighbor_actions[j])] += 1
    valid_groups = [action for action, count in groups.items() if count >= min_group_size]
    if len(valid_groups) < min_groups:
        return {
            "eligible": False,
            "eligible_query_fraction": None,
            "h_state": None,
            "h_state_action": None,
            "action_information_gain": None,
            "eligible_count": int(np.count_nonzero(eligible)),
            "valid_group_count": len(valid_groups),
        }

    def _entropy(values: np.ndarray) -> float:
        counts = Counter(float(value) for value in values.tolist())
        total = sum(counts.values())
        return float(-sum((count / total) * np.log(count / total) for count in counts.values()))

    total_eligible = float(np.count_nonzero(eligible))
    h_state = _entropy(neighbor_targets[eligible])
    h_state_action = 0.0
    # Once eligible, all mutually-legal action groups enter the decomposition,
    # including groups smaller than min_group_size.  min_group_size only gates
    # whether the query has enough support for readout.
    for action, count in groups.items():
        mask = eligible & (neighbor_actions == action)
        weight = float(count) / total_eligible
        h_state_action += weight * _entropy(neighbor_targets[mask])
    return {
        "eligible": True,
        "eligible_query_fraction": None,
        "h_state": h_state,
        "h_state_action": h_state_action,
        "action_information_gain": h_state - h_state_action,
        "eligible_count": int(np.count_nonzero(eligible)),
        "valid_group_count": len(valid_groups),
    }


def action_credit_route_stats(
    z: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    hanchan_ids: np.ndarray,
    k: int = K_CREDIT,
) -> dict[str, object]:
    """Aggregate action-credit descriptive stats for one route.

    Uses exact k=64 same-route neighborhoods with same-hanchan exclusion.
    """
    from training.mortal.k0_representation_audit_core import (
        knn_neighbor_indices_blockwise,
    )

    z = np.asarray(z, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    targets = np.asarray(targets, dtype=np.float64)
    hanchan_ids = np.asarray(hanchan_ids, dtype=np.int64)
    n = z.shape[0]
    neighbor_indices = knn_neighbor_indices_blockwise(z, z, hanchan_ids, hanchan_ids, k=k)
    h_state_values: list[float] = []
    h_state_action_values: list[float] = []
    ig_values: list[float] = []
    eligible = 0
    for i in range(n):
        result = action_credit_stats(
            int(actions[i]),
            masks[i],
            actions[neighbor_indices[i]],
            masks[neighbor_indices[i]],
            targets[neighbor_indices[i]],
            k=k,
        )
        if result["eligible"]:
            eligible += 1
            h_state_values.append(float(result["h_state"]))
            h_state_action_values.append(float(result["h_state_action"]))
            ig_values.append(float(result["action_information_gain"]))
    return {
        "rows": n,
        "eligible_queries": eligible,
        "eligible_query_fraction": float(eligible) / float(n) if n else 0.0,
        "h_state_mean": float(np.mean(h_state_values)) if h_state_values else None,
        "h_state_action_mean": float(np.mean(h_state_action_values)) if h_state_action_values else None,
        "action_information_gain_mean": float(np.mean(ig_values)) if ig_values else None,
    }


def adam_alignment_metrics(
    grad_flat: np.ndarray,
    exp_avg_flat: np.ndarray,
    exp_avg_sq_flat: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, float | None]:
    """Descriptive Adam alignment metrics on flattened vectors."""
    grad_flat = np.asarray(grad_flat, dtype=np.float64)
    exp_avg_flat = np.asarray(exp_avg_flat, dtype=np.float64)
    exp_avg_sq_flat = np.asarray(exp_avg_sq_flat, dtype=np.float64)
    if grad_flat.shape != exp_avg_flat.shape or grad_flat.shape != exp_avg_sq_flat.shape:
        raise ValueError("gradient and Adam state vectors must have equal shape")
    g_norm = float(np.linalg.norm(grad_flat))
    m_norm = float(np.linalg.norm(exp_avg_flat))
    cos_m = float(np.dot(grad_flat, exp_avg_flat) / (g_norm * m_norm)) if g_norm > 0 and m_norm > 0 else None
    denom = exp_avg_flat / (np.sqrt(exp_avg_sq_flat) + eps)
    d_norm = float(np.linalg.norm(denom))
    cos_m_den = float(np.dot(grad_flat, denom) / (g_norm * d_norm)) if g_norm > 0 and d_norm > 0 else None
    return {"cos_g_m": cos_m, "cos_g_m_den": cos_m_den}


def sample_microbatch_rows(
    row_hanchan: np.ndarray,
    batch_size: int = 32,
    n_batches: int = 32,
    seed: int = MICROBATCH_SEED,
) -> np.ndarray:
    """Deterministic hanchan-balanced microbatch row sampling.

    Each batch draws distinct hanchans when possible and then selects one row
    from each selected hanchan.  This is used by the descriptive Adam
    diagnostic to fix sample identities.
    """
    row_hanchan = np.asarray(row_hanchan)
    unique = np.unique(row_hanchan)
    if unique.size == 0:
        raise ValueError("row_hanchan must be non-empty")
    rng = np.random.default_rng(seed)
    out = np.empty((n_batches, batch_size), dtype=np.int64)
    for batch_index in range(n_batches):
        selected = rng.choice(unique, size=batch_size, replace=unique.size < batch_size)
        for j, hanchan in enumerate(selected):
            rows = np.flatnonzero(row_hanchan == hanchan)
            out[batch_index, j] = int(rng.choice(rows))
    return out


# ---------------------------------------------------------------- rehydration gate
def row_rehydration_matches(
    z_stored: np.ndarray,
    z_recomputed: np.ndarray,
    behavior_action_legal: bool,
    target_matches: bool,
    perspective_matches: bool,
    rtol: float = REHYDRATION_RTOL,
    atol: float = REHYDRATION_ATOL,
) -> bool:
    """Frozen row rehydration identity gate for one row."""
    if not behavior_action_legal or not target_matches or not perspective_matches:
        return False
    z_stored = np.asarray(z_stored, dtype=np.float64)
    z_recomputed = np.asarray(z_recomputed, dtype=np.float64)
    if z_stored.shape != z_recomputed.shape:
        return False
    return bool(np.allclose(z_stored, z_recomputed, rtol=rtol, atol=atol, equal_nan=False))


# ---------------------------------------------------------------- verdict
def authoritative_verdict(readout: str, gates_pass: bool, complete: bool) -> dict[str, object]:
    """Gate-aware report verdict.

    Any failed gate or incomplete analysis produces
    ``no_verdict_gates_failed`` with ``authoritative=false``.
    """
    if not gates_pass or not complete or readout == "no_verdict_gates_failed":
        return {"readout": "no_verdict_gates_failed", "authoritative": False}
    return {"readout": readout, "authoritative": True}


def combine_decision_verdict(support_signal: bool, q_gradient_signal: bool) -> str:
    if support_signal and q_gradient_signal:
        return "both_signals"
    if support_signal:
        return "state_action_support_signal"
    if q_gradient_signal:
        return "objective_gradient_shift_signal"
    return "inconclusive"
