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
    rff_mmd2_weighted,
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
    switchable_rate = float(np.count_nonzero(switched)) / float(k)
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


def g1_rff_mmd2_weighted(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
) -> float:
    """Frozen G1 RFF-MMD distance on unit gradient directions."""
    return rff_mmd2_weighted(values_a, weights_a, values_b, weights_b, omega, bias)


def gradient_family_vote(result: dict[str, Any]) -> bool:
    return bool(result["delta1_ci"][0] > 0.0 and result["delta3_ci"][0] > 0.0)


def q_gradient_signal_from_family_votes(votes: list[bool]) -> bool:
    return sum(bool(vote) for vote in votes) >= 2


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

    h_state = _entropy(neighbor_targets[eligible])
    h_state_action = 0.0
    for action in valid_groups:
        mask = eligible & (neighbor_actions == action)
        weight = float(np.count_nonzero(mask)) / float(np.count_nonzero(eligible))
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
def combine_decision_verdict(support_signal: bool, q_gradient_signal: bool) -> str:
    if support_signal and q_gradient_signal:
        return "both_signals"
    if support_signal:
        return "state_action_support_signal"
    if q_gradient_signal:
        return "objective_gradient_shift_signal"
    return "inconclusive"
