#!/usr/bin/env python3
"""Pure analysis core for K0 representation-space audit (prereg v1).

This module implements only the frozen numeric protocol from:

    training/docs/mortal/experiments_zh/
    2026-08_K0表示空间机制审计_预注册设计.md

It deliberately contains no loader, model, corpus, or report code.
All functions are deterministic for a fixed seed and fixed input ordering.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------- frozen knobs
PHI_DIM = 1024
PROJECTION_DIM = 256
RFF_DIM = 2048
KNN_K = 16
RESERVOIR_ROWS_PER_HANCHAN = 3
SW_QUANTILE_COUNT = 4096
SW_DIRECTION_COUNT = 256
PAIR_COUNT_PER_ROUTE = 65536
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260817
PERMUTATION_REPS = 999
PERMUTATION_SEED = 20260818
PAIR_SEED = 20260819
RFF_SEED = 20260820
PROJECTION_SEED = 20260815
SW_DIRECTION_SEED = 20260816
CREDIT_ALPHA = 0.05
EXPOSURE_PROXY_SIZE = 18000
EXPOSURE_PROXY_SEED = 20260821
FALLBACK_SIGMA = 0.1
ROUTE_ORDER = ("M0", "D1", "D2", "D3")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def percentile(values: np.ndarray, q: float) -> float:
    """Frozen quantile method: numpy linear interpolation (same as R default type 7)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("percentile requires a non-empty array")
    return float(np.quantile(values, q, method="linear"))


def weighted_quantile_values(
    values: np.ndarray,
    weights: np.ndarray,
    n_quantiles: int = SW_QUANTILE_COUNT,
) -> np.ndarray:
    """Deterministic inverse-CDF over weighted empirical distribution.

    The CDF is the piecewise-linear CDF through sorted observations; duplicate
    observations are allowed.  Quantile positions are the n_quantiles cell
    midpoints ``(k + 0.5) / n_quantiles``.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be 1-D arrays of equal length")
    if values.size == 0:
        raise ValueError("weighted_quantile_values requires at least one value")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    order = np.argsort(values, kind="stable")
    return weighted_quantile_values_from_sorted(values[order], weights[order] / total, n_quantiles)


def weighted_quantile_values_from_sorted(
    sorted_values: np.ndarray,
    sorted_weights: np.ndarray,
    n_quantiles: int = SW_QUANTILE_COUNT,
) -> np.ndarray:
    """Quantile inversion for already sorted values and normalized weights."""
    sorted_values = np.asarray(sorted_values, dtype=np.float64)
    sorted_weights = np.asarray(sorted_weights, dtype=np.float64)
    if sorted_values.ndim != 1 or sorted_weights.shape != sorted_values.shape:
        raise ValueError("sorted values and weights must be 1-D arrays of equal length")
    if sorted_values.size == 0:
        raise ValueError("weighted_quantile_values_from_sorted requires at least one value")
    if np.any(sorted_weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(sorted_weights.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    sorted_weights = sorted_weights / total
    cumulative = np.cumsum(sorted_weights)
    n = sorted_values.size
    lower_cumulative = np.empty_like(cumulative)
    lower_cumulative[0] = 0.0
    lower_cumulative[1:] = cumulative[:-1]

    # Precompute the next positive-weight bin for zero-weight runs.
    next_positive = np.full(n, -1, dtype=np.int64)
    nearest = -1
    for index in range(n - 1, -1, -1):
        if sorted_weights[index] > 0:
            nearest = index
        next_positive[index] = nearest

    query_points = (np.arange(n_quantiles, dtype=np.float64) + 0.5) / float(n_quantiles)
    hits = np.searchsorted(cumulative, query_points, side="left").astype(np.int64)
    output = np.empty(n_quantiles, dtype=np.float64)
    inside = hits < n
    if inside.any():
        hit = hits[inside]
        prev_hit = np.maximum(hit - 1, 0)
        upper = cumulative[hit]
        lower = lower_cumulative[hit]
        value_at_hit = sorted_values[hit]
        value_prev = sorted_values[prev_hit]
        interpolated = value_prev + (value_at_hit - value_prev) * (query_points[inside] - lower) / np.maximum(upper - lower, np.finfo(np.float64).tiny)
        zero_weight = upper <= lower
        later = next_positive[hit]
        fallback = np.where(later >= 0, sorted_values[np.maximum(later, 0)], sorted_values[-1])
        output[inside] = np.where(zero_weight, fallback, interpolated)
    if not inside.all():
        output[~inside] = sorted_values[-1]
    return output


def sliced_wasserstein_weighted(
    z_a: np.ndarray,
    weights_a: np.ndarray,
    z_b: np.ndarray,
    weights_b: np.ndarray,
    directions: np.ndarray,
) -> float:
    """Sliced Wasserstein-1 with frozen weighted-quantile inverse CDF.

    ``z_a`` and ``z_b`` are L2-normalized projected reservoir rows.
    ``directions`` has shape ``[L, d]`` with unit-norm rows.
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if z_a.shape[1] != z_b.shape[1] or z_a.shape[1] != directions.shape[1]:
        raise ValueError("SW dimension mismatch")
    if z_a.ndim != 2 or z_b.ndim != 2 or directions.ndim != 2:
        raise ValueError("SW expects 2-D z_a/z_b/directions")
    projected_a = z_a @ directions.T
    projected_b = z_b @ directions.T
    total = 0.0
    for direction_index in range(directions.shape[0]):
        qa = weighted_quantile_values(projected_a[:, direction_index], weights_a)
        qb = weighted_quantile_values(projected_b[:, direction_index], weights_b)
        total += float(np.abs(qa - qb).sum())
    return total / float(directions.shape[0] * SW_QUANTILE_COUNT)


def make_projection_matrix(
    seed: int = PROJECTION_SEED,
    projected_dim: int = PROJECTION_DIM,
    phi_dim: int = PHI_DIM,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(projected_dim, phi_dim, generator=generator, dtype=dtype)
    matrix = matrix / torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    return matrix


def make_sw_directions(
    seed: int = SW_DIRECTION_SEED,
    dim: int = PROJECTION_DIM,
    count: int = SW_DIRECTION_COUNT,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(count, dim, generator=generator, dtype=dtype)
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    return directions


def sample_cross_hanchan_pairs(
    row_hanchan_ids: np.ndarray,
    n_pairs: int,
    seed: int,
    route_index: int,
) -> np.ndarray:
    """Deterministic cross-hanchan unordered row pairs.

    Rows are grouped by lexicographically sorted ``row_hanchan_ids``.  Each
    route uses ``seed + route_index * 1_000_000``.  Pairs are unique unordered
    hanchan pairs; each pair selects one row uniformly from each hanchan.
    """
    row_hanchan_ids = np.asarray(row_hanchan_ids)
    if row_hanchan_ids.ndim != 1:
        raise ValueError("row_hanchan_ids must be 1-D")
    unique_hanchans = sorted(set(row_hanchan_ids.tolist()))
    if len(unique_hanchans) < 2:
        raise ValueError("cross-hanchan pairs require at least two hanchans")
    rows_by_hanchan: dict[Any, np.ndarray] = {}
    for hanchan in unique_hanchans:
        rows_by_hanchan[hanchan] = np.flatnonzero(row_hanchan_ids == hanchan).astype(np.int64)
    rng = np.random.default_rng(seed + route_index * 1_000_000)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    hanchan_count = len(unique_hanchans)
    while len(pairs) < n_pairs:
        left_pos = int(rng.integers(0, hanchan_count))
        right_pos = int(rng.integers(0, hanchan_count))
        if left_pos == right_pos:
            continue
        key = (min(left_pos, right_pos), max(left_pos, right_pos))
        if key in seen:
            continue
        seen.add(key)
        left_rows = rows_by_hanchan[unique_hanchans[key[0]]]
        right_rows = rows_by_hanchan[unique_hanchans[key[1]]]
        left_row = int(left_rows[rng.integers(0, left_rows.size)])
        right_row = int(right_rows[rng.integers(0, right_rows.size)])
        pairs.append((min(int(left_row), int(right_row)), max(int(left_row), int(right_row))))
    return np.asarray(pairs, dtype=np.int64)


def estimate_sigma_from_pairs(
    z_by_route: dict[str, np.ndarray],
    hanchan_ids_by_route: dict[str, np.ndarray],
    pair_seed: int = PAIR_SEED,
    pairs_per_route: int = PAIR_COUNT_PER_ROUTE,
) -> float:
    distances: list[float] = []
    for route_index, route in enumerate(ROUTE_ORDER):
        pairs = sample_cross_hanchan_pairs(
            hanchan_ids_by_route[route], pairs_per_route, pair_seed, route_index
        )
        z = np.asarray(z_by_route[route], dtype=np.float64)
        pair_distances = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
        distances.extend(float(value) for value in pair_distances)
    median = float(np.median(np.asarray(distances, dtype=np.float64)))
    return FALLBACK_SIGMA if median == 0 else median


def make_rff_features(
    sigma: float,
    seed: int = RFF_SEED,
    input_dim: int = PROJECTION_DIM,
    n_features: int = RFF_DIM,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn(n_features, input_dim, generator=generator, dtype=dtype) / float(sigma)
    bias = torch.rand(n_features, generator=generator, dtype=dtype) * (2.0 * float(np.pi))
    return omega, bias


def rff_mmd2_weighted(
    z_a: np.ndarray,
    weights_a: np.ndarray,
    z_b: np.ndarray,
    weights_b: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
) -> float:
    """Squared RFF-MMD with frozen Gaussian random Fourier features."""
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    weights_a = np.asarray(weights_a, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if weights_a.sum() <= 0 or weights_b.sum() <= 0:
        raise ValueError("RFF-MMD requires positive weight sums")
    weights_a = weights_a / weights_a.sum()
    weights_b = weights_b / weights_b.sum()
    scale = float(np.sqrt(2.0 / omega.shape[0]))
    features_a = scale * np.cos(z_a @ omega.T + bias[None, :])
    features_b = scale * np.cos(z_b @ omega.T + bias[None, :])
    mean_a = weights_a @ features_a
    mean_b = weights_b @ features_b
    return float(np.sum((mean_a - mean_b) ** 2))


def build_global_hanchan_ids(
    sorted_hashes_by_route: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Canonical-hash -> global integer identity shared across all routes.

    Route-local bootstrap indices remain separate; cross-route neighbor
    exclusion MUST use these global IDs so that equal sorted positions in
    different routes are never treated as the same hanchan.
    """
    union = sorted({value for values in sorted_hashes_by_route.values() for value in values})
    hash_to_id = {value: index for index, value in enumerate(union)}
    route_ids = {
        route: np.asarray([hash_to_id[value] for value in sorted_hashes_by_route[route]], dtype=np.int32)
        for route in sorted_hashes_by_route
    }
    return route_ids, hash_to_id


def knn_neighbor_indices_blockwise(
    z_query: np.ndarray,
    z_reference: np.ndarray,
    query_hanchan: np.ndarray,
    reference_hanchan: np.ndarray,
    k: int = KNN_K,
    block_size: int = 512,
) -> np.ndarray:
    """Exact k nearest allowed reference indices for each query row.

    Self and same-hanchan rows are masked before ranking.  Ties are broken by
    smaller reference index because ``np.argpartition``/``np.sort`` is stable
    only with an explicit index key below.
    """
    z_query = np.asarray(z_query, dtype=np.float64)
    z_reference = np.asarray(z_reference, dtype=np.float64)
    query_hanchan = np.asarray(query_hanchan)
    reference_hanchan = np.asarray(reference_hanchan)
    if z_query.ndim != 2 or z_reference.ndim != 2 or z_query.shape[1] != z_reference.shape[1]:
        raise ValueError("kNN dimension mismatch")
    if query_hanchan.shape != (z_query.shape[0],) or reference_hanchan.shape != (z_reference.shape[0],):
        raise ValueError("hanchan id length mismatch")
    if k <= 0 or k >= z_reference.shape[0]:
        raise ValueError("k must be positive and smaller than reference size")
    output = np.empty((z_query.shape[0], k), dtype=np.int64)
    reference_index = np.arange(z_reference.shape[0], dtype=np.int64)
    for start in range(0, z_query.shape[0], block_size):
        stop = min(start + block_size, z_query.shape[0])
        similarity = z_query[start:stop] @ z_reference.T
        distances = 1.0 - similarity
        same_hanchan = query_hanchan[start:stop, None] == reference_hanchan[None, :]
        distances[same_hanchan] = np.inf
        for row in range(start, stop):
            ordered = np.lexsort((reference_index, distances[row - start]))
            output[row] = reference_index[ordered[:k]]
    return output


def _knn_mean_distances_blockwise(
    z_query: np.ndarray,
    z_reference: np.ndarray,
    query_hanchan: np.ndarray,
    reference_hanchan: np.ndarray,
    k: int = KNN_K,
    block_size: int = 512,
) -> np.ndarray:
    """Exact per-query mean cosine distance to k nearest allowed references.

    Self and every row with the same hanchan id are excluded from neighbor
    selection before distance ranking.
    """
    z_query = np.asarray(z_query, dtype=np.float64)
    z_reference = np.asarray(z_reference, dtype=np.float64)
    query_hanchan = np.asarray(query_hanchan)
    reference_hanchan = np.asarray(reference_hanchan)
    if z_query.ndim != 2 or z_reference.ndim != 2 or z_query.shape[1] != z_reference.shape[1]:
        raise ValueError("kNN dimension mismatch")
    if query_hanchan.shape != (z_query.shape[0],) or reference_hanchan.shape != (z_reference.shape[0],):
        raise ValueError("hanchan id length mismatch")
    if k <= 0 or k >= z_reference.shape[0]:
        raise ValueError("k must be positive and smaller than reference size")
    output = np.empty(z_query.shape[0], dtype=np.float64)
    for start in range(0, z_query.shape[0], block_size):
        stop = min(start + block_size, z_query.shape[0])
        similarity = z_query[start:stop] @ z_reference.T
        distances = 1.0 - similarity
        same_hanchan = query_hanchan[start:stop, None] == reference_hanchan[None, :]
        distances[same_hanchan] = np.inf
        # With normalization, two distinct rows can still have cosine distance 0;
        # keep those finite (they are distinct hanchans by mask).
        sorted_distances = np.sort(distances, axis=1)
        output[start:stop] = sorted_distances[:, :k].mean(axis=1)
    return output


def knn_row_stats_and_indicators(
    z_a: np.ndarray,
    z_b: np.ndarray,
    hanchan_a: np.ndarray,
    hanchan_b: np.ndarray,
    k: int = KNN_K,
) -> dict[str, np.ndarray]:
    """Frozen cross-corpus kNN row statistics and missing indicators.

    Returns:
      a_to_b_mean: per-A-row mean distance to its k nearest B rows
      b_to_a_mean: per-B-row mean distance to its k nearest A rows
      b_self_mean: per-B-row mean distance to its k nearest B rows
      a_missing_in_b: indicator (a_to_b_mean > percentile_95(b_self_mean))
      b_missing_in_a: indicator (b_to_a_mean > percentile_95(a_self_mean))
    """
    a_to_b_mean = _knn_mean_distances_blockwise(z_a, z_b, hanchan_a, hanchan_b, k)
    b_to_a_mean = _knn_mean_distances_blockwise(z_b, z_a, hanchan_b, hanchan_a, k)
    b_self_mean = _knn_mean_distances_blockwise(z_b, z_b, hanchan_b, hanchan_b, k)
    a_self_mean = _knn_mean_distances_blockwise(z_a, z_a, hanchan_a, hanchan_a, k)
    threshold_b = percentile(b_self_mean, 0.95)
    threshold_a = percentile(a_self_mean, 0.95)
    return {
        "a_to_b_mean": a_to_b_mean,
        "b_to_a_mean": b_to_a_mean,
        "b_self_mean": b_self_mean,
        "a_self_mean": a_self_mean,
        "a_missing_in_b": (a_to_b_mean > threshold_b).astype(np.int8),
        "b_missing_in_a": (b_to_a_mean > threshold_a).astype(np.int8),
    }


def query_to_reference_density_stats(
    query_z: np.ndarray,
    reference_z: np.ndarray,
    query_hanchan: np.ndarray,
    reference_hanchan: np.ndarray,
    k: int = KNN_K,
) -> dict[str, np.ndarray]:
    """One-way descriptive density stats used by D3 and exposure groups.

    This intentionally does NOT compute query-self or reference->query
    geometry; those are not needed for the prereg readout and would make the
    82k-row D3 descriptive categories unnecessarily quadratic.
    """
    query_to_reference_mean = _knn_mean_distances_blockwise(
        query_z, reference_z, query_hanchan, reference_hanchan, k
    )
    reference_self_mean = _knn_mean_distances_blockwise(
        reference_z, reference_z, reference_hanchan, reference_hanchan, k
    )
    sorted_reference = np.sort(reference_self_mean)
    left = np.searchsorted(sorted_reference, query_to_reference_mean, side="left")
    right = np.searchsorted(sorted_reference, query_to_reference_mean, side="right")
    density_percentile = (left.astype(np.float64) + right.astype(np.float64)) / (2.0 * sorted_reference.size)
    threshold = percentile(reference_self_mean, 0.95)
    return {
        "query_to_reference_mean": query_to_reference_mean,
        "reference_self_mean": reference_self_mean,
        "density_percentile": density_percentile,
        "q95_reference_self": np.asarray([threshold], dtype=np.float64),
        "low_density": (query_to_reference_mean > threshold).astype(np.int8),
    }


def bootstrap_hanchan_draws(
    n_hanchans: int,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Frozen cluster-bootstrap multiplicity draws.

    D1 and D2 share ``d12``.  M0 and D3 each have independent draws.
    """
    if n_hanchans <= 0:
        raise ValueError("n_hanchans must be positive")
    rng = np.random.default_rng(seed)
    m0 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    d12 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    d3 = rng.integers(0, n_hanchans, size=(reps, n_hanchans), dtype=np.int32)
    return {"m0": m0, "d12": d12, "d3": d3}


def hanchan_multiplicity_weights(
    draws: np.ndarray,
    row_hanchan_index: np.ndarray,
) -> np.ndarray:
    """Convert one replicate's hanchan draws to per-row multiplicity weights."""
    counts = np.bincount(draws.astype(np.int64), minlength=row_hanchan_index.max() + 1)
    return counts[row_hanchan_index].astype(np.float64)


def entropy(values: np.ndarray) -> float:
    values = np.asarray(values)
    counts = Counter(int(value) for value in values.tolist())
    total = values.size
    if total == 0:
        return 0.0
    return float(-sum((count / total) * np.log(count / total) for count in counts.values() if count))


def permutation_draws(
    n_hanchans: int,
    reps: int = PERMUTATION_REPS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, np.ndarray]:
    """Frozen hanchan-level target permutation indices.

    ``d12`` is one shared draw used by both D1 and D2 after both routes are
    aligned to sorted canonical hanchan hash order.
    """
    if n_hanchans <= 0:
        raise ValueError("n_hanchans must be positive")
    rng = np.random.default_rng(seed)
    base = np.arange(n_hanchans, dtype=np.int32)
    output: dict[str, np.ndarray] = {}
    for key in ("m0", "d12", "d3"):
        draws = np.empty((reps, n_hanchans), dtype=np.int32)
        for rep in range(reps):
            draws[rep] = rng.permutation(base)
        output[key] = draws
    return output


def permuted_hanchan_targets(
    hanchan_targets: np.ndarray,
    permutation_indices: np.ndarray,
) -> np.ndarray:
    """Apply one cluster permutation to per-hanchan target values."""
    hanchan_targets = np.asarray(hanchan_targets)
    permutation_indices = np.asarray(permutation_indices, dtype=np.int64)
    return hanchan_targets[permutation_indices]


def credit_p_predict(u_obs: float, u_null: np.ndarray) -> float:
    """Frozen p_predict = (1 + count(U_null >= U_obs)) / 1000."""
    u_null = np.asarray(u_null, dtype=np.float64)
    return float((1 + int(np.count_nonzero(u_null >= u_obs))) / float(u_null.size + 1))


def credit_ambiguity_vote(u_obs: float, u_null: np.ndarray, alpha: float = CREDIT_ALPHA) -> dict[str, float | bool]:
    p_predict = credit_p_predict(u_obs, u_null)
    return {
        "p_predict": p_predict,
        "vote": bool(p_predict > alpha),
        "interpretation": "shuffle-like ambiguity / no detectable local predictability",
    }


def combine_verdict(coverage: bool, credit: bool) -> str:
    if coverage and credit:
        return "both_signals"
    if coverage:
        return "latent_coverage_signal"
    if credit:
        return "latent_credit_ambiguity_signal"
    return "inconclusive"


def save_float64_npy(tensor: torch.Tensor, path: Path) -> str:
    array = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)
    np.save(path, array, allow_pickle=False)
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
