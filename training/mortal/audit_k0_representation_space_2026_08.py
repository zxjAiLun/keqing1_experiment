#!/usr/bin/env python3
"""K0 representation-space mechanism audit runner (prereg v1).

Read-only diagnostic audit.  It scans the same frozen M0/D1/D2/D3 indexes as
cross-corpus audit v2, runs only the frozen K0 Brain, stores projected
reservoirs and the D3 event-row projection set, then applies the frozen
four-state adjudication protocol.

Formal corpus scanning is disabled until the implementation review gate is
explicitly opened.  The implementation phase only permits unit/synthetic
tests and ``--checkpoint-smoke``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

from training.mortal.audit_cross_corpus_mechanisms_2026_08 import (
    D1_PREP,
    D2_DATASET,
    D3_INDEX,
    D3_INDEX_SHA,
    DATA_ROOT,
    FULL_ROUTE_NAMES,
    K0_MODEL,
    TRAIN_PTS,
    _canonical_log_hash,
    _log_key,
    _read_log,
    load_d3_events_by_key,
    load_index,
    native_path,
    read_json,
    sha256,
)
from training.mortal.audit_replay_distribution import load_checkpoint
from training.mortal.k0_representation_audit_core import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    KNN_K,
    PAIR_COUNT_PER_ROUTE,
    PAIR_SEED,
    PERMUTATION_REPS,
    PERMUTATION_SEED,
    PHI_DIM,
    PROJECTION_DIM,
    RESERVOIR_ROWS_PER_HANCHAN,
    ROUTE_ORDER,
    bootstrap_hanchan_draws,
    combine_verdict,
    credit_ambiguity_vote,
    entropy,
    estimate_sigma_from_pairs,
    hanchan_multiplicity_weights,
    knn_neighbor_indices_blockwise,
    knn_row_stats_and_indicators,
    make_projection_matrix,
    make_rff_features,
    make_sw_directions,
    percentile,
    permutation_draws,
    permuted_hanchan_targets,
    rff_mmd2_weighted,
    sha256_array,
    sliced_wasserstein_weighted,
)

# ---------------------------------------------------------------- prereg gate
PREREG_COMMIT = "729741ad585eea93e8a2fa02d04020a59ae95716"
PREREG_FILE = (
    REPO
    / "training/docs/mortal/experiments_zh"
    / "2026-08_K0表示空间机制审计_预注册设计.md"
)
PREREG_FILE_SHA256 = "3ebd88b5afc8e7fda28c1c5e61aff5cdf6babf90d082a9096a1529545088b359"
OUTPUT_ROOT = DATA_ROOT / "mortal/authoritative/K0_representation_audit_2026_08"
CROSS_CORPUS_OUTPUT = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08"
    / "diagnostics/cross_corpus_mechanism_audit"
)
FORMAL_RUN_AUTHORIZED = False
RUN_AUTHORIZATION_NOTE = "implementation review gate is not open; formal CUDA run is NOT AUTHORIZED"


def check_preregistration() -> dict[str, Any]:
    actual = sha256(PREREG_FILE)
    ok = actual == PREREG_FILE_SHA256
    return {
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_file": str(PREREG_FILE),
        "preregistration_file_sha256": actual,
        "preregistration_sha_matches": ok,
    }


def git_worktree_metadata() -> dict[str, Any]:
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        cwd=REPO,
    ).strip().splitlines()
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip(),
        "git_worktree_clean": not any(line.strip() for line in git_status),
        "git_worktree_status": git_status,
    }


def _to_i64(raw: Any, signed: bool = False) -> np.ndarray:
    if isinstance(raw, (bytes, bytearray)):
        return np.frombuffer(raw, dtype=np.int8 if signed else np.uint8).astype(np.int64)
    return np.asarray(list(raw), dtype=np.int64)


def _rank_and_gap_vectorized(scores: np.ndarray, player_id: int) -> tuple[np.ndarray, np.ndarray]:
    own = scores[:, player_id]
    rank = np.ones(scores.shape[0], dtype=np.int64)
    for other in range(4):
        if other == player_id:
            continue
        other_score = scores[:, other]
        rank += (other_score > own).astype(np.int64)
        rank += ((other_score == own) & (np.arange(4)[other] < player_id)).astype(np.int64)
    opp_max = np.max(np.stack([scores[:, other] for other in range(4) if other != player_id], axis=1), axis=1)
    return rank, own - opp_max


def build_route_table() -> dict[str, dict[str, Any]]:
    m0_index = load_index(D1_PREP / "file_index_m0.pth")
    d1_index = load_index(D1_PREP / "file_index_d1.pth")
    d2_index = load_index(D2_DATASET / "file_index_d2.pth")
    d3_index = load_index(D3_INDEX)
    if sha256(D3_INDEX) != D3_INDEX_SHA:
        raise ValueError("D3 frozen file index SHA mismatch")
    mapping = json.loads((D2_DATASET / "player_names_by_file.json").read_text(encoding="utf-8"))
    mapping_normalized = {str(native_path(k)): str(v) for k, v in mapping.items()}
    return {
        "M0": {"index": m0_index, "labels": ["ext_mortal"], "by_file": None},
        "D1": {"index": d1_index, "labels": ["K0_70k"], "by_file": None},
        "D2": {
            "index": d2_index,
            "labels": sorted(set(mapping_normalized.values())),
            "by_file": mapping_normalized,
        },
        "D3": {"index": d3_index, "labels": ["K0_70k"], "by_file": None},
    }


def load_brain_only(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int, dict[str, Any]]:
    from model import Brain

    state = load_checkpoint(checkpoint_path)
    config = state["config"]
    version = int(config["control"].get("version", 4))
    brain = Brain(version=version, **config["resnet"]).to(device).eval()
    brain.load_state_dict(state["mortal"])
    return brain, version, config


def checkpoint_smoke(device: torch.device) -> dict[str, Any]:
    """Checkpoint-only phi shape smoke.  No corpus file is opened."""
    from model import obs_shape

    brain, version, _config = load_brain_only(K0_MODEL, device)
    shape = obs_shape(version)
    obs = torch.zeros(16, *shape, device=device)
    with torch.inference_mode():
        phi = brain(obs)
    phi_np = phi.detach().cpu().numpy()
    result = {
        "checkpoint_sha256": sha256(K0_MODEL),
        "version": version,
        "obs_shape": list(shape),
        "phi_ndim": int(phi_np.ndim),
        "phi_shape": list(phi_np.shape),
        "phi_dim": int(phi_np.shape[1]) if phi_np.ndim == 2 else None,
        "phi_finite_fraction": float(np.isfinite(phi_np).mean()),
        "smoke_pass": bool(
            phi_np.ndim == 2
            and phi_np.shape[1] == PHI_DIM
            and float(np.isfinite(phi_np).mean()) == 1.0
        ),
    }
    if not result["smoke_pass"]:
        raise RuntimeError(f"checkpoint smoke failed: {result}")
    return result


def _d3_event_category_by_loader_row(
    path: Path,
    seat: int,
    actions: np.ndarray,
    legal_counts: np.ndarray,
    at_kyoku: np.ndarray,
    file_events: list[dict[str, Any]],
) -> dict[int, str]:
    """Return ``{loader_row_index: event_category}`` exactly-once map.

    Mirrors the v2 gate-F wiring in ``audit_cross_corpus_mechanisms_2026_08``.
    """
    from training.mortal.d3_native_scene import reconstruct_native_scenes
    from training.mortal.d3_production_audit_core import primary_row_flags

    if not file_events:
        return {}
    flags = primary_row_flags(int(action) for action in actions)
    loader_rows = [
        {"action": int(actions[i]), "legal_count": int(legal_counts[i]), "kyoku": int(at_kyoku[i])}
        for i, is_primary in enumerate(flags)
        if is_primary
    ]
    recon = reconstruct_native_scenes(path, seat, loader_rows)
    arena_to_row = {}
    for entry in recon["scenes"]:
        if entry["arena_index"] is not None and entry["loader_row_index"] is not None and entry["arena_consulted"]:
            arena_to_row[(entry["kyoku"], entry["arena_index"])] = entry["loader_row_index"]
    event_by_context = {
        (int(event["kyoku_index"]), int(event["decision_index"])): event for event in file_events
    }
    result: dict[int, str] = {}
    loader_index_by_kyoku: dict[int, int] = {}
    for row_index, kyoku in enumerate(at_kyoku.tolist()):
        loader_index = loader_index_by_kyoku.get(kyoku, 0)
        loader_index_by_kyoku[kyoku] = loader_index + 1
        event = event_by_context.get((int(kyoku), loader_index))
        if event is not None:
            category = (
                "explored"
                if event.get("explored")
                else "hash_rejected"
                if event.get("reason") == "hash_rejected"
                else "budget_exhausted"
            )
            result[int(row_index)] = category
    return result


def _reservoir_row_indices(n_rows: int) -> np.ndarray:
    if n_rows < 3:
        raise ValueError(f"canonical hanchan has fewer than 3 rows: {n_rows}")
    indices = np.asarray(
        [int(((2 * j + 1) * n_rows) // 6) for j in range(RESERVOIR_ROWS_PER_HANCHAN)],
        dtype=np.int64,
    )
    if np.unique(indices).size != RESERVOIR_ROWS_PER_HANCHAN:
        raise ValueError(f"reservoir row indices are not distinct for n_rows={n_rows}")
    return indices


def _update_streaming_moments(
    mean: np.ndarray,
    outer_sum: np.ndarray,
    count: int,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    batch = np.asarray(batch, dtype=np.float64)
    batch_n = batch.shape[0]
    new_count = count + batch_n
    delta = batch - mean[None, :]
    mean = mean + delta.sum(axis=0) / new_count
    outer_sum = outer_sum + batch.T @ batch
    return mean, outer_sum, new_count


def _weighted_proxy_sample(
    file_index_by_row: np.ndarray,
    consumed_per_file: np.ndarray,
    n_samples: int,
    seed: int,
    route_index: int,
) -> np.ndarray:
    weights = np.asarray(consumed_per_file, dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("consumed_per_file weights must have positive sum")
    probabilities = weights[file_index_by_row] / weights.sum()
    rng = np.random.default_rng(seed + route_index * 1_000_000)
    return rng.choice(file_index_by_row.size, size=n_samples, replace=True, p=probabilities).astype(np.int64)


# ---------------------------------------------------------------- formal run
# The formal extraction implementation below is intentionally gated by
# FORMAL_RUN_AUTHORIZED.  It is written against prereg v1, but is not allowed
# to touch the four frozen corpora during the implementation-review phase.
def extract_route_representation(
    route_name: str,
    route_spec: dict[str, Any],
    version: int,
    brain: torch.nn.Module,
    device: torch.device,
    out_root: Path,
    projection: np.ndarray,
    d3_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stream one route, save projected reservoir/event rows, return manifest."""
    from libriichi.dataset import GameplayLoader

    files = route_spec["index"]
    labels = route_spec["labels"]
    by_file = route_spec["by_file"]
    route_dir = out_root / "route_artifacts" / route_name
    route_dir.mkdir(parents=True, exist_ok=True)

    canonical_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    mean_phi = np.zeros(PHI_DIM, dtype=np.float64)
    outer_phi = np.zeros((PHI_DIM, PHI_DIM), dtype=np.float64)
    count_phi = 0
    perspective_count = 0
    malformed: list[str] = []
    hanchan_hashes: list[str] = []
    rows_per_hanchan: list[int] = []

    by_label: dict[str, list[tuple[int, Path]]] = {}
    for file_index, path in enumerate(files):
        label = by_file[str(path.resolve())] if by_file is not None else labels[0]
        by_label.setdefault(label, []).append((file_index, path))

    load_chunk = 8
    for label, tuples in by_label.items():
        loader = GameplayLoader(version=version, oracle=False, player_names=[label], excludes=None, augmented=False)
        for start in range(0, len(tuples), load_chunk):
            chunk = tuples[start : start + load_chunk]
            loaded = loader.load_gz_log_files([str(path) for _, path in chunk])
            if len(loaded) != len(chunk):
                malformed.append(f"{label}: batch size mismatch {len(loaded)} != {len(chunk)}")
                continue
            for (file_index, path), perspectives in zip(chunk, loaded, strict=True):
                if len(perspectives) != 1:
                    malformed.append(f"{path.name}: expected one perspective, got {len(perspectives)}")
                    continue
                try:
                    game = perspectives[0]
                    obs = game.take_obs()
                    actions = game.take_actions()
                    masks = game.take_masks()
                    at_kyoku = game.take_at_kyoku()
                    grp = game.take_grp()
                    player_id = int(game.take_player_id())
                    final_rank = int(grp.take_rank_by_player()[player_id])
                except Exception as exc:  # noqa: BLE001
                    malformed.append(f"{path.name}: {exc}")
                    continue
                n_rows = len(obs)
                if n_rows < 3:
                    malformed.append(f"{path.name}: fewer than 3 loader rows")
                    continue
                if not (len(masks) == len(actions) == len(at_kyoku) == n_rows):
                    malformed.append(f"{path.name}: inconsistent array lengths")
                    continue
                actions_arr = np.asarray(actions, dtype=np.int64) if not isinstance(actions, (bytes, bytearray)) else _to_i64(actions, signed=True)
                masks_arr = np.asarray(masks, dtype=np.bool_)
                legal_counts = masks_arr.sum(axis=1).astype(np.int64)
                at_kyoku_arr = _to_i64(at_kyoku)
                target = float(TRAIN_PTS[final_rank] - TRAIN_PTS.mean())
                canonical_hanchan_hash = _canonical_log_hash(_read_log(path))
                perspective_count += 1
                rows_per_hanchan.append(n_rows)

                event_category_by_row: dict[int, str] = {}
                file_events: list[dict[str, Any]] = []
                if route_name == "D3" and d3_events_by_key is not None:
                    game_seed_key = _log_key(_read_log(path), path)
                    file_events = d3_events_by_key.get(game_seed_key, [])
                    event_category_by_row = _d3_event_category_by_loader_row(
                        path, player_id, actions_arr, legal_counts, at_kyoku_arr, file_events
                    )
                selected_indices = _reservoir_row_indices(n_rows)
                selected_set = {int(value) for value in selected_indices}
                event_set = set(event_category_by_row)
                downstream_set: set[int] = set()
                for row_index in sorted(event_category_by_row):
                    if event_category_by_row[row_index] == "explored":
                        for downstream_index in range(row_index + 1, min(row_index + 9, n_rows)):
                            downstream_set.add(downstream_index)
                union_indices = sorted(selected_set | event_set | downstream_set)
                obs_rows = [np.ascontiguousarray(np.asarray(obs[i], dtype=np.float32)) for i in range(n_rows)]
                all_phi = np.empty((n_rows, PHI_DIM), dtype=np.float64)
                batch_size = 512
                for batch_start in range(0, n_rows, batch_size):
                    batch_stop = min(batch_start + batch_size, n_rows)
                    obs_tensor = torch.as_tensor(np.stack(obs_rows[batch_start:batch_stop]), device=device)
                    with torch.inference_mode():
                        phi = brain(obs_tensor)
                    phi_batch = phi.detach().to("cpu", dtype=torch.float64).numpy()
                    if phi_batch.ndim != 2 or phi_batch.shape[1] != PHI_DIM or not np.isfinite(phi_batch).all():
                        raise ValueError(f"{path.name}: invalid Brain output shape/finiteness")
                    all_phi[batch_start:batch_stop] = phi_batch
                    mean_phi, outer_phi, count_phi = _update_streaming_moments(
                        mean_phi, outer_phi, count_phi, phi_batch
                    )
                projected = all_phi @ projection.T
                projected_norms = np.linalg.norm(projected, axis=1, keepdims=True)
                if not np.isfinite(projected).all() or np.any(projected_norms <= 0):
                    raise ValueError(f"{path.name}: projected representation contains zero/non-finite norm")
                projected = projected / projected_norms
                phi_norms = np.linalg.norm(all_phi, axis=1)
                hanchan_hashes.append(canonical_hanchan_hash)
                hanchan_index = len(hanchan_hashes) - 1
                for row_index in union_indices:
                    row = {
                        "route": route_name,
                        "file_index": int(file_index),
                        "row_index": int(row_index),
                        "hanchan_index": int(hanchan_index),
                        "canonical_hanchan_hash": canonical_hanchan_hash,
                        "perspective_label": label,
                        "target": target,
                        "phi_l2_norm": float(phi_norms[row_index]),
                        "z": projected[row_index].astype(np.float32),
                    }
                    if row_index in selected_set:
                        canonical_rows.append(row)
                    if row_index in event_set:
                        row = dict(row)
                        row["event_category"] = event_category_by_row[row_index]
                        event_rows.append(row)
                    if row_index in downstream_set:
                        row = dict(row)
                        row["downstream_of_explored"] = True
                        downstream_rows.append(row)

    if malformed or perspective_count != len(files) or len(hanchan_hashes) != len(files):
        raise ValueError(f"{route_name}: malformed={malformed[:5]} perspectives={perspective_count}/{len(files)}")

    def _save_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        if not rows:
            return {"rows": 0}
        z_array = np.stack([np.asarray(row["z"], dtype=np.float32) for row in rows])
        metadata = {
            "file_index": np.asarray([row["file_index"] for row in rows], dtype=np.int64),
            "row_index": np.asarray([row["row_index"] for row in rows], dtype=np.int64),
            "hanchan_index": np.asarray([row["hanchan_index"] for row in rows], dtype=np.int32),
            "target": np.asarray([row["target"] for row in rows], dtype=np.float64),
            "phi_l2_norm": np.asarray([row["phi_l2_norm"] for row in rows], dtype=np.float64),
        }
        np.save(route_dir / f"{prefix}_z.npy", z_array, allow_pickle=False)
        np.savez(route_dir / f"{prefix}_metadata.npz", **metadata)
        if prefix == "event":
            np.save(
                route_dir / f"{prefix}_category.npy",
                np.asarray([row.get("event_category", "ordinary") for row in rows]),
                allow_pickle=False,
            )
        return {"rows": len(rows), "z_sha256": sha256_array(z_array)}

    covariance = outer_phi / count_phi - np.outer(mean_phi, mean_phi)
    manifest = {
        "route": route_name,
        "perspectives": perspective_count,
        "rows": count_phi,
        "hanchan_hashes": hanchan_hashes,
        "rows_per_hanchan": rows_per_hanchan,
        "canonical_rows": _save_rows(canonical_rows, "canonical"),
        "event_rows": _save_rows(event_rows, "event"),
        "downstream_rows": _save_rows(downstream_rows, "downstream"),
        "mean_phi": mean_phi,
        "outer_phi": outer_phi,
        "covariance": covariance,
    }
    (route_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_canonical_route(out_root: Path, route_name: str) -> dict[str, Any]:
    route_dir = out_root / "route_artifacts" / route_name
    manifest = read_json(route_dir / "manifest.json")
    z = np.load(route_dir / "canonical_z.npy").astype(np.float64)
    metadata = np.load(route_dir / "canonical_metadata.npz")
    hashes = manifest["hanchan_hashes"]
    sorted_hashes = sorted(hashes)
    hash_to_sorted_index = {value: index for index, value in enumerate(sorted_hashes)}
    row_sorted_hanchan = np.asarray(
        [hash_to_sorted_index[hashes[int(index)]] for index in metadata["hanchan_index"]],
        dtype=np.int32,
    )
    row_original_hanchan = np.asarray(metadata["hanchan_index"], dtype=np.int32)
    return {
        "z": z,
        "target": np.asarray(metadata["target"], dtype=np.float64),
        "hanchan_index": row_sorted_hanchan,
        "original_hanchan_index": row_original_hanchan,
        "sorted_hashes": sorted_hashes,
        "manifest": manifest,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("weights must have positive sum")
    return float((values * weights).sum() / weights.sum())


def _metric_bootstrap_deltas(
    routes: dict[str, dict[str, Any]],
    directions: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
    knn_stats: dict[tuple[str, str], dict[str, np.ndarray]],
    draws: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Frozen cluster-bootstrap metric/delta computation.

    SW and RFF-MMD use hanchan multiplicity weighted empirical distributions.
    kNN/missing-mass reuse precomputed row statistics and never duplicate
    reference rows.
    """
    row_hanchan = {name: routes[name]["hanchan_index"] for name in ROUTE_ORDER}
    families = {
        "sliced_wasserstein": {},
        "rbf_mmd": {},
        "cross_knn_distance": {},
        "latent_missing_mass": {},
    }
    support_d1: list[float] = []
    support_d3: list[float] = []
    for rep in range(BOOTSTRAP_REPS):
        weights = {
            "M0": hanchan_multiplicity_weights(draws["m0"][rep], row_hanchan["M0"]),
            "D1": hanchan_multiplicity_weights(draws["d12"][rep], row_hanchan["D1"]),
            "D2": hanchan_multiplicity_weights(draws["d12"][rep], row_hanchan["D2"]),
            "D3": hanchan_multiplicity_weights(draws["d3"][rep], row_hanchan["D3"]),
        }
        point: dict[str, dict[str, float]] = {}
        for family, family_values in families.items():
            point[family] = {}
            for left, right in (("M0", "D1"), ("M0", "D3"), ("D1", "D2")):
                if family == "sliced_wasserstein":
                    value = sliced_wasserstein_weighted(
                        routes[left]["z"], weights[left], routes[right]["z"], weights[right], directions
                    )
                elif family == "rbf_mmd":
                    value = rff_mmd2_weighted(
                        routes[left]["z"], weights[left], routes[right]["z"], weights[right], omega, bias
                    )
                elif family == "cross_knn_distance":
                    stats = knn_stats[(left, right)]
                    value = 0.5 * (
                        _weighted_mean(stats["a_to_b_mean"], weights[left])
                        + _weighted_mean(stats["b_to_a_mean"], weights[right])
                    )
                else:
                    stats = knn_stats[(left, right)]
                    value = 0.5 * (
                        _weighted_mean(stats["a_missing_in_b"], weights[left])
                        + _weighted_mean(stats["b_missing_in_a"], weights[right])
                    )
                point[family][f"{left}_{right}"] = value
            family_values.setdefault("d_m0_d1", []).append(point[family]["M0_D1"])
            family_values.setdefault("d_m0_d3", []).append(point[family]["M0_D3"])
            family_values.setdefault("d_d1_d2", []).append(point[family]["D1_D2"])
            family_values.setdefault("delta1", []).append(point[family]["M0_D1"] - point[family]["D1_D2"])
            family_values.setdefault("delta3", []).append(point[family]["M0_D3"] - point[family]["D1_D2"])
        stats_m0_d1 = knn_stats[("M0", "D1")]
        stats_m0_d3 = knn_stats[("M0", "D3")]
        support_d1.append(
            _weighted_mean(stats_m0_d1["a_missing_in_b"], weights["M0"])
            - _weighted_mean(stats_m0_d1["b_missing_in_a"], weights["D1"])
        )
        support_d3.append(
            _weighted_mean(stats_m0_d3["a_missing_in_b"], weights["M0"])
            - _weighted_mean(stats_m0_d3["b_missing_in_a"], weights["D3"])
        )
    family_results: dict[str, Any] = {}
    coverage_family_votes = 0
    for family, values in families.items():
        family_results[family] = {
            "point": {
                "d_m0_d1": float(np.mean(values["d_m0_d1"])),
                "d_m0_d3": float(np.mean(values["d_m0_d3"])),
                "d_d1_d2": float(np.mean(values["d_d1_d2"])),
            },
            "delta1_ci95": [percentile(np.asarray(values["delta1"]), 0.025), percentile(np.asarray(values["delta1"]), 0.975)],
            "delta3_ci95": [percentile(np.asarray(values["delta3"]), 0.025), percentile(np.asarray(values["delta3"]), 0.975)],
        }
        if family_results[family]["delta1_ci95"][0] > 0 and family_results[family]["delta3_ci95"][0] > 0:
            coverage_family_votes += 1
    support_results = {
        "D1": {
            "point": float(np.mean(support_d1)),
            "ci95": [percentile(np.asarray(support_d1), 0.025), percentile(np.asarray(support_d1), 0.975)],
        },
        "D3": {
            "point": float(np.mean(support_d3)),
            "ci95": [percentile(np.asarray(support_d3), 0.025), percentile(np.asarray(support_d3), 0.975)],
        },
    }
    coverage_signal = bool(
        coverage_family_votes >= 3
        and support_results["D1"]["ci95"][0] > 0
        and support_results["D3"]["ci95"][0] > 0
    )
    return {
        "families": family_results,
        "coverage_family_votes": coverage_family_votes,
        "support_results": support_results,
        "coverage_signal": coverage_signal,
    }


def _credit_route_result(
    z: np.ndarray,
    target: np.ndarray,
    row_hanchan: np.ndarray,
    permutation_indices: np.ndarray,
) -> dict[str, Any]:
    """Frozen same-route credit ambiguity test for one route."""
    if np.unique(row_hanchan).size != permutation_indices.shape[1]:
        raise ValueError("hanchan count does not match permutation shape")
    hanchan_index = np.asarray(row_hanchan, dtype=np.int64)
    targets_per_hanchan = np.empty(int(hanchan_index.max()) + 1, dtype=np.float64)
    for hanchan in range(targets_per_hanchan.size):
        rows = np.flatnonzero(hanchan_index == hanchan)
        if rows.size == 0:
            raise ValueError("empty hanchan")
        values = np.unique(target[rows])
        if values.size != 1:
            raise ValueError("reservoir target is not constant within hanchan")
        targets_per_hanchan[hanchan] = values[0]
    neighbors = knn_neighbor_indices_blockwise(z, z, hanchan_index, hanchan_index, KNN_K)
    neighbor_targets = target[neighbors]

    def local_entropy(row_targets: np.ndarray) -> tuple[float, float]:
        row_entropies = np.asarray([entropy(row_targets[index]) for index in range(row_targets.shape[0])])
        return float(row_entropies.mean()), float(entropy(target))

    h_local, h_global = local_entropy(neighbor_targets)
    u_obs = 1.0 - h_local / h_global if h_global > 0 else 0.0
    null_u: list[float] = []
    for rep in range(permutation_indices.shape[0]):
        permuted = permuted_hanchan_targets(targets_per_hanchan, permutation_indices[rep])
        null_target = permuted[hanchan_index]
        null_local, null_global = local_entropy(null_target[neighbors])
        null_u.append(1.0 - null_local / null_global if null_global > 0 else 0.0)
    vote = credit_ambiguity_vote(u_obs, np.asarray(null_u))
    return {
        "h_global": h_global,
        "h_local": h_local,
        "u_obs": u_obs,
        "u_null": null_u,
        **vote,
    }


def analyze_route_artifacts(
    out_root: Path,
    exposure_json_path: Path,
) -> dict[str, Any]:
    """Frozen v1 adjudication over already-extracted canonical reservoirs."""
    routes = {name: _load_canonical_route(out_root, name) for name in ROUTE_ORDER}
    for name in ROUTE_ORDER:
        if len(routes[name]["sorted_hashes"]) != 6000:
            raise ValueError(f"{name}: expected 6000 canonical hanchans")
        if routes[name]["z"].shape != (18000, PROJECTION_DIM):
            raise ValueError(f"{name}: canonical reservoir shape mismatch")
    if routes["D1"]["sorted_hashes"] != routes["D2"]["sorted_hashes"]:
        raise ValueError("D1/D2 sorted canonical hanchan hashes are not identical")

    projection = np.asarray(make_projection_matrix().numpy(), dtype=np.float64)
    directions = np.asarray(make_sw_directions().numpy(), dtype=np.float64)
    sigma = estimate_sigma_from_pairs(
        {name: routes[name]["z"] for name in ROUTE_ORDER},
        {name: routes[name]["hanchan_index"] for name in ROUTE_ORDER},
        pair_seed=PAIR_SEED,
        pairs_per_route=PAIR_COUNT_PER_ROUTE,
    )
    omega_tensor, bias_tensor = make_rff_features(sigma)
    omega = np.asarray(omega_tensor.numpy(), dtype=np.float64)
    bias = np.asarray(bias_tensor.numpy(), dtype=np.float64)
    draws = bootstrap_hanchan_draws(6000, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    knn_stats: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for left, right in (("M0", "D1"), ("M0", "D3"), ("D1", "D2")):
        knn_stats[(left, right)] = knn_row_stats_and_indicators(
            routes[left]["z"],
            routes[right]["z"],
            routes[left]["hanchan_index"],
            routes[right]["hanchan_index"],
            KNN_K,
        )
    coverage = _metric_bootstrap_deltas(routes, directions, omega, bias, knn_stats, draws)
    permutation_draws_all = permutation_draws(6000, PERMUTATION_REPS, PERMUTATION_SEED)
    credit_results = {
        "M0": _credit_route_result(
            routes["M0"]["z"], routes["M0"]["target"], routes["M0"]["hanchan_index"], permutation_draws_all["m0"]
        ),
        "D1": _credit_route_result(
            routes["D1"]["z"], routes["D1"]["target"], routes["D1"]["hanchan_index"], permutation_draws_all["d12"]
        ),
        "D2": _credit_route_result(
            routes["D2"]["z"], routes["D2"]["target"], routes["D2"]["hanchan_index"], permutation_draws_all["d12"]
        ),
        "D3": _credit_route_result(
            routes["D3"]["z"], routes["D3"]["target"], routes["D3"]["hanchan_index"], permutation_draws_all["d3"]
        ),
    }
    credit_signal = bool(
        credit_results["M0"]["vote"]
        and credit_results["D3"]["vote"]
        and (credit_results["D1"]["vote"] or credit_results["D2"]["vote"])
    )
    exposure = read_json(exposure_json_path)
    exposure_artifact = {
        "sha256": sha256(exposure_json_path),
        "routes": {name: exposure[name] for name in ROUTE_ORDER},
    }
    return {
        "projection_sha256": sha256_array(projection),
        "sw_directions_sha256": sha256_array(directions),
        "rff_sigma": sigma,
        "rff_omega_sha256": sha256_array(omega),
        "rff_bias_sha256": sha256_array(bias),
        "coverage": coverage,
        "credit": credit_results,
        "credit_signal": credit_signal,
        "verdict": combine_verdict(coverage["coverage_signal"], credit_signal),
        "exposure_proxy_source": exposure_artifact,
    }


def run_formal_audit(device: torch.device, out_root: Path) -> None:
    if not FORMAL_RUN_AUTHORIZED:
        raise RuntimeError(f"formal run is not authorized: {RUN_AUTHORIZATION_NOTE}")
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict[str, Any] = {
        "schema": "keqing.mortal.k0_representation_space_audit.v1",
        "preregistration": check_preregistration(),
        **git_worktree_metadata(),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_index": 0 if torch.cuda.is_available() else None,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "route_timings_seconds": {},
    }
    route_table = build_route_table()
    brain, version, _ = load_brain_only(K0_MODEL, device)
    projection_tensor = make_projection_matrix()
    projection = projection_tensor.numpy()
    projection_path = out_root / "projection_matrix.npy"
    np.save(projection_path, projection, allow_pickle=False)
    report["projection"] = {"path": str(projection_path), "sha256": sha256_array(projection)}
    d3_events = load_d3_events_by_key() if "D3" in FULL_ROUTE_NAMES else None
    route_manifests: dict[str, Any] = {}
    for route_name in ROUTE_ORDER:
        route_t0 = time.time()
        route_manifests[route_name] = extract_route_representation(
            route_name,
            route_table[route_name],
            version,
            brain,
            device,
            out_root,
            projection,
            d3_events_by_key=d3_events if route_name == "D3" else None,
        )
        report["route_timings_seconds"][route_name] = time.time() - route_t0
    report["route_manifests"] = route_manifests
    report["analysis"] = analyze_route_artifacts(out_root, CROSS_CORPUS_OUTPUT / "training_exposure.json")
    report["status"] = "complete"
    report["elapsed_total_seconds"] = time.time() - t0
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-smoke", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prereg = check_preregistration()
    if not prereg["preregistration_sha_matches"]:
        raise RuntimeError("preregistration document SHA mismatch")
    print(json.dumps({"preregistration": prereg, **git_worktree_metadata()}, ensure_ascii=False, indent=2), flush=True)
    if args.checkpoint_smoke:
        print(json.dumps(checkpoint_smoke(device), ensure_ascii=False, indent=2), flush=True)
        return
    if not FORMAL_RUN_AUTHORIZED:
        print(json.dumps({"status": "implementation_review", "note": RUN_AUTHORIZATION_NOTE}, ensure_ascii=False, indent=2), flush=True)
        return
    run_formal_audit(device, args.output_root)


if __name__ == "__main__":
    main()
