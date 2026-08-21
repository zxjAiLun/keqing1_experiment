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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import riichi
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
    SEEDS,
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
    EXPOSURE_PROXY_SEED,
    EXPOSURE_PROXY_SIZE,
    KNN_K,
    PAIR_COUNT_PER_ROUTE,
    PAIR_SEED,
    PERMUTATION_REPS,
    PERMUTATION_SEED,
    PHI_DIM,
    PROJECTION_DIM,
    RESERVOIR_ROWS_PER_HANCHAN,
    ROUTE_ORDER,
    SW_QUANTILE_COUNT,
    bootstrap_hanchan_draws,
    build_global_hanchan_ids,
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
    query_to_reference_density_stats,
    rff_mmd2_weighted,
    sha256_array,
    sliced_wasserstein_weighted,
    weighted_quantile_values_from_sorted,
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
RUN_AUTHORIZATION_NOTE = (
    "formal representation audit COMPLETE; "
    "formal gate closed to prevent accidental rerun"
)
FROZEN_INPUT_SHA256 = {
    "k0_checkpoint": (K0_MODEL, "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"),
    "m0_index": (D1_PREP / "file_index_m0.pth", "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2"),
    "d1_index": (D1_PREP / "file_index_d1.pth", "e357bdb00d5bf3cd7e0afa6960ee43af656421cfed381a3320f6b83ac56087f0"),
    "d2_index": (D2_DATASET / "file_index_d2.pth", "9c86b29204e86df0be8a5d3b0c4e211c010b07bd52e8e491fcdfc7e79e104bb1"),
    "d3_index": (D3_INDEX, "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"),
    "d2_mapping": (D2_DATASET / "player_names_by_file.json", "23b6d5a1589c4b3cba731332896dd33d3f23d50e8b987cb56de0789fdeca5970"),
    "v2_report": (CROSS_CORPUS_OUTPUT / "cross_corpus_mechanism_audit.json", "57f03bbe5653fc3168341df790e0c3dc7ec87860efcd4803f0ba233476a383e0"),
    "v2_route_cache_manifest": (CROSS_CORPUS_OUTPUT / "route_cache_manifest.json", "cd79b0a6c4cb19e48501a7ecaf762c54b80263aa6db1f0be12999f593a98fa44"),
    "v2_training_exposure": (CROSS_CORPUS_OUTPUT / "training_exposure.json", "430cc9eef8fcf495619c115736a04abd6a1f003c764eb7e06368a22305f540e9"),
    "riichi_extension": (Path(riichi.__file__).resolve(), "da687ececbae8c803c99fe58fb8f66d0e4b9e762eb2bb7257a2115c57e5dd82b"),
}
ROUTE_CACHE_EXPECTED_SHA256 = {
    "M0": "61353d4d2910944bd9b7d02c8ded084268a09d152f98c60f0b20dfd29d869cbb",
    "D1": "2ff175c1bf0e9358475110d453539c002d393b6f9242d5e28c956b079ed1e7be",
    "D2": "22ac7d3ecef85d3e6a2fc4c34c17f9d4ded1972da825d39defb28ae997f01ea6",
    "D3": "f7253d68f6adca241b45763d2e21a932ca718a1d0202d5ba7596fdcc994eca67",
}


def formal_preflight(device: torch.device, out_root: Path) -> dict[str, Any]:
    """Fail-closed v1 Gate A and formal-run environment checks."""
    checks: dict[str, bool] = {}
    for key, (path, expected) in FROZEN_INPUT_SHA256.items():
        checks[key] = path.is_file() and sha256(path) == expected
    for route_name, expected in ROUTE_CACHE_EXPECTED_SHA256.items():
        cache_path = CROSS_CORPUS_OUTPUT / "route_agg_cache" / f"{route_name}.json"
        checks[f"route_cache_{route_name.lower()}"] = cache_path.is_file() and sha256(cache_path) == expected
    worktree = git_worktree_metadata()
    checks["git_worktree_clean"] = bool(worktree["git_worktree_clean"])
    checks["device_is_cuda0"] = bool(device.type == "cuda" and getattr(device, "index", -1) == 0)
    checks["torch_cuda_available"] = bool(torch.cuda.is_available())
    checks["output_dir_absent_or_empty"] = not out_root.exists() or not any(out_root.iterdir())
    gate_a = {
        key: value
        for key, value in checks.items()
        if key not in {"git_worktree_clean", "device_is_cuda0", "torch_cuda_available", "output_dir_absent_or_empty"}
    }
    all_checks = all(checks.values())
    return {
        "checks": checks,
        "gate_a": gate_a,
        "gate_a_pass": all(gate_a.values()),
        "all_pass": all_checks,
        "worktree": worktree,
    }


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
    event_row_map: dict[tuple[int, int], dict[str, Any]] = {}
    for event in file_events:
        context = (int(event["kyoku_index"]), int(event["decision_index"]))
        loader_idx = arena_to_row.get(context)
        if loader_idx is not None:
            key = (int(event["kyoku_index"]), loader_idx)
            if key in event_row_map:
                raise ValueError(f"duplicate D3 event/loader-row mapping in {path}: {key}")
            event_row_map[key] = event
    result: dict[int, str] = {}
    loader_index_by_kyoku: dict[int, int] = {}
    for row_index, kyoku in enumerate(at_kyoku.tolist()):
        loader_index = loader_index_by_kyoku.get(int(kyoku), 0)
        loader_index_by_kyoku[int(kyoku)] = loader_index + 1
        event = event_row_map.pop((int(kyoku), loader_index), None)
        if event is None:
            continue
        category = (
            "explored"
            if event.get("explored")
            else "hash_rejected"
            if event.get("reason") == "hash_rejected"
            else "budget_exhausted"
        )
        if row_index in result:
            raise ValueError(f"duplicate D3 loader-row mapping for {path}")
        result[int(row_index)] = category
    if event_row_map:
        raise ValueError(f"unmapped D3 events remain for {path}: {len(event_row_map)}")
    if len(result) != len(file_events):
        raise ValueError(f"D3 event count mismatch for {path}: mapped={len(result)} events={len(file_events)}")
    return result


def _downstream_relations(
    event_category_by_row: dict[int, str],
    n_rows: int,
) -> list[tuple[int, int, int]]:
    """Per-explored-row downstream relations, without deduplicating shared rows."""
    relations: list[tuple[int, int, int]] = []
    for source_row_index in sorted(event_category_by_row):
        if event_category_by_row[source_row_index] == "explored":
            for offset, downstream_index in enumerate(
                range(source_row_index + 1, min(source_row_index + 9, n_rows)),
                start=1,
            ):
                relations.append((int(source_row_index), int(downstream_index), int(offset)))
    return relations


def _authoritative_verdict(readout: str, preflight_all_pass: bool) -> dict[str, Any]:
    return {
        "readout": readout if preflight_all_pass else "no_verdict_gates_failed",
        "authoritative": bool(preflight_all_pass and readout != "no_verdict_gates_failed"),
    }


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


def _exposure_proxy_probabilities(
    file_index_by_row: np.ndarray,
    consumed_per_file: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(consumed_per_file, dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("consumed_per_file weights must have positive sum")
    row_weights = weights[file_index_by_row]
    if row_weights.sum() <= 0:
        raise ValueError("selected row weights must have positive sum")
    return row_weights / row_weights.sum()


def _weighted_proxy_sample(
    file_index_by_row: np.ndarray,
    consumed_per_file: np.ndarray,
    n_samples: int,
    seed: int,
    route_index: int,
) -> np.ndarray:
    probabilities = _exposure_proxy_probabilities(file_index_by_row, consumed_per_file)
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
                downstream_relations = _downstream_relations(event_category_by_row, n_rows)
                downstream_needed = {downstream_index for _, downstream_index, _ in downstream_relations}
                union_indices = sorted(selected_set | event_set | downstream_needed)
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
                    if route_name == "D3":
                        row["event_category"] = event_category_by_row.get(row_index, "ordinary")
                    if row_index in selected_set:
                        canonical_rows.append(dict(row))
                    if row_index in event_set:
                        event_rows.append(dict(row))
                for source_explored_row_index, downstream_index, offset in downstream_relations:
                    row = {
                        "route": route_name,
                        "file_index": int(file_index),
                        "row_index": int(downstream_index),
                        "hanchan_index": int(hanchan_index),
                        "canonical_hanchan_hash": canonical_hanchan_hash,
                        "perspective_label": label,
                        "target": target,
                        "phi_l2_norm": float(phi_norms[downstream_index]),
                        "z": projected[downstream_index].astype(np.float32),
                        "source_explored_row_index": source_explored_row_index,
                        "downstream_offset": offset,
                    }
                    if route_name == "D3":
                        row["event_category"] = "ordinary"
                    downstream_rows.append(row)

    if malformed or perspective_count != len(files) or len(hanchan_hashes) != len(files):
        raise ValueError(f"{route_name}: malformed={malformed[:5]} perspectives={perspective_count}/{len(files)}")
    if route_name == "D3":
        event_counts = Counter(row["event_category"] for row in event_rows)
        expected_event_counts = {"explored": 27506, "hash_rejected": 82455, "budget_exhausted": 41321}
        if dict(event_counts) != expected_event_counts:
            raise ValueError(f"D3 event accounting mismatch: {dict(event_counts)} != {expected_event_counts}")

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
        if prefix == "downstream":
            metadata["source_explored_row_index"] = np.asarray(
                [row["source_explored_row_index"] for row in rows], dtype=np.int64
            )
            metadata["downstream_offset"] = np.asarray([row["downstream_offset"] for row in rows], dtype=np.int8)
        np.save(route_dir / f"{prefix}_z.npy", z_array, allow_pickle=False)
        np.savez(route_dir / f"{prefix}_metadata.npz", **metadata)
        labels_path = route_dir / f"{prefix}_perspective_labels.json"
        labels_path.write_text(
            json.dumps([row["perspective_label"] for row in rows], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hashes_path = route_dir / f"{prefix}_canonical_hanchan_hashes.json"
        hashes_path.write_text(
            json.dumps([row["canonical_hanchan_hash"] for row in rows], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if "event_category" in rows[0]:
            np.save(
                route_dir / f"{prefix}_category.npy",
                np.asarray([row.get("event_category", "ordinary") for row in rows]),
                allow_pickle=False,
            )
        return {
            "rows": len(rows),
            "z_sha256": sha256_array(z_array),
            "perspective_labels_sha256": sha256(labels_path),
            "canonical_hanchan_hashes_sha256": sha256(hashes_path),
        }

    covariance = outer_phi / count_phi - np.outer(mean_phi, mean_phi)
    moment_artifacts: dict[str, Any] = {}
    for name, array in (("mean_phi", mean_phi), ("outer_phi", outer_phi), ("covariance", covariance)):
        moment_path = route_dir / f"{name}.npy"
        np.save(moment_path, array, allow_pickle=False)
        moment_artifacts[name] = {
            "path": str(moment_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_array(array),
        }
    manifest = {
        "route": route_name,
        "perspectives": perspective_count,
        "rows": count_phi,
        "hanchan_hashes": hanchan_hashes,
        "rows_per_hanchan": rows_per_hanchan,
        "canonical_rows": _save_rows(canonical_rows, "canonical"),
        "event_rows": _save_rows(event_rows, "event"),
        "downstream_rows": _save_rows(downstream_rows, "downstream"),
        "moments": moment_artifacts,
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
        "file_index": np.asarray(metadata["file_index"], dtype=np.int64),
        "hanchan_index": row_sorted_hanchan,
        "original_hanchan_index": row_original_hanchan,
        "sorted_hashes": sorted_hashes,
        "manifest": manifest,
    }


def _load_extra_route_rows(out_root: Path, route_name: str, prefix: str) -> dict[str, Any]:
    route_dir = out_root / "route_artifacts" / route_name
    z_path = route_dir / f"{prefix}_z.npy"
    if not z_path.is_file():
        return {"z": np.empty((0, PROJECTION_DIM), dtype=np.float64), "rows": 0}
    z = np.load(z_path).astype(np.float64)
    metadata = np.load(route_dir / f"{prefix}_metadata.npz")
    labels = json.loads((route_dir / f"{prefix}_perspective_labels.json").read_text(encoding="utf-8"))
    hashes = json.loads((route_dir / f"{prefix}_canonical_hanchan_hashes.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "z": z,
        "target": np.asarray(metadata["target"], dtype=np.float64),
        "file_index": np.asarray(metadata["file_index"], dtype=np.int64),
        "row_index": np.asarray(metadata["row_index"], dtype=np.int64),
        "hanchan_index": np.asarray(metadata["hanchan_index"], dtype=np.int32),
        "perspective_labels": labels,
        "canonical_hanchan_hashes": hashes,
        "rows": int(z.shape[0]),
    }
    category_path = route_dir / f"{prefix}_category.npy"
    if category_path.is_file():
        result["event_category"] = np.load(category_path, allow_pickle=False)
    return result


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
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Frozen cluster-bootstrap metric/delta computation.

    SW and RFF-MMD use hanchan multiplicity weighted empirical distributions.
    kNN/missing-mass reuse precomputed row statistics and never duplicate
    reference rows.
    """
    row_hanchan = {name: routes[name]["hanchan_index"] for name in ROUTE_ORDER}
    n_hanchans = int(max(int(np.max(value)) for value in row_hanchan.values()) + 1)
    if any(int(np.min(value)) != 0 or int(np.max(value)) != n_hanchans - 1 for value in row_hanchan.values()):
        raise ValueError("route-local hanchan indices must cover 0..n_hanchans-1")
    sw_precomputed: dict[str, dict[str, np.ndarray]] = {}
    rff_precomputed: dict[str, dict[str, np.ndarray]] = {}
    for name in ROUTE_ORDER:
        projected = routes[name]["z"] @ directions.T
        sw_precomputed[name] = {
            "projected": projected,
            "order": np.argsort(projected, axis=0, kind="stable"),
        }
        scale = float(np.sqrt(2.0 / omega.shape[0]))
        features = scale * np.cos(routes[name]["z"] @ omega.T + bias[None, :])
        rff_precomputed[name] = {"features": features}
        hanchan_sums = np.zeros((n_hanchans, features.shape[1]), dtype=np.float64)
        for hanchan in range(n_hanchans):
            rows = np.flatnonzero(row_hanchan[name] == hanchan)
            if rows.size != 3:
                raise ValueError(f"{name}: expected 3 canonical rows per hanchan, got {rows.size}")
            hanchan_sums[hanchan] = features[rows].sum(axis=0)
        rff_precomputed[name]["hanchan_sums"] = hanchan_sums
        del rff_precomputed[name]["features"]

    def sw_fast(left: str, right: str, weights_left: np.ndarray, weights_right: np.ndarray) -> float:
        total = 0.0
        left_order = sw_precomputed[left]["order"]
        right_order = sw_precomputed[right]["order"]
        left_projected = sw_precomputed[left]["projected"]
        right_projected = sw_precomputed[right]["projected"]
        left_sum = float(weights_left.sum())
        right_sum = float(weights_right.sum())
        for direction_index in range(directions.shape[0]):
            left_order_col = left_order[:, direction_index]
            right_order_col = right_order[:, direction_index]
            q_left = weighted_quantile_values_from_sorted(
                left_projected[left_order_col, direction_index],
                weights_left[left_order_col] / left_sum,
            )
            q_right = weighted_quantile_values_from_sorted(
                right_projected[right_order_col, direction_index],
                weights_right[right_order_col] / right_sum,
            )
            total += float(np.abs(q_left - q_right).sum())
        return total / float(directions.shape[0] * SW_QUANTILE_COUNT)

    def mmd_fast(left: str, right: str, counts_left: np.ndarray, counts_right: np.ndarray) -> float:
        left_mean = (counts_left @ rff_precomputed[left]["hanchan_sums"]) / (3.0 * counts_left.sum())
        right_mean = (counts_right @ rff_precomputed[right]["hanchan_sums"]) / (3.0 * counts_right.sum())
        return float(np.sum((left_mean - right_mean) ** 2))

    def pair_metric_value(
        family: str,
        left: str,
        right: str,
        weights: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
    ) -> float:
        if family == "sliced_wasserstein":
            return sw_fast(left, right, weights[left], weights[right])
        if family == "rbf_mmd":
            return mmd_fast(left, right, counts[left], counts[right])
        stats = knn_stats[(left, right)]
        if family == "cross_knn_distance":
            return 0.5 * (
                _weighted_mean(stats["a_to_b_mean"], weights[left])
                + _weighted_mean(stats["b_to_a_mean"], weights[right])
            )
        return 0.5 * (
            _weighted_mean(stats["a_missing_in_b"], weights[left])
            + _weighted_mean(stats["b_missing_in_a"], weights[right])
        )

    full_weights = {name: np.ones(routes[name]["z"].shape[0], dtype=np.float64) for name in ROUTE_ORDER}
    full_counts = {name: np.ones(n_hanchans, dtype=np.float64) for name in ROUTE_ORDER}
    full_point: dict[str, dict[str, float]] = {}
    for family in ("sliced_wasserstein", "rbf_mmd", "cross_knn_distance", "latent_missing_mass"):
        full_point[family] = {
            "d_m0_d1": pair_metric_value(family, "M0", "D1", full_weights, full_counts),
            "d_m0_d3": pair_metric_value(family, "M0", "D3", full_weights, full_counts),
            "d_d1_d2": pair_metric_value(family, "D1", "D2", full_weights, full_counts),
        }
    stats_m0_d1_full = knn_stats[("M0", "D1")]
    stats_m0_d3_full = knn_stats[("M0", "D3")]
    full_support = {
        "D1": float(stats_m0_d1_full["a_missing_in_b"].mean() - stats_m0_d1_full["b_missing_in_a"].mean()),
        "D3": float(stats_m0_d3_full["a_missing_in_b"].mean() - stats_m0_d3_full["b_missing_in_a"].mean()),
    }

    families = {
        "sliced_wasserstein": {},
        "rbf_mmd": {},
        "cross_knn_distance": {},
        "latent_missing_mass": {},
    }
    support_d1: list[float] = []
    support_d3: list[float] = []
    for rep in range(reps):
        hanchan_counts = {
            "M0": np.bincount(draws["m0"][rep], minlength=n_hanchans).astype(np.float64),
            "D1": np.bincount(draws["d12"][rep], minlength=n_hanchans).astype(np.float64),
            "D2": np.bincount(draws["d12"][rep], minlength=n_hanchans).astype(np.float64),
            "D3": np.bincount(draws["d3"][rep], minlength=n_hanchans).astype(np.float64),
        }
        weights = {
            name: hanchan_multiplicity_weights(draws[group][rep], row_hanchan[name])
            for name, group in (("M0", "m0"), ("D1", "d12"), ("D2", "d12"), ("D3", "d3"))
        }
        point: dict[str, dict[str, float]] = {}
        for family, family_values in families.items():
            point[family] = {}
            for left, right in (("M0", "D1"), ("M0", "D3"), ("D1", "D2")):
                if family == "sliced_wasserstein":
                    value = sw_fast(left, right, weights[left], weights[right])
                elif family == "rbf_mmd":
                    value = mmd_fast(left, right, hanchan_counts[left], hanchan_counts[right])
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
            "point": full_point[family],
            "delta1_ci95": [percentile(np.asarray(values["delta1"]), 0.025), percentile(np.asarray(values["delta1"]), 0.975)],
            "delta3_ci95": [percentile(np.asarray(values["delta3"]), 0.025), percentile(np.asarray(values["delta3"]), 0.975)],
        }
        if family_results[family]["delta1_ci95"][0] > 0 and family_results[family]["delta3_ci95"][0] > 0:
            coverage_family_votes += 1
    support_results = {
        "D1": {
            "point": full_support["D1"],
            "ci95": [percentile(np.asarray(support_d1), 0.025), percentile(np.asarray(support_d1), 0.975)],
        },
        "D3": {
            "point": full_support["D3"],
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
    neighbor_hanchan: np.ndarray,
    permutation_indices: np.ndarray,
) -> dict[str, Any]:
    """Frozen same-route credit ambiguity test for one route.

    ``row_hanchan`` is the route-local bootstrap cluster index; kNN uses
    ``neighbor_hanchan`` (global canonical-hash identity).
    """
    if np.unique(row_hanchan).size != permutation_indices.shape[1]:
        raise ValueError("hanchan count does not match permutation shape")
    hanchan_index = np.asarray(row_hanchan, dtype=np.int64)
    neighbor_hanchan = np.asarray(neighbor_hanchan)
    targets_per_hanchan = np.empty(int(hanchan_index.max()) + 1, dtype=np.float64)
    for hanchan in range(targets_per_hanchan.size):
        rows = np.flatnonzero(hanchan_index == hanchan)
        if rows.size == 0:
            raise ValueError("empty hanchan")
        values = np.unique(target[rows])
        if values.size != 1:
            raise ValueError("reservoir target is not constant within hanchan")
        targets_per_hanchan[hanchan] = values[0]
    neighbors = knn_neighbor_indices_blockwise(z, z, neighbor_hanchan, neighbor_hanchan, KNN_K)
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


def _descriptive_d3_and_exposure(
    out_root: Path,
    routes: dict[str, dict[str, Any]],
    neighbor_hanchan: dict[str, np.ndarray],
    global_hash_to_id: dict[str, int],
    directions: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
    exposure_json_path: Path,
) -> dict[str, Any]:
    """Prereg groups 3 and 4: descriptive-only, never enter the four-state vote."""
    event_rows = _load_extra_route_rows(out_root, "D3", "event")
    downstream_rows = _load_extra_route_rows(out_root, "D3", "downstream")
    event_global_ids = np.asarray(
        [global_hash_to_id[value] for value in event_rows["canonical_hanchan_hashes"]],
        dtype=np.int64,
    )
    downstream_global_ids = np.asarray(
        [global_hash_to_id[value] for value in downstream_rows["canonical_hanchan_hashes"]],
        dtype=np.int64,
    )

    def distance_summary(query: dict[str, Any], query_global_ids: np.ndarray, label: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for reference_name in ("D1", "M0"):
            reference = routes[reference_name]
            stats = query_to_reference_density_stats(
                query["z"],
                reference["z"],
                query_global_ids,
                neighbor_hanchan[reference_name],
                KNN_K,
            )
            output[reference_name] = {
                "rows": int(query["z"].shape[0]),
                "mean_cross_knn_distance": float(stats["query_to_reference_mean"].mean()),
                "missing_mass_in_reference": float(stats["low_density"].mean()),
                "density_percentile_median": float(percentile(stats["density_percentile"], 0.50)),
                "density_percentile_p90": float(percentile(stats["density_percentile"], 0.90)),
                "low_density_fraction": float(stats["low_density"].mean()),
            }
        output["_row_kind"] = label
        return output

    d3_event_output: dict[str, Any] = {"rows": event_rows["rows"]}
    if event_rows["rows"]:
        categories = np.asarray(event_rows["event_category"])
        for category in ("explored", "hash_rejected", "budget_exhausted"):
            mask = categories == category
            if not mask.any():
                d3_event_output[category] = {"rows": 0}
                continue
            query = {key: value[mask] for key, value in event_rows.items() if isinstance(value, np.ndarray)}
            query["perspective_labels"] = [value for index, value in enumerate(event_rows["perspective_labels"]) if mask[index]]
            query["canonical_hanchan_hashes"] = [value for index, value in enumerate(event_rows["canonical_hanchan_hashes"]) if mask[index]]
            d3_event_output[category] = distance_summary(query, event_global_ids[mask], category)
    downstream_output = distance_summary(downstream_rows, downstream_global_ids, "explored_downstream")

    exposure = read_json(exposure_json_path)
    proxy_index_dir = out_root / "proxy_indices"
    proxy_index_dir.mkdir(parents=True, exist_ok=True)
    exposure_output: dict[str, Any] = {}
    for seed in SEEDS:
        proxies: dict[str, dict[str, np.ndarray]] = {}
        seed_output: dict[str, Any] = {"proxy_indices": {}}
        for route_name in ROUTE_ORDER:
            route = routes[route_name]
            consumed = np.asarray(
                exposure[route_name][str(seed)]["simulation"]["consumed_per_file"],
                dtype=np.float64,
            )
            if consumed.shape[0] != 6000:
                raise ValueError(f"{route_name} seed {seed}: exposure weights length != 6000")
            sample_indices = _weighted_proxy_sample(
                route["file_index"],
                consumed,
                EXPOSURE_PROXY_SIZE,
                EXPOSURE_PROXY_SEED,
                ROUTE_ORDER.index(route_name),
            )
            index_path = proxy_index_dir / f"{route_name}_{seed}.npy"
            np.save(index_path, sample_indices, allow_pickle=False)
            proxies[route_name] = {
                "z": route["z"][sample_indices],
                "hanchan": neighbor_hanchan[route_name][sample_indices],
            }
            seed_output["proxy_indices"][route_name] = {
                "path": str(index_path),
                "n": EXPOSURE_PROXY_SIZE,
                "sha256": sha256(index_path),
            }

        def matrix(
            kind: str,
            proxies: dict[str, dict[str, np.ndarray]] = proxies,
        ) -> dict[str, dict[str, float]]:
            output: dict[str, dict[str, float]] = {name: {} for name in ROUTE_ORDER}
            for left in ROUTE_ORDER:
                for right in ROUTE_ORDER:
                    if kind == "sliced_wasserstein":
                        value = sliced_wasserstein_weighted(
                            proxies[left]["z"], np.ones(proxies[left]["z"].shape[0]),
                            proxies[right]["z"], np.ones(proxies[right]["z"].shape[0]),
                            directions,
                        )
                    elif kind == "rbf_mmd":
                        value = rff_mmd2_weighted(
                            proxies[left]["z"], np.ones(proxies[left]["z"].shape[0]),
                            proxies[right]["z"], np.ones(proxies[right]["z"].shape[0]),
                            omega, bias,
                        )
                    else:
                        left_to_right = query_to_reference_density_stats(
                            proxies[left]["z"], proxies[right]["z"],
                            proxies[left]["hanchan"], proxies[right]["hanchan"], KNN_K,
                        )
                        right_to_left = query_to_reference_density_stats(
                            proxies[right]["z"], proxies[left]["z"],
                            proxies[right]["hanchan"], proxies[left]["hanchan"], KNN_K,
                        )
                        value = 0.5 * (
                            float(left_to_right["query_to_reference_mean"].mean())
                            + float(right_to_left["query_to_reference_mean"].mean())
                        )
                    output[left][right] = value
            return output

        seed_output["matrix_sliced_wasserstein"] = matrix("sliced_wasserstein")
        seed_output["matrix_rbf_mmd"] = matrix("rbf_mmd")
        seed_output["matrix_cross_knn"] = matrix("cross_knn")
        seed_output["within_route_exposure_drift"] = {}
        for route_name in ROUTE_ORDER:
            route = routes[route_name]
            seed_output["within_route_exposure_drift"][route_name] = {
                "sliced_wasserstein": sliced_wasserstein_weighted(
                    route["z"], np.ones(route["z"].shape[0]),
                    proxies[route_name]["z"], np.ones(proxies[route_name]["z"].shape[0]),
                    directions,
                ),
                "rbf_mmd2": rff_mmd2_weighted(
                    route["z"], np.ones(route["z"].shape[0]),
                    proxies[route_name]["z"], np.ones(proxies[route_name]["z"].shape[0]),
                    omega, bias,
                ),
            }
        exposure_output[str(seed)] = seed_output
    return {"d3_event_rows": d3_event_output, "d3_downstream": downstream_output, "exposure_proxy": exposure_output}


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
    global_hanchan_ids_by_route, global_hash_to_id = build_global_hanchan_ids(
        {name: routes[name]["sorted_hashes"] for name in ROUTE_ORDER}
    )
    neighbor_hanchan = {
        name: np.asarray(global_hanchan_ids_by_route[name], dtype=np.int64)[
            np.asarray(routes[name]["hanchan_index"], dtype=np.int64)
        ]
        for name in ROUTE_ORDER
    }

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
            neighbor_hanchan[left],
            neighbor_hanchan[right],
            KNN_K,
        )
    coverage = _metric_bootstrap_deltas(routes, directions, omega, bias, knn_stats, draws)
    permutation_draws_all = permutation_draws(6000, PERMUTATION_REPS, PERMUTATION_SEED)
    credit_results = {
        "M0": _credit_route_result(
            routes["M0"]["z"],
            routes["M0"]["target"],
            routes["M0"]["hanchan_index"],
            neighbor_hanchan["M0"],
            permutation_draws_all["m0"],
        ),
        "D1": _credit_route_result(
            routes["D1"]["z"],
            routes["D1"]["target"],
            routes["D1"]["hanchan_index"],
            neighbor_hanchan["D1"],
            permutation_draws_all["d12"],
        ),
        "D2": _credit_route_result(
            routes["D2"]["z"],
            routes["D2"]["target"],
            routes["D2"]["hanchan_index"],
            neighbor_hanchan["D2"],
            permutation_draws_all["d12"],
        ),
        "D3": _credit_route_result(
            routes["D3"]["z"],
            routes["D3"]["target"],
            routes["D3"]["hanchan_index"],
            neighbor_hanchan["D3"],
            permutation_draws_all["d3"],
        ),
    }
    credit_signal = bool(
        credit_results["M0"]["vote"]
        and credit_results["D3"]["vote"]
        and (credit_results["D1"]["vote"] or credit_results["D2"]["vote"])
    )
    descriptive = _descriptive_d3_and_exposure(
        out_root,
        routes,
        neighbor_hanchan,
        global_hash_to_id,
        directions,
        omega,
        bias,
        exposure_json_path,
    )
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
        "descriptive": descriptive,
    }


def run_formal_audit(device: torch.device, out_root: Path) -> None:
    if not FORMAL_RUN_AUTHORIZED:
        raise RuntimeError(f"formal run is not authorized: {RUN_AUTHORIZATION_NOTE}")
    preflight = formal_preflight(device, out_root)
    if not preflight["all_pass"]:
        failure_report = {
            "schema": "keqing.mortal.k0_representation_space_audit.v1",
            "preregistration": check_preregistration(),
            "formal_preflight": preflight,
            **git_worktree_metadata(),
            "verdict": _authoritative_verdict("no_verdict_gates_failed", False),
        }
        failure_path = out_root.with_suffix(out_root.suffix + ".preflight_failed.json")
        failure_path.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"preflight_failed": True, "report": str(failure_path)}, ensure_ascii=False, indent=2), flush=True)
        raise RuntimeError(f"formal preflight failed: report={failure_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict[str, Any] = {
        "schema": "keqing.mortal.k0_representation_space_audit.v1",
        "preregistration": check_preregistration(),
        "formal_preflight": preflight,
        **git_worktree_metadata(),
        "device": str(device),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_index": 0 if torch.cuda.is_available() else None,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "riichi_extension_path": str(Path(riichi.__file__).resolve()),
        "riichi_extension_sha256": sha256(Path(riichi.__file__).resolve()),
        "route_timings_seconds": {},
        "verdict": _authoritative_verdict("no_verdict_gates_failed", preflight["all_pass"]),
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
    report["verdict"] = _authoritative_verdict(report["analysis"]["verdict"], preflight["all_pass"])
    report["status"] = "complete"
    report["elapsed_total_seconds"] = time.time() - t0
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-smoke", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
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
