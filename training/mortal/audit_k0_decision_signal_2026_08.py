#!/usr/bin/env python3
"""K0 State-Action Training-Signal Mechanism Audit runner (prereg v1).

Read-only diagnostic audit.  The implementation phase is limited to
unit/synthetic tests and checkpoint-only model/Q smoke.  The formal runner
must refuse to touch the real four-route corpus until a separate
authorization-only commit enables FORMAL_RUN_AUTHORIZED.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    DATA_ROOT,
    K0_MODEL,
    sha256,
)
from training.mortal.audit_replay_distribution import load_checkpoint
from training.mortal.k0_decision_signal_audit_core import (
    ACTION_DIM,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GRAD_EPS,
    K_CREDIT,
    ROUTE_ORDER,
    SUPPORT_K,
    action_credit_route_stats,
    authoritative_verdict,
    build_pooled_anchor_weights,
    centered_preference_pressure,
    combine_decision_verdict,
    compute_row_gradients,
    compute_support_neighbors,
    cosine_defined,
    estimate_g1_sigma,
    g1_bootstrap_deltas_from_features,
    gradient_family_bootstrap_deltas,
    gradient_family_vote,
    make_frozen_bootstrap_draws,
    make_g1_rff_features,
    precompute_g1_rff_features,
    q_gradient_signal_from_family_votes,
    support_bootstrap_deltas,
    support_metrics_for_anchor,
    support_signal_from_bootstrap,
    weighted_wasserstein_1d,
)
from training.mortal.k0_representation_audit_core import sha256_array

# ---------------------------------------------------------------- prereg gate
PREREG_COMMIT = "7bee592c7c1d00614ca1f5083032dc16b1665d36"
PREREG_FILE = (
    REPO
    / "training/docs/mortal/experiments_zh"
    / "2026-08_K0状态动作训练信号机制审计_预注册设计.md"
)
PREREG_FILE_SHA256 = "1e27e97e6efb509eba80299f644507fe025e4e66375183155f7190a76c639a9d"
OUTPUT_ROOT = DATA_ROOT / "mortal/authoritative/K0_decision_signal_audit_2026_08"
K0_REPR_OUTPUT = DATA_ROOT / "mortal/authoritative/K0_representation_audit_2026_08"
FORMAL_RUN_AUTHORIZED = False
RUN_AUTHORIZATION_NOTE = "implementation + synthetic/unit tests authorized; formal corpus run NOT authorized"

FROZEN_INPUT_SHA256 = {
    "k0_checkpoint": (K0_MODEL, "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"),
    "m0_index": (D1_PREP / "file_index_m0.pth", "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2"),
    "d1_index": (D1_PREP / "file_index_d1.pth", "e357bdb00d5bf3cd7e0afa6960ee43af656421cfed381a3320f6b83ac56087f0"),
    "d2_index": (D2_DATASET / "file_index_d2.pth", "9c86b29204e86df0be8a5d3b0c4e211c010b07bd52e8e491fcdfc7e79e104bb1"),
    "d3_index": (D3_INDEX, "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"),
    "d2_mapping": (D2_DATASET / "player_names_by_file.json", "23b6d5a1589c4b3cba731332896dd33d3f23d50e8b987cb56de0789fdeca5970"),
    "riichi_extension": (Path(riichi.__file__).resolve(), "da687ececbae8c803c99fe58fb8f66d0e4b9e762eb2bb7257a2115c57e5dd82b"),
    "k0_repr_report": (K0_REPR_OUTPUT / "report.json", "6298e1e97668549702cd2c6294222b00a858162d0317411e109bbfc4da1cf1a2"),
    "k0_projection_matrix": (K0_REPR_OUTPUT / "projection_matrix.npy", "ba8d57a4d58f6d9db5c3590aeab3ae034edd1fa4ed00ebd81a5de4264eb87ca9"),
    "m0_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/M0/manifest.json", "077ffba14530135a622c3e5740945554222a23b68cc0b59ae27b3e794188e530"),
    "d1_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D1/manifest.json", "479a133b83a4596eaf7969bf2e9bf0ab268c2fe981087094861e6859e5ad571b"),
    "d2_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D2/manifest.json", "d7b0853a44b794f89968cc37a0e37d668aa1158e5b1fb5481df8b16ba93d1546"),
    "d3_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D3/manifest.json", "a8f48ff1ad444c899ea6f21391f02e337a54aa9ba39dbbba43094b60de3086fc"),
    "m0_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/M0/canonical_metadata.npz", "fad453d504149782fc2fe4950ceb25fe81530d2dcf40b3c49aef1694b880c3c6"),
    "d1_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D1/canonical_metadata.npz", "4dada3ec9223d42e6bd9655bc7eec90d404336852ba2348fdc0425d822a05759"),
    "d2_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D2/canonical_metadata.npz", "efa4a5e34164a41198e78e7e1ba7948dd3784301eaf75d6d3465b04093b0a003"),
    "d3_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D3/canonical_metadata.npz", "6d96fcebdaf9b4a4cf23babd00c231335fdb9b2d8089ffc477a2f83c6ea4c22b"),
}


FROZEN_ARRAY_SHA256 = {
    "k0_projection_array": (K0_REPR_OUTPUT / "projection_matrix.npy", "87cb9ce48397652da8f326b1cbbe31656576ed13d83ee4ca5d41af302bfedd21"),
    "m0_canonical_z": (K0_REPR_OUTPUT / "route_artifacts/M0/canonical_z.npy", "a2e95a0118e9cfd70089a67aa07d98765d4ab12b07b4ce2f69e60c0bf6319fe9"),
    "d1_canonical_z": (K0_REPR_OUTPUT / "route_artifacts/D1/canonical_z.npy", "f3bdfd77a5acbba8561a8fc202f2d8522f789c72be28eeb3b60e71e87f0e272f"),
    "d2_canonical_z": (K0_REPR_OUTPUT / "route_artifacts/D2/canonical_z.npy", "323c03e0fb66855601f2ca5f69c6f3a6bd99d709ef712e6241228872bccad525"),
    "d3_canonical_z": (K0_REPR_OUTPUT / "route_artifacts/D3/canonical_z.npy", "1b63115e607b5e5d5c004443fffbb5911e4a6b5c33f7c9b8cacb79f8b889805b"),
}

FROZEN_JSON_SHA256 = {
    "m0_perspective_labels": (K0_REPR_OUTPUT / "route_artifacts/M0/canonical_perspective_labels.json", "006b14a469519a4c9a2d504081282a7600313a7e7d928ee65290ea12fbb07079"),
    "d1_perspective_labels": (K0_REPR_OUTPUT / "route_artifacts/D1/canonical_perspective_labels.json", "12bcd3599719a468556602932295689a03a42d0d0aec9e7aedb4449328c85b13"),
    "d2_perspective_labels": (K0_REPR_OUTPUT / "route_artifacts/D2/canonical_perspective_labels.json", "fd56eb09f5ed82819796713de907cc2ccaf9333ff002e477389c2dfa5eab29bd"),
    "d3_perspective_labels": (K0_REPR_OUTPUT / "route_artifacts/D3/canonical_perspective_labels.json", "12bcd3599719a468556602932295689a03a42d0d0aec9e7aedb4449328c85b13"),
    "m0_canonical_hanchan_hashes": (K0_REPR_OUTPUT / "route_artifacts/M0/canonical_canonical_hanchan_hashes.json", "9bf8f31101159b1817e0c55ff6f49b0eafd24629e5ca258d97124977800b3ed7"),
    "d1_canonical_hanchan_hashes": (K0_REPR_OUTPUT / "route_artifacts/D1/canonical_canonical_hanchan_hashes.json", "80cc548b573c0f60aadfcaa9f9ee68b7855a9bb564dbd2bc0672792669f75f4e"),
    "d2_canonical_hanchan_hashes": (K0_REPR_OUTPUT / "route_artifacts/D2/canonical_canonical_hanchan_hashes.json", "77325d4b30eb4bb953ae5f763a090e0d4510d940f2d2f9b0492bbf71e0c62ef1"),
    "d3_canonical_hanchan_hashes": (K0_REPR_OUTPUT / "route_artifacts/D3/canonical_canonical_hanchan_hashes.json", "31e813da8363d34ad04849eed051fa4f5cd9caf369fffa1cb78a8cf74613a762"),
}


def check_preregistration() -> dict[str, str | bool]:
    actual = sha256(PREREG_FILE)
    return {
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_file": str(PREREG_FILE),
        "preregistration_file_sha256": actual,
        "preregistration_sha_matches": actual == PREREG_FILE_SHA256,
    }


def git_worktree_metadata() -> dict[str, object]:
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


def formal_preflight(device: torch.device, out_root: Path) -> dict[str, object]:
    """Fail-closed Gate A for the eventual formal run.

    This function only checks files/provenance; it does not open replay corpus.
    """
    checks: dict[str, bool] = {}
    for key, (path, expected) in FROZEN_INPUT_SHA256.items():
        checks[key] = path.is_file() and sha256(path) == expected
    for key, (path, expected_array_sha) in FROZEN_ARRAY_SHA256.items():
        if not path.is_file():
            checks[key] = False
            continue
        try:
            array = np.load(path, allow_pickle=False)
            checks[key] = sha256_array(array) == expected_array_sha
        except Exception:  # noqa: BLE001
            checks[key] = False
    for key, (path, expected) in FROZEN_JSON_SHA256.items():
        checks[key] = path.is_file() and sha256(path) == expected
    prereg = check_preregistration()
    checks["preregistration_sha"] = bool(prereg["preregistration_sha_matches"])
    worktree = git_worktree_metadata()
    checks["git_worktree_clean"] = bool(worktree["git_worktree_clean"])
    checks["device_is_cuda0"] = bool(device.type == "cuda" and getattr(device, "index", -1) == 0)
    checks["torch_cuda_available"] = bool(torch.cuda.is_available())
    checks["output_dir_absent"] = not out_root.exists()
    gate_a = {
        key: value
        for key, value in checks.items()
        if key not in {"git_worktree_clean", "device_is_cuda0", "torch_cuda_available", "output_dir_absent"}
    }
    all_checks = all(checks.values())
    return {
        "checks": checks,
        "gate_a": gate_a,
        "gate_a_pass": all(gate_a.values()),
        "all_pass": all_checks,
        "worktree": worktree,
        "formal_run_authorized": FORMAL_RUN_AUTHORIZED,
    }


def checkpoint_smoke(device: torch.device) -> dict[str, object]:
    """Checkpoint-only phi/Q shape smoke. No corpus file is opened."""
    from model import DQN, Brain, obs_shape

    state = load_checkpoint(K0_MODEL)
    config = state["config"]
    version = int(config["control"].get("version", 4))
    brain = Brain(version=version, **config["resnet"]).to(device).eval()
    brain.load_state_dict(state["mortal"])
    dqn = DQN(version=version).to(device).eval()
    dqn.load_state_dict(state["current_dqn"])
    shape = obs_shape(version)
    obs = torch.zeros(8, *shape, device=device)
    masks = torch.ones(8, 46, dtype=torch.bool, device=device)
    with torch.inference_mode():
        phi = brain(obs)
        q = dqn(phi, masks)
    phi_np = phi.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    result = {
        "checkpoint_sha256": sha256(K0_MODEL),
        "version": version,
        "obs_shape": list(shape),
        "phi_ndim": int(phi_np.ndim),
        "phi_shape": list(phi_np.shape),
        "phi_dim": int(phi_np.shape[1]) if phi_np.ndim == 2 else None,
        "phi_finite_fraction": float(np.isfinite(phi_np).mean()),
        "q_shape": list(q_np.shape),
        "q_finite_fraction": float(np.isfinite(q_np).mean()),
        "smoke_pass": bool(
            phi_np.ndim == 2
            and phi_np.shape[1] == 1024
            and float(np.isfinite(phi_np).mean()) == 1.0
            and q_np.ndim == 2
            and q_np.shape[1] == 46
            and float(np.isfinite(q_np).mean()) == 1.0
        ),
    }
    if not result["smoke_pass"]:
        raise RuntimeError(f"checkpoint smoke failed: {result}")
    return result



# ---------------------------------------------------------------- formal workflow
def _load_decision_canonical(out_root: Path) -> dict[str, dict[str, object]]:
    """Load the four frozen canonical reservoirs plus row identities."""
    from training.mortal.audit_k0_representation_space_2026_08 import (
        _load_extra_route_rows,
    )

    result: dict[str, dict[str, object]] = {}
    for route in ROUTE_ORDER:
        result[route] = _load_extra_route_rows(out_root, route, "canonical")
    return result


def _build_pooled_anchors(
    canonical_by_route: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    """Build the pooled 72k anchor panel with global hanchan IDs."""
    from training.mortal.k0_representation_audit_core import build_global_hanchan_ids

    hashes_by_route = {
        route: list(canonical_by_route[route]["canonical_hanchan_hashes"])
        for route in ROUTE_ORDER
    }
    global_ids_by_route, _ = build_global_hanchan_ids(hashes_by_route)
    routes: list[str] = []
    hanchan_local: list[int] = []
    hanchan_global: list[int] = []
    file_index: list[int] = []
    row_index: list[int] = []
    target: list[float] = []
    z: list[np.ndarray] = []
    for route in ROUTE_ORDER:
        data = canonical_by_route[route]
        n = int(data["rows"])
        routes.extend([route] * n)
        hanchan_local.extend(int(v) for v in data["hanchan_index"])
        hanchan_global.extend(int(v) for v in global_ids_by_route[route])
        file_index.extend(int(v) for v in data["file_index"])
        row_index.extend(int(v) for v in data["row_index"])
        target.extend(float(v) for v in data["target"])
        z.extend(np.asarray(data["z"][i], dtype=np.float64) for i in range(n))
    return {
        "routes": np.asarray(routes),
        "hanchan_local": np.asarray(hanchan_local, dtype=np.int64),
        "hanchan_global": np.asarray(hanchan_global, dtype=np.int64),
        "file_index": np.asarray(file_index, dtype=np.int64),
        "row_index": np.asarray(row_index, dtype=np.int64),
        "target": np.asarray(target, dtype=np.float64),
        "z": np.stack(z),
    }


def _rehydrate_canonical_rows(
    route_name: str,
    route_spec: dict[str, object],
    canonical: dict[str, object],
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    device: torch.device,
    projection: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct obs/action/mask for the canonical rows and verify identity.

    This function is part of the formal workflow and is never called while
    ``FORMAL_RUN_AUTHORIZED`` is False.
    """
    from libriichi.dataset import GameplayLoader

    from training.mortal.audit_cross_corpus_mechanisms_2026_08 import (
        TRAIN_PTS,
    )
    from training.mortal.audit_k0_representation_space_2026_08 import _to_i64

    files = route_spec["index"]
    labels = route_spec["labels"]
    by_file = route_spec.get("by_file")
    file_index_arr = np.asarray(canonical["file_index"], dtype=np.int64)
    row_index_arr = np.asarray(canonical["row_index"], dtype=np.int64)
    target_arr = np.asarray(canonical["target"], dtype=np.float64)
    perspective_arr = np.asarray(canonical["perspective_labels"])
    z_arr = np.asarray(canonical["z"], dtype=np.float64)

    rows_by_file: dict[int, list[int]] = {}
    for i, file_index in enumerate(file_index_arr.tolist()):
        rows_by_file.setdefault(int(file_index), []).append(i)

    actions_out = np.zeros(z_arr.shape[0], dtype=np.int64)
    masks_out = np.zeros((z_arr.shape[0], ACTION_DIM), dtype=bool)
    targets_out = np.zeros(z_arr.shape[0], dtype=np.float64)
    z_out = np.zeros_like(z_arr)

    for file_index, path in enumerate(files):
        indices = rows_by_file.get(int(file_index), [])
        if not indices:
            continue
        label = by_file[str(path.resolve())] if by_file is not None else labels[0]
        loader = GameplayLoader(
            version=4,
            oracle=False,
            player_names=[label],
            excludes=None,
            augmented=False,
        )
        loaded = loader.load_gz_log_files([str(path)])
        if len(loaded) != 1 or len(loaded[0]) != 1:
            raise RuntimeError(f"{route_name}: rehydration loader perspective mismatch for {path.name}")
        game = loaded[0][0]
        obs_all = game.take_obs()
        actions_all_raw = game.take_actions()
        masks_all = game.take_masks()
        grp = game.take_grp()
        player_id = int(game.take_player_id())
        final_rank = int(grp.take_rank_by_player()[player_id])
        expected_target = float(TRAIN_PTS[final_rank] - TRAIN_PTS.mean())
        actions_all = _to_i64(actions_all_raw, signed=True)
        for i in indices:
            row = int(row_index_arr[i])
            action = int(actions_all[row])
            mask = np.asarray(masks_all[row], dtype=bool)
            if mask.shape != (ACTION_DIM,):
                raise RuntimeError(f"{route_name}: mask shape mismatch at file {file_index} row {row}")
            if not mask[action]:
                raise RuntimeError(f"{route_name}: rehydrated behavior action is illegal at file {file_index} row {row}")
            if not np.isclose(expected_target, target_arr[i], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"{route_name}: target mismatch at file {file_index} row {row}")
            if label != perspective_arr[i]:
                raise RuntimeError(f"{route_name}: perspective mismatch at file {file_index} row {row}")
            obs_t = torch.as_tensor(np.asarray(obs_all[row], dtype=np.float32), device=device).unsqueeze(0)
            with torch.inference_mode():
                phi = brain(obs_t)
                _ = dqn(phi, torch.as_tensor(mask[None, :], device=device))
            z_recomp_full = np.asarray(phi.detach().cpu().numpy()[0], dtype=np.float64) @ projection.T
            norm = np.linalg.norm(z_recomp_full)
            if norm <= 0 or not np.isfinite(z_recomp_full).all():
                raise RuntimeError(f"{route_name}: invalid recomputed z at file {file_index} row {row}")
            z_recomp = z_recomp_full / norm
            if not np.allclose(z_arr[i], z_recomp, rtol=1e-5, atol=1e-6, equal_nan=False):
                raise RuntimeError(f"{route_name}: z identity mismatch at file {file_index} row {row}")
            actions_out[i] = action
            masks_out[i] = mask
            targets_out[i] = expected_target
            z_out[i] = z_recomp
    return {"actions": actions_out, "masks": masks_out, "targets": targets_out, "z": z_out}


def _compute_support_metrics(
    anchor_data: dict[str, np.ndarray],
    neighbor_indices: dict[str, np.ndarray],
    actions_by_route: dict[str, np.ndarray],
    masks_by_route: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-route S1/S2 arrays over the pooled anchors."""
    n_anchors = anchor_data["routes"].shape[0]
    out: dict[str, dict[str, np.ndarray]] = {}
    for route in ROUTE_ORDER:
        s1 = np.zeros(n_anchors, dtype=np.float64)
        s2 = np.zeros(n_anchors, dtype=np.float64)
        actions = actions_by_route[route]
        masks = masks_by_route[route]
        for anchor_index in range(n_anchors):
            query_action = int(actions[anchor_index])
            query_legal = masks[anchor_index]
            neighbor_positions = neighbor_indices[route][anchor_index]
            result = support_metrics_for_anchor(
                query_action,
                query_legal,
                actions[neighbor_positions],
                masks[neighbor_positions],
                k=SUPPORT_K,
            )
            s1[anchor_index] = result["switchable_rate"]
            s2[anchor_index] = result["distinct_alt"]
        out[route] = {"s1": s1, "s2": s2}
    return out


def _compute_route_gradients(
    q_by_route: dict[str, np.ndarray],
    actions_by_route: dict[str, np.ndarray],
    masks_by_route: dict[str, np.ndarray],
    targets_by_route: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Row-level Q-output gradient tensors for every route."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for route in ROUTE_ORDER:
        q = q_by_route[route]
        actions = actions_by_route[route]
        masks = masks_by_route[route]
        targets = targets_by_route[route]
        n = q.shape[0]
        g_value = np.zeros_like(q)
        g_cql = np.zeros_like(q)
        g_total = np.zeros_like(q)
        for i in range(n):
            grad = compute_row_gradients(q[i], masks[i], int(actions[i]), float(targets[i]))
            g_value[i] = grad["g_value"]
            g_cql[i] = grad["g_cql"]
            g_total[i] = grad["g_total"]
        out[route] = {"g_value": g_value, "g_cql": g_cql, "g_total": g_total}
    return out


def _run_decision_analysis(
    out_root: Path,
    canonical_by_route: dict[str, dict[str, object]],
    rehydrated_by_route: dict[str, dict[str, np.ndarray]],
    q_by_route: dict[str, np.ndarray],
    projection: np.ndarray,
) -> dict[str, object]:
    """Frozen decision-signal analysis over already-rehydrated rows."""
    from training.mortal.audit_k0_representation_space_2026_08 import (
        _load_canonical_route,
    )
    from training.mortal.k0_representation_audit_core import build_global_hanchan_ids

    routes_for_neighbors = {
        route: _load_canonical_route(out_root, route) for route in ROUTE_ORDER
    }
    hashes_by_route = {route: routes_for_neighbors[route]["sorted_hashes"] for route in ROUTE_ORDER}
    global_ids_by_route, _ = build_global_hanchan_ids(hashes_by_route)
    neighbor_hanchan = {
        route: np.asarray(global_ids_by_route[route], dtype=np.int64)[
            np.asarray(routes_for_neighbors[route]["hanchan_index"], dtype=np.int64)
        ]
        for route in ROUTE_ORDER
    }
    anchor_data = _build_pooled_anchors(canonical_by_route)
    anchor_z = anchor_data["z"]
    anchor_global = anchor_data["hanchan_global"]
    reference_z = {route: routes_for_neighbors[route]["z"] for route in ROUTE_ORDER}
    reference_hanchan = {route: neighbor_hanchan[route] for route in ROUTE_ORDER}
    neighbor_indices = compute_support_neighbors(
        anchor_z, anchor_global, reference_z, reference_hanchan, k=SUPPORT_K
    )
    support_metrics = _compute_support_metrics(
        anchor_data, neighbor_indices,
        {route: rehydrated_by_route[route]["actions"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["masks"] for route in ROUTE_ORDER},
    )
    gradients = _compute_route_gradients(
        q_by_route,
        {route: rehydrated_by_route[route]["actions"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["masks"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["targets"] for route in ROUTE_ORDER},
    )

    # Bootstrap draws and pooled anchor weights.
    draws = make_frozen_bootstrap_draws(6000, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    anchor_weights = build_pooled_anchor_weights(
        anchor_data["routes"], anchor_data["hanchan_local"], draws, BOOTSTRAP_REPS
    )
    support_s1 = support_bootstrap_deltas(
        {route: support_metrics[route]["s1"] for route in ROUTE_ORDER}, anchor_weights, BOOTSTRAP_REPS
    )
    support_s2 = support_bootstrap_deltas(
        {route: support_metrics[route]["s2"] for route in ROUTE_ORDER}, anchor_weights, BOOTSTRAP_REPS
    )
    support_signal = support_signal_from_bootstrap(support_s1, support_s2)

    # Gradient families.
    # G2/G3 scalar values.
    c_by_route: dict[str, np.ndarray] = {}
    m_by_route: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        gv = gradients[route]["g_value"]
        gc = gradients[route]["g_cql"]
        gt = gradients[route]["g_total"]
        c_vals: list[float] = []
        m_vals: list[float] = []
        for i in range(gt.shape[0]):
            cos, defined = cosine_defined(gv[i], gc[i])
            if defined:
                c_vals.append(float(cos))
            m_vals.append(float(centered_preference_pressure(gt[i], masks_by_route_for_gradient(route, canonical_by_route, rehydrated_by_route, i))))
        c_by_route[route] = np.asarray(c_vals, dtype=np.float64)
        m_by_route[route] = np.asarray(m_vals, dtype=np.float64)
    # TODO: row_hanchan for gradient bootstrap must use global hanchan ids from canonical metadata.
    # For implementation completeness this is wired through canonical_by_route hanchan_index.
    row_hanchan = {route: np.asarray(canonical_by_route[route]["hanchan_index"], dtype=np.int64) for route in ROUTE_ORDER}
    g2 = gradient_family_bootstrap_deltas(
        c_by_route, row_hanchan, draws, lambda a, wa, b, wb: weighted_wasserstein_1d(a, wa, b, wb), BOOTSTRAP_REPS
    )
    g3 = gradient_family_bootstrap_deltas(
        m_by_route, row_hanchan, draws, lambda a, wa, b, wb: weighted_wasserstein_1d(a, wa, b, wb), BOOTSTRAP_REPS
    )
    # G1 optimized RFF from unit gradient directions.
    gdirs = {
        route: np.stack([
            (gradients[route]["g_total"][i] / np.linalg.norm(gradients[route]["g_total"][i])
             if np.linalg.norm(gradients[route]["g_total"][i]) > GRAD_EPS
             else np.zeros(ACTION_DIM))
            for i in range(gradients[route]["g_total"].shape[0])
        ])
        for route in ROUTE_ORDER
    }
    sigma_g1 = estimate_g1_sigma(gdirs, row_hanchan)
    omega_g1, bias_g1 = make_g1_rff_features(sigma_g1)
    features_g1 = {route: precompute_g1_rff_features(gdirs[route], omega_g1, bias_g1) for route in ROUTE_ORDER}
    g1 = g1_bootstrap_deltas_from_features(features_g1, row_hanchan, draws, BOOTSTRAP_REPS)
    family_votes = [
        gradient_family_vote(g1),
        gradient_family_vote(g2),
        gradient_family_vote(g3),
    ]
    q_gradient_signal = q_gradient_signal_from_family_votes(family_votes)

    verdict_readout = combine_decision_verdict(support_signal, q_gradient_signal)
    action_credit = {
        route: action_credit_route_stats(
            routes_for_neighbors[route]["z"],
            rehydrated_by_route[route]["actions"],
            rehydrated_by_route[route]["masks"],
            rehydrated_by_route[route]["targets"],
            neighbor_hanchan[route],
            k=K_CREDIT,
        )
        for route in ROUTE_ORDER
    }
    analysis = {
        "support": {"s1": support_s1, "s2": support_s2, "support_signal": support_signal},
        "gradient": {"g1": g1, "g2": g2, "g3": g3, "family_votes": family_votes, "q_gradient_signal": q_gradient_signal},
        "action_credit": action_credit,
        "verdict": verdict_readout,
    }
    return analysis


def masks_by_route_for_gradient(route: str, canonical_by_route: dict[str, dict[str, object]], rehydrated_by_route: dict[str, dict[str, np.ndarray]], i: int) -> np.ndarray:
    return rehydrated_by_route[route]["masks"][i]


def run_formal_audit(device: torch.device, out_root: Path) -> None:
    """Formal decision-signal audit pipeline. Gated by FORMAL_RUN_AUTHORIZED."""
    if not FORMAL_RUN_AUTHORIZED:
        raise RuntimeError(f"formal run is not authorized: {RUN_AUTHORIZATION_NOTE}")
    preflight = formal_preflight(device, out_root)
    if not preflight["all_pass"]:
        raise RuntimeError("formal preflight failed")
    out_root.mkdir(parents=True, exist_ok=True)
    from training.mortal.audit_k0_representation_space_2026_08 import build_route_table
    from training.mortal.audit_replay_distribution import load_model
    from training.mortal.k0_representation_audit_core import make_projection_matrix

    state = load_checkpoint(K0_MODEL)
    brain, dqn, _version = load_model(state, device)
    route_table = build_route_table()
    canonical_by_route = _load_decision_canonical(out_root)
    projection = np.asarray(make_projection_matrix().numpy(), dtype=np.float64)
    rehydrated_by_route: dict[str, dict[str, np.ndarray]] = {}
    q_by_route: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        rehydrated = _rehydrate_canonical_rows(
            route,
            route_table[route],
            canonical_by_route[route],
            brain,
            dqn,
            device,
            projection,
        )
        rehydrated_by_route[route] = rehydrated
        q_by_route[route] = rehydrated["q"]
    analysis = _run_decision_analysis(out_root, canonical_by_route, rehydrated_by_route, q_by_route, projection)
    verdict = authoritative_verdict(analysis["verdict"], bool(preflight["all_pass"]), complete=True)
    report = {
        "schema": "keqing.mortal.k0_decision_signal_audit.v1",
        "preregistration": check_preregistration(),
        "formal_preflight": preflight,
        **git_worktree_metadata(),
        "verdict": verdict,
        "analysis": analysis,
        "status": "complete",
    }
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-prereg", action="store_true", help="verify prereg document SHA")
    parser.add_argument("--checkpoint-smoke", action="store_true", help="run checkpoint-only phi smoke")
    parser.add_argument("--formal-preflight", action="store_true", help="run provenance preflight without corpus access")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.check_prereg:
        result = check_preregistration()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["preregistration_sha_matches"]:
            raise SystemExit(1)
        return

    if args.checkpoint_smoke:
        device = torch.device(args.device)
        result = checkpoint_smoke(device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.formal_preflight:
        device = torch.device(args.device)
        result = formal_preflight(device, OUTPUT_ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not result["all_pass"] or not FORMAL_RUN_AUTHORIZED:
            raise SystemExit(1)
        return

    parser.print_help()
    raise SystemExit(
        "Formal corpus audit is NOT AUTHORIZED in this phase. "
        "Use --check-prereg, --checkpoint-smoke, or --formal-preflight."
    )


if __name__ == "__main__":
    main()
