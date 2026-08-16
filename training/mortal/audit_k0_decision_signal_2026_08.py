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
    adam_alignment_metrics,
    alternative_action_from_q,
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
    greedy_action_from_q,
    make_frozen_bootstrap_draws,
    make_g1_rff_features,
    precompute_g1_rff_features,
    q_gradient_signal_from_family_votes,
    rff_mmd2_from_features,
    sample_microbatch_rows,
    support_bootstrap_deltas,
    support_metrics_for_anchor,
    support_signal_from_bootstrap,
    weighted_wasserstein_1d,
)
from training.mortal.k0_representation_audit_core import (
    entropy,
    sha256_array,
    sha256_file,
)

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
    # Include illegal actions to verify the Mortal DQN contract: legal finite,
    # illegal -inf.
    masks[:, 30:] = False
    with torch.inference_mode():
        phi = brain(obs)
        q = dqn(phi, masks)
    phi_np = phi.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    legal_finite = bool(np.isfinite(q_np[masks.cpu().numpy()]).all())
    illegal_neginf = bool(np.isneginf(q_np[~masks.cpu().numpy()]).all())
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
        "legal_q_finite": legal_finite,
        "illegal_q_neginf": illegal_neginf,
        "smoke_pass": bool(
            phi_np.ndim == 2
            and phi_np.shape[1] == 1024
            and float(np.isfinite(phi_np).mean()) == 1.0
            and q_np.ndim == 2
            and q_np.shape[1] == 46
            and legal_finite
            and illegal_neginf
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
    q_out = np.zeros((z_arr.shape[0], ACTION_DIM), dtype=np.float64)

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
            mask_t = torch.as_tensor(mask[None, :], device=device)
            with torch.inference_mode():
                phi = brain(obs_t)
                q_np = dqn(phi, mask_t).detach().cpu().numpy()[0].astype(np.float64)
            if q_np.shape != (ACTION_DIM,):
                raise RuntimeError(f"{route_name}: invalid DQN output shape at file {file_index} row {row}")
            if not np.isfinite(q_np[mask]).all():
                raise RuntimeError(f"{route_name}: illegal DQN output at legal coordinates at file {file_index} row {row}")
            if not np.isneginf(q_np[~mask]).all():
                raise RuntimeError(f"{route_name}: DQN output at illegal coordinates is not -inf at file {file_index} row {row}")
            q_out[i] = q_np
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
    return {"actions": actions_out, "masks": masks_out, "targets": targets_out, "z": z_out, "q": q_out}


def _compute_support_metrics(
    anchor_data: dict[str, np.ndarray],
    anchor_actions: np.ndarray,
    anchor_masks: np.ndarray,
    neighbor_indices: dict[str, np.ndarray],
    actions_by_route: dict[str, np.ndarray],
    masks_by_route: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-route S1/S2 arrays over the pooled anchors.

    Query action/legal always come from the pooled anchor row itself; only
    neighbor action/legal come from the per-route reference reservoir.
    """
    n_anchors = anchor_data["routes"].shape[0]
    if anchor_actions.shape != (n_anchors,) or anchor_masks.shape != (n_anchors, ACTION_DIM):
        raise ValueError("anchor_actions/anchor_masks must match pooled anchor count")
    out: dict[str, dict[str, np.ndarray]] = {}
    for route, actions in actions_by_route.items():
        s1 = np.zeros(n_anchors, dtype=np.float64)
        s2 = np.zeros(n_anchors, dtype=np.float64)
        masks = masks_by_route[route]
        k = int(neighbor_indices[route].shape[1])
        for anchor_index in range(n_anchors):
            query_action = int(anchor_actions[anchor_index])
            query_legal = anchor_masks[anchor_index]
            neighbor_positions = neighbor_indices[route][anchor_index]
            result = support_metrics_for_anchor(
                query_action,
                query_legal,
                actions[neighbor_positions],
                masks[neighbor_positions],
                k=k,
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
    repr_artifact_root: Path,
    canonical_by_route: dict[str, dict[str, object]],
    rehydrated_by_route: dict[str, dict[str, np.ndarray]],
    q_by_route: dict[str, np.ndarray],
    projection: np.ndarray,
) -> dict[str, object]:
    """Frozen decision-signal analysis over already-rehydrated rows.

    ``repr_artifact_root`` must be the prior K0 representation audit output
    root, not the new decision-signal output root.
    """
    from training.mortal.audit_k0_representation_space_2026_08 import (
        _load_canonical_route,
    )
    from training.mortal.k0_representation_audit_core import build_global_hanchan_ids

    routes_for_neighbors = {
        route: _load_canonical_route(repr_artifact_root, route) for route in ROUTE_ORDER
    }
    hashes_by_route = {route: routes_for_neighbors[route]["sorted_hashes"] for route in ROUTE_ORDER}
    global_ids_by_route, _ = build_global_hanchan_ids(hashes_by_route)
    neighbor_hanchan = {
        route: np.asarray(global_ids_by_route[route], dtype=np.int64)[
            np.asarray(routes_for_neighbors[route]["hanchan_index"], dtype=np.int64)
        ]
        for route in ROUTE_ORDER
    }

    # Bootstrap identity must be per-route sorted canonical hash index, not
    # original file-order hanchan index.  D1/D2 share the same sorted hash
    # ordering and therefore the same bootstrap cluster indices.
    canonical_for_anchors = {route: dict(canonical_by_route[route]) for route in ROUTE_ORDER}
    for route in ROUTE_ORDER:
        canonical_for_anchors[route]["hanchan_index"] = routes_for_neighbors[route]["hanchan_index"]

    anchor_data = _build_pooled_anchors(canonical_for_anchors)
    anchor_z = anchor_data["z"]
    anchor_global = anchor_data["hanchan_global"]
    reference_z = {route: routes_for_neighbors[route]["z"] for route in ROUTE_ORDER}
    reference_hanchan = {route: neighbor_hanchan[route] for route in ROUTE_ORDER}
    neighbor_indices = compute_support_neighbors(
        anchor_z, anchor_global, reference_z, reference_hanchan, k=SUPPORT_K
    )

    # Pooled anchor action/mask come from the anchor source route, not from the
    # reference route being queried.
    anchor_actions = np.concatenate(
        [rehydrated_by_route[route]["actions"] for route in ROUTE_ORDER]
    ).astype(np.int64)
    anchor_masks = np.concatenate(
        [rehydrated_by_route[route]["masks"] for route in ROUTE_ORDER]
    ).astype(bool)

    support_metrics = _compute_support_metrics(
        anchor_data,
        anchor_actions,
        anchor_masks,
        neighbor_indices,
        {route: rehydrated_by_route[route]["actions"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["masks"] for route in ROUTE_ORDER},
    )
    gradients = _compute_route_gradients(
        q_by_route,
        {route: rehydrated_by_route[route]["actions"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["masks"] for route in ROUTE_ORDER},
        {route: rehydrated_by_route[route]["targets"] for route in ROUTE_ORDER},
    )

    draws = make_frozen_bootstrap_draws(6000, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    anchor_weights = build_pooled_anchor_weights(
        anchor_data["routes"], anchor_data["hanchan_local"], draws, BOOTSTRAP_REPS
    )
    # Local Action Support descriptive: raw 46-action entropy on the same
    # frozen 16-neighbor panels, and behavior agreement on route canonical rows.
    local_entropy_by_route: dict[str, np.ndarray] = {}
    effective_count_by_route: dict[str, np.ndarray] = {}
    for route in ROUTE_ORDER:
        actions = rehydrated_by_route[route]["actions"]
        entropies: list[float] = []
        effective: list[float] = []
        for anchor_index in range(anchor_actions.shape[0]):
            neighbor_positions = neighbor_indices[route][anchor_index]
            neighbor_actions = actions[neighbor_positions]
            h = entropy(neighbor_actions)
            entropies.append(h)
            effective.append(float(np.exp(h)))
        local_entropy_by_route[route] = np.asarray(entropies, dtype=np.float64)
        effective_count_by_route[route] = np.asarray(effective, dtype=np.float64)

    behavior_agreement: dict[str, dict[str, float]] = {}
    for route in ROUTE_ORDER:
        q_route = q_by_route[route]
        masks = rehydrated_by_route[route]["masks"]
        actions = rehydrated_by_route[route]["actions"]
        agreements = 0
        for i in range(q_route.shape[0]):
            legal_indices = np.flatnonzero(masks[i])
            greedy = int(legal_indices[int(np.argmax(q_route[i][legal_indices]))])
            agreements += int(greedy == int(actions[i]))
        behavior_agreement[route] = {
            "count": agreements,
            "rate": float(agreements / q_route.shape[0]) if q_route.shape[0] else 0.0,
        }

    support_s1 = support_bootstrap_deltas(
        {route: support_metrics[route]["s1"] for route in ROUTE_ORDER}, anchor_weights, BOOTSTRAP_REPS
    )
    support_s2 = support_bootstrap_deltas(
        {route: support_metrics[route]["s2"] for route in ROUTE_ORDER}, anchor_weights, BOOTSTRAP_REPS
    )
    support_signal = support_signal_from_bootstrap(support_s1, support_s2)

    # G2/G3 scalar values and G2 defined-row masks.
    c_by_route: dict[str, np.ndarray] = {}
    m_by_route: dict[str, np.ndarray] = {}
    g2_hanchan: dict[str, np.ndarray] = {}
    g3_hanchan: dict[str, np.ndarray] = {}
    cosine_defined_counts: dict[str, int] = {}
    cosine_defined_rates: dict[str, float] = {}
    conflict_rates: dict[str, float] = {}
    zero_gradient_rates: dict[str, float] = {}
    behavior_push_by_route: dict[str, np.ndarray] = {}
    flip_pressure_by_route: dict[str, np.ndarray] = {}
    alt_suppression_by_route: dict[str, np.ndarray] = {}
    alt_suppression_defined_rates: dict[str, float] = {}
    for route in ROUTE_ORDER:
        gv = gradients[route]["g_value"]
        gc = gradients[route]["g_cql"]
        gt = gradients[route]["g_total"]
        masks = rehydrated_by_route[route]["masks"]
        actions = rehydrated_by_route[route]["actions"]
        row_hanchan_route = np.asarray(routes_for_neighbors[route]["hanchan_index"], dtype=np.int64)
        c_vals: list[float] = []
        c_defined: list[bool] = []
        m_vals: list[float] = []
        zero_count = 0
        conflict_count = 0
        behavior_push: list[float] = []
        flip_pressure: list[float] = []
        alt_suppression: list[float] = []
        alt_defined: list[bool] = []
        q_route = q_by_route[route]
        for i in range(gt.shape[0]):
            g_norm = float(np.linalg.norm(gt[i]))
            if g_norm <= GRAD_EPS:
                zero_count += 1
            cos, defined = cosine_defined(gv[i], gc[i])
            c_defined.append(defined)
            if defined:
                c_vals.append(float(cos))
                if cos < 0.0:
                    conflict_count += 1
            m_vals.append(float(centered_preference_pressure(gt[i], masks[i])["m_centered"]))
            a = int(actions[i])
            behavior_push.append(float(-gt[i][a]))
            legal = masks[i]
            legal_indices = np.flatnonzero(legal)
            if legal_indices.size >= 2:
                # Frozen prereg: greedy and alternative are selected from K0 Q,
                # not from the Q-gradient.
                a_greedy = greedy_action_from_q(q_route[i], legal)
                flip_pressure.append(float(-(gt[i][a] - gt[i][a_greedy])))
                alt_indices = legal_indices[legal_indices != a]
                if alt_indices.size:
                    a_alt = alternative_action_from_q(q_route[i], legal, a)
                    alt_suppression.append(float(gt[i][a_alt]))
                    alt_defined.append(True)
                else:
                    alt_suppression.append(float("nan"))
                    alt_defined.append(False)
            else:
                flip_pressure.append(float("nan"))
                alt_suppression.append(float("nan"))
                alt_defined.append(False)
        c_arr = np.asarray(c_vals, dtype=np.float64)
        m_arr = np.asarray(m_vals, dtype=np.float64)
        c_mask = np.asarray(c_defined, dtype=bool)
        c_by_route[route] = c_arr
        m_by_route[route] = m_arr
        g2_hanchan[route] = row_hanchan_route[c_mask]
        g3_hanchan[route] = row_hanchan_route
        cosine_defined_counts[route] = int(c_mask.sum())
        cosine_defined_rates[route] = float(c_mask.mean()) if c_mask.size else 0.0
        conflict_rates[route] = float(conflict_count / c_mask.sum()) if c_mask.sum() else 0.0
        zero_gradient_rates[route] = float(zero_count / gt.shape[0]) if gt.shape[0] else 0.0
        behavior_push_by_route[route] = np.asarray(behavior_push, dtype=np.float64)
        flip_pressure_by_route[route] = np.asarray(flip_pressure, dtype=np.float64)
        alt_suppression_by_route[route] = np.asarray(alt_suppression, dtype=np.float64)
        alt_suppression_defined_rates[route] = float(np.mean(alt_defined)) if alt_defined else 0.0

    row_hanchan = {route: np.asarray(routes_for_neighbors[route]["hanchan_index"], dtype=np.int64) for route in ROUTE_ORDER}
    g2 = gradient_family_bootstrap_deltas(
        c_by_route, g2_hanchan, draws, lambda a, wa, b, wb: weighted_wasserstein_1d(a, wa, b, wb), BOOTSTRAP_REPS
    )
    g3 = gradient_family_bootstrap_deltas(
        m_by_route, g3_hanchan, draws, lambda a, wa, b, wb: weighted_wasserstein_1d(a, wa, b, wb), BOOTSTRAP_REPS
    )

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
    # Full-sample point estimates (P2 provenance convenience).
    s1_point = {route: float(np.mean(support_metrics[route]["s1"])) for route in ROUTE_ORDER}
    s2_point = {route: float(np.mean(support_metrics[route]["s2"])) for route in ROUTE_ORDER}
    support_point = {
        "s1": s1_point,
        "s2": s2_point,
        "view_gap_s1": abs(s1_point["D1"] - s1_point["D2"]),
        "view_gap_s2": abs(s2_point["D1"] - s2_point["D2"]),
        "delta1_s1": (s1_point["M0"] - s1_point["D1"]) - abs(s1_point["D1"] - s1_point["D2"]),
        "delta3_s1": (s1_point["M0"] - s1_point["D3"]) - abs(s1_point["D1"] - s1_point["D2"]),
        "delta1_s2": (s2_point["M0"] - s2_point["D1"]) - abs(s2_point["D1"] - s2_point["D2"]),
        "delta3_s2": (s2_point["M0"] - s2_point["D3"]) - abs(s2_point["D1"] - s2_point["D2"]),
    }

    def _w1_point(values: dict[str, np.ndarray], left: str, right: str) -> float:
        return weighted_wasserstein_1d(values[left], np.ones(values[left].shape[0]), values[right], np.ones(values[right].shape[0]))

    g2_point = {
        "d_m0_d1": _w1_point(c_by_route, "M0", "D1"),
        "d_m0_d3": _w1_point(c_by_route, "M0", "D3"),
        "d_d1_d2": _w1_point(c_by_route, "D1", "D2"),
    }
    g3_point = {
        "d_m0_d1": _w1_point(m_by_route, "M0", "D1"),
        "d_m0_d3": _w1_point(m_by_route, "M0", "D3"),
        "d_d1_d2": _w1_point(m_by_route, "D1", "D2"),
    }
    g1_point = {
        "d_m0_d1": rff_mmd2_from_features(features_g1["M0"], np.ones(features_g1["M0"].shape[0]), features_g1["D1"], np.ones(features_g1["D1"].shape[0])),
        "d_m0_d3": rff_mmd2_from_features(features_g1["M0"], np.ones(features_g1["M0"].shape[0]), features_g1["D3"], np.ones(features_g1["D3"].shape[0])),
        "d_d1_d2": rff_mmd2_from_features(features_g1["D1"], np.ones(features_g1["D1"].shape[0]), features_g1["D2"], np.ones(features_g1["D2"].shape[0])),
    }
    for name, point in (("g1", g1_point), ("g2", g2_point), ("g3", g3_point)):
        point["delta1"] = point["d_m0_d1"] - point["d_d1_d2"]
        point["delta3"] = point["d_m0_d3"] - point["d_d1_d2"]

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
        "support": {
            "s1": support_s1,
            "s2": support_s2,
            "support_signal": support_signal,
            "point": support_point,
            "local_action_entropy": local_entropy_by_route,
            "effective_action_count": effective_count_by_route,
            "behavior_agreement": behavior_agreement,
        },
        "gradient": {
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "point": {"g1": g1_point, "g2": g2_point, "g3": g3_point},
            "family_votes": family_votes,
            "q_gradient_signal": q_gradient_signal,
            "cosine_defined_rows": cosine_defined_counts,
            "cosine_defined_rate": cosine_defined_rates,
            "conflict_rate": conflict_rates,
            "zero_total_gradient_rate": zero_gradient_rates,
            "behavior_push": behavior_push_by_route,
            "flip_pressure": flip_pressure_by_route,
            "alt_suppression": alt_suppression_by_route,
            "alt_suppression_defined_rate": alt_suppression_defined_rates,
        },
        "action_credit": action_credit,
        "verdict": verdict_readout,
    }
    return analysis



def _adam_diagnostic_from_optimizer(
    optimizer: torch.optim.Optimizer,
    gradients: dict[torch.nn.Parameter, torch.Tensor],
) -> dict[str, object]:
    """Descriptive Adam alignment using live optimizer state.

    This intentionally uses ``optimizer.state[param]`` keyed by the live
    Parameter object, avoiding integer-id guessing.
    """
    results: dict[str, object] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        cos_m: list[float] = []
        cos_m_den: list[float] = []
        for param in group["params"]:
            if param not in gradients or param not in optimizer.state:
                continue
            state = optimizer.state[param]
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            grad_flat = gradients[param].detach().float().reshape(-1)
            exp_avg_flat = state["exp_avg"].detach().float().reshape(-1)
            exp_avg_sq_flat = state["exp_avg_sq"].detach().float().reshape(-1)
            metrics = adam_alignment_metrics(
                grad_flat.numpy(), exp_avg_flat.numpy(), exp_avg_sq_flat.numpy()
            )
            if metrics["cos_g_m"] is not None:
                cos_m.append(float(metrics["cos_g_m"]))
            if metrics["cos_g_m_den"] is not None:
                cos_m_den.append(float(metrics["cos_g_m_den"]))
        results[f"group_{group_index}"] = {
            "cos_g_m_mean": float(np.mean(cos_m)) if cos_m else None,
            "cos_g_m_den_mean": float(np.mean(cos_m_den)) if cos_m_den else None,
            "param_count": len(group["params"]),
        }
    return results


def _json_safe_report(value: object, out_root: Path, path_parts: tuple[str, ...] = ()) -> object:
    """Recursively convert analysis arrays into JSON-safe artifact references."""
    if isinstance(value, np.ndarray):
        artifact_dir = out_root / "bootstrap"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        name = "_".join(path_parts) if path_parts else "array"
        path = artifact_dir / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        return {
            "artifact": str(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_file(path),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe_report(item, out_root, path_parts + (str(key),)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_report(item, out_root, path_parts + (str(i),)) for i, item in enumerate(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value



# ---------------------------------------------------------------- Adam formal
def _build_preserved_optimizer(
    state: dict[str, object],
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    aux: torch.nn.Module,
) -> torch.optim.Optimizer:
    """Construct the production optimizer and load frozen preserved Adam state."""
    from training.mortal.preflight_optimizer_ab import make_optimizer, make_scheduler
    from training.run_mortal_dqn_offline import (
        _optimizer_group_metadata,
        _validate_preserved_optimizer,
    )

    config = state["config"]
    optimizer = make_optimizer(config, (brain, dqn, aux))
    # Reproduce production fresh-optimizer recipe before capturing metadata.
    # The scheduler may adjust optimizer param-group metadata (e.g. lr /
    # initial_lr); we only need its construction side effect, not its state.
    _scheduler = make_scheduler(config, optimizer)
    fresh_groups = _optimizer_group_metadata(optimizer)
    if "optimizer" not in state:
        raise RuntimeError("frozen K0 checkpoint has no preserved optimizer state")
    optimizer.load_state_dict(state["optimizer"])
    expected_tensors = sum(1 for module in (brain, dqn, aux) for _ in module.parameters())
    _validate_preserved_optimizer(
        optimizer,
        fresh_groups=fresh_groups,
        expected_parameter_tensors=expected_tensors,
    )
    return optimizer


def _collect_microbatch_records(
    route_name: str,
    route_spec: dict[str, object],
    canonical: dict[str, object],
    sampled_positions: np.ndarray,
    config: dict[str, object],
) -> list[object]:
    """Collect LearnabilityRecord objects for the frozen 32x32 sampled rows."""
    from libriichi.dataset import GameplayLoader

    from training.mortal.audit_cross_corpus_mechanisms_2026_08 import TRAIN_PTS
    from training.mortal.audit_objective_learnability import _record_game

    files = route_spec["index"]
    labels = route_spec["labels"]
    by_file = route_spec.get("by_file")
    file_index_arr = np.asarray(canonical["file_index"], dtype=np.int64)
    row_index_arr = np.asarray(canonical["row_index"], dtype=np.int64)
    sampled_positions = np.asarray(sampled_positions, dtype=np.int64)
    needed_by_file: dict[int, set[int]] = {}
    for pos in sampled_positions.tolist():
        file_index = int(file_index_arr[pos])
        needed_by_file.setdefault(file_index, set()).add(int(row_index_arr[pos]))

    records_by_pos: dict[int, object] = {}
    for file_index, path in enumerate(files):
        needed_rows = needed_by_file.get(int(file_index))
        if not needed_rows:
            continue
        label = by_file[str(path.resolve())] if by_file is not None else labels[0]
        loader = GameplayLoader(
            version=int(config["control"].get("version", 4)),
            oracle=False,
            player_names=[label],
            excludes=None,
            augmented=False,
        )
        loaded = loader.load_gz_log_files([str(path)])
        if len(loaded) != 1 or len(loaded[0]) != 1:
            raise RuntimeError(f"{route_name}: Adam microbatch loader mismatch for {path.name}")
        game = loaded[0][0]
        gamma = float(config["env"].get("gamma", 1.0))
        for row_index, record in enumerate(_record_game(game, TRAIN_PTS, gamma)):
            if row_index in needed_rows:
                # Find the canonical position for this (file, row).
                for pos in sampled_positions.tolist():
                    if int(file_index_arr[pos]) == int(file_index) and int(row_index_arr[pos]) == row_index:
                        records_by_pos[int(pos)] = record
                        break
    missing = [int(pos) for pos in sampled_positions.tolist() if int(pos) not in records_by_pos]
    if missing:
        raise RuntimeError(f"{route_name}: failed to collect {len(missing)} Adam microbatch rows")
    return [records_by_pos[int(pos)] for pos in sampled_positions.tolist()]


def _flatten_group_gradient(
    all_params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    group_params: list[torch.nn.Parameter],
) -> torch.Tensor | None:
    """Flatten a module group's gradient from the full autograd tuple.

    ``None`` gradients are replaced by zero vectors so the flattened gradient
    remains in the same parameter space as Adam state.
    """
    parts: list[torch.Tensor] = []
    for param in group_params:
        index = next((i for i, candidate in enumerate(all_params) if candidate is param), None)
        if index is None:
            raise RuntimeError("module param not found in all_params")
        grad = grads[index]
        if grad is None:
            grad = torch.zeros_like(param)
        parts.append(grad.detach().float().reshape(-1))
    return torch.cat(parts) if parts else None


def _module_gradient_cosines(
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    aux: torch.nn.Module,
    all_params: list[torch.nn.Parameter],
    slices: dict[str, list[torch.nn.Parameter]],
    grad_dqn: tuple[torch.Tensor | None, ...],
    grad_cql: tuple[torch.Tensor | None, ...],
) -> dict[str, float | None]:
    """DQN-vs-CQL parameter-gradient cosine per requested module group."""
    from training.mortal.audit_objective_learnability import _cosine

    groups = {
        "brain": slices["brain"],
        "dqn": slices["dqn"],
        "brain_dqn": slices["brain"] + slices["dqn"],
    }
    out: dict[str, float | None] = {}
    for name, params in groups.items():
        flat_dqn = _flatten_group_gradient(all_params, grad_dqn, params)
        flat_cql = _flatten_group_gradient(all_params, grad_cql, params)
        out[name] = _cosine(flat_dqn, flat_cql)
    return out


def _adam_diagnostic_for_route(
    route_name: str,
    records: list[object],
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    aux: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    """Compute frozen descriptive Adam/microbatch diagnostics for one route."""
    from training.mortal.audit_objective_learnability import (
        _cosine,
    )

    all_params = list(brain.parameters()) + list(dqn.parameters()) + list(aux.parameters())
    slices = {
        "brain": list(brain.parameters()),
        "dqn": list(dqn.parameters()),
        "aux": list(aux.parameters()),
        "brain_dqn": list(brain.parameters()) + list(dqn.parameters()),
    }
    cql_weight = float(config["cql"]["min_q_weight"])
    aux_weight = float(config["aux"]["next_rank_weight"])
    eps = float(config["optim"]["eps"])
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 32
    n_batches = len(records) // batch_size
    per_batch: list[dict[str, object]] = []

    for batch_index in range(n_batches):
        batch = records[batch_index * batch_size : (batch_index + 1) * batch_size]
        obs = torch.as_tensor(np.stack([r.obs for r in batch]), device=device, dtype=torch.float32)
        masks = torch.as_tensor(np.stack([r.mask for r in batch]), device=device, dtype=torch.bool)
        actions = torch.as_tensor([r.action for r in batch], device=device, dtype=torch.int64)
        targets = torch.as_tensor([r.q_target for r in batch], device=device, dtype=torch.float32)
        player_ranks = torch.as_tensor([r.player_rank for r in batch], device=device, dtype=torch.int64)

        phi = brain(obs)
        q_out = dqn(phi, masks)
        q = q_out[torch.arange(batch_size, device=device), actions]
        dqn_loss = 0.5 * torch.mean((q - targets) ** 2)
        cql_loss = q_out.logsumexp(-1).mean() - q.mean()
        (next_rank_logits,) = aux(phi)
        aux_loss = criterion(next_rank_logits, player_ranks)
        total_loss = dqn_loss + cql_weight * cql_loss + aux_weight * aux_loss

        grad_dqn = torch.autograd.grad(dqn_loss, all_params, retain_graph=True, allow_unused=True)
        grad_cql = torch.autograd.grad(cql_loss, all_params, retain_graph=True, allow_unused=True)
        grad_total = torch.autograd.grad(total_loss, all_params, allow_unused=True)

        group_values: dict[str, dict[str, float | None]] = {}
        for group_name, params in slices.items():
            flat_g = _flatten_group_gradient(all_params, grad_total, params)
            flat_m_parts = []
            flat_v_parts = []
            for param in params:
                state = optimizer.state.get(param)
                if state is None or "exp_avg" not in state or "exp_avg_sq" not in state:
                    raise RuntimeError(f"{route_name}: missing Adam state for a live parameter in {group_name}")
                flat_m_parts.append(state["exp_avg"].detach().float().reshape(-1))
                flat_v_parts.append(state["exp_avg_sq"].detach().float().reshape(-1))
            flat_m = torch.cat(flat_m_parts) if flat_m_parts else None
            flat_v = torch.cat(flat_v_parts) if flat_v_parts else None
            if flat_g is not None and flat_m is not None and flat_v is not None:
                denom = flat_m / (torch.sqrt(flat_v) + eps)
                group_values[group_name] = {
                    "cos_g_m": _cosine(flat_g, flat_m),
                    "cos_g_m_den": _cosine(flat_g, denom),
                }
            else:
                group_values[group_name] = {"cos_g_m": None, "cos_g_m_den": None}

        per_batch.append({
            "batch_index": batch_index,
            "dqn_cql_cosines": _module_gradient_cosines(
                brain, dqn, aux, all_params, slices, grad_dqn, grad_cql
            ),
            "adam": group_values,
        })

    summary: dict[str, object] = {}
    for group_name in slices:
        cos_m_values = [float(b["adam"][group_name]["cos_g_m"]) for b in per_batch if b["adam"][group_name]["cos_g_m"] is not None]
        cos_m_den_values = [float(b["adam"][group_name]["cos_g_m_den"]) for b in per_batch if b["adam"][group_name]["cos_g_m_den"] is not None]
        summary[group_name] = {
            "cos_g_m_mean": float(np.mean(cos_m_values)) if cos_m_values else None,
            "cos_g_m_median": float(np.median(cos_m_values)) if cos_m_values else None,
            "cos_g_m_den_mean": float(np.mean(cos_m_den_values)) if cos_m_den_values else None,
            "cos_g_m_den_median": float(np.median(cos_m_den_values)) if cos_m_den_values else None,
            "defined_batches": len(cos_m_values),
        }
    dqn_cql_summary: dict[str, object] = {}
    for group_name in ("brain", "dqn", "brain_dqn"):
        values = [float(b["dqn_cql_cosines"][group_name]) for b in per_batch if b["dqn_cql_cosines"][group_name] is not None]
        dqn_cql_summary[group_name] = {
            "cos_mean": float(np.mean(values)) if values else None,
            "defined_batches": len(values),
        }
    return {
        "per_batch": per_batch,
        "summary": summary,
        "dqn_cql_parameter_cosine": dqn_cql_summary,
        "batch_count": n_batches,
        "batch_size": batch_size,
    }


def run_formal_audit(device: torch.device, out_root: Path) -> None:
    """Formal decision-signal audit pipeline. Gated by FORMAL_RUN_AUTHORIZED."""
    if not FORMAL_RUN_AUTHORIZED:
        raise RuntimeError(f"formal run is not authorized: {RUN_AUTHORIZATION_NOTE}")
    preflight = formal_preflight(device, out_root)
    if not preflight["all_pass"]:
        raise RuntimeError("formal preflight failed")
    out_root.mkdir(parents=True, exist_ok=True)
    from training.mortal.audit_k0_representation_space_2026_08 import (
        _load_canonical_route,
        build_route_table,
    )
    from training.mortal.audit_objective_learnability import _build_model
    from training.mortal.audit_replay_distribution import load_model

    state = load_checkpoint(K0_MODEL)
    analysis_brain, analysis_dqn, _version = load_model(state, device)
    route_table = build_route_table()
    canonical_by_route = _load_decision_canonical(K0_REPR_OUTPUT)
    projection = np.load(K0_REPR_OUTPUT / "projection_matrix.npy", allow_pickle=False).astype(np.float64)
    routes_for_neighbors = {
        route: _load_canonical_route(K0_REPR_OUTPUT, route) for route in ROUTE_ORDER
    }
    rehydrated_by_route: dict[str, dict[str, np.ndarray]] = {}
    q_by_route: dict[str, np.ndarray] = {}
    adam_diagnostics: dict[str, object] = {}
    for route in ROUTE_ORDER:
        sorted_hanchan = np.asarray(routes_for_neighbors[route]["hanchan_index"], dtype=np.int64)
        sampled_positions = sample_microbatch_rows(
            sorted_hanchan, batch_size=32, n_batches=32, seed=20260824
        ).reshape(-1)
        # Fresh per-route Adam models prevent route-order contamination.
        adam_brain, adam_dqn, adam_aux, _ = _build_model(state, device)
        adam_optimizer = _build_preserved_optimizer(state, adam_brain, adam_dqn, adam_aux)
        rehydrated = _rehydrate_canonical_rows(
            route,
            route_table[route],
            canonical_by_route[route],
            analysis_brain,
            analysis_dqn,
            device,
            projection,
        )
        rehydrated_by_route[route] = rehydrated
        q_by_route[route] = rehydrated["q"]
        records = _collect_microbatch_records(
            route,
            route_table[route],
            canonical_by_route[route],
            sampled_positions,
            state["config"],
        )
        canonical_hashes = canonical_by_route[route]["canonical_hanchan_hashes"]
        perspective_labels = canonical_by_route[route]["perspective_labels"]
        identities = [
            (
                route,
                str(canonical_hashes[int(pos)]),
                int(canonical_by_route[route]["file_index"][pos]),
                int(canonical_by_route[route]["row_index"][pos]),
                str(perspective_labels[int(pos)]),
            )
            for pos in sampled_positions.tolist()
        ]
        sampled_sha = __import__("hashlib").sha256(
            json.dumps(identities, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        adam_diagnostics[route] = _adam_diagnostic_for_route(
            route, records, adam_brain, adam_dqn, adam_aux, adam_optimizer, state["config"], device
        )
        adam_diagnostics[route]["sampled_rows_sha256"] = sampled_sha
    analysis = _run_decision_analysis(K0_REPR_OUTPUT, canonical_by_route, rehydrated_by_route, q_by_route, projection)
    analysis["adam_diagnostic"] = adam_diagnostics
    verdict = authoritative_verdict(analysis["verdict"], bool(preflight["all_pass"]), complete=True)
    riichi_path = Path(riichi.__file__).resolve()
    report = {
        "schema": "keqing.mortal.k0_decision_signal_audit.v1",
        "preregistration": check_preregistration(),
        "formal_preflight": preflight,
        **git_worktree_metadata(),
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp": False,
            "riichi_extension_path": str(riichi_path),
            "riichi_extension_sha256": sha256(riichi_path),
        },
        "verdict": verdict,
        "analysis": _json_safe_report(analysis, out_root),
        "status": "complete",
    }
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-prereg", action="store_true", help="verify prereg document SHA")
    parser.add_argument("--checkpoint-smoke", action="store_true", help="run checkpoint-only phi smoke")
    parser.add_argument("--formal-preflight", action="store_true", help="run provenance preflight without corpus access")
    parser.add_argument("--formal-run", action="store_true", help="run formal audit (requires FORMAL_RUN_AUTHORIZED)")
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

    if args.formal_run:
        device = torch.device(args.device)
        if not FORMAL_RUN_AUTHORIZED:
            print(json.dumps({
                "status": "formal_run_not_authorized",
                "formal_run_authorized": False,
                "note": RUN_AUTHORIZATION_NOTE,
            }, ensure_ascii=False, indent=2), flush=True)
            raise SystemExit(1)
        run_formal_audit(device, OUTPUT_ROOT)
        return

    parser.print_help()
    raise SystemExit(
        "Formal corpus audit is NOT AUTHORIZED in this phase. "
        "Use --check-prereg, --checkpoint-smoke, --formal-preflight, or --formal-run."
    )


if __name__ == "__main__":
    main()
