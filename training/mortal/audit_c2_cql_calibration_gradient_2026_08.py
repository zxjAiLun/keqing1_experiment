"""Formal runner for C2 CQL calibration and gradient attribution audit.

Experiment ID: C2_cql_calibration_gradient_attribution_2026_08

Runs forward Q-calibration evaluation and autograd parameter gradient conflict
evaluation across 12 C1 checkpoints over a balanced shared-panel of 12,000 canonical rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

import riichi
from libriichi.dataset import GameplayLoader
from model import Brain, DQN

from training.mortal.audit_cross_corpus_mechanisms_2026_08 import (
    D1_PREP,
    DATA_ROOT,
    TRAIN_PTS,
    load_index,
)
from training.mortal.c2_cql_mechanism_core import (
    ACTION_DIM,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CONDITIONS,
    LEGAL_TARGET_SET,
    ROUTES,
    SEEDS,
    bootstrap_ci95,
    check_target_values,
    compute_batch_parameter_gradients,
    compute_delta_and_interaction,
    determine_verdict,
    forward_q_and_metrics,
    gradient_conflict_vote,
    q_calibration_vote,
    sha256_array,
    sha256_file,
)

SCHEMA = "keqing.mortal.c2_cql_mechanism_summary.v1"
EXPERIMENT_ID = "C2_cql_calibration_gradient_attribution_2026_08"
K0_REPR_OUTPUT = DATA_ROOT / "mortal/authoritative/K0_representation_audit_2026_08"
C1_EVAL_ROOT = REPO / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/evaluation_implementation_2026_08"
C1_TRAIN_ROOT = REPO / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/training_implementation_2026_08"
OUTPUT_ROOT = REPO / "artifacts/experiments/C2_cql_calibration_gradient_attribution_2026_08"

# 12 Target Checkpoints Map: (route, condition, seed) -> (path, expected_sha256)
CHECKPOINT_SPEC: dict[tuple[str, str, int], tuple[Path, str]] = {
    # M0 CURRENT
    ("M0", "CURRENT", 20260806): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/M0_control/seed_20260806/checkpoints/mortal_72000.pth"),
        "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
    ),
    ("M0", "CURRENT", 20260807): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/M0_control/seed_20260807/checkpoints/mortal_72000.pth"),
        "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
    ),
    ("M0", "CURRENT", 20260808): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/M0_control/seed_20260808/checkpoints/mortal_72000.pth"),
        "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
    ),
    # D1 CURRENT
    ("D1", "CURRENT", 20260806): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/D1_variant/seed_20260806/checkpoints/mortal_72000.pth"),
        "9425109b2562eb48a86ca7b3a250738b5691503f9156f29bc50a2b20e7a922aa",
    ),
    ("D1", "CURRENT", 20260807): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/D1_variant/seed_20260807/checkpoints/mortal_72000.pth"),
        "e2718ee8d572071b8d46d04beaf5f2aa6d90ad847762254f80648de9639a0b3d",
    ),
    ("D1", "CURRENT", 20260808): (
        Path("/media/bailan/DISK/AUbuntuProject/project/keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/D1_variant/seed_20260808/checkpoints/mortal_72000.pth"),
        "985a3e532ef13cd7fab945c92839a941390fd9f7cc5dc0e177d4d4182a116f41",
    ),
    # M0 CQL_OFF
    ("M0", "CQL_OFF", 20260806): (
        C1_TRAIN_ROOT / "M0_CQL_OFF/seed_20260806/checkpoints/mortal_72000.pth",
        "7e74aad0f0389973e5e7516e8fa2d2b64caa53599a0fe1c61b6b258e3c9f35c5",
    ),
    ("M0", "CQL_OFF", 20260807): (
        C1_TRAIN_ROOT / "M0_CQL_OFF/seed_20260807/checkpoints/mortal_72000.pth",
        "e1bd426b8c9b2004cfc9728021f3c00ff77a57db6878e92a9515ce21eeb5ded5",
    ),
    ("M0", "CQL_OFF", 20260808): (
        C1_TRAIN_ROOT / "M0_CQL_OFF/seed_20260808/checkpoints/mortal_72000.pth",
        "8c68a89ddc3164c07b43308a56113a7b15177c65b6dde47e1c400b024d9fd793",
    ),
    # D1 CQL_OFF
    ("D1", "CQL_OFF", 20260806): (
        C1_TRAIN_ROOT / "D1_CQL_OFF/seed_20260806/checkpoints/mortal_72000.pth",
        "99ee9985753dedd11453fab0a0e142793f8a94af13ce4bcc4526b9e28643ca95",
    ),
    ("D1", "CQL_OFF", 20260807): (
        C1_TRAIN_ROOT / "D1_CQL_OFF/seed_20260807/checkpoints/mortal_72000.pth",
        "57b9c7dff51e118595d83bb838492c74374b49c52edb9ffe8c4c991a112ca661",
    ),
    ("D1", "CQL_OFF", 20260808): (
        C1_TRAIN_ROOT / "D1_CQL_OFF/seed_20260808/checkpoints/mortal_72000.pth",
        "697f6ce021d2f4379a8943d464dcb387e7a7ed8ece70ddaba394e220b3563022",
    ),
}


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Brain, DQN]:
    """Load Brain and DQN from checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = state["config"]
    version = int(model_config["control"].get("version", 4))
    brain = Brain(version=version, **model_config["resnet"]).to(device)
    dqn = DQN(version=version).to(device)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    brain.eval()
    dqn.eval()
    return brain, dqn


def load_shared_panel_data(device: torch.device, smoke: bool = False) -> dict[str, Any]:
    """Load 6000 M0 and 6000 D1 canonical rows and rehydrate obs/action/mask with strict memory control."""
    cache_path = OUTPUT_ROOT / "shared_panel_cache.npz"
    if not smoke and cache_path.exists():
        print(f"Loading shared panel from cache {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        panel_by_route = {}
        for route in ["M0", "D1"]:
            panel_by_route[route] = {
                "obs": cached[f"{route}_obs"],
                "actions": cached[f"{route}_actions"],
                "masks": cached[f"{route}_masks"],
                "targets": cached[f"{route}_targets"],
                "hanchan_indices": cached[f"{route}_hanchan_indices"],
                "file_indices": cached[f"{route}_file_indices"],
                "row_indices": cached[f"{route}_row_indices"],
            }
        return panel_by_route

    m0_index = load_index(D1_PREP / "file_index_m0.pth")
    d1_index = load_index(D1_PREP / "file_index_d1.pth")
    
    panel_by_route = {}
    routes_to_load = [("M0", m0_index, "ext_mortal")] if smoke else [("M0", m0_index, "ext_mortal"), ("D1", d1_index, "K0_70k")]
    
    for route, files, label in routes_to_load:
        meta = np.load(K0_REPR_OUTPUT / f"route_artifacts/{route}/canonical_metadata.npz", allow_pickle=True)
        hanchan_indices = meta["hanchan_indices"] if "hanchan_indices" in meta else meta["hanchan_index"]
        file_indices = meta["file_index"]
        row_indices = meta["row_index"]
        targets = meta["target"]
        
        # Select first occurrence of each unique hanchan_index in array order
        _, first_pos = np.unique(hanchan_indices, return_index=True)
        first_pos = np.sort(first_pos)
        assert len(first_pos) == 6000, f"{route} must have exactly 6000 unique hanchans"
        
        sel_file_idx = file_indices[first_pos]
        sel_row_idx = row_indices[first_pos]
        sel_targets = targets[first_pos]
        sel_hanchan_idx = hanchan_indices[first_pos]
        
        # Verify target set
        if not check_target_values(sel_targets):
            raise ValueError(f"{route} targets contain non-centered values")
        
        # Map file_index -> (sel_index, row_index)
        rows_by_file: dict[int, tuple[int, int]] = {}
        for i, (f_idx, r_idx) in enumerate(zip(sel_file_idx, sel_row_idx)):
            rows_by_file[int(f_idx)] = (i, int(r_idx))
        
        n_rows = 100 if smoke else 6000
        obs_arr = np.empty((n_rows, 1012, 34), dtype=np.float32)
        actions_arr = np.empty(n_rows, dtype=np.int64)
        masks_arr = np.empty((n_rows, 46), dtype=bool)
        
        loader_instance = GameplayLoader(
            version=4,
            oracle=False,
            player_names=[label],
            excludes=None,
            augmented=False,
        )
        
        t0 = time.time()
        # Chunked streaming extraction (chunk_size=50) for high C++ speed and tight memory bounds
        chunk_size = 50
        active_f_indices = [f_idx for f_idx in range(n_rows) if f_idx in rows_by_file]
        for c_start in range(0, len(active_f_indices), chunk_size):
            c_chunk = active_f_indices[c_start : c_start + chunk_size]
            c_paths = [str(files[f_idx]) for f_idx in c_chunk]
            loaded_games = loader_instance.load_gz_log_files(c_paths)
            for f_idx, game_outer in zip(c_chunk, loaded_games):
                game = game_outer[0]
                sel_idx, row = rows_by_file[f_idx]
                out_idx = f_idx if smoke else sel_idx
                
                obs_all = game.take_obs()
                actions_all = game.take_actions()
                masks_all = game.take_masks()
                grp = game.take_grp()
                player_id = int(game.take_player_id())
                final_rank = int(grp.take_rank_by_player()[player_id])
                expected_target = float(TRAIN_PTS[final_rank] - TRAIN_PTS.mean())
                
                if not np.isclose(expected_target, sel_targets[sel_idx], atol=1e-12):
                    raise ValueError(f"{route} target mismatch at file {f_idx} row {row}")
                
                obs_arr[out_idx] = np.asarray(obs_all[row], dtype=np.float32)
                actions_arr[out_idx] = int(actions_all[row])
                masks_arr[out_idx] = np.asarray(masks_all[row], dtype=bool)
                
            del loaded_games
            if (c_start + chunk_size) % 1000 == 0 or (c_start + chunk_size) >= len(active_f_indices):
                elapsed = time.time() - t0
                print(f"  [{route}] {min(c_start + chunk_size, len(active_f_indices))}/{len(active_f_indices)} rows rehydrated in {elapsed:.1f}s", flush=True)
        
        # Verify legal action
        for i in range(len(actions_arr)):
            if not masks_arr[i, actions_arr[i]]:
                raise ValueError(f"{route} rehydrated action is illegal at row {i}")
        
        panel_by_route[route] = {
            "obs": obs_arr,
            "actions": actions_arr,
            "masks": masks_arr,
            "targets": sel_targets[:n_rows] if smoke else sel_targets,
            "hanchan_indices": sel_hanchan_idx[:n_rows] if smoke else sel_hanchan_idx,
            "file_indices": sel_file_idx[:n_rows] if smoke else sel_file_idx,
            "row_indices": sel_row_idx[:n_rows] if smoke else sel_row_idx,
        }
    
    if not smoke:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            M0_obs=panel_by_route["M0"]["obs"],
            M0_actions=panel_by_route["M0"]["actions"],
            M0_masks=panel_by_route["M0"]["masks"],
            M0_targets=panel_by_route["M0"]["targets"],
            M0_hanchan_indices=panel_by_route["M0"]["hanchan_indices"],
            M0_file_indices=panel_by_route["M0"]["file_indices"],
            M0_row_indices=panel_by_route["M0"]["row_indices"],
            D1_obs=panel_by_route["D1"]["obs"],
            D1_actions=panel_by_route["D1"]["actions"],
            D1_masks=panel_by_route["D1"]["masks"],
            D1_targets=panel_by_route["D1"]["targets"],
            D1_hanchan_indices=panel_by_route["D1"]["hanchan_indices"],
            D1_file_indices=panel_by_route["D1"]["file_indices"],
            D1_row_indices=panel_by_route["D1"]["row_indices"],
        )
        print(f"Saved shared panel cache to {cache_path}", flush=True)
        
    return panel_by_route


def run_audit(device: torch.device, smoke: bool = False) -> dict[str, Any]:
    t0 = time.time()
    gates = {}
    
    # 1. Verify Checkpoints existence and SHA
    checkpoint_shas = {}
    ckpt_pass = True
    for key, (path, expected_sha) in CHECKPOINT_SPEC.items():
        if not path.exists():
            ckpt_pass = False
            checkpoint_shas[f"{key[0]}_{key[1]}_{key[2]}"] = {"path": str(path), "status": "missing"}
            continue
        actual_sha = sha256_file(path)
        is_match = (actual_sha == expected_sha)
        if not is_match:
            ckpt_pass = False
        checkpoint_shas[f"{key[0]}_{key[1]}_{key[2]}"] = {
            "path": str(path),
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "sha_match": is_match,
        }
    gates["checkpoints_verified"] = ckpt_pass
    
    # 2. Load panel data
    print("Loading shared panel data...")
    panel = load_shared_panel_data(device, smoke=smoke)
    m0_data = panel["M0"]
    d1_data = panel.get("D1")
    
    gates["canonical_rows_rehydrated"] = True
    gates["target_values_centered"] = (
        check_target_values(m0_data["targets"]) and (d1_data is None or check_target_values(d1_data["targets"]))
    )
    
    # In smoke mode, just run 1 batch of 100 rows on 1 checkpoint
    if smoke:
        print("Running smoke verification on 1 batch...")
        ckpt_key = ("M0", "CURRENT", 20260806)
        path, _ = CHECKPOINT_SPEC[ckpt_key]
        brain, dqn = load_model(path, device)
        
        obs_sub = torch.as_tensor(m0_data["obs"][:100], device=device)
        masks_sub = torch.as_tensor(m0_data["masks"][:100], device=device)
        actions_sub = torch.as_tensor(m0_data["actions"][:100], device=device)
        targets_sub = torch.as_tensor(m0_data["targets"][:100], dtype=torch.float32, device=device)
        
        with torch.inference_mode():
            q_out = forward_q_and_metrics(brain, dqn, obs_sub, masks_sub, actions_sub, targets_sub)
        
        cos_all, cos_brain, cos_dqn = compute_batch_parameter_gradients(
            brain, dqn, obs_sub, masks_sub, actions_sub, targets_sub
        )
        
        print("Smoke Forward Q finite:", torch.isfinite(q_out["q_behavior"]).all().item())
        print("Smoke Gradient Cosine:", cos_all, "Brain:", cos_brain, "DQN:", cos_dqn)
        print("SMOKE PASS")
        return {"smoke_pass": True}

    # Full Formal Run
    print("Running formal audit across 12 checkpoints...")
    # Results store: (route_data, ckpt_key) -> metrics dict
    # We evaluate each checkpoint on BOTH M0 (6000) and D1 (6000) rows -> balanced shared panel
    
    q_metrics_by_cell: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    grad_cos_by_cell: dict[tuple[str, str, int], list[float]] = {}
    
    # Setup microbatches: 60 batches of 100 rows for M0 and D1
    batch_size = 100
    n_batches = 60
    
    # Checkpoint loop
    models_eval_mode_pass = True
    for (c_route, condition, seed), (ckpt_path, _) in CHECKPOINT_SPEC.items():
        cell_key = (c_route, condition, seed)
        print(f"Evaluating checkpoint {c_route}_{condition}_{seed}...")
        brain, dqn = load_model(ckpt_path, device)
        if brain.training or dqn.training:
            models_eval_mode_pass = False
        
        # Forward pass on all 12,000 rows (6000 M0 + 6000 D1) in chunks of 500
        all_res = {"abs_residual": [], "residual": [], "overestimate": [], "cql_penalty": [], "legal_entropy": []}
        
        for p_route in ["M0", "D1"]:
            p_obs = panel[p_route]["obs"]
            p_masks = panel[p_route]["masks"]
            p_actions = panel[p_route]["actions"]
            p_targets = panel[p_route]["targets"]
            
            for start in range(0, 6000, 500):
                stop = min(start + 500, 6000)
                obs_t = torch.as_tensor(p_obs[start:stop], device=device)
                masks_t = torch.as_tensor(p_masks[start:stop], device=device)
                actions_t = torch.as_tensor(p_actions[start:stop], device=device)
                targets_t = torch.as_tensor(p_targets[start:stop], dtype=torch.float32, device=device)
                
                with torch.inference_mode():
                    q_res = forward_q_and_metrics(brain, dqn, obs_t, masks_t, actions_t, targets_t)
                
                all_res["abs_residual"].append(q_res["abs_residual"].cpu().numpy())
                all_res["residual"].append(q_res["residual"].cpu().numpy())
                all_res["overestimate"].append(q_res["overestimate"].cpu().numpy())
                all_res["cql_penalty"].append(q_res["cql_penalty"].cpu().numpy())
                all_res["legal_entropy"].append(q_res["legal_entropy"].cpu().numpy())
        
        q_metrics_by_cell[cell_key] = {k: np.concatenate(v) for k, v in all_res.items()}
        
        # Parameter gradients over 60 fixed microbatches for M0 and 60 for D1 (total 120 shared microbatches)
        cos_list = []
        for p_route in ["M0", "D1"]:
            p_obs = panel[p_route]["obs"]
            p_masks = panel[p_route]["masks"]
            p_actions = panel[p_route]["actions"]
            p_targets = panel[p_route]["targets"]
            
            for b in range(n_batches):
                start = b * batch_size
                stop = start + batch_size
                obs_b = torch.as_tensor(p_obs[start:stop], device=device)
                masks_b = torch.as_tensor(p_masks[start:stop], device=device)
                actions_b = torch.as_tensor(p_actions[start:stop], device=device)
                targets_b = torch.as_tensor(p_targets[start:stop], dtype=torch.float32, device=device)
                
                cos_all, _, _ = compute_batch_parameter_gradients(
                    brain, dqn, obs_b, masks_b, actions_b, targets_b
                )
                cos_list.append(cos_all)
        
        grad_cos_by_cell[cell_key] = cos_list
    
    gates["models_eval_mode"] = models_eval_mode_pass
    
    # 3. Compute Seed-Level Metrics & Interactions
    # Total shared panel rows = 12000 (0..5999 M0, 6000..11999 D1)
    # Total shared gradient batches = 120 (0..59 M0 batches, 60..119 D1 batches)
    
    # Q calibration metrics
    abs_res_means = {}
    overestimate_means = {}
    for cell_key, m_dict in q_metrics_by_cell.items():
        abs_res_means[cell_key] = float(np.mean(m_dict["abs_residual"]))
        overestimate_means[cell_key] = float(np.mean(m_dict["overestimate"]))
    
    i_abs_res_by_seed = {}
    i_overest_by_seed = {}
    for s in SEEDS:
        _, _, i_abs = compute_delta_and_interaction(
            abs_res_means[("D1", "CQL_OFF", s)],
            abs_res_means[("D1", "CURRENT", s)],
            abs_res_means[("M0", "CQL_OFF", s)],
            abs_res_means[("M0", "CURRENT", s)],
        )
        _, _, i_over = compute_delta_and_interaction(
            overestimate_means[("D1", "CQL_OFF", s)],
            overestimate_means[("D1", "CURRENT", s)],
            overestimate_means[("M0", "CQL_OFF", s)],
            overestimate_means[("M0", "CURRENT", s)],
        )
        i_abs_res_by_seed[s] = i_abs
        i_overest_by_seed[s] = i_over
    
    # Gradient metrics
    grad_cos_means = {}
    grad_conflict_means = {}
    for cell_key, cos_l in grad_cos_by_cell.items():
        cos_arr = np.array(cos_l)
        grad_cos_means[cell_key] = float(np.mean(cos_arr))
        grad_conflict_means[cell_key] = float(np.mean(cos_arr < 0))
    
    i_grad_cos_by_seed = {}
    i_grad_conflict_by_seed = {}
    for s in SEEDS:
        _, _, i_cos = compute_delta_and_interaction(
            grad_cos_means[("D1", "CQL_OFF", s)],
            grad_cos_means[("D1", "CURRENT", s)],
            grad_cos_means[("M0", "CQL_OFF", s)],
            grad_cos_means[("M0", "CURRENT", s)],
        )
        _, _, i_conf = compute_delta_and_interaction(
            grad_conflict_means[("D1", "CQL_OFF", s)],
            grad_conflict_means[("D1", "CURRENT", s)],
            grad_conflict_means[("M0", "CQL_OFF", s)],
            grad_conflict_means[("M0", "CURRENT", s)],
        )
        i_grad_cos_by_seed[s] = i_cos
        i_grad_conflict_by_seed[s] = i_conf

    # 4. Bootstrap CI95
    print("Running bootstrap (reps=5000, seed=20260824)...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    
    # Q bootstrap: cluster unit is hanchan (6000 M0, 6000 D1)
    # Balanced panel: resample 6000 from M0 panel and 6000 from D1 panel
    boot_i_abs_res = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    boot_i_overest = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    
    n_m0 = 6000
    n_d1 = 6000
    for b in range(BOOTSTRAP_REPS):
        idx_m0 = rng.integers(0, n_m0, size=n_m0)
        idx_d1 = rng.integers(0, n_d1, size=n_d1) + n_m0
        boot_idx = np.concatenate([idx_m0, idx_d1])
        
        i_abs_seeds = []
        i_over_seeds = []
        for s in SEEDS:
            m_d1_off = np.mean(q_metrics_by_cell[("D1", "CQL_OFF", s)]["abs_residual"][boot_idx])
            m_d1_curr = np.mean(q_metrics_by_cell[("D1", "CURRENT", s)]["abs_residual"][boot_idx])
            m_m0_off = np.mean(q_metrics_by_cell[("M0", "CQL_OFF", s)]["abs_residual"][boot_idx])
            m_m0_curr = np.mean(q_metrics_by_cell[("M0", "CURRENT", s)]["abs_residual"][boot_idx])
            _, _, i_abs_draw = compute_delta_and_interaction(m_d1_off, m_d1_curr, m_m0_off, m_m0_curr)
            i_abs_seeds.append(i_abs_draw)
            
            o_d1_off = np.mean(q_metrics_by_cell[("D1", "CQL_OFF", s)]["overestimate"][boot_idx])
            o_d1_curr = np.mean(q_metrics_by_cell[("D1", "CURRENT", s)]["overestimate"][boot_idx])
            o_m0_off = np.mean(q_metrics_by_cell[("M0", "CQL_OFF", s)]["overestimate"][boot_idx])
            o_m0_curr = np.mean(q_metrics_by_cell[("M0", "CURRENT", s)]["overestimate"][boot_idx])
            _, _, i_over_draw = compute_delta_and_interaction(o_d1_off, o_d1_curr, o_m0_off, o_m0_curr)
            i_over_seeds.append(i_over_draw)
            
        boot_i_abs_res[b] = np.mean(i_abs_seeds)
        boot_i_overest[b] = np.mean(i_over_seeds)
        
    ci_abs_res = bootstrap_ci95(boot_i_abs_res)
    ci_overest = bootstrap_ci95(boot_i_overest)
    
    # Gradient bootstrap: cluster unit is microbatch (60 M0, 60 D1, total 120)
    boot_i_grad_cos = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    boot_i_grad_conf = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    
    for b in range(BOOTSTRAP_REPS):
        b_idx_m0 = rng.integers(0, 60, size=60)
        b_idx_d1 = rng.integers(0, 60, size=60) + 60
        b_idx = np.concatenate([b_idx_m0, b_idx_d1])
        
        i_cos_seeds = []
        i_conf_seeds = []
        for s in SEEDS:
            cos_d1_off = np.array(grad_cos_by_cell[("D1", "CQL_OFF", s)])[b_idx]
            cos_d1_curr = np.array(grad_cos_by_cell[("D1", "CURRENT", s)])[b_idx]
            cos_m0_off = np.array(grad_cos_by_cell[("M0", "CQL_OFF", s)])[b_idx]
            cos_m0_curr = np.array(grad_cos_by_cell[("M0", "CURRENT", s)])[b_idx]
            
            _, _, i_cos_draw = compute_delta_and_interaction(
                np.mean(cos_d1_off), np.mean(cos_d1_curr), np.mean(cos_m0_off), np.mean(cos_m0_curr)
            )
            _, _, i_conf_draw = compute_delta_and_interaction(
                np.mean(cos_d1_off < 0), np.mean(cos_d1_curr < 0), np.mean(cos_m0_off < 0), np.mean(cos_m0_curr < 0)
            )
            i_cos_seeds.append(i_cos_draw)
            i_conf_seeds.append(i_conf_draw)
            
        boot_i_grad_cos[b] = np.mean(i_cos_seeds)
        boot_i_grad_conf[b] = np.mean(i_conf_seeds)
        
    ci_grad_cos = bootstrap_ci95(boot_i_grad_cos)
    ci_grad_conf = bootstrap_ci95(boot_i_grad_conf)

    # 5. Mechanism Family Votes & Verdict
    calib_pass = q_calibration_vote(
        i_abs_residual_seeds=tuple(i_abs_res_by_seed[s] for s in SEEDS),
        ci_abs_residual=ci_abs_res,
        i_overestimate_seeds=tuple(i_overest_by_seed[s] for s in SEEDS),
        ci_overestimate=ci_overest,
    )
    
    grad_pass = gradient_conflict_vote(
        i_cosine_seeds=tuple(i_grad_cos_by_seed[s] for s in SEEDS),
        ci_cosine=ci_grad_cos,
        i_conflict_seeds=tuple(i_grad_conflict_by_seed[s] for s in SEEDS),
        ci_conflict=ci_grad_conf,
    )
    
    all_gates_pass = all(gates.values())
    verdict = determine_verdict(all_gates_pass, calib_pass, grad_pass)
    
    elapsed = time.time() - t0
    
    # 6. Assemble Summary & Manifest
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoints": checkpoint_shas,
        "shared_panel": {
            "m0_canonical_metadata": str(K0_REPR_OUTPUT / "route_artifacts/M0/canonical_metadata.npz"),
            "d1_canonical_metadata": str(K0_REPR_OUTPUT / "route_artifacts/D1/canonical_metadata.npz"),
            "m0_rows": 6000,
            "d1_rows": 6000,
            "total_shared_rows": 12000,
            "microbatches_per_route": 60,
            "rows_per_microbatch": 100,
        },
    }
    
    cache_path = OUTPUT_ROOT / "shared_panel_cache.npz"
    cache_sha = sha256_file(cache_path) if cache_path.exists() else None
    runner_sha = sha256_file(Path(__file__).resolve())
    
    summary = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "provenance": {
            "implementation_commit": "ca894adf614eb9391da9aa761abca7d616dc3b00",
            "runner_sha256": runner_sha,
            "shared_panel_cache_sha256": cache_sha,
        },
        "runtime_seconds": elapsed,
        "device": str(device),
        "hard_gates": gates,
        "calibration_family": {
            "abs_residual": {
                "equal_seed_interaction_mean": float(np.mean([i_abs_res_by_seed[s] for s in SEEDS])),
                "seed_interactions": i_abs_res_by_seed,
                "hierarchical_ci95": ci_abs_res,
                "pass_all_seeds_negative": all(i_abs_res_by_seed[s] < 0 for s in SEEDS),
                "pass_ci_upper_negative": ci_abs_res[1] < 0,
            },
            "overestimate_rate": {
                "equal_seed_interaction_mean": float(np.mean([i_overest_by_seed[s] for s in SEEDS])),
                "seed_interactions": i_overest_by_seed,
                "hierarchical_ci95": ci_overest,
                "pass_all_seeds_negative": all(i_overest_by_seed[s] < 0 for s in SEEDS),
                "pass_ci_upper_negative": ci_overest[1] < 0,
            },
            "vote": calib_pass,
        },
        "gradient_family": {
            "gradient_cosine": {
                "equal_seed_interaction_mean": float(np.mean([i_grad_cos_by_seed[s] for s in SEEDS])),
                "seed_interactions": i_grad_cos_by_seed,
                "hierarchical_ci95": ci_grad_cos,
                "pass_all_seeds_positive": all(i_grad_cos_by_seed[s] > 0 for s in SEEDS),
                "pass_ci_lower_positive": ci_grad_cos[0] > 0,
            },
            "gradient_conflict_rate": {
                "equal_seed_interaction_mean": float(np.mean([i_grad_conflict_by_seed[s] for s in SEEDS])),
                "seed_interactions": i_grad_conflict_by_seed,
                "hierarchical_ci95": ci_grad_conf,
                "pass_all_seeds_negative": all(i_grad_conflict_by_seed[s] < 0 for s in SEEDS),
                "pass_ci_upper_negative": ci_grad_conf[1] < 0,
            },
            "vote": grad_pass,
        },
        "machine_verdict": verdict,
    }
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "c2_input_manifest.json"
    summary_path = OUTPUT_ROOT / "c2_summary.json"
    
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    summary_sha = sha256_file(summary_path)
    print(f"Summary written to {summary_path} (SHA256: {summary_sha})")
    
    return {
        "runtime_seconds": elapsed,
        "hard_gates": gates,
        "i_abs_res": (i_abs_res_by_seed, ci_abs_res),
        "i_overest": (i_overest_by_seed, ci_overest),
        "i_grad_cos": (i_grad_cos_by_seed, ci_grad_cos),
        "i_grad_conf": (i_grad_conflict_by_seed, ci_grad_conf),
        "machine_verdict": verdict,
        "summary_sha256": summary_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run smoke verification only")
    parser.add_argument("--run", action="store_true", help="Run full formal C2 mechanism audit")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.smoke:
        run_audit(device, smoke=True)
    elif args.run:
        run_audit(device, smoke=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
