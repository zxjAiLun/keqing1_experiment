"""Core numerical routines, bootstrap estimators, and gates for C2 mechanism audit.

Experiment ID: C2_cql_calibration_gradient_attribution_2026_08
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DIM = 46
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260824
LEGAL_TARGET_SET = {-3.0, -1.0, 1.0, 3.0}
SEEDS = (20260806, 20260807, 20260808)
ROUTES = ("M0", "D1")
CONDITIONS = ("CURRENT", "CQL_OFF")


def sha256_file(path: str | Path) -> str:
    """Compute sha256 of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Deterministic hash of numpy array data."""
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def check_target_values(targets: np.ndarray) -> bool:
    """Verify all targets belong to {-3.0, -1.0, +1.0, +3.0}."""
    unique_vals = set(np.unique(targets).tolist())
    return unique_vals.issubset(LEGAL_TARGET_SET)


def compute_delta_and_interaction(
    m_d1_off: float, m_d1_curr: float, m_m0_off: float, m_m0_curr: float
) -> tuple[float, float, float]:
    """Compute Delta_D1, Delta_M0, and interaction I."""
    delta_d1 = m_d1_off - m_d1_curr
    delta_m0 = m_m0_off - m_m0_curr
    interaction = delta_d1 - delta_m0
    return delta_d1, delta_m0, interaction


def bootstrap_ci95(draws: np.ndarray) -> tuple[float, float]:
    """Compute empirical 95% confidence interval [2.5%, 97.5%]."""
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def q_calibration_vote(
    i_abs_residual_seeds: tuple[float, float, float],
    ci_abs_residual: tuple[float, float],
    i_overestimate_seeds: tuple[float, float, float],
    ci_overestimate: tuple[float, float],
) -> bool:
    """q_calibration_signal vote rule:
    1. all 3 seed I(abs_residual) < 0
    2. CI95 upper of I(abs_residual) < 0
    3. all 3 seed I(overestimate_rate) < 0
    4. CI95 upper of I(overestimate_rate) < 0
    """
    cond1 = all(val < 0 for val in i_abs_residual_seeds)
    cond2 = ci_abs_residual[1] < 0
    cond3 = all(val < 0 for val in i_overestimate_seeds)
    cond4 = ci_overestimate[1] < 0
    return bool(cond1 and cond2 and cond3 and cond4)


def gradient_conflict_vote(
    i_cosine_seeds: tuple[float, float, float],
    ci_cosine: tuple[float, float],
    i_conflict_seeds: tuple[float, float, float],
    ci_conflict: tuple[float, float],
) -> bool:
    """parameter_gradient_conflict_signal vote rule:
    1. all 3 seed I(gradient_cosine) > 0
    2. CI95 lower of I(gradient_cosine) > 0
    3. all 3 seed I(gradient_conflict_rate) < 0
    4. CI95 upper of I(gradient_conflict_rate) < 0
    """
    cond1 = all(val > 0 for val in i_cosine_seeds)
    cond2 = ci_cosine[0] > 0
    cond3 = all(val < 0 for val in i_conflict_seeds)
    cond4 = ci_conflict[1] < 0
    return bool(cond1 and cond2 and cond3 and cond4)


def determine_verdict(
    gates_pass: bool,
    calib_pass: bool,
    grad_pass: bool,
) -> str:
    """Determine machine verdict string."""
    if not gates_pass:
        return "no_verdict_gates_failed"
    if calib_pass and grad_pass:
        return "calibration_and_gradient_signal"
    if calib_pass and not grad_pass:
        return "q_calibration_signal"
    if not calib_pass and grad_pass:
        return "parameter_gradient_conflict_signal"
    return "inconclusive"


def forward_q_and_metrics(
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    obs: torch.Tensor,
    masks: torch.Tensor,
    behavior_actions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forward pass over a batch to compute Q-values and Q metrics.
    
    obs: (N, C, H, W) float32
    masks: (N, 46) bool
    behavior_actions: (N,) int64
    targets: (N,) float32
    """
    phi = brain(obs)
    q = dqn(phi, masks)  # (N, 46), masked with -inf at illegal coordinates
    
    # behavior Q
    q_behavior = q.gather(1, behavior_actions.unsqueeze(1)).squeeze(1)  # (N,)
    residual = q_behavior - targets
    abs_residual = torch.abs(residual)
    overestimate = (residual > 0).to(torch.float32)
    
    # descriptive metrics (without vote)
    # legal softmax entropy & legal logsumexp
    # Q at illegal actions is -inf, so logsumexp over legal actions:
    legal_logsumexp = torch.logsumexp(q, dim=1)  # (N,)
    cql_penalty = legal_logsumexp - q_behavior
    
    # legal softmax entropy: sum -p log p
    # p = softmax(q, dim=1), illegal has q=-inf -> p=0, log p = -inf -> p log p = 0
    probs = F.softmax(q, dim=1)
    log_probs = F.log_softmax(q, dim=1)
    # mask out illegal to avoid nan in 0 * -inf
    legal_p_log_p = torch.where(masks, probs * log_probs, torch.zeros_like(probs))
    legal_entropy = -torch.sum(legal_p_log_p, dim=1)
    
    return {
        "q": q,
        "q_behavior": q_behavior,
        "residual": residual,
        "abs_residual": abs_residual,
        "overestimate": overestimate,
        "cql_penalty": cql_penalty,
        "legal_entropy": legal_entropy,
    }


def compute_batch_parameter_gradients(
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    obs: torch.Tensor,
    masks: torch.Tensor,
    behavior_actions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute gradients of L_value and L_cql w.r.t shared parameters, and their cosine.
    
    L_value = 0.5 * mean((Q_behavior - target)^2)
    L_cql = mean(logsumexp(Q_legal) - Q_behavior)
    
    Returns (cosine_all, cosine_brain, cosine_dqn).
    """
    brain.zero_grad(set_to_none=True)
    dqn.zero_grad(set_to_none=True)
    
    # Forward pass with autograd enabled
    phi = brain(obs)
    q = dqn(phi, masks)
    q_behavior = q.gather(1, behavior_actions.unsqueeze(1)).squeeze(1)
    
    l_value = 0.5 * torch.mean((q_behavior - targets) ** 2)
    l_cql = torch.mean(torch.logsumexp(q, dim=1) - q_behavior)
    
    # Get parameters in fixed order
    brain_params = [p for p in brain.parameters() if p.requires_grad]
    dqn_params = [p for p in dqn.parameters() if p.requires_grad]
    all_params = brain_params + dqn_params
    
    # Gradient of L_value
    grads_value = torch.autograd.grad(
        l_value, all_params, retain_graph=True, allow_unused=False
    )
    # Gradient of L_cql
    grads_cql = torch.autograd.grad(
        l_cql, all_params, retain_graph=False, allow_unused=False
    )
    
    # Flatten
    gv_brain = torch.cat([g.reshape(-1) for g in grads_value[:len(brain_params)]])
    gc_brain = torch.cat([g.reshape(-1) for g in grads_cql[:len(brain_params)]])
    
    gv_dqn = torch.cat([g.reshape(-1) for g in grads_value[len(brain_params):]])
    gc_dqn = torch.cat([g.reshape(-1) for g in grads_cql[len(brain_params):]])
    
    gv_all = torch.cat([gv_brain, gv_dqn])
    gc_all = torch.cat([gc_brain, gc_dqn])
    
    eps = 1e-12
    cos_all = float(
        (torch.dot(gv_all, gc_all) / (gv_all.norm() * gc_all.norm() + eps)).item()
    )
    cos_brain = float(
        (torch.dot(gv_brain, gc_brain) / (gv_brain.norm() * gc_brain.norm() + eps)).item()
    )
    cos_dqn = float(
        (torch.dot(gv_dqn, gc_dqn) / (gv_dqn.norm() * gc_dqn.norm() + eps)).item()
    )
    
    brain.zero_grad(set_to_none=True)
    dqn.zero_grad(set_to_none=True)
    
    return cos_all, cos_brain, cos_dqn
