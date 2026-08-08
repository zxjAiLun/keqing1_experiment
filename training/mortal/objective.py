"""Project-owned Mortal objective variants.

The network and inference contract stay unchanged. This module only controls
which statistic of the legal Q table receives the final-rank MC target.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_OBJECTIVE_MODE = "behavior_action_mc"
SUPPORTED_OBJECTIVE_MODES = {DEFAULT_OBJECTIVE_MODE, "legal_mean_mc"}


def objective_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    objective = config.get("objective", {})
    if not isinstance(objective, Mapping):
        raise ValueError("[objective] must be a TOML table")
    mode = str(objective.get("mode", DEFAULT_OBJECTIVE_MODE))
    if mode not in SUPPORTED_OBJECTIVE_MODES:
        raise ValueError(
            f"unsupported objective mode {mode!r}; expected one of {sorted(SUPPORTED_OBJECTIVE_MODES)}"
        )
    return {
        "mode": mode,
        "value_statistic": "mean_legal_q" if mode == "legal_mean_mc" else "behavior_action_q",
        "preference_loss": "existing_cql",
        "reward_mode": str(config.get("reward", {}).get("mode", "final_rank_mc")),
    }


def compute_objective_losses(
    *,
    q_out: Tensor,
    masks: Tensor,
    actions: Tensor,
    q_target_mc: Tensor,
    next_rank_logits: Tensor,
    player_ranks: Tensor,
    mode: str = DEFAULT_OBJECTIVE_MODE,
    cql_weight: float,
    aux_weight: float,
) -> dict[str, Tensor]:
    """Compute value, CQL and auxiliary losses plus shared Q diagnostics.

    ``q_out`` is expected to be the DQN output after illegal actions have been
    set to ``-inf``. ``masks`` remains explicit so the legal mean never depends
    on the numerical representation of illegal actions.
    """

    if mode not in SUPPORTED_OBJECTIVE_MODES:
        raise ValueError(f"unsupported objective mode: {mode}")
    if q_out.ndim != 2 or masks.shape != q_out.shape:
        raise ValueError(f"q_out/masks shape mismatch: {tuple(q_out.shape)} vs {tuple(masks.shape)}")
    batch_size = q_out.shape[0]
    if actions.shape != (batch_size,) or q_target_mc.shape != (batch_size,):
        raise ValueError("actions and q_target_mc must be one value per batch row")
    if not bool(masks[torch.arange(batch_size, device=masks.device), actions].all().item()):
        raise ValueError("behavior action is outside the legal action mask")

    row_index = torch.arange(batch_size, device=q_out.device)
    behavior_q = q_out[row_index, actions]
    legal_count_int = masks.sum(dim=-1)
    legal_count = legal_count_int.clamp_min(1).to(q_out.dtype)
    legal_q_sum = q_out.masked_fill(~masks, 0.0).sum(dim=-1)
    legal_q_mean = legal_q_sum / legal_count

    if mode == "legal_mean_mc":
        value_prediction = legal_q_mean
    else:
        value_prediction = behavior_q
    value_loss = 0.5 * F.mse_loss(value_prediction, q_target_mc)

    # Keep the existing CQL preference term exactly unchanged. DQN has already
    # masked illegal actions to -inf, so logsumexp only sees legal actions.
    preference_loss = q_out.logsumexp(-1).mean() - behavior_q.mean()
    next_rank_loss = F.cross_entropy(next_rank_logits, player_ranks)
    total_loss = value_loss + float(cql_weight) * preference_loss + float(aux_weight) * next_rank_loss

    legal_q_centered = q_out - legal_q_mean.unsqueeze(-1)
    legal_q_std = torch.sqrt(
        legal_q_centered.masked_fill(~masks, 0.0).square().sum(dim=-1) / legal_count
    )
    centered_advantage_abs_mean = (
        legal_q_centered.abs().masked_fill(~masks, 0.0).sum(dim=-1) / legal_count
    )
    top_two = q_out.masked_fill(~masks, -torch.inf).topk(k=min(2, q_out.shape[-1]), dim=-1).values
    raw_margin = top_two[:, 0] - top_two[:, 1] if top_two.shape[-1] >= 2 else torch.zeros_like(legal_q_mean)
    greedy_margin = torch.where(
        legal_count_int >= 2,
        raw_margin,
        torch.zeros_like(legal_q_mean),
    )

    return {
        "value_loss": value_loss,
        "dqn_loss": value_loss,
        "preference_loss": preference_loss,
        "cql_loss": preference_loss,
        "next_rank_loss": next_rank_loss,
        "total_loss": total_loss,
        "value_prediction": value_prediction,
        "legal_q_mean": legal_q_mean,
        "legal_q_std": legal_q_std,
        "behavior_q": behavior_q,
        "behavior_centered_advantage": behavior_q - legal_q_mean,
        "greedy_margin": greedy_margin,
        "value_target_abs_error": (value_prediction - q_target_mc).abs(),
        "value_abs_err": (value_prediction - q_target_mc).abs(),
        "centered_advantage_abs_mean": centered_advantage_abs_mean,
    }
