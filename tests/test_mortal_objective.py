from __future__ import annotations

import torch

from training.mortal.objective import compute_objective_losses, objective_contract_from_config


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    raw_q = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    masks = torch.tensor(
        [
            [True, True, False, True, False, False],
            [True, False, True, True, True, False],
            [False, True, True, False, True, True],
        ]
    )
    q_out = raw_q.masked_fill(~masks, -torch.inf)
    actions = torch.tensor([0, 3, 5])
    targets = torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64)
    next_rank_logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    player_ranks = torch.tensor([0, 2, 1])
    return q_out, masks, actions, targets, next_rank_logits, player_ranks


def _kwargs(q_out, masks, actions, targets, next_rank_logits, player_ranks):
    return {
        "q_out": q_out,
        "masks": masks,
        "actions": actions,
        "q_target_mc": targets,
        "next_rank_logits": next_rank_logits,
        "player_ranks": player_ranks,
        "cql_weight": 5.0,
        "aux_weight": 0.2,
    }


def test_control_matches_legacy_loss_formula() -> None:
    fixture = _fixture()
    q_out, masks, actions, targets, next_rank_logits, player_ranks = fixture
    result = compute_objective_losses(**_kwargs(*fixture), mode="behavior_action_mc")
    behavior_q = q_out[torch.arange(3), actions]
    legacy_value = 0.5 * torch.mean((behavior_q - targets) ** 2)
    legacy_cql = q_out.logsumexp(-1).mean() - behavior_q.mean()
    legacy_aux = torch.nn.functional.cross_entropy(next_rank_logits, player_ranks)
    assert torch.allclose(result["value_loss"], legacy_value)
    assert torch.allclose(result["preference_loss"], legacy_cql)
    assert torch.allclose(result["next_rank_loss"], legacy_aux)
    assert torch.allclose(
        result["total_loss"], legacy_value + 5.0 * legacy_cql + 0.2 * legacy_aux
    )
    assert torch.allclose(result["value_prediction"], behavior_q)


def test_legal_mean_uses_only_legal_actions() -> None:
    fixture = _fixture()
    result = compute_objective_losses(**_kwargs(*fixture), mode="legal_mean_mc")
    q_out, masks, _, _, _, _ = fixture
    expected = q_out.masked_fill(~masks, 0.0).sum(-1) / masks.sum(-1)
    assert torch.allclose(result["value_prediction"], expected)
    assert torch.allclose(result["legal_q_mean"], expected)


def test_single_legal_action_has_finite_zero_margin() -> None:
    q_out = torch.tensor([[2.0, -torch.inf, -torch.inf]], requires_grad=True)
    masks = torch.tensor([[True, False, False]])
    actions = torch.tensor([0])
    targets = torch.tensor([1.0])
    next_rank_logits = torch.zeros(1, 4, requires_grad=True)
    result = compute_objective_losses(
        q_out=q_out,
        masks=masks,
        actions=actions,
        q_target_mc=targets,
        next_rank_logits=next_rank_logits,
        player_ranks=torch.tensor([0]),
        mode="legal_mean_mc",
        cql_weight=5.0,
        aux_weight=0.2,
    )
    assert torch.isfinite(result["total_loss"])
    assert torch.isfinite(result["legal_q_mean"]).all()
    assert torch.isfinite(result["legal_q_std"]).all()
    assert torch.isfinite(result["centered_advantage_abs_mean"]).all()
    assert torch.isfinite(result["greedy_margin"]).all()
    assert result["greedy_margin"].item() == 0.0


def test_cql_is_invariant_to_per_row_common_offset() -> None:
    fixture = _fixture()
    kwargs = _kwargs(*fixture)
    original = compute_objective_losses(**kwargs, mode="legal_mean_mc")["preference_loss"]
    offsets = torch.tensor([[2.0], [-1.5], [0.75]], dtype=torch.float64)
    shifted = compute_objective_losses(
        **{**kwargs, "q_out": kwargs["q_out"] + offsets}, mode="legal_mean_mc"
    )["preference_loss"]
    assert torch.allclose(original, shifted, atol=1e-12, rtol=1e-12)


def test_gradient_separation_in_q_output_space() -> None:
    fixture = _fixture()
    q_out, masks, actions, targets, next_rank_logits, player_ranks = fixture
    value = compute_objective_losses(
        **_kwargs(*fixture), mode="legal_mean_mc"
    )["value_loss"]
    value_grad = torch.autograd.grad(value, q_out, retain_graph=True)[0]
    for row in range(q_out.shape[0]):
        legal = value_grad[row][masks[row]]
        assert torch.allclose(legal, legal[0].expand_as(legal))

    preference = compute_objective_losses(
        **_kwargs(*fixture), mode="legal_mean_mc"
    )["preference_loss"]
    preference_grad = torch.autograd.grad(preference, q_out)[0]
    row_sums = preference_grad.masked_fill(~masks, 0.0).sum(-1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-12, rtol=1e-12)


def test_objective_contract_separates_reward_from_objective() -> None:
    control = objective_contract_from_config({"reward": {"mode": "final_rank_mc"}})
    variant = objective_contract_from_config(
        {"reward": {"mode": "final_rank_mc"}, "objective": {"mode": "legal_mean_mc"}}
    )
    assert control["mode"] == "behavior_action_mc"
    assert control["reward_mode"] == variant["reward_mode"] == "final_rank_mc"
    assert variant["value_statistic"] == "mean_legal_q"
