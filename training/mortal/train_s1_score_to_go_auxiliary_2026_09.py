"""Training runner for S1 score_to_go auxiliary multiseed experiment (variant-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.mainline_dataloader import FileDatasetsIter
from training.mortal.objective import compute_objective_losses
from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import (
    ADAM_BETAS,
    ADAM_EPS,
    AUX_WEIGHT,
    BATCH_SIZE,
    CQL_MIN_Q_WEIGHT,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FILE_BATCH_SIZE,
    GAMMA,
    HEAD_LR,
    HEAD_WEIGHT_DECAY,
    LEARNING_RATE,
    MAIN_REWARD_MODE,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    RANK_PTS,
    ROW_IDENTITY_FIELDS,
    S1_TRAINING_DIR,
    SCORE_AUX_HEAD_IN_FEATURES,
    SCORE_AUX_LOSS_WEIGHT,
    SCORE_AUX_MSE_FACTOR,
    STEPS_START,
    STEPS_TARGET,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEEDS,
    WEIGHT_DECAY,
    ContractError,
    check_directory_empty_or_nonexistent,
    init_score_to_go_head,
    native_path,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    sha256_file,
    update_auxiliary_target_digest,
    update_row_identity_digest,
    validate_s1_seed_set,
)

logger = logging.getLogger("s1_training")


class ScoreToGoHead(nn.Module):
    """Independent linear head mapping Brain's 1024-d phi to one score_to_go prediction."""

    def __init__(self, in_features: int = SCORE_AUX_HEAD_IN_FEATURES):
        super().__init__()
        self.net = nn.Linear(in_features, 1, bias=False)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).squeeze(-1)


# Main Q target domain: centered final_rank_mc values.
ALLOWED_Q_TARGET_VALUES = {3.0, 1.0, -1.0, -3.0}


def _build_models(
    k0_path: Path,
    device: str,
    training_seed: int,
) -> tuple[model.Brain, model.DQN, model.AuxNet, ScoreToGoHead]:
    """Reconstruct the formal Brain/DQN/AuxNet from K0 plus a fresh ScoreToGoHead.

    The head is initialized via the contract's single canonical
    init_score_to_go_head (Normal(0, 0.01), local generator seeded by the
    training seed); global/dataloader RNG state is untouched.
    """
    state = torch.load(k0_path, map_location="cpu")

    brain = model.Brain(version=4, conv_channels=192, num_blocks=40)
    dqn = model.DQN(version=4)
    aux_net = model.AuxNet((4,))

    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    aux_net.load_state_dict(state["aux_net"])

    brain.to(device).train()
    dqn.to(device).train()
    aux_net.to(device).train()
    brain.freeze_bn(False)

    head = ScoreToGoHead().to(device).train()
    init_score_to_go_head(head, training_seed)
    return brain, dqn, aux_net, head


def _build_optimizer(
    k0_path: Path,
    brain: model.Brain,
    dqn: model.DQN,
    aux_net: model.AuxNet,
    head: ScoreToGoHead,
) -> torch.optim.AdamW:
    """Restore K0's 410 parent moments exactly, then append one fresh head param group."""
    state = torch.load(k0_path, map_location="cpu")

    all_models = (brain, dqn, aux_net)
    decay_params = []
    no_decay_params = []
    for m in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in m.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)) and name.endswith("weight"):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))

    # Parent groups must reproduce K0's exact 410 moments first.
    param_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LEARNING_RATE,
        weight_decay=0,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    optimizer.load_state_dict(state["optimizer"])

    # Enforce current learning rate on parent groups.
    for g in optimizer.param_groups:
        g["lr"] = LEARNING_RATE

    # Append a fresh head group with no optimizer state.
    optimizer.add_param_group({
        "params": list(head.parameters()),
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
    })
    return optimizer


def _build_dataloader(
    file_index_path: Path,
    seed: int,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build reproducible ext_mortal-only dataloader with the 7th score_to_go target field."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    raw_files = torch.load(file_index_path, map_location="cpu")["file_list"]
    file_list = [str(native_path(f)) for f in raw_files]

    dataset = FileDatasetsIter(
        version=4,
        file_list=file_list,
        pts=RANK_PTS,
        oracle=False,
        file_batch_size=FILE_BATCH_SIZE,
        reserve_ratio=0,
        player_names=list(TRAINABLE_PLAYER_NAMES),
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        reward_mode=MAIN_REWARD_MODE,
        include_score_to_go_target=True,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def _verify_optimizer_structure(optimizer: torch.optim.AdamW) -> None:
    """Fail-closed: 3 groups sized [165, 245, 1], 410 parent moments, fresh head group config."""
    if len(optimizer.param_groups) != 3:
        raise ContractError(f"Expected 3 optimizer param groups, got {len(optimizer.param_groups)}")
    sizes = [len(g["params"]) for g in optimizer.param_groups]
    if sizes != [165, 245, 1]:
        raise ContractError(f"Optimizer param group sizes {sizes} != [165, 245, 1]")
    if len(optimizer.state) != 410:
        raise ContractError(f"Optimizer must restore exactly 410 parent moments, got {len(optimizer.state)}")
    head_group = optimizer.param_groups[2]
    if head_group.get("lr") != HEAD_LR or head_group.get("weight_decay") != HEAD_WEIGHT_DECAY:
        raise ContractError(
            f"Head group config mismatch: lr={head_group.get('lr')} wd={head_group.get('weight_decay')}"
        )
    for p in head_group["params"]:
        if p in optimizer.state:
            raise ContractError("Head parameter must start with fresh (empty) optimizer state")


def _verify_q_target_values(q_target_mc: torch.Tensor) -> None:
    """Fail-closed: every main Q target value must be in {+3, +1, -1, -3}."""
    unique = torch.unique(q_target_mc.detach().cpu().to(torch.float64))
    for value in unique.tolist():
        if not any(abs(value - allowed) < 1e-6 for allowed in ALLOWED_Q_TARGET_VALUES):
            raise ContractError(
                f"Main Q target contains value {value} outside the final_rank_mc domain {sorted(ALLOWED_Q_TARGET_VALUES)}"
            )


def _verify_auxiliary_target(score_to_go_target: torch.Tensor) -> None:
    """Fail-closed: auxiliary target must be finite and within [-1, +1]."""
    t = score_to_go_target.detach()
    if not bool(torch.isfinite(t).all()):
        raise ContractError("Auxiliary target contains non-finite values")
    if float(t.min()) < -1.0 - 1e-6 or float(t.max()) > 1.0 + 1e-6:
        raise ContractError(
            f"Auxiliary target out of [-1, +1]: min={float(t.min())}, max={float(t.max())}"
        )


def _verify_score_loss_gradient_routing(
    brain: model.Brain,
    dqn: model.DQN,
    aux_net: model.AuxNet,
    head: ScoreToGoHead,
    obs: torch.Tensor,
    score_to_go_target: torch.Tensor,
    *,
    probe_rows: int = 8,
) -> None:
    """First-batch autograd verification of the score loss gradient routing.

    Runs BEFORE the full-batch forward on a deterministic prefix (the first
    `probe_rows` rows) so only one small transient probe graph exists; the
    graph and its tensors are released immediately after verification. The
    isolated score loss must route finite non-zero gradients into Brain and
    ScoreToGoHead parameters; DQN and the original AuxNet must receive
    None/zero gradients.
    """
    # Backward on the probe accumulates into .grad; clear everything first and
    # restore a clean state afterwards so training is untouched.
    saved_grads: list[tuple[torch.nn.Parameter, torch.Tensor | None]] = []
    for module in (brain, dqn, aux_net, head):
        for p in module.parameters():
            saved_grads.append((p, p.grad))
            p.grad = None

    n = min(probe_rows, obs.shape[0], score_to_go_target.shape[0])
    obs_probe = obs[:n].detach().to(dtype=torch.float32)
    target_probe = score_to_go_target[:n].detach()

    phi = brain(obs_probe)
    loss = SCORE_AUX_MSE_FACTOR * torch.nn.functional.mse_loss(head(phi), target_probe)
    loss.backward()
    del loss, phi  # release the probe graph immediately

    def _grad_magnitude(p: torch.nn.Parameter) -> float:
        return 0.0 if p.grad is None else float(p.grad.detach().abs().sum())

    brain_total = sum(_grad_magnitude(p) for p in brain.parameters())
    head_total = sum(_grad_magnitude(p) for p in head.parameters())
    if brain_total == 0.0:
        raise ContractError("Score loss does not route a non-zero gradient into Brain parameters")
    if head_total == 0.0:
        raise ContractError("Score loss does not route a non-zero gradient into ScoreToGoHead parameters")
    for owner, module in (("dqn", dqn), ("aux_net", aux_net)):
        for pname, p in module.named_parameters():
            if p.grad is not None and float(p.grad.detach().abs().sum()) != 0.0:
                raise ContractError(f"Score loss leaked gradient into {owner}.{pname}")
    for owner, module in (("brain", brain), ("head", head)):
        for pname, p in module.named_parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                raise ContractError(f"Non-finite score-loss gradient in {owner}.{pname}")

    # Restore pre-verification grad state.
    for p, g in saved_grads:
        p.grad = g


def _verify_parameters_finite(brain, dqn, aux_net, head) -> None:
    """Fail-closed: all model parameters finite."""
    for name, module in (("brain", brain), ("dqn", dqn), ("aux_net", aux_net), ("head", head)):
        for pname, p in module.named_parameters():
            if not bool(torch.isfinite(p).all()):
                raise ContractError(f"Non-finite parameter {pname} in {name}")


def train_s1_aux_variant(
    seed: int,
    device: str = "cuda",
    output_dir: Path = S1_TRAINING_DIR,
) -> tuple[Path, str, list[dict[str, Any]], str, str, dict[str, bool]]:
    """Train exactly 400 steps of the S1 auxiliary variant on one seed.

    Returns (checkpoint_path, checkpoint_sha, step_logs, row_identity_digest,
    auxiliary_target_digest, science_gates).
    """
    k0_path, k0_sha = resolve_k0_checkpoint()
    m0_index_path, _ = resolve_m0_dataset_index()
    # Fail-closed binding to the frozen R2 Control arm before training.
    resolve_r2_control_checkpoint(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain, dqn, aux_net, head = _build_models(k0_path, device, training_seed=seed)
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net, head)
    _verify_optimizer_structure(optimizer)
    dataloader = _build_dataloader(m0_index_path, seed=seed)
    data_iter = iter(dataloader)

    science_gates: dict[str, bool] = {
        "main_q_target_final_rank_mc_verified": True,
        "score_loss_routed_to_brain_and_head_only": True,
        "optimizer_410_parent_moments_plus_fresh_head_group": True,
        "exact_step_counts_verified": True,
        "all_losses_and_parameters_finite": True,
    }

    step_logs: list[dict[str, Any]] = []
    row_digest = hashlib.sha256()
    aux_digest = hashlib.sha256()
    t0 = time.time()

    for step_idx in range(1, OPTIMIZER_STEPS + 1):
        batch = next(data_iter)
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, score_to_go_target = batch

        # Rolling digests over all 400 batches.
        update_row_identity_digest(
            row_digest,
            obs=obs,
            actions=actions,
            masks=masks,
            steps_to_done=steps_to_done,
            player_ranks=player_ranks,
        )
        update_auxiliary_target_digest(aux_digest, score_to_go_target)

        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device)
        score_to_go_target = score_to_go_target.to(dtype=torch.float32, device=device)

        # Per-batch science gates: Q target domain and auxiliary target bounds.
        _verify_auxiliary_target(score_to_go_target)

        # Main Q target: final_rank_mc only; score_to_go NEVER enters q_target_mc.
        q_target_mc = (float(GAMMA) ** steps_to_done * kyoku_rewards).to(torch.float32)
        _verify_q_target_values(q_target_mc)

        # First batch: autograd verification of score-loss gradient routing on a
        # deterministic 8-row prefix, BEFORE the full-batch forward so only one
        # small transient probe graph exists; released inside the verifier.
        if step_idx == 1:
            _verify_score_loss_gradient_routing(brain, dqn, aux_net, head, obs, score_to_go_target)

        phi = brain(obs)
        q_out = dqn(phi, masks)
        (next_rank_logits,) = aux_net(phi)
        score_pred = head(phi)

        main_losses = compute_objective_losses(
            q_out=q_out,
            masks=masks,
            actions=actions,
            q_target_mc=q_target_mc,
            next_rank_logits=next_rank_logits,
            player_ranks=player_ranks,
            mode=OBJECTIVE_MODE,
            cql_weight=CQL_MIN_Q_WEIGHT,
            aux_weight=AUX_WEIGHT,
        )
        score_aux_loss = SCORE_AUX_MSE_FACTOR * torch.nn.functional.mse_loss(score_pred, score_to_go_target)
        total_loss = main_losses["total_loss"] + SCORE_AUX_LOSS_WEIGHT * score_aux_loss

        for lname, lval in (
            ("total", total_loss),
            ("value", main_losses["value_loss"]),
            ("cql", main_losses["cql_loss"]),
            ("next_rank", main_losses["next_rank_loss"]),
            ("score_aux", score_aux_loss),
        ):
            if not bool(torch.isfinite(lval)):
                science_gates["all_losses_and_parameters_finite"] = False
                raise ContractError(f"Non-finite {lname} loss at step {step_idx}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for pname, p in brain.named_parameters():
            if not bool(torch.isfinite(p).all()):
                science_gates["all_losses_and_parameters_finite"] = False
                raise ContractError(f"Non-finite brain parameter {pname} at step {step_idx}")
        for pname, p in head.named_parameters():
            if not bool(torch.isfinite(p).all()):
                science_gates["all_losses_and_parameters_finite"] = False
                raise ContractError(f"Non-finite head parameter {pname} at step {step_idx}")

        if step_idx % 100 == 0 or step_idx == OPTIMIZER_STEPS:
            logger.info(
                "[Seed %d | aux_variant] Step %d/%d (step %d) | total: %.4f | value: %.4f | cql: %.4f | score_aux: %.6f",
                seed,
                step_idx,
                OPTIMIZER_STEPS,
                STEPS_START + step_idx,
                float(total_loss.item()),
                float(main_losses["value_loss"].item()),
                float(main_losses["cql_loss"].item()),
                float(score_aux_loss.item()),
            )

        step_logs.append({
            "step": STEPS_START + step_idx,
            "step_in_pilot": step_idx,
            "total_loss": float(total_loss.item()),
            "value_loss": float(main_losses["value_loss"].item()),
            "cql_loss": float(main_losses["cql_loss"].item()),
            "next_rank_loss": float(main_losses["next_rank_loss"].item()),
            "score_aux_loss": float(score_aux_loss.item()),
        })

    if len(step_logs) != OPTIMIZER_STEPS:
        science_gates["exact_step_counts_verified"] = False
        raise ContractError(f"Executed {len(step_logs)} steps, expected exactly {OPTIMIZER_STEPS}")

    _verify_parameters_finite(brain, dqn, aux_net, head)

    elapsed = time.time() - t0
    logger.info("[Seed %d | aux_variant] Completed 400 optimizer steps in %.2f seconds", seed, elapsed)

    checkpoint_name = f"mortal_aux_variant_70400_seed_{seed}.pth"
    checkpoint_path = output_dir / checkpoint_name
    save_state = {
        "mortal": brain.state_dict(),
        "current_dqn": dqn.state_dict(),
        "aux_net": aux_net.state_dict(),
        "score_aux_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": STEPS_TARGET,
        "condition": "score_to_go_auxiliary",
        "training_seed": seed,
        "experiment_id": EXPERIMENT_ID,
        "parent_model_sha256": k0_sha,
        "config": {
            "control": {"version": 4, "online": False, "batch_size": BATCH_SIZE},
            "resnet": {"conv_channels": 192, "num_blocks": 40},
            "objective": {"mode": OBJECTIVE_MODE},
            "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
            "reward": {"mode": MAIN_REWARD_MODE},
            "score_auxiliary": {
                "head": "ScoreToGoHead(1024, 1, bias=False)",
                "loss_weight": SCORE_AUX_LOSS_WEIGHT,
                "mse_factor": SCORE_AUX_MSE_FACTOR,
                "target_range": [-1.0, 1.0],
                "excluded_from_q_target": True,
            },
            "env": {"pts": list(RANK_PTS), "gamma": GAMMA},
        },
    }
    torch.save(save_state, checkpoint_path)
    ckpt_sha = sha256_file(checkpoint_path)

    return checkpoint_path, ckpt_sha, step_logs, row_digest.hexdigest(), aux_digest.hexdigest(), science_gates


def run_s1_training(
    seeds: list[int] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = S1_TRAINING_DIR,
) -> dict[str, Any]:
    """Execute variant-only 400-step training for all 3 seeds under S1 protocol."""
    target_seeds = validate_s1_seed_set(seeds)
    check_directory_empty_or_nonexistent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, k0_sha = resolve_k0_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    checkpoints_manifest: dict[str, Any] = {}
    row_identity_by_seed: dict[str, Any] = {}
    all_rows_match_r2 = True
    # Science gates are computed per-seed by the runner; any False fails closed.
    science_gate_results: dict[str, bool] = {
        "main_q_target_final_rank_mc_verified": True,
        "score_loss_routed_to_brain_and_head_only": True,
        "optimizer_410_parent_moments_plus_fresh_head_group": True,
        "exact_step_counts_verified": True,
    }

    for s in target_seeds:
        r2c_path, r2c_sha, r2c_row_digest = resolve_r2_control_checkpoint(s)

        logger.info("================ S1 Seed %d (AuxVariant: final_rank_mc main + score_to_go aux) ================", s)
        var_path, var_sha, _var_logs, var_row_digest, var_aux_digest, seed_science_gates = train_s1_aux_variant(
            seed=s, device=device, output_dir=output_dir
        )
        for gate_name, gate_value in seed_science_gates.items():
            if gate_name in science_gate_results:
                science_gate_results[gate_name] = science_gate_results[gate_name] and gate_value

        matches = var_row_digest == r2c_row_digest
        if not matches:
            all_rows_match_r2 = False

        row_identity_by_seed[f"seed_{s}"] = {
            "r2_control_sha256": r2c_row_digest,
            "aux_variant_sha256": var_row_digest,
            "matches_r2_control": matches,
            "auxiliary_target_sha256": var_aux_digest,
        }

        checkpoints_manifest[f"seed_{s}"] = {
            "r2_control": {
                "name": r2c_path.name,
                "path": str(r2c_path),
                "sha256": r2c_sha,
            },
            "aux_variant": {
                "name": var_path.name,
                "path": str(var_path),
                "sha256": var_sha,
            },
        }

    hard_gates: dict[str, bool] = {
        "k0_parent_verified": True,
        "m0_dataset_verified": True,
        "r2_control_checkpoints_verified": True,
        "all_3_seeds_completed": (len(checkpoints_manifest) == len(target_seeds)),
        "all_3_variant_checkpoints_saved": all(
            Path(data["aux_variant"]["path"]).exists() for data in checkpoints_manifest.values()
        ),
        "all_seeds_row_identity_matches_r2_control": all_rows_match_r2,
        "main_q_target_final_rank_mc_verified": science_gate_results["main_q_target_final_rank_mc_verified"],
        "score_loss_routed_to_brain_and_head_only": science_gate_results["score_loss_routed_to_brain_and_head_only"],
        "optimizer_410_parent_moments_plus_fresh_head_group": science_gate_results["optimizer_410_parent_moments_plus_fresh_head_group"],
        "exact_step_counts_verified": science_gate_results["exact_step_counts_verified"],
    }

    if set(hard_gates.keys()) != set(EXPECTED_TRAINING_HARD_GATES):
        raise ContractError(f"Training hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_TRAINING_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Training hard gate failed: {hard_gates}")

    manifest = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "objective": {
            "mode": OBJECTIVE_MODE,
            "value_statistic": OBJECTIVE_VALUE_STATISTIC,
            "preference_loss": "existing_cql",
        },
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [float(v) for v in RANK_PTS]},
        "score_auxiliary": {
            "loss_weight": SCORE_AUX_LOSS_WEIGHT,
            "mse_factor": SCORE_AUX_MSE_FACTOR,
            "target_range": [-1.0, 1.0],
            "excluded_from_q_target": True,
        },
        "training_config": {
            "training_seeds": target_seeds,
            "steps_start": STEPS_START,
            "steps_target": STEPS_TARGET,
            "optimizer_steps": OPTIMIZER_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "head_learning_rate": HEAD_LR,
            "head_weight_decay": HEAD_WEIGHT_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "cql_min_q_weight": CQL_MIN_Q_WEIGHT,
            "aux_weight": AUX_WEIGHT,
            "score_aux_weight": SCORE_AUX_LOSS_WEIGHT,
            "gamma": GAMMA,
            "device": device,
        },
        "checkpoints": checkpoints_manifest,
        "row_identity": {
            "fields": [name for name, _ in ROW_IDENTITY_FIELDS],
            "excluded_field": "kyoku_rewards",
            "by_seed": row_identity_by_seed,
        },
        "hard_gates": hard_gates,
        "verdict": "training_completed" if all(hard_gates.values()) else "training_failed",
    }

    manifest_path = output_dir / "s1_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=S1_TRAINING_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_s1_training(device=args.device, output_dir=args.output_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
