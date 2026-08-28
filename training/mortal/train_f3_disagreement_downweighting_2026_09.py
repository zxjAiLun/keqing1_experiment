"""Training runner for F3 disagreement-downweighting pilot (variant-only)."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.mainline_dataloader import FileDatasetsIter
from training.mortal.objective import compute_objective_losses
from training.mortal.f3_disagreement_downweighting_contract_2026_09 import (
    ADAM_BETAS,
    ADAM_EPS,
    AGREEMENT_WEIGHT,
    ALLOWED_Q_TARGET_VALUES,
    AUX_WEIGHT,
    BATCH_SIZE,
    CQL_MIN_Q_WEIGHT,
    DISAGREEMENT_WEIGHT,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FILE_BATCH_SIZE,
    F3_TRAINING_DIR,
    GAMMA,
    LEARNING_RATE,
    MAIN_REWARD_MODE,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    RANK_PTS,
    ROW_IDENTITY_FIELDS,
    STEPS_START,
    STEPS_TARGET,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEEDS,
    WEIGHT_DECAY,
    ContractError,
    check_directory_empty_or_nonexistent,
    compute_f3_row_weights,
    native_path,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    sha256_file,
    update_row_identity_digest,
    validate_f3_seed_set,
    validate_f3_weight_values,
)

logger = logging.getLogger("f3_training")


def _build_models(k0_path: Path, device: str) -> tuple[model.Brain, model.DQN, model.AuxNet]:
    """Reconstruct the formal Brain/DQN/AuxNet from K0 (identical to R2 Control)."""
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
    return brain, dqn, aux_net


def _build_frozen_scorer(k0_path: Path, device: str) -> tuple[model.Brain, model.DQN]:
    """Frozen K0 scorer in eval mode; its parameters are never registered in any optimizer."""
    state = torch.load(k0_path, map_location="cpu")

    scorer_brain = model.Brain(version=4, conv_channels=192, num_blocks=40)
    scorer_dqn = model.DQN(version=4)
    scorer_brain.load_state_dict(state["mortal"])
    scorer_dqn.load_state_dict(state["current_dqn"])

    scorer_brain.to(device).eval()
    scorer_dqn.to(device).eval()
    for p in scorer_brain.parameters():
        p.requires_grad_(False)
    for p in scorer_dqn.parameters():
        p.requires_grad_(False)
    return scorer_brain, scorer_dqn


def _build_optimizer(
    k0_path: Path,
    brain: model.Brain,
    dqn: model.DQN,
    aux_net: model.AuxNet,
) -> torch.optim.AdamW:
    """Restore K0's exact 410 parent moments (two param groups [165, 245], no new group)."""
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
    for g in optimizer.param_groups:
        g["lr"] = LEARNING_RATE
    return optimizer


def _build_dataloader(
    file_index_path: Path,
    seed: int,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Reproducible ext_mortal-only dataloader consuming the same base stream as R2 Control."""
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
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def _verify_optimizer_structure(optimizer: torch.optim.AdamW) -> None:
    """Fail-closed: 2 groups sized [165, 245], exactly 410 restored parent moments."""
    if len(optimizer.param_groups) != 2:
        raise ContractError(f"Expected 2 optimizer param groups, got {len(optimizer.param_groups)}")
    sizes = [len(g["params"]) for g in optimizer.param_groups]
    if sizes != [165, 245]:
        raise ContractError(f"Optimizer param group sizes {sizes} != [165, 245]")
    if len(optimizer.state) != 410:
        raise ContractError(f"Optimizer must restore exactly 410 parent moments, got {len(optimizer.state)}")


def _scorer_parameter_digest(brain: model.Brain, dqn: model.DQN) -> str:
    """Bit-level digest over every frozen scorer tensor value."""
    digest = hashlib.sha256()
    for module_name, module in (("brain", brain), ("dqn", dqn)):
        for pname, p in module.named_parameters():
            digest.update(f"{module_name}.{pname}".encode("utf-8"))
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        for bname, buf in module.named_buffers():
            digest.update(f"{module_name}.{bname}".encode("utf-8"))
            digest.update(buf.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compute_high_signal_mask(
    scorer_brain: model.Brain,
    scorer_dqn: model.DQN,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """disagreement = actions != argmax legal Q of frozen K0 (no margin threshold; F1 parity)."""
    if not bool(masks.gather(1, actions.unsqueeze(1)).all()):
        raise ContractError("Behavior action illegal (mask false) in batch")
    with torch.inference_mode():
        phi = scorer_brain(obs)
        q = scorer_dqn(phi, masks)
        greedy = q.argmax(dim=1)
    disagreement = (actions != greedy).to(torch.bool)
    return disagreement.detach().cpu().numpy(), greedy.detach().cpu().numpy()


def _verify_q_target_values(q_target_mc: torch.Tensor) -> None:
    """Fail-closed: every main Q target value must be in {+3, +1, -1, -3}."""
    unique = torch.unique(q_target_mc.detach().cpu().to(torch.float64))
    for value in unique.tolist():
        if not any(abs(value - allowed) < 1e-6 for allowed in ALLOWED_Q_TARGET_VALUES):
            raise ContractError(
                f"Main Q target contains value {value} outside the final_rank_mc domain {sorted(ALLOWED_Q_TARGET_VALUES)}"
            )


def _verify_parameters_finite(brain, dqn, aux_net) -> None:
    for name, module in (("brain", brain), ("dqn", dqn), ("aux_net", aux_net)):
        for pname, p in module.named_parameters():
            if not bool(torch.isfinite(p).all()):
                raise ContractError(f"Non-finite parameter {pname} in {name}")


def compute_weighted_f3_losses(
    *,
    q_out: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    q_target_mc: torch.Tensor,
    next_rank_logits: torch.Tensor,
    player_ranks: torch.Tensor,
    row_weights: torch.Tensor,
    cql_weight: float,
    aux_weight: float,
) -> dict[str, torch.Tensor]:
    """F3 weighted objective: weighted_mean over per-row value/CQL/next-rank losses.

    weighted_mean(x) = sum(weight * x_row) / sum(weight)

    total = weighted_mean(value_loss_row)
          + cql_weight * weighted_mean(cql_loss_row)
          + aux_weight * weighted_mean(next_rank_loss_row)

    Per-row formulas match the unweighted objective exactly (0.5*MSE_row,
    logsumexp_row - behavior_q_row, CE_row). Row weights are constant with
    respect to autograd, so no importance correction is applied or needed.
    """
    batch_size = q_out.shape[0]
    if row_weights.shape != (batch_size,):
        raise ContractError(f"row_weights shape {tuple(row_weights.shape)} != ({batch_size},)")
    if bool((row_weights <= 0).any()) or not bool(torch.isfinite(row_weights).all()):
        raise ContractError("row_weights must be positive and finite")
    weight_sum = row_weights.sum()

    row_index = torch.arange(batch_size, device=q_out.device)
    behavior_q = q_out[row_index, actions]

    value_row = 0.5 * F.mse_loss(q_out[row_index, actions], q_target_mc, reduction="none")
    cql_row = q_out.logsumexp(-1) - behavior_q
    next_rank_row = F.cross_entropy(next_rank_logits, player_ranks, reduction="none")

    value_loss = (row_weights * value_row).sum() / weight_sum
    preference_loss = (row_weights * cql_row).sum() / weight_sum
    next_rank_loss = (row_weights * next_rank_row).sum() / weight_sum
    total_loss = value_loss + float(cql_weight) * preference_loss + float(aux_weight) * next_rank_loss

    return {
        "value_loss": value_loss,
        "dqn_loss": value_loss,
        "preference_loss": preference_loss,
        "cql_loss": preference_loss,
        "next_rank_loss": next_rank_loss,
        "total_loss": total_loss,
        "behavior_q": behavior_q,
        "weight_sum": weight_sum,
    }


def train_f3_downweighted_variant(
    seed: int,
    device: str = "cuda",
    output_dir: Path = F3_TRAINING_DIR,
) -> tuple[Path, str, str, dict[str, Any]]:
    """Train exactly 400 steps of the F3 downweighted variant on one seed.

    Every base row is used exactly once per batch; only the loss weighting
    changes. Returns (checkpoint_path, checkpoint_sha, base_row_digest, downweighting_stats).
    """
    k0_path, k0_sha = resolve_k0_checkpoint()
    m0_index_path, _ = resolve_m0_dataset_index()
    resolve_r2_control_checkpoint(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain, dqn, aux_net = _build_models(k0_path, device)
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)
    _verify_optimizer_structure(optimizer)

    scorer_brain, scorer_dqn = _build_frozen_scorer(k0_path, device)
    scorer_digest_before = _scorer_parameter_digest(scorer_brain, scorer_dqn)

    dataloader = _build_dataloader(m0_index_path, seed=seed)
    data_iter = iter(dataloader)

    base_digest = hashlib.sha256()
    per_step: list[dict[str, Any]] = []
    t0 = time.time()

    for step_idx in range(1, OPTIMIZER_STEPS + 1):
        batch = next(data_iter)
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch

        # Base stream digest: reward-excluded fields over all 512 rows; must equal R2 Control.
        update_row_identity_digest(
            base_digest,
            obs=obs,
            actions=actions,
            masks=masks,
            steps_to_done=steps_to_done,
            player_ranks=player_ranks,
        )

        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device)

        # Disagreement mask and row weights from the frozen K0 scorer.
        disagreement, _greedy = compute_high_signal_mask(scorer_brain, scorer_dqn, obs, actions, masks)
        row_weights_np = compute_f3_row_weights(disagreement)
        disagreement_count = int(disagreement.sum())
        weight_sum_expected = float(
            disagreement_count * DISAGREEMENT_WEIGHT + (obs.shape[0] - disagreement_count) * AGREEMENT_WEIGHT
        )
        row_weights = torch.as_tensor(row_weights_np, dtype=torch.float32, device=device)
        if abs(float(row_weights.sum().item()) - weight_sum_expected) > 1e-6:
            raise ContractError(f"Row weight sum mismatch at step {step_idx}")
        weights_unique = sorted({float(w) for w in np.unique(row_weights_np)})
        try:
            validate_f3_weight_values(weights_unique)
        except ContractError:
            raise ContractError(f"Illegal weight values at step {step_idx}: {weights_unique}") from None

        q_target_mc = (float(GAMMA) ** steps_to_done * kyoku_rewards).to(torch.float32)
        _verify_q_target_values(q_target_mc)

        phi = brain(obs)
        q_out = dqn(phi, masks)
        (next_rank_logits,) = aux_net(phi)

        losses = compute_weighted_f3_losses(
            q_out=q_out,
            masks=masks,
            actions=actions,
            q_target_mc=q_target_mc,
            next_rank_logits=next_rank_logits,
            player_ranks=player_ranks,
            row_weights=row_weights,
            cql_weight=CQL_MIN_Q_WEIGHT,
            aux_weight=AUX_WEIGHT,
        )
        total_loss = losses["total_loss"]
        for lname, lval in (
            ("total", total_loss),
            ("value", losses["value_loss"]),
            ("cql", losses["cql_loss"]),
            ("next_rank", losses["next_rank_loss"]),
        ):
            if not bool(torch.isfinite(lval)):
                raise ContractError(f"Non-finite {lname} loss at step {step_idx}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for pname, p in brain.named_parameters():
            if not bool(torch.isfinite(p).all()):
                raise ContractError(f"Non-finite brain parameter {pname} at step {step_idx}")

        per_step.append({
            "step": step_idx,
            "rows_used": int(obs.shape[0]),
            "disagreement_count": disagreement_count,
            "disagreement_rate": disagreement_count / float(obs.shape[0]),
            "weights_unique": weights_unique,
            "weight_sum": weight_sum_expected,
        })

        if step_idx % 100 == 0 or step_idx == OPTIMIZER_STEPS:
            logger.info(
                "[Seed %d | downweighted_variant] Step %d/%d (step %d) | total: %.4f | value: %.4f | cql: %.4f | disagree: %d (%.1f%%)",
                seed, step_idx, OPTIMIZER_STEPS, STEPS_START + step_idx,
                float(total_loss.item()), float(losses["value_loss"].item()), float(losses["cql_loss"].item()),
                disagreement_count, 100.0 * disagreement_count / float(obs.shape[0]),
            )

    if len(per_step) != OPTIMIZER_STEPS:
        raise ContractError(f"Executed {len(per_step)} steps, expected exactly {OPTIMIZER_STEPS}")

    scorer_digest_after = _scorer_parameter_digest(scorer_brain, scorer_dqn)
    if scorer_digest_before != scorer_digest_after:
        raise ContractError("Frozen scorer parameters changed during training")

    _verify_parameters_finite(brain, dqn, aux_net)

    ds_counts = [rec["disagreement_count"] for rec in per_step]
    downweighting_stats = {
        "batches": len(per_step),
        "rows_per_batch": BATCH_SIZE,
        "disagreement_count_min": int(min(ds_counts)),
        "disagreement_count_max": int(max(ds_counts)),
        "disagreement_rate_mean": float(np.mean([rec["disagreement_rate"] for rec in per_step])),
        "per_step": per_step,
    }

    elapsed = time.time() - t0
    logger.info("[Seed %d | downweighted_variant] Completed 400 optimizer steps in %.2f seconds", seed, elapsed)

    checkpoint_name = f"mortal_downweighted_variant_70400_seed_{seed}.pth"
    checkpoint_path = output_dir / checkpoint_name
    save_state = {
        "mortal": brain.state_dict(),
        "current_dqn": dqn.state_dict(),
        "aux_net": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": STEPS_TARGET,
        "condition": "disagreement_downweighting_without_resampling",
        "training_seed": seed,
        "experiment_id": EXPERIMENT_ID,
        "parent_model_sha256": k0_sha,
        "config": {
            "control": {"version": 4, "online": False, "batch_size": BATCH_SIZE},
            "resnet": {"conv_channels": 192, "num_blocks": 40},
            "objective": {"mode": OBJECTIVE_MODE},
            "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
            "reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [float(v) for v in RANK_PTS]},
            "disagreement_downweighting": {
                "disagreement_weight": DISAGREEMENT_WEIGHT,
                "agreement_weight": AGREEMENT_WEIGHT,
                "resampling": False,
                "row_usage": "each_row_exactly_once",
                "normalization": "batch_weight_sum",
                "applied_to": ["value_loss_row", "cql_loss_row", "next_rank_loss_row"],
                "q_margin_threshold": None,
            },
            "env": {"pts": list(RANK_PTS), "gamma": GAMMA},
        },
    }
    torch.save(save_state, checkpoint_path)
    ckpt_sha = sha256_file(checkpoint_path)

    return checkpoint_path, ckpt_sha, base_digest.hexdigest(), downweighting_stats


def run_f3_training(
    seeds: list[int] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = F3_TRAINING_DIR,
) -> dict[str, Any]:
    """Execute variant-only 400-step training for all 3 seeds under F3 protocol."""
    target_seeds = validate_f3_seed_set(seeds)
    check_directory_empty_or_nonexistent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, k0_sha = resolve_k0_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    checkpoints_manifest: dict[str, Any] = {}
    row_identity_by_seed: dict[str, Any] = {}
    all_rows_match_r2 = True
    all_row_usage_ok = True
    all_weights_ok = True

    for s in target_seeds:
        r2c_path, r2c_sha, r2c_row_digest = resolve_r2_control_checkpoint(s)

        logger.info(
            "================ F3 Seed %d (DownweightedVariant: rows once, disagreement weight 0.5) ================",
            s,
        )
        var_path, var_sha, var_base_digest, downweighting_stats = train_f3_downweighted_variant(
            seed=s, device=device, output_dir=output_dir
        )

        matches = var_base_digest == r2c_row_digest
        if not matches:
            all_rows_match_r2 = False
        row_usage_ok = all(rec["rows_used"] == BATCH_SIZE for rec in downweighting_stats["per_step"])
        all_row_usage_ok = all_row_usage_ok and row_usage_ok
        weights_ok = True
        for rec in downweighting_stats["per_step"]:
            try:
                validate_f3_weight_values(rec["weights_unique"])
            except ContractError:
                weights_ok = False
                break
        all_weights_ok = all_weights_ok and weights_ok

        row_identity_by_seed[f"seed_{s}"] = {
            "r2_control_sha256": r2c_row_digest,
            "base_row_sha256": var_base_digest,
            "matches_r2_control": matches,
            "downweighting_stats": downweighting_stats,
        }

        checkpoints_manifest[f"seed_{s}"] = {
            "r2_control": {
                "name": r2c_path.name,
                "path": str(r2c_path),
                "sha256": r2c_sha,
            },
            "downweighted_variant": {
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
            Path(data["downweighted_variant"]["path"]).exists() for data in checkpoints_manifest.values()
        ),
        "all_seeds_base_row_identity_matches_r2_control": all_rows_match_r2,
        "no_resampling_row_usage_exact": all_row_usage_ok,
        "weights_0_5_1_normalized_all_batches": all_weights_ok,
        "scorer_parameters_bit_exact": True,
        "main_q_target_final_rank_mc_verified": True,
        "optimizer_preserved_k0_moments_410": True,
        "exact_step_counts_verified": True,
    }

    if set(hard_gates.keys()) != set(EXPECTED_TRAINING_HARD_GATES):
        raise ContractError(f"Training hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_TRAINING_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Training hard gate failed: {hard_gates}")

    manifest = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "frozen_scorer": {
            "name": "K0_70k",
            "sha256": k0_sha,
            "mode": "eval",
            "amp": False,
            "dtype": "float32",
            "parameters_bit_exact": True,
        },
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "objective": {
            "mode": OBJECTIVE_MODE,
            "value_statistic": OBJECTIVE_VALUE_STATISTIC,
            "preference_loss": "existing_cql",
        },
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [float(v) for v in RANK_PTS]},
        "disagreement_downweighting": {
            "disagreement_weight": DISAGREEMENT_WEIGHT,
            "agreement_weight": AGREEMENT_WEIGHT,
            "resampling": False,
            "row_usage": "each_row_exactly_once",
            "normalization": "batch_weight_sum",
            "applied_to": ["value_loss_row", "cql_loss_row", "next_rank_loss_row"],
            "q_margin_threshold": None,
            "uses_reward_or_eval_signal": False,
        },
        "training_config": {
            "training_seeds": target_seeds,
            "steps_start": STEPS_START,
            "steps_target": STEPS_TARGET,
            "optimizer_steps": OPTIMIZER_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "cql_min_q_weight": CQL_MIN_Q_WEIGHT,
            "aux_weight": AUX_WEIGHT,
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

    manifest_path = output_dir / "f3_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=F3_TRAINING_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_f3_training(device=args.device, output_dir=args.output_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
