"""Read-only gradient-scale calibration for the frozen T1 policy-anchor pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party/Mortal/mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party/Mortal/mortal"))

import model  # noqa: E402

from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: E402
from training.mortal.objective import compute_objective_losses  # noqa: E402
from training.mortal.t1_k0_policy_anchor_contract_2026_09 import (  # noqa: E402
    ANCHOR_DIRECTION,
    ANCHOR_LOSS,
    ANCHOR_TEMPERATURE,
    AUX_WEIGHT,
    BATCH_SIZE,
    CALIBRATION_BATCHES_PER_SEED,
    CALIBRATION_SCHEMA,
    CQL_MIN_Q_WEIGHT,
    EXPERIMENT_ID,
    FILE_BATCH_SIZE,
    GAMMA,
    MAIN_REWARD_MODE,
    OBJECTIVE_MODE,
    RANK_PTS,
    TARGET_GRADIENT_RATIO,
    T1_CALIBRATION_DIR,
    T1_CALIBRATION_PATH,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_SEEDS,
    ContractError,
    check_directory_empty_or_nonexistent,
    legal_policy_kl_rows,
    native_path,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    sha256_file,
    update_row_identity_digest,
    validate_t1_seed_set,
    verify_calibration,
)

logger = logging.getLogger("t1_calibration")


def _load_model(checkpoint: Path, device: str, *, training: bool) -> tuple[model.Brain, model.DQN, model.AuxNet]:
    state = torch.load(checkpoint, map_location="cpu")
    brain = model.Brain(version=4, conv_channels=192, num_blocks=40)
    dqn = model.DQN(version=4)
    aux_net = model.AuxNet((4,))
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    aux_net.load_state_dict(state["aux_net"])
    brain.to(device)
    dqn.to(device)
    aux_net.to(device)
    if training:
        brain.train().freeze_bn(False)
        dqn.train()
        aux_net.train()
    else:
        brain.eval()
        dqn.eval()
        aux_net.eval()
        for module in (brain, dqn, aux_net):
            module.requires_grad_(False)
    return brain, dqn, aux_net


def _dataloader(index_path: Path, seed: int) -> DataLoader:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw_files = torch.load(index_path, map_location="cpu")["file_list"]
    dataset = FileDatasetsIter(
        version=4,
        file_list=[str(native_path(path)) for path in raw_files],
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
    return DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=0, pin_memory=False)


def _tensor_digest(brain: model.Brain, dqn: model.DQN) -> str:
    digest = hashlib.sha256()
    for module_name, module in (("brain", brain), ("dqn", dqn)):
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            digest.update(f"{module_name}.{name}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    for grad in grads:
        if grad is not None:
            squared = squared + grad.detach().to(torch.float64).square().sum()
    value = float(torch.sqrt(squared).cpu())
    if not math.isfinite(value) or value <= 0:
        raise ContractError(f"Gradient norm must be positive and finite, got {value}")
    return value


def calibrate_t1_lambda(
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = T1_CALIBRATION_DIR,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Select one lambda from existing R2 controls without training or game outcomes."""
    target_seeds = validate_t1_seed_set(seeds)
    check_directory_empty_or_nonexistent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k0_path, k0_sha = resolve_k0_checkpoint()
    index_path, index_sha = resolve_m0_dataset_index()
    parent_brain, parent_dqn, _ = _load_model(k0_path, device, training=False)
    parent_digest_before = _tensor_digest(parent_brain, parent_dqn)

    by_seed: dict[str, Any] = {}
    lambda_values: list[float] = []
    for seed in target_seeds:
        control_path, control_sha, _ = resolve_r2_control_checkpoint(seed)
        brain, dqn, aux_net = _load_model(control_path, device, training=True)
        batch = next(iter(_dataloader(index_path, seed)))
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
        row_digest = hashlib.sha256()
        update_row_identity_digest(
            row_digest,
            obs=obs,
            actions=actions,
            masks=masks,
            steps_to_done=steps_to_done,
            player_ranks=player_ranks,
        )
        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.int64)
        masks = masks.to(device=device, dtype=torch.bool)
        steps_to_done = steps_to_done.to(device=device, dtype=torch.int64)
        kyoku_rewards = kyoku_rewards.to(device=device, dtype=torch.float64)
        player_ranks = player_ranks.to(device=device, dtype=torch.int64)

        brain.train().freeze_bn(False)
        phi = brain(obs)
        q_out = dqn(phi, masks)
        (rank_logits,) = aux_net(phi)
        q_target = (float(GAMMA) ** steps_to_done * kyoku_rewards).to(torch.float32)
        base_loss = compute_objective_losses(
            q_out=q_out,
            masks=masks,
            actions=actions,
            q_target_mc=q_target,
            next_rank_logits=rank_logits,
            player_ranks=player_ranks,
            mode=OBJECTIVE_MODE,
            cql_weight=CQL_MIN_Q_WEIGHT,
            aux_weight=AUX_WEIGHT,
        )["total_loss"]

        brain.eval()
        q_current_anchor = dqn(brain(obs), masks)
        with torch.inference_mode():
            q_parent = parent_dqn(parent_brain(obs), masks)
        anchor_loss = legal_policy_kl_rows(q_current_anchor, q_parent, masks).mean()

        params = list(brain.parameters()) + list(dqn.parameters())
        base_norm = _gradient_norm(base_loss, params)
        anchor_norm = _gradient_norm(anchor_loss, params)
        lambda_value = TARGET_GRADIENT_RATIO * base_norm / anchor_norm
        if not math.isfinite(lambda_value) or lambda_value <= 0:
            raise ContractError(f"Invalid lambda for seed {seed}: {lambda_value}")
        lambda_values.append(lambda_value)
        by_seed[f"seed_{seed}"] = {
            "r2_control": {"path": str(control_path), "sha256": control_sha},
            "r2_control_sha256": control_sha,
            "calibration_rows": BATCH_SIZE * CALIBRATION_BATCHES_PER_SEED,
            "calibration_row_sha256": row_digest.hexdigest(),
            "base_loss": float(base_loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "base_gradient_norm": base_norm,
            "anchor_gradient_norm": anchor_norm,
            "lambda_for_target_ratio": lambda_value,
        }
        logger.info("seed=%d base_grad=%.6g anchor_grad=%.6g lambda=%.6g", seed, base_norm, anchor_norm, lambda_value)
        del brain, dqn, aux_net, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_lambda = float(np.median(np.asarray(lambda_values, dtype=np.float64)))
    parent_digest_after = _tensor_digest(parent_brain, parent_dqn)
    hard_gates = {
        "k0_parent_verified": k0_sha == sha256_file(k0_path),
        "m0_dataset_verified": index_sha == sha256_file(index_path),
        "all_3_r2_controls_verified": len(by_seed) == len(TRAINING_SEEDS),
        "exact_fixed_seed_set": target_seeds == TRAINING_SEEDS,
        "positive_finite_gradient_norms": all(math.isfinite(v) and v > 0 for v in lambda_values),
        "selected_lambda_is_median": selected_lambda == float(np.median(lambda_values)),
        "frozen_scorer_bit_exact": parent_digest_before == parent_digest_after,
        "no_optimizer_step": True,
        "no_reward_or_evaluation_signal": True,
    }
    if not all(hard_gates.values()):
        raise ContractError(f"Calibration hard gate failed: {hard_gates}")
    result = {
        "schema": CALIBRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol": {
            "anchor_loss": ANCHOR_LOSS,
            "direction": ANCHOR_DIRECTION,
            "temperature": ANCHOR_TEMPERATURE,
            "target_gradient_ratio": TARGET_GRADIENT_RATIO,
            "aggregation": "median_of_three_seed_specific_lambdas",
            "calibration_source": "existing_frozen_R2_Control_70400",
            "calibration_batches_per_seed": CALIBRATION_BATCHES_PER_SEED,
            "student_anchor_forward_mode": "eval",
            "uses_reward_or_evaluation_signal": False,
        },
        "parent_model": {"path": str(k0_path), "sha256": k0_sha},
        "dataset": {"path": str(index_path), "sha256": index_sha},
        "by_seed": by_seed,
        "selected_lambda": selected_lambda,
        "hard_gates": hard_gates,
        "verdict": "calibration_completed",
    }
    output_path = output_dir / T1_CALIBRATION_PATH.name
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_calibration(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=T1_CALIBRATION_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(calibrate_t1_lambda(device=args.device, output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
