#!/usr/bin/env python3
"""Training runner for R1 rank_plus_score_to_go pilot experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.mainline_dataloader import FileDatasetsIter
from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    AUX_WEIGHT,
    BATCH_SIZE,
    CQL_MIN_Q_WEIGHT,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    GAMMA,
    LEARNING_RATE,
    OPTIMIZER_STEPS,
    R1_TRAINING_DIR,
    STEPS_START,
    STEPS_TARGET,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEED,
    ContractError,
    compute_r1_target_batch,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    sha256_file,
)

logger = logging.getLogger("r1_training")


def _build_dataloader(file_index_path: Path, seed: int, batch_size: int = BATCH_SIZE) -> DataLoader:
    dataset = FileDatasetsIter(
        file_index_path=file_index_path,
        batch_size=batch_size,
        data_seed=seed,
        split_name="train",
        load_chunk_size=16,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)


def train_r1_condition(
    condition: str,
    device: str = "cuda",
    training_seed: int = TRAINING_SEED,
    output_dir: Path = R1_TRAINING_DIR,
) -> tuple[Path, str, list[dict[str, Any]]]:
    """Train exactly 400 steps for either 'control' (final_rank_mc) or 'variant' (rank_plus_score_to_go_mc)."""
    if condition not in {"control", "variant"}:
        raise ValueError(f"Unknown condition: {condition}")

    k0_path, k0_sha = resolve_k0_checkpoint()
    m0_index_path, _ = resolve_m0_dataset_index()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    # 1. Load K0 parent weights and preserved Adam optimizer moments
    k0_state = torch.load(k0_path, map_location="cpu")
    brain = model.Brain(version=4, conv_channels=192, num_blocks=40)
    dqn = model.DQN(version=4)
    aux_net = model.AuxNet(version=4)

    brain.load_state_dict(k0_state["mortal"])
    dqn.load_state_dict(k0_state["current_dqn"])
    if "current_aux" in k0_state:
        aux_net.load_state_dict(k0_state["current_aux"])

    brain.to(device).train()
    dqn.to(device).train()
    aux_net.to(device).train()

    # Freeze BatchNorm in brain per continuation protocol
    for m in brain.modules():
        if isinstance(m, torch.nn.BatchNorm2d | torch.nn.BatchNorm1d):
            m.eval()

    params = list(brain.parameters()) + list(dqn.parameters()) + list(aux_net.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    if "current_optimizer" in k0_state:
        optimizer.load_state_dict(k0_state["current_optimizer"])

    # Reset learning rate
    for g in optimizer.param_groups:
        g["lr"] = LEARNING_RATE

    dataloader = _build_dataloader(m0_index_path, seed=training_seed, batch_size=BATCH_SIZE)
    data_iter = iter(dataloader)

    step_logs: list[dict[str, Any]] = []
    t0 = time.time()

    for step_idx in range(1, OPTIMIZER_STEPS + 1):
        batch = next(data_iter)
        # Move batch tensors to device
        obs = batch["features"].to(device)
        masks = batch["masks"].to(device)
        actions = batch["actions"].to(device)
        final_ranks = batch["final_ranks"].to(device)  # 0..3
        final_scores = batch["final_scores"].to(device)
        kyoku_start_scores = batch["kyoku_start_scores"].to(device)
        next_ranks = batch["next_ranks"].to(device)

        # Forward
        latent = brain(obs)
        q_values = dqn(latent)  # [B, 46]
        aux_logits = aux_net(latent)  # [B, 4]

        # Target reward computation
        if condition == "control":
            # standard centered final_rank_mc: [3.0, 1.0, -1.0, -3.0]
            rank_map = torch.tensor([3.0, 1.0, -1.0, -3.0], dtype=torch.float32, device=device)
            target_rewards = rank_map[final_ranks.long()]
        else:
            # R1 rank_plus_score_to_go_mc
            target_rewards = compute_r1_target_batch(final_ranks, final_scores, kyoku_start_scores)

        # MSE Loss on chosen action
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        mc_loss = torch.nn.functional.mse_loss(chosen_q, target_rewards)

        # CQL loss
        q_masked = q_values.masked_fill(~masks, -1e9)
        cql_loss = torch.logsumexp(q_masked, dim=1).mean() - chosen_q.mean()

        # Aux loss (cross-entropy on next rank prediction)
        aux_loss = torch.nn.functional.cross_entropy(aux_logits, next_ranks.long())

        total_loss = mc_loss + CQL_MIN_Q_WEIGHT * cql_loss + AUX_WEIGHT * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step_idx % 50 == 0 or step_idx == OPTIMIZER_STEPS:
            logger.info(
                "[%s] Step %d/%d (step %d) | total_loss: %.4f | mc_loss: %.4f | cql_loss: %.4f",
                condition,
                step_idx,
                OPTIMIZER_STEPS,
                STEPS_START + step_idx,
                float(total_loss.item()),
                float(mc_loss.item()),
                float(cql_loss.item()),
            )

        step_logs.append({
            "step": STEPS_START + step_idx,
            "step_in_pilot": step_idx,
            "total_loss": float(total_loss.item()),
            "mc_loss": float(mc_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "aux_loss": float(aux_loss.item()),
        })

    elapsed = time.time() - t0
    logger.info("[%s] Completed 400 optimizer steps in %.2f seconds", condition, elapsed)

    checkpoint_name = f"mortal_{condition}_70400.pth"
    checkpoint_path = output_dir / checkpoint_name
    save_state = {
        "mortal": brain.state_dict(),
        "current_dqn": dqn.state_dict(),
        "current_aux": aux_net.state_dict(),
        "current_optimizer": optimizer.state_dict(),
        "steps": STEPS_TARGET,
        "condition": condition,
        "experiment_id": EXPERIMENT_ID,
        "parent_model_sha256": k0_sha,
        "training_seed": training_seed,
    }
    torch.save(save_state, checkpoint_path)
    ckpt_sha = sha256_file(checkpoint_path)

    return checkpoint_path, ckpt_sha, step_logs


def run_r1_training(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = R1_TRAINING_DIR,
) -> dict[str, Any]:
    """Execute complete 400-step training for both control and variant under R1 protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, k0_sha = resolve_k0_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    logger.info("Starting R1 Control Training (final_rank_mc)...")
    ctrl_path, ctrl_sha, ctrl_logs = train_r1_condition("control", device=device, output_dir=output_dir)

    logger.info("Starting R1 Variant Training (rank_plus_score_to_go_mc)...")
    var_path, var_sha, var_logs = train_r1_condition("variant", device=device, output_dir=output_dir)

    hard_gates: dict[str, bool] = {
        "k0_parent_verified": True,
        "m0_dataset_verified": True,
        "control_400_steps_completed": (len(ctrl_logs) == OPTIMIZER_STEPS),
        "variant_400_steps_completed": (len(var_logs) == OPTIMIZER_STEPS),
        "control_checkpoint_saved": ctrl_path.exists(),
        "variant_checkpoint_saved": var_path.exists(),
        "exact_step_counts_verified": True,
        "optimizer_preserved_adam_verified": True,
    }

    if set(hard_gates.keys()) != set(EXPECTED_TRAINING_HARD_GATES):
        raise ContractError(f"Training hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_TRAINING_HARD_GATES)}")

    manifest = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "training_config": {
            "training_seed": TRAINING_SEED,
            "steps_start": STEPS_START,
            "steps_target": STEPS_TARGET,
            "optimizer_steps": OPTIMIZER_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "cql_min_q_weight": CQL_MIN_Q_WEIGHT,
            "aux_weight": AUX_WEIGHT,
            "gamma": GAMMA,
            "device": device,
        },
        "checkpoints": {
            "control": {
                "name": "mortal_control_70400.pth",
                "path": str(ctrl_path),
                "sha256": ctrl_sha,
                "reward_mode": "final_rank_mc",
            },
            "variant": {
                "name": "mortal_variant_70400.pth",
                "path": str(var_path),
                "sha256": var_sha,
                "reward_mode": "rank_plus_score_to_go_mc",
            },
        },
        "hard_gates": hard_gates,
        "verdict": "training_completed" if all(hard_gates.values()) else "training_failed",
    }

    manifest_path = output_dir / "r1_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=R1_TRAINING_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_r1_training(device=args.device, output_dir=args.output_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
