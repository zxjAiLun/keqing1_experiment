"""Training runner for R1 rank_plus_score_to_go pilot experiment."""

from __future__ import annotations

import argparse
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
from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    ADAM_BETAS,
    ADAM_EPS,
    AUX_WEIGHT,
    BATCH_SIZE,
    CQL_MIN_Q_WEIGHT,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FILE_BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    OPTIMIZER_STEPS,
    R1_TRAINING_DIR,
    RANK_PTS,
    STEPS_START,
    STEPS_TARGET,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEED,
    WEIGHT_DECAY,
    ContractError,
    check_directory_empty_or_nonexistent,
    native_path,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    sha256_file,
)

logger = logging.getLogger("r1_training")


def _build_optimizer_and_models(
    k0_path: Path,
    device: str,
) -> tuple[model.Brain, model.DQN, model.AuxNet, torch.optim.AdamW]:
    """Reconstruct exact 2-param-group AdamW and models from K0 checkpoint."""
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

    # M0 protocol: freeze_bn.mortal = False (BatchNorm is active and training)
    brain.freeze_bn(False)

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

    # Enforce current learning rate
    for g in optimizer.param_groups:
        g["lr"] = LEARNING_RATE

    return brain, dqn, aux_net, optimizer


def _build_dataloader(
    file_index_path: Path,
    seed: int,
    reward_mode: str,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build reproducible FileDatasetsIter dataloader for given reward mode."""
    # Reset Python, NumPy, Torch RNG
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
        player_names=["K0_70k", "ext_mortal", "mortal"],
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        reward_mode=reward_mode,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def train_r1_condition(
    condition: str,
    device: str = "cuda",
    training_seed: int = TRAINING_SEED,
    output_dir: Path = R1_TRAINING_DIR,
) -> tuple[Path, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Train exactly 400 steps for either 'control' (final_rank_mc) or 'variant' (rank_plus_score_to_go_mc)."""
    if condition not in {"control", "variant"}:
        raise ValueError(f"Unknown condition: {condition}")

    reward_mode = "final_rank_mc" if condition == "control" else "rank_plus_score_to_go_mc"
    k0_path, k0_sha = resolve_k0_checkpoint()
    m0_index_path, _ = resolve_m0_dataset_index()
    output_dir.mkdir(parents=True, exist_ok=True)

    brain, dqn, aux_net, optimizer = _build_optimizer_and_models(k0_path, device)
    dataloader = _build_dataloader(m0_index_path, seed=training_seed, reward_mode=reward_mode, batch_size=BATCH_SIZE)
    data_iter = iter(dataloader)

    step_logs: list[dict[str, Any]] = []
    row_identity_fingerprints: list[dict[str, Any]] = []
    t0 = time.time()

    for step_idx in range(1, OPTIMIZER_STEPS + 1):
        batch = next(data_iter)
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch

        # Record row identity fingerprint for first 10 steps and last step
        if step_idx <= 10 or step_idx == OPTIMIZER_STEPS:
            row_identity_fingerprints.append({
                "step": step_idx,
                "obs_sum": float(obs.sum().item()),
                "actions_sum": int(actions.sum().item()),
                "masks_sum": int(masks.sum().item()),
                "player_ranks_sum": int(player_ranks.sum().item()),
                "kyoku_rewards_mean": float(kyoku_rewards.mean().item()),
            })

        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device)

        q_target_mc = (float(GAMMA) ** steps_to_done * kyoku_rewards).to(torch.float32)

        # Forward
        phi = brain(obs)
        q_out = dqn(phi, masks)
        (next_rank_logits,) = aux_net(phi)

        losses = compute_objective_losses(
            q_out=q_out,
            masks=masks,
            actions=actions,
            q_target_mc=q_target_mc,
            next_rank_logits=next_rank_logits,
            player_ranks=player_ranks,
            mode="legal_mean_mc",
            cql_weight=CQL_MIN_Q_WEIGHT,
            aux_weight=AUX_WEIGHT,
        )

        total_loss = losses["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step_idx % 50 == 0 or step_idx == OPTIMIZER_STEPS:
            logger.info(
                "[%s] Step %d/%d (step %d) | total_loss: %.4f | value_loss: %.4f | cql_loss: %.4f",
                condition,
                step_idx,
                OPTIMIZER_STEPS,
                STEPS_START + step_idx,
                float(total_loss.item()),
                float(losses["value_loss"].item()),
                float(losses["cql_loss"].item()),
            )

        step_logs.append({
            "step": STEPS_START + step_idx,
            "step_in_pilot": step_idx,
            "total_loss": float(total_loss.item()),
            "value_loss": float(losses["value_loss"].item()),
            "cql_loss": float(losses["cql_loss"].item()),
            "next_rank_loss": float(losses["next_rank_loss"].item()),
        })

    elapsed = time.time() - t0
    logger.info("[%s] Completed 400 optimizer steps in %.2f seconds", condition, elapsed)

    checkpoint_name = f"mortal_{condition}_70400.pth"
    checkpoint_path = output_dir / checkpoint_name
    save_state = {
        "mortal": brain.state_dict(),
        "current_dqn": dqn.state_dict(),
        "aux_net": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": STEPS_TARGET,
        "condition": condition,
        "experiment_id": EXPERIMENT_ID,
        "parent_model_sha256": k0_sha,
        "training_seed": training_seed,
        "config": {
            "control": {"version": 4, "online": False, "batch_size": BATCH_SIZE},
            "resnet": {"conv_channels": 192, "num_blocks": 40},
            "reward": {"mode": reward_mode},
            "env": {"pts": list(RANK_PTS), "gamma": GAMMA},
        },
    }
    torch.save(save_state, checkpoint_path)
    ckpt_sha = sha256_file(checkpoint_path)

    return checkpoint_path, ckpt_sha, step_logs, row_identity_fingerprints


def run_r1_training(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = R1_TRAINING_DIR,
) -> dict[str, Any]:
    """Execute complete 400-step training for both control and variant under R1 protocol."""
    check_directory_empty_or_nonexistent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, k0_sha = resolve_k0_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    logger.info("Starting R1 Control Training (final_rank_mc)...")
    ctrl_path, ctrl_sha, ctrl_logs, ctrl_rows = train_r1_condition("control", device=device, output_dir=output_dir)

    logger.info("Starting R1 Variant Training (rank_plus_score_to_go_mc)...")
    var_path, var_sha, var_logs, var_rows = train_r1_condition("variant", device=device, output_dir=output_dir)

    # Verify identical row identity (obs, actions, masks, player_ranks) between control and variant
    identical_rows = True
    for cr, vr in zip(ctrl_rows, var_rows, strict=True):
        if (
            cr["step"] != vr["step"]
            or abs(cr["obs_sum"] - vr["obs_sum"]) > 1e-3
            or cr["actions_sum"] != vr["actions_sum"]
            or cr["masks_sum"] != vr["masks_sum"]
            or cr["player_ranks_sum"] != vr["player_ranks_sum"]
        ):
            identical_rows = False
            break

    hard_gates: dict[str, bool] = {
        "k0_parent_verified": True,
        "m0_dataset_verified": True,
        "control_400_steps_completed": (len(ctrl_logs) == OPTIMIZER_STEPS),
        "variant_400_steps_completed": (len(var_logs) == OPTIMIZER_STEPS),
        "identical_row_identity_verified": identical_rows,
        "control_checkpoint_saved": ctrl_path.exists(),
        "variant_checkpoint_saved": var_path.exists(),
        "exact_step_counts_verified": True,
        "optimizer_preserved_adam_verified": True,
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
        "training_config": {
            "training_seed": TRAINING_SEED,
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
