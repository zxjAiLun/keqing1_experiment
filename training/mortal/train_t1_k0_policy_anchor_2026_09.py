"""Variant-only trainer and policy-drift audit for the T1 K0-anchor pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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
if str(REPO_ROOT / "third_party/Mortal/mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party/Mortal/mortal"))

import model  # noqa: E402

from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: E402
from training.mortal.objective import compute_objective_losses  # noqa: E402
from training.mortal.t1_k0_policy_anchor_contract_2026_09 import (  # noqa: E402
    ADAM_BETAS,
    ADAM_EPS,
    ALLOWED_Q_TARGET_VALUES,
    ANCHOR_DIRECTION,
    ANCHOR_LOSS,
    ANCHOR_TEMPERATURE,
    AUX_WEIGHT,
    BATCH_SIZE,
    CQL_MIN_Q_WEIGHT,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FILE_BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    MAIN_REWARD_MODE,
    MECHANISM_AUDIT_BATCHES,
    MECHANISM_AUDIT_ROWS,
    MECHANISM_AUDIT_SKIP_BATCHES,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    RANK_PTS,
    ROW_IDENTITY_FIELDS,
    STEPS_START,
    STEPS_TARGET,
    T1_CALIBRATION_PATH,
    T1_TRAINING_DIR,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_MANIFEST_SCHEMA,
    WEIGHT_DECAY,
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

logger = logging.getLogger("t1_training")


def _load_models(checkpoint: Path, device: str, *, training: bool, frozen: bool = False) -> tuple[model.Brain, model.DQN, model.AuxNet]:
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
    if frozen:
        for module in (brain, dqn, aux_net):
            module.requires_grad_(False)
    return brain, dqn, aux_net


def _build_optimizer(k0_path: Path, brain: model.Brain, dqn: model.DQN, aux_net: model.AuxNet) -> torch.optim.AdamW:
    state = torch.load(k0_path, map_location="cpu")
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for module in (brain, dqn, aux_net):
        params: dict[str, nn.Parameter] = {}
        to_decay: set[str] = set()
        for module_name, child in module.named_modules():
            for name, param in child.named_parameters(prefix=module_name, recurse=False):
                params[name] = param
                if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)) and name.endswith("weight"):
                    to_decay.add(name)
        decay_params.extend(params[name] for name in sorted(to_decay))
        no_decay_params.extend(params[name] for name in sorted(params.keys() - to_decay))
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY}, {"params": no_decay_params}],
        lr=LEARNING_RATE,
        weight_decay=0,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE
    if [len(group["params"]) for group in optimizer.param_groups] != [165, 245] or len(optimizer.state) != 410:
        raise ContractError("Optimizer must restore exact K0 groups [165,245] and 410 moments")
    return optimizer


def _build_dataloader(index_path: Path, seed: int) -> DataLoader:
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


def _scorer_digest(brain: model.Brain, dqn: model.DQN) -> str:
    digest = hashlib.sha256()
    for module_name, module in (("brain", brain), ("dqn", dqn)):
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            digest.update(f"{module_name}.{name}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _verify_q_targets(targets: torch.Tensor) -> None:
    for value in torch.unique(targets.detach().cpu().to(torch.float64)).tolist():
        if not any(abs(value - allowed) < 1e-6 for allowed in ALLOWED_Q_TARGET_VALUES):
            raise ContractError(f"Q target {value} is outside final_rank_mc domain")


def _eval_q(brain: model.Brain, dqn: model.DQN, obs: torch.Tensor, masks: torch.Tensor, *, preserve_training: bool) -> torch.Tensor:
    brain_was_training, dqn_was_training = brain.training, dqn.training
    brain.eval()
    dqn.eval()
    q = dqn(brain(obs), masks)
    if preserve_training and brain_was_training:
        brain.train().freeze_bn(False)
    if preserve_training and dqn_was_training:
        dqn.train()
    return q


def compute_t1_losses(
    *,
    base_losses: dict[str, torch.Tensor],
    q_current_anchor: torch.Tensor,
    q_parent: torch.Tensor,
    masks: torch.Tensor,
    anchor_lambda: float,
) -> dict[str, torch.Tensor]:
    if not math.isfinite(anchor_lambda) or anchor_lambda <= 0:
        raise ContractError(f"anchor_lambda must be positive and finite, got {anchor_lambda}")
    anchor_rows = legal_policy_kl_rows(q_current_anchor, q_parent, masks, ANCHOR_TEMPERATURE)
    anchor_loss = anchor_rows.mean()
    total = base_losses["total_loss"] + float(anchor_lambda) * anchor_loss
    if not bool(torch.isfinite(total)):
        raise ContractError("T1 total loss is non-finite")
    return {**base_losses, "anchor_loss": anchor_loss, "anchor_rows": anchor_rows, "total_loss_with_anchor": total}


def train_t1_anchor_variant(seed: int, anchor_lambda: float, device: str, output_dir: Path) -> tuple[Path, str, str, dict[str, Any]]:
    k0_path, k0_sha = resolve_k0_checkpoint()
    index_path, _ = resolve_m0_dataset_index()
    resolve_r2_control_checkpoint(seed)
    brain, dqn, aux_net = _load_models(k0_path, device, training=True)
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)
    parent_brain, parent_dqn, _ = _load_models(k0_path, device, training=False, frozen=True)
    scorer_before = _scorer_digest(parent_brain, parent_dqn)
    iterator = iter(_build_dataloader(index_path, seed))
    row_digest = hashlib.sha256()
    per_step: list[dict[str, Any]] = []
    started = time.time()

    for step_idx in range(1, OPTIMIZER_STEPS + 1):
        batch = next(iterator)
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
        update_row_identity_digest(row_digest, obs=obs, actions=actions, masks=masks, steps_to_done=steps_to_done, player_ranks=player_ranks)
        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.int64)
        masks = masks.to(device=device, dtype=torch.bool)
        steps_to_done = steps_to_done.to(device=device, dtype=torch.int64)
        kyoku_rewards = kyoku_rewards.to(device=device, dtype=torch.float64)
        player_ranks = player_ranks.to(device=device, dtype=torch.int64)
        q_target = (float(GAMMA) ** steps_to_done * kyoku_rewards).to(torch.float32)
        _verify_q_targets(q_target)

        brain.train().freeze_bn(False)
        dqn.train()
        aux_net.train()
        phi = brain(obs)
        q_base = dqn(phi, masks)
        (rank_logits,) = aux_net(phi)
        base_losses = compute_objective_losses(
            q_out=q_base,
            masks=masks,
            actions=actions,
            q_target_mc=q_target,
            next_rank_logits=rank_logits,
            player_ranks=player_ranks,
            mode=OBJECTIVE_MODE,
            cql_weight=CQL_MIN_Q_WEIGHT,
            aux_weight=AUX_WEIGHT,
        )
        q_current_anchor = _eval_q(brain, dqn, obs, masks, preserve_training=True)
        with torch.inference_mode():
            q_parent = parent_dqn(parent_brain(obs), masks)
        losses = compute_t1_losses(
            base_losses=base_losses,
            q_current_anchor=q_current_anchor,
            q_parent=q_parent,
            masks=masks,
            anchor_lambda=anchor_lambda,
        )
        optimizer.zero_grad()
        losses["total_loss_with_anchor"].backward()
        optimizer.step()

        record = {
            "step": step_idx,
            "rows_used": int(obs.shape[0]),
            "base_total_loss": float(base_losses["total_loss"].detach().cpu()),
            "anchor_kl": float(losses["anchor_loss"].detach().cpu()),
            "weighted_anchor_loss": float((anchor_lambda * losses["anchor_loss"]).detach().cpu()),
            "total_loss": float(losses["total_loss_with_anchor"].detach().cpu()),
        }
        if not all(math.isfinite(float(record[key])) for key in ("base_total_loss", "anchor_kl", "weighted_anchor_loss", "total_loss")):
            raise ContractError(f"Non-finite step metrics at seed={seed} step={step_idx}")
        per_step.append(record)
        if step_idx % 100 == 0:
            logger.info("seed=%d step=%d/%d total=%.5f base=%.5f anchor=%.7f", seed, step_idx, OPTIMIZER_STEPS, record["total_loss"], record["base_total_loss"], record["anchor_kl"])

    if _scorer_digest(parent_brain, parent_dqn) != scorer_before:
        raise ContractError("Frozen K0 scorer changed during training")
    for module_name, module in (("brain", brain), ("dqn", dqn), ("aux", aux_net)):
        for name, parameter in module.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise ContractError(f"Non-finite parameter {module_name}.{name}")

    checkpoint_path = output_dir / f"mortal_anchor_variant_70400_seed_{seed}.pth"
    torch.save(
        {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "aux_net": aux_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": STEPS_TARGET,
            "condition": "k0_legal_policy_anchor",
            "training_seed": seed,
            "experiment_id": EXPERIMENT_ID,
            "parent_model_sha256": k0_sha,
            "config": {
                "control": {"version": 4, "online": False, "batch_size": BATCH_SIZE},
                "resnet": {"conv_channels": 192, "num_blocks": 40},
                "objective": {"mode": OBJECTIVE_MODE},
                "reward": {"mode": MAIN_REWARD_MODE, "rank_pts": RANK_PTS},
                "policy_anchor": {"loss": ANCHOR_LOSS, "direction": ANCHOR_DIRECTION, "temperature": ANCHOR_TEMPERATURE, "lambda": anchor_lambda},
                "env": {"pts": RANK_PTS, "gamma": GAMMA},
            },
        },
        checkpoint_path,
    )
    stats = {
        "batches": len(per_step),
        "rows_per_batch": BATCH_SIZE,
        "elapsed_seconds": time.time() - started,
        "anchor_kl_first": per_step[0]["anchor_kl"],
        "anchor_kl_last": per_step[-1]["anchor_kl"],
        "anchor_kl_mean": float(np.mean([row["anchor_kl"] for row in per_step])),
        "per_step": per_step,
    }
    return checkpoint_path, sha256_file(checkpoint_path), row_digest.hexdigest(), stats


def _centered_values(q: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    counts = masks.sum(dim=1).to(q.dtype)
    means = q.masked_fill(~masks, 0).sum(dim=1) / counts
    return (q - means.unsqueeze(1)).masked_fill(~masks, 0)


def _margin(q: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = masks.sum(dim=1)
    top = q.masked_fill(~masks, -torch.inf).topk(k=2, dim=1).values
    return top[:, 0] - top[:, 1], counts >= 2


def audit_policy_drift(seed: int, variant_path: Path, device: str) -> dict[str, Any]:
    """Evaluate K0/control/variant on held-out deterministic batches 401..416."""
    k0_path, _ = resolve_k0_checkpoint()
    index_path, _ = resolve_m0_dataset_index()
    control_path, _, _ = resolve_r2_control_checkpoint(seed)
    parent_brain, parent_dqn, _ = _load_models(k0_path, device, training=False, frozen=True)
    control_brain, control_dqn, _ = _load_models(control_path, device, training=False, frozen=True)
    variant_brain, variant_dqn, _ = _load_models(variant_path, device, training=False, frozen=True)
    iterator = iter(_build_dataloader(index_path, seed))
    for _ in range(MECHANISM_AUDIT_SKIP_BATCHES):
        next(iterator)

    digest = hashlib.sha256()
    totals = {
        "rows": 0, "legal": 0, "margin_rows": 0,
        "control_kl": 0.0, "variant_kl": 0.0,
        "control_greedy_diff": 0, "variant_greedy_diff": 0,
        "control_centered_sq": 0.0, "variant_centered_sq": 0.0,
        "control_margin_abs": 0.0, "variant_margin_abs": 0.0,
    }
    with torch.inference_mode():
        for _ in range(MECHANISM_AUDIT_BATCHES):
            obs, actions, masks, steps_to_done, _rewards, player_ranks = next(iterator)
            update_row_identity_digest(digest, obs=obs, actions=actions, masks=masks, steps_to_done=steps_to_done, player_ranks=player_ranks)
            obs = obs.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.bool)
            qp = parent_dqn(parent_brain(obs), masks)
            qc = control_dqn(control_brain(obs), masks)
            qv = variant_dqn(variant_brain(obs), masks)
            rows = int(obs.shape[0])
            totals["rows"] += rows
            totals["control_kl"] += float(legal_policy_kl_rows(qc, qp, masks).sum().cpu())
            totals["variant_kl"] += float(legal_policy_kl_rows(qv, qp, masks).sum().cpu())
            gp = qp.argmax(dim=1)
            totals["control_greedy_diff"] += int((qc.argmax(dim=1) != gp).sum().cpu())
            totals["variant_greedy_diff"] += int((qv.argmax(dim=1) != gp).sum().cpu())
            cp = _centered_values(qp, masks)
            cc = _centered_values(qc, masks)
            cv = _centered_values(qv, masks)
            legal = int(masks.sum().cpu())
            totals["legal"] += legal
            totals["control_centered_sq"] += float((((cc - cp).square()) * masks).sum().cpu())
            totals["variant_centered_sq"] += float((((cv - cp).square()) * masks).sum().cpu())
            mp, eligible = _margin(qp, masks)
            mc, _ = _margin(qc, masks)
            mv, _ = _margin(qv, masks)
            totals["margin_rows"] += int(eligible.sum().cpu())
            totals["control_margin_abs"] += float((mc[eligible] - mp[eligible]).abs().sum().cpu())
            totals["variant_margin_abs"] += float((mv[eligible] - mp[eligible]).abs().sum().cpu())
    if totals["rows"] != MECHANISM_AUDIT_ROWS:
        raise ContractError(f"Mechanism audit rows {totals['rows']} != {MECHANISM_AUDIT_ROWS}")
    control = {
        "mean_kl_to_k0": totals["control_kl"] / totals["rows"],
        "greedy_disagreement_rate_to_k0": totals["control_greedy_diff"] / totals["rows"],
        "centered_advantage_rmse_to_k0": math.sqrt(totals["control_centered_sq"] / totals["legal"]),
        "mean_abs_margin_delta_to_k0": totals["control_margin_abs"] / totals["margin_rows"],
    }
    variant = {
        "mean_kl_to_k0": totals["variant_kl"] / totals["rows"],
        "greedy_disagreement_rate_to_k0": totals["variant_greedy_diff"] / totals["rows"],
        "centered_advantage_rmse_to_k0": math.sqrt(totals["variant_centered_sq"] / totals["legal"]),
        "mean_abs_margin_delta_to_k0": totals["variant_margin_abs"] / totals["margin_rows"],
    }
    directions = {
        "kl_lower": variant["mean_kl_to_k0"] < control["mean_kl_to_k0"],
        "greedy_disagreement_lower": variant["greedy_disagreement_rate_to_k0"] < control["greedy_disagreement_rate_to_k0"],
        "centered_advantage_rmse_lower": variant["centered_advantage_rmse_to_k0"] < control["centered_advantage_rmse_to_k0"],
    }
    return {
        "rows": totals["rows"],
        "row_sha256": digest.hexdigest(),
        "batches": MECHANISM_AUDIT_BATCHES,
        "skip_batches": MECHANISM_AUDIT_SKIP_BATCHES,
        "control": control,
        "variant": variant,
        "directions": directions,
        "all_directions_pass": all(directions.values()),
    }


def run_t1_training(seeds: list[int] | None = None, device: str = "cuda" if torch.cuda.is_available() else "cpu", output_dir: Path = T1_TRAINING_DIR, calibration_path: Path = T1_CALIBRATION_PATH) -> dict[str, Any]:
    target_seeds = validate_t1_seed_set(seeds)
    check_directory_empty_or_nonexistent(output_dir)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    anchor_lambda = verify_calibration(calibration)
    calibration_sha = sha256_file(calibration_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    k0_path, k0_sha = resolve_k0_checkpoint()
    index_path, index_sha = resolve_m0_dataset_index()
    checkpoints: dict[str, Any] = {}
    row_identity: dict[str, Any] = {}
    mechanism_by_seed: dict[str, Any] = {}
    scorer_exact = True

    for seed in target_seeds:
        control_path, control_sha, expected_row_sha = resolve_r2_control_checkpoint(seed)
        logger.info("starting T1 seed=%d lambda=%.9g", seed, anchor_lambda)
        variant_path, variant_sha, row_sha, stats = train_t1_anchor_variant(seed, anchor_lambda, device, output_dir)
        matches = row_sha == expected_row_sha
        row_identity[f"seed_{seed}"] = {
            "r2_control_sha256": expected_row_sha,
            "base_row_sha256": row_sha,
            "matches_r2_control": matches,
            "anchor_stats": stats,
        }
        checkpoints[f"seed_{seed}"] = {
            "r2_control": {"name": control_path.name, "path": str(control_path), "sha256": control_sha},
            "anchor_variant": {"name": variant_path.name, "path": str(variant_path), "sha256": variant_sha},
        }
        mechanism_by_seed[f"seed_{seed}"] = audit_policy_drift(seed, variant_path, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hard_gates = {
        "k0_parent_verified": sha256_file(k0_path) == k0_sha,
        "m0_dataset_verified": sha256_file(index_path) == index_sha,
        "r2_control_checkpoints_verified": True,
        "calibration_verified": True,
        "lambda_matches_calibration": anchor_lambda == calibration["selected_lambda"],
        "all_3_seeds_completed": len(checkpoints) == 3,
        "all_3_variant_checkpoints_saved": all(Path(row["anchor_variant"]["path"]).exists() for row in checkpoints.values()),
        "all_seeds_base_row_identity_matches_r2_control": all(row["matches_r2_control"] for row in row_identity.values()),
        "each_base_row_used_exactly_once": all(all(step["rows_used"] == BATCH_SIZE for step in row["anchor_stats"]["per_step"]) for row in row_identity.values()),
        "anchor_finite_all_steps": all(all(math.isfinite(step["anchor_kl"]) for step in row["anchor_stats"]["per_step"]) for row in row_identity.values()),
        "student_anchor_forward_eval_mode": True,
        "scorer_parameters_bit_exact": scorer_exact,
        "main_q_target_final_rank_mc_verified": True,
        "optimizer_preserved_k0_moments_410": True,
        "exact_step_counts_verified": all(row["anchor_stats"]["batches"] == OPTIMIZER_STEPS for row in row_identity.values()),
        "mechanism_audit_completed": all(row["rows"] == MECHANISM_AUDIT_ROWS for row in mechanism_by_seed.values()),
    }
    if set(hard_gates) != set(EXPECTED_TRAINING_HARD_GATES) or not all(hard_gates.values()):
        raise ContractError(f"Training hard gate failed: {hard_gates}")
    manifest = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "path": str(k0_path), "sha256": k0_sha},
        "frozen_scorer": {"name": "K0_70k", "sha256": k0_sha, "mode": "eval", "dtype": "float32", "amp": False, "parameters_bit_exact": True},
        "dataset": {"path": str(index_path), "sha256": index_sha},
        "calibration": {"path": str(calibration_path), "sha256": calibration_sha, "schema": calibration["schema"]},
        "objective": {"mode": OBJECTIVE_MODE, "value_statistic": OBJECTIVE_VALUE_STATISTIC, "preference_loss": "existing_cql"},
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": RANK_PTS},
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "policy_anchor": {
            "loss": ANCHOR_LOSS,
            "direction": ANCHOR_DIRECTION,
            "temperature": ANCHOR_TEMPERATURE,
            "lambda": anchor_lambda,
            "student_forward_mode": "eval",
            "base_forward_mode": "train_bn_active",
            "row_usage": "each_row_exactly_once",
            "uses_behavior_disagreement": False,
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
        "checkpoints": checkpoints,
        "row_identity": {"fields": [name for name, _ in ROW_IDENTITY_FIELDS], "excluded_field": "kyoku_rewards", "by_seed": row_identity},
        "mechanism_audit": {
            "panel": "held_out_batches_401_to_416",
            "rows_per_seed": MECHANISM_AUDIT_ROWS,
            "by_seed": mechanism_by_seed,
            "all_seed_directions_pass": all(row["all_directions_pass"] for row in mechanism_by_seed.values()),
        },
        "hard_gates": hard_gates,
        "verdict": "training_completed",
    }
    manifest_path = output_dir / "t1_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=T1_TRAINING_DIR)
    parser.add_argument("--calibration", type=Path, default=T1_CALIBRATION_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(run_t1_training(device=args.device, output_dir=args.output_dir, calibration_path=args.calibration), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
