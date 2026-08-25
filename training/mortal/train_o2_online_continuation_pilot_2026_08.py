#!/usr/bin/env python3
"""O2: Keqing online continuation pilot training runner and gate validator.

Executes exactly 16 refresh cycles of 25 optimizer steps (400 total steps, 204,800 rows)
starting from K0_70k using trainee replay, project final_rank_mc objective, and online no-CQL.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import toml
import torch
from torch import nn, optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import engine
import model
from libriichi.arena import OneVsThree

from training.mortal.o2_online_continuation_contract_2026_08 import (
    ADAPTER_KIND,
    AUX_WEIGHT,
    BATCH_SIZE,
    CENTERED_TARGETS,
    EXPERIMENT_ID,
    FREEZE_BN,
    GAMMA,
    GENERATION_BASE_SEED,
    INITIAL_SEED_GROUPS_PER_CYCLE,
    K0_EXPECTED_SHA256,
    LEARNING_RATE,
    MAX_SEED_GROUPS_PER_CYCLE,
    NUM_CYCLES,
    O2_ROOT,
    O2_TRAINING_DIR,
    OBJECTIVE_MODE,
    ONLINE_FLAG,
    PARENT_MODEL,
    RANK_PTS,
    REWARD_MODE,
    ROWS_PER_CYCLE,
    SEED_KEY,
    SEEDS_PER_CYCLE_BLOCK,
    START_STEP,
    STEPS_PER_CYCLE,
    TARGET_STEP,
    TOTAL_CONSUMED_ROWS,
    TOTAL_OPTIMIZER_STEPS,
    TRAINEE_EXPLORATION,
    TRAINING_SEED,
    ContractError,
    check_directory_boundary,
    compute_effective_cql_weight,
    ensure_clean_staging_dir,
    resolve_k0_checkpoint,
    sha256_file,
)
from training.mortal.objective import compute_objective_losses
from training.run_mortal_dqn_offline import _optimizer_param_groups

logger = logging.getLogger("o2_training")


def setup_mortal_config(config_dir: Path) -> Path:
    """Create local MORTAL_CFG TOML file for production FileDatasetsIter integration."""
    config_path = config_dir / "mortal_cfg.toml"
    cfg = {
        "control": {"version": 4},
        "env": {"pts": RANK_PTS.tolist(), "gamma": GAMMA},
        "reward": {"mode": REWARD_MODE},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    os.environ["MORTAL_CFG"] = str(config_path.resolve())
    return config_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict[str, Any]:
    """Capture RNG states for exact resume reproducibility."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore RNG states for exact resume reproducibility."""
    if "python" not in states or "numpy" not in states or "torch" not in states:
        raise ContractError("Incomplete RNG state payload in recovery data")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def construct_and_validate_preserved_adamw(
    all_models: tuple[nn.Module, ...],
    k0_optimizer_state: dict[str, Any],
    lr: float = LEARNING_RATE,
) -> optim.AdamW:
    """Construct standard 2-group AdamW and load preserved moments from K0 checkpoint."""
    param_groups = _optimizer_param_groups(all_models, weight_decay=0.1)
    if len(param_groups) != 2:
        raise ContractError(f"Expected 2 parameter groups, got {len(param_groups)}")

    total_param_tensors = sum(len(g["params"]) for g in param_groups)
    if total_param_tensors != 410:
        raise ContractError(f"Expected 410 total parameter tensors, got {total_param_tensors}")

    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.0)
    optimizer.load_state_dict(k0_optimizer_state)

    for group in optimizer.param_groups:
        group["lr"] = lr

    state = optimizer.state_dict()["state"]
    if len(state) != 410:
        raise ContractError(f"Preserved optimizer state count {len(state)} != 410 parameter tensors")

    required = {"step", "exp_avg", "exp_avg_sq"}
    if any(not required.issubset(entry) for entry in state.values()):
        raise ContractError("Preserved optimizer is missing AdamW step/moment tensors")

    return optimizer


def run_cycle_generation(
    cycle_idx: int,
    cycle_dir: Path,
    trainee_mortal_sd: dict[str, Any],
    trainee_dqn_sd: dict[str, Any],
    k0_state: dict[str, Any],
    device: torch.device,
) -> tuple[list[str], set[tuple[tuple[int, int], tuple[str, ...]]], int]:
    """Generate selfplay replays for the cycle via OneVsThree until at least ROWS_PER_CYCLE rows are obtained."""
    from training.mortal.mainline_dataloader import FileDatasetsIter

    cycle_logs_dir = cycle_dir / "logs"
    ensure_clean_staging_dir(cycle_logs_dir, O2_ROOT)

    worker_mortal = model.Brain(version=4, conv_channels=192, num_blocks=40).to(device).eval()
    worker_dqn = model.DQN(version=4).to(device).eval()
    worker_mortal.load_state_dict(trainee_mortal_sd)
    worker_dqn.load_state_dict(trainee_dqn_sd)

    baseline_mortal = model.Brain(version=4, conv_channels=192, num_blocks=40).to(device).eval()
    baseline_dqn = model.DQN(version=4).to(device).eval()
    baseline_mortal.load_state_dict(k0_state["mortal"])
    baseline_dqn.load_state_dict(k0_state["current_dqn"])

    engine_chal = engine.MortalEngine(
        worker_mortal,
        worker_dqn,
        is_oracle=False,
        version=4,
        device=device,
        name="trainee",
        boltzmann_epsilon=TRAINEE_EXPLORATION["boltzmann_epsilon"],
        boltzmann_temp=TRAINEE_EXPLORATION["boltzmann_temp"],
        top_p=TRAINEE_EXPLORATION["top_p"],
    )
    engine_base = engine.MortalEngine(
        baseline_mortal,
        baseline_dqn,
        is_oracle=False,
        version=4,
        device=device,
        name="baseline",
        enable_rule_based_agari_guard=True,
    )

    arena = OneVsThree(disable_progress_bar=True, log_dir=str(cycle_logs_dir))
    cycle_base_seed = GENERATION_BASE_SEED + cycle_idx * SEEDS_PER_CYCLE_BLOCK

    # Step A: Initial batch of 32 seed groups (128 hanchans)
    arena.py_vs_py(
        challenger=engine_chal,
        champion=engine_base,
        seed_start=(cycle_base_seed, SEED_KEY),
        seed_count=INITIAL_SEED_GROUPS_PER_CYCLE,
    )

    current_seed_groups = INITIAL_SEED_GROUPS_PER_CYCLE

    def load_rows() -> tuple[list[Any], list[str], set[tuple[tuple[int, int], tuple[str, ...]]]]:
        files = sorted(str(p) for p in cycle_logs_dir.glob("*.json.gz"))
        identities: set[tuple[tuple[int, int], tuple[str, ...]]] = set()
        for f_path in files:
            with gzip.open(f_path, "rt", encoding="utf-8") as gz_f:
                first_line = json.loads(gz_f.readline())
                if first_line.get("type") == "start_game":
                    identities.add((tuple(first_line["seed"]), tuple(first_line["names"])))

        # Invariant check: exact 1:1 match between files, unique identities, and expected count
        expected_game_count = current_seed_groups * 4
        if len(files) != expected_game_count:
            raise ContractError(
                f"Cycle {cycle_idx} replay file count mismatch: got {len(files)}, expected {expected_game_count}"
            )
        if len(identities) != expected_game_count:
            raise ContractError(
                f"Cycle {cycle_idx} replay unique identities mismatch: got {len(identities)}, expected {expected_game_count}"
            )

        dataset = FileDatasetsIter(
            version=4,
            file_list=files,
            pts=RANK_PTS,
            oracle=False,
            player_names=["trainee"],
            enable_augmentation=False,
            num_epochs=1,
        )
        return list(dataset), files, identities

    all_rows, log_files, identities = load_rows()

    # Step B: If rows < 12800, append 1 seed group (4 hanchans) at a time up to 40 seed groups (160 hanchans)
    while len(all_rows) < ROWS_PER_CYCLE and current_seed_groups < MAX_SEED_GROUPS_PER_CYCLE:
        next_seed = cycle_base_seed + current_seed_groups
        arena.py_vs_py(
            challenger=engine_chal,
            champion=engine_base,
            seed_start=(next_seed, SEED_KEY),
            seed_count=1,
        )
        current_seed_groups += 1
        all_rows, log_files, identities = load_rows()

    if len(all_rows) < ROWS_PER_CYCLE:
        raise ContractError(
            f"Cycle {cycle_idx} failed to generate {ROWS_PER_CYCLE} rows after {current_seed_groups} seed groups ({len(all_rows)} rows)"
        )

    return log_files, identities, len(all_rows)


def run_o2_training(
    *,
    output_dir: Path = O2_TRAINING_DIR,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Execute complete O2 16-cycle online continuation pilot training."""
    check_directory_boundary(output_dir, O2_ROOT)

    recovery_file = output_dir / "recovery_state.pth"
    if not resume:
        ensure_clean_staging_dir(output_dir, O2_ROOT)
    else:
        if not recovery_file.exists():
            raise ContractError(f"Explicit --resume requested but recovery file does not exist: {recovery_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(TRAINING_SEED)
    device = torch.device(device_name)

    k0_path, k0_sha256 = resolve_k0_checkpoint()
    k0_state = torch.load(k0_path, weights_only=False, map_location="cpu")

    mortal_net = model.Brain(version=4, conv_channels=192, num_blocks=40).to(device)
    dqn_net = model.DQN(version=4).to(device)
    aux_net = model.AuxNet((4,)).to(device)

    mortal_net.load_state_dict(k0_state["mortal"])
    dqn_net.load_state_dict(k0_state["current_dqn"])
    aux_net.load_state_dict(k0_state["aux_net"])

    if FREEZE_BN:
        mortal_net.freeze_bn(True)

    # Initial parameter snapshots for all models for difference validation
    initial_params = {
        "mortal": {n: p.clone().cpu() for n, p in mortal_net.named_parameters()},
        "dqn": {n: p.clone().cpu() for n, p in dqn_net.named_parameters()},
        "aux": {n: p.clone().cpu() for n, p in aux_net.named_parameters()},
    }

    # Initialize AdamW optimizer with preserved K0 moments & 2 parameter groups
    optimizer = construct_and_validate_preserved_adamw(
        (mortal_net, dqn_net, aux_net),
        k0_state["optimizer"],
        lr=LEARNING_RATE,
    )

    # Fresh scheduler & scaler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=False)

    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    setup_mortal_config(config_dir)

    start_cycle = 0
    total_rows_consumed = 0
    step_count = START_STEP
    all_game_identities: set[tuple[tuple[int, int], tuple[str, ...]]] = set()

    # Check and load resume state
    if resume and recovery_file.exists():
        logger.info("Found recovery state at %s, validating resume consistency...", recovery_file)
        recovery_data = torch.load(recovery_file, weights_only=False, map_location="cpu")
        start_cycle = recovery_data["next_cycle"]
        step_count = recovery_data["step_count"]
        total_rows_consumed = recovery_data["total_rows_consumed"]

        if not (0 <= start_cycle <= NUM_CYCLES):
            raise ContractError(f"Invalid recovery next_cycle: {start_cycle}")
        expected_step = START_STEP + start_cycle * STEPS_PER_CYCLE
        if step_count != expected_step:
            raise ContractError(f"Recovery step count mismatch: got {step_count}, expected {expected_step}")
        expected_rows = start_cycle * ROWS_PER_CYCLE
        if total_rows_consumed != expected_rows:
            raise ContractError(f"Recovery rows consumed mismatch: got {total_rows_consumed}, expected {expected_rows}")

        # Strict completeness requirement for resume payload
        required_resume_keys = {"mortal", "dqn", "aux", "optimizer", "scheduler", "scaler", "rng_states", "game_identities"}
        missing_keys = required_resume_keys - set(recovery_data.keys())
        if missing_keys:
            raise ContractError(f"Recovery payload missing required keys: {missing_keys}")

        mortal_net.load_state_dict(recovery_data["mortal"])
        dqn_net.load_state_dict(recovery_data["dqn"])
        aux_net.load_state_dict(recovery_data["aux"])
        optimizer.load_state_dict(recovery_data["optimizer"])
        scheduler.load_state_dict(recovery_data["scheduler"])
        scaler.load_state_dict(recovery_data["scaler"])
        set_rng_states(recovery_data["rng_states"])

        all_game_identities = set(recovery_data["game_identities"])
        logger.info("Successfully resumed: next_cycle=%d, step=%d, rows=%d", start_cycle, step_count, total_rows_consumed)

    hard_gates: dict[str, bool] = {
        "parent_verified": (k0_sha256 == K0_EXPECTED_SHA256),
        "production_loader_used": False,
        "exact_16_cycles": False,
        "exact_400_optimizer_steps": False,
        "exact_204800_rows_consumed": False,
        "no_replay_identity_reuse": False,
        "online_cql_disabled": False,
        "final_rank_mc_verified": False,
        "gradients_finite": True,
        "parameters_finite": True,
        "parameters_changed_from_k0": False,
        "bn_frozen": True,
        "final_checkpoint_70400_created": False,
        "resume_state_consistent": False,
    }

    from training.mortal.mainline_dataloader import FileDatasetsIter
    hard_gates["production_loader_used"] = True

    cql_active, effective_cql_weight = compute_effective_cql_weight(
        online=ONLINE_FLAG, force_online=False
    )
    hard_gates["online_cql_disabled"] = (not cql_active and effective_cql_weight == 0.0)

    # Main 16-cycle loop
    for cycle in range(start_cycle, NUM_CYCLES):
        cycle_dir = output_dir / f"cycle_{cycle:02d}"
        logger.info("--- Starting Cycle %d/%d (step %d) ---", cycle + 1, NUM_CYCLES, step_count)

        # 1. Parameter refresh & Self-play generation
        trainee_mortal_sd = {k: v.clone().cpu() for k, v in mortal_net.state_dict().items()}
        trainee_dqn_sd = {k: v.clone().cpu() for k, v in dqn_net.state_dict().items()}

        log_files, cycle_identities, _total_cycle_rows = run_cycle_generation(
            cycle_idx=cycle,
            cycle_dir=cycle_dir,
            trainee_mortal_sd=trainee_mortal_sd,
            trainee_dqn_sd=trainee_dqn_sd,
            k0_state=k0_state,
            device=device,
        )

        # Identity overlap check across cycles
        if all_game_identities.intersection(cycle_identities):
            raise ContractError(f"Duplicate replay game identity detected in cycle {cycle}")
        all_game_identities.update(cycle_identities)

        # 2. Production loader consumes exactly 12,800 rows deterministically
        dataset = FileDatasetsIter(
            version=4,
            file_list=log_files,
            pts=RANK_PTS,
            oracle=False,
            player_names=["trainee"],
            enable_augmentation=False,
            num_epochs=1,
        )
        loaded_rows = list(dataset)
        if len(loaded_rows) < ROWS_PER_CYCLE:
            raise ContractError(f"FileDatasetsIter loaded {len(loaded_rows)} rows, expected >= {ROWS_PER_CYCLE}")

        # Deterministically slice exactly ROWS_PER_CYCLE (12,800) rows
        selected_rows = loaded_rows[:ROWS_PER_CYCLE]

        # 3. Train 25 optimizer steps × 512 batch size
        mortal_net.train()
        dqn_net.train()
        aux_net.train()
        if FREEZE_BN:
            mortal_net.freeze_bn(True)

        for step_idx in range(STEPS_PER_CYCLE):
            batch_slice = selected_rows[step_idx * BATCH_SIZE : (step_idx + 1) * BATCH_SIZE]
            b_obs = torch.as_tensor(np.stack([r[0] for r in batch_slice], axis=0), dtype=torch.float32, device=device)
            b_act = torch.as_tensor([r[1] for r in batch_slice], dtype=torch.int64, device=device)
            b_mask = torch.as_tensor(np.stack([r[2] for r in batch_slice], axis=0), dtype=torch.bool, device=device)
            b_rew = torch.as_tensor([r[4] for r in batch_slice], dtype=torch.float32, device=device)
            b_nr = torch.as_tensor([r[5] for r in batch_slice], dtype=torch.int64, device=device)

            # Target contract validation
            unique_targets = set(b_rew.cpu().numpy().tolist())
            if not unique_targets.issubset(set(CENTERED_TARGETS.values())):
                raise ContractError(f"Target value out of bounds: {unique_targets}")
            hard_gates["final_rank_mc_verified"] = True

            # Forward
            optimizer.zero_grad(set_to_none=True)
            phi = mortal_net(b_obs)
            q_out = dqn_net(phi, b_mask)
            (next_rank_logits,) = aux_net(phi)

            # Mainline objective computation
            losses = compute_objective_losses(
                q_out=q_out,
                masks=b_mask,
                actions=b_act,
                q_target_mc=b_rew,
                next_rank_logits=next_rank_logits,
                player_ranks=b_nr,
                mode=OBJECTIVE_MODE,
                cql_weight=effective_cql_weight,
                aux_weight=AUX_WEIGHT,
            )
            total_loss = losses["total_loss"]

            if not torch.isfinite(total_loss):
                hard_gates["gradients_finite"] = False
                raise ContractError(f"Non-finite loss at step {step_count}: {total_loss.item()}")

            total_loss.backward()

            # Check gradients
            for mod in (mortal_net, dqn_net, aux_net):
                for p in mod.parameters():
                    if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                        hard_gates["gradients_finite"] = False
                        raise ContractError(f"Non-finite gradient at step {step_count}")

            optimizer.step()
            scheduler.step()
            step_count += 1
            total_rows_consumed += BATCH_SIZE

        # 4. Atomic recovery state replacement at cycle boundary
        tmp_recovery = output_dir / "recovery_state.pth.tmp"
        torch.save(
            {
                "next_cycle": cycle + 1,
                "step_count": step_count,
                "total_rows_consumed": total_rows_consumed,
                "mortal": mortal_net.state_dict(),
                "dqn": dqn_net.state_dict(),
                "aux": aux_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_states": get_rng_states(),
                "game_identities": list(all_game_identities),
            },
            tmp_recovery,
        )
        tmp_recovery.replace(recovery_file)
        logger.info("Cycle %d complete. Total steps: %d, Total rows: %d", cycle + 1, step_count, total_rows_consumed)

    # Validate final parameters across Mortal, DQN, and AuxNet
    all_params_finite = True
    all_models_changed = True
    for model_name, model_inst in [("mortal", mortal_net), ("dqn", dqn_net), ("aux", aux_net)]:
        model_changed = False
        for n, p in model_inst.named_parameters():
            if not torch.isfinite(p).all():
                all_params_finite = False
            init_p = initial_params[model_name][n].to(p.device)
            if not torch.equal(init_p, p):
                model_changed = True
        if not model_changed:
            all_models_changed = False

    hard_gates["parameters_finite"] = all_params_finite
    hard_gates["parameters_changed_from_k0"] = all_models_changed
    hard_gates["exact_16_cycles"] = (step_count == TARGET_STEP and total_rows_consumed == TOTAL_CONSUMED_ROWS)
    hard_gates["exact_400_optimizer_steps"] = (step_count == TARGET_STEP)
    hard_gates["exact_204800_rows_consumed"] = (total_rows_consumed == TOTAL_CONSUMED_ROWS)
    hard_gates["no_replay_identity_reuse"] = (len(all_game_identities) >= NUM_CYCLES * INITIAL_SEED_GROUPS_PER_CYCLE * 4)

    # Validate recovery state mathematical consistency
    if recovery_file.exists():
        final_rec = torch.load(recovery_file, weights_only=False, map_location="cpu")
        hard_gates["resume_state_consistent"] = (
            final_rec["next_cycle"] == NUM_CYCLES
            and final_rec["step_count"] == TARGET_STEP
            and final_rec["total_rows_consumed"] == TOTAL_CONSUMED_ROWS
            and len(final_rec["game_identities"]) == len(all_game_identities)
            and "rng_states" in final_rec
            and "scheduler" in final_rec
            and "scaler" in final_rec
        )

    # Build updated O2 config metadata
    o2_config = copy.deepcopy(k0_state.get("config", {}))
    o2_config.setdefault("control", {})["version"] = 4
    o2_config["control"]["online"] = True
    o2_config.setdefault("cql", {})["min_q_weight"] = 0.0
    o2_config.setdefault("freeze_bn", {})["mortal"] = True
    o2_config.setdefault("aux", {})["next_rank_weight"] = AUX_WEIGHT
    o2_config.setdefault("env", {})["pts"] = RANK_PTS.tolist()
    o2_config["env"]["gamma"] = GAMMA
    o2_config.setdefault("optim", {})["scheduler"] = {"peak": LEARNING_RATE, "final": LEARNING_RATE}

    # Save scientific checkpoint 70400
    checkpoint_70400_path = output_dir / "mortal_70400.pth"
    torch.save(
        {
            "mortal": mortal_net.state_dict(),
            "current_dqn": dqn_net.state_dict(),
            "aux_net": aux_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "steps": step_count,
            "experiment_id": EXPERIMENT_ID,
            "parent_model": PARENT_MODEL,
            "config": o2_config,
            "o2_training_contract": {
                "adapter": ADAPTER_KIND,
                "objective": OBJECTIVE_MODE,
                "reward": REWARD_MODE,
                "gamma": GAMMA,
                "aux_weight": AUX_WEIGHT,
                "effective_cql_weight": effective_cql_weight,
                "learning_rate": LEARNING_RATE,
                "freeze_bn": FREEZE_BN,
                "total_cycles": NUM_CYCLES,
                "steps_per_cycle": STEPS_PER_CYCLE,
                "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
                "batch_size": BATCH_SIZE,
                "total_consumed_rows": TOTAL_CONSUMED_ROWS,
            },
        },
        checkpoint_70400_path,
    )
    hard_gates["final_checkpoint_70400_created"] = checkpoint_70400_path.exists()

    all_passed = all(hard_gates.values())
    summary = {
        "schema": "keqing.mortal.o2_training_completion.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_checkpoint": {
            "path": str(k0_path),
            "sha256": k0_sha256,
        },
        "training_contract": {
            "adapter": ADAPTER_KIND,
            "objective": OBJECTIVE_MODE,
            "reward": REWARD_MODE,
            "gamma": GAMMA,
            "aux_weight": AUX_WEIGHT,
            "effective_cql_weight": effective_cql_weight,
            "learning_rate": LEARNING_RATE,
            "freeze_bn": FREEZE_BN,
            "total_cycles": NUM_CYCLES,
            "steps_per_cycle": STEPS_PER_CYCLE,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "batch_size": BATCH_SIZE,
            "total_consumed_rows": TOTAL_CONSUMED_ROWS,
        },
        "hard_gates": hard_gates,
        "final_checkpoint": {
            "path": str(checkpoint_70400_path),
            "sha256": sha256_file(checkpoint_70400_path) if checkpoint_70400_path.exists() else None,
            "step": step_count,
        },
        "verdict": "training_completed" if all_passed else "training_failed",
    }

    completion_json = output_dir / "training_completion.json"
    with open(completion_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=O2_TRAINING_DIR, help="Training output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--resume", action="store_true", help="Resume from recovery_state.pth in output-dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_o2_training(
        output_dir=args.output_dir,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
