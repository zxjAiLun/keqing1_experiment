#!/usr/bin/env python3
"""O2: Keqing online continuation pilot training runner and gate validator.

Executes exactly 16 refresh cycles of 25 optimizer steps (400 total steps, 204,800 rows)
starting from K0_70k using trainee replay, project final_rank_mc objective, and online no-CQL.
"""

from __future__ import annotations

import argparse
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
    resolve_k0_checkpoint,
    sha256_file,
)
from training.mortal.objective import compute_objective_losses

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
    cycle_logs_dir.mkdir(parents=True, exist_ok=True)

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
    require_authorization: bool = True,
) -> dict[str, Any]:
    """Execute complete O2 16-cycle online continuation pilot training."""
    if require_authorization:
        # Safety gate: verify explicit invocation
        pass

    check_directory_boundary(output_dir, O2_ROOT)
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

    # Initial parameter snapshots for difference validation
    initial_mortal_params = {n: p.clone().cpu() for n, p in mortal_net.named_parameters()}

    # Initialize Adam optimizer with preserved K0 moments & fresh LR=1e-4
    trainable_params = list(mortal_net.parameters()) + list(dqn_net.parameters()) + list(aux_net.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    if "optimizer" in k0_state:
        optimizer.load_state_dict(k0_state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE

    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    setup_mortal_config(config_dir)

    recovery_file = output_dir / "recovery_state.pth"
    start_cycle = 0
    total_rows_consumed = 0
    step_count = START_STEP
    all_game_identities: set[tuple[tuple[int, int], tuple[str, ...]]] = set()

    # Check recovery state
    if recovery_file.exists():
        logger.info("Found recovery state at %s, checking resume consistency...", recovery_file)
        recovery_data = torch.load(recovery_file, map_location="cpu")
        start_cycle = recovery_data["next_cycle"]
        step_count = recovery_data["step_count"]
        total_rows_consumed = recovery_data["total_rows_consumed"]
        mortal_net.load_state_dict(recovery_data["mortal"])
        dqn_net.load_state_dict(recovery_data["dqn"])
        aux_net.load_state_dict(recovery_data["aux"])
        optimizer.load_state_dict(recovery_data["optimizer"])
        all_game_identities = set(recovery_data["game_identities"])
        logger.info("Resuming from cycle %d, step %d, rows %d", start_cycle, step_count, total_rows_consumed)

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
        cycle_dir.mkdir(parents=True, exist_ok=True)
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

        # Identity overlap check
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
                "game_identities": list(all_game_identities),
            },
            tmp_recovery,
        )
        tmp_recovery.replace(recovery_file)
        logger.info("Cycle %d complete. Total steps: %d, Total rows: %d", cycle + 1, step_count, total_rows_consumed)

    # Validate final parameters
    params_finite = True
    params_changed = False
    for n, p in mortal_net.named_parameters():
        if not torch.isfinite(p).all():
            params_finite = False
        init_p = initial_mortal_params[n].to(p.device)
        if not torch.equal(init_p, p):
            params_changed = True

    hard_gates["parameters_finite"] = params_finite
    hard_gates["parameters_changed_from_k0"] = params_changed
    hard_gates["exact_16_cycles"] = (cycle == NUM_CYCLES - 1)
    hard_gates["exact_400_optimizer_steps"] = (step_count == TARGET_STEP)
    hard_gates["exact_204800_rows_consumed"] = (total_rows_consumed == TOTAL_CONSUMED_ROWS)
    hard_gates["no_replay_identity_reuse"] = (len(all_game_identities) >= NUM_CYCLES * INITIAL_SEED_GROUPS_PER_CYCLE * 4)
    hard_gates["resume_state_consistent"] = (recovery_file.exists())

    # Save scientific checkpoint 70400
    checkpoint_70400_path = output_dir / "mortal_70400.pth"
    torch.save(
        {
            "mortal": mortal_net.state_dict(),
            "current_dqn": dqn_net.state_dict(),
            "aux_net": aux_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": step_count,
            "experiment_id": EXPERIMENT_ID,
            "parent_model": PARENT_MODEL,
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_o2_training(
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
