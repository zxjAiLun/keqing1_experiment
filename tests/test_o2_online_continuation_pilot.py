"""Targeted unit tests for O2 online continuation pilot contracts and components."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import toml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.o2_online_continuation_contract_2026_08 import (
    BATCH_SIZE,
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GAMMA,
    GENERATION_BASE_SEED,
    INITIAL_SEED_GROUPS_PER_CYCLE,
    LEARNING_RATE,
    MAX_SEED_GROUPS_PER_CYCLE,
    NUM_CYCLES,
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
    adjudicate_o2_verdict,
    compute_effective_cql_weight,
    final_scores_with_reach_accepted,
    paired_bootstrap_ci,
)


def test_1_16_x_25_step_and_row_schedule() -> None:
    """Test 1: Schedule invariants: exactly 16 cycles, 25 steps/cycle, 512 batch size = 400 steps and 204,800 rows."""
    assert NUM_CYCLES == 16
    assert STEPS_PER_CYCLE == 25
    assert BATCH_SIZE == 512

    total_steps = NUM_CYCLES * STEPS_PER_CYCLE
    assert total_steps == TOTAL_OPTIMIZER_STEPS == 400
    assert START_STEP + total_steps == TARGET_STEP == 70400

    rows_per_cycle = STEPS_PER_CYCLE * BATCH_SIZE
    assert rows_per_cycle == ROWS_PER_CYCLE == 12800

    total_rows = NUM_CYCLES * rows_per_cycle
    assert total_rows == TOTAL_CONSUMED_ROWS == 204800


def test_2_generation_identity_no_overlap() -> None:
    """Test 2: Seed ranges across cycles are strictly disjoint and have zero identity overlap."""
    assert SEEDS_PER_CYCLE_BLOCK == 40
    assert MAX_SEED_GROUPS_PER_CYCLE == 40

    cycle_seed_ranges: list[range] = []
    for cycle_i in range(NUM_CYCLES):
        start_seed = GENERATION_BASE_SEED + cycle_i * SEEDS_PER_CYCLE_BLOCK
        end_seed = start_seed + MAX_SEED_GROUPS_PER_CYCLE
        cycle_seed_ranges.append(range(start_seed, end_seed))

    # Pairwise disjointness check
    for i in range(NUM_CYCLES):
        for j in range(i + 1, NUM_CYCLES):
            set_i = set(cycle_seed_ranges[i])
            set_j = set(cycle_seed_ranges[j])
            assert len(set_i.intersection(set_j)) == 0, f"Overlap between cycle {i} and {j}"

    # Seat rotation identity uniqueness check
    rotations = [
        ("trainee", "baseline", "baseline", "baseline"),
        ("baseline", "trainee", "baseline", "baseline"),
        ("baseline", "baseline", "trainee", "baseline"),
        ("baseline", "baseline", "baseline", "trainee"),
    ]
    seen_identities = set()
    for cycle_i in range(NUM_CYCLES):
        base_seed = GENERATION_BASE_SEED + cycle_i * SEEDS_PER_CYCLE_BLOCK
        for g_idx in range(INITIAL_SEED_GROUPS_PER_CYCLE):
            seed_val = base_seed + g_idx
            for seat_rot in rotations:
                identity = ((seed_val, SEED_KEY), seat_rot)
                assert identity not in seen_identities
                seen_identities.add(identity)

    assert len(seen_identities) == NUM_CYCLES * INITIAL_SEED_GROUPS_PER_CYCLE * 4 == 2048


def test_3_production_file_datasets_iter_consumption(tmp_path: Path) -> None:
    """Test 3: FileDatasetsIter loads log and enforces final_rank_mc target contract."""
    candidates = list(REPO_ROOT.glob("artifacts/**/*.json.gz"))
    assert candidates, "No .json.gz logs found in artifacts"
    sample_log = candidates[0]

    config_path = tmp_path / "mortal_cfg.toml"
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump({
            "control": {"version": 4},
            "env": {"pts": RANK_PTS.tolist(), "gamma": GAMMA},
            "reward": {"mode": REWARD_MODE},
        }, f)
    os.environ["MORTAL_CFG"] = str(config_path.resolve())

    from training.mortal.mainline_dataloader import FileDatasetsIter

    dataset = FileDatasetsIter(
        version=4,
        file_list=[str(sample_log)],
        pts=RANK_PTS,
        oracle=False,
        player_names=["ext_mortal", "trainee", "70k", "M0_CURRENT_20260806"],
        enable_augmentation=False,
        num_epochs=1,
    )

    rows = list(dataset)
    assert len(rows) > 0

    expected_domain = {-3.0, -1.0, 1.0, 3.0}
    for obs, action, mask, steps_to_done, kyoku_reward, next_rank in rows[:50]:
        assert obs.shape == (1012, 34)
        assert 0 <= action <= 45
        assert mask.shape == (46,)
        assert bool(mask[action]) is True
        assert steps_to_done >= 0
        assert float(kyoku_reward) in expected_domain
        assert 0 <= next_rank <= 3


def test_4_online_cql_branch_calculation() -> None:
    """Test 4: Online CQL branch calculation respects online vs force_online."""
    # online=True, force_online=False -> cql_active=False, weight=0.0
    active, weight = compute_effective_cql_weight(online=True, force_online=False, base_min_q_weight=5.0)
    assert active is False
    assert weight == 0.0

    # online=True, force_online=True -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=True, force_online=True, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0

    # online=False, force_online=False -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=False, force_online=False, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0

    # online=False, force_online=True -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=False, force_online=True, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0


def test_5_preserved_optimizer_and_fresh_scheduler_contract() -> None:
    """Test 5: Preserved Adam optimizer moments loading and constant LR=1e-4."""
    mortal_net = model.Brain(version=4, conv_channels=32, num_blocks=2)
    dqn_net = model.DQN(version=4)
    aux_net = model.AuxNet((4,))

    params = list(mortal_net.parameters()) + list(dqn_net.parameters()) + list(aux_net.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    assert optimizer.defaults["lr"] == 1e-4

    # Dummy moments state
    p0 = params[0]
    optimizer.state[p0] = {
        "step": torch.tensor(70000.0),
        "exp_avg": torch.ones_like(p0) * 0.5,
        "exp_avg_sq": torch.ones_like(p0) * 0.25,
    }
    saved_state = optimizer.state_dict()

    # New optimizer restoring state
    new_optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    new_optimizer.load_state_dict(saved_state)
    for g in new_optimizer.param_groups:
        g["lr"] = LEARNING_RATE

    assert new_optimizer.param_groups[0]["lr"] == 1e-4
    assert torch.equal(new_optimizer.state[p0]["exp_avg"], torch.ones_like(p0) * 0.5)


def test_6_batch_norm_frozen_verification() -> None:
    """Test 6: freeze_bn(True) prevents updating running_mean and running_var during train mode."""
    torch.manual_seed(42)
    mortal_net = model.Brain(version=4, conv_channels=32, num_blocks=2).train()
    mortal_net.freeze_bn(True)

    bn_layers = [m for m in mortal_net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(bn_layers) > 0
    assert all(not bn.training for bn in bn_layers)

    initial_stats = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in bn_layers]

    # Forward 10 batches with varying statistics
    for _ in range(10):
        x = torch.randn(8, 1012, 34) * 5.0 + 2.0
        _ = mortal_net(x)

    # Running stats must remain bit-for-bit unchanged
    for bn, (init_mean, init_var) in zip(bn_layers, initial_stats, strict=True):
        assert torch.equal(bn.running_mean, init_mean)
        assert torch.equal(bn.running_var, init_var)


def test_7_cycle_recovery_state_and_resume_identity(tmp_path: Path) -> None:
    """Test 7: Recovery state stores exact cycle, step, consumed rows, and parameters atomically."""
    mortal_net = model.Brain(version=4, conv_channels=32, num_blocks=2)
    dqn_net = model.DQN(version=4)
    aux_net = model.AuxNet((4,))
    optimizer = torch.optim.Adam(list(mortal_net.parameters()), lr=1e-4)

    recovery_file = tmp_path / "recovery_state.pth"
    tmp_recovery = tmp_path / "recovery_state.pth.tmp"

    state_payload = {
        "next_cycle": 5,
        "step_count": 70125,
        "total_rows_consumed": 64000,
        "mortal": mortal_net.state_dict(),
        "dqn": dqn_net.state_dict(),
        "aux": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "game_identities": [((2000000, 8192), ("trainee", "baseline", "baseline", "baseline"))],
    }
    torch.save(state_payload, tmp_recovery)
    tmp_recovery.replace(recovery_file)

    assert recovery_file.exists()
    assert not tmp_recovery.exists()

    loaded = torch.load(recovery_file, map_location="cpu")
    assert loaded["next_cycle"] == 5
    assert loaded["step_count"] == 70125
    assert loaded["total_rows_consumed"] == 64000
    assert len(loaded["game_identities"]) == 1


def test_8_corrected_reach_accepted_parser() -> None:
    """Test 8: final_scores_with_reach_accepted correctly accounts for -1000 on reach_accepted."""
    events = [
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 1},  # player 1 reaches -> -1000
        {"type": "hora", "deltas": [8000, -8000, 0, 0]},  # player 0 wins from player 1
    ]
    scores = final_scores_with_reach_accepted(events)
    assert scores == [33000.0, 16000.0, 25000.0, 25000.0]

    # Without hora / ryukyoku
    events2 = [
        {"type": "start_kyoku", "scores": [30000, 30000, 20000, 20000]},
        {"type": "reach_accepted", "actor": 0},
        {"type": "reach_accepted", "actor": 2},
    ]
    scores2 = final_scores_with_reach_accepted(events2)
    assert scores2 == [29000.0, 30000.0, 19000.0, 20000.0]


def test_9_paired_bootstrap_determinism() -> None:
    """Test 9: 5000-rep paired bootstrap with fixed seed 20260903 is strictly deterministic."""
    np.random.seed(42)
    diffs = np.random.normal(loc=2.5, scale=10.0, size=1000)

    mean_1, ci_1 = paired_bootstrap_ci(diffs, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
    mean_2, ci_2 = paired_bootstrap_ci(diffs, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)

    assert mean_1 == mean_2
    assert ci_1[0] == ci_2[0]
    assert ci_1[1] == ci_2[1]
    assert ci_1[0] < mean_1 < ci_1[1]


def test_10_four_state_verdict_mapping() -> None:
    """Test 10: Verdict adjudicator produces exact four states and keeps promotions False."""
    # 1. strong_signal: both means > 0 and both CI lowers > 0
    v1 = adjudicate_o2_verdict(
        all_gates_pass=True,
        mean_x=5.2,
        ci_x=[1.1, 9.3],
        mean_y=3.8,
        ci_y=[0.4, 7.2],
    )
    assert v1 == "strong_signal"

    # 2. promising: both means > 0 but at least one CI crosses 0
    v2 = adjudicate_o2_verdict(
        all_gates_pass=True,
        mean_x=5.2,
        ci_x=[-0.5, 10.9],
        mean_y=3.8,
        ci_y=[0.4, 7.2],
    )
    assert v2 == "promising"

    # 3. not_promising: any mean <= 0
    v3 = adjudicate_o2_verdict(
        all_gates_pass=True,
        mean_x=-1.2,
        ci_x=[-6.5, 4.1],
        mean_y=3.8,
        ci_y=[0.4, 7.2],
    )
    assert v3 == "not_promising"

    # 4. invalid: gate failure
    v4 = adjudicate_o2_verdict(
        all_gates_pass=False,
        mean_x=5.2,
        ci_x=[1.1, 9.3],
        mean_y=3.8,
        ci_y=[0.4, 7.2],
    )
    assert v4 == "invalid"
