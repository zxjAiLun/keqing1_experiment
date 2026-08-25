"""Targeted unit tests for O2 online continuation pilot contracts, optimizer, recovery, and evaluator integration."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Import libriichi before adding third_party/Mortal/mortal to sys.path
import libriichi.arena  # noqa: F401
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.eval_o2_online_continuation_pilot_2026_08 import build_shard_spec
from training.mortal.four_player_native import _load_engine
from training.mortal.o2_online_continuation_contract_2026_08 import (
    BATCH_SIZE,
    EVALUATION_GAMES_PER_SHARD,
    EVALUATION_LINEUP,
    EVALUATION_MANIFEST_SCHEMA,
    EVALUATION_SEED_START,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    GENERATION_BASE_SEED,
    INITIAL_SEED_GROUPS_PER_CYCLE,
    K0_EXPECTED_SHA256,
    LEARNING_RATE,
    MAX_SEED_GROUPS_PER_CYCLE,
    NUM_CYCLES,
    ROWS_PER_CYCLE,
    SEED_KEY,
    SEEDS_PER_CYCLE_BLOCK,
    START_STEP,
    STEPS_PER_CYCLE,
    TARGET_STEP,
    TOTAL_CONSUMED_ROWS,
    TOTAL_OPTIMIZER_STEPS,
    TRAINING_COMPLETION_SCHEMA,
    ContractError,
    adjudicate_o2_verdict,
    compute_effective_cql_weight,
    resolve_k0_checkpoint,
    sha256_file,
)
from training.mortal.summary_o2_online_continuation_pilot_2026_08 import (
    LOG_NAME_RE,
    adjudicate_o2_evaluation,
    parse_evaluation_log,
)
from training.mortal.train_o2_online_continuation_pilot_2026_08 import (
    construct_and_validate_preserved_adamw,
    get_rng_states,
    set_rng_states,
)
from training.run_mortal_dqn_offline import _optimizer_param_groups

if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import engine
import model
from lr_scheduler import LinearWarmUpCosineAnnealingLR


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


def test_3_real_k0_two_param_groups_adamw_preservation() -> None:
    """Test 3: Preserved AdamW constructs exact 2 parameter groups and loads all 410 moments from real K0."""
    k0_path, k0_sha256 = resolve_k0_checkpoint()
    assert k0_sha256 == K0_EXPECTED_SHA256
    k0_state = torch.load(k0_path, map_location="cpu")

    mortal_net = model.Brain(version=4, conv_channels=192, num_blocks=40)
    dqn_net = model.DQN(version=4)
    aux_net = model.AuxNet((4,))
    all_models = (mortal_net, dqn_net, aux_net)

    optimizer = construct_and_validate_preserved_adamw(
        all_models,
        k0_state["optimizer"],
        lr=LEARNING_RATE,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 1e-4

    state = optimizer.state_dict()["state"]
    assert len(state) == 410
    for s_entry in state.values():
        assert "step" in s_entry
        assert "exp_avg" in s_entry
        assert "exp_avg_sq" in s_entry

    # Verify that during 400 optimizer steps and LinearWarmUpCosineAnnealingLR steps, LR is exactly 1e-4 throughout
    sched_cfg = {"peak": LEARNING_RATE, "final": LEARNING_RATE, "warm_up_steps": 0, "max_steps": 0}
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **sched_cfg)
    for _ in range(400):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[1]["lr"] == 1e-4


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


def test_5_recovery_rng_and_cycle_identity_consistency(tmp_path: Path) -> None:
    """Test 5: Recovery state stores exact cycle, step, consumed rows, RNGs, and parameters atomically."""
    mortal_net = model.Brain(version=4, conv_channels=32, num_blocks=2)
    dqn_net = model.DQN(version=4)
    aux_net = model.AuxNet((4,))
    optimizer = torch.optim.AdamW(_optimizer_param_groups((mortal_net, dqn_net, aux_net), weight_decay=0.1), lr=1e-4)
    sched_cfg = {"peak": 1e-4, "final": 1e-4, "warm_up_steps": 0, "max_steps": 0}
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **sched_cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    rng_before = get_rng_states()
    recovery_file = tmp_path / "recovery_state.pth"
    tmp_recovery = tmp_path / "recovery_state.pth.tmp"

    cycle_idx = 4
    step_count = START_STEP + cycle_idx * STEPS_PER_CYCLE  # 70100
    rows_consumed = cycle_idx * ROWS_PER_CYCLE            # 51200

    state_payload = {
        "next_cycle": cycle_idx,
        "step_count": step_count,
        "total_rows_consumed": rows_consumed,
        "mortal": mortal_net.state_dict(),
        "dqn": dqn_net.state_dict(),
        "aux": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_states": rng_before,
        "game_identities": [((2000000, 8192), ("trainee", "baseline", "baseline", "baseline"))],
    }
    torch.save(state_payload, tmp_recovery)
    tmp_recovery.replace(recovery_file)

    assert recovery_file.exists()
    assert not tmp_recovery.exists()

    loaded = torch.load(recovery_file, weights_only=False, map_location="cpu")
    assert loaded["next_cycle"] == 4
    assert loaded["step_count"] == 70100
    assert loaded["total_rows_consumed"] == 51200
    assert len(loaded["game_identities"]) == 1
    assert "rng_states" in loaded

    set_rng_states(loaded["rng_states"])


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


def test_7_o2_checkpoint_loader_parity_with_four_player_native(tmp_path: Path) -> None:
    """Test 7: O2 checkpoint saved with updated 'config' metadata is loadable by four_player_native._load_engine."""
    k0_path, _ = resolve_k0_checkpoint()
    k0_state = torch.load(k0_path, map_location="cpu")

    o2_config = dict(k0_state["config"])
    o2_config["control"]["online"] = True
    o2_config["cql"] = {"min_q_weight": 0.0}
    o2_config["freeze_bn"] = {"mortal": True}

    o2_ckpt_path = tmp_path / "mortal_70400.pth"
    torch.save(
        {
            "mortal": k0_state["mortal"],
            "current_dqn": k0_state["current_dqn"],
            "aux_net": k0_state["aux_net"],
            "optimizer": k0_state["optimizer"],
            "steps": 70400,
            "experiment_id": EXPERIMENT_ID,
            "config": o2_config,
            "o2_training_contract": {
                "adapter": "keqing_project_online",
                "objective": "behavior_action_mc",
                "freeze_bn": True,
            },
        },
        o2_ckpt_path,
    )

    engine_chal = _load_engine(
        label="O2_70400",
        state_file=o2_ckpt_path,
        mortal_root=REPO_ROOT / "third_party" / "Mortal",
        device="cpu",
        enable_amp=False,
        enable_profile=False,
    )
    assert isinstance(engine_chal, engine.MortalEngine)
    assert engine_chal.name == "O2_70400"


def test_8_evaluator_cli_spec_parity(tmp_path: Path) -> None:
    """Test 8: build_shard_spec produces CLI flags strictly aligned with four_player_native parser."""
    dummy_o2 = tmp_path / "mortal_70400.pth"
    dummy_o2.touch()

    spec = build_shard_spec(
        shard_id=1,
        output_dir=tmp_path / "eval",
        o2_checkpoint_path=dummy_o2,
        device="cpu",
    )

    cmd = spec["command"]
    assert "--output-dir" in cmd
    assert "--seed-start" in cmd
    idx_seed = cmd.index("--seed-start")
    assert cmd[idx_seed + 1] == str(EVALUATION_SEED_START + EVALUATION_GAMES_PER_SHARD)

    assert "--games" in cmd
    idx_games = cmd.index("--games")
    assert cmd[idx_games + 1] == "250"

    assert "--rank-points-profile" in cmd
    idx_prof = cmd.index("--rank-points-profile")
    assert cmd[idx_prof + 1] == "tenhou_reference"

    assert "--native-batch-games" in cmd
    idx_nbg = cmd.index("--native-batch-games")
    assert cmd[idx_nbg + 1] == "250"


def test_9_real_json_gz_log_parsing_and_regex_match(tmp_path: Path) -> None:
    """Test 9: LOG_NAME_RE regex and parse_evaluation_log parse valid gzipped logs with reach_accepted correctly."""
    log_name = "2100042.json.gz"
    assert LOG_NAME_RE.match(log_name) is not None
    assert LOG_NAME_RE.match(log_name).group("seed") == "2100042"

    # Synthetic gzipped log
    log_file = tmp_path / log_name
    events = [
        {"type": "start_game", "seed": [2100042, SEED_KEY], "names": ["K0_70k", "ext_mortal", "M0_CURRENT_20260807", "O2_70400"]},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 3},  # O2 reaches -> 24000
        {"type": "hora", "deltas": [0, 0, 0, 8000]},  # O2 wins -> 32000
        {"type": "end_game"},
    ]
    with gzip.open(log_file, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    parsed = parse_evaluation_log(log_file)
    assert parsed["game_id"] == 2100042
    assert parsed["label_to_rank"]["O2_70400"] == 0  # 1st rank
    assert parsed["label_to_pt"]["O2_70400"] == 90.0


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


def test_11_eval_and_summary_checkpoint_sha_and_gate_binding(tmp_path: Path) -> None:
    """Test 11: Summarizer and evaluator strictly fail closed if gates are not exact 14 or SHA mismatch."""
    k0_path, _k0_sha = resolve_k0_checkpoint()
    k0_state = torch.load(k0_path, map_location="cpu")

    train_dir = tmp_path / "training"
    eval_dir = tmp_path / "evaluation"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    o2_ckpt_path = train_dir / "mortal_70400.pth"
    torch.save({"dummy": 1, "config": k0_state["config"]}, o2_ckpt_path)
    real_sha = sha256_file(o2_ckpt_path)

    # Valid completion JSON with exact 14 hard gates
    valid_gates = {g: True for g in EXPECTED_TRAINING_HARD_GATES}
    comp_json = train_dir / "training_completion.json"
    comp_json.write_text(json.dumps({
        "schema": TRAINING_COMPLETION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "hard_gates": valid_gates,
        "final_checkpoint": {"path": str(o2_ckpt_path), "sha256": real_sha, "step": 70400},
        "verdict": "training_completed",
    }))
    real_comp_sha = sha256_file(comp_json)

    # 1. Manifest with mismatching checkpoint SHA
    eval_manifest_bad_ckpt = eval_dir / "evaluation_manifest.json"
    eval_manifest_bad_ckpt.write_text(json.dumps({
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "lineup": EVALUATION_LINEUP,
        "total_games": 1000,
        "shards": [{"shard_id": i} for i in range(4)],
        "o2_checkpoint": {"path": str(o2_ckpt_path), "sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
        "training_completion": {"path": str(comp_json), "sha256": real_comp_sha},
        "verdict": "evaluation_completed",
    }))
    with pytest.raises(ContractError, match="Evaluation manifest checkpoint SHA mismatch"):
        adjudicate_o2_evaluation(evaluation_dir=eval_dir, training_dir=train_dir, allowed_root=tmp_path)

    # 2. Manifest with mismatching training completion SHA
    eval_manifest_bad_comp = eval_dir / "evaluation_manifest.json"
    eval_manifest_bad_comp.write_text(json.dumps({
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "lineup": EVALUATION_LINEUP,
        "total_games": 1000,
        "shards": [{"shard_id": i} for i in range(4)],
        "o2_checkpoint": {"path": str(o2_ckpt_path), "sha256": real_sha},
        "training_completion": {"path": str(comp_json), "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
        "verdict": "evaluation_completed",
    }))
    with pytest.raises(ContractError, match="Evaluation manifest training completion SHA mismatch"):
        adjudicate_o2_evaluation(evaluation_dir=eval_dir, training_dir=train_dir, allowed_root=tmp_path)

    # 3. Incomplete gate dictionary in training completion (e.g. {"all": True})
    comp_json.write_text(json.dumps({
        "schema": TRAINING_COMPLETION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "hard_gates": {"all": True},
        "final_checkpoint": {"path": str(o2_ckpt_path), "sha256": real_sha, "step": 70400},
        "verdict": "training_completed",
    }))
    new_comp_sha = sha256_file(comp_json)
    eval_manifest_valid = eval_dir / "evaluation_manifest.json"
    eval_manifest_valid.write_text(json.dumps({
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "lineup": EVALUATION_LINEUP,
        "total_games": 1000,
        "shards": [{"shard_id": i} for i in range(4)],
        "o2_checkpoint": {"path": str(o2_ckpt_path), "sha256": real_sha},
        "training_completion": {"path": str(comp_json), "sha256": new_comp_sha},
        "verdict": "evaluation_completed",
    }))
    with pytest.raises(ContractError, match="Training hard gates key set mismatch"):
        adjudicate_o2_evaluation(evaluation_dir=eval_dir, training_dir=train_dir, allowed_root=tmp_path)
