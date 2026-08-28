"""Targeted unit tests for F1 behavior-disagreement prioritized replay experiment contracts and runners."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

from training.mortal.f1_disagreement_prioritized_replay_contract_2026_09 import (
    BOOTSTRAP_SEED,
    CANONICAL_K1_SEED,
    EVAL_GAMES_PER_PANEL,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_TOTAL_GAMES,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    FROZEN_R2_CONTROL,
    MAIN_REWARD_MODE,
    MIN_HIGH_SIGNAL_ROWS,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    PRIORITY_ROWS,
    ROW_IDENTITY_FIELDS,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_SEEDS,
    UNIFORM_ROWS,
    ContractError,
    adjudicate_f1_verdict,
    canonical_step_rng,
    crossed_bootstrap_ci,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    select_priority_batch,
    sha256_file,
    update_row_identity_digest,
    verify_training_manifest,
)
from training.mortal.summary_f1_disagreement_prioritized_replay_2026_09 import adjudicate_f1_prioritized_replay
from training.mortal.train_f1_disagreement_prioritized_replay_2026_09 import (
    _build_dataloader,
    _build_frozen_scorer,
    _build_models,
    _build_optimizer,
    _verify_optimizer_structure,
    _verify_q_target_values,
    compute_high_signal_mask,
    _scorer_parameter_digest,
)
import model


def test_1_f1_contract_invariants() -> None:
    """Test 1: Frozen constants, frozen R2 Control binding, and gate sets."""
    assert EXPERIMENT_ID == "F1_m0_behavior_disagreement_prioritized_replay_2026_09"
    assert TRAINING_SEEDS == [20260910, 20260911, 20260912]
    assert CANONICAL_K1_SEED == 20260911
    assert OPTIMIZER_STEPS == 400
    assert BOOTSTRAP_SEED == 20260930
    assert EVAL_GAMES_PER_PANEL == 1000
    assert EVAL_TOTAL_GAMES == 3000
    assert EVAL_SEED_START == 2500000

    # Protocol unchanged from operational.
    assert MAIN_REWARD_MODE == "final_rank_mc"
    assert OBJECTIVE_MODE == "behavior_action_mc"
    assert OBJECTIVE_VALUE_STATISTIC == "behavior_action_q"
    assert TRAINABLE_PLAYER_NAMES == ("ext_mortal",)

    # Prioritized replay constants.
    assert UNIFORM_ROWS == 256
    assert PRIORITY_ROWS == 256
    assert MIN_HIGH_SIGNAL_ROWS == 16

    assert len(EXPECTED_TRAINING_HARD_GATES) == 12
    assert len(EXPECTED_EVAL_HARD_GATES) == 7
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    for s in TRAINING_SEEDS:
        path, ckpt_sha, row_sha = resolve_r2_control_checkpoint(s)
        assert path.exists()
        assert ckpt_sha == FROZEN_R2_CONTROL[s]["checkpoint_sha256"]
        assert row_sha == FROZEN_R2_CONTROL[s]["row_identity_sha256"]

    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists() and len(k0_sha) == 64
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    assert ext_path.exists() and len(ext_sha) == 64
    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists() and len(m0_sha) == 64


def test_2_disagreement_determination() -> None:
    """Test 2: high_signal = actions != frozen-K0 greedy over legal actions; legality enforced."""
    k0_path, _ = resolve_k0_checkpoint()
    scorer_brain, scorer_dqn = _build_frozen_scorer(k0_path, "cpu")
    assert not scorer_brain.training and not scorer_dqn.training
    for p in list(scorer_brain.parameters()) + list(scorer_dqn.parameters()):
        assert not p.requires_grad

    obs = torch.randn(8, 1012, 34)
    masks = torch.zeros(8, 46, dtype=torch.bool)
    masks[:, :10] = True  # at least 10 legal actions each row
    actions = torch.zeros(8, dtype=torch.int64)
    # Ensure at least one definite disagreement: pick action 1 for all rows, then
    # disagreement depends on greedy; instead construct both cases explicitly.
    actions[:] = 0

    disagreement, greedy = compute_high_signal_mask(scorer_brain, scorer_dqn, obs, actions, masks)
    assert disagreement.shape == (8,)
    assert greedy.shape == (8,)
    # Greedy actions are legal by DQN's masked_fill(-inf) construction.
    assert bool(masks.gather(1, torch.as_tensor(greedy, dtype=torch.long).unsqueeze(1)).all())
    # disagreement is exactly actions != greedy.
    assert np.array_equal(disagreement, (actions.numpy() != greedy))

    # Determinism: repeated calls produce identical output.
    d2, g2 = compute_high_signal_mask(scorer_brain, scorer_dqn, obs, actions, masks)
    assert np.array_equal(disagreement, d2) and np.array_equal(greedy, g2)

    # Illegal behavior action -> fail closed.
    bad_masks = torch.zeros(8, 46, dtype=torch.bool)
    bad_masks[:, 0] = True
    illegal_actions = torch.full((8,), 5, dtype=torch.int64)  # 5 not legal
    with pytest.raises(ContractError):
        compute_high_signal_mask(scorer_brain, scorer_dqn, obs, illegal_actions, bad_masks)


def test_3_deterministic_256_256_sampler_and_rng_isolation() -> None:
    """Test 3: Deterministic sampler bound to (experiment, seed, step); global RNG untouched."""
    mask = np.zeros(512, dtype=bool)
    mask[[3, 10, 77, 200, 451]] = True  # 5 disagreement rows (>= 16? no: below min -> separate check)

    # Below-minimum batch fails closed.
    with pytest.raises(ContractError):
        select_priority_batch(mask, canonical_step_rng(EXPERIMENT_ID, 20260910, 1))

    # A batch with exactly 16 high-signal rows is accepted.
    mask16 = np.zeros(512, dtype=bool)
    mask16[:16] = True
    rng_a = canonical_step_rng(EXPERIMENT_ID, 20260910, 1)
    sel_a = select_priority_batch(mask16, rng_a)

    assert sel_a["high_signal_count"] == 16
    assert sel_a["high_signal_rate"] == 16 / 512
    assert int(sel_a["uniform_idx"].size) == UNIFORM_ROWS
    assert int(np.unique(sel_a["uniform_idx"]).size) == UNIFORM_ROWS  # without replacement
    assert int(sel_a["priority_idx"].size) == PRIORITY_ROWS
    assert np.isin(sel_a["priority_idx"], np.flatnonzero(mask16)).all()  # only high-signal rows
    assert sel_a["final_idx"].shape == (512,)
    # final is a permutation of uniform+priority concatenation.
    concat = np.concatenate([sel_a["uniform_idx"], sel_a["priority_idx"]])
    assert sorted(sel_a["final_idx"].tolist()) == sorted(concat.tolist())

    # Determinism: same identity -> same selection.
    sel_b = select_priority_batch(mask16, canonical_step_rng(EXPERIMENT_ID, 20260910, 1))
    assert np.array_equal(sel_a["final_idx"], sel_b["final_idx"])

    # Different step -> (almost surely) different selection.
    sel_c = select_priority_batch(mask16, canonical_step_rng(EXPERIMENT_ID, 20260910, 2))
    assert not np.array_equal(sel_a["final_idx"], sel_c["final_idx"])

    # Different seed -> different selection.
    sel_d = select_priority_batch(mask16, canonical_step_rng(EXPERIMENT_ID, 20260911, 1))
    assert not np.array_equal(sel_a["final_idx"], sel_d["final_idx"])

    # Canonical RNG derivation: distinct identities -> distinct streams.
    r1 = canonical_step_rng(EXPERIMENT_ID, 20260910, 1)
    r2 = canonical_step_rng(EXPERIMENT_ID, 20260910, 2)
    assert not np.array_equal(r1.integers(0, 512, size=64), r2.integers(0, 512, size=64))

    # Global RNG untouched by sampler and canonical RNG.
    import random as _r
    state_before = _r.getstate()
    np_state_before = np.random.get_state()
    torch_state_before = torch.get_rng_state()

    select_priority_batch(mask16, canonical_step_rng(EXPERIMENT_ID, 20260910, 3))
    canonical_step_rng(EXPERIMENT_ID, 20260910, 4)

    assert _r.getstate() == state_before
    assert np.random.get_state()[1].tolist() == np_state_before[1].tolist()
    assert torch.equal(torch.get_rng_state(), torch_state_before)

    # Duplicate rate sanity: with 16 high-signal rows and 256 draws, duplicates occur.
    assert sel_a["priority_duplicate_rate"] > 0.0
    assert 0.0 <= sel_a["priority_duplicate_rate"] <= 1.0


def test_4_base_digest_matches_r2_control_and_selected_differs() -> None:
    """Test 4: Base stream digest equals R2 Control's first-batch prefix; selected digest differs."""
    m0_path, _ = resolve_m0_dataset_index()
    seed = 20260910

    dl = _build_dataloader(m0_path, seed=seed, batch_size=32)
    batch = next(iter(dl))
    assert len(batch) == 6  # F1 uses the standard 6-field interface; dataloader unmodified
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch

    base = hashlib.sha256()
    update_row_identity_digest(
        base, obs=obs, actions=actions, masks=masks, steps_to_done=steps_to_done, player_ranks=player_ranks
    )

    # Build a synthetic selection differing from identity to verify the selected digest
    # tracks the selected rows rather than the base rows.
    rng = canonical_step_rng(EXPERIMENT_ID, seed, 1)
    mask16 = np.zeros(32, dtype=bool)
    mask16[:16] = True
    sel = select_priority_batch(mask16, rng, batch_size=32, uniform_rows=16, priority_rows=16)

    sel_obs = obs[torch.as_tensor(sel["final_idx"], dtype=torch.long)]
    sel_actions = actions[torch.as_tensor(sel["final_idx"], dtype=torch.long)]
    sel_masks = masks[torch.as_tensor(sel["final_idx"], dtype=torch.long)]
    sel_steps = steps_to_done[torch.as_tensor(sel["final_idx"], dtype=torch.long)]
    sel_ranks = player_ranks[torch.as_tensor(sel["final_idx"], dtype=torch.long)]

    selected = hashlib.sha256()
    update_row_identity_digest(
        selected, obs=sel_obs, actions=sel_actions, masks=sel_masks, steps_to_done=sel_steps, player_ranks=sel_ranks
    )
    # With duplicates in the priority draw, the selected digest differs from base.
    assert selected.hexdigest() != base.hexdigest()

    # Determinism of selected digest given the same identity.
    selected2 = hashlib.sha256()
    rng2 = canonical_step_rng(EXPERIMENT_ID, seed, 1)
    sel2 = select_priority_batch(mask16, rng2, batch_size=32, uniform_rows=16, priority_rows=16)
    idx2 = torch.as_tensor(sel2["final_idx"], dtype=torch.long)
    update_row_identity_digest(
        selected2,
        obs=obs[idx2],
        actions=actions[idx2],
        masks=masks[idx2],
        steps_to_done=steps_to_done[idx2],
        player_ranks=player_ranks[idx2],
    )
    assert selected.hexdigest() == selected2.hexdigest()


def test_5_frozen_scorer_bit_exact_and_eval_mode() -> None:
    """Test 5: Frozen scorer parameters bit-exact before/after use; eval mode; no optimizer registration."""
    k0_path, k0_sha = resolve_k0_checkpoint()
    scorer_brain, scorer_dqn = _build_frozen_scorer(k0_path, "cpu")

    digest_before = _scorer_parameter_digest(scorer_brain, scorer_dqn)

    # Run the scorer on real-shaped input (inference mode; no grad).
    obs = torch.randn(16, 1012, 34)
    masks = torch.ones(16, 46, dtype=torch.bool)
    actions = torch.zeros(16, dtype=torch.int64)
    disagreement, _ = compute_high_signal_mask(scorer_brain, scorer_dqn, obs, actions, masks)

    digest_after = _scorer_parameter_digest(scorer_brain, scorer_dqn)
    assert digest_before == digest_after

    # The training optimizer never contains scorer parameters.
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)
    _verify_optimizer_structure(optimizer)
    scorer_ids = {id(p) for p in list(scorer_brain.parameters()) + list(scorer_dqn.parameters())}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert scorer_ids.isdisjoint(opt_ids)

    # Scorer weights identical to the training Brain's initial weights (same K0).
    for (n1, p1), (n2, p2) in zip(scorer_brain.named_parameters(), brain.named_parameters(), strict=True):
        assert n1 == n2 and torch.equal(p1.detach(), p2.detach())

    # Disagreement on all-agree actions yields a valid boolean distribution.
    assert disagreement.dtype == np.bool_


def test_6_optimizer_preserved_k0_moments_410() -> None:
    """Test 6: Optimizer restores exactly 410 K0 moments in two groups [165, 245]."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net)

    assert len(optimizer.param_groups) == 2
    assert [len(g["params"]) for g in optimizer.param_groups] == [165, 245]
    assert len(optimizer.state) == 410
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert optimizer.param_groups[0]["lr"] == 1e-4
    _verify_optimizer_structure(optimizer)


def test_7_q_target_domain_and_evaluator_cli_shape() -> None:
    """Test 7: Main Q target restricted to final_rank_mc domain; evaluator CLI composed correctly."""
    _verify_q_target_values(torch.tensor([3.0, 1.0, -1.0, -3.0]))
    _verify_q_target_values(torch.tensor([1.0, 1.0, -3.0]))
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([3.0, 0.25]))  # foreign value leaked in
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([2.0]))

    # Evaluator builds the exact frozen lineup and CLI from the contract.
    from training.mortal.eval_f1_disagreement_prioritized_replay_2026_09 import _lineup_for_seed
    for s in TRAINING_SEEDS:
        lineup = _lineup_for_seed(s)
        assert lineup == (
            "K0_70k",
            "ext_mortal",
            f"R2_Control_seed_{s}",
            f"F1_PriorityVariant_seed_{s}",
        )


def _write_game_log(log_path: Path, seed: int, names: list[str], *, variant_wins: bool = True) -> None:
    # ranks: seat order maps to names order; variant seat 3.
    #hora deltas such that seat 3 tops (or seat 0 tops when variant_wins False)
    if variant_wins:
        deltas = [-8000, -6000, -4000, 30000]
    else:
        deltas = [30000, -8000, -6000, -4000]
    events = [
        {"type": "start_game", "seed": [seed, EVAL_SEED_KEY], "names": names},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": deltas},
        {"type": "end_game"},
    ]
    with gzip.open(log_path, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _build_strict_fixture(root: Path) -> dict:
    """Build a complete, contract-conformant F1 artifact fixture with real a/b/c/d logs and real SHAs."""
    tr_dir = root / "training"
    ev_dir = root / "evaluation"
    sm_dir = root / "summary"
    tr_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)
    sm_dir.mkdir(parents=True)

    _, k0_sha = resolve_k0_checkpoint()
    _, ext_sha = resolve_ext_mortal_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()

    checkpoints_manifest: dict = {}
    row_by_seed: dict = {}
    for s in TRAINING_SEEDS:
        r2c_path, r2c_sha, r2c_row = resolve_r2_control_checkpoint(s)
        v_path = tr_dir / f"mortal_priority_variant_70400_seed_{s}.pth"
        v_path.write_bytes(f"pr_variant_{s}".encode())
        v_sha = sha256_file(v_path)
        checkpoints_manifest[f"seed_{s}"] = {
            "r2_control": {"name": r2c_path.name, "path": str(r2c_path), "sha256": r2c_sha},
            "priority_variant": {"name": v_path.name, "path": str(v_path), "sha256": v_sha},
        }
        per_step = [
            {
                "step": i + 1,
                "high_signal_count": 50,
                "high_signal_rate": 50 / 512,
                "priority_duplicate_rate": 0.7,
                "uniform_rows": UNIFORM_ROWS,
                "priority_rows": PRIORITY_ROWS,
            }
            for i in range(OPTIMIZER_STEPS)
        ]
        row_by_seed[f"seed_{s}"] = {
            "r2_control_sha256": r2c_row,
            "base_row_sha256": r2c_row,  # must equal frozen R2 control digest
            "selected_row_sha256": hashlib.sha256(f"selected_{s}".encode()).hexdigest(),
            "matches_r2_control": True,
            "selection_stats": {
                "batches": OPTIMIZER_STEPS,
                "high_signal_count_min": 50,
                "high_signal_count_max": 60,
                "high_signal_rate_mean": 0.107,
                "priority_duplicate_rate_mean": 0.7,
                "per_step": per_step,
            },
        }

    tr_man = {
        "schema": "keqing.mortal.f1_training_manifest.v1",
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
        "objective": {"mode": OBJECTIVE_MODE, "value_statistic": OBJECTIVE_VALUE_STATISTIC, "preference_loss": "existing_cql"},
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [6.0, 4.0, 2.0, 0.0]},
        "prioritized_replay": {
            "uniform_rows": UNIFORM_ROWS,
            "priority_rows": PRIORITY_ROWS,
            "min_high_signal_rows": MIN_HIGH_SIGNAL_ROWS,
            "q_margin_threshold": None,
            "uniform_sampling": "without_replacement",
            "priority_sampling": "with_replacement",
            "sampler": "sha256_canonical_local_rng",
            "uses_reward_or_eval_signal": False,
        },
        "training_config": {"training_seeds": TRAINING_SEEDS, "device": "cpu"},
        "checkpoints": checkpoints_manifest,
        "row_identity": {"fields": [name for name, _ in ROW_IDENTITY_FIELDS], "excluded_field": "kyoku_rewards", "by_seed": row_by_seed},
        "hard_gates": {g: True for g in EXPECTED_TRAINING_HARD_GATES},
        "verdict": "training_completed",
    }
    tr_path = tr_dir / "f1_training_manifest.json"
    tr_path.write_text(json.dumps(tr_man), encoding="utf-8")
    tr_sha = sha256_file(tr_path)

    for s in TRAINING_SEEDS:
        panel_dir = ev_dir / f"panel_seed_{s}"
        lineup = ["K0_70k", "ext_mortal", f"R2_Control_seed_{s}", f"F1_PriorityVariant_seed_{s}"]
        for shard_idx in range(4):
            logs_dir = panel_dir / f"shard_{shard_idx:03d}" / "logs"
            logs_dir.mkdir(parents=True)
            for g_id in range(2500000 + shard_idx * 250, 2500000 + (shard_idx + 1) * 250):
                suffix = ["a", "b", "c", "d"][g_id % 4]
                _write_game_log(logs_dir / f"{g_id}_{EVAL_SEED_KEY}_{suffix}.json.gz", g_id, lineup, variant_wins=True)

    ev_man = {
        "schema": "keqing.mortal.f1_eval_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_path), "sha256": tr_sha},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "ext_mortal_model": {"name": "ext_mortal", "path": "mock", "sha256": ext_sha},
        "eval_config": {"panels_count": 3, "games_per_panel": 1000, "total_games": 3000},
        "panels": {
            f"seed_{s}": {
                "training_seed": s,
                "panel_dir": str(ev_dir / f"panel_seed_{s}"),
                "games_count": 1000,
                "models": {
                    "r2_control": {"sha256": checkpoints_manifest[f"seed_{s}"]["r2_control"]["sha256"]},
                    "priority_variant": {"sha256": checkpoints_manifest[f"seed_{s}"]["priority_variant"]["sha256"]},
                },
            }
            for s in TRAINING_SEEDS
        },
        "hard_gates": {g: True for g in EXPECTED_EVAL_HARD_GATES},
        "total_games_evaluated": 3000,
        "verdict": "evaluation_completed",
    }
    ev_path = ev_dir / "f1_eval_manifest.json"
    ev_path.write_text(json.dumps(ev_man), encoding="utf-8")

    return {"tr_dir": tr_dir, "ev_dir": ev_dir, "sm_dir": sm_dir, "tr_path": tr_path, "ev_path": ev_path}


def test_8_f1_summary_pipeline_and_fail_closed(tmp_path: Path) -> None:
    """Test 8: Full summary pipeline plus fail-closed rejections for all tamper classes."""
    fx = _build_strict_fixture(tmp_path / "valid")
    summary = adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])
    assert summary["schema"] == "keqing.mortal.f1_summary.v1"
    assert summary["hard_gates"]["all_3000_logs_verified"] is True
    assert summary["metrics"]["total_games"] == 3000
    assert summary["training_manifest"]["sha256"] == sha256_file(fx["tr_path"])
    assert summary["eval_manifest"]["sha256"] == sha256_file(fx["ev_path"])
    # Variant wins over both Control and K0 in the mock logs -> promotion_supported.
    assert summary["verdict"] == "promotion_supported"
    assert summary["promotion"]["k1"] == "mortal_priority_variant_70400_seed_20260911.pth"
    assert summary["selection_stats"]["seed_20260910"]["high_signal_count_min"] == 50

    # (a) Base row digest mismatch with frozen R2 Control must be rejected.
    fx = _build_strict_fixture(tmp_path / "row_mismatch")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260910"]["base_row_sha256"] = "0" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (b) Q-margin threshold introduction must be rejected.
    fx = _build_strict_fixture(tmp_path / "margin_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["prioritized_replay"]["q_margin_threshold"] = 0.5
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (c) Reward shaping leakage into main_reward must be rejected.
    fx = _build_strict_fixture(tmp_path / "main_reward_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["main_reward"]["mode"] = "rank_plus_score_to_go_mc"
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (d) Frozen R2 Control checkpoint SHA tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "r2c_sha_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["checkpoints"]["seed_20260911"]["r2_control"]["sha256"] = "f" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (e) Checkpoint disk tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "ckpt_tamper")
    (fx["tr_dir"] / "mortal_priority_variant_70400_seed_20260910.pth").write_bytes(b"tampered")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (f) Duplicate game ID must be rejected.
    fx = _build_strict_fixture(tmp_path / "dup_id")
    logs = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs"
    (logs / f"{2500001}_{EVAL_SEED_KEY}_b.json.gz").unlink()
    shutil.copy(logs / f"{2500000}_{EVAL_SEED_KEY}_a.json.gz", logs / f"{2500000}_{EVAL_SEED_KEY}_c.json.gz")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (g) Wrong lineup in one log must be rejected.
    fx = _build_strict_fixture(tmp_path / "lineup_tamper")
    bad = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs" / f"{2500000}_{EVAL_SEED_KEY}_a.json.gz"
    _write_game_log(bad, 2500000, ["K0_70k", "ext_mortal", "R2_Control_seed_20260910", "Intruder"])
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (h) Training-manifest SHA chain rewrite must be rejected.
    fx = _build_strict_fixture(tmp_path / "sha_chain")
    with open(fx["tr_path"], "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (i) Missing eval hard gate must be rejected.
    fx = _build_strict_fixture(tmp_path / "gate_tamper")
    data = json.loads(fx["ev_path"].read_text(encoding="utf-8"))
    data["hard_gates"].pop("zero_missing_games")
    fx["ev_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (j) Per-step high_signal_count below the fail-closed minimum must be rejected.
    fx = _build_strict_fixture(tmp_path / "hs_below_min")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260912"]["selection_stats"]["per_step"][0]["high_signal_count"] = 15
    data["row_identity"]["by_seed"]["seed_20260912"]["selection_stats"]["high_signal_count_min"] = 15
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (k) Scorer bit-exact flag false must be rejected.
    fx = _build_strict_fixture(tmp_path / "scorer_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["frozen_scorer"]["parameters_bit_exact"] = False
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])


def test_9_crossed_bootstrap_and_four_state_verdict() -> None:
    """Test 9: Crossed bootstrap shared resampling and four-state F1 verdict logic."""
    rng = np.random.default_rng(42)
    mat1 = rng.normal(loc=5.0, scale=20.0, size=(3, 1000))
    mat2 = mat1 + 2.0

    m1, ci1, idx = crossed_bootstrap_ci(mat1, reps=500, seed=20260930, ci=95.0, return_sampled_indices=True)
    assert idx is not None
    assert idx[0].shape == (500, 3)
    assert idx[1].shape == (500, 1000)
    m2, ci2, _ = crossed_bootstrap_ci(mat2, reps=500, seed=20260930, ci=95.0, shared_indices=idx)
    assert abs(m2 - (m1 + 2.0)) < 1e-6
    assert abs(ci2[0] - (ci1[0] + 2.0)) < 1e-6
    assert abs(ci2[1] - (ci1[1] + 2.0)) < 1e-6

    # promotion_supported: both primary and absolute fully pass.
    v, r, c, k = adjudicate_f1_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "promotion_supported" and r is True and c is True
    assert k == "mortal_priority_variant_70400_seed_20260911.pth"

    # sampling_effect_only: primary passes, absolute fails.
    v, r, c, k = adjudicate_f1_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, -1.0, 10.0],
        absolute_ci_lower=-0.5,
    )
    assert v == "sampling_effect_only" and r is False and c is False and k is None

    # not_supported: primary seed mean negative.
    v, r, c, k = adjudicate_f1_verdict(
        primary_seed_means=[5.0, -2.0, 7.0],
        primary_ci_lower=-1.0,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None

    # not_supported: primary CI crosses zero.
    v, r, c, k = adjudicate_f1_verdict(
        primary_seed_means=[2.0, 1.0, 3.0],
        primary_ci_lower=-0.5,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None

    # not_supported: primary fewer than 3 seed means.
    v, r, c, k = adjudicate_f1_verdict(
        primary_seed_means=[2.0, 1.0],
        primary_ci_lower=1.0,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None


def test_10_exact_seed_set_enforcement() -> None:
    """Test 10: All F1 runners reject any seed set other than the frozen three."""
    from training.mortal.f1_disagreement_prioritized_replay_contract_2026_09 import validate_f1_seed_set

    assert validate_f1_seed_set(None) == [20260910, 20260911, 20260912]
    assert validate_f1_seed_set([20260910, 20260911, 20260912]) == [20260910, 20260911, 20260912]

    for bad in (
        [20260910],
        [20260910, 20260911],
        [20260910, 20260911, 20260912, 20260913],
        [20260910, 20260912, 20260911],
        [20260911, 20260912, 20260913],
    ):
        with pytest.raises(ContractError):
            validate_f1_seed_set(bad)

    # Runners fail closed on wrong seed sets before touching artifacts.
    from training.mortal.eval_f1_disagreement_prioritized_replay_2026_09 import run_f1_evaluation
    from training.mortal.summary_f1_disagreement_prioritized_replay_2026_09 import adjudicate_f1_prioritized_replay
    from training.mortal.train_f1_disagreement_prioritized_replay_2026_09 import run_f1_training

    with pytest.raises(ContractError):
        run_f1_training(seeds=[20260910])
    with pytest.raises(ContractError):
        run_f1_evaluation(seeds=[20260910, 20260911])
    with pytest.raises(ContractError):
        adjudicate_f1_prioritized_replay(seeds=[20260913])


def test_11_checkpoint_evaluator_compatibility() -> None:
    """Test 11: F1 checkpoint keeps the standard four_player_native-loadable structure."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net = _build_models(k0_path, "cpu")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "mortal_priority_variant_70400_seed_20260910.pth"
        save_state = {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "aux_net": aux_net.state_dict(),
            "steps": 70400,
            "condition": "behavior_disagreement_prioritized_replay",
            "config": {
                "control": {"version": 4, "online": False, "batch_size": 512},
                "resnet": {"conv_channels": 192, "num_blocks": 40},
            },
        }
        torch.save(save_state, ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        m = model.Brain(version=4, conv_channels=192, num_blocks=40)
        q = model.DQN(version=4)
        a = model.AuxNet((4,))
        m.load_state_dict(state["mortal"])
        q.load_state_dict(state["current_dqn"])
        a.load_state_dict(state["aux_net"])
        # No new heads, no structural additions.
        assert "score_aux_head" not in state
        assert state["condition"] == "behavior_disagreement_prioritized_replay"


def test_12_dataloader_interface_unchanged() -> None:
    """Test 12: F1 dataloader consumes the standard 6-field interface (mainline_dataloader untouched)."""
    m0_path, _ = resolve_m0_dataset_index()
    seed = 20260910
    dl = _build_dataloader(m0_path, seed=seed, batch_size=16)
    batch = next(iter(dl))
    assert len(batch) == 6
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    assert np.asarray(kyoku_rewards).dtype == np.float64
    # Determinism: identical rebuild reproduces the identical first batch.
    dl2 = _build_dataloader(m0_path, seed=seed, batch_size=16)
    batch2 = next(iter(dl2))
    for a, b in zip(batch, batch2, strict=True):
        assert torch.equal(a, b)


def test_13_sampler_fail_closed_below_16() -> None:
    """Test 13: Fewer than 16 disagreement rows in any batch fails closed; no threshold fallback."""
    for count in (0, 1, 15):
        mask = np.zeros(512, dtype=bool)
        mask[:count] = True
        with pytest.raises(ContractError):
            select_priority_batch(mask, canonical_step_rng(EXPERIMENT_ID, 20260910, 1))
    # 16 exactly passes (no temporary threshold lowering anywhere).
    mask = np.zeros(512, dtype=bool)
    mask[:16] = True
    sel = select_priority_batch(mask, canonical_step_rng(EXPERIMENT_ID, 20260910, 1))
    assert sel["high_signal_count"] == 16
