"""Targeted unit tests for R1 rank_plus_score_to_go pilot contracts, math, and runner stubs."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

from training.mortal.objective import compute_objective_losses
from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    ADAM_BETAS,
    ADAM_EPS,
    BATCH_SIZE,
    BOOTSTRAP_SEED,
    EVAL_GAMES_PER_SHARD,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_SHARDS,
    EVAL_TOTAL_GAMES,
    EVALUATION_LINEUP,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPECTED_TRAINING_HARD_GATES,
    EXPERIMENT_ID,
    LEARNING_RATE,
    LOG_NAME_RE,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    ROW_IDENTITY_FIELDS,
    SCORE_TO_GO_SCALE,
    SCORE_TO_GO_WEIGHT,
    STEPS_START,
    STEPS_TARGET,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEED,
    WEIGHT_DECAY,
    ContractError,
    adjudicate_r1_verdict,
    compute_r1_target,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    reward_contract_for_condition,
    sha256_file,
    update_row_identity_digest,
    verify_training_manifest,
)
from training.mortal.summary_r1_rank_plus_score_to_go_2026_09 import (
    adjudicate_r1_pilot,
)
from training.mortal.train_r1_rank_plus_score_to_go_2026_09 import (
    _build_dataloader,
    _build_optimizer_and_models,
)


def test_1_r1_contract_invariants() -> None:
    """Test 1: Check R1 frozen constants, reward-only protocol fields, and hard gate sets."""
    assert EXPERIMENT_ID == "R1_rank_plus_score_to_go_pilot_2026_09"
    assert TRAINING_SEED == 20260807
    assert STEPS_START == 70000
    assert STEPS_TARGET == 70400
    assert OPTIMIZER_STEPS == 400
    assert BATCH_SIZE == 512
    assert LEARNING_RATE == 1e-4
    assert WEIGHT_DECAY == 0.1
    assert ADAM_BETAS == (0.9, 0.999)
    assert ADAM_EPS == 1e-8
    assert SCORE_TO_GO_WEIGHT == 0.25
    assert SCORE_TO_GO_SCALE == 10000.0
    assert BOOTSTRAP_SEED == 20260906

    # R1 is reward-only: M0 objective and the single authoritative trainable label.
    assert OBJECTIVE_MODE == "behavior_action_mc"
    assert OBJECTIVE_VALUE_STATISTIC == "behavior_action_q"
    assert TRAINABLE_PLAYER_NAMES == ("ext_mortal",)

    # Row identity covers every batch field except the reward.
    assert [name for name, _ in ROW_IDENTITY_FIELDS] == [
        "obs",
        "actions",
        "masks",
        "steps_to_done",
        "player_ranks",
    ]

    # Evaluation lineup and log filename contract.
    assert EVALUATION_LINEUP == ("K0_70k", "ext_mortal", "Control_70400", "Variant_70400")
    assert LOG_NAME_RE.match(f"{EVAL_SEED_START}_{EVAL_SEED_KEY}_0.json.gz") is not None
    assert LOG_NAME_RE.match("bad_name.json.gz") is None

    # Reward contracts for both conditions.
    ctrl_reward = reward_contract_for_condition("control")
    assert ctrl_reward == {"mode": "final_rank_mc", "rank_pts": [6.0, 4.0, 2.0, 0.0]}
    var_reward = reward_contract_for_condition("variant")
    assert var_reward["mode"] == "rank_plus_score_to_go_mc"
    assert var_reward["score_to_go_weight"] == 0.25
    assert var_reward["score_to_go_scale"] == 10000.0
    assert var_reward["score_to_go_clip_min"] == -3.0
    assert var_reward["score_to_go_clip_max"] == 3.0

    assert len(EXPECTED_TRAINING_HARD_GATES) == 9
    assert len(EXPECTED_EVAL_HARD_GATES) == 7
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    # Verify K0, ext_mortal checkpoint, and M0 index paths exist
    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists()
    assert len(k0_sha) == 64

    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    assert ext_path.exists()
    assert len(ext_sha) == 64

    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists()
    assert len(m0_sha) == 64


def test_2_r1_target_mathematics_and_clipping() -> None:
    """Test 2: Test scalar R1 target calculation and clipping."""
    # Centered pts: [3.0, 1.0, -1.0, -3.0]
    # Case 1: 1st place (+3.0), +15000 score gained -> diff=+15000 -> 15000/10000 = +1.5 -> +3.0 + 0.25*1.5 = +3.375
    t1 = compute_r1_target(final_rank=0, final_score=40000.0, score_at_current_kyoku_start=25000.0)
    assert t1 == 3.0 + 0.25 * 1.5
    assert t1 == 3.375

    # Case 2: 1st place (+3.0), +50000 score gained -> diff=+50000 -> 50000/10000 = +5.0, clipped to +3.0 -> +3.0 + 0.25*3.0 = +3.75
    t2 = compute_r1_target(final_rank=0, final_score=75000.0, score_at_current_kyoku_start=25000.0)
    assert t2 == 3.0 + 0.25 * 3.0
    assert t2 == 3.75

    # Case 3: 4th place (-3.0), -50000 score lost -> diff=-50000 -> -5.0, clipped to -3.0 -> -3.0 + 0.25*(-3.0) = -3.75
    t3 = compute_r1_target(final_rank=3, final_score=-25000.0, score_at_current_kyoku_start=25000.0)
    assert t3 == -3.75

    # Case 4: 2nd place (+1.0), +5000 score gained -> diff=+5000 -> score_to_go = +0.5 -> target = +1.0 + 0.25*0.5 = +1.125
    t4 = compute_r1_target(final_rank=1, final_score=30000.0, score_at_current_kyoku_start=25000.0)
    assert t4 == 1.0 + 0.25 * 0.5
    assert t4 == 1.125


def test_3_real_k0_and_m0_one_step_zero_update_smoke() -> None:
    """Test 3: Real K0 + real M0 ext_mortal view: full-batch row-identity digests match, rewards differ, losses finite."""
    k0_path, _ = resolve_k0_checkpoint()
    m0_path, _ = resolve_m0_dataset_index()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    brain, dqn, aux_net, _optimizer = _build_optimizer_and_models(k0_path, device)
    dl_ctrl = _build_dataloader(m0_path, seed=20260807, reward_mode="final_rank_mc", batch_size=32)
    batch_ctrl = next(iter(dl_ctrl))

    dl_var = _build_dataloader(m0_path, seed=20260807, reward_mode="rank_plus_score_to_go_mc", batch_size=32)
    batch_var = next(iter(dl_var))

    # 1. Rolling row-identity digests (reward excluded) must match exactly.
    digest_ctrl = hashlib.sha256()
    update_row_identity_digest(
        digest_ctrl,
        obs=batch_ctrl[0],
        actions=batch_ctrl[1],
        masks=batch_ctrl[2],
        steps_to_done=batch_ctrl[3],
        player_ranks=batch_ctrl[5],
    )
    digest_var = hashlib.sha256()
    update_row_identity_digest(
        digest_var,
        obs=batch_var[0],
        actions=batch_var[1],
        masks=batch_var[2],
        steps_to_done=batch_var[3],
        player_ranks=batch_var[5],
    )
    assert digest_ctrl.hexdigest() == digest_var.hexdigest()

    # 2. Verify kyoku_rewards are different due to score_to_go
    assert not torch.allclose(batch_ctrl[4], batch_var[4])

    # 3. Test forward and objective losses on control with the frozen M0 objective
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = [t.to(device) for t in batch_ctrl]
    q_target_mc = (1.0 ** steps_to_done * kyoku_rewards).to(torch.float32)

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
        mode=OBJECTIVE_MODE,
        cql_weight=5.0,
        aux_weight=0.2,
    )
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["value_loss"])
    assert torch.isfinite(losses["cql_loss"])
    assert torch.isfinite(losses["next_rank_loss"])


def _write_game_log(log_path: Path, seed: int, names: list[str]) -> None:
    events = [
        {"type": "start_game", "seed": [seed, EVAL_SEED_KEY], "names": names},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": [0, 0, -8000, 8000]},  # Variant wins over Control
        {"type": "end_game"},
    ]
    with gzip.open(log_path, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _build_valid_fixture(root: Path, with_logs: bool = True) -> dict:
    """Build a complete, contract-conformant R1 artifact fixture under root."""
    tr_dir = root / "training"
    ev_dir = root / "evaluation"
    sm_dir = root / "summary"
    tr_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)
    sm_dir.mkdir(parents=True)

    ctrl_ckpt = tr_dir / "mortal_control_70400.pth"
    var_ckpt = tr_dir / "mortal_variant_70400.pth"
    ctrl_ckpt.write_bytes(b"ctrl_mock_checkpoint_bytes")
    var_ckpt.write_bytes(b"variant_mock_checkpoint_bytes")
    ctrl_sha = sha256_file(ctrl_ckpt)
    var_sha = sha256_file(var_ckpt)

    _, k0_sha = resolve_k0_checkpoint()
    _, ext_sha = resolve_ext_mortal_checkpoint()
    m0_path, m0_sha = resolve_m0_dataset_index()
    row_digest = hashlib.sha256(b"r1_row_identity").hexdigest()

    tr_manifest = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "objective": {
            "mode": OBJECTIVE_MODE,
            "value_statistic": OBJECTIVE_VALUE_STATISTIC,
            "preference_loss": "existing_cql",
        },
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "training_config": {"optimizer_steps": OPTIMIZER_STEPS},
        "checkpoints": {
            "control": {
                "name": ctrl_ckpt.name,
                "path": str(ctrl_ckpt),
                "sha256": ctrl_sha,
                "reward": reward_contract_for_condition("control"),
            },
            "variant": {
                "name": var_ckpt.name,
                "path": str(var_ckpt),
                "sha256": var_sha,
                "reward": reward_contract_for_condition("variant"),
            },
        },
        "row_identity": {
            "fields": [name for name, _ in ROW_IDENTITY_FIELDS],
            "excluded_field": "kyoku_rewards",
            "control_sha256": row_digest,
            "variant_sha256": row_digest,
            "identical": True,
        },
        "hard_gates": {g: True for g in EXPECTED_TRAINING_HARD_GATES},
        "verdict": "training_completed",
    }
    tr_manifest_path = tr_dir / "r1_training_manifest.json"
    tr_manifest_path.write_text(json.dumps(tr_manifest), encoding="utf-8")
    tr_manifest_sha = sha256_file(tr_manifest_path)

    if with_logs:
        for shard_idx in range(EVAL_SHARDS):
            logs_dir = ev_dir / f"shard_{shard_idx:03d}" / "logs"
            logs_dir.mkdir(parents=True)
            s_start = EVAL_SEED_START + shard_idx * EVAL_GAMES_PER_SHARD
            for s in range(s_start, s_start + EVAL_GAMES_PER_SHARD):
                _write_game_log(logs_dir / f"{s}_{EVAL_SEED_KEY}_0.json.gz", s, list(EVALUATION_LINEUP))

    ev_manifest = {
        "schema": EVAL_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_manifest_path), "sha256": tr_manifest_sha},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "lineup": list(EVALUATION_LINEUP),
        "models": {
            "k0": {"name": "K0_70k", "path": "mock_k0", "sha256": k0_sha},
            "ext_mortal": {"name": "ext_mortal", "path": "mock_ext", "sha256": ext_sha},
            "control": {"name": "Control_70400", "path": str(ctrl_ckpt), "sha256": ctrl_sha},
            "variant": {"name": "Variant_70400", "path": str(var_ckpt), "sha256": var_sha},
        },
        "eval_config": {"total_games": EVAL_TOTAL_GAMES},
        "game_id_range": [EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE],
        "hard_gates": {g: True for g in EXPECTED_EVAL_HARD_GATES},
        "games_count": EVAL_TOTAL_GAMES,
        "verdict": "evaluation_completed",
    }
    (ev_dir / "r1_eval_manifest.json").write_text(json.dumps(ev_manifest), encoding="utf-8")

    return {
        "tr_dir": tr_dir,
        "ev_dir": ev_dir,
        "sm_dir": sm_dir,
        "tr_manifest_path": tr_manifest_path,
        "ctrl_ckpt": ctrl_ckpt,
        "var_ckpt": var_ckpt,
    }


def _tamper_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_4_adjudicate_r1_pilot_verdicts_and_mock(tmp_path: Path) -> None:
    """Test 4: Fail-closed adjudication pipeline accepts a contract-conformant fixture and yields the expected verdict."""
    # Test 4-level verdict logic
    assert adjudicate_r1_verdict(primary_mean=2.5, primary_ci_lower=0.5) == "strong_positive"
    assert adjudicate_r1_verdict(primary_mean=2.5, primary_ci_lower=-0.5) == "weak_positive"
    assert adjudicate_r1_verdict(primary_mean=-1.0, primary_ci_lower=-3.0) == "not_promising"
    assert adjudicate_r1_verdict(primary_mean=0.0, primary_ci_lower=-2.0) == "not_promising"

    fx = _build_valid_fixture(tmp_path / "valid")
    summary = adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])
    assert summary["schema"] == "keqing.mortal.r1_summary.v1"
    assert summary["experiment_id"] == EXPERIMENT_ID
    assert summary["hard_gates"] == {g: True for g in EXPECTED_SUMMARY_HARD_GATES}
    assert summary["metrics"]["total_games"] == 1000
    assert summary["evaluation_protocol"]["lineup"] == list(EVALUATION_LINEUP)
    assert summary["evaluation_protocol"]["game_id_range"] == [EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE]
    assert summary["metrics"]["primary_contrast_variant_minus_control"]["mean_pt"] > 0
    assert summary["verdict"] == "strong_positive"
    # SHA chain bindings are recorded.
    assert summary["training_manifest"]["sha256"] == sha256_file(fx["tr_manifest_path"])


def test_5_summary_fail_closed_rejections(tmp_path: Path) -> None:
    """Test 5: Adjudication rejects every tampering class with ContractError."""
    # (a) Duplicate / missing game IDs break the unique-contiguous requirement.
    fx = _build_valid_fixture(tmp_path / "duplicate_id")
    shard3_logs = fx["ev_dir"] / "shard_003" / "logs"
    source_log = fx["ev_dir"] / "shard_000" / "logs" / f"{EVAL_SEED_START}_{EVAL_SEED_KEY}_0.json.gz"
    (shard3_logs / f"{EVAL_SEED_START + EVAL_TOTAL_GAMES - 1}_{EVAL_SEED_KEY}_0.json.gz").unlink()
    shutil.copy(source_log, shard3_logs / f"{EVAL_SEED_START}_{EVAL_SEED_KEY}_dup.json.gz")
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (b) Tampered checkpoint bytes break the three-way SHA chain.
    fx = _build_valid_fixture(tmp_path / "ckpt_tamper", with_logs=False)
    fx["ctrl_ckpt"].write_bytes(b"tampered_checkpoint")
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (c) Objective other than behavior_action_mc is rejected.
    fx = _build_valid_fixture(tmp_path / "objective_tamper", with_logs=False)
    _tamper_json(fx["tr_manifest_path"], lambda d: d["objective"].__setitem__("mode", "legal_mean_mc"))
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (d) Trainable labels other than ext_mortal are rejected.
    fx = _build_valid_fixture(tmp_path / "label_tamper", with_logs=False)
    _tamper_json(fx["tr_manifest_path"], lambda d: d.__setitem__("trainable_player_names", ["mortal"]))
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (e) A missing eval hard gate is rejected.
    fx = _build_valid_fixture(tmp_path / "gate_tamper", with_logs=False)
    _tamper_json(
        fx["ev_dir"] / "r1_eval_manifest.json",
        lambda d: d["hard_gates"].pop("zero_missing_games"),
    )
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (f) Rewriting the training manifest after eval binding breaks the SHA chain.
    fx = _build_valid_fixture(tmp_path / "sha_chain_tamper", with_logs=False)
    with open(fx["tr_manifest_path"], "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (g) A wrong lineup name in any game log is rejected.
    fx = _build_valid_fixture(tmp_path / "lineup_tamper")
    bad_log = fx["ev_dir"] / "shard_000" / "logs" / f"{EVAL_SEED_START}_{EVAL_SEED_KEY}_0.json.gz"
    _write_game_log(bad_log, EVAL_SEED_START, ["K0_70k", "ext_mortal", "Control_70400", "Intruder"])
    with pytest.raises(ContractError):
        adjudicate_r1_pilot(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])


def test_6_row_identity_digest_and_log_parser_contracts(tmp_path: Path) -> None:
    """Test 6: Row-identity digest properties and fail-closed game-log identity parsing."""
    obs = torch.randn(8, 16)
    actions = torch.randint(0, 5, (8,))
    masks = torch.ones(8, 5, dtype=torch.bool)
    steps = torch.randint(0, 3, (8,))
    ranks = torch.randint(0, 4, (8,))

    def digest(obs_t=obs) -> str:
        d = hashlib.sha256()
        update_row_identity_digest(
            d, obs=obs_t, actions=actions, masks=masks, steps_to_done=steps, player_ranks=ranks
        )
        return d.hexdigest()

    # Deterministic, and any field change flips the digest.
    assert digest() == digest()
    assert digest(obs_t=obs.clone()) == digest()
    assert digest(obs_t=obs + 1.0) != digest()

    # verify_training_manifest: a valid manifest passes, and each newly-checked field is enforced.
    fx = _build_valid_fixture(tmp_path / "row_digest", with_logs=False)
    valid_man = json.loads(fx["tr_manifest_path"].read_text(encoding="utf-8"))
    assert verify_training_manifest(valid_man) is True

    def reject_after(mutate) -> None:
        data = json.loads(json.dumps(valid_man))
        mutate(data)
        with pytest.raises(ContractError):
            verify_training_manifest(data)

    reject_after(lambda d: d["parent_model"].__setitem__("sha256", "f" * 64))
    reject_after(lambda d: d["dataset"].__setitem__("sha256", "f" * 64))
    reject_after(lambda d: d["objective"].__setitem__("value_statistic", "mean_legal_q"))
    reject_after(lambda d: d["row_identity"].__setitem__("fields", ["obs", "actions"]))
    reject_after(lambda d: d["row_identity"].__setitem__("excluded_field", "something_else"))
    reject_after(lambda d: d["row_identity"].__setitem__("identical", False))
    reject_after(lambda d: d["row_identity"].__setitem__("variant_sha256", "0" * 64))

    # Parser: valid log passes.
    seed = EVAL_SEED_START
    good_log = tmp_path / f"{seed}_{EVAL_SEED_KEY}_0.json.gz"
    _write_game_log(good_log, seed, list(EVALUATION_LINEUP))
    ident = parse_game_identity(good_log)
    assert ident["game_id"] == seed
    assert set(ident["names"]) == set(EVALUATION_LINEUP)

    # Parser rejects wrong seed key in filename.
    bad_key_log = tmp_path / f"{seed}_9999_0.json.gz"
    _write_game_log(bad_key_log, seed, list(EVALUATION_LINEUP))
    with pytest.raises(ContractError):
        parse_game_identity(bad_key_log)

    # Parser rejects filename/start_game seed disagreement.
    mismatch_log = tmp_path / f"{seed + 1}_{EVAL_SEED_KEY}_0.json.gz"
    _write_game_log(mismatch_log, seed, list(EVALUATION_LINEUP))
    with pytest.raises(ContractError):
        parse_game_identity(mismatch_log)

    # Parser rejects non-conforming filenames and wrong lineups.
    bad_name_log = tmp_path / "not_a_game.json.gz"
    _write_game_log(bad_name_log, seed, list(EVALUATION_LINEUP))
    with pytest.raises(ContractError):
        parse_game_identity(bad_name_log)

    bad_lineup_log = tmp_path / f"{seed + 2}_{EVAL_SEED_KEY}_0.json.gz"
    _write_game_log(bad_lineup_log, seed + 2, ["K0_70k", "ext_mortal", "Control_70400", "Intruder"])
    with pytest.raises(ContractError):
        parse_game_identity(bad_lineup_log)
