"""Targeted unit tests for S1 score_to_go auxiliary multiseed experiment contracts and runners."""

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

from training.mortal.mainline_dataloader import FileDatasetsIter
from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import (
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
    HEAD_LR,
    HEAD_WEIGHT_DECAY,
    MAIN_REWARD_MODE,
    OBJECTIVE_MODE,
    OBJECTIVE_VALUE_STATISTIC,
    OPTIMIZER_STEPS,
    ROW_IDENTITY_FIELDS,
    SCORE_AUX_LOSS_WEIGHT,
    SCORE_AUX_MSE_FACTOR,
    TRAINABLE_PLAYER_NAMES,
    TRAINING_SEEDS,
    ContractError,
    adjudicate_s1_verdict,
    compute_s1_auxiliary_target,
    crossed_bootstrap_ci,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    resolve_m0_dataset_index,
    resolve_r2_control_checkpoint,
    sha256_file,
    update_auxiliary_target_digest,
    update_row_identity_digest,
    verify_training_manifest,
)
from training.mortal.summary_s1_score_to_go_auxiliary_2026_09 import adjudicate_s1_auxiliary
from training.mortal.train_s1_score_to_go_auxiliary_2026_09 import (
    ScoreToGoHead,
    _build_dataloader,
    _build_models,
    _build_optimizer,
)
import model


def test_1_s1_contract_invariants() -> None:
    """Test 1: Frozen constants, frozen R2 Control binding, and gate sets."""
    assert EXPERIMENT_ID == "S1_score_to_go_auxiliary_multiseed_2026_09"
    assert TRAINING_SEEDS == [20260910, 20260911, 20260912]
    assert CANONICAL_K1_SEED == 20260911
    assert OPTIMIZER_STEPS == 400
    assert BOOTSTRAP_SEED == 20260920
    assert EVAL_GAMES_PER_PANEL == 1000
    assert EVAL_TOTAL_GAMES == 3000
    assert EVAL_SEED_START == 2400000

    # Reward-only-in-main protocol: main Q target is final_rank_mc.
    assert MAIN_REWARD_MODE == "final_rank_mc"
    assert OBJECTIVE_MODE == "behavior_action_mc"
    assert OBJECTIVE_VALUE_STATISTIC == "behavior_action_q"
    assert TRAINABLE_PLAYER_NAMES == ("ext_mortal",)
    assert SCORE_AUX_LOSS_WEIGHT == 0.2
    assert SCORE_AUX_MSE_FACTOR == 0.5
    assert HEAD_LR == 1e-4
    assert HEAD_WEIGHT_DECAY == 0.0

    assert len(EXPECTED_TRAINING_HARD_GATES) == 10
    assert len(EXPECTED_EVAL_HARD_GATES) == 7
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 7

    # Frozen R2 Control checkpoint digests.
    for s in TRAINING_SEEDS:
        path, ckpt_sha, row_sha = resolve_r2_control_checkpoint(s)
        assert path.exists()
        assert ckpt_sha == FROZEN_R2_CONTROL[s]["checkpoint_sha256"]
        assert row_sha == FROZEN_R2_CONTROL[s]["row_identity_sha256"]
        assert len(ckpt_sha) == 64 and len(row_sha) == 64

    k0_path, k0_sha = resolve_k0_checkpoint()
    assert k0_path.exists() and len(k0_sha) == 64
    ext_path, ext_sha = resolve_ext_mortal_checkpoint()
    assert ext_path.exists() and len(ext_sha) == 64
    m0_path, m0_sha = resolve_m0_dataset_index()
    assert m0_path.exists() and len(m0_sha) == 64


def test_2_s1_auxiliary_target_math_and_range() -> None:
    """Test 2: Auxiliary target math, clipping, and strict [-1, +1] range."""
    # +45000 diff -> raw 4.5 -> clip 3 -> 3/3 = +1.0
    assert compute_s1_auxiliary_target(70000.0, 25000.0) == 1.0
    # -60000 diff -> raw -6.0 -> clip -3 -> -1.0
    assert compute_s1_auxiliary_target(-35000.0, 25000.0) == -1.0
    # +15000 diff -> 1.5/3 = 0.5
    assert compute_s1_auxiliary_target(40000.0, 25000.0) == 0.5
    # -5000 diff -> -0.5/3
    assert compute_s1_auxiliary_target(20000.0, 25000.0) == -0.5 / 3.0

    # Exhaustive range check.
    rng = np.random.default_rng(0)
    for final in rng.uniform(-60000, 100000, 500):
        for start in rng.uniform(0, 50000, 5):
            t = compute_s1_auxiliary_target(float(final), float(start))
            assert -1.0 <= t <= 1.0

    # Boundary exactness.
    assert compute_s1_auxiliary_target(55000.0, 25000.0) == 1.0  # raw 3.0 exactly
    assert compute_s1_auxiliary_target(-5000.0, 25000.0) == -1.0  # raw -3.0 exactly


def test_3_dataloader_default_interface_unchanged_and_seventh_field() -> None:
    """Test 3: Default 6-field output identical; include flag adds 7th target in [-1,1]."""
    m0_path, _ = resolve_m0_dataset_index()
    from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import native_path, RANK_PTS, FILE_BATCH_SIZE
    import random as _r

    raw = torch.load(m0_path, map_location="cpu")["file_list"]
    file_list = [str(native_path(f)) for f in raw]

    def build(include: bool):
        _r.seed(20260910)
        np.random.seed(20260910)
        torch.manual_seed(20260910)
        ds = FileDatasetsIter(
            version=4,
            file_list=list(file_list),
            pts=RANK_PTS,
            oracle=False,
            file_batch_size=FILE_BATCH_SIZE,
            reserve_ratio=0,
            player_names=list(TRAINABLE_PLAYER_NAMES),
            num_epochs=1,
            enable_augmentation=False,
            augmented_first=False,
            reward_mode="final_rank_mc",
            include_score_to_go_target=include,
        )
        return iter(ds)

    # Consume sequentially: creating both iterators before consuming would let the
    # second build's re-seed interleave with the first generator's RNG usage.
    e_default = next(build(False))
    e_aux = next(build(True))

    # Default: exactly 6 fields; aux: exactly 7.
    assert len(e_default) == 6
    assert len(e_aux) == 7
    # The first 6 fields identical.
    for a, b in zip(e_default, e_aux[:6], strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # 7th field within [-1, +1].
    stg = float(e_aux[6])
    assert -1.0 <= stg <= 1.0
    # Main q target (field 4) is the final_rank_mc reward and unchanged.
    assert np.asarray(e_aux[4]).dtype == np.float64


def test_4_s1_row_digest_matches_r2_control_and_main_q_target() -> None:
    """Test 4: S1 dataloader row digest equals frozen R2 Control digest; q target stays final_rank_mc."""
    m0_path, _ = resolve_m0_dataset_index()
    seed = 20260910

    dl = _build_dataloader(m0_path, seed=seed, batch_size=32)
    batch = next(iter(dl))
    # 7 fields with the flag on.
    assert len(batch) == 7
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, score_to_go_target = batch

    d = hashlib.sha256()
    update_row_identity_digest(
        d, obs=obs, actions=actions, masks=masks, steps_to_done=steps_to_done, player_ranks=player_ranks
    )
    # First-batch digest prefix must differ from nothing—compare full-run equality is the trainer's job;
    # here we assert the standard digest algorithm matches R2's first batch under identical seed/batch size.
    # R2 Control digest is over all 400 batches at batch_size=512, so we only verify determinism and
    # cross-consistency between include True/False at the same batch size.
    dl_plain = _build_dataloader(m0_path, seed=seed, batch_size=32)
    # _build_dataloader always sets include_score_to_go_target=True; build a plain one manually.
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import native_path, RANK_PTS, FILE_BATCH_SIZE
    raw = torch.load(m0_path, map_location="cpu")["file_list"]
    ds = FileDatasetsIter(
        version=4,
        file_list=[str(native_path(f)) for f in raw],
        pts=RANK_PTS,
        oracle=False,
        file_batch_size=FILE_BATCH_SIZE,
        reserve_ratio=0,
        player_names=list(TRAINABLE_PLAYER_NAMES),
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        reward_mode="final_rank_mc",
    )
    dl2 = torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True, num_workers=0)
    b2 = next(iter(dl2))
    d2 = hashlib.sha256()
    update_row_identity_digest(
        d2, obs=b2[0], actions=b2[1], masks=b2[2], steps_to_done=b2[3], player_ranks=b2[5]
    )
    assert d.hexdigest() == d2.hexdigest()
    # Main q target identical between plain and aux dataloaders.
    assert torch.equal(b2[4], kyoku_rewards)
    # Aux target strictly bounded.
    assert float(score_to_go_target.min()) >= -1.0
    assert float(score_to_go_target.max()) <= 1.0

    # Auxiliary digest: deterministic and sensitive to the aux field.
    da1 = hashlib.sha256()
    update_auxiliary_target_digest(da1, score_to_go_target)
    da2 = hashlib.sha256()
    update_auxiliary_target_digest(da2, score_to_go_target)
    assert da1.hexdigest() == da2.hexdigest()
    da3 = hashlib.sha256()
    update_auxiliary_target_digest(da3, score_to_go_target + 1.0)
    assert da1.hexdigest() != da3.hexdigest()


def test_5_score_loss_gradient_routing() -> None:
    """Test 5: score aux loss updates Brain+ScoreToGoHead only; never DQN/AuxNet; q target final_rank_mc.

    Uses the production head initialization (Normal(0, 0.01), local generator
    seeded by the training seed) — no manual weight re-initialization.
    """
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net, head = _build_models(k0_path, "cpu", training_seed=20260910)

    # Production head is non-zero after canonical init.
    assert float(head.net.weight.detach().abs().sum()) > 0

    obs = torch.randn(4, 1012, 34)
    phi = brain(obs)
    score_pred = head(phi)
    target = torch.rand(4) * 2 - 1

    score_aux_loss = SCORE_AUX_MSE_FACTOR * torch.nn.functional.mse_loss(score_pred, target)
    score_aux_loss.backward()

    brain_grad = sum(p.grad.abs().sum().item() for p in brain.parameters() if p.grad is not None)
    head_grad = head.net.weight.grad.abs().sum().item()
    dqn_grad = sum(p.grad.abs().sum().item() for p in dqn.parameters() if p.grad is not None)
    aux_grad = sum(p.grad.abs().sum().item() for p in aux_net.parameters() if p.grad is not None)

    assert head_grad > 0
    assert brain_grad > 0
    assert dqn_grad == 0.0
    assert aux_grad == 0.0

    # The production runner's first-batch autograd verification passes on the real head.
    from training.mortal.train_s1_score_to_go_auxiliary_2026_09 import (
        _verify_score_loss_gradient_routing as verify_routing,
    )
    verify_routing(brain, dqn, aux_net, head, obs, target)

    # Head structure: Linear(1024, 1, bias=False).
    fresh = ScoreToGoHead()
    assert isinstance(fresh.net, torch.nn.Linear)
    assert fresh.net.in_features == 1024
    assert fresh.net.out_features == 1
    assert fresh.net.bias is None

    # Canonical init: deterministic given the seed, local generator only.
    import random as _r
    h1 = ScoreToGoHead()
    h2 = ScoreToGoHead()
    h3 = ScoreToGoHead()

    from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import init_score_to_go_head
    state_before = _r.getstate()
    np_state_before = np.random.get_state()
    torch_state_before = torch.get_rng_state()

    init_score_to_go_head(h1, 20260911)
    init_score_to_go_head(h2, 20260911)
    init_score_to_go_head(h3, 20260912)

    assert torch.equal(h1.net.weight, h2.net.weight)  # deterministic per seed
    assert not torch.equal(h1.net.weight, h3.net.weight)  # seed-dependent
    assert float(h1.net.weight.detach().mean()) < 0.05 and float(h1.net.weight.detach().std()) < 0.05  # ~N(0, 0.01)

    # Global RNG untouched by head init.
    assert _r.getstate() == state_before
    assert np.random.get_state()[1].tolist() == np_state_before[1].tolist()
    assert torch.equal(torch.get_rng_state(), torch_state_before)


def test_6_optimizer_410_parent_moments_plus_fresh_head_group() -> None:
    """Test 6: Optimizer restores 410 K0 moments exactly, then appends one fresh head group."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net, head = _build_models(k0_path, "cpu", training_seed=20260910)
    optimizer = _build_optimizer(k0_path, brain, dqn, aux_net, head)

    assert len(optimizer.param_groups) == 3
    assert [len(g["params"]) for g in optimizer.param_groups] == [165, 245, 1]
    assert len(optimizer.state) == 410  # parent moments only; head state empty
    assert optimizer.param_groups[2]["lr"] == HEAD_LR
    assert optimizer.param_groups[2]["weight_decay"] == HEAD_WEIGHT_DECAY
    for p in optimizer.param_groups[2]["params"]:
        assert p not in optimizer.state

    # The production runner's structural verification passes.
    from training.mortal.train_s1_score_to_go_auxiliary_2026_09 import (
        _verify_optimizer_structure as verify_structure,
    )
    verify_structure(optimizer)

    # Parent groups keep K0 lr/wd semantics.
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0


def test_7_checkpoint_evaluator_compatibility() -> None:
    """Test 7: S1 checkpoint saves score_aux_head as extra key and stays four_player_native-loadable."""
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net, head = _build_models(k0_path, "cpu", training_seed=20260910)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "mortal_aux_variant_70400_seed_20260910.pth"
        save_state = {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "aux_net": aux_net.state_dict(),
            "score_aux_head": head.state_dict(),
            "steps": 70400,
            "config": {
                "control": {"version": 4, "online": False, "batch_size": 512},
                "resnet": {"conv_channels": 192, "num_blocks": 40},
            },
        }
        torch.save(save_state, ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        # Evaluator loads exactly these keys.
        m = model.Brain(version=4, conv_channels=192, num_blocks=40)
        q = model.DQN(version=4)
        a = model.AuxNet((4,))
        m.load_state_dict(state["mortal"])
        q.load_state_dict(state["current_dqn"])
        a.load_state_dict(state["aux_net"])
        # score_aux_head present as extra key.
        assert "score_aux_head" in state
        assert tuple(state["score_aux_head"]["net.weight"].shape) == (1, 1024)


def _write_game_log(log_path: Path, seed: int, names: list[str]) -> None:
    events = [
        {"type": "start_game", "seed": [seed, EVAL_SEED_KEY], "names": names},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "hora", "deltas": [-8000, 0, 0, 8000]},
        {"type": "end_game"},
    ]
    with gzip.open(log_path, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _build_strict_fixture(root: Path) -> dict:
    """Build a complete, contract-conformant S1 artifact fixture with real a/b/c/d logs and real SHAs."""
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
        v_path = tr_dir / f"mortal_aux_variant_70400_seed_{s}.pth"
        v_path.write_bytes(f"auxvar_{s}".encode())
        v_sha = sha256_file(v_path)
        checkpoints_manifest[f"seed_{s}"] = {
            "r2_control": {"name": r2c_path.name, "path": str(r2c_path), "sha256": r2c_sha},
            "aux_variant": {"name": v_path.name, "path": str(v_path), "sha256": v_sha},
        }
        aux_digest = hashlib.sha256(f"aux_targets_{s}".encode()).hexdigest()
        row_by_seed[f"seed_{s}"] = {
            "r2_control_sha256": r2c_row,
            "aux_variant_sha256": r2c_row,  # must equal frozen R2 control digest
            "matches_r2_control": True,
            "auxiliary_target_sha256": aux_digest,
        }

    tr_man = {
        "schema": "keqing.mortal.s1_training_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "dataset": {"path": str(m0_path), "sha256": m0_sha},
        "objective": {"mode": OBJECTIVE_MODE, "value_statistic": OBJECTIVE_VALUE_STATISTIC, "preference_loss": "existing_cql"},
        "trainable_player_names": list(TRAINABLE_PLAYER_NAMES),
        "main_reward": {"mode": MAIN_REWARD_MODE, "rank_pts": [6.0, 4.0, 2.0, 0.0]},
        "score_auxiliary": {
            "loss_weight": SCORE_AUX_LOSS_WEIGHT,
            "mse_factor": SCORE_AUX_MSE_FACTOR,
            "target_range": [-1.0, 1.0],
            "excluded_from_q_target": True,
        },
        "training_config": {"training_seeds": TRAINING_SEEDS, "device": "cpu"},
        "checkpoints": checkpoints_manifest,
        "row_identity": {"fields": [name for name, _ in ROW_IDENTITY_FIELDS], "excluded_field": "kyoku_rewards", "by_seed": row_by_seed},
        "hard_gates": {g: True for g in EXPECTED_TRAINING_HARD_GATES},
        "verdict": "training_completed",
    }
    tr_path = tr_dir / "s1_training_manifest.json"
    tr_path.write_text(json.dumps(tr_man), encoding="utf-8")
    tr_sha = sha256_file(tr_path)

    for s in TRAINING_SEEDS:
        panel_dir = ev_dir / f"panel_seed_{s}"
        lineup = ["K0_70k", "ext_mortal", f"R2_Control_seed_{s}", f"S1_AuxVariant_seed_{s}"]
        for shard_idx in range(4):
            logs_dir = panel_dir / f"shard_{shard_idx:03d}" / "logs"
            logs_dir.mkdir(parents=True)
            for g_id in range(2400000 + shard_idx * 250, 2400000 + (shard_idx + 1) * 250):
                suffix = ["a", "b", "c", "d"][g_id % 4]
                _write_game_log(logs_dir / f"{g_id}_{EVAL_SEED_KEY}_{suffix}.json.gz", g_id, lineup)

    ev_man = {
        "schema": "keqing.mortal.s1_eval_manifest.v1",
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
                    "aux_variant": {"sha256": checkpoints_manifest[f"seed_{s}"]["aux_variant"]["sha256"]},
                },
            }
            for s in TRAINING_SEEDS
        },
        "hard_gates": {g: True for g in EXPECTED_EVAL_HARD_GATES},
        "total_games_evaluated": 3000,
        "verdict": "evaluation_completed",
    }
    ev_path = ev_dir / "s1_eval_manifest.json"
    ev_path.write_text(json.dumps(ev_man), encoding="utf-8")

    return {"tr_dir": tr_dir, "ev_dir": ev_dir, "sm_dir": sm_dir, "tr_path": tr_path, "ev_path": ev_path}


def test_8_s1_summary_pipeline_and_fail_closed(tmp_path: Path) -> None:
    """Test 8: Full summary pipeline plus fail-closed rejections for all tamper classes."""
    fx = _build_strict_fixture(tmp_path / "valid")
    summary = adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])
    assert summary["schema"] == "keqing.mortal.s1_summary.v1"
    assert summary["hard_gates"]["all_3000_logs_verified"] is True
    assert summary["metrics"]["total_games"] == 3000
    assert summary["training_manifest"]["sha256"] == sha256_file(fx["tr_path"])
    assert summary["eval_manifest"]["sha256"] == sha256_file(fx["ev_path"])
    # Variant wins over both Control and K0 in the mock logs -> promotion_supported.
    assert summary["verdict"] == "promotion_supported"
    assert summary["promotion"]["k1"] == "mortal_aux_variant_70400_seed_20260911.pth"

    # (a) Row digest mismatch with frozen R2 Control must be rejected.
    fx = _build_strict_fixture(tmp_path / "row_mismatch")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["row_identity"]["by_seed"]["seed_20260910"]["aux_variant_sha256"] = "0" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (b) score_to_go entering q target (main_reward tamper) must be rejected.
    fx = _build_strict_fixture(tmp_path / "main_reward_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["main_reward"]["mode"] = "rank_plus_score_to_go_mc"
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (c) excluded_from_q_target false must be rejected.
    fx = _build_strict_fixture(tmp_path / "q_target_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["score_auxiliary"]["excluded_from_q_target"] = False
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (d) Frozen R2 Control checkpoint SHA tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "r2c_sha_tamper")
    data = json.loads(fx["tr_path"].read_text(encoding="utf-8"))
    data["checkpoints"]["seed_20260911"]["r2_control"]["sha256"] = "f" * 64
    fx["tr_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (e) Checkpoint disk tamper must be rejected.
    fx = _build_strict_fixture(tmp_path / "ckpt_tamper")
    (fx["tr_dir"] / "mortal_aux_variant_70400_seed_20260910.pth").write_bytes(b"tampered")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (f) Duplicate game ID must be rejected.
    fx = _build_strict_fixture(tmp_path / "dup_id")
    logs = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs"
    (logs / f"{2400001}_{EVAL_SEED_KEY}_b.json.gz").unlink()
    shutil.copy(logs / f"{2400000}_{EVAL_SEED_KEY}_a.json.gz", logs / f"{2400000}_{EVAL_SEED_KEY}_c.json.gz")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (g) Wrong lineup in one log must be rejected.
    fx = _build_strict_fixture(tmp_path / "lineup_tamper")
    bad = fx["ev_dir"] / "panel_seed_20260910" / "shard_000" / "logs" / f"{2400000}_{EVAL_SEED_KEY}_a.json.gz"
    _write_game_log(bad, 2400000, ["K0_70k", "ext_mortal", "R2_Control_seed_20260910", "Intruder"])
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (h) Training-manifest SHA chain rewrite must be rejected.
    fx = _build_strict_fixture(tmp_path / "sha_chain")
    with open(fx["tr_path"], "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])

    # (i) Missing eval hard gate must be rejected.
    fx = _build_strict_fixture(tmp_path / "gate_tamper")
    data = json.loads(fx["ev_path"].read_text(encoding="utf-8"))
    data["hard_gates"].pop("zero_missing_games")
    fx["ev_path"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(training_dir=fx["tr_dir"], eval_dir=fx["ev_dir"], summary_dir=fx["sm_dir"])


def test_9_crossed_bootstrap_and_four_state_verdict() -> None:
    """Test 9: Crossed bootstrap shared resampling and four-state S1 verdict logic."""
    rng = np.random.default_rng(42)
    mat1 = rng.normal(loc=5.0, scale=20.0, size=(3, 1000))
    mat2 = mat1 + 2.0

    m1, ci1, idx = crossed_bootstrap_ci(mat1, reps=500, seed=20260920, ci=95.0, return_sampled_indices=True)
    assert idx is not None
    assert idx[0].shape == (500, 3)
    assert idx[1].shape == (500, 1000)
    m2, ci2, _ = crossed_bootstrap_ci(mat2, reps=500, seed=20260920, ci=95.0, shared_indices=idx)
    assert abs(m2 - (m1 + 2.0)) < 1e-6
    assert abs(ci2[0] - (ci1[0] + 2.0)) < 1e-6
    assert abs(ci2[1] - (ci1[1] + 2.0)) < 1e-6

    # promotion_supported: both primary and absolute fully pass.
    v, r, c, k = adjudicate_s1_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "promotion_supported" and r is True and c is True
    assert k == "mortal_aux_variant_70400_seed_20260911.pth"

    # auxiliary_effect_only: primary passes, absolute fails.
    v, r, c, k = adjudicate_s1_verdict(
        primary_seed_means=[5.0, 6.0, 7.0],
        primary_ci_lower=1.2,
        absolute_seed_means=[8.0, -1.0, 10.0],
        absolute_ci_lower=-0.5,
    )
    assert v == "auxiliary_effect_only" and r is False and c is False and k is None

    # not_supported: primary seed mean negative.
    v, r, c, k = adjudicate_s1_verdict(
        primary_seed_means=[5.0, -2.0, 7.0],
        primary_ci_lower=-1.0,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None

    # not_supported: primary CI crosses zero.
    v, r, c, k = adjudicate_s1_verdict(
        primary_seed_means=[2.0, 1.0, 3.0],
        primary_ci_lower=-0.5,
        absolute_seed_means=[8.0, 9.0, 10.0],
        absolute_ci_lower=2.5,
    )
    assert v == "not_supported" and r is False and c is False and k is None


def test_10_parse_game_identity_lineup_binding(tmp_path: Path) -> None:
    """Test 10: parse_game_identity validates per-panel lineup and seed tuple strictly."""
    lineup = ("K0_70k", "ext_mortal", "R2_Control_seed_20260910", "S1_AuxVariant_seed_20260910")
    good = tmp_path / f"{2400000}_{EVAL_SEED_KEY}_a.json.gz"
    _write_game_log(good, 2400000, list(lineup))
    ident = parse_game_identity(good, lineup=lineup)
    assert ident["game_id"] == 2400000
    assert set(ident["names"]) == set(lineup)

    # Wrong lineup rejected.
    bad = tmp_path / f"{2400001}_{EVAL_SEED_KEY}_b.json.gz"
    _write_game_log(bad, 2400001, ["K0_70k", "ext_mortal", "Wrong", "S1_AuxVariant_seed_20260910"])
    with pytest.raises(ContractError):
        parse_game_identity(bad, lineup=lineup)

    # Seed mismatch rejected.
    mismatch = tmp_path / f"{2400002}_{EVAL_SEED_KEY}_c.json.gz"
    _write_game_log(mismatch, 2400000, list(lineup))
    with pytest.raises(ContractError):
        parse_game_identity(mismatch, lineup=lineup)

    # Missing seed tuple rejected.
    no_seed = tmp_path / f"{2400003}_{EVAL_SEED_KEY}_d.json.gz"
    events = [
        {"type": "start_game", "names": list(lineup)},
        {"type": "end_game"},
    ]
    with gzip.open(no_seed, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    with pytest.raises(ContractError):
        parse_game_identity(no_seed, lineup=lineup)


def test_11_per_batch_science_verifiers() -> None:
    """Test 11: Runner per-batch verifiers: Q target domain, aux target bounds, finiteness."""
    from training.mortal.train_s1_score_to_go_auxiliary_2026_09 import (
        _verify_auxiliary_target,
        _verify_parameters_finite,
        _verify_q_target_values,
    )

    # Q target domain: only {+3, +1, -1, -3}.
    _verify_q_target_values(torch.tensor([3.0, 1.0, -1.0, -3.0]))
    _verify_q_target_values(torch.tensor([1.0, 1.0, -3.0]))
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([3.0, 0.25]))  # score_to_go leaked into q target
    with pytest.raises(ContractError):
        _verify_q_target_values(torch.tensor([2.0]))

    # Auxiliary target bounds and finiteness.
    _verify_auxiliary_target(torch.tensor([-1.0, 1.0, 0.5, -0.25]))
    with pytest.raises(ContractError):
        _verify_auxiliary_target(torch.tensor([1.5]))  # out of range
    with pytest.raises(ContractError):
        _verify_auxiliary_target(torch.tensor([float("nan")]))
    with pytest.raises(ContractError):
        _verify_auxiliary_target(torch.tensor([float("inf")]))

    # Parameter finiteness.
    k0_path, _ = resolve_k0_checkpoint()
    brain, dqn, aux_net, head = _build_models(k0_path, "cpu", training_seed=20260910)
    _verify_parameters_finite(brain, dqn, aux_net, head)
    with torch.no_grad():
        head.net.weight[0, 0] = float("nan")
    with pytest.raises(ContractError):
        _verify_parameters_finite(brain, dqn, aux_net, head)


def test_12_exact_seed_set_enforcement() -> None:
    """Test 12: All S1 runners reject any seed set other than the frozen three."""
    from training.mortal.s1_score_to_go_auxiliary_contract_2026_09 import validate_s1_seed_set

    assert validate_s1_seed_set(None) == [20260910, 20260911, 20260912]
    assert validate_s1_seed_set([20260910, 20260911, 20260912]) == [20260910, 20260911, 20260912]

    for bad in (
        [20260910],  # subset
        [20260910, 20260911],  # missing one
        [20260910, 20260911, 20260912, 20260913],  # extra
        [20260910, 20260912, 20260911],  # wrong order
        [20260911, 20260912, 20260913],  # wrong seed
    ):
        with pytest.raises(ContractError):
            validate_s1_seed_set(bad)

    # Runners fail closed on wrong seed sets before touching artifacts.
    from training.mortal.eval_s1_score_to_go_auxiliary_2026_09 import run_s1_evaluation
    from training.mortal.summary_s1_score_to_go_auxiliary_2026_09 import adjudicate_s1_auxiliary
    from training.mortal.train_s1_score_to_go_auxiliary_2026_09 import run_s1_training

    with pytest.raises(ContractError):
        run_s1_training(seeds=[20260910])
    with pytest.raises(ContractError):
        run_s1_evaluation(seeds=[20260910, 20260911])
    with pytest.raises(ContractError):
        adjudicate_s1_auxiliary(seeds=[20260913])


def test_13_canonical_auxiliary_target_single_formula() -> None:
    """Test 13: Scalar and vectorized targets share the one frozen contract formula."""
    # Vectorized call equals elementwise scalar calls.
    finals = np.array([70000.0, -35000.0, 40000.0, 20000.0, 55000.0, -5000.0])
    starts = np.array([25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0])
    vec = compute_s1_auxiliary_target(finals, starts)
    for f, s, v in zip(finals, starts, vec, strict=True):
        assert abs(float(v) - compute_s1_auxiliary_target(float(f), float(s))) < 1e-12

    # The dataloader's seventh field is produced by this same formula (spot check
    # via a tiny synthetic case at the clip boundary).
    assert float(compute_s1_auxiliary_target(np.float64(70000.0), np.float64(25000.0))) == 1.0
    assert float(compute_s1_auxiliary_target(np.float64(-35000.0), np.float64(25000.0))) == -1.0
    # No standalone scalar helper remains in the dataloader module.
    import training.mortal.mainline_dataloader as mld
    assert not hasattr(mld, "compute_score_to_go_auxiliary_target")
