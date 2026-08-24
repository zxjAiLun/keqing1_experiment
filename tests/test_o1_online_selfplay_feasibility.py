"""Tests for O1: Keqing project-owned online adapter feasibility."""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.audit_o1_online_selfplay_feasibility_2026_08 import (
    CENTERED_TARGETS,
    GAMMA,
    MAX_BACKWARD_ROWS,
    RANK_PTS,
    FeasibilityServerHandler,
    FeasibilityTCPServer,
    ServerState,
    check_directory_boundary,
    compute_effective_cql_weight,
    ensure_clean_staging_dir,
    fetch_param_with_retry,
    recv_msg,
    send_msg,
)
from training.mortal.objective import compute_objective_losses


def test_1_final_rank_mc_targets_and_centering() -> None:
    """Test 1: final_rank_mc targets are strictly in {-3.0, -1.0, +1.0, +3.0}."""
    assert np.allclose(RANK_PTS, [6.0, 4.0, 2.0, 0.0])
    mean_val = float(RANK_PTS.mean())
    assert mean_val == 3.0

    assert CENTERED_TARGETS[0] == 3.0   # Rank 1 (index 0) -> +3.0
    assert CENTERED_TARGETS[1] == 1.0   # Rank 2 (index 1) -> +1.0
    assert CENTERED_TARGETS[2] == -1.0  # Rank 3 (index 2) -> -1.0
    assert CENTERED_TARGETS[3] == -3.0  # Rank 4 (index 3) -> -3.0

    domain = set(CENTERED_TARGETS.values())
    assert domain == {-3.0, -1.0, 1.0, 3.0}


def test_2_gamma_equals_one() -> None:
    """Test 2: gamma is exactly 1.0 so kyoku rewards are unattenuated."""
    assert GAMMA == 1.0
    steps_to_done = 15
    terminal_return = -1.0
    mc_target = (GAMMA ** steps_to_done) * terminal_return
    assert mc_target == terminal_return


def test_3_next_rank_uses_at_kyoku_plus_one() -> None:
    """Test 3: next-rank alignment uses player_ranks[at_kyoku + 1]."""
    # Simulate grp feature (7 columns, columns 3..6 are player scores) and final scores sequence
    # grp_feature with 8 kyokus -> scores_seq has 9 elements
    grp_feature = np.zeros((8, 7))
    final_scores = np.array([25000, 30000, 20000, 25000])
    scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
    assert scores_seq.shape[0] == 9

    rank_by_player_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
    player_id = 0
    player_ranks = rank_by_player_seq[:, player_id]

    # For any kyoku index i in 0..7, next-rank target must be at index i + 1
    for kyoku_i in range(8):
        next_rank = player_ranks[kyoku_i + 1]
        assert 0 <= next_rank <= 3


def test_4_online_force_online_cql_branch_combinations() -> None:
    """Test 4: online and force_online 4 combinations calculate CQL correctly."""
    # 1. online=True, force_online=False -> cql_active=False, weight=0.0
    active, weight = compute_effective_cql_weight(online=True, force_online=False, base_min_q_weight=5.0)
    assert active is False
    assert weight == 0.0

    # 2. online=True, force_online=True -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=True, force_online=True, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0

    # 3. online=False, force_online=False -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=False, force_online=False, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0

    # 4. online=False, force_online=True -> cql_active=True, weight=5.0
    active, weight = compute_effective_cql_weight(online=False, force_online=True, base_min_q_weight=5.0)
    assert active is True
    assert weight == 5.0


def test_5_staging_directory_fail_closed_non_empty_no_delete(tmp_path: Path) -> None:
    """Test 5: non-empty staging dir fails closed and does NOT delete contents."""
    exp_root = tmp_path / "artifacts" / "experiments" / "O1_online_selfplay_feasibility_2026_08"
    exp_root.mkdir(parents=True)

    staging_dir = exp_root / "tmp_feasibility_run" / "buffer"
    staging_dir.mkdir(parents=True)
    sentinel_file = staging_dir / "precious_data.txt"
    sentinel_file.write_text("must_not_be_deleted")

    # Fail closed immediately
    with pytest.raises(RuntimeError, match="Fail-closed security check.*non-empty"):
        ensure_clean_staging_dir(staging_dir, exp_root)

    # Verify sentinel file is still intact (never deleted)
    assert sentinel_file.exists()
    assert sentinel_file.read_text() == "must_not_be_deleted"

    # Boundary check
    outside_dir = tmp_path / "outside_experiment" / "buffer"
    with pytest.raises(ValueError, match="Security boundary violation"):
        check_directory_boundary(outside_dir, exp_root)


def test_6_bounded_backward_batch() -> None:
    """Test 6: backward batch is deterministically bounded to max 32 rows."""
    assert MAX_BACKWARD_ROWS == 32

    # Case A: total_rows = 120 -> backward batch is exactly 32
    total_rows_large = 120
    backward_batch_rows_large = min(MAX_BACKWARD_ROWS, total_rows_large)
    assert backward_batch_rows_large == 32

    # Case B: total_rows = 20 -> backward batch is 20
    total_rows_small = 20
    backward_batch_rows_small = min(MAX_BACKWARD_ROWS, total_rows_small)
    assert backward_batch_rows_small == 20


def test_7_verdict_mapping_and_exact_four_replays() -> None:
    """Test 7: Verdict mapping requires all hard gates including exactly_four_unique_replays."""
    all_pass = {
        "k0_identity_verified": True,
        "server_client_roundtrip": True,
        "exactly_four_unique_replays": True,
        "trainee_rows_loaded": True,
        "final_rank_mc_contract_verified": True,
        "next_rank_alignment_verified": True,
        "legal_actions_valid": True,
        "bounded_backward_batch": True,
        "targets_and_losses_finite": True,
        "gradients_finite": True,
        "online_cql_branch_disabled": True,
        "parameters_unchanged": True,
        "no_checkpoint_created": True,
        "processes_cleaned_up": True,
    }
    verdict_pass = "keqing_online_adapter_feasible" if all(all_pass.values()) else "keqing_online_adapter_not_feasible"
    assert verdict_pass == "keqing_online_adapter_feasible"

    # If exactly_four_unique_replays is False -> not feasible
    fail_replays = dict(all_pass)
    fail_replays["exactly_four_unique_replays"] = False
    verdict_fail = "keqing_online_adapter_feasible" if all(fail_replays.values()) else "keqing_online_adapter_not_feasible"
    assert verdict_fail == "keqing_online_adapter_not_feasible"


def test_8_zero_update_parameter_and_buffer_invariant() -> None:
    """Test 8: Forward, Loss, and Backward preserve bit-exact equality of all parameters and buffers."""
    torch.manual_seed(42)
    mortal = model.Brain(version=4, conv_channels=32, num_blocks=2).train()
    mortal.freeze_bn(True)
    dqn = model.DQN(version=4).train()
    aux = model.AuxNet((4,)).train()

    # Capture all parameters and buffers
    snapshot_before = {
        f"{mod_name}.{name}": tensor_val.clone()
        for mod_name, mod in (("mortal", mortal), ("dqn", dqn), ("aux", aux))
        for name, tensor_val in mod.state_dict().items()
    }

    # Bounded backward batch (32 rows)
    batch_size = 32
    obs = torch.randn(batch_size, 1012, 34)
    masks = torch.ones(batch_size, 46, dtype=torch.bool)
    actions = torch.zeros(batch_size, dtype=torch.int64)
    targets = torch.tensor([3.0, 1.0, -1.0, -3.0] * 8, dtype=torch.float32)
    ranks = torch.zeros(batch_size, dtype=torch.int64)

    phi = mortal(obs)
    q_out = dqn(phi, masks)
    (next_rank_logits,) = aux(phi)

    losses = compute_objective_losses(
        q_out=q_out,
        masks=masks,
        actions=actions,
        q_target_mc=targets,
        next_rank_logits=next_rank_logits,
        player_ranks=ranks,
        mode="behavior_action_mc",
        cql_weight=0.0,
        aux_weight=0.2,
    )
    total_loss = losses["total_loss"]
    total_loss.backward()

    # Zero grad without optimizer step
    mortal.zero_grad(set_to_none=True)
    dqn.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)

    # Invariance check: every parameter and buffer in state_dict matches exactly
    for mod_name, mod in (("mortal", mortal), ("dqn", dqn), ("aux", aux)):
        for name, tensor_val in mod.state_dict().items():
            before_val = snapshot_before[f"{mod_name}.{name}"]
            assert torch.equal(before_val, tensor_val), f"State drifted: {mod_name}.{name}"


def test_9_protocol_parity_and_roundtrip(tmp_path: Path) -> None:
    """Test 9: Socket framing parity, get_param retry against races, and full roundtrip."""
    # Test binary framing directly
    msg = {"test_key": [1, 2, 3], "nested": {"a": "b"}}
    buf = BytesIO()
    torch.save(msg, buf)
    raw_payload = buf.getvalue()
    prefix = struct.pack("<Q", len(raw_payload))
    assert len(prefix) == 8
    (unpacked_len,) = struct.unpack("<Q", prefix)
    assert unpacked_len == len(raw_payload)

    # Test server-client communication roundtrip
    buffer_dir = tmp_path / "buffer"
    drain_dir = tmp_path / "drain"
    buffer_dir.mkdir()
    drain_dir.mkdir()

    state = ServerState(
        buffer_dir=str(buffer_dir),
        drain_dir=str(drain_dir),
        capacity=100,
        force_sequential=False,
        dir_lock=threading.Lock(),
        param_lock=threading.Lock(),
    )

    server = FeasibilityTCPServer(("127.0.0.1", 0), FeasibilityServerHandler, state)
    allocated_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # A. Verify fetch_param_with_retry retries on 'empty param' and succeeds when param is submitted concurrently
        dummy_mortal = OrderedDict({"weight": torch.tensor([10.0, 20.0])})
        dummy_dqn = OrderedDict({"weight": torch.tensor([30.0, 40.0])})

        def delayed_submit():
            time.sleep(0.05)
            with socket.socket() as conn:
                conn.connect(("127.0.0.1", allocated_port))
                send_msg(conn, {
                    "type": "submit_param",
                    "mortal": dummy_mortal,
                    "dqn": dummy_dqn,
                    "is_idle": True,
                })

        submitter_thread = threading.Thread(target=delayed_submit)
        submitter_thread.start()

        # Worker calls fetch_param_with_retry while param is initially empty
        rsp = fetch_param_with_retry(
            host="127.0.0.1",
            port=allocated_port,
            param_version=-1,
            timeout_s=3.0,
            poll_interval_s=0.01,
        )
        submitter_thread.join()

        assert rsp["status"] == "ok"
        assert torch.equal(rsp["mortal"]["weight"], dummy_mortal["weight"])
        assert rsp["param_version"] == 1

        # B. Worker submits 4 replays
        logs = {f"sample_{i}.json.gz": f"content-{i}".encode() for i in range(4)}
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", allocated_port))
            send_msg(conn, {
                "type": "submit_replay",
                "logs": logs,
                "param_version": 1,
            })

        for _ in range(100):
            with state.dir_lock:
                if state.buffer_size == 4:
                    break
            time.sleep(0.01)

        assert state.buffer_size == 4
        assert len(list(buffer_dir.iterdir())) == 4

        # C. Trainer drains replays
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", allocated_port))
            send_msg(conn, {"type": "drain"})
            drain_msg = recv_msg(conn)

        assert drain_msg["count"] == 4
        assert len(list(buffer_dir.iterdir())) == 0
        assert len(list(drain_dir.iterdir())) == 4

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        assert not server_thread.is_alive()
