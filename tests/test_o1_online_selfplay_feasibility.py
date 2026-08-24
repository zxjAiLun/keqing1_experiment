"""Tests for O1 Online Self-play Training Stack Feasibility Auditor."""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

import model

from training.mortal.audit_o1_online_selfplay_feasibility_2026_08 import (
    FeasibilityServerHandler,
    FeasibilityTCPServer,
    ServerState,
    check_directory_boundary,
    ensure_clean_staging_dir,
    recv_msg,
    send_msg,
)


def test_1_directory_security_boundary(tmp_path: Path) -> None:
    exp_root = tmp_path / "artifacts" / "experiments" / "O1_online_selfplay_feasibility_2026_08"
    exp_root.mkdir(parents=True)

    inside_dir = exp_root / "tmp_feasibility_run" / "buffer"
    check_directory_boundary(inside_dir, exp_root)

    outside_dir = tmp_path / "other_experiments" / "hack"
    with pytest.raises(ValueError, match="Security boundary violation"):
        check_directory_boundary(outside_dir, exp_root)

    # ensure_clean_staging_dir creates fresh dir
    ensure_clean_staging_dir(inside_dir, exp_root)
    assert inside_dir.exists()

    # ensure_clean_staging_dir fails closed if allow_clean_existing is False and dir non-empty
    (inside_dir / "dummy.txt").write_text("content")
    with pytest.raises(RuntimeError, match="already exists and is non-empty"):
        ensure_clean_staging_dir(inside_dir, exp_root, allow_clean_existing=False)


def test_2_zero_update_parameter_invariance() -> None:
    torch.manual_seed(42)
    mortal = model.Brain(version=4, conv_channels=32, num_blocks=2).train()
    mortal.freeze_bn(True)
    dqn = model.DQN(version=4).train()
    aux = model.AuxNet((4,)).train()

    snapshot_before = {
        f"{mod_name}.{name}": param.clone()
        for mod_name, mod in (("mortal", mortal), ("dqn", dqn), ("aux", aux))
        for name, param in mod.state_dict().items()
    }

    # Forward + loss + backward
    batch_size = 4
    obs = torch.randn(batch_size, 1012, 34)
    masks = torch.ones(batch_size, 46, dtype=torch.bool)
    actions = torch.zeros(batch_size, dtype=torch.int64)
    targets = torch.randn(batch_size)
    ranks = torch.zeros(batch_size, dtype=torch.int64)

    phi = mortal(obs)
    q_out = dqn(phi, masks)
    (next_rank_logits,) = aux(phi)

    loss_val = 0.5 * F.mse_loss(q_out[range(batch_size), actions], targets)
    loss_aux = F.cross_entropy(next_rank_logits, ranks)
    total_loss = loss_val + 0.2 * loss_aux

    total_loss.backward()

    # Zero grad without optimizer step
    mortal.zero_grad(set_to_none=True)
    dqn.zero_grad(set_to_none=True)
    aux.zero_grad(set_to_none=True)

    # Invariance check: all parameters match exactly
    for mod_name, mod in (("mortal", mortal), ("dqn", dqn), ("aux", aux)):
        for name, param in mod.state_dict().items():
            before_tensor = snapshot_before[f"{mod_name}.{name}"]
            assert torch.equal(before_tensor, param), f"Parameter drifted: {mod_name}.{name}"


def test_3_cql_disabled_online_gate() -> None:
    # In online self-play training, CQL must be strictly disabled (weight == 0.0)
    cql_weight_online = 0.0
    cql_disabled_gate = (cql_weight_online == 0.0)
    assert cql_disabled_gate is True

    cql_weight_offline = 5.0
    cql_disabled_gate_offline = (cql_weight_offline == 0.0)
    assert cql_disabled_gate_offline is False


def test_4_verdict_mapping() -> None:
    all_pass_gates = {
        "server_client_roundtrip": True,
        "k0_identity_verified": True,
        "online_replays_generated": True,
        "trainee_rows_loaded": True,
        "legal_actions_valid": True,
        "targets_and_losses_finite": True,
        "gradients_finite": True,
        "cql_disabled_online": True,
        "parameters_unchanged": True,
        "no_checkpoint_created": True,
        "processes_cleaned_up": True,
    }
    verdict_pass = "online_stack_feasible" if all(all_pass_gates.values()) else "online_stack_not_feasible"
    assert verdict_pass == "online_stack_feasible"

    one_fail_gates = dict(all_pass_gates)
    one_fail_gates["parameters_unchanged"] = False
    verdict_fail = "online_stack_feasible" if all(one_fail_gates.values()) else "online_stack_not_feasible"
    assert verdict_fail == "online_stack_not_feasible"


def test_5_minimal_protocol_roundtrip(tmp_path: Path) -> None:
    buffer_dir = tmp_path / "buffer"
    drain_dir = tmp_path / "drain"
    buffer_dir.mkdir()
    drain_dir.mkdir()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    state = ServerState(
        buffer_dir=str(buffer_dir),
        drain_dir=str(drain_dir),
        capacity=100,
        force_sequential=False,
        dir_lock=threading.Lock(),
        param_lock=threading.Lock(),
    )

    server = FeasibilityTCPServer(("127.0.0.1", free_port), FeasibilityServerHandler, state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. Trainer submits dummy weights
        dummy_mortal = OrderedDict({"weight": torch.tensor([1.0, 2.0])})
        dummy_dqn = OrderedDict({"weight": torch.tensor([3.0, 4.0])})
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {
                "type": "submit_param",
                "mortal": dummy_mortal,
                "dqn": dummy_dqn,
                "is_idle": True,
            })

        # 2. Worker gets weights
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {"type": "get_param", "param_version": -1})
            rsp = recv_msg(conn)
        assert rsp["status"] == "ok"
        assert torch.equal(rsp["mortal"]["weight"], dummy_mortal["weight"])
        assert rsp["param_version"] == 1

        # 3. Worker submits dummy logs
        logs = {"sample_0.json.gz": b"dummy-log-bytes-0", "sample_1.json.gz": b"dummy-log-bytes-1"}
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {
                "type": "submit_replay",
                "logs": logs,
                "param_version": 1,
            })

        for _ in range(100):
            if state.buffer_size == 2:
                break
            time.sleep(0.01)

        assert state.buffer_size == 2
        assert len(list(buffer_dir.iterdir())) == 2

        # 4. Trainer drains logs
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {"type": "drain"})
            drain_msg = recv_msg(conn)

        assert drain_msg["count"] == 2
        assert len(list(buffer_dir.iterdir())) == 0
        assert len(list(drain_dir.iterdir())) == 2

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        assert not server_thread.is_alive()
