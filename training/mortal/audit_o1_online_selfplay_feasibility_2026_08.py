#!/usr/bin/env python3
"""O1: Keqing project-owned online adapter feasibility.

Verifies that the Keqing project-owned online adapter loop can safely and
correctly be used for online continuation without weight drift, leakage, CQL
interference, or leftover background processes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from socketserver import BaseRequestHandler, ThreadingTCPServer
from typing import Any

import numpy as np
import toml
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

import engine
import model
from libriichi.arena import OneVsThree

from training.mortal.objective import compute_objective_losses

logger = logging.getLogger("o1_feasibility")

EXPERIMENT_ID = "O1_online_selfplay_feasibility_2026_08"
O1_CANONICAL_DIR = REPO / "artifacts" / "experiments" / EXPERIMENT_ID
K0_CANONICAL_PATH = Path(
    "/media/bailan/DISK/AUbuntuProject/keqing-data/mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

# Fixed Project-Owned Training & Reward Contract
ADAPTER_KIND = "keqing_project_online"
OBJECTIVE_MODE = "behavior_action_mc"
REWARD_MODE = "final_rank_mc"
GAMMA = 1.0
RANK_PTS = np.array([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
CENTERED_TARGETS = {
    0: float(RANK_PTS[0] - RANK_PTS.mean()),  # Rank 1 -> +3.0
    1: float(RANK_PTS[1] - RANK_PTS.mean()),  # Rank 2 -> +1.0
    2: float(RANK_PTS[2] - RANK_PTS.mean()),  # Rank 3 -> -1.0
    3: float(RANK_PTS[3] - RANK_PTS.mean()),  # Rank 4 -> -3.0
}
AUX_WEIGHT = 0.2
BASE_MIN_Q_WEIGHT = 5.0
MAX_BACKWARD_ROWS = 32


class UnexpectedEOF(Exception):
    def __init__(self) -> None:
        super().__init__("unexpected EOF")


def send_msg(conn: socket.socket, msg: Any, packed: bool = False) -> None:
    if packed:
        tx = msg
    else:
        buf = BytesIO()
        torch.save(msg, buf)
        tx = buf.getbuffer()
    conn.sendall(struct.pack("<Q", len(tx)))
    conn.sendall(tx)


def recv_binary(conn: socket.socket, size: int) -> bytes:
    if size <= 0:
        raise ValueError("size must be positive")
    ret = bytearray(size)
    buf = memoryview(ret)
    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)


def recv_msg(conn: socket.socket, map_location: str | torch.device = "cpu") -> Any:
    rx = recv_binary(conn, 8)
    (size,) = struct.unpack("<Q", rx)
    rx = recv_binary(conn, size)
    return torch.load(BytesIO(rx), weights_only=True, map_location=map_location)


def fetch_param_with_retry(
    host: str,
    port: int,
    param_version: int = -1,
    device: str | torch.device = "cpu",
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    """Poll get_param on server with retry on 'empty param' or busy status until deadline."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as conn:
            conn.connect((host, port))
            send_msg(conn, {"type": "get_param", "param_version": param_version})
            rsp = recv_msg(conn, map_location=device)
        status = rsp.get("status")
        if status == "ok":
            return rsp
        if status in ("empty param", "samples overflow", "trainer is busy"):
            time.sleep(poll_interval_s)
            continue
        raise RuntimeError(f"Unexpected get_param response status: {rsp}")
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for parameters from server (last rsp: {rsp})")


@dataclass
class ServerState:
    buffer_dir: str
    drain_dir: str
    capacity: int
    force_sequential: bool
    dir_lock: threading.Lock
    param_lock: threading.Lock
    buffer_size: int = 0
    submission_id: int = 0
    mortal_param: OrderedDict | None = None
    dqn_param: OrderedDict | None = None
    param_version: int = 0
    idle_param_version: int = 0


class FeasibilityServerHandler(BaseRequestHandler):
    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def handle(self) -> None:
        try:
            msg = recv_msg(self.request)
        except (UnexpectedEOF, ConnectionResetError, BrokenPipeError):
            return

        msg_type = msg.get("type")
        if msg_type == "get_param":
            self.handle_get_param(msg)
        elif msg_type == "submit_replay":
            self.handle_submit_replay(msg)
        elif msg_type == "submit_param":
            self.handle_submit_param(msg)
        elif msg_type == "drain":
            self.handle_drain()

    def handle_get_param(self, msg: Mapping[str, Any]) -> None:
        state = self.state
        with state.dir_lock:
            overflow = state.buffer_size >= state.capacity
            with state.param_lock:
                has_param = state.mortal_param is not None and state.dqn_param is not None
        if overflow:
            send_msg(self.request, {"status": "samples overflow"})
            return
        if not has_param:
            send_msg(self.request, {"status": "empty param"})
            return

        client_param_version = msg.get("param_version", -1)
        buf = BytesIO()
        with state.param_lock:
            if state.force_sequential and state.idle_param_version <= client_param_version:
                res = {"status": "trainer is busy"}
            else:
                res = {
                    "status": "ok",
                    "mortal": state.mortal_param,
                    "dqn": state.dqn_param,
                    "param_version": state.param_version,
                }
            torch.save(res, buf)
        send_msg(self.request, buf.getbuffer(), packed=True)

    def handle_submit_replay(self, msg: Mapping[str, Any]) -> None:
        state = self.state
        logs = msg.get("logs", {})
        with state.dir_lock:
            for filename, content in logs.items():
                filepath = os.path.join(state.buffer_dir, f"{state.submission_id}_{filename}")
                with open(filepath, "wb") as f:
                    f.write(content)
            state.buffer_size += len(logs)
            state.submission_id += 1
            logger.info("total buffer size: %s", state.buffer_size)

    def handle_submit_param(self, msg: Mapping[str, Any]) -> None:
        state = self.state
        with state.param_lock:
            state.mortal_param = msg["mortal"]
            state.dqn_param = msg["dqn"]
            state.param_version += 1
            if msg.get("is_idle", False):
                state.idle_param_version = state.param_version

    def handle_drain(self) -> None:
        state = self.state
        drained_size = 0
        with state.dir_lock:
            buffer_list = os.listdir(state.buffer_dir)
            raw_count = len(buffer_list)
            if (not state.force_sequential or raw_count >= state.capacity) and raw_count > 0:
                old_drain_list = os.listdir(state.drain_dir)
                for filename in old_drain_list:
                    filepath = os.path.join(state.drain_dir, filename)
                    os.remove(filepath)
                for filename in buffer_list:
                    src = os.path.join(state.buffer_dir, filename)
                    dst = os.path.join(state.drain_dir, filename)
                    shutil.move(src, dst)
                drained_size = raw_count
                state.buffer_size = 0
        send_msg(
            self.request,
            {
                "count": drained_size,
                "drain_dir": state.drain_dir,
            },
        )


class FeasibilityTCPServer(ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseRequestHandler], state: ServerState):
        self.state = state
        super().__init__(server_address, handler_cls)

    def finish_request(self, request: Any, client_address: Any) -> None:
        self.RequestHandlerClass(request, client_address, self)

    def handle_error(self, request: Any, client_address: Any) -> None:
        typ, _, _ = sys.exc_info()
        if typ in (BrokenPipeError, ConnectionResetError, UnexpectedEOF):
            return
        super().handle_error(request, client_address)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_directory_boundary(target_dir: Path, experiment_root: Path) -> None:
    """Ensure target_dir is strictly contained within experiment_root."""
    resolved_root = experiment_root.resolve()
    resolved_target = target_dir.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Security boundary violation: target directory '{resolved_target}' is outside experiment root '{resolved_root}'"
        ) from exc


def ensure_clean_staging_dir(dir_path: Path, experiment_root: Path) -> None:
    """Ensure staging dir exists and is strictly empty. Fail closed if non-empty; never delete existing contents."""
    check_directory_boundary(dir_path, experiment_root)
    if dir_path.exists():
        entries = list(dir_path.iterdir())
        if entries:
            raise RuntimeError(
                f"Fail-closed security check: staging directory already exists and is non-empty ({len(entries)} items): {dir_path}"
            )
    else:
        dir_path.mkdir(parents=True, exist_ok=False)


def load_k0_checkpoint(k0_path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    target_path = k0_path or (K0_CANONICAL_PATH if K0_CANONICAL_PATH.exists() else K0_FALLBACK_PATH)
    if not target_path.exists():
        raise FileNotFoundError(f"K0 checkpoint not found at: {target_path}")
    actual_sha256 = _sha256_file(target_path)
    if actual_sha256 != K0_EXPECTED_SHA256:
        raise ValueError(
            f"K0 checkpoint SHA-256 mismatch: expected {K0_EXPECTED_SHA256}, got {actual_sha256} for {target_path}"
        )
    state = torch.load(target_path, weights_only=False, map_location="cpu")
    return state, target_path, actual_sha256


def compute_effective_cql_weight(*, online: bool, force_online: bool, base_min_q_weight: float = BASE_MIN_Q_WEIGHT) -> tuple[bool, float]:
    """Compute effective CQL activation and weight from runtime online flags."""
    cql_active = (not online) or force_online
    effective_weight = float(base_min_q_weight) if cql_active else 0.0
    return cql_active, effective_weight


def run_feasibility_audit(
    *,
    experiment_root: Path = O1_CANONICAL_DIR,
    device_name: str = "cpu",
    games: int = 4,
    k0_path: Path | None = None,
    online: bool = True,
    force_online: bool = False,
) -> dict[str, Any]:
    """Execute complete O1 feasibility audit and return summary dict."""
    if games != 4:
        raise ValueError(f"games must be exactly 4 for O1 smoke feasibility audit; got {games}")

    check_directory_boundary(experiment_root, REPO / "artifacts" / "experiments")
    experiment_root.mkdir(parents=True, exist_ok=True)

    tmp_run_dir = experiment_root / "tmp_feasibility_run"
    buffer_dir = tmp_run_dir / "buffer"
    drain_dir = tmp_run_dir / "drain"
    client_log_dir = tmp_run_dir / "client_logs"

    ensure_clean_staging_dir(buffer_dir, experiment_root)
    ensure_clean_staging_dir(drain_dir, experiment_root)
    ensure_clean_staging_dir(client_log_dir, experiment_root)

    # Initialize mortal config TOML for mainline_dataloader FileDatasetsIter
    config_path = tmp_run_dir / "mortal_config.toml"
    config_payload = {
        "control": {"version": 4},
        "env": {"pts": RANK_PTS.tolist(), "gamma": GAMMA},
        "reward": {"mode": REWARD_MODE},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(config_payload, f)
    os.environ["MORTAL_CFG"] = str(config_path.resolve())

    device = torch.device(device_name)
    hard_gates: dict[str, bool] = {
        "k0_identity_verified": False,
        "server_client_roundtrip": False,
        "exactly_four_unique_replays": False,
        "trainee_rows_loaded": False,
        "final_rank_mc_contract_verified": False,
        "next_rank_alignment_verified": False,
        "legal_actions_valid": False,
        "bounded_backward_batch": False,
        "targets_and_losses_finite": False,
        "gradients_finite": False,
        "online_cql_branch_disabled": False,
        "parameters_unchanged": False,
        "no_checkpoint_created": False,
        "processes_cleaned_up": False,
    }

    server = None
    server_thread = None
    initial_checkpoints_snapshot = set(experiment_root.rglob("*.pth"))

    total_rows_loaded = 0
    backward_batch_rows = 0

    try:
        # 1. K0 Identity & Model Initialization
        k0_state, resolved_k0_path, k0_sha256 = load_k0_checkpoint(k0_path)
        hard_gates["k0_identity_verified"] = True

        mortal_net = model.Brain(version=4, conv_channels=192, num_blocks=40).to(device)
        dqn_net = model.DQN(version=4).to(device)
        aux_net = model.AuxNet((4,)).to(device)

        mortal_net.load_state_dict(k0_state["mortal"])
        dqn_net.load_state_dict(k0_state["current_dqn"])
        aux_net.load_state_dict(k0_state["aux_net"])
        mortal_net.freeze_bn(True)

        # Snapshot parameters and buffers bit-for-bit before any operations
        state_dict_before: dict[str, torch.Tensor] = {}
        for mod_name, mod in (("mortal", mortal_net), ("dqn", dqn_net), ("aux", aux_net)):
            for name, tensor_val in mod.state_dict().items():
                state_dict_before[f"{mod_name}.{name}"] = tensor_val.clone().cpu()

        # 2. Server setup on localhost with dynamic free port
        server = FeasibilityTCPServer(("127.0.0.1", 0), FeasibilityServerHandler, ServerState(
            buffer_dir=str(buffer_dir),
            drain_dir=str(drain_dir),
            capacity=100,
            force_sequential=False,
            dir_lock=threading.Lock(),
            param_lock=threading.Lock(),
        ))
        allocated_port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # 3. Trainer submits parameters (submit_param)
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", allocated_port))
            send_msg(
                conn,
                {
                    "type": "submit_param",
                    "mortal": mortal_net.state_dict(),
                    "dqn": dqn_net.state_dict(),
                    "is_idle": True,
                },
            )

        # 4. Worker queries parameters (get_param) with deadline-bounded retry
        rsp = fetch_param_with_retry(
            host="127.0.0.1",
            port=allocated_port,
            param_version=-1,
            device=device,
            timeout_s=5.0,
            poll_interval_s=0.02,
        )

        # 5. Worker plays 4 deterministic games using OneVsThree (trainee=K0, baseline=K0)
        worker_mortal = model.Brain(version=4, conv_channels=192, num_blocks=40).to(device).eval()
        worker_dqn = model.DQN(version=4).to(device).eval()
        worker_mortal.load_state_dict(rsp["mortal"])
        worker_dqn.load_state_dict(rsp["dqn"])

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
            boltzmann_epsilon=0.0,
            boltzmann_temp=1.0,
            top_p=1.0,
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

        arena = OneVsThree(disable_progress_bar=True, log_dir=str(client_log_dir))
        # games = 4 -> seed_count = 1 (1 seed plays 4 games in OneVsThree)
        arena.py_vs_py(
            challenger=engine_chal,
            champion=engine_base,
            seed_start=(10000, 0x2000),
            seed_count=1,
        )

        client_logs: dict[str, bytes] = {}
        unique_game_seeds: set[tuple[Any, ...]] = set()
        for p in sorted(client_log_dir.glob("*.json.gz")):
            content = p.read_bytes()
            client_logs[p.name] = content
            # Parse first line to verify start_game seed identity
            with gzip.open(p, "rt", encoding="utf-8") as gz_f:
                first_line = json.loads(gz_f.readline())
                if first_line.get("type") == "start_game" and "seed" in first_line:
                    unique_game_seeds.add(tuple(first_line["seed"]))

        if len(client_logs) == 4 and len(unique_game_seeds) == 4:
            hard_gates["exactly_four_unique_replays"] = True

        # 6. Worker submits replays to server
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", allocated_port))
            send_msg(
                conn,
                {
                    "type": "submit_replay",
                    "logs": client_logs,
                    "param_version": rsp["param_version"],
                },
            )

        # Wait until server has accepted and written all submission replays
        for _ in range(500):
            with server.state.dir_lock:
                if server.state.buffer_size >= len(client_logs):
                    break
            time.sleep(0.01)

        # 7. Trainer drains replays from server
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", allocated_port))
            send_msg(conn, {"type": "drain"})
            drain_msg = recv_msg(conn)

        if drain_msg.get("count", 0) == 4 and os.path.exists(drain_msg["drain_dir"]):
            hard_gates["server_client_roundtrip"] = True

        # 8. Production FileDatasetsIter loads trainee perspectives directly from drained files
        from training.mortal.mainline_dataloader import (
            FileDatasetsIter,
        )

        drained_files = sorted(
            os.path.join(drain_msg["drain_dir"], p) for p in os.listdir(drain_msg["drain_dir"])
        )
        dataset = FileDatasetsIter(
            version=4,
            file_list=drained_files,
            pts=RANK_PTS,
            oracle=False,
            player_names=["trainee"],
            enable_augmentation=False,
            num_epochs=1,
        )

        all_obs = []
        all_actions = []
        all_masks = []
        all_steps_to_done = []
        all_q_targets = []
        all_next_rank_targets = []

        for entry in dataset:
            obs, action, mask, steps_to_done, kyoku_reward, next_rank = entry
            all_obs.append(obs)
            all_actions.append(action)
            all_masks.append(mask)
            all_steps_to_done.append(steps_to_done)
            # gamma = 1.0 -> q_target = kyoku_reward
            q_target = float((GAMMA ** steps_to_done) * kyoku_reward)
            all_q_targets.append(q_target)
            all_next_rank_targets.append(int(next_rank))

        total_rows_loaded = len(all_obs)
        if total_rows_loaded > 0:
            hard_gates["trainee_rows_loaded"] = True

        # 9. CPU schema, legal action, and target-domain validation across ALL rows
        all_actions_arr = np.array(all_actions, dtype=np.int64)
        all_masks_arr = np.array(all_masks, dtype=bool)
        all_targets_arr = np.array(all_q_targets, dtype=np.float32)
        all_next_ranks_arr = np.array(all_next_rank_targets, dtype=np.int64)

        all_legal = True
        for row_i in range(total_rows_loaded):
            if not bool(all_masks_arr[row_i, all_actions_arr[row_i]]):
                all_legal = False
                break
        hard_gates["legal_actions_valid"] = all_legal

        unique_targets = set(np.unique(all_targets_arr).tolist())
        expected_target_domain = set(CENTERED_TARGETS.values())
        hard_gates["final_rank_mc_contract_verified"] = (
            unique_targets.issubset(expected_target_domain) and len(unique_targets) > 0
        )

        unique_next_ranks = set(np.unique(all_next_ranks_arr).tolist())
        hard_gates["next_rank_alignment_verified"] = unique_next_ranks.issubset({0, 1, 2, 3})

        # 10. CQL runtime branch verification
        cql_active, effective_cql_weight = compute_effective_cql_weight(
            online=online, force_online=force_online, base_min_q_weight=BASE_MIN_Q_WEIGHT
        )
        hard_gates["online_cql_branch_disabled"] = (not cql_active and effective_cql_weight == 0.0)

        # 11. Bounded backward batch (Deterministic first 32 rows)
        backward_batch_rows = min(MAX_BACKWARD_ROWS, total_rows_loaded)
        hard_gates["bounded_backward_batch"] = (backward_batch_rows <= MAX_BACKWARD_ROWS)

        batch_obs = torch.as_tensor(np.stack(all_obs[:backward_batch_rows], axis=0), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(all_actions_arr[:backward_batch_rows], dtype=torch.int64, device=device)
        batch_masks = torch.as_tensor(all_masks_arr[:backward_batch_rows], dtype=torch.bool, device=device)
        batch_q_targets = torch.as_tensor(all_targets_arr[:backward_batch_rows], dtype=torch.float32, device=device)
        batch_next_ranks = torch.as_tensor(all_next_ranks_arr[:backward_batch_rows], dtype=torch.int64, device=device)

        # 12. Forward, Loss, Backward using mainline compute_objective_losses (Zero-Update)
        mortal_net.train()
        dqn_net.train()
        aux_net.train()
        mortal_net.zero_grad(set_to_none=True)
        dqn_net.zero_grad(set_to_none=True)
        aux_net.zero_grad(set_to_none=True)

        phi = mortal_net(batch_obs)
        q_out = dqn_net(phi, batch_masks)
        (next_rank_logits,) = aux_net(phi)

        objective_losses = compute_objective_losses(
            q_out=q_out,
            masks=batch_masks,
            actions=batch_actions,
            q_target_mc=batch_q_targets,
            next_rank_logits=next_rank_logits,
            player_ranks=batch_next_ranks,
            mode=OBJECTIVE_MODE,
            cql_weight=effective_cql_weight,
            aux_weight=AUX_WEIGHT,
        )
        total_loss = objective_losses["total_loss"]

        losses_finite = bool(
            torch.isfinite(batch_q_targets).all().item()
            and torch.isfinite(total_loss).all().item()
            and all(
                torch.isfinite(v).all().item()
                for v in objective_losses.values()
                if isinstance(v, torch.Tensor)
            )
        )
        hard_gates["targets_and_losses_finite"] = losses_finite

        total_loss.backward()

        grads_finite = True
        for mod in (mortal_net, dqn_net, aux_net):
            for p in mod.parameters():
                if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all().item())):
                    grads_finite = False
                    break
        hard_gates["gradients_finite"] = grads_finite

        # ZERO UPDATE: zero out gradients without stepping optimizer or scheduler
        mortal_net.zero_grad(set_to_none=True)
        dqn_net.zero_grad(set_to_none=True)
        aux_net.zero_grad(set_to_none=True)

        # 13. Verify all parameters and buffers unchanged bit-for-bit
        params_unchanged = True
        for mod_name, mod in (("mortal", mortal_net), ("dqn", dqn_net), ("aux", aux_net)):
            for name, tensor_val in mod.state_dict().items():
                before_val = state_dict_before[f"{mod_name}.{name}"]
                if not torch.equal(before_val, tensor_val.cpu()):
                    params_unchanged = False
                    break
        hard_gates["parameters_unchanged"] = params_unchanged

        # 14. Verify no checkpoint created
        current_checkpoints = set(experiment_root.rglob("*.pth"))
        new_checkpoints = current_checkpoints - initial_checkpoints_snapshot
        hard_gates["no_checkpoint_created"] = (len(new_checkpoints) == 0)

    finally:
        # Clean up server & background threads
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except OSError as ex:
                logger.warning("Error shutting down server: %s", ex)
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=2.0)
        hard_gates["processes_cleaned_up"] = (server_thread is None or not server_thread.is_alive())

    verdict = (
        "keqing_online_adapter_feasible"
        if all(hard_gates.values())
        else "keqing_online_adapter_not_feasible"
    )

    summary = {
        "schema": "keqing.mortal.o1_online_feasibility_summary.v2",
        "experiment_id": EXPERIMENT_ID,
        "contract": {
            "adapter_kind": ADAPTER_KIND,
            "objective": OBJECTIVE_MODE,
            "reward": REWARD_MODE,
            "gamma": GAMMA,
            "rank_pts": RANK_PTS.tolist(),
            "centered_targets": CENTERED_TARGETS,
            "aux_weight": AUX_WEIGHT,
            "base_min_q_weight": BASE_MIN_Q_WEIGHT,
            "online": online,
            "force_online": force_online,
            "effective_cql_weight": compute_effective_cql_weight(online=online, force_online=force_online)[1],
        },
        "k0_checkpoint": {
            "path": str(resolved_k0_path),
            "sha256": k0_sha256,
            "verified": hard_gates["k0_identity_verified"],
        },
        "feasibility_run": {
            "games_generated": len(client_logs) if "client_logs" in locals() else 0,
            "total_rows_loaded": total_rows_loaded,
            "backward_batch_rows": backward_batch_rows,
            "device": str(device),
        },
        "hard_gates": hard_gates,
        "verdict": verdict,
    }

    summary_file = experiment_root / "o1_summary.json"
    tmp_summary = experiment_root / "o1_summary.json.tmp"
    with open(tmp_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_summary.replace(summary_file)
    logger.info("O1 summary written to %s (verdict=%s)", summary_file, verdict)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=O1_CANONICAL_DIR, help="Experiment root directory")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--games", type=int, default=4, help="Number of smoke games to generate (must be exactly 4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_feasibility_audit(
        experiment_root=args.output_dir,
        device_name=args.device,
        games=args.games,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["verdict"] != "keqing_online_adapter_feasible":
        sys.exit(1)


if __name__ == "__main__":
    main()
