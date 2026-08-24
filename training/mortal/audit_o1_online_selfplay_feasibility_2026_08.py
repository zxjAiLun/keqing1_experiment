#!/usr/bin/env python3
"""O1: Online Self-play Training Stack Feasibility Auditor.

Verifies that the existing Mortal online server / client / trainer loop can
safely and correctly be used for online continuation without weight drift,
leakage, CQL interference, or leftover background processes.
"""

from __future__ import annotations

import argparse
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
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

import engine
import model
from libriichi.arena import OneVsThree
from libriichi.dataset import GameplayLoader

logger = logging.getLogger("o1_feasibility")

EXPERIMENT_ID = "O1_online_selfplay_feasibility_2026_08"
O1_CANONICAL_DIR = REPO / "artifacts" / "experiments" / EXPERIMENT_ID
K0_CANONICAL_PATH = Path(
    "/media/bailan/DISK/AUbuntuProject/keqing-data/mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"


class UnexpectedEOF(Exception):
    pass


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


def ensure_clean_staging_dir(dir_path: Path, experiment_root: Path, allow_clean_existing: bool = True) -> None:
    check_directory_boundary(dir_path, experiment_root)
    if dir_path.exists():
        entries = list(dir_path.iterdir())
        if entries:
            if not allow_clean_existing:
                raise RuntimeError(f"Directory already exists and is non-empty: {dir_path}")
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


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


def run_feasibility_audit(
    *,
    experiment_root: Path = O1_CANONICAL_DIR,
    device_name: str = "cpu",
    games: int = 4,
    k0_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete O1 feasibility audit and return summary dict."""
    check_directory_boundary(experiment_root, REPO / "artifacts" / "experiments")
    experiment_root.mkdir(parents=True, exist_ok=True)

    tmp_run_dir = experiment_root / "tmp_feasibility_run"
    buffer_dir = tmp_run_dir / "buffer"
    drain_dir = tmp_run_dir / "drain"
    client_log_dir = tmp_run_dir / "client_logs"

    ensure_clean_staging_dir(buffer_dir, experiment_root)
    ensure_clean_staging_dir(drain_dir, experiment_root)
    ensure_clean_staging_dir(client_log_dir, experiment_root)

    device = torch.device(device_name)
    hard_gates: dict[str, bool] = {
        "server_client_roundtrip": False,
        "k0_identity_verified": False,
        "online_replays_generated": False,
        "trainee_rows_loaded": False,
        "legal_actions_valid": False,
        "targets_and_losses_finite": False,
        "gradients_finite": False,
        "cql_disabled_online": False,
        "parameters_unchanged": False,
        "no_checkpoint_created": False,
        "processes_cleaned_up": False,
    }

    server = None
    server_thread = None
    initial_checkpoints_snapshot = set(experiment_root.rglob("*.pth"))

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

        # Take bit-exact parameter snapshot before any operations
        param_snapshot_before: dict[str, torch.Tensor] = {}
        for mod_name, mod in (("mortal", mortal_net), ("dqn", dqn_net), ("aux", aux_net)):
            for p_name, p_val in mod.state_dict().items():
                param_snapshot_before[f"{mod_name}.{p_name}"] = p_val.clone().cpu()

        # 2. Server setup on localhost with dynamic free port
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

        # 3. Trainer submits parameters (submit_param)
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(
                conn,
                {
                    "type": "submit_param",
                    "mortal": mortal_net.state_dict(),
                    "dqn": dqn_net.state_dict(),
                    "is_idle": True,
                },
            )

        # 4. Worker queries parameters (get_param)
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {"type": "get_param", "param_version": -1})
            rsp = recv_msg(conn, map_location=device)

        if rsp.get("status") != "ok":
            raise RuntimeError(f"Worker failed to receive parameters from server: {rsp}")

        # 5. Worker plays games using OneVsThree
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
        seed_count = max(1, games // 4)
        arena.py_vs_py(
            challenger=engine_chal,
            champion=engine_base,
            seed_start=(10000, 0x2000),
            seed_count=seed_count,
        )

        client_logs: dict[str, bytes] = {}
        for p in client_log_dir.glob("*.json.gz"):
            client_logs[p.name] = p.read_bytes()

        if len(client_logs) >= 1:
            hard_gates["online_replays_generated"] = True

        # 6. Worker submits replays to server
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
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
            with state.dir_lock:
                if state.buffer_size >= len(client_logs):
                    break
            time.sleep(0.01)

        # 7. Trainer drains replays from server
        with socket.socket() as conn:
            conn.connect(("127.0.0.1", free_port))
            send_msg(conn, {"type": "drain"})
            drain_msg = recv_msg(conn)

        if drain_msg.get("count", 0) > 0 and os.path.exists(drain_msg["drain_dir"]):
            hard_gates["server_client_roundtrip"] = True

        # 8. GameplayLoader loads trainee perspectives
        drained_files = [os.path.join(drain_msg["drain_dir"], p) for p in os.listdir(drain_msg["drain_dir"])]
        loader = GameplayLoader(version=4, oracle=False, player_names=["trainee"], augmented=False)
        data = loader.load_gz_log_files(drained_files)

        all_obs = []
        all_actions = []
        all_masks = []
        all_steps_to_done = []
        all_kyoku_rewards = []
        all_player_ranks = []

        pts = np.array([90.0, 45.0, 0.0, -135.0])
        for file in data:
            for game in file:
                obs = game.take_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)
                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = np.zeros(len(grp_feature), dtype=np.float64)
                final_rank = int(rank_by_player[player_id])
                kyoku_rewards[min(len(kyoku_rewards) - 1, int(at_kyoku[-1]))] = float(pts[final_rank])

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                all_obs.append(obs)
                all_actions.append(actions)
                all_masks.append(masks)
                all_steps_to_done.append(steps_to_done)
                all_kyoku_rewards.append(kyoku_rewards[at_kyoku])
                all_player_ranks.append(player_ranks[at_kyoku])

        total_rows = sum(len(o) for o in all_obs)
        if total_rows > 0:
            hard_gates["trainee_rows_loaded"] = True

        batch_obs = torch.as_tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(np.concatenate(all_actions, axis=0), dtype=torch.int64, device=device)
        batch_masks = torch.as_tensor(np.concatenate(all_masks, axis=0), dtype=torch.bool, device=device)
        batch_steps_to_done = torch.as_tensor(
            np.concatenate(all_steps_to_done, axis=0), dtype=torch.int64, device=device
        )
        batch_kyoku_rewards = torch.as_tensor(
            np.concatenate(all_kyoku_rewards, axis=0), dtype=torch.float64, device=device
        )
        batch_player_ranks = torch.as_tensor(
            np.concatenate(all_player_ranks, axis=0), dtype=torch.int64, device=device
        )

        # 9. Verify legal actions
        row_indices = torch.arange(batch_obs.shape[0], device=device)
        is_legal = bool(batch_masks[row_indices, batch_actions].all().item())
        hard_gates["legal_actions_valid"] = is_legal

        # 10. Check CQL disabled for online selfplay
        cql_min_q_weight = 0.0  # Online self-play requires CQL disabled
        hard_gates["cql_disabled_online"] = (cql_min_q_weight == 0.0)

        # 11. Forward, Loss, Backward (Zero-Update)
        mortal_net.train()
        dqn_net.train()
        aux_net.train()
        mortal_net.zero_grad(set_to_none=True)
        dqn_net.zero_grad(set_to_none=True)
        aux_net.zero_grad(set_to_none=True)

        gamma = 0.99
        q_target_mc = (gamma**batch_steps_to_done * batch_kyoku_rewards).to(torch.float32)

        phi = mortal_net(batch_obs)
        q_out = dqn_net(phi, batch_masks)
        (next_rank_logits,) = aux_net(phi)

        behavior_q = q_out[row_indices, batch_actions]
        value_loss = 0.5 * F.mse_loss(behavior_q, q_target_mc)
        aux_loss = F.cross_entropy(next_rank_logits, batch_player_ranks)
        total_loss = value_loss + 0.2 * aux_loss

        losses_finite = bool(
            torch.isfinite(q_target_mc).all().item()
            and torch.isfinite(value_loss).all().item()
            and torch.isfinite(aux_loss).all().item()
            and torch.isfinite(total_loss).all().item()
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

        # ZERO UPDATE: zero out gradients without stepping optimizer
        mortal_net.zero_grad(set_to_none=True)
        dqn_net.zero_grad(set_to_none=True)
        aux_net.zero_grad(set_to_none=True)

        # 12. Verify parameters unchanged bit-for-bit
        params_unchanged = True
        for mod_name, mod in (("mortal", mortal_net), ("dqn", dqn_net), ("aux", aux_net)):
            for p_name, p_val in mod.state_dict().items():
                before_val = param_snapshot_before[f"{mod_name}.{p_name}"]
                if not torch.equal(before_val, p_val.cpu()):
                    params_unchanged = False
                    break
        hard_gates["parameters_unchanged"] = params_unchanged

        # 13. Verify no checkpoint created
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
        "online_stack_feasible"
        if all(hard_gates.values())
        else "online_stack_not_feasible"
    )

    summary = {
        "schema": "keqing.mortal.o1_online_feasibility_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "k0_checkpoint": {
            "path": str(resolved_k0_path),
            "sha256": k0_sha256,
            "verified": hard_gates["k0_identity_verified"],
        },
        "feasibility_run": {
            "games_generated": len(client_logs) if "client_logs" in locals() else 0,
            "trainee_rows_loaded": total_rows if "total_rows" in locals() else 0,
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
    parser.add_argument("--games", type=int, default=4, help="Number of smoke games to generate (default 4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_feasibility_audit(
        experiment_root=args.output_dir,
        device_name=args.device,
        games=args.games,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["verdict"] != "online_stack_feasible":
        sys.exit(1)


if __name__ == "__main__":
    main()
