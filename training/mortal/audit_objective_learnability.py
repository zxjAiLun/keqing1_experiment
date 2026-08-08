#!/usr/bin/env python3
"""Audit whether final-rank MC carries learnable action-level signal.

This is an analysis-only pass. It compares the 70k parent with the matched M0
and S0 continuation checkpoints on deterministic samples from both replay
routes. It does not update parameters or create a training checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys
import tomllib
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))

from training.mortal import audit_replay_distribution as distribution


RANK_POINTS_DEFAULT = np.asarray([6.0, 4.0, 2.0, 0.0], dtype=np.float64)


@dataclass
class LearnabilityRecord:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    q_target: float
    terminal_return: float
    steps_to_done: int
    player_rank: int
    phase: str
    current_rank: int
    score_gap: float
    own_riichi: bool


class ScalarStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        self.sum_sq += float(value) * float(value)

    def to_json(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None}
        mean = self.sum / self.count
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        return {"count": self.count, "mean": mean, "std": math.sqrt(variance)}


class RegressionStats:
    def __init__(self, sample_limit: int = 50_000) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_residual = 0.0
        self.sum_residual_sq = 0.0
        self.sample_limit = sample_limit
        self.sample_x: list[float] = []
        self.sample_y: list[float] = []

    def add(self, x: float, y: float) -> None:
        self.count += 1
        self.sum_x += float(x)
        self.sum_y += float(y)
        self.sum_x2 += float(x) * float(x)
        self.sum_y2 += float(y) * float(y)
        self.sum_xy += float(x) * float(y)
        residual = float(y) - float(x)
        self.sum_residual += residual
        self.sum_residual_sq += residual * residual
        if len(self.sample_x) < self.sample_limit:
            self.sample_x.append(float(x))
            self.sample_y.append(float(y))

    def to_json(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        var_x = max(0.0, self.sum_x2 / self.count - mean_x * mean_x)
        var_y = max(0.0, self.sum_y2 / self.count - mean_y * mean_y)
        covariance = self.sum_xy / self.count - mean_x * mean_y
        pearson = covariance / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else None
        slope = covariance / var_x if var_x > 0 else None
        intercept = mean_y - slope * mean_x if slope is not None else None
        residual_mean = self.sum_residual / self.count
        residual_var = max(0.0, self.sum_residual_sq / self.count - residual_mean * residual_mean)
        pearson_r_squared = pearson * pearson if pearson is not None else None
        identity_residual_variance_ratio = residual_var / var_y if var_y > 0 else None
        identity_explained = (
            1.0 - identity_residual_variance_ratio
            if identity_residual_variance_ratio is not None
            else None
        )
        return {
            "count": self.count,
            "q_mean": mean_x,
            "q_std": math.sqrt(var_x),
            "target_mean": mean_y,
            "target_std": math.sqrt(var_y),
            "pearson_r": pearson,
            "ols_r_squared": pearson_r_squared,
            "prefix_sample_spearman": _spearman(self.sample_x, self.sample_y),
            "prefix_sample_count": len(self.sample_x),
            "linear_slope": slope,
            "linear_intercept": intercept,
            "identity_residual_target_minus_q_mean": residual_mean,
            "identity_residual_target_minus_q_std": math.sqrt(residual_var),
            "identity_residual_variance_ratio": identity_residual_variance_ratio,
            "identity_explained_variance": identity_explained,
        }


class QDecompositionStats:
    def __init__(self) -> None:
        self.count = 0
        self.greedy_changes = 0
        self.rank_agreements = 0
        self.raw_abs = ScalarStats()
        self.raw_signed = ScalarStats()
        self.centered_abs = ScalarStats()
        self.offset = ScalarStats()
        self.offset_abs = ScalarStats()
        self.behavior_delta = ScalarStats()
        self.centered_behavior_delta = ScalarStats()
        self.parent_margin = ScalarStats()
        self.candidate_margin = ScalarStats()

    def add(self, parent_q: np.ndarray, candidate_q: np.ndarray, record: LearnabilityRecord) -> None:
        valid = record.mask.astype(bool) & np.isfinite(parent_q) & np.isfinite(candidate_q)
        if int(valid.sum()) < 2 or not valid[record.action]:
            return
        parent = parent_q[valid].astype(np.float64)
        candidate = candidate_q[valid].astype(np.float64)
        parent_mean = float(parent.mean())
        candidate_mean = float(candidate.mean())
        delta = candidate - parent
        self.count += 1
        self.raw_abs.add(float(np.abs(delta).mean()))
        self.raw_signed.add(float(delta.mean()))
        self.centered_abs.add(float(np.abs((candidate - candidate_mean) - (parent - parent_mean)).mean()))
        self.offset.add(candidate_mean - parent_mean)
        self.offset_abs.add(abs(candidate_mean - parent_mean))
        parent_action = float(parent_q[record.action])
        candidate_action = float(candidate_q[record.action])
        self.behavior_delta.add(candidate_action - parent_action)
        self.centered_behavior_delta.add(
            (candidate_action - candidate_mean) - (parent_action - parent_mean)
        )
        parent_sorted = np.partition(parent, -2)[-2:]
        candidate_sorted = np.partition(candidate, -2)[-2:]
        self.parent_margin.add(float(np.max(parent_sorted) - np.min(parent_sorted)))
        self.candidate_margin.add(float(np.max(candidate_sorted) - np.min(candidate_sorted)))
        self.greedy_changes += int(int(np.argmax(parent)) != int(np.argmax(candidate)))
        self.rank_agreements += int(np.array_equal(np.argsort(parent), np.argsort(candidate)))

    def to_json(self) -> dict[str, Any]:
        return {
            "states": self.count,
            "greedy_change_rate": self.greedy_changes / self.count if self.count else None,
            "legal_action_rank_agreement_rate": self.rank_agreements / self.count if self.count else None,
            "mean_raw_q_abs_delta": self.raw_abs.to_json()["mean"],
            "mean_raw_q_signed_delta": self.raw_signed.to_json()["mean"],
            "mean_centered_advantage_abs_delta": self.centered_abs.to_json()["mean"],
            "mean_common_q_offset": self.offset.to_json()["mean"],
            "mean_abs_common_q_offset": self.offset_abs.to_json()["mean"],
            "mean_behavior_q_delta": self.behavior_delta.to_json()["mean"],
            "mean_centered_behavior_q_delta": self.centered_behavior_delta.to_json()["mean"],
            "parent_margin_mean": self.parent_margin.to_json()["mean"],
            "candidate_margin_mean": self.candidate_margin.to_json()["mean"],
        }


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denominator = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    return float(np.sum(dx * dy) / denominator) if denominator > 0 else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except (RuntimeError, pickle.UnpicklingError):
        return torch.load(path, weights_only=False, map_location="cpu")


def _load_file_list(path: Path) -> list[Path]:
    return distribution.load_file_list(path.resolve())


def _steps_to_done(dones: np.ndarray, apply_gamma: np.ndarray) -> np.ndarray:
    result = np.zeros(len(dones), dtype=np.int64)
    for index in reversed(range(len(dones))):
        if not bool(dones[index]) and index + 1 < len(dones):
            result[index] = result[index + 1] + int(apply_gamma[index])
    return result


def _bucket_names(record: LearnabilityRecord) -> list[str]:
    names = [
        "all",
        f"steps_{_steps_bucket(record.steps_to_done)}",
        f"phase_{record.phase}",
        f"current_rank_{record.current_rank}",
        f"score_gap_{distribution.score_gap_bucket(record.score_gap)}",
        f"own_riichi_{str(record.own_riichi).lower()}",
    ]
    return names


def _steps_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 4:
        return "1_4"
    if value <= 12:
        return "5_12"
    if value <= 24:
        return "13_24"
    return "25_plus"


def _reservoir_add(
    reservoir: list[LearnabilityRecord],
    record: LearnabilityRecord,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(record)
        return
    index = rng.randrange(seen + 1)
    if index < limit:
        reservoir[index] = record


def _record_game(game: Any, pts: np.ndarray, gamma: float) -> Iterable[LearnabilityRecord]:
    obs = game.take_obs()
    masks = game.take_masks()
    actions = game.take_actions()
    at_kyoku_raw = game.take_at_kyoku()
    if isinstance(at_kyoku_raw, (bytes, bytearray)):
        at_kyoku = np.frombuffer(at_kyoku_raw, dtype=np.uint8).astype(np.int64)
    else:
        at_kyoku = np.asarray(at_kyoku_raw, dtype=np.int64)
    dones = np.asarray(game.take_dones(), dtype=bool)
    apply_gamma = np.asarray(game.take_apply_gamma(), dtype=np.int64)
    grp = game.take_grp()
    features = np.asarray(grp.take_feature(), dtype=np.float64)
    player_id = int(game.take_player_id())
    final_rank = int(grp.take_rank_by_player()[player_id])
    terminal_return = float(pts[final_rank] - pts.mean())
    final_scores = grp.take_final_scores()
    scores_seq = np.concatenate((features[:, 3:] * 1e4, [final_scores]))
    rank_seq = (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
    player_ranks = rank_seq[:, player_id]
    steps = _steps_to_done(dones, apply_gamma)
    own_riichi_kyoku: int | None = None
    for index, (obs_i, mask_i, action_i) in enumerate(zip(obs, masks, actions, strict=True)):
        kyoku = int(at_kyoku[index])
        rank, score_gap = distribution.current_rank_and_gap(features, player_id, kyoku)
        action = int(action_i)
        own_riichi = own_riichi_kyoku == kyoku
        q_target = float((gamma**int(steps[index])) * terminal_return)
        yield LearnabilityRecord(
            obs=np.asarray(obs_i, dtype=np.float32).copy(),
            mask=np.asarray(mask_i, dtype=np.bool_).copy(),
            action=action,
            q_target=q_target,
            terminal_return=terminal_return,
            steps_to_done=int(steps[index]),
            player_rank=int(player_ranks[min(kyoku + 1, len(player_ranks) - 1)]),
            phase=distribution.phase_bucket(kyoku),
            current_rank=rank,
            score_gap=score_gap,
            own_riichi=own_riichi,
        )
        if action == 37:
            own_riichi_kyoku = kyoku


def collect_route(
    *,
    file_index: Path,
    player_name: str,
    version: int,
    pts: np.ndarray,
    gamma: float,
    sample_limit: int,
    seed: int,
    max_files: int,
    file_batch_size: int,
    progress_every: int,
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader

    files = _load_file_list(file_index)
    if max_files > 0:
        files = files[:max_files]
    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=[player_name],
        augmented=False,
    )
    rng = random.Random(seed)
    reservoir: list[LearnabilityRecord] = []
    target_buckets: dict[str, ScalarStats] = defaultdict(ScalarStats)
    total_decisions = 0
    perspectives = 0
    target_sum = 0.0
    target_sq_sum = 0.0
    for batch_start in range(0, len(files), file_batch_size):
        file_batch = files[batch_start : batch_start + file_batch_size]
        try:
            loaded_batch = loader.load_gz_log_files([str(path) for path in file_batch])
            if len(loaded_batch) != len(file_batch):
                raise ValueError(
                    f"loader returned {len(loaded_batch)} files for {len(file_batch)} paths"
                )
        except Exception as batch_exc:  # noqa: BLE001
            # Preserve per-file error isolation if a native batch parser rejects
            # one member. This is slower, but keeps the audit diagnosable.
            loaded_batch = []
            for path in file_batch:
                try:
                    loaded = loader.load_gz_log_files([str(path)])
                    if len(loaded) != 1:
                        raise ValueError(f"loader returned {len(loaded)} files")
                    loaded_batch.append(loaded)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"{path}: batch={batch_exc}; single={exc}") from exc

        for path, loaded in zip(file_batch, loaded_batch, strict=True):
            if len(loaded) != 1:
                raise ValueError(f"{path}: expected one {player_name} perspective, got {len(loaded)}")
            records = list(_record_game(loaded[0], pts, gamma))
            if not records:
                raise ValueError(f"{path}: no decisions for {player_name}")
            perspectives += 1
            for record in records:
                total_decisions += 1
                target_sum += record.q_target
                target_sq_sum += record.q_target * record.q_target
                for bucket in _bucket_names(record):
                    target_buckets[bucket].add(record.q_target)
                _reservoir_add(reservoir, record, total_decisions - 1, sample_limit, rng)
        file_index_position = batch_start + len(file_batch)
        if progress_every > 0 and file_index_position % progress_every == 0:
            print(
                f"[objective] {player_name}: files={file_index_position}/{len(files)} "
                f"decisions={total_decisions} sample={len(reservoir)}",
                flush=True,
            )
    target_mean = target_sum / total_decisions if total_decisions else 0.0
    target_variance = max(0.0, target_sq_sum / total_decisions - target_mean * target_mean)
    return {
        "player_name": player_name,
        "file_index": str(file_index.resolve()),
        "file_index_sha256": _sha256_file(file_index.resolve()),
        "files": len(files),
        "perspectives": perspectives,
        "decisions": total_decisions,
        "sample_decisions": len(reservoir),
        "target_mean": target_mean,
        "target_std": math.sqrt(target_variance),
        "target_buckets": {key: value.to_json() for key, value in sorted(target_buckets.items())},
        "sample": reservoir,
    }


def _build_model(state: dict[str, Any], device: torch.device):
    from model import AuxNet, Brain, DQN

    model_config = state["config"]
    version = int(model_config["control"].get("version", 4))
    brain = Brain(version=version, **model_config["resnet"]).to(device)
    dqn = DQN(version=version).to(device)
    aux = AuxNet((4,)).to(device)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    if "aux_net" in state:
        aux.load_state_dict(state["aux_net"])
    return brain, dqn, aux, version


def _model_q(
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    records: list[LearnabilityRecord],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    brain.eval()
    dqn.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            obs = torch.as_tensor(np.stack([record.obs for record in batch]), device=device)
            masks = torch.as_tensor(np.stack([record.mask for record in batch]), device=device)
            phi = brain(obs)
            rows.append(dqn(phi, masks).float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _q_calibration(
    records: list[LearnabilityRecord],
    q_values: np.ndarray,
) -> dict[str, Any]:
    buckets: dict[str, RegressionStats] = defaultdict(RegressionStats)
    agreement_count = 0
    legal_count = 0
    margin_stats = ScalarStats()
    for record, q in zip(records, q_values, strict=True):
        valid = record.mask & np.isfinite(q)
        if not valid[record.action]:
            continue
        legal = q[valid]
        action_indices = np.flatnonzero(valid)
        greedy_position = int(np.argmax(legal))
        greedy_action = int(action_indices[greedy_position])
        legal_count += 1
        agreement_count += int(greedy_action == record.action)
        if len(legal) >= 2:
            top = np.partition(legal, -2)[-2:]
            margin_stats.add(float(np.max(top) - np.min(top)))
        names = _bucket_names(record)
        names.append(f"parent_agreement_{str(greedy_action == record.action).lower()}")
        margin = float(np.max(np.partition(legal, -2)[-2:]) - np.min(np.partition(legal, -2)[-2:])) if len(legal) >= 2 else 0.0
        names.append(f"parent_margin_{_margin_bucket(margin)}")
        for name in names:
            buckets[name].add(float(q[record.action]), record.q_target)
    return {
        "sample_decisions": len(records),
        "legal_behavior_rate": legal_count / len(records) if records else 0.0,
        "agreement_rate": agreement_count / legal_count if legal_count else 0.0,
        "mean_greedy_margin": margin_stats.to_json()["mean"],
        "buckets": {key: value.to_json() for key, value in sorted(buckets.items())},
    }


def _margin_bucket(value: float) -> str:
    if value < 1.0:
        return "lt_1"
    if value < 3.0:
        return "1_3"
    return "ge_3"


def _q_decomposition(
    records: list[LearnabilityRecord],
    parent_q: np.ndarray,
    candidate_q: np.ndarray,
) -> dict[str, Any]:
    buckets: dict[str, QDecompositionStats] = defaultdict(QDecompositionStats)
    for record, parent_row, candidate_row in zip(records, parent_q, candidate_q, strict=True):
        for name in _bucket_names(record):
            buckets[name].add(parent_row, candidate_row, record)
    return {key: value.to_json() for key, value in sorted(buckets.items())}


def _flatten_gradients(params: list[torch.nn.Parameter], grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor | None:
    values = [grad.detach().float().reshape(-1) for param, grad in zip(params, grads, strict=True) if grad is not None]
    return torch.cat(values) if values else None


def _grad_norm(value: torch.Tensor | None) -> float | None:
    return float(torch.linalg.vector_norm(value).detach().cpu()) if value is not None else None


def _cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None or left.numel() != right.numel():
        return None
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _gradient_diagnostic(
    checkpoint_path: Path,
    records: list[LearnabilityRecord],
    *,
    device: torch.device,
    batch_size: int,
    batch_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    state = _load_checkpoint(checkpoint_path)
    brain, dqn, aux, _ = _build_model(state, device)
    del state
    brain.train()
    dqn.train()
    aux.train()
    all_params = list(brain.parameters()) + list(dqn.parameters()) + list(aux.parameters())
    slices = {
        "brain": list(brain.parameters()),
        "dqn": list(dqn.parameters()),
        "aux_net": list(aux.parameters()),
    }
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name, params in slices.items():
        end = start + len(params)
        offsets[name] = (start, end)
        start = end
    gamma = float(config["env"].get("gamma", 1.0))
    cql_weight = float(config["cql"]["min_q_weight"])
    aux_weight = float(config["aux"]["next_rank_weight"])
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, Any]] = []
    total = min(batch_count, len(records) // batch_size)
    for batch_index in range(total):
        batch = records[batch_index * batch_size : (batch_index + 1) * batch_size]
        obs = torch.as_tensor(np.stack([record.obs for record in batch]), device=device, dtype=torch.float32)
        masks = torch.as_tensor(np.stack([record.mask for record in batch]), device=device, dtype=torch.bool)
        actions = torch.as_tensor([record.action for record in batch], device=device, dtype=torch.int64)
        steps = torch.as_tensor([record.steps_to_done for record in batch], device=device, dtype=torch.int64)
        targets = torch.as_tensor([record.q_target for record in batch], device=device, dtype=torch.float32)
        player_ranks = torch.as_tensor([record.player_rank for record in batch], device=device, dtype=torch.int64)
        with torch.autocast(device.type, enabled=bool(config["control"].get("enable_amp", False))):
            phi = brain(obs)
            q_out = dqn(phi, masks)
            q = q_out[torch.arange(batch_size, device=device), actions]
            # q_target is already the gamma-discounted final-rank MC target
            # reconstructed by _record_game. Do not discount it a second time.
            q_target = targets.to(torch.float32)
            dqn_loss = 0.5 * torch.mean((q - q_target) ** 2)
            cql_loss = q_out.logsumexp(-1).mean() - q.mean()
            (next_rank_logits,) = aux(phi)
            next_rank_loss = criterion(next_rank_logits, player_ranks)
        losses = {"dqn": dqn_loss, "cql": cql_loss, "aux": next_rank_loss}
        gradients: dict[str, dict[str, torch.Tensor | None]] = {}
        for loss_name, loss in losses.items():
            raw = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)
            gradients[loss_name] = {}
            for module_name, (lo, hi) in offsets.items():
                gradients[loss_name][module_name] = _flatten_gradients(slices[module_name], raw[lo:hi])
        row = {
            "batch_index": batch_index,
            "batch_size": batch_size,
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "raw_gradient_norms": {
                loss_name: {module: _grad_norm(value) for module, value in module_values.items()}
                for loss_name, module_values in gradients.items()
            },
            "gradient_cosines": {
                module: {
                    "dqn_cql": _cosine(gradients["dqn"][module], gradients["cql"][module]),
                    "dqn_aux": _cosine(gradients["dqn"][module], gradients["aux"][module]),
                    "cql_aux": _cosine(gradients["cql"][module], gradients["aux"][module]),
                }
                for module in slices
            },
            "weighted_gradient_norms": {
                "dqn": {module: _grad_norm(value) for module, value in gradients["dqn"].items()},
                "cql": {module: _grad_norm(value * cql_weight if value is not None else None) for module, value in gradients["cql"].items()},
                "aux": {module: _grad_norm(value * aux_weight if value is not None else None) for module, value in gradients["aux"].items()},
            },
        }
        rows.append(row)
        brain.zero_grad(set_to_none=True)
        dqn.zero_grad(set_to_none=True)
        aux.zero_grad(set_to_none=True)
    if not rows:
        raise ValueError("gradient diagnostic did not have enough complete batches")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "batches": rows,
        "mean_losses": {
            key: float(np.mean([row["losses"][key] for row in rows])) for key in rows[0]["losses"]
        },
        "mean_raw_gradient_norms": {
            loss: {
                module: _mean_optional([row["raw_gradient_norms"][loss][module] for row in rows])
                for module in ("brain", "dqn", "aux_net")
            }
            for loss in ("dqn", "cql", "aux")
        },
        "mean_gradient_cosines": {
            module: {
                key: _mean_optional([row["gradient_cosines"][module][key] for row in rows])
                for key in ("dqn_cql", "dqn_aux", "cql_aux")
            }
            for module in ("brain", "dqn", "aux_net")
        },
        "mean_weighted_gradient_norms": {
            loss: {
                module: _mean_optional([row["weighted_gradient_norms"][loss][module] for row in rows])
                for module in ("brain", "dqn", "aux_net")
            }
            for loss in ("dqn", "cql", "aux")
        },
    }


def _mean_optional(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Objective Learnability Audit",
        "",
        "Analysis-only audit of final-rank MC target calibration, Q drift and loss gradients.",
        "No parameters were updated.",
        "",
        "## Route Summary",
        "",
        "| Route | Files | Decisions | Target mean | Target std | Sample |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, route in report["routes"].items():
        lines.append(
            f"| {name} | {route['files']} | {route['decisions']} | {route['target_mean']:.6f} | "
            f"{route['target_std']:.6f} | {route['sample_decisions']} |"
        )
    lines.extend(["", "## Parent Target Calibration", ""])
    for name, route in report["routes"].items():
        overall = route["parent_calibration"]["buckets"]["all"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Parent behavior-action agreement: `{route['parent_calibration']['agreement_rate']:.4%}`.",
                f"- Mean greedy margin: `{route['parent_calibration']['mean_greedy_margin']:.6f}`.",
                f"- Pearson Q/target: `{_fmt_optional(overall.get('pearson_r'))}`; "
                f"OLS R^2: `{_fmt_optional(overall.get('ols_r_squared'))}`; "
                f"prefix Spearman: `{_fmt_optional(overall.get('prefix_sample_spearman'))}`.",
                f"- Identity target-minus-Q mean/std: "
                f"`{_fmt_optional(overall.get('identity_residual_target_minus_q_mean'))}` / "
                f"`{_fmt_optional(overall.get('identity_residual_target_minus_q_std'))}`; "
                f"identity explained variance: `{_fmt_optional(overall.get('identity_explained_variance'))}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Q Decomposition",
            "",
            "| Candidate | Greedy change | Raw Q abs delta | Centered advantage abs delta | Abs common offset | Behavior Q delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, candidate in report["q_decomposition"].items():
        overall = candidate["all"]
        lines.append(
            f"| {name} | {_fmt_optional(overall['greedy_change_rate'], 3)} | "
            f"{_fmt_optional(overall['mean_raw_q_abs_delta'], 6)} | "
            f"{_fmt_optional(overall['mean_centered_advantage_abs_delta'], 6)} | "
            f"{_fmt_optional(overall['mean_abs_common_q_offset'], 6)} | "
            f"{_fmt_optional(overall['mean_behavior_q_delta'], 6)} |"
        )
    lines.extend(["", "## Gradient Diagnostic", ""])
    for name, gradient in report["gradient_diagnostic"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Mean raw losses: `{gradient['mean_losses']}`.")
        lines.append(f"- Mean raw gradient norms: `{gradient['mean_raw_gradient_norms']}`.")
        lines.append(f"- Mean loss-gradient cosines: `{gradient['mean_gradient_cosines']}`.")
        lines.append(f"- Mean weighted gradient norms: `{gradient['mean_weighted_gradient_norms']}`.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "This report is diagnostic. It does not promote a reward, checkpoint, "
            "architecture or data route. Use the target variance, centered-Q drift "
            "and gradient alignment together before opening one new objective experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_optional(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-file-index", type=Path, required=True)
    parser.add_argument("--s0-file-index", type=Path, required=True)
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--s0-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--m0-checkpoint", type=Path, required=True)
    parser.add_argument("--s0-checkpoint", type=Path, required=True)
    parser.add_argument("--m0-player-name", default="ext_mortal")
    parser.add_argument("--s0-player-name", default="train_ext")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-decisions", type=int, default=8192)
    parser.add_argument("--probe-states", type=int, default=4096)
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--gradient-batch-size", type=int, default=512)
    parser.add_argument("--gradient-batches", type=int, default=2)
    parser.add_argument("--file-batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as handle:
        config = tomllib.load(handle)
    if str(config.get("reward", {}).get("mode", "final_rank_mc")) != "final_rank_mc":
        raise ValueError(f"{path}: objective audit requires reward.mode=final_rank_mc")
    return config


def main() -> None:
    args = parse_args()
    if args.sample_decisions <= 0 or args.probe_states <= 0 or args.file_batch_size <= 0:
        raise ValueError("sample, probe and file-batch sizes must be positive")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    parent_path = args.parent.resolve()
    m0_checkpoint = args.m0_checkpoint.resolve()
    s0_checkpoint = args.s0_checkpoint.resolve()
    for path in (parent_path, m0_checkpoint, s0_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    m0_config = _load_config(args.m0_config)
    s0_config = _load_config(args.s0_config)
    parent_state = _load_checkpoint(parent_path)
    version = int(parent_state["config"]["control"].get("version", 4))
    for config in (m0_config, s0_config):
        if int(config["control"]["version"]) != version:
            raise ValueError("route config/model version differs from parent")
    pts = np.asarray(m0_config["env"].get("pts", RANK_POINTS_DEFAULT), dtype=np.float64)
    gamma = float(m0_config["env"].get("gamma", 1.0))
    if pts.shape != (4,):
        raise ValueError(f"expected four rank points, got {pts}")

    m0 = collect_route(
        file_index=args.m0_file_index,
        player_name=args.m0_player_name,
        version=version,
        pts=pts,
        gamma=gamma,
        sample_limit=args.sample_decisions,
        seed=args.seed,
        max_files=args.max_files,
        file_batch_size=args.file_batch_size,
        progress_every=args.progress_every,
    )
    s0 = collect_route(
        file_index=args.s0_file_index,
        player_name=args.s0_player_name,
        version=version,
        pts=pts,
        gamma=gamma,
        sample_limit=args.sample_decisions,
        seed=args.seed + 1,
        max_files=args.max_files,
        file_batch_size=args.file_batch_size,
        progress_every=args.progress_every,
    )
    route_payloads = {"M0": m0, "S0": s0}
    m0_sample = m0.pop("sample")
    s0_sample = s0.pop("sample")
    all_probe_source = m0_sample + s0_sample
    probe_rng = random.Random(args.seed + 2)
    probe = probe_rng.sample(all_probe_source, min(args.probe_states, len(all_probe_source)))
    route_samples = {"M0": m0_sample, "S0": s0_sample}

    print(f"[objective] parent Q on probe routes: {len(probe)} states", flush=True)
    parent_brain, parent_dqn, _, _ = _build_model(parent_state, device)
    parent_q_by_route = {
        "M0": _model_q(parent_brain, parent_dqn, m0_sample, device, args.q_batch_size),
        "S0": _model_q(parent_brain, parent_dqn, s0_sample, device, args.q_batch_size),
    }
    parent_q_probe = _model_q(parent_brain, parent_dqn, probe, device, args.q_batch_size)
    del parent_brain, parent_dqn, parent_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, route in route_payloads.items():
        route["parent_calibration"] = _q_calibration(route_samples[name], parent_q_by_route[name])

    q_decomposition: dict[str, Any] = {}
    checkpoint_paths = {"M0_72000": m0_checkpoint, "S0_72000": s0_checkpoint}
    for name, checkpoint_path in checkpoint_paths.items():
        state = _load_checkpoint(checkpoint_path)
        brain, dqn, _, _ = _build_model(state, device)
        candidate_q_probe = _model_q(brain, dqn, probe, device, args.q_batch_size)
        q_decomposition[name] = _q_decomposition(probe, parent_q_probe, candidate_q_probe)
        del state, brain, dqn, candidate_q_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gradient_diagnostic = {
        "M0_72000": _gradient_diagnostic(
            m0_checkpoint,
            m0_sample,
            device=device,
            batch_size=args.gradient_batch_size,
            batch_count=args.gradient_batches,
            config=m0_config,
        ),
        "S0_72000": _gradient_diagnostic(
            s0_checkpoint,
            s0_sample,
            device=device,
            batch_size=args.gradient_batch_size,
            batch_count=args.gradient_batches,
            config=s0_config,
        ),
    }
    report = {
        "schema": "keqing.mortal.objective_learnability_audit.v2",
        "analysis_only": True,
        "device": str(device),
        "parent": {
            "path": str(parent_path),
            "sha256": _sha256_file(parent_path),
        },
        "configs": {
            "M0": str(args.m0_config.resolve()),
            "S0": str(args.s0_config.resolve()),
        },
        "reward": {"mode": "final_rank_mc", "rank_points": [float(value) for value in pts], "gamma": gamma},
        "sampling": {
            "sample_decisions_per_route": args.sample_decisions,
            "probe_states": len(probe),
            "q_batch_size": args.q_batch_size,
            "gradient_batch_size": args.gradient_batch_size,
            "gradient_batches": args.gradient_batches,
            "file_batch_size": args.file_batch_size,
            "seed": args.seed,
            "selection": "deterministic reservoir per route plus deterministic mixed probe",
        },
        "routes": route_payloads,
        "q_decomposition": q_decomposition,
        "gradient_diagnostic": gradient_diagnostic,
        "checkpoints": {
            "M0_72000": {"path": str(m0_checkpoint), "sha256": _sha256_file(m0_checkpoint)},
            "S0_72000": {"path": str(s0_checkpoint), "sha256": _sha256_file(s0_checkpoint)},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "objective_learnability_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "objective_learnability_audit.md").write_text(
        _build_markdown(report), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "routes": {key: value["decisions"] for key, value in route_payloads.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
