#!/usr/bin/env python3
"""Audit model drift on a deterministic replay-state probe set.

The probe is taken from arena logs that are separate from the offline training
file index.  The script compares every checkpoint with a common parent and
reports action, Q-value, and parameter drift without starting a new training
run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON = MORTAL_ROOT / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))


ACTION_SPACE = 46


@dataclass(frozen=True)
class ProbeState:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    kyoku: int
    current_rank: int
    score_gap: float
    own_riichi: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="candidate checkpoint; repeat for every F/G checkpoint",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        required=True,
        help="directory containing held-out arena logs",
    )
    parser.add_argument("--probe-games", type=int, default=128)
    parser.add_argument("--probe-states", type=int, default=4096)
    parser.add_argument("--probe-seed", type=int, default=20260722)
    parser.add_argument("--player-name", default=None)
    parser.add_argument("--performance-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must use LABEL=PATH: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip()).resolve()
        if not label or label in seen:
            raise ValueError(f"duplicate or empty checkpoint label: {label!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(label)
        specs.append((label, path))
    return specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_probe_logs(root: Path) -> list[Path]:
    logs = sorted(root.glob("**/*.json.gz"))
    if not logs:
        raise FileNotFoundError(f"no .json.gz logs under {root.resolve()}")
    return logs


def choose_probe_logs(logs: list[Path], count: int, seed: int) -> list[Path]:
    if count <= 0:
        raise ValueError("--probe-games must be positive")
    if count >= len(logs):
        return logs
    rng = random.Random(seed)
    return sorted(rng.sample(logs, count))


def action_kind(action: int) -> str:
    if action < 37:
        return "discard"
    return {
        37: "reach",
        38: "chi_low",
        39: "chi_mid",
        40: "chi_high",
        41: "pon",
        42: "kan",
        43: "agari",
        44: "ryukyoku",
        45: "pass",
    }.get(action, "other")


def score_gap_bucket(score_gap: float) -> str:
    if score_gap >= 12000:
        return "ahead_big"
    if score_gap >= 0:
        return "ahead"
    if score_gap > -12000:
        return "behind"
    return "behind_big"


def phase_bucket(kyoku: int) -> str:
    if kyoku <= 1:
        return "early"
    if kyoku <= 5:
        return "middle"
    return "late"


def reservoir_add(
    reservoir: list[ProbeState],
    item: ProbeState,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    index = rng.randrange(seen + 1)
    if index < limit:
        reservoir[index] = item


def collect_probe(
    logs: list[Path],
    *,
    version: int,
    player_name: str | None,
    state_limit: int,
    seed: int,
) -> tuple[list[ProbeState], dict[str, Any]]:
    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=[player_name] if player_name else [],
        augmented=False,
    )
    rng = random.Random(seed)
    reservoir: list[ProbeState] = []
    seen = 0
    games_seen = 0
    perspective_count = 0
    for path in logs:
        loaded = loader.load_gz_log_files([str(path)])[0]
        games_seen += 1
        for game in loaded:
            obs = game.take_obs()
            masks = game.take_masks()
            actions = game.take_actions()
            at_kyoku = game.take_at_kyoku()
            player_id = int(game.take_player_id())
            grp = game.take_grp()
            features = np.asarray(grp.take_feature(), dtype=np.float32)
            if features.ndim != 2 or features.shape[1] < 7:
                raise ValueError(f"unexpected GRP feature shape in {path}: {features.shape}")
            perspective_count += 1
            reached_kyoku: int | None = None
            for obs_i, mask_i, action, kyoku_raw in zip(obs, masks, actions, at_kyoku):
                kyoku = int(kyoku_raw)
                scores = features[min(kyoku, len(features) - 1), 3:] * 10000.0
                own_score = float(scores[player_id])
                opponent_max = max(float(scores[i]) for i in range(4) if i != player_id)
                rank_order = np.argsort(-scores, kind="stable")
                current_rank = int(np.where(rank_order == player_id)[0][0]) + 1
                state = ProbeState(
                    obs=np.asarray(obs_i, dtype=np.float32),
                    mask=np.asarray(mask_i, dtype=bool),
                    action=int(action),
                    kyoku=kyoku,
                    current_rank=current_rank,
                    score_gap=own_score - opponent_max,
                    own_riichi=reached_kyoku == kyoku,
                )
                reservoir_add(reservoir, state, seen, state_limit, rng)
                seen += 1
                if int(action) == 37:
                    reached_kyoku = kyoku
            # The game object is consumed above.  The next perspective is a
            # separate object and keeps the probe balanced across seats.
    if not reservoir:
        raise ValueError("probe contained no Mortal decision states")
    metadata = {
        "log_count": len(logs),
        "games_loaded": games_seen,
        "perspectives_loaded": perspective_count,
        "candidate_states_seen": seen,
        "states_selected": len(reservoir),
        "selection": "deterministic reservoir sampling",
        "probe_seed": seed,
        "player_name_filter": player_name,
        "log_sha256": [sha256_file(path) for path in logs],
        "log_paths": [str(path.resolve()) for path in logs],
    }
    return reservoir, metadata


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except (RuntimeError, pickle.UnpicklingError):  # type: ignore[name-defined]
        return torch.load(path, weights_only=False, map_location="cpu")


def build_models(state: dict[str, Any], device: torch.device):
    from model import Brain, DQN

    config = state["config"]
    version = int(config["control"].get("version", 4))
    brain = Brain(version=version, **config["resnet"]).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return brain, dqn, version


def model_q(
    brain: torch.nn.Module,
    dqn: torch.nn.Module,
    states: list[ProbeState],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            obs = torch.as_tensor(np.stack([s.obs for s in batch]), device=device)
            masks = torch.as_tensor(np.stack([s.mask for s in batch]), device=device)
            phi = brain(obs)
            q = dqn(phi, masks)
            rows.append(q.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def finite_q(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return q[mask & np.isfinite(q)]


def q_metrics(parent_q: np.ndarray, candidate_q: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    parent_greedy = np.argmax(parent_q, axis=1)
    candidate_greedy = np.argmax(candidate_q, axis=1)
    margins_parent: list[float] = []
    margins_candidate: list[float] = []
    abs_q_delta: list[float] = []
    signed_q_delta: list[float] = []
    q_scale_parent: list[float] = []
    q_scale_candidate: list[float] = []
    q_offsets: list[float] = []
    centered_abs_delta: list[float] = []
    legal_std_parent: list[float] = []
    legal_std_candidate: list[float] = []
    legal_rank_agreement: list[float] = []
    for i, mask in enumerate(masks):
        valid = mask & np.isfinite(parent_q[i]) & np.isfinite(candidate_q[i])
        p = parent_q[i, valid]
        c = candidate_q[i, valid]
        if len(p) < 2:
            continue
        p_sorted = np.sort(p)
        c_sorted = np.sort(c)
        margins_parent.append(float(p_sorted[-1] - p_sorted[-2]))
        margins_candidate.append(float(c_sorted[-1] - c_sorted[-2]))
        abs_q_delta.extend(np.abs(c - p).tolist())
        signed_q_delta.extend((c - p).tolist())
        q_scale_parent.append(float(np.mean(np.abs(p))))
        q_scale_candidate.append(float(np.mean(np.abs(c))))
        p_offset = float(np.mean(p))
        c_offset = float(np.mean(c))
        q_offsets.append(c_offset - p_offset)
        centered_abs_delta.extend(np.abs((c - c_offset) - (p - p_offset)).tolist())
        legal_std_parent.append(float(np.std(p)))
        legal_std_candidate.append(float(np.std(c)))
        legal_rank_agreement.append(float(np.array_equal(np.argsort(p), np.argsort(c))))
    return {
        "states": float(len(parent_q)),
        "greedy_agreement_rate": float(np.mean(parent_greedy == candidate_greedy)),
        "greedy_change_rate": float(np.mean(parent_greedy != candidate_greedy)),
        "mean_abs_q_delta": float(np.mean(abs_q_delta)) if abs_q_delta else 0.0,
        "mean_signed_q_delta": float(np.mean(signed_q_delta)) if signed_q_delta else 0.0,
        "parent_q_abs_mean": float(np.mean(q_scale_parent)) if q_scale_parent else 0.0,
        "candidate_q_abs_mean": float(np.mean(q_scale_candidate)) if q_scale_candidate else 0.0,
        "q_offset_delta_mean": float(np.mean(q_offsets)) if q_offsets else 0.0,
        "q_offset_abs_mean": float(np.mean(np.abs(q_offsets))) if q_offsets else 0.0,
        "centered_q_abs_delta": float(np.mean(centered_abs_delta)) if centered_abs_delta else 0.0,
        "parent_legal_q_std_mean": float(np.mean(legal_std_parent)) if legal_std_parent else 0.0,
        "candidate_legal_q_std_mean": float(np.mean(legal_std_candidate)) if legal_std_candidate else 0.0,
        "legal_q_std_ratio": float(np.mean(legal_std_candidate) / np.mean(legal_std_parent))
        if legal_std_parent and np.mean(legal_std_parent) > 0
        else 0.0,
        "legal_action_rank_agreement_rate": float(np.mean(legal_rank_agreement))
        if legal_rank_agreement
        else 0.0,
        "parent_margin_mean": float(np.mean(margins_parent)) if margins_parent else 0.0,
        "candidate_margin_mean": float(np.mean(margins_candidate)) if margins_candidate else 0.0,
        "margin_delta_mean": float(np.mean(margins_candidate) - np.mean(margins_parent))
        if margins_parent and margins_candidate
        else 0.0,
    }


def parameter_drift(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for module_key, output_key in (("mortal", "brain"), ("current_dqn", "dqn"), ("aux_net", "aux_net")):
        parent_state = parent.get(module_key)
        candidate_state = candidate.get(module_key)
        if not isinstance(parent_state, dict) or not isinstance(candidate_state, dict):
            result[output_key] = {"available": False}
            continue
        param_diff_sq = 0.0
        param_parent_sq = 0.0
        param_abs_diff = 0.0
        param_count = 0
        buffer_diff_sq = 0.0
        buffer_parent_sq = 0.0
        buffer_abs_diff = 0.0
        buffer_count = 0
        for key, parent_value in parent_state.items():
            if key not in candidate_state:
                continue
            p = parent_value.detach().float()
            c = candidate_state[key].detach().float()
            if p.shape != c.shape:
                continue
            diff = c - p
            is_buffer = key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked")
            if is_buffer:
                buffer_diff_sq += float(torch.sum(diff * diff))
                buffer_parent_sq += float(torch.sum(p * p))
                buffer_abs_diff += float(torch.sum(torch.abs(diff)))
                buffer_count += p.numel()
            else:
                param_diff_sq += float(torch.sum(diff * diff))
                param_parent_sq += float(torch.sum(p * p))
                param_abs_diff += float(torch.sum(torch.abs(diff)))
                param_count += p.numel()
        result[output_key] = {
            "available": param_count > 0,
            "parameter_count": param_count,
            "trainable_parameter_relative_l2": math.sqrt(param_diff_sq / param_parent_sq)
            if param_parent_sq > 0
            else None,
            "trainable_parameter_mean_abs_delta": param_abs_diff / param_count if param_count else None,
            "floating_buffer_count": buffer_count,
            "floating_buffer_relative_l2": math.sqrt(buffer_diff_sq / buffer_parent_sq)
            if buffer_parent_sq > 0
            else None,
            "floating_buffer_mean_abs_delta": buffer_abs_diff / buffer_count if buffer_count else None,
        }
    return result


def stratum_names(state: ProbeState) -> list[str]:
    names = [
        "all",
        f"phase_{phase_bucket(state.kyoku)}",
        f"rank_{state.current_rank}",
        f"score_{score_gap_bucket(state.score_gap)}",
        f"action_{action_kind(state.action)}",
    ]
    if state.own_riichi:
        names.append("own_riichi_active")
    else:
        names.append("own_riichi_inactive")
    return names


def summarize_strata(
    states: list[ProbeState],
    parent_q: np.ndarray,
    candidate_q: np.ndarray,
) -> dict[str, dict[str, float]]:
    indexes: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        for name in stratum_names(state):
            indexes[name].append(index)
    masks = np.stack([state.mask for state in states])
    return {
        name: q_metrics(parent_q[indexes[name]], candidate_q[indexes[name]], masks[indexes[name]])
        for name in sorted(indexes)
        if indexes[name]
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Checkpoint Drift Audit",
        "",
        f"- Probe states: `{report['probe']['states_selected']}`",
        f"- Probe logs: `{report['probe']['log_count']}`",
        f"- Device: `{report['device']}`",
        "- Probe source is arena evaluation data and is separate from the 6000-file training index.",
        "",
        "## Parameter Drift",
        "",
        "| checkpoint | brain parameter L2 | DQN parameter L2 | AuxNet parameter L2 | greedy change | raw Q delta | centered Q delta | Q offset |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["checkpoints"].items():
        param = row["parameter_drift"]
        overall = row["overall"]
        lines.append(
            f"| {label} | {param['brain'].get('trainable_parameter_relative_l2', 0):.6g} | "
            f"{param['dqn'].get('trainable_parameter_relative_l2', 0):.6g} | "
            f"{param['aux_net'].get('trainable_parameter_relative_l2', 0):.6g} | "
            f"{overall['greedy_change_rate'] * 100:.3f}% | {overall['mean_abs_q_delta']:.6g} | "
            f"{overall['centered_q_abs_delta']:.6g} | {overall['q_offset_abs_mean']:.6g} |"
        )
    lines.extend(["", "## Interpretation Inputs", "", "Strata are reported in JSON under `strata`; they are diagnostic slices, not promotion gates.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.probe_states <= 0:
        raise ValueError("--batch-size and --probe-states must be positive")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    parent_path = args.parent.resolve()
    if not parent_path.exists():
        raise FileNotFoundError(parent_path)
    specs = parse_checkpoint_specs(args.checkpoint)
    parent_state = load_checkpoint(parent_path)
    _, _, version = build_models(parent_state, torch.device("cpu"))
    probe_logs = choose_probe_logs(
        discover_probe_logs(args.probe_root.resolve()),
        int(args.probe_games),
        int(args.probe_seed),
    )
    states, probe_meta = collect_probe(
        probe_logs,
        version=version,
        player_name=args.player_name,
        state_limit=int(args.probe_states),
        seed=int(args.probe_seed),
    )
    masks = np.stack([state.mask for state in states])
    parent_brain, parent_dqn, _ = build_models(parent_state, device)
    print(f"[drift] evaluating parent on {len(states)} states", flush=True)
    parent_q = model_q(parent_brain, parent_dqn, states, device, int(args.batch_size))
    del parent_brain, parent_dqn

    report: dict[str, Any] = {
        "schema": "keqing.mortal.checkpoint_drift.v1",
        "parent": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
        "device": str(device),
        "probe": probe_meta,
        "performance_summary": str(args.performance_summary.resolve()) if args.performance_summary else None,
        "checkpoints": {},
    }
    for label, path in specs:
        print(f"[drift] {label}", flush=True)
        state = load_checkpoint(path)
        brain, dqn, candidate_version = build_models(state, device)
        if candidate_version != version:
            raise ValueError(f"version mismatch for {label}: {candidate_version} != {version}")
        candidate_q = model_q(brain, dqn, states, device, int(args.batch_size))
        overall = q_metrics(parent_q, candidate_q, masks)
        report["checkpoints"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "steps": int(state.get("steps", -1)),
            "parameter_drift": parameter_drift(parent_state, state),
            "overall": overall,
            "strata": summarize_strata(states, parent_q, candidate_q),
        }
        del brain, dqn, candidate_q
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_drift_audit.json"
    md_path = output_dir / "checkpoint_drift_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
