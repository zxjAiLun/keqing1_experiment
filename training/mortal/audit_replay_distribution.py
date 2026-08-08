#!/usr/bin/env python3
"""Audit the learning distribution of the retained Mortal replay corpus.

This is an analysis-only pass.  It does not rewrite logs, build a new file
index, or start training.  The audit treats decisions as the observed sample
unit, while retaining hanchan clusters for contribution and uncertainty
diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import tomllib
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))


ACTION_NAMES = {
    **{action: "discard" for action in range(37)},
    37: "reach",
    38: "chi_low",
    39: "chi_mid",
    40: "chi_high",
    41: "pon",
    42: "kan",
    43: "agari",
    44: "ryukyoku",
    45: "pass",
}
TARGET_VALUES = (-3.0, -1.0, 1.0, 3.0)


@dataclass
class DecisionRecord:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    target: float
    target_rank: int
    phase: str
    kyoku: int
    current_rank: int
    score_gap: float
    own_riichi: bool
    legal_actions: int
    shanten: int
    action_kind: str


class SupportAccumulator:
    def __init__(self) -> None:
        self.states = 0
        self.legal_behavior = 0
        self.agreement = 0
        self.q_rank_counts: Counter[int] = Counter()
        self.q_regret_sum = 0.0
        self.margin_sum = 0.0
        self.behavior_q_sum = 0.0
        self.greedy_q_sum = 0.0
        self.q_scale_sum = 0.0
        self.margin_states = 0

    def update(self, *, q: np.ndarray, record: DecisionRecord) -> None:
        valid = record.mask.astype(bool) & np.isfinite(q)
        self.states += 1
        if not valid[record.action]:
            return
        self.legal_behavior += 1
        legal_actions = np.flatnonzero(valid)
        legal_q = q[legal_actions]
        greedy_index = int(np.argmax(legal_q))
        greedy_action = int(legal_actions[greedy_index])
        behavior_q = float(q[record.action])
        greedy_q = float(legal_q[greedy_index])
        self.agreement += int(greedy_action == record.action)
        self.q_rank_counts[int(1 + np.sum(legal_q > behavior_q))] += 1
        self.q_regret_sum += greedy_q - behavior_q
        self.behavior_q_sum += behavior_q
        self.greedy_q_sum += greedy_q
        self.q_scale_sum += float(np.mean(np.abs(legal_q)))
        if legal_q.size >= 2:
            top_two = np.partition(legal_q, -2)[-2:]
            self.margin_sum += float(np.max(top_two) - np.min(top_two))
            self.margin_states += 1

    def to_json(self) -> dict[str, Any]:
        legal = self.legal_behavior
        return {
            "states": self.states,
            "behavior_action_legal_rate": self.legal_behavior / self.states if self.states else 0.0,
            "greedy_agreement_rate": self.agreement / legal if legal else 0.0,
            "greedy_disagreement_rate": 1.0 - self.agreement / legal if legal else 0.0,
            "behavior_q_rank_counts": {str(key): int(value) for key, value in sorted(self.q_rank_counts.items())},
            "mean_behavior_q_rank": (
                sum(key * value for key, value in self.q_rank_counts.items()) / legal if legal else 0.0
            ),
            "mean_q_regret_greedy_minus_behavior": self.q_regret_sum / legal if legal else 0.0,
            "mean_greedy_margin": self.margin_sum / self.margin_states if self.margin_states else 0.0,
            "mean_behavior_q": self.behavior_q_sum / legal if legal else 0.0,
            "mean_greedy_q": self.greedy_q_sum / legal if legal else 0.0,
            "mean_legal_q_abs": self.q_scale_sum / legal if legal else 0.0,
        }


class BucketAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.targets: Counter[str] = Counter()
        self.actions: Counter[str] = Counter()

    def update(self, record: DecisionRecord) -> None:
        self.count += 1
        self.target_sum += record.target
        self.target_sq_sum += record.target * record.target
        self.targets[str(record.target)] += 1
        self.actions[record.action_kind] += 1

    def to_json(self) -> dict[str, Any]:
        mean = self.target_sum / self.count if self.count else 0.0
        variance = max(0.0, self.target_sq_sum / self.count - mean * mean) if self.count else 0.0
        probs = [count / self.count for count in self.actions.values()] if self.count else []
        entropy = max(0.0, -sum(prob * math.log(prob) for prob in probs if prob > 0.0))
        return {
            "count": self.count,
            "target_mean": mean,
            "target_std": math.sqrt(variance),
            "target_counts": {key: int(value) for key, value in sorted(self.targets.items())},
            "action_counts": {key: int(value) for key, value in sorted(self.actions.items())},
            "action_entropy_nats": entropy,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="ext_mortal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--file-batch-size", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except (RuntimeError, pickle.UnpicklingError):
        return torch.load(path, weights_only=False, map_location="cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_file_list(path: Path) -> list[Path]:
    payload = torch.load(path.resolve(), weights_only=False, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("file_list")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"file index does not contain a non-empty file_list: {path}")
    files = []
    for value in payload:
        file_path = Path(str(value))
        if not file_path.is_absolute():
            file_path = (REPO_ROOT / file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        files.append(file_path)
    return files


def phase_bucket(kyoku: int) -> str:
    if kyoku < 4:
        return "early"
    if kyoku < 8:
        return "middle"
    return "late"


def score_gap_bucket(score_gap: float) -> str:
    if score_gap >= 12000:
        return "ahead_big"
    if score_gap >= 0:
        return "ahead"
    if score_gap > -12000:
        return "behind"
    return "behind_big"


def legal_count_bucket(count: int) -> str:
    if count <= 5:
        return "1_5"
    if count <= 10:
        return "6_10"
    return "11_plus"


def shanten_bucket(value: int) -> str:
    if value <= 0:
        return "tenpai_or_agari"
    if value <= 2:
        return str(value)
    return "3_plus"


def action_name(action: int) -> str:
    return ACTION_NAMES.get(action, "other")


def current_rank_and_gap(features: np.ndarray, player_id: int, kyoku: int) -> tuple[int, float]:
    row = features[min(kyoku, len(features) - 1), 3:7] * 10000.0
    own_score = float(row[player_id])
    opponent_max = max(float(row[index]) for index in range(4) if index != player_id)
    order = np.argsort(-row, kind="stable")
    rank = int(np.flatnonzero(order == player_id)[0]) + 1
    return rank, own_score - opponent_max


def decision_hash(record: DecisionRecord) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(record.obs, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(record.mask, dtype=np.bool_).tobytes())
    digest.update(int(record.action).to_bytes(2, "little", signed=False))
    return digest.digest()


def state_hash(record: DecisionRecord) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(record.obs, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(record.mask, dtype=np.bool_).tobytes())
    return digest.digest()


def records_from_game(game: Any, pts: np.ndarray) -> Iterable[DecisionRecord]:
    obs = game.take_obs()
    masks = game.take_masks()
    actions = game.take_actions()
    at_kyoku = game.take_at_kyoku()
    at_turns = game.take_at_turns()
    shantens = game.take_shantens()
    grp = game.take_grp()
    features = np.asarray(grp.take_feature(), dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < 7:
        raise ValueError(f"unexpected GRP feature shape: {features.shape}")
    player_id = int(game.take_player_id())
    final_rank = int(grp.take_rank_by_player()[player_id])
    target = float(pts[final_rank] - pts.mean())
    own_riichi_kyoku: int | None = None
    lengths = {len(obs), len(masks), len(actions), len(at_kyoku), len(at_turns), len(shantens)}
    if len(lengths) != 1:
        raise ValueError(f"decision arrays have inconsistent lengths: {sorted(lengths)}")
    for obs_i, mask_i, action_i, kyoku_i, turn_i, shanten_i in zip(
        obs, masks, actions, at_kyoku, at_turns, shantens, strict=True
    ):
        kyoku = int(kyoku_i)
        rank, score_gap = current_rank_and_gap(features, player_id, kyoku)
        action = int(action_i)
        own_riichi = own_riichi_kyoku == kyoku
        record = DecisionRecord(
            obs=np.asarray(obs_i, dtype=np.float32),
            mask=np.asarray(mask_i, dtype=np.bool_),
            action=action,
            target=target,
            target_rank=final_rank + 1,
            phase=phase_bucket(kyoku),
            kyoku=kyoku,
            current_rank=rank,
            score_gap=score_gap,
            own_riichi=own_riichi,
            legal_actions=int(np.asarray(mask_i, dtype=np.bool_).sum()),
            shanten=int(shanten_i),
            action_kind=action_name(action),
        )
        yield record
        if action == 37:
            own_riichi_kyoku = kyoku


def bucket_key(record: DecisionRecord) -> str:
    return json.dumps(
        {
            "phase": record.phase,
            "rank": record.current_rank,
            "score_gap": score_gap_bucket(record.score_gap),
            "own_riichi": record.own_riichi,
            "legal_actions": legal_count_bucket(record.legal_actions),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def support_key(record: DecisionRecord) -> str:
    return json.dumps(
        {
            "phase": record.phase,
            "rank": record.current_rank,
            "score_gap": score_gap_bucket(record.score_gap),
            "own_riichi": record.own_riichi,
            "legal_actions": legal_count_bucket(record.legal_actions),
            "shanten": shanten_bucket(record.shanten),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def model_q(brain: torch.nn.Module, dqn: torch.nn.Module, records: list[DecisionRecord], device: torch.device) -> np.ndarray:
    obs = torch.as_tensor(np.stack([record.obs for record in records]), device=device)
    masks = torch.as_tensor(np.stack([record.mask for record in records]), device=device)
    with torch.inference_mode():
        phi = brain(obs)
        q = dqn(phi, masks)
    return q.float().cpu().numpy()


def load_model(state: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, int]:
    from model import Brain, DQN

    model_config = state["config"]
    version = int(model_config["control"].get("version", 4))
    brain = Brain(version=version, **model_config["resnet"]).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return brain, dqn, version


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * ordered) / (n * ordered.sum()))


def contribution_summary(decision_counts: list[int]) -> dict[str, Any]:
    values = np.asarray(decision_counts, dtype=np.float64)
    total = float(values.sum())
    ordered = np.sort(values)[::-1]

    def share(percent: float) -> float:
        count = max(1, math.ceil(len(ordered) * percent))
        return float(ordered[:count].sum() / total) if total else 0.0

    return {
        "hanchans": len(decision_counts),
        "total_decisions": int(total),
        "decisions_per_hanchan_quantiles": {
            quantile: float(np.quantile(values, quantile)) if len(values) else 0.0
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "top_1pct_decision_share": share(0.01),
        "top_5pct_decision_share": share(0.05),
        "top_10pct_decision_share": share(0.10),
        "decision_weight_gini": gini(decision_counts),
        "decision_weight_ess": float(total * total / np.square(values).sum()) if total else 0.0,
    }


def counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def finalize_buckets(buckets: dict[str, BucketAccumulator]) -> dict[str, dict[str, Any]]:
    return {key: buckets[key].to_json() for key in sorted(buckets)}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    corpus = report["corpus"]
    contribution = report["hanchan_contribution"]
    support = report["support_audit"]["overall"]
    duplicates = report["duplicates"]
    conflicts = report["target_conflict"]["buckets"]
    ranked_conflicts = sorted(
        conflicts.items(),
        key=lambda item: (item[1]["target_std"], item[1]["count"]),
        reverse=True,
    )[:20]
    lines = [
        "# Replay Distribution Audit",
        "",
        "Analysis-only audit of the retained 6000-hanchan training corpus. No logs or checkpoints were modified.",
        "",
        "## Corpus",
        "",
        f"- Hanchans scanned: `{corpus['hanchans']}`; malformed files: `{corpus['malformed_count']}`.",
        f"- Trainable perspectives: `{corpus['trainable_perspectives']}`; total decisions: `{corpus['total_decisions']}`.",
        f"- Final-rank target counts: `{corpus['final_rank_counts']}`; target values are centered `{list(TARGET_VALUES)}`.",
        "",
        "## Decision Contribution",
        "",
        f"- Decision-weighted ESS: `{contribution['decision_weight_ess']:.1f}` out of `{contribution['hanchans']}` hanchans.",
        f"- Gini: `{contribution['decision_weight_gini']:.4f}`; top 1/5/10% decision shares: `{contribution['top_1pct_decision_share']:.2%}` / `{contribution['top_5pct_decision_share']:.2%}` / `{contribution['top_10pct_decision_share']:.2%}`.",
        f"- Decision-count quantiles: `{contribution['decisions_per_hanchan_quantiles']}`.",
        "",
        "## 70k Behavior Support",
        "",
        f"- Behavior action legal rate: `{support['behavior_action_legal_rate']:.4%}`.",
        f"- ext_mortal action equals 70k greedy action: `{support['greedy_agreement_rate']:.4%}`.",
        f"- Mean behavior-action Q rank: `{support['mean_behavior_q_rank']:.3f}`; mean greedy-minus-behavior Q regret: `{support['mean_q_regret_greedy_minus_behavior']:.6g}`.",
        f"- Mean 70k greedy margin: `{support['mean_greedy_margin']:.6g}`.",
        "",
        "## Exact Decision Repeats",
        "",
        f"- Unique `(obs, legal_mask, behavior_action)` hashes: `{duplicates['unique_decision_count']}`; exact duplicate rate: `{duplicates['duplicate_decision_rate']:.2%}`.",
        f"- Unique `(obs, legal_mask)` hashes: `{duplicates['unique_state_count']}`; state-only duplicate rate: `{duplicates['state_duplicate_rate']:.2%}`.",
        "",
        "## Highest-Variance Target Buckets",
        "",
        "Buckets use phase, current rank, score-gap bucket, own-riichi state, and legal-action count; action kind is reported inside each bucket. Opponent-riichi count is intentionally omitted from this first pass because the loader does not expose a reliable per-decision global count.",
        "",
        "| Bucket | Count | Target mean | Target std | Action entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in ranked_conflicts:
        lines.append(
            f"| `{key}` | {value['count']} | {value['target_mean']:+.4f} | {value['target_std']:.4f} | {value['action_entropy_nats']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scope Notes",
            "",
            "- The Q regret is an internal 70k support metric, not a ground-truth regret estimate.",
            "- This pass does not perform embedding clustering, data reweighting, reward changes, or training.",
            "- Use the JSON strata and target buckets to choose the next project-owned lineage data design.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.q_batch_size <= 0 or args.file_batch_size <= 0:
        raise ValueError("--q-batch-size and --file-batch-size must be positive")
    if args.max_files < 0:
        raise ValueError("--max-files must be non-negative")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False")

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    reward_mode = str(config.get("reward", {}).get("mode", "final_rank_mc"))
    if reward_mode != "final_rank_mc":
        raise ValueError(f"replay distribution audit requires reward.mode=final_rank_mc, got {reward_mode!r}")
    version = int(config["control"]["version"])
    pts = np.asarray(config["env"].get("pts", [6.0, 4.0, 2.0, 0.0]), dtype=np.float64)
    if pts.shape != (4,):
        raise ValueError(f"expected four rank-point values, got {pts}")

    file_index = args.file_index.resolve()
    parent_path = args.parent.resolve()
    files = load_file_list(file_index)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise ValueError("file index selected no files")

    state = load_checkpoint(parent_path)
    brain, dqn, model_version = load_model(state, device)
    if model_version != version:
        raise ValueError(f"parent model version {model_version} differs from config version {version}")
    del state

    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=[str(args.model_label)],
        augmented=False,
    )
    decision_counts: list[int] = []
    hanchan_rows: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    score_gap_counts: Counter[str] = Counter()
    own_riichi_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    legal_count_counts: Counter[str] = Counter()
    shanten_counts: Counter[str] = Counter()
    final_rank_counts: Counter[int] = Counter()
    target_counts: Counter[float] = Counter()
    target_buckets: dict[str, BucketAccumulator] = {}
    support_overall = SupportAccumulator()
    support_strata: dict[str, SupportAccumulator] = {}
    decision_hashes: Counter[bytes] = Counter()
    state_hashes: Counter[bytes] = Counter()
    malformed: list[dict[str, str]] = []
    total_decisions = 0
    trainable_perspectives = 0

    def consume_loaded_game(file_path: Path, loaded_games: list[Any]) -> int:
        nonlocal trainable_perspectives

        if len(loaded_games) != 1:
            raise ValueError(f"expected one {args.model_label} perspective, got {len(loaded_games)}")
        game = loaded_games[0]
        records = list(records_from_game(game, pts))
        if not records:
            raise ValueError("perspective contains no decisions")

        # Run model inference before mutating aggregate counters so a failed
        # model batch cannot leave a partially counted hanchan behind.
        q_records: list[tuple[DecisionRecord, np.ndarray]] = []
        for start in range(0, len(records), args.q_batch_size):
            batch = records[start : start + args.q_batch_size]
            q_values = model_q(brain, dqn, batch, device)
            q_records.extend(zip(batch, q_values, strict=True))

        for record, q in q_records:
            phase_counts[record.phase] += 1
            rank_counts[record.current_rank] += 1
            score_gap_counts[score_gap_bucket(record.score_gap)] += 1
            own_riichi_counts[str(record.own_riichi).lower()] += 1
            action_counts[record.action_kind] += 1
            legal_count_counts[legal_count_bucket(record.legal_actions)] += 1
            shanten_counts[shanten_bucket(record.shanten)] += 1
            target_counts[record.target] += 1
            decision_hashes[decision_hash(record)] += 1
            state_hashes[state_hash(record)] += 1
            target_buckets.setdefault(bucket_key(record), BucketAccumulator()).update(record)
            support_strata.setdefault(support_key(record), SupportAccumulator()).update(q=q, record=record)
            support_overall.update(q=q, record=record)

        trainable_perspectives += 1
        decision_counts.append(len(records))
        final_rank_counts[records[0].target_rank] += 1
        hanchan_rows.append(
            {
                "source_log": str(file_path),
                "decisions": len(records),
                "kyoku_count": max(record.kyoku for record in records) + 1,
                "late_phase": any(record.phase == "late" for record in records),
                "final_rank": records[0].target_rank,
                "target": records[0].target,
            }
        )
        return len(records)

    processed_files = 0
    for batch_start in range(0, len(files), args.file_batch_size):
        file_batch = files[batch_start : batch_start + args.file_batch_size]
        try:
            loaded_batch = loader.load_gz_log_files([str(path) for path in file_batch])
            if len(loaded_batch) != len(file_batch):
                raise ValueError(f"loader returned {len(loaded_batch)} files for {len(file_batch)} paths")
            for file_path, loaded_games in zip(file_batch, loaded_batch, strict=True):
                try:
                    total_decisions += consume_loaded_game(file_path, loaded_games)
                except Exception as exc:  # noqa: BLE001
                    malformed.append({"path": str(file_path), "error": str(exc)})
                processed_files += 1
        except Exception as batch_exc:  # noqa: BLE001
            # Keep malformed-file isolation if a native batch parser rejects
            # one member of the batch; the successful files still count.
            for file_path in file_batch:
                try:
                    loaded = loader.load_gz_log_files([str(file_path)])
                    if len(loaded) != 1:
                        raise ValueError(f"loader returned {len(loaded)} files")
                    total_decisions += consume_loaded_game(file_path, loaded[0])
                except Exception as exc:  # noqa: BLE001
                    malformed.append(
                        {"path": str(file_path), "error": f"batch={batch_exc}; single={exc}"}
                    )
                processed_files += 1
        del file_batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.progress_every and processed_files % args.progress_every == 0:
            print(
                f"[replay-audit] files {processed_files}/{len(files)} decisions={total_decisions} malformed={len(malformed)}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    contribution = contribution_summary(decision_counts)
    unique_decisions = len(decision_hashes)
    report = {
        "schema": "keqing.mortal.replay_distribution_audit.v1",
        "inputs": {
            "file_index": str(file_index),
            "file_index_sha256": sha256_file(file_index),
            "parent": str(parent_path),
            "parent_sha256": sha256_file(parent_path),
            "config": str(config_path),
            "config_reward_mode": reward_mode,
            "model_label": str(args.model_label),
            "device": str(device),
            "q_batch_size": int(args.q_batch_size),
            "file_batch_size": int(args.file_batch_size),
        },
        "corpus": {
            "files_selected": len(files),
            "hanchans": len(hanchan_rows),
            "trainable_perspectives": trainable_perspectives,
            "total_decisions": total_decisions,
            "malformed_count": len(malformed),
            "malformed": malformed[:50],
            "final_rank_counts": counter_json(final_rank_counts),
            "target_counts": {str(key): int(value) for key, value in sorted(target_counts.items())},
        },
        "hanchan_contribution": contribution,
        "decision_distribution": {
            "phase_counts": counter_json(phase_counts),
            "current_rank_counts": counter_json(rank_counts),
            "score_gap_counts": counter_json(score_gap_counts),
            "own_riichi_counts": counter_json(own_riichi_counts),
            "action_counts": counter_json(action_counts),
            "legal_action_count_buckets": counter_json(legal_count_counts),
            "shanten_buckets": counter_json(shanten_counts),
        },
        "support_audit": {
            "overall": support_overall.to_json(),
            "by_state_stratum": {key: value.to_json() for key, value in sorted(support_strata.items())},
            "q_regret_definition": "Q(parent greedy action) - Q(data behavior action), legal actions only",
        },
        "target_conflict": {
            "bucket_dimensions": ["phase", "current_rank", "score_gap", "own_riichi", "legal_actions"],
            "omitted_dimensions": {
                "opponent_riichi_count": "not exposed as a reliable per-decision field by GameplayLoader in this first pass"
            },
            "buckets": finalize_buckets(target_buckets),
        },
        "duplicates": {
            "hash_definition": "blake2b(obs float32 bytes + legal mask bytes + behavior action)",
            "total_decision_count": total_decisions,
            "unique_decision_count": unique_decisions,
            "duplicate_decision_count": total_decisions - unique_decisions,
            "duplicate_decision_rate": (total_decisions - unique_decisions) / total_decisions if total_decisions else 0.0,
            "max_exact_repeat_count": max(decision_hashes.values(), default=0),
            "state_hash_definition": "blake2b(obs float32 bytes + legal mask bytes)",
            "unique_state_count": len(state_hashes),
            "state_duplicate_count": total_decisions - len(state_hashes),
            "state_duplicate_rate": (total_decisions - len(state_hashes)) / total_decisions if total_decisions else 0.0,
            "max_state_repeat_count": max(state_hashes.values(), default=0),
        },
        "hanchans": hanchan_rows,
        "scope_notes": [
            "analysis only; no logs, file index, checkpoint, or training state was modified",
            "no embedding clustering or automatic reweighting was performed",
            "Q metrics are internal parent-model support metrics, not ground-truth regret",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_distribution_audit.json"
    md_path = output_dir / "data_distribution_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "decisions": total_decisions}, ensure_ascii=False), flush=True)
    if malformed:
        raise SystemExit("replay distribution audit found malformed files")


if __name__ == "__main__":
    main()
