#!/usr/bin/env python3
"""Run the read-only D3 uncertainty-guided exploration feasibility audit.

The audit compares the retained M0 and D1 replay indexes under the same K0
parent Q function.  It measures legal top-1/top-2 margins, the parent-Q gap
incurred by selecting the second legal action, semantic action crossings, and
coverage under diagnostic margin thresholds.  It does not generate logs,
modify indexes, or start training.  The measured thresholds are descriptive
bins only; they are not a D3 sampling contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import sys
import tomllib
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.audit_replay_distribution import (  # noqa: E402
    ACTION_NAMES,
    DecisionRecord,
    load_checkpoint,
    load_file_list,
    load_model,
    model_q,
    records_from_game,
    sha256_file,
)


DIAGNOSTIC_THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 4.0)
MARGIN_BINS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-file-index", type=Path, required=True)
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--m0-model-label", default="ext_mortal")
    parser.add_argument("--d1-file-index", type=Path, required=True)
    parser.add_argument("--d1-config", type=Path, required=True)
    parser.add_argument("--d1-model-label", default="K0_70k")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--file-batch-size", type=int, default=15)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def action_semantic(action: int) -> str:
    if action < 37:
        return "discard"
    if action == 37:
        return "reach"
    if 38 <= action <= 40:
        return "chi"
    if action == 41:
        return "pon"
    if action == 42:
        return "kan"
    if action == 43:
        return "agari"
    if action == 44:
        return "ryukyoku"
    if action == 45:
        return "pass"
    return "other"


def action_kind(action: int) -> str:
    return ACTION_NAMES.get(action, "other")


def legal_count_bucket(count: int) -> str:
    if count <= 5:
        return "1_5"
    if count <= 10:
        return "6_10"
    return "11_plus"


def phase_bucket(kyoku: int) -> str:
    if kyoku < 4:
        return "early"
    if kyoku < 8:
        return "middle"
    return "late"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    reward_mode = str(config.get("reward", {}).get("mode", "final_rank_mc"))
    objective_mode = str(config.get("objective", {}).get("mode", "behavior_action_mc"))
    if reward_mode != "final_rank_mc":
        raise ValueError(f"D3 feasibility audit requires final_rank_mc, got {reward_mode!r}: {path}")
    if objective_mode != "behavior_action_mc":
        raise ValueError(f"D3 feasibility audit expects behavior_action_mc, got {objective_mode!r}: {path}")
    version = int(config["control"]["version"])
    pts = np.asarray(config["env"].get("pts", [6.0, 4.0, 2.0, 0.0]), dtype=np.float64)
    if pts.shape != (4,):
        raise ValueError(f"expected four rank-point values, got {pts}: {path}")
    return {"config": config, "version": version, "pts": pts, "reward_mode": reward_mode, "objective_mode": objective_mode}


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def margin_bin(value: float) -> str:
    for lower, upper in zip(MARGIN_BINS, MARGIN_BINS[1:]):
        if value < upper:
            return f"[{lower:g},{upper:g})"
    return f"[{MARGIN_BINS[-1]:g},inf)"


class MarginAccumulator:
    def __init__(self) -> None:
        self.states = 0
        self.two_action_states = 0
        self.single_action_states = 0
        self.zero_finite_legal_action_states = 0
        self.nonfinite_q_states = 0
        self.nonfinite_legal_q_values = 0
        self.behavior_action_illegal_count = 0
        self.margins = RunningStats()
        self.histogram: Counter[str] = Counter()
        self.threshold_counts: Counter[str] = Counter()
        self.transition_counts: Counter[str] = Counter()
        self.top1_semantics: Counter[str] = Counter()
        self.top2_semantics: Counter[str] = Counter()
        self.greedy_agreement = 0
        self.behavior_legal = 0
        self.behavior_regret = RunningStats()

    def update(self, record: DecisionRecord, q: np.ndarray) -> None:
        self.states += 1
        mask = record.mask.astype(bool)
        if q.shape != mask.shape:
            raise ValueError(f"Q/mask shape mismatch: q={q.shape} mask={mask.shape}")
        finite_legal = mask & np.isfinite(q)
        finite_count = int(finite_legal.sum())
        nonfinite_count = int((mask & ~np.isfinite(q)).sum())
        self.nonfinite_legal_q_values += nonfinite_count
        self.nonfinite_q_states += int(nonfinite_count > 0)
        if finite_count == 0:
            self.zero_finite_legal_action_states += 1
        elif finite_count == 1:
            self.single_action_states += 1
        else:
            self.two_action_states += 1

        behavior_is_legal = 0 <= record.action < len(finite_legal) and bool(finite_legal[record.action])
        if behavior_is_legal:
            self.behavior_legal += 1
        else:
            self.behavior_action_illegal_count += 1
        if finite_count == 0:
            return

        legal_actions = np.flatnonzero(finite_legal)
        legal_q = q[legal_actions]
        order = np.argsort(-legal_q, kind="stable")
        top1_action = int(legal_actions[order[0]])
        top1_q = float(legal_q[order[0]])
        top1_semantic = action_semantic(top1_action)
        self.top1_semantics[top1_semantic] += 1
        if finite_count >= 2:
            top2_action = int(legal_actions[order[1]])
            margin = top1_q - float(legal_q[order[1]])
            self.margins.update(margin)
            self.histogram[margin_bin(margin)] += 1
            for threshold in DIAGNOSTIC_THRESHOLDS:
                if margin <= threshold:
                    self.threshold_counts[str(threshold)] += 1
            top2_semantic = action_semantic(top2_action)
            self.top2_semantics[top2_semantic] += 1
            self.transition_counts[f"{top1_semantic}->{top2_semantic}"] += 1
        if behavior_is_legal:
            behavior_q = float(q[record.action])
            self.greedy_agreement += int(top1_action == record.action)
            self.behavior_regret.update(top1_q - behavior_q)

    def to_json(self) -> dict[str, Any]:
        two = self.two_action_states
        states = self.states
        thresholds = {
            str(threshold): {
                "count": int(self.threshold_counts[str(threshold)]),
                "rate_all_states": self.threshold_counts[str(threshold)] / states if states else 0.0,
                "rate_two_action_states": self.threshold_counts[str(threshold)] / two if two else 0.0,
            }
            for threshold in DIAGNOSTIC_THRESHOLDS
        }
        same_semantic = sum(
            count
            for transition, count in self.transition_counts.items()
            if transition.split("->", maxsplit=1)[0] == transition.split("->", maxsplit=1)[1]
        )
        return {
            "states": states,
            "behavior_action_legal_count": self.behavior_legal,
            "behavior_action_legal_rate": self.behavior_legal / states if states else 0.0,
            "greedy_agreement_rate": self.greedy_agreement / self.behavior_legal if self.behavior_legal else 0.0,
            "single_legal_action_states": self.single_action_states,
            "two_or_more_legal_action_states": two,
            "zero_finite_legal_action_states": self.zero_finite_legal_action_states,
            "nonfinite_q_states": self.nonfinite_q_states,
            "nonfinite_legal_q_values": self.nonfinite_legal_q_values,
            "behavior_action_illegal_count": self.behavior_action_illegal_count,
            "two_or_more_legal_action_rate": two / states if states else 0.0,
            "top1_top2_margin": self.margins.summary(),
            "second_action_parent_q_regret": self.margins.summary(),
            "margin_histogram": {key: int(value) for key, value in sorted(self.histogram.items())},
            "diagnostic_margin_thresholds": thresholds,
            "top1_action_semantic_counts": dict(sorted(self.top1_semantics.items())),
            "top2_action_semantic_counts": dict(sorted(self.top2_semantics.items())),
            "top1_to_top2_semantic_transitions": dict(sorted(self.transition_counts.items())),
            "same_semantic_top2_rate": same_semantic / two if two else 0.0,
            "cross_semantic_top2_rate": (two - same_semantic) / two if two else 0.0,
            "discard_to_discard_top2_rate": self.transition_counts["discard->discard"] / two if two else 0.0,
            "behavior_q_regret_greedy_minus_behavior": self.behavior_regret.summary(),
        }


def hard_finite_checks(summary: dict[str, Any]) -> dict[str, bool]:
    states = int(summary["states"])
    return {
        "behavior_action_legal_count_equals_states": int(summary["behavior_action_legal_count"]) == states,
        "finite_legal_action_partition_covers_states": (
            int(summary["single_legal_action_states"])
            + int(summary["two_or_more_legal_action_states"])
            == states
        ),
        "zero_finite_legal_action_states_is_zero": int(summary["zero_finite_legal_action_states"]) == 0,
        "nonfinite_legal_q_values_is_zero": int(summary["nonfinite_legal_q_values"]) == 0,
    }


def counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def stratum_key(record: DecisionRecord) -> str:
    return json.dumps(
        {
            "phase": record.phase,
            "behavior_action_kind": action_kind(record.action),
            "behavior_action_semantic": action_semantic(record.action),
            "own_riichi": record.own_riichi,
            "legal_actions": legal_count_bucket(record.legal_actions),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_report(
    *,
    name: str,
    file_index: Path,
    config_path: Path,
    model_label: str,
    files: list[Path],
    config_meta: dict[str, Any],
    margin_overall: MarginAccumulator,
    margin_strata: dict[str, MarginAccumulator],
    phase_counts: Counter[str],
    action_kind_counts: Counter[str],
    action_semantic_counts: Counter[str],
    own_riichi_counts: Counter[str],
    legal_counts: Counter[str],
    final_rank_counts: Counter[int],
    decision_counts: list[int],
    decisions: int,
    malformed: list[dict[str, str]],
) -> dict[str, Any]:
    values = np.asarray(decision_counts, dtype=np.float64)
    overall = margin_overall.to_json()
    return {
        "name": name,
        "inputs": {
            "file_index": str(file_index),
            "file_index_sha256": sha256_file(file_index),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "model_label": model_label,
            "version": config_meta["version"],
            "pts": config_meta["pts"].tolist(),
            "reward_mode": config_meta["reward_mode"],
            "objective_mode": config_meta["objective_mode"],
        },
        "corpus": {
            "files_selected": len(files),
            "trainable_perspectives": len(decision_counts),
            "total_decisions": decisions,
            "malformed_count": len(malformed),
            "malformed": malformed[:50],
            "final_rank_counts": counter_json(final_rank_counts),
            "decision_count_quantiles": {
                str(quantile): float(np.quantile(values, quantile)) if len(values) else 0.0
                for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
        },
        "decision_distribution": {
            "phase_counts": counter_json(phase_counts),
            "behavior_action_kind_counts": counter_json(action_kind_counts),
            "behavior_action_semantic_counts": counter_json(action_semantic_counts),
            "own_riichi_counts": counter_json(own_riichi_counts),
            "legal_action_count_buckets": counter_json(legal_counts),
        },
        "top2_feasibility": {
            "dimensions": ["phase", "behavior_action_kind", "behavior_action_semantic", "own_riichi", "legal_actions"],
            "overall": overall,
            "hard_finite_checks": hard_finite_checks(overall),
            "by_stratum": {key: value.to_json() for key, value in sorted(margin_strata.items())},
        },
        "scope_notes": [
            "analysis only; no logs, indexes, checkpoints, or training state were modified",
            "diagnostic margin thresholds are measurement bins, not frozen D3 sampling parameters",
            "parent-Q regret means Q(top-1 legal action) - Q(top-2 legal action), not ground-truth regret",
            "opponent_riichi is omitted because GameplayLoader does not expose a reliable per-decision global field",
        ],
    }


def audit_source(
    *,
    name: str,
    file_index: Path,
    config_path: Path,
    model_label: str,
    parent: tuple[torch.nn.Module, torch.nn.Module, int],
    device: torch.device,
    q_batch_size: int,
    file_batch_size: int,
    max_files: int,
    progress_every: int,
) -> dict[str, Any]:
    config_meta = load_config(config_path)
    files = load_file_list(file_index)
    if max_files:
        files = files[:max_files]
    if not files:
        raise ValueError(f"{name} selected no files")
    brain, dqn, model_version = parent
    if model_version != config_meta["version"]:
        raise ValueError(f"parent version {model_version} differs from {name} config version {config_meta['version']}")

    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(
        version=config_meta["version"],
        oracle=False,
        player_names=[model_label],
        augmented=False,
    )
    margin_overall = MarginAccumulator()
    margin_strata: dict[str, MarginAccumulator] = {}
    phase_counts: Counter[str] = Counter()
    action_kind_counts: Counter[str] = Counter()
    action_semantic_counts: Counter[str] = Counter()
    own_riichi_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    final_rank_counts: Counter[int] = Counter()
    decision_counts: list[int] = []
    malformed: list[dict[str, str]] = []
    total_decisions = 0
    processed_files = 0

    def consume(file_path: Path, loaded_games: list[Any]) -> int:
        if len(loaded_games) != 1:
            raise ValueError(f"expected one {model_label} perspective, got {len(loaded_games)}")
        records = list(records_from_game(loaded_games[0], config_meta["pts"]))
        if not records:
            raise ValueError("perspective contains no decisions")
        q_records: list[tuple[DecisionRecord, np.ndarray]] = []
        for start in range(0, len(records), q_batch_size):
            batch = records[start : start + q_batch_size]
            q_values = model_q(brain, dqn, batch, device)
            q_records.extend(zip(batch, q_values, strict=True))
        for record, q in q_records:
            phase_counts[record.phase] += 1
            action_kind_counts[action_kind(record.action)] += 1
            action_semantic_counts[action_semantic(record.action)] += 1
            own_riichi_counts[str(record.own_riichi).lower()] += 1
            legal_counts[legal_count_bucket(record.legal_actions)] += 1
            margin_overall.update(record, q)
            margin_strata.setdefault(stratum_key(record), MarginAccumulator()).update(record, q)
        final_rank_counts[records[0].target_rank] += 1
        decision_counts.append(len(records))
        return len(records)

    for batch_start in range(0, len(files), file_batch_size):
        file_batch = files[batch_start : batch_start + file_batch_size]
        try:
            loaded_batch = loader.load_gz_log_files([str(path) for path in file_batch])
            if len(loaded_batch) != len(file_batch):
                raise ValueError(f"loader returned {len(loaded_batch)} files for {len(file_batch)} paths")
            for file_path, loaded_games in zip(file_batch, loaded_batch, strict=True):
                try:
                    total_decisions += consume(file_path, loaded_games)
                except Exception as exc:  # noqa: BLE001
                    malformed.append({"path": str(file_path), "error": str(exc)})
                processed_files += 1
        except Exception as batch_exc:  # noqa: BLE001
            for file_path in file_batch:
                try:
                    loaded = loader.load_gz_log_files([str(file_path)])
                    if len(loaded) != 1:
                        raise ValueError(f"loader returned {len(loaded)} files")
                    total_decisions += consume(file_path, loaded[0])
                except Exception as exc:  # noqa: BLE001
                    malformed.append({"path": str(file_path), "error": f"batch={batch_exc}; single={exc}"})
                processed_files += 1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress_every and processed_files % progress_every == 0:
            print(
                f"[d3-feasibility] {name} files {processed_files}/{len(files)} "
                f"decisions={total_decisions} malformed={len(malformed)}",
                flush=True,
            )

    report = source_report(
        name=name,
        file_index=file_index.resolve(),
        config_path=config_path.resolve(),
        model_label=model_label,
        files=files,
        config_meta=config_meta,
        margin_overall=margin_overall,
        margin_strata=margin_strata,
        phase_counts=phase_counts,
        action_kind_counts=action_kind_counts,
        action_semantic_counts=action_semantic_counts,
        own_riichi_counts=own_riichi_counts,
        legal_counts=legal_counts,
        final_rank_counts=final_rank_counts,
        decision_counts=decision_counts,
        decisions=total_decisions,
        malformed=malformed,
    )
    report["audit_passed"] = (
        not malformed
        and len(decision_counts) == len(files)
        and all(report["top2_feasibility"]["hard_finite_checks"].values())
    )
    return report


def comparison_report(sources: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {source["name"]: source for source in sources}
    if not {"M0", "D1"}.issubset(by_name):
        return {}
    metrics = {
        "total_decisions": lambda source: source["corpus"]["total_decisions"],
        "greedy_agreement_rate": lambda source: source["top2_feasibility"]["overall"]["greedy_agreement_rate"],
        "two_or_more_legal_action_rate": lambda source: source["top2_feasibility"]["overall"]["two_or_more_legal_action_rate"],
        "mean_top1_top2_margin": lambda source: source["top2_feasibility"]["overall"]["top1_top2_margin"]["mean"],
        "mean_second_action_parent_q_regret": lambda source: source["top2_feasibility"]["overall"]["second_action_parent_q_regret"]["mean"],
        "cross_semantic_top2_rate": lambda source: source["top2_feasibility"]["overall"]["cross_semantic_top2_rate"],
        "discard_to_discard_top2_rate": lambda source: source["top2_feasibility"]["overall"]["discard_to_discard_top2_rate"],
    }
    result = {"M0": {}, "D1": {}, "D1_minus_M0": {}}
    for name in ("M0", "D1"):
        result[name] = {metric: getter(by_name[name]) for metric, getter in metrics.items()}
    for metric in metrics:
        left = result["D1"][metric]
        right = result["M0"][metric]
        result["D1_minus_M0"][metric] = left - right if left is not None and right is not None else None
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# D3 uncertainty-guided exploration feasibility audit",
        "",
        "> 只读审计：不生成牌谱、不修改 file index、不训练。下面的 margin threshold 只是观测分桶，不是已经冻结的 D3 参数。",
        "",
        "## Corpus summary",
        "",
        "| Source | Files | Decisions | 70k agreement | Top-2 eligible | Mean top-1/top-2 margin | Cross-semantic top-2 | Malformed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in report["sources"]:
        overall = source["top2_feasibility"]["overall"]
        lines.append(
            f"| {source['name']} | {source['corpus']['files_selected']} | {source['corpus']['total_decisions']} | "
            f"{overall['greedy_agreement_rate']:.4%} | {overall['two_or_more_legal_action_rate']:.4%} | "
            f"{overall['top1_top2_margin']['mean']:.6g} | {overall['cross_semantic_top2_rate']:.4%} | "
            f"{source['corpus']['malformed_count']} |"
        )
        checks = source["top2_feasibility"]["hard_finite_checks"]
        lines.append(f"| {source['name']} hard checks | {'PASS' if all(checks.values()) else 'FAIL'} |  |  |  |  |  |  |")
    lines.extend(["", "## Diagnostic margin coverage", "", "The denominator is all decisions; the conditional column is among states with at least two finite legal Q values.", ""])
    for source in report["sources"]:
        overall = source["top2_feasibility"]["overall"]
        lines.extend([f"### {source['name']}", "", "| Margin <= | Count | All decisions | Top-2 eligible states |", "|---:|---:|---:|---:|"])
        for threshold, values in overall["diagnostic_margin_thresholds"].items():
            lines.append(f"| {threshold} | {values['count']} | {values['rate_all_states']:.4%} | {values['rate_two_action_states']:.4%} |")
        lines.extend([
            "",
            f"- Top-1/top-2 margin mean/std: `{overall['top1_top2_margin']['mean']}` / `{overall['top1_top2_margin']['std']}`.",
            f"- Top-1/top-2 margin range: `{overall['top1_top2_margin']['min']}` to `{overall['top1_top2_margin']['max']}`.",
            f"- Same-semantic top-2 rate: `{overall['same_semantic_top2_rate']:.4%}`; cross-semantic rate: `{overall['cross_semantic_top2_rate']:.4%}`.",
            f"- Discard-to-discard top-2 rate: `{overall['discard_to_discard_top2_rate']:.4%}`.",
            f"- Mean parent-Q gap for top-2: `{overall['second_action_parent_q_regret']['mean']}`.",
            "",
            "Top-1 to top-2 semantic transitions:",
            "",
        ])
        for transition, count in overall["top1_to_top2_semantic_transitions"].items():
            lines.append(f"- `{transition}`: `{count}`")
        lines.append("")
    lines.extend([
        "## Comparison",
        "",
        "The comparison is descriptive and is not a training result.",
        "",
        "```json",
        json.dumps(report["comparison"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Stratified view",
        "",
        "Each source also contains `top2_feasibility.by_stratum`, keyed by phase, observed behavior action kind/semantic, own-riichi state, and legal-action-count bucket.",
        "Opponent-riichi is omitted because the current GameplayLoader does not expose a reliable per-decision global field.",
        "",
        "## Scope and next gate",
        "",
        "- This audit does not choose a margin threshold, exploration probability, or candidate-action policy.",
        "- It does not inspect later training/evaluation outcomes to select a rule.",
        "- A D3 generation contract may be written only after these feasibility facts are reviewed and the trigger, candidate set, sampler, budget, and stop conditions are separately pre-registered.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
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

    m0_meta = load_config(args.m0_config.resolve())
    d1_meta = load_config(args.d1_config.resolve())
    if m0_meta["version"] != d1_meta["version"] or not np.array_equal(m0_meta["pts"], d1_meta["pts"]):
        raise ValueError("M0/D1 configs have different model version or rank-point values")
    parent_path = args.parent.resolve()
    state = load_checkpoint(parent_path)
    parent = load_model(state, device)
    del state

    sources = [
        audit_source(
            name="M0",
            file_index=args.m0_file_index.resolve(),
            config_path=args.m0_config.resolve(),
            model_label=args.m0_model_label,
            parent=parent,
            device=device,
            q_batch_size=args.q_batch_size,
            file_batch_size=args.file_batch_size,
            max_files=args.max_files,
            progress_every=args.progress_every,
        ),
        audit_source(
            name="D1",
            file_index=args.d1_file_index.resolve(),
            config_path=args.d1_config.resolve(),
            model_label=args.d1_model_label,
            parent=parent,
            device=device,
            q_batch_size=args.q_batch_size,
            file_batch_size=args.file_batch_size,
            max_files=args.max_files,
            progress_every=args.progress_every,
        ),
    ]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "schema": "keqing.mortal.d3_exploration_feasibility.v1",
        "status": "passed" if all(source["audit_passed"] for source in sources) else "failed",
        "analysis_only": True,
        "inputs": {
            "parent": str(parent_path),
            "parent_sha256": sha256_file(parent_path),
            "device": str(device),
            "require_cuda": bool(args.require_cuda),
            "q_batch_size": int(args.q_batch_size),
            "file_batch_size": int(args.file_batch_size),
            "max_files": int(args.max_files),
            "diagnostic_thresholds": list(DIAGNOSTIC_THRESHOLDS),
            "margin_bins": [*MARGIN_BINS, "inf"],
        },
        "sources": sources,
        "comparison": comparison_report(sources),
        "scope_notes": [
            "M0 and D1 are audited as separate replay lineages; this is not a paired training experiment",
            "parent-Q regret is an internal support diagnostic, not a correctness label",
            "opponent-riichi is not reported because the loader lacks a reliable per-decision global field",
            "no D3 generation or training parameter is frozen by this report",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "d3_exploration_feasibility.json"
    markdown_path = output_dir / "d3_exploration_feasibility.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(f"[d3-feasibility] status={report['status']} json={json_path} markdown={markdown_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
