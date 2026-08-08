#!/usr/bin/env python3
"""Aggregate the fixed 24-shard M0/D1 distribution audits.

The per-shard auditor keeps exact decision and target buckets.  This script
merges those reports without rescanning the replay corpus.  Duplicate counts
remain explicitly shard-local because the shard auditor does not retain a
global decision-hash set.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--m0-outcomes", type=Path, required=True)
    parser.add_argument("--d1-outcomes", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def counter_add(target: Counter[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def shard_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(root.glob("shard_*/data_distribution_audit.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if len(reports) != 24:
        raise ValueError(f"expected 24 shard reports under {root}, found {len(reports)}")
    for index, report in enumerate(reports):
        corpus = report["corpus"]
        if corpus["files_selected"] != 250 or corpus["hanchans"] != 250:
            raise ValueError(f"shard {index:02d} does not contain exactly 250 hanchans")
        if corpus["trainable_perspectives"] != 250 or corpus["malformed_count"] != 0:
            raise ValueError(f"shard {index:02d} failed trainable/malformed checks")
    return reports


def merge_bucket(target: dict[str, dict[str, Any]], buckets: dict[str, Any]) -> None:
    for key, bucket in buckets.items():
        existing = target.setdefault(
            key,
            {
                "count": 0,
                "target_sum": 0.0,
                "target_sum_sq": 0.0,
                "target_counts": Counter(),
                "action_counts": Counter(),
            },
        )
        count = int(bucket["count"])
        mean = float(bucket["target_mean"])
        std = float(bucket["target_std"])
        existing["count"] += count
        existing["target_sum"] += count * mean
        existing["target_sum_sq"] += count * (std * std + mean * mean)
        counter_add(existing["target_counts"], bucket.get("target_counts", {}))
        counter_add(existing["action_counts"], bucket.get("action_counts", {}))


def finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, bucket in sorted(buckets.items()):
        count = bucket["count"]
        mean = bucket["target_sum"] / count if count else 0.0
        variance = max(0.0, bucket["target_sum_sq"] / count - mean * mean) if count else 0.0
        result[key] = {
            "count": count,
            "target_mean": mean,
            "target_std": math.sqrt(variance),
            "target_counts": dict(sorted(bucket["target_counts"].items())),
            "action_counts": dict(sorted(bucket["action_counts"].items())),
        }
    return result


def aggregate(label: str, root: Path, outcome_path: Path) -> dict[str, Any]:
    reports = shard_reports(root)
    corpus = {
        "files_selected": 0,
        "hanchans": 0,
        "trainable_perspectives": 0,
        "total_decisions": 0,
        "malformed_count": 0,
        "final_rank_counts": Counter(),
        "target_counts": Counter(),
    }
    decision_distribution = {
        name: Counter()
        for name in (
            "phase_counts",
            "current_rank_counts",
            "score_gap_counts",
            "own_riichi_counts",
            "action_counts",
            "legal_action_count_buckets",
            "shanten_buckets",
        )
    }
    contribution_decisions: list[int] = []
    support_sums = Counter()
    support_rank_counts = Counter()
    support_states = 0
    bucket_accumulator: dict[str, dict[str, Any]] = {}
    shard_duplicate_rows = []

    for report in reports:
        report_corpus = report["corpus"]
        for key in ("files_selected", "hanchans", "trainable_perspectives", "total_decisions", "malformed_count"):
            corpus[key] += int(report_corpus[key])
        counter_add(corpus["final_rank_counts"], report_corpus["final_rank_counts"])
        counter_add(corpus["target_counts"], report_corpus["target_counts"])
        for name in decision_distribution:
            counter_add(decision_distribution[name], report["decision_distribution"][name])
        contribution_decisions.extend(int(row["decisions"]) for row in report["hanchans"])
        support = report["support_audit"]["overall"]
        states = int(support["states"])
        support_states += states
        for key in (
            "behavior_action_legal_rate",
            "greedy_agreement_rate",
            "greedy_disagreement_rate",
            "mean_behavior_q_rank",
            "mean_q_regret_greedy_minus_behavior",
            "mean_greedy_margin",
            "mean_behavior_q",
            "mean_greedy_q",
            "mean_legal_q_abs",
        ):
            support_sums[key] += states * float(support[key])
        counter_add(support_rank_counts, support["behavior_q_rank_counts"])
        duplicate = report["duplicates"]
        shard_duplicate_rows.append(
            {
                "unique_decision_count": int(duplicate["unique_decision_count"]),
                "duplicate_decision_count": int(duplicate["duplicate_decision_count"]),
                "unique_state_count": int(duplicate["unique_state_count"]),
                "state_duplicate_count": int(duplicate["state_duplicate_count"]),
            }
        )
        merge_bucket(bucket_accumulator, report["target_conflict"]["buckets"])

    support = {
        "states": support_states,
        "behavior_action_legal_rate": support_sums["behavior_action_legal_rate"] / support_states,
        "greedy_agreement_rate": support_sums["greedy_agreement_rate"] / support_states,
        "greedy_disagreement_rate": support_sums["greedy_disagreement_rate"] / support_states,
        "behavior_q_rank_counts": dict(sorted(support_rank_counts.items())),
    }
    for key in (
        "mean_behavior_q_rank",
        "mean_q_regret_greedy_minus_behavior",
        "mean_greedy_margin",
        "mean_behavior_q",
        "mean_greedy_q",
        "mean_legal_q_abs",
    ):
        support[key] = support_sums[key] / support_states

    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    if outcomes["hanchans"] != 6000 or outcomes["malformed_count"] != 0:
        raise ValueError(f"invalid outcome audit for {label}: {outcome_path}")
    report = {
        "schema": "keqing.mortal.d1_distribution_summary.v1",
        "label": label,
        "shards": 24,
        "corpus": {
            **corpus,
            "final_rank_counts": dict(sorted(corpus["final_rank_counts"].items())),
            "target_counts": dict(sorted(corpus["target_counts"].items())),
        },
        "hanchan_contribution": {
            "hanchans": len(contribution_decisions),
            "total_decisions": sum(contribution_decisions),
            "decisions_per_hanchan_min": min(contribution_decisions),
            "decisions_per_hanchan_max": max(contribution_decisions),
            "decisions_per_hanchan_mean": sum(contribution_decisions) / len(contribution_decisions),
            "decision_weight_ess": sum(contribution_decisions) ** 2 / sum(value * value for value in contribution_decisions),
        },
        "decision_distribution": {name: dict(sorted(values.items())) for name, values in decision_distribution.items()},
        "support_audit_overall": support,
        "target_conflict_buckets": finalize_buckets(bucket_accumulator),
        "duplicates": {
            "scope": "shard_local_only",
            "note": "Shard-local duplicate counts are not a global duplicate audit.",
            "sum_unique_decisions": sum(row["unique_decision_count"] for row in shard_duplicate_rows),
            "sum_duplicate_decisions": sum(row["duplicate_decision_count"] for row in shard_duplicate_rows),
            "sum_unique_states": sum(row["unique_state_count"] for row in shard_duplicate_rows),
            "sum_state_duplicates": sum(row["state_duplicate_count"] for row in shard_duplicate_rows),
        },
        "outcomes": outcomes,
    }
    return report


def markdown(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    support = summary["support_audit_overall"]
    contribution = summary["hanchan_contribution"]
    outcomes = summary["outcomes"]
    counts = outcomes["raw_event_counts"]
    lines = [
        f"# {summary['label']} D1 Distribution Summary",
        "",
        "Aggregated from 24 fixed 250-hanchan audit shards. This is an analysis artifact; no logs or checkpoints were modified.",
        "",
        "## Corpus",
        "",
        f"- Files/hanchans: `{corpus['files_selected']}` / `{corpus['hanchans']}`; trainable perspectives: `{corpus['trainable_perspectives']}`.",
        f"- Total decisions: `{corpus['total_decisions']}`; malformed: `{corpus['malformed_count']}`.",
        f"- Final rank counts: `{corpus['final_rank_counts']}`; target counts: `{corpus['target_counts']}`.",
        f"- Decisions/hanchan: mean `{contribution['decisions_per_hanchan_mean']:.2f}`, min/max `{contribution['decisions_per_hanchan_min']}/{contribution['decisions_per_hanchan_max']}`, decision ESS `{contribution['decision_weight_ess']:.1f}`.",
        "",
        "## Parent Support",
        "",
        f"- Behavior action legal rate: `{support['behavior_action_legal_rate']:.4%}`; 70k greedy agreement: `{support['greedy_agreement_rate']:.4%}`.",
        f"- Mean behavior-Q rank: `{support['mean_behavior_q_rank']:.4f}`; mean greedy-minus-behavior Q regret: `{support['mean_q_regret_greedy_minus_behavior']:.6g}`.",
        f"- Mean greedy margin: `{support['mean_greedy_margin']:.6g}`; mean behavior Q: `{support['mean_behavior_q']:.6g}`.",
        "",
        "## Raw Trainable-View Outcomes",
        "",
        f"- Agari/houjuu/fuuro/riichi events: `{counts['agari']}` / `{counts['houjuu']}` / `{counts['fuuro']}` / `{counts['riichi']}`.",
        f"- Hanchans with agari/houjuu/fuuro/riichi: `{outcomes['hanchans_with_event']}`.",
        f"- Reconstructed final ranks: `{outcomes['final_rank_counts']}`.",
        "",
        "## Duplicate Scope",
        "",
        "- The duplicate totals below are sums of shard-local audits, not a global decision-hash result.",
        f"- Shard-local duplicate decisions: `{summary['duplicates']['sum_duplicate_decisions']}`; shard-local state duplicates: `{summary['duplicates']['sum_state_duplicates']}`.",
        "",
        "## Interpretation",
        "",
        "- Use M0 versus D1 differences as data-lineage diagnostics, not as a strength result.",
        "- The six training configs remain blocked only by the final preflight contract; this summary does not select a checkpoint or training seed.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for label, root, outcomes in (
        ("M0", args.m0_root, args.m0_outcomes),
        ("D1", args.d1_root, args.d1_outcomes),
    ):
        report = aggregate(label, root, outcomes)
        json_path = args.output_prefix.with_name(f"{args.output_prefix.name}_{label.lower()}.json")
        md_path = args.output_prefix.with_name(f"{args.output_prefix.name}_{label.lower()}.md")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"label": label, "json": str(json_path), "markdown": str(md_path), "decisions": report["corpus"]["total_decisions"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
