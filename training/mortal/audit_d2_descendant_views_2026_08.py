#!/usr/bin/env python3
"""Audit the V2/V3 descendant-view mix before D2 training.

This orchestrator runs the existing raw-event and parent-Q auditors on the two
fixed 3,000-hanchan indexes, then writes a weighted 50/50 D2 summary.  It does
not alter replay logs, checkpoints, or training state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D2_project_owned_descendant_view_mix_2026_08"
DEFAULT_PARENT = REPO_ROOT / "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
DEFAULT_CONFIG = DEFAULT_ROOT / "training_prep_2026_08/D2_variant/seed_20260806/config.toml"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_list(path: Path) -> list[str]:
    payload = torch.load(path.resolve(), weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"file index has no file_list: {path}")
    files = [
        str((Path(str(value)) if Path(str(value)).is_absolute() else REPO_ROOT / str(value)).resolve())
        for value in values
    ]
    if len(files) != len(set(files)) or any(not Path(value).is_file() for value in files):
        raise ValueError(f"file index contains duplicate or missing files: {path}")
    return files


def counter_add(target: Counter[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def weighted_support(reports: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "behavior_action_legal_rate",
        "greedy_agreement_rate",
        "greedy_disagreement_rate",
        "mean_behavior_q_rank",
        "mean_q_regret_greedy_minus_behavior",
        "mean_greedy_margin",
        "mean_behavior_q",
        "mean_greedy_q",
        "mean_legal_q_abs",
    )
    sums = Counter()
    rank_counts = Counter()
    states = 0
    for report in reports:
        support = report["support_audit"]["overall"]
        count = int(support["states"])
        states += count
        for field in fields:
            sums[field] += count * float(support[field])
        counter_add(rank_counts, support["behavior_q_rank_counts"])
    return {"states": states, **{field: sums[field] / states for field in fields}, "behavior_q_rank_counts": dict(sorted(rank_counts.items()))}


def weighted_corpus(reports: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = {key: 0 for key in ("files_selected", "hanchans", "trainable_perspectives", "total_decisions", "malformed_count")}
    final_ranks = Counter()
    targets = Counter()
    for report in reports:
        source = report["corpus"]
        for key in corpus:
            corpus[key] += int(source[key])
        counter_add(final_ranks, source["final_rank_counts"])
        counter_add(targets, source["target_counts"])
    corpus["final_rank_counts"] = dict(sorted(final_ranks.items()))
    corpus["target_counts"] = dict(sorted(targets.items()))
    return corpus


def weighted_decisions(reports: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = (
        "phase_counts",
        "current_rank_counts",
        "score_gap_counts",
        "own_riichi_counts",
        "action_counts",
        "legal_action_count_buckets",
        "shanten_buckets",
    )
    result = {}
    for name in names:
        values = Counter()
        for report in reports:
            counter_add(values, report["decision_distribution"][name])
        result[name] = dict(sorted(values.items()))
    return result


def combine_outcomes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    event_keys = ("kyoku", "agari", "houjuu", "fuuro", "riichi", "ryukyoku", "dahai", "tsumo", "reach_accepted")
    events = Counter()
    hanchans_with = Counter()
    ranks = Counter()
    malformed = 0
    hanchans = 0
    decisions = 0
    for report in reports:
        hanchans += int(report["hanchans"])
        malformed += int(report["malformed_count"])
        decisions += int(report["decision_like_event_count"])
        counter_add(events, report["raw_event_counts"])
        counter_add(hanchans_with, report["hanchans_with_event"])
        counter_add(ranks, report["final_rank_counts"])
    rates = {f"{key}_per_hanchan": events[key] / hanchans for key in ("agari", "houjuu", "fuuro", "riichi")}
    rates.update({f"{key}_hanchan_rate": hanchans_with[key] / hanchans for key in ("agari", "houjuu", "fuuro", "riichi")})
    return {
        "hanchans": hanchans,
        "raw_event_counts": {key: int(events[key]) for key in event_keys},
        "hanchans_with_event": {key: int(hanchans_with[key]) for key in ("agari", "houjuu", "fuuro", "riichi")},
        "rates": rates,
        "final_rank_counts": {str(key): int(value) for key, value in sorted(ranks.items())},
        "decision_like_event_count": decisions,
        "malformed_count": malformed,
    }


def write_index(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"file_list": [str(value) for value in files]}, path)


def aggregate_q_shards(shard_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard audits while keeping memory bounded per child process."""

    corpus = {key: 0 for key in ("files_selected", "hanchans", "trainable_perspectives", "total_decisions", "malformed_count")}
    final_ranks = Counter()
    targets = Counter()
    distributions = {
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
    support_fields = (
        "behavior_action_legal_rate",
        "greedy_agreement_rate",
        "greedy_disagreement_rate",
        "mean_behavior_q_rank",
        "mean_q_regret_greedy_minus_behavior",
        "mean_greedy_margin",
        "mean_behavior_q",
        "mean_greedy_q",
        "mean_legal_q_abs",
    )
    support_sums = Counter()
    support_rank_counts = Counter()
    hanchan_rows = []
    duplicate_totals = Counter()
    for report in shard_reports:
        source = report["corpus"]
        for key in corpus:
            corpus[key] += int(source[key])
        counter_add(final_ranks, source["final_rank_counts"])
        counter_add(targets, source["target_counts"])
        for name in distributions:
            counter_add(distributions[name], report["decision_distribution"][name])
        support = report["support_audit"]["overall"]
        states = int(support["states"])
        for field in support_fields:
            support_sums[field] += states * float(support[field])
        counter_add(support_rank_counts, support["behavior_q_rank_counts"])
        hanchan_rows.extend(report["hanchans"])
        duplicate = report["duplicates"]
        for key in ("unique_decision_count", "duplicate_decision_count", "unique_state_count", "state_duplicate_count"):
            duplicate_totals[key] += int(duplicate[key])
    states = sum(int(report["support_audit"]["overall"]["states"]) for report in shard_reports)
    support = {
        "states": states,
        "behavior_q_rank_counts": dict(sorted(support_rank_counts.items())),
        **{field: support_sums[field] / states for field in support_fields},
    }
    decisions = [int(row["decisions"]) for row in hanchan_rows]
    q_report = {
        "schema": "keqing.mortal.replay_distribution_audit.aggregate.v1",
        "corpus": {
            **corpus,
            "final_rank_counts": dict(sorted(final_ranks.items())),
            "target_counts": dict(sorted(targets.items())),
        },
        "hanchan_contribution": {
            "hanchans": len(decisions),
            "total_decisions": sum(decisions),
            "decisions_per_hanchan_min": min(decisions),
            "decisions_per_hanchan_max": max(decisions),
            "decisions_per_hanchan_mean": sum(decisions) / len(decisions),
            "decision_weight_ess": sum(decisions) ** 2 / sum(value * value for value in decisions),
        },
        "decision_distribution": {name: dict(sorted(values.items())) for name, values in distributions.items()},
        "support_audit": {"overall": support},
        "hanchans": hanchan_rows,
        "duplicates": {
            "scope": "shard_local_only",
            "sum_unique_decisions": duplicate_totals["unique_decision_count"],
            "sum_duplicate_decisions": duplicate_totals["duplicate_decision_count"],
            "sum_unique_states": duplicate_totals["unique_state_count"],
            "sum_state_duplicates": duplicate_totals["state_duplicate_count"],
        },
    }
    return q_report


def run_child(command: list[str]) -> None:
    print("[d2-audit] running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def markdown(report: dict[str, Any]) -> str:
    rows = report["views"]
    lines = [
        "# D2 Descendant-View Audit",
        "",
        "This is a pre-training audit of the fixed D2 50/50 V2/V3 view assignment. It does not select checkpoints or use outcomes for assignment.",
        "",
        "| view | files | decisions | 70k agreement | mean behavior-Q rank | mean Q regret | agari/houjuu/fuuro/riichi per hanchan | decision ESS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("K0_70k_reference", "V2_74000", "V3_74000", "D2_50_50"):
        row = rows[name]
        events = row["outcomes"]
        lines.append(
            f"| {name} | {row['files']} | {row['decisions']} | {row['greedy_agreement_rate']:.4%} | "
            f"{row['mean_behavior_q_rank']:.4f} | {row['mean_q_regret_greedy_minus_behavior']:.6g} | "
            f"{events['rates']['agari_per_hanchan']:.3f}/{events['rates']['houjuu_per_hanchan']:.3f}/"
            f"{events['rates']['fuuro_per_hanchan']:.3f}/{events['rates']['riichi_per_hanchan']:.3f} | "
            f"{row['decision_weight_ess']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Source hanchans: `{report['source']['source_file_count']}`; D2 assignment: `{report['assignment']['counts']}`.",
            f"- Single trainable perspective per file: `{report['assignment']['single_perspective_per_file']}`.",
            f"- Malformed V2/V3 reports: `{report['malformed_count']}`.",
            "- Agreement is a parent-support diagnostic, not a ground-truth quality score.",
            "- A high agreement result is not automatically a reason to continue training; inspect action support and Q regret together.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-q-audit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    parent = args.parent.resolve()
    config = args.config.resolve()
    prep = root / "training_prep_2026_08"
    dataset = root / "dataset"
    output = prep / "distribution_d2"
    output.mkdir(parents=True, exist_ok=True)
    indexes = {label: dataset / f"file_index_{suffix}.pth" for label, suffix in (("V2_74000", "v2"), ("V3_74000", "v3"))}
    if any(not path.is_file() for path in indexes.values()):
        raise FileNotFoundError("D2 V2/V3 file indexes are missing; run prepare_d2_descendant_view_mix_2026_08.py first")

    q_shard_reports: dict[str, list[dict[str, Any]]] = {}
    for label, index in indexes.items():
        outcome_path = output / f"outcomes_{label}.json"
        if not outcome_path.is_file():
            run_child(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/mortal/audit_trainable_view_outcomes_2026_07.py"),
                    "--file-index",
                    str(index),
                    "--model-label",
                    label,
                    "--output",
                    str(outcome_path),
                    "--progress-every",
                    "250",
                ]
            )
        if not args.skip_q_audit:
            selected_files = file_list(index)
            if len(selected_files) != 3000:
                raise ValueError(f"expected 3000 files for {label}, got {len(selected_files)}")
            shard_root = output / "q_shards" / label
            index_root = output / "q_indexes" / label
            shard_reports: list[dict[str, Any]] = []
            for shard_number in range(12):
                shard_files = [Path(value) for value in selected_files[shard_number * 250 : (shard_number + 1) * 250]]
                shard_index = index_root / f"file_index_{shard_number:02d}.pth"
                shard_output = shard_root / f"shard_{shard_number:02d}"
                write_index(shard_index, shard_files)
                shard_report_path = shard_output / "data_distribution_audit.json"
                if not shard_report_path.is_file():
                    run_child(
                        [
                            sys.executable,
                            str(REPO_ROOT / "scripts/mortal/audit_replay_distribution.py"),
                            "--file-index",
                            str(shard_index),
                            "--parent",
                            str(parent),
                            "--config",
                            str(config),
                            "--output-dir",
                            str(shard_output),
                            "--model-label",
                            label,
                            "--device",
                            "cuda",
                            "--require-cuda",
                            "--q-batch-size",
                            "4096",
                            "--file-batch-size",
                            "50",
                            "--progress-every",
                            "250",
                        ]
                    )
                shard_report = load_json(shard_report_path)
                if shard_report["corpus"]["files_selected"] != 250 or shard_report["corpus"]["malformed_count"] != 0:
                    raise SystemExit(f"{label} shard {shard_number:02d} failed Q audit")
                shard_reports.append(shard_report)
            aggregate = aggregate_q_shards(shard_reports)
            aggregate["inputs"] = {
                "model_label": label,
                "shards": 12,
                "files_per_shard": 250,
                "source_file_index": str(index.resolve()),
            }
            aggregate_path = output / label / "data_distribution_audit.json"
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            q_shard_reports[label] = shard_reports

    reports = {}
    for label in indexes:
        outcome = load_json(output / f"outcomes_{label}.json")
        q_report_path = output / label / "data_distribution_audit.json"
        if args.skip_q_audit:
            q_report = None
        else:
            q_report = load_json(q_report_path)
        if outcome["hanchans"] != 3000 or outcome["malformed_count"] != 0:
            raise SystemExit(f"{label} outcome audit failed: {outcome}")
        if q_report is not None and (
            q_report["corpus"]["files_selected"] != 3000
            or q_report["corpus"]["trainable_perspectives"] != 3000
            or q_report["corpus"]["malformed_count"] != 0
        ):
            raise SystemExit(f"{label} Q audit failed corpus gate")
        reports[label] = {"outcomes": outcome, "q": q_report}

    if args.skip_q_audit:
        print(json.dumps({"status": "outcome_audit_only", "output": str(output)}, indent=2), flush=True)
        return

    view_rows = {}
    for label, value in reports.items():
        q = value["q"]
        support = q["support_audit"]["overall"]
        decisions = int(q["corpus"]["total_decisions"])
        contribution = q["hanchan_contribution"]
        view_rows[label] = {
            "files": int(q["corpus"]["files_selected"]),
            "decisions": decisions,
            "greedy_agreement_rate": float(support["greedy_agreement_rate"]),
            "mean_behavior_q_rank": float(support["mean_behavior_q_rank"]),
            "mean_q_regret_greedy_minus_behavior": float(support["mean_q_regret_greedy_minus_behavior"]),
            "mean_greedy_margin": float(support["mean_greedy_margin"]),
            "mean_behavior_q": float(support["mean_behavior_q"]),
            "mean_greedy_q": float(support["mean_greedy_q"]),
            "mean_legal_q_abs": float(support["mean_legal_q_abs"]),
            "decision_weight_ess": float(contribution["decision_weight_ess"]),
            "outcomes": value["outcomes"],
            "q_report": str((output / label / "data_distribution_audit.json").resolve()),
        }
    v2_q = reports["V2_74000"]["q"]
    v3_q = reports["V3_74000"]["q"]
    combined_support = weighted_support([v2_q, v3_q])
    combined_corpus = weighted_corpus([v2_q, v3_q])
    combined_outcomes = combine_outcomes([reports["V2_74000"]["outcomes"], reports["V3_74000"]["outcomes"]])
    combined_decisions = weighted_decisions([v2_q, v3_q])
    combined_contribution = {
        "hanchans": combined_corpus["hanchans"],
        "total_decisions": combined_corpus["total_decisions"],
        "decision_weight_ess": combined_corpus["total_decisions"] ** 2
        / sum(int(row["decisions"]) ** 2 for report in (v2_q, v3_q) for row in report["hanchans"]),
    }
    view_rows["D2_50_50"] = {
        "files": combined_corpus["files_selected"],
        "decisions": combined_corpus["total_decisions"],
        **{key: float(value) for key, value in combined_support.items() if key != "behavior_q_rank_counts"},
        "decision_weight_ess": float(combined_contribution["decision_weight_ess"]),
        "outcomes": combined_outcomes,
        "decision_distribution": combined_decisions,
    }

    d1_reference = (
        root.parent
        / "D1_project_owned_population_2026_07"
        / "training_prep_2026_07/distribution/summary/d1_distribution_d1.json"
    )
    k0 = load_json(d1_reference) if d1_reference.is_file() else None
    if k0:
        k0_support = k0["support_audit_overall"]
        view_rows["K0_70k_reference"] = {
            "files": int(k0["corpus"]["files_selected"]),
            "decisions": int(k0["corpus"]["total_decisions"]),
            "greedy_agreement_rate": float(k0_support["greedy_agreement_rate"]),
            "mean_behavior_q_rank": float(k0_support["mean_behavior_q_rank"]),
            "mean_q_regret_greedy_minus_behavior": float(k0_support["mean_q_regret_greedy_minus_behavior"]),
            "mean_greedy_margin": float(k0_support["mean_greedy_margin"]),
            "mean_behavior_q": float(k0_support["mean_behavior_q"]),
            "mean_greedy_q": float(k0_support["mean_greedy_q"]),
            "mean_legal_q_abs": float(k0_support["mean_legal_q_abs"]),
            "decision_weight_ess": float(k0["hanchan_contribution"]["decision_weight_ess"]),
            "outcomes": k0["outcomes"],
            "reference": str(d1_reference.resolve()),
        }

    report = {
        "schema": "keqing.mortal.d2_descendant_view_audit.v1",
        "passed": all(
            row["files"] == (6000 if name == "D2_50_50" else 3000)
            for name, row in view_rows.items()
            if name in {"V2_74000", "V3_74000", "D2_50_50"}
        ),
        "source": {
            "d1_root": str(root),
            "source_file_count": 6000,
            "reuse_same_hanchans": True,
        },
        "assignment": {
            "method": "canonical_hash_parity_50_50",
            "counts": {"V2_74000": 3000, "V3_74000": 3000},
            "single_perspective_per_file": True,
        },
        "malformed_count": sum(int(value["outcomes"]["malformed_count"]) for value in reports.values()),
        "views": view_rows,
        "decision_distributions": {
            "V2_74000": reports["V2_74000"]["q"]["decision_distribution"],
            "V3_74000": reports["V3_74000"]["q"]["decision_distribution"],
            "D2_50_50": combined_decisions,
        },
        "notes": [
            "V2/V3 assignment is independent of outcome, rank, and target.",
            "D2_50_50 metrics are decision-weighted sums of the two fixed 3000-file views.",
            "Parent-Q regret is an internal support diagnostic, not a ground-truth regret estimate.",
        ],
    }
    json_path = output / "d2_descendant_view_audit.json"
    md_path = output / "d2_descendant_view_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("D2 descendant-view audit failed")


if __name__ == "__main__":
    main()
