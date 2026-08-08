#!/usr/bin/env python3
"""Audit C/V Q diagnostics over the existing 70k->72k training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SEEDS = (20260803, 20260804, 20260805)
GROUPS = {
    "C": "C_behavior_action_mc",
    "V": "V_legal_mean_mc",
}
TAGS = {
    "legal_q_mean": "q/legal_q_mean_window",
    "behavior_centered_advantage": "q/behavior_centered_advantage_window",
    "greedy_margin": "q/greedy_margin_window",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/experiments/model_pool_2026_07/legal_mean_value_ab_2026_07"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=70001)
    parser.add_argument("--end-step", type=int, default=72000)
    return parser.parse_args()


def _series(tb_dir: Path, tag: str) -> tuple[dict[int, float], list[dict[str, Any]]]:
    values: dict[int, float] = {}
    conflicts: list[dict[str, Any]] = []
    for event_file in sorted(tb_dir.glob("events.*")):
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            value = float(event.value)
            previous = values.get(step)
            if previous is not None and not np.isclose(previous, value, rtol=1e-6, atol=1e-7):
                conflicts.append(
                    {
                        "tag": tag,
                        "step": step,
                        "previous": previous,
                        "replacement": value,
                        "event_file": event_file.name,
                    }
                )
            values[step] = value
    return values, conflicts


def _read_seed(experiment_root: Path, seed: int, start_step: int, end_step: int) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for model_key, group in GROUPS.items():
        tb_dir = experiment_root / group / f"seed_{seed}" / "tb_mortal"
        if not tb_dir.is_dir():
            raise FileNotFoundError(tb_dir)
        metrics: dict[str, Any] = {}
        for metric, tag in TAGS.items():
            series, series_conflicts = _series(tb_dir, tag)
            conflicts.extend(series_conflicts)
            selected = {step: value for step, value in series.items() if start_step <= step <= end_step}
            if not selected or min(selected) != start_step or max(selected) != end_step:
                raise ValueError(f"{tb_dir}: incomplete {tag} range {start_step}->{end_step}")
            ordered = [selected[step] for step in sorted(selected)]
            metrics[metric] = {
                "step_count": len(ordered),
                "start": float(ordered[0]),
                "end": float(ordered[-1]),
                "delta": float(ordered[-1] - ordered[0]),
                "mean": float(np.mean(ordered)),
                "abs_mean": float(np.mean(np.abs(ordered))),
                "min": float(np.min(ordered)),
                "max": float(np.max(ordered)),
            }
        models[model_key] = metrics

    paired: dict[str, dict[str, float]] = {}
    for metric in TAGS:
        control = models["C"][metric]
        variant = models["V"][metric]
        paired[metric] = {
            "start_v_minus_c": variant["start"] - control["start"],
            "end_v_minus_c": variant["end"] - control["end"],
            "delta_v_minus_c": variant["delta"] - control["delta"],
            "mean_v_minus_c": variant["mean"] - control["mean"],
        }
    return {"seed": seed, "models": models, "paired": paired, "duplicate_step_conflicts": conflicts}


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    args = parse_args()
    if args.end_step < args.start_step:
        raise ValueError("end step must be >= start step")
    per_seed = [_read_seed(args.experiment_root, seed, args.start_step, args.end_step) for seed in SEEDS]
    document = {
        "schema": "keqing.mortal.legal_mean_training_drift.v1",
        "experiment_root": str(args.experiment_root.resolve()),
        "training_seeds": list(SEEDS),
        "step_range": [args.start_step, args.end_step],
        "tags": TAGS,
        "per_seed": per_seed,
        "interpretation": {
            "legal_q_mean": "scalar-Q/common-offset proxy; not a direct action-preference measure",
            "behavior_centered_advantage": "behavior Q relative to legal-Q mean; centered action-preference proxy",
            "greedy_margin": "top-two legal action Q margin",
            "duplicate_step_conflicts": "resume-overlap scalar values are resolved by latest event-file order and retained for audit",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "legal_mean_training_drift.json"
    md_path = args.output_dir / "legal_mean_training_drift.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Legal-Mean Training Drift Audit",
        "",
        f"Existing TensorBoard window metrics from steps `{args.start_step}` to `{args.end_step}`. This is analysis-only and does not select a checkpoint.",
        "",
        "`legal_q_mean` is used as a scalar-Q/common-offset proxy; `behavior_centered_advantage` is the centered action-preference proxy; `greedy_margin` is the top-two legal-action Q margin.",
        "",
        "## V-C Changes",
        "",
        "| Seed | Metric | V-C at start | V-C at end | Change in V-C |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for item in per_seed:
        for metric in TAGS:
            values = item["paired"][metric]
            lines.append(
                f"| {item['seed']} | {metric} | {_fmt(values['start_v_minus_c'])} | "
                f"{_fmt(values['end_v_minus_c'])} | {_fmt(values['delta_v_minus_c'])} |"
            )
    lines.extend(["", "## TensorBoard Resume Overlap", "", "Duplicate scalar steps are resolved by latest event-file order and retained here instead of being silently overwritten.", ""])
    for item in per_seed:
        conflicts = item["duplicate_step_conflicts"]
        if conflicts:
            lines.append(f"- Seed `{item['seed']}`: `{len(conflicts)}` conflicting duplicate scalar values; latest event file wins.")
        else:
            lines.append(f"- Seed `{item['seed']}`: no conflicting duplicate scalar values.")
    lines.extend(
        [
            "",
            "## Per-Model Endpoints",
            "",
            "| Seed | Model | Legal-Q mean start->end | Centered advantage start->end | Greedy margin start->end |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in per_seed:
        for model in ("C", "V"):
            values = item["models"][model]
            lines.append(
                f"| {item['seed']} | {model} | {_fmt(values['legal_q_mean']['start'])}->{_fmt(values['legal_q_mean']['end'])} | "
                f"{_fmt(values['behavior_centered_advantage']['start'])}->{_fmt(values['behavior_centered_advantage']['end'])} | "
                f"{_fmt(values['greedy_margin']['start'])}->{_fmt(values['greedy_margin']['end'])} |"
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The audit records whether legal-mean training changes scalar Q calibration, centered action preference, and greedy margin relative to behavior-action control. It does not establish a causal mechanism or a strength result. The legal-mean branch remains closed by the B250 paired evaluation.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
