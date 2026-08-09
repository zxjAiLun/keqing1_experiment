"""Descriptive distributions and Markdown output for the D3 B250 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.mortal.d3_exploration_engine import CONTRACT_ID
from training.mortal.d3_production_contract import (
    SEED_END_EXCLUSIVE, SEED_KEY, SEED_START, expected_seed_keys,
)
from training.mortal.d3_production_audit_core import (
    DecisionSnapshot, _bucket, _event_context, _quantiles,
)

def _exploration_distribution(
    events: list[dict[str, Any]],
    snapshots: dict[tuple[int, int, int, int, int], DecisionSnapshot],
    all_kyoku_keys: Iterable[Iterable[int]],
) -> dict[str, Any]:
    hanchan_counts: Counter[tuple[int, int]] = Counter()
    kyoku_counts: Counter[tuple[int, int, int, int]] = Counter()
    phase_counts: Counter[str] = Counter()
    shanten_counts: Counter[int] = Counter()
    legal_counts: Counter[int] = Counter()
    action_counts: Counter[str] = Counter()
    from training.mortal.audit_replay_distribution import action_name  # noqa: PLC0415

    for event in events:
        if not bool(event.get("explored")):
            continue
        context = _event_context(event)
        seed, key, seat, kyoku_index, _ = context
        hanchan_counts[(seed, key)] += 1
        kyoku_counts[(seed, key, seat, kyoku_index)] += 1
        snapshot = snapshots[context]
        phase_counts[snapshot.phase] += 1
        shanten_counts[snapshot.shanten] += 1
        legal_counts[snapshot.legal_action_count] += 1
        action_counts[action_name(snapshot.action)] += 1

    hanchan_values = [hanchan_counts.get(key, 0) for key in sorted(expected_seed_keys())]
    kyoku_key_tuples = [tuple(int(value) for value in key) for key in all_kyoku_keys]
    kyoku_values = [kyoku_counts.get(key, 0) for key in kyoku_key_tuples]
    return {
        "explorations_per_hanchan": _quantiles(hanchan_values),
        "explorations_per_hanchan_counts": _bucket(hanchan_values),
        "explorations_per_kyoku": _quantiles(kyoku_values),
        "explorations_per_kyoku_counts": _bucket(kyoku_values),
        "phase_counts": {key: int(value) for key, value in sorted(phase_counts.items())},
        "shanten_counts": _bucket(shanten_counts.elements()),
        "legal_action_count_distribution": _bucket(legal_counts.elements()),
        "action_mix": {key: int(value) for key, value in sorted(action_counts.items())},
    }

def _write_markdown(report: dict[str, Any], path: Path) -> None:
    gate = report["gate"]
    metrics = report["descriptive_metrics"]
    behavior = metrics["k0_behavior"]
    exploration = metrics["exploration"]
    lines = [
        "# D3 首个 250h 生产 Gate 机器审计 v2",
        "",
        f"- Gate：`{gate['verdict']}`",
        f"- Contract：`{CONTRACT_ID}`",
        f"- Seeds：`{SEED_START}..{SEED_END_EXCLUSIVE - 1}`；seed key `{SEED_KEY}`",
        f"- 真实单次 B250：`{gate['checks']['provenance']['native_call_count']}`",
        f"- Generation commit：`{gate['generation_commit']}`",
        f"- Auditor commit：`{gate['auditor_commit']}`",
        f"- 日志：`{report['data_integrity']['file_count']}`；唯一 canonical hanchans：`{report['data_integrity']['unique_canonical_hanchans']}`",
        f"- Primary decisions：`{behavior['total_primary_decisions']}`；parent greedy agreement：`{behavior['parent_greedy_agreement_rate']:.4%}`",
        f"- Eligible / explored：`{exploration['eligible_count']}` / `{exploration['explored_count']}`；realized rate：`{exploration['realized_explored_over_eligible']:.4%}`",
        f"- 事件映射：`{report['event_audit']['event_count'] - report['event_audit']['mapping_violations'].get('unmapped_context', 0)}` / `{report['event_audit']['event_count']}`；behavior mismatch：`{report['event_audit']['mapping_violations'].get('behavior_mismatch', 0)}`",
        "",
        "## Hard checks",
        "",
    ]
    for section, checks in gate["checks"].items():
        lines.append(f"### {section}")
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
        lines.append("")
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "本报告只判断冻结 recipe 是否完整、可训练并严格遵守预注册合同。rank/Pt、胡率、放铳率、副露率、立直率及分布指标只作描述，不用于修改 margin、probability、预算或选择参数；本审计不会启动剩余 5750h，也不会启动训练。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
