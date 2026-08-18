#!/usr/bin/env python3
"""Validate and render the machine-readable Mortal research registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BEGIN = "<!-- BEGIN AUTO REGISTRY STATUS -->"
END = "<!-- END AUTO REGISTRY STATUS -->"
REQUIRED_RECORD_FIELDS = {
    "experiment_id",
    "name_zh",
    "category",
    "parent_model",
    "control",
    "variant",
    "changed_variable",
    "fixed_variables",
    "training_seeds",
    "evaluation_protocol",
    "status",
    "primary_result",
    "promoted_data_route",
    "promoted_checkpoint",
    "recipe_promotion",
    "checkpoint_promotion",
    "closure_reason",
    "predecessor",
    "next_experiment",
    "commits",
    "artifact_paths",
    "report_paths",
}
ALLOWED_STATUSES = {
    "operational",
    "closed",
    "rejected",
    "promoted_recipe_only",
    "analysis_only",
    "proposal_only_not_started",
    "preregistered_not_started",
    "preregistered_frozen",
    "gate_passed",
    "generation_closed",
    "not_selected",
}


def load_registry(path: Path) -> dict[str, Any]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"registry is not valid JSON: {exc}") from exc
    if registry.get("schema") != "keqing.mortal.research_registry.v1":
        raise ValueError("unsupported registry schema")
    state = registry.get("current_state")
    if not isinstance(state, dict):
        raise ValueError("current_state must be an object")  # noqa: TRY004
    for key in ("current_formal_lineage", "K0", "K1", "operational_control", "next_experiment", "prohibitions"):
        if key not in state:
            raise ValueError(f"current_state missing {key}")
    records = registry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty array")
    ids: set[str] = set()
    for record in records:
        missing = REQUIRED_RECORD_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {record.get('experiment_id')} missing {sorted(missing)}")
        experiment_id = record["experiment_id"]
        if experiment_id in ids:
            raise ValueError(f"duplicate experiment_id: {experiment_id}")
        ids.add(experiment_id)
        if record["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported status for {experiment_id}: {record['status']}")
        if not isinstance(record["fixed_variables"], list) or not isinstance(record["training_seeds"], list):
            raise ValueError(f"record {experiment_id} has invalid list fields")  # noqa: TRY004
    if state["K1"] is not None:
        raise ValueError("K1 must remain null until a formal K1 is actually promoted")
    record_by_id = {record["experiment_id"]: record for record in records}
    next_id = state["next_experiment"]
    if next_id is not None:
        if next_id not in record_by_id:
            raise ValueError(f"current_state.next_experiment is not registered: {next_id}")
        next_record = record_by_id[next_id]
        if next_record["status"] != state["next_experiment_status"]:
            raise ValueError(
                "next experiment status mismatch: "
                f"current_state={state['next_experiment_status']} record={next_record['status']}"
            )
    operational_id = state["operational_control"]
    if operational_id not in record_by_id:
        raise ValueError(f"operational control is not registered: {operational_id}")
    if record_by_id[operational_id]["status"] != "operational":
        raise ValueError(f"operational control is not operational: {operational_id}")
    reference_fields = ("predecessor", "next_experiment", "mechanistic_reference", "primary_promotion_control")
    for record in records:
        for field in reference_fields:
            reference = record.get(field)
            if reference is not None and reference not in record_by_id:
                raise ValueError(f"record {record['experiment_id']} references unknown {field}: {reference}")
        promotion_control = record.get("primary_promotion_control")
        if promotion_control is not None and record_by_id[promotion_control]["status"] != "operational":
            raise ValueError(
                f"record {record['experiment_id']} primary_promotion_control is not operational: "
                f"{promotion_control}"
            )
    return registry


def status_text(status: str) -> str:
    return {
        "operational": "运行中",
        "closed": "已关闭",
        "rejected": "拒绝/不晋级",
        "promoted_recipe_only": "仅 recipe 晋级",
        "analysis_only": "仅分析",
        "proposal_only_not_started": "仅提案，未启动",
        "preregistered_not_started": "已预注册，未启动",
        "preregistered_frozen": "预注册已冻结，未启动",
        "gate_passed": "首个 B250 gate 已通过",
        "generation_closed": "6000h 生成已闭环",
        "not_selected": "未选择",
    }[status]


def render_block(registry: dict[str, Any]) -> str:
    state = registry["current_state"]
    records = registry["records"]
    lines = [
        BEGIN,
        "## 自动状态（由 `research_registry.json` 生成）",
        "",
        f"- 正式当前 lineage：`{state['current_formal_lineage']}`；K1：`{state['K1'] or '尚未产生'}`。",
        f"- 当前 operational control：`{state['operational_control']}`。",
        f"- 当前 objective/reward：`{state['operational_objective']}` / `{state['operational_reward']}`。",
        f"- 70k legacy continuation optimizer：`{state['legacy_continuation_optimizer']}`。",
        f"- 当前阶段：{state['current_stage']}",
        "",
        "| 实验 ID | 类别 | 状态 | 主要结论 |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        result = str(record["primary_result"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{record['experiment_id']}` | {record['category']} | {status_text(record['status'])} | {result} |")
    lines.extend([
        "",
        f"- 下一实验提案：`{state['next_experiment'] or '尚未选择'}`（{state['next_experiment_status']}）。",
        "- 当前禁止事项：",
    ])
    lines.extend(f"  - {item}" for item in state["prohibitions"])
    lines.append(END)
    return "\n".join(lines) + "\n"


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", re.DOTALL)
    if not pattern.search(text):
        raise ValueError("overview does not contain the registry markers")
    return pattern.sub(block, text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("training/docs/mortal/research_registry.json"))
    parser.add_argument("--overview", type=Path, default=Path("training/docs/mortal/研发总览_当前.md"))
    parser.add_argument("--write", action="store_true", help="replace the marked block in the overview")
    parser.add_argument("--check", action="store_true", help="fail when the overview is not up to date")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")
    try:
        registry = load_registry(args.registry)
        expected = replace_block(args.overview.read_text(encoding="utf-8"), render_block(registry))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    current = args.overview.read_text(encoding="utf-8")
    if args.write:
        args.overview.write_text(expected, encoding="utf-8", newline="\n")
        print(f"updated {args.overview}")
        return 0
    if args.check and current != expected:
        print(f"error: generated registry block is stale in {args.overview}", file=sys.stderr)
        return 1
    print("registry valid; overview is up to date" if args.check else "registry valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
