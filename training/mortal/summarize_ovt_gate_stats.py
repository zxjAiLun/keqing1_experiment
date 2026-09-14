#!/usr/bin/env python3
"""Chinese summary of a bidirectional one-vs-three gate from the detailed reports.

Why this exists: ``one_vs_three_native.py`` writes ``detailed_stats.json`` per
direction, but a direction on its own cannot answer "where does the candidate
differ from its opponent" -- the candidate is the *solo seat* in one direction
and one of the *three trio seats* in the other.  This reads both directions per
gate, averages each side's per-seat rates over the two directions, and prints a
Chinese table with the per-seat sample basis spelled out.

It only reads reports.  It never plays a game and never touches the GPU.

    python training/mortal/summarize_ovt_gate_stats.py \
        --gate "Gate A：U32 vs K0_70k=artifacts/eval/ovt_p4m11_gateA_solo|artifacts/eval/ovt_p4m11_gateA_mirror" \
        --out artifacts/eval/p4m11_gate_stats_zh.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# (Chinese label, derived field, formatter).  Lower is better only for avg_rank.
SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("平均顺位（越低越好）", "avg_rank", "rank"),
    ("首位率", "rank_1_rate", "rate"),
    ("二位率", "rank_2_rate", "rate"),
    ("三位率", "rank_3_rate", "rate"),
    ("四位率", "rank_4_rate", "rate"),
    ("和牌率", "agari_rate", "rate"),
    ("放铳率", "houjuu_rate", "rate"),
    ("立直率", "riichi_rate", "rate"),
    ("副露率", "fuuro_rate", "rate"),
    ("平均和牌打点", "avg_point_per_agari", "point"),
    ("平均和牌巡目", "avg_agari_jun", "turn"),
    ("平均放铳巡目", "avg_houjuu_jun", "turn"),
    ("平均每局得分", "avg_point_per_game", "point"),
    ("平均顺位点", "avg_rank_pt", "rankpt"),
)


class GateSpecError(ValueError):
    """Raised when a --gate spec or a report cannot be interpreted."""


def parse_gate_spec(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec or "|" not in spec:
        raise GateSpecError(f"--gate must be LABEL=SOLO_DIR|MIRROR_DIR, got: {spec}")
    label, dirs = spec.split("=", 1)
    solo_raw, mirror_raw = dirs.split("|", 1)
    label, solo_raw, mirror_raw = label.strip(), solo_raw.strip(), mirror_raw.strip()
    if not label or not solo_raw or not mirror_raw:
        raise GateSpecError(f"--gate must be LABEL=SOLO_DIR|MIRROR_DIR, got: {spec}")
    return label, Path(solo_raw), Path(mirror_raw)


def _derived(report: dict[str, Any], label: str) -> dict[str, Any]:
    players = report.get("players")
    if not isinstance(players, dict):
        raise GateSpecError("report has no players mapping")
    if label not in players:
        raise GateSpecError(f"report has no player {label!r}; it has {sorted(players)}")
    return players[label]["derived"]


def _games(report: dict[str, Any], label: str) -> int:
    return int(report["players"][label]["raw"]["game"])


def read_direction(directory: Path, candidate_label: str) -> dict[str, Any]:
    """Read one direction and classify who sat solo and who sat in the trio."""
    path = directory / "detailed_stats.json"
    if not path.exists():
        raise GateSpecError(f"missing {path}; run stat_report.py for this direction first")
    report = json.loads(path.read_text(encoding="utf-8"))
    players = list(report["players"])
    if len(players) != 2:
        raise GateSpecError(f"{path} describes {len(players)} players, expected exactly 2")
    if candidate_label not in players:
        raise GateSpecError(f"{path} does not describe {candidate_label!r}; it has {players}")
    counts = {label: _games(report, label) for label in players}
    for label, count in counts.items():
        if count <= 0:
            raise GateSpecError(f"{path} reports 0 games for {label!r}")
    solo = min(counts, key=lambda label: counts[label])
    trio = [label for label in players if label != solo][0]
    if counts[trio] != 3 * counts[solo]:
        raise GateSpecError(
            f"{path}: trio games {counts[trio]} != 3 x solo games {counts[solo]}; "
            "this is not a one-vs-three log set"
        )
    return {
        "dir": str(directory),
        "hanchans": counts[solo],
        "solo_label": solo,
        "trio_label": trio,
        "games": counts,
        "derived": {label: _derived(report, label) for label in players},
    }


def summarize_gate(label: str, solo_dir: Path, mirror_dir: Path, candidate_label: str) -> dict[str, Any]:
    solo = read_direction(solo_dir, candidate_label)
    mirror = read_direction(mirror_dir, candidate_label)
    if solo["hanchans"] != mirror["hanchans"]:
        raise GateSpecError(
            f"{label}: the two directions cover different hanchan counts "
            f"({solo['hanchans']} vs {mirror['hanchans']})"
        )
    if solo["solo_label"] != mirror["trio_label"] or solo["trio_label"] != mirror["solo_label"]:
        raise GateSpecError(
            f"{label}: the two directions are not mirror images "
            f"(solo {solo['solo_label']!r}/{mirror['trio_label']!r}, "
            f"trio {solo['trio_label']!r}/{mirror['solo_label']!r})"
        )
    candidate = solo["solo_label"]
    opponent = solo["trio_label"]
    if candidate != candidate_label:
        raise GateSpecError(
            f"{label}: the candidate {candidate_label!r} is not the solo seat of the solo direction "
            f"(found {candidate!r})"
        )
    averaged: dict[str, dict[str, float]] = {}
    # Per-seat rates are directly comparable across the two directions, so each
    # side is averaged over the direction where it sat solo and the one where it
    # sat in the trio.  (Do not name this loop variable ``label``: that would
    # shadow the gate label returned below.)
    for side in (candidate, opponent):
        averaged[side] = {
            key: (float(solo["derived"][side][key]) + float(mirror["derived"][side][key])) / 2.0
            for _title, key, _kind in SUMMARY_ROWS
        }
    return {
        "label": label,
        "candidate_label": candidate,
        "opponent_label": opponent,
        "hanchans": solo["hanchans"],
        "averaged": averaged,
        "directions": {"solo": solo, "mirror": mirror},
    }


def _format(value: float | None, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "rate":
        return f"{value * 100:.2f}%"
    if kind == "rank":
        return f"{value:.4f}"
    if kind == "point":
        return f"{value:+.1f}"
    if kind == "turn":
        return f"{value:.2f}"
    return f"{value:+.1f}"


def _delta(value: float, other: float, kind: str) -> str:
    diff = value - other
    if kind == "rate":
        return f"{diff * 100:+.2f}pp"
    if kind == "rank":
        return f"{diff:+.4f}"
    if kind == "turn":
        return f"{diff:+.2f}"
    return f"{diff:+.1f}"


def format_markdown(summaries: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# P4-M11 双向门详细统计摘要（中文）",
        "",
        "本文件由 `training/mortal/summarize_ovt_gate_stats.py` 从各方向的",
        "`detailed_stats.json` 生成，**不重打任何对局**。",
        "",
        "## 样本口径",
        "",
        "- 每个方向：256 seeds × 4 座位轮换 = **1,024 半庄**。",
        "- 单挑方每半庄占 **1 座**，三人组占 **3 座**（同一批半庄）。",
        "- 因此三人组的 `Games` 计数是单挑方的 3 倍：**只比较逐座比率，不比较计数**。",
        "- 下表把候选与其对手各自在两个方向上的逐座比率取平均（两个方向是镜像，",
        "  平均后每一方都同时包含 1 座与 3 座两种角色）。",
        "- `ryukyoku_rate`（流局率）是牌桌级属性，双方相同，故不列入对比。",
        "",
    ]
    for summary in summaries:
        candidate = summary["candidate_label"]
        opponent = summary["opponent_label"]
        lines += [
            f"## {summary['label']}",
            "",
            f"方向：`solo` = {candidate} 单挑 vs 3×{opponent}；",
            f"`mirror` = {opponent} 单挑 vs 3×{candidate}（各 {summary['hanchans']} 半庄）。",
            "",
            f"| 指标 | {candidate}（逐座均值） | {opponent}（逐座均值） | 差（{candidate} − {opponent}） |",
            "| --- | --- | --- | --- |",
        ]
        for title, key, kind in SUMMARY_ROWS:
            mine = summary["averaged"][candidate][key]
            theirs = summary["averaged"][opponent][key]
            lines.append(
                f"| {title} | {_format(mine, kind)} | {_format(theirs, kind)} | {_delta(mine, theirs, kind)} |"
            )
        lines += ["", "逐方向原始值（未平均）：", ""]
        lines.append(f"| 指标 | solo {candidate} | solo {opponent} | mirror {opponent} | mirror {candidate} |")
        lines.append("| --- | --- | --- | --- | --- |")
        solo = summary["directions"]["solo"]
        mirror = summary["directions"]["mirror"]
        for title, key, kind in SUMMARY_ROWS:
            cells = [
                _format(float(solo["derived"][candidate][key]), kind),
                _format(float(solo["derived"][opponent][key]), kind),
                _format(float(mirror["derived"][opponent][key]), kind),
                _format(float(mirror["derived"][candidate][key]), kind),
            ]
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gate",
        action="append",
        required=True,
        help="LABEL=SOLO_DIR|MIRROR_DIR. Repeat for several gates, e.g. one per opponent.",
    )
    parser.add_argument("--candidate", default="p4m11_u32", help="label of the candidate under test")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown here instead of stdout")
    parser.add_argument("--json-out", type=Path, default=None, help="optional machine-readable copy")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries = [
        summarize_gate(*parse_gate_spec(spec), candidate_label=args.candidate) for spec in args.gate
    ]
    markdown = format_markdown(summaries)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(markdown)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
