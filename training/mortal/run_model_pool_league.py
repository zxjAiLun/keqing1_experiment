#!/usr/bin/env python3
"""Run the balanced final model-pool league and aggregate family statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

# Allow direct execution via ``python training/mortal/run_model_pool_league.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.stat_report import build_stat_report


LINEUPS = (
    ("L1", ("ext_mortal", "70k", "V2_74000", "V3_74000")),
    ("L2", ("ext_mortal_a", "ext_mortal_b", "V2_74000", "V3_74000")),
    ("L3", ("70k_a", "70k_b", "V2_74000", "V3_74000")),
    ("L4", ("V2_74000_a", "V2_74000_b", "ext_mortal", "70k")),
    ("L5", ("V3_74000_a", "V3_74000_b", "ext_mortal", "70k")),
)
BASE_LABELS = {
    "ext_mortal_a": "ext_mortal",
    "ext_mortal_b": "ext_mortal",
    "70k_a": "70k",
    "70k_b": "70k",
    "V2_74000_a": "V2_74000",
    "V2_74000_b": "V2_74000",
    "V3_74000_a": "V3_74000",
    "V3_74000_b": "V3_74000",
}
RANK_PTS = [90.0, 45.0, 0.0, -135.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ext-mortal", type=Path, default=Path("artifacts/external_mortal_20240308_best_min.pth"))
    parser.add_argument("--70k", dest="anchor_70k", type=Path, default=Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"))
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=991000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--games-per-lineup", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def family_label(label: str) -> str:
    return BASE_LABELS.get(label, label)


def model_path_for_label(label: str, args: argparse.Namespace) -> Path:
    return {
        "ext_mortal": args.ext_mortal,
        "70k": args.anchor_70k,
        "V2_74000": args.v2,
        "V3_74000": args.v3,
    }[family_label(label)]


def aggregate_family_stats(stat_report: dict) -> dict[str, dict]:
    raw_totals: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for label, player in stat_report["players"].items():
        family = family_label(label)
        raw = player["raw"]
        for key, value in raw.items():
            raw_totals[family][key] += float(value)

    result: dict[str, dict] = {}
    for family, raw in sorted(raw_totals.items()):
        games = raw["game"]
        rounds = raw["round"]
        total_rank_pt = sum(raw[f"rank_{rank}"] * RANK_PTS[rank - 1] for rank in range(1, 5))
        result[family] = {
            "games": int(games),
            "rounds": int(rounds),
            "rank_counts": [int(raw[f"rank_{rank}"]) for rank in range(1, 5)],
            "avg_rank": sum(rank * raw[f"rank_{rank}"] for rank in range(1, 5)) / games,
            "avg_rank_pt": total_rank_pt / games,
            "avg_game_delta_score": raw["point"] / games,
            "agari_rate": raw["agari"] / rounds,
            "houjuu_rate": raw["houjuu"] / rounds,
            "fuuro_rate": raw["fuuro"] / rounds,
            "riichi_rate": raw["riichi"] / rounds,
            "ryukyoku_rate": raw["ryukyoku"] / rounds,
            "tobi_rate": raw["tobi"] / games,
        }
    return result


def write_summary(output_dir: Path, stats: dict[str, dict], *, games_total: int) -> None:
    document = {
        "schema": "keqing.mortal.model_pool_league.v1",
        "lineups": [name for name, _ in LINEUPS],
        "games_per_lineup": games_total // len(LINEUPS),
        "games_total": games_total,
        "rank_points": RANK_PTS,
        "models": stats,
    }
    (output_dir / "league_summary.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Balanced Model Pool League",
        "",
        f"- Games: `{report['games']}`",
        "- Each model family: `1000` seat-hanchans",
        "",
        "| Model | Games | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | Rank counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model, row in stats.items():
        lines.append(
            f"| {model} | {row['games']} | {row['avg_rank']:.4f} | {row['avg_rank_pt']:.2f} | "
            f"{row['agari_rate']:.4%} | {row['houjuu_rate']:.4%} | {row['fuuro_rate']:.4%} | "
            f"{row['riichi_rate']:.4%} | {row['rank_counts']} |"
        )
    (output_dir / "league_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.games_per_lineup != 200:
        raise ValueError("the balanced league requires exactly 200 games per lineup")
    for checkpoint in (args.ext_mortal, args.anchor_70k, args.v2, args.v3):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_dirs: list[Path] = []
    for index, (lineup_name, labels) in enumerate(LINEUPS):
        block_dir = output_dir / "lineups" / lineup_name
        block_dirs.append(block_dir)
        command = [
            sys.executable,
            str(Path("training/mortal/four_player_native.py").resolve()),
            "--output-dir", str(block_dir),
            "--device", str(args.device),
            "--games", str(args.games_per_lineup),
            "--seed-start", str(args.seed_start + index * args.games_per_lineup),
            "--seed-key", str(args.seed_key),
            "--seat-mode", "random",
            "--progress-every", "25",
            "--native-batch-games", "25",
            "--require-cuda" if args.require_cuda else "",
        ]
        command = [item for item in command if item]
        for label in labels:
            command.extend(["--model", f"{label}={model_path_for_label(label, args)}"])
        print(f"[league] start {lineup_name}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=Path.cwd(), check=True)

    combined_dir = output_dir / "combined_logs"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for block_dir in block_dirs:
        for source in sorted((block_dir / "logs").glob("*.json.gz")):
            shutil.copy2(source, combined_dir / f"{block_dir.name}_{source.name}")

    stat_report = build_stat_report(
        log_dir=combined_dir,
        players={label: label for label in sorted({label for _, labels in LINEUPS for label in labels})},
        mortal_root=Path("third_party/Mortal"),
        rank_pts=RANK_PTS,
        rank_points_profile="custom",
    )
    stats = aggregate_family_stats(stat_report)
    games_total = sum(int(player["raw"]["game"]) for player in stat_report["players"].values()) // 4
    write_summary(output_dir, stats, games_total=games_total)
    manifest = {
        "schema": "keqing.mortal.model_pool_league_run.v1",
        "seed_start": args.seed_start,
        "seed_key": args.seed_key,
        "games_per_lineup": args.games_per_lineup,
        "lineups": [{"name": name, "labels": labels} for name, labels in LINEUPS],
        "models": {"ext_mortal": str(args.ext_mortal), "70k": str(args.anchor_70k), "V2_74000": str(args.v2), "V3_74000": str(args.v3)},
        "summary": str(output_dir / "league_summary.json"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
