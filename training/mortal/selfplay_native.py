#!/usr/bin/env python3
"""Generate one independent Mortal selfplay hanchan per seed."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.eval_metrics import (
    add_rank_point_args,
    build_metrics_document,
    resolve_rank_points,
    summarize_rank_counts_with_references,
    write_metrics,
)
from training.mortal.stat_report import write_stat_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", default="ext_mortal")
    parser.add_argument(
        "--player-names",
        nargs=4,
        metavar=("TRAIN", "OPP1", "OPP2", "OPP3"),
        help="optional per-seat log names for pure selfplay; all four use the same engine",
    )
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--games", type=int, required=True, help="One game is generated for each seed.")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--native-batch-games",
        type=int,
        default=0,
        help="hanchans per Rust arena batch; 0 preserves the progress-sized batch behavior",
    )
    parser.add_argument("--profile", action="store_true", help="record inference batch and timing telemetry")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--defer-reports", action="store_true", help="generate logs only; build reports in a final pass")
    add_rank_point_args(parser)
    return parser.parse_args()


def _load_engine(args: argparse.Namespace) -> Any:
    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    from engine import MortalEngine  # noqa: PLC0415
    from model import Brain, DQN  # noqa: PLC0415

    state = torch.load(args.model, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = int(cfg["control"].get("version", 4))
    mortal = Brain(
        version=version,
        conv_channels=int(cfg["resnet"]["conv_channels"]),
        num_blocks=int(cfg["resnet"]["num_blocks"]),
    ).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(args.device),
        enable_amp=bool(args.enable_amp),
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name=str(args.model_label),
        enable_profile=bool(args.profile),
    )


def _completed_prefix(log_dir: Path, seed_start: int, seed_key: int, total_games: int) -> int:
    existing = {path.name for path in log_dir.glob("*.json.gz")} if log_dir.exists() else set()
    completed = 0
    for offset in range(total_games):
        filename = f"{seed_start + offset}_{seed_key}.json.gz"
        if filename not in existing:
            break
        completed += 1
    expected = {f"{seed_start + offset}_{seed_key}.json.gz" for offset in range(total_games)}
    unexpected = sorted(existing - expected)
    trailing = sorted(
        name
        for name in existing & expected
        if int(name.split("_", 1)[0]) >= seed_start + completed
    )
    if unexpected or trailing:
        examples = (unexpected + trailing)[:5]
        raise ValueError(f"selfplay resume log set is not a contiguous prefix: {examples}")
    return completed


def _random_train_seat(seed: int, key: int) -> int:
    rotated_key = ((key << 13) & ((1 << 64) - 1)) | (key >> 51)
    return int((seed ^ rotated_key) % 4)


def _rewrite_aliases_for_batch(log_dir: Path, seed_start: int, seed_key: int, count: int, names: list[str]) -> None:
    """Fallback for an older locally installed libriichi extension.

    The current Rust extension writes aliases directly.  This keeps the Python
    runner usable if a stale `riichi.pyd` still exposes the older py_selfplay
    signature.
    """
    if len(names) != 4:
        raise ValueError("player names must contain exactly four aliases")
    for offset in range(count):
        seed = seed_start + offset
        path = log_dir / f"{seed}_{seed_key}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = handle.readlines()
        start_game = json.loads(lines[0])
        train_seat = _random_train_seat(seed, seed_key)
        aliases = [""] * 4
        opponent_index = 1
        for seat in range(4):
            if seat == train_seat:
                aliases[seat] = names[0]
            else:
                aliases[seat] = names[opponent_index]
                opponent_index += 1
        start_game["names"] = aliases
        lines[0] = json.dumps(start_game, ensure_ascii=False, separators=(",", ":")) + "\n"
        temp_path = path.with_name(path.name + ".tmp")
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            handle.writelines(lines)
        os.replace(temp_path, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    if int(args.games) <= 0:
        raise ValueError("--games must be positive")

    rank_points_profile, rank_points = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    completed = _completed_prefix(log_dir, int(args.seed_start), int(args.seed_key), int(args.games))
    if completed and not args.resume:
        raise ValueError(f"found {completed} existing logs; pass --resume to continue")

    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from libriichi.arena import OneVsThree  # noqa: PLC0415

    load_started = time.perf_counter()
    engine = _load_engine(args)
    model_load_time = time.perf_counter() - load_started
    print(f"loaded {args.model_label} in {model_load_time:.1f}s on {args.device}", flush=True)
    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))

    total_games = int(args.games)
    initial_completed = completed
    progress_every = int(args.progress_every or 0)
    requested_batch_size = int(getattr(args, "native_batch_games", 0) or 0)
    batch_size = requested_batch_size or (total_games if progress_every <= 0 else max(1, progress_every))
    if batch_size <= 0:
        raise ValueError("--native-batch-games must be positive when provided")
    all_seat_rank_counts = [completed * 4, completed * 4, completed * 4, completed * 4]
    started_at = time.monotonic()
    while completed < total_games:
        count = min(batch_size, total_games - completed)
        batch_seed_start = int(args.seed_start) + completed
        print(
            f"[selfplay] games {completed + 1}-{completed + count}/{total_games} "
            f"seed_start={batch_seed_start} device={args.device}",
            file=sys.stderr,
            flush=True,
        )
        try:
            batch_counts = list(
                env.py_selfplay(
                    engine=engine,
                    seed_start=(batch_seed_start, int(args.seed_key)),
                    game_count=count,
                    player_names=args.player_names,
                )
            )
        except TypeError:
            if args.player_names is None:
                raise
            print("[selfplay] installed libriichi lacks player_names; rewriting aliases in Python", flush=True)
            batch_counts = list(
                env.py_selfplay(
                    engine=engine,
                    seed_start=(batch_seed_start, int(args.seed_key)),
                    game_count=count,
                )
            )
            _rewrite_aliases_for_batch(
                log_dir,
                batch_seed_start,
                int(args.seed_key),
                count,
                list(args.player_names),
            )
        for index, value in enumerate(batch_counts):
            all_seat_rank_counts[index] += int(value)
        completed += count
        elapsed = time.monotonic() - started_at
        generated = completed - initial_completed
        rate = generated / elapsed if elapsed > 0 else 0.0
        remaining = (total_games - completed) / rate if rate > 0 else 0.0
        print(
            f"[selfplay] completed {completed}/{total_games}, elapsed={elapsed:.1f}s "
            f"eta={remaining:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    if args.defer_reports:
        print(f"[selfplay] log generation complete: {completed}/{total_games}; reports deferred", flush=True)
        return {"completed_games": completed, "log_dir": str(log_dir)}

    metrics = {}
    document = build_metrics_document(
        run={
            "kind": "native_selfplay",
            "backend": "libriichi.arena.OneVsThree.py_selfplay",
            "model": str(args.model),
            "model_label": str(args.model_label),
            "seed_start": int(args.seed_start),
            "seed_key": int(args.seed_key),
            "games": total_games,
            "native_batch_games": int(batch_size),
            "all_seat_rank_counts": all_seat_rank_counts,
            "device": str(args.device),
            "model_load_time_sec": model_load_time,
            "rank_points_profile": rank_points_profile,
            "rank_points_values": [float(value) for value in rank_points],
        },
        metrics={str(args.model_label): metrics},
        artifacts={"log_dir": str(log_dir)},
        rank_points_profile=rank_points_profile,
        rank_points_values=rank_points,
    )
    write_metrics(args.output_dir / "metrics.json", document)
    if bool(args.profile):
        profile_path = args.output_dir / "inference_profile.json"
        inference_profile = engine.profile_snapshot()
        profile_path.write_text(json.dumps(inference_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        document["artifacts"]["inference_profile_json"] = str(profile_path)
        document["inference_profile"] = inference_profile
        write_metrics(args.output_dir / "metrics.json", document)
    stat_report = write_stat_report(
        output_dir=args.output_dir,
        log_dir=log_dir,
        players={str(args.model_label): str(args.model_label)},
        mortal_root=args.mortal_root,
        rank_pts=rank_points,
        rank_points_profile=rank_points_profile,
    )
    train_raw = stat_report["players"][str(args.model_label)]["raw"]
    train_rank_counts = [int(train_raw[f"rank_{rank}"]) for rank in range(1, 5)]
    metrics = {str(args.model_label): summarize_rank_counts_with_references(train_rank_counts, rank_points=rank_points)}
    document["metrics"] = metrics
    document["artifacts"]["detailed_stats_json"] = str(args.output_dir / "detailed_stats.json")
    document["artifacts"]["detailed_stats_md"] = str(args.output_dir / "detailed_stats.md")
    document["detailed_stats_schema"] = stat_report["schema"]
    write_metrics(args.output_dir / "metrics.json", document)
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2), flush=True)
    return document


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
