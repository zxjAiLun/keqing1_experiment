#!/usr/bin/env python3
"""Run a four-engine Mortal/libriichi native arena with seat rotation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

# Import the installed native arena before adding third_party/Mortal/mortal to
# sys.path; that directory contains a legacy libriichi.pyd which shadows the
# current Python package and does not expose the FourPlayer arena wrapper.
from libriichi.arena import FourPlayer as NativeFourPlayer

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


_ANCHOR_70K = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
_V2_CANDIDATE = Path("artifacts/experiments/model_pool_2026_07/V2_population_mixed_v4_warmstart_2026_07/checkpoints/mortal_74000.pth")
_EXT_MORTAL = Path("artifacts/external_mortal_20240308_best_min.pth")
DEFAULT_MODELS = {
    "ext_mortal": _EXT_MORTAL,
    "70k": _ANCHOR_70K,
    "candidate": _V2_CANDIDATE if _V2_CANDIDATE.exists() else _ANCHOR_70K,
    "ext_mortal_control": _EXT_MORTAL,
}


def _parse_model_specs(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return dict(DEFAULT_MODELS)
    models: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"model spec must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"model spec must be LABEL=PATH, got: {spec}")
        models[label] = Path(path)
    if len(models) != 4:
        raise ValueError(f"exactly four models are required, got {len(models)}")
    return models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native four-player Mortal arena")
    parser.add_argument("--model", action="append", help="LABEL=CHECKPOINT. Repeat exactly four times.")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/four_player_native"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed-start", type=int, default=600000)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--games", type=int, default=1, help="number of hanchans to run")
    parser.add_argument(
        "--seat-mode",
        choices=("random", "rotation"),
        default="random",
        help="random seats per hanchan, or four fixed rotation splits per seed",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="emit progress every N hanchans")
    parser.add_argument(
        "--native-batch-games",
        type=int,
        default=0,
        help="hanchans per Rust arena batch; 0 preserves the progress-sized batch behavior",
    )
    parser.add_argument("--profile", action="store_true", help="record per-engine inference batch and timing telemetry")
    parser.add_argument("--resume", action="store_true", help="resume from existing native logs in output-dir/logs")
    parser.add_argument("--enable-amp", action="store_true")
    add_rank_point_args(parser)
    return parser.parse_args()


def _load_engine(
    *,
    label: str,
    state_file: Path,
    mortal_root: Path,
    device: str,
    enable_amp: bool,
    enable_profile: bool,
) -> Any:
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    from engine import MortalEngine  # noqa: PLC0415
    from model import Brain, DQN  # noqa: PLC0415

    state = torch.load(state_file, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = int(cfg["control"].get("version", 4))
    conv_channels = int(cfg["resnet"]["conv_channels"])
    num_blocks = int(cfg["resnet"]["num_blocks"])

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(device),
        enable_amp=bool(enable_amp),
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name=label,
        enable_profile=enable_profile,
    )


def _load_engines(
    *,
    models: dict[str, Path],
    mortal_root: Path,
    device: str,
    enable_amp: bool,
    enable_profile: bool,
) -> tuple[list[str], list[Any], dict[str, float]]:
    labels = list(models)
    engines: list[Any] = []
    load_times: dict[str, float] = {}
    for label in labels:
        started = time.perf_counter()
        engines.append(
            _load_engine(
                label=label,
                state_file=models[label],
                mortal_root=mortal_root,
                device=device,
                enable_amp=enable_amp,
                enable_profile=enable_profile,
            )
        )
        load_times[label] = time.perf_counter() - started
        print(f"loaded {label:<10} in {load_times[label]:.1f}s", flush=True)
    return labels, engines, load_times


def _rank_counts_from_stat_report(
    stat_report: dict[str, Any],
    labels: list[str],
) -> tuple[dict[str, list[int]], int]:
    rank_counts: dict[str, list[int]] = {}
    completed_games = 0
    for label in labels:
        raw = stat_report["players"][label]["raw"]
        counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
        rank_counts[label] = counts
        completed_games = max(completed_games, int(raw["game"]))
    return rank_counts, completed_games


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")

    models = _parse_model_specs(getattr(args, "model", None))
    rank_points_profile, rank_points = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    labels, engines, model_load_times = _load_engines(
        models=models,
        mortal_root=args.mortal_root,
        device=str(args.device),
        enable_amp=bool(args.enable_amp),
        enable_profile=bool(args.profile),
    )

    env = NativeFourPlayer(disable_progress_bar=True, log_dir=str(log_dir))
    total_games = int(args.games)
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    requested_batch_size = int(getattr(args, "native_batch_games", 0) or 0)
    batch_size = requested_batch_size or (total_games if progress_every <= 0 else max(1, progress_every))
    if batch_size <= 0:
        raise ValueError("--native-batch-games must be positive when provided")
    rank_counts = {label: [0, 0, 0, 0] for label in labels}
    completed = 0
    if bool(getattr(args, "resume", False)) and log_dir.exists():
        existing_stat_report = write_stat_report(
            output_dir=output_dir,
            log_dir=log_dir,
            players={label: label for label in labels},
            mortal_root=args.mortal_root,
            rank_pts=rank_points,
            rank_points_profile=rank_points_profile,
        )
        rank_counts, completed = _rank_counts_from_stat_report(existing_stat_report, labels)
        completed = min(completed, total_games)
        if completed:
            print(f"resuming from {completed}/{total_games} existing games in {log_dir}", flush=True)
    started_at = time.monotonic()
    while completed < total_games:
        count = min(batch_size, total_games - completed)
        batch_seed_start = int(args.seed_start) + completed
        if progress_every > 0:
            print(
                (
                    f"[four_player_native] games {completed + 1}-{completed + count}/"
                    f"{total_games} start={batch_seed_start} device={args.device} "
                    f"seat_mode={args.seat_mode}"
                ),
                file=sys.stderr,
                flush=True,
            )

        if str(args.seat_mode) == "random":
            batch_rank_counts = env.py_vs_py_random_seats(
                engines[0],
                engines[1],
                engines[2],
                engines[3],
                (batch_seed_start, int(args.seed_key)),
                count,
            )
            batch_games = count
        else:
            if count % 4 != 0:
                raise ValueError("--seat-mode rotation requires --games/progress chunk sizes divisible by 4")
            batch_rank_counts = env.py_vs_py(
                engines[0],
                engines[1],
                engines[2],
                engines[3],
                (batch_seed_start, int(args.seed_key)),
                count // 4,
            )
            batch_games = count
        for label, counts in zip(labels, batch_rank_counts, strict=True):
            for i, value in enumerate(counts):
                rank_counts[label][i] += int(value)

        completed += batch_games
        if progress_every > 0:
            elapsed = time.monotonic() - started_at
            seeds_per_sec = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total_games - completed) / seeds_per_sec if seeds_per_sec > 0 else 0.0
            print(
                (
                    f"[four_player_native] completed {completed}/{total_games} games, "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s rank_counts={rank_counts}"
                ),
                file=sys.stderr,
                flush=True,
            )

    metrics = {
        label: summarize_rank_counts_with_references(counts, rank_points=rank_points)
        for label, counts in rank_counts.items()
    }
    document = build_metrics_document(
        run={
            "kind": "four_player_native",
            "backend": "libriichi.arena.FourPlayer",
            "models": {label: str(path) for label, path in models.items()},
            "seed_start": int(args.seed_start),
            "seed_key": int(args.seed_key),
            "games": int(args.games),
            "seat_mode": str(args.seat_mode),
            "native_batch_games": int(batch_size),
            "device": str(args.device),
            "rank_points_profile": rank_points_profile,
            "rank_points_values": [float(value) for value in rank_points],
            "model_load_time_sec": {key: float(value) for key, value in model_load_times.items()},
        },
        metrics=metrics,
        artifacts={"log_dir": str(log_dir)},
        rank_points_profile=rank_points_profile,
        rank_points_values=rank_points,
    )
    write_metrics(output_dir / "metrics.json", document)

    if bool(args.profile):
        inference_profile = {label: engine.profile_snapshot() for label, engine in zip(labels, engines, strict=True)}
        profile_path = output_dir / "inference_profile.json"
        profile_path.write_text(json.dumps(inference_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        document["artifacts"]["inference_profile_json"] = str(profile_path)
        document["inference_profile"] = inference_profile

    stat_report = write_stat_report(
        output_dir=output_dir,
        log_dir=log_dir,
        players={label: label for label in labels},
        mortal_root=args.mortal_root,
        rank_pts=rank_points,
        rank_points_profile=rank_points_profile,
    )
    document["artifacts"]["detailed_stats_json"] = str(output_dir / "detailed_stats.json")
    document["artifacts"]["detailed_stats_md"] = str(output_dir / "detailed_stats.md")
    document["detailed_stats_schema"] = stat_report["schema"]
    write_metrics(output_dir / "metrics.json", document)
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2), flush=True)
    return document


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
