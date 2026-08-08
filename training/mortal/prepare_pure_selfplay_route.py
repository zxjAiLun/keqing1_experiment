#!/usr/bin/env python3
"""Freeze the pure-ext_mortal selfplay data-route contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT_ID = "S0_pure_ext_selfplay_6000h"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
DEFAULT_MODEL = Path("artifacts/external_mortal_20240308_best_min.pth")
TRAIN_LABEL = "train_ext"
PLAYER_NAMES = ("train_ext", "opp_ext_1", "opp_ext_2", "opp_ext_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--native-batch-games", type=int, default=100)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.games <= 0 or args.native_batch_games <= 0:
        raise ValueError("games and native-batch-games must be positive")
    output_dir = args.output_root / EXPERIMENT_ID
    log_dir = output_dir / "logs"
    manifest = {
        "schema": "keqing.mortal.pure_selfplay_route.v1",
        "experiment_id": EXPERIMENT_ID,
        "model": str(args.model.resolve()),
        "model_label": TRAIN_LABEL,
        "player_names": list(PLAYER_NAMES),
        "trainable_player_name": TRAIN_LABEL,
        "seed_start": int(args.seed_start),
        "seed_key": int(args.seed_key),
        "games": int(args.games),
        "native_batch_games": int(args.native_batch_games),
        "one_log_per_seed": True,
        "seat_schedule": "deterministic uniform train_ext alias rotation derived from seed/key",
        "opponent_population": "same checkpoint as train_ext",
        "training_perspectives_per_hanchan": 1,
        "log_dir": str(log_dir.resolve()),
        "file_index": str((output_dir / "file_index.pth").resolve()),
        "generation_command": [
            "uv", "run", "--no-sync", "python", "training/mortal/selfplay_native.py",
            "--model", str(args.model),
            "--model-label", TRAIN_LABEL,
            "--player-names", *PLAYER_NAMES,
            "--output-dir", str(output_dir),
            "--device", "cuda", "--require-cuda",
            "--seed-start", str(args.seed_start), "--seed-key", str(args.seed_key),
            "--games", str(args.games), "--native-batch-games", str(args.native_batch_games),
            "--progress-every", str(args.native_batch_games),
        ],
        "audit_requirements": {
            "files": int(args.games),
            "trainable_perspectives": int(args.games),
            "malformed": 0,
            "seed_overlap": 0,
            "exact_duplicate_rate": 0.0,
            "behavior_action_legal_rate": 1.0,
        },
        "status": "planned",
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.write:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
