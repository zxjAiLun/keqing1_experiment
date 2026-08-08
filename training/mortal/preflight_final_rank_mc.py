#!/usr/bin/env python3
"""Validate the project-owned final-rank Monte Carlo reward on all training logs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from libriichi.dataset import GameplayLoader
from training.mortal.prepare_v2_population_mixed_warmstart import POOL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/experiments/model_pool_2026_07/V2_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/reward_preflight.json"),
    )
    parser.add_argument("--model-label", default="ext_mortal")
    parser.add_argument("--max-files", type=int, default=0, help="limit files for a smoke preflight")
    return parser.parse_args()


def iter_files(data_root: Path, max_files: int) -> list[Path]:
    files: list[Path] = []
    for pool_id, _ in POOL_SPECS:
        files.extend(sorted((data_root / pool_id / "logs").glob("*.json.gz")))
    files = sorted(files)
    return files if max_files <= 0 else files[:max_files]


def main() -> None:
    args = parse_args()
    files = iter_files(args.data_root, int(args.max_files))
    if not files:
        raise SystemExit("no training logs found")

    pts = np.asarray([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
    centered_pts = pts - pts.mean()
    loader = GameplayLoader(version=4, oracle=False, player_names=[str(args.model_label)], augmented=False)
    rank_counts: Counter[int] = Counter()
    target_counts: Counter[float] = Counter()
    decision_samples = 0
    train_games = 0
    malformed = []

    for start in range(0, len(files), 25):
        batch = files[start:start + 25]
        loaded_batch = loader.load_gz_log_files([str(path) for path in batch])
        for path, file_games in zip(batch, loaded_batch, strict=True):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    start_game = json.loads(next(handle))
                names = start_game.get("names")
                if not isinstance(names, list) or names.count(str(args.model_label)) != 1:
                    raise ValueError(f"expected exactly one {args.model_label} seat, found {names!r}")
                if len(file_games) != 1:
                    raise ValueError(f"expected one trainable game, found {len(file_games)}")
                game = file_games[0]
                actions = game.take_actions()
                grp = game.take_grp()
                player_id = int(game.take_player_id())
                rank_by_player = grp.take_rank_by_player()
                final_rank = int(rank_by_player[player_id])
                target = float(centered_pts[final_rank])
                rank_counts[final_rank] += 1
                target_counts[target] += len(actions)
                decision_samples += len(actions)
                train_games += 1
            except Exception as exc:  # noqa: BLE001
                malformed.append({"path": str(path), "error": str(exc)})

    target_sum = sum(float(target) * count for target, count in target_counts.items())
    target_sq_sum = sum(float(target) ** 2 * count for target, count in target_counts.items())
    target_mean = target_sum / decision_samples
    target_variance = max(0.0, target_sq_sum / decision_samples - target_mean**2)
    report = {
        "schema": "keqing.mortal.final_rank_mc_preflight.v1",
        "data_root": str(args.data_root),
        "model_label": str(args.model_label),
        "files_scanned": len(files),
        "train_games": train_games,
        "decision_samples": decision_samples,
        "final_rank_counts": {str(rank): int(rank_counts[rank]) for rank in range(4)},
        "target_sample_counts": {str(target): int(target_counts[target]) for target in sorted(target_counts)},
        "target_mean": target_mean,
        "target_std": float(target_variance**0.5),
        "reward_nonzero_rate": sum(target_counts.values()) / decision_samples if decision_samples else 0.0,
        "malformed_count": len(malformed),
        "malformed": malformed[:20],
    }
    report["passed"] = (
        len(malformed) == 0
        and train_games == len(files)
        and decision_samples > 0
        and set(target_counts) <= {-3.0, -1.0, 1.0, 3.0}
        and report["reward_nonzero_rate"] == 1.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("files_scanned", "train_games", "decision_samples", "target_mean", "target_std", "reward_nonzero_rate", "passed")}, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("final_rank_mc preflight failed")


if __name__ == "__main__":
    main()
