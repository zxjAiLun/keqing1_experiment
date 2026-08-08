#!/usr/bin/env python3
"""Evaluate a GRP checkpoint on a fixed validation or holdout split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.run_grp_training import (  # noqa: E402
    GrpBatchStream,
    _evaluate,
    _load_file_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "holdout"), required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--data-seed", type=int, default=20260719)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    device = torch.device(args.device or config["grp"]["control"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    files = _load_file_list(config, args.split)
    dataset_config = config["grp"]["dataset"]
    stream = GrpBatchStream(
        files,
        batch_size=int(config["grp"]["control"].get("batch_size", 512)),
        file_batch_size=int(dataset_config.get("file_batch_size", 50)),
        seed=args.data_seed,
    )
    from model import GRP  # noqa: PLC0415

    model = GRP(**config["grp"]["network"]).to(device)
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    pts = torch.tensor(config.get("env", {}).get("pts", [6.0, 4.0, 2.0, 0.0]), dtype=torch.float64, device=device)
    metrics = _evaluate(model, stream, device=device, steps=args.steps, pts=pts)
    report = {
        "schema": "keqing.mortal.grp_checkpoint_eval.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_steps": int(state.get("steps", -1)),
        "split": args.split,
        "file_count": len(files),
        "steps": args.steps,
        "data_seed": args.data_seed,
        "metrics": metrics,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
