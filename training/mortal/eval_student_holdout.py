#!/usr/bin/env python3
"""Full-holdout evaluation of a phase-1 student checkpoint.

The trainer's built-in holdout metric samples 24 uniformly spaced shards (one
batch each) so it stays cheap inside the training loop.  This script is the
one-off complement requested for a milestone checkpoint: it walks every
holdout shard, every row, and reports overall plus per-pool cross-entropy and
teacher-greedy agreement.  It is evaluation-only: no gradients, no weight
updates, no shard writes.

Reference line for interpretation: ``behavior_vs_teacher_agreement`` is the
agreement between the logged (behavior) action and the teacher label.  The
student is trained toward the teacher label, so this bounds how much the
student could agree with the recorded play by imitation alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.four_player_native import _model_dimensions  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-holdout evaluation of a student checkpoint")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON result path")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    return parser.parse_args()


def _build_models(*, version: int, conv_channels: int, num_blocks: int, device: torch.device, mortal_root: Path):
    import sys as _sys

    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in _sys.path:
        _sys.path.insert(0, str(mortal_python_dir))
    from model import Brain, DQN  # noqa: PLC0415

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).to(device)
    dqn = DQN(version=version).to(device)
    return mortal, dqn


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.resolve()
    device = torch.device(args.device)

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    pool_names = list(manifest.get("pools", {}))
    if not pool_names:
        raise RuntimeError("dataset manifest does not record pool identities")
    expected_rows = int(manifest["splits"]["holdout"]["flushed_rows"])

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    version, conv_channels, num_blocks = _model_dimensions(state)
    mortal, dqn = _build_models(
        version=version,
        conv_channels=conv_channels,
        num_blocks=num_blocks,
        device=device,
        mortal_root=args.mortal_root,
    )
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    mortal.eval()
    dqn.eval()

    names = ["overall", *pool_names]
    stats = {name: {"ce": 0.0, "agree": 0.0, "behavior_agree": 0.0, "rows": 0} for name in names}
    shards = sorted(dataset_dir.glob("holdout_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no holdout shards in {dataset_dir}")

    with torch.inference_mode():
        for shard_path in shards:
            with np.load(shard_path) as shard:
                rows = len(shard["obs"])
                for start in range(0, rows, args.batch_size):
                    stop = min(start + args.batch_size, rows)
                    obs = torch.as_tensor(shard["obs"][start:stop], dtype=torch.float32, device=device)
                    mask = torch.as_tensor(shard["mask"][start:stop], dtype=torch.bool, device=device)
                    teacher = torch.as_tensor(shard["teacher_action"][start:stop], dtype=torch.int64, device=device)
                    behavior = torch.as_tensor(shard["behavior_action"][start:stop], dtype=torch.int64, device=device)
                    pool_ids = torch.as_tensor(shard["pool"][start:stop], dtype=torch.int64, device=device)
                    q_out = dqn(mortal(obs), mask).float()
                    log_probs = q_out.log_softmax(dim=-1)
                    ce = -log_probs.gather(1, teacher.unsqueeze(1)).squeeze(1)
                    agree = (q_out.argmax(dim=-1) == teacher).to(torch.float32)
                    behavior_agree = (behavior == teacher).to(torch.float32)
                    buckets = [("overall", torch.ones_like(pool_ids, dtype=torch.bool))]
                    buckets += [
                        (name, pool_ids == index) for index, name in enumerate(pool_names)
                    ]
                    for name, selected in buckets:
                        count = int(selected.sum().item())
                        if not count:
                            continue
                        entry = stats[name]
                        entry["ce"] += float(ce[selected].sum().cpu())
                        entry["agree"] += float(agree[selected].sum().cpu())
                        entry["behavior_agree"] += float(behavior_agree[selected].sum().cpu())
                        entry["rows"] += count

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": int(state["steps"]),
        "dataset": str(dataset_dir),
        "holdout_shards": len(shards),
        "expected_rows": expected_rows,
        "pools": {},
    }
    for name in names:
        entry = stats[name]
        rows = entry["rows"]
        result["pools"][name] = {
            "rows": rows,
            "ce": entry["ce"] / rows if rows else None,
            "agreement": entry["agree"] / rows if rows else None,
            "behavior_vs_teacher_agreement": entry["behavior_agree"] / rows if rows else None,
        }
    total_rows = stats["overall"]["rows"]
    result["rows"] = total_rows
    result["rows_match_manifest"] = total_rows == expected_rows

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
