#!/usr/bin/env python3
"""Train a randomly initialized Mortal-native student on external teacher labels.

Phase 1 of the 2026-09-07 mainline shift.  This trainer is deliberately a new,
separate entry point (not a renamed old runner):

- Student: fresh random-init ``Brain(version=4, 192, 40)`` + ``DQN(version=4)``
  (K0-family capacity).  No K0 weights, no K0 Adam, no external weights.
- Objective: pure legal-action policy loss only — cross entropy between the
  student's masked Q logits and the teacher's greedy hard label.  No MC return,
  no GRP, no next-rank auxiliary head, no K0 policy anchoring, no value head loss.
- Data: ``.npz`` shards produced by ``relabel_ext_teacher.py`` (all-perspective
  states, teacher greedy labels, behavior action kept for diagnostics only).
  Only the holdout split is used for evaluation metrics.
- Resume: the checkpoint stores the shard/row cursor and Python/NumPy/Torch RNG
  states so an interrupted run continues the exact sample stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fresh Mortal-native student on external teacher labels")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="directory containing train_*.npz/holdout_*.npz + manifest.json from relabel_ext_teacher.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--conv-channels", type=int, default=192)
    parser.add_argument("--num-blocks", type=int, default=40)
    parser.add_argument("--holdout-every", type=int, default=2000, help="steps between holdout evaluations (0 disables)")
    parser.add_argument("--holdout-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers for shard reading (0: main process)")
    parser.add_argument("--enable-amp", action="store_true")
    return parser.parse_args()


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "keqing.mortal.ext_relabel.v1":
        raise ValueError(f"unexpected dataset manifest schema: {manifest.get('schema')}")
    return manifest


def _shard_paths(dataset_dir: Path, split: str) -> list[Path]:
    paths = sorted(dataset_dir.glob(f"{split}_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {split}_*.npz shards in {dataset_dir}")
    return paths


class ShardStream:
    """Deterministic row stream over the relabeled .npz shards.

    Rows are yielded in (shard, row) order.  The stream is chunked by
    ``chunk_rows`` so the caller controls memory; the cursor is
    ``(shard_index, row_index)`` and resume replays the skip exactly.
    """

    def __init__(self, shard_paths: list[Path], seed: int):
        self.shard_paths = shard_paths
        self.rng = random.Random(seed)
        self._open_index: int | None = None
        self._arrays: dict[str, np.ndarray] | None = None

    def _ensure_open(self, shard_index: int) -> None:
        if self._open_index == shard_index:
            return
        if self._open_index is not None:
            self._close()
        with np.load(self.shard_paths[shard_index]) as payload:
            self._arrays = {key: payload[key] for key in (
                "obs", "mask", "teacher_action", "behavior_action", "teacher_q",
                "pool", "player_id", "file_row",
            )}
        self._open_index = shard_index

    def _close(self) -> None:
        self._arrays = None
        self._open_index = None

    def read_chunk(self, shard_index: int, row_index: int, chunk_rows: int) -> tuple[dict[str, np.ndarray], int]:
        """Read up to ``chunk_rows`` rows starting at (shard_index, row_index).

        Returns the payload and the next (shard_index, row_index) cursor.  When
        the cursor passes the end of the last shard it wraps to (0, 0).
        """
        shard_count = len(self.shard_paths)
        if shard_index < 0 or shard_index >= shard_count:
            raise IndexError(f"shard index out of range: {shard_index}")
        self._ensure_open(shard_index)
        total = len(self._arrays["obs"])
        if row_index >= total:
            raise IndexError(f"row index out of range: {row_index} >= {total}")
        take = min(chunk_rows, total - row_index)
        payload = {key: value[row_index : row_index + take] for key, value in self._arrays.items()}
        next_shard, next_row = shard_index, row_index + take
        if next_row >= total:
            if shard_index + 1 >= shard_count:
                next_shard, next_row = 0, 0
                self._close()
            else:
                next_shard, next_row = shard_index + 1, 0
        return payload, (next_shard, next_row)


def _rows_from_manifest(manifest: dict[str, Any], split: str, shard_paths: list[Path]) -> int:
    """Row count for a split, read from the manifest instead of decompressing
    every shard.  Falls back to summing ``shard_rows`` by shard filename order;
    only if the manifest lacks both does it scan shards (loud, not silent)."""
    split_info = manifest.get("splits", {}).get(split, {})
    shard_rows = split_info.get("shard_rows")
    if shard_rows is not None and len(shard_rows) == len(shard_paths):
        return int(sum(shard_rows))
    flushed = split_info.get("flushed_rows")
    if flushed is not None and int(split_info.get("flushed_shards", -1)) == len(shard_paths):
        return int(flushed)
    raise RuntimeError(
        f"manifest does not record a usable row count for split {split!r} "
        f"(shard_rows={shard_rows!r}, shards on disk={len(shard_paths)}); "
        "regenerate the label cache manifest"
    )


def _build_models(*, version: int, conv_channels: int, num_blocks: int, device: torch.device, mortal_root: Path):
    mortal_python_dir = (mortal_root / "mortal").resolve()
    import sys

    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from model import Brain, DQN  # noqa: PLC0415

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).to(device)
    dqn = DQN(version=version).to(device)
    return mortal, dqn


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.holdout_every < 0 or args.holdout_batches < 0:
        raise ValueError("--holdout-every/--holdout-batches must be non-negative")

    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "student.pth"
    manifest = _load_manifest(args.dataset_dir.resolve())
    train_shards = _shard_paths(args.dataset_dir.resolve(), "train")
    holdout_shards = _shard_paths(args.dataset_dir.resolve(), "holdout")

    logging.info("train shards: %d, holdout shards: %d", len(train_shards), len(holdout_shards))
    train_rows = _rows_from_manifest(manifest, "train", train_shards)
    holdout_rows = _rows_from_manifest(manifest, "holdout", holdout_shards)
    logging.info("train rows: %s, holdout rows: %s (from manifest)", f"{train_rows:,}", f"{holdout_rows:,}")

    version = 4
    mortal, dqn = _build_models(
        version=version,
        conv_channels=args.conv_channels,
        num_blocks=args.num_blocks,
        device=device,
        mortal_root=args.mortal_root,
    )
    parameters = [p for p in mortal.parameters()] + [p for p in dqn.parameters()]
    optimizer = optim.AdamW(
        parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Lambda is a pure multiplier on the optimizer's base LR: linear warm-up
    # from 0 to 1, then constant 1.  The peak LR itself lives in the optimizer.
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    from torch.optim.lr_scheduler import LambdaLR

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(args.enable_amp))

    stream = ShardStream(train_shards, seed=int(args.seed))
    holdout_stream = ShardStream(holdout_shards, seed=int(args.seed) + 1)

    steps = 0
    cursor = (0, 0)
    python_rng_state = None
    torch_rng_state = None
    cuda_rng_states = None
    history: list[dict[str, Any]] = []

    if state_file.exists():
        state = torch.load(state_file, map_location=device, weights_only=False)
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        steps = int(state["steps"])
        cursor = (int(state["cursor"][0]), int(state["cursor"][1]))
        history = list(state.get("history", []))
        python_rng_state = state.get("python_rng_state")
        torch_rng_state = state.get("torch_rng_state")
        cuda_rng_states = state.get("cuda_rng_states")
        logging.info("resumed checkpoint: steps=%s cursor=%s", steps, cursor)
        if python_rng_state is not None:
            random.setstate(python_rng_state)
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.detach().cpu())
        if cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.detach().cpu() for s in cuda_rng_states])
    else:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))

    writer = SummaryWriter(str(output_dir / "tb_student"))

    contract = {
        "schema": "keqing.mortal.student_policy_v1",
        "student": {
            "init": "random",
            "version": version,
            "conv_channels": int(args.conv_channels),
            "num_blocks": int(args.num_blocks),
        },
        "objective": "masked_q_cross_entropy_teacher_greedy",
        "dataset_manifest": {
            "path": str(args.dataset_dir.resolve()),
            "teacher_sha256": manifest.get("teacher_sha256"),
            "totals": manifest.get("totals"),
            "holdout_ratio": manifest.get("holdout_ratio"),
            "train_rows": int(train_rows),
            "holdout_rows": int(holdout_rows),
        },
        "optim": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_steps": int(args.warmup_steps),
            "batch_size": int(args.batch_size),
            "enable_amp": bool(args.enable_amp),
        },
        "git_commit": _git_revision(_REPO_ROOT),
        "seed": int(args.seed),
    }
    (output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def save_checkpoint() -> None:
        checkpoint = {
            "mortal": mortal.state_dict(),
            "current_dqn": dqn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "steps": steps,
            "cursor": list(cursor),
            "history": history,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_contract": contract,
        }
        tmp_path = state_file.with_name(state_file.name + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(state_file)
        logging.info("saved checkpoint: %s steps=%s", state_file, steps)

    @torch.inference_mode()
    def evaluate_holdout() -> dict[str, float]:
        mortal.eval()
        dqn.eval()
        total_ce = 0.0
        total_agree = 0.0
        total_rows = 0
        # Sequential scan over holdout batches (deterministic, fixed count).
        hcursor = (0, 0)
        for _ in range(int(args.holdout_batches)):
            payload, hcursor = holdout_stream.read_chunk(hcursor[0], hcursor[1], int(args.batch_size))
            obs = torch.as_tensor(payload["obs"], dtype=torch.float32, device=device)
            mask = torch.as_tensor(payload["mask"], dtype=torch.bool, device=device)
            teacher_action = torch.as_tensor(payload["teacher_action"], dtype=torch.int64, device=device)
            with torch.autocast(device.type, enabled=bool(args.enable_amp)):
                phi = mortal(obs)
                q_out = dqn(phi, mask)
            log_probs = q_out.float().log_softmax(dim=-1)
            ce = -log_probs.gather(1, teacher_action.unsqueeze(1)).squeeze(1)
            agree = (q_out.float().argmax(dim=-1) == teacher_action).to(torch.float32)
            total_ce += float(ce.sum().cpu())
            total_agree += float(agree.sum().cpu())
            total_rows += int(obs.shape[0])
        mortal.train()
        dqn.train()
        return {
            "holdout_ce": total_ce / max(1, total_rows),
            "holdout_agreement": total_agree / max(1, total_rows),
            "holdout_rows": float(total_rows),
        }

    window_ce = 0.0
    window_agree = 0.0
    window_rows = 0
    window_count = 0
    started = time.time()

    while steps < args.target_steps:
        payload, next_cursor = stream.read_chunk(cursor[0], cursor[1], int(args.batch_size))
        if len(payload["obs"]) < int(args.batch_size):
            # Partial chunks also occur at plain shard boundaries; only the
            # dataset end (wrap to (0, 0)) needs padding from the beginning.
            if next_cursor == (0, 0) and cursor != (0, 0):
                remaining = int(args.batch_size) - len(payload["obs"])
                payload2, _ = stream.read_chunk(0, 0, remaining)
                payload = {key: np.concatenate([payload[key], payload2[key]]) for key in payload}
                next_cursor = (0, remaining)
            else:
                # Shard-boundary tail: keep the short batch (last partial batch
                # of a shard).  This is deterministic and resume-safe.
                pass
        obs = torch.as_tensor(payload["obs"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(payload["mask"], dtype=torch.bool, device=device)
        teacher_action = torch.as_tensor(payload["teacher_action"], dtype=torch.int64, device=device)
        if not bool(mask[torch.arange(len(teacher_action)), teacher_action].all().item()):
            raise RuntimeError("teacher label outside legal mask — dataset corruption")

        with torch.autocast(device.type, enabled=bool(args.enable_amp)):
            phi = mortal(obs)
            q_out = dqn(phi, mask)
            log_probs = q_out.log_softmax(dim=-1)
            ce = -log_probs.gather(1, teacher_action.unsqueeze(1)).squeeze(1)
            loss = ce.mean()
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError(f"non-finite loss at step {steps + 1}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.inference_mode():
            agree = (q_out.float().argmax(dim=-1) == teacher_action).to(torch.float32).mean().item()
        window_ce += float(loss.detach().cpu())
        window_agree += agree
        window_count += 1
        cursor = next_cursor
        steps += 1

        if steps % int(args.log_every) == 0 or steps >= args.target_steps:
            avg_ce = window_ce / max(1, window_count)
            avg_agree = window_agree / max(1, window_count)
            rate = steps / max(1e-9, time.time() - started)
            logging.info(
                "steps=%s/%s train_ce=%.4f train_agreement=%.4f lr=%.3e rows/s=%.0f cursor=%s",
                steps,
                args.target_steps,
                avg_ce,
                avg_agree,
                scheduler.get_last_lr()[0],
                rate * int(args.batch_size),
                cursor,
            )
            writer.add_scalar("train/ce", avg_ce, steps)
            writer.add_scalar("train/agreement", avg_agree, steps)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)
            writer.flush()
            window_ce = 0.0
            window_agree = 0.0
            window_count = 0

        if int(args.holdout_every) > 0 and (steps % int(args.holdout_every) == 0 or steps >= args.target_steps):
            metrics = evaluate_holdout()
            metrics["steps"] = steps
            history.append(metrics)
            logging.info(
                "holdout: steps=%s ce=%.4f agreement=%.4f rows=%.0f",
                steps, metrics["holdout_ce"], metrics["holdout_agreement"], metrics["holdout_rows"],
            )
            writer.add_scalar("holdout/ce", metrics["holdout_ce"], steps)
            writer.add_scalar("holdout/agreement", metrics["holdout_agreement"], steps)

        if int(args.save_every) > 0 and steps % int(args.save_every) == 0:
            save_checkpoint()

    save_checkpoint()
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    writer.close()
    return {"steps": steps, "state_file": str(state_file), "history": history}


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
