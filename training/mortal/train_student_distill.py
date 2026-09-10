#!/usr/bin/env python3
"""Short soft-target distillation stage on top of an archived student checkpoint.

This is a *new stage*, not a continuation of the hard-label run, so its resume
semantics differ from ``train_student_policy.py`` deliberately:

- initialization copies the parent model weights (and nothing else): the
  optimizer, scheduler, scaler, and step counter all start fresh, so a new
  learning rate cannot be smuggled in by copying a checkpoint and editing CLI
  flags;
- the parent's data position (shuffled shard order + row cursor) is reused, so
  the stage trains on the parent's next batches rather than restarting the epoch;
- stage weights are written to dedicated files (step 0 / 2500 / 5000) that are
  never overwritten; the rolling state file is only for resuming this stage;
- ``--resume`` restores this stage's own full state and fails closed unless the
  recorded recipe (parent hash, temperature, LR, schedule, batch, seed) matches.

Objective (pure teacher soft targets, no hard CE, no auxiliary heads):

    p_teacher      = softmax(teacher_q / T)      over legal actions only
    log_p_student  = log_softmax(student_q)      over legal actions only
    loss           = mean_over_states KL(p_teacher || p_student)

Illegal actions are masked out of the sum as well as the softmax, so the
``0 * -inf`` term cannot produce NaN.  The KL value is not comparable with the
hard-label CE from the parent stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.train_student_policy import (  # noqa: E402
    ShardStream,
    _build_models,
    _load_manifest,
    _rows_from_manifest,
    _shard_paths,
    _uniform_shard_indices,
)

_SAMPLE_FIELDS = ("obs", "mask", "teacher_q", "teacher_action", "pool")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short teacher soft-target distillation stage")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stage-saves", type=int, nargs="*", default=[0, 2500, 5000])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--seed", type=int, default=20260908, help="parent's shard-shuffle seed")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--state-every", type=int, default=500, help="rolling resume-state interval")
    parser.add_argument("--monitor-every", type=int, default=1000)
    parser.add_argument("--monitor-batches", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _kl_per_row(*, student_q: torch.Tensor, teacher_q: torch.Tensor, legal: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per-state KL(p_teacher || p_student) over legal actions."""
    if not bool(legal.any(dim=1).all().item()):
        raise RuntimeError("row with no legal action")
    masked_teacher = teacher_q.float().masked_fill(~legal, -torch.inf)
    log_p_teacher = torch.log_softmax(masked_teacher / float(temperature), dim=-1)
    masked_student = student_q.float().masked_fill(~legal, -torch.inf)
    log_p_student = torch.log_softmax(masked_student, dim=-1)
    terms = log_p_teacher.exp() * (log_p_teacher - log_p_student)
    # 0 * -inf on illegal entries would be NaN; mask before summing.
    terms = torch.where(legal, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1)


def _kl_loss(*, student_q: torch.Tensor, teacher_q: torch.Tensor, legal: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(p_teacher || p_student) over legal actions, averaged over states."""
    return _kl_per_row(student_q=student_q, teacher_q=teacher_q, legal=legal, temperature=temperature).mean()


def _check_resume(recipe_path: Path, recipe: dict[str, Any], *, state_exists: bool, resume: bool) -> None:
    """Fail closed unless this stage's recorded recipe matches exactly."""
    if not state_exists:
        return
    if not resume:
        raise RuntimeError(f"{recipe_path.parent / 'distill_state.pth'} exists; pass --resume to continue this stage or use a new output dir")
    recorded = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recorded != recipe:
        raise RuntimeError(
            f"refusing --resume: recipe differs from {recipe_path} "
            "(parent hash, temperature, LR, schedule, batch, or seed changed)"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be positive")
    if float(args.temperature) <= 0:
        raise ValueError("--temperature must be positive")

    device = torch.device(args.device)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "distill_state.pth"
    recipe_path = output_dir / "distill_recipe.json"

    manifest = _load_manifest(dataset_dir)
    train_shards = _shard_paths(dataset_dir, "train")
    holdout_shards = _shard_paths(dataset_dir, "holdout")
    train_rows = _rows_from_manifest(manifest, "train", train_shards)
    holdout_rows = _rows_from_manifest(manifest, "holdout", holdout_shards)

    parent_state = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    parent_contract = parent_state.get("training_contract", {})
    student = parent_contract.get("student", {})
    version = int(student.get("version", 4))
    conv_channels = int(student.get("conv_channels", 192))
    num_blocks = int(student.get("num_blocks", 40))
    parent_cursor = (int(parent_state["cursor"][0]), int(parent_state["cursor"][1]))
    parent_steps = int(parent_state["steps"])

    recipe = {
        "schema": "keqing.mortal.student_distill.v1",
        "objective": "kl_teacher_soft_target_only",
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_sha256": _sha256_file(args.parent_checkpoint.resolve()),
        "parent_steps": parent_steps,
        "parent_epoch_position": {"cursor": [parent_cursor[0], parent_cursor[1]], "shuffle_seed": int(args.seed)},
        "student": {"version": version, "conv_channels": conv_channels, "num_blocks": num_blocks},
        "optim": {
            "fresh_optimizer": True,
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_steps": int(args.warmup_steps),
            "grad_clip": 1.0,
            "schedule": "linear_warmup_then_constant",
            "batch_size": int(args.batch_size),
            "target_steps": int(args.target_steps),
            "enable_amp": False,
        },
        "temperature": {"teacher": float(args.temperature), "student": 1.0},
        "dataset": {
            "path": str(dataset_dir),
            "teacher_sha256": manifest.get("teacher_sha256"),
            "train_rows": int(train_rows),
            "holdout_rows": int(holdout_rows),
        },
        "git_commit": _git_revision(),
    }

    _check_resume(recipe_path, recipe, state_exists=state_file.exists(), resume=bool(args.resume))
    if state_file.exists():
        logging.info("resume: recipe verified against %s", recipe_path)
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")

    mortal, dqn = _build_models(
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
        device=device, mortal_root=args.mortal_root,
    )
    mortal.load_state_dict(parent_state["mortal"])
    dqn.load_state_dict(parent_state["current_dqn"])

    parameters = [p for p in mortal.parameters()] + [p for p in dqn.parameters()]
    optimizer = optim.AdamW(parameters, lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step: int) -> float:
        if step < int(args.warmup_steps):
            return step / max(1, int(args.warmup_steps))
        return 1.0

    from torch.optim.lr_scheduler import LambdaLR

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    # Same shard order as the parent stage, continued from the parent's cursor.
    import random as _random

    _random.Random(int(args.seed)).shuffle(train_shards)
    stream = ShardStream(train_shards, seed=int(args.seed))
    monitor_shards = [holdout_shards[i] for i in _uniform_shard_indices(len(holdout_shards), int(args.monitor_batches))]
    pool_names = list(manifest.get("pools", {}))

    steps = 0
    cursor = parent_cursor
    history: list[dict[str, Any]] = []
    if state_file.exists():
        state = torch.load(state_file, map_location=device, weights_only=False)
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        steps = int(state["steps"])
        cursor = (int(state["cursor"][0]), int(state["cursor"][1]))
        history = list(state.get("history", []))
        logging.info("resumed stage: steps=%s cursor=%s", steps, cursor)

    def save_stage(path: Path) -> None:
        torch.save(
            {
                "mortal": mortal.state_dict(),
                "current_dqn": dqn.state_dict(),
                "steps": steps,
                "cursor": list(cursor),
                "recipe": recipe,
                "parent_checkpoint": str(args.parent_checkpoint.resolve()),
            },
            path,
        )
        logging.info("saved stage checkpoint: %s steps=%s", path.name, steps)

    def save_state() -> None:
        tmp = state_file.with_name(state_file.name + ".tmp")
        torch.save(
            {
                "mortal": mortal.state_dict(),
                "current_dqn": dqn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "steps": steps,
                "cursor": list(cursor),
                "history": history,
                "recipe": recipe,
            },
            tmp,
        )
        tmp.replace(state_file)

    @torch.inference_mode()
    def monitor() -> dict[str, float]:
        mortal.eval()
        dqn.eval()
        totals = {name: {"kl": 0.0, "agree": 0.0, "rows": 0} for name in ["overall", *pool_names]}
        for shard_path in monitor_shards:
            with np.load(shard_path) as shard:
                take = min(int(args.batch_size), len(shard["obs"]))
                payload = {key: shard[key][:take] for key in _SAMPLE_FIELDS}
            obs = torch.as_tensor(payload["obs"], dtype=torch.float32, device=device)
            legal = torch.as_tensor(payload["mask"], dtype=torch.bool, device=device)
            teacher_q = torch.as_tensor(payload["teacher_q"], dtype=torch.float32, device=device)
            teacher_action = torch.as_tensor(payload["teacher_action"], dtype=torch.int64, device=device)
            pool_ids = torch.as_tensor(payload["pool"], dtype=torch.int64, device=device)
            q_out = dqn(mortal(obs), legal).float()
            kl_per_row = _kl_per_row(
                student_q=q_out, teacher_q=teacher_q, legal=legal, temperature=float(args.temperature)
            )
            agree = (q_out.argmax(dim=-1) == teacher_action).to(torch.float32)
            for name, selected in [("overall", torch.ones_like(pool_ids, dtype=torch.bool))] + [
                (name, pool_ids == index) for index, name in enumerate(pool_names)
            ]:
                count = int(selected.sum().item())
                if not count:
                    continue
                totals[name]["kl"] += float(kl_per_row[selected].sum().cpu())
                totals[name]["agree"] += float(agree[selected].sum().cpu())
                totals[name]["rows"] += count
        mortal.train()
        dqn.train()
        metrics: dict[str, float] = {}
        for name, entry in totals.items():
            rows = entry["rows"]
            if rows == 0:
                raise RuntimeError(f"monitor sample has no rows for {name!r}")
            prefix = "monitor" if name == "overall" else f"monitor_{name}"
            metrics[f"{prefix}_kl"] = entry["kl"] / rows
            metrics[f"{prefix}_agreement"] = entry["agree"] / rows
            metrics[f"{prefix}_rows"] = float(rows)
        return metrics

    writer = SummaryWriter(str(output_dir / "tb_distill"))
    stage_saves = sorted({int(value) for value in args.stage_saves})
    if steps == 0 and 0 in stage_saves:
        save_stage(output_dir / "student_distill_step_000000.pth")

    window_kl = 0.0
    window_agree = 0.0
    window_count = 0
    started = time.time()

    def payload_batch() -> tuple[dict[str, np.ndarray], tuple[int, int]]:
        nonlocal cursor
        payload, next_cursor = stream.read_chunk(cursor[0], cursor[1], int(args.batch_size))
        if len(payload["obs"]) < int(args.batch_size):
            if next_cursor == (0, 0) and cursor != (0, 0):
                remaining = int(args.batch_size) - len(payload["obs"])
                tail, _ = stream.read_chunk(0, 0, remaining)
                payload = {key: np.concatenate([payload[key], tail[key]]) for key in payload}
                next_cursor = (0, remaining)
        return payload, next_cursor

    while steps < int(args.target_steps):
        payload, next_cursor = payload_batch()
        obs = torch.as_tensor(payload["obs"], dtype=torch.float32, device=device)
        legal = torch.as_tensor(payload["mask"], dtype=torch.bool, device=device)
        teacher_q = torch.as_tensor(payload["teacher_q"], dtype=torch.float32, device=device)
        teacher_action = torch.as_tensor(payload["teacher_action"], dtype=torch.int64, device=device)
        if not bool(legal[torch.arange(len(teacher_action)), teacher_action].all().item()):
            raise RuntimeError("teacher label outside legal mask; dataset corruption")

        q_out = dqn(mortal(obs), legal)
        loss = _kl_loss(student_q=q_out, teacher_q=teacher_q, legal=legal, temperature=float(args.temperature))
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError(f"non-finite KL at step {steps + 1}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()

        with torch.inference_mode():
            agree = float((q_out.detach().float().argmax(dim=-1) == teacher_action).float().mean().cpu())
        window_kl += float(loss.detach().cpu())
        window_agree += agree
        window_count += 1
        cursor = next_cursor
        steps += 1

        if steps % int(args.log_every) == 0 or steps >= int(args.target_steps):
            avg_kl = window_kl / max(1, window_count)
            avg_agree = window_agree / max(1, window_count)
            logging.info(
                "steps=%s/%s kl=%.5f hard_agreement=%.4f lr=%.3e cursor=%s",
                steps, args.target_steps, avg_kl, avg_agree, scheduler.get_last_lr()[0], cursor,
            )
            writer.add_scalar("distill/kl", avg_kl, steps)
            writer.add_scalar("distill/hard_agreement", avg_agree, steps)
            writer.add_scalar("distill/lr", scheduler.get_last_lr()[0], steps)
            window_kl = window_agree = 0.0
            window_count = 0

        if steps in stage_saves:
            save_stage(output_dir / f"student_distill_step_{steps:06d}.pth")
        if int(args.state_every) > 0 and steps % int(args.state_every) == 0:
            save_state()
        if int(args.monitor_every) > 0 and steps % int(args.monitor_every) == 0:
            metrics = monitor()
            metrics["steps"] = steps
            history.append(metrics)
            logging.info(
                "monitor: steps=%s overall kl=%.5f agreement=%.4f rows=%.0f",
                steps, metrics["monitor_kl"], metrics["monitor_agreement"], metrics["monitor_rows"],
            )
            for pool_name in pool_names:
                logging.info(
                    "monitor: steps=%s pool=%s kl=%.5f agreement=%.4f",
                    steps, pool_name, metrics[f"monitor_{pool_name}_kl"], metrics[f"monitor_{pool_name}_agreement"],
                )
            writer.add_scalar("monitor/kl", metrics["monitor_kl"], steps)
            writer.add_scalar("monitor/agreement", metrics["monitor_agreement"], steps)
            for pool_name in pool_names:
                writer.add_scalar(f"monitor/{pool_name}/kl", metrics[f"monitor_{pool_name}_kl"], steps)
                writer.add_scalar(f"monitor/{pool_name}/agreement", metrics[f"monitor_{pool_name}_agreement"], steps)

    if steps in stage_saves:
        save_stage(output_dir / f"student_distill_step_{steps:06d}.pth")
    save_state()
    (output_dir / "distill_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    writer.close()
    elapsed = time.time() - started
    return {"steps": steps, "cursor": list(cursor), "elapsed_sec": elapsed, "history": history, "recipe": recipe}


def main() -> None:
    result = run(_parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
