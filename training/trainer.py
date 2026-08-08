"""共享训练循环：基础 policy/value 训练 + task 级扩展 loss。"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.specs import TaskSpec

# action index → 类型名（dahai 合并为一个大类，chi_*/kan 系列合并显示）
_ACTION_LABELS = (
    ["dahai"] * 34
    + ["reach", "chi_low", "chi_mid", "chi_high", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku", "none"]
)

# 合并显示时的分组映射
_MERGE_MAP = {
    "chi_low": "chi",
    "chi_mid": "chi",
    "chi_high": "chi",
    "daiminkan": "kan",
    "ankan": "kan",
    "kakan": "kan",
}


def _action_type_name(idx: int) -> str:
    if 0 <= idx < len(_ACTION_LABELS):
        return _ACTION_LABELS[idx]
    return f"unknown_{idx}"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_val_loss: float,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    payload_builder=None,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if payload_builder is not None:
        payload = payload_builder(
            base_payload=payload,
            cfg=cfg or {},
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            best_val_loss=best_val_loss,
        )
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("best_val_loss", float("inf"))


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE loss，mask 为 0 的动作设为 -1e4 再计算 softmax。"""
    logits = logits.masked_fill(mask == 0, -1e4)
    return nn.functional.cross_entropy(logits, labels)


def _is_finite_scalar(value: float) -> bool:
    return math.isfinite(float(value))


def _format_nonfinite_debug(
    *,
    tag: str,
    batch_idx: int,
    reason: str,
    loss_value: float,
    ce_value: float,
    val_loss_value: float,
    extra_loss_value: float,
    extra_metrics: Dict,
    lr: float,
    grad_norm: float | None = None,
) -> str:
    extra_str = " ".join(f"{key}={float(extra_metrics.get(key, 0.0)):.4f}" for key in sorted(extra_metrics))
    grad_str = f" grad_norm={grad_norm:.4f}" if grad_norm is not None else ""
    return (
        f"  [{tag}] nonfinite reason={reason} batch={batch_idx:5d} "
        f"loss={loss_value:.4f} ce={ce_value:.4f} val={val_loss_value:.4f} "
        f"extra={extra_loss_value:.4f} {extra_str}{grad_str} lr={lr:.2e}"
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    accumulation_steps: int,
    policy_loss_weight: float,
    value_loss_weight: float,
    task: TaskSpec,
    is_train: bool,
    step: int,
    log_interval: int = 100,
    max_batches: int | None = None,
) -> Dict:
    import sys

    model.train(is_train)
    total_ce = total_val_loss = total_acc = 0.0
    total_extra_loss = 0.0
    total_grad_norm = 0.0
    grad_norm_steps = 0
    nonfinite_steps = 0
    extra_totals = {k: 0.0 for k in task.log_metric_keys}
    n_batches = 0
    tag = "train" if is_train else "val"

    correct_by_type: Dict[str, int] = defaultdict(int)
    total_by_type: Dict[str, int] = defaultdict(int)

    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and n_batches >= max_batches:
                break
            batch_data = task.unpack_batch(batch, device)
            tile_feat = batch_data["tile_feat"]
            scalar = batch_data["scalar"]
            mask = batch_data["mask"]
            action_idx = batch_data["action_idx"]
            value_target = batch_data["value_target"]
            model_kwargs = batch_data.get("model_kwargs", {})

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                policy_logits, value_pred = model(tile_feat, scalar, **model_kwargs)
                ce = masked_ce_loss(policy_logits, action_idx, mask)
                val_loss = nn.functional.mse_loss(value_pred.squeeze(-1), value_target)
                extra_loss, extra_metrics = task.compute_extra_loss(
                    model, device, batch_data, is_train, i
                )
                loss = policy_loss_weight * ce + value_loss_weight * val_loss + extra_loss

            if is_train:
                loss_value = float(loss.detach().item())
                ce_value = float(ce.detach().item())
                val_loss_value = float(val_loss.detach().item())
                extra_loss_value = float(extra_loss.detach().item())
                current_lr = float(optimizer.param_groups[0]["lr"])
                if not _is_finite_scalar(loss_value):
                    nonfinite_steps += 1
                    print(
                        _format_nonfinite_debug(
                            tag=tag,
                            batch_idx=n_batches + 1,
                            reason="loss",
                            loss_value=loss_value,
                            ce_value=ce_value,
                            val_loss_value=val_loss_value,
                            extra_loss_value=extra_loss_value,
                            extra_metrics=extra_metrics,
                            lr=current_lr,
                        )
                    )
                    optimizer.zero_grad()
                    continue
                loss_scaled = loss / accumulation_steps
                scaler.scale(loss_scaled).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    if not _is_finite_scalar(grad_norm):
                        nonfinite_steps += 1
                        print(
                            _format_nonfinite_debug(
                                tag=tag,
                                batch_idx=n_batches + 1,
                                reason="grad_norm",
                                loss_value=loss_value,
                                ce_value=ce_value,
                                val_loss_value=val_loss_value,
                                extra_loss_value=extra_loss_value,
                                extra_metrics=extra_metrics,
                                lr=current_lr,
                                grad_norm=grad_norm,
                            )
                        )
                        optimizer.zero_grad()
                        scaler.update()
                        continue
                    total_grad_norm += grad_norm
                    grad_norm_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    step += 1

            with torch.no_grad():
                masked_logits = policy_logits.masked_fill(mask == 0, -1e4)
                pred = masked_logits.argmax(dim=-1)
                correct_mask = pred == action_idx
                total_acc += correct_mask.float().mean().item()

                for atype, correct in zip(action_idx, correct_mask):
                    name = _action_type_name(atype.item())
                    total_by_type[name] += 1
                    if correct:
                        correct_by_type[name] += 1

            total_ce += ce.item()
            total_val_loss += val_loss.item()
            total_extra_loss += float(extra_loss.item())
            for key in task.log_metric_keys:
                extra_totals[key] += float(extra_metrics.get(key, 0.0))
            n_batches += 1

            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
            grad_norm_str = ""
            if is_train and grad_norm_steps > 0:
                grad_norm_str = f" gnorm={total_grad_norm/grad_norm_steps:.3f}"
            nonfinite_str = f" skipped={nonfinite_steps}" if is_train and nonfinite_steps > 0 else ""
            if n_batches == 1 or (log_interval > 0 and n_batches % log_interval == 0):
                print(
                    f"  [{tag}] batch={n_batches:5d} | "
                    f"ce={total_ce/n_batches:.4f} "
                    f"val={total_val_loss/n_batches:.4f}"
                    f" acc={total_acc/n_batches:.3f}"
                    f"{grad_norm_str}"
                    f"{nonfinite_str}"
                    f"{lr_str}   "
                )

    merged_cor: Dict[str, int] = defaultdict(int)
    merged_tot: Dict[str, int] = defaultdict(int)
    for name in total_by_type:
        group = _MERGE_MAP.get(name, name)
        merged_cor[group] += correct_by_type[name]
        merged_tot[group] += total_by_type[name]
    acc_lines = []
    for name in sorted(merged_tot):
        tot = merged_tot[name]
        cor = merged_cor[name]
        acc = cor / tot if tot > 0 else 0.0
        acc_lines.append(f"    {name:>8s}: {cor:5d}/{tot:5d} = {acc:.3f}")
    if acc_lines:
        print("\n".join(acc_lines))

    n = max(1, n_batches)
    acc_by_type = {k: correct_by_type[k] / total_by_type[k] for k in total_by_type}
    stats = {
        "ce": total_ce / n,
        "val_loss": total_val_loss / n,
        "extra_loss": total_extra_loss / n,
        "objective": policy_loss_weight * (total_ce / n) + value_loss_weight * (total_val_loss / n) + (total_extra_loss / n),
        "acc": total_acc / n,
        "num_batches": n_batches,
        "grad_norm": (total_grad_norm / grad_norm_steps if grad_norm_steps > 0 else None),
        "nonfinite_steps": nonfinite_steps,
        "step": step,
        "acc_by_type": acc_by_type,
        "total_by_type": dict(total_by_type),
    }
    for key in task.log_metric_keys:
        stats[key] = extra_totals[key] / n
    return stats


def _is_better_metric(candidate: float, best: float, mode: str) -> bool:
    if mode == "max":
        return candidate > best
    return candidate < best


def _meld_metric_from_stats(stats: Dict) -> float | None:
    acc_by_type = stats.get("acc_by_type", {}) or {}
    total_by_type = stats.get("total_by_type", {}) or {}
    response_types = ("none", "chi", "pon", "daiminkan")
    weighted_correct = 0.0
    weighted_total = 0.0
    for name in response_types:
        total = float(total_by_type.get(name, 0.0))
        if total <= 0:
            continue
        weighted_total += total
        weighted_correct += total * float(acc_by_type.get(name, 0.0))
    if weighted_total <= 0:
        return None
    return weighted_correct / weighted_total


def train_model(
    model: nn.Module,
    *,
    train_loader: Optional[DataLoader],
    train_loader_factory: Optional[Callable[[int], DataLoader]],
    val_loader: DataLoader,
    task: TaskSpec,
    cfg: Dict,
    output_dir: Path,
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
    device_str: str = "cuda",
    checkpoint_payload_builder=None,
    checkpoint_loader=None,
) -> nn.Module:
    if train_loader is None and train_loader_factory is None:
        raise ValueError("train_loader or train_loader_factory is required")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    num_epochs = cfg.get("num_epochs", 10)
    accumulation_steps = cfg.get("accumulation_steps", 4)
    warmup_steps = cfg.get("warmup_steps", 500)
    steps_per_epoch_cfg = cfg.get("steps_per_epoch", None)
    steps_per_epoch = steps_per_epoch_cfg if steps_per_epoch_cfg is not None else 5000
    val_steps_per_epoch_cfg = cfg.get("val_steps_per_epoch", None)
    total_steps = steps_per_epoch * num_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf") if task.best_metric_mode == "max" else float("inf")
    best_meld = -float("inf")

    if resume_path is not None and resume_path.exists():
        checkpoint_label = f"checkpoint {resume_path}"
        if weights_only:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint_loader is not None:
                checkpoint_loader(
                    ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_label=checkpoint_label,
                    weights_only=True,
                )
            else:
                model.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded weights from {resume_path} (optimizer/scheduler/epoch reset)")
        else:
            if checkpoint_loader is not None:
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                start_epoch, global_step, best_metric = checkpoint_loader(
                    ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_label=checkpoint_label,
                    weights_only=False,
                )
            else:
                start_epoch, global_step, best_metric = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_loss_weight = cfg.get("policy_loss_weight", 1.0)
    value_loss_weight = cfg.get("value_loss_weight", 0.5)
    log_interval = cfg.get("log_interval", 100)

    end_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, end_epoch):
        t0 = time.time()
        print(f"\n[Epoch {epoch+1}/{end_epoch}]")

        current_train_loader = train_loader_factory(epoch) if train_loader_factory is not None else train_loader
        train_stats = _run_epoch(
            model, current_train_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, policy_loss_weight, value_loss_weight,
            task, is_train=True, step=global_step, log_interval=log_interval,
            max_batches=int(steps_per_epoch_cfg) if steps_per_epoch_cfg is not None else None,
        )
        if epoch == start_epoch and steps_per_epoch_cfg is None:
            actual_spe = train_stats["step"] - global_step
            print(f"  [auto] 实际 steps/epoch={actual_spe}，重建 scheduler（原估算={steps_per_epoch}）")
            remaining_epochs = end_epoch - start_epoch - 1
            scheduler = build_scheduler(optimizer, warmup_steps, actual_spe * remaining_epochs)

        global_step = train_stats["step"]

        val_stats = _run_epoch(
            model, val_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, policy_loss_weight, value_loss_weight,
            task, is_train=False, step=global_step, log_interval=log_interval,
            max_batches=int(val_steps_per_epoch_cfg) if val_steps_per_epoch_cfg is not None else None,
        )

        elapsed = time.time() - t0
        train_extra = " ".join(
            f"{key}={train_stats[key]:.4f}" for key in task.log_metric_keys
        )
        val_extra = " ".join(
            f"{key}={val_stats[key]:.4f}" for key in task.log_metric_keys
        )
        train_extra = f" {train_extra}" if train_extra else ""
        val_extra = f" {val_extra}" if val_extra else ""
        grad_norm = train_stats.get("grad_norm")
        gnorm_str = f" gnorm={grad_norm:.3f}" if grad_norm is not None else ""
        skipped_str = (
            f" skipped={train_stats['nonfinite_steps']}"
            if train_stats.get("nonfinite_steps", 0) > 0
            else ""
        )
        print(
            f"  train ce={train_stats['ce']:.4f}{train_extra} acc={train_stats['acc']:.3f}{gnorm_str}{skipped_str} "
            f"| val ce={val_stats['ce']:.4f}{val_extra} acc={val_stats['acc']:.3f} "
            f"| {elapsed:.0f}s"
        )

        save_checkpoint(
            output_dir / "last.pth", model, optimizer, scheduler,
            epoch + 1, global_step, best_metric,
            cfg=cfg,
            payload_builder=checkpoint_payload_builder,
        )

        metric_value = val_stats[task.best_metric_name]
        if _is_better_metric(metric_value, best_metric, task.best_metric_mode):
            best_metric = metric_value
            save_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_metric,
                cfg=cfg,
                payload_builder=checkpoint_payload_builder,
            )
            print(f"  [best checkpoint saved, val_{task.best_metric_name}={best_metric:.4f}]")

        meld_metric = _meld_metric_from_stats(val_stats)
        if meld_metric is not None and meld_metric > best_meld:
            best_meld = meld_metric
            save_checkpoint(
                output_dir / "best_meld.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_meld,
                cfg=cfg,
                payload_builder=checkpoint_payload_builder,
            )
            print(f"  [best_meld checkpoint saved, val_meld={best_meld:.4f}]")

        log_row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_ce": train_stats["ce"],
            "train_value_loss": train_stats["val_loss"],
            "train_extra_loss": train_stats["extra_loss"],
            "train_objective": train_stats["objective"],
            "train_acc": train_stats["acc"],
            "train_num_batches": train_stats.get("num_batches"),
            "train_grad_norm": train_stats.get("grad_norm"),
            "train_nonfinite_steps": train_stats.get("nonfinite_steps", 0),
            "val_ce": val_stats["ce"],
            "val_value_loss": val_stats["val_loss"],
            "val_extra_loss": val_stats["extra_loss"],
            "val_objective": val_stats["objective"],
            "val_acc": val_stats["acc"],
            "val_num_batches": val_stats.get("num_batches"),
            "lr": optimizer.param_groups[0]["lr"],
            "val_acc_by_type": val_stats.get("acc_by_type", {}),
            "val_total_by_type": val_stats.get("total_by_type", {}),
            "val_meld_metric": meld_metric,
        }
        for key in task.log_metric_keys:
            log_row[f"train_{key}"] = train_stats.get(key)
            log_row[f"val_{key}"] = val_stats.get(key)
        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(log_row) + "\n")

    print(f"\nTraining complete. Best val_{task.best_metric_name}={best_metric:.4f}")
    return model
