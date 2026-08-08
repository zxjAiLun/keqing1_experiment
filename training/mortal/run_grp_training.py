#!/usr/bin/env python3
"""Finite-step, resumable training runner for the project-owned GRP model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import random
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON_ROOT = MORTAL_ROOT / "mortal"
for import_root in (REPO_ROOT, MORTAL_ROOT, MORTAL_PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from libriichi.dataset import Grp  # noqa: E402
from model import GRP  # noqa: E402


LOGGER = logging.getLogger("keqing_grp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive-steps", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_file_list(config: dict[str, Any], key: str) -> list[str]:
    dataset = config["grp"]["dataset"]
    list_file = dataset.get(f"{key}_files_file")
    if list_file:
        path = Path(str(list_file)).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        files = [str(Path(value).resolve()) for value in values]
    else:
        from glob import glob

        files = []
        for pattern in dataset.get(f"{key}_globs", []):
            files.extend(glob(str(pattern), recursive=True))
        files = [str(Path(value).resolve()) for value in files]
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"GRP {key} file list is empty")
    missing = [value for value in files if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"GRP {key} list contains missing files: {missing[0]}")
    return files


def _list_hash(files: list[str]) -> str:
    return _sha256_bytes(("\n".join(files) + "\n").encode("utf-8"))


def _training_contract(config: dict[str, Any], train_files: list[str], val_files: list[str], holdout_files: list[str]) -> dict[str, Any]:
    dataset = config["grp"]["dataset"]
    manifest_file = dataset.get("manifest_file")
    manifest_sha = None
    if manifest_file:
        manifest_path = Path(str(manifest_file)).resolve()
        if manifest_path.exists():
            manifest_sha = _sha256_file(manifest_path)
    return {
        "schema": "keqing.mortal.grp_training_contract.v1",
        "network": {
            "hidden_size": int(config["grp"]["network"]["hidden_size"]),
            "num_layers": int(config["grp"]["network"]["num_layers"]),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(config["grp"]["optim"]["lr"]),
            "weight_decay": float(config["grp"]["optim"].get("weight_decay", 0.01)),
            "betas": [float(value) for value in config["grp"]["optim"].get("betas", [0.9, 0.999])],
        },
        "dataset": {
            "train_count": len(train_files),
            "validation_count": len(val_files),
            "holdout_count": len(holdout_files),
            "train_file_list_sha256": _list_hash(train_files),
            "validation_file_list_sha256": _list_hash(val_files),
            "holdout_file_list_sha256": _list_hash(holdout_files),
            "manifest_sha256": manifest_sha,
            "file_batch_size": int(config["grp"]["dataset"].get("file_batch_size", 50)),
        },
        "project_git_revision": _git_revision(REPO_ROOT),
        "mortal_git_revision": _git_revision(MORTAL_ROOT),
    }


def _make_sample_batch(samples: list[tuple[np.ndarray, list[int]]]) -> tuple[Any, torch.Tensor, torch.Tensor]:
    sequences = [torch.as_tensor(feature, dtype=torch.float64) for feature, _ in samples]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
    padded = pad_sequence(sequences, batch_first=True)
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    ranks = torch.tensor([rank for _, rank in samples], dtype=torch.int64)
    stages = torch.tensor([int(feature[-1, 0]) for feature, _ in samples], dtype=torch.int64)
    return packed, ranks, stages


class GrpBatchStream:
    """Deterministic chunked stream with a small, serializable cursor."""

    def __init__(self, files: list[str], *, batch_size: int, file_batch_size: int, seed: int) -> None:
        self.files = files
        self.batch_size = batch_size
        self.file_batch_size = file_batch_size
        self.seed = seed
        self.epoch = 0
        self.chunk_index = 0
        self.batch_index = 0
        self._cache_key: tuple[int, int] | None = None
        self._cache_samples: list[tuple[np.ndarray, list[int]]] | None = None

    def _ordered_files(self, epoch: int) -> list[str]:
        ordered = list(self.files)
        random.Random(self.seed + epoch * 1_000_003).shuffle(ordered)
        return ordered

    def _load_chunk(self, epoch: int, chunk_index: int) -> list[tuple[np.ndarray, list[int]]]:
        key = (epoch, chunk_index)
        if self._cache_key == key and self._cache_samples is not None:
            return self._cache_samples
        ordered = self._ordered_files(epoch)
        start = chunk_index * self.file_batch_size
        chunk_files = ordered[start : start + self.file_batch_size]
        if not chunk_files:
            self.epoch += 1
            self.chunk_index = 0
            self.batch_index = 0
            return self._load_chunk(self.epoch, self.chunk_index)

        games = Grp.load_gz_log_files(chunk_files)
        samples: list[tuple[np.ndarray, list[int]]] = []
        for game in games:
            feature = np.asarray(game.take_feature(), dtype=np.float64)
            rank_by_player = [int(value) for value in game.take_rank_by_player()]
            for prefix_end in range(1, len(feature) + 1):
                samples.append((feature[:prefix_end].copy(), rank_by_player))
        random.Random(self.seed + epoch * 1_000_003 + chunk_index * 97_003).shuffle(samples)
        self._cache_key = key
        self._cache_samples = samples
        return samples

    def next_batch(self) -> tuple[Any, torch.Tensor, torch.Tensor]:
        while True:
            samples = self._load_chunk(self.epoch, self.chunk_index)
            start = self.batch_index * self.batch_size
            end = start + self.batch_size
            self.batch_index += 1
            if end <= len(samples):
                return _make_sample_batch(samples[start:end])
            self.chunk_index += 1
            self.batch_index = 0
            self._cache_key = None
            self._cache_samples = None

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "chunk_index": self.chunk_index,
            "batch_index": self.batch_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.chunk_index = int(state["chunk_index"])
        self.batch_index = int(state["batch_index"])
        self._cache_key = None
        self._cache_samples = None


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _stage_name(stage: int) -> str:
    if stage <= 1:
        return "east_front"
    if stage <= 3:
        return "east_back"
    if stage <= 5:
        return "south_front"
    if stage <= 7:
        return "south_back"
    return "extension"


def _evaluate(model: GRP, stream: GrpBatchStream, *, device: torch.device, steps: int, pts: torch.Tensor) -> dict[str, Any]:
    model.eval()
    total = defaultdict(float)
    stage_totals: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    with torch.inference_mode():
        for _ in range(steps):
            packed, ranks, stages = stream.next_batch()
            packed = packed.to(device=device)
            ranks = ranks.to(device=device)
            logits = model.forward_packed(packed)
            labels = model.get_label(ranks)
            probs = logits.softmax(-1)
            one_hot = F.one_hot(labels, num_classes=24).to(dtype=probs.dtype)
            matrix = model.calc_matrix(logits).to(device=device)
            actual_pt = pts[ranks]
            expected_pt = matrix @ pts
            losses = F.cross_entropy(logits, labels, reduction="none")
            pt_errors = (expected_pt - actual_pt).abs().mean(-1)
            batch_size = int(labels.shape[0])
            total["nll"] += float(F.cross_entropy(logits, labels).item()) * batch_size
            total["top1_correct"] += float((logits.argmax(-1) == labels).sum().item())
            total["brier"] += float(((probs - one_hot) ** 2).sum(-1).sum().item())
            total["expected_pt_mae"] += float((expected_pt - actual_pt).abs().mean(-1).sum().item())
            for player in range(4):
                for rank in range(4):
                    confidence = matrix[:, player, rank]
                    target = (ranks[:, player] == rank).to(dtype=confidence.dtype)
                    total["marginal_ece"] += float((confidence - target).abs().sum().item())
            for stage in torch.unique(stages).tolist():
                mask = stages == int(stage)
                name = _stage_name(int(stage))
                stage_totals[name]["count"] += int(mask.sum().item())
                stage_totals[name]["nll"] += float(losses[mask].sum().item())
                stage_totals[name]["expected_pt_mae"] += float(pt_errors[mask].sum().item())
            count += batch_size
    model.train()
    result = {
        "samples": count,
        "nll": total["nll"] / count,
        "top1_accuracy": total["top1_correct"] / count,
        "brier": total["brier"] / count,
        "expected_pt_mae": total["expected_pt_mae"] / count,
        "marginal_ece_proxy": total["marginal_ece"] / (count * 16),
        "stage": {},
    }
    for name, values in stage_totals.items():
        stage_count = values["count"]
        result["stage"][name] = {
            "samples": int(stage_count),
            "nll": values["nll"] / stage_count,
            "expected_pt_mae": values["expected_pt_mae"] / stage_count,
        }
    return result


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    steps: int,
    best_validation_nll: float,
    contract: dict[str, Any],
    data_stream: GrpBatchStream,
    validation_stream: GrpBatchStream,
    seed: int,
    data_seed: int,
    validation: dict[str, Any] | None,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": steps,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "best_validation_nll": best_validation_nll,
        "validation": validation,
        "contract": contract,
        "seed": seed,
        "data_seed": data_seed,
        "data_cursor": data_stream.state_dict(),
        "validation_cursor": validation_stream.state_dict(),
        "rng": _capture_rng(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    grp_config = config["grp"]
    control = grp_config["control"]
    dataset_config = grp_config["dataset"]
    device = torch.device(args.device or control.get("device", "cuda"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("GRP training requires CUDA, but torch.cuda.is_available() is False")
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_files = _load_file_list(config, "train")
    validation_files = _load_file_list(config, "validation")
    holdout_files = _load_file_list(config, "holdout")
    contract = _training_contract(config, train_files, validation_files, holdout_files)
    data_seed = int(args.data_seed if args.data_seed is not None else args.seed)
    batch_size = int(control.get("batch_size", 512))
    file_batch_size = int(dataset_config.get("file_batch_size", 50))
    train_stream = GrpBatchStream(train_files, batch_size=batch_size, file_batch_size=file_batch_size, seed=data_seed)
    validation_stream = GrpBatchStream(validation_files, batch_size=batch_size, file_batch_size=file_batch_size, seed=data_seed + 1)
    model = GRP(**grp_config["network"]).to(device)
    optimizer_config = grp_config["optim"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        betas=tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )
    state_file = Path(str(grp_config["state_file"])).resolve()
    best_state_file = Path(str(grp_config.get("best_state_file", state_file.with_name(state_file.stem + "_best.pth")))).resolve()
    steps = 0
    best_validation_nll = float("inf")
    last_validation: dict[str, Any] | None = None
    if args.resume:
        if not state_file.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is missing: {state_file}")
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        if state.get("contract", {}).get("dataset") != contract["dataset"]:
            raise RuntimeError("GRP resume refused: dataset contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        steps = int(state["steps"])
        best_validation_nll = float(state.get("best_validation_nll", float("inf")))
        last_validation = state.get("validation")
        train_stream.load_state_dict(state["data_cursor"])
        validation_stream.load_state_dict(state["validation_cursor"])
        _restore_rng(state["rng"])
        LOGGER.info("resumed GRP checkpoint steps=%s", steps)

    archive_steps = int(args.archive_steps or control.get("archive_steps", 2000))
    val_steps = int(args.val_steps or control.get("val_steps", 400))
    pts = torch.tensor(config.get("env", {}).get("pts", [6.0, 4.0, 2.0, 0.0]), dtype=torch.float64, device=device)
    while steps < args.target_steps:
        packed, ranks, _ = train_stream.next_batch()
        packed = packed.to(device=device)
        ranks = ranks.to(device=device)
        logits = model.forward_packed(packed)
        labels = model.get_label(ranks)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % args.log_every == 0 or steps == args.target_steps:
            LOGGER.info("step=%s/%s loss=%.6f", steps, args.target_steps, float(loss.item()))
        if steps % archive_steps == 0 or steps == args.target_steps:
            validation_stream.load_state_dict({"epoch": 0, "chunk_index": 0, "batch_index": 0})
            validation = _evaluate(model, validation_stream, device=device, steps=val_steps, pts=pts)
            last_validation = validation
            LOGGER.info(
                "validation step=%s nll=%.6f top1=%.4f brier=%.6f pt_mae=%.6f",
                steps,
                validation["nll"],
                validation["top1_accuracy"],
                validation["brier"],
                validation["expected_pt_mae"],
            )
            candidate_best = min(best_validation_nll, float(validation["nll"]))
            _save_checkpoint(
                state_file,
                model=model,
                optimizer=optimizer,
                steps=steps,
                best_validation_nll=candidate_best,
                contract=contract,
                data_stream=train_stream,
                validation_stream=validation_stream,
                seed=args.seed,
                data_seed=data_seed,
                validation=validation,
            )
            if validation["nll"] < best_validation_nll:
                best_validation_nll = float(validation["nll"])
                _save_checkpoint(
                    best_state_file,
                    model=model,
                    optimizer=optimizer,
                    steps=steps,
                    best_validation_nll=best_validation_nll,
                    contract=contract,
                    data_stream=train_stream,
                    validation_stream=validation_stream,
                    seed=args.seed,
                    data_seed=data_seed,
                    validation=validation,
                )

    contract_path = state_file.with_name("training_contract.json")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed steps=%s checkpoint=%s best=%s", steps, state_file, best_state_file)


if __name__ == "__main__":
    main()
