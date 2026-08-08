#!/usr/bin/env python3
"""Hash the first deterministic Mortal dataloader batches for one config."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random
import sys
import tomllib
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))


def hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def hash_batch(batch: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in batch:
        hash_tensor(digest, tensor)
    return digest.hexdigest()


def load_labels(config: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for path in config["dataset"]["player_names_files"]:
        values.update(
            line.strip()
            for line in Path(str(path)).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return sorted(values)


def load_player_names_by_file(config: dict[str, Any]) -> dict[str, str] | None:
    path = config["dataset"].get("player_names_by_file")
    if not path:
        return None
    payload = __import__("json").loads(Path(str(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"player_names_by_file must be a JSON object: {path}")
    return {str(Path(str(key)).resolve()): str(value) for key, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--batch-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_count <= 0:
        raise ValueError("--batch-count must be positive")

    config_path = args.config.resolve()
    os.environ["MORTAL_CFG"] = str(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)
    from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: PLC0415

    index_path = Path(str(config["dataset"]["file_index"])).resolve()
    index_payload = torch.load(index_path, weights_only=False, map_location="cpu")
    file_list = list(index_payload["file_list"] if isinstance(index_payload, dict) else index_payload)
    if len(file_list) != 6000:
        raise ValueError(f"expected 6000 indexed files, got {len(file_list)}")
    dataset = FileDatasetsIter(
        version=int(config["control"]["version"]),
        file_list=file_list,
        pts=config["env"]["pts"],
        file_batch_size=int(config["dataset"]["file_batch_size"]),
        reserve_ratio=float(config["dataset"]["reserve_ratio"]),
        player_names=load_labels(config),
        num_epochs=int(config["dataset"]["num_epochs"]),
        enable_augmentation=bool(config["dataset"]["enable_augmentation"]),
        augmented_first=bool(config["dataset"]["augmented_first"]),
        player_names_by_file=load_player_names_by_file(config),
    )
    loader = iter(
        DataLoader(
            dataset=dataset,
            batch_size=int(config["control"]["batch_size"]),
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )
    )
    rows: list[dict[str, Any]] = []
    for batch_index in range(args.batch_count):
        batch = next(loader)
        if not isinstance(batch, (tuple, list)) or len(batch) != 6:
            raise ValueError(f"unexpected batch structure at {batch_index}")
        tensors = tuple(value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in batch)
        rows.append(
            {
                "batch": batch_index,
                "sha256": hash_batch(tensors),
                "samples": int(tensors[0].shape[0]),
                "shapes": [list(value.shape) for value in tensors],
            }
        )
    report = {
        "schema": "keqing.mortal.dataloader_batch_preview.v1",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "data_seed": args.data_seed,
        "label": load_labels(config),
        "file_index": str(index_path),
        "file_count": len(file_list),
        "batches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(__import__("json").dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
