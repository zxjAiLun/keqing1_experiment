#!/usr/bin/env python3
"""Compute global exact decision/state duplicate rates without model inference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))


def file_list(index_path: Path) -> list[str]:
    payload = torch.load(index_path.resolve(), weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 6000:
        raise ValueError(f"expected 6000 files in {index_path}")
    return [str(value) for value in values]


def digest_record(obs: np.ndarray, mask: np.ndarray, action: int | None = None) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(obs, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.bool_).tobytes())
    if action is not None:
        digest.update(int(action).to_bytes(2, "little", signed=False))
    return digest.digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--file-batch-size", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

    files = file_list(args.file_index)
    loader = GameplayLoader(version=args.version, player_names=[args.model_label], oracle=False, augmented=False)
    decision_hashes: Counter[bytes] = Counter()
    state_hashes: Counter[bytes] = Counter()
    malformed: list[dict[str, str]] = []
    decisions = 0
    perspectives = 0
    for start in range(0, len(files), args.file_batch_size):
        batch = files[start : start + args.file_batch_size]
        try:
            loaded = loader.load_gz_log_files(batch)
            if len(loaded) != len(batch):
                raise ValueError(f"loader returned {len(loaded)} files for {len(batch)}")
            for path, games in zip(batch, loaded, strict=True):
                if len(games) != 1:
                    raise ValueError(f"expected one {args.model_label} perspective, got {len(games)}")
                game = games[0]
                obs = game.take_obs()
                masks = game.take_masks()
                actions = game.take_actions()
                if not (len(obs) == len(masks) == len(actions)):
                    raise ValueError("decision arrays have inconsistent lengths")
                for row_obs, row_mask, action in zip(obs, masks, actions, strict=True):
                    decision_hashes[digest_record(row_obs, row_mask, int(action))] += 1
                    state_hashes[digest_record(row_obs, row_mask)] += 1
                decisions += len(actions)
                perspectives += 1
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(batch[0]) if batch else "", "error": str(exc)})
        if (start // args.file_batch_size + 1) * args.file_batch_size % args.progress_every == 0:
            print(f"[duplicate-audit] files {min(start + len(batch), len(files))}/{len(files)} decisions={decisions}", flush=True)

    report = {
        "schema": "keqing.mortal.trainable_view_duplicates.v1",
        "inputs": {"file_index": str(args.file_index.resolve()), "model_label": args.model_label, "file_count": len(files)},
        "hanchans": len(files),
        "trainable_perspectives": perspectives,
        "total_decisions": decisions,
        "unique_decision_count": len(decision_hashes),
        "duplicate_decision_count": decisions - len(decision_hashes),
        "duplicate_decision_rate": (decisions - len(decision_hashes)) / decisions if decisions else 0.0,
        "max_exact_repeat_count": max(decision_hashes.values(), default=0),
        "unique_state_count": len(state_hashes),
        "state_duplicate_count": decisions - len(state_hashes),
        "state_duplicate_rate": (decisions - len(state_hashes)) / decisions if decisions else 0.0,
        "max_state_repeat_count": max(state_hashes.values(), default=0),
        "malformed_count": len(malformed),
        "malformed": malformed[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "decisions": decisions, "malformed": len(malformed)}, ensure_ascii=False), flush=True)
    if malformed:
        raise SystemExit("duplicate audit found malformed files")


if __name__ == "__main__":
    main()
