#!/usr/bin/env python3
"""Run the existing trainable-view audit in resumable 250-hanchan shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_files(index_path: Path) -> list[str]:
    payload = torch.load(index_path.resolve(), weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 6000:
        raise ValueError(f"expected 6000 files in {index_path}")
    return [str(value) for value in values]


def complete(path: Path) -> bool:
    report = path / "data_distribution_audit.json"
    if not report.is_file():
        return False
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        corpus = payload["corpus"]
        return int(corpus["files_selected"]) == 250 and int(corpus["malformed_count"]) == 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--q-batch-size", type=int, default=2048)
    parser.add_argument("--file-batch-size", type=int, default=100)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.start_shard <= args.end_shard <= 24:
        raise ValueError("shard range must satisfy 0 <= start <= end <= 24")

    files = read_files(args.file_index)
    audit_script = REPO_ROOT / "scripts/mortal/audit_replay_distribution.py"
    output_root = args.output_root.resolve()
    shard_index_root = output_root / "shard_indexes"
    shard_index_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for shard in range(args.start_shard, args.end_shard):
        shard_output = output_root / f"shard_{shard:02d}"
        if complete(shard_output) and not args.force:
            print(f"[distribution-shards] skip shard_{shard:02d}", flush=True)
            completed += 1
            continue
        shard_output.mkdir(parents=True, exist_ok=True)
        shard_files = files[shard * 250 : (shard + 1) * 250]
        if len(shard_files) != 250:
            raise ValueError(f"shard {shard} does not contain 250 files")
        shard_index = shard_index_root / f"file_index_{shard:02d}.pth"
        torch.save({"file_list": shard_files}, shard_index)
        command = [
            sys.executable,
            str(audit_script),
            "--file-index",
            str(shard_index),
            "--parent",
            str(args.parent.resolve()),
            "--config",
            str(args.config.resolve()),
            "--output-dir",
            str(shard_output),
            "--model-label",
            args.model_label,
            "--device",
            args.device,
            "--q-batch-size",
            str(args.q_batch_size),
            "--file-batch-size",
            str(args.file_batch_size),
            "--progress-every",
            "50",
        ]
        if args.require_cuda:
            command.append("--require-cuda")
        print(f"[distribution-shards] start shard_{shard:02d} ({shard + 1}/24)", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not complete(shard_output):
            raise RuntimeError(f"shard_{shard:02d} completed without a valid audit report")
        completed += 1
        print(f"[distribution-shards] complete shard_{shard:02d} completed={completed}", flush=True)
    print(json.dumps({"output_root": str(output_root), "completed": completed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
