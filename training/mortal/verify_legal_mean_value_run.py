#!/usr/bin/env python3
"""Verify a completed legal-mean objective run before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ARCHIVE_STEPS = (70001, 70010, 70100, 70500, 71000, 72000)
DATA_STREAM_COMPARISON_FIELDS = (
    "schema",
    "data_seed",
    "dataset_file_count",
    "num_workers",
    "batches_consumed",
    "samples_consumed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--peer-run-dir", type=Path, required=True)
    parser.add_argument("--expected-objective", required=True)
    parser.add_argument("--expected-peer-objective", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=True, map_location="cpu")


def assert_finite(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"non-finite tensor in {label}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{label}[{index}]")


def canonical_data_stream(data_stream: dict[str, Any]) -> dict[str, Any]:
    """Return fields that describe the consumed stream, excluding resume provenance."""
    return {key: data_stream.get(key) for key in DATA_STREAM_COMPARISON_FIELDS}


def validate_run(
    run_dir: Path,
    *,
    expected_objective: str,
    expected_seed: int,
    parent_sha: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    state_path = run_dir / "mortal.pth"
    state = load_checkpoint(state_path)
    if int(state.get("steps", -1)) != 72000:
        raise ValueError(f"{run_dir}: final checkpoint is not at step 72000")
    assert_finite(state.get("mortal"), f"{run_dir}/mortal")
    assert_finite(state.get("current_dqn"), f"{run_dir}/current_dqn")
    assert_finite(state.get("aux_net"), f"{run_dir}/aux_net")
    assert_finite(state.get("optimizer"), f"{run_dir}/optimizer")
    contract = state.get("training_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{run_dir}: missing training contract")
    if contract.get("objective", {}).get("mode") != expected_objective:
        raise ValueError(f"{run_dir}: objective contract mismatch")
    if contract.get("reward", {}).get("mode") != "final_rank_mc":
        raise ValueError(f"{run_dir}: reward contract mismatch")
    if contract.get("git_commit") != expected_git_commit or bool(contract.get("git_dirty")):
        raise ValueError(f"{run_dir}: git provenance mismatch")
    initialization = state.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError(f"{run_dir}: missing initialization metadata")
    if initialization.get("mode") != "weights_plus_optimizer_warm_start":
        raise ValueError(f"{run_dir}: optimizer was not preserved")
    if initialization.get("parent_sha256") != parent_sha:
        raise ValueError(f"{run_dir}: parent SHA mismatch")
    if initialization.get("optimizer_checkpoint_sha256") != parent_sha:
        raise ValueError(f"{run_dir}: optimizer parent SHA mismatch")
    data_stream = state.get("data_stream")
    if not isinstance(data_stream, dict):
        raise ValueError(f"{run_dir}: missing data stream metadata")
    if int(data_stream.get("data_seed", -1)) != expected_seed:
        raise ValueError(f"{run_dir}: data seed mismatch")
    batches = int(data_stream.get("batches_consumed", 0))
    samples = int(data_stream.get("samples_consumed", 0))
    batch_size = int(state.get("config", {}).get("control", {}).get("batch_size", 0))
    if batches <= 0 or samples <= 0:
        raise ValueError(f"{run_dir}: no data stream progress")
    if batch_size <= 0 or samples != batches * batch_size:
        raise ValueError(f"{run_dir}: data stream sample/batch counts are inconsistent")
    archive_paths: list[str] = []
    for step in ARCHIVE_STEPS:
        archive_path = run_dir / "checkpoints" / f"mortal_{step}.pth"
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        archive = load_checkpoint(archive_path)
        if int(archive.get("steps", -1)) != step:
            raise ValueError(f"{archive_path}: embedded step mismatch")
        assert_finite(archive.get("mortal"), str(archive_path))
        archive_paths.append(str(archive_path))
    return {
        "run_dir": str(run_dir.resolve()),
        "objective": expected_objective,
        "steps": 72000,
        "data_seed": expected_seed,
        "data_stream": data_stream,
        "batch_size": batch_size,
        "archive_steps": list(ARCHIVE_STEPS),
        "archive_paths": archive_paths,
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "parent_sha256": parent_sha,
        "optimizer_checkpoint_sha256": parent_sha,
    }


def main() -> None:
    args = parse_args()
    parent_sha = sha256_file(args.parent.resolve())
    current = validate_run(
        args.run_dir.resolve(),
        expected_objective=args.expected_objective,
        expected_seed=args.expected_seed,
        parent_sha=parent_sha,
        expected_git_commit=args.expected_git_commit,
    )
    peer = validate_run(
        args.peer_run_dir.resolve(),
        expected_objective=args.expected_peer_objective,
        expected_seed=args.expected_seed,
        parent_sha=parent_sha,
        expected_git_commit=args.expected_git_commit,
    )
    current_stream = canonical_data_stream(current["data_stream"])
    peer_stream = canonical_data_stream(peer["data_stream"])
    if current_stream != peer_stream:
        raise ValueError(
            "control and variant consumed data stream differs: "
            f"control={current_stream!r} variant={peer_stream!r}"
        )
    report = {
        "schema": "keqing.mortal.legal_mean_value_run_verification.v2",
        "passed": True,
        "parent": str(args.parent.resolve()),
        "parent_sha256": parent_sha,
        "expected_git_commit": args.expected_git_commit,
        "control": current,
        "variant": peer,
        "data_stream_identical": True,
        "data_stream_comparison_fields": list(DATA_STREAM_COMPARISON_FIELDS),
        "resume_provenance_differs": (
            current["data_stream"].get("resume_skipped_batches")
            != peer["data_stream"].get("resume_skipped_batches")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
