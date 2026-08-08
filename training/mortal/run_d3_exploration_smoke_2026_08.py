#!/usr/bin/env python3
"""Run the preregistered D3 v1 functional smoke only."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from libriichi import _riichi

# Import the installed native package before adding the repository's Mortal
# Python directory, which contains a legacy extension that can shadow it.
from libriichi.arena import FourPlayer as NativeFourPlayer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_exploration_engine import CONTRACT_ID, D3ExplorationEngine
from training.mortal.four_player_native import _load_engine, _parse_model_specs


DEFAULT_OUTPUT = Path(
    "artifacts/experiments/model_pool_2026_07/"
    "D3_uncertainty_guided_exploration_2026_08/generation_smoke/run_a"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_value_at(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="LABEL=CHECKPOINT; repeat exactly four times")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed-start", type=int, default=1_799_000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--native-batch-games", type=int, default=25)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda":
        raise ValueError("D3 smoke is fixed to device=cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    if args.games != 25 or args.seed_start != 1_799_000:
        raise ValueError("D3 smoke is fixed at 25 games and seed_start=1799000")
    if args.seed_key != 8192:
        raise ValueError("D3 smoke is fixed at seed_key=8192")
    if args.native_batch_games != 25:
        raise ValueError("D3 smoke is fixed at native_batch_games=25")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"smoke output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    native_root = args.mortal_root.resolve()
    patch_path = REPO_ROOT / "scripts/mortal/patches/libriichi_d3_decision_context.patch"
    if not patch_path.is_file():
        raise FileNotFoundError(f"D3 native patch not found: {patch_path}")
    extension_path = Path(_riichi.__file__).resolve()

    models = _parse_model_specs(args.model)
    required_labels = {"K0_70k", "V2_74000", "V3_74000", "ext_mortal"}
    if set(models) != required_labels:
        raise ValueError(f"D3 smoke requires exactly {sorted(required_labels)}, got {sorted(models)}")

    loaded: dict[str, Any] = {}
    for label, checkpoint in models.items():
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found for {label}: {checkpoint}")
        print(f"[d3-smoke] loading {label}: {checkpoint}", flush=True)
        loaded[label] = _load_engine(
            label=label,
            state_file=checkpoint,
            mortal_root=args.mortal_root,
            device=str(args.device),
            enable_amp=False,
            enable_profile=False,
        )

    d3_engine = D3ExplorationEngine(loaded["K0_70k"], name="K0_70k")
    engines = [
        d3_engine,
        loaded["V2_74000"],
        loaded["V3_74000"],
        loaded["ext_mortal"],
    ]
    labels = ["K0_70k", "V2_74000", "V3_74000", "ext_mortal"]

    project_git_commit = _git_value("rev-parse", "HEAD")
    project_git_dirty = bool(_git_value("status", "--porcelain"))
    mortal_source_commit = _git_value_at(native_root, "rev-parse", "HEAD")
    mortal_source_dirty = bool(_git_value_at(native_root, "status", "--porcelain"))
    protocol = {
        "schema": "keqing.mortal.d3_generation_smoke_protocol.v1",
        "contract_id": CONTRACT_ID,
        "seed_start": args.seed_start,
        "seed_end_exclusive": args.seed_start + args.games,
        "seed_key": args.seed_key,
        "games": args.games,
        "native_batch_games": args.native_batch_games,
        "seat_mode": "random",
        "amp": False,
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {
            label: {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
            for label, path in models.items()
        },
        "engine_order": labels,
        "git_commit": project_git_commit,
        "git_dirty": project_git_dirty,
        "project_git_commit": project_git_commit,
        "project_git_dirty": project_git_dirty,
        "mortal_source_commit": mortal_source_commit,
        "mortal_source_dirty": mortal_source_dirty,
        "d3_native_patch_sha256": _sha256_file(patch_path),
        "loaded_libriichi_path": str(extension_path),
        "loaded_libriichi_sha256": _sha256_file(extension_path),
        "native_build_profile": "release",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "contract_semantics": {
            "probability": 0.25,
            "margin_threshold": 0.5,
            "kyoku_budget": 1,
            "hanchan_budget": 8,
            "index_encoding": "decimal ASCII without leading zeros",
            "tie_break": "stable descending finite legal Q; lower action id first",
        },
    }
    _write_json(output_dir / "protocol.json", protocol)

    env = NativeFourPlayer(disable_progress_bar=True, log_dir=str(log_dir))
    print(
        f"[d3-smoke] games 1-{args.games}/{args.games} seed_start={args.seed_start} "
        f"batch={args.native_batch_games} device={args.device} amp=false",
        flush=True,
    )
    started = time.monotonic()
    rank_counts = env.py_vs_py_random_seats(
        engines[0],
        engines[1],
        engines[2],
        engines[3],
        (args.seed_start, args.seed_key),
        args.games,
    )
    elapsed = time.monotonic() - started
    print(f"[d3-smoke] completed {args.games}/{args.games} in {elapsed:.1f}s", flush=True)

    audit_dir = output_dir / "exploration"
    audit_paths = d3_engine.write_audit_files(audit_dir)
    summary = d3_engine.summary()
    summary["logs"] = {"directory": str(log_dir), "file_count": len(list(log_dir.glob("*.json.gz")))}
    summary["rank_counts"] = {
        label: [int(value) for value in counts]
        for label, counts in zip(labels, rank_counts, strict=True)
    }
    summary["elapsed_seconds"] = elapsed
    _write_json(output_dir / "smoke_summary.json", summary)
    protocol["artifacts"] = {
        "logs": str(log_dir),
        "exploration_events": str(audit_paths["events"]),
        "exploration_summary": str(audit_paths["summary"]),
        "smoke_summary": str(output_dir / "smoke_summary.json"),
    }
    protocol["exploration_counters"] = summary["counters"]
    _write_json(output_dir / "protocol.json", protocol)
    print(json.dumps(summary["counters"], ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
