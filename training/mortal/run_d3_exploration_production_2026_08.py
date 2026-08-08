#!/usr/bin/env python3
"""Preflight or execute the frozen first D3 production B250 gate exactly once."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_exploration_engine import CONTRACT_ID, D3ExplorationEngine
from training.mortal.d3_production_contract import (
    AMP,
    CONFIRMATION_TOKEN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMOKE_PROTOCOL,
    DEVICE,
    GAMES,
    GATE_ID,
    NATIVE_BATCH_GAMES,
    PRODUCTION_SCHEMA,
    RANK_POINTS,
    REQUIRED_LABELS,
    SEAT_MODE,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    ContractError,
    archive_mortal_lineage,
    assert_empty_output,
    sha256_file,
    write_json,
)
from training.mortal.d3_production_preflight import build_preflight, final_call_guard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="LABEL=CHECKPOINT; repeat exactly four times")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--authoritative-smoke-protocol",
        type=Path,
        default=DEFAULT_SMOKE_PROTOCOL,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the single B250 call; without this flag only preflight runs",
    )
    parser.add_argument(
        "--confirmation-token",
        default="",
        help=f"required with --execute; must equal {CONFIRMATION_TOKEN}",
    )
    return parser.parse_args(argv)


def run_single_native_b250(arena: Any, engines: list[Any]) -> Any:
    """Make the one and only native generation call authorized by this entrypoint."""

    if len(engines) != len(REQUIRED_LABELS):
        raise ContractError(f"expected four engines, got {len(engines)}")
    return arena.py_vs_py_random_seats(
        engines[0],
        engines[1],
        engines[2],
        engines[3],
        (SEED_START, SEED_KEY),
        GAMES,
    )


def _execute(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    if args.confirmation_token != CONFIRMATION_TOKEN:
        raise ContractError(
            f"--execute requires --confirmation-token {CONFIRMATION_TOKEN}"
        )
    if not preflight["passed"]:
        raise ContractError(f"preflight failed: {preflight['errors']}")

    from libriichi.arena import FourPlayer as NativeFourPlayer  # noqa: PLC0415
    from training.mortal.four_player_native import _load_engine  # noqa: PLC0415

    output_dir = Path(preflight["output_dir"])
    assert_empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight.json", preflight)
    lineage_archives = archive_mortal_lineage(
        args.mortal_root.resolve(), output_dir / "lineage" / "nested_mortal"
    )
    protocol: dict[str, Any] = {
        "schema": PRODUCTION_SCHEMA,
        "gate_id": GATE_ID,
        "contract_id": CONTRACT_ID,
        "status": "preflight_passed_generation_not_started",
        "fixed_protocol": preflight["execution_shape"],
        "authoritative_smoke": preflight["authoritative_smoke"],
        "project_lineage": preflight["project_lineage"],
        "mortal_lineage": preflight["mortal_lineage"],
        "runtime": preflight["runtime"],
        "d3_native_patch": preflight["d3_native_patch"],
        "production_implementation": preflight["production_implementation"],
        "ignored_artifacts": preflight["ignored_artifacts"],
        "models": preflight["models"],
        "engine_order": list(REQUIRED_LABELS),
        "lineage_archives": lineage_archives,
        "failure_policy": {
            "resume": "forbidden",
            "on_interruption": "delete the entire shard directory and restart from seed 1800000",
            "partial_shard_reuse": "forbidden",
        },
    }
    write_json(output_dir / "protocol.json", protocol)

    models = {label: Path(preflight["models"][label]["path"]) for label in REQUIRED_LABELS}
    loaded: dict[str, Any] = {}
    for label in REQUIRED_LABELS:
        actual_sha = sha256_file(models[label])
        expected_sha = preflight["models"][label]["sha256"]
        if actual_sha != expected_sha:
            raise ContractError(f"model changed before load: {label}")
        print(f"[d3-production] loading {label}: {models[label]}", flush=True)
        loaded[label] = _load_engine(
            label=label,
            state_file=models[label],
            mortal_root=args.mortal_root,
            device=DEVICE,
            enable_amp=False,
            enable_profile=False,
        )

    d3_engine = D3ExplorationEngine(loaded["K0_70k"], name="K0_70k")
    engines = [d3_engine, loaded["V2_74000"], loaded["V3_74000"], loaded["ext_mortal"]]
    log_dir = output_dir / "logs"
    arena = NativeFourPlayer(disable_progress_bar=True, log_dir=str(log_dir))
    call_guard = final_call_guard(args, preflight)
    protocol["final_call_guard"] = call_guard
    if not call_guard["passed"]:
        write_json(output_dir / "protocol.json", protocol)
        raise ContractError(f"final call guard failed: {call_guard['errors']}")
    protocol["status"] = "generation_running_single_b250"
    protocol["started_unix_time"] = time.time()
    write_json(output_dir / "protocol.json", protocol)

    print(
        "[d3-production] ONE native call: games=250 seeds=1800000..1800249 "
        "seed_key=8192 seat=random device=cuda amp=false",
        flush=True,
    )
    started = time.monotonic()
    rank_counts_raw = run_single_native_b250(arena, engines)
    elapsed = time.monotonic() - started
    rank_counts = {
        label: [int(value) for value in counts]
        for label, counts in zip(REQUIRED_LABELS, rank_counts_raw, strict=True)
    }
    if any(len(counts) != 4 or sum(counts) != GAMES for counts in rank_counts.values()):
        raise RuntimeError(f"invalid rank counts returned by single B250: {rank_counts}")
    if any(sum(counts[rank] for counts in rank_counts.values()) != GAMES for rank in range(4)):
        raise RuntimeError(f"rank columns do not each sum to {GAMES}: {rank_counts}")
    audit_paths = d3_engine.write_audit_files(output_dir / "exploration")
    exploration = d3_engine.summary()
    log_count = len(list(log_dir.glob("*.json.gz")))
    if log_count != GAMES:
        raise RuntimeError(f"single B250 returned {log_count} logs instead of {GAMES}")

    summary = {
        "schema": "keqing.mortal.d3_production_gate_run_summary.v1",
        "gate_id": GATE_ID,
        "contract_id": CONTRACT_ID,
        "status": "generation_completed_audit_pending",
        "games": GAMES,
        "seed_start": SEED_START,
        "seed_end_exclusive": SEED_END_EXCLUSIVE,
        "seed_key": SEED_KEY,
        "native_batch_games": NATIVE_BATCH_GAMES,
        "native_call_count": 1,
        "log_count": log_count,
        "rank_counts": rank_counts,
        "rank_points": list(RANK_POINTS),
        "elapsed_seconds": elapsed,
        "exploration_counters": exploration["counters"],
        "artifacts": {
            "logs": str(log_dir),
            "exploration_events": str(audit_paths["events"]),
            "exploration_summary": str(audit_paths["summary"]),
        },
        "scope": "generation only; no training, no remaining 5750h, no recipe selection",
    }
    write_json(output_dir / "production_summary.json", summary)
    protocol["status"] = "generation_completed_audit_pending"
    protocol["completed_unix_time"] = time.time()
    protocol["elapsed_seconds"] = elapsed
    protocol["artifacts"] = summary["artifacts"] | {
        "production_summary": str(output_dir / "production_summary.json")
    }
    protocol["exploration_counters"] = exploration["counters"]
    protocol["rank_counts"] = rank_counts
    write_json(output_dir / "protocol.json", protocol)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        preflight = build_preflight(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        if not preflight["passed"]:
            raise SystemExit(2)
        return

    output_dir = args.output_dir.resolve()
    try:
        _execute(args, preflight)
    except BaseException as exc:  # noqa: BLE001
        if output_dir.exists():
            failure = {
                "schema": "keqing.mortal.d3_production_gate_failure.v1",
                "gate_id": GATE_ID,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "required_action": (
                    "Delete the entire shard directory. Do not resume or keep partial logs; "
                    "restart from seed 1800000 with one B250 call."
                ),
            }
            write_json(output_dir / "RUN_FAILED_DELETE_WHOLE_SHARD.json", failure)
        raise


if __name__ == "__main__":
    main()
