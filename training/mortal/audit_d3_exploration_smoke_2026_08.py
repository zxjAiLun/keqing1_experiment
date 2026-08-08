#!/usr/bin/env python3
"""Audit two independent D3 v1 smoke runs for correctness and determinism."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_exploration_engine import (
    CONTRACT_ID,
    EXPLORATION_PROBABILITY,
    HANCHAN_BUDGET,
    KYOKU_BUDGET,
    canonical_hash_u,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1_799_000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--expected-project-commit", required=True)
    parser.add_argument("--expected-mortal-source-commit", required=True)
    parser.add_argument("--expected-native-patch-sha256", required=True)
    parser.add_argument("--expected-native-binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_log(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _log_key(events: list[dict[str, Any]], path: Path) -> tuple[int, int]:
    if not events or events[0].get("type") != "start_game":
        raise ValueError(f"first event is not start_game: {path}")
    seed = events[0].get("seed")
    if not isinstance(seed, list) or len(seed) != 2:
        raise ValueError(f"invalid start_game seed: {path}")
    return int(seed[0]), int(seed[1])


def _canonical_log_hash(events: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for original in events:
        event = copy.deepcopy(original)
        event.pop("meta", None)
        if event.get("type") == "start_game":
            event.pop("names", None)
        digest.update((json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _load_events(root: Path) -> list[dict[str, Any]]:
    path = root / "exploration" / "exploration_events.jsonl"
    if not path.is_file():
        raise ValueError(f"missing exploration events: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return sorted(
        events,
        key=lambda event: (
            int(event["generation_seed"]),
            int(event["seed_key"]),
            int(event["seat"]),
            int(event["kyoku_index"]),
            int(event["decision_index"]),
        ),
    )


def _audit_events(events: list[dict[str, Any]], expected_seed_keys: set[tuple[int, int]]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    kyoku_counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    hanchan_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    explored_by_kyoku: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    explored_by_hanchan: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    reason_counts: defaultdict[str, int] = defaultdict(int)

    for event in events:
        try:
            if event.get("contract_id") != CONTRACT_ID:
                errors.append("event has wrong contract_id")
            generation_seed = int(event["generation_seed"])
            seed_key = int(event["seed_key"])
            seat = int(event["seat"])
            kyoku_index = int(event["kyoku_index"])
            decision_index = int(event["decision_index"])
            identity = (generation_seed, seed_key, seat, kyoku_index, decision_index)
            if identity in seen:
                errors.append(f"duplicate decision context: {identity}")
            seen.add(identity)
            if (generation_seed, seed_key) not in expected_seed_keys:
                errors.append(f"event seed outside expected range: {identity}")
            if seat not in range(4) or kyoku_index < 0 or decision_index < 0:
                errors.append(f"invalid context indices: {identity}")
            top1_action = int(event["top1_action"])
            top2_action = int(event["top2_action"])
            if not (0 <= top1_action < 37 and 0 <= top2_action < 37 and top1_action != top2_action):
                errors.append(f"eligible event is not discard->discard: {identity}")
            top1_q = float(event["top1_q"])
            top2_q = float(event["top2_q"])
            margin = float(event["margin"])
            if not (math.isfinite(top1_q) and math.isfinite(top2_q) and math.isfinite(margin)):
                errors.append(f"non-finite event Q: {identity}")
            if margin > 0.5 + 1e-12:
                errors.append(f"margin threshold violation: {identity}")
            if bool(event["own_riichi"]):
                errors.append(f"own-riichi event: {identity}")
            if event.get("context_kind") != "primary_action" or event.get("exploration_allowed") is not True:
                errors.append(f"event is not a primary action context: {identity}")

            expected_canonical, expected_digest, expected_u = canonical_hash_u(
                generation_seed, seed_key, seat, kyoku_index, decision_index
            )
            if event.get("hash_input") != expected_canonical or event.get("hash_sha256") != expected_digest:
                errors.append(f"hash canonical mismatch: {identity}")
            if abs(float(event["hash_u"]) - expected_u) > 1e-18:
                errors.append(f"hash u mismatch: {identity}")

            kyoku_key = (generation_seed, seed_key, seat, kyoku_index)
            hanchan_key = (generation_seed, seed_key, seat)
            kyoku_before = kyoku_counts[kyoku_key]
            hanchan_before = hanchan_counts[hanchan_key]
            if int(event["kyoku_exploration_count_before"]) != kyoku_before:
                errors.append(f"kyoku budget counter mismatch: {identity}")
            if int(event["hanchan_exploration_count_before"]) != hanchan_before:
                errors.append(f"hanchan budget counter mismatch: {identity}")

            reason = str(event["reason"])
            reason_counts[reason] += 1
            explored = bool(event["explored"])
            actual_action = int(event["actual_action"])
            if reason == "explored":
                if not explored or actual_action != top2_action or float(event["hash_u"]) >= EXPLORATION_PROBABILITY:
                    errors.append(f"explored decision contract violation: {identity}")
                kyoku_counts[kyoku_key] += 1
                hanchan_counts[hanchan_key] += 1
                explored_by_kyoku[kyoku_key] += 1
                explored_by_hanchan[hanchan_key] += 1
            elif reason == "hash_rejected":
                if explored or actual_action != top1_action or float(event["hash_u"]) < EXPLORATION_PROBABILITY:
                    errors.append(f"hash rejection contract violation: {identity}")
            elif reason == "kyoku_budget_exhausted":
                if explored or actual_action != top1_action or kyoku_before < KYOKU_BUDGET:
                    errors.append(f"kyoku budget contract violation: {identity}")
            elif reason == "hanchan_budget_exhausted":
                if explored or actual_action != top1_action or hanchan_before < HANCHAN_BUDGET:
                    errors.append(f"hanchan budget contract violation: {identity}")
            else:
                errors.append(f"unknown event reason {reason!r}: {identity}")
            if actual_action not in {top1_action, top2_action}:
                errors.append(f"actual action is not top1/top2: {identity}")
            if int(event["base_action"]) != top1_action:
                errors.append(f"base greedy action differs from stable top1: {identity}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"malformed exploration event: {exc}")

    errors.extend(
        f"kyoku exploration budget exceeded: {key} -> {value}"
        for key, value in explored_by_kyoku.items()
        if value > KYOKU_BUDGET
    )
    errors.extend(
        f"hanchan exploration budget exceeded: {key} -> {value}"
        for key, value in explored_by_hanchan.items()
        if value > HANCHAN_BUDGET
    )
    return {
        "event_count": len(events),
        "eligible_count": len(events),
        "explored_count": reason_counts.get("explored", 0),
        "hash_rejected_count": reason_counts.get("hash_rejected", 0),
        "reason_counts": dict(sorted(reason_counts.items())),
        "unique_decision_contexts": len(seen),
        "errors": errors,
        "passed": not errors,
    }


def _audit_run(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = root.resolve()
    protocol = _read_json(root / "protocol.json")
    expected_seeds = {(args.seed_start + offset, args.seed_key) for offset in range(args.games)}
    logs = sorted((root / "logs").glob("*.json.gz"))
    log_rows: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    for path in logs:
        try:
            events = _read_log(path)
            key = _log_key(events, path)
            key_text = f"{key[0]}_{key[1]}"
            if key_text in log_rows:
                malformed.append(f"duplicate log seed: {key}")
            log_rows[key_text] = {"path": str(path), "canonical_sha256": _canonical_log_hash(events)}
            start_names = events[0].get("names", [])
            if sorted(start_names) != sorted(["K0_70k", "V2_74000", "V3_74000", "ext_mortal"]):
                malformed.append(f"wrong model names in {path}: {start_names!r}")
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{path}: {exc}")
    events = _load_events(root)
    event_audit = _audit_events(events, expected_seeds)
    checks = {
        "protocol_contract_id": protocol.get("contract_id") == CONTRACT_ID,
        "protocol_seed_range": protocol.get("seed_start") == args.seed_start
        and protocol.get("seed_end_exclusive") == args.seed_start + args.games,
        "protocol_seed_key": protocol.get("seed_key") == args.seed_key,
        "protocol_games": protocol.get("games") == args.games,
        "protocol_native_batch_games": protocol.get("native_batch_games") == 25,
        "protocol_amp_false": protocol.get("amp") is False,
        "protocol_device_cuda": protocol.get("device") == "cuda",
        "protocol_cuda_available": protocol.get("cuda_available") is True,
        "protocol_project_clean": protocol.get("project_git_dirty") is False
        and protocol.get("git_dirty") is False,
        "protocol_project_commit": protocol.get("project_git_commit") == args.expected_project_commit
        and protocol.get("git_commit") == args.expected_project_commit,
        "protocol_mortal_clean": protocol.get("mortal_source_dirty") is False,
        "protocol_mortal_commit": protocol.get("mortal_source_commit")
        == args.expected_mortal_source_commit,
        "protocol_native_patch": protocol.get("d3_native_patch_sha256")
        == args.expected_native_patch_sha256,
        "protocol_native_binary": protocol.get("loaded_libriichi_sha256")
        == args.expected_native_binary_sha256,
        "protocol_native_profile": protocol.get("native_build_profile") == "release",
        "protocol_native_path_present": bool(protocol.get("loaded_libriichi_path")),
        "protocol_model_manifest": set(protocol.get("models", {}))
        == {"K0_70k", "V2_74000", "V3_74000", "ext_mortal"}
        and all(
            isinstance(value, dict) and bool(value.get("sha256"))
            for value in protocol.get("models", {}).values()
        ),
        "protocol_auxiliary_exploration_zero": protocol.get("exploration_counters", {}).get(
            "auxiliary_exploration_count", 0
        )
        == 0,
        "log_count": len(logs) == args.games,
        "unique_seed_count": len(log_rows) == args.games,
        "expected_seed_set": set(log_rows)
        == {f"{seed}_{key}" for seed, key in expected_seeds},
        "malformed_log_count": not malformed,
        "event_eligible_count_positive": event_audit["eligible_count"] > 0,
        "event_explored_count_positive": event_audit["explored_count"] > 0,
        "event_hash_rejected_count_positive": event_audit["hash_rejected_count"] > 0,
        "event_contract": event_audit["passed"],
    }
    return {
        "root": str(root),
        "protocol": protocol,
        "checks": checks,
        "logs": log_rows,
        "malformed": malformed,
        "events": event_audit,
        "passed": all(checks.values()),
    }


def _compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    left_keys = set(left["logs"])
    right_keys = set(right["logs"])
    if left_keys != right_keys:
        errors.append(f"run seed sets differ: {sorted(left_keys)} vs {sorted(right_keys)}")
    for key in sorted(left_keys & right_keys):
        if left["logs"][key]["canonical_sha256"] != right["logs"][key]["canonical_sha256"]:
            errors.append(f"canonical log hash differs for seed {key}")

    left_events = _load_events(Path(left["root"]))
    right_events = _load_events(Path(right["root"]))
    left_digest = hashlib.sha256(
        (json.dumps(left_events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    right_digest = hashlib.sha256(
        (json.dumps(right_events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    if left_events != right_events:
        errors.append("sorted exploration event records differ")
    left_protocol = left["protocol"]
    right_protocol = right["protocol"]
    for field in (
        "project_git_commit",
        "mortal_source_commit",
        "d3_native_patch_sha256",
        "loaded_libriichi_sha256",
        "native_build_profile",
        "device",
        "cuda_version",
        "torch_version",
    ):
        if left_protocol.get(field) != right_protocol.get(field):
            errors.append(f"protocol field differs: {field}")
    if left_protocol.get("models") != right_protocol.get("models"):
        errors.append("model manifest differs")
    return {
        "canonical_log_hashes_equal": not any("canonical log hash" in error for error in errors),
        "event_records_equal": left_events == right_events,
        "event_digest_a": left_digest,
        "event_digest_b": right_digest,
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    args = parse_args()
    run_a = _audit_run(args.run_a, args)
    run_b = _audit_run(args.run_b, args)
    comparison = _compare_runs(run_a, run_b)
    report = {
        "schema": "keqing.mortal.d3_generation_smoke_audit.v1",
        "contract_id": CONTRACT_ID,
        "run_a": run_a,
        "run_b": run_b,
        "determinism": comparison,
        "passed": bool(run_a["passed"] and run_b["passed"] and comparison["passed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    lines = [
        "# D3 generation v1 smoke audit",
        "",
        f"- status: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- contract: `{CONTRACT_ID}`",
        f"- runs: `{args.run_a.resolve()}` and `{args.run_b.resolve()}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in run_a["checks"].items():
        lines.append(f"- run A `{name}`: {'PASS' if value else 'FAIL'}")
    for name, value in run_b["checks"].items():
        lines.append(f"- run B `{name}`: {'PASS' if value else 'FAIL'}")
    lines.extend([
        f"- deterministic log/event comparison: {'PASS' if comparison['passed'] else 'FAIL'}",
        "",
        "## Exploration counts",
        "",
        f"- run A: `{run_a['events']}`",
        f"- run B: `{run_b['events']}`",
        "",
        "Pt/rank/behavior metrics are not used as a smoke gate.",
    ])
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
