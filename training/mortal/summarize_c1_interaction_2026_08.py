#!/usr/bin/env python3
"""Parse complete C1 hanchans and adjudicate the frozen interaction gate.

The formal command requires all 24 B250 shards and all 6000 raw logs.  The
pure functions in this module intentionally accept synthetic rows so the
contract can be tested without importing or running the arena.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.c1_evaluation_contract_2026_08 import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    C1_ID,
    CONDITIONS,
    EVALUATION_PLAN_PATH,
    EVALUATION_ROOT,
    GAMES_PER_SHARD,
    IMPLEMENTATION_PREFLIGHT_PATH,
    RANK_POINTS,
    SHARD_STARTS,
    TOTAL_SHARDS,
    TRAINING_SEEDS,
    ContractError,
    current_checkpoint_records,
    dump_json,
    load_json,
    off_model_label,
    sha256_file,
    validate_runtime_provenance,
    validate_source_provenance,
)

LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)(?:_[^/]*)?\.json\.gz$")
ROLE_ORDER = ("70k", "ext_mortal", "M0", "D1")


def final_scores(events: list[dict[str, Any]]) -> list[float] | None:
    """Reconstruct final scores with the corrected ReachAccepted semantics."""

    scores: list[float] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(value) for value in values]
        elif event_type == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif event_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def ranks_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    if not events or events[0].get("type") != "start_game":
        raise ContractError("raw hanchan is missing the first start_game event")
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4 or len(set(names)) != 4:
        raise ContractError(f"raw hanchan has invalid four-seat names: {names!r}")
    scores = final_scores(events)
    if scores is None:
        raise ContractError("raw hanchan has no reconstructible final scores")
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    return {str(names[seat]): order.index(seat) + 1 for seat in range(4)}


def authoritative_ranks_from_log(raw_text: str, stat_cls: Any = None) -> list[int]:
    """Return strict native Stat ranks; no actual-evaluation fallback exists."""

    if stat_cls is None:
        try:
            from libriichi.stat import Stat as stat_cls
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise ContractError("libriichi.stat.Stat is required for real C1 rank equivalence") from exc
    ranks: list[int] = []
    for seat in range(4):
        stat = stat_cls.from_log(raw_text, seat)
        candidates = [rank for rank in (1, 2, 3, 4) if getattr(stat, f"rank_{rank}") == 1]
        if len(candidates) != 1:
            raise ContractError(f"native Stat rank is ambiguous for seat {seat}: {candidates}")
        ranks.append(candidates[0])
    return ranks


def normalize_role(label: str) -> str:
    if label in {"70k", "ext_mortal"}:
        return label
    if label.startswith("M0_"):
        return "M0"
    if label.startswith("D1_"):
        return "D1"
    raise ContractError(f"unknown C1 lineup label: {label}")


def points_for_ranks(ranks: Mapping[str, int]) -> dict[str, float]:
    if set(ranks) != set(ROLE_ORDER) or len(ranks) != 4:
        raise ContractError(f"rank labels are not a four-seat C1 lineup: {sorted(ranks)}")
    values = {str(label): int(rank) for label, rank in ranks.items()}
    if sorted(values.values()) != [1, 2, 3, 4]:
        raise ContractError(f"ranks are not a permutation of 1..4: {values}")
    return {label: RANK_POINTS[rank - 1] for label, rank in values.items()}


def _hanchan_seed_from_filename(path: Path) -> int:
    match = LOG_NAME_RE.match(path.name)
    if match is None:
        raise ContractError(f"raw hanchan filename has no integer seed prefix: {path.name}")
    return int(match.group("seed"))


def _log_seed_key(events: list[dict[str, Any]]) -> tuple[int, int]:
    start = events[0]
    raw_seed = start.get("seed")
    if not isinstance(raw_seed, (list, tuple)) or len(raw_seed) != 2:
        raise ContractError("start_game does not carry [hanchan_seed, seed_key]")
    try:
        return int(raw_seed[0]), int(raw_seed[1])
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid start_game seed: {raw_seed!r}") from exc


def parse_raw_log(
    path: Path,
    *,
    condition: str,
    training_seed: int,
    expected_seed_start: int,
    expected_seed_end: int,
    stat_cls: Any = None,
) -> dict[str, Any]:
    if condition not in CONDITIONS or training_seed not in TRAINING_SEEDS:
        raise ContractError(f"invalid raw-log identity: {condition}/{training_seed}")
    hanchan_seed = _hanchan_seed_from_filename(path)
    if not expected_seed_start <= hanchan_seed < expected_seed_end:
        raise ContractError(f"hanchan seed outside frozen shard range: {path.name}")
    try:
        raw_bytes = path.read_bytes()
        raw_text = gzip.decompress(raw_bytes).decode("utf-8")
        events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse raw hanchan log: {path}: {exc}") from exc
    if not events or not all(isinstance(event, dict) for event in events):
        raise ContractError(f"raw hanchan log is empty or not event objects: {path}")
    if events[0].get("type") != "start_game":
        raise ContractError(f"raw hanchan log does not begin with start_game: {path}")
    if sum(event.get("type") == "start_game" for event in events) != 1:
        raise ContractError(f"raw log contains more than one hanchan: {path}")
    if not any(event.get("type") == "start_kyoku" for event in events):
        raise ContractError(f"raw hanchan has no kyoku: {path}")
    if not any(event.get("type") in {"end_kyoku", "end_game"} for event in events):
        raise ContractError(f"raw hanchan has no complete-game marker: {path}")
    log_seed, seed_key = _log_seed_key(events)
    if log_seed != hanchan_seed or seed_key != 8192:
        raise ContractError(f"raw hanchan seed/key mismatch: {path}")
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4 or len(set(names)) != 4:
        raise ContractError(f"raw hanchan seat order is invalid: {path}")
    roles = [normalize_role(str(name)) for name in names]
    if set(roles) != set(ROLE_ORDER) or len(set(roles)) != 4:
        raise ContractError(f"raw hanchan lineup role order is not frozen: {path}: {roles}")
    ranks = ranks_from_events(events)
    points = points_for_ranks({role: ranks[str(name)] for role, name in zip(roles, names, strict=True)})
    native_ranks = authoritative_ranks_from_log(raw_text, stat_cls=stat_cls)
    reconstructed_seat_ranks = [ranks[str(name)] for name in names]
    if reconstructed_seat_ranks != native_ranks:
        raise ContractError(f"native Stat rank mismatch: {path}")
    role_to_seat = {role: seat for seat, role in enumerate(roles)}
    return {
        "condition": condition,
        "training_seed": training_seed,
        "hanchan_seed": hanchan_seed,
        "seed_key": seed_key,
        "source_log": str(path.resolve()),
        "source_log_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "seat_order": [str(name) for name in names],
        "role_order": list(ROLE_ORDER),
        "role_to_seat": role_to_seat,
        "ranks": {str(name): ranks[str(name)] for name in names},
        "ranks_by_role": {role: ranks[str(name)] for role, name in zip(roles, names, strict=True)},
        "pts": points,
        "gap_pt_d1_minus_m0": points["D1"] - points["M0"],
        "native_stat_ranks_by_seat": native_ranks,
    }


def validate_hanchan_seed_set(rows: Iterable[Mapping[str, Any]], *, start: int, games: int = 1000) -> None:
    values = [int(row["hanchan_seed"]) for row in rows]
    expected = set(range(start, start + games))
    actual = set(values)
    if len(values) != games:
        raise ContractError(f"expected {games} complete hanchans, found {len(values)}")
    if len(actual) != len(values):
        raise ContractError("duplicate hanchan seed in evaluation block")
    if actual != expected:
        missing = sorted(expected - actual)[:8]
        extra = sorted(actual - expected)[:8]
        raise ContractError(f"hanchan seed range mismatch; missing={missing}, extra={extra}")


def validate_row_integrity(row: Mapping[str, Any]) -> None:
    required = {"condition", "training_seed", "hanchan_seed", "seed_key", "role_order", "role_to_seat", "ranks_by_role", "pts", "gap_pt_d1_minus_m0"}
    missing = sorted(required - set(row))
    if missing:
        raise ContractError(f"interaction input row is missing fields: {missing}")
    if tuple(row["role_order"]) != ROLE_ORDER:
        raise ContractError("interaction input role order mismatch")
    role_to_seat = {str(key): int(value) for key, value in dict(row["role_to_seat"]).items()}
    if set(role_to_seat) != set(ROLE_ORDER) or sorted(role_to_seat.values()) != [0, 1, 2, 3]:
        raise ContractError("interaction input role-to-seat assignment is not a seat permutation")
    ranks = {str(key): int(value) for key, value in dict(row["ranks_by_role"]).items()}
    points = points_for_ranks(ranks)
    if dict(row["pts"]) != points:
        raise ContractError("interaction input rank-point mapping mismatch")
    expected_gap = points["D1"] - points["M0"]
    if not math.isclose(float(row["gap_pt_d1_minus_m0"]), expected_gap, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("interaction input D1-M0 point gap mismatch")


def pair_current_off(current_rows: Iterable[Mapping[str, Any]], off_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    current = list(current_rows)
    off = list(off_rows)
    if not current or not off:
        raise ContractError("CURRENT/OFF pair is empty")
    current_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    off_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in current:
        validate_row_integrity(row)
        if row.get("condition") != "CURRENT":
            raise ContractError("CURRENT pair input has wrong condition")
        key = (int(row["training_seed"]), int(row["hanchan_seed"]))
        if key in current_by_key:
            raise ContractError(f"duplicate CURRENT pair key: {key}")
        current_by_key[key] = row
    for row in off:
        validate_row_integrity(row)
        if row.get("condition") != "CQL_OFF":
            raise ContractError("CQL_OFF pair input has wrong condition")
        key = (int(row["training_seed"]), int(row["hanchan_seed"]))
        if key in off_by_key:
            raise ContractError(f"duplicate CQL_OFF pair key: {key}")
        off_by_key[key] = row
    if set(current_by_key) != set(off_by_key):
        raise ContractError("CURRENT/OFF hanchan identities are not exactly paired")
    paired: list[dict[str, Any]] = []
    for key in sorted(current_by_key):
        current_row = current_by_key[key]
        off_row = off_by_key[key]
        for field in ("training_seed", "seed_key", "role_order", "role_to_seat"):
            if current_row.get(field) != off_row.get(field):
                raise ContractError(f"CURRENT/OFF pairing gate mismatch at {key}: {field}")
        d_current = float(current_row["gap_pt_d1_minus_m0"])
        d_off = float(off_row["gap_pt_d1_minus_m0"])
        paired.append(
            {
                "training_seed": key[0],
                "hanchan_seed": key[1],
                "seed_key": int(current_row["seed_key"]),
                "role_order": list(ROLE_ORDER),
                "role_to_seat": dict(current_row["role_to_seat"]),
                "current_source_log": current_row.get("source_log"),
                "off_source_log": off_row.get("source_log"),
                "d_current": d_current,
                "d_off": d_off,
                "interaction_row": d_off - d_current,
            }
        )
    return paired


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    if ordered.size == 0:
        raise ContractError("cannot calculate a percentile of no bootstrap values")
    return float(np.quantile(ordered, probability, method="linear"))


def hierarchical_bootstrap(
    interaction_rows_by_seed: Mapping[int, Iterable[float]],
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if reps != BOOTSTRAP_REPS or seed != BOOTSTRAP_SEED:
        raise ContractError("C1 bootstrap B/seed are frozen at 5000/20260818")
    if tuple(sorted(interaction_rows_by_seed)) != TRAINING_SEEDS:
        raise ContractError("hierarchical bootstrap requires exactly the three frozen training seeds")
    arrays = [np.asarray(list(interaction_rows_by_seed[training_seed]), dtype=np.float64) for training_seed in TRAINING_SEEDS]
    if any(array.size != 1000 for array in arrays):
        raise ContractError("hierarchical bootstrap requires exactly 1000 interaction rows per seed")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(reps, dtype=np.float64)
    for replicate in range(reps):
        selected_seed_indexes = rng.integers(0, len(arrays), size=3)
        selected_means: list[float] = []
        for index in selected_seed_indexes:
            values = arrays[int(index)]
            sample_indexes = rng.integers(0, values.size, size=values.size)
            selected_means.append(float(values[sample_indexes].mean()))
        bootstrap_means[replicate] = float(np.mean(selected_means))
    ci95 = [percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)]
    return {
        "method": "equal_seed_hierarchical_interaction_rows",
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "hierarchical_equal_seed_bootstrap_ci95": ci95,
    }


def machine_adjudication(
    interaction_seed_means: Mapping[int, float],
    ci95: Iterable[float],
    gates: Mapping[str, bool],
) -> dict[str, Any]:
    if tuple(sorted(interaction_seed_means)) != TRAINING_SEEDS:
        raise ContractError("machine adjudication requires exactly three seed means")
    interval = list(ci95)
    if len(interval) != 2:
        raise ContractError("machine adjudication requires a two-sided CI")
    gates_result = {str(name): bool(value) for name, value in gates.items()}
    all_gates_pass = all(gates_result.values()) and bool(gates_result)
    all_seed_positive = all(float(interaction_seed_means[seed]) > 0 for seed in TRAINING_SEEDS)
    ci_lower_positive = float(interval[0]) > 0
    if not all_gates_pass:
        verdict = "no_verdict_gates_failed"
    elif all_seed_positive and ci_lower_positive:
        verdict = "interaction_supported"
    else:
        verdict = "interaction_not_confirmed"
    return {
        "verdict": verdict,
        "interaction_seed_means": {str(seed): float(interaction_seed_means[seed]) for seed in TRAINING_SEEDS},
        "all_three_interaction_seed_means_positive": all_seed_positive,
        "hierarchical_ci95": [float(interval[0]), float(interval[1])],
        "hierarchical_ci_lower_positive": ci_lower_positive,
        "gates": gates_result,
        "all_gates_pass": all_gates_pass,
    }


def summarize_interaction_rows(
    paired_rows_by_seed: Mapping[int, Iterable[Mapping[str, Any]]],
    *,
    gates: Mapping[str, bool],
) -> dict[str, Any]:
    if tuple(sorted(paired_rows_by_seed)) != TRAINING_SEEDS:
        raise ContractError("C1 interaction summary requires exactly three training seeds")
    rows_by_seed = {seed: list(rows) for seed, rows in paired_rows_by_seed.items()}
    for seed in TRAINING_SEEDS:
        if len(rows_by_seed[seed]) != 1000:
            raise ContractError(f"C1 interaction summary requires 1000 pairs for seed {seed}")
        validate_hanchan_seed_set(rows_by_seed[seed], start=SHARD_STARTS[seed][0])
    interaction_rows_by_seed = {
        seed: [float(row["interaction_row"]) for row in rows_by_seed[seed]] for seed in TRAINING_SEEDS
    }
    seed_means = {seed: float(np.mean(values)) for seed, values in interaction_rows_by_seed.items()}
    primary_mean = float(np.mean(list(seed_means.values())))
    bootstrap = hierarchical_bootstrap(interaction_rows_by_seed)
    decision = machine_adjudication(
        seed_means,
        bootstrap["hierarchical_equal_seed_bootstrap_ci95"],
        gates,
    )
    return {
        "schema": "keqing.mortal.c1_interaction_summary.v1",
        "experiment_id": C1_ID,
        "training_seed_counts": {str(seed): len(rows_by_seed[seed]) for seed in TRAINING_SEEDS},
        "interaction_seed_means": {str(seed): seed_means[seed] for seed in TRAINING_SEEDS},
        "primary_interaction_mean": primary_mean,
        "bootstrap": bootstrap,
        "adjudication": decision,
        "paired_rows": [row for seed in TRAINING_SEEDS for row in rows_by_seed[seed]],
    }


def _plan_models(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): item for item in plan.get("models", []) if isinstance(item, dict) and item.get("label")}


def _formal_provenance_gates(plan: dict[str, Any], preflight: dict[str, Any], execution: dict[str, Any]) -> dict[str, bool]:
    gates = {
        "training_provenance": False,
        "evaluation_provenance": False,
        "runtime_provenance": False,
        "pairing_gate": True,
    }
    try:
        if preflight.get("implementation_preflight_passed") is not True:
            return gates
        execution_rows = {
            (str(row.get("route")), int(row.get("training_seed", -1))): row
            for row in execution.get("runs", [])
            if isinstance(row, dict)
        }
        expected_execution = {(f"{route}_CQL_OFF", seed) for route in ("M0", "D1") for seed in TRAINING_SEEDS}
        if set(execution_rows) != expected_execution:
            return gates
        models = _plan_models(plan)
        for route in ("M0", "D1"):
            for seed in TRAINING_SEEDS:
                label = off_model_label(route, seed)
                row = execution_rows[(f"{route}_CQL_OFF", seed)]
                model = models.get(label)
                if model is None or model.get("path") != row.get("final_checkpoint_path"):
                    return gates
                if model.get("sha256") is not None and model.get("sha256") != row.get("final_checkpoint_sha256"):
                    return gates
                checkpoint = Path(str(row.get("final_checkpoint_path")))
                if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("final_checkpoint_sha256"):
                    return gates
        current = current_checkpoint_records()
        if any(models.get(label, {}).get("sha256") != row.get("sha256") for label, row in current.items()):
            return gates
        gates["training_provenance"] = True
        expected_sources = {
            "evaluator": plan["evaluator_provenance"],
            "direct_dependencies": plan["evaluation_dependency_sources"],
            "mortal_revision": plan["mortal_revision"],
        }
        validate_source_provenance(expected_sources)
        gates["evaluation_provenance"] = True
        validate_runtime_provenance(plan["runtime_provenance"])
        gates["runtime_provenance"] = True
    except (ContractError, KeyError, OSError, ValueError):
        return gates
    return gates


def _read_formal_evaluation(plan: dict[str, Any], eval_root: Path, stat_cls: Any = None) -> tuple[dict[int, list[dict[str, Any]]], dict[str, bool]]:
    models = _plan_models(plan)
    if len(plan.get("runs", [])) != TOTAL_SHARDS:
        raise ContractError("formal C1 summary requires exactly 24 planned shards")
    current_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in TRAINING_SEEDS}
    off_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in TRAINING_SEEDS}
    for run in plan["runs"]:
        condition = str(run["condition"])
        seed = int(run["training_seed"])
        shard = int(run["shard"])
        output_dir = Path(str(run["output_dir"])).resolve()
        if output_dir.parent.parent.parent != eval_root.resolve():
            raise ContractError(f"evaluation output path escaped the frozen root: {output_dir}")
        log_dir = output_dir / "logs"
        logs = sorted(log_dir.glob("*.json.gz"))
        if len(logs) != GAMES_PER_SHARD:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: expected 250 raw logs, found {len(logs)}")
        for path in logs:
            row = parse_raw_log(
                path,
                condition=condition,
                training_seed=seed,
                expected_seed_start=int(run["hanchan_seed_start"]),
                expected_seed_end=int(run["hanchan_seed_end_exclusive"]),
                stat_cls=stat_cls,
            )
            (current_rows if condition == "CURRENT" else off_rows)[seed].append(row)
    for seed in TRAINING_SEEDS:
        validate_hanchan_seed_set(current_rows[seed], start=SHARD_STARTS[seed][0])
        validate_hanchan_seed_set(off_rows[seed], start=SHARD_STARTS[seed][0])
        if len(current_rows[seed]) != 1000 or len(off_rows[seed]) != 1000:
            raise ContractError(f"formal C1 seed block is not exactly 1000/1000: {seed}")
    paired = {seed: pair_current_off(current_rows[seed], off_rows[seed]) for seed in TRAINING_SEEDS}
    return paired, {"pairing_gate": True, "plan_models_complete": len(models) == 14}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--plan", type=Path, default=EVALUATION_PLAN_PATH)
    parser.add_argument("--preflight", type=Path, default=IMPLEMENTATION_PREFLIGHT_PATH)
    parser.add_argument("--execution-manifest", type=Path, default=EVALUATION_ROOT / "execution_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_json(args.plan.resolve())
    preflight = load_json(args.preflight.resolve())
    execution = load_json(args.execution_manifest.resolve())
    paired, structural_gates = _read_formal_evaluation(plan, args.eval_root.resolve())
    gates = _formal_provenance_gates(plan, preflight, execution)
    gates.update(structural_gates)
    summary = summarize_interaction_rows(paired, gates=gates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "c1_interaction_summary.json"
    dump_json(output, summary)
    print(json.dumps({"summary": str(output), "verdict": summary["adjudication"]["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
