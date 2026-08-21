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
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal import run_c1_evaluation_2026_08 as evaluation_launcher
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
    SHARDS,
    TOTAL_GAMES,
    TOTAL_SHARDS,
    TRAINING_SEEDS,
    ContractError,
    assert_exact_run_matrix,
    current_checkpoint_records,
    evaluation_shard_dir,
    git_blob_oid,
    git_info,
    load_json,
    model_order,
    off_model_label,
    resolve_execution_manifest,
    sha256_file,
    validate_git_scope,
    validate_runtime_provenance,
    validate_source_provenance,
)

LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)(?:_[^/]*)?\.json\.gz$")
ROLE_ORDER = ("70k", "ext_mortal", "M0", "D1")

EVALUATION_EXECUTION_INVENTORY_PATH = EVALUATION_ROOT / "evaluation_execution_inventory.json"
EVALUATION_EXECUTION_INVENTORY_SHA256 = (
    "6d9cc23a8a5de778e2e2bb743c11aa995f196bac8e2f037eadf59f3a595dd648"
)
EVALUATION_AUTHORIZATION_COMMIT = "e12c0991b8753a865f09db1590232755ea358201"
AUTHORIZED_ARTIFACT_SHA256 = {
    "plan": "2d9f85144492cbd2f86c786ebc5d6ad10722ce8449bdcde5176bf8f6578f18f7",
    "preflight": "9f1ecbe20473b3ceccfcdc70aa1a26b041890c532f0f15fc5a6b476abd20c0d0",
    "completion": "cdaaaa8d67bc8497ad7dcf279de78db5bc0110d071882ab47daaf1b3e07b2b9f",
    "execution": "241cfcb5559598fa5c103161ceb28363c80af65e6944f894d3fc2fde7fd1a151",
}
CANONICAL_FORMAL_ADJUDICATION_DIR = EVALUATION_ROOT / "formal_adjudication"
FORMAL_SUMMARY_FILENAME = "c1_interaction_summary.json"


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
    log_seed, seed_key = _log_seed_key(events)
    if log_seed != hanchan_seed or seed_key != 8192:
        raise ContractError(f"raw hanchan seed/key mismatch: {path}")
    names = events[0].get("names")
    if (
        not isinstance(names, list)
        or len(names) != 4
        or not all(isinstance(name, str) for name in names)
        or len(set(names)) != 4
    ):
        raise ContractError(f"raw hanchan seat order is invalid: {path}")
    expected_labels = set(model_order(condition, training_seed))
    if set(names) != expected_labels:
        raise ContractError(
            f"raw hanchan model identity mismatch: {path}: "
            f"{sorted(names)} != {sorted(expected_labels)}"
        )
    roles = [normalize_role(name) for name in names]
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
                "current_source_log_sha256": current_row.get("source_log_sha256"),
                "off_source_log": off_row.get("source_log"),
                "off_source_log_sha256": off_row.get("source_log_sha256"),
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


TRAINING_COMPLETION_CLOSURE_PATH = EVALUATION_ROOT / "training_completion_closure.json"
EXECUTION_MANIFEST_PATH = EVALUATION_ROOT / "execution_manifest.json"
EXECUTION_FIELDS = (
    "route",
    "training_seed",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "steps",
    "trained_optimizer_steps",
    "parent_checkpoint_sha256",
    "cql_min_q_weight",
    "objective",
    "reward",
    "initialization",
    "data_seed",
)


def _plan_models(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_models = plan.get("models")
    if not isinstance(raw_models, list):
        raise ContractError("evaluation plan has no model list")
    models: dict[str, dict[str, Any]] = {}
    for item in raw_models:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"]:
            raise ContractError("evaluation plan contains an invalid model record")
        label = str(item["label"])
        if label in models:
            raise ContractError(f"evaluation plan contains duplicate model label: {label}")
        models[label] = item
    return models


def _expected_model_labels() -> set[str]:
    labels = {"70k", "ext_mortal"}
    labels.update(model_order(condition, seed)[2] for condition in CONDITIONS for seed in TRAINING_SEEDS)
    labels.update(model_order(condition, seed)[3] for condition in CONDITIONS for seed in TRAINING_SEEDS)
    return labels


def _execution_rows(manifest: dict[str, Any], *, name: str) -> dict[tuple[str, int], dict[str, Any]]:
    if manifest.get("schema") != "keqing.mortal.c1_evaluation_execution_manifest.v1":
        raise ContractError(f"{name} schema mismatch")
    if manifest.get("experiment_id") != C1_ID:
        raise ContractError(f"{name} experiment mismatch")
    if manifest.get("status") != "resolved_not_authorized":
        raise ContractError(f"{name} status mismatch")
    if manifest.get("evaluation_authorized") is not False or manifest.get("evaluation_games_run") != 0:
        raise ContractError(f"{name} records authorization or evaluation")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise ContractError(f"{name} must contain exactly six runs")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_runs:
        if not isinstance(row, dict):
            raise ContractError(f"{name} contains a non-object run")
        route = str(row.get("route", ""))
        try:
            seed = int(row.get("training_seed", -1))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name} contains an invalid training seed") from exc
        key = (route, seed)
        if key in result:
            raise ContractError(f"{name} contains duplicate run: {key}")
        result[key] = row
    expected = {(f"{route}_CQL_OFF", seed) for route in ("M0", "D1") for seed in TRAINING_SEEDS}
    if set(result) != expected:
        raise ContractError(f"{name} does not contain exactly M0_CQL_OFF/D1_CQL_OFF x three seeds")
    for key, row in result.items():
        missing = [field for field in EXECUTION_FIELDS if field not in row]
        if missing:
            raise ContractError(f"{name} run {key} is missing fields: {missing}")
    return result


def _require_formal_authorization() -> tuple[str, dict[str, Path], dict[str, str]]:
    if evaluation_launcher.EVALUATION_AUTHORIZED is not True:
        raise ContractError("C1 formal summary is not authorized: EVALUATION_AUTHORIZED is not true")
    approved_commit = evaluation_launcher.APPROVED_EVALUATION_IMPLEMENTATION_COMMIT
    if not isinstance(approved_commit, str) or not approved_commit.strip():
        raise ContractError("C1 formal summary has no approved implementation commit binding")
    paths = {
        "plan": EVALUATION_PLAN_PATH,
        "preflight": IMPLEMENTATION_PREFLIGHT_PATH,
        "completion": TRAINING_COMPLETION_CLOSURE_PATH,
        "execution": EXECUTION_MANIFEST_PATH,
    }
    bindings = {
        "plan": evaluation_launcher.AUTHORIZED_EVALUATION_PLAN_SHA256,
        "preflight": evaluation_launcher.AUTHORIZED_EVALUATION_PREFLIGHT_SHA256,
        "completion": evaluation_launcher.AUTHORIZED_TRAINING_COMPLETION_SHA256,
        "execution": evaluation_launcher.AUTHORIZED_EXECUTION_MANIFEST_SHA256,
    }
    actual: dict[str, str] = {}
    for name, expected in bindings.items():
        if not isinstance(expected, str) or not expected.strip():
            raise ContractError(f"C1 formal summary has an empty authorization binding: {name}")
        path = paths[name]
        try:
            actual[name] = sha256_file(path.resolve())
        except OSError as exc:
            raise ContractError(f"authorized C1 artifact is missing: {path}") from exc
        if actual[name] != expected:
            raise ContractError(f"authorized C1 artifact SHA mismatch: {name}")
    return approved_commit, paths, actual


def _validate_execution_inventory() -> tuple[dict[str, Any], str]:
    """Validate the immutable E1 execution inventory before formal adjudication."""

    path = EVALUATION_EXECUTION_INVENTORY_PATH.resolve()
    if not path.is_file():
        raise ContractError(f"C1 execution inventory is missing: {path}")
    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        raise ContractError(f"cannot hash C1 execution inventory: {path}") from exc
    if actual_sha256 != EVALUATION_EXECUTION_INVENTORY_SHA256:
        raise ContractError("C1 execution inventory SHA mismatch")
    inventory = load_json(path)

    if inventory.get("schema") != "keqing.mortal.c1_evaluation_execution_inventory.v1":
        raise ContractError("C1 execution inventory schema mismatch")
    if inventory.get("experiment_id") != C1_ID:
        raise ContractError("C1 execution inventory experiment mismatch")
    if inventory.get("evaluation_authorization_commit") != EVALUATION_AUTHORIZATION_COMMIT:
        raise ContractError("C1 execution inventory authorization commit mismatch")
    for field, expected in AUTHORIZED_ARTIFACT_SHA256.items():
        inventory_field = {
            "plan": "plan_sha256",
            "preflight": "preflight_sha256",
            "completion": "closure_sha256",
            "execution": "execution_manifest_sha256",
        }[field]
        if inventory.get(inventory_field) != expected:
            raise ContractError(f"C1 execution inventory {inventory_field} mismatch")

    expected_counts = {
        "total_shards": TOTAL_SHARDS,
        "games_per_shard": GAMES_PER_SHARD,
        "total_games": TOTAL_GAMES,
        "total_raw_logs": TOTAL_GAMES,
        "metrics_file_count": TOTAL_SHARDS,
        "detailed_stats_file_count": TOTAL_SHARDS,
    }
    for field, expected in expected_counts.items():
        if inventory.get(field) != expected:
            raise ContractError(f"C1 execution inventory {field} mismatch")

    raw_shards = inventory.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != TOTAL_SHARDS:
        raise ContractError("C1 execution inventory must contain exactly 24 shards")
    expected_keys = {
        (condition, training_seed, shard)
        for condition in CONDITIONS
        for training_seed in TRAINING_SEEDS
        for shard in SHARDS
    }
    actual_keys: set[tuple[str, int, int]] = set()
    for item in raw_shards:
        if not isinstance(item, dict):
            raise ContractError("C1 execution inventory contains a non-object shard")
        try:
            condition = str(item["condition"])
            training_seed = int(item["training_seed"])
            shard = int(item["shard"])
            seed_start = int(item["seed_start"])
            seed_end = int(item["seed_end_exclusive"])
            raw_log_count = int(item["raw_log_count"])
            artifact_file_count = int(item["artifact_file_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("C1 execution inventory contains an invalid shard record") from exc
        key = (condition, training_seed, shard)
        if key in actual_keys:
            raise ContractError(f"C1 execution inventory contains duplicate shard: {key}")
        actual_keys.add(key)
        if key not in expected_keys:
            raise ContractError(f"C1 execution inventory contains an unexpected shard: {key}")
        expected_start = SHARD_STARTS[training_seed][shard]
        if (seed_start, seed_end) != (expected_start, expected_start + GAMES_PER_SHARD):
            raise ContractError(f"C1 execution inventory seed range mismatch: {key}")
        if raw_log_count != GAMES_PER_SHARD or artifact_file_count != GAMES_PER_SHARD + 2:
            raise ContractError(f"C1 execution inventory shard count mismatch: {key}")
        shard_inventory_sha = item.get("inventory_sha256")
        if not isinstance(shard_inventory_sha, str) or re.fullmatch(r"[0-9a-f]{64}", shard_inventory_sha) is None:
            raise ContractError(f"C1 execution inventory shard digest is invalid: {key}")
    if actual_keys != expected_keys:
        raise ContractError("C1 execution inventory shard matrix mismatch")
    return inventory, actual_sha256


def formal_summary_source_provenance() -> dict[str, str]:
    """Return the current tracked source identity used to create a formal result."""

    try:
        info = git_info()
        validate_git_scope(info)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise ContractError("C1 formal summary requires a clean tracked main worktree") from exc
    source_path = Path(__file__).resolve()
    try:
        relative_path = source_path.relative_to(SCRIPT_REPO_ROOT)
    except ValueError as exc:
        raise ContractError("C1 formal summary source is outside the repository") from exc
    return {
        "path": relative_path.as_posix(),
        "git_commit": str(info["commit"]),
        "content_sha256": sha256_file(source_path),
        "git_blob_oid": git_blob_oid(source_path),
    }


def _repo_or_absolute_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(SCRIPT_REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_formal_provenance(
    *,
    approved_commit: str,
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
    inventory: Mapping[str, Any],
    inventory_path: Path,
    inventory_sha256: str,
    plan: Mapping[str, Any],
    source_provenance: Mapping[str, str],
    paired_rows_by_seed: Mapping[int, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the provenance block attached to the future formal result."""

    artifact_keys = {
        "plan": "evaluation_plan",
        "preflight": "implementation_preflight",
        "completion": "training_completion_closure",
        "execution": "execution_manifest",
    }
    authorized_artifacts: dict[str, dict[str, str]] = {}
    for source_key, result_key in artifact_keys.items():
        if source_key not in artifact_paths or source_key not in artifact_hashes:
            raise ContractError(f"formal provenance is missing authorized artifact: {source_key}")
        digest = str(artifact_hashes[source_key])
        if not digest:
            raise ContractError(f"formal provenance has an empty artifact SHA: {source_key}")
        authorized_artifacts[result_key] = {
            "path": _repo_or_absolute_path(Path(artifact_paths[source_key])),
            "sha256": digest,
        }

    if inventory.get("evaluation_authorization_commit") != EVALUATION_AUTHORIZATION_COMMIT:
        raise ContractError("formal provenance inventory authorization commit mismatch")
    if not isinstance(plan.get("evaluator_provenance"), dict) or not isinstance(plan.get("runtime_provenance"), dict):
        raise ContractError("formal provenance is missing frozen evaluator/runtime provenance")
    if not approved_commit:
        raise ContractError("formal provenance has no approved evaluation implementation commit")

    row_counts = {int(seed): len(list(rows)) for seed, rows in paired_rows_by_seed.items()}
    if tuple(sorted(row_counts)) != TRAINING_SEEDS or any(count != 1000 for count in row_counts.values()):
        raise ContractError("formal provenance requires exactly 1000 paired rows for each training seed")
    paired_rows = sum(row_counts.values())
    return {
        "evaluation_authorization_commit": EVALUATION_AUTHORIZATION_COMMIT,
        "approved_evaluation_implementation_commit": approved_commit,
        "authorized_artifacts": authorized_artifacts,
        "evaluation_execution_inventory": {
            "path": _repo_or_absolute_path(inventory_path),
            "sha256": inventory_sha256,
        },
        "formal_summary_source": dict(source_provenance),
        "evaluator": dict(plan["evaluator_provenance"]),
        "runtime": dict(plan["runtime_provenance"]),
        "evaluation_counts": {
            "shards": TOTAL_SHARDS,
            "raw_logs": paired_rows * 2,
            "current_hanchans": paired_rows,
            "cql_off_hanchans": paired_rows,
            "paired_interaction_rows": paired_rows,
        },
    }


def publish_atomic(output_path: Path, value: dict[str, Any]) -> None:
    """Publish one JSON result atomically, refusing an existing final path."""

    output_path = output_path.resolve()
    if output_path.exists():
        raise ContractError(f"formal C1 result already exists; refusing overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.exists():
            raise ContractError(f"formal C1 result appeared during publication; refusing overwrite: {output_path}")
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_formal_artifact_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], str]:
    approved_commit, paths, hashes = _require_formal_authorization()
    plan = load_json(paths["plan"].resolve())
    preflight = load_json(paths["preflight"].resolve())
    completion = load_json(paths["completion"].resolve())
    execution = load_json(paths["execution"].resolve())

    if plan.get("schema") != "keqing.mortal.c1_evaluation_plan.v1" or plan.get("experiment_id") != C1_ID:
        raise ContractError("authorized C1 evaluation plan schema/experiment mismatch")
    if plan.get("evaluation_authorized") is not False or plan.get("evaluation_games_run") != 0:
        raise ContractError("authorized C1 evaluation plan records authorization or games")
    if plan.get("git_scope", {}).get("commit") != approved_commit:
        raise ContractError("authorized C1 plan implementation commit mismatch")
    if preflight.get("implementation_preflight_passed") is not True or preflight.get("passed") is not True:
        raise ContractError("authorized C1 implementation preflight is not passed")
    if preflight.get("plan_sha256") != hashes["plan"]:
        raise ContractError("authorized C1 preflight does not bind the exact plan SHA")
    if preflight.get("git", {}).get("commit") != approved_commit:
        raise ContractError("authorized C1 preflight implementation commit mismatch")
    if preflight.get("evaluation_games_run") != 0 or preflight.get("new_checkpoints") != 0:
        raise ContractError("authorized C1 preflight records execution")
    if completion.get("experiment_id") != C1_ID:
        raise ContractError("training completion closure experiment mismatch")

    models = _plan_models(plan)
    if len(models) != 14 or set(models) != _expected_model_labels():
        raise ContractError("authorized C1 plan must contain exactly the frozen 14 model records")
    assert_exact_run_matrix(plan.get("runs", []), models)

    resolved = resolve_execution_manifest(plan, completion)
    actual_rows = _execution_rows(execution, name="execution manifest")
    resolved_rows = _execution_rows(resolved, name="resolved execution manifest")
    for key in sorted(resolved_rows):
        actual = actual_rows[key]
        expected = resolved_rows[key]
        for field in (*EXECUTION_FIELDS, "label"):
            if actual.get(field) != expected.get(field):
                raise ContractError(f"execution manifest mismatch at {key}/{field}")

    effective_models = {label: dict(record) for label, record in models.items()}
    for (route, seed), row in resolved_rows.items():
        label = str(row["label"])
        effective_models[label]["path"] = str(Path(str(row["final_checkpoint_path"])).resolve())
        effective_models[label]["sha256"] = str(row["final_checkpoint_sha256"])
        if route != f"{label.split('_', 1)[0]}_CQL_OFF" or seed not in TRAINING_SEEDS:
            raise ContractError(f"resolved execution label mismatch at {route}/{seed}")
    return plan, preflight, completion, execution, effective_models, approved_commit


def _validate_formal_provenance(
    plan: dict[str, Any],
    preflight: dict[str, Any],
    execution: dict[str, Any],
    effective_models: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    if preflight.get("implementation_preflight_passed") is not True:
        raise ContractError("implementation preflight gate failed")
    execution_rows = _execution_rows(execution, name="execution manifest")
    current = current_checkpoint_records()
    for label, current_record in current.items():
        planned = effective_models.get(label)
        if planned is None:
            raise ContractError(f"CURRENT/anchor model is absent from the authorized plan: {label}")
        if Path(str(planned.get("path"))).resolve() != Path(str(current_record.get("path"))).resolve():
            raise ContractError(f"CURRENT/anchor checkpoint path mismatch: {label}")
        if planned.get("sha256") != current_record.get("sha256"):
            raise ContractError(f"CURRENT/anchor checkpoint SHA mismatch: {label}")
    for route in ("M0", "D1"):
        for seed in TRAINING_SEEDS:
            key = (f"{route}_CQL_OFF", seed)
            row = execution_rows[key]
            label = off_model_label(route, seed)
            model = effective_models.get(label)
            if model is None:
                raise ContractError(f"CQL_OFF model is absent from the authorized plan: {label}")
            checkpoint = Path(str(row["final_checkpoint_path"])).resolve()
            if Path(str(model["path"])).resolve() != checkpoint:
                raise ContractError(f"CQL_OFF checkpoint path mismatch: {label}")
            if model.get("sha256") != row["final_checkpoint_sha256"]:
                raise ContractError(f"CQL_OFF checkpoint SHA mismatch: {label}")
            if not checkpoint.is_file() or sha256_file(checkpoint) != row["final_checkpoint_sha256"]:
                raise ContractError(f"CQL_OFF checkpoint artifact mismatch: {label}")
    expected_sources = {
        "evaluator": plan["evaluator_provenance"],
        "direct_dependencies": plan["evaluation_dependency_sources"],
        "mortal_revision": plan["mortal_revision"],
    }
    validate_source_provenance(expected_sources)
    validate_runtime_provenance(plan["runtime_provenance"])
    return {
        "training_provenance": True,
        "evaluation_provenance": True,
        "runtime_provenance": True,
        "pairing_gate": True,
    }


def _formal_provenance_gates(plan: dict[str, Any], preflight: dict[str, Any], execution: dict[str, Any]) -> dict[str, bool]:
    """Compatibility wrapper for callers that want a boolean gate snapshot."""

    gates = {
        "training_provenance": False,
        "evaluation_provenance": False,
        "runtime_provenance": False,
        "pairing_gate": True,
    }
    try:
        models = _plan_models(plan)
        gates.update(_validate_formal_provenance(plan, preflight, execution, models))
    except (ContractError, KeyError, OSError, ValueError, TypeError):
        pass
    return gates


def _validate_shard_artifacts(
    *,
    run: dict[str, Any],
    output_dir: Path,
    rows: list[dict[str, Any]],
    effective_models: dict[str, dict[str, Any]],
) -> None:
    condition = str(run["condition"])
    seed = int(run["training_seed"])
    shard = int(run["shard"])
    labels = tuple(model_order(condition, seed))
    expected_paths = {
        label: str(Path(str(effective_models[label]["path"])).resolve()) for label in labels
    }
    metrics = load_json(output_dir / "metrics.json")
    detailed = load_json(output_dir / "detailed_stats.json")
    run_metrics = metrics.get("run")
    if not isinstance(run_metrics, dict):
        raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics.run is missing")
    expected_run = {
        "kind": "four_player_native",
        "backend": "libriichi.arena.FourPlayer",
        "seed_start": int(run["hanchan_seed_start"]),
        "seed_key": 8192,
        "games": 250,
        "native_batch_games": 250,
        "seat_mode": "random",
        "device": "cuda",
        "rank_points_values": [90.0, 45.0, 0.0, -135.0],
    }
    for field, expected in expected_run.items():
        if run_metrics.get(field) != expected:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics.run.{field} mismatch")
    actual_models = run_metrics.get("models")
    if not isinstance(actual_models, dict) or set(actual_models) != set(labels):
        raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics model labels mismatch")
    for label in labels:
        if actual_models.get(label) != expected_paths[label]:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: checkpoint path mismatch for {label}")

    metrics_by_label = metrics.get("metrics")
    if not isinstance(metrics_by_label, dict) or set(metrics_by_label) != set(labels):
        raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics labels mismatch")
    players = detailed.get("players")
    if not isinstance(players, dict) or set(players) != set(labels):
        raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: detailed-stats player labels mismatch")

    role_by_label = {label: normalize_role(label) for label in labels}
    reconstructed: dict[str, list[int]] = {}
    for label in labels:
        role = role_by_label[label]
        reconstructed[label] = [
            sum(int(row["ranks_by_role"][role]) == rank for row in rows) for rank in (1, 2, 3, 4)
        ]
        metric_row = metrics_by_label[label]
        if not isinstance(metric_row, dict) or metric_row.get("games") != 250:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics game count mismatch for {label}")
        if metric_row.get("rank_counts") != reconstructed[label]:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: raw/metrics rank mismatch for {label}")
        player = players[label]
        if not isinstance(player, dict) or not isinstance(player.get("raw"), dict):
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: detailed stats raw block missing for {label}")
        raw = player["raw"]
        if raw.get("game") != 250:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: detailed stats game count mismatch for {label}")
        detailed_counts = [raw.get(f"rank_{rank}") for rank in (1, 2, 3, 4)]
        if detailed_counts != reconstructed[label]:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: raw/detailed rank mismatch for {label}")


def _read_formal_evaluation(
    plan: dict[str, Any],
    eval_root: Path,
    stat_cls: Any = None,
    *,
    effective_models: dict[str, dict[str, Any]] | None = None,
    validate_shard_artifacts: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, bool]]:
    models = _plan_models(plan)
    if len(models) != 14 or set(models) != _expected_model_labels():
        raise ContractError("formal C1 summary requires exactly the frozen 14 model records")
    runs = plan.get("runs", [])
    assert_exact_run_matrix(runs, models)
    if len(runs) != TOTAL_SHARDS:
        raise ContractError("formal C1 summary requires exactly 24 planned shards")
    effective_models = effective_models or models
    current_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in TRAINING_SEEDS}
    off_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in TRAINING_SEEDS}
    frozen_root = eval_root.resolve()
    for run in runs:
        condition = str(run["condition"])
        seed = int(run["training_seed"])
        shard = int(run["shard"])
        output_dir = Path(str(run["output_dir"])).resolve()
        expected_output = evaluation_shard_dir(condition, seed, shard).resolve()
        if validate_shard_artifacts and output_dir != expected_output:
            raise ContractError(f"evaluation output path is not the frozen shard path: {output_dir}")
        if output_dir.parent.parent.parent != frozen_root:
            raise ContractError(f"evaluation output path is not the frozen shard path: {output_dir}")
        log_dir = output_dir / "logs"
        logs = sorted(log_dir.glob("*.json.gz"))
        if len(logs) != GAMES_PER_SHARD:
            raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: expected 250 raw logs, found {len(logs)}")
        shard_rows: list[dict[str, Any]] = []
        for path in logs:
            row = parse_raw_log(
                path,
                condition=condition,
                training_seed=seed,
                expected_seed_start=int(run["hanchan_seed_start"]),
                expected_seed_end=int(run["hanchan_seed_end_exclusive"]),
                stat_cls=stat_cls,
            )
            shard_rows.append(row)
            (current_rows if condition == "CURRENT" else off_rows)[seed].append(row)
        if validate_shard_artifacts:
            if not (output_dir / "metrics.json").is_file() or not (output_dir / "detailed_stats.json").is_file():
                raise ContractError(f"{condition}/{seed}/shard_{shard:02d}: metrics/detailed stats artifact is missing")
            _validate_shard_artifacts(
                run=run,
                output_dir=output_dir,
                rows=shard_rows,
                effective_models=effective_models,
            )
    for seed in TRAINING_SEEDS:
        validate_hanchan_seed_set(current_rows[seed], start=SHARD_STARTS[seed][0])
        validate_hanchan_seed_set(off_rows[seed], start=SHARD_STARTS[seed][0])
        if len(current_rows[seed]) != 1000 or len(off_rows[seed]) != 1000:
            raise ContractError(f"formal C1 seed block is not exactly 1000/1000: {seed}")
    paired = {seed: pair_current_off(current_rows[seed], off_rows[seed]) for seed in TRAINING_SEEDS}
    return paired, {"pairing_gate": True, "plan_models_complete": True, "shard_artifacts_gate": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = args.output_dir.resolve()
        if output_dir != CANONICAL_FORMAL_ADJUDICATION_DIR.resolve():
            raise ContractError(
                "formal C1 summary output directory is not the canonical formal_adjudication directory"
            )
        output = output_dir / FORMAL_SUMMARY_FILENAME
        if output.exists():
            raise ContractError(f"formal C1 result already exists; refusing overwrite: {output}")

        inventory, inventory_sha256 = _validate_execution_inventory()
        plan, preflight, _completion, execution, effective_models, approved_commit = _validate_formal_artifact_chain()
        source_provenance = formal_summary_source_provenance()
        gates = _validate_formal_provenance(plan, preflight, execution, effective_models)
        paired, structural_gates = _read_formal_evaluation(
            plan,
            EVALUATION_ROOT,
            effective_models=effective_models,
            validate_shard_artifacts=True,
        )
        gates.update(structural_gates)
        if not all(gates.values()):
            raise ContractError("formal C1 gate set is incomplete")
        result = summarize_interaction_rows(paired, gates=gates)
        result["provenance"] = build_formal_provenance(
            approved_commit=approved_commit,
            artifact_paths={
                "plan": EVALUATION_PLAN_PATH,
                "preflight": IMPLEMENTATION_PREFLIGHT_PATH,
                "completion": TRAINING_COMPLETION_CLOSURE_PATH,
                "execution": EXECUTION_MANIFEST_PATH,
            },
            artifact_hashes={
                "plan": sha256_file(EVALUATION_PLAN_PATH.resolve()),
                "preflight": sha256_file(IMPLEMENTATION_PREFLIGHT_PATH.resolve()),
                "completion": sha256_file(TRAINING_COMPLETION_CLOSURE_PATH.resolve()),
                "execution": sha256_file(EXECUTION_MANIFEST_PATH.resolve()),
            },
            inventory=inventory,
            inventory_path=EVALUATION_EXECUTION_INVENTORY_PATH,
            inventory_sha256=inventory_sha256,
            plan=plan,
            source_provenance=source_provenance,
            paired_rows_by_seed=paired,
        )
    except (ContractError, OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        print(f"C1-I2 formal summary refused: {exc}", file=sys.stderr)
        return 2

    try:
        publish_atomic(output, result)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        print(f"C1-I2 formal summary refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": str(output), "verdict": result["adjudication"]["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
