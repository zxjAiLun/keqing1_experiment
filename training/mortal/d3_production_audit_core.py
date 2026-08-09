"""Log, context, and summary primitives for the D3 B250 production audit."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from training.mortal.d3_exploration_engine import DISCARD_ACTION_LIMIT
from training.mortal.d3_production_contract import GAMES, REQUIRED_LABELS, SMOKE_SEEDS, expected_seed_keys

@dataclass(frozen=True)
class DecisionSnapshot:
    context: tuple[int, int, int, int, int]
    action: int
    own_riichi: bool
    shanten: int
    phase: str
    legal_action_count: int
    finite_legal_actions: tuple[int, ...]
    top1_action: int | None
    top2_action: int | None
    top1_q: float | None
    top2_q: float | None
    margin: float | None
    eligible: bool

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
        digest.update(
            (
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()

def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "exploration" / "exploration_events.jsonl"
    if not path.is_file():
        raise ValueError(f"missing exploration events: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed exploration event line {line_number}: {exc}") from exc
    return sorted(rows, key=_event_context)

def _event_context(event: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(event["generation_seed"]),
        int(event["seed_key"]),
        int(event["seat"]),
        int(event["kyoku_index"]),
        int(event["decision_index"]),
    )

def _phase(kyoku_index: int) -> str:
    if kyoku_index < 4:
        return "early"
    if kyoku_index < 8:
        return "middle"
    return "late"

def _bucket(values: Iterable[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}

def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "p90", "max", "mean")}
    ordered = sorted(values)

    def q(fraction: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(ordered[low])
        return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))

    return {
        "min": float(ordered[0]),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.9),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }

def primary_row_flags(actions: Iterable[int]) -> list[bool]:
    """Identify native primary rows in GameplayLoader output.

    With ``always_include_kan_select=True``, every primary action 42 is immediately
    followed by exactly one auxiliary tile-selection row. The auxiliary row is
    model input used by the kan guard, but it must not increment D3 decision_index.
    """

    flags: list[bool] = []
    expect_auxiliary = False
    for raw_action in actions:
        action = int(raw_action)
        if expect_auxiliary:
            if not 0 <= action < DISCARD_ACTION_LIMIT:
                raise ValueError(f"invalid auxiliary kan-selection action: {action}")
            flags.append(False)
            expect_auxiliary = False
            continue
        flags.append(True)
        expect_auxiliary = action == 42
    if expect_auxiliary:
        raise ValueError("kan action is missing its adjacent auxiliary selection row")
    return flags

def _load_log_manifest(
    log_dir: Path, expected: set[tuple[int, int]] | None = None
) -> dict[str, Any]:
    if expected is None:
        expected = expected_seed_keys()
    paths = sorted(log_dir.glob("*.json.gz"))
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    malformed: list[str] = []
    canonical_hashes: list[str] = []
    for path in paths:
        try:
            events = _read_log(path)
            key = _log_key(events, path)
            if key in rows:
                malformed.append(f"duplicate log seed: {key}")
            names = events[0].get("names")
            if not isinstance(names, list):
                raise ValueError("start_game names is not a list")
            if sorted(names) != sorted(REQUIRED_LABELS):
                malformed.append(f"wrong lineup in {path}: {names!r}")
            if names.count("K0_70k") != 1:
                malformed.append(f"expected exactly one K0 trainable perspective in {path}")
            kyoku_count = sum(event.get("type") == "start_kyoku" for event in events)
            if kyoku_count <= 0:
                malformed.append(f"log has no start_kyoku events: {path}")
            canonical_sha = _canonical_log_hash(events)
            canonical_hashes.append(canonical_sha)
            rows[key] = {
                "path": str(path),
                "names": names,
                "k0_seat": names.index("K0_70k") if names.count("K0_70k") == 1 else None,
                "kyoku_count": kyoku_count,
                "canonical_sha256": canonical_sha,
            }
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{path}: {exc}")
    actual = set(rows)
    smoke_mixed = sorted(seed for seed, _ in actual if seed in SMOKE_SEEDS)
    return {
        "paths": paths,
        "rows": rows,
        "file_count": len(paths),
        "unique_seed_count": len(actual),
        "expected_seed_set": actual == expected,
        "missing_seeds": sorted(expected - actual),
        "unexpected_seeds": sorted(actual - expected),
        "smoke_seeds_mixed": smoke_mixed,
        "unique_canonical_hanchans": len(set(canonical_hashes)),
        "malformed": malformed,
        "passed": (
            len(paths) == GAMES
            and len(actual) == GAMES
            and actual == expected
            and not smoke_mixed
            and len(set(canonical_hashes)) == GAMES
            and not malformed
        ),
    }
