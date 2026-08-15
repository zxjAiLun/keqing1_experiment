#!/usr/bin/env python3
"""Cross-corpus mechanism audit: M0 vs D1/D2/D3 (read-only, no training).

Memory-frugal rewrite: no per-row storage. Per route we keep only per-hanchan
aggregate counters (strata/action-kind/shanten histograms), streaming Q-target
covariance accumulators, state/decision hash sets, and per-file row counters
(for exposure simulation). This keeps the audit within laptop RAM while the
K0 forward pass streams over every row exactly once.

Repair-1 adjudication contract (frozen):
  * structured strata use the trusted audit_replay_distribution.phase_bucket
    (early/middle/late) as the first dimension -- NOT exact kyoku.
  * action families use the trusted audit_replay_distribution.action_name
    semantics (discard/reach/chi_low/chi_mid/chi_high/pon/kan/agari/
    ryukyoku/pass) -- NOT raw action IDs.
  * within-stratum target variance is a Q-free quantity over ALL rows.

Gates:
  A provenance (frozen indexes; D2 hanchans == D1 6000/6000, V2/V3 3000/3000)
  B loader-view integrity (6000 files / 6000 perspectives / 0 malformed /
    100% legal / augmented false)
  C canonical row audit + K0 Q readout aggregates
  D actual training exposure (RNG-faithful FileDatasetsIter simulation +
     real preview runs vs frozen batch hashes) + exposure-weighted readouts
  E overlap / distance (hanchan, structured strata weighted mass, JSD, TV;
     D1-D2 hanchan overlap == 100%)
  F D3 exploration diagnostic (every frozen event mapped to exactly one
    loader row, per category; hard gate)

Verdict readout: A_coverage_priority / inconclusive / no_verdict_gates_failed.
A uses the frozen relative bootstrap rule (majority of families with both
delta CI lowers above the D1->D2 view baseline + positive M0-exclusive mass).
B_credit_assignment_priority is NEVER machine-promoted: its observational
diagnostics are descriptive only because no quantitative promotion threshold
was frozen before data was seen. Any A-F gate false blocks the verdict.
Priority, never a causal proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

from training.mortal.audit_replay_distribution import (
    action_name,
    decision_hash,
    load_checkpoint,
    load_model,
    model_q,
    phase_bucket,
    records_from_game,
    state_hash,
)
from training.mortal.d3_production_audit_core import (
    _canonical_log_hash,
    _log_key,
    _read_log,
)

TRAIN_PTS = np.asarray([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
SEEDS = (20260806, 20260807, 20260808)
CONSUMED_SAMPLES = 2000 * 512
FULL_ROUTE_NAMES = ("M0", "D1", "D2", "D3")

# Repository layout is stable on both OSes:
#   <project>/keqing1_experiment  (this repo)
#   <project>/keqing1             (training mainline)
#   <project>/../keqing-data       (shared authoritative data)
PROJECT_ROOT = REPO.parent
DATA_ROOT = PROJECT_ROOT.parent / "keqing-data"
KEQING1_REPO = PROJECT_ROOT / "keqing1"
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08"
    / "diagnostics/cross_corpus_mechanism_audit"
)
D1_PREP = (
    KEQING1_REPO
    / "artifacts/experiments/model_pool_2026_07"
    / "D1_project_owned_population_2026_07/training_prep_2026_07"
)
D2_PREP = (
    KEQING1_REPO
    / "artifacts/experiments/model_pool_2026_07"
    / "D2_project_owned_descendant_view_mix_2026_08/training_prep_2026_08"
)
D2_DATASET = D2_PREP.parent / "dataset"
D3_RECIPE = (
    REPO
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
    / "training_recipe_2026_08"
)
D3_INDEX = D3_RECIPE.parent / "training_contract_2026_08/file_index_d3_k0.pth"
D3_INDEX_SHA = "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"
K0_MODEL = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08"
    / "models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
D3_EXP_ROOT = (
    REPO
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
)
PREVIEW_SCRIPT = REPO / "training/mortal/preview_dataloader_batches_2026_07.py"


def native_path(raw: str | Path) -> Path:
    """Resolve frozen Windows paths from repo artifacts on the current OS.

    D1/D2/D3 indexes were frozen on Windows and store absolute paths such as
    ``E:\\AUbuntuProject\\...``.  On Windows those are valid as-is; on POSIX
    they are remapped to the mounted AUbuntuProject root that contains this
    repo, so the audit can run without modifying the frozen index files.
    """
    text = str(raw)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        parts = text.replace("\\", "/").split("/")
        repo_parts = REPO.parts
        if "AUbuntuProject" in parts and "AUbuntuProject" in repo_parts:
            root_idx = repo_parts.index("AUbuntuProject")
            path_idx = parts.index("AUbuntuProject")
            return Path(*repo_parts[: root_idx + 1], *parts[path_idx + 1 :]).resolve()
    return Path(text).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_index(path: Path) -> list[Path]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 6000:
        raise ValueError(f"expected 6000 indexed files in {path}")
    files = [native_path(value) for value in values]
    for file_path in files:
        if not file_path.is_file():
            raise ValueError(f"indexed file does not exist on this OS: {file_path}")
    return files


def _as_hash_set(hashes: Any) -> set[bytes]:
    if isinstance(hashes, bytes):
        return {hashes[i : i + 16] for i in range(0, len(hashes), 16)}
    return set(hashes)


def bucket_score_gap(value: float) -> str:
    return "ahead_big" if value >= 12000 else "ahead" if value >= 0 else "behind" if value > -12000 else "behind_big"


def bucket_legal(value: int) -> str:
    return "1_5" if value <= 5 else "6_10" if value <= 10 else "11_plus"


def bucket_shanten(value: int) -> str:
    return "tenpai" if value <= 0 else str(value) if value <= 2 else "3_plus"


def strata_key(kyoku: int, current_rank: int, score_gap: float, own_riichi: bool, legal: int, shanten: int) -> tuple[str, ...]:
    # First dimension is the trusted phase_bucket (early/middle/late), the
    # frozen structured-state definition from audit_replay_distribution.py.
    return (
        phase_bucket(kyoku),
        str(current_rank),
        bucket_score_gap(score_gap),
        str(bool(own_riichi)),
        bucket_legal(legal),
        bucket_shanten(shanten),
    )


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    eps = 1e-12
    return float(np.sqrt(0.5 * (np.sum(p * np.log((p + eps) / (m + eps))) + np.sum(q * np.log((q + eps) / (m + eps))))))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def counter_array(counter: Counter) -> np.ndarray:
    total = sum(counter.values())
    order = sorted(counter)
    return np.asarray([counter[k] / total for k in order], dtype=np.float64) if total else np.zeros(1)


def weighted_missing_mass(source: Counter, target: Counter) -> float:
    total = sum(source.values())
    return sum(count for key, count in source.items() if target.get(key, 0) == 0) / total if total else 0.0


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum())) - (n + 1) / n)


def covariance_accumulators() -> dict[str, float]:
    return {"n": 0.0, "sum_q": 0.0, "sum_t": 0.0, "sum_qt": 0.0, "sum_q2": 0.0, "sum_t2": 0.0}


def corr_from_accumulators(acc: dict[str, float]) -> float:
    n = acc["n"]
    if n < 2:
        return 0.0
    cov = acc["sum_qt"] - acc["sum_q"] * acc["sum_t"] / n
    var_q = acc["sum_q2"] - acc["sum_q"] ** 2 / n
    var_t = acc["sum_t2"] - acc["sum_t"] ** 2 / n
    denom = math.sqrt(max(var_q, 0.0) * max(var_t, 0.0))
    return float(cov / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------- adjudication
VERDICT_NO_GATES = "no_verdict_gates_failed"
VERDICT_A = "A_coverage_priority"
VERDICT_INCONCLUSIVE = "inconclusive"
B_MACHINE_GATE = "not_preregistered_quantitatively"
ROUTE_AGG_CACHE_SCHEMA = "keqing.mortal.route_agg_cache.v2"


def decide_readout(*, coverage_votes: int, num_families: int, m0_exclusive_mass: dict[str, float], gates_ok: bool) -> str:
    """Frozen adjudication contract (repair 1).

    A_coverage_priority: majority of bootstrap families have BOTH the M0->D1
    and M0->D3 delta CI lowers strictly above the D1->D2 view baseline AND at
    least one descendant route carries positive M0-exclusive support mass.
    This is a relative/bootstrap rule -- no absolute thresholds.

    B_credit_assignment_priority is NOT machine-promotable: the observational
    diagnostics may inform a future experiment, but no quantitative promotion
    threshold was frozen before data was seen, so the machine never outputs B.

    Any A-F gate false blocks the verdict entirely.
    """
    if not gates_ok:
        return VERDICT_NO_GATES
    if coverage_votes / num_families > 0.5 and any(value > 0 for value in m0_exclusive_mass.values()):
        return VERDICT_A
    return VERDICT_INCONCLUSIVE


def gate_f_checks(summary: dict[str, Any]) -> dict[str, bool]:
    """Hard checks for gate F: every frozen D3 exploration event must map to
    exactly one loader row (per category, nothing unconsumed, and the mapped
    count must be an EVENT count, never histogram-cell increments)."""
    totals = summary.get("category_totals", {})
    mapped = summary.get("mapped_event_totals", {})
    total_events = int(summary.get("total_events", 0))
    total_mapped = int(summary.get("total_mapped_events", 0))
    total_unconsumed = int(summary.get("total_unconsumed_events", -1))
    return {
        "diagnostic_present": total_events > 0,
        "all_events_mapped_exactly_once": total_events > 0 and total_mapped == total_events,
        "no_unconsumed_events": total_events > 0 and total_unconsumed == 0,
        "category_counts_exact": bool(totals) and all(int(mapped.get(key, 0)) == int(totals[key]) for key in totals),
    }


# ------------------------------------------------------------ exposure readout
def _jsd_counters(first: Counter, second: Counter) -> float:
    keys = sorted(set(first) | set(second))
    pa = np.asarray([first.get(k, 0) for k in keys], dtype=np.float64)
    pb = np.asarray([second.get(k, 0) for k in keys], dtype=np.float64)
    if pa.sum() == 0 or pb.sum() == 0:
        return 0.0
    return jsd(pa / pa.sum(), pb / pb.sum())


def _jsd_matrix(routes: dict[str, Counter]) -> dict[str, dict[str, float]]:
    return {a: {b: _jsd_counters(routes[a], routes[b]) for b in routes} for a in routes}


def _json_distribution(counter: Counter) -> dict[str, float]:
    return {key if isinstance(key, str) else "|".join(str(part) for part in key): float(value) for key, value in counter.items()}


def exposure_weighted_distribution(weights: np.ndarray, per_file: list[Counter]) -> Counter:
    total: Counter = Counter()
    for weight, counter in zip(weights, per_file):
        if weight > 0:
            for key, value in counter.items():
                total[key] += weight * value
    return total


def exposure_weighted_readout(
    route_scan: dict[str, dict[str, Any]],
    exposure: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
    names: tuple[str, ...] = ("M0", "D1", "D2", "D3"),
) -> dict[str, Any]:
    """Exposure-weighted readouts: the 1,024,000 samples the training loop
    actually consumes, weighted by the RNG-faithful simulation, per route and
    seed, with route x seed JSD comparisons (canonical vs consumed)."""
    canonical: dict[str, dict[str, Counter]] = {name: {} for name in names}
    weighted: dict[str, dict[str, dict[str, Counter]]] = {str(seed): {name: {} for name in names} for seed in seeds}
    for name in names:
        scan = route_scan[name]
        agg_by_index = {int(entry["hanchan"]): entry for entry in scan["hanchan_agg"]}
        n_files = len(scan["hanchan_agg"])
        per_file_target = [scan["per_file_target_counts"].get(index, Counter()) for index in range(n_files)]
        for family in ("target", "action_kind", "strata"):
            if family == "target":
                per_file = per_file_target
            elif family == "action_kind":
                per_file = [agg_by_index[index]["actions"] for index in range(n_files)]
            else:
                per_file = [agg_by_index[index]["strata"] for index in range(n_files)]
            canonical[name][family] = exposure_weighted_distribution(np.ones(n_files), per_file)
            for seed in seeds:
                weights = np.asarray(exposure[name][str(seed)]["simulation"]["consumed_per_file"], dtype=np.float64)
                if weights.size != n_files:
                    raise ValueError(f"{name} seed {seed}: exposure weights {weights.size} != files {n_files}")
                weighted[str(seed)][name][family] = exposure_weighted_distribution(weights, per_file)
    out: dict[str, Any] = {"seeds": [str(seed) for seed in seeds]}
    for family in ("target", "action_kind", "strata"):
        per_seed: dict[str, Any] = {}
        between_seed_jsd: dict[str, float] = {}
        for name in names:
            dists = [weighted[str(seed)][name][family] for seed in seeds]
            pairs = [_jsd_counters(dists[i], dists[j]) for i in range(len(dists)) for j in range(i + 1, len(dists))]
            between_seed_jsd[name] = float(np.mean(pairs)) if pairs else 0.0
        for seed in seeds:
            routes_dist = {name: weighted[str(seed)][name][family] for name in names}
            per_seed[str(seed)] = {
                "jsd_matrix": _jsd_matrix(routes_dist),
                "distributions": {name: _json_distribution(routes_dist[name]) for name in names},
            }
        out[family] = {
            "canonical_jsd_matrix": _jsd_matrix({name: canonical[name][family] for name in names}),
            "per_seed": per_seed,
            "between_seed_mean_jsd": between_seed_jsd,
        }
    return out


# ---------------------------------------------------------------- route scan
def scan_route(
    name: str,
    files: list[Path],
    labels: list[str],
    by_file: dict[str, str] | None,
    version: int,
    brain,
    dqn,
    q_batch_size: int,
    device: torch.device,
    d3_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader

    hanchan_agg: list[dict[str, Any]] = []
    per_file_counters: dict[int, Counter] = {}
    per_file_target_counts: dict[int, Counter] = {}
    state_hashes: set[bytes] = set()
    decision_hashes: set[bytes] = set()
    final_rank_counts: Counter[int] = Counter()
    target_counts: Counter[float] = Counter()
    behavior_legal = 0
    total_rows = 0
    rows_per_hanchan: list[int] = []
    global_tacc = covariance_accumulators()
    global_target_acc = covariance_accumulators()
    strata_tacc: dict[tuple[str, ...], dict[str, float]] = {}
    action_hist: Counter[str] = Counter()
    shanten_hist: Counter[str] = Counter()
    phase_hist: Counter[str] = Counter()
    rank_hist: Counter[str] = Counter()
    score_gap_hist: Counter[str] = Counter()
    legal_hist: Counter[str] = Counter()
    behavior_q_hist: Counter[str] = Counter()
    q_regret_hist: Counter[str] = Counter()
    margin_hist: Counter[str] = Counter()
    greedy_agreement = 0
    malformed: list[str] = []
    perspective_count = 0
    d3_diag: dict[str, Any] | None = None
    if name == "D3" and d3_events_by_key is not None:
        d3_diag = {
            "category_hanchan": defaultdict(lambda: defaultdict(Counter)),
            "category_total": Counter(),
            "category_mapped_events": Counter(),
            "category_unconsumed": Counter(),
            "category_hanchan_events": defaultdict(Counter),
        }

    q_pending: list[tuple[Any, dict[str, Any]]] = []

    def flush_q() -> None:
        nonlocal greedy_agreement
        if not q_pending:
            return
        q = model_q(brain, dqn, [item[0] for item in q_pending], device)
        for index, (_, row) in enumerate(q_pending):
            q_row = np.asarray(q[index], dtype=np.float64)
            mask = row["mask"]
            legal = np.flatnonzero(mask & np.isfinite(q_row))
            action = row["action"]
            target = row["target"]
            if legal.size == 0:
                continue
            greedy = int(legal[np.argmax(q_row[legal])])
            behavior_q = float(q_row[action]) if action in legal else None
            if behavior_q is None:
                continue
            greedy_q = float(q_row[greedy])
            top2_min = float(np.partition(q_row[legal], -2)[-2:].min()) if legal.size >= 2 else greedy_q
            behavior_q_hist[f"{behavior_q:+.3f}"] += 1
            q_regret_hist[f"{greedy_q - behavior_q:+.3f}"] += 1
            margin_hist[f"{(q_row[legal].max() - top2_min):+.3f}"] += 1
            if greedy == action:
                greedy_agreement += 1
            acc = global_tacc
            acc["n"] += 1
            acc["sum_q"] += behavior_q
            acc["sum_t"] += target
            acc["sum_qt"] += behavior_q * target
            acc["sum_q2"] += behavior_q * behavior_q
            acc["sum_t2"] += target * target
            category = row.get("event_category")
            if category is not None:
                hanchan = row["hanchan"]
                for bucket_name, value in (
                    ("target", f"{target:+.0f}"),
                    ("final_rank", str(row["final_rank"])),
                    ("behavior_q", f"{behavior_q:+.3f}"),
                    ("q_regret", f"{greedy_q - behavior_q:+.3f}"),
                    ("margin", f"{(q_row[legal].max() - top2_min):+.3f}"),
                    ("phase", phase_bucket(row["kyoku"])),
                    ("rank", str(row["current_rank"])),
                    ("score_gap", bucket_score_gap(row["score_gap"])),
                    ("shanten", bucket_shanten(row["shanten"])),
                ):
                    d3_diag["category_hanchan"][category][hanchan][bucket_name + ":" + value] += 1
        q_pending.clear()

    for file_index, path in enumerate(files):
        if file_index % 50 == 0:
            print(f"[trace] {name} file {file_index} {path.name}", flush=True)
        label = by_file[str(path.resolve())] if by_file is not None else labels[0]
        loader = GameplayLoader(version=version, oracle=False, player_names=[label], excludes=None, augmented=False)
        try:
            loaded = loader.load_gz_log_files([str(path)])
            if len(loaded) != 1 or len(loaded[0]) != 1:
                raise ValueError(f"expected one perspective for {path.name}")
            records = list(records_from_game(loaded[0][0], TRAIN_PTS))
            if not records:
                raise ValueError(f"zero rows for {path.name}")
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{path.name}: {exc}")
            continue
        perspective_count += 1
        rows_per_hanchan.append(len(records))
        per_file_counters[file_index] = Counter()
        per_file_target_counts[file_index] = Counter()
        hanchan_strata: Counter = Counter()
        hanchan_actions: Counter = Counter()
        hanchan_shanten: Counter = Counter()
        # D3 gate-F wiring: native-scene event -> loader row mapping
        event_row_map: dict[tuple[int, int], dict[str, Any]] = {}
        file_events: list[dict[str, Any]] = []
        if d3_events_by_key is not None:
            from training.mortal.d3_native_scene import (
                reconstruct_native_scenes,
            )
            from training.mortal.d3_production_audit_core import (
                primary_row_flags,
            )

            game_seed_key = _log_key(_read_log(path), path)
            file_events = d3_events_by_key.get(game_seed_key, [])
            if file_events:
                flags = primary_row_flags(r.action for r in records)
                loader_rows = [
                    {"action": int(r.action), "legal_count": int(np.asarray(r.mask, dtype=np.bool_).sum()), "kyoku": int(r.kyoku)}
                    for r, isp in zip(records, flags, strict=True)
                    if isp
                ]
                seat = int(loaded[0][0].take_player_id())
                recon = reconstruct_native_scenes(path, seat, loader_rows)
                arena_to_row = {}
                for entry in recon["scenes"]:
                    if entry["arena_index"] is not None and entry["loader_row_index"] is not None and entry["arena_consulted"]:
                        arena_to_row[(entry["kyoku"], entry["arena_index"])] = entry["loader_row_index"]
                for event in file_events:
                    context = (int(event["kyoku_index"]), int(event["decision_index"]))
                    loader_idx = arena_to_row.get(context)
                    if loader_idx is not None:
                        event_row_map[(int(event["kyoku_index"]), loader_idx)] = event
        loader_index_by_kyoku: dict[int, int] = {}
        for record in records:
            action = int(record.action)
            mask = np.asarray(record.mask, dtype=np.bool_)
            legal_count = int(mask.sum())
            state_hashes.add(state_hash(record))
            decision_hashes.add(decision_hash(record))
            if action in np.flatnonzero(mask):
                behavior_legal += 1
            total_rows += 1
            final_rank_counts[int(record.target_rank) - 1] += 1
            target_counts[float(record.target)] += 1
            target = float(record.target)
            kyoku = int(record.kyoku)
            rank = int(record.current_rank)
            score_gap = float(record.score_gap)
            own_riichi = bool(record.own_riichi)
            shanten = int(record.shanten)
            key = strata_key(kyoku, rank, score_gap, own_riichi, legal_count, shanten)
            hanchan_strata[key] += 1
            hanchan_actions[action_name(action)] += 1
            hanchan_shanten[bucket_shanten(shanten)] += 1
            action_hist[action_name(action)] += 1
            shanten_hist[bucket_shanten(shanten)] += 1
            phase_hist[phase_bucket(kyoku)] += 1
            rank_hist[str(rank)] += 1
            score_gap_hist[bucket_score_gap(score_gap)] += 1
            legal_hist[bucket_legal(legal_count)] += 1
            per_file_counters[file_index]["rows"] += 1
            per_file_target_counts[file_index][str(target)] += 1
            gacc = global_target_acc
            gacc["n"] += 1
            gacc["sum_t"] += target
            gacc["sum_t2"] += target * target
            sacc = strata_tacc.setdefault(key, covariance_accumulators())
            sacc["n"] += 1
            sacc["sum_t"] += target
            sacc["sum_t2"] += target * target
            event_category = None
            if event_row_map:
                loader_index = loader_index_by_kyoku.get(kyoku, 0)
                loader_index_by_kyoku[kyoku] = loader_index + 1
                event = event_row_map.pop((kyoku, loader_index), None)
                if event is not None:
                    event_category = (
                        "explored"
                        if event.get("explored")
                        else "hash_rejected"
                        if event.get("reason") == "hash_rejected"
                        else "budget_exhausted"
                    )
                    d3_diag["category_mapped_events"][event_category] += 1
                    d3_diag["category_hanchan_events"][event_category][file_index] += 1
            row = {
                "action": action,
                "target": target,
                "mask": mask,
                "strata": key,
                "hanchan": file_index,
                "kyoku": kyoku,
                "current_rank": rank,
                "score_gap": score_gap,
                "shanten": shanten,
                "final_rank": int(record.target_rank) - 1,
            }
            if event_category is not None:
                row["event_category"] = event_category
            q_pending.append((record, row))
            if len(q_pending) >= q_batch_size:
                flush_q()
        for event in file_events:
            category = "explored" if event.get("explored") else "hash_rejected" if event.get("reason") == "hash_rejected" else "budget_exhausted"
            d3_diag["category_total"][category] += 1
        if event_row_map:
            for event in event_row_map.values():
                category = "explored" if event.get("explored") else "hash_rejected" if event.get("reason") == "hash_rejected" else "budget_exhausted"
                d3_diag["category_unconsumed"][category] += 1
        hanchan_agg.append(
            {
                "hanchan": file_index,
                "strata": hanchan_strata,
                "actions": hanchan_actions,
                "shanten": hanchan_shanten,
                "rows": len(records),
            }
        )
        if (file_index + 1) % 1000 == 0:
            print(f"[audit] {name} {file_index + 1}/6000", flush=True)
    flush_q()
    if malformed:
        raise ValueError(f"{name}: malformed: {malformed[:10]}")
    if perspective_count != len(files) or behavior_legal != total_rows:
        raise ValueError(f"{name}: integrity {perspective_count}/{len(files)} legal {behavior_legal}/{total_rows}")
    within_sum = sum(
        acc["sum_t2"] - acc["sum_t"] ** 2 / acc["n"] for acc in strata_tacc.values() if acc["n"] >= 2
    )
    total_n = global_target_acc["n"]
    total_var = global_target_acc["sum_t2"] - global_target_acc["sum_t"] ** 2 / total_n if total_n > 1 else 0.0
    return {
        "route": name,
        "meta": {
            "perspectives": perspective_count,
            "total_rows": total_rows,
            "rows_per_hanchan": {"min": min(rows_per_hanchan), "median": float(np.median(rows_per_hanchan)), "max": max(rows_per_hanchan)},
            "final_rank_counts": {str(k): v for k, v in sorted(final_rank_counts.items())},
            "target_counts": {str(k): v for k, v in sorted(target_counts.items())},
            "unique_state_hashes": len(state_hashes),
            "unique_decision_hashes": len(decision_hashes),
            "greedy_agreement_rate": greedy_agreement / total_rows if total_rows else 0.0,
            "behavior_q_target_corr": corr_from_accumulators(global_tacc),
            "within_stratum_target_var_ratio": within_sum / total_var if total_var > 0 else 1.0,
        },
        "hanchan_agg": hanchan_agg,
        "per_file_counters": per_file_counters,
        "per_file_target_counts": per_file_target_counts,
        "distributions": {
            "action_kind": dict(action_hist),
            "shanten": dict(shanten_hist),
            "phase": dict(phase_hist),
            "rank": dict(rank_hist),
            "score_gap": dict(score_gap_hist),
            "legal": dict(legal_hist),
            "behavior_q": dict(behavior_q_hist),
            "q_regret": dict(q_regret_hist),
            "greedy_margin": dict(margin_hist),
        },
        "state_hashes": state_hashes,
        "decision_hashes": decision_hashes,
        "d3_diag": d3_diag,
    }


# ---------------------------------------------------------------- fast scan
def _rank_and_gap_vectorized(scores: np.ndarray, player_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (current rank 1..4, score gap) from (N,4) raw scores.

    Matches records_from_game.current_rank_and_gap: stable descending sort,
    seat order for ties.
    """
    own = scores[:, player_id]
    rank = np.ones(scores.shape[0], dtype=np.int64)
    for other in range(4):
        if other == player_id:
            continue
        other_score = scores[:, other]
        rank += (other_score > own).astype(np.int64)
        rank += ((other_score == own) & (np.arange(4)[other] < player_id)).astype(np.int64)
    opp_max = np.max(np.stack([scores[:, other] for other in range(4) if other != player_id], axis=1), axis=1)
    gap = own - opp_max
    return rank, gap


def scan_route_fast(
    name: str,
    files: list[Path],
    labels: list[str],
    by_file: dict[str, str] | None,
    version: int,
    brain,
    dqn,
    q_batch_size: int,
    device: torch.device,
    max_files: int | None = None,
    d3_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader

    files = files if max_files is None else files[:max_files]
    hanchan_agg: list[dict[str, Any]] = []
    per_file_counters: dict[int, Counter] = {}
    per_file_target_counts: dict[int, Counter] = {}
    state_hash_buf: bytearray = bytearray()
    decision_hash_buf: bytearray = bytearray()
    final_rank_counts: Counter[int] = Counter()
    target_counts: Counter[float] = Counter()
    behavior_legal = 0
    total_rows = 0
    rows_per_hanchan: list[int] = []
    global_tacc = covariance_accumulators()
    global_target_acc = covariance_accumulators()
    strata_tacc: dict[tuple[str, ...], dict[str, float]] = {}
    action_hist: Counter[str] = Counter()
    shanten_hist: Counter[str] = Counter()
    phase_hist: Counter[str] = Counter()
    rank_hist: Counter[str] = Counter()
    score_gap_hist: Counter[str] = Counter()
    legal_hist: Counter[str] = Counter()
    behavior_q_hist: Counter[str] = Counter()
    q_regret_hist: Counter[str] = Counter()
    margin_hist: Counter[str] = Counter()
    greedy_agreement = 0
    malformed: list[str] = []
    perspective_count = 0
    d3_diag: dict[str, Any] | None = None
    if name == "D3" and d3_events_by_key is not None:
        d3_diag = {
            "category_hanchan": defaultdict(lambda: defaultdict(Counter)),
            "category_total": Counter(),
            "category_mapped_events": Counter(),
            "category_unconsumed": Counter(),
            "category_hanchan_events": defaultdict(Counter),
        }

    pending_obs: list[np.ndarray] = []
    pending_masks: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []

    def flush_q() -> None:
        nonlocal greedy_agreement
        if not pending_obs:
            return
        obs_tensor = torch.as_tensor(np.ascontiguousarray(np.stack(pending_obs)), device=device)
        mask_tensor = torch.as_tensor(np.ascontiguousarray(np.stack(pending_masks)), device=device)
        with torch.inference_mode():
            q = dqn(brain(obs_tensor), mask_tensor)
        q_np = q.float().cpu().numpy()
        for index, meta in enumerate(pending_meta):
            q_row = q_np[index]
            mask = meta["mask"]
            legal_idx = np.flatnonzero(mask & np.isfinite(q_row))
            if legal_idx.size == 0:
                continue
            action = meta["action"]
            greedy = int(legal_idx[np.argmax(q_row[legal_idx])])
            behavior_q = float(q_row[action]) if action in legal_idx else None
            if behavior_q is None:
                continue
            greedy_q = float(q_row[greedy])
            top2_min = float(np.partition(q_row[legal_idx], -2)[-2:].min()) if legal_idx.size >= 2 else greedy_q
            margin = float(q_row[legal_idx].max() - top2_min) if legal_idx.size >= 2 else 0.0
            behavior_q_hist[f"{behavior_q:+.3f}"] += 1
            q_regret_hist[f"{greedy_q - behavior_q:+.3f}"] += 1
            margin_hist[f"{margin:+.3f}"] += 1
            if greedy == action:
                greedy_agreement += 1
            acc = global_tacc
            acc["n"] += 1
            acc["sum_q"] += behavior_q
            acc["sum_t"] += meta["target"]
            acc["sum_qt"] += behavior_q * meta["target"]
            acc["sum_q2"] += behavior_q * behavior_q
            acc["sum_t2"] += meta["target"] * meta["target"]
            category = meta.get("event_category")
            if category is not None:
                hanchan = meta["hanchan"]
                for bucket_name, value in (
                    ("target", f"{meta['target']:+.0f}"),
                    ("final_rank", str(meta["final_rank"])),
                    ("behavior_q", f"{behavior_q:+.3f}"),
                    ("q_regret", f"{greedy_q - behavior_q:+.3f}"),
                    ("margin", f"{margin:+.3f}"),
                    ("phase", phase_bucket(meta["kyoku"])),
                    ("rank", str(meta["current_rank"])),
                    ("score_gap", bucket_score_gap(meta["score_gap"])),
                    ("shanten", bucket_shanten(meta["shanten"])),
                ):
                    d3_diag["category_hanchan"][category][hanchan][bucket_name + ":" + value] += 1
        pending_obs.clear()
        pending_masks.clear()
        pending_meta.clear()

    by_label: dict[str, list[tuple[int, Path]]] = {}
    for file_index, path in enumerate(files):
        label = by_file[str(path.resolve())] if by_file is not None else labels[0]
        by_label.setdefault(label, []).append((file_index, path))

    LOAD_CHUNK = 8
    for label, tuples in by_label.items():
        loader = GameplayLoader(version=version, oracle=False, player_names=[label], excludes=None, augmented=False)
        for start in range(0, len(tuples), LOAD_CHUNK):
            chunk = tuples[start : start + LOAD_CHUNK]
            loaded = loader.load_gz_log_files([str(p) for _, p in chunk])
            if len(loaded) != len(chunk):
                malformed.append(f"{label}: batch size mismatch {len(loaded)} != {len(chunk)}")
                continue
            for (file_index, path), perspectives in zip(chunk, loaded, strict=True):
                if len(perspectives) != 1:
                    malformed.append(f"{path.name}: expected one perspective, got {len(perspectives)}")
                    continue
                try:
                    game = perspectives[0]
                    obs = game.take_obs()
                    masks = game.take_masks()
                    actions = game.take_actions()
                    at_kyoku = game.take_at_kyoku()
                    shantens = game.take_shantens()
                    grp = game.take_grp()
                    features = np.asarray(grp.take_feature(), dtype=np.float64)
                    player_id = int(game.take_player_id())
                    final_rank = int(grp.take_rank_by_player()[player_id])
                except Exception as exc:  # noqa: BLE001
                    malformed.append(f"{path.name}: {exc}")
                    continue
                n_rows = len(obs)
                if n_rows == 0:
                    malformed.append(f"{path.name}: zero rows")
                    continue
                if not (len(masks) == len(actions) == len(at_kyoku) == len(shantens) == n_rows):
                    malformed.append(f"{path.name}: inconsistent array lengths")
                    continue
                # pyo3 returns Vec<u8>/Vec<i8> as bytes; Vec<i64>/obs/masks as lists/arrays
                def _to_int64(raw, signed: bool = False) -> np.ndarray:
                    if isinstance(raw, (bytes, bytearray)):
                        return np.frombuffer(raw, dtype=np.int8 if signed else np.uint8).astype(np.int64)
                    return np.asarray(list(raw), dtype=np.int64)

                at_kyoku_arr = _to_int64(at_kyoku)
                shantens_arr = _to_int64(shantens, signed=True)
                actions_arr = np.asarray(actions, dtype=np.int64) if not isinstance(actions, (bytes, bytearray)) else _to_int64(actions, signed=True)
                target = float(TRAIN_PTS[final_rank] - TRAIN_PTS.mean())
                perspective_count += 1
                rows_per_hanchan.append(n_rows)
                per_file_counters[file_index] = Counter()
                per_file_target_counts[file_index] = Counter()
                final_rank_counts[final_rank] += n_rows
                target_counts[target] += n_rows
                kyoku_indices = np.clip(at_kyoku_arr, 0, features.shape[0] - 1)
                scores = features[kyoku_indices, 3:7] * 10000.0
                ranks, gaps = _rank_and_gap_vectorized(scores, player_id)
                # own_riichi per-row (sequential over actions)
                own_riichi = np.zeros(n_rows, dtype=bool)
                riichi_kyoku = None
                for i in range(n_rows):
                    action = int(actions_arr[i])
                    kyoku = int(at_kyoku_arr[i])
                    if riichi_kyoku == kyoku:
                        own_riichi[i] = True
                    if action == 37:
                        riichi_kyoku = kyoku
                masks_arr = np.asarray(masks, dtype=np.bool_)
                legal_counts = masks_arr.sum(axis=1).astype(np.int64)
                obs_rows = [np.ascontiguousarray(np.asarray(obs[i], dtype=np.float32)) for i in range(n_rows)]
                # D3 gate-F event mapping
                event_row_map: dict[tuple[int, int], dict[str, Any]] = {}
                file_events: list[dict[str, Any]] = []
                if d3_events_by_key is not None:
                    from training.mortal.d3_native_scene import (
                        reconstruct_native_scenes,
                    )
                    from training.mortal.d3_production_audit_core import (
                        primary_row_flags,
                    )

                    game_seed_key = _log_key(_read_log(path), path)
                    file_events = d3_events_by_key.get(game_seed_key, [])
                    if file_events:
                        flags = primary_row_flags(int(r) for r in actions)
                        loader_rows = [
                            {"action": int(actions_arr[i]), "legal_count": int(legal_counts[i]), "kyoku": int(at_kyoku_arr[i])}
                            for i, isp in enumerate(flags)
                            if isp
                        ]
                        recon = reconstruct_native_scenes(path, player_id, loader_rows)
                        arena_to_row = {}
                        for entry in recon["scenes"]:
                            if entry["arena_index"] is not None and entry["loader_row_index"] is not None and entry["arena_consulted"]:
                                arena_to_row[(entry["kyoku"], entry["arena_index"])] = entry["loader_row_index"]
                        for event in file_events:
                            context = (int(event["kyoku_index"]), int(event["decision_index"]))
                            loader_idx = arena_to_row.get(context)
                            if loader_idx is not None:
                                event_row_map[(int(event["kyoku_index"]), loader_idx)] = event
                hanchan_strata: Counter = Counter()
                hanchan_actions: Counter = Counter()
                hanchan_shanten: Counter = Counter()
                loader_index_by_kyoku: dict[int, int] = {}
                for i in range(n_rows):
                    action = int(actions_arr[i])
                    kyoku = int(at_kyoku_arr[i])
                    legal_count = int(legal_counts[i])
                    shanten = int(shantens_arr[i])
                    rank = int(ranks[i])
                    score_gap = float(gaps[i])
                    if action in np.flatnonzero(masks_arr[i]):
                        behavior_legal += 1
                    total_rows += 1
                    key = strata_key(kyoku, rank, score_gap, own_riichi[i], legal_count, shanten)
                    hanchan_strata[key] += 1
                    hanchan_actions[action_name(action)] += 1
                    hanchan_shanten[bucket_shanten(shanten)] += 1
                    action_hist[action_name(action)] += 1
                    shanten_hist[bucket_shanten(shanten)] += 1
                    phase_hist[phase_bucket(kyoku)] += 1
                    rank_hist[str(rank)] += 1
                    score_gap_hist[bucket_score_gap(score_gap)] += 1
                    legal_hist[bucket_legal(legal_count)] += 1
                    per_file_counters[file_index]["rows"] += 1
                    per_file_target_counts[file_index][f"{target:+.0f}"] += 1
                    gacc = global_target_acc
                    gacc["n"] += 1
                    gacc["sum_t"] += target
                    gacc["sum_t2"] += target * target
                    sacc = strata_tacc.setdefault(key, covariance_accumulators())
                    sacc["n"] += 1
                    sacc["sum_t"] += target
                    sacc["sum_t2"] += target * target
                    state_hash_buf += hashlib.blake2b(np.ascontiguousarray(obs_rows[i]).tobytes() + np.ascontiguousarray(masks_arr[i]).tobytes(), digest_size=16).digest()
                    decision_hash_buf += hashlib.blake2b(
                        np.ascontiguousarray(obs_rows[i]).tobytes() + np.ascontiguousarray(masks_arr[i]).tobytes() + action.to_bytes(2, "little", signed=False),
                        digest_size=16,
                    ).digest()
                    event_category = None
                    if event_row_map:
                        loader_index = loader_index_by_kyoku.get(kyoku, 0)
                        loader_index_by_kyoku[kyoku] = loader_index + 1
                        event = event_row_map.pop((kyoku, loader_index), None)
                        if event is not None:
                            event_category = (
                                "explored"
                                if event.get("explored")
                                else "hash_rejected"
                                if event.get("reason") == "hash_rejected"
                                else "budget_exhausted"
                            )
                            d3_diag["category_mapped_events"][event_category] += 1
                            d3_diag["category_hanchan_events"][event_category][file_index] += 1
                    meta = {
                        "action": action,
                        "target": target,
                        "mask": masks_arr[i],
                        "strata": key,
                        "hanchan": file_index,
                        "kyoku": kyoku,
                        "current_rank": rank,
                        "score_gap": score_gap,
                        "shanten": shanten,
                        "final_rank": final_rank,
                    }
                    if event_category is not None:
                        meta["event_category"] = event_category
                    pending_obs.append(obs_rows[i])
                    pending_masks.append(masks_arr[i])
                    pending_meta.append(meta)
                    if len(pending_obs) >= q_batch_size:
                        flush_q()
                for event in file_events:
                    category = "explored" if event.get("explored") else "hash_rejected" if event.get("reason") == "hash_rejected" else "budget_exhausted"
                    d3_diag["category_total"][category] += 1
                if event_row_map:
                    for event in event_row_map.values():
                        category = "explored" if event.get("explored") else "hash_rejected" if event.get("reason") == "hash_rejected" else "budget_exhausted"
                        d3_diag["category_unconsumed"][category] += 1
                hanchan_agg.append(
                    {
                        "hanchan": file_index,
                        "strata": hanchan_strata,
                        "actions": hanchan_actions,
                        "shanten": hanchan_shanten,
                        "rows": n_rows,
                    }
                )
                if (file_index + 1) % 1000 == 0:
                    try:
                        import psutil

                        rss_mb = int(psutil.Process().memory_info().rss / 1_000_000)
                    except Exception:  # noqa: BLE001
                        rss_mb = -1
                    print(f"[audit] {name} {file_index + 1}/{len(files)} rows={total_rows} rss={rss_mb}MB", flush=True)
    flush_q()
    if malformed:
        raise ValueError(f"{name}: malformed: {malformed[:10]}")
    if perspective_count != len(files) or behavior_legal != total_rows:
        raise ValueError(f"{name}: integrity {perspective_count}/{len(files)} legal {behavior_legal}/{total_rows}")
    within_sum = sum(
        acc["sum_t2"] - acc["sum_t"] ** 2 / acc["n"] for acc in strata_tacc.values() if acc["n"] >= 2
    )
    total_n = global_target_acc["n"]
    total_var = global_target_acc["sum_t2"] - global_target_acc["sum_t"] ** 2 / total_n if total_n > 1 else 0.0

    def _unique_count(buf: bytearray) -> int:
        if not buf:
            return 0
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(-1, 16)
        return int(np.unique(arr, axis=0).shape[0])

    return {
        "route": name,
        "meta": {
            "perspectives": perspective_count,
            "total_rows": total_rows,
            "rows_per_hanchan": {"min": min(rows_per_hanchan) if rows_per_hanchan else 0, "median": float(np.median(rows_per_hanchan)) if rows_per_hanchan else 0.0, "max": max(rows_per_hanchan) if rows_per_hanchan else 0},
            "final_rank_counts": {str(k): v for k, v in sorted(final_rank_counts.items())},
            "target_counts": {str(k): v for k, v in sorted(target_counts.items())},
            "unique_state_hashes": _unique_count(state_hash_buf),
            "unique_decision_hashes": _unique_count(decision_hash_buf),
            "greedy_agreement_rate": greedy_agreement / total_rows if total_rows else 0.0,
            "behavior_q_target_corr": corr_from_accumulators(global_tacc),
            "within_stratum_target_var_ratio": within_sum / total_var if total_var > 0 else 1.0,
        },
        "hanchan_agg": hanchan_agg,
        "per_file_counters": per_file_counters,
        "per_file_target_counts": per_file_target_counts,
        "distributions": {
            "action_kind": dict(action_hist),
            "shanten": dict(shanten_hist),
            "phase": dict(phase_hist),
            "rank": dict(rank_hist),
            "score_gap": dict(score_gap_hist),
            "legal": dict(legal_hist),
            "behavior_q": dict(behavior_q_hist),
            "q_regret": dict(q_regret_hist),
            "greedy_margin": dict(margin_hist),
        },
        "state_hashes": bytes(state_hash_buf),
        "decision_hashes": bytes(decision_hash_buf),
        "d3_diag": d3_diag,
    }


# ---------------------------------------------------------------- exposure
def simulate_exposure(
    files: list[Path],
    seed: int,
    file_batch_size: int,
    num_epochs: int,
    per_file_counters: dict[int, Counter],
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    file_list = list(files)
    path_to_index = {str(path.resolve()): i for i, path in enumerate(files)}
    rows_by_index = np.asarray(
        [int(per_file_counters.get(i, Counter()).get("rows", 0)) for i in range(len(files))],
        dtype=np.int64,
    )
    consumed_per_file: Counter[int] = Counter()
    consumed = 0
    for _ in range(num_epochs):
        random.shuffle(file_list)
        for start in range(0, len(file_list), file_batch_size):
            entries: list[int] = []
            for path in file_list[start : start + file_batch_size]:
                file_index = path_to_index[str(path.resolve())]
                entries.extend([file_index] * int(rows_by_index[file_index]))
            random.shuffle(entries)
            for file_index in entries:
                if consumed >= CONSUMED_SAMPLES:
                    break
                consumed_per_file[file_index] += 1
                consumed += 1
            if consumed >= CONSUMED_SAMPLES:
                break
        if consumed >= CONSUMED_SAMPLES:
            break
    counts = np.asarray([consumed_per_file[i] for i in range(len(files))], dtype=np.float64)
    unique = int(np.count_nonzero(counts))
    return {
        "seed": seed,
        "samples_consumed": consumed,
        "consumed_per_file": counts.astype(np.int64).tolist(),
        "unique_hanchans_exposed": unique,
        "repeat_rate": float((consumed - unique) / consumed) if consumed else 0.0,
        "exposure_gini": gini(counts),
        "top_1pct_share": float(counts[np.argsort(-counts)[: max(1, len(counts) // 100)]].sum() / consumed) if consumed else 0.0,
        "top_5pct_share": float(counts[np.argsort(-counts)[: max(1, len(counts) // 20)]].sum() / consumed) if consumed else 0.0,
        "top_10pct_share": float(counts[np.argsort(-counts)[: max(1, len(counts) // 10)]].sum() / consumed) if consumed else 0.0,
        "effective_hanchan_n": float(consumed**2 / np.sum(counts**2)) if np.sum(counts**2) else 0.0,
    }


def run_preview(config: Path, seed: int) -> list[dict[str, Any]]:
    output = config.parent / f"_audit_preview_{seed}.json"
    subprocess.run(
        [sys.executable, str(PREVIEW_SCRIPT), "--config", str(config), "--data-seed", str(seed), "--batch-count", "3", "--output", str(output)],
        cwd=REPO,
        check=True,
    )
    report = read_json(output)
    output.unlink(missing_ok=True)
    return [row["sha256"] for row in report["batches"]]


# ---------------------------------------------------------------- bootstrap
def cluster_bootstrap_delta(
    family: str,
    route_agg: dict[str, list[dict[str, Any]]],
    view_agg: dict[str, list[dict[str, Any]]],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    def combine(agg: list[dict[str, Any]], chosen: np.ndarray) -> Counter:
        if family in ("structured_jsd", "structured_tv"):
            field = "strata"
        elif family == "shanten_jsd":
            field = "shanten"
        elif family == "action_kind_jsd":
            field = "actions"
        else:
            raise ValueError(family)
        result: Counter = Counter()
        for idx in chosen:
            result.update(agg[int(idx)][field])
        return result

    def dist(a: Counter, b: Counter) -> float:
        keys = sorted(set(a) | set(b))
        pa = np.asarray([a.get(k, 0) for k in keys], dtype=np.float64)
        pb = np.asarray([b.get(k, 0) for k in keys], dtype=np.float64)
        pa = pa / pa.sum() if pa.sum() else pa
        pb = pb / pb.sum() if pb.sum() else pb
        return jsd(pa, pb) if family.endswith("jsd") else tv_distance(pa, pb)

    n = len(route_agg["M0"])
    n1 = len(route_agg["D1"])
    n3 = len(route_agg["D3"])
    nv = len(view_agg["D2"])
    point_view = dist(combine(view_agg["D1"], np.arange(n1)), combine(view_agg["D2"], np.arange(nv)))
    point_d01 = dist(combine(route_agg["M0"], np.arange(n)), combine(route_agg["D1"], np.arange(n1)))
    point_d03 = dist(combine(route_agg["M0"], np.arange(n)), combine(route_agg["D3"], np.arange(n3)))
    d01_samples = []
    d03_samples = []
    view_samples = []
    for _ in range(reps):
        view_samples.append(
            dist(
                combine(view_agg["D1"], rng.integers(0, n1, size=n1)),
                combine(view_agg["D2"], rng.integers(0, nv, size=nv)),
            )
        )
        d01_samples.append(
            dist(
                combine(route_agg["M0"], rng.integers(0, n, size=n)),
                combine(route_agg["D1"], rng.integers(0, n1, size=n1)),
            )
        )
        d03_samples.append(
            dist(
                combine(route_agg["M0"], rng.integers(0, n, size=n)),
                combine(route_agg["D3"], rng.integers(0, n3, size=n3)),
            )
        )

    def ci(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.025))]), float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.975))])]

    return {
        "family": family,
        "point_d_m0_d1": point_d01,
        "point_d_m0_d3": point_d03,
        "point_d1_d2_view": point_view,
        "delta1_ci95": ci([d01_samples[i] - view_samples[i] for i in range(reps)]),
        "delta3_ci95": ci([d03_samples[i] - view_samples[i] for i in range(reps)]),
        "reps": reps,
        "bootstrap_seed": seed,
    }


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k0-model", type=Path, default=K0_MODEL)
    parser.add_argument("--device", type=str, default="cuda", help="torch device for the K0 Q readout")
    parser.add_argument(
        "--routes",
        type=str,
        default=",".join(FULL_ROUTE_NAMES),
        help="comma-separated route subset to scan; the default runs the complete audit",
    )
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--skip-exposure", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="cap files per route (benchmarking)")
    parser.add_argument("--equivalence", type=int, default=None, help="run reference vs fast on N files per route and exit")
    args = parser.parse_args(argv)
    route_names = tuple(dict.fromkeys(name.strip() for name in args.routes.split(",") if name.strip()))
    if not route_names:
        raise ValueError("--routes must contain at least one route name")
    unknown_routes = sorted(set(route_names) - set(FULL_ROUTE_NAMES))
    if unknown_routes:
        raise ValueError(f"unknown routes: {unknown_routes}; expected {FULL_ROUTE_NAMES}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Fix the NVIDIA driver/kernel-module state (usually reboot after a driver update) "
            "or use --device cpu explicitly."
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        cwd=REPO,
    ).strip().splitlines()
    git_worktree_clean = not any(line.strip() for line in git_status)
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    report: dict[str, Any] = {
        "schema": "keqing.mortal.cross_corpus_mechanism_audit.v2",
        "verdict": {
            "readout": "pending",
            "authoritative": False,
            "b_machine_gate": B_MACHINE_GATE,
            "new_experiment_created": False,
            "training_performed": False,
            "generation_performed": False,
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip(),
        "git_worktree_clean": git_worktree_clean,
        "git_worktree_status": git_status,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "device": str(device),
        "cuda_device_index": 0 if cuda_available and device.type == "cuda" else None,
        "cuda_device_name": cuda_device_name if cuda_available and device.type == "cuda" else None,
        "routes_requested": list(route_names),
        "source_files": {
            "audit": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "preview": {
                "path": str(PREVIEW_SCRIPT.resolve()),
                "sha256": sha256(PREVIEW_SCRIPT),
            },
            "setup_dev": {
                "path": str((REPO / "scripts/setup-dev.sh").resolve()),
                "sha256": sha256(REPO / "scripts/setup-dev.sh"),
            },
        },
        "checks": checks,
    }
    cache_manifest_path = output_dir / "route_cache_manifest.json"
    cache_manifest = read_json(cache_manifest_path) if cache_manifest_path.is_file() else None
    if cache_manifest is not None and cache_manifest.get("schema") != "keqing.mortal.route_cache_manifest.v1":
        raise ValueError(f"unsupported route cache manifest schema in {cache_manifest_path}")

    # gate A
    m0_index = load_index(D1_PREP / "file_index_m0.pth")
    d1_index = load_index(D1_PREP / "file_index_d1.pth")
    d2_index = load_index(D2_DATASET / "file_index_d2.pth")
    d3_index = load_index(D3_INDEX)
    if sha256(D3_INDEX) != D3_INDEX_SHA:
        raise ValueError("D3 frozen file index SHA mismatch")
    mapping = json.loads((D2_DATASET / "player_names_by_file.json").read_text(encoding="utf-8"))
    mapping_normalized = {str(native_path(k)): str(v) for k, v in mapping.items()}
    d2_paths = {str(path.resolve()) for path in d2_index}
    d1_paths = {str(path.resolve()) for path in d1_index}
    d2_labels = Counter(mapping_normalized.values())
    checks["gate_a_d2_hanchans_equal_d1"] = d2_paths == d1_paths
    checks["gate_a_d2_v2_v3_3000_3000"] = d2_labels.get("V2_74000", 0) == 3000 and d2_labels.get("V3_74000", 0) == 3000
    routes = {
        "M0": {"index": m0_index, "labels": ["ext_mortal"], "by_file": None, "configs": {seed: D1_PREP / f"M0_control/seed_{seed}/config.toml" for seed in SEEDS}},
        "D1": {"index": d1_index, "labels": ["K0_70k"], "by_file": None, "configs": {seed: D1_PREP / f"D1_variant/seed_{seed}/config.toml" for seed in SEEDS}},
        "D2": {"index": d2_index, "labels": sorted(set(mapping_normalized.values())), "by_file": mapping_normalized, "configs": {seed: D2_PREP / f"D2_variant/seed_{seed}/config.toml" for seed in SEEDS}},
        "D3": {"index": d3_index, "labels": ["K0_70k"], "by_file": None, "configs": {seed: D3_RECIPE / f"seed_{seed}/config.toml" for seed in SEEDS}},
    }
    report["provenance"] = {
        "M0_index": str(D1_PREP / "file_index_m0.pth"),
        "D1_index": str(D1_PREP / "file_index_d1.pth"),
        "D2_index": str(D2_DATASET / "file_index_d2.pth"),
        "D3_index": str(D3_INDEX),
        "D3_index_sha256": D3_INDEX_SHA,
        "D2_v2": d2_labels.get("V2_74000", 0),
        "D2_v3": d2_labels.get("V3_74000", 0),
    }

    # gates B+C
    state = load_checkpoint(args.k0_model.resolve())
    version = int(state["config"]["control"].get("version", 4))
    del state
    brain, dqn, _ = load_model(load_checkpoint(args.k0_model.resolve()), device)

    if args.equivalence:
        d3_keys = None
        result = {"comparisons": {}}
        for name in route_names:
            ref_files = routes[name]["index"][: args.equivalence]
            ref = scan_route(
                name, ref_files, routes[name]["labels"], routes[name]["by_file"],
                version, brain, dqn, args.q_batch_size, device,
                d3_events_by_key=d3_keys if name == "D3" else None,
            )
            fast = scan_route_fast(
                name, routes[name]["index"], routes[name]["labels"], routes[name]["by_file"],
                version, brain, dqn, args.q_batch_size, device, max_files=args.equivalence,
                d3_events_by_key=d3_keys if name == "D3" else None,
            )
            rm = ref["meta"]
            fm = fast["meta"]
            subchecks = {
                "perspectives": rm["perspectives"] == fm["perspectives"],
                "total_rows": rm["total_rows"] == fm["total_rows"],
                "final_rank_counts": rm["final_rank_counts"] == fm["final_rank_counts"],
                "target_counts": rm["target_counts"] == fm["target_counts"],
                "unique_state_hashes": rm["unique_state_hashes"] == fm["unique_state_hashes"],
                "unique_decision_hashes": rm["unique_decision_hashes"] == fm["unique_decision_hashes"],
                "action_kind": dict(ref["distributions"]["action_kind"]) == dict(fast["distributions"]["action_kind"]),
                "shanten": dict(ref["distributions"]["shanten"]) == dict(fast["distributions"]["shanten"]),
                "phase": dict(ref["distributions"]["phase"]) == dict(fast["distributions"]["phase"]),
                "rank": dict(ref["distributions"]["rank"]) == dict(fast["distributions"]["rank"]),
                "score_gap": dict(ref["distributions"]["score_gap"]) == dict(fast["distributions"]["score_gap"]),
                "legal": dict(ref["distributions"]["legal"]) == dict(fast["distributions"]["legal"]),
                "state_hash_set": _as_hash_set(ref["state_hashes"]) == _as_hash_set(fast["state_hashes"]),
                "decision_hash_set": _as_hash_set(ref["decision_hashes"]) == _as_hash_set(fast["decision_hashes"]),
            }
            exact_ok = all(subchecks.values())
            numeric_ok = (
                abs(rm["greedy_agreement_rate"] - fm["greedy_agreement_rate"]) < 1e-6
                and abs(rm["behavior_q_target_corr"] - fm["behavior_q_target_corr"]) < 1e-3
                and abs(rm["within_stratum_target_var_ratio"] - fm["within_stratum_target_var_ratio"]) < 1e-3
            )
            result["comparisons"][name] = {
                "exact_ok": exact_ok,
                "numeric_ok": numeric_ok,
                "ref_rows": rm["total_rows"],
                "fast_rows": fm["total_rows"],
                "ref_greedy_agree": rm["greedy_agreement_rate"],
                "fast_greedy_agree": fm["greedy_agreement_rate"],
            }
            print(json.dumps({"route": name, "exact_ok": exact_ok, "numeric_ok": numeric_ok, "failed": [k for k, v in subchecks.items() if not v], "ref_rank": rm["final_rank_counts"], "fast_rank": fm["final_rank_counts"], "ref_target": rm["target_counts"], "fast_target": fm["target_counts"]}, ensure_ascii=False), flush=True)
            if not (exact_ok and numeric_ok):
                raise ValueError(f"equivalence FAILED for {name}")
        print(json.dumps({"equivalence": "PASS", "routes": result["comparisons"]}, ensure_ascii=False, indent=2), flush=True)
        return

    route_scan: dict[str, dict[str, Any]] = {}
    route_timings: dict[str, float] = {}
    t0 = time.time()
    report["provenance"]["agg_source"] = {}
    cache_dir = output_dir / "route_agg_cache"
    for name in route_names:
        cache_path = cache_dir / f"{name}.json"
        if cache_path.is_file():
            route_scan[name] = load_route_agg(cache_path)
            report["provenance"]["agg_source"][name] = "cache"
            print(f"[audit] {name} loaded from agg cache", flush=True)
            continue
        d3_events = None
        if name == "D3":
            d3_events = load_d3_events_by_key()
        route_t0 = time.time()
        route_scan[name] = scan_route_fast(
            name, routes[name]["index"], routes[name]["labels"], routes[name]["by_file"],
            version, brain, dqn, args.q_batch_size, device, max_files=args.max_files,
            d3_events_by_key=d3_events,
        )
        route_elapsed = time.time() - route_t0
        route_timings[name] = route_elapsed
        report["provenance"]["agg_source"][name] = "live"
        print(
            f"[audit] {name} rows={route_scan[name]['meta']['total_rows']} "
            f"route_elapsed={route_elapsed:.1f}s elapsed_total={time.time() - t0:.1f}s",
            flush=True,
        )
        dump_route_agg(route_scan[name], cache_path)
    report["route_cache"] = {}
    for name in route_names:
        cache_path = cache_dir / f"{name}.json"
        if not cache_path.is_file():
            raise ValueError(f"missing route agg cache: {cache_path}")
        actual_cache_sha = sha256(cache_path)
        entry: dict[str, Any] = {
            "sha256": actual_cache_sha,
            "device": str(device),
            "rows": int(route_scan[name]["meta"]["total_rows"]),
        }
        if cache_manifest is not None:
            manifest_route = cache_manifest.get("routes", {}).get(name)
            if not isinstance(manifest_route, dict):
                raise ValueError(f"route cache manifest is missing route {name}")
            if manifest_route.get("sha256") != actual_cache_sha:
                raise ValueError(
                    f"route cache manifest SHA mismatch for {name}: "
                    f"manifest={manifest_route.get('sha256')} actual={actual_cache_sha}"
                )
            entry["route_elapsed_seconds"] = manifest_route.get("route_elapsed_seconds")
            entry["cache_sha256_matches_manifest"] = True
            entry["cache_generated_under_commit"] = cache_manifest.get("source_commit")
        report["route_cache"][name] = entry
    report["route_cache_manifest"] = cache_manifest
    if cache_manifest is not None and set(cache_manifest.get("routes", {})) >= set(FULL_ROUTE_NAMES):
        report["route_timings_seconds"] = {
            name: cache_manifest["routes"][name].get("route_elapsed_seconds") for name in FULL_ROUTE_NAMES
        }
    else:
        report["route_timings_seconds"] = {name: route_timings.get(name) for name in route_names}
    report["corpus_rows_summary"] = {name: route_scan[name]["meta"] for name in route_names}

    if route_names != FULL_ROUTE_NAMES:
        partial = {
            "schema": "keqing.mortal.cross_corpus_route_scan_partial.v2",
            "git_commit": report["git_commit"],
            "device": str(device),
            "routes_scanned": list(route_names),
            "corpus_rows_summary": report["corpus_rows_summary"],
            "note": "route scan only; gates D/E/F and the verdict are not computed for a route subset",
        }
        (output_dir / "route_scan_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "partial_route_scan": list(route_names),
                    "corpus_rows_summary": partial["corpus_rows_summary"],
                    "output": str(output_dir / "route_scan_partial.json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    (output_dir / "corpus_rows_summary.json").write_text(json.dumps(report["corpus_rows_summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # gate D
    if not args.skip_exposure:
        import tomllib

        frozen_previews = {
            "M0": {"20260806": D1_PREP / "preflight/m0_control_batch_preview.json"},
            "D1": {"20260806": D1_PREP / "preflight/d1_variant_batch_preview.json"},
            "D2": {str(seed): D2_PREP / f"preflight/d2_batch_preview_{seed}.json" for seed in SEEDS},
            "D3": {str(seed): D3_RECIPE / f"preflight/batch_preview_{seed}_repeat_0.json" for seed in SEEDS},
        }
        exposure: dict[str, Any] = {}
        for name in ("M0", "D1", "D2", "D3"):
            exposure[name] = {}
            for seed in SEEDS:
                with routes[name]["configs"][seed].open("rb") as handle:
                    config = tomllib.load(handle)
                file_batch_size = int(config["dataset"]["file_batch_size"])
                num_epochs = int(config["dataset"]["num_epochs"])
                sim = simulate_exposure(routes[name]["index"], seed, file_batch_size, num_epochs, route_scan[name]["per_file_counters"])
                sim_repeat = simulate_exposure(routes[name]["index"], seed, file_batch_size, num_epochs, route_scan[name]["per_file_counters"])
                frozen = frozen_previews[name].get(str(seed))
                preview_match = None
                if frozen is not None and frozen.is_file():
                    current_hashes = run_preview(routes[name]["configs"][seed], seed)
                    frozen_hashes = [row["sha256"] for row in read_json(frozen)["batches"]]
                    preview_match = current_hashes == frozen_hashes
                    if not preview_match:
                        raise ValueError(f"{name} seed {seed}: preview hashes differ from frozen reference")
                exposure[name][str(seed)] = {"simulation": sim, "stream_deterministic": sim == sim_repeat, "frozen_preview_match": preview_match}
        report["training_exposure"] = exposure
        exposure_weighted = exposure_weighted_readout(route_scan, exposure, SEEDS, ("M0", "D1", "D2", "D3"))
        report["exposure_weighted"] = exposure_weighted
        (output_dir / "exposure_weighted.json").write_text(json.dumps(exposure_weighted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "training_exposure.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checks["gate_d_exposure_simulation"] = True
        checks["gate_d_stream_deterministic"] = all(
            exposure[name][str(seed)]["stream_deterministic"] for name in ("M0", "D1", "D2", "D3") for seed in SEEDS
        )
        checks["gate_d_frozen_previews_match"] = all(
            exposure[name][str(seed)]["frozen_preview_match"] is not False
            for name in ("M0", "D1", "D2", "D3") for seed in SEEDS
        )
    else:
        report["training_exposure"] = {"skipped": True}
        report["exposure_weighted"] = {"skipped": True}
        checks["gate_d_exposure_simulation"] = True
        checks["gate_d_stream_deterministic"] = True
        checks["gate_d_frozen_previews_match"] = True

    # gate E
    hanchan_hashes: dict[str, dict[str, str]] = {}
    for name in ("M0", "D1", "D2", "D3"):
        hashes = {}
        for path in routes[name]["index"]:
            hashes[str(path.resolve())] = _canonical_log_hash(_read_log(path))
        hanchan_hashes[name] = hashes
    overlap_matrix = {
        a: {b: len(set(hanchan_hashes[a].values()) & set(hanchan_hashes[b].values())) for b in ("M0", "D1", "D2", "D3")}
        for a in ("M0", "D1", "D2", "D3")
    }
    checks["gate_e_d1_d2_hanchan_overlap_100pct"] = overlap_matrix["D1"]["D2"] == 6000
    strata = {name: Counter({k: v for entry in route_scan[name]["hanchan_agg"] for k, v in entry["strata"].items()}) for name in ("M0", "D1", "D2", "D3")}
    m0_exclusive = {name: weighted_missing_mass(strata["M0"], strata[name]) for name in ("D1", "D2", "D3")}
    jsd_matrix, tv_matrix = {}, {}
    for a in ("M0", "D1", "D2", "D3"):
        jsd_matrix[a], tv_matrix[a] = {}, {}
        for b in ("M0", "D1", "D2", "D3"):
            keys = sorted(set(strata[a]) | set(strata[b]))
            pa = np.asarray([strata[a].get(k, 0) for k in keys], dtype=np.float64)
            pb = np.asarray([strata[b].get(k, 0) for k in keys], dtype=np.float64)
            pa = pa / pa.sum()
            pb = pb / pb.sum()
            jsd_matrix[a][b] = jsd(pa, pb)
            tv_matrix[a][b] = tv_distance(pa, pb)
    report["support_overlap"] = {
        "hanchan_overlap_matrix": overlap_matrix,
        "m0_exclusive_weighted_mass": m0_exclusive,
        "structured_jsd_matrix": jsd_matrix,
        "structured_tv_matrix": tv_matrix,
    }
    (output_dir / "support_overlap.json").write_text(json.dumps(report["support_overlap"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks["gate_e_overlap_distance"] = True

    # gate F (D3 exploration diagnostic; hard gate on exactly-once mapping)
    d3_diag = route_scan["D3"]["d3_diag"]
    d3_summary = summarize_d3_diag(d3_diag) if d3_diag else None
    report["d3_exploration_readout"] = d3_summary
    f_checks = (
        gate_f_checks(d3_summary)
        if d3_summary
        else {"diagnostic_present": False, "all_events_mapped_exactly_once": False, "no_unconsumed_events": False, "category_counts_exact": False}
    )
    checks.update({f"gate_f_{key}": value for key, value in f_checks.items()})
    checks["gate_f_d3_exploration_diagnostic"] = all(f_checks.values())

    # verdict (frozen repair-1 adjudication; B is descriptive only)
    view_agg = {"D1": route_scan["D1"]["hanchan_agg"], "D2": route_scan["D2"]["hanchan_agg"]}
    route_agg = {name: route_scan[name]["hanchan_agg"] for name in ("M0", "D1", "D3")}
    families = {}
    for family in ("structured_jsd", "structured_tv", "shanten_jsd", "action_kind_jsd"):
        families[family] = cluster_bootstrap_delta(family, route_agg, view_agg, args.bootstrap_reps, args.bootstrap_seed)
    coverage_votes = sum(1 for f in families.values() if f["delta1_ci95"][0] > 0 and f["delta3_ci95"][0] > 0)
    gates_ok = all(value for key, value in checks.items() if isinstance(value, bool))
    readout = decide_readout(
        coverage_votes=coverage_votes,
        num_families=len(families),
        m0_exclusive_mass=m0_exclusive,
        gates_ok=gates_ok,
    )
    report["verdict"]["readout"] = readout
    report["verdict"]["authoritative"] = gates_ok and readout != VERDICT_NO_GATES and git_worktree_clean
    report["mechanism_bootstrap"] = families
    report["coverage_support"] = {"votes": coverage_votes, "families": len(families), "m0_exclusive_mass": m0_exclusive}
    report["credit_assignment_diagnostics"] = {
        "machine_gate": B_MACHINE_GATE,
        "note": (
            "descriptive only: no quantitative B promotion threshold was frozen before data, "
            "so B_credit_assignment_priority is never machine-promoted"
        ),
        "within_stratum_target_var_ratio": {
            name: route_scan[name]["meta"]["within_stratum_target_var_ratio"] for name in ("M0", "D1", "D2", "D3")
        },
        "behavior_q_target_corr": {
            name: route_scan[name]["meta"]["behavior_q_target_corr"] for name in ("M0", "D1", "D2", "D3")
        },
    }
    report["checks"] = checks
    out_json = output_dir / "cross_corpus_mechanism_audit.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": readout,
                "authoritative": report["verdict"]["authoritative"],
                "checks": checks,
                "m0_exclusive_mass": m0_exclusive,
                "coverage_votes": f"{coverage_votes}/{len(families)}",
                "output": str(out_json),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def load_d3_events_by_key() -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group frozen D3 exploration events (24 shards) by (seed, key)."""
    events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    roots = [
        D3_EXP_ROOT / "generation_production/shard_000_1800000_1800249",
        *sorted((D3_EXP_ROOT / "generation_continuation").glob("shard_*/")),
    ]
    for run_dir in roots:
        events_path = run_dir / "exploration/exploration_events.jsonl"
        if not events_path.is_file():
            raise ValueError(f"missing exploration events: {events_path}")
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            events_by_key[(int(event["generation_seed"]), int(event["seed_key"]))].append(event)
    return dict(events_by_key)


def summarize_d3_diag(d3_diag: dict[str, Any]) -> dict[str, Any]:
    """Per-category aggregate + per-hanchan rate stats (cluster unit = hanchan).

    'events_mapped_to_rows' counts EVENTS consumed by exactly one loader row,
    never histogram-cell increments. Histogram cells stay descriptive only.
    """
    category_total = d3_diag["category_total"]
    category_hanchan = d3_diag["category_hanchan"]
    category_mapped = d3_diag["category_mapped_events"]
    category_unconsumed = d3_diag["category_unconsumed"]
    category_hanchan_events = d3_diag["category_hanchan_events"]
    result: dict[str, Any] = {
        "category_totals": {str(key): int(value) for key, value in sorted(category_total.items())},
        "mapped_event_totals": {str(key): int(category_mapped.get(key, 0)) for key in sorted(category_total)},
        "unconsumed_event_totals": {str(key): int(category_unconsumed.get(key, 0)) for key in sorted(category_total)},
        "total_events": int(sum(category_total.values())),
        "total_mapped_events": int(sum(category_mapped.values())),
        "total_unconsumed_events": int(sum(category_unconsumed.values())),
    }
    for category in sorted(category_total):
        hanchan_rows = category_hanchan[category]
        hanchan_event_counts = [int(category_hanchan_events[category].get(hanchan, 0)) for hanchan in hanchan_rows]
        histograms: Counter = Counter()
        for hanchan in hanchan_rows:
            histograms.update(hanchan_rows[hanchan])
        result[category] = {
            "events": int(category_total[category]),
            "events_mapped_to_rows": int(category_mapped.get(category, 0)),
            "hanchans_with_events": int(sum(1 for count in hanchan_event_counts if count > 0)),
            "per_hanchan_event_count": {
                "min": min(hanchan_event_counts) if hanchan_event_counts else 0,
                "median": float(np.median(hanchan_event_counts)) if hanchan_event_counts else 0.0,
                "max": max(hanchan_event_counts) if hanchan_event_counts else 0,
            },
            "histograms": {key: value for key, value in sorted(histograms.items())},
        }
    return result


# ---------------------------------------------------------------- agg cache
def _counter_to_json(counter: Counter, tuple_keys: bool) -> dict[str, int]:
    return {("|".join(key) if tuple_keys else str(key)): int(value) for key, value in counter.items()}


def _json_to_counter(payload: dict[str, int], tuple_keys: bool) -> Counter:
    counter: Counter = Counter()
    for key, value in payload.items():
        counter[tuple(key.split("|")) if tuple_keys else key] = value
    return counter


def _load_diag(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "category_hanchan": defaultdict(
            lambda: defaultdict(Counter),
            {
                category: defaultdict(Counter, {int(hanchan): Counter(cells) for hanchan, cells in hanchans.items()})
                for category, hanchans in payload["category_hanchan"].items()
            },
        ),
        "category_total": Counter(payload["category_total"]),
        "category_mapped_events": Counter(payload["category_mapped_events"]),
        "category_unconsumed": Counter(payload["category_unconsumed"]),
        "category_hanchan_events": defaultdict(
            Counter,
            {
                category: Counter({int(hanchan): int(count) for hanchan, count in hanchans.items()})
                for category, hanchans in payload["category_hanchan_events"].items()
            },
        ),
    }


def dump_route_agg(scan: dict[str, Any], path: Path) -> None:
    """Persist per-route aggregation so later adjudication rounds can recompute
    bootstrap / exposure-weighted readouts without rescanning the corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    diag = scan.get("d3_diag")
    serialized_diag = None
    if diag is not None:
        serialized_diag = {
            "category_hanchan": {
                category: {str(hanchan): dict(cells) for hanchan, cells in hanchans.items()}
                for category, hanchans in diag["category_hanchan"].items()
            },
            "category_total": dict(diag["category_total"]),
            "category_mapped_events": dict(diag["category_mapped_events"]),
            "category_unconsumed": dict(diag["category_unconsumed"]),
            "category_hanchan_events": {
                category: {str(hanchan): int(count) for hanchan, count in hanchans.items()}
                for category, hanchans in diag["category_hanchan_events"].items()
            },
        }
    payload = {
        "cache_schema": ROUTE_AGG_CACHE_SCHEMA,
        "route": scan["route"],
        "meta": scan["meta"],
        "hanchan_agg": [
            {
                "hanchan": int(entry["hanchan"]),
                "rows": int(entry["rows"]),
                "strata": _counter_to_json(entry["strata"], tuple_keys=True),
                "actions": _counter_to_json(entry["actions"], tuple_keys=False),
                "shanten": _counter_to_json(entry["shanten"], tuple_keys=False),
            }
            for entry in scan["hanchan_agg"]
        ],
        "per_file_counters": {str(index): dict(counter) for index, counter in scan["per_file_counters"].items()},
        "per_file_target_counts": {str(index): dict(counter) for index, counter in scan["per_file_target_counts"].items()},
        "distributions": scan["distributions"],
        "d3_diag": serialized_diag,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def load_route_agg(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("cache_schema") != ROUTE_AGG_CACHE_SCHEMA:
        raise ValueError(f"stale aggregation cache {path.name}: delete route_agg_cache and rescan")
    return {
        "route": payload["route"],
        "meta": payload["meta"],
        "hanchan_agg": [
            {
                "hanchan": int(entry["hanchan"]),
                "rows": int(entry["rows"]),
                "strata": _json_to_counter(entry["strata"], tuple_keys=True),
                "actions": _json_to_counter(entry["actions"], tuple_keys=False),
                "shanten": _json_to_counter(entry["shanten"], tuple_keys=False),
            }
            for entry in payload["hanchan_agg"]
        ],
        "per_file_counters": {int(index): Counter(counter) for index, counter in payload["per_file_counters"].items()},
        "per_file_target_counts": {int(index): Counter(counter) for index, counter in payload["per_file_target_counts"].items()},
        "distributions": payload["distributions"],
        "state_hashes": b"",
        "decision_hashes": b"",
        "d3_diag": _load_diag(payload["d3_diag"]),
    }


if __name__ == "__main__":
    import faulthandler
    import time as _t

    faulthandler.enable()
    print("[startup]", _t.strftime("%H:%M:%S"), flush=True)
    main()
