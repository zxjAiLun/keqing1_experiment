#!/usr/bin/env python3
"""Cross-corpus mechanism audit: M0 vs D1/D2/D3 (read-only, no training).

Memory-frugal rewrite: no per-row storage. Per route we keep only per-hanchan
aggregate counters (strata/action/shanten histograms), streaming Q-target
covariance accumulators, state/decision hash sets, and per-file row counters
(for exposure simulation). This keeps the audit within laptop RAM while the
K0 forward pass streams over every row exactly once.

Gates:
  A provenance (frozen indexes; D2 hanchans == D1 6000/6000, V2/V3 3000/3000)
  B loader-view integrity (6000 files / 6000 perspectives / 0 malformed /
    100% legal / augmented false)
  C canonical row audit + K0 Q readout aggregates
  D actual training exposure (RNG-faithful FileDatasetsIter simulation +
     real preview runs vs frozen batch hashes)
  E overlap / distance (hanchan, structured strata weighted mass, JSD, TV;
     D1-D2 hanchan overlap == 100%)
  F D3 exploration diagnostic (explored vs hash_rejected, hanchan-cluster
    uncertainty, descriptive only)

Verdict readout: A_coverage_priority / B_credit_assignment_priority /
inconclusive. Priority, never a causal proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

from training.mortal.d3_production_audit_core import _canonical_log_hash, _read_log, _log_key  # noqa: E402
from training.mortal.audit_replay_distribution import (  # noqa: E402
    decision_hash,
    load_checkpoint,
    load_model,
    model_q,
    records_from_game,
    state_hash,
)

TRAIN_PTS = np.asarray([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
SEEDS = (20260806, 20260807, 20260808)
CONSUMED_SAMPLES = 2000 * 512
DEFAULT_OUTPUT_DIR = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\diagnostics\cross_corpus_mechanism_audit"
)
D1_PREP = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D1_project_owned_population_2026_07\training_prep_2026_07"
)
D2_PREP = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D2_project_owned_descendant_view_mix_2026_08\training_prep_2026_08"
)
D2_DATASET = D2_PREP.parent / "dataset"
D3_RECIPE = (
    REPO
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
    / "training_recipe_2026_08"
)
D3_INDEX = D3_RECIPE.parent / "training_contract_2026_08/file_index_d3_k0.pth"
D3_INDEX_SHA = "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"
K0_MODEL = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\models\K0_70k\mortal_default_70k_promoted_candidate.pth"
)
D3_EXP_ROOT = (
    REPO
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
)
PREVIEW_SCRIPT = REPO / "training/mortal/preview_dataloader_batches_2026_07.py"


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
    return [Path(str(value)) for value in values]


def _as_hash_set(hashes: Any) -> set[bytes]:
    if isinstance(hashes, bytes):
        return set(hashes[i : i + 16] for i in range(0, len(hashes), 16))
    return set(hashes)


def bucket_score_gap(value: float) -> str:
    return "ahead_big" if value >= 12000 else "ahead" if value >= 0 else "behind" if value > -12000 else "behind_big"


def bucket_legal(value: int) -> str:
    return "1_5" if value <= 5 else "6_10" if value <= 10 else "11_plus"


def bucket_shanten(value: int) -> str:
    return "tenpai" if value <= 0 else str(value) if value <= 2 else "3_plus"


def strata_key(kyoku: int, current_rank: int, score_gap: float, own_riichi: bool, legal: int, shanten: int) -> tuple[str, ...]:
    return (
        str(kyoku),
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
    d3_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

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
        }

    q_pending: list[tuple[Any, dict[str, Any]]] = []

    def flush_q() -> None:
        nonlocal greedy_agreement
        if not q_pending:
            return
        q = model_q(brain, dqn, [item[0] for item in q_pending], torch.device("cuda"))
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
            key = row["strata"]
            if key not in strata_tacc:
                strata_tacc[key] = covariance_accumulators()
            sacc = strata_tacc[key]
            sacc["n"] += 1
            sacc["sum_t"] += target
            sacc["sum_t2"] += target * target
            category = row.get("event_category")
            if category is not None:
                hanchan = row["hanchan"]
                for bucket_name, value in (
                    ("target", f"{target:+.0f}"),
                    ("final_rank", str(row["final_rank"])),
                    ("behavior_q", f"{behavior_q:+.3f}"),
                    ("q_regret", f"{greedy_q - behavior_q:+.3f}"),
                    ("margin", f"{(q_row[legal].max() - top2_min):+.3f}"),
                    ("phase", str(row["kyoku"])),
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
            from training.mortal.d3_native_scene import reconstruct_native_scenes  # noqa: PLC0415
            from training.mortal.d3_production_audit_core import primary_row_flags  # noqa: PLC0415

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
            kyoku = int(record.kyoku)
            rank = int(record.current_rank)
            score_gap = float(record.score_gap)
            own_riichi = bool(record.own_riichi)
            shanten = int(record.shanten)
            key = strata_key(kyoku, rank, score_gap, own_riichi, legal_count, shanten)
            hanchan_strata[key] += 1
            hanchan_actions[str(action)] += 1
            hanchan_shanten[bucket_shanten(shanten)] += 1
            action_hist[str(action)] += 1
            shanten_hist[bucket_shanten(shanten)] += 1
            phase_hist[str(kyoku)] += 1
            rank_hist[str(rank)] += 1
            score_gap_hist[bucket_score_gap(score_gap)] += 1
            legal_hist[bucket_legal(legal_count)] += 1
            per_file_counters[file_index]["rows"] += 1
            per_file_target_counts[file_index][str(float(record.target))] += 1
            event_category = None
            if event_row_map:
                loader_index = loader_index_by_kyoku.get(kyoku, 0)
                loader_index_by_kyoku[kyoku] = loader_index + 1
                event = event_row_map.get((kyoku, loader_index))
                if event is not None:
                    event_category = (
                        "explored"
                        if event.get("explored")
                        else "hash_rejected"
                        if event.get("reason") == "hash_rejected"
                        else "budget_exhausted"
                    )
            row = {
                "action": action,
                "target": float(record.target),
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
    ) / total_rows
    total_var = (global_tacc["sum_t2"] - global_tacc["sum_t"] ** 2 / global_tacc["n"]) / total_rows if global_tacc["n"] > 1 else 0.0
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
            "action": dict(action_hist),
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
    max_files: int | None = None,
    d3_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

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
        d3_diag = {"category_hanchan": defaultdict(lambda: defaultdict(Counter)), "category_total": Counter()}

    pending_obs: list[np.ndarray] = []
    pending_masks: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []

    def flush_q() -> None:
        nonlocal greedy_agreement
        if not pending_obs:
            return
        obs_tensor = torch.as_tensor(np.ascontiguousarray(np.stack(pending_obs)), device=torch.device("cuda"))
        mask_tensor = torch.as_tensor(np.ascontiguousarray(np.stack(pending_masks)), device=torch.device("cuda"))
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
            key = meta["strata"]
            sacc = strata_tacc.setdefault(key, covariance_accumulators())
            sacc["n"] += 1
            sacc["sum_t"] += meta["target"]
            sacc["sum_t2"] += meta["target"] * meta["target"]
            category = meta.get("event_category")
            if category is not None:
                hanchan = meta["hanchan"]
                for bucket_name, value in (
                    ("target", f"{meta['target']:+.0f}"),
                    ("final_rank", str(meta["final_rank"])),
                    ("behavior_q", f"{behavior_q:+.3f}"),
                    ("q_regret", f"{greedy_q - behavior_q:+.3f}"),
                    ("margin", f"{margin:+.3f}"),
                    ("phase", str(meta["kyoku"])),
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
                    from training.mortal.d3_native_scene import reconstruct_native_scenes  # noqa: PLC0415
                    from training.mortal.d3_production_audit_core import primary_row_flags  # noqa: PLC0415

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
                    hanchan_actions[str(action)] += 1
                    hanchan_shanten[bucket_shanten(shanten)] += 1
                    action_hist[str(action)] += 1
                    shanten_hist[bucket_shanten(shanten)] += 1
                    phase_hist[str(kyoku)] += 1
                    rank_hist[str(rank)] += 1
                    score_gap_hist[bucket_score_gap(score_gap)] += 1
                    legal_hist[bucket_legal(legal_count)] += 1
                    per_file_counters[file_index]["rows"] += 1
                    per_file_target_counts[file_index][f"{target:+.0f}"] += 1
                    state_hash_buf += hashlib.blake2b(np.ascontiguousarray(obs_rows[i]).tobytes() + np.ascontiguousarray(masks_arr[i]).tobytes(), digest_size=16).digest()
                    decision_hash_buf += hashlib.blake2b(
                        np.ascontiguousarray(obs_rows[i]).tobytes() + np.ascontiguousarray(masks_arr[i]).tobytes() + action.to_bytes(2, "little", signed=False),
                        digest_size=16,
                    ).digest()
                    event_category = None
                    if event_row_map:
                        loader_index = loader_index_by_kyoku.get(kyoku, 0)
                        loader_index_by_kyoku[kyoku] = loader_index + 1
                        event = event_row_map.get((kyoku, loader_index))
                        if event is not None:
                            event_category = (
                                "explored"
                                if event.get("explored")
                                else "hash_rejected"
                                if event.get("reason") == "hash_rejected"
                                else "budget_exhausted"
                            )
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
                        import psutil  # noqa: PLC0415

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
    ) / total_rows
    total_var = (global_tacc["sum_t2"] - global_tacc["sum_t"] ** 2 / global_tacc["n"]) / total_rows if global_tacc["n"] > 1 else 0.0

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
            "action": dict(action_hist),
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
        elif family == "action_jsd":
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
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--skip-exposure", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="cap files per route (benchmarking)")
    parser.add_argument("--equivalence", type=int, default=None, help="run reference vs fast on N files per route and exit")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {
        "schema": "keqing.mortal.cross_corpus_mechanism_audit.v1",
        "verdict": {"readout": "inconclusive", "new_experiment_created": False, "training_performed": False, "generation_performed": False},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip(),
        "checks": checks,
    }

    # gate A
    m0_index = load_index(D1_PREP / "file_index_m0.pth")
    d1_index = load_index(D1_PREP / "file_index_d1.pth")
    d2_index = load_index(D2_DATASET / "file_index_d2.pth")
    d3_index = load_index(D3_INDEX)
    if sha256(D3_INDEX) != D3_INDEX_SHA:
        raise ValueError("D3 frozen file index SHA mismatch")
    mapping = json.loads((D2_DATASET / "player_names_by_file.json").read_text(encoding="utf-8"))
    mapping_normalized = {str(Path(str(k)).resolve()): str(v) for k, v in mapping.items()}
    d2_paths = {str(path.resolve()) for path in d2_index}
    d1_paths = {str(path.resolve()) for path in d1_index}
    d2_labels = Counter(mapping_normalized.values())
    checks["gate_a_d2_hanchans_equal_d1"] = d2_paths == d1_paths
    checks["gate_a_d2_v2_v3_3000_3000"] = d2_labels.get("V2_74000", 0) == 3000 and d2_labels.get("V3_74000", 0) == 3000
    if not (checks["gate_a_d2_hanchans_equal_d1"] and checks["gate_a_d2_v2_v3_3000_3000"]):
        raise ValueError("gate A failed")
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
    brain, dqn, _ = load_model(load_checkpoint(args.k0_model.resolve()), torch.device("cuda"))

    if args.equivalence:
        d3_keys = None
        result = {"comparisons": {}}
        for name in ("M0", "D1", "D2", "D3"):
            ref_files = routes[name]["index"][: args.equivalence]
            ref = scan_route(
                name, ref_files, routes[name]["labels"], routes[name]["by_file"],
                version, brain, dqn, args.q_batch_size,
                d3_events_by_key=d3_keys if name == "D3" else None,
            )
            fast = scan_route_fast(
                name, routes[name]["index"], routes[name]["labels"], routes[name]["by_file"],
                version, brain, dqn, args.q_batch_size, max_files=args.equivalence,
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
                "action": dict(ref["distributions"]["action"]) == dict(fast["distributions"]["action"]),
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
    t0 = time.time()
    for name in ("M0", "D1", "D2", "D3"):
        d3_events = None
        if name == "D3":
            d3_events = load_d3_events_by_key()
        route_scan[name] = scan_route_fast(
            name, routes[name]["index"], routes[name]["labels"], routes[name]["by_file"],
            version, brain, dqn, args.q_batch_size, max_files=args.max_files,
            d3_events_by_key=d3_events,
        )
        print(f"[audit] {name} rows={route_scan[name]['meta']['total_rows']} elapsed={time.time() - t0:.1f}s", flush=True)
    checks["gate_b_c_loader_and_q"] = True
    report["corpus_rows_summary"] = {name: route_scan[name]["meta"] for name in ("M0", "D1", "D2", "D3")}
    (output_dir / "corpus_rows_summary.json").write_text(json.dumps(report["corpus_rows_summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # gate D
    if not args.skip_exposure:
        import tomllib  # noqa: PLC0415

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

    # gate F (D3 exploration diagnostic)
    d3_diag = route_scan["D3"]["d3_diag"]
    report["d3_exploration_readout"] = summarize_d3_diag(d3_diag) if d3_diag else None
    checks["gate_f_d3_exploration_diagnostic"] = True

    # verdict
    view_agg = {"D1": route_scan["D1"]["hanchan_agg"], "D2": route_scan["D2"]["hanchan_agg"]}
    route_agg = {name: route_scan[name]["hanchan_agg"] for name in ("M0", "D1", "D3")}
    families = {}
    for family in ("structured_jsd", "structured_tv", "shanten_jsd", "action_jsd"):
        families[family] = cluster_bootstrap_delta(family, route_agg, view_agg, args.bootstrap_reps, args.bootstrap_seed)
    coverage_votes = sum(1 for f in families.values() if f["delta1_ci95"][0] > 0 and f["delta3_ci95"][0] > 0)
    supports_a = coverage_votes / len(families) > 0.5 and any(value > 0 for value in m0_exclusive.values())
    prioritize_b = False
    if not supports_a:
        prioritize_b = all(
            route_scan[name]["meta"]["within_stratum_target_var_ratio"] >= 0.8
            and abs(route_scan[name]["meta"]["behavior_q_target_corr"]) <= 0.3
            for name in ("M0", "D1", "D2", "D3")
        )
    readout = "A_coverage_priority" if supports_a else ("B_credit_assignment_priority" if prioritize_b else "inconclusive")
    report["verdict"]["readout"] = readout
    report["mechanism_bootstrap"] = families
    report["coverage_support"] = {"votes": coverage_votes, "families": len(families), "m0_exclusive_mass": m0_exclusive}
    report["checks"] = checks
    out_json = output_dir / "cross_corpus_mechanism_audit.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": readout, "checks": checks, "m0_exclusive_mass": m0_exclusive, "coverage_votes": f"{coverage_votes}/{len(families)}", "output": str(out_json)}, ensure_ascii=False, indent=2), flush=True)


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
    """Per-category aggregate + per-hanchan rate stats (cluster unit = hanchan)."""
    category_total = d3_diag["category_total"]
    category_hanchan = d3_diag["category_hanchan"]
    result: dict[str, Any] = {"category_totals": dict(category_total)}
    for category in sorted(category_total):
        hanchan_rows = category_hanchan[category]
        per_hanchan_counts = [sum(hanchan_rows[hanchan].values()) for hanchan in hanchan_rows]
        histograms: Counter = Counter()
        for hanchan in hanchan_rows:
            histograms.update(hanchan_rows[hanchan])
        result[category] = {
            "events": int(category_total[category]),
            "events_mapped_to_rows": sum(per_hanchan_counts),
            "hanchans_with_events": len(per_hanchan_counts),
            "per_hanchan_event_count": {
                "min": min(per_hanchan_counts) if per_hanchan_counts else 0,
                "median": float(np.median(per_hanchan_counts)) if per_hanchan_counts else 0.0,
                "max": max(per_hanchan_counts) if per_hanchan_counts else 0,
            },
            "histograms": {k: v for k, v in sorted(histograms.items())},
        }
    return result


if __name__ == "__main__":
    import faulthandler
    import time as _t

    faulthandler.enable()
    print("[startup]", _t.strftime("%H:%M:%S"), flush=True)
    main()
