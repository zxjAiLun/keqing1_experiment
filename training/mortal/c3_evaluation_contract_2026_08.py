#!/usr/bin/env python3
"""Frozen C3 D1_CQL_OFF Absolute-Promotion Evaluation Contract.

This module defines:
- The 4-player lineup and frozen checkpoint contracts for seeds 20260806, 20260807, 20260808
- The 12 shards (3000 games total, 250 games/shard, 1000 games/seed) across fresh game ID spans
- The paired evaluation statistics:
    x = Pt(D1_CQL_OFF) - Pt(K0_70k)
    y = Pt(D1_CQL_OFF) - Pt(M0_CURRENT)
- The equal-seed hierarchical paired bootstrap CI95 protocol (reps=5000, seed=20260826):
    outer training-seed resampling with replacement (sample 3 seeds from {20260806, 20260807, 20260808})
    inner hanchan resampling with replacement (sample 1000 games from each sampled seed's 1000 games)
    x and y share identical outer and inner bootstrap sample draws
- The strict promotion gates for K1 (designated canonical checkpoint D1_CQL_OFF_20260807).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
C3_EXPERIMENT_ID = "C3_d1_cql_off_absolute_promotion_2026_08"

# Shard and game parameters
SEEDS = (20260806, 20260807, 20260808)
GAMES_PER_SHARD = 250
SHARDS_PER_SEED = 4
TOTAL_SHARDS = 12
TOTAL_GAMES = 3000
SHARDS = tuple(range(TOTAL_SHARDS))

SEED_KEY = 8192
SEAT_MODE = "random"
DEVICE = "cuda"
AMP = False
RANK_POINTS = [90.0, 45.0, 0.0, -135.0]

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_CI = 95.0

# Fresh independent hanchan ranges (no overlap with C1)
# Seed 20260806: 1900000..1900999 (4 shards: 0..3)
# Seed 20260807: 1910000..1910999 (4 shards: 4..7)
# Seed 20260808: 1920000..1920999 (4 shards: 8..11)
SEED_HANCHAN_SPANS = {
    20260806: (1900000, 1900999),
    20260807: (1910000, 1910999),
    20260808: (1920000, 1920999),
}

SHARD_CONFIG: list[dict[str, Any]] = []
for _s_idx, _seed in enumerate(SEEDS):
    _start_base, _end_base = SEED_HANCHAN_SPANS[_seed]
    for _sub in range(SHARDS_PER_SEED):
        _shard_id = _s_idx * SHARDS_PER_SEED + _sub
        _h_start = _start_base + _sub * GAMES_PER_SHARD
        _h_end = _h_start + GAMES_PER_SHARD - 1
        SHARD_CONFIG.append({
            "shard_id": _shard_id,
            "training_seed": _seed,
            "sub_shard": _sub,
            "start_hanchan": _h_start,
            "end_hanchan": _h_end,
            "games_count": GAMES_PER_SHARD,
        })

# Checkpoint paths & expected SHA-256
DATA_ROOT = REPO_ROOT.parents[1] / "keqing-data"
K0_70K_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_70K_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

EXT_MORTAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/ext_mortal/external_mortal_20240308_best_min.pth"
)
EXT_MORTAL_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

# Operational M0 control checkpoints from D1 prep
D1_PREP_ROOT = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07"
)
M0_CURRENT_CHECKPOINTS = {
    20260806: {
        "path": D1_PREP_ROOT / "M0_control/seed_20260806/checkpoints/mortal_72000.pth",
        "sha256": "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
    },
    20260807: {
        "path": D1_PREP_ROOT / "M0_control/seed_20260807/checkpoints/mortal_72000.pth",
        "sha256": "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
    },
    20260808: {
        "path": D1_PREP_ROOT / "M0_control/seed_20260808/checkpoints/mortal_72000.pth",
        "sha256": "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
    },
}

# D1_CQL_OFF checkpoints from C1 training completion closure
C1_TRAINING_ROOT = (
    REPO_ROOT
    / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/training_implementation_2026_08"
)
D1_CQL_OFF_CHECKPOINTS = {
    20260806: {
        "path": C1_TRAINING_ROOT / "D1_CQL_OFF/seed_20260806/checkpoints/mortal_72000.pth",
        "sha256": "99ee9985753dedd11453fab0a0e142793f8a94af13ce4bcc4526b9e28643ca95",
    },
    20260807: {
        "path": C1_TRAINING_ROOT / "D1_CQL_OFF/seed_20260807/checkpoints/mortal_72000.pth",
        "sha256": "57b9c7dff51e118595d83bb838492c74374b49c52edb9ffe8c4c991a112ca661",
    },
    20260808: {
        "path": C1_TRAINING_ROOT / "D1_CQL_OFF/seed_20260808/checkpoints/mortal_72000.pth",
        "sha256": "697f6ce021d2f4379a8943d464dcb387e7a7ed8ece70ddaba394e220b3563022",
    },
}

# Pre-designated canonical checkpoint for K1 promotion
CANONICAL_PROMOTION_CHECKPOINT = "D1_CQL_OFF_20260807"


def sha256_file(path: Path) -> str:
    """Compute sha256 hex digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoints() -> tuple[bool, dict[str, Any]]:
    """Verify all 8 unique checkpoint files exist and match exact SHA256."""
    records: dict[str, Any] = {}
    all_ok = True

    # 1. 70k
    p_70k = Path(K0_70K_PATH)
    if not p_70k.exists():
        all_ok = False
        records["70k"] = {"path": str(p_70k), "status": "missing"}
    else:
        sha = sha256_file(p_70k)
        match = (sha == K0_70K_SHA256)
        all_ok = all_ok and match
        records["70k"] = {"path": str(p_70k), "sha256": sha, "match": match}

    # 2. ext_mortal
    p_ext = Path(EXT_MORTAL_PATH)
    if not p_ext.exists():
        all_ok = False
        records["ext_mortal"] = {"path": str(p_ext), "status": "missing"}
    else:
        sha = sha256_file(p_ext)
        match = (sha == EXT_MORTAL_SHA256)
        all_ok = all_ok and match
        records["ext_mortal"] = {"path": str(p_ext), "sha256": sha, "match": match}

    # 3. M0_CURRENT (3 seeds)
    for s, meta in M0_CURRENT_CHECKPOINTS.items():
        p = Path(meta["path"])
        label = f"M0_CURRENT_{s}"
        if not p.exists():
            all_ok = False
            records[label] = {"path": str(p), "status": "missing"}
        else:
            sha = sha256_file(p)
            match = (sha == meta["sha256"])
            all_ok = all_ok and match
            records[label] = {"path": str(p), "sha256": sha, "match": match}

    # 4. D1_CQL_OFF (3 seeds)
    for s, meta in D1_CQL_OFF_CHECKPOINTS.items():
        p = Path(meta["path"])
        label = f"D1_CQL_OFF_{s}"
        if not p.exists():
            all_ok = False
            records[label] = {"path": str(p), "status": "missing"}
        else:
            sha = sha256_file(p)
            match = (sha == meta["sha256"])
            all_ok = all_ok and match
            records[label] = {"path": str(p), "sha256": sha, "match": match}

    return all_ok, records


def model_lineup_for_seed(training_seed: int) -> list[dict[str, Any]]:
    """Return the ordered 4-player lineup for given training seed."""
    return [
        {"label": "70k", "path": str(K0_70K_PATH), "sha256": K0_70K_SHA256},
        {"label": "ext_mortal", "path": str(EXT_MORTAL_PATH), "sha256": EXT_MORTAL_SHA256},
        {
            "label": f"M0_CURRENT_{training_seed}",
            "path": str(M0_CURRENT_CHECKPOINTS[training_seed]["path"]),
            "sha256": M0_CURRENT_CHECKPOINTS[training_seed]["sha256"],
        },
        {
            "label": f"D1_CQL_OFF_{training_seed}",
            "path": str(D1_CQL_OFF_CHECKPOINTS[training_seed]["path"]),
            "sha256": D1_CQL_OFF_CHECKPOINTS[training_seed]["sha256"],
        },
    ]


def equal_seed_hierarchical_bootstrap(
    x_by_seed: dict[int, np.ndarray],
    y_by_seed: dict[int, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Compute paired equal-seed hierarchical bootstrap CI95 with outer seed resampling and inner hanchan resampling.

    Protocol:
    1. For each of B reps:
       a. Sample 3 training seeds with replacement from {20260806, 20260807, 20260808}.
       b. For each sampled seed, sample 1000 paired rows with replacement from that seed's 1000 games.
       c. Compute mean(x) and mean(y) across the 3 sampled seed draws (each seed draw = mean of 1000 resampled games).
       d. x and y share identical outer and inner bootstrap sample draws.
    2. Compute percentile 95% confidence intervals from the bootstrap distributions.
    """
    rng = np.random.default_rng(seed)
    
    seeds_list = list(SEEDS)
    num_seeds = len(seeds_list)

    x_means = [float(np.mean(x_by_seed[s])) for s in SEEDS]
    y_means = [float(np.mean(y_by_seed[s])) for s in SEEDS]
    x_equal_mean = float(np.mean(x_means))
    y_equal_mean = float(np.mean(y_means))

    boot_x = np.empty(reps, dtype=np.float64)
    boot_y = np.empty(reps, dtype=np.float64)

    for b in range(reps):
        # Outer bootstrap: resample 3 seeds with replacement from {0, 1, 2}
        sampled_seed_indices = rng.integers(0, num_seeds, size=num_seeds)
        
        b_x_seeds = []
        b_y_seeds = []
        for s_idx in sampled_seed_indices:
            s = seeds_list[s_idx]
            n_games = len(x_by_seed[s])
            assert len(y_by_seed[s]) == n_games, f"Game count mismatch for seed {s}"
            
            # Inner bootstrap: resample 1000 games with replacement
            game_indices = rng.integers(0, n_games, size=n_games)
            b_x_seeds.append(np.mean(x_by_seed[s][game_indices]))
            b_y_seeds.append(np.mean(y_by_seed[s][game_indices]))
        
        boot_x[b] = np.mean(b_x_seeds)
        boot_y[b] = np.mean(b_y_seeds)

    alpha = (100.0 - BOOTSTRAP_CI) / 2.0
    x_ci = (float(np.percentile(boot_x, alpha)), float(np.percentile(boot_x, 100.0 - alpha)))
    y_ci = (float(np.percentile(boot_y, alpha)), float(np.percentile(boot_y, 100.0 - alpha)))

    return x_ci, y_ci, x_equal_mean, y_equal_mean


def adjudicate_c3_promotion(
    x_seed_means: dict[int, float],
    x_ci95: tuple[float, float],
    y_seed_means: dict[int, float],
    y_ci95: tuple[float, float],
    gates_pass: bool,
) -> dict[str, Any]:
    """Apply frozen C3 adjudication rules to determine promotion."""
    if not gates_pass:
        return {
            "verdict": "invalid",
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "promoted_k1_checkpoint": None,
            "reason": "Hard gates failed",
        }

    x_all_positive = all(x_seed_means[s] > 0.0 for s in SEEDS)
    x_ci_positive = x_ci95[0] > 0.0
    x_pass = x_all_positive and x_ci_positive

    y_all_positive = all(y_seed_means[s] > 0.0 for s in SEEDS)
    y_ci_positive = y_ci95[0] > 0.0
    y_pass = y_all_positive and y_ci_positive

    if x_pass and y_pass:
        return {
            "verdict": "promotion_supported",
            "recipe_promotion": True,
            "checkpoint_promotion": True,
            "promoted_k1_checkpoint": CANONICAL_PROMOTION_CHECKPOINT,
            "reason": "All x (vs 70k) and y (vs M0_CURRENT) criteria satisfied (3/3 seeds > 0 and CI95 lower > 0)",
        }
    else:
        reasons = []
        if not x_pass:
            reasons.append(f"x vs 70k failed (all_pos={x_all_positive}, ci_lower={x_ci95[0]:.4f})")
        if not y_pass:
            reasons.append(f"y vs M0_CURRENT failed (all_pos={y_all_positive}, ci_lower={y_ci95[0]:.4f})")
        return {
            "verdict": "not_supported",
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "promoted_k1_checkpoint": None,
            "reason": "; ".join(reasons),
        }
