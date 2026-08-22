#!/usr/bin/env python3
"""Frozen M1 ext_mortal Dataset Expansion & Training/Evaluation Contract.

Experiment ID: M1_ext_mixed_expansion_2026_08

This module defines:
1. Dataset Contract:
   - Control: M0 authoritative file index (6,000 hanchans, ext_mortal perspective)
   - Variant: M1 = M0 (6,000) + D1 generation (6,000), both using ext_mortal perspective
   - Total: 12,000 unique hanchans, 12,000 trainable perspectives
   - Integrity verification: 0 hanchan overlap, 1 ext_mortal per game, legal actions, centered targets.
2. Training Contract:
   - Parent: K0_70k (mortal_default_70k_promoted_candidate.pth)
   - Steps: 70000 -> 72000 (2000 steps)
   - Seeds: 20260806, 20260807, 20260808
   - Preserved K0 Adam optimizer, fresh scaler/scheduler
   - Batch size 512, opt_step_every 1, AMP false, device cuda:0
   - Objective: behavior_action_mc, Reward: final_rank_mc
   - CQL min_q_weight: 5.0, next_rank_weight: 0.2
3. Evaluation Contract:
   - Lineup: 70k, ext_mortal, M0_CURRENT_<seed>, M1_CURRENT_<seed>
   - Spans:
     - 20260806: 1930000..1930999 (shards 0..3)
     - 20260807: 1940000..1940999 (shards 4..7)
     - 20260808: 1950000..1950999 (shards 8..11)
   - Total 3000 games across 12 shards (250 games/shard)
   - seed_key: 8192, random seats, rank points [+90, +45, 0, -135]
   - Statistics:
     x = Pt(M1) - Pt(M0_CURRENT)
     y = Pt(M1) - Pt(K0_70k)
   - Equal-seed hierarchical paired bootstrap (reps=5000, seed=20260830, outer seed + inner game resampling)
   - Promotion gate:
     x: 3/3 seeds > 0 AND CI95 lower > 0
     y: 3/3 seeds > 0 AND CI95 lower > 0
     If passed: K1 = M1_CURRENT_20260807
     Else: verdict = not_supported, K1 = null
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
M1_EXPERIMENT_ID = "M1_ext_mixed_expansion_2026_08"

SEEDS = (20260806, 20260807, 20260808)
START_STEP = 70000
TARGET_STEP = 72000
OPTIMIZER_STEPS = 2000
ARCHIVE_STEPS = (70001, 70010, 70100, 70500, 71000, 72000)

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
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_CI = 95.0

# Fresh hanchan ranges for M1 evaluation
SEED_HANCHAN_SPANS = {
    20260806: (1930000, 1930999),
    20260807: (1940000, 1940999),
    20260808: (1950000, 1950999),
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

M1_TRAINING_DIR = REPO_ROOT / "artifacts/experiments/M1_ext_mixed_expansion_2026_08/training_implementation_2026_08"
M1_CHECKPOINTS = {
    20260806: M1_TRAINING_DIR / "M1_variant/seed_20260806/checkpoints/mortal_72000.pth",
    20260807: M1_TRAINING_DIR / "M1_variant/seed_20260807/checkpoints/mortal_72000.pth",
    20260808: M1_TRAINING_DIR / "M1_variant/seed_20260808/checkpoints/mortal_72000.pth",
}

CANONICAL_PROMOTION_CHECKPOINT = "M1_CURRENT_20260807"

WINDOWS_PREFIX_MAP = [
    ("E:/AUbuntuProject/project/keqing1/", "/media/bailan/DISK/AUbuntuProject/project/keqing1/"),
    ("E:\\AUbuntuProject\\project\\keqing1\\", "/media/bailan/DISK/AUbuntuProject/project/keqing1/"),
    ("E:/AUbuntuProject/", "/media/bailan/DISK/AUbuntuProject/"),
    ("E:\\AUbuntuProject\\", "/media/bailan/DISK/AUbuntuProject/"),
]


def sha256_file(path: Path) -> str:
    """Compute sha256 hex digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def normalize_runtime_path(p: str | Path) -> Path:
    """Normalize Windows paths from legacy index files to Linux paths."""
    s = str(p)
    for win_prefix, linux_prefix in WINDOWS_PREFIX_MAP:
        if s.startswith(win_prefix):
            s = linux_prefix + s[len(win_prefix):].replace("\\", "/")
            break
    s = s.replace("\\", "/")
    return Path(s).resolve()


def load_file_index(pth_path: Path) -> list[Path]:
    """Load and normalize file index list from a PyTorch .pth file."""
    raw_list = torch.load(pth_path, weights_only=False)
    return [normalize_runtime_path(p) for p in raw_list]


def build_m1_dataset_files(
    output_dir: Path,
    m0_index_path: Path = D1_PREP_ROOT / "file_index_m0.pth",
    d1_index_path: Path = D1_PREP_ROOT / "file_index_d1.pth",
    expected_m0_count: int = 6000,
    expected_d1_count: int = 6000,
) -> tuple[Path, Path]:
    """Build M1 concatenated 12000 file index and train labels file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    m1_index_path = output_dir / "file_index_m1.pth"
    m1_labels_path = output_dir / "m1_train_labels.txt"

    m0_files = load_file_index(m0_index_path)
    d1_files = load_file_index(d1_index_path)

    if len(m0_files) != expected_m0_count:
        raise ValueError(f"M0 file count is {len(m0_files)}, expected {expected_m0_count}")
    if len(d1_files) != expected_d1_count:
        raise ValueError(f"D1 file count is {len(d1_files)}, expected {expected_d1_count}")

    m1_files = [str(p) for p in m0_files] + [str(p) for p in d1_files]
    total_expected = expected_m0_count + expected_d1_count
    if len(m1_files) != total_expected:
        raise ValueError(f"M1 total file count is {len(m1_files)}, expected {total_expected}")

    # Save file_index_m1.pth
    torch.save(m1_files, m1_index_path)

    # Save m1_train_labels.txt: "ext_mortal" for all files
    with open(m1_labels_path, "w", encoding="utf-8") as f:
        for _ in range(total_expected):
            f.write("ext_mortal\n")

    return m1_index_path, m1_labels_path


def validate_m1_dataset_integrity(
    m1_index_path: Path,
) -> dict[str, Any]:
    """Verify integrity of M1 12,000 files: zero overlap, exactly 1 ext_mortal per game, all files exist."""
    files = load_file_index(m1_index_path)
    if len(files) != 12000:
        raise ValueError(f"Expected 12000 files in M1 index, found {len(files)}")

    m0_slice = files[:6000]
    d1_slice = files[6000:]

    m0_seeds: set[int] = set()
    d1_seeds: set[int] = set()

    for idx, p in enumerate(files):
        if not p.exists():
            raise FileNotFoundError(f"File missing at index {idx}: {p}")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            ev0 = json.loads(f.readline())
        if ev0.get("type") != "start_game":
            raise ValueError(f"Event 0 is not start_game in {p.name}")
        names = ev0.get("names", [])
        if names.count("ext_mortal") != 1:
            raise ValueError(f"Expected exactly 1 ext_mortal in {p.name}, found {names.count('ext_mortal')}")
        
        seed_id = int(ev0["seed"][0])
        if idx < 6000:
            m0_seeds.add(seed_id)
        else:
            d1_seeds.add(seed_id)

    if len(m0_seeds) != 6000:
        raise ValueError(f"Expected 6000 unique M0 seeds, found {len(m0_seeds)}")
    if len(d1_seeds) != 6000:
        raise ValueError(f"Expected 6000 unique D1 seeds, found {len(d1_seeds)}")

    overlap = m0_seeds.intersection(d1_seeds)
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping seeds between M0 and D1 parts!")

    return {
        "total_files": 12000,
        "m0_files": 6000,
        "d1_files": 6000,
        "m0_unique_seeds": len(m0_seeds),
        "d1_unique_seeds": len(d1_seeds),
        "seed_overlap": len(overlap),
        "all_files_exist": True,
        "all_single_ext_mortal": True,
    }


def generate_m1_training_config(
    seed: int,
    output_run_dir: Path,
    m1_index_path: Path,
    m1_labels_path: Path,
) -> dict[str, Any]:
    """Generate exact Mortal training configuration dictionary for M1."""
    checkpoints_dir = output_run_dir / "checkpoints"
    tb_dir = output_run_dir / "tb_mortal"
    state_file = output_run_dir / "mortal.pth"
    best_state_file = output_run_dir / "mortal_best.pth"

    return {
        "control": {
            "version": 4,
            "state_file": str(state_file.resolve()),
            "best_state_file": str(best_state_file.resolve()),
            "tensorboard_dir": str(tb_dir.resolve()),
            "device": "cuda:0",
            "enable_amp": False,
            "batch_size": 512,
            "opt_step_every": 1,
            "save_every": 400,
        },
        "dataset": {
            "globs": [],
            "file_index": str(m1_index_path.resolve()),
            "file_batch_size": 15,
            "reserve_ratio": 0.0,
            "num_workers": 0,
            "player_names_files": [str(m1_labels_path.resolve())],
            "num_epochs": 3,
            "enable_augmentation": False,
            "augmented_first": False,
        },
        "env": {
            "gamma": 1.0,
            "pts": [6.0, 4.0, 2.0, 0.0],
        },
        "reward": {
            "mode": "final_rank_mc",
        },
        "resnet": {
            "conv_channels": 192,
            "num_blocks": 40,
        },
        "cql": {
            "min_q_weight": 5.0,
        },
        "aux": {
            "next_rank_weight": 0.2,
        },
        "freeze_bn": {
            "mortal": False,
        },
        "optim": {
            "eps": 1e-08,
            "betas": [0.9, 0.999],
            "weight_decay": 0.1,
            "max_grad_norm": 0.0,
            "scheduler": {
                "peak": 0.0001,
                "final": 0.0001,
                "warm_up_steps": 0,
                "max_steps": 0,
            },
        },
        "objective": {
            "mode": "behavior_action_mc",
        },
        "experiment": {
            "route": "M1_variant",
            "trainable_label": "ext_mortal",
            "training_seed": seed,
            "parent_steps": START_STEP,
            "reward_mode": "final_rank_mc",
        },
    }


def validate_checkpoints() -> tuple[bool, dict[str, Any]]:
    """Verify all required baseline checkpoints exist and match expected SHA256."""
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

    return all_ok, records


def model_lineup_for_seed(training_seed: int) -> list[dict[str, Any]]:
    """Return the ordered 4-player lineup for given training seed."""
    m1_path = M1_CHECKPOINTS[training_seed]
    return [
        {"label": "70k", "path": str(K0_70K_PATH), "sha256": K0_70K_SHA256},
        {"label": "ext_mortal", "path": str(EXT_MORTAL_PATH), "sha256": EXT_MORTAL_SHA256},
        {
            "label": f"M0_CURRENT_{training_seed}",
            "path": str(M0_CURRENT_CHECKPOINTS[training_seed]["path"]),
            "sha256": M0_CURRENT_CHECKPOINTS[training_seed]["sha256"],
        },
        {
            "label": f"M1_CURRENT_{training_seed}",
            "path": str(m1_path),
            "sha256": sha256_file(m1_path) if m1_path.exists() else None,
        },
    ]


def equal_seed_hierarchical_bootstrap(
    x_by_seed: dict[int, np.ndarray],
    y_by_seed: dict[int, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Compute paired equal-seed hierarchical bootstrap CI95 with outer seed resampling and inner hanchan resampling."""
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
        sampled_seed_indices = rng.integers(0, num_seeds, size=num_seeds)
        
        b_x_seeds = []
        b_y_seeds = []
        for s_idx in sampled_seed_indices:
            s = seeds_list[s_idx]
            n_games = len(x_by_seed[s])
            assert len(y_by_seed[s]) == n_games, f"Game count mismatch for seed {s}"
            
            game_indices = rng.integers(0, n_games, size=n_games)
            b_x_seeds.append(np.mean(x_by_seed[s][game_indices]))
            b_y_seeds.append(np.mean(y_by_seed[s][game_indices]))
        
        boot_x[b] = np.mean(b_x_seeds)
        boot_y[b] = np.mean(b_y_seeds)

    alpha = (100.0 - BOOTSTRAP_CI) / 2.0
    x_ci = (float(np.percentile(boot_x, alpha)), float(np.percentile(boot_x, 100.0 - alpha)))
    y_ci = (float(np.percentile(boot_y, alpha)), float(np.percentile(boot_y, 100.0 - alpha)))

    return x_ci, y_ci, x_equal_mean, y_equal_mean


def adjudicate_m1_promotion(
    x_seed_means: dict[int, float],
    x_ci95: tuple[float, float],
    y_seed_means: dict[int, float],
    y_ci95: tuple[float, float],
    gates_pass: bool,
) -> dict[str, Any]:
    """Apply frozen M1 adjudication rules to determine promotion."""
    if not gates_pass:
        return {
            "verdict": "invalid",
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "promoted_k1_checkpoint": None,
            "reason": "Hard gates failed",
        }

    # x = Pt(M1) - Pt(M0_CURRENT)
    x_all_positive = all(x_seed_means[s] > 0.0 for s in SEEDS)
    x_ci_positive = x_ci95[0] > 0.0
    x_pass = x_all_positive and x_ci_positive

    # y = Pt(M1) - Pt(K0_70k)
    y_all_positive = all(y_seed_means[s] > 0.0 for s in SEEDS)
    y_ci_positive = y_ci95[0] > 0.0
    y_pass = y_all_positive and y_ci_positive

    if x_pass and y_pass:
        return {
            "verdict": "promotion_supported",
            "recipe_promotion": True,
            "checkpoint_promotion": True,
            "promoted_k1_checkpoint": CANONICAL_PROMOTION_CHECKPOINT,
            "reason": "All x (vs M0_CURRENT) and y (vs 70k) criteria satisfied (3/3 seeds > 0 and CI95 lower > 0)",
        }
    else:
        reasons = []
        if not x_pass:
            reasons.append(f"x vs M0_CURRENT failed (all_pos={x_all_positive}, ci_lower={x_ci95[0]:.4f})")
        if not y_pass:
            reasons.append(f"y vs 70k failed (all_pos={y_all_positive}, ci_lower={y_ci95[0]:.4f})")
        return {
            "verdict": "not_supported",
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "promoted_k1_checkpoint": None,
            "reason": "; ".join(reasons),
        }
