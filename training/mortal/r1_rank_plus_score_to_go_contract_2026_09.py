"""Frozen contract, paths, reward computation, and schema definitions for R1 pilot experiment."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "R1_rank_plus_score_to_go_pilot_2026_09"
R1_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
R1_TRAINING_DIR = R1_ROOT / "training"
R1_EVAL_DIR = R1_ROOT / "evaluation"
R1_SUMMARY_DIR = R1_ROOT / "summary"

# Schemas
TRAINING_MANIFEST_SCHEMA = "keqing.mortal.r1_training_manifest.v1"
EVAL_MANIFEST_SCHEMA = "keqing.mortal.r1_eval_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.r1_summary.v1"

# Parent & models
PARENT_MODEL = "K0_70k"
DATA_ROOT = Path("/media/bailan/DISK/AUbuntuProject/keqing-data")
K0_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO_ROOT / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

# External mortal model for evaluation
EXT_MORTAL_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/ext_mortal/external_mortal_20240308_best_min.pth"
)
EXT_MORTAL_FALLBACK_PATH = REPO_ROOT.parent / "keqing1/artifacts/external_mortal_20240308_best_min.pth"
EXT_MORTAL_EXPECTED_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

# M0 mixed replay corpus index
M0_DATA_INDEX_PATH = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/file_index_m0.pth"
)
M0_EXPECTED_SHA256 = "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2"

# Training hyperparams
TRAINING_SEED = 20260807
STEPS_START = 70000
STEPS_TARGET = 70400
OPTIMIZER_STEPS = 400
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
CQL_MIN_Q_WEIGHT = 5.0
AUX_WEIGHT = 0.2
GAMMA = 1.0
FILE_BATCH_SIZE = 15

# Reward parameters for R1: rank_target + 0.25 * score_to_go
RANK_PTS = [6.0, 4.0, 2.0, 0.0]
SCORE_TO_GO_WEIGHT = 0.25
SCORE_TO_GO_SCALE = 10000.0
SCORE_TO_GO_CLIP_MIN = -3.0
SCORE_TO_GO_CLIP_MAX = 3.0

# Tenhou rank points for evaluation
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

# Evaluation protocol
EVAL_TOTAL_GAMES = 1000
EVAL_SHARDS = 4
EVAL_GAMES_PER_SHARD = 250
EVAL_SEED_START = 2200000
EVAL_SEED_END_EXCLUSIVE = 2201000
EVAL_SEED_KEY = 8192
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_CI = 95.0

EXPECTED_TRAINING_HARD_GATES: frozenset[str] = frozenset({
    "k0_parent_verified",
    "m0_dataset_verified",
    "control_400_steps_completed",
    "variant_400_steps_completed",
    "identical_row_identity_verified",
    "control_checkpoint_saved",
    "variant_checkpoint_saved",
    "exact_step_counts_verified",
    "optimizer_preserved_adam_verified",
})

EXPECTED_EVAL_HARD_GATES: frozenset[str] = frozenset({
    "training_manifest_verified",
    "checkpoints_verified",
    "ext_mortal_verified",
    "all_4_shards_completed",
    "exact_1000_games_evaluated",
    "reach_accepted_semantics_enforced",
    "zero_missing_games",
})

EXPECTED_SUMMARY_HARD_GATES: frozenset[str] = frozenset({
    "training_manifest_verified",
    "eval_manifest_verified",
    "all_logs_verified",
    "paired_metrics_recalculated",
    "primary_contrast_computed",
    "secondary_contrast_computed",
    "bootstrap_computed",
})


class ContractError(RuntimeError):
    """Raised when any R1 contract invariant is breached."""


def native_path(raw: str | Path) -> Path:
    """Resolve frozen Windows paths from repo artifacts on the current OS."""
    text = str(raw)
    if os.name != "nt" and (re.match(r"^[A-Za-z]:", text) or "\\" in text or "AUbuntuProject" in text):
        norm = text.replace("\\", "/")
        parts = norm.split("/")
        repo_parts = REPO_ROOT.parts
        if "AUbuntuProject" in parts and "AUbuntuProject" in repo_parts:
            root_idx = repo_parts.index("AUbuntuProject")
            path_idx = parts.index("AUbuntuProject")
            return Path(*repo_parts[: root_idx + 1], *parts[path_idx + 1 :]).resolve()
    return Path(text).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_directory_empty_or_nonexistent(target_dir: Path) -> None:
    if target_dir.exists():
        entries = list(target_dir.iterdir())
        if len(entries) > 0:
            raise ContractError(
                f"Directory {target_dir} is not empty (contains {len(entries)} items). "
                "Overwriting non-empty directory is forbidden."
            )


def resolve_k0_checkpoint() -> tuple[Path, str]:
    target = K0_CANONICAL_PATH if K0_CANONICAL_PATH.exists() else K0_FALLBACK_PATH
    if not target.exists():
        raise FileNotFoundError(f"K0 checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"K0 SHA256 mismatch: expected {K0_EXPECTED_SHA256}, got {actual_sha}")
    return target, actual_sha


def resolve_ext_mortal_checkpoint() -> tuple[Path, str]:
    target = EXT_MORTAL_CANONICAL_PATH if EXT_MORTAL_CANONICAL_PATH.exists() else EXT_MORTAL_FALLBACK_PATH
    if not target.exists():
        raise FileNotFoundError(f"External Mortal checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != EXT_MORTAL_EXPECTED_SHA256:
        raise ContractError(f"External Mortal SHA256 mismatch: expected {EXT_MORTAL_EXPECTED_SHA256}, got {actual_sha}")
    return target, actual_sha


def resolve_m0_dataset_index() -> tuple[Path, str]:
    if not M0_DATA_INDEX_PATH.exists():
        raise FileNotFoundError(f"M0 dataset index not found at: {M0_DATA_INDEX_PATH}")
    actual_sha = sha256_file(M0_DATA_INDEX_PATH)
    if actual_sha != M0_EXPECTED_SHA256:
        raise ContractError(f"M0 dataset index SHA mismatch: expected {M0_EXPECTED_SHA256}, got {actual_sha}")
    return M0_DATA_INDEX_PATH, actual_sha


def compute_r1_target(
    final_rank: int,
    final_score: float,
    score_at_current_kyoku_start: float,
) -> float:
    """Compute R1 target = base_rank_target + 0.25 * clip((final_score - start_score)/10000, -3, +3)."""
    if not (0 <= final_rank < 4):
        raise ValueError(f"Invalid final_rank: {final_rank}")
    pts_arr = np.asarray(RANK_PTS, dtype=np.float64)
    base_rank_target = float(pts_arr[final_rank] - pts_arr.mean())
    score_diff = final_score - score_at_current_kyoku_start
    score_to_go = float(np.clip(score_diff / SCORE_TO_GO_SCALE, SCORE_TO_GO_CLIP_MIN, SCORE_TO_GO_CLIP_MAX))
    return base_rank_target + SCORE_TO_GO_WEIGHT * score_to_go


def adjudicate_r1_verdict(primary_mean: float, primary_ci_lower: float) -> str:
    """Adjudicate frozen pilot verdict according to contract."""
    if primary_mean > 0.0:
        if primary_ci_lower > 0.0:
            return "strong_positive"
        return "weak_positive"
    return "not_promising"


def paired_bootstrap_ci(
    diffs: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, list[float]]:
    """Compute mean and deterministic percentile bootstrap confidence interval for paired differences."""
    n = len(diffs)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array")
    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(diffs))
    indices = rng.integers(0, n, size=(reps, n))
    bootstrap_means = np.mean(diffs[indices], axis=1)
    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 - alpha))
    return mean_val, [ci_lower, ci_upper]
