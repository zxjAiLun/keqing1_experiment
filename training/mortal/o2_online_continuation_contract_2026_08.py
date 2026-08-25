"""Frozen contract, paths, and mathematical helpers for O2 online continuation pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "O2_keqing_online_continuation_pilot_2026_08"
O2_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
O2_TRAINING_DIR = O2_ROOT / "training"
O2_EVALUATION_DIR = O2_ROOT / "evaluation"

# Linchpin models & parent
PARENT_MODEL = "K0_70k"
DATA_ROOT = REPO_ROOT.parents[1] / "keqing-data"
K0_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO_ROOT / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

EXT_MORTAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/ext_mortal/external_mortal_20240308_best_min.pth"
)
EXT_MORTAL_FALLBACK_PATH = REPO_ROOT / "artifacts" / "external_mortal_20240308_best_min.pth"
EXT_MORTAL_EXPECTED_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

M0_20260807_PATH = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/M0_control/seed_20260807/checkpoints/mortal_72000.pth"
)
M0_20260807_EXPECTED_SHA256 = "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d"

# Training schedule & hyperparameters
TRAINING_SEED = 20260831
START_STEP = 70000
TARGET_STEP = 70400
TOTAL_OPTIMIZER_STEPS = 400
BATCH_SIZE = 512
TOTAL_CONSUMED_ROWS = 204800

NUM_CYCLES = 16
STEPS_PER_CYCLE = 25
ROWS_PER_CYCLE = 12800  # 25 * 512

INITIAL_SEED_GROUPS_PER_CYCLE = 32  # 32 * 4 = 128 hanchans
MAX_SEED_GROUPS_PER_CYCLE = 40      # 40 * 4 = 160 hanchans
SEEDS_PER_CYCLE_BLOCK = 40          # cycle i starts at 2000000 + i * 40
GENERATION_BASE_SEED = 2000000
SEED_KEY = 8192  # 0x2000

TRAINEE_EXPLORATION = {
    "boltzmann_epsilon": 0.005,
    "boltzmann_temp": 0.05,
    "top_p": 1.0,
}

# Project-owned Objective & Reward Contract
ADAPTER_KIND = "keqing_project_online"
OBJECTIVE_MODE = "behavior_action_mc"
REWARD_MODE = "final_rank_mc"
GAMMA = 1.0
RANK_PTS = np.array([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
CENTERED_TARGETS = {
    0: float(RANK_PTS[0] - RANK_PTS.mean()),  # Rank 1 -> +3.0
    1: float(RANK_PTS[1] - RANK_PTS.mean()),  # Rank 2 -> +1.0
    2: float(RANK_PTS[2] - RANK_PTS.mean()),  # Rank 3 -> -1.0
    3: float(RANK_PTS[3] - RANK_PTS.mean()),  # Rank 4 -> -3.0
}
AUX_WEIGHT = 0.2
BASE_MIN_Q_WEIGHT = 5.0
ONLINE_FLAG = True
FORCE_ONLINE_FLAG = False
LEARNING_RATE = 1e-4
FREEZE_BN = True
AMP = False

# Evaluation Configuration
EVALUATION_GAMES = 1000
EVALUATION_SEED_START = 2100000
EVALUATION_SEED_END_EXCLUSIVE = 2101000
EVALUATION_SHARDS = 4
EVALUATION_GAMES_PER_SHARD = 250
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_CI = 95.0
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

# Lineup for 4-player evaluation
EVALUATION_LINEUP = ["K0_70k", "ext_mortal", "M0_CURRENT_20260807", "O2_70400"]


class ContractError(RuntimeError):
    """Raised when any contract invariant is breached."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_directory_boundary(target_dir: Path, allowed_root: Path) -> None:
    """Ensure target_dir is strictly contained within allowed_root."""
    resolved_root = allowed_root.resolve()
    resolved_target = target_dir.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(
            f"Security boundary violation: '{resolved_target}' is outside root '{resolved_root}'"
        ) from exc


def ensure_clean_staging_dir(dir_path: Path, allowed_root: Path) -> None:
    """Ensure staging dir exists and is strictly empty. Fail closed if non-empty; never delete existing contents."""
    check_directory_boundary(dir_path, allowed_root)
    if dir_path.exists():
        entries = list(dir_path.iterdir())
        if entries:
            raise ContractError(
                f"Fail-closed check: staging directory already exists and is non-empty ({len(entries)} items): {dir_path}"
            )
    else:
        dir_path.mkdir(parents=True, exist_ok=False)


def compute_effective_cql_weight(
    *, online: bool = ONLINE_FLAG, force_online: bool = FORCE_ONLINE_FLAG, base_min_q_weight: float = BASE_MIN_Q_WEIGHT
) -> tuple[bool, float]:
    """Compute effective CQL activation and weight from runtime online flags."""
    cql_active = (not online) or force_online
    effective_weight = float(base_min_q_weight) if cql_active else 0.0
    return cql_active, effective_weight


def resolve_k0_checkpoint() -> tuple[Path, str]:
    target = K0_CANONICAL_PATH if K0_CANONICAL_PATH.exists() else K0_FALLBACK_PATH
    if not target.exists():
        raise FileNotFoundError(f"K0 checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"K0 SHA256 mismatch: expected {K0_EXPECTED_SHA256}, got {actual_sha}")
    return target, actual_sha


def resolve_ext_mortal_checkpoint() -> tuple[Path, str]:
    target = EXT_MORTAL_PATH if EXT_MORTAL_PATH.exists() else EXT_MORTAL_FALLBACK_PATH
    if not target.exists():
        raise FileNotFoundError(f"ext_mortal checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != EXT_MORTAL_EXPECTED_SHA256:
        raise ContractError(f"ext_mortal SHA256 mismatch: expected {EXT_MORTAL_EXPECTED_SHA256}, got {actual_sha}")
    return target, actual_sha


def resolve_m0_20260807_checkpoint() -> tuple[Path, str]:
    target = M0_20260807_PATH
    if not target.exists():
        raise FileNotFoundError(f"M0 20260807 checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != M0_20260807_EXPECTED_SHA256:
        raise ContractError(f"M0 20260807 SHA256 mismatch: expected {M0_20260807_EXPECTED_SHA256}, got {actual_sha}")
    return target, actual_sha


def final_scores_with_reach_accepted(events: list[dict[str, Any]]) -> list[float] | None:
    """Reconstruct final scores with corrected ReachAccepted semantics (-1000)."""
    scores: list[float] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(val) for val in values]
        elif event_type == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif event_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def compute_final_ranks(scores: list[float]) -> list[int]:
    """Rank players from 0 (1st) to 3 (4th) based on descending final scores (tie-breaking: seat order)."""
    indexed = [(score, -seat) for seat, score in enumerate(scores)]
    sorted_seats = [-seat for _, seat in sorted(indexed, reverse=True)]
    ranks = [0] * 4
    for r, seat in enumerate(sorted_seats):
        ranks[seat] = r
    return ranks


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


def adjudicate_o2_verdict(
    *,
    all_gates_pass: bool,
    mean_x: float,
    ci_x: list[float],
    mean_y: float,
    ci_y: list[float],
) -> str:
    """Four-state verdict mapping: invalid | strong_signal | promising | not_promising."""
    if not all_gates_pass:
        return "invalid"

    ci_x_lower = ci_x[0]
    ci_y_lower = ci_y[0]

    if mean_x > 0.0 and ci_x_lower > 0.0 and mean_y > 0.0 and ci_y_lower > 0.0:
        return "strong_signal"
    if mean_x > 0.0 and mean_y > 0.0:
        return "promising"
    return "not_promising"
