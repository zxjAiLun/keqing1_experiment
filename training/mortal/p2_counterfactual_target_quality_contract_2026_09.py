"""Frozen contract, paths, and mathematical helpers for P2 counterfactual target quality evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "P2_seed_replay_counterfactual_target_quality_2026_09"
P2_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
P2_PANEL_DIR = P2_ROOT / "counterfactual_panel"
P2_SUMMARY_DIR = P2_ROOT / "summary"

# Parent & models
PARENT_MODEL = "K0_70k"
DATA_ROOT = REPO_ROOT.parents[1] / "keqing-data"
K0_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO_ROOT / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

# Schemas
PANEL_MANIFEST_SCHEMA = "keqing.mortal.p2_counterfactual_panel_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.p2_counterfactual_summary.v1"

# Contract configuration
PANEL_GAMES = 128
SEED_START = 3000000
SEED_END_EXCLUSIVE = SEED_START + PANEL_GAMES  # 3000128
SEED_KEY = 8192
FOCAL_SEAT = 0  # We evaluate focal engine on seat 0 (split 'a')
SPLIT_NAME = "a"
DISCARD_ACTION_LIMIT = 34

# Standard Tenhou rank points
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260904
BOOTSTRAP_CI = 95.0


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


def resolve_k0_checkpoint() -> tuple[Path, str]:
    target = K0_CANONICAL_PATH if K0_CANONICAL_PATH.exists() else K0_FALLBACK_PATH
    if not target.exists():
        raise FileNotFoundError(f"K0 checkpoint not found at: {target}")
    actual_sha = sha256_file(target)
    if actual_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"K0 SHA256 mismatch: expected {K0_EXPECTED_SHA256}, got {actual_sha}")
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
