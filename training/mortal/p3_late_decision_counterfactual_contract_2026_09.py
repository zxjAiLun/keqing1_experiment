"""Frozen contract, paths, and mathematical helpers for P3 late-decision counterfactual signal density evaluation."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "P3_late_decision_counterfactual_signal_density_2026_09"
P3_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
P3_PANEL_DIR = P3_ROOT / "counterfactual_panel"
P3_SUMMARY_DIR = P3_ROOT / "summary"

# Parent & models
PARENT_MODEL = "K0_70k"
DATA_ROOT = REPO_ROOT.parents[1] / "keqing-data"
K0_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO_ROOT / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

# P2 predecessor artifacts for comparative density calculation
P2_EXPERIMENT_ID = "P2_seed_replay_counterfactual_target_quality_2026_09"
P2_SUMMARY_SCHEMA = "keqing.mortal.p2_counterfactual_summary.v1"
P2_SUMMARY_PATH = (
    REPO_ROOT
    / "artifacts"
    / "experiments"
    / P2_EXPERIMENT_ID
    / "summary"
    / "p2_summary.json"
)
P2_SUMMARY_EXPECTED_SHA256 = "014726b14a3dd1e9f40fc0db8cc98becb00cf2788febc9e9677adc5bbdbb3f4a"

# Schemas
PANEL_MANIFEST_SCHEMA = "keqing.mortal.p3_counterfactual_panel_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.p3_counterfactual_summary.v1"

# Contract configuration
PANEL_GAMES = 128
SEED_START = 3100000
SEED_END_EXCLUSIVE = SEED_START + PANEL_GAMES  # 3100128
SEED_KEY = 8192
FOCAL_SEAT = 0  # Evaluate focal engine on seat 0 (split 'a')
SPLIT_NAME = "a"
DISCARD_ACTION_LIMIT = 34

# Discard action to base tile mapping (0..33)
ACTION_TO_TILE: list[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]

# Standard Tenhou rank points
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_CI = 95.0

# Exact hard gates expected in panel manifest
EXPECTED_PANEL_HARD_GATES: frozenset[str] = frozenset({
    "k0_parent_verified",
    "exact_128_pairs_generated",
    "seeds_strictly_contiguous",
    "focal_seat_verified",
    "all_target_contexts_intervened_exactly_once",
    "all_prefixes_exact_matched",
    "all_first_divergences_verified_dahai",
    "all_branches_completed_end_game",
    "scores_and_ranks_valid",
})

# Exact hard gates expected in summary
EXPECTED_SUMMARY_HARD_GATES: frozenset[str] = frozenset({
    "manifest_verified",
    "k0_parent_verified",
    "exact_128_pairs_analyzed",
    "seeds_contiguous",
    "all_branch_logs_verified",
    "canonical_content_hashes_verified",
    "independent_metrics_recalculated_match",
    "p2_comparison_verified",
    "bootstrap_computed",
})


class ContractError(RuntimeError):
    """Raised when any contract invariant is breached."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_event_for_canonical_hash(ev: Any) -> Any:
    """Strip non-deterministic wall-clock timing fields from event metadata before hashing."""
    if not isinstance(ev, dict):
        return ev
    ev_copy = dict(ev)
    if "meta" in ev_copy and isinstance(ev_copy["meta"], dict):
        m_copy = dict(ev_copy["meta"])
        m_copy.pop("eval_time_ns", None)
        ev_copy["meta"] = m_copy
    return ev_copy


def canonical_log_content_sha256(path: Path) -> str:
    """Compute deterministic SHA256 of decompressed canonical JSONL events, excluding wall-clock nanoseconds."""
    import json
    with gzip.open(path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    normalized = [normalize_event_for_canonical_hash(e) for e in events]
    text = "\n".join(json.dumps(e, sort_keys=True) for e in normalized)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def action_matches_pai(action_id: int, pai_str: str) -> bool:
    """Verify that a dahai event pai string matches the corresponding discrete discard action ID."""
    if not (0 <= action_id < len(ACTION_TO_TILE)):
        return False
    expected_tile = ACTION_TO_TILE[action_id]
    if pai_str == expected_tile:
        return True
    if expected_tile == "5m" and pai_str == "5mr":
        return True
    if expected_tile == "5p" and pai_str == "5pr":
        return True
    return bool(expected_tile == "5s" and pai_str == "5sr")


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
    """Compute mean and deterministic percentile bootstrap confidence interval for a single series."""
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


def two_sample_rate_diff_bootstrap_ci(
    binary_a: np.ndarray,
    binary_b: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, list[float]]:
    """Compute difference in rates (mean(a) - mean(b)) and two-sample bootstrap confidence interval."""
    na = len(binary_a)
    nb = len(binary_b)
    if na == 0 or nb == 0:
        raise ValueError("Arrays cannot be empty")
    rng = np.random.default_rng(seed)
    rate_a = float(np.mean(binary_a))
    rate_b = float(np.mean(binary_b))
    diff_rate = rate_a - rate_b

    indices_a = rng.integers(0, na, size=(reps, na))
    indices_b = rng.integers(0, nb, size=(reps, nb))

    boot_a = np.mean(binary_a[indices_a], axis=1)
    boot_b = np.mean(binary_b[indices_b], axis=1)
    boot_diffs = boot_a - boot_b

    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha))
    ci_upper = float(np.percentile(boot_diffs, 100.0 - alpha))
    return diff_rate, [ci_lower, ci_upper]


def adjudicate_p3_verdict(
    p3_score_nonzero_rate: float,
    diff_score_nonzero_rate_ci: list[float],
) -> str:
    """Adjudicate P3 late-decision density outcome."""
    if p3_score_nonzero_rate >= 0.40 and diff_score_nonzero_rate_ci[0] > 0.0:
        return "late_decision_targets_promising"
    return "counterfactual_targets_insufficiently_dense"
