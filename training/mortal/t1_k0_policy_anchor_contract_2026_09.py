"""Frozen contract and pure helpers for the T1 K0 policy-anchor pilot."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "T1_k0_policy_anchor_continuation_pilot_2026_09"
T1_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
T1_CALIBRATION_DIR = T1_ROOT / "calibration"
T1_CALIBRATION_PATH = T1_CALIBRATION_DIR / "t1_lambda_calibration.json"
T1_TRAINING_DIR = T1_ROOT / "training"
T1_EVAL_DIR = T1_ROOT / "evaluation"
T1_SUMMARY_DIR = T1_ROOT / "summary"

CALIBRATION_SCHEMA = "keqing.mortal.t1_lambda_calibration.v1"
TRAINING_MANIFEST_SCHEMA = "keqing.mortal.t1_training_manifest.v1"
EVAL_MANIFEST_SCHEMA = "keqing.mortal.t1_eval_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.t1_summary.v1"

R2_EXPERIMENT_ID = "R2_rank_plus_score_to_go_multiseed_confirmation_2026_09"
R2_ROOT = REPO_ROOT / "artifacts" / "experiments" / R2_EXPERIMENT_ID
R2_TRAINING_DIR = R2_ROOT / "training"

PARENT_MODEL = "K0_70k"
DATA_ROOT = Path("/media/bailan/DISK/AUbuntuProject/keqing-data")
K0_CANONICAL_PATH = DATA_ROOT / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
K0_FALLBACK_PATH = REPO_ROOT / "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

EXT_MORTAL_CANONICAL_PATH = DATA_ROOT / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/ext_mortal/external_mortal_20240308_best_min.pth"
EXT_MORTAL_FALLBACK_PATH = REPO_ROOT.parent / "keqing1/artifacts/external_mortal_20240308_best_min.pth"
EXT_MORTAL_EXPECTED_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

M0_DATA_INDEX_PATH = REPO_ROOT.parent / "keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/file_index_m0.pth"
M0_EXPECTED_SHA256 = "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2"

TRAINING_SEEDS = [20260910, 20260911, 20260912]
CANONICAL_K1_SEED = 20260911
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

RANK_PTS = [6.0, 4.0, 2.0, 0.0]
OBJECTIVE_MODE = "behavior_action_mc"
OBJECTIVE_VALUE_STATISTIC = "behavior_action_q"
TRAINABLE_PLAYER_NAMES: tuple[str, ...] = ("ext_mortal",)
MAIN_REWARD_MODE = "final_rank_mc"
ALLOWED_Q_TARGET_VALUES = {3.0, 1.0, -1.0, -3.0}

ANCHOR_LOSS = "legal_action_kl"
ANCHOR_DIRECTION = "KL(P_K0||P_current)"
ANCHOR_TEMPERATURE = 1.0
TARGET_GRADIENT_RATIO = 0.10
CALIBRATION_BATCHES_PER_SEED = 1
MECHANISM_AUDIT_SKIP_BATCHES = OPTIMIZER_STEPS
MECHANISM_AUDIT_BATCHES = 16
MECHANISM_AUDIT_ROWS = BATCH_SIZE * MECHANISM_AUDIT_BATCHES

FROZEN_R2_CONTROL: dict[int, dict[str, str]] = {
    20260910: {"checkpoint_sha256": "fe7fecf86d8e99b107ee974ab73ad7f590d8986ce493b1165241dc33cbc9df92", "row_identity_sha256": "2e1e41ad31487fa19953d6e1cd1cd777c76d2229592ffdaf2a62943b7c30c013"},
    20260911: {"checkpoint_sha256": "8b3cad1a0a5ab43c529f7f8ce7c6c8c017f079f578e9ba3db28b4168ab9ed849", "row_identity_sha256": "e503dc4043b21ae05382172a9071b358dc504bd55713b8c40373a6c93dd569c9"},
    20260912: {"checkpoint_sha256": "cee3fd1e780f8d0b1bf13d550e90084802ec1cf22ca9ceffe38f003d92b1eeb2", "row_identity_sha256": "fdeb82219f8e0b7bd9e306949377f07e0f0e2424d505751fe1a648149f8b3fba"},
}

ROW_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("obs", "float32"), ("actions", "int64"), ("masks", "bool"),
    ("steps_to_done", "int64"), ("player_ranks", "int64"),
)

EVALUATION_LINEUP_TEMPLATE: tuple[str, ...] = (
    "K0_70k", "ext_mortal", "R2_Control_seed_{seed}", "T1_AnchorVariant_seed_{seed}",
)
EVAL_GAMES_PER_PANEL = 1000
EVAL_TOTAL_GAMES = 3000
EVAL_SHARDS_PER_PANEL = 4
EVAL_GAMES_PER_SHARD = 250
EVAL_SEED_START = 2800000
EVAL_SEED_END_EXCLUSIVE = 2801000
EVAL_SEED_KEY = 8192
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)
LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<seed_key>\d+)_[^/]+\.json\.gz$")

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_CI = 95.0

EXPECTED_TRAINING_HARD_GATES = frozenset({
    "k0_parent_verified", "m0_dataset_verified", "r2_control_checkpoints_verified",
    "calibration_verified", "lambda_matches_calibration", "all_3_seeds_completed",
    "all_3_variant_checkpoints_saved", "all_seeds_base_row_identity_matches_r2_control",
    "each_base_row_used_exactly_once", "anchor_finite_all_steps",
    "student_anchor_forward_eval_mode", "scorer_parameters_bit_exact",
    "main_q_target_final_rank_mc_verified", "optimizer_preserved_k0_moments_410",
    "exact_step_counts_verified", "mechanism_audit_completed",
})
EXPECTED_EVAL_HARD_GATES = frozenset({
    "training_manifest_verified", "all_checkpoints_verified", "ext_mortal_verified",
    "all_3_panels_completed", "exact_3000_games_evaluated",
    "reach_accepted_semantics_enforced", "zero_missing_games",
})
EXPECTED_SUMMARY_HARD_GATES = frozenset({
    "training_manifest_verified", "eval_manifest_verified", "all_3000_logs_verified",
    "paired_metrics_recalculated", "crossed_bootstrap_computed",
    "primary_contrast_evaluated", "absolute_contrast_evaluated",
    "mechanism_audit_evaluated",
})


class ContractError(RuntimeError):
    """Raised when a T1 invariant is breached."""


def native_path(raw: str | Path) -> Path:
    text = str(raw)
    if os.name != "nt" and (re.match(r"^[A-Za-z]:", text) or "\\" in text or "AUbuntuProject" in text):
        parts = text.replace("\\", "/").split("/")
        repo_parts = REPO_ROOT.parts
        if "AUbuntuProject" in parts and "AUbuntuProject" in repo_parts:
            return Path(*repo_parts[: repo_parts.index("AUbuntuProject") + 1], *parts[parts.index("AUbuntuProject") + 1 :]).resolve()
    return Path(text).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_directory_empty_or_nonexistent(target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise ContractError(f"Directory {target_dir} is not empty; overwrite is forbidden")


def _resolve_hashed(primary: Path, fallback: Path | None, expected: str, label: str) -> tuple[Path, str]:
    target = primary if primary.exists() else fallback
    if target is None or not target.exists():
        raise FileNotFoundError(f"{label} not found at: {target}")
    actual = sha256_file(target)
    if actual != expected:
        raise ContractError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return target, actual


def resolve_k0_checkpoint() -> tuple[Path, str]:
    return _resolve_hashed(K0_CANONICAL_PATH, K0_FALLBACK_PATH, K0_EXPECTED_SHA256, "K0 checkpoint")


def resolve_ext_mortal_checkpoint() -> tuple[Path, str]:
    return _resolve_hashed(EXT_MORTAL_CANONICAL_PATH, EXT_MORTAL_FALLBACK_PATH, EXT_MORTAL_EXPECTED_SHA256, "external Mortal checkpoint")


def resolve_m0_dataset_index() -> tuple[Path, str]:
    return _resolve_hashed(M0_DATA_INDEX_PATH, None, M0_EXPECTED_SHA256, "M0 dataset index")


def resolve_r2_control_checkpoint(seed: int) -> tuple[Path, str, str]:
    if seed not in FROZEN_R2_CONTROL:
        raise ContractError(f"Seed {seed} is not in the frozen R2 Control set")
    path = R2_TRAINING_DIR / f"mortal_control_70400_seed_{seed}.pth"
    if not path.exists():
        raise FileNotFoundError(f"R2 Control checkpoint not found at: {path}")
    actual = sha256_file(path)
    expected = FROZEN_R2_CONTROL[seed]["checkpoint_sha256"]
    if actual != expected:
        raise ContractError(f"R2 Control SHA mismatch for seed {seed}: {actual} != {expected}")
    return path, actual, FROZEN_R2_CONTROL[seed]["row_identity_sha256"]


def validate_t1_seed_set(seeds: list[int] | None) -> list[int]:
    if seeds is None:
        return list(TRAINING_SEEDS)
    if list(seeds) != TRAINING_SEEDS:
        raise ContractError(f"T1 requires exact seeds {TRAINING_SEEDS}; got {seeds}")
    return list(seeds)


def update_row_identity_digest(digest: Any, *, obs, actions, masks, steps_to_done, player_ranks) -> None:
    tensors = {"obs": obs, "actions": actions, "masks": masks, "steps_to_done": steps_to_done, "player_ranks": player_ranks}
    for name, dtype_str in ROW_IDENTITY_FIELDS:
        arr = tensors[name].detach().cpu().numpy().astype(np.dtype(dtype_str))
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())


def legal_centered_q(q: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if q.shape != masks.shape or masks.dtype != torch.bool:
        raise ContractError(f"Q/mask mismatch: q={tuple(q.shape)} mask={tuple(masks.shape)} dtype={masks.dtype}")
    if not bool(masks.any(dim=1).all()):
        raise ContractError("Every row must contain at least one legal action")
    if not bool(torch.isfinite(q[masks]).all()):
        raise ContractError("Legal Q values must be finite")
    count = masks.sum(dim=1).to(q.dtype)
    mean = q.masked_fill(~masks, 0.0).sum(dim=1) / count
    return (q - mean.unsqueeze(1)).masked_fill(~masks, -1.0e9)


def legal_policy_kl_rows(q_current: torch.Tensor, q_parent: torch.Tensor, masks: torch.Tensor, temperature: float = ANCHOR_TEMPERATURE) -> torch.Tensor:
    """KL(P_K0 || P_current) over legal centered Q values, one value per row."""
    if q_current.shape != q_parent.shape:
        raise ContractError(f"Current/parent Q shape mismatch: {q_current.shape} vs {q_parent.shape}")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ContractError(f"temperature must be positive and finite, got {temperature}")
    current_logits = legal_centered_q(q_current, masks) / float(temperature)
    parent_logits = legal_centered_q(q_parent, masks) / float(temperature)
    log_current = torch.log_softmax(current_logits, dim=1)
    log_parent = torch.log_softmax(parent_logits, dim=1)
    parent_prob = torch.softmax(parent_logits, dim=1)
    rows = (parent_prob * (log_parent - log_current)).sum(dim=1)
    if not bool(torch.isfinite(rows).all()):
        raise ContractError("Policy-anchor KL produced non-finite values")
    return rows.clamp_min(0.0)


def parse_game_identity(log_path: Path, lineup: tuple[str, ...] | None = None) -> dict[str, Any]:
    m = LOG_NAME_RE.match(log_path.name)
    if not m:
        raise ContractError(f"Invalid game log filename: {log_path.name}")
    file_seed, file_key = int(m.group("seed")), int(m.group("seed_key"))
    if file_key != EVAL_SEED_KEY:
        raise ContractError(f"Seed key mismatch in {log_path.name}")
    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events or events[0].get("type") != "start_game":
        raise ContractError(f"Log {log_path.name} does not start with start_game")
    seed_tuple = events[0].get("seed")
    if not isinstance(seed_tuple, (list, tuple)) or len(seed_tuple) != 2:
        raise ContractError(f"Invalid start_game seed tuple in {log_path.name}: {seed_tuple}")
    if int(seed_tuple[0]) != file_seed or int(seed_tuple[1]) != EVAL_SEED_KEY:
        raise ContractError(f"start_game seed mismatch in {log_path.name}: {seed_tuple}")
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ContractError(f"Invalid names array in {log_path.name}: {names}")
    if lineup is not None and set(names) != set(lineup):
        raise ContractError(f"Lineup mismatch in {log_path.name}: {sorted(names)} != {sorted(lineup)}")
    return {"game_id": file_seed, "names": names, "events": events}


def verify_calibration(cal: dict[str, Any]) -> float:
    if cal.get("schema") != CALIBRATION_SCHEMA or cal.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("Calibration schema or experiment ID mismatch")
    if cal.get("verdict") != "calibration_completed" or not all(cal.get("hard_gates", {}).values()):
        raise ContractError("Calibration is not complete with all gates passing")
    protocol = cal.get("protocol", {})
    if protocol.get("anchor_loss") != ANCHOR_LOSS or protocol.get("direction") != ANCHOR_DIRECTION:
        raise ContractError("Calibration anchor definition mismatch")
    if protocol.get("temperature") != ANCHOR_TEMPERATURE or protocol.get("target_gradient_ratio") != TARGET_GRADIENT_RATIO:
        raise ContractError("Calibration temperature or target ratio mismatch")
    if protocol.get("uses_reward_or_evaluation_signal") is not False:
        raise ContractError("Calibration must not use reward or evaluation signal")
    selected = float(cal.get("selected_lambda", float("nan")))
    if not math.isfinite(selected) or selected <= 0:
        raise ContractError(f"Invalid selected lambda: {selected}")
    by_seed = cal.get("by_seed", {})
    if set(by_seed) != {f"seed_{s}" for s in TRAINING_SEEDS}:
        raise ContractError("Calibration seed set mismatch")
    for seed in TRAINING_SEEDS:
        row = by_seed[f"seed_{seed}"]
        if row.get("r2_control_sha256") != FROZEN_R2_CONTROL[seed]["checkpoint_sha256"]:
            raise ContractError(f"Calibration R2 SHA mismatch for seed {seed}")
        for key in ("base_gradient_norm", "anchor_gradient_norm", "lambda_for_target_ratio"):
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0:
                raise ContractError(f"Calibration {key} invalid for seed {seed}: {value}")
    return selected


def verify_training_manifest(tr_man: dict[str, Any]) -> bool:
    if tr_man.get("schema") != TRAINING_MANIFEST_SCHEMA or tr_man.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("Training manifest schema or experiment ID mismatch")
    if tr_man.get("verdict") != "training_completed":
        raise ContractError(f"Training manifest verdict is {tr_man.get('verdict')}")
    gates = tr_man.get("hard_gates", {})
    if set(gates) != set(EXPECTED_TRAINING_HARD_GATES) or not all(gates.values()):
        raise ContractError(f"Training hard gates invalid: {gates}")
    _, k0_sha = resolve_k0_checkpoint()
    _, m0_sha = resolve_m0_dataset_index()
    if tr_man.get("parent_model", {}).get("sha256") != k0_sha or tr_man.get("frozen_scorer", {}).get("sha256") != k0_sha:
        raise ContractError("Training parent/scorer SHA mismatch")
    scorer = tr_man["frozen_scorer"]
    if scorer.get("mode") != "eval" or scorer.get("dtype") != "float32" or scorer.get("amp") is not False or scorer.get("parameters_bit_exact") is not True:
        raise ContractError("Frozen scorer protocol mismatch")
    if tr_man.get("dataset", {}).get("sha256") != m0_sha:
        raise ContractError("Training M0 dataset SHA mismatch")
    cal_ref = tr_man.get("calibration", {})
    cal_path = Path(cal_ref.get("path", ""))
    if not cal_path.exists() or sha256_file(cal_path) != cal_ref.get("sha256"):
        raise ContractError("Training calibration path/SHA binding mismatch")
    selected = verify_calibration(json.loads(cal_path.read_text(encoding="utf-8")))
    anchor = tr_man.get("policy_anchor", {})
    if anchor.get("loss") != ANCHOR_LOSS or anchor.get("direction") != ANCHOR_DIRECTION or anchor.get("temperature") != ANCHOR_TEMPERATURE:
        raise ContractError("Training policy-anchor definition mismatch")
    if float(anchor.get("lambda", float("nan"))) != selected:
        raise ContractError("Training policy-anchor lambda does not match calibration")
    if anchor.get("uses_behavior_disagreement") is not False or anchor.get("uses_reward_or_eval_signal") is not False:
        raise ContractError("T1 must not use disagreement, reward, or evaluation signals")
    if anchor.get("row_usage") != "each_row_exactly_once" or anchor.get("student_forward_mode") != "eval":
        raise ContractError("T1 row usage or student anchor-forward mode mismatch")
    if tr_man.get("objective", {}).get("mode") != OBJECTIVE_MODE or tr_man.get("main_reward", {}).get("mode") != MAIN_REWARD_MODE:
        raise ContractError("Training objective/reward mismatch")
    checkpoints = tr_man.get("checkpoints", {})
    rows = tr_man.get("row_identity", {}).get("by_seed", {})
    for seed in TRAINING_SEEDS:
        key = f"seed_{seed}"
        if checkpoints.get(key, {}).get("r2_control", {}).get("sha256") != FROZEN_R2_CONTROL[seed]["checkpoint_sha256"]:
            raise ContractError(f"R2 checkpoint binding mismatch for {seed}")
        variant = checkpoints.get(key, {}).get("anchor_variant", {})
        if len(variant.get("sha256", "")) != 64 or not variant.get("path"):
            raise ContractError(f"Variant checkpoint missing for {seed}")
        row = rows.get(key, {})
        if row.get("base_row_sha256") != FROZEN_R2_CONTROL[seed]["row_identity_sha256"] or row.get("matches_r2_control") is not True:
            raise ContractError(f"Row identity mismatch for {seed}")
        stats = row.get("anchor_stats", {})
        if stats.get("batches") != OPTIMIZER_STEPS or len(stats.get("per_step", [])) != OPTIMIZER_STEPS:
            raise ContractError(f"Anchor step log incomplete for {seed}")
        if any(rec.get("rows_used") != BATCH_SIZE or not math.isfinite(float(rec.get("anchor_kl", float("nan")))) for rec in stats["per_step"]):
            raise ContractError(f"Anchor step log invalid for {seed}")
    mechanism = tr_man.get("mechanism_audit", {})
    if mechanism.get("rows_per_seed") != MECHANISM_AUDIT_ROWS or set(mechanism.get("by_seed", {})) != {f"seed_{s}" for s in TRAINING_SEEDS}:
        raise ContractError("Mechanism audit missing or incomplete")
    return True


def crossed_bootstrap_ci(matrix_3x1000: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED, ci: float = BOOTSTRAP_CI, return_sampled_indices: bool = False, shared_indices: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, list[float], tuple[np.ndarray, np.ndarray] | None]:
    if matrix_3x1000.shape != (3, 1000):
        raise ValueError(f"Expected (3, 1000), got {matrix_3x1000.shape}")
    if shared_indices is None:
        rng = np.random.default_rng(seed)
        seed_idx = rng.integers(0, 3, size=(reps, 3))
        game_idx = rng.integers(0, 1000, size=(reps, 1000))
    else:
        seed_idx, game_idx = shared_indices
    values = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        values[i] = matrix_3x1000[seed_idx[i], :][:, game_idx[i]].mean()
    alpha = (100.0 - ci) / 2.0
    sampled = (seed_idx, game_idx) if return_sampled_indices else None
    return float(matrix_3x1000.mean()), [float(np.percentile(values, alpha)), float(np.percentile(values, 100.0 - alpha))], sampled


def adjudicate_t1_verdict(*, mechanism_pass: bool, primary_seed_means: list[float], primary_ci_lower: float, absolute_seed_means: list[float], absolute_ci_lower: float) -> tuple[str, bool, bool, None]:
    primary_pass = len(primary_seed_means) == 3 and all(v > 0 for v in primary_seed_means) and primary_ci_lower > 0
    absolute_pass = len(absolute_seed_means) == 3 and all(v > 0 for v in absolute_seed_means) and absolute_ci_lower > 0
    if not mechanism_pass:
        verdict = "mechanism_not_supported"
    elif primary_pass and absolute_pass:
        verdict = "anchor_promising"
    elif primary_pass:
        verdict = "stability_only"
    else:
        verdict = "not_supported"
    return verdict, False, False, None
