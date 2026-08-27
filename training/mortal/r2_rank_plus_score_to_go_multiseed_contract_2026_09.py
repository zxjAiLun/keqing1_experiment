"""Frozen contract, paths, reward computation, and schema definitions for R2 multi-seed confirmation experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "R2_rank_plus_score_to_go_multiseed_confirmation_2026_09"
R2_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
R2_TRAINING_DIR = R2_ROOT / "training"
R2_EVAL_DIR = R2_ROOT / "evaluation"
R2_SUMMARY_DIR = R2_ROOT / "summary"

# Schemas
TRAINING_MANIFEST_SCHEMA = "keqing.mortal.r2_training_manifest.v1"
EVAL_MANIFEST_SCHEMA = "keqing.mortal.r2_eval_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.r2_summary.v1"

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

# Fresh 3 training seeds for R2
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

# Reward parameters for R2: rank_target + 0.25 * score_to_go
RANK_PTS = [6.0, 4.0, 2.0, 0.0]
SCORE_TO_GO_WEIGHT = 0.25
SCORE_TO_GO_SCALE = 10000.0
SCORE_TO_GO_CLIP_MIN = -3.0
SCORE_TO_GO_CLIP_MAX = 3.0

# R2 is reward-only: identical objective/trainable protocol to R1/M0.
OBJECTIVE_MODE = "behavior_action_mc"
OBJECTIVE_VALUE_STATISTIC = "behavior_action_q"
TRAINABLE_PLAYER_NAMES: tuple[str, ...] = ("ext_mortal",)

ROW_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("obs", "float32"),
    ("actions", "int64"),
    ("masks", "bool"),
    ("steps_to_done", "int64"),
    ("player_ranks", "int64"),
)

EVALUATION_LINEUP: tuple[str, ...] = ("K0_70k", "ext_mortal", "Control_70400", "Variant_70400")
LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<seed_key>\d+)_[^/]+\.json\.gz$")

# Tenhou rank points for evaluation
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

# Common-random-number evaluation protocol
EVAL_GAMES_PER_PANEL = 1000
EVAL_TOTAL_GAMES = 3000
EVAL_SHARDS_PER_PANEL = 4
EVAL_GAMES_PER_SHARD = 250
EVAL_SEED_START = 2300000
EVAL_SEED_END_EXCLUSIVE = 2301000
EVAL_SEED_KEY = 8192
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_CI = 95.0

EXPECTED_TRAINING_HARD_GATES: frozenset[str] = frozenset({
    "k0_parent_verified",
    "m0_dataset_verified",
    "all_3_training_seeds_completed",
    "all_6_checkpoints_saved",
    "all_seeds_identical_row_identity_verified",
    "exact_step_counts_verified",
    "optimizer_preserved_adam_verified",
})

EXPECTED_EVAL_HARD_GATES: frozenset[str] = frozenset({
    "training_manifest_verified",
    "all_checkpoints_verified",
    "ext_mortal_verified",
    "all_3_panels_completed",
    "exact_3000_games_evaluated",
    "reach_accepted_semantics_enforced",
    "zero_missing_games",
})

EXPECTED_SUMMARY_HARD_GATES: frozenset[str] = frozenset({
    "training_manifest_verified",
    "eval_manifest_verified",
    "all_3000_logs_verified",
    "paired_metrics_recalculated",
    "crossed_bootstrap_computed",
    "primary_contrast_evaluated",
    "absolute_contrast_evaluated",
})


class ContractError(RuntimeError):
    """Raised when any R2 contract invariant is breached."""


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


def reward_contract_for_condition(condition: str) -> dict:
    """Return the frozen reward-parameter record for one training condition."""
    if condition == "control":
        return {
            "mode": "final_rank_mc",
            "rank_pts": [float(v) for v in RANK_PTS],
        }
    if condition == "variant":
        return {
            "mode": "rank_plus_score_to_go_mc",
            "rank_pts": [float(v) for v in RANK_PTS],
            "score_to_go_weight": SCORE_TO_GO_WEIGHT,
            "score_to_go_scale": SCORE_TO_GO_SCALE,
            "score_to_go_clip_min": SCORE_TO_GO_CLIP_MIN,
            "score_to_go_clip_max": SCORE_TO_GO_CLIP_MAX,
        }
    raise ValueError(f"Unknown condition: {condition}")


def update_row_identity_digest(
    digest: hashlib._Hash,
    *,
    obs,
    actions,
    masks,
    steps_to_done,
    player_ranks,
) -> None:
    """Feed one batch's reward-excluded fields into the rolling row-identity SHA256 (R1 parity)."""
    tensors = {
        "obs": obs,
        "actions": actions,
        "masks": masks,
        "steps_to_done": steps_to_done,
        "player_ranks": player_ranks,
    }
    for name, dtype_str in ROW_IDENTITY_FIELDS:
        arr = tensors[name].detach().cpu().numpy().astype(np.dtype(dtype_str))
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())


def parse_game_identity(log_path: Path) -> dict:
    """Parse one game log and fail-closed on filename, seed, or lineup violations (R1 parity)."""
    m = LOG_NAME_RE.match(log_path.name)
    if not m:
        raise ContractError(f"Invalid game log filename: {log_path.name}")
    file_seed = int(m.group("seed"))
    file_seed_key = int(m.group("seed_key"))
    if file_seed_key != EVAL_SEED_KEY:
        raise ContractError(
            f"Seed key mismatch in {log_path.name}: file={file_seed_key}, expected={EVAL_SEED_KEY}"
        )

    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events or events[0].get("type") != "start_game":
        raise ContractError(f"Log {log_path.name} does not start with start_game")

    seed_tuple = events[0].get("seed")
    if not isinstance(seed_tuple, (list, tuple)) or len(seed_tuple) != 2:
        raise ContractError(f"Invalid start_game seed tuple in {log_path.name}: {seed_tuple}")
    try:
        log_seed, log_key = int(seed_tuple[0]), int(seed_tuple[1])
    except Exception as exc:
        raise ContractError(f"Invalid start_game seed tuple in {log_path.name}: {seed_tuple}") from exc
    if log_seed != file_seed:
        raise ContractError(
            f"Seed mismatch in {log_path.name}: filename={file_seed}, start_game={log_seed}"
        )
    if log_key != EVAL_SEED_KEY:
        raise ContractError(
            f"Seed key mismatch in {log_path.name}: start_game={log_key}, expected={EVAL_SEED_KEY}"
        )
    game_id = log_seed

    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ContractError(f"Invalid names array in {log_path.name}: {names}")
    if set(names) != set(EVALUATION_LINEUP):
        raise ContractError(
            f"Lineup mismatch in {log_path.name}: got {sorted(names)}, expected {sorted(EVALUATION_LINEUP)}"
        )

    return {"game_id": game_id, "names": names, "events": events}


def verify_training_manifest(tr_man: dict) -> bool:
    """Fail-closed validation of a R2 training manifest (R1 parity extended to 3 seeds)."""
    if tr_man.get("schema") != TRAINING_MANIFEST_SCHEMA:
        raise ContractError(f"Training manifest schema mismatch: {tr_man.get('schema')}")
    if tr_man.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"Training manifest experiment_id mismatch: {tr_man.get('experiment_id')}")
    if tr_man.get("verdict") != "training_completed":
        raise ContractError(f"Training manifest verdict is not training_completed: {tr_man.get('verdict')}")

    gates = tr_man.get("hard_gates", {})
    if set(gates.keys()) != set(EXPECTED_TRAINING_HARD_GATES):
        raise ContractError(f"Training manifest hard gate set mismatch: {sorted(gates.keys())}")
    if not all(gates.values()):
        raise ContractError(f"Training manifest hard gates not all passed: {gates}")

    _, k0_sha = resolve_k0_checkpoint()
    if tr_man.get("parent_model", {}).get("sha256") != k0_sha:
        raise ContractError(
            f"Training manifest parent K0 SHA mismatch: manifest={tr_man.get('parent_model', {}).get('sha256')}, canonical={k0_sha}"
        )
    dataset_info = tr_man.get("dataset")
    if not isinstance(dataset_info, dict) or not dataset_info.get("sha256"):
        raise ContractError(f"Training manifest dataset missing or invalid: {dataset_info}")
    _, m0_sha = resolve_m0_dataset_index()
    if dataset_info.get("sha256") != m0_sha:
        raise ContractError(
            f"Training manifest M0 dataset SHA mismatch: manifest={dataset_info.get('sha256')}, canonical={m0_sha}"
        )

    objective = tr_man.get("objective")
    if not isinstance(objective, dict):
        raise ContractError(f"Training manifest objective missing or invalid: {objective}")
    if objective.get("mode") != OBJECTIVE_MODE:
        raise ContractError(f"Training objective is not {OBJECTIVE_MODE}: {objective.get('mode')}")
    if objective.get("value_statistic") != OBJECTIVE_VALUE_STATISTIC:
        raise ContractError(
            f"Training objective value_statistic is not {OBJECTIVE_VALUE_STATISTIC}: {objective.get('value_statistic')}"
        )
    trainable = tr_man.get("trainable_player_names")
    if trainable != list(TRAINABLE_PLAYER_NAMES):
        raise ContractError(f"Training trainable labels mismatch: {trainable}")

    checkpoints = tr_man.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ContractError(f"Training manifest checkpoints missing or invalid: {checkpoints}")
    for s in TRAINING_SEEDS:
        key = f"seed_{s}"
        if key not in checkpoints:
            raise ContractError(f"Training manifest missing checkpoint seed {key}")
        entry = checkpoints[key]
        for condition in ("control", "variant"):
            if condition not in entry:
                raise ContractError(f"Training manifest missing {key}/{condition}")
            recorded = entry[condition].get("reward")
            if not isinstance(recorded, dict):
                raise ContractError(f"Training manifest reward missing for {key}/{condition}: {recorded}")
            expected = reward_contract_for_condition(condition)
            if recorded != expected:
                raise ContractError(f"Training reward contract mismatch for {key}/{condition}: {recorded} vs {expected}")
            if not entry[condition].get("sha256") or len(entry[condition].get("sha256", "")) != 64:
                raise ContractError(f"Training manifest checkpoint sha missing/invalid for {key}/{condition}")
            if not entry[condition].get("path"):
                raise ContractError(f"Training manifest checkpoint path missing for {key}/{condition}")

    row_identity = tr_man.get("row_identity")
    if not isinstance(row_identity, dict):
        raise ContractError(f"Training manifest row_identity missing or invalid: {row_identity}")
    if row_identity.get("fields") != [name for name, _ in ROW_IDENTITY_FIELDS]:
        raise ContractError(f"Training row-identity fields mismatch: {row_identity.get('fields')}")
    if row_identity.get("excluded_field") != "kyoku_rewards":
        raise ContractError(f"Training row-identity excluded_field mismatch: {row_identity.get('excluded_field')}")
    by_seed = row_identity.get("by_seed")
    if not isinstance(by_seed, dict):
        raise ContractError(f"Training row_identity by_seed missing or invalid: {by_seed}")
    for s in TRAINING_SEEDS:
        ks = f"seed_{s}"
        if ks not in by_seed:
            raise ContractError(f"Training row-identity missing seed {ks}")
        seg = by_seed[ks]
        if not seg.get("control_sha256") or not seg.get("variant_sha256"):
            raise ContractError(f"Training row-identity digests missing for {ks}: {seg}")
        if seg.get("control_sha256") != seg.get("variant_sha256"):
            raise ContractError(f"Training row-identity digests differ for {ks}: {seg}")
        if seg.get("identical") is not True:
            raise ContractError(f"Training row-identity identical flag not True for {ks}: {seg.get('identical')}")
        if len(seg.get("control_sha256", "")) != 64 or len(seg.get("variant_sha256", "")) != 64:
            raise ContractError(f"Training row-identity SHA length invalid for {ks}")

    return True


def compute_r2_target(
    final_rank: int,
    final_score: float,
    score_at_current_kyoku_start: float,
) -> float:
    """Compute R2 target = base_rank_target + 0.25 * clip((final_score - start_score)/10000, -3, +3)."""
    if not (0 <= final_rank < 4):
        raise ValueError(f"Invalid final_rank: {final_rank}")
    pts_arr = np.asarray(RANK_PTS, dtype=np.float64)
    base_rank_target = float(pts_arr[final_rank] - pts_arr.mean())
    score_diff = final_score - score_at_current_kyoku_start
    score_to_go = float(np.clip(score_diff / SCORE_TO_GO_SCALE, SCORE_TO_GO_CLIP_MIN, SCORE_TO_GO_CLIP_MAX))
    return base_rank_target + SCORE_TO_GO_WEIGHT * score_to_go


def crossed_bootstrap_ci(
    matrix_3x1000: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
    return_sampled_indices: bool = False,
    shared_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, list[float], tuple[np.ndarray, np.ndarray] | None]:
    """Compute crossed bootstrap CI across training-seed axis (3) and shared game-ID axis (1000)."""
    if matrix_3x1000.shape != (3, 1000):
        raise ValueError(f"Expected matrix of shape (3, 1000), got {matrix_3x1000.shape}")

    grand_mean = float(np.mean(matrix_3x1000))

    if shared_indices is not None:
        seed_idx_mat, game_idx_mat = shared_indices
    else:
        rng = np.random.default_rng(seed)
        seed_idx_mat = rng.integers(0, 3, size=(reps, 3))
        game_idx_mat = rng.integers(0, 1000, size=(reps, 1000))

    bootstrap_means = np.zeros(reps, dtype=np.float64)
    for r in range(reps):
        sub_matrix = matrix_3x1000[seed_idx_mat[r], :]
        sampled = sub_matrix[:, game_idx_mat[r]]
        bootstrap_means[r] = np.mean(sampled)

    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 - alpha))

    sampled_indices_out = (seed_idx_mat, game_idx_mat) if return_sampled_indices else None
    return grand_mean, [ci_lower, ci_upper], sampled_indices_out


def adjudicate_r2_verdict(
    primary_seed_means: list[float],
    primary_ci_lower: float,
    absolute_seed_means: list[float],
    absolute_ci_lower: float,
) -> tuple[str, bool, bool, str | None]:
    """Adjudicate formal R2 confirmation verdict and promotion status."""
    primary_pass = (
        len(primary_seed_means) == 3
        and all(m > 0 for m in primary_seed_means)
        and primary_ci_lower > 0
    )
    absolute_pass = (
        len(absolute_seed_means) == 3
        and all(m > 0 for m in absolute_seed_means)
        and absolute_ci_lower > 0
    )

    if primary_pass and absolute_pass:
        verdict = "promotion_supported"
        recipe_promotion = True
        checkpoint_promotion = True
        k1 = f"mortal_variant_70400_seed_{CANONICAL_K1_SEED}.pth"
    elif primary_pass and not absolute_pass:
        verdict = "reward_effect_only"
        recipe_promotion = False
        checkpoint_promotion = False
        k1 = None
    else:
        verdict = "not_supported"
        recipe_promotion = False
        checkpoint_promotion = False
        k1 = None

    return verdict, recipe_promotion, checkpoint_promotion, k1
