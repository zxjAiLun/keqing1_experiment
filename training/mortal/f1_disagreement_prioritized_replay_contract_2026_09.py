"""Frozen contract, paths, and schemas for F1 M0 behavior-disagreement prioritized replay experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "F1_m0_behavior_disagreement_prioritized_replay_2026_09"
F1_ROOT = REPO_ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
F1_TRAINING_DIR = F1_ROOT / "training"
F1_EVAL_DIR = F1_ROOT / "evaluation"
F1_SUMMARY_DIR = F1_ROOT / "summary"

TRAINING_MANIFEST_SCHEMA = "keqing.mortal.f1_training_manifest.v1"
EVAL_MANIFEST_SCHEMA = "keqing.mortal.f1_eval_manifest.v1"
SUMMARY_SCHEMA = "keqing.mortal.f1_summary.v1"

R2_EXPERIMENT_ID = "R2_rank_plus_score_to_go_multiseed_confirmation_2026_09"
R2_ROOT = REPO_ROOT / "artifacts" / "experiments" / R2_EXPERIMENT_ID
R2_TRAINING_DIR = R2_ROOT / "training"

PARENT_MODEL = "K0_70k"
DATA_ROOT = Path("/media/bailan/DISK/AUbuntuProject/keqing-data")
K0_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/K0_70k/mortal_default_70k_promoted_candidate.pth"
)
K0_FALLBACK_PATH = REPO_ROOT / "artifacts" / "mortal_training" / "checkpoints" / "mortal_default_70k_promoted_candidate.pth"
K0_EXPECTED_SHA256 = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"

EXT_MORTAL_CANONICAL_PATH = (
    DATA_ROOT
    / "mortal/authoritative/D3_top2_discard_v1_2026_08/models/ext_mortal/external_mortal_20240308_best_min.pth"
)
EXT_MORTAL_FALLBACK_PATH = REPO_ROOT.parent / "keqing1/artifacts/external_mortal_20240308_best_min.pth"
EXT_MORTAL_EXPECTED_SHA256 = "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"

M0_DATA_INDEX_PATH = (
    REPO_ROOT.parent
    / "keqing1/artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/file_index_m0.pth"
)
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

UNIFORM_ROWS = 256
PRIORITY_ROWS = 256
MIN_HIGH_SIGNAL_ROWS = 16
Q_MARGIN_THRESHOLD = None

FROZEN_R2_CONTROL: dict[int, dict[str, str]] = {
    20260910: {
        "checkpoint_sha256": "fe7fecf86d8e99b107ee974ab73ad7f590d8986ce493b1165241dc33cbc9df92",
        "row_identity_sha256": "2e1e41ad31487fa19953d6e1cd1cd777c76d2229592ffdaf2a62943b7c30c013",
    },
    20260911: {
        "checkpoint_sha256": "8b3cad1a0a5ab43c529f7f8ce7c6c8c017f079f578e9ba3db28b4168ab9ed849",
        "row_identity_sha256": "e503dc4043b21ae05382172a9071b358dc504bd55713b8c40373a6c93dd569c9",
    },
    20260912: {
        "checkpoint_sha256": "cee3fd1e780f8d0b1bf13d550e90084802ec1cf22ca9ceffe38f003d92b1eeb2",
        "row_identity_sha256": "fdeb82219f8e0b7bd9e306949377f07e0f0e2424d505751fe1a648149f8b3fba",
    },
}

ROW_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("obs", "float32"),
    ("actions", "int64"),
    ("masks", "bool"),
    ("steps_to_done", "int64"),
    ("player_ranks", "int64"),
)

EVALUATION_LINEUP_TEMPLATE: tuple[str, ...] = (
    "K0_70k",
    "ext_mortal",
    "R2_Control_seed_{seed}",
    "F1_PriorityVariant_seed_{seed}",
)
EVAL_GAMES_PER_PANEL = 1000
EVAL_TOTAL_GAMES = 3000
EVAL_SHARDS_PER_PANEL = 4
EVAL_GAMES_PER_SHARD = 250
EVAL_SEED_START = 2500000
EVAL_SEED_END_EXCLUSIVE = 2501000
EVAL_SEED_KEY = 8192
TENHOU_RANK_POINTS = np.array([90.0, 45.0, 0.0, -135.0], dtype=np.float64)

LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<seed_key>\d+)_[^/]+\.json\.gz$")

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_CI = 95.0

EXPECTED_TRAINING_HARD_GATES: frozenset[str] = frozenset({
    "k0_parent_verified",
    "m0_dataset_verified",
    "r2_control_checkpoints_verified",
    "all_3_seeds_completed",
    "all_3_variant_checkpoints_saved",
    "all_seeds_base_row_identity_matches_r2_control",
    "high_signal_min_16_all_batches",
    "uniform_priority_split_256_256_all_batches",
    "scorer_parameters_bit_exact",
    "main_q_target_final_rank_mc_verified",
    "optimizer_preserved_k0_moments_410",
    "exact_step_counts_verified",
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

ALLOWED_Q_TARGET_VALUES = {3.0, 1.0, -1.0, -3.0}


class ContractError(RuntimeError):
    """Raised when any F1 contract invariant is breached."""


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


def resolve_r2_control_checkpoint(seed: int) -> tuple[Path, str, str]:
    """Resolve a frozen R2 Control checkpoint; returns (path, sha256, row_identity_sha256)."""
    if seed not in FROZEN_R2_CONTROL:
        raise ContractError(f"Seed {seed} is not part of the frozen R2 Control set")
    path = R2_TRAINING_DIR / f"mortal_control_70400_seed_{seed}.pth"
    if not path.exists():
        raise FileNotFoundError(f"R2 Control checkpoint not found at: {path}")
    actual_sha = sha256_file(path)
    expected_sha = FROZEN_R2_CONTROL[seed]["checkpoint_sha256"]
    if actual_sha != expected_sha:
        raise ContractError(
            f"R2 Control checkpoint SHA mismatch for seed {seed}: expected {expected_sha}, got {actual_sha}"
        )
    return path, actual_sha, FROZEN_R2_CONTROL[seed]["row_identity_sha256"]


def canonical_step_rng(experiment_id: str, training_seed: int, optimizer_step: int) -> np.random.Generator:
    """SHA-256-derived local RNG bound to (experiment ID, training seed, optimizer step).

    Never touches the dataloader or global random/np/torch RNG state.
    """
    material = f"{experiment_id}|{training_seed}|{optimizer_step}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big")
    return np.random.default_rng(seed_int)


def select_priority_batch(
    high_signal_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    batch_size: int = BATCH_SIZE,
    uniform_rows: int = UNIFORM_ROWS,
    priority_rows: int = PRIORITY_ROWS,
    min_high_signal: int = MIN_HIGH_SIGNAL_ROWS,
) -> dict[str, np.ndarray | int | float]:
    """Deterministic 256 uniform (without replacement) + 256 priority (with
    replacement from high_signal rows) selection followed by a final 512-row
    permutation. Fail-closed when the batch has fewer than 16 disagreement rows.
    """
    mask = np.asarray(high_signal_mask, dtype=bool)
    if mask.shape != (batch_size,):
        raise ContractError(f"high_signal_mask shape {mask.shape} != ({batch_size},)")
    hs_idx = np.flatnonzero(mask)
    hs_count = int(hs_idx.size)
    if hs_count < min_high_signal:
        raise ContractError(
            f"Batch has {hs_count} high-signal rows, below the fail-closed minimum of {min_high_signal}"
        )

    uniform_idx = rng.choice(batch_size, size=uniform_rows, replace=False)
    priority_idx = rng.choice(hs_idx, size=priority_rows, replace=True)
    concatenated = np.concatenate([uniform_idx, priority_idx])
    permutation = rng.permutation(batch_size)
    final_idx = concatenated[permutation]

    duplicates = priority_rows - int(np.unique(priority_idx).size)
    duplicate_rate = duplicates / priority_rows

    if int(np.unique(uniform_idx).size) != uniform_rows:
        raise ContractError("Uniform selection must be without replacement")
    if not np.isin(priority_idx, hs_idx).all():
        raise ContractError("Priority selection leaked a non-high-signal row")
    if final_idx.shape != (batch_size,):
        raise ContractError(f"Selected batch size {final_idx.shape[0]} != {batch_size}")
    if sorted(final_idx.tolist()) != sorted(concatenated.tolist()):
        raise ContractError("Final selection is not a permutation of uniform+priority rows")

    return {
        "uniform_idx": uniform_idx,
        "priority_idx": priority_idx,
        "final_idx": final_idx,
        "high_signal_count": hs_count,
        "high_signal_rate": hs_count / batch_size,
        "priority_duplicate_rate": duplicate_rate,
    }


def validate_f1_seed_set(seeds: list[int] | None) -> list[int]:
    """Fail-closed: F1 runners accept only the exact frozen seed set."""
    if seeds is None:
        return list(TRAINING_SEEDS)
    if list(seeds) != list(TRAINING_SEEDS):
        raise ContractError(
            f"F1 requires the exact frozen seed set {TRAINING_SEEDS}; got {seeds}"
        )
    return list(seeds)


def update_row_identity_digest(
    digest: hashlib._Hash,
    *,
    obs,
    actions,
    masks,
    steps_to_done,
    player_ranks,
) -> None:
    """Feed one batch's reward-excluded fields into the rolling row-identity SHA256 (R1/R2 parity)."""
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


def parse_game_identity(log_path: Path, lineup: tuple[str, ...] | None = None) -> dict:
    """Parse one game log and fail-closed on filename, seed, or lineup violations."""
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
    if lineup is not None and set(names) != set(lineup):
        raise ContractError(
            f"Lineup mismatch in {log_path.name}: got {sorted(names)}, expected {sorted(lineup)}"
        )

    return {"game_id": game_id, "names": names, "events": events}


def verify_training_manifest(tr_man: dict) -> bool:
    """Fail-closed validation of an F1 training manifest."""
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
    scorer = tr_man.get("frozen_scorer")
    if not isinstance(scorer, dict) or scorer.get("sha256") != k0_sha:
        raise ContractError(f"Training manifest frozen_scorer missing or SHA mismatch: {scorer}")
    if scorer.get("mode") != "eval" or scorer.get("amp") is not False or scorer.get("dtype") != "float32":
        raise ContractError(f"Training manifest frozen_scorer protocol mismatch: {scorer}")
    if scorer.get("parameters_bit_exact") is not True:
        raise ContractError("Training manifest frozen_scorer parameters_bit_exact must be true")

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
    main_reward = tr_man.get("main_reward")
    if not isinstance(main_reward, dict) or main_reward.get("mode") != MAIN_REWARD_MODE:
        raise ContractError(f"Training main Q target must be {MAIN_REWARD_MODE}: {main_reward}")

    pr = tr_man.get("prioritized_replay")
    if not isinstance(pr, dict):
        raise ContractError(f"Training manifest prioritized_replay missing or invalid: {pr}")
    if pr.get("uniform_rows") != UNIFORM_ROWS or pr.get("priority_rows") != PRIORITY_ROWS:
        raise ContractError(f"prioritized_replay split mismatch: {pr.get('uniform_rows')}/{pr.get('priority_rows')}")
    if pr.get("min_high_signal_rows") != MIN_HIGH_SIGNAL_ROWS:
        raise ContractError(f"prioritized_replay min_high_signal_rows mismatch: {pr.get('min_high_signal_rows')}")
    if pr.get("q_margin_threshold") is not None:
        raise ContractError("prioritized_replay must not set a Q-margin threshold")
    if pr.get("uniform_sampling") != "without_replacement":
        raise ContractError(f"uniform_sampling mismatch: {pr.get('uniform_sampling')}")
    if pr.get("priority_sampling") != "with_replacement":
        raise ContractError(f"priority_sampling mismatch: {pr.get('priority_sampling')}")
    if pr.get("sampler") != "sha256_canonical_local_rng":
        raise ContractError(f"sampler mismatch: {pr.get('sampler')}")
    if pr.get("uses_reward_or_eval_signal") is not False:
        raise ContractError("prioritized_replay must not use reward or evaluation signals")

    checkpoints = tr_man.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ContractError(f"Training manifest checkpoints missing or invalid: {checkpoints}")
    for s in TRAINING_SEEDS:
        key = f"seed_{s}"
        if key not in checkpoints:
            raise ContractError(f"Training manifest missing checkpoint seed {key}")
        entry = checkpoints[key]

        r2c = entry.get("r2_control")
        if not isinstance(r2c, dict):
            raise ContractError(f"Training manifest missing r2_control for {key}")
        if r2c.get("sha256") != FROZEN_R2_CONTROL[s]["checkpoint_sha256"]:
            raise ContractError(f"r2_control SHA mismatch for {key}: {r2c.get('sha256')}")

        var = entry.get("priority_variant")
        if not isinstance(var, dict):
            raise ContractError(f"Training manifest missing priority_variant for {key}")
        if not var.get("sha256") or len(var.get("sha256", "")) != 64:
            raise ContractError(f"priority_variant sha missing/invalid for {key}")
        if not var.get("path"):
            raise ContractError(f"priority_variant path missing for {key}")

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
        if seg.get("base_row_sha256") != FROZEN_R2_CONTROL[s]["row_identity_sha256"]:
            raise ContractError(
                f"Training row-identity base digest mismatch for {ks}: {seg.get('base_row_sha256')}"
            )
        if seg.get("matches_r2_control") is not True:
            raise ContractError(f"Training row-identity matches_r2_control flag not True for {ks}")
        selected = seg.get("selected_row_sha256")
        if not selected or len(selected) != 64:
            raise ContractError(f"Training selected-row digest missing/invalid for {ks}")
        stats = seg.get("selection_stats")
        if not isinstance(stats, dict):
            raise ContractError(f"Training selection_stats missing for {ks}")
        if int(stats.get("high_signal_count_min", -1)) < MIN_HIGH_SIGNAL_ROWS:
            raise ContractError(f"Training selection_stats high_signal_count_min below {MIN_HIGH_SIGNAL_ROWS} for {ks}")
        if int(stats.get("batches", -1)) != OPTIMIZER_STEPS:
            raise ContractError(f"Training selection_stats batches mismatch for {ks}")
        per_step = stats.get("per_step")
        if not isinstance(per_step, list) or len(per_step) != OPTIMIZER_STEPS:
            raise ContractError(f"Training per-step selection log missing/incomplete for {ks}")
        for rec in per_step:
            if int(rec.get("high_signal_count", -1)) < MIN_HIGH_SIGNAL_ROWS:
                raise ContractError(f"Training per-step high_signal_count below minimum in {ks}")
            if int(rec.get("uniform_rows", -1)) != UNIFORM_ROWS or int(rec.get("priority_rows", -1)) != PRIORITY_ROWS:
                raise ContractError(f"Training per-step split mismatch in {ks}")

    return True


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


def adjudicate_f1_verdict(
    primary_seed_means: list[float],
    primary_ci_lower: float,
    absolute_seed_means: list[float],
    absolute_ci_lower: float,
) -> tuple[str, bool, bool, str | None]:
    """Adjudicate the four-state F1 verdict and promotion status.

    The fourth state, `invalid`, is realized fail-closed: any data, sampling,
    training, log, or provenance gate failure raises ContractError upstream and
    no verdict is ever produced from invalid artifacts.
    """
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
        k1 = f"mortal_priority_variant_70400_seed_{CANONICAL_K1_SEED}.pth"
    elif primary_pass and not absolute_pass:
        verdict = "sampling_effect_only"
        recipe_promotion = False
        checkpoint_promotion = False
        k1 = None
    else:
        verdict = "not_supported"
        recipe_promotion = False
        checkpoint_promotion = False
        k1 = None

    return verdict, recipe_promotion, checkpoint_promotion, k1
