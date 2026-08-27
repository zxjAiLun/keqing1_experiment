"""Statistical summary and adjudication for R1 pilot experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.r1_rank_plus_score_to_go_contract_2026_09 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_START,
    EVAL_SHARDS,
    EVAL_TOTAL_GAMES,
    EVALUATION_LINEUP,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    EXT_MORTAL_EXPECTED_SHA256,
    R1_EVAL_DIR,
    R1_SUMMARY_DIR,
    R1_TRAINING_DIR,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    ContractError,
    adjudicate_r1_verdict,
    check_directory_empty_or_nonexistent,
    paired_bootstrap_ci,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    sha256_file,
    verify_training_manifest,
)

logger = logging.getLogger("r1_summary")


def _scores_and_ranks_from_events(events: list[dict[str, Any]], log_name: str) -> tuple[list[float], list[int]]:
    """Reconstruct final scores with exact reach_accepted semantics, then ranks."""
    scores: list[float] | None = None

    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "start_kyoku" and isinstance(ev.get("scores"), list):
            values = ev["scores"]
            if len(values) == 4:
                scores = [float(v) for v in values]
        elif ev_type == "reach_accepted" and scores is not None:
            actor = ev.get("actor")
            if actor is not None and 0 <= int(actor) < 4:
                scores[int(actor)] -= 1000.0
        elif ev_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = ev.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [s + float(d) for s, d in zip(scores, deltas, strict=True)]

    if scores is None:
        raise ContractError(f"Failed to reconstruct scores for {log_name}")

    # Ranks (tie-breaking: seat order)
    indexed = [(score, -seat) for seat, score in enumerate(scores)]
    sorted_seats = [-seat for _, seat in sorted(indexed, reverse=True)]
    ranks = [0] * 4
    for r, seat in enumerate(sorted_seats):
        ranks[seat] = r

    return scores, ranks


def _verify_eval_manifest(
    ev_man: dict[str, Any],
    *,
    tr_man: dict[str, Any],
    tr_man_sha: str,
    ctrl_disk_sha: str,
    var_disk_sha: str,
) -> None:
    """Fail-closed validation of the eval manifest and its SHA chain bindings."""
    if ev_man.get("schema") != EVAL_MANIFEST_SCHEMA:
        raise ContractError(f"Eval manifest schema mismatch: {ev_man.get('schema')}")
    if ev_man.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"Eval manifest experiment_id mismatch: {ev_man.get('experiment_id')}")
    if ev_man.get("verdict") != "evaluation_completed":
        raise ContractError(f"Eval manifest verdict is not evaluation_completed: {ev_man.get('verdict')}")

    gates = ev_man.get("hard_gates", {})
    if set(gates.keys()) != set(EXPECTED_EVAL_HARD_GATES):
        raise ContractError(f"Eval manifest hard gate set mismatch: {sorted(gates.keys())}")
    if not all(gates.values()):
        raise ContractError(f"Eval manifest hard gates not all passed: {gates}")

    # Eval manifest must bind the exact training manifest it consumed.
    bound = ev_man.get("training_manifest", {})
    if bound.get("sha256") != tr_man_sha:
        raise ContractError(
            f"Eval manifest training-manifest SHA mismatch: bound={bound.get('sha256')}, disk={tr_man_sha}"
        )

    if ev_man.get("lineup") != list(EVALUATION_LINEUP):
        raise ContractError(f"Eval manifest lineup mismatch: {ev_man.get('lineup')}")
    if ev_man.get("game_id_range") != [EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE]:
        raise ContractError(f"Eval manifest game_id_range mismatch: {ev_man.get('game_id_range')}")
    if ev_man.get("games_count") != EVAL_TOTAL_GAMES:
        raise ContractError(f"Eval manifest games_count mismatch: {ev_man.get('games_count')}")

    # Three-way checkpoint SHA chain: eval manifest vs training manifest vs disk.
    models = ev_man.get("models", {})
    tr_checkpoints = tr_man.get("checkpoints", {})
    for key, condition, disk_sha in (("control", "control", ctrl_disk_sha), ("variant", "variant", var_disk_sha)):
        ev_sha = models.get(key, {}).get("sha256")
        tr_sha = tr_checkpoints.get(condition, {}).get("sha256")
        if not (ev_sha == tr_sha == disk_sha):
            raise ContractError(
                f"Three-way checkpoint SHA mismatch for {condition}: "
                f"eval_manifest={ev_sha}, training_manifest={tr_sha}, disk={disk_sha}"
            )


def adjudicate_r1_pilot(
    training_dir: Path = R1_TRAINING_DIR,
    eval_dir: Path = R1_EVAL_DIR,
    summary_dir: Path = R1_SUMMARY_DIR,
) -> dict[str, Any]:
    """Load manifests, verify logs, compute primary and secondary contrasts, and generate summary."""
    check_directory_empty_or_nonexistent(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "r1_training_manifest.json"
    ev_man_path = eval_dir / "r1_eval_manifest.json"

    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    if not ev_man_path.exists():
        raise FileNotFoundError(f"Eval manifest not found at {ev_man_path}")

    tr_man_sha = sha256_file(tr_man_path)
    ev_man_sha = sha256_file(ev_man_path)

    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    ev_man = json.loads(ev_man_path.read_text(encoding="utf-8"))

    # Fail-closed manifest chain: exact gate sets, objective/label/reward fields,
    # row-identity digests, and SHA bindings across both manifests.
    tr_manifest_ok = verify_training_manifest(tr_man)

    _, k0_sha = resolve_k0_checkpoint()
    _, ext_sha = resolve_ext_mortal_checkpoint()
    if tr_man.get("parent_model", {}).get("sha256") != k0_sha:
        raise ContractError(
            f"Training manifest parent SHA mismatch: manifest={tr_man.get('parent_model', {}).get('sha256')}, canonical={k0_sha}"
        )

    ctrl_path = training_dir / "mortal_control_70400.pth"
    var_path = training_dir / "mortal_variant_70400.pth"
    if not ctrl_path.exists():
        raise FileNotFoundError(f"Control checkpoint not found at {ctrl_path}")
    if not var_path.exists():
        raise FileNotFoundError(f"Variant checkpoint not found at {var_path}")
    ctrl_disk_sha = sha256_file(ctrl_path)
    var_disk_sha = sha256_file(var_path)

    models = ev_man.get("models", {})
    if models.get("k0", {}).get("sha256") != k0_sha:
        raise ContractError(f"Eval manifest K0 SHA mismatch: {models.get('k0', {}).get('sha256')} vs {k0_sha}")
    if models.get("ext_mortal", {}).get("sha256") != ext_sha:
        raise ContractError(
            f"Eval manifest ext_mortal SHA mismatch: {models.get('ext_mortal', {}).get('sha256')} vs {ext_sha}"
        )
    if ext_sha != EXT_MORTAL_EXPECTED_SHA256:
        raise ContractError(f"ext_mortal canonical SHA mismatch: {ext_sha}")

    _verify_eval_manifest(
        ev_man,
        tr_man=tr_man,
        tr_man_sha=tr_man_sha,
        ctrl_disk_sha=ctrl_disk_sha,
        var_disk_sha=var_disk_sha,
    )
    ev_manifest_ok = True

    # Collect per-hanchan game logs from all 4 shards with identity verification.
    diff_var_minus_ctrl: list[float] = []
    diff_var_minus_k0: list[float] = []
    diff_ctrl_minus_k0: list[float] = []
    game_ids: list[int] = []

    for shard_idx in range(EVAL_SHARDS):
        shard_dir = eval_dir / f"shard_{shard_idx:03d}"
        logs_dir = shard_dir / "logs"
        if not logs_dir.exists():
            raise FileNotFoundError(f"Missing logs directory in {shard_dir}")

        log_files = sorted(logs_dir.glob("*.json.gz"))
        for log_path in log_files:
            ident = parse_game_identity(log_path)
            game_ids.append(int(ident["game_id"]))

            events = ident["events"]
            if events[-1].get("type") != "end_game":
                raise ContractError(f"Incomplete log file: {log_path}")
            _, ranks = _scores_and_ranks_from_events(events, log_path.name)
            n2s = {name: seat for seat, name in enumerate(ident["names"])}

            seat_k0 = n2s["K0_70k"]
            seat_ctrl = n2s["Control_70400"]
            seat_var = n2s["Variant_70400"]

            pt_k0 = TENHOU_RANK_POINTS[ranks[seat_k0]]
            pt_ctrl = TENHOU_RANK_POINTS[ranks[seat_ctrl]]
            pt_var = TENHOU_RANK_POINTS[ranks[seat_var]]

            diff_var_minus_ctrl.append(float(pt_var - pt_ctrl))
            diff_var_minus_k0.append(float(pt_var - pt_k0))
            diff_ctrl_minus_k0.append(float(pt_ctrl - pt_k0))

    # 1000 unique and strictly contiguous game IDs.
    expected_ids = list(range(EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE))
    logs_verified = (
        len(game_ids) == EVAL_TOTAL_GAMES
        and len(set(game_ids)) == EVAL_TOTAL_GAMES
        and sorted(game_ids) == expected_ids
    )
    if not logs_verified:
        raise ContractError(
            f"Game ID verification failed: parsed={len(game_ids)}, unique={len(set(game_ids))}, "
            f"expected contiguous range {EVAL_SEED_START}..{EVAL_SEED_END_EXCLUSIVE - 1}"
        )

    arr_var_ctrl = np.array(diff_var_minus_ctrl, dtype=np.float64)
    arr_var_k0 = np.array(diff_var_minus_k0, dtype=np.float64)
    arr_ctrl_k0 = np.array(diff_ctrl_minus_k0, dtype=np.float64)
    paired_ok = (
        len(arr_var_ctrl) == EVAL_TOTAL_GAMES
        and bool(np.isfinite(arr_var_ctrl).all())
        and bool(np.isfinite(arr_var_k0).all())
        and bool(np.isfinite(arr_ctrl_k0).all())
    )

    mean_vc, ci_vc = paired_bootstrap_ci(arr_var_ctrl, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
    mean_vk0, ci_vk0 = paired_bootstrap_ci(arr_var_k0, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 1, ci=BOOTSTRAP_CI)
    mean_ck0, ci_ck0 = paired_bootstrap_ci(arr_ctrl_k0, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 2, ci=BOOTSTRAP_CI)

    primary_ok = bool(np.isfinite(mean_vc) and np.isfinite(ci_vc).all())
    secondary_ok = bool(np.isfinite(mean_vk0) and np.isfinite(mean_ck0) and np.isfinite(ci_vk0).all() and np.isfinite(ci_ck0).all())
    bootstrap_ok = primary_ok and secondary_ok

    # Pilot verdict adjudication: strong_positive / weak_positive / not_promising
    verdict = adjudicate_r1_verdict(primary_mean=mean_vc, primary_ci_lower=ci_vc[0])

    hard_gates: dict[str, bool] = {
        "training_manifest_verified": tr_manifest_ok,
        "eval_manifest_verified": ev_manifest_ok,
        "all_logs_verified": logs_verified,
        "paired_metrics_recalculated": paired_ok,
        "primary_contrast_computed": primary_ok,
        "secondary_contrast_computed": secondary_ok,
        "bootstrap_computed": bootstrap_ok,
    }

    if set(hard_gates.keys()) != set(EXPECTED_SUMMARY_HARD_GATES):
        raise ContractError(f"Summary hard gates mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_SUMMARY_HARD_GATES)}")
    if not all(hard_gates.values()):
        raise ContractError(f"Summary hard gate failed: {hard_gates}")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "training_manifest": {"path": str(tr_man_path), "sha256": tr_man_sha},
        "eval_manifest": {"path": str(ev_man_path), "sha256": ev_man_sha},
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "evaluation_protocol": {
            "total_games": EVAL_TOTAL_GAMES,
            "game_id_range": [EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE],
            "lineup": list(EVALUATION_LINEUP),
            "rank_points": TENHOU_RANK_POINTS.tolist(),
        },
        "hard_gates": hard_gates,
        "metrics": {
            "total_games": len(arr_var_ctrl),
            "primary_contrast_variant_minus_control": {
                "mean_pt": mean_vc,
                "ci95": ci_vc,
            },
            "secondary_contrast_variant_minus_k0": {
                "mean_pt": mean_vk0,
                "ci95": ci_vk0,
            },
            "reference_contrast_control_minus_k0": {
                "mean_pt": mean_ck0,
                "ci95": ci_ck0,
            },
        },
        "verdict": verdict,
        "promotion": {
            "recipe_promotion": False,
            "checkpoint_promotion": False,
            "k1": None,
        },
    }

    summary_path = summary_dir / "r1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=R1_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=R1_EVAL_DIR)
    parser.add_argument("--summary-dir", type=Path, default=R1_SUMMARY_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = adjudicate_r1_pilot(training_dir=args.training_dir, eval_dir=args.eval_dir, summary_dir=args.summary_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
