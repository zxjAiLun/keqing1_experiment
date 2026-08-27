"""Statistical summary and crossed bootstrap adjudication for R2 multi-seed confirmation experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.r2_rank_plus_score_to_go_multiseed_contract_2026_09 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CANONICAL_K1_SEED,
    EVAL_GAMES_PER_PANEL,
    EVAL_GAMES_PER_SHARD,
    EVAL_MANIFEST_SCHEMA,
    EVAL_SEED_END_EXCLUSIVE,
    EVAL_SEED_KEY,
    EVAL_SEED_START,
    EVAL_SHARDS_PER_PANEL,
    EVALUATION_LINEUP,
    EXPECTED_EVAL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    EXT_MORTAL_EXPECTED_SHA256,
    K0_EXPECTED_SHA256,
    R2_EVAL_DIR,
    R2_SUMMARY_DIR,
    R2_TRAINING_DIR,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    TRAINING_MANIFEST_SCHEMA,
    TRAINING_SEEDS,
    ContractError,
    adjudicate_r2_verdict,
    check_directory_empty_or_nonexistent,
    crossed_bootstrap_ci,
    parse_game_identity,
    resolve_ext_mortal_checkpoint,
    resolve_k0_checkpoint,
    sha256_file,
    verify_training_manifest,
)

logger = logging.getLogger("r2_summary")


def _scores_and_ranks_from_events(events: list[dict[str, Any]], log_name: str) -> tuple[list[float], list[int]]:
    """Reconstruct final scores with exact reach_accepted semantics, then ranks (R1 parity)."""
    scores: list[float] | None = None
    has_kyoku_scores = False
    reach_ok = True
    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "start_kyoku" and isinstance(ev.get("scores"), list):
            values = ev["scores"]
            if len(values) == 4:
                scores = [float(v) for v in values]
                has_kyoku_scores = True
        elif ev_type == "reach_accepted" and scores is not None:
            actor = ev.get("actor")
            if not isinstance(actor, int) or not (0 <= actor < 4):
                reach_ok = False
            else:
                scores[int(actor)] -= 1000.0
        elif ev_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = ev.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [s + float(d) for s, d in zip(scores, deltas, strict=True)]
    if not has_kyoku_scores:
        raise ContractError(f"Failed reach_accepted semantics for {log_name}: missing start_kyoku scores")
    if not reach_ok:
        raise ContractError(f"Failed reach_accepted semantics for {log_name}: invalid actor")
    if scores is None:
        raise ContractError(f"Failed to reconstruct scores for {log_name}")

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
    k0_sha: str,
    ext_sha: str,
) -> None:
    """Fail-closed validation of the eval manifest and its SHA chain bindings (R1 parity)."""
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

    bound = ev_man.get("training_manifest")
    if not isinstance(bound, dict) or bound.get("sha256") != tr_man_sha:
        raise ContractError(
            f"Eval manifest training-manifest SHA mismatch: bound={bound.get('sha256') if isinstance(bound, dict) else None}, disk={tr_man_sha}"
        )

    parent = ev_man.get("parent_model")
    if not isinstance(parent, dict) or parent.get("sha256") != k0_sha:
        raise ContractError(f"Eval manifest K0 SHA mismatch: {parent.get('sha256') if isinstance(parent, dict) else None} vs {k0_sha}")
    ext_model = ev_man.get("ext_mortal_model")
    if not isinstance(ext_model, dict) or ext_model.get("sha256") != ext_sha:
        raise ContractError(f"Eval manifest ext_mortal SHA mismatch: {ext_model.get('sha256') if isinstance(ext_model, dict) else None} vs {ext_sha}")
    panels = ev_man.get("panels")
    if not isinstance(panels, dict):
        raise ContractError(f"Eval manifest panels missing or invalid: {panels}")
    tr_checkpoints = tr_man.get("checkpoints", {})
    for seed_val in TRAINING_SEEDS:
        pkey = f"seed_{seed_val}"
        panel_entry = panels.get(pkey)
        tr_entry = tr_checkpoints.get(pkey)
        if not isinstance(panel_entry, dict) or not isinstance(tr_entry, dict):
            raise ContractError(f"Eval manifest missing panel or training checkpoint for {pkey}")
        for cond in ("control", "variant"):
            ev_sha = panel_entry.get("models", {}).get(cond, {}).get("sha256")
            tr_sha = tr_entry.get(cond, {}).get("sha256")
            if not ev_sha or len(ev_sha) != 64:
                raise ContractError(f"Eval manifest panel checkpoint sha missing/invalid for {pkey}/{cond}")
            if not tr_sha or len(tr_sha) != 64:
                raise ContractError(f"Training manifest checkpoint sha missing/invalid for {pkey}/{cond}")
            if ev_sha != tr_sha:
                raise ContractError(f"Eval vs training checkpoint SHA mismatch for {pkey}/{cond}: eval={ev_sha} training={tr_sha}")


def adjudicate_r2_multiseed(
    training_dir: Path = R2_TRAINING_DIR,
    eval_dir: Path = R2_EVAL_DIR,
    summary_dir: Path = R2_SUMMARY_DIR,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Load manifests, verify all 3000 logs across 3 panels, run crossed bootstrap, and adjudicate."""
    target_seeds = seeds or TRAINING_SEEDS
    check_directory_empty_or_nonexistent(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    tr_man_path = training_dir / "r2_training_manifest.json"
    ev_man_path = eval_dir / "r2_eval_manifest.json"

    if not tr_man_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {tr_man_path}")
    if not ev_man_path.exists():
        raise FileNotFoundError(f"Eval manifest not found at {ev_man_path}")

    tr_man_sha = sha256_file(tr_man_path)
    ev_man_sha = sha256_file(ev_man_path)

    tr_man = json.loads(tr_man_path.read_text(encoding="utf-8"))
    ev_man = json.loads(ev_man_path.read_text(encoding="utf-8"))

    # Fail-closed manifest verification (exact gate sets, objective/label/reward, row-identity, SHA bindings)
    verify_training_manifest(tr_man)
    tr_manifest_ok = True

    _, k0_sha = resolve_k0_checkpoint()
    _, ext_sha = resolve_ext_mortal_checkpoint()
    if ext_sha != EXT_MORTAL_EXPECTED_SHA256 or k0_sha != K0_EXPECTED_SHA256:
        raise ContractError(f"Canonical model SHA mismatch: k0={k0_sha} ext={ext_sha}")

    _verify_eval_manifest(ev_man, tr_man=tr_man, tr_man_sha=tr_man_sha, k0_sha=k0_sha, ext_sha=ext_sha)
    ev_manifest_ok = True

    # Verify six checkpoints manifest ↔ disk ↔ eval SHA (strict three-way binding)
    for s in target_seeds:
        key = f"seed_{s}"
        tr_entry = tr_man.get("checkpoints", {}).get(key)
        if not isinstance(tr_entry, dict):
            raise ContractError(f"Training manifest missing checkpoint entry for {key}")
        for cond in ("control", "variant"):
            rec = tr_entry.get(cond)
            if not isinstance(rec, dict):
                raise ContractError(f"Training manifest missing {key}/{cond}")
            path_str = rec.get("path")
            sha_rec = rec.get("sha256")
            if not path_str or not sha_rec or len(sha_rec) != 64:
                raise ContractError(f"Training manifest checkpoint sha/path missing/invalid for {key}/{cond}: path={path_str} sha={sha_rec}")
            p = Path(path_str)
            if not p.exists():
                p2 = training_dir / f"mortal_{cond}_70400_seed_{s}.pth"
                if p2.exists():
                    p = p2
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {path_str}")
            disk_sha = sha256_file(p)
            if disk_sha != sha_rec:
                raise ContractError(f"Checkpoint SHA mismatch for {key}/{cond}: manifest={sha_rec} disk={disk_sha}")
            panels = ev_man.get("panels", {})
            if not isinstance(panels, dict) or key not in panels:
                raise ContractError(f"Eval manifest missing panel {key}")
            ev_panel = panels[key]
            ev_sha = ev_panel.get("models", {}).get(cond, {}).get("sha256")
            if not ev_sha or len(ev_sha) != 64:
                raise ContractError(f"Eval manifest panel checkpoint sha missing/invalid for {key}/{cond}")
            if ev_sha != sha_rec:
                raise ContractError(f"Three-way checkpoint SHA mismatch for {key}/{cond}: training={sha_rec} eval={ev_sha} disk={disk_sha}")
            if ev_sha != disk_sha:
                raise ContractError(f"Three-way checkpoint SHA mismatch for {key}/{cond}: eval={ev_sha} disk={disk_sha}")

    # Shape: (3, 1000) for Primary (Variant - Control) and Absolute (Variant - K0)
    primary_matrix = np.zeros((len(target_seeds), EVAL_GAMES_PER_PANEL), dtype=np.float64)
    absolute_matrix = np.zeros((len(target_seeds), EVAL_GAMES_PER_PANEL), dtype=np.float64)

    total_logs_verified = 0
    # Track per-panel IDs for contiguous check
    per_panel_ok = True

    for s_idx, t_seed in enumerate(target_seeds):
        panel_dir = eval_dir / f"panel_seed_{t_seed}"
        if not panel_dir.exists():
            raise FileNotFoundError(f"Panel dir {panel_dir} does not exist")

        # Glob all logs recursively; four_player_native --seat-mode=random generates a/b/c/d suffix, not just _0
        # Use rglob to find any *.json.gz under panel_dir, but validate via parse_game_identity
        all_logs = sorted(panel_dir.rglob("*.json.gz"))
        # Filter to only those under shard_*/logs to avoid stray files?
        # Keep all that match LOG_NAME_RE via parse_game_identity – will fail-closed on bad names
        game_ids_this_panel: list[int] = []
        # Track seen game_ids to detect duplicate
        seen_ids: set[int] = set()
        # For precise failure messages, also check per-shard count if shards exist
        shard_counts: dict[int, int] = {i: 0 for i in range(EVAL_SHARDS_PER_PANEL)}
        # We'll also verify that per-shard directory exists and contains expected structure
        for shard_idx in range(EVAL_SHARDS_PER_PANEL):
            sd = panel_dir / f"shard_{shard_idx:03d}" / "logs"
            if sd.exists():
                # Count files per shard via glob
                cnt = len(list(sd.glob("*.json.gz")))
                shard_counts[shard_idx] = cnt
                # Enforce exactly 250 per shard if shard dir exists (fail-closed); but allow legacy where shards maybe missing?
                # For existing mock test, shards are 4 x 250 = 1000, so enforce.
                # For future strict, same.
                if cnt != EVAL_GAMES_PER_SHARD:
                    # Only raise if we have rglob total also off? To avoid false positive on missing shard dir
                    # But if shard exists, it must have 250
                    raise ContractError(f"Panel {panel_dir.name} shard {shard_idx} contains {cnt} logs, expected {EVAL_GAMES_PER_SHARD}")

        if len(all_logs) != EVAL_GAMES_PER_PANEL:
            raise ContractError(f"Panel {panel_dir.name} total logs {len(all_logs)} != {EVAL_GAMES_PER_PANEL}")

        # Temporary arrays to hold values before placing into matrix to detect duplicates
        col_to_vals: dict[int, tuple[float, float]] = {}
        for log_path in all_logs:
            ident = parse_game_identity(log_path)
            game_id = int(ident["game_id"])
            events = ident["events"]
            if events[-1].get("type") != "end_game":
                raise ContractError(f"Incomplete log file: {log_path}")
            # Validate reach semantics and reconstruct scores/ranks
            _, ranks = _scores_and_ranks_from_events(events, log_path.name)
            n2s = {name: seat for seat, name in enumerate(ident["names"])}
            # Exact lineup already validated in parse_game_identity, but double-check seat mapping
            seat_k0 = n2s["K0_70k"]
            seat_ctrl = n2s["Control_70400"]
            seat_var = n2s["Variant_70400"]

            pt_k0 = TENHOU_RANK_POINTS[ranks[seat_k0]]
            pt_ctrl = TENHOU_RANK_POINTS[ranks[seat_ctrl]]
            pt_var = TENHOU_RANK_POINTS[ranks[seat_var]]

            col = game_id - EVAL_SEED_START
            if col < 0 or col >= EVAL_GAMES_PER_PANEL:
                raise ContractError(f"Game ID {game_id} out of expected range for panel {panel_dir.name}")
            if col in col_to_vals:
                raise ContractError(f"Duplicate game ID {game_id} in panel {panel_dir.name}")
            if game_id in seen_ids:
                raise ContractError(f"Duplicate game ID {game_id} in panel {panel_dir.name}")
            seen_ids.add(game_id)
            game_ids_this_panel.append(game_id)
            col_to_vals[col] = (float(pt_var - pt_ctrl), float(pt_var - pt_k0))

        # Verify unique contiguous 1000 IDs per panel
        expected_ids = list(range(EVAL_SEED_START, EVAL_SEED_END_EXCLUSIVE))
        if sorted(game_ids_this_panel) != expected_ids:
            per_panel_ok = False
            raise ContractError(
                f"Panel {panel_dir.name} game ID verification failed: sorted IDs mismatch expected contiguous range {EVAL_SEED_START}..{EVAL_SEED_END_EXCLUSIVE-1}"
            )
        if len(set(game_ids_this_panel)) != EVAL_GAMES_PER_PANEL:
            per_panel_ok = False
            raise ContractError(f"Panel {panel_dir.name} duplicate/missing game IDs: unique {len(set(game_ids_this_panel))}")

        # Fill matrix columns in game_id order (column = game_id - 2300000)
        for col, (prim, absv) in col_to_vals.items():
            primary_matrix[s_idx, col] = prim
            absolute_matrix[s_idx, col] = absv

        total_logs_verified += len(game_ids_this_panel)

    if total_logs_verified != len(target_seeds) * EVAL_GAMES_PER_PANEL:
        raise ContractError(f"Total verified logs mismatch: {total_logs_verified} vs {len(target_seeds) * EVAL_GAMES_PER_PANEL}")

    logs_verified = (total_logs_verified == len(target_seeds) * EVAL_GAMES_PER_PANEL and per_panel_ok)

    # Paired metrics are recalculated via _scores_and_ranks_from_events; ensure finite
    paired_ok = (
        primary_matrix.shape == (len(target_seeds), EVAL_GAMES_PER_PANEL)
        and absolute_matrix.shape == (len(target_seeds), EVAL_GAMES_PER_PANEL)
        and bool(np.isfinite(primary_matrix).all())
        and bool(np.isfinite(absolute_matrix).all())
    )
    if not paired_ok:
        raise ContractError("Paired metrics not finite")

    # Seed-level means
    primary_seed_means = [float(np.mean(primary_matrix[i, :])) for i in range(len(target_seeds))]
    absolute_seed_means = [float(np.mean(absolute_matrix[i, :])) for i in range(len(target_seeds))]

    # Crossed Bootstrap with shared resampling indices across primary and absolute
    grand_primary_mean, primary_ci, sampled_indices = crossed_bootstrap_ci(
        primary_matrix, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI, return_sampled_indices=True
    )
    grand_absolute_mean, absolute_ci, _ = crossed_bootstrap_ci(
        absolute_matrix, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI, shared_indices=sampled_indices
    )

    crossed_ok = bool(np.isfinite(grand_primary_mean) and np.isfinite(primary_ci).all() and np.isfinite(grand_absolute_mean) and np.isfinite(absolute_ci).all())

    verdict, recipe_promo, ckpt_promo, k1_target = adjudicate_r2_verdict(
        primary_seed_means=primary_seed_means,
        primary_ci_lower=primary_ci[0],
        absolute_seed_means=absolute_seed_means,
        absolute_ci_lower=absolute_ci[0],
    )

    primary_ok = bool(np.isfinite(grand_primary_mean) and np.isfinite(primary_ci).all())
    absolute_ok = bool(np.isfinite(grand_absolute_mean) and np.isfinite(absolute_ci).all())

    hard_gates: dict[str, bool] = {
        "training_manifest_verified": tr_manifest_ok,
        "eval_manifest_verified": ev_manifest_ok,
        "all_3000_logs_verified": logs_verified,
        "paired_metrics_recalculated": paired_ok,
        "crossed_bootstrap_computed": crossed_ok,
        "primary_contrast_evaluated": primary_ok,
        "absolute_contrast_evaluated": absolute_ok,
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
        "training_seeds": target_seeds,
        "canonical_k1_seed": CANONICAL_K1_SEED,
        "hard_gates": hard_gates,
        "metrics": {
            "total_games": total_logs_verified,
            "primary_contrast_variant_minus_control": {
                "grand_mean_pt": grand_primary_mean,
                "ci95": primary_ci,
                "seed_means_pt": {
                    f"seed_{s}": primary_seed_means[i] for i, s in enumerate(target_seeds)
                },
                "all_seed_means_positive": all(m > 0 for m in primary_seed_means),
                "ci_lower_positive": (primary_ci[0] > 0),
            },
            "absolute_contrast_variant_minus_k0": {
                "grand_mean_pt": grand_absolute_mean,
                "ci95": absolute_ci,
                "seed_means_pt": {
                    f"seed_{s}": absolute_seed_means[i] for i, s in enumerate(target_seeds)
                },
                "all_seed_means_positive": all(m > 0 for m in absolute_seed_means),
                "ci_lower_positive": (absolute_ci[0] > 0),
            },
        },
        "verdict": verdict,
        "promotion": {
            "recipe_promotion": recipe_promo,
            "checkpoint_promotion": ckpt_promo,
            "k1": k1_target,
        },
    }

    summary_path = summary_dir / "r2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=R2_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=R2_EVAL_DIR)
    parser.add_argument("--summary-dir", type=Path, default=R2_SUMMARY_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = adjudicate_r2_multiseed(training_dir=args.training_dir, eval_dir=args.eval_dir, summary_dir=args.summary_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
