#!/usr/bin/env python3
"""Deterministic late-decision counterfactual panel generator for P3 signal density evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import libriichi
import libriichi.arena
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Mortal" / "mortal"))

import engine
import model

from training.mortal.p3_late_decision_counterfactual_contract_2026_09 import (
    DISCARD_ACTION_LIMIT,
    EXPECTED_PANEL_HARD_GATES,
    EXPERIMENT_ID,
    FOCAL_SEAT,
    P3_PANEL_DIR,
    P3_ROOT,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_KEY,
    SEED_START,
    SPLIT_NAME,
    TENHOU_RANK_POINTS,
    ContractError,
    action_matches_pai,
    canonical_log_content_sha256,
    check_directory_boundary,
    compute_final_ranks,
    ensure_clean_staging_dir,
    final_scores_with_reach_accepted,
    normalize_event_for_canonical_hash,
    resolve_k0_checkpoint,
)

logger = logging.getLogger("p3_generator")


class CounterfactualBranchingEngine:
    """Wrapper that records decision contexts and forces an explicit action ID at a target context."""

    def __init__(
        self,
        base_engine: Any,
        target_context: tuple[int, int, int, int, int] | None = None,
        forced_action: int | None = None,
    ) -> None:
        self.base_engine = base_engine
        self.target_context = target_context
        self.forced_action = forced_action
        self.recorded_contexts: list[dict[str, Any]] = []
        self.intervened = False
        self.intervention_count = 0
        self.supports_decision_context = True

        self.name = base_engine.name
        self.is_oracle = base_engine.is_oracle
        self.version = base_engine.version
        self.enable_quick_eval = base_engine.enable_quick_eval
        self.enable_rule_based_agari_guard = base_engine.enable_rule_based_agari_guard
        self.enable_amp = base_engine.enable_amp
        self.device = base_engine.device
        self.engine_type = base_engine.engine_type

    def react_batch(
        self,
        obs: Any,
        masks: Any,
        invisible_obs: Any,
        decision_contexts: Any,
    ) -> Any:
        actions, q_values, returned_masks, is_greedy = self.base_engine.react_batch(obs, masks, invisible_obs)
        out_actions = list(actions)
        for i, ctx in enumerate(decision_contexts):
            gen_seed, seed_key, seat, kyoku_idx, dec_idx, own_riichi, _ = ctx
            c_tuple = (int(gen_seed), int(seed_key), int(seat), int(kyoku_idx), int(dec_idx))

            q_row = [float(v) for v in q_values[i]]
            m_row = [bool(v) for v in returned_masks[i]]
            legal = [act for act, ok in enumerate(m_row) if ok and torch.isfinite(torch.tensor(q_row[act]))]
            ranked = sorted(legal, key=lambda a: (-q_row[a], a))

            if len(legal) >= 2 and not own_riichi and ranked[0] < DISCARD_ACTION_LIMIT and ranked[1] < DISCARD_ACTION_LIMIT:
                self.recorded_contexts.append({
                    "ctx": c_tuple,
                    "top1": ranked[0],
                    "top2": ranked[1],
                    "top1_q": q_row[ranked[0]],
                    "top2_q": q_row[ranked[1]],
                    "margin": q_row[ranked[0]] - q_row[ranked[1]],
                })

            if self.target_context is not None and c_tuple == self.target_context:
                if self.forced_action is not None:
                    if self.forced_action not in legal:
                        raise ContractError(
                            f"Forced action {self.forced_action} is illegal in context {c_tuple} (legal: {legal})"
                        )
                    out_actions[i] = self.forced_action
                self.intervened = True
                self.intervention_count += 1

        return out_actions, q_values, returned_masks, is_greedy


def _create_k0_engine(name: str = "K0_70k", device: str = "cuda") -> engine.MortalEngine:
    k0_path, _ = resolve_k0_checkpoint()
    k0_state = torch.load(k0_path, map_location=device)
    m = model.Brain(version=4, conv_channels=192, num_blocks=40).eval()
    d = model.DQN(version=4).eval()
    m.load_state_dict(k0_state["mortal"])
    d.load_state_dict(k0_state["current_dqn"])
    return engine.MortalEngine(
        m,
        d,
        is_oracle=False,
        version=4,
        device=torch.device(device),
        name=name,
        enable_rule_based_agari_guard=True,
    )


def generate_single_counterfactual_pair(
    seed: int,
    seed_key: int,
    raw_logs_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate and strictly validate one paired counterfactual rollout for a single seed at the LAST eligible decision context."""
    pair_dir = raw_logs_dir / f"seed_{seed}"
    dir_a = pair_dir / "branch_a"
    dir_b = pair_dir / "branch_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    # 1. Baseline discovery run (in temp dir)
    with tempfile.TemporaryDirectory() as tmp_base:
        eng_base = CounterfactualBranchingEngine(_create_k0_engine("E0", device=device))
        e1 = _create_k0_engine("E1", device=device)
        e2 = _create_k0_engine("E2", device=device)
        e3 = _create_k0_engine("E3", device=device)

        arena_base = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=tmp_base)
        arena_base.py_vs_py(eng_base, e1, e2, e3, (seed, seed_key), 1)

        focal_contexts = [
            c for c in eng_base.recorded_contexts
            if c["ctx"][2] == FOCAL_SEAT
        ]
        if not focal_contexts:
            raise ContractError(f"No eligible non-riichi discard decision found for focal seat in seed {seed}")

        # Deterministic selection: LAST eligible context
        target = focal_contexts[-1]
        target_ctx = target["ctx"]
        top1_action = target["top1"]
        top2_action = target["top2"]

    # 2. Branch A: forced frozen top1 action ID
    eng_a = CounterfactualBranchingEngine(_create_k0_engine("E0", device=device), target_context=target_ctx, forced_action=top1_action)
    arena_a = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=str(dir_a))
    arena_a.py_vs_py(eng_a, _create_k0_engine("E1", device=device), _create_k0_engine("E2", device=device), _create_k0_engine("E3", device=device), (seed, seed_key), 1)
    if eng_a.intervention_count != 1:
        raise ContractError(f"Branch A for seed {seed} intervened {eng_a.intervention_count} times, expected exactly 1")

    # 3. Branch B: forced frozen top2 action ID
    eng_b = CounterfactualBranchingEngine(_create_k0_engine("E0", device=device), target_context=target_ctx, forced_action=top2_action)
    arena_b = libriichi.arena.FourPlayer(disable_progress_bar=True, log_dir=str(dir_b))
    arena_b.py_vs_py(eng_b, _create_k0_engine("E1", device=device), _create_k0_engine("E2", device=device), _create_k0_engine("E3", device=device), (seed, seed_key), 1)
    if eng_b.intervention_count != 1:
        raise ContractError(f"Branch B for seed {seed} intervened {eng_b.intervention_count} times, expected exactly 1")

    # 4. Strict explicit log matching: {seed}_{seed_key}_{split}.json.gz
    expected_log_name = f"{seed}_{seed_key}_{SPLIT_NAME}.json.gz"
    log_a_path = dir_a / expected_log_name
    log_b_path = dir_b / expected_log_name

    if not log_a_path.exists():
        raise FileNotFoundError(f"Branch A log {expected_log_name} not found in {dir_a}")
    if not log_b_path.exists():
        raise FileNotFoundError(f"Branch B log {expected_log_name} not found in {dir_b}")

    with gzip.open(log_a_path, "rt", encoding="utf-8") as f:
        events_a = [json.loads(line) for line in f]
    with gzip.open(log_b_path, "rt", encoding="utf-8") as f:
        events_b = [json.loads(line) for line in f]

    if not events_a or events_a[0].get("type") != "start_game" or events_a[-1].get("type") != "end_game":
        raise ContractError(f"Branch A log {expected_log_name} is incomplete or invalid")
    if not events_b or events_b[0].get("type") != "start_game" or events_b[-1].get("type") != "end_game":
        raise ContractError(f"Branch B log {expected_log_name} is incomplete or invalid")

    # 5. Bit-for-bit prefix check and divergence check (using normalized events to ignore wall-clock nanoseconds)
    norm_events_a = [normalize_event_for_canonical_hash(e) for e in events_a]
    norm_events_b = [normalize_event_for_canonical_hash(e) for e in events_b]

    div_idx = -1
    for idx, (ea, eb) in enumerate(zip(norm_events_a, norm_events_b, strict=False)):
        if ea != eb:
            div_idx = idx
            break

    if div_idx <= 0:
        raise ContractError(f"No divergence found between Branch A and B for seed {seed}")

    ev_a_div = events_a[div_idx]
    ev_b_div = events_b[div_idx]

    if ev_a_div.get("type") != "dahai" or ev_b_div.get("type") != "dahai":
        raise ContractError(f"Divergence at event {div_idx} is not dahai: A={ev_a_div} vs B={ev_b_div}")
    if ev_a_div.get("actor") != FOCAL_SEAT or ev_b_div.get("actor") != FOCAL_SEAT:
        raise ContractError(f"Divergence actor mismatch at event {div_idx}: A={ev_a_div.get('actor')} vs B={ev_b_div.get('actor')}")
    if ev_a_div.get("pai") == ev_b_div.get("pai"):
        raise ContractError(f"Divergence dahai pai identical at event {div_idx}: {ev_a_div.get('pai')}")

    pai_a = ev_a_div.get("pai", "")
    pai_b = ev_b_div.get("pai", "")
    if not action_matches_pai(top1_action, pai_a):
        raise ContractError(f"Branch A dahai pai '{pai_a}' does not match frozen top1_action {top1_action}")
    if not action_matches_pai(top2_action, pai_b):
        raise ContractError(f"Branch B dahai pai '{pai_b}' does not match frozen top2_action {top2_action}")

    log_a_content_sha = canonical_log_content_sha256(log_a_path)
    log_b_content_sha = canonical_log_content_sha256(log_b_path)

    # 6. Score and rank computation
    scores_a = final_scores_with_reach_accepted(events_a)
    scores_b = final_scores_with_reach_accepted(events_b)
    if scores_a is None or scores_b is None:
        raise ContractError(f"Could not reconstruct scores for seed {seed}")

    ranks_a = compute_final_ranks(scores_a)
    ranks_b = compute_final_ranks(scores_b)

    score_top1 = float(scores_a[FOCAL_SEAT])
    score_top2 = float(scores_b[FOCAL_SEAT])
    rank_top1 = int(ranks_a[FOCAL_SEAT])
    rank_top2 = int(ranks_b[FOCAL_SEAT])

    pt_top1 = float(TENHOU_RANK_POINTS[rank_top1])
    pt_top2 = float(TENHOU_RANK_POINTS[rank_top2])

    delta_final_score = score_top2 - score_top1
    delta_rank_point = pt_top2 - pt_top1

    return {
        "seed": seed,
        "seed_key": seed_key,
        "focal_seat": FOCAL_SEAT,
        "target_context": target_ctx,
        "top1_action": top1_action,
        "top2_action": top2_action,
        "top1_q": target["top1_q"],
        "top2_q": target["top2_q"],
        "margin": target["margin"],
        "divergence_event_index": div_idx,
        "branch_a_dahai_pai": pai_a,
        "branch_b_dahai_pai": pai_b,
        "branch_a_total_events": len(events_a),
        "branch_b_total_events": len(events_b),
        "branch_a_log_path": str(log_a_path),
        "branch_b_log_path": str(log_b_path),
        "branch_a_canonical_content_sha256": log_a_content_sha,
        "branch_b_canonical_content_sha256": log_b_content_sha,
        "rank_top1": rank_top1,
        "rank_top2": rank_top2,
        "score_top1": score_top1,
        "score_top2": score_top2,
        "pt_top1": pt_top1,
        "pt_top2": pt_top2,
        "delta_rank_point": delta_rank_point,
        "delta_final_score": delta_final_score,
    }


def generate_p3_late_decision_panel(
    panel_games: int = PANEL_GAMES,
    seed_start: int = SEED_START,
    seed_key: int = SEED_KEY,
    output_dir: Path = P3_PANEL_DIR,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute complete generation of 128 late-decision counterfactual pairs and write panel manifest."""
    check_directory_boundary(output_dir, P3_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_logs_dir = output_dir / "raw_logs"
    ensure_clean_staging_dir(raw_logs_dir, output_dir)

    _, k0_sha = resolve_k0_checkpoint()

    hard_gates: dict[str, bool] = {
        "k0_parent_verified": True,
        "exact_128_pairs_generated": False,
        "seeds_strictly_contiguous": False,
        "focal_seat_verified": False,
        "all_target_contexts_intervened_exactly_once": False,
        "all_prefixes_exact_matched": False,
        "all_first_divergences_verified_dahai": False,
        "all_branches_completed_end_game": False,
        "scores_and_ranks_valid": False,
    }

    pairs: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, seed in enumerate(range(seed_start, seed_start + panel_games)):
        logger.info("Generating late-decision pair %d/%d (seed %d)...", idx + 1, panel_games, seed)
        pair_res = generate_single_counterfactual_pair(
            seed=seed,
            seed_key=seed_key,
            raw_logs_dir=raw_logs_dir,
            device=device,
        )
        pairs.append(pair_res)

    elapsed = time.time() - t0
    logger.info("Generated %d late-decision pairs in %.2f seconds", len(pairs), elapsed)

    hard_gates["exact_128_pairs_generated"] = (len(pairs) == panel_games)
    hard_gates["seeds_strictly_contiguous"] = (
        [p["seed"] for p in pairs] == list(range(seed_start, seed_start + panel_games))
    )
    hard_gates["focal_seat_verified"] = all(p["focal_seat"] == FOCAL_SEAT for p in pairs)
    hard_gates["all_target_contexts_intervened_exactly_once"] = True
    hard_gates["all_prefixes_exact_matched"] = all(p["divergence_event_index"] > 0 for p in pairs)
    hard_gates["all_first_divergences_verified_dahai"] = all(
        p["branch_a_dahai_pai"] != p["branch_b_dahai_pai"]
        and action_matches_pai(p["top1_action"], p["branch_a_dahai_pai"])
        and action_matches_pai(p["top2_action"], p["branch_b_dahai_pai"])
        for p in pairs
    )
    hard_gates["all_branches_completed_end_game"] = all(
        p["branch_a_total_events"] > 0 and p["branch_b_total_events"] > 0 for p in pairs
    )
    hard_gates["scores_and_ranks_valid"] = all(
        0 <= p["rank_top1"] <= 3 and 0 <= p["rank_top2"] <= 3 for p in pairs
    )

    if set(hard_gates.keys()) != set(EXPECTED_PANEL_HARD_GATES):
        raise ContractError(
            f"Panel hard gates key mismatch: {set(hard_gates.keys())} vs {set(EXPECTED_PANEL_HARD_GATES)}"
        )

    manifest = {
        "schema": PANEL_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {
            "name": "K0_70k",
            "sha256": k0_sha,
        },
        "panel_config": {
            "total_pairs": len(pairs),
            "seed_start": seed_start,
            "seed_end_exclusive": seed_start + panel_games,
            "seed_key": seed_key,
            "focal_seat": FOCAL_SEAT,
            "split_name": SPLIT_NAME,
            "device": device,
        },
        "hard_gates": hard_gates,
        "pairs": pairs,
        "verdict": "panel_generation_completed" if all(hard_gates.values()) else "panel_generation_failed",
    }

    manifest_path = output_dir / "counterfactual_panel_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=PANEL_GAMES, help="Number of paired games to generate")
    parser.add_argument("--seed-start", type=int, default=SEED_START, help="Start seed")
    parser.add_argument("--output-dir", type=Path, default=P3_PANEL_DIR, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = generate_p3_late_decision_panel(
        panel_games=args.games,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
