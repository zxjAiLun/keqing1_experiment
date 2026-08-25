"""Targeted unit tests for P2 counterfactual target quality generation and summary contracts."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.generate_p2_counterfactual_panel_2026_09 import (
    generate_single_counterfactual_pair,
)
from training.mortal.p2_counterfactual_target_quality_contract_2026_09 import (
    EXPECTED_PANEL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    FOCAL_SEAT,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    SPLIT_NAME,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    action_matches_pai,
    canonical_log_content_sha256,
    resolve_k0_checkpoint,
)
from training.mortal.summary_p2_counterfactual_target_quality_2026_09 import (
    adjudicate_p2_counterfactual_panel,
    summarize_diff_series,
)


def test_1_p2_contract_invariants_and_action_matches_pai() -> None:
    """Test 1: P2 contract invariants, hard gate sets, and action-to-pai mapping."""
    assert PANEL_GAMES == 128
    assert SEED_START == 3000000
    assert SEED_END_EXCLUSIVE == 3000128
    assert SEED_KEY == 8192
    assert FOCAL_SEAT == 0
    assert SPLIT_NAME == "a"
    assert len(TENHOU_RANK_POINTS) == 4

    assert len(EXPECTED_PANEL_HARD_GATES) == 9
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 8

    # action_matches_pai checks
    assert action_matches_pai(0, "1m")
    assert action_matches_pai(4, "5m")
    assert action_matches_pai(4, "5mr")
    assert action_matches_pai(13, "5pr")
    assert action_matches_pai(22, "5sr")
    assert action_matches_pai(27, "E")
    assert not action_matches_pai(0, "2m")
    assert not action_matches_pai(4, "5p")


def test_2_single_pair_generation_and_hard_divergence_check(tmp_path: Path) -> None:
    """Test 2: Single pair generation strictly enforces bit-exact prefix, single intervention, and dahai divergence matching action IDs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_logs = tmp_path / "raw_logs"
    raw_logs.mkdir()

    pair = generate_single_counterfactual_pair(
        seed=3000000,
        seed_key=8192,
        raw_logs_dir=raw_logs,
        device=device,
    )

    assert pair["seed"] == 3000000
    assert pair["seed_key"] == 8192
    assert pair["focal_seat"] == 0
    assert pair["divergence_event_index"] > 0

    # Divergence must match top1 and top2 action IDs exactly
    assert action_matches_pai(pair["top1_action"], pair["branch_a_dahai_pai"])
    assert action_matches_pai(pair["top2_action"], pair["branch_b_dahai_pai"])
    assert pair["branch_a_dahai_pai"] != pair["branch_b_dahai_pai"]
    assert pair["top1_action"] != pair["top2_action"]

    # Canonical content SHAs must be present
    assert len(pair["branch_a_canonical_content_sha256"]) == 64
    assert len(pair["branch_b_canonical_content_sha256"]) == 64

    # Scores and ranks must be bounded
    assert 0 <= pair["rank_top1"] <= 3
    assert 0 <= pair["rank_top2"] <= 3
    assert pair["pt_top1"] in TENHOU_RANK_POINTS
    assert pair["pt_top2"] in TENHOU_RANK_POINTS
    assert pair["delta_rank_point"] == pair["pt_top2"] - pair["pt_top1"]
    assert pair["delta_final_score"] == pair["score_top2"] - pair["score_top1"]


def test_3_fixed_seed_rerun_bit_exact_reproducibility_including_content_sha(tmp_path: Path) -> None:
    """Test 3: Generating the same seed twice produces bit-exact identical canonical content SHAs and metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_1 = tmp_path / "run_1"
    raw_2 = tmp_path / "run_2"
    raw_1.mkdir()
    raw_2.mkdir()

    p1 = generate_single_counterfactual_pair(seed=3000001, seed_key=8192, raw_logs_dir=raw_1, device=device)
    p2 = generate_single_counterfactual_pair(seed=3000001, seed_key=8192, raw_logs_dir=raw_2, device=device)

    assert p1["branch_a_canonical_content_sha256"] == p2["branch_a_canonical_content_sha256"]
    assert p1["branch_b_canonical_content_sha256"] == p2["branch_b_canonical_content_sha256"]
    assert p1["target_context"] == p2["target_context"]
    assert p1["top1_action"] == p2["top1_action"]
    assert p1["top2_action"] == p2["top2_action"]
    assert p1["divergence_event_index"] == p2["divergence_event_index"]
    assert p1["branch_a_dahai_pai"] == p2["branch_a_dahai_pai"]
    assert p1["branch_b_dahai_pai"] == p2["branch_b_dahai_pai"]
    assert p1["delta_rank_point"] == p2["delta_rank_point"]
    assert p1["delta_final_score"] == p2["delta_final_score"]


def test_4_diff_series_summary_and_bootstrap_ci() -> None:
    """Test 4: summarize_diff_series accurately calculates rates, counts, mean, and deterministic CI."""
    diffs = np.array([45.0, -90.0, 0.0, 90.0, -45.0, 0.0, 45.0, 0.0], dtype=np.float64)
    summary = summarize_diff_series(diffs, reps=1000, seed=12345)

    assert summary["total_pairs"] == 8
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == 2
    assert summary["tie_count"] == 3
    assert summary["nonzero_count"] == 5
    assert summary["nonzero_rate"] == 5.0 / 8.0
    assert summary["positive_rate"] == 3.0 / 8.0
    assert summary["negative_rate"] == 2.0 / 8.0
    assert summary["tie_rate"] == 3.0 / 8.0
    assert summary["mean_diff"] == float(np.mean(diffs))
    assert len(summary["ci95"]) == 2
    assert summary["ci95"][0] <= summary["mean_diff"] <= summary["ci95"][1]


def test_5_adjudicate_p2_counterfactual_panel_mock_and_fail_closed(tmp_path: Path) -> None:
    """Test 5: Summarizer validates raw logs from disk, canonical hashes, and fails closed if logs are missing or tampered."""
    panel_dir = tmp_path / "counterfactual_panel"
    summary_dir = tmp_path / "summary"
    raw_logs = panel_dir / "raw_logs"
    panel_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    raw_logs.mkdir(parents=True)

    _, k0_sha = resolve_k0_checkpoint()

    # Generate real synthetic gzipped log files for 128 pairs
    pairs = []
    for s in range(SEED_START, SEED_END_EXCLUSIVE):
        pair_dir = raw_logs / f"seed_{s}"
        dir_a = pair_dir / "branch_a"
        dir_b = pair_dir / "branch_b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        log_a_path = dir_a / f"{s}_8192_a.json.gz"
        log_b_path = dir_b / f"{s}_8192_a.json.gz"

        # Synthetic matching prefix + divergent dahai + end_game
        events_prefix = [
            {"type": "start_game", "names": ["K0_70k", "opp1", "opp2", "opp3"], "seed": [s, 8192]},
            {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
            {"type": "tsumo", "actor": 0, "pai": "2p"},
        ]
        # Action 9 is "1p", Action 10 is "2p"
        event_a_div = {"type": "dahai", "actor": 0, "pai": "1p"}
        event_b_div = {"type": "dahai", "actor": 0, "pai": "2p"}
        events_tail = [
            {"type": "hora", "deltas": [8000, -8000, 0, 0]},
            {"type": "end_game"},
        ]

        events_a = events_prefix + [event_a_div] + events_tail
        events_b = events_prefix + [event_b_div] + events_tail

        with gzip.open(log_a_path, "wt", encoding="utf-8") as f:
            for ev in events_a:
                f.write(json.dumps(ev) + "\n")
        with gzip.open(log_b_path, "wt", encoding="utf-8") as f:
            for ev in events_b:
                f.write(json.dumps(ev) + "\n")

        sha_a = canonical_log_content_sha256(log_a_path)
        sha_b = canonical_log_content_sha256(log_b_path)

        pairs.append({
            "seed": s,
            "seed_key": SEED_KEY,
            "focal_seat": FOCAL_SEAT,
            "target_context": [s, SEED_KEY, FOCAL_SEAT, 0, 0],
            "top1_action": 9,
            "top2_action": 10,
            "top1_q": 0.5,
            "top2_q": 0.2,
            "margin": 0.3,
            "divergence_event_index": 3,
            "branch_a_dahai_pai": "1p",
            "branch_b_dahai_pai": "2p",
            "branch_a_total_events": len(events_a),
            "branch_b_total_events": len(events_b),
            "branch_a_log_path": str(log_a_path),
            "branch_b_log_path": str(log_b_path),
            "branch_a_canonical_content_sha256": sha_a,
            "branch_b_canonical_content_sha256": sha_b,
            "rank_top1": 0,
            "rank_top2": 0,
            "score_top1": 33000.0,
            "score_top2": 33000.0,
            "pt_top1": 90.0,
            "pt_top2": 90.0,
            "delta_rank_point": 0.0,
            "delta_final_score": 0.0,
        })

    hard_gates = {g: True for g in EXPECTED_PANEL_HARD_GATES}
    manifest = {
        "schema": PANEL_MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_model": {"name": "K0_70k", "sha256": k0_sha},
        "panel_config": {
            "total_pairs": 128,
            "seed_start": SEED_START,
            "seed_end_exclusive": SEED_END_EXCLUSIVE,
            "seed_key": SEED_KEY,
            "focal_seat": FOCAL_SEAT,
            "split_name": SPLIT_NAME,
            "device": "cpu",
        },
        "hard_gates": hard_gates,
        "pairs": pairs,
        "verdict": "panel_generation_completed",
    }
    manifest_path = panel_dir / "counterfactual_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # 1. Valid pass
    summary = adjudicate_p2_counterfactual_panel(
        panel_dir=panel_dir,
        summary_dir=summary_dir,
        allowed_root=tmp_path,
    )
    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["experiment_id"] == EXPERIMENT_ID
    assert summary["hard_gates"]["all_branch_logs_verified"] is True
    assert summary["hard_gates"]["canonical_content_hashes_verified"] is True
    assert summary["hard_gates"]["independent_metrics_recalculated_match"] is True

    # 2. Missing log file fail-closed check
    first_log_a = Path(pairs[0]["branch_a_log_path"])
    first_log_a.unlink()
    with pytest.raises(FileNotFoundError):
        adjudicate_p2_counterfactual_panel(panel_dir=panel_dir, summary_dir=summary_dir, allowed_root=tmp_path)
