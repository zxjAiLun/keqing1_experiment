"""Targeted unit tests for P3 late-decision counterfactual signal density evaluation."""

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

from training.mortal.generate_p3_late_decision_panel_2026_09 import (
    generate_single_counterfactual_pair,
)
from training.mortal.p2_counterfactual_target_quality_contract_2026_09 import (
    EXPECTED_SUMMARY_HARD_GATES as EXPECTED_P2_SUMMARY_HARD_GATES,
)
from training.mortal.p3_late_decision_counterfactual_contract_2026_09 import (
    BOOTSTRAP_SEED,
    EXPECTED_PANEL_HARD_GATES,
    EXPECTED_SUMMARY_HARD_GATES,
    EXPERIMENT_ID,
    FOCAL_SEAT,
    P2_EXPERIMENT_ID,
    P2_SUMMARY_EXPECTED_SHA256,
    P2_SUMMARY_PATH,
    P2_SUMMARY_SCHEMA,
    PANEL_GAMES,
    PANEL_MANIFEST_SCHEMA,
    SEED_END_EXCLUSIVE,
    SEED_KEY,
    SEED_START,
    SPLIT_NAME,
    SUMMARY_SCHEMA,
    TENHOU_RANK_POINTS,
    ContractError,
    action_matches_pai,
    adjudicate_p3_verdict,
    canonical_log_content_sha256,
    resolve_k0_checkpoint,
    sha256_file,
    two_sample_rate_diff_bootstrap_ci,
)
from training.mortal.summary_p3_late_decision_counterfactual_2026_09 import (
    adjudicate_p3_counterfactual_panel,
)


def test_1_p3_contract_invariants() -> None:
    """Test 1: P3 contract invariants, seeds range, schemas, and gate sets."""
    assert PANEL_GAMES == 128
    assert SEED_START == 3100000
    assert SEED_END_EXCLUSIVE == 3100128
    assert SEED_KEY == 8192
    assert FOCAL_SEAT == 0
    assert SPLIT_NAME == "a"
    assert BOOTSTRAP_SEED == 20260905
    assert len(TENHOU_RANK_POINTS) == 4
    assert len(EXPECTED_PANEL_HARD_GATES) == 9
    assert len(EXPECTED_SUMMARY_HARD_GATES) == 9

    # P2 expected SHA must match real authoritative P2 artifact
    assert P2_SUMMARY_PATH.exists()
    assert sha256_file(P2_SUMMARY_PATH) == P2_SUMMARY_EXPECTED_SHA256


def test_2_single_pair_late_decision_generation_and_hard_divergence(tmp_path: Path) -> None:
    """Test 2: Single pair late-decision generation produces bit-exact prefix and divergent dahai matching frozen action IDs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_logs = tmp_path / "raw_logs"
    raw_logs.mkdir()

    pair = generate_single_counterfactual_pair(
        seed=3100000,
        seed_key=8192,
        raw_logs_dir=raw_logs,
        device=device,
    )

    assert pair["seed"] == 3100000
    assert pair["seed_key"] == 8192
    assert pair["focal_seat"] == 0
    assert pair["divergence_event_index"] > 0

    assert action_matches_pai(pair["top1_action"], pair["branch_a_dahai_pai"])
    assert action_matches_pai(pair["top2_action"], pair["branch_b_dahai_pai"])
    assert pair["branch_a_dahai_pai"] != pair["branch_b_dahai_pai"]
    assert pair["top1_action"] != pair["top2_action"]

    assert len(pair["branch_a_canonical_content_sha256"]) == 64
    assert len(pair["branch_b_canonical_content_sha256"]) == 64

    assert 0 <= pair["rank_top1"] <= 3
    assert 0 <= pair["rank_top2"] <= 3
    assert pair["pt_top1"] in TENHOU_RANK_POINTS
    assert pair["pt_top2"] in TENHOU_RANK_POINTS


def test_3_two_sample_bootstrap_and_verdict_adjudication() -> None:
    """Test 3: Two-sample rate difference bootstrap CI and formal adjudication logic."""
    bin_p3 = np.array([1.0] * 60 + [0.0] * 68)  # 60/128 = 46.875%
    bin_p2 = np.array([1.0] * 35 + [0.0] * 93)  # 35/128 = 27.344%

    diff, ci = two_sample_rate_diff_bootstrap_ci(bin_p3, bin_p2, reps=1000, seed=BOOTSTRAP_SEED)
    assert diff > 0.15
    assert ci[0] <= diff <= ci[1]

    v1 = adjudicate_p3_verdict(p3_score_nonzero_rate=float(np.mean(bin_p3)), diff_score_nonzero_rate_ci=ci)
    if ci[0] > 0:
        assert v1 == "late_decision_targets_promising"

    v2 = adjudicate_p3_verdict(p3_score_nonzero_rate=0.35, diff_score_nonzero_rate_ci=[0.02, 0.15])
    assert v2 == "counterfactual_targets_insufficiently_dense"

    v3 = adjudicate_p3_verdict(p3_score_nonzero_rate=0.42, diff_score_nonzero_rate_ci=[-0.05, 0.20])
    assert v3 == "counterfactual_targets_insufficiently_dense"


def test_4_adjudicate_p3_counterfactual_panel_mock_and_fail_closed(tmp_path: Path) -> None:
    """Test 4: Summarizer validates P3 raw logs, canonical content SHAs, P2 comparison, and produces summary."""
    panel_dir = tmp_path / "counterfactual_panel"
    summary_dir = tmp_path / "summary"
    raw_logs = panel_dir / "raw_logs"
    panel_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    raw_logs.mkdir(parents=True)

    _, k0_sha = resolve_k0_checkpoint()

    # Synthetic P2 summary file with full schema and hard gates
    p2_summary_file = tmp_path / "p2_summary.json"
    p2_summary_mock = {
        "schema": P2_SUMMARY_SCHEMA,
        "experiment_id": P2_EXPERIMENT_ID,
        "hard_gates": {g: True for g in EXPECTED_P2_SUMMARY_HARD_GATES},
        "metrics": {
            "delta_rank_point": {"total_pairs": 128, "nonzero_count": 21, "nonzero_rate": 21.0 / 128.0},
            "delta_final_score": {"total_pairs": 128, "nonzero_count": 35, "nonzero_rate": 35.0 / 128.0},
        },
    }
    p2_summary_file.write_text(json.dumps(p2_summary_mock))
    mock_p2_sha = sha256_file(p2_summary_file)

    # Generate 128 synthetic pair logs
    pairs = []
    for s in range(SEED_START, SEED_END_EXCLUSIVE):
        pair_dir = raw_logs / f"seed_{s}"
        dir_a = pair_dir / "branch_a"
        dir_b = pair_dir / "branch_b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        log_a_path = dir_a / f"{s}_8192_a.json.gz"
        log_b_path = dir_b / f"{s}_8192_a.json.gz"

        events_prefix = [
            {"type": "start_game", "names": ["K0_70k", "opp1", "opp2", "opp3"], "seed": [s, 8192]},
            {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
            {"type": "tsumo", "actor": 0, "pai": "2p"},
        ]
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

    summary = adjudicate_p3_counterfactual_panel(
        panel_dir=panel_dir,
        summary_dir=summary_dir,
        p2_summary_path=p2_summary_file,
        expected_p2_sha=mock_p2_sha,
        allowed_root=tmp_path,
    )
    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["experiment_id"] == EXPERIMENT_ID
    assert summary["hard_gates"]["p2_comparison_verified"] is True
    assert summary["hard_gates"]["canonical_content_hashes_verified"] is True
    assert summary["metrics"]["comparative_signal_density_vs_p2"]["p2_summary_sha256"] == mock_p2_sha

    # Missing log fail-closed check
    first_log_a = Path(pairs[0]["branch_a_log_path"])
    first_log_a.unlink()
    with pytest.raises(FileNotFoundError):
        adjudicate_p3_counterfactual_panel(
            panel_dir=panel_dir,
            summary_dir=summary_dir,
            p2_summary_path=p2_summary_file,
            expected_p2_sha=mock_p2_sha,
            allowed_root=tmp_path,
        )


def test_5_p2_summary_tamper_fails_closed(tmp_path: Path) -> None:
    """Test 5: Tampering with P2 summary SHA, schema, gates, or pair counts fails closed."""
    panel_dir = tmp_path / "panel"
    summary_dir = tmp_path / "summary"
    p2_summary_file = tmp_path / "p2_summary_tampered.json"
    panel_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)

    # Minimal valid manifest
    _, k0_sha = resolve_k0_checkpoint()
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
        "hard_gates": {g: True for g in EXPECTED_PANEL_HARD_GATES},
        "pairs": [{"seed": s} for s in range(SEED_START, SEED_END_EXCLUSIVE)],
        "verdict": "panel_generation_completed",
    }
    manifest_path = panel_dir / "counterfactual_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # 1. SHA mismatch fail-closed check
    p2_summary_file.write_text(json.dumps({"schema": P2_SUMMARY_SCHEMA, "experiment_id": P2_EXPERIMENT_ID}))
    with pytest.raises(ContractError, match="P2 summary SHA256 mismatch"):
        adjudicate_p3_counterfactual_panel(
            panel_dir=panel_dir,
            summary_dir=summary_dir,
            p2_summary_path=p2_summary_file,
            expected_p2_sha="0000000000000000000000000000000000000000000000000000000000000000",
            allowed_root=tmp_path,
        )

    # 2. Schema mismatch fail-closed check
    p2_summary_file.write_text(json.dumps({"schema": "invalid.schema", "experiment_id": P2_EXPERIMENT_ID}))
    with pytest.raises(ContractError, match="P2 summary schema mismatch"):
        adjudicate_p3_counterfactual_panel(
            panel_dir=panel_dir,
            summary_dir=summary_dir,
            p2_summary_path=p2_summary_file,
            expected_p2_sha=None,
            allowed_root=tmp_path,
        )

    # 3. Failing hard gate fail-closed check
    p2_tampered_gates = {g: True for g in EXPECTED_P2_SUMMARY_HARD_GATES}
    p2_tampered_gates["all_branch_logs_verified"] = False
    p2_summary_file.write_text(json.dumps({
        "schema": P2_SUMMARY_SCHEMA,
        "experiment_id": P2_EXPERIMENT_ID,
        "hard_gates": p2_tampered_gates,
    }))
    with pytest.raises(ContractError, match="P2 summary contains failing hard gate"):
        adjudicate_p3_counterfactual_panel(
            panel_dir=panel_dir,
            summary_dir=summary_dir,
            p2_summary_path=p2_summary_file,
            expected_p2_sha=None,
            allowed_root=tmp_path,
        )
