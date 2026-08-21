"""Targeted tests for C2 CQL calibration and gradient attribution audit.

Tests cover:
- Centered target rejects uncentered [6,4,2,0]
- Shared-panel interaction formula
- Calibration vote logic
- Gradient conflict vote logic
- Deterministic bootstrap
- Machine verdict selection
- Gate failure handling
"""
import numpy as np
import pytest
import torch

from training.mortal.c2_cql_mechanism_core import (
    LEGAL_TARGET_SET,
    bootstrap_ci95,
    check_target_values,
    compute_delta_and_interaction,
    determine_verdict,
    gradient_conflict_vote,
    q_calibration_vote,
)


def test_centered_target_rejects_uncentered_ranks() -> None:
    valid_targets = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
    assert check_target_values(valid_targets) is True

    uncentered_targets = np.array([6.0, 4.0, 2.0, 0.0], dtype=np.float64)
    assert check_target_values(uncentered_targets) is False

    invalid_mixed = np.array([-3.0, 0.0, 1.0, 3.0], dtype=np.float64)
    assert check_target_values(invalid_mixed) is False


def test_delta_and_interaction_formula() -> None:
    # Delta_R(m) = m(R, OFF) - m(R, CURRENT)
    # I(m) = Delta_D1 - Delta_M0
    m_d1_off = 1.0
    m_d1_curr = 2.5
    m_m0_off = 1.2
    m_m0_curr = 1.5

    delta_d1, delta_m0, interaction = compute_delta_and_interaction(
        m_d1_off, m_d1_curr, m_m0_off, m_m0_curr
    )
    assert pytest.approx(delta_d1) == 1.0 - 2.5  # -1.5
    assert pytest.approx(delta_m0) == 1.2 - 1.5  # -0.3
    assert pytest.approx(interaction) == (-1.5) - (-0.3)  # -1.2


def test_calibration_vote_logic() -> None:
    # Pass case
    pass_res = q_calibration_vote(
        i_abs_residual_seeds=(-0.2, -0.15, -0.3),
        ci_abs_residual=(-0.35, -0.05),
        i_overestimate_seeds=(-0.05, -0.08, -0.04),
        ci_overestimate=(-0.10, -0.01),
    )
    assert pass_res is True

    # Fail case: one seed >= 0
    fail_res1 = q_calibration_vote(
        i_abs_residual_seeds=(-0.2, 0.01, -0.3),
        ci_abs_residual=(-0.35, -0.05),
        i_overestimate_seeds=(-0.05, -0.08, -0.04),
        ci_overestimate=(-0.10, -0.01),
    )
    assert fail_res1 is False

    # Fail case: CI upper >= 0
    fail_res2 = q_calibration_vote(
        i_abs_residual_seeds=(-0.2, -0.15, -0.3),
        ci_abs_residual=(-0.35, 0.02),
        i_overestimate_seeds=(-0.05, -0.08, -0.04),
        ci_overestimate=(-0.10, -0.01),
    )
    assert fail_res2 is False


def test_gradient_conflict_vote_logic() -> None:
    # Pass case: cosine interaction > 0, conflict rate interaction < 0
    pass_res = gradient_conflict_vote(
        i_cosine_seeds=(0.15, 0.20, 0.12),
        ci_cosine=(0.05, 0.25),
        i_conflict_seeds=(-0.10, -0.15, -0.08),
        ci_conflict=(-0.20, -0.02),
    )
    assert pass_res is True

    # Fail case: cosine CI lower <= 0
    fail_res1 = gradient_conflict_vote(
        i_cosine_seeds=(0.15, 0.20, 0.12),
        ci_cosine=(-0.01, 0.25),
        i_conflict_seeds=(-0.10, -0.15, -0.08),
        ci_conflict=(-0.20, -0.02),
    )
    assert fail_res1 is False


def test_deterministic_bootstrap_ci95() -> None:
    rng = np.random.default_rng(20260824)
    data = rng.normal(loc=0.0, scale=1.0, size=5000)
    low1, high1 = bootstrap_ci95(data)
    low2, high2 = bootstrap_ci95(data)
    assert low1 == low2
    assert high1 == high2
    assert low1 < 0.0 < high1


def test_determine_verdict_mapping() -> None:
    assert determine_verdict(gates_pass=False, calib_pass=True, grad_pass=True) == "no_verdict_gates_failed"
    assert determine_verdict(gates_pass=True, calib_pass=True, grad_pass=True) == "calibration_and_gradient_signal"
    assert determine_verdict(gates_pass=True, calib_pass=True, grad_pass=False) == "q_calibration_signal"
    assert determine_verdict(gates_pass=True, calib_pass=False, grad_pass=True) == "parameter_gradient_conflict_signal"
    assert determine_verdict(gates_pass=True, calib_pass=False, grad_pass=False) == "inconclusive"
