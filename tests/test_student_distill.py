"""Targeted tests for the soft-target distillation stage.

Covers the two things that would silently corrupt the candidate or its
bookkeeping: the masked KL objective (including the ``0 * -inf`` NaN trap and
illegal-action exclusion) and the stage's fail-closed recipe identity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _legal_mask(rows: int, legal: list[list[int]], width: int = 46) -> torch.Tensor:
    mask = torch.zeros(rows, width, dtype=torch.bool)
    for row, indices in enumerate(legal):
        for index in indices:
            mask[row, index] = True
    return mask


def test_kl_is_zero_when_student_matches_teacher() -> None:
    from training.mortal.train_student_distill import _kl_loss

    teacher_q = torch.tensor([[1.0, 0.5, -2.0, -9.0], [3.0, 1.0, 0.0, -1.0]])
    mask = _legal_mask(2, [[0, 1, 2], [0, 1, 2, 3]], width=4)
    legal_q = teacher_q.clone()
    # a constant shift does not change a softmax, so a shifted student still matches
    student_q = teacher_q + 5.0
    loss = _kl_loss(student_q=student_q, teacher_q=legal_q, legal=mask, temperature=1.0)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)
    # ... but the objective is not scale invariant: rescaling the student logits
    # changes its distribution away from the teacher's
    scaled_loss = _kl_loss(student_q=teacher_q * 2.0, teacher_q=legal_q, legal=mask, temperature=1.0)
    assert float(scaled_loss) > 1e-6


def test_kl_ignores_illegal_actions_without_nan() -> None:
    from training.mortal.train_student_distill import _kl_loss, _kl_per_row

    # teacher_q carries -inf on illegal actions, as the relabel cache stores it
    raw_teacher = torch.tensor([[1.0, 0.5, -float("inf"), -float("inf")]])
    mask = _legal_mask(1, [[0, 1]], width=4)
    # a student that puts huge but still finite mass on illegal logits must be
    # unaffected by them: masking happens before the softmax
    student_q = torch.tensor([[0.9, 1.0, 50.0, 60.0]])
    loss = _kl_loss(student_q=student_q, teacher_q=raw_teacher, legal=mask, temperature=1.0)
    assert torch.isfinite(loss)
    per_row = _kl_per_row(student_q=student_q, teacher_q=raw_teacher, legal=mask, temperature=1.0)
    p_teacher = torch.softmax(torch.tensor([1.0, 0.5]), dim=-1)
    p_student = torch.softmax(torch.tensor([0.9, 1.0]), dim=-1)
    expected = (p_teacher * (p_teacher.log() - p_student.log())).sum()
    assert torch.allclose(per_row, expected.unsqueeze(0), atol=1e-6)
    # the illegal logits must not move the student distribution at all
    shifted = torch.tensor([[0.9, 1.0, -50.0, -60.0]])
    assert torch.allclose(
        _kl_per_row(student_q=student_q, teacher_q=raw_teacher, legal=mask, temperature=1.0),
        _kl_per_row(student_q=shifted, teacher_q=raw_teacher, legal=mask, temperature=1.0),
        atol=1e-6,
    )


def test_temperature_sharpens_and_rejects_single_legal_action() -> None:
    from training.mortal.train_student_distill import _kl_loss

    teacher_q = torch.tensor([[2.0, 0.0, -1.0]])
    mask = _legal_mask(1, [[0, 1, 2]], width=3)
    student_q = torch.tensor([[0.0, 0.0, 0.0]])
    uniform_kl = float(_kl_loss(student_q=student_q, teacher_q=teacher_q, legal=mask, temperature=1.0))
    # a lower temperature concentrates the teacher, so the same student is worse
    sharp_kl = float(_kl_loss(student_q=student_q, teacher_q=teacher_q, legal=mask, temperature=0.5))
    assert sharp_kl > uniform_kl
    # one legal action carries no information and must give exactly zero
    single = _legal_mask(1, [[1]], width=3)
    assert float(_kl_loss(student_q=student_q, teacher_q=teacher_q, legal=single, temperature=1.0)) == pytest.approx(0.0)


def test_recipe_identity_fails_closed(tmp_path: Path) -> None:
    """A changed hyper-parameter must not be able to resume an existing stage."""
    from training.mortal.train_student_distill import _check_resume

    recorded = {
        "schema": "keqing.mortal.student_distill.v1",
        "parent_sha256": "abc",
        "temperature": {"teacher": 1.0, "student": 1.0},
        "optim": {"lr": 1e-5, "warmup_steps": 100, "target_steps": 5000},
    }
    recipe_path = tmp_path / "distill_recipe.json"
    recipe_path.write_text(json.dumps(recorded), encoding="utf-8")

    # no state on disk: a fresh run is always allowed, resume flag or not
    _check_resume(recipe_path, recorded, state_exists=False, resume=True)
    # identical recipe: resume is allowed
    _check_resume(recipe_path, recorded, state_exists=True, resume=True)
    # existing state without --resume must not silently overwrite the stage
    with pytest.raises(RuntimeError, match="pass --resume"):
        _check_resume(recipe_path, recorded, state_exists=True, resume=False)

    for key, changed in (
        ("lr", 1e-4),
        ("warmup_steps", 0),
        ("target_steps", 10_000),
    ):
        mutated = json.loads(json.dumps(recorded))
        mutated["optim"][key] = changed
        with pytest.raises(RuntimeError, match="reflect.*|refusing --resume"):
            _check_resume(recipe_path, mutated, state_exists=True, resume=True)

    for mutated in (
        {**recorded, "parent_sha256": "def"},
        {**recorded, "temperature": {"teacher": 0.5, "student": 1.0}},
    ):
        with pytest.raises(RuntimeError, match="refusing --resume"):
            _check_resume(recipe_path, mutated, state_exists=True, resume=True)


def test_stage_checkpoint_contract_is_readable_by_shared_tooling() -> None:
    """Regression: stage checkpoints must be loadable by every consumer.

    The native arena, the holdout evaluation, and the supervision-gap diagnostic
    all read architecture through ``four_player_native._model_dimensions``. A
    stage checkpoint without that contract is unusable in evaluation, which is
    how the first saved stage files failed.
    """
    from training.mortal.four_player_native import _model_dimensions
    from training.mortal.train_student_distill import _stage_contract

    recipe = {
        "objective": "kl_teacher_soft_target_only",
        "parent_checkpoint": "parent.pth",
        "parent_sha256": "deadbeef",
        "parent_steps": 50000,
        "student": {"version": 4, "conv_channels": 192, "num_blocks": 40},
        "temperature": {"teacher": 1.0, "student": 1.0},
        "optim": {"lr": 1e-5},
        "git_commit": None,
    }
    contract = _stage_contract(recipe=recipe, dataset_manifest={"train_rows": 1, "holdout_rows": 1})
    assert _model_dimensions({"training_contract": contract}) == (4, 192, 40)
    # the parent stage's own hard-label contract must keep working unchanged
    assert _model_dimensions({
        "training_contract": {
            "schema": "keqing.mortal.student_policy_v1",
            "student": {"version": 4, "conv_channels": 192, "num_blocks": 40},
        }
    }) == (4, 192, 40)
