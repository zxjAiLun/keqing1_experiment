"""Regression tests for relabel manifest snapshot/resume sequencing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


class _FakeGame:
    def __init__(self, marker: int, rows: int = 3) -> None:
        self._obs = np.full((rows, 1, 34), marker, dtype=np.float32)
        self._masks = np.ones((rows, 46), dtype=bool)
        self._actions = np.full(rows, marker % 46, dtype=np.int64)

    def take_obs(self) -> np.ndarray:
        return self._obs

    def take_masks(self) -> np.ndarray:
        return self._masks

    def take_actions(self) -> np.ndarray:
        return self._actions

    def take_player_id(self) -> int:
        return 2


class _FakeBrain(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class _FakeDQN(torch.nn.Module):
    def forward(self, _phi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros(mask.shape, dtype=torch.float32, device=mask.device).masked_fill(~mask, -torch.inf)


def _payload_sequence(output_dir: Path) -> dict[str, np.ndarray]:
    fields = ("obs", "mask", "teacher_action", "behavior_action", "pool", "player_id", "file_row")
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for shard in sorted(output_dir.glob("train_*.npz")):
        with np.load(shard) as payload:
            for field in fields:
                chunks[field].append(payload[field])
    return {field: np.concatenate(values) for field, values in chunks.items()}


def _args(source_dir: Path, output_dir: Path, teacher: Path, *, resume: bool) -> SimpleNamespace:
    return SimpleNamespace(
        teacher=teacher,
        mortal_root=Path("unused"),
        output_dir=output_dir,
        pool=[f"S0={source_dir}"],
        holdout_ratio=0.0,
        holdout_salt="relabel-resume-regression",
        rows_per_shard=64,
        # Three rows per source file keeps periodic snapshots below this batch
        # size: the pre-fix implementation would mark the file complete while
        # all of its rows were still buffered.
        inference_batch=8,
        enable_amp=False,
        device="cpu",
        limit_files=0,
        manifest_snapshot_every=1,
        resume=resume,
    )


def test_snapshot_resume_preserves_buffered_source_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A snapshot interruption must equal an uninterrupted payload sequence.

    Source files use distinct observation markers.  This compares the complete
    emitted payload (obs, masks, labels, and metadata), rather than the
    non-unique ``(pool, player_id, file_row)`` tuple alone.
    """
    from training.mortal import relabel_ext_teacher as relabel

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for marker in (11, 22, 33):
        (source_dir / f"game_{marker}.json.gz").write_bytes(str(marker).encode())
    teacher = tmp_path / "teacher.pth"
    teacher.write_bytes(b"synthetic teacher identity")

    fake_dataset = ModuleType("libriichi.dataset")

    class FakeGameplayLoader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load_gz_log_files(self, paths: list[str]) -> list[list[_FakeGame]]:
            marker = int(Path(paths[0]).name.removesuffix(".json.gz").split("_")[-1])
            return [[_FakeGame(marker)]]

    fake_dataset.GameplayLoader = FakeGameplayLoader
    monkeypatch.setitem(sys.modules, "libriichi.dataset", fake_dataset)
    monkeypatch.setattr(relabel, "_load_teacher", lambda *_args: (_FakeBrain(), _FakeDQN(), 4))

    continuous_dir = tmp_path / "continuous"
    continuous = relabel.run(_args(source_dir, continuous_dir, teacher, resume=False))

    resumed_dir = tmp_path / "resumed"
    original_write_manifest = relabel._write_manifest
    snapshots = 0

    def interrupt_after_first_snapshot(*args: object, **kwargs: object) -> None:
        nonlocal snapshots
        original_write_manifest(*args, **kwargs)
        snapshots += 1
        if snapshots == 1:
            raise RuntimeError("simulated process interruption after manifest snapshot")

    monkeypatch.setattr(relabel, "_write_manifest", interrupt_after_first_snapshot)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        relabel.run(_args(source_dir, resumed_dir, teacher, resume=False))
    assert json.loads((resumed_dir / "manifest.json").read_text(encoding="utf-8"))["files"]

    monkeypatch.setattr(relabel, "_write_manifest", original_write_manifest)
    resumed = relabel.run(_args(source_dir, resumed_dir, teacher, resume=True))

    assert continuous["totals"]["rows"] == resumed["totals"]["rows"] == 9
    assert continuous["totals"]["hanchans"] == resumed["totals"]["hanchans"] == 3
    expected = _payload_sequence(continuous_dir)
    actual = _payload_sequence(resumed_dir)
    assert actual.keys() == expected.keys()
    for field in expected:
        assert np.array_equal(actual[field], expected[field]), field
    assert actual["obs"][:, 0, 0].tolist() == [11.0] * 3 + [22.0] * 3 + [33.0] * 3
