from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from training.mortal import audit_c1_loader_compatibility_2026_08 as audit


def _config() -> dict:
    return {
        "control": {"version": 4, "batch_size": 2},
        "dataset": {
            "file_batch_size": 1,
            "reserve_ratio": 0.0,
            "num_epochs": 1,
            "enable_augmentation": False,
            "augmented_first": False,
            "num_workers": 0,
        },
        "env": {"pts": [6.0, 4.0, 2.0, 0.0]},
    }


class _SyntheticLoader(IterableDataset):
    offset = 0

    def __init__(self, **kwargs) -> None:
        del kwargs

    def __iter__(self):
        for index in range(4):
            value = index + self.offset
            yield (
                torch.full((3,), value, dtype=torch.float64),
                torch.tensor(value, dtype=torch.int32),
                torch.tensor([True, False, True], dtype=torch.uint8),
                torch.tensor(value, dtype=torch.int32),
                torch.tensor(float(value), dtype=torch.float32),
                torch.tensor(value, dtype=torch.int32),
            )


class _SyntheticCurrentLoader(_SyntheticLoader):
    offset = 1


def test_canonical_batch_uses_runner_dtypes_and_contiguous_bytes(monkeypatch) -> None:
    monkeypatch.setattr(audit, "BATCH_SIZE", 2)
    batch = audit.canonical_batch(
        (
            torch.ones(2, 3, dtype=torch.float64),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.uint8),
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int32),
        )
    )
    assert [value.dtype for value in batch] == list(audit.CANONICAL_DTYPES)
    assert all(value.is_contiguous() for value in batch)
    assert audit.hash_batch(batch) == audit.hash_batch(batch)


def test_first_batch_mismatch_reports_tensor() -> None:
    left = tuple(torch.zeros(2) for _ in audit.CANONICAL_NAMES)
    right = tuple(torch.zeros(2) for _ in audit.CANONICAL_NAMES)
    right[2][0] = 1
    assert audit.first_batch_mismatch(left, right) == "masks.bytes"


def test_synthetic_stream_pair_compares_complete_ordered_stream(monkeypatch) -> None:
    monkeypatch.setattr(audit, "BATCHES", 2)
    monkeypatch.setattr(audit, "BATCH_SIZE", 2)
    monkeypatch.setattr(audit, "EXPECTED_SAMPLES", 4)
    result = audit.run_stream_pair(
        historical_loader=_SyntheticLoader,
        current_loader=_SyntheticLoader,
        config=_config(),
        file_list=["synthetic"],
        labels=["synthetic"],
        seed=123,
    )
    assert result["batches_compared"] == 2
    assert result["samples_compared"] == 4
    assert result["exact_match"] is True
    assert result["first_mismatch_batch"] is None
    assert result["first_mismatch_tensor"] is None


def test_synthetic_stream_pair_finds_late_batch_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(audit, "BATCHES", 2)
    monkeypatch.setattr(audit, "BATCH_SIZE", 2)
    monkeypatch.setattr(audit, "EXPECTED_SAMPLES", 4)
    result = audit.run_stream_pair(
        historical_loader=_SyntheticLoader,
        current_loader=_SyntheticCurrentLoader,
        config=_config(),
        file_list=["synthetic"],
        labels=["synthetic"],
        seed=123,
    )
    assert result["exact_match"] is False
    assert result["first_mismatch_batch"] == 0
    assert result["first_mismatch_tensor"] == "obs"


def test_separate_stream_comparator_locates_component_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(audit, "BATCHES", 2)
    monkeypatch.setattr(audit, "EXPECTED_SAMPLES", 4)
    same_hash = "00" * 32
    historical_hashes = [same_hash, "11" * 32]
    current_hashes = [same_hash, "22" * 32]
    historical_components = [["same"] * 6, ["h"] * 6]
    current_components = [["same"] * 6, ["h", "different", "h", "h", "h", "h"]]

    result = audit.compare_stream_hashes(
        historical_hashes=historical_hashes,
        current_hashes=current_hashes,
        historical_components=historical_components,
        current_components=current_components,
    )

    assert result["exact_match"] is False
    assert result["first_mismatch_batch"] == 1
    assert result["first_mismatch_tensor"] == "actions"
