from __future__ import annotations

from pathlib import Path

from training.mortal.audit_d3_generation_aggregate_2026_08 import (
    latest_ledger_entries,
    load_ledger,
    shard_audit_dir,
    shard_seed_start_end,
)
from training.mortal.d3_continuation_contract import (
    SHARD_COUNT,
    shard_dir_name,
    shard_seed_end_exclusive,
    shard_seed_start,
)
from training.mortal.d3_production_contract import expected_seed_keys


def test_aggregate_shard_grid_covers_exact_6000_seed_range() -> None:
    b250 = expected_seed_keys()
    assert {seed for seed, _ in b250} == set(range(1_800_000, 1_800_250))
    all_seeds = set()
    for index in range(1, SHARD_COUNT + 1):
        all_seeds.update(range(shard_seed_start(index), shard_seed_end_exclusive(index)))
    assert len(all_seeds) == 5750
    global_seeds = {seed for seed, _ in b250} | all_seeds
    assert global_seeds == set(range(1_800_000, 1_806_000))
    assert len(global_seeds) == 6000


def test_aggregate_shard_seed_start_end_is_inclusive() -> None:
    assert shard_seed_start_end(0) == (1_800_000, 1_800_249)
    assert shard_seed_start_end(1) == (1_800_250, 1_800_499)
    assert shard_seed_start_end(23) == (1_805_750, 1_805_999)


def test_aggregate_shard_audit_dirs() -> None:
    assert shard_audit_dir(0).name == "shard_000_1800000_1800249"
    assert shard_audit_dir(5).name == "shard_005_1801250_1801499"
    assert "generation_production" in str(shard_audit_dir(0))
    assert "generation_continuation" in str(shard_audit_dir(1))


def test_ledger_load_and_latest_pass_entries(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                '{"shard_index":1,"verdict":"FAIL","seed_start":1800250}',
                '{"shard_index":1,"verdict":"PASS","seed_start":1800250,"auditor_commit":"a"}',
                '{"shard_index":2,"verdict":"PASS","seed_start":1800500,"auditor_commit":"b"}',
                '{"shard_index":1,"verdict":"PASS","seed_start":1800250,"auditor_commit":"c"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_ledger(ledger)
    assert len(rows) == 4
    latest = latest_ledger_entries(ledger)
    assert set(latest) == {1, 2}
    assert latest[1]["auditor_commit"] == "c"  # last PASS wins
    assert latest[2]["auditor_commit"] == "b"
