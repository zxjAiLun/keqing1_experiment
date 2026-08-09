from __future__ import annotations

import pytest

from training.mortal.d3_continuation_contract import (
    CONTINUATION_GOVERNANCE,
    D3_SEMANTIC_ANCHOR,
    FIRST_GATE_AUDITOR,
    GAMES_PER_SHARD,
    PRODUCTION_SEED_START,
    SEED_KEY,
    SHARD_COUNT,
    continuation_lineage,
    shard_confirmation_token,
    shard_dir_name,
    shard_output_dir,
    shard_seed_end_exclusive,
    shard_seed_keys,
    shard_seed_start,
)
from training.mortal.d3_production_contract import expected_seed_keys
from training.mortal.run_d3_continuation_shard_2026_08 import (
    parse_args as parse_continuation_args,
)


def test_continuation_shard_grid_is_exactly_23_x_250() -> None:
    assert SHARD_COUNT == 23
    assert GAMES_PER_SHARD == 250
    assert 23 * 250 == 5750
    assert shard_seed_start(1) == 1_800_250
    assert shard_seed_end_exclusive(1) == 1_800_500
    assert shard_seed_start(23) == 1_805_750
    assert shard_seed_end_exclusive(23) == 1_806_000


def test_continuation_seeds_are_contiguous_and_disjoint_from_b250() -> None:
    b250 = expected_seed_keys()
    all_continuation = set()
    for index in range(1, SHARD_COUNT + 1):
        keys = shard_seed_keys(index)
        assert len(keys) == 250
        assert all(key[1] == SEED_KEY for key in keys)
        assert keys.isdisjoint(b250)
        assert all(1_800_250 <= seed <= 1_805_999 for seed, _ in keys)
        all_continuation |= keys
    assert len(all_continuation) == 5750
    seeds = sorted(seed for seed, _ in all_continuation)
    assert seeds[0] == 1_800_250
    assert seeds[-1] == 1_805_999
    assert seeds == list(range(1_800_250, 1_806_000))


def test_continuation_dir_names_and_tokens_are_shard_bound() -> None:
    assert shard_dir_name(1) == "shard_001_1800250_1800499"
    assert shard_dir_name(23) == "shard_023_1805750_1805999"
    assert shard_output_dir(1).as_posix() == (
        "artifacts/experiments/model_pool_2026_07/"
        "D3_uncertainty_guided_exploration_2026_08/generation_continuation/"
        "shard_001_1800250_1800499"
    )
    assert (
        shard_confirmation_token(1)
        == "D3_CONTINUE_SHARD_001_1800250_1800499_SINGLE_SHOT"
    )
    assert (
        shard_confirmation_token(23)
        == "D3_CONTINUE_SHARD_023_1805750_1805999_SINGLE_SHOT"
    )


def test_continuation_shard_index_bounds() -> None:
    for bad in (0, 24, -1, 100):
        with pytest.raises(ValueError):
            shard_seed_start(bad)
    with pytest.raises(ValueError):
        shard_confirmation_token(0)


def test_continuation_lineage_anchors_are_frozen() -> None:
    assert D3_SEMANTIC_ANCHOR == "2cc12b46f81850da11c6e669d1c54b039476b440"
    assert FIRST_GATE_AUDITOR == "cf9bb86a40e1e52e24deea8d5b2af8ab12e1a63b"
    assert CONTINUATION_GOVERNANCE == "67f2ccb96fe932abc5c2c4b889ad396d4f584823"


def test_continuation_lineage_passes_on_current_main() -> None:
    lineage = continuation_lineage()
    assert lineage["branch"] == "main"
    assert lineage["governance_is_ancestor"] is True
    assert lineage["semantic_diff_paths"] == []
    assert lineage["passed"] is True, lineage["errors"]


def test_continuation_runner_requires_shard_index_only() -> None:
    args = parse_continuation_args(["--shard-index", "3"])
    assert args.shard_index == 3
    assert args.execute is False
    with pytest.raises(SystemExit):
        parse_continuation_args([])
    with pytest.raises(SystemExit):
        parse_continuation_args(["--shard-index", "0"])
    with pytest.raises(SystemExit):
        parse_continuation_args(["--shard-index", "99"])
    with pytest.raises(SystemExit):
        parse_continuation_args(["--shard-index", "3", "--seed-start", "1800000"])
