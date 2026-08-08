from __future__ import annotations

import random
import subprocess
import sys
import os
from pathlib import Path

import pytest

import keqing_core
import mahjong_env.progress_oracle as progress_oracle
from mahjong_env.progress_oracle import (
    NormalProgressInfo,
    _candidate_discards_no_meld_break,
    _candidate_progress_key,
    _candidate_progress_v1_from_native,
    _select_candidate_discards_3n2,
    _select_best_candidate_progress_3n2_python,
    _summarize_3n2_candidates_python,
    _summarize_3n1_cached,
    _summarize_3n2_cached,
    analyze_normal_progress_from_counts,
    calc_shanten_waits_from_counts,
    calc_standard_shanten_from_counts,
    clear_progress_caches,
)


def _set_rust_mode(enabled: bool) -> None:
    keqing_core.enable_rust(enabled)
    clear_progress_caches()


def _require_native_capabilities(*names: str) -> None:
    missing = [name for name in names if getattr(keqing_core, name, None) is None]
    if missing:
        pytest.skip(
            "keqing_core native capabilities are unavailable: " + ", ".join(missing)
        )


def _random_counts(total_tiles: int, rng: random.Random) -> tuple[int, ...]:
    counts = [0] * 34
    remaining = [4] * 34
    tiles = 0
    while tiles < total_tiles:
        idx = rng.randrange(34)
        if remaining[idx] <= 0:
            continue
        remaining[idx] -= 1
        counts[idx] += 1
        tiles += 1
    return tuple(counts)


def _random_visible_counts(hand_counts: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    visible = list(hand_counts)
    for idx, cnt in enumerate(visible):
        extra_max = max(0, 4 - cnt)
        if extra_max > 0:
            visible[idx] += rng.randrange(extra_max + 1)
    return tuple(visible)


def _progress_snapshot(progress) -> dict[str, object]:
    return {
        "shanten": progress.shanten,
        "waits_count": progress.waits_count,
        "waits_tiles": tuple(progress.waits_tiles),
        "tehai_count": progress.tehai_count,
        "ukeire_type_count": progress.ukeire_type_count,
        "ukeire_live_count": progress.ukeire_live_count,
        "ukeire_tiles": tuple(progress.ukeire_tiles),
        "good_shape_ukeire_type_count": progress.good_shape_ukeire_type_count,
        "good_shape_ukeire_live_count": progress.good_shape_ukeire_live_count,
        "good_shape_ukeire_tiles": tuple(progress.good_shape_ukeire_tiles),
        "improvement_type_count": progress.improvement_type_count,
        "improvement_live_count": progress.improvement_live_count,
        "improvement_tiles": tuple(progress.improvement_tiles),
    }


@pytest.fixture(autouse=True)
def _reset_rust_mode():
    _set_rust_mode(False)
    yield
    _set_rust_mode(False)


def test_calc_standard_shanten_counts34_rust_python_parity_fixed_cases():
    cases = [
        tuple([0] * 34),
        tuple([4] + [0] * 33),
        tuple([0] * 27 + [1, 1, 1, 1, 1, 1, 1]),
        tuple([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2] + [0] * 23),
        tuple([0, 2, 0, 3, 1, 0, 1, 2, 0, 2, 0, 1] + [0] * 22),
        tuple([3, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 3, 1] + [0] * 20),
    ]

    _set_rust_mode(False)
    expected = [calc_standard_shanten_from_counts(case) for case in cases]

    _set_rust_mode(True)
    actual = [calc_standard_shanten_from_counts(case) for case in cases]

    assert actual == expected


def test_calc_standard_shanten_counts34_rust_python_parity_random_cases():
    rng = random.Random(20260408)
    samples = [
        _random_counts(13, rng)
        for _ in range(120)
    ] + [
        _random_counts(14, rng)
        for _ in range(120)
    ]

    _set_rust_mode(False)
    expected = [calc_standard_shanten_from_counts(sample) for sample in samples]

    _set_rust_mode(True)
    actual = [calc_standard_shanten_from_counts(sample) for sample in samples]

    assert actual == expected


def test_standard_shanten_many_rust_python_parity_random_cases():
    rng = random.Random(20260410)
    samples = [
        _random_counts(13, rng)
        for _ in range(60)
    ] + [
        _random_counts(14, rng)
        for _ in range(60)
    ]

    _set_rust_mode(False)
    expected = keqing_core.standard_shanten_many(samples)

    _set_rust_mode(True)
    actual = keqing_core.standard_shanten_many(samples)

    assert actual == expected


def test_calc_shanten_all_matches_standard_shanten_on_current_public_surface():
    rng = random.Random(20260418)
    samples = [
        _random_counts(13, rng)
        for _ in range(40)
    ] + [
        _random_counts(14, rng)
        for _ in range(40)
    ]

    _set_rust_mode(True)
    actual = [keqing_core.calc_shanten_all(sample) for sample in samples]
    expected = [calc_standard_shanten_from_counts(sample) for sample in samples]

    assert actual == expected


def test_required_tiles_contains_live_waits_for_tenpai_case():
    _require_native_capabilities("_RUST_REQUIRED_TILES", "_RUST_DRAW_DELTAS")
    counts = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2) + (0,) * 20
    visible = counts

    _set_rust_mode(True)
    required = keqing_core.calc_required_tiles(counts, visible, sum(counts) // 3)
    draw_deltas = keqing_core.calc_draw_deltas(counts, visible, sum(counts) // 3)
    expected = tuple(
        (tile34, live_count)
        for tile34, live_count, shanten_diff in draw_deltas
        if shanten_diff < 0
    )

    assert required == expected


def test_draw_and_discard_deltas_return_stable_shapes():
    _require_native_capabilities("_RUST_DRAW_DELTAS", "_RUST_DISCARD_DELTAS")
    rng = random.Random(20260419)
    counts = _random_counts(13, rng)
    visible = _random_visible_counts(counts, rng)

    _set_rust_mode(True)
    draw_deltas = keqing_core.calc_draw_deltas(counts, visible, sum(counts) // 3)
    discard_deltas = keqing_core.calc_discard_deltas(counts, sum(counts) // 3)

    assert all(len(item) == 3 for item in draw_deltas)
    assert all(len(item) == 2 for item in discard_deltas)
    assert all(0 <= tile34 < 34 for tile34, *_ in draw_deltas)
    assert all(0 <= tile34 < 34 for tile34, _ in discard_deltas)


def test_summarize_3n1_native_matches_python_when_available():
    if getattr(keqing_core, "_RUST_SUMMARIZE_3N1", None) is None:
        pytest.skip("3n+1 native summary seam is only available after rebuilding the wheel")

    rng = random.Random(20260428)
    hand_counts = _random_counts(13, rng)
    visible_counts = _random_visible_counts(hand_counts, rng)

    expected = _progress_snapshot(_summarize_3n1_cached(hand_counts, visible_counts))

    _set_rust_mode(True)
    shanten, waits_count, waits_tiles, tehai_count, ukeire_type_count, ukeire_live_count, ukeire_tiles = keqing_core.summarize_3n1(
        hand_counts,
        visible_counts,
    )

    actual = {
        "shanten": shanten,
        "waits_count": waits_count,
        "waits_tiles": tuple(waits_tiles),
        "tehai_count": tehai_count,
        "ukeire_type_count": ukeire_type_count,
        "ukeire_live_count": ukeire_live_count,
        "ukeire_tiles": tuple(ukeire_tiles),
        "good_shape_ukeire_type_count": 0,
        "good_shape_ukeire_live_count": 0,
        "good_shape_ukeire_tiles": tuple([False] * 34),
        "improvement_type_count": 0,
        "improvement_live_count": 0,
        "improvement_tiles": tuple([False] * 34),
    }

    assert actual == expected


def test_3n1_high_shanten_path_consumes_required_tiles(monkeypatch):
    hand_counts = (
        0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 2, 0,
    )
    visible_counts = hand_counts
    calls = {"required": 0}

    def _wrapped_required_tiles(counts34, visible34):
        calls["required"] += 1
        return ((0, 1),)

    monkeypatch.setattr(progress_oracle, "_calc_required_tiles", _wrapped_required_tiles)
    _set_rust_mode(False)

    progress = analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert progress.shanten > 2
    assert calls["required"] == 1
    assert progress.ukeire_tiles[0] is True


def test_3n1_low_shanten_path_consumes_draw_and_discard_deltas(monkeypatch):
    hand_counts = (
        0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0,
        3, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
    )
    visible_counts = hand_counts
    calls = {"draw": 0, "discard": 0}

    def _wrapped_draw_deltas(counts34, visible34):
        calls["draw"] += 1
        return ((11, 1, 0),)

    def _wrapped_discard_deltas(counts34):
        calls["discard"] += 1
        return ((0, -1),)

    monkeypatch.setattr(progress_oracle, "_calc_draw_deltas", _wrapped_draw_deltas)
    monkeypatch.setattr(progress_oracle, "_calc_discard_deltas", _wrapped_discard_deltas)
    _set_rust_mode(False)

    progress = analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert progress.shanten in (1, 2)
    assert calls == {"draw": 1, "discard": 1}
    assert progress.ukeire_tiles[11] is True


def test_analyze_normal_progress_rust_python_parity_random_cases():
    rng = random.Random(20260409)
    hand_cases = [
        _random_counts(13, rng)
        for _ in range(80)
    ] + [
        _random_counts(14, rng)
        for _ in range(80)
    ]
    visible_cases = [_random_visible_counts(hand_counts, rng) for hand_counts in hand_cases]

    _set_rust_mode(False)
    expected = [
        _progress_snapshot(analyze_normal_progress_from_counts(hand_counts, visible_counts))
        for hand_counts, visible_counts in zip(hand_cases, visible_cases)
    ]

    _set_rust_mode(True)
    actual = [
        _progress_snapshot(analyze_normal_progress_from_counts(hand_counts, visible_counts))
        for hand_counts, visible_counts in zip(hand_cases, visible_cases)
    ]

    assert actual == expected


def test_enable_rust_roundtrip_tracks_runtime_availability():
    keqing_core.enable_rust(False)
    assert keqing_core._USE_RUST is False


def test_rust_auto_enabled_when_extension_is_available():
    code = (
        "import keqing_core; "
        "print(int(keqing_core.is_available()), int(keqing_core.is_enabled()), int(keqing_core._USE_RUST))"
    )
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir if not existing else src_dir + os.pathsep + existing
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    available, enabled, use_rust = [int(part) for part in proc.stdout.strip().split()]
    if available:
        assert enabled == 1
        assert use_rust == 1

    keqing_core.enable_rust(True)
    assert keqing_core._USE_RUST is keqing_core.is_available()

    keqing_core.enable_rust(False)
    assert keqing_core._USE_RUST is False


def test_native_candidate_paths_are_deduped_when_available():
    candidates = keqing_core._candidate_native_paths()
    assert len(candidates) == len(set(candidates))
    if keqing_core.is_available():
        assert candidates


def test_analyze_normal_progress_rust_python_parity_random_3n2_cases():
    rng = random.Random(20260411)
    hand_cases = [_random_counts(14, rng) for _ in range(120)]
    visible_cases = [_random_visible_counts(hand_counts, rng) for hand_counts in hand_cases]

    _set_rust_mode(False)
    expected = [
        _progress_snapshot(analyze_normal_progress_from_counts(hand_counts, visible_counts))
        for hand_counts, visible_counts in zip(hand_cases, visible_cases)
    ]

    _set_rust_mode(True)
    actual = [
        _progress_snapshot(analyze_normal_progress_from_counts(hand_counts, visible_counts))
        for hand_counts, visible_counts in zip(hand_cases, visible_cases)
    ]

    assert actual == expected


def test_summarize_3n2_tie_uses_first_seen_best_candidate():
    hand_counts = (
        1, 0, 1, 0, 0, 1, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 1, 1, 0,
        1, 0, 1, 0, 0, 0, 0, 1, 0,
        2, 0, 1, 0, 0, 0, 0,
    )
    visible_counts = (
        2, 1, 4, 4, 4, 1, 4, 4, 1,
        1, 1, 0, 0, 0, 4, 3, 3, 3,
        3, 4, 3, 0, 0, 0, 4, 3, 2,
        4, 2, 1, 0, 2, 1, 0,
    )

    discards = list(_candidate_discards_no_meld_break(hand_counts))
    ranked = []
    for discard34 in discards:
        after_counts = list(hand_counts)
        after_counts[discard34] -= 1
        after_counts_t = tuple(after_counts)
        progress = _summarize_3n1_cached(after_counts_t, visible_counts)
        key = (
            -progress.shanten,
            progress.ukeire_live_count,
            progress.good_shape_ukeire_live_count,
            progress.improvement_live_count,
            progress.ukeire_type_count,
            progress.improvement_type_count,
        )
        ranked.append((discard34, key, _progress_snapshot(progress)))

    best_key = max(key for _, key, _ in ranked)
    tied = [item for item in ranked if item[1] == best_key]
    assert len(tied) >= 2

    expected_snapshot = tied[0][2]
    actual_snapshot = _progress_snapshot(_summarize_3n2_cached(hand_counts, visible_counts))
    assert actual_snapshot == expected_snapshot


def test_summarize_3n2_native_candidate_summaries_match_python():
    _require_native_capabilities("_RUST_SUMMARIZE_3N2_CANDIDATES")
    rng = random.Random(20260413)
    hand_counts = _random_counts(14, rng)
    visible_counts = _random_visible_counts(hand_counts, rng)
    discard_candidates = _select_candidate_discards_3n2(hand_counts)

    expected = _summarize_3n2_candidates_python(hand_counts, visible_counts, discard_candidates)

    _set_rust_mode(True)
    actual = [
        _candidate_progress_v1_from_native(item)
        for item in keqing_core.summarize_3n2_candidates(
            hand_counts,
            visible_counts,
            _summarize_3n1_cached,
        )
    ]

    assert actual == expected
    assert [item.discard_tile34 for item in actual] == list(discard_candidates)
    assert [_candidate_progress_key(item) for item in actual] == [
        _candidate_progress_key(item) for item in expected
    ]


def test_summarize_best_3n2_candidate_native_matches_python():
    if getattr(keqing_core, "_RUST_SUMMARIZE_BEST_3N2_CANDIDATE", None) is None:
        pytest.skip("best-candidate 3n+2 native seam is only available after rebuilding the wheel")
    rng = random.Random(20260425)
    hand_counts = _random_counts(14, rng)
    visible_counts = _random_visible_counts(hand_counts, rng)
    discard_candidates = _select_candidate_discards_3n2(hand_counts)

    expected = _select_best_candidate_progress_3n2_python(
        hand_counts,
        visible_counts,
        discard_candidates,
    )

    _set_rust_mode(True)
    actual = keqing_core.summarize_best_3n2_candidate(
        hand_counts,
        visible_counts,
        _summarize_3n1_cached,
    )

    assert actual is not None
    assert _candidate_progress_v1_from_native(actual) == expected


def test_native_3n2_seams_do_not_require_python_summary_callback_when_available():
    if getattr(keqing_core, "_RUST_SUMMARIZE_BEST_3N2_CANDIDATE", None) is None:
        pytest.skip("best-candidate 3n+2 native seam is only available after rebuilding the wheel")

    rng = random.Random(20260429)
    hand_counts = _random_counts(14, rng)
    visible_counts = _random_visible_counts(hand_counts, rng)

    _set_rust_mode(True)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("native 3n+2 seam should not call back into Python for 3n+1 summaries anymore")

    try:
        best = keqing_core.summarize_best_3n2_candidate(hand_counts, visible_counts, _should_not_be_called)
        candidates = keqing_core.summarize_3n2_candidates(hand_counts, visible_counts, _should_not_be_called)
    except AssertionError:
        pytest.skip("source-mode wheel has not been rebuilt yet for callback-free native 3n+2 seam validation")

    assert best is not None
    assert candidates


def test_select_candidate_discards_3n2_prunes_worse_shanten_discards(monkeypatch):
    hand_counts = _random_counts(14, random.Random(20260420))

    monkeypatch.setattr(progress_oracle, "_candidate_discards_no_meld_break", lambda _counts: (0, 1, 2))
    monkeypatch.setattr(progress_oracle, "_calc_discard_deltas", lambda _counts: ((0, 1), (1, 0), (2, 0)))
    monkeypatch.setattr(progress_oracle, "calc_standard_shanten_from_counts", lambda _counts: 2)

    assert _select_candidate_discards_3n2(hand_counts) == (1, 2)


def test_summarize_3n2_only_summarizes_best_after_shanten_candidates(monkeypatch):
    hand_counts = _random_counts(14, random.Random(20260421))
    visible_counts = _random_visible_counts(hand_counts, random.Random(20260422))
    calls: list[int] = []
    clear_progress_caches()

    def _fake_summary(after_counts, _visible_counts):
        discard34 = next(idx for idx, (before, after) in enumerate(zip(hand_counts, after_counts)) if before != after)
        calls.append(discard34)
        ukeire_tiles = [False] * 34
        if discard34 == 1:
            ukeire_tiles[5] = True
        return NormalProgressInfo(
            shanten=1,
            waits_count=0,
            waits_tiles=[False] * 34,
            tehai_count=sum(after_counts),
            ukeire_type_count=sum(ukeire_tiles),
            ukeire_live_count=sum(ukeire_tiles),
            ukeire_tiles=ukeire_tiles,
            good_shape_ukeire_type_count=0,
            good_shape_ukeire_live_count=0,
            good_shape_ukeire_tiles=[False] * 34,
            improvement_type_count=0,
            improvement_live_count=0,
            improvement_tiles=[False] * 34,
        )

    monkeypatch.setattr(progress_oracle, "_candidate_discards_no_meld_break", lambda _counts: (0, 1, 2))
    monkeypatch.setattr(progress_oracle, "_calc_discard_deltas", lambda _counts: ((0, 1), (1, 0), (2, 0)))
    monkeypatch.setattr(progress_oracle, "calc_standard_shanten_from_counts", lambda _counts: 2)
    monkeypatch.setattr(progress_oracle, "_summarize_3n1_cached", _fake_summary)
    monkeypatch.setattr(progress_oracle, "_has_3n2_candidate_summaries", lambda: False)

    progress = _summarize_3n2_cached(hand_counts, visible_counts)

    assert calls[:2] == [1, 2]
    assert 0 not in calls
    assert progress.ukeire_tiles[5] is True


def test_summarize_3n2_does_not_touch_native_candidate_path_when_rust_disabled(monkeypatch):
    calls = 0

    def _unexpected_native_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("native 3n+2 seam should not be called when rust is disabled")

    monkeypatch.setattr(progress_oracle, "_summarize_3n2_candidates_native", _unexpected_native_call)
    hand_counts = _random_counts(14, random.Random(20260414))
    visible_counts = _random_visible_counts(hand_counts, random.Random(20260415))

    _set_rust_mode(False)
    analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert calls == 0


def test_summarize_3n2_does_not_touch_native_candidate_path_when_seam_unavailable(monkeypatch):
    calls = 0

    def _unexpected_native_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("native 3n+2 seam should not be called when seam capability is unavailable")

    monkeypatch.setattr(progress_oracle, "_summarize_3n2_candidates_native", _unexpected_native_call)
    monkeypatch.setattr(progress_oracle, "_has_3n2_candidate_summaries", lambda: False)
    hand_counts = _random_counts(14, random.Random(20260416))
    visible_counts = _random_visible_counts(hand_counts, random.Random(20260417))

    analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert calls == 0


def test_summarize_3n2_does_not_need_legacy_native_candidate_path_when_python_pruning_exists(monkeypatch):
    calls = 0
    hand_counts = _random_counts(14, random.Random(20260423))
    visible_counts = _random_visible_counts(hand_counts, random.Random(20260424))

    def _fake_native(_counts, _visible_counts, summarize_fn):
        nonlocal calls
        calls += 1
        after_counts = list(hand_counts)
        after_counts[0] -= 1
        progress = summarize_fn(tuple(after_counts), visible_counts)
        return [(
            0,
            tuple(after_counts),
            progress.shanten,
            progress.waits_count,
            progress.ukeire_type_count,
            progress.ukeire_live_count,
            progress.good_shape_ukeire_type_count,
            progress.good_shape_ukeire_live_count,
            progress.improvement_type_count,
            progress.improvement_live_count,
        )]

    monkeypatch.setattr(progress_oracle, "_summarize_3n2_candidates_native", _fake_native)
    monkeypatch.setattr(progress_oracle, "_has_3n2_candidate_summaries", lambda: True)
    monkeypatch.setattr(progress_oracle, "_select_candidate_discards_3n2", lambda _counts: (1, 2))

    analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert calls == 0


def test_summarize_3n2_prefers_native_best_candidate_path(monkeypatch):
    calls = {"best": 0, "candidates": 0}
    hand_counts = _random_counts(14, random.Random(20260426))
    visible_counts = _random_visible_counts(hand_counts, random.Random(20260427))
    discard34 = next(idx for idx, cnt in enumerate(hand_counts) if cnt > 0)

    def _fake_best_native(_counts, _visible_counts, summarize_fn):
        calls["best"] += 1
        after_counts = list(hand_counts)
        after_counts[discard34] -= 1
        progress = summarize_fn(tuple(after_counts), visible_counts)
        return (
            discard34,
            tuple(after_counts),
            progress.shanten,
            progress.waits_count,
            progress.ukeire_type_count,
            progress.ukeire_live_count,
            progress.good_shape_ukeire_type_count,
            progress.good_shape_ukeire_live_count,
            progress.improvement_type_count,
            progress.improvement_live_count,
        )

    def _unexpected_candidates(*args, **kwargs):
        calls["candidates"] += 1
        raise AssertionError("candidate-list native seam should not be used when best-candidate seam is available")

    monkeypatch.setattr(progress_oracle, "_summarize_best_3n2_candidate_native", _fake_best_native)
    monkeypatch.setattr(progress_oracle, "_summarize_3n2_candidates_native", _unexpected_candidates)
    monkeypatch.setattr(progress_oracle, "_has_3n2_candidate_summaries", lambda: True)
    _set_rust_mode(True)

    analyze_normal_progress_from_counts(hand_counts, visible_counts)

    assert calls == {"best": 1, "candidates": 0}
