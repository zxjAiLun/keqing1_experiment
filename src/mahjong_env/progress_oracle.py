from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from keqing_core import calc_discard_deltas as _calc_discard_deltas_native
from keqing_core import calc_draw_deltas as _calc_draw_deltas_native
from keqing_core import calc_required_tiles as _calc_required_tiles_native
from keqing_core import has_3n2_candidate_summaries as _has_3n2_candidate_summaries
from keqing_core import calc_shanten_normal as _calc_shanten_normal_native
from keqing_core import summarize_one_shanten_draw_metrics as _summarize_one_shanten_draw_metrics_native
from keqing_core import calc_standard_shanten as _calc_standard_shanten_native
from keqing_core import summarize_3n1 as _summarize_3n1_native
from keqing_core import summarize_best_3n2_candidate as _summarize_best_3n2_candidate_native
from keqing_core import summarize_3n2_candidates as _summarize_3n2_candidates_native
from keqing_core import standard_shanten_many as _standard_shanten_many_native


_REGULAR_WAITS_CACHE_SIZE = 10000
_STANDARD_SHANTEN_CACHE_SIZE = 20000
_SUMMARY_CACHE_SIZE = 10000
_ACTIVE_PROGRESS_PROFILER: Optional[dict[str, float]] = None


_TILE34_TO_STR: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]

@dataclass(frozen=True)
class NormalProgressInfo:
    shanten: int
    waits_count: int
    waits_tiles: List[bool]
    tehai_count: int
    ukeire_type_count: int
    ukeire_live_count: int
    ukeire_tiles: List[bool]
    good_shape_ukeire_type_count: int
    good_shape_ukeire_live_count: int
    good_shape_ukeire_tiles: List[bool]
    improvement_type_count: int
    improvement_live_count: int
    improvement_tiles: List[bool]


@dataclass(frozen=True)
class CandidateProgressV1:
    discard_tile34: int
    after_counts34: tuple[int, ...]
    shanten: int
    waits_count: int
    ukeire_type_count: int
    ukeire_live_count: int
    good_shape_ukeire_type_count: int
    good_shape_ukeire_live_count: int
    improvement_type_count: int
    improvement_live_count: int


def counts_to_hand(counts: Sequence[int]) -> List[str]:
    hand: List[str] = []
    for t34, cnt in enumerate(counts):
        if cnt > 0:
            hand.extend([_TILE34_TO_STR[t34]] * cnt)
    return hand


def _progress_profiler_add(key: str, delta_s: float) -> None:
    global _ACTIVE_PROGRESS_PROFILER
    if _ACTIVE_PROGRESS_PROFILER is not None:
        _ACTIVE_PROGRESS_PROFILER[key] = _ACTIVE_PROGRESS_PROFILER.get(key, 0.0) + delta_s


def _progress_profiler_inc(key: str, delta: int = 1) -> None:
    global _ACTIVE_PROGRESS_PROFILER
    if _ACTIVE_PROGRESS_PROFILER is not None:
        _ACTIVE_PROGRESS_PROFILER[key] = _ACTIVE_PROGRESS_PROFILER.get(key, 0) + delta


def _calc_standard_shanten_many(counts34_items: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    if not counts34_items:
        return ()
    if _ACTIVE_PROGRESS_PROFILER is not None:
        return tuple(calc_standard_shanten_from_counts(counts34) for counts34 in counts34_items)
    return tuple(_standard_shanten_many_native(counts34_items))


def _is_suited_sequence_start(tile34: int) -> bool:
    return tile34 < 27 and (tile34 % 9) <= 6


def _tile_in_obvious_meld(counts34: tuple[int, ...] | List[int], tile34: int) -> bool:
    cnt = counts34[tile34]
    if cnt <= 0:
        return False
    if cnt >= 3:
        return True
    if tile34 >= 27:
        return False
    pos = tile34 % 9
    base = tile34 - pos
    if pos >= 2 and counts34[base + pos - 1] > 0 and counts34[base + pos - 2] > 0:
        return True
    if 1 <= pos <= 7 and counts34[base + pos - 1] > 0 and counts34[base + pos + 1] > 0:
        return True
    if pos <= 6 and counts34[base + pos + 1] > 0 and counts34[base + pos + 2] > 0:
        return True
    return False


def _candidate_discards_no_meld_break(counts34: tuple[int, ...] | List[int]) -> tuple[int, ...]:
    preferred: list[int] = []
    fallback: list[int] = []
    seen: Set[int] = set()
    for tile34, cnt in enumerate(counts34):
        if cnt <= 0 or tile34 in seen:
            continue
        seen.add(tile34)
        fallback.append(tile34)
        if not _tile_in_obvious_meld(counts34, tile34):
            preferred.append(tile34)
    return tuple(preferred) if preferred else tuple(fallback)


def _calc_required_tiles(counts34: tuple[int, ...], visible_counts34: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    try:
        return tuple(_calc_required_tiles_native(counts34, visible_counts34, sum(counts34) // 3))
    except RuntimeError:
        shanten = calc_standard_shanten_from_counts(counts34)
        out: list[tuple[int, int]] = []
        for tile34 in range(34):
            live_count = max(0, 4 - visible_counts34[tile34])
            if live_count <= 0 or counts34[tile34] >= 4:
                continue
            work = list(counts34)
            work[tile34] += 1
            after = calc_standard_shanten_from_counts(tuple(work))
            if after < shanten:
                out.append((tile34, live_count))
        return tuple(out)


def _calc_draw_deltas(
    counts34: tuple[int, ...],
    visible_counts34: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    try:
        return tuple(_calc_draw_deltas_native(counts34, visible_counts34, sum(counts34) // 3))
    except RuntimeError:
        shanten = calc_standard_shanten_from_counts(counts34)
        out: list[tuple[int, int, int]] = []
        for tile34 in range(34):
            live_count = max(0, 4 - visible_counts34[tile34])
            if live_count <= 0 or counts34[tile34] >= 4:
                continue
            work = list(counts34)
            work[tile34] += 1
            after = calc_standard_shanten_from_counts(tuple(work))
            out.append((tile34, live_count, after - shanten))
        return tuple(out)


def _calc_discard_deltas(counts34: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    try:
        return tuple(_calc_discard_deltas_native(counts34, sum(counts34) // 3))
    except RuntimeError:
        shanten = calc_standard_shanten_from_counts(counts34)
        out: list[tuple[int, int]] = []
        for tile34, cnt in enumerate(counts34):
            if cnt <= 0:
                continue
            work = list(counts34)
            work[tile34] -= 1
            after = calc_standard_shanten_from_counts(tuple(work))
            out.append((tile34, after - shanten))
        return tuple(out)


def _populate_ukeire_tiles_3n1(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
    shanten: int,
    ukeire_tiles: List[bool],
) -> None:
    if shanten == 0:
        return
    if shanten > 2:
        for tile34, _live_count in _calc_required_tiles(counts_3n1, visible_counts_local):
            ukeire_tiles[tile34] = True
        return

    draw_deltas = _calc_draw_deltas(counts_3n1, visible_counts_local)
    for tile34, _live_count, draw_shanten_diff in draw_deltas:
        if draw_shanten_diff < 0:
            ukeire_tiles[tile34] = True
            continue

        counts_3n2 = list(counts_3n1)
        counts_3n2[tile34] += 1
        discard_deltas = _calc_discard_deltas(tuple(counts_3n2))
        if any(draw_shanten_diff + discard_diff < 0 for _discard34, discard_diff in discard_deltas):
            ukeire_tiles[tile34] = True


def _summarize_3n1_python(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    t0 = time.perf_counter()
    _progress_profiler_inc("standard_shanten_calls")
    standard_shanten = calc_standard_shanten_from_counts(counts_3n1)
    _progress_profiler_add("standard_shanten_s", time.perf_counter() - t0)
    shanten = standard_shanten
    tehai_count = sum(counts_3n1)
    waits_count = 0
    waits_tiles: tuple[bool, ...] = tuple([False] * 34)
    ukeire_tiles = [False] * 34
    good_shape_ukeire_tiles = [False] * 34
    improvement_tiles = [False] * 34
    regular_shanten: Optional[int] = None
    use_regular_detail = False
    if shanten <= 1:
        t0 = time.perf_counter()
        _progress_profiler_inc("waits_calls")
        regular_shanten, waits_count, waits_tiles, tehai_count = calc_shanten_waits_from_counts(counts_3n1)
        _progress_profiler_add("waits_s", time.perf_counter() - t0)
        use_regular_detail = regular_shanten == standard_shanten
        if not use_regular_detail:
            waits_count = 0
            waits_tiles = tuple([False] * 34)

    loop_t0 = time.perf_counter()
    _populate_ukeire_tiles_3n1(counts_3n1, visible_counts_local, shanten, ukeire_tiles)
    _progress_profiler_add("ukeire_improvement_s", time.perf_counter() - loop_t0)

    return _make_progress(
        shanten, waits_count, waits_tiles, tehai_count,
        ukeire_tiles, good_shape_ukeire_tiles, improvement_tiles, visible_counts_local,
    )


def _is_complete_regular_counts(
    counts: tuple[int, ...],
    cache: Dict[tuple[tuple[int, ...], int, bool], bool],
    melds_needed: Optional[int] = None,
    need_pair: bool = True,
) -> bool:
    if melds_needed is None:
        tile_count = sum(counts)
        if tile_count % 3 != 2:
            return False
        melds_needed = (tile_count - 2) // 3

    key = (counts, melds_needed, need_pair)
    cached = cache.get(key)
    if cached is not None:
        return cached

    first = next((i for i, cnt in enumerate(counts) if cnt > 0), None)
    if first is None:
        result = melds_needed == 0 and not need_pair
        cache[key] = result
        return result

    work = list(counts)
    result = False

    if need_pair and work[first] >= 2:
        work[first] -= 2
        if _is_complete_regular_counts(tuple(work), cache, melds_needed, False):
            result = True
        work[first] += 2

    if not result and melds_needed > 0 and work[first] >= 3:
        work[first] -= 3
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True
        work[first] += 3

    if (
        not result
        and melds_needed > 0
        and _is_suited_sequence_start(first)
        and work[first + 1] > 0
        and work[first + 2] > 0
    ):
        work[first] -= 1
        work[first + 1] -= 1
        work[first + 2] -= 1
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True

    cache[key] = result
    return result


def find_regular_waits(counts: tuple[int, ...]) -> List[bool]:
    waits = [False] * 34
    if sum(counts) % 3 != 1:
        return waits

    complete_cache: Dict[tuple[tuple[int, ...], int, bool], bool] = {}
    for tile34, cnt in enumerate(counts):
        if cnt >= 4:
            continue
        work = list(counts)
        work[tile34] += 1
        if _is_complete_regular_counts(tuple(work), complete_cache):
            waits[tile34] = True
    return waits


@lru_cache(maxsize=_REGULAR_WAITS_CACHE_SIZE)
def calc_shanten_waits_from_counts(
    counts34: tuple[int, ...],
) -> tuple[int, int, tuple[bool, ...], int]:
    tehai_count = sum(counts34)
    if tehai_count == 0:
        return 8, 0, tuple([False] * 34), 0
    shanten = int(_calc_shanten_normal_native(counts34))
    waits34 = tuple(find_regular_waits(counts34)) if shanten == 0 else tuple([False] * 34)
    return shanten, sum(waits34), waits34, tehai_count


@lru_cache(maxsize=_STANDARD_SHANTEN_CACHE_SIZE)
def calc_standard_shanten_from_counts(counts34: tuple[int, ...]) -> int:
    return int(_calc_standard_shanten_native(counts34))


def calc_shanten_waits_from_hand(hand: List[str], counts_builder) -> tuple[int, int, List[bool], int]:
    counts = tuple(counts_builder(hand))
    shanten, waits_count, waits, tehai_count = calc_shanten_waits_from_counts(counts)
    return shanten, waits_count, list(waits), tehai_count


def _make_progress(
    shanten: int,
    waits_count: int,
    waits_tiles: tuple[bool, ...] | List[bool],
    tehai_count: int,
    ukeire_tiles: List[bool],
    good_shape_ukeire_tiles: List[bool],
    improvement_tiles: List[bool],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    return NormalProgressInfo(
        shanten=shanten,
        waits_count=waits_count,
        waits_tiles=list(waits_tiles),
        tehai_count=tehai_count,
        ukeire_type_count=sum(ukeire_tiles),
        ukeire_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(ukeire_tiles) if flag),
        ukeire_tiles=ukeire_tiles,
        good_shape_ukeire_type_count=sum(good_shape_ukeire_tiles),
        good_shape_ukeire_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(good_shape_ukeire_tiles) if flag),
        good_shape_ukeire_tiles=good_shape_ukeire_tiles,
        improvement_type_count=sum(improvement_tiles),
        improvement_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(improvement_tiles) if flag),
        improvement_tiles=improvement_tiles,
    )


def _candidate_progress_key(candidate: CandidateProgressV1) -> tuple[int, int, int, int, int, int]:
    return (
        -candidate.shanten,
        candidate.ukeire_live_count,
        candidate.ukeire_type_count,
        candidate.waits_count,
        0,
        0,
    )


def _candidate_progress_v1_from_native(
    item: tuple[int, tuple[int, ...], int, int, int, int, int, int, int, int],
) -> CandidateProgressV1:
    return CandidateProgressV1(
        discard_tile34=int(item[0]),
        after_counts34=tuple(int(value) for value in item[1]),
        shanten=int(item[2]),
        waits_count=int(item[3]),
        ukeire_type_count=int(item[4]),
        ukeire_live_count=int(item[5]),
        good_shape_ukeire_type_count=int(item[6]),
        good_shape_ukeire_live_count=int(item[7]),
        improvement_type_count=int(item[8]),
        improvement_live_count=int(item[9]),
    )


def _select_candidate_discards_3n2(counts_3n2: tuple[int, ...]) -> tuple[int, ...]:
    candidate_discards = _candidate_discards_no_meld_break(counts_3n2)
    if not candidate_discards:
        return ()

    current_shanten = calc_standard_shanten_from_counts(counts_3n2)
    discard_delta_map = {tile34: shanten_diff for tile34, shanten_diff in _calc_discard_deltas(counts_3n2)}
    best_after_shanten: Optional[int] = None
    selected: list[int] = []

    for discard34 in candidate_discards:
        shanten_diff = discard_delta_map.get(discard34)
        if shanten_diff is None:
            continue
        after_shanten = current_shanten + shanten_diff
        if best_after_shanten is None or after_shanten < best_after_shanten:
            best_after_shanten = after_shanten
            selected = [discard34]
        elif after_shanten == best_after_shanten:
            selected.append(discard34)

    return tuple(selected) if selected else tuple(candidate_discards)


def _summarize_3n2_candidates_python(
    counts_3n2: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
    discard_candidates: Optional[Sequence[int]] = None,
) -> list[CandidateProgressV1]:
    out: list[CandidateProgressV1] = []
    for discard34 in (discard_candidates or _candidate_discards_no_meld_break(counts_3n2)):
        counts_3n1 = list(counts_3n2)
        counts_3n1[discard34] -= 1
        counts_3n1_t = tuple(counts_3n1)
        progress_3n1 = _summarize_3n1_cached(counts_3n1_t, visible_counts_local)
        out.append(
            CandidateProgressV1(
                discard_tile34=discard34,
                after_counts34=counts_3n1_t,
                shanten=progress_3n1.shanten,
                waits_count=progress_3n1.waits_count,
                ukeire_type_count=progress_3n1.ukeire_type_count,
                ukeire_live_count=progress_3n1.ukeire_live_count,
                good_shape_ukeire_type_count=progress_3n1.good_shape_ukeire_type_count,
                good_shape_ukeire_live_count=progress_3n1.good_shape_ukeire_live_count,
                improvement_type_count=progress_3n1.improvement_type_count,
                improvement_live_count=progress_3n1.improvement_live_count,
            )
        )
    return out


def _select_best_candidate_progress_3n2_python(
    counts_3n2: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
    discard_candidates: Optional[Sequence[int]] = None,
) -> Optional[CandidateProgressV1]:
    best: Optional[tuple[tuple[int, int, int, int, int, int], CandidateProgressV1]] = None
    for candidate in _summarize_3n2_candidates_python(counts_3n2, visible_counts_local, discard_candidates):
        key = _candidate_progress_key(candidate)
        if best is None or key > best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


@lru_cache(maxsize=_SUMMARY_CACHE_SIZE)
def summarize_one_shanten_draw_metrics(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> tuple[int, int]:
    try:
        return tuple(
            int(value)
            for value in _summarize_one_shanten_draw_metrics_native(counts_3n1, visible_counts_local)
        )
    except RuntimeError:
        pass
    current = _summarize_3n1_cached(counts_3n1, visible_counts_local)
    current_key = (
        -current.shanten,
        current.ukeire_live_count,
        current.ukeire_type_count,
        current.waits_count,
        0,
        0,
    )
    good_shape_live = 0
    improvement_live = 0
    for tile34, live_count, _draw_shanten_diff in _calc_draw_deltas(counts_3n1, visible_counts_local):
        if live_count <= 0 or counts_3n1[tile34] >= 4:
            continue
        counts_3n2 = list(counts_3n1)
        counts_3n2[tile34] += 1
        counts_3n2_t = tuple(counts_3n2)
        after_best: Optional[CandidateProgressV1] = None
        if _has_3n2_candidate_summaries():
            try:
                native_best = _summarize_best_3n2_candidate_native(
                    counts_3n2_t,
                    visible_counts_local,
                    _summarize_3n1_cached,
                )
            except RuntimeError:
                native_best = None
            if native_best is not None:
                after_best = _candidate_progress_v1_from_native(native_best)
        if after_best is None:
            discard_candidates = _select_candidate_discards_3n2(counts_3n2_t)
            after_best = _select_best_candidate_progress_3n2_python(
                counts_3n2_t,
                visible_counts_local,
                discard_candidates,
            )
        if after_best is None:
            continue
        if after_best.shanten == 0:
            if after_best.ukeire_live_count > 4:
                good_shape_live += live_count
            continue
        if after_best.shanten == current.shanten and _candidate_progress_key(after_best) > current_key:
            improvement_live += live_count
    return good_shape_live, improvement_live


@lru_cache(maxsize=_SUMMARY_CACHE_SIZE)
def _summarize_3n1_cached(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    t0 = time.perf_counter()
    try:
        (
            shanten,
            waits_count,
            waits_tiles,
            tehai_count,
            _ukeire_type_count,
            _ukeire_live_count,
            ukeire_tiles_t,
        ) = _summarize_3n1_native(counts_3n1, visible_counts_local)
    except RuntimeError:
        return _summarize_3n1_python(counts_3n1, visible_counts_local)
    _progress_profiler_add("ukeire_improvement_s", time.perf_counter() - t0)
    return _make_progress(
        shanten,
        waits_count,
        waits_tiles,
        tehai_count,
        list(ukeire_tiles_t),
        [False] * 34,
        [False] * 34,
        visible_counts_local,
    )


@lru_cache(maxsize=_SUMMARY_CACHE_SIZE)
def _summarize_3n2_cached(
    counts_3n2: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    loop_t0 = time.perf_counter()
    best_after_counts: Optional[tuple[int, ...]] = None
    if _has_3n2_candidate_summaries():
        try:
            best_candidate_native = _summarize_best_3n2_candidate_native(
                    counts_3n2,
                    visible_counts_local,
                    _summarize_3n1_cached,
                )
            if best_candidate_native is not None:
                best_after_counts = _candidate_progress_v1_from_native(best_candidate_native).after_counts34
        except RuntimeError:
            discard_candidates = _select_candidate_discards_3n2(counts_3n2)
            best_candidate = _select_best_candidate_progress_3n2_python(
                counts_3n2,
                visible_counts_local,
                discard_candidates,
            )
            if best_candidate is not None:
                best_after_counts = best_candidate.after_counts34
    else:
        discard_candidates = _select_candidate_discards_3n2(counts_3n2)
        best_candidate = _select_best_candidate_progress_3n2_python(
            counts_3n2,
            visible_counts_local,
            discard_candidates,
        )
        if best_candidate is not None:
            best_after_counts = best_candidate.after_counts34

    _progress_profiler_add("ukeire_improvement_s", time.perf_counter() - loop_t0)
    if best_after_counts is not None:
        return _summarize_3n1_cached(best_after_counts, visible_counts_local)
    return _summarize_3n1_cached(tuple(counts_3n2), visible_counts_local)


def analyze_normal_progress_from_counts(
    hand_counts34: tuple[int, ...],
    visible_counts34: tuple[int, ...],
) -> NormalProgressInfo:
    tile_count = sum(hand_counts34)
    mod = tile_count % 3
    if mod == 1:
        return _summarize_3n1_cached(hand_counts34, visible_counts34)
    if mod == 2:
        return _summarize_3n2_cached(hand_counts34, visible_counts34)
    return _make_progress(8, 0, tuple([False] * 34), tile_count, [False] * 34, [False] * 34, [False] * 34, visible_counts34)


def analyze_normal_progress_with_timings(
    hand_counts34: tuple[int, ...],
    visible_counts34: tuple[int, ...],
) -> tuple[NormalProgressInfo, dict[str, float]]:
    global _ACTIVE_PROGRESS_PROFILER
    timings = {
        "standard_shanten_s": 0.0,
        "waits_s": 0.0,
        "ukeire_improvement_s": 0.0,
        "shape_s": 0.0,
        "standard_shanten_calls": 0,
        "waits_calls": 0,
        "standard_shanten_hits": 0,
        "standard_shanten_misses": 0,
        "waits_hits": 0,
        "waits_misses": 0,
    }
    std_before = calc_standard_shanten_from_counts.cache_info()
    waits_before = calc_shanten_waits_from_counts.cache_info()
    prev = _ACTIVE_PROGRESS_PROFILER
    _ACTIVE_PROGRESS_PROFILER = timings
    try:
        progress = analyze_normal_progress_from_counts(hand_counts34, visible_counts34)
    finally:
        _ACTIVE_PROGRESS_PROFILER = prev
    std_after = calc_standard_shanten_from_counts.cache_info()
    waits_after = calc_shanten_waits_from_counts.cache_info()
    timings["standard_shanten_hits"] = std_after.hits - std_before.hits
    timings["standard_shanten_misses"] = std_after.misses - std_before.misses
    timings["waits_hits"] = waits_after.hits - waits_before.hits
    timings["waits_misses"] = waits_after.misses - waits_before.misses
    return progress, timings


def clear_progress_caches() -> None:
    calc_shanten_waits_from_counts.cache_clear()
    calc_standard_shanten_from_counts.cache_clear()
    summarize_one_shanten_draw_metrics.cache_clear()
    _summarize_3n1_cached.cache_clear()
    _summarize_3n2_cached.cache_clear()
