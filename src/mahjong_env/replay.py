from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from mahjong.shanten import Shanten
from mahjong_env.feature_tracker import SnapshotFeatureTracker
from mahjong_env.progress_oracle import (
    NormalProgressInfo,
    analyze_normal_progress_from_counts as _oracle_calc_normal_progress_from_counts,
    calc_shanten_waits_from_counts as _oracle_calc_shanten_waits_from_counts,
)
from mahjong_env.replay_normalizer import (
    normalize_replay_events,
    normalize_replay_label_for_legal_compare,
    replay_label_matches_legal,
)
from mahjong_env.tiles import AKA_DORA_TILES, tile_to_34 as _to_34
from mahjong_env.types import MjaiEvent

# ---------------------------------------------------------------------------
# generic 3n+1 / 3n+2 standard-hand shanten/waits helpers
# ---------------------------------------------------------------------------


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    if tile.endswith("r"):
        return tile[:-1]
    return tile


_TILE34_TO_STR: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]


def _counts34_from_hand(hand: List[str]) -> List[int]:
    counts = [0] * 34
    for tile in hand:
        t34 = _to_34(tile)
        if 0 <= t34 < 34:
            counts[t34] += 1
    return counts


def _is_suited_sequence_start(tile34: int) -> bool:
    return tile34 < 27 and (tile34 % 9) <= 6


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


def _find_regular_waits(counts: tuple[int, ...]) -> List[bool]:
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


def _find_special_waits(
    counts34: tuple[int, ...],
    *,
    shanten_calc: Shanten,
) -> tuple[int, List[bool]]:
    chiitoi_shanten = int(shanten_calc.calculate_shanten_for_chiitoitsu_hand(list(counts34)))
    kokushi_shanten = int(shanten_calc.calculate_shanten_for_kokushi_hand(list(counts34)))
    special_shanten = min(chiitoi_shanten, kokushi_shanten)
    waits = [False] * 34
    if special_shanten != 0:
        return special_shanten, waits

    for tile34, cnt in enumerate(counts34):
        if cnt >= 4:
            continue
        work = list(counts34)
        work[tile34] += 1
        if (
            (chiitoi_shanten == 0 and int(shanten_calc.calculate_shanten_for_chiitoitsu_hand(work)) == -1)
            or (kokushi_shanten == 0 and int(shanten_calc.calculate_shanten_for_kokushi_hand(work)) == -1)
        ):
            waits[tile34] = True
    return special_shanten, waits


def _calc_shanten_waits(hand: List[str], melds: List[dict]):
    """返回 (standard_shanten, waits_count, waits_tile34_bools, tehai_count)。"""
    counts34 = _counts34_from_hand(hand)
    try:
        regular_shanten, waits_count, waits34, tehai_count = _oracle_calc_shanten_waits_from_counts(tuple(counts34))
        waits = list(waits34)
        shanten = regular_shanten
        if not melds:
            shanten_calc = Shanten()
            special_shanten, special_waits = _find_special_waits(tuple(counts34), shanten_calc=shanten_calc)
            if special_shanten < shanten:
                shanten = special_shanten
                waits = special_waits if special_shanten == 0 else [False] * 34
            elif special_shanten == shanten == 0:
                waits = [a or b for a, b in zip(waits, special_waits)]
        return shanten, sum(waits), waits, tehai_count
    except Exception:
        del melds
        tehai_count = sum(counts34)
        return 8, 0, [False] * 34, tehai_count


def _expand_hand_counter(counter) -> List[str]:
    """把 PlayerState.hand (Counter) 展开成 list[tile_str] 供 _calc_shanten_waits 使用。"""
    hand_list: List[str] = []
    for tile, cnt in sorted(counter.items()):
        hand_list.extend([tile] * int(cnt))
    return hand_list


def _compute_opp_tenpai_target(
    sample_state: "GameState",
    actor: int,
) -> Tuple[float, float, float]:
    """在决策时刻从上帝视角计算 3 个对手的 tenpai 状态。

    顺序相对 actor:(actor+1)%4, (actor+2)%4, (actor+3)%4。
    这是 xmodel1 opp_tenpai_head 的 BCE 监督标签。

    实现:用 `keqing_core.calc_shanten_all`(Rust shanten 表)对对手 counts34 做判断,
    shanten ≤ 0 视为 tenpai。这样和 Rust preprocess (`xmodel1_export.rs` 中同名逻辑)
    使用**同一个 shanten source of truth**,保证 Python/Rust 两端导出的标签逐元素一致。

    之前用 `_calc_shanten_waits` 的 Python 递归版本,在 degenerate fixture
    (例如全 13 张同一种 tile) 下会和 Rust 表产生分歧;对合法牌局无差异,但会污染
    parity 测试。改用 Rust 后该问题消失。

    如果 Rust 扩展不可用或调用异常,退化为 shanten=8(非 tenpai),安全兜底。
    """
    try:
        from keqing_core import calc_shanten_all as _rust_calc_shanten_all
    except ImportError:  # pragma: no cover - Rust 扩展缺失时兜底
        _rust_calc_shanten_all = None

    result: List[float] = []
    for rel in (1, 2, 3):
        opp_idx = (actor + rel) % 4
        if opp_idx < 0 or opp_idx >= len(sample_state.players):
            result.append(0.0)
            continue
        player = sample_state.players[opp_idx]
        hand_list = _expand_hand_counter(player.hand)
        melds_opp = list(player.melds)
        if not hand_list and not melds_opp:
            result.append(0.0)
            continue

        counts34 = _hand_tile34_counts(hand_list)
        # 与 Rust tracker (`apply_hand_count_delta` 的 clamp(0,4)) 对齐:
        # 实战数据一张 tile 最多 4 张,但测试 fixture 可能造出 >4 张同牌的
        # degenerate 状态。Rust preprocess 对这类状态会 clamp,如果 Python
        # 不 clamp,Rust/Python 两端算出的 shanten 就会分叉,破坏 parity。
        counts34 = [min(c, 4) for c in counts34]
        tile_sum = sum(counts34)
        if tile_sum == 0:
            result.append(0.0)
            continue

        shanten_opp: int
        if _rust_calc_shanten_all is not None:
            try:
                shanten_opp = int(_rust_calc_shanten_all(list(counts34)))
            except Exception:
                shanten_opp = 8
        else:  # pragma: no cover - 依赖 riichienv 的 fallback 路径
            try:
                shanten_opp, *_ = _calc_shanten_waits(hand_list, melds_opp)
            except Exception:
                shanten_opp = 8
        result.append(1.0 if shanten_opp is not None and shanten_opp <= 0 else 0.0)
    return (result[0], result[1], result[2])


def _meld_tile34_counts(melds: List[dict]) -> List[int]:
    counts = [0] * 34
    for meld in melds:
        for p in meld.get("consumed", []) + ([meld.get("pai")] if meld.get("pai") else []):
            t34 = _to_34(p)
            if 0 <= t34 < 34:
                counts[t34] += 1
    return counts


def _hand_tile34_counts(hand: List[str]) -> List[int]:
    counts = [0] * 34
    for tile in hand:
        t34 = _to_34(tile)
        if 0 <= t34 < 34:
            counts[t34] += 1
    return counts


def _default_visible_counts(hand: List[str], melds: List[dict]) -> List[int]:
    counts = _hand_tile34_counts(hand)
    meld_counts = _meld_tile34_counts(melds)
    for i in range(34):
        counts[i] += meld_counts[i]
    return counts


def _counts_to_hand(counts: Sequence[int]) -> List[str]:
    hand: List[str] = []
    for t34, cnt in enumerate(counts):
        if cnt > 0:
            hand.extend([_TILE34_TO_STR[t34]] * cnt)
    return hand


def _tenpai_live_wait_count(
    counts13: tuple[int, ...],
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> int:
    cached = shanten_cache.get(counts13)
    if cached is None:
        cached = _calc_shanten_waits(_counts_to_hand(counts13), melds)
        shanten_cache[counts13] = cached
    shanten, _waits_count, waits_tiles, _tehai_count = cached
    if shanten != 0:
        return 0
    return sum(
        max(0, 4 - visible_counts_local[t34])
        for t34, flag in enumerate(waits_tiles)
        if flag
    )


def _best_tenpai_wait_live_after_draw(
    counts14: tuple[int, ...],
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> int:
    best_wait_live = 0
    seen_discards: Set[int] = set()
    for discard34, cnt in enumerate(counts14):
        if cnt <= 0 or discard34 in seen_discards:
            continue
        seen_discards.add(discard34)
        after_counts13 = list(counts14)
        after_counts13[discard34] -= 1
        after_counts13_t = tuple(after_counts13)
        after_shanten, _w_cnt, _w_tiles, _tehai = shanten_cache.get(after_counts13_t, (None, None, None, None))
        if after_shanten is None:
            after_shanten, _w_cnt, _w_tiles, _tehai = _calc_shanten_waits(_counts_to_hand(after_counts13_t), melds)
            shanten_cache[after_counts13_t] = (after_shanten, _w_cnt, _w_tiles, _tehai)
        if after_shanten != 0:
            continue
        wait_live = _tenpai_live_wait_count(after_counts13_t, melds, visible_counts_local, shanten_cache)
        if wait_live > best_wait_live:
            best_wait_live = wait_live
    return best_wait_live


def _is_good_shape_draw_for_one_shanten(
    counts13: tuple[int, ...],
    draw_tile34: int,
    melds: List[dict],
    visible_counts_local: tuple[int, ...],
    shanten_cache: Dict[tuple[int, ...], tuple[int, int, List[bool], int]],
) -> bool:
    """严格复刻一向听好型判定：

    摸入有效牌后，枚举所有去重打牌分支；只要存在一个分支能形成一般形听牌，
    且该听牌的 live waits > 4，就判该摸牌为好型进张。
    """

    counts14 = list(counts13)
    counts14[draw_tile34] += 1
    counts14_t = tuple(counts14)
    return _best_tenpai_wait_live_after_draw(
        counts14_t, melds, visible_counts_local, shanten_cache
    ) > 4


def _calc_normal_progress(
    hand: List[str],
    melds: List[dict],
    visible_counts: Optional[List[int]] = None,
) -> NormalProgressInfo:
    if visible_counts is None:
        visible_counts = _default_visible_counts(hand, melds)
    return _oracle_calc_normal_progress_from_counts(
        tuple(_hand_tile34_counts(hand)),
        tuple(visible_counts),
    )


@dataclass
class ReplaySample:
    state: Dict
    actor: int
    actor_name: str
    label_action: Dict
    legal_actions: List[Dict]
    value_target: float
    score_delta_target: float = 0.0
    win_target: float = 0.0
    dealin_target: float = 0.0
    pts_given_win_target: float = 0.0
    pts_given_dealin_target: float = 0.0
    ryukyoku_tenpai_target: float = 0.0
    # 决策时刻 3 个对手的 tenpai 状态 (相对 actor,顺序为下家/对家/上家 = (actor+1)%4, (actor+2)%4, (actor+3)%4)。
    # 作为 xmodel1 opp_tenpai_head 的 BCE 监督标签,1.0 表示该对手当前向听 ≤ 0。
    # Stage 2 Python 原型:这个字段曾由旧 replay sample builder 在样本产出时通过 _calc_shanten_waits
    # 对上帝视角下的对手手牌直接计算;Stage 2 Rust 迁移后改由 Rust preprocess emit。
    opp_tenpai_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    score_before_action: int = 0
    final_score_delta_points_target: int = 0
    final_rank_target: int = 0
    # 触发该监督样本的原始 replay 事件索引。xmodel1 preprocess 构造 event_history
    # 时必须使用真实事件位置，而不是样本序号。
    event_index: int = 0
    # 规范化后的原始事件序,供 parity / smoke preprocess 构造与生产一致的
    # event_history。
    events: Optional[List[MjaiEvent]] = None


class IllegalLabelActionError(ValueError):
    """Raised when a replay label action is not contained in the reconstructed legal set."""


def _inject_replay_sample_snapshot_features(snapshot: Dict, label_action: Dict, actor: int) -> Dict:
    snap = dict(snapshot)
    if label_action.get("type") != "none":
        snap["tsumo_pai"] = None
    hand = snap.get("hand", [])
    melds = (snap.get("melds") or [[], [], [], []])[actor]
    shanten, waits_count, waits_tiles, _ = _calc_shanten_waits(hand, melds)
    snap["shanten"] = shanten
    snap["waits_count"] = waits_count
    snap["waits_tiles"] = waits_tiles
    if label_action.get("type") == "hora":
        if label_action.get("is_haitei") is True:
            snap["_hora_is_haitei"] = True
        if label_action.get("is_houtei") is True:
            snap["_hora_is_houtei"] = True
        if label_action.get("is_rinshan") is True:
            snap["_hora_is_rinshan"] = True
        if label_action.get("is_chankan") is True:
            snap["_hora_is_chankan"] = True
    if "feature_tracker" not in snap:
        snap["feature_tracker"] = asdict(SnapshotFeatureTracker.from_state(snap, actor))
    else:
        tracker = dict(snap["feature_tracker"])
        for key in ("hand_counts34", "meld_counts34", "visible_counts34", "suit_counts", "aka_counts"):
            if key in tracker:
                tracker[key] = tuple(tracker[key])
        snap["feature_tracker"] = tracker
    return snap


def _illegal_label_message(
    *,
    event_index: int,
    actor: int,
    actor_name: str,
    label: Dict,
    legal_dicts: List[Dict],
    snapshot: Dict,
) -> str:
    melds = (snapshot.get("melds") or [[], [], [], []])[actor]
    return (
        "replay label action not in reconstructed legal set: "
        f"event_index={event_index} actor={actor} actor_name={actor_name} "
        f"label={label} legal={legal_dicts} "
        f"snap={{'bakaze': {snapshot.get('bakaze')!r}, 'kyoku': {snapshot.get('kyoku')!r}, "
        f"'honba': {snapshot.get('honba')!r}, 'hand': {snapshot.get('hand')!r}, "
        f"'last_discard': {snapshot.get('last_discard')!r}, 'melds': {melds!r}}}"
    )


def build_replay_samples_mc_return(
    events: List[MjaiEvent],
    actor_filter: Optional[Set[int]] = None,
    actor_name_filter: Optional[Set[str]] = None,
    strict_legal_labels: bool = True,
) -> List[ReplaySample]:
    events = normalize_replay_events(events)
    try:
        import keqing_core
    except ImportError:
        raise RuntimeError("keqing_core native replay sample builder is not available")

    try:
        native_records = keqing_core.build_replay_decision_records_mc_return(events)
    except Exception as exc:
        if keqing_core.is_missing_rust_capability_error(exc):
            raise RuntimeError("keqing_core native replay sample builder capability is not available") from exc
        raise

    actor_names = extract_actor_names(events)
    samples: List[ReplaySample] = []
    for record in native_records:
        actor = int(record["actor"])
        if actor_filter is not None and actor not in actor_filter:
            continue
        actor_name = actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
        if actor_name_filter is not None and actor_name not in actor_name_filter:
            continue

        label = normalize_replay_label_for_legal_compare(dict(record["label_action"]))
        legal_dicts = [dict(item) for item in record["legal_actions"]]
        snapshot = _inject_replay_sample_snapshot_features(dict(record["state"]), label, actor)
        if strict_legal_labels and not replay_label_matches_legal(label, legal_dicts):
            raise IllegalLabelActionError(
                _illegal_label_message(
                    event_index=int(record["event_index"]),
                    actor=actor,
                    actor_name=actor_name,
                    label=label,
                    legal_dicts=legal_dicts,
                    snapshot=snapshot,
                )
            )
        samples.append(
            ReplaySample(
                state=snapshot,
                actor=actor,
                actor_name=actor_name,
                label_action=label,
                legal_actions=legal_dicts,
                value_target=float(record.get("value_target", 0.0)),
                score_delta_target=float(record.get("score_delta_target", 0.0)),
                win_target=float(record.get("win_target", 0.0)),
                dealin_target=float(record.get("dealin_target", 0.0)),
                pts_given_win_target=float(record.get("pts_given_win_target", 0.0)),
                pts_given_dealin_target=float(record.get("pts_given_dealin_target", 0.0)),
                ryukyoku_tenpai_target=float(record.get("ryukyoku_tenpai_target", 0.0)),
                opp_tenpai_target=tuple(float(v) for v in record.get("opp_tenpai_target", (0.0, 0.0, 0.0))),
                score_before_action=int(record.get("score_before_action", snapshot.get("scores", [0, 0, 0, 0])[actor])),
                final_score_delta_points_target=int(record.get("final_score_delta_points_target", 0)),
                final_rank_target=int(record.get("final_rank_target", 0)),
                event_index=int(record.get("event_index", 0)),
                events=events,
            )
        )
    return samples


def read_mjai_jsonl(path: str | Path) -> List[MjaiEvent]:
    events: List[MjaiEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def extract_actor_names(events: Sequence[MjaiEvent]) -> List[str]:
    for e in events:
        if (
            e.get("type") == "start_game"
            and isinstance(e.get("names"), list)
            and len(e["names"]) == 4
        ):
            return [str(x) for x in e["names"]]
    return ["p0", "p1", "p2", "p3"]
