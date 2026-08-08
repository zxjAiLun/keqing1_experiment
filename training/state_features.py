"""Shared tile/scalar state feature encoder used by active model lines."""

from __future__ import annotations

from collections import Counter
import time
from typing import Dict, List

import numpy as np
from mahjong.shanten import Shanten

from mahjong_env.tiles import tile_to_34 as _to_34, tile_is_aka as _is_aka
from mahjong_env.feature_tracker import SnapshotFeatureTracker
from mahjong_env.progress_oracle import (
    analyze_normal_progress_from_counts,
    analyze_normal_progress_with_timings,
)


_DORA_NEXT = {
    '1m':'2m','2m':'3m','3m':'4m','4m':'5m','5m':'6m','6m':'7m','7m':'8m','8m':'9m','9m':'1m',
    '1p':'2p','2p':'3p','3p':'4p','4p':'5p','5p':'6p','6p':'7p','7p':'8p','8p':'9p','9p':'1p',
    '1s':'2s','2s':'3s','3s':'4s','4s':'5s','5s':'6s','6s':'7s','7s':'8s','8s':'9s','9s':'1s',
    'E':'S','S':'W','W':'N','N':'E','P':'F','F':'C','C':'P',
}

# ---------------------------------------------------------------------------
# 34 种牌名顺序（与 action_space.py 保持一致，tile34 index 对应）
# ---------------------------------------------------------------------------
N_TILES = 34
C_TILE = 57
N_SCALAR = 56

_WIND_ORDER = ["E", "S", "W", "N"]
_BAKAZE_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}


def _shape_proxy_from_hand_count(hand_count: Counter) -> tuple[int, int, int]:
    pair_count = sum(1 for c in hand_count.values() if c >= 2)
    ankoutsu_count = sum(1 for c in hand_count.values() if c >= 3)
    taatsu_count = 0
    for base in (0, 9, 18):
        suit = [hand_count.get(base + i, 0) for i in range(9)]
        for i in range(8):
            if suit[i] > 0 and suit[i + 1] > 0:
                taatsu_count += 1
        for i in range(7):
            if suit[i] > 0 and suit[i + 2] > 0:
                taatsu_count += 1
    return pair_count, taatsu_count, ankoutsu_count


def _regular_shanten_counts(all34_list: List[int]) -> tuple[int, int]:
    counts = [0] * 34
    for t34 in all34_list:
        counts[t34] += 1
    shanten_calc = Shanten()
    chiitoi = int(shanten_calc.calculate_shanten_for_chiitoitsu_hand(counts))
    kokushi = int(shanten_calc.calculate_shanten_for_kokushi_hand(counts))
    return chiitoi, kokushi


def encode(state: Dict, actor: int):
    """返回 (tile_features, scalar_features)。"""
    tile_feat, scalar, _timings = _encode_impl(state, actor, collect_timings=False)
    return tile_feat, scalar


def encode_with_timings(state: Dict, actor: int):
    """返回 (tile_features, scalar_features, timings)。"""
    return _encode_impl(state, actor, collect_timings=True)


def _encode_impl(state: Dict, actor: int, *, collect_timings: bool):
    timings = {
        "tracker_s": 0.0,
        "progress_s": 0.0,
        "fill_s": 0.0,
    }
    t_fill0 = time.perf_counter() if collect_timings else 0.0
    tile_feat = np.zeros((C_TILE, N_TILES), dtype=np.float32)
    scalar = np.zeros(N_SCALAR, dtype=np.float32)

    tracker = SnapshotFeatureTracker.from_state(state, actor)
    if collect_timings:
        timings["tracker_s"] = time.perf_counter() - t_fill0
        t_fill0 = time.perf_counter()
    discards: List[List] = state.get("discards", [[], [], [], []])

    def _disc_pai(d) -> str:
        return d["pai"] if isinstance(d, dict) else d

    def _disc_tsumogiri(d) -> bool:
        return d.get("tsumogiri", False) if isinstance(d, dict) else False

    def _disc_reach_declared(d) -> bool:
        return d.get("reach_declared", False) if isinstance(d, dict) else False
    melds: List[List[dict]] = state.get("melds", [[], [], [], []])
    scores: List[int] = state.get("scores", [25000] * 4)
    dora_markers: List[str] = state.get("dora_markers", [])
    reached: List[bool] = state.get("reached", [False] * 4)
    bakaze: str = state.get("bakaze", "E")
    kyoku: int = state.get("kyoku", 1)
    honba: int = state.get("honba", 0)
    kyotaku: int = state.get("kyotaku", 0)
    oya: int = state.get("oya", 0)
    hand_counts34 = tracker.hand_counts34
    meld_counts34 = tracker.meld_counts34
    visible_counts = tracker.visible_counts34

    # ---- ch 0-3: 自家手牌计数 planes ----
    hand_count: Counter = Counter({i: c for i, c in enumerate(hand_counts34) if c > 0})
    for tile34, cnt in hand_count.items():
        for k in range(min(cnt, 4)):
            tile_feat[k, tile34] = 1.0

    discard_infos: list[list[tuple[int, bool, bool]]] = []
    total_discards = 0
    latest_riichi_pid = None
    latest_riichi_turn = -1
    for pid in range(4):
        pid_disc_infos: list[tuple[int, bool, bool]] = []
        pid_discards = discards[pid] if pid < len(discards) else []
        total_discards += len(pid_discards)
        for turn, d in enumerate(pid_discards):
            idx = _to_34(_disc_pai(d))
            if idx < 0:
                continue
            tsumogiri = _disc_tsumogiri(d)
            reach_declared = _disc_reach_declared(d)
            pid_disc_infos.append((idx, tsumogiri, reach_declared))
            if pid != actor and reach_declared and turn > latest_riichi_turn:
                latest_riichi_turn = turn
                latest_riichi_pid = pid
        discard_infos.append(pid_disc_infos)

    chi_pon_calls = 0
    kan_calls = 0
    opponent_meld_counts = [0.0, 0.0, 0.0]

    # ---- ch 4-7: 自家副露 presence ----
    actor_melds = melds[actor] if actor < len(melds) else []
    for meld_slot, meld in enumerate(actor_melds[:4]):
        ch = 4 + meld_slot
        pais = meld.get("consumed", []) + ([meld.get("pai")] if meld.get("pai") else [])
        for p in pais:
            idx = _to_34(p)
            if idx >= 0:
                tile_feat[ch, idx] = 1.0

    # ---- ch 8-13: 他家舍牌（3家 × 2通道，ch_base+0: 立直宣言牌, ch_base+1: 手切flag）----
    other_pids = [pid for pid in range(4) if pid != actor]
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        ch_base = 8 + slot * 2
        for idx, tsumogiri, reach_declared in discard_infos[pid]:
            if reach_declared:
                tile_feat[ch_base,     idx] = 1.0  # 立直宣言牌
            if not tsumogiri:
                tile_feat[ch_base + 1, idx] = 1.0  # 手切

    # ---- ch 14-16: 他家副露 presence（3家 × 1通道）----
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        pid_melds_list = melds[pid] if pid < len(melds) else []
        opponent_meld_counts[slot] = len(pid_melds_list) / 4.0
        for meld in pid_melds_list:
            pais = meld.get('consumed', []) + ([meld.get('pai')] if meld.get('pai') else [])
            mtype = meld.get("type")
            if mtype in ("chi", "pon"):
                chi_pon_calls += 1
            if mtype in ("daiminkan", "ankan", "kakan"):
                kan_calls += 1
            for p in pais:
                idx = _to_34(p)
                if idx >= 0:
                    tile_feat[14 + slot, idx] = 1.0
    # ---- ch 17: 自家舍牌 presence ----
    for idx, _tsumogiri, _reach_declared in discard_infos[actor]:
        tile_feat[17, idx] = 1.0

    # ---- ch 18: dora 实际牌（指示牌下一张，累加）----
    for dm in dora_markers:
        dt = _DORA_NEXT.get(dm)
        if dt:
            idx = _to_34(dt)
            if idx >= 0:
                tile_feat[18, idx] += 1.0

    if collect_timings:
        t_progress0 = time.perf_counter()
        progress, progress_timings = analyze_normal_progress_with_timings(hand_counts34, visible_counts)
        timings["progress_s"] = time.perf_counter() - t_progress0
        timings.update(progress_timings)
        t_fill0 = time.perf_counter()
    else:
        progress = analyze_normal_progress_from_counts(hand_counts34, visible_counts)

    # ---- ch 19: 听牌 waits tiles ----
    waits_tiles = state.get("waits_tiles")
    if waits_tiles is None:
        waits_tiles = progress.waits_tiles
    for i, w in enumerate(waits_tiles[:34]):
        if w:
            tile_feat[19, i] = 1.0

    # ---- ch 20-51: 舍牌巡目分段编码（4家 × 8段，ch = 20 + slot*8 + seg）----
    # slot: 他家 0/1/2（与 ch 8-13 相同相对顺序），自家 slot=3
    # 每3巡一段（0-2巡→seg0, 3-5→seg1, ..., 21+→seg7）
    slot_for_pid = {pid: slot for slot, pid in enumerate(other_pids)}
    slot_for_pid[actor] = 3
    for pid in range(4):
        slot = slot_for_pid[pid]
        for turn, (idx, _tsumogiri, _reach_declared) in enumerate(discard_infos[pid]):
            seg = min(turn // 3, 7)
            ch = 20 + slot * 8 + seg
            tile_feat[ch, idx] = 1.0

    # ch 52: tsumo 牌位置
    tsumo_pai = state.get("tsumo_pai")
    if tsumo_pai:
        idx = _to_34(tsumo_pai)
        if idx >= 0:
            tile_feat[52, idx] = 1.0

    # ch 53: 最新立直家立直后其他家打过的牌（对该立直家的真安全牌）
    # 找最后一个立直的他家 pid，取其立直宣言牌之后其他家的舍牌
    if latest_riichi_pid is not None:
        # 收集立直后所有其他家（含 actor）的舍牌
        after_riichi_tiles: set = set()
        for pid in range(4):
            if pid == latest_riichi_pid:
                continue
            for idx, _tsumogiri, _reach_declared in discard_infos[pid]:
                after_riichi_tiles.add(idx)
        for idx in after_riichi_tiles:
            tile_feat[53, idx] = 1.0

    # ---- ch 54-55: 标准型进展 tiles ----
    normal_ukeire_tiles = state.get("normal_ukeire_tiles", progress.ukeire_tiles)
    for i, flag in enumerate(normal_ukeire_tiles[:34]):
        if flag:
            tile_feat[54, i] = 1.0
    # v3 no longer models improvement tiles; keep the channel zero-filled for shape stability.

    # ---- ch 56: 已见枚数 / 4 ----
    tile_feat[56, :] = np.clip(np.asarray(visible_counts, dtype=np.float32) / 4.0, 0.0, 1.0)

    actor_disc_n = tracker.discards_count
    actor_score = scores[actor] if actor < len(scores) else 25000
    jikaze = (actor - oya) % 4

    # ===== 役种特征计算基础 =====
    is_open = len(actor_melds) > 0

    dora_set: Counter = Counter()
    for dm in dora_markers:
        dt = _DORA_NEXT.get(dm)
        if dt:
            dora_set[_to_34(dt)] += 1
    # 手牌+副露的所有牌（tile34 list），仅构建一次，供 dora 计数和役种判断共用
    all34_list: List[int] = list(hand_count.elements()) + [i for i, c in enumerate(meld_counts34) for _ in range(c)]
    dora_count = sum(dora_set[t] for t in all34_list)

    _YAOCHUU_34 = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    _MAN_34   = set(range(0, 9))
    _PIN_34   = set(range(9, 18))
    _SOU_34   = set(range(18, 27))
    _HONOR_34 = set(range(27, 34))
    all34_set = set(all34_list)
    total_tiles = len(all34_list)

    man_cnt, pin_cnt, sou_cnt, honor_cnt = tracker.suit_counts
    yaochuu_cnt = sum(1 for t in all34_list if t in _YAOCHUU_34)

    has_tanyao = len(all34_set & _YAOCHUU_34) == 0 and total_tiles > 0
    suupai_yaochuu_cnt = sum(1 for t in all34_list if t in {0,8,9,17,18,26})
    has_junchan = (not is_open) and (honor_cnt == 0) and total_tiles > 0 and (suupai_yaochuu_cnt * 3 >= total_tiles)
    has_chanta = total_tiles > 0 and (yaochuu_cnt * 3 >= total_tiles)
    suit_counts = [c for c in (man_cnt, pin_cnt, sou_cnt) if c > 0]
    has_honitsu = len(suit_counts) == 1 and total_tiles > 0
    has_chinitsu = len(suit_counts) == 1 and honor_cnt == 0 and total_tiles > 0
    pair_count = tracker.pair_count
    _pair_count_unused, taatsu_count, _ank_unused = _shape_proxy_from_hand_count(hand_count)
    ankoutsu_cnt = tracker.ankoutsu_count
    if is_open:
        chiitoi_shanten, kokushi_shanten = 6, 13
    else:
        chiitoi_shanten, kokushi_shanten = _regular_shanten_counts(list(hand_count.elements()))

    estimated_wall_draws = total_discards - chi_pon_calls + kan_calls
    if tsumo_pai:
        estimated_wall_draws += 1
    tiles_left = max(0, 70 - estimated_wall_draws)
    top_gap = max(0, max(scores) - actor_score)
    bottom_gap = max(0, actor_score - min(scores))
    riichi_threat_count = sum(1 for pid, flag in enumerate(reached) if pid != actor and flag)
    is_all_last_like = 1.0 if (bakaze in ("W", "N") or (bakaze == "S" and kyoku >= 4)) else 0.0
    tenpai_live_tile_count = sum(
        max(0, 4 - visible_counts[t34])
        for t34, flag in enumerate(waits_tiles)
        if flag
    )
    wait_hiddenness_ratio = (
        tenpai_live_tile_count / max(4 * progress.waits_count, 1)
        if progress.waits_count > 0
        else 0.0
    )
    # ===== Scalar features =====
    # [0-2] bakaze 3-bit（E/S/W，N场三者均为0）
    scalar[0] = 1.0 if bakaze == "E" else 0.0
    scalar[1] = 1.0 if bakaze == "S" else 0.0
    scalar[2] = 1.0 if bakaze == "W" else 0.0
    scalar[3] = kyoku / 4.0
    scalar[4] = honba / 8.0
    scalar[5] = kyotaku / 8.0
    scalar[6] = actor_score / 50000.0

    # [7] score rank (0=1st, 3=4th)
    rank = sum(1 for s in scores if s > actor_score)
    scalar[7] = rank / 3.0

    scalar[8] = progress.shanten / 8.0
    scalar[9] = progress.waits_count / 34.0
    scalar[10] = actor_disc_n / 18.0
    scalar[11] = tracker.meld_count / 4.0

    # [12-15] 立直 flag，相对 slot（slot=0: actor, 1: 下家, 2: 对家, 3: 上家）
    all_pids_by_slot = [actor] + other_pids  # slot 0=actor, 1/2/3=他家
    for slot, pid in enumerate(all_pids_by_slot):
        scalar[12 + slot] = 1.0 if (pid < len(reached) and reached[pid]) else 0.0

    # [16] 赤5总数（手牌+副露）/ 4.0
    aka_total, aka_m, aka_p, aka_s = tracker.aka_counts
    scalar[16] = aka_total / 4.0
    # [17] 断幺九
    scalar[17] = 1.0 if has_tanyao else 0.0
    # [18-20] 赤5 flag（手牌+副露）
    scalar[18] = 1.0 if aka_m > 0 else 0.0
    scalar[19] = 1.0 if aka_p > 0 else 0.0
    scalar[20] = 1.0 if aka_s > 0 else 0.0
    scalar[21] = opponent_meld_counts[0] if len(other_pids) > 0 else 0.0
    scalar[22] = opponent_meld_counts[1] if len(other_pids) > 1 else 0.0
    scalar[23] = opponent_meld_counts[2] if len(other_pids) > 2 else 0.0
    scalar[24] = ((scores[other_pids[0]] if len(other_pids) > 0 else actor_score) - actor_score) / 30000.0
    scalar[25] = ((scores[other_pids[1]] if len(other_pids) > 1 else actor_score) - actor_score) / 30000.0
    scalar[26] = ((scores[other_pids[2]] if len(other_pids) > 2 else actor_score) - actor_score) / 30000.0
    scalar[27] = jikaze / 3.0
    scalar[28] = min((dora_count + aka_total) / 10.0, 1.0)

    # [29] 确定番数下限（已确定能得到的番，不含进张后可能役种）
    _SANGEN_34 = [31, 32, 33]    # 白/发/中
    bakaze_34 = 27 + _BAKAZE_IDX.get(bakaze, 0)
    jikaze_34 = 27 + jikaze
    confirmed_han = 0
    honor_koutsu = set()
    for meld in actor_melds:
        if meld.get('type') in ('pon', 'daiminkan', 'ankan', 'kakan'):
            t34 = _to_34(meld.get('pai', ''))
            if t34 in _HONOR_34:
                honor_koutsu.add(t34)
    for t34, cnt in hand_count.items():
        if t34 in _HONOR_34 and cnt >= 3:
            honor_koutsu.add(t34)
    if bakaze_34 in honor_koutsu:
        confirmed_han += 1
    if jikaze_34 in honor_koutsu and jikaze_34 != bakaze_34:
        confirmed_han += 1
    confirmed_han += sum(1 for t in _SANGEN_34 if t in honor_koutsu)
    if has_tanyao:
        confirmed_han += 1
    if has_chinitsu:
        confirmed_han += 6 if not is_open else 5
    elif has_honitsu:
        confirmed_han += 3 if not is_open else 2
    confirmed_han += dora_count + aka_total
    scalar[29] = min(confirmed_han / 8.0, 1.0)

    # [30-37] 花色/做牌方向
    if total_tiles > 0:
        scalar[30] = man_cnt / total_tiles
        scalar[31] = pin_cnt / total_tiles
        scalar[32] = sou_cnt / total_tiles
        scalar[33] = honor_cnt / total_tiles
    scalar[34] = 1.0 if has_junchan else 0.0
    scalar[35] = 1.0 if has_chanta else 0.0
    scalar[36] = 1.0 if has_honitsu else 0.0
    scalar[37] = 1.0 if has_chinitsu else 0.0

    # [38-45] 进展 / 风险
    scalar[38] = min(progress.ukeire_live_count / 34.0, 1.0)
    scalar[39] = 0.0
    scalar[40] = min(tenpai_live_tile_count / 136.0, 1.0)
    scalar[41] = pair_count / 7.0
    scalar[42] = taatsu_count / 6.0
    scalar[43] = ankoutsu_cnt / 4.0
    furiten_list = state.get("furiten", [False] * 4)
    scalar[44] = 1.0 if (actor < len(furiten_list) and furiten_list[actor]) else 0.0
    scalar[45] = riichi_threat_count / 3.0

    # [46-52] 特殊型 / 场况
    scalar[46] = chiitoi_shanten / 6.0
    scalar[47] = kokushi_shanten / 13.0
    scalar[48] = 1.0 if actor == oya else 0.0
    scalar[49] = is_all_last_like
    scalar[50] = top_gap / 30000.0
    scalar[51] = bottom_gap / 30000.0
    scalar[52] = tiles_left / 70.0
    # [53-55] v3 no longer models good-shape/improvement; keep reserved slots stable.
    scalar[53] = 0.0
    scalar[54] = 0.0
    scalar[55] = float(np.clip(wait_hiddenness_ratio, 0.0, 1.0))

    if collect_timings:
        timings["fill_s"] = time.perf_counter() - t_fill0
        return tile_feat, scalar, timings
    return tile_feat, scalar, timings
