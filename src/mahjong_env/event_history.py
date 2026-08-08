from __future__ import annotations

from typing import Sequence

import numpy as np

from mahjong_env.tiles import normalize_tile, tile_to_34

EVENT_HISTORY_LEN = 48
EVENT_HISTORY_FEATURE_DIM = 5
EVENT_MAX_TURN_IDX = 15
EVENT_TYPE_PAD = 0
EVENT_TYPE_TSUMO = 1
EVENT_TYPE_DAHAI = 2
EVENT_TYPE_PON = 3
EVENT_TYPE_CHI = 4
EVENT_TYPE_DAIMINKAN = 5
EVENT_TYPE_ANKAN = 6
EVENT_TYPE_KAKAN = 7
EVENT_TYPE_REACH = 8
EVENT_TYPE_DORA = 9
EVENT_TYPE_HORA = 10
EVENT_TYPE_RYUKYOKU = 11
EVENT_TYPE_UNKNOWN = 12
EVENT_NO_ACTOR = 4
EVENT_NO_TILE = -1


def empty_event_history() -> np.ndarray:
    out = np.zeros((EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM), dtype=np.int16)
    out[:, 0] = EVENT_NO_ACTOR
    out[:, 1] = EVENT_TYPE_PAD
    out[:, 2] = EVENT_NO_TILE
    return out


def event_type_id_from_str(event_type: str) -> int:
    return {
        "tsumo": EVENT_TYPE_TSUMO,
        "dahai": EVENT_TYPE_DAHAI,
        "pon": EVENT_TYPE_PON,
        "chi": EVENT_TYPE_CHI,
        "daiminkan": EVENT_TYPE_DAIMINKAN,
        "ankan": EVENT_TYPE_ANKAN,
        "kakan": EVENT_TYPE_KAKAN,
        "reach": EVENT_TYPE_REACH,
        "dora": EVENT_TYPE_DORA,
        "hora": EVENT_TYPE_HORA,
        "ryukyoku": EVENT_TYPE_RYUKYOKU,
    }.get(event_type, EVENT_TYPE_UNKNOWN)


def event_tile_id_from_event(event: dict, event_type: str) -> int:
    if event_type not in {
        "tsumo",
        "dahai",
        "pon",
        "chi",
        "daiminkan",
        "ankan",
        "kakan",
        "dora",
    }:
        return EVENT_NO_TILE
    pai = event.get("pai")
    if not isinstance(pai, str):
        return EVENT_NO_TILE
    try:
        return int(tile_to_34(normalize_tile(pai)))
    except Exception:
        return EVENT_NO_TILE


def compute_event_history(events: Sequence[dict], event_index: int) -> np.ndarray:
    out = empty_event_history()
    if event_index <= 0:
        return out
    end = min(int(event_index), len(events))
    if end <= 0:
        return out

    kyoku_start = 0
    for idx in range(end - 1, -1, -1):
        if events[idx].get("type") == "start_kyoku":
            kyoku_start = idx + 1
            break
    if kyoku_start >= end:
        return out

    slice_start = max(end - EVENT_HISTORY_LEN, kyoku_start)
    dahai_count_so_far = sum(
        1 for item in events[kyoku_start:slice_start] if item.get("type") == "dahai"
    )
    token_count = end - slice_start
    pad_len = EVENT_HISTORY_LEN - token_count

    for offset, idx in enumerate(range(slice_start, end)):
        event = events[idx]
        event_type = str(event.get("type", ""))
        actor_raw = event.get("actor")
        actor = int(actor_raw) if isinstance(actor_raw, int) and 0 <= actor_raw <= 3 else EVENT_NO_ACTOR
        turn_idx = min(max(dahai_count_so_far // 4, 0), EVENT_MAX_TURN_IDX)
        is_tedashi = 0
        if event_type == "dahai":
            is_tedashi = 0 if bool(event.get("tsumogiri", False)) else 1
        slot = pad_len + offset
        out[slot, 0] = actor
        out[slot, 1] = event_type_id_from_str(event_type)
        out[slot, 2] = event_tile_id_from_event(event, event_type)
        out[slot, 3] = turn_idx
        out[slot, 4] = is_tedashi
        if event_type == "dahai":
            dahai_count_so_far += 1

    return out


__all__ = [
    "EVENT_HISTORY_FEATURE_DIM",
    "EVENT_HISTORY_LEN",
    "EVENT_NO_ACTOR",
    "EVENT_NO_TILE",
    "EVENT_TYPE_DAHAI",
    "EVENT_TYPE_PAD",
    "EVENT_TYPE_TSUMO",
    "compute_event_history",
    "empty_event_history",
]
