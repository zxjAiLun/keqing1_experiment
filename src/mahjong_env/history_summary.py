from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HISTORY_SUMMARY_DIM = 20
_RECENCY_CAP = 16
_REL_ORDER = (0, 1, 2, 3)
_EVENT_TYPES = ("discard", "call", "kan", "riichi")


def empty_history_summary() -> np.ndarray:
    return np.zeros((HISTORY_SUMMARY_DIM,), dtype=np.float16)


def _meaningful_event_kind(event_type: str) -> str | None:
    if event_type == "dahai":
        return "discard"
    if event_type in {"chi", "pon"}:
        return "call"
    if event_type in {"daiminkan", "ankan", "kakan_accepted"}:
        return "kan"
    if event_type == "reach":
        return "riichi"
    return None


def _recency_norm(distance: int | None) -> float:
    if distance is None or distance <= 0:
        return 0.0
    return max(0.0, 1.0 - min(distance, _RECENCY_CAP) / float(_RECENCY_CAP))


def compute_history_summary(
    events: Sequence[dict],
    event_index: int,
    actor: int,
) -> np.ndarray:
    out = empty_history_summary()
    if event_index <= 0 or not events:
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

    meaningful_events: list[tuple[int, int, str]] = []
    discard_count = 0
    call_count = 0
    kan_count = 0
    riichi_count = 0
    last_seen: dict[tuple[int, str], int] = {}

    for idx in range(kyoku_start, end):
        event = events[idx]
        kind = _meaningful_event_kind(str(event.get("type", "")))
        if kind is None:
            continue
        event_actor_raw = event.get("actor")
        event_actor = (
            int(event_actor_raw)
            if isinstance(event_actor_raw, int) and 0 <= event_actor_raw <= 3
            else 4
        )
        meaningful_events.append((idx, event_actor, kind))
        current_pos = len(meaningful_events)
        if kind == "discard":
            discard_count += 1
        elif kind == "call":
            call_count += 1
        elif kind == "kan":
            kan_count += 1
        elif kind == "riichi":
            riichi_count += 1
        if event_actor != 4:
            last_seen[(event_actor, kind)] = current_pos

    if not meaningful_events:
        return out

    out[0] = np.float16(min(discard_count / 60.0, 1.0))
    out[1] = np.float16(min(call_count / 16.0, 1.0))
    out[2] = np.float16(min(kan_count / 8.0, 1.0))
    out[3] = np.float16(min(riichi_count / 4.0, 1.0))

    total_meaningful = len(meaningful_events)
    offset = 4
    for rel in _REL_ORDER:
        who = (int(actor) + rel) % 4
        for kind in _EVENT_TYPES:
            last_pos = last_seen.get((who, kind))
            distance = None if last_pos is None else total_meaningful - last_pos + 1
            out[offset] = np.float16(_recency_norm(distance))
            offset += 1
    return out


__all__ = [
    "HISTORY_SUMMARY_DIM",
    "compute_history_summary",
    "empty_history_summary",
]
