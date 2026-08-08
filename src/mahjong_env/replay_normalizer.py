from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from mahjong_env.state import GameState, TileStateError, apply_event
from mahjong_env.tiles import AKA_DORA_TILES, normalize_tile
from mahjong_env.types import Action, MjaiEvent, action_dict_to_spec, action_specs_match

REPLAY_META_EVENT_TYPES = frozenset({"reach_accepted", "dora", "new_dora"})


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    return normalize_tile(tile)


def is_replay_meta_event(event_or_type: MjaiEvent | str | None) -> bool:
    if isinstance(event_or_type, dict):
        event_type = event_or_type.get("type")
    else:
        event_type = event_or_type
    return event_type in REPLAY_META_EVENT_TYPES


def normalize_replay_event(event: MjaiEvent) -> MjaiEvent:
    out = dict(event)
    if "pai" in out and isinstance(out["pai"], str):
        out["pai"] = _normalize_or_keep_aka(out["pai"])
    if "pai_raw" in out and isinstance(out["pai_raw"], str):
        out["pai_raw"] = _normalize_or_keep_aka(out["pai_raw"])
    if "consumed" in out and isinstance(out["consumed"], list):
        out["consumed"] = [_normalize_or_keep_aka(tile) for tile in out["consumed"]]
    if "dora_marker" in out and isinstance(out["dora_marker"], str):
        out["dora_marker"] = _normalize_or_keep_aka(out["dora_marker"])
    if "ura_markers" in out and isinstance(out["ura_markers"], list):
        out["ura_markers"] = [_normalize_or_keep_aka(tile) for tile in out["ura_markers"]]
    if "ura_dora_markers" in out and isinstance(out["ura_dora_markers"], list):
        out["ura_dora_markers"] = [_normalize_or_keep_aka(tile) for tile in out["ura_dora_markers"]]
    if "tehais" in out and isinstance(out["tehais"], list):
        out["tehais"] = [
            [_normalize_or_keep_aka(tile) if tile != "?" else tile for tile in tehai]
            for tehai in out["tehais"]
        ]
    return out


def _canonicalize_hora_event(event: MjaiEvent, state: GameState) -> MjaiEvent:
    out = dict(event)
    actor = out.get("actor")
    target = out.get("target")
    if actor is None or target is None:
        return out

    if actor == target:
        pai = out.get("pai")
        if pai is None:
            pai = state.last_tsumo_raw[actor] or state.last_tsumo[actor]
            if pai is not None:
                out["pai"] = pai
        if state.players[actor].rinshan_tsumo:
            out["is_rinshan"] = True
        if state.remaining_wall == 0:
            out["is_haitei"] = True
        return out

    if state.last_kakan and state.last_kakan.get("actor") == target:
        pai = out.get("pai")
        kakan_pai = state.last_kakan.get("pai_raw") or state.last_kakan.get("pai")
        if pai is None and kakan_pai is not None:
            out["pai"] = kakan_pai
        if kakan_pai is not None and normalize_tile(out.get("pai", kakan_pai)) == normalize_tile(kakan_pai):
            out["is_chankan"] = True
            return out

    last_discard = state.last_discard
    if isinstance(last_discard, dict) and last_discard.get("actor") == target:
        pai = out.get("pai")
        discard_pai = last_discard.get("pai_raw") or last_discard.get("pai")
        if pai is None and discard_pai is not None:
            out["pai"] = discard_pai
        if state.remaining_wall == 0:
            out["is_houtei"] = True
    return out


def _probe_apply_event(state: GameState, event: MjaiEvent) -> None:
    try:
        apply_event(state, event)
    except TileStateError:
        # Probe state is only used to infer canonical replay context.
        # Incomplete synthetic fixtures or partially lossy legacy streams should
        # not break normalization itself; we simply stop gaining extra hints.
        return


def normalize_replay_label_for_legal_compare(label: Dict) -> Dict:
    out = dict(label)
    if "pai" in out and isinstance(out["pai"], str):
        out["pai"] = _normalize_or_keep_aka(out["pai"])
    if "consumed" in out and isinstance(out["consumed"], list):
        out["consumed"] = [_normalize_or_keep_aka(tile) for tile in out["consumed"]]
    if "ura_markers" in out and isinstance(out["ura_markers"], list):
        out["ura_markers"] = [_normalize_or_keep_aka(tile) for tile in out["ura_markers"]]
    return out


def replay_label_matches_legal(label: Dict, legal_actions: List[Dict]) -> bool:
    label = normalize_replay_label_for_legal_compare(label)
    label_spec = action_dict_to_spec(label)
    return any(action_specs_match(label_spec, action_dict_to_spec(legal)) for legal in legal_actions)


def replay_label_to_legal_mjai(label: Dict, actor: int) -> Dict:
    label = normalize_replay_label_for_legal_compare(label)
    action_kwargs = {"type": label["type"], "actor": actor}
    if "consumed" in label:
        action_kwargs["consumed"] = label["consumed"]
    if "pai" in label:
        action_kwargs["pai"] = label["pai"]
    if "target" in label:
        action_kwargs["target"] = label["target"]
    if "tsumogiri" in label:
        action_kwargs["tsumogiri"] = bool(label["tsumogiri"])
    return Action(**action_kwargs).to_mjai()


def _auto_accept_from_pending_kakan(pending_kakan: Optional[MjaiEvent]) -> Optional[MjaiEvent]:
    if pending_kakan is None:
        return None
    return {
        "type": "kakan_accepted",
        "actor": pending_kakan["actor"],
        "pai": pending_kakan["pai"],
        "pai_raw": pending_kakan.get("pai_raw", pending_kakan["pai"]),
        "consumed": list(pending_kakan.get("consumed", [])),
        "target": pending_kakan.get("target"),
    }


def normalize_replay_events(events: Iterable[MjaiEvent]) -> List[MjaiEvent]:
    """Canonicalize external mjai logs into the project's replay event stream.

    Current responsibilities:
    - normalize tile fields while preserving aka tiles
    - inject legacy `kakan_accepted` when old logs emit `kakan` and then move on
    """

    normalized: List[MjaiEvent] = []
    pending_kakan: Optional[MjaiEvent] = None
    probe_state = GameState()

    for raw_event in events:
        event = normalize_replay_event(raw_event)
        et = event.get("type")

        if pending_kakan is not None and et not in {"hora", "kakan_accepted"}:
            auto_accept = _auto_accept_from_pending_kakan(pending_kakan)
            if auto_accept is not None:
                normalized.append(auto_accept)
                _probe_apply_event(probe_state, auto_accept)
            pending_kakan = None

        if et == "hora":
            event = _canonicalize_hora_event(event, probe_state)
        normalized.append(event)
        _probe_apply_event(probe_state, event)

        if et == "kakan":
            pending_kakan = event
        elif et == "kakan_accepted":
            pending_kakan = None
        elif not is_replay_meta_event(et) and et != "hora":
            # Any other meaningful event closes the pending legacy kakan window.
            pending_kakan = None

    return normalized


__all__ = [
    "normalize_replay_event",
    "normalize_replay_events",
    "normalize_replay_label_for_legal_compare",
    "REPLAY_META_EVENT_TYPES",
    "is_replay_meta_event",
    "replay_label_matches_legal",
    "replay_label_to_legal_mjai",
]
