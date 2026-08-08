from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from mahjong_env.tiles import normalize_tile


MjaiEvent = Dict[str, Any]


@dataclass(frozen=True)
class ActionSpec:
    type: str
    actor: Optional[int] = None
    pai: Optional[str] = None
    consumed: Tuple[str, ...] = ()
    target: Optional[int] = None
    tsumogiri: Optional[bool] = None

    def to_action(self) -> "Action":
        return Action(
            type=self.type,
            actor=0 if self.actor is None else self.actor,
            pai=self.pai,
            consumed=list(self.consumed) if self.consumed else None,
            target=self.target,
            tsumogiri=self.tsumogiri,
        )

    def to_mjai(self) -> Dict[str, Any]:
        return self.to_action().to_mjai()


@dataclass
class Action:
    type: str
    actor: int
    pai: Optional[str] = None
    consumed: Optional[List[str]] = None
    target: Optional[int] = None
    tsumogiri: Optional[bool] = None

    def to_spec(self) -> ActionSpec:
        return ActionSpec(
            type=self.type,
            actor=None if self.type == "none" else self.actor,
            pai=self.pai,
            consumed=tuple(self.consumed or ()),
            target=self.target,
            tsumogiri=self.tsumogiri,
        )

    def to_mjai(self) -> Dict[str, Any]:
        # mjai protocol: "none" must not include "actor".
        if self.type == "none":
            return {"type": "none"}

        out: Dict[str, Any] = {"type": self.type, "actor": self.actor}
        if self.pai is not None:
            out["pai"] = self.pai
        if self.consumed is not None:
            out["consumed"] = self.consumed
        if self.target is not None:
            out["target"] = self.target
        if self.tsumogiri is not None:
            out["tsumogiri"] = self.tsumogiri
        return out


def canonical_meld_pai(action: Dict[str, Any]) -> Optional[str]:
    pai = action.get("pai")
    if pai:
        return normalize_or_keep_aka(pai)
    consumed = action.get("consumed") or []
    if not consumed:
        return None
    return normalize_or_keep_aka(consumed[0])


def normalize_or_keep_aka(tile: Optional[str]) -> Optional[str]:
    if tile is None:
        return None
    if tile in {"5mr", "5pr", "5sr"}:
        return tile
    return normalize_tile(tile)


def _normalized_tile_counter(tiles: Tuple[str, ...]) -> Counter:
    return Counter(normalize_tile(tile) for tile in tiles)


def action_dict_to_spec(action: Dict[str, Any], *, actor_hint: Optional[int] = None) -> ActionSpec:
    action_type = action.get("type")
    actor = action.get("actor", actor_hint)
    target = action.get("target")
    pai = action.get("pai")
    consumed = tuple(normalize_or_keep_aka(t) for t in (action.get("consumed") or ()))
    tsumogiri = action.get("tsumogiri")

    if action_type in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        pai = canonical_meld_pai(action)
    elif pai is not None:
        pai = normalize_or_keep_aka(pai)

    return ActionSpec(
        type=action_type,
        actor=None if action_type == "none" else actor,
        pai=pai,
        consumed=consumed,
        target=target,
        tsumogiri=tsumogiri,
    )


def action_specs_match(left: ActionSpec, right: ActionSpec) -> bool:
    if left.type != right.type:
        return False
    if left.type in {"none", "ryukyoku", "reach"}:
        return True
    if left.type == "dahai":
        # `tsumogiri` is not a legality discriminator. With duplicate tiles in hand,
        # replay reconstruction cannot always recover whether a same-face discard came
        # from the draw slot or the closed hand. Keep tsumogiri for UI/analysis, but
        # treat discard semantics as "discard this tile face".
        return left.pai == right.pai
    if left.type == "hora":
        return (
            left.target == right.target
            and (left.pai is None or right.pai is None or left.pai == right.pai)
        )
    if left.type in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        left_pai = normalize_tile(left.pai) if left.pai is not None else None
        right_pai = normalize_tile(right.pai) if right.pai is not None else None
        return (
            (left_pai == right_pai or left.pai is None or right.pai is None)
            and left.target == right.target
            and _normalized_tile_counter(left.consumed)
            == _normalized_tile_counter(right.consumed)
        )
    return left == right
