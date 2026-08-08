from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from mahjong_env.feature_tracker import RoundFeatureTracker
from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES
from mahjong_env.types import MjaiEvent


class TileStateError(RuntimeError):
    """状态机试图移除不存在的牌时抛出。"""


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    return normalize_tile(tile)


@dataclass
class PlayerState:
    hand: Counter = field(default_factory=Counter)
    discards: List[dict] = field(default_factory=list)  # each: {"pai": str, "tsumogiri": bool, "reach_declared": bool}
    melds: List[dict] = field(default_factory=list)
    reached: bool = False
    pending_reach: bool = False
    furiten: bool = False          # 舍牌振听 or 立直振听
    sutehai_furiten: bool = False  # 舍牌振听（自己曾打出进张牌）
    riichi_furiten: bool = False   # 立直振听（立直后摸到进张但未自摸）
    doujun_furiten: bool = False   # 同巡振听（本巡放弃荣和）
    ippatsu_eligible: bool = False
    rinshan_tsumo: bool = False


@dataclass
class GameState:
    bakaze: str = "E"
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    dora_markers: List[str] = field(default_factory=list)
    ura_dora_markers: List[str] = field(default_factory=list)
    scores: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    players: List[PlayerState] = field(default_factory=lambda: [PlayerState() for _ in range(4)])
    actor_to_move: Optional[int] = None
    last_discard: Optional[dict] = None
    last_kakan: Optional[dict] = None
    last_tsumo: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    last_tsumo_raw: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    remaining_wall: Optional[int] = None
    pending_rinshan_actor: Optional[int] = None
    ryukyoku_tenpai_players: List[int] = field(default_factory=list)
    in_game: bool = False
    feature_tracker: Optional[RoundFeatureTracker] = None

    def snapshot(self, actor: int) -> Dict:
        if self.feature_tracker is not None:
            hand_list = sorted(self.feature_tracker.players[actor].hand_tiles)
        else:
            hand_counter = self.players[actor].hand
            hand_list: List[str] = []
            for tile, cnt in sorted(hand_counter.items()):
                hand_list.extend([tile] * cnt)
        snap = {
            "bakaze": self.bakaze,
            "kyoku": self.kyoku,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "oya": self.oya,
            "scores": self.scores[:],
            "dora_markers": self.dora_markers[:],
            "ura_dora_markers": self.ura_dora_markers[:],
            "actor": actor,
            "hand": hand_list,
            "discards": [p.discards[:] for p in self.players],
            "melds": [p.melds[:] for p in self.players],
            "reached": [p.reached for p in self.players],
            "pending_reach": [p.pending_reach for p in self.players],
            "actor_to_move": self.actor_to_move,
            "last_discard": self.last_discard.copy() if self.last_discard else None,
            "last_kakan": self.last_kakan.copy() if self.last_kakan else None,
            "last_tsumo": self.last_tsumo[:],
            "last_tsumo_raw": self.last_tsumo_raw[:],
            "remaining_wall": self.remaining_wall,
            "pending_rinshan_actor": self.pending_rinshan_actor,
            "ryukyoku_tenpai_players": self.ryukyoku_tenpai_players[:],
            "furiten": [p.furiten for p in self.players],
            "sutehai_furiten": [p.sutehai_furiten for p in self.players],
            "riichi_furiten": [p.riichi_furiten for p in self.players],
            "doujun_furiten": [p.doujun_furiten for p in self.players],
            "ippatsu_eligible": [p.ippatsu_eligible for p in self.players],
            "rinshan_tsumo": [p.rinshan_tsumo for p in self.players],
        }
        if self.feature_tracker is not None:
            snap["feature_tracker"] = self.feature_tracker.snapshot_for_actor(
                actor,
                tsumo_pai=self.last_tsumo[actor],
            )
        return snap


def _remove_tile(counter: Counter, tile: str, n: int = 1, *, context: str = "") -> bool:
    actual = counter.get(tile, 0)
    if actual < n:
        detail = f"missing tile {tile}: need {n}, have {actual}, hand={dict(counter)}"
        if context:
            detail = f"{context}: {detail}"
        raise TileStateError(detail)
    counter[tile] -= n
    if counter[tile] == 0:
        del counter[tile]
    return True


def _upgrade_pon_meld_to_kakan(player: PlayerState, pai: str, added_tile: str) -> None:
    """将已有 pon 副露升级为 kakan；若未找到则退化为追加新 meld。"""
    pai_family = normalize_tile(pai)
    for meld in player.melds:
        if meld.get("type") != "pon":
            continue
        meld_pai = meld.get("pai", "")
        if not meld_pai or normalize_tile(_normalize_or_keep_aka(meld_pai)) != pai_family:
            continue
        meld["type"] = "kakan"
        base_tiles = list(meld.get("consumed", []))
        called_tile = _normalize_or_keep_aka(meld.get("pai_raw", meld.get("pai", pai)))
        meld["consumed"] = base_tiles + [called_tile, added_tile]
        meld["pai"] = pai
        meld["pai_raw"] = pai
        meld["target"] = meld.get("target")
        return

    player.melds.append(
        {
            "type": "kakan",
            "pai": pai,
            "pai_raw": pai,
            "consumed": [pai] * 4,
            "target": None,
        }
    )


def _consume_wall_for_kan(state: GameState) -> None:
    """A successful kan reduces future live-wall draws by one.

    The following rinshan draw then comes from the dead wall and must not
    decrement `remaining_wall` again.
    """
    if state.remaining_wall is not None:
        state.remaining_wall = max(0, state.remaining_wall - 1)


def _resolve_dahai_tile_key(state: GameState, actor: int, pai_raw: str, tsumogiri: bool) -> tuple[str, str]:
    if pai_raw != "?":
        return _normalize_or_keep_aka(pai_raw), pai_raw
    if tsumogiri:
        last_raw = state.last_tsumo_raw[actor]
        last_key = state.last_tsumo[actor]
        if last_raw is not None and last_key is not None:
            return last_key, last_raw
    raise TileStateError(
        f"apply_event dahai actor={actor}: unresolved unknown discard pai='?' "
        f"(tsumogiri={tsumogiri}, last_tsumo={state.last_tsumo[actor]!r}, "
        f"last_tsumo_raw={state.last_tsumo_raw[actor]!r})"
    )


def apply_event(state: GameState, event: MjaiEvent) -> None:
    et = event["type"]
    if et == "start_game":
        state.in_game = True
        return
    if et == "start_kyoku":
        state.bakaze = event["bakaze"]
        state.kyoku = event["kyoku"]
        state.honba = event["honba"]
        state.kyotaku = event["kyotaku"]
        state.oya = event["oya"]
        state.scores = event["scores"][:]
        state.dora_markers = [_normalize_or_keep_aka(event["dora_marker"])]
        state.ura_dora_markers = []
        state.last_discard = None
        state.last_kakan = None
        state.actor_to_move = state.oya
        state.last_tsumo = [None, None, None, None]
        state.last_tsumo_raw = [None, None, None, None]
        state.remaining_wall = 70
        state.pending_rinshan_actor = None
        for pid in range(4):
            state.players[pid] = PlayerState()
            for tile in event["tehais"][pid]:
                if tile != "?":
                    state.players[pid].hand[_normalize_or_keep_aka(tile)] += 1
        state.feature_tracker = RoundFeatureTracker.from_start_kyoku(
            tehais=[
                [_normalize_or_keep_aka(tile) for tile in tehai if tile != "?"]
                for tehai in event["tehais"]
            ],
            dora_markers=state.dora_markers,
        )
        return

    if et == "tsumo":
        actor = event["actor"]
        pai = event["pai"]
        is_rinshan = bool(event.get("rinshan", False)) or state.pending_rinshan_actor == actor
        if pai != "?":
            tile_key = _normalize_or_keep_aka(pai)
            state.players[actor].hand[tile_key] += 1
            if state.feature_tracker is not None:
                state.feature_tracker.on_tsumo(actor, tile_key)
            state.last_tsumo[actor] = tile_key
            state.last_tsumo_raw[actor] = pai
            if not is_rinshan and state.remaining_wall is not None:
                state.remaining_wall = max(0, state.remaining_wall - 1)
        else:
            state.last_tsumo[actor] = None
            state.last_tsumo_raw[actor] = None
        state.actor_to_move = actor
        state.last_discard = None
        state.last_kakan = None
        state.pending_rinshan_actor = None
        # 摸牌时解除同巡振听
        state.players[actor].doujun_furiten = False
        state.players[actor].rinshan_tsumo = is_rinshan
        return

    if et == "dahai":
        actor = event["actor"]
        pai_raw = event["pai"]
        tsumogiri = bool(event.get("tsumogiri", False))
        tile_key, resolved_pai_raw = _resolve_dahai_tile_key(state, actor, pai_raw, tsumogiri)
        if not bool(event.get("skip_hand_update", False)):
            _remove_tile(
                state.players[actor].hand,
                tile_key,
                1,
                context=f"apply_event dahai actor={actor}",
            )
        if state.feature_tracker is not None:
            state.feature_tracker.on_dahai(actor, tile_key)
        reach_declared = state.players[actor].pending_reach
        state.players[actor].discards.append({"pai": tile_key, "tsumogiri": tsumogiri, "reach_declared": reach_declared})
        state.last_discard = {"actor": actor, "pai": tile_key, "pai_raw": resolved_pai_raw}
        state.last_kakan = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        state.actor_to_move = (actor + 1) % 4
        state.players[actor].ippatsu_eligible = False
        state.players[actor].rinshan_tsumo = False
        if state.players[actor].pending_reach:
            state.players[actor].reached = True
            state.players[actor].pending_reach = False
        return

    if et in ("pon", "chi", "daiminkan"):
        actor = event["actor"]
        for p in state.players:
            p.ippatsu_eligible = False
        consumed = [_normalize_or_keep_aka(x) for x in event.get("consumed", [])]
        if not bool(event.get("skip_hand_update", False)):
            for t in consumed:
                _remove_tile(
                    state.players[actor].hand,
                    t,
                    1,
                    context=f"apply_event {et} actor={actor}",
                )
        if state.feature_tracker is not None:
            state.feature_tracker.on_open_meld(
                actor,
                consumed,
                _normalize_or_keep_aka(event["pai"]),
            )
        state.players[actor].melds.append(
            {
                "type": et,
                "pai": _normalize_or_keep_aka(event["pai"]),
                "pai_raw": _normalize_or_keep_aka(event.get("pai_raw", event["pai"])),
                "consumed": consumed,
                "target": event.get("target"),
            }
        )
        state.last_discard = None
        state.last_kakan = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        state.actor_to_move = actor
        if et == "daiminkan":
            _consume_wall_for_kan(state)
            state.pending_rinshan_actor = actor
        return

    if et == "ankan":
        actor = event["actor"]
        for p in state.players:
            p.ippatsu_eligible = False
        consumed = [_normalize_or_keep_aka(x) for x in event["consumed"]]
        if not bool(event.get("skip_hand_update", False)):
            for t in consumed:
                _remove_tile(
                    state.players[actor].hand,
                    t,
                    1,
                    context=f"apply_event ankan actor={actor}",
                )
        ankan_pai = _normalize_or_keep_aka(event["consumed"][0]) if "pai" not in event else _normalize_or_keep_aka(event["pai"])
        if state.feature_tracker is not None:
            state.feature_tracker.on_ankan(actor, consumed, ankan_pai)
        state.players[actor].melds.append(
            {
                "type": "ankan",
                "pai": ankan_pai,
                "pai_raw": _normalize_or_keep_aka(event.get("pai_raw", ankan_pai)),
                "consumed": consumed,
                "target": actor,
            }
        )
        state.actor_to_move = actor
        state.last_discard = None
        state.last_kakan = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        state.ryukyoku_tenpai_players = []
        _consume_wall_for_kan(state)
        state.pending_rinshan_actor = actor
        return

    if et == "kakan":
        actor = event["actor"]
        for p in state.players:
            p.ippatsu_eligible = False
        state.last_kakan = {
            "actor": actor,
            "pai": _normalize_or_keep_aka(event["pai"]),
            "pai_raw": _normalize_or_keep_aka(event.get("pai_raw", event["pai"])),
            "consumed": [_normalize_or_keep_aka(x) for x in event.get("consumed", [])],
            "target": event.get("target"),
        }
        state.actor_to_move = actor
        state.last_discard = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        return

    if et == "kakan_accepted":
        actor = event["actor"]
        pai = _normalize_or_keep_aka(event["pai"])
        added_tile = pai
        if state.players[actor].hand.get(added_tile, 0) <= 0:
            consumed = [_normalize_or_keep_aka(x) for x in event.get("consumed", [])]
            for t in consumed:
                if state.players[actor].hand.get(t, 0) > 0:
                    added_tile = t
                    break
        if not bool(event.get("skip_hand_update", False)):
            _remove_tile(
                state.players[actor].hand,
                added_tile,
                1,
                context=f"apply_event kakan_accepted actor={actor}",
            )
        if state.feature_tracker is not None:
            state.feature_tracker.on_kakan_accepted(actor, added_tile, pai)
        _upgrade_pon_meld_to_kakan(state.players[actor], pai, added_tile)
        state.actor_to_move = actor
        state.last_discard = None
        state.last_kakan = None
        state.last_tsumo[actor] = None
        state.last_tsumo_raw[actor] = None
        _consume_wall_for_kan(state)
        state.pending_rinshan_actor = actor
        return

    if et == "reach":
        actor = event["actor"]
        state.players[actor].pending_reach = True
        return

    if et == "reach_accepted":
        actor = event["actor"]
        state.players[actor].ippatsu_eligible = True
        if "scores" in event:
            state.scores = list(event["scores"])
        if "kyotaku" in event:
            state.kyotaku = int(event["kyotaku"])
        return

    if et in ("hora", "ryukyoku"):
        state.pending_rinshan_actor = None
        if "scores" in event:
            state.scores = list(event["scores"])
        if "state_honba" in event:
            state.honba = int(event["state_honba"])
        elif "honba" in event:
            state.honba = int(event["honba"])
        if "state_kyotaku" in event:
            state.kyotaku = int(event["state_kyotaku"])
        elif "kyotaku" in event:
            state.kyotaku = int(event["kyotaku"])
        if "oya" in event:
            state.oya = int(event["oya"])
        if "ura_dora_markers" in event:
            state.ura_dora_markers = [
                _normalize_or_keep_aka(tile) for tile in event["ura_dora_markers"]
            ]
        if "tenpai_players" in event:
            state.ryukyoku_tenpai_players = [int(pid) for pid in event["tenpai_players"]]
        elif et == "hora":
            state.ryukyoku_tenpai_players = []
        state.actor_to_move = None
        state.last_discard = None
        state.last_kakan = None
        return

    if et in ("end_kyoku", "end_game"):
        state.actor_to_move = None
        state.last_discard = None
        state.last_kakan = None
        state.pending_rinshan_actor = None
        state.ryukyoku_tenpai_players = []
        return

    if et == "dora":
        state.dora_markers.append(normalize_tile(event["dora_marker"]))
        if state.feature_tracker is not None:
            state.feature_tracker.on_dora(normalize_tile(event["dora_marker"]))
        return


def visible_tiles_for_actor(state: GameState, actor: int) -> Set[str]:
    visible: Set[str] = set(state.players[actor].hand.keys())
    for p in state.players:
        visible.update(d["pai"] if isinstance(d, dict) else d for d in p.discards)
        for meld in p.melds:
            visible.update(meld.get("consumed", []))
            if "pai" in meld:
                visible.add(meld["pai"])
    return visible
