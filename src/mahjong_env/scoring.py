from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol

import keqing_core
from keqing_core import build_136_pool_entries as _build_136_pool_entries_native
from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.tile import TilesConverter
from mahjong.utils import is_aka_dora, plus_dora

from mahjong_env.state import GameState, PlayerState
from mahjong_env.tiles import normalize_tile


_CALCULATOR = HandCalculator()
_RULES = OptionalRules(has_open_tanyao=True, has_aka_dora=True)
_WIND_TO_CONST = {"E": EAST, "S": SOUTH, "W": WEST, "N": NORTH}
_HONOR_TO_DIGIT = {"E": "1", "S": "2", "W": "3", "N": "4", "P": "5", "F": "6", "C": "7"}
_ONE_LINE_TO_TILE = {
    "0m": "5mr",
    "0p": "5pr",
    "0s": "5sr",
    "1z": "E",
    "2z": "S",
    "3z": "W",
    "4z": "N",
    "5z": "P",
    "6z": "F",
    "7z": "C",
}


class _ScoreBackend(Protocol):
    def estimate_hand_value(
        self,
        *,
        tiles136: List[int],
        win_tile: int,
        melds: List[Meld],
        dora_indicators: List[int],
        config: HandConfig,
    ) -> Any: ...


class _PythonMahjongBackend:
    def estimate_hand_value(
        self,
        *,
        tiles136: List[int],
        win_tile: int,
        melds: List[Meld],
        dora_indicators: List[int],
        config: HandConfig,
    ):
        return _CALCULATOR.estimate_hand_value(
            tiles136,
            win_tile,
            melds=melds,
            dora_indicators=dora_indicators,
            config=config,
        )


_SCORE_BACKEND: _ScoreBackend = _PythonMahjongBackend()


@dataclass
class HoraResult:
    han: int
    fu: int
    yaku: List[str]
    yaku_details: List[Dict[str, int | str]]
    is_open_hand: bool
    cost: Dict
    deltas: List[int]


@dataclass(frozen=True)
class _HoraTruth:
    han: int
    fu: int
    yaku: List[str]
    yaku_details: List[Dict[str, int | str]]
    is_open_hand: bool
    cost: Dict
    deltas: List[int]
    dora_count: int
    ura_count: int
    aka_count: int
    backend_name: str
    truth_source: str


def _build_prepared_hora_payload_from_view(
    *,
    context: "_ScoreContext",
    player_view: "_ScorePlayerView",
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    ura_dora_markers: Optional[List[str]] = None,
    is_rinshan: bool = False,
    is_chankan: bool = False,
    is_haitei: bool = False,
    is_houtei: bool = False,
) -> dict:
    closed_tiles = list(player_view.hand_tiles)
    if not is_tsumo:
        closed_tiles.append(pai)
    return {
        "closed_tiles": closed_tiles,
        "melds": [
            {
                "type": meld.get("type"),
                "opened": meld.get("type") in {"chi", "pon", "daiminkan", "kakan"},
                "tiles": _meld_tiles(meld),
            }
            for meld in player_view.melds
        ],
        "bakaze": context.bakaze,
        "honba": context.honba,
        "kyotaku": context.kyotaku,
        "oya": context.oya,
        "dora_markers": list(context.dora_markers),
        "active_ura_markers": list(ura_dora_markers or []) if player_view.reached else [],
        "reached": bool(player_view.reached),
        "ippatsu_eligible": bool(player_view.ippatsu_eligible),
        "is_tsumo": bool(is_tsumo),
        "is_chankan": bool(is_chankan),
        "resolved_is_rinshan": bool(is_rinshan),
        "resolved_is_haitei": bool(is_haitei),
        "resolved_is_houtei": bool(is_houtei),
        "pai": pai,
        "actor": actor,
        "target": target,
        "_remaining_wall": context.remaining_wall,
    }


def _extract_hora_truth_from_native_payload(payload: dict) -> _HoraTruth:
    if not payload.get("cost"):
        raise RuntimeError("malformed Rust hora truth: missing cost")
    dora_count = int(payload.get("dora_count") or 0)
    ura_count = int(payload.get("ura_count") or 0)
    aka_count = int(payload.get("aka_count") or 0)
    return _HoraTruth(
        han=int(payload.get("han") or 0),
        fu=int(payload.get("fu") or 0),
        yaku=[str(name) for name in (payload.get("yaku") or [])],
        yaku_details=[dict(item) for item in (payload.get("yaku_details") or [])],
        is_open_hand=bool(payload.get("is_open_hand")),
        cost=dict(payload.get("cost") or {}),
        deltas=[int(v) for v in (payload.get("deltas") or [])],
        dora_count=dora_count,
        ura_count=ura_count,
        aka_count=aka_count,
        backend_name=str(payload.get("backend_name") or ""),
        truth_source=str(payload.get("truth_source") or ""),
    )


def _evaluate_hora_truth_from_prepared_payload(prepared: dict) -> tuple[str, _HoraTruth | None]:
    if not keqing_core.is_enabled():
        return "missing_capability", None
    try:
        payload = keqing_core.evaluate_hora_truth_from_prepared(prepared)
    except RuntimeError as exc:
        if keqing_core.is_missing_rust_capability_error(exc):
            return "missing_capability", None
        if "no cost" in str(exc):
            return "invalid_hora", None
        raise
    return "truth", _extract_hora_truth_from_native_payload(payload)


@dataclass(frozen=True)
class _ScoreContext:
    bakaze: str
    honba: int
    kyotaku: int
    oya: int
    dora_markers: List[str]
    remaining_wall: Optional[int]


@dataclass(frozen=True)
class _ScorePlayerView:
    hand_tiles: List[str]
    melds: List[dict]
    reached: bool
    ippatsu_eligible: bool
    rinshan_tsumo: bool


def _state_from_snapshot(snapshot: dict, *, actor: int | None = None) -> GameState:
    state = GameState()
    state.bakaze = snapshot.get("bakaze", "E")
    state.kyoku = int(snapshot.get("kyoku", 1))
    state.honba = int(snapshot.get("honba", 0))
    state.kyotaku = int(snapshot.get("kyotaku", 0))
    state.oya = int(snapshot.get("oya", 0))
    state.dora_markers = list(snapshot.get("dora_markers", []))
    state.scores = list(snapshot.get("scores", [25000, 25000, 25000, 25000]))
    state.actor_to_move = snapshot.get("actor_to_move")
    state.last_discard = snapshot.get("last_discard")
    state.last_tsumo = list(snapshot.get("last_tsumo", [None, None, None, None]))
    state.last_tsumo_raw = list(snapshot.get("last_tsumo_raw", [None, None, None, None]))
    state.remaining_wall = snapshot.get("remaining_wall")
    state.players = [PlayerState() for _ in range(4)]
    hand_owner = actor if actor is not None else snapshot.get("actor")
    for pid in range(4):
        player = state.players[pid]
        hand = snapshot.get("hand", []) if pid == hand_owner else []
        player.hand = Counter(hand)
        player.discards = list((snapshot.get("discards") or [[], [], [], []])[pid])
        player.melds = list((snapshot.get("melds") or [[], [], [], []])[pid])
        player.reached = bool((snapshot.get("reached") or [False] * 4)[pid])
        player.pending_reach = bool((snapshot.get("pending_reach") or [False] * 4)[pid])
        player.ippatsu_eligible = bool(
            (snapshot.get("ippatsu_eligible") or [False] * 4)[pid]
        )
        player.rinshan_tsumo = bool(
            (snapshot.get("rinshan_tsumo") or [False] * 4)[pid]
        )
    return state


def _context_from_state(state: GameState) -> _ScoreContext:
    return _ScoreContext(
        bakaze=state.bakaze,
        honba=int(state.honba),
        kyotaku=int(state.kyotaku),
        oya=int(state.oya),
        dora_markers=list(state.dora_markers),
        remaining_wall=state.remaining_wall,
    )


def _player_view_from_state(state: GameState, actor: int) -> _ScorePlayerView:
    player = state.players[actor]
    return _ScorePlayerView(
        hand_tiles=list(player.hand.elements()),
        melds=list(player.melds),
        reached=bool(player.reached),
        ippatsu_eligible=bool(player.ippatsu_eligible),
        rinshan_tsumo=bool(player.rinshan_tsumo),
    )


def _player_view_from_snapshot(snapshot: dict, *, actor: int) -> _ScorePlayerView:
    hand_tiles = list(snapshot.get("hand", []))
    melds_all = snapshot.get("melds") or [[], [], [], []]
    reached_all = snapshot.get("reached") or [False] * 4
    ippatsu_all = snapshot.get("ippatsu_eligible") or [False] * 4
    rinshan_all = snapshot.get("rinshan_tsumo") or [False] * 4
    return _ScorePlayerView(
        hand_tiles=hand_tiles,
        melds=list(melds_all[actor]),
        reached=bool(reached_all[actor]),
        ippatsu_eligible=bool(ippatsu_all[actor]),
        rinshan_tsumo=bool(rinshan_all[actor]),
    )


def _context_from_snapshot(snapshot: dict) -> _ScoreContext:
    return _ScoreContext(
        bakaze=snapshot.get("bakaze", "E"),
        honba=int(snapshot.get("honba", 0)),
        kyotaku=int(snapshot.get("kyotaku", 0)),
        oya=int(snapshot.get("oya", 0)),
        dora_markers=list(snapshot.get("dora_markers", [])),
        remaining_wall=snapshot.get("remaining_wall"),
    )


def _tile_to_one_line(tile: str) -> str:
    if tile == "5mr":
        return "0m"
    if tile == "5pr":
        return "0p"
    if tile == "5sr":
        return "0s"
    if tile in _HONOR_TO_DIGIT:
        return _HONOR_TO_DIGIT[tile] + "z"
    return tile


def _tiles_to_converter_args(tiles: List[str]) -> Dict[str, str]:
    grouped = defaultdict(list)
    for tile in tiles:
        one_line = _tile_to_one_line(tile)
        grouped[one_line[-1]].append(one_line[0])

    def _sorted_group(suit: str) -> str | None:
        values = grouped[suit]
        if not values:
            return None
        # Under aka-dora rules, a suit cannot contain four plain 5s.
        # If all four copies of 5 are present and none is explicitly marked
        # red, promote one to aka so the 136-tile pool stays physically valid.
        if suit in "mps":
            five_count = values.count("5")
            aka_count = values.count("0")
            if five_count >= 4 and aka_count == 0:
                promoted = False
                adjusted: list[str] = []
                for value in values:
                    if not promoted and value == "5":
                        adjusted.append("0")
                        promoted = True
                    else:
                        adjusted.append(value)
                values = adjusted
        return "".join(sorted(values))

    return {
        "man": _sorted_group("m"),
        "pin": _sorted_group("p"),
        "sou": _sorted_group("s"),
        "honors": _sorted_group("z"),
    }


def _build_136_pool_python(tiles: List[str]) -> Dict[str, List[int]]:
    converter_args = _tiles_to_converter_args(tiles)
    ids = TilesConverter.string_to_136_array(has_aka_dora=True, **converter_args)
    pool: Dict[str, List[int]] = defaultdict(list)
    for tile_id in sorted(ids):
        one_line = TilesConverter.to_one_line_string([tile_id], print_aka_dora=True)
        pool[_ONE_LINE_TO_TILE.get(one_line, one_line)].append(tile_id)
    return pool


def _build_136_pool(tiles: List[str]) -> Dict[str, List[int]]:
    try:
        return {tile: list(ids) for tile, ids in _build_136_pool_entries_native(tiles)}
    except RuntimeError:
        return _build_136_pool_python(tiles)


def _take_tile_id(pool: Dict[str, List[int]], tile: str) -> int:
    exact = pool.get(tile)
    if exact:
        return exact.pop(0)

    normalized = normalize_tile(tile)
    fallback = pool.get(normalized)
    if fallback:
        return fallback.pop(0)

    if len(normalized) == 2 and normalized[0] == "5" and normalized[1] in "mps":
        aka = normalized + "r"
        aka_pool = pool.get(aka)
        if aka_pool:
            return aka_pool.pop(0)

    raise ValueError(f"failed to allocate tile id for {tile} from pool")


def _meld_tile_sort_key(tile: str) -> tuple[int, int, int]:
    normalized = normalize_tile(tile)
    suit = normalized[-1]
    if suit in "mps":
        return ("mps".index(suit), int(normalized[0]), 0 if tile.endswith("r") else 1)
    honor_digit = _HONOR_TO_DIGIT.get(normalized, "9")
    return (3, int(honor_digit), 0 if tile.endswith("r") else 1)


def _meld_tiles(meld: dict) -> List[str]:
    consumed = list(meld.get("consumed", []))
    called_tile = meld.get("pai_raw") or meld.get("pai")
    if meld.get("type") in {"ankan", "kakan"}:
        return sorted(consumed, key=_meld_tile_sort_key)
    if called_tile:
        return sorted(consumed + [called_tile], key=_meld_tile_sort_key)
    return sorted(consumed, key=_meld_tile_sort_key)


def _to_mahjong_meld(meld: dict, pool: Dict[str, List[int]]) -> Meld:
    meld_type = meld.get("type")
    tiles = [_take_tile_id(pool, tile) for tile in _meld_tiles(meld)]
    opened = meld_type in ("chi", "pon", "daiminkan", "kakan")
    return Meld(meld_type="kan" if "kan" in meld_type else meld_type, tiles=tiles, opened=opened)


def _to_mahjong_meld_from_payload(meld: dict, pool: Dict[str, List[int]]) -> Meld:
    meld_type = meld.get("type")
    tiles = [_take_tile_id(pool, tile) for tile in (meld.get("tiles") or [])]
    opened = bool(meld.get("opened", False))
    return Meld(meld_type="kan" if "kan" in str(meld_type) else meld_type, tiles=tiles, opened=opened)


def _to_mahjong_meld_from_id_payload(meld: dict) -> Meld:
    meld_type = meld.get("type")
    tiles = [int(v) for v in (meld.get("tile_ids") or [])]
    opened = bool(meld.get("opened", False))
    return Meld(meld_type="kan" if "kan" in str(meld_type) else meld_type, tiles=tiles, opened=opened)


def _prepared_hora_payload_from_snapshot(
    snapshot: dict,
    *,
    actor: int,
    target: Optional[int] = None,
    pai: str,
    is_tsumo: bool,
    is_chankan: bool = False,
    is_rinshan: Optional[bool] = None,
    is_haitei: Optional[bool] = None,
    is_houtei: Optional[bool] = None,
):
    if not keqing_core.is_enabled():
        return None
    snapshot_payload = dict(snapshot)
    if not is_tsumo and target is not None:
        if is_chankan:
            if not snapshot_payload.get("last_kakan"):
                snapshot_payload["last_kakan"] = {"actor": target, "pai": pai, "pai_raw": pai}
        else:
            if not snapshot_payload.get("last_discard"):
                snapshot_payload["last_discard"] = {"actor": target, "pai": pai, "pai_raw": pai}
    try:
        if not keqing_core.can_hora_shape_from_snapshot(snapshot_payload, actor, pai, is_tsumo):
            return False
        return keqing_core.prepare_hora_evaluation_from_snapshot(
            snapshot_payload,
            actor,
            pai,
            is_tsumo,
            is_chankan=is_chankan,
            is_rinshan=is_rinshan,
            is_haitei=is_haitei,
            is_houtei=is_houtei,
        )
    except RuntimeError as exc:
        if keqing_core.is_missing_rust_capability_error(exc):
            return None
        raise


def _prepared_hora_payload_from_state(
    state: GameState,
    *,
    actor: int,
    target: Optional[int] = None,
    pai: str,
    is_tsumo: bool,
    is_chankan: bool = False,
    is_rinshan: Optional[bool] = None,
    is_haitei: Optional[bool] = None,
    is_houtei: Optional[bool] = None,
    ura_dora_markers: Optional[List[str]] = None,
):
    snapshot = state.snapshot(actor)
    snapshot.pop("feature_tracker", None)
    prepared = _prepared_hora_payload_from_snapshot(
        snapshot,
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        is_chankan=is_chankan,
        is_rinshan=is_rinshan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
    )
    if isinstance(prepared, dict):
        prepared["active_ura_markers"] = list(ura_dora_markers or []) if prepared.get("reached") else []
        prepared["_remaining_wall"] = state.remaining_wall
        prepared.setdefault("actor", actor)
        if target is not None:
            prepared["target"] = target
    return prepared


def _estimate_hand_value_from_prepared_payload(
    prepared: dict,
    *,
    actor: int,
    pai: str,
):
    dora_markers = list(prepared.get("dora_markers", []))
    allocation = None
    if keqing_core.is_enabled():
        try:
            allocation = keqing_core.prepare_hora_tile_allocation(prepared)
        except RuntimeError as exc:
            if keqing_core.is_missing_rust_capability_error(exc):
                allocation = None
            else:
                raise
    if allocation is not None:
        closed_tile_ids = [int(v) for v in allocation.get("closed_tile_ids", [])]
        win_tile = int(allocation["win_tile"])
        meld_objects = []
        meld_tile_ids: List[int] = []
        for meld in allocation.get("melds", []):
            meld_object = _to_mahjong_meld_from_id_payload(meld)
            meld_objects.append(meld_object)
            meld_tile_ids.extend(list(meld_object.tiles))
        tiles136 = closed_tile_ids + meld_tile_ids
        dora_ids = [int(v) for v in allocation.get("dora_ids", [])]
        ura_ids = [int(v) for v in allocation.get("ura_ids", [])]
    else:
        hand_tiles = list(prepared.get("closed_tiles", []))
        melds = list(prepared.get("melds", []))
        all_tiles = hand_tiles[:]
        for meld in melds:
            all_tiles.extend(meld.get("tiles", []))
        active_ura_markers = list(prepared.get("active_ura_markers", []))
        all_tiles.extend(dora_markers)
        all_tiles.extend(active_ura_markers)
        pool = _build_136_pool(all_tiles)
        closed_tile_ids = [_take_tile_id(pool, tile) for tile in hand_tiles]
        win_tile = _find_win_tile_id(closed_tile_ids, hand_tiles, pai)
        meld_objects = []
        meld_tile_ids: List[int] = []
        for meld in melds:
            meld_object = _to_mahjong_meld_from_payload(meld, pool)
            meld_objects.append(meld_object)
            meld_tile_ids.extend(list(meld_object.tiles))
        tiles136 = closed_tile_ids + meld_tile_ids
        dora_ids = [_take_tile_id(pool, marker) for marker in dora_markers]
        ura_ids = [_take_tile_id(pool, marker) for marker in active_ura_markers]
    context = _ScoreContext(
        bakaze=str(prepared["bakaze"]),
        honba=int(prepared["honba"]),
        kyotaku=int(prepared["kyotaku"]),
        oya=int(prepared["oya"]),
        dora_markers=dora_markers,
        remaining_wall=prepared.get("_remaining_wall"),
    )
    config = _cached_hand_config(
        is_tsumo=bool(prepared["is_tsumo"]),
        is_riichi=bool(prepared["reached"]),
        is_ippatsu=bool(prepared["ippatsu_eligible"]),
        is_rinshan=bool(prepared["resolved_is_rinshan"]),
        is_chankan=bool(prepared["is_chankan"]),
        is_haitei=bool(prepared["resolved_is_haitei"]),
        is_houtei=bool(prepared["resolved_is_houtei"]),
        player_wind=_player_wind_from_context(context, actor),
        round_wind=_round_wind_from_context(context),
        kyoutaku_number=int(prepared["kyotaku"]),
        tsumi_number=int(prepared["honba"]),
    )
    response = _estimate_hand_value_backend(
        tiles136=tiles136,
        win_tile=win_tile,
        melds=meld_objects,
        dora_indicators=dora_ids + ura_ids,
        config=config,
    )
    return response, context, tiles136, dora_ids, ura_ids


def _estimate_hand_value_backend(
    *,
    tiles136: List[int],
    win_tile: int,
    melds,
    dora_indicators: List[int],
    config: HandConfig,
):
    return _SCORE_BACKEND.estimate_hand_value(
        tiles136=tiles136,
        win_tile=win_tile,
        melds=melds,
        dora_indicators=dora_indicators,
        config=config,
    )


def _find_win_tile_id(hand_tile_ids: List[int], hand_tiles: List[str], pai: str) -> int:
    for tile_id, tile in zip(reversed(hand_tile_ids), reversed(hand_tiles)):
        if tile == pai:
            return tile_id
    pai_norm = normalize_tile(pai)
    for tile_id, tile in zip(reversed(hand_tile_ids), reversed(hand_tiles)):
        if normalize_tile(tile) == pai_norm:
            return tile_id
    raise ValueError(f"failed to find win tile id for {pai}")


def _player_wind(state: GameState, actor: int) -> int:
    order = ["E", "S", "W", "N"]
    return _WIND_TO_CONST[order[(actor - state.oya) % 4]]


def _round_wind(state: GameState) -> int:
    return _WIND_TO_CONST.get(state.bakaze, EAST)


def _player_wind_from_context(context: _ScoreContext, actor: int) -> int:
    order = ["E", "S", "W", "N"]
    return _WIND_TO_CONST[order[(actor - context.oya) % 4]]


def _round_wind_from_context(context: _ScoreContext) -> int:
    return _WIND_TO_CONST.get(context.bakaze, EAST)


@lru_cache(maxsize=256)
def _cached_hand_config(
    *,
    is_tsumo: bool,
    is_riichi: bool,
    is_ippatsu: bool,
    is_rinshan: bool,
    is_chankan: bool,
    is_haitei: bool,
    is_houtei: bool,
    player_wind: int,
    round_wind: int,
    kyoutaku_number: int,
    tsumi_number: int,
) -> HandConfig:
    return HandConfig(
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        is_ippatsu=is_ippatsu,
        is_rinshan=is_rinshan,
        is_chankan=is_chankan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        player_wind=player_wind,
        round_wind=round_wind,
        kyoutaku_number=kyoutaku_number,
        tsumi_number=tsumi_number,
        options=_RULES,
    )


def _yaku_han(yaku_obj, is_open_hand: bool) -> int:
    han = yaku_obj.han_open if is_open_hand else yaku_obj.han_closed
    if han is None:
        han = yaku_obj.han_closed if yaku_obj.han_closed is not None else yaku_obj.han_open
    return int(han or 0)


def _build_yaku_details(
    response,
    *,
    tile_ids: List[int],
    dora_indicator_ids: List[int],
    ura_indicator_ids: List[int],
) -> List[Dict[str, int | str]]:
    details = _base_yaku_details(response)

    dora_count = sum(
        plus_dora(tile_id, dora_indicator_ids, add_aka_dora=False) for tile_id in tile_ids
    )
    ura_count = sum(
        plus_dora(tile_id, ura_indicator_ids, add_aka_dora=False) for tile_id in tile_ids
    )
    aka_count = sum(1 for tile_id in tile_ids if is_aka_dora(tile_id, aka_enabled=True))

    if dora_count:
        details.append({"key": "Dora", "name": "Dora", "han": int(dora_count)})
    if ura_count:
        details.append({"key": "Ura Dora", "name": "Ura Dora", "han": int(ura_count)})
    if aka_count:
        details.append({"key": "Aka Dora", "name": "Aka Dora", "han": int(aka_count)})

    return details


def _base_yaku_details(response) -> List[Dict[str, int | str]]:
    details: List[Dict[str, int | str]] = []
    dora_like = {"Dora", "Ura Dora", "Aka Dora"}

    for yaku_obj in response.yaku or []:
        name = str(yaku_obj.name)
        if name in dora_like:
            continue
        details.append(
            {
                "key": name,
                "name": name,
                "han": _yaku_han(yaku_obj, bool(response.is_open_hand)),
            }
        )
    return details


def _extract_hora_truth(
    response,
    *,
    context: _ScoreContext,
    actor: int,
    target: int,
    is_tsumo: bool,
    tile_ids: List[int],
    dora_indicator_ids: List[int],
    ura_indicator_ids: List[int],
) -> _HoraTruth:
    dora_count = sum(
        plus_dora(tile_id, dora_indicator_ids, add_aka_dora=False) for tile_id in tile_ids
    )
    ura_count = sum(
        plus_dora(tile_id, ura_indicator_ids, add_aka_dora=False) for tile_id in tile_ids
    )
    aka_count = sum(1 for tile_id in tile_ids if is_aka_dora(tile_id, aka_enabled=True))
    cost = dict(response.cost or {})
    if is_tsumo:
        deltas = [0, 0, 0, 0]
        if actor == context.oya:
            payment = int(cost["main"] + cost["main_bonus"])
            for pid in range(4):
                if pid == actor:
                    continue
                deltas[pid] -= payment
                deltas[actor] += payment
        else:
            dealer_payment = int(cost["main"] + cost["main_bonus"])
            non_dealer_payment = int(cost["additional"] + cost["additional_bonus"])
            for pid in range(4):
                if pid == actor:
                    continue
                payment = dealer_payment if pid == context.oya else non_dealer_payment
                deltas[pid] -= payment
                deltas[actor] += payment
        deltas[actor] += int(cost.get("kyoutaku_bonus", 0))
    else:
        payment = int(cost["main"] + cost["main_bonus"] + cost.get("kyoutaku_bonus", 0))
        deltas = [0, 0, 0, 0]
        deltas[target] -= payment
        deltas[actor] += payment
    return _HoraTruth(
        han=int(response.han or 0),
        fu=int(response.fu or 0),
        yaku=[str(y.name) for y in (response.yaku or [])],
        yaku_details=_build_yaku_details(
            response,
            tile_ids=tile_ids,
            dora_indicator_ids=dora_indicator_ids,
            ura_indicator_ids=ura_indicator_ids,
        ),
        is_open_hand=bool(response.is_open_hand),
        cost=cost,
        deltas=deltas,
        dora_count=int(dora_count),
        ura_count=int(ura_count),
        aka_count=int(aka_count),
        backend_name="python-mahjong",
        truth_source="python-emergency-fallback",
    )


def _finalize_hora_result(*, truth: _HoraTruth) -> HoraResult:
    return HoraResult(
        han=truth.han,
        fu=truth.fu,
        yaku=truth.yaku,
        yaku_details=truth.yaku_details,
        is_open_hand=truth.is_open_hand,
        cost=dict(truth.cost),
        deltas=truth.deltas,
    )


def score_hora(
    state: GameState,
    *,
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    ura_dora_markers: Optional[List[str]] = None,
    is_rinshan: bool = False,
    is_chankan: bool = False,
    is_haitei: bool = False,
    is_houtei: bool = False,
) -> HoraResult:
    context = _context_from_state(state)
    prepared = _prepared_hora_payload_from_state(
        state,
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        is_chankan=is_chankan,
        is_rinshan=is_rinshan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        ura_dora_markers=ura_dora_markers,
    )
    if prepared is False:
        raise ValueError(f"invalid hora for actor={actor}: rust shape precheck failed")
    if isinstance(prepared, dict):
        status, truth = _evaluate_hora_truth_from_prepared_payload(prepared)
        if status == "truth":
            assert truth is not None
            return _finalize_hora_result(truth=truth)
        if status == "invalid_hora":
            raise ValueError(f"invalid hora for actor={actor}: no cost")
        if status != "missing_capability":
            raise RuntimeError(f"unexpected native hora truth status: {status}")
        response, _context, tiles136, dora_ids, ura_ids = _estimate_hand_value_from_prepared_payload(
            prepared,
            actor=actor,
            pai=pai,
        )
        if response.error or not response.cost:
            raise ValueError(f"invalid hora for actor={actor}: {response.error or 'no cost'}")
        truth = _extract_hora_truth(
            response,
            context=context,
            actor=actor,
            target=target,
            is_tsumo=is_tsumo,
            tile_ids=tiles136,
            dora_indicator_ids=dora_ids,
            ura_indicator_ids=ura_ids,
        )
        return _finalize_hora_result(truth=truth)
    return _score_hora_from_view(
        context=context,
        player_view=_player_view_from_state(state, actor),
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        ura_dora_markers=ura_dora_markers,
        is_rinshan=is_rinshan,
        is_chankan=is_chankan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
    )


def _score_hora_from_view(
    *,
    context: _ScoreContext,
    player_view: _ScorePlayerView,
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    ura_dora_markers: Optional[List[str]] = None,
    is_rinshan: bool = False,
    is_chankan: bool = False,
    is_haitei: bool = False,
    is_houtei: bool = False,
) -> HoraResult:
    hand_tiles = list(player_view.hand_tiles)
    if not is_tsumo:
        hand_tiles = hand_tiles + [pai]

    melds = list(player_view.melds)
    all_tiles = hand_tiles[:]
    for meld in melds:
        all_tiles.extend(_meld_tiles(meld))
    all_tiles.extend(context.dora_markers)
    active_ura_markers = list(ura_dora_markers or [])
    if not player_view.reached:
        active_ura_markers = []
    all_tiles.extend(active_ura_markers)
    pool = _build_136_pool(all_tiles)

    closed_tile_ids = [_take_tile_id(pool, tile) for tile in hand_tiles]
    win_tile = _find_win_tile_id(closed_tile_ids, hand_tiles, pai)
    meld_objects = []
    meld_tile_ids: List[int] = []
    for meld in melds:
        meld_object = _to_mahjong_meld(meld, pool)
        meld_objects.append(meld_object)
        meld_tile_ids.extend(list(meld_object.tiles))
    tiles136 = closed_tile_ids + meld_tile_ids
    dora_ids = [_take_tile_id(pool, marker) for marker in context.dora_markers]
    ura_ids = [_take_tile_id(pool, marker) for marker in active_ura_markers]
    prepared = _build_prepared_hora_payload_from_view(
        context=context,
        player_view=player_view,
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        ura_dora_markers=ura_dora_markers,
        is_rinshan=is_rinshan,
        is_chankan=is_chankan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
    )
    status, truth = _evaluate_hora_truth_from_prepared_payload(prepared)
    if status == "truth":
        assert truth is not None
        return _finalize_hora_result(truth=truth)
    if status == "invalid_hora":
        raise ValueError(f"invalid hora for actor={actor}: no cost")
    if status != "missing_capability":
        raise RuntimeError(f"unexpected native hora truth status: {status}")

    config = _cached_hand_config(
        is_tsumo=is_tsumo,
        is_riichi=bool(player_view.reached),
        is_ippatsu=bool(player_view.ippatsu_eligible),
        is_rinshan=is_rinshan,
        is_chankan=is_chankan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        player_wind=_player_wind_from_context(context, actor),
        round_wind=_round_wind_from_context(context),
        kyoutaku_number=context.kyotaku,
        tsumi_number=context.honba,
    )
    response = _estimate_hand_value_backend(
        tiles136=tiles136,
        win_tile=win_tile,
        melds=meld_objects,
        dora_indicators=dora_ids + ura_ids,
        config=config,
    )
    if response.error or not response.cost:
        raise ValueError(f"invalid hora for actor={actor}: {response.error or 'no cost'}")

    truth = _extract_hora_truth(
        response,
        context=context,
        actor=actor,
        target=target,
        is_tsumo=is_tsumo,
        tile_ids=tiles136,
        dora_indicator_ids=dora_ids,
        ura_indicator_ids=ura_ids,
    )
    return _finalize_hora_result(truth=truth)


def can_hora(
    state: GameState,
    *,
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    is_chankan: bool = False,
    is_rinshan: Optional[bool] = None,
    is_haitei: Optional[bool] = None,
    is_houtei: Optional[bool] = None,
) -> bool:
    prepared = _prepared_hora_payload_from_state(
        state,
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        is_chankan=is_chankan,
        is_rinshan=is_rinshan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
    )
    if prepared is False:
        return False
    if isinstance(prepared, dict):
        status, truth = _evaluate_hora_truth_from_prepared_payload(prepared)
        if status == "truth":
            return truth is not None
        if status == "invalid_hora":
            if bool(prepared["is_tsumo"]) and bool(prepared["resolved_is_rinshan"]) and bool(prepared["resolved_is_haitei"]):
                prepared = {**prepared, "resolved_is_haitei": False}
                retry_status, retry_truth = _evaluate_hora_truth_from_prepared_payload(prepared)
                if retry_status == "truth":
                    return retry_truth is not None
                if retry_status == "invalid_hora":
                    return False
                response, _context, _tiles136, _dora_ids, _ura_ids = _estimate_hand_value_from_prepared_payload(
                    prepared,
                    actor=actor,
                    pai=pai,
                )
                return not (response.error or not response.cost)
            return False
        response, _context, _tiles136, _dora_ids, _ura_ids = _estimate_hand_value_from_prepared_payload(
            prepared,
            actor=actor,
            pai=pai,
        )
        return not (response.error or not response.cost)
    player_view = _player_view_from_state(state, actor)
    context = _context_from_state(state)
    remaining_wall = context.remaining_wall
    resolved_is_rinshan = bool(player_view.rinshan_tsumo) if is_rinshan is None else is_rinshan
    resolved_is_haitei = (
        bool(remaining_wall == 0) and is_tsumo
    ) if is_haitei is None else is_haitei
    resolved_is_houtei = (
        bool(remaining_wall == 0) and not is_tsumo
    ) if is_houtei is None else is_houtei
    try:
        _score_hora_from_view(
            context=context,
            player_view=player_view,
            actor=actor,
            target=target,
            pai=pai,
            is_tsumo=is_tsumo,
            is_rinshan=resolved_is_rinshan,
            is_chankan=is_chankan,
            is_haitei=resolved_is_haitei,
            is_houtei=resolved_is_houtei,
        )
    except Exception:
        # Project replay semantics allow the final live-wall kan draw to be
        # tagged as both rinshan and haitei. The scoring backend rejects that
        # combination, but legality reconstruction should still recognize the
        # winning shape so strict replay preprocessing does not fail.
        if is_tsumo and resolved_is_rinshan and resolved_is_haitei:
            try:
                _score_hora_from_view(
                    context=context,
                    player_view=player_view,
                    actor=actor,
                    target=target,
                    pai=pai,
                    is_tsumo=is_tsumo,
                    is_rinshan=resolved_is_rinshan,
                    is_chankan=is_chankan,
                    is_haitei=False,
                    is_houtei=resolved_is_houtei,
                )
            except Exception:
                return False
            return True
        return False
    return True


def can_hora_from_snapshot(
    snapshot: dict,
    *,
    actor: int,
    target: int,
    pai: str,
    is_tsumo: bool,
    is_chankan: bool = False,
    is_rinshan: Optional[bool] = None,
    is_haitei: Optional[bool] = None,
    is_houtei: Optional[bool] = None,
) -> bool:
    prepared = _prepared_hora_payload_from_snapshot(
        snapshot,
        actor=actor,
        target=target,
        pai=pai,
        is_tsumo=is_tsumo,
        is_chankan=is_chankan,
        is_rinshan=is_rinshan,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
    )
    if prepared is False:
        return False
    if isinstance(prepared, dict):
        prepared["_remaining_wall"] = snapshot.get("remaining_wall")
        status, truth = _evaluate_hora_truth_from_prepared_payload(prepared)
        if status == "truth":
            return truth is not None
        if status == "invalid_hora":
            if bool(prepared["is_tsumo"]) and bool(prepared["resolved_is_rinshan"]) and bool(prepared["resolved_is_haitei"]):
                prepared = {**prepared, "resolved_is_haitei": False}
                retry_status, retry_truth = _evaluate_hora_truth_from_prepared_payload(prepared)
                if retry_status == "truth":
                    return retry_truth is not None
                if retry_status == "invalid_hora":
                    return False
                response, _context, _tiles136, _dora_ids, _ura_ids = _estimate_hand_value_from_prepared_payload(
                    prepared,
                    actor=actor,
                    pai=pai,
                )
                return not (response.error or not response.cost)
            return False
        response, _context, _tiles136, _dora_ids, _ura_ids = _estimate_hand_value_from_prepared_payload(
            prepared,
            actor=actor,
            pai=pai,
        )
        return not (response.error or not response.cost)
    context = _context_from_snapshot(snapshot)
    player_view = _player_view_from_snapshot(snapshot, actor=actor)
    remaining_wall = context.remaining_wall
    resolved_is_rinshan = bool(player_view.rinshan_tsumo) if is_rinshan is None else is_rinshan
    resolved_is_haitei = (
        bool(remaining_wall == 0) and is_tsumo
    ) if is_haitei is None else is_haitei
    resolved_is_houtei = (
        bool(remaining_wall == 0) and not is_tsumo
    ) if is_houtei is None else is_houtei
    try:
        _score_hora_from_view(
            context=context,
            player_view=player_view,
            actor=actor,
            target=target,
            pai=pai,
            is_tsumo=is_tsumo,
            is_rinshan=resolved_is_rinshan,
            is_chankan=is_chankan,
            is_haitei=resolved_is_haitei,
            is_houtei=resolved_is_houtei,
        )
    except Exception:
        if is_tsumo and resolved_is_rinshan and resolved_is_haitei:
            try:
                _score_hora_from_view(
                    context=context,
                    player_view=player_view,
                    actor=actor,
                    target=target,
                    pai=pai,
                    is_tsumo=is_tsumo,
                    is_rinshan=resolved_is_rinshan,
                    is_chankan=is_chankan,
                    is_haitei=False,
                    is_houtei=resolved_is_houtei,
                )
            except Exception:
                return False
            return True
        return False
    return True
