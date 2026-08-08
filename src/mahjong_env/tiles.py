from __future__ import annotations

from typing import Dict, List

SUITED = ("m", "p", "s")
HONORS = ("E", "S", "W", "N", "P", "F", "C")

AKA_DORA_TILES = ("5mr", "5pr", "5sr")

NORMAL_SUITED_TILES = (
    "1m",
    "2m",
    "3m",
    "4m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "6s",
    "7s",
    "8s",
    "9s",
)


def normalize_tile(tile: str) -> str:
    if tile.endswith("r"):
        return tile[:-1]
    return tile


def is_aka_dora(tile: str) -> bool:
    return tile in AKA_DORA_TILES


def all_discardable_tiles() -> List[str]:
    tiles: List[str] = []
    for suit in SUITED:
        for n in range(1, 10):
            tiles.append(f"{n}{suit}")
    tiles.extend(HONORS)
    return tiles


def all_discardable_tiles_with_aka() -> List[str]:
    tiles = []
    for suit in SUITED:
        for n in range(1, 10):
            if n == 5:
                tiles.append(f"5{suit}r")
            else:
                tiles.append(f"{n}{suit}")
    tiles.extend(HONORS)
    return tiles


def tile_without_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return normalize_tile(tile)
    return tile


# ---------------------------------------------------------------------------
# tile136 / tile34 查找表（依赖 mahjong 包）
# ---------------------------------------------------------------------------
_STR_TO_136: Dict[str, int] = {}


def _build_str_to_136() -> None:
    try:
        from mahjong.tile import (
            TilesConverter,
            FIVE_RED_MAN,
            FIVE_RED_PIN,
            FIVE_RED_SOU,
        )
    except ImportError:
        return
    for suit, kw in (("m", "man"), ("p", "pin"), ("s", "sou")):
        for n in range(1, 10):
            t = TilesConverter.string_to_136_array(**{kw: str(n)}, has_aka_dora=True)
            _STR_TO_136[f"{n}{suit}"] = t[0]
    _STR_TO_136["5mr"] = FIVE_RED_MAN
    _STR_TO_136["5pr"] = FIVE_RED_PIN
    _STR_TO_136["5sr"] = FIVE_RED_SOU
    for name, z in (
        ("E", "1"),
        ("S", "2"),
        ("W", "3"),
        ("N", "4"),
        ("P", "5"),
        ("F", "6"),
        ("C", "7"),
    ):
        _STR_TO_136[name] = TilesConverter.string_to_136_array(honors=z)[0]


_build_str_to_136()

try:
    from mahjong.tile import (
        FIVE_RED_MAN as _FRM,
        FIVE_RED_PIN as _FRP,
        FIVE_RED_SOU as _FRS,
    )

    _AKA_136: frozenset = frozenset({_FRM, _FRP, _FRS})
except ImportError:
    _AKA_136 = frozenset()


def tile_to_136(tile: str) -> int:
    """项目字符串 → tile136；未知牌返回 -1。"""
    return _STR_TO_136.get(tile, -1)


def tile_to_34(tile: str) -> int:
    """项目字符串 → tile34（0-33）；未知牌返回 -1。"""
    t136 = _STR_TO_136.get(tile, -1)
    return t136 // 4 if t136 >= 0 else -1


def tile_is_aka(tile: str) -> bool:
    """是否赤宝牌（通过 tile136 判断）。"""
    return _STR_TO_136.get(tile, -1) in _AKA_136
