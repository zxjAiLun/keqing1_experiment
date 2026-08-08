"""动作空间定义，共 45 个动作。

索引布局：
  0-33 : dahai（弃牌，34 种牌种，赤宝牌归并到对应牌种）
  34   : reach（立直）
  35   : chi_low （吃，摸到的牌是顺子最低位）
  36   : chi_mid （吃，摸到的牌是顺子中间位）
  37   : chi_high（吃，摸到的牌是顺子最高位）
  38   : pon（碰）
  39   : daiminkan（明杠）
  40   : ankan（暗杠）
  41   : kakan（加杠）
  42   : hora（荣和/自摸统一）
  43   : ryukyoku（九种九牌流局）
  44   : none / pass

ChiType 判断（对齐 Mortal chi_type.rs）：
  去赤后，设 pai_rank = 摸到的牌数字，lo/hi = consumed 两张牌数字 min/max。
  pai_rank < lo  → chi_low
  lo ≤ pai_rank < hi → chi_mid  （即 pai_rank 在两张牌之间）
  pai_rank ≥ hi  → chi_high
"""

from __future__ import annotations
from typing import Dict, List
from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES

# ---------------------------------------------------------------------------
# 34 种牌名（不含赤宝牌）
# ---------------------------------------------------------------------------
_SUITS = ("m", "p", "s")
_TILE_NAMES: List[str] = []
for _suit in _SUITS:
    for _n in range(1, 10):
        _TILE_NAMES.append(f"{_n}{_suit}")
_TILE_NAMES.extend(["E", "S", "W", "N", "P", "F", "C"])

TILE_NAME_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(_TILE_NAMES)}
IDX_TO_TILE_NAME: Dict[int, str] = {i: t for i, t in enumerate(_TILE_NAMES)}
N_TILES = 34

# ---------------------------------------------------------------------------
# 动作索引常量
# ---------------------------------------------------------------------------
ACTION_SPACE = 45

DAHAI_OFFSET  = 0
REACH_IDX     = 34
CHI_LOW_IDX   = 35
CHI_MID_IDX   = 36
CHI_HIGH_IDX  = 37
PON_IDX       = 38
DAIMINKAN_IDX = 39
ANKAN_IDX     = 40
KAKAN_IDX     = 41
HORA_IDX      = 42
RYUKYOKU_IDX  = 43
NONE_IDX      = 44


def _deaka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return normalize_tile(tile)
    return tile


def _tile_rank(tile: str) -> int:
    """返回牌在同花色内的数字（1-9）。荣誉牌返回 -1。"""
    tile = _deaka(tile)
    if len(tile) >= 2 and tile[-1] in _SUITS:
        return int(tile[:-1])
    return -1


def chi_type_idx(pai: str, consumed: List[str]) -> int:
    """根据 Mortal ChiType 逻辑返回吃牌的动作索引。"""
    pai_rank = _tile_rank(pai)
    ranks = sorted(_tile_rank(t) for t in consumed)
    lo, hi = ranks[0], ranks[1]
    if pai_rank < lo:
        return CHI_LOW_IDX
    elif pai_rank < hi:
        return CHI_MID_IDX
    else:
        return CHI_HIGH_IDX


def action_to_idx(action: dict) -> int:
    """将 mjai 格式的动作 dict 转换为动作索引。"""
    t = action.get("type", "none")
    if t == "dahai":
        tile = _deaka(action.get("pai", ""))
        return DAHAI_OFFSET + TILE_NAME_TO_IDX.get(tile, 0)
    if t == "reach":
        return REACH_IDX
    if t == "chi":
        return chi_type_idx(action["pai"], action["consumed"])
    if t == "pon":
        return PON_IDX
    if t == "daiminkan":
        return DAIMINKAN_IDX
    if t == "ankan":
        return ANKAN_IDX
    if t == "kakan":
        return KAKAN_IDX
    if t == "hora":
        return HORA_IDX
    if t == "ryukyoku":
        return RYUKYOKU_IDX
    return NONE_IDX


def build_legal_mask(legal_actions: List[dict]) -> List[bool]:
    """返回长度为 ACTION_SPACE 的布尔 mask，合法动作位置为 True。"""
    mask = [False] * ACTION_SPACE
    for a in legal_actions:
        idx = action_to_idx(a)
        mask[idx] = True
    return mask
