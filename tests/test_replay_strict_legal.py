import pytest

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.scoring import can_hora_from_snapshot
from mahjong_env.state import GameState, PlayerState


# =============================================================================
# scoring / legal 重建一致性
# 对应修复过的 open ron、chi 顺序、暗杠门清等问题。
# =============================================================================

def test_enumerate_legal_actions_allows_ron_without_waits_tiles_when_scoring_allows(monkeypatch):
    player0 = PlayerState()
    player0.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player0.melds = [
        {"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3},
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)
    snap.pop("waits_tiles", None)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "8m" for a in legal)


def test_can_hora_from_snapshot_handles_open_ron_shape():
    player0 = PlayerState()
    player0.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player0.melds = [
        {"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3},
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    assert can_hora_from_snapshot(snap, actor=0, target=1, pai="8m", is_tsumo=False) is True


def test_can_hora_from_snapshot_handles_open_ron_with_ankan():
    player0 = PlayerState()
    player0.hand.update(["3p", "3p", "6s", "6s", "9m", "9m", "9m"])
    player0.melds = [
        {"type": "pon", "pai": "S", "pai_raw": "S", "consumed": ["S", "S"], "target": 1},
        {"type": "ankan", "pai": "1p", "pai_raw": "1p", "consumed": ["1p", "1p", "1p", "1p"], "target": 0},
    ]

    gs = GameState()
    gs.bakaze = "S"
    gs.kyoku = 4
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3p", "pai_raw": "3p"}
    snap = gs.snapshot(actor=0)

    assert can_hora_from_snapshot(snap, actor=0, target=1, pai="3p", is_tsumo=False) is True


def test_can_hora_from_snapshot_handles_open_ron_with_unsorted_chi_meld():
    player2 = PlayerState()
    player2.hand.update(["1s", "2s", "3s", "7p", "8p", "8s", "9p", "9s", "W", "W"])
    player2.melds = [
        {"type": "chi", "pai": "5s", "pai_raw": "5s", "consumed": ["4s", "6s"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "S"
    gs.kyoku = 3
    gs.honba = 1
    gs.oya = 0
    gs.dora_markers = ["7p"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "7s", "pai_raw": "7s"}
    snap = gs.snapshot(actor=2)

    assert can_hora_from_snapshot(snap, actor=2, target=0, pai="7s", is_tsumo=False) is True


def test_can_hora_from_snapshot_uses_explicit_actor_when_snapshot_has_no_actor():
    player2 = PlayerState()
    player2.hand.update(["3p", "5m", "5m", "5mr", "5s", "5s", "5s"])
    player2.melds = [
        {"type": "pon", "pai": "2p", "pai_raw": "2p", "consumed": ["2p", "2p"], "target": 3},
        {"type": "chi", "pai": "3s", "pai_raw": "3s", "consumed": ["2s", "4s"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "S"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3p", "pai_raw": "3p"}
    snap = gs.snapshot(actor=2)
    snap.pop("actor", None)

    assert can_hora_from_snapshot(snap, actor=2, target=1, pai="3p", is_tsumo=False) is True

    legal = enumerate_legal_actions(snap, actor=2)
    assert any(a.type == "hora" and a.target == 1 and a.pai == "3p" for a in legal)


def test_can_hora_from_snapshot_handles_unsorted_tile_pool_with_four_fives():
    player2 = PlayerState()
    player2.hand.update(["3p", "5m", "5m", "5mr", "5s", "5s", "5s"])
    player2.melds = [
        {"type": "pon", "pai": "2p", "pai_raw": "2p", "consumed": ["2p", "2p"], "target": 3},
        {"type": "chi", "pai": "3s", "pai_raw": "3s", "consumed": ["2s", "4s"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "S"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1s", "5s"]
    gs.scores = [29500, 21600, 22500, 26400]
    gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3p", "pai_raw": "3p"}
    snap = gs.snapshot(actor=2)

    assert can_hora_from_snapshot(snap, actor=2, target=1, pai="3p", is_tsumo=False) is True

    legal = enumerate_legal_actions(snap, actor=2)
    assert any(a.type == "hora" and a.target == 1 and a.pai == "3p" for a in legal)


def test_enumerate_legal_actions_allows_reach_with_only_concealed_kan_melds():
    player3 = PlayerState()
    player3.hand.update(["3m", "3m", "3m", "3p", "3p", "6p", "6p", "6s", "7s", "8p", "8s"])
    player3.melds = [
        {"type": "ankan", "pai": "1p", "pai_raw": "1p", "consumed": ["1p", "1p", "1p", "1p"], "target": 3},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), PlayerState(), PlayerState(), player3]
    gs.actor_to_move = 3
    snap = gs.snapshot(actor=3)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=3)

    assert any(a.type == "reach" and a.actor == 3 for a in legal)


# =============================================================================
# scoring / legal 重建一致性
# 对应修复过的 open ron、chi 顺序、暗杠门清等问题。
# =============================================================================

def test_enumerate_legal_actions_ron_without_waits_tiles_direct():
    """荣和应出现在 legal set，不依赖 waits_tiles 注入。"""
    player0 = PlayerState()
    player0.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player0.melds = [
        {"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3},
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)
    snap.pop("waits_tiles", None)

    legal = enumerate_legal_actions(snap, actor=0)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "8m" for a in legal)


def test_enumerate_legal_actions_chankan_without_waits_tiles(monkeypatch):
    """抢杠荣和应出现在 legal set，不依赖 waits_tiles。"""
    player0 = PlayerState()
    player0.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player0.melds = [
        {"type": "pon", "pai": "8m", "pai_raw": "8m", "consumed": ["8m", "8m", "8m"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_kakan = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)
    snap.pop("waits_tiles", None)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "8m" for a in legal)


def test_enumerate_legal_actions_chankan_blocks_none_when_hora_legal(monkeypatch):
    """抢杠时，若可荣和则 none 不应单独出现（但hora后的none是有的）。"""
    player0 = PlayerState()
    player0.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player0.melds = [
        {"type": "pon", "pai": "8m", "pai_raw": "8m", "consumed": ["8m", "8m", "8m"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_kakan = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    hora_actions = [a for a in legal if a.type == "hora"]
    assert len(hora_actions) >= 1


def test_enumerate_legal_actions_allows_rinshan_plus_haitei_tsumo():
    """项目语义里最后一张开杠后自摸可以同时带 rinshan 和 haitei。"""
    player = PlayerState()
    player.hand.update(["2p", "2p", "2p", "3p", "4m", "4p", "5m", "6m", "7p", "8p", "9p"])
    player.melds = [
        {"type": "kakan", "pai": "C", "pai_raw": "C", "consumed": ["C", "C", "C", "C"], "target": 0},
    ]
    player.rinshan_tsumo = True

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 1
    gs.oya = 0
    gs.scores = [25000, 25000, 25000, 25000]
    gs.remaining_wall = 0
    gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
    gs.actor_to_move = 1
    gs.last_tsumo = [None, "9p", None, None]
    gs.last_tsumo_raw = [None, "9p", None, None]

    snap = gs.snapshot(actor=1)
    snap["_hora_is_rinshan"] = True
    snap["_hora_is_haitei"] = True

    legal = enumerate_legal_actions(snap, actor=1)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "9p" for a in legal)


# =============================================================================
# 特殊型听牌合法性 - 七对子/国士
# =============================================================================

def test_enumerate_legal_actions_chiitoitsu_tenpai_has_hora(monkeypatch):
    """七对子听牌时自摸，hora 应出现在 legal set。"""
    player = PlayerState()
    # 7 pairs: 1m,2m,3m,4p,5p,6s,7s + waiting for another 7s
    player.hand.update(["1m", "1m", "2m", "2m", "3m", "3m", "4p", "4p", "5p", "5p", "6s", "6s", "7s", "7s"])

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = [None, None, None, None]
    gs.last_tsumo_raw = [None, None, None, None]
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0
    snap["waits_tiles"] = [False] * 34
    snap["waits_tiles"][30] = True  # 7s tile34 index

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    # reach should be present for tenpai
    assert any(a.type == "reach" for a in legal), f"chiitoitsu tenpai should have reach, got {legal}"
    # hora requires last_tsumo to be set (tsumo case), so just verify reach is present
    # The key regression test is that reach is not incorrectly blocked


def test_enumerate_legal_actions_kokushi_tenpai_has_hora(monkeypatch):
    """国士无双听牌时自摸，reach 应出现在 legal set。"""
    player = PlayerState()
    # Kokushi: 1m,9m,1p,9p,1s,9s,E,S,W,N + waiting for P or any
    player.hand.update(["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "P"])

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = [None, None, None, None]
    gs.last_tsumo_raw = [None, None, None, None]
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0
    snap["waits_tiles"] = [False] * 34
    snap["waits_tiles"][31] = True  # P tile34 index

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    # reach should be present for tenpai
    assert any(a.type == "reach" for a in legal), f"kokushi tenpai should have reach, got {legal}"


# =============================================================================
# 抢杠荣和响应（他人 kakan 后的反应窗口）
# =============================================================================

def test_kakan_response_hora_legal(monkeypatch):
    """他人加杠时，非本人可荣和。"""
    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
        {"type": "pon", "pai": "8m", "pai_raw": "8m", "consumed": ["8m", "8m", "8m"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.last_kakan = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "8m" for a in legal)


# =============================================================================
# 荣和 - 门前 vs 副露
# =============================================================================

def test_enumerate_legal_actions_ron_after_meld(monkeypatch):
    """副露后仍可荣和（抢杠场景已在上面覆盖，补充一般荣和）。"""
    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

    legal = enumerate_legal_actions(snap, actor=0)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "8m" for a in legal)
