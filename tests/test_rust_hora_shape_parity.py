from __future__ import annotations

import pytest

import keqing_core
from mahjong_env.scoring import can_hora, can_hora_from_snapshot, score_hora
from mahjong_env.state import GameState, PlayerState


def _set_rust_mode(enabled: bool) -> None:
    keqing_core.enable_rust(enabled)


@pytest.fixture(autouse=True)
def _reset_rust_mode():
    _set_rust_mode(False)
    yield
    _set_rust_mode(False)


def _require_rust_hora_shape() -> None:
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")
    _set_rust_mode(True)
    try:
        keqing_core.can_hora_shape_from_snapshot({"hand": [], "melds": [[], [], [], []]}, 0, "1m", False)
    except RuntimeError:
        pytest.skip("Rust hora shape capability is not available in the installed native module")
    except Exception:
        pass
    finally:
        _set_rust_mode(False)


def test_rust_hora_shape_matches_known_open_ron_shape():
    _require_rust_hora_shape()

    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
        {"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3},
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    _set_rust_mode(True)
    assert keqing_core.can_hora_shape_from_snapshot(snap, 0, "8m", False) is True


def test_rust_hora_shape_rejects_known_non_tenpai_snapshot():
    _require_rust_hora_shape()

    player = PlayerState()
    player.hand.update(["1s", "2s", "3s", "4p", "5p", "6p", "7s", "8s", "E", "W", "N", "P", "F"])

    gs = GameState()
    gs.players = [PlayerState(), PlayerState(), player, PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "E", "pai_raw": "E"}
    snap = gs.snapshot(actor=2)

    _set_rust_mode(True)
    assert keqing_core.can_hora_shape_from_snapshot(snap, 2, "E", False) is False


def test_can_hora_from_snapshot_keeps_behavior_with_rust_shape_precheck_enabled():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
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
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=0)

    _set_rust_mode(False)
    expected = can_hora_from_snapshot(snap, actor=0, target=1, pai="8m", is_tsumo=False)
    _set_rust_mode(True)
    actual = can_hora_from_snapshot(snap, actor=0, target=1, pai="8m", is_tsumo=False)

    assert actual == expected


def test_score_hora_keeps_same_ron_result_with_rust_delta_rules():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
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
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku_details == expected.yaku_details


def test_score_hora_keeps_same_tsumo_result_with_rust_delta_rules():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5m", "5m"])

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.remaining_wall = 0

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=0, pai="5m", is_tsumo=True, is_haitei=True)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=0, pai="5m", is_tsumo=True, is_haitei=True)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku_details == expected.yaku_details


def test_can_hora_state_keeps_behavior_with_rust_state_prep_enabled():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
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
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]

    _set_rust_mode(False)
    expected = can_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)
    _set_rust_mode(True)
    actual = can_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)

    assert actual == expected


def test_score_hora_keeps_same_haitei_tsumo_result_with_rust_truth():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(
        [
            "1m", "2m", "3m",
            "1p", "2p", "3p",
            "1s", "2s", "3s",
            "7m", "8m", "9m",
            "5p", "5p",
        ]
    )

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.remaining_wall = 0

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=0, pai="5p", is_tsumo=True, is_haitei=True)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=0, pai="5p", is_tsumo=True, is_haitei=True)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku == expected.yaku
    assert actual.yaku_details == expected.yaku_details


def test_score_hora_keeps_same_houtei_ron_result_with_rust_truth():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(
        [
            "1m", "2m", "3m",
            "1p", "2p", "3p",
            "1s", "2s", "3s",
            "7m", "8m", "9m",
            "5p",
        ]
    )

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.remaining_wall = 0

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=1, pai="5p", is_tsumo=False, is_houtei=True)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=1, pai="5p", is_tsumo=False, is_houtei=True)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku == expected.yaku
    assert actual.yaku_details == expected.yaku_details


def test_score_hora_keeps_same_chankan_result_with_rust_truth():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(
        [
            "1m", "2m", "3m",
            "1p", "2p", "3p",
            "1s", "2s", "3s",
            "7m", "8m", "9m",
            "5p",
        ]
    )

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=1, pai="5p", is_tsumo=False, is_chankan=True)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=1, pai="5p", is_tsumo=False, is_chankan=True)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku == expected.yaku
    assert actual.yaku_details == expected.yaku_details


def test_score_hora_keeps_same_kokushi_13_wait_result_with_rust_truth():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(
        [
            "1m", "9m", "1p", "9p", "1s", "9s",
            "E", "S", "W", "N", "P", "F", "C",
        ]
    )

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]

    _set_rust_mode(False)
    expected = score_hora(gs, actor=0, target=1, pai="E", is_tsumo=False)
    _set_rust_mode(True)
    actual = score_hora(gs, actor=0, target=1, pai="E", is_tsumo=False)

    assert actual.han == expected.han
    assert actual.fu == expected.fu
    assert actual.cost == expected.cost
    assert actual.deltas == expected.deltas
    assert actual.yaku == expected.yaku
    assert actual.yaku_details == expected.yaku_details
