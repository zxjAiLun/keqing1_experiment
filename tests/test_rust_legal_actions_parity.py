from __future__ import annotations

import pytest

import keqing_core
from mahjong_env.legal_actions import enumerate_legal_action_specs
from mahjong_env.state import GameState, PlayerState


def _set_rust_mode(enabled: bool) -> None:
    keqing_core.enable_rust(enabled)


@pytest.fixture(autouse=True)
def _reset_rust_mode():
    _set_rust_mode(False)
    yield
    _set_rust_mode(False)


def _specs_to_mjai(specs):
    return [spec.to_mjai() for spec in specs]


def _compare_enabled_vs_disabled(snapshot: dict, actor: int):
    _set_rust_mode(False)
    expected = _specs_to_mjai(enumerate_legal_action_specs(snapshot, actor))
    _set_rust_mode(True)
    actual = _specs_to_mjai(enumerate_legal_action_specs(snapshot, actor))
    assert actual == expected


def test_rust_legal_actions_matches_python_for_discard_reaction_structure():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    reactor = PlayerState()
    reactor.hand.update(["2m", "4m", "5m", "5mr", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"])

    gs = GameState()
    gs.players = [PlayerState(), reactor, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "3m", "pai_raw": "3m"}
    snap = gs.snapshot(actor=1)

    _compare_enabled_vs_disabled(snap, actor=1)


def test_rust_legal_actions_matches_python_for_own_turn_structural_branch():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["1m", "1m", "1m", "1m", "5mr"])
    player.melds = [
        {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 2},
    ]

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)

    _compare_enabled_vs_disabled(snap, actor=0)


def test_rust_legal_actions_matches_python_for_pending_reach_discards():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["4m", "4s", "5mr", "5sr", "6m", "6p", "6p", "6p", "6s", "7m", "8m", "9m", "C", "C"])
    player.pending_reach = True

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = ["5mr", None, None, None]
    gs.last_tsumo_raw = ["5mr", None, None, None]
    snap = gs.snapshot(actor=0)

    _compare_enabled_vs_disabled(snap, actor=0)


def test_rust_legal_actions_keeps_python_hora_injection_for_chankan(monkeypatch):
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    reactor = PlayerState()
    reactor.hand.update(["1m"] * 13)

    gs = GameState()
    gs.players = [PlayerState(), reactor, PlayerState(), PlayerState()]
    gs.last_kakan = {"actor": 0, "pai": "2m", "pai_raw": "2m", "consumed": ["2m"], "target": 1}
    snap = gs.snapshot(actor=1)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )

    _compare_enabled_vs_disabled(snap, actor=1)


def test_rust_legal_actions_keeps_python_hora_injection_for_discard_ron(monkeypatch):
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    reactor = PlayerState()
    reactor.hand.update(["1m"] * 13)

    gs = GameState()
    gs.players = [PlayerState(), reactor, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "8m", "pai_raw": "8m"}
    snap = gs.snapshot(actor=1)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )

    _compare_enabled_vs_disabled(snap, actor=1)


def test_rust_legal_actions_keeps_python_hora_injection_for_own_turn_tsumo(monkeypatch):
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = ["5mr", None, None, None]
    gs.last_tsumo_raw = ["5mr", None, None, None]
    snap = gs.snapshot(actor=0)

    monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )

    _compare_enabled_vs_disabled(snap, actor=0)


def test_rust_legal_actions_matches_python_for_reached_only_tsumogiri():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])
    player.reached = True

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = ["5mr", None, None, None]
    gs.last_tsumo_raw = ["5mr", None, None, None]
    snap = gs.snapshot(actor=0)

    _compare_enabled_vs_disabled(snap, actor=0)


def test_rust_legal_actions_matches_python_for_reached_ankan_guard():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    player = PlayerState()
    player.hand.update(["1s", "2s", "3s", "3s", "3s", "3s", "1m", "1m", "1m", "2m", "2m", "2m", "4m"])
    player.reached = True

    gs = GameState()
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = ["4m", None, None, None]
    gs.last_tsumo_raw = ["4m", None, None, None]
    snap = gs.snapshot(actor=0)

    _compare_enabled_vs_disabled(snap, actor=0)


def test_rust_public_legal_actions_fall_back_only_for_missing_capability(monkeypatch):
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    reactor = PlayerState()
    reactor.hand.update(["2m", "4m", "5m", "5mr", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"])

    gs = GameState()
    gs.players = [PlayerState(), reactor, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "3m", "pai_raw": "3m"}
    snap = gs.snapshot(actor=1)

    _set_rust_mode(True)
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )

    actual = _specs_to_mjai(enumerate_legal_action_specs(snap, actor=1))

    _set_rust_mode(False)
    expected = _specs_to_mjai(enumerate_legal_action_specs(snap, actor=1))
    assert actual == expected


def test_rust_public_legal_actions_propagate_unexpected_bridge_errors(monkeypatch):
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    reactor = PlayerState()
    reactor.hand.update(["2m", "4m", "5m", "5mr", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"])

    gs = GameState()
    gs.players = [PlayerState(), reactor, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "3m", "pai_raw": "3m"}
    snap = gs.snapshot(actor=1)

    _set_rust_mode(True)
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("public legal bridge drift")),
    )
    monkeypatch.setattr(
        "mahjong_env.legal_actions._enumerate_legal_action_specs_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    with pytest.raises(RuntimeError, match="public legal bridge drift"):
        enumerate_legal_action_specs(snap, actor=1)
