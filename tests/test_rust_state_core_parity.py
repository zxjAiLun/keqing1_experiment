from __future__ import annotations

import copy

import pytest

import keqing_core
from mahjong_env.state import GameState, apply_event


def _set_rust_mode(enabled: bool) -> None:
    keqing_core.enable_rust(enabled)


@pytest.fixture(autouse=True)
def _reset_rust_mode():
    _set_rust_mode(False)
    yield
    _set_rust_mode(False)


def _python_snapshot(events: list[dict], actor: int) -> dict:
    state = GameState()
    for event in events:
        apply_event(state, copy.deepcopy(event))
    snapshot = state.snapshot(actor)
    snapshot.pop("feature_tracker", None)
    return snapshot


def _rust_snapshot(events: list[dict], actor: int) -> dict:
    _set_rust_mode(True)
    return keqing_core.replay_state_snapshot(events, actor)


def test_rust_replay_state_snapshot_matches_python_for_basic_turn_cycle():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "9p"},
    ]

    expected = _python_snapshot(events, 1)

    actual = _rust_snapshot(events, 1)

    assert actual == expected


def test_rust_replay_state_snapshot_preserves_last_tsumo_and_last_discard_shape():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
    ]

    expected = _python_snapshot(events, 0)

    actual = _rust_snapshot(events, 0)

    assert actual["last_tsumo"] == expected["last_tsumo"]
    assert actual["last_tsumo_raw"] == expected["last_tsumo_raw"]
    assert actual["hand"] == expected["hand"]
    assert actual["dora_markers"] == expected["dora_markers"]


def test_rust_replay_state_snapshot_matches_python_for_pon_reach_and_hora_terminal():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["2m"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "pon", "actor": 1, "target": 0, "pai": "2m", "consumed": ["2m", "2m"]},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": False},
        {"type": "reach", "actor": 2},
        {"type": "reach_accepted", "actor": 2, "scores": [25000, 25000, 24000, 25000], "kyotaku": 1},
        {"type": "hora", "actor": 3, "target": 2, "scores": [25000, 25000, 16000, 34000], "honba": 0, "kyotaku": 0},
    ]

    expected = _python_snapshot(events, 1)

    actual = _rust_snapshot(events, 1)

    assert actual == expected


def test_rust_replay_state_snapshot_matches_python_for_daiminkan_dora_and_ryukyoku():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 2,
            "honba": 1,
            "kyotaku": 0,
            "oya": 1,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1p",
            "tehais": [
                ["1m"] * 13,
                ["4p", "4p", "4p", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4s", "5s"],
                ["4p"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "dahai", "actor": 2, "pai": "4p", "tsumogiri": False},
        {"type": "daiminkan", "actor": 1, "target": 2, "pai": "4p", "consumed": ["4p", "4p", "4p"]},
        {"type": "dora", "dora_marker": "9s"},
        {"type": "ryukyoku", "scores": [25500, 25500, 24500, 24500], "tenpai_players": [0, 1]},
        {"type": "end_kyoku"},
    ]

    expected = _python_snapshot(events, 1)

    actual = _rust_snapshot(events, 1)

    assert actual == expected


def test_rust_replay_state_snapshot_matches_python_for_ankan_and_rinshan_tsumo():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 3,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "3p",
            "tehais": [
                ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p"],
                ["2p"] * 13,
                ["3p"] * 13,
                ["4p"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "ankan", "actor": 0, "consumed": ["1m", "1m", "1m", "1m"], "pai": "1m"},
        {"type": "tsumo", "actor": 0, "pai": "2s"},
    ]

    expected = _python_snapshot(events, 0)
    actual = _rust_snapshot(events, 0)

    assert actual == expected
    assert actual["pending_rinshan_actor"] is None
    assert actual["rinshan_tsumo"][0] is True


def test_rust_replay_state_snapshot_matches_python_for_kakan_pending_state():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 4,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "6s",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
                ["4m"] * 13,
                ["5m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "pon", "actor": 1, "target": 0, "pai": "2m", "consumed": ["2m", "2m"]},
        {"type": "kakan", "actor": 1, "target": 0, "pai": "2m", "consumed": ["2m"]},
    ]

    expected = _python_snapshot(events, 1)
    actual = _rust_snapshot(events, 1)

    assert actual == expected
    assert actual["last_kakan"] is not None
    assert actual["pending_rinshan_actor"] is None


def test_rust_replay_state_snapshot_matches_python_for_kakan_accepted_and_rinshan_tsumo():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 4,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "6s",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
                ["4m"] * 13,
                ["5m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "pon", "actor": 1, "target": 0, "pai": "2m", "consumed": ["2m", "2m"]},
        {"type": "kakan", "actor": 1, "target": 0, "pai": "2m", "consumed": ["2m"]},
        {"type": "kakan_accepted", "actor": 1, "pai": "2m", "consumed": ["2m"]},
        {"type": "tsumo", "actor": 1, "pai": "7s"},
    ]

    expected = _python_snapshot(events, 1)
    actual = _rust_snapshot(events, 1)

    assert actual == expected
    assert actual["last_kakan"] is None
    assert actual["rinshan_tsumo"][1] is True


def test_rust_replay_state_snapshot_validation_api_reports_match_and_mismatch():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ]
    expected = _python_snapshot(events, 0)
    _set_rust_mode(True)

    matched = keqing_core.validate_replay_state_snapshot(events, 0, expected)

    assert matched["ok"] is True
    assert matched["event_count"] == len(events)
    assert matched["rust_snapshot"] == expected

    expected["scores"] = [1, 2, 3, 4]
    mismatched = keqing_core.validate_replay_state_snapshot(events, 0, expected)

    assert mismatched["ok"] is False
    assert mismatched["mismatch"] is True
    assert mismatched["rust_snapshot"]["scores"] == [25000, 25000, 25000, 25000]


def test_rust_replay_state_snapshot_matches_python_for_terminal_runtime_state_fields():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 2,
            "kyotaku": 1,
            "oya": 0,
            "scores": [24000, 26000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {
            "type": "hora",
            "actor": 1,
            "target": 0,
            "pai": "2m",
            "scores": [20000, 30000, 25000, 25000],
            "honba": 2,
            "state_honba": 0,
            "kyotaku": 1,
            "state_kyotaku": 0,
            "oya": 1,
        },
        {"type": "end_kyoku"},
    ]

    expected = _python_snapshot(events, 1)
    actual = _rust_snapshot(events, 1)

    assert actual == expected
    assert actual["honba"] == 0
    assert actual["kyotaku"] == 0
    assert actual["oya"] == 1
    assert actual["actor_to_move"] is None
