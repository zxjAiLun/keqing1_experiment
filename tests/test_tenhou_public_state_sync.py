from __future__ import annotations

from mahjong_env.state import GameState, apply_event


def test_public_opponent_chi_can_skip_hidden_hand_updates() -> None:
    state = GameState()
    apply_event(
        state,
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
                ["?"] * 13,
                ["?"] * 13,
                ["?"] * 13,
            ],
        },
    )

    apply_event(
        state,
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "6m",
            "consumed": ["5m", "7m"],
            "skip_hand_update": True,
        },
    )

    assert state.players[1].melds[-1]["type"] == "chi"
    assert state.players[1].melds[-1]["consumed"] == ["5m", "7m"]


def test_public_opponent_kakan_accepted_can_skip_hidden_hand_updates() -> None:
    state = GameState()
    apply_event(
        state,
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
                ["?"] * 13,
                ["?"] * 13,
                ["?"] * 13,
            ],
        },
    )
    state.players[1].melds.append(
        {
            "type": "pon",
            "pai": "6m",
            "pai_raw": "6m",
            "consumed": ["6m", "6m", "6m"],
            "target": 0,
        }
    )

    apply_event(
        state,
        {
            "type": "kakan_accepted",
            "actor": 1,
            "pai": "6m",
            "consumed": ["6m", "6m", "6m", "6m"],
            "skip_hand_update": True,
        },
    )

    assert state.players[1].melds[-1]["type"] == "kakan"


def test_public_opponent_post_meld_discard_can_skip_hidden_hand_updates() -> None:
    state = GameState()
    apply_event(
        state,
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
                ["?"] * 13,
                ["?"] * 13,
                ["?"] * 13,
            ],
        },
    )

    apply_event(
        state,
        {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "pai": "8p",
            "consumed": ["8p", "8p"],
            "skip_hand_update": True,
        },
    )
    apply_event(
        state,
        {
            "type": "dahai",
            "actor": 1,
            "pai": "6m",
            "tsumogiri": False,
            "skip_hand_update": True,
        },
    )

    assert state.players[1].discards[-1]["pai"] == "6m"
