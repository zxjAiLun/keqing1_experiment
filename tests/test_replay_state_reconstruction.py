from mahjong_env.replay_normalizer import normalize_replay_events
from mahjong_env.state import GameState, PlayerState, apply_event


def test_normalize_replay_events_injects_legacy_kakan_accepted():
    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {"type": "kakan", "actor": 1, "pai": "W", "consumed": ["W", "W", "W"]},
        {"type": "dora", "dora_marker": "9s"},
        {"type": "tsumo", "actor": 3, "pai": "4p"},
    ]

    normalized = normalize_replay_events(events)
    assert normalized[1]["type"] == "kakan"
    assert normalized[2]["type"] == "kakan_accepted"
    assert normalized[3]["type"] == "dora"


def test_apply_event_kakan_accepted_upgrades_existing_pon_with_aka_family():
    player = PlayerState()
    player.hand.update(["5pr"])
    player.melds = [
        {"type": "pon", "pai": "5p", "pai_raw": "5p", "consumed": ["5p", "5p"], "target": 0},
    ]

    gs = GameState()
    gs.players = [PlayerState(), PlayerState(), player, PlayerState()]
    gs.last_kakan = {
        "actor": 2,
        "pai": "5pr",
        "pai_raw": "5pr",
        "consumed": ["5p", "5p", "5p"],
        "target": 0,
    }

    apply_event(
        gs,
        {
            "type": "kakan_accepted",
            "actor": 2,
            "pai": "5pr",
            "pai_raw": "5pr",
            "consumed": ["5p", "5p", "5p"],
            "target": 0,
        },
    )

    melds = gs.players[2].melds
    assert len(melds) == 1
    assert melds[0]["type"] == "kakan"
    assert sorted(melds[0]["consumed"]) == ["5p", "5p", "5p", "5pr"]
