from mahjong_env.legal_actions import enumerate_legal_actions, enumerate_legal_action_specs
from mahjong_env.state import GameState, PlayerState


def test_enumerate_legal_action_specs_projects_canonical_specs():
    player0 = PlayerState()
    player0.hand.update(["5mr", "5m", "5m"])

    gs = GameState()
    gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "5m", "pai_raw": "5m"}
    snap = gs.snapshot(actor=0)

    specs = enumerate_legal_action_specs(snap, actor=0)

    assert any(
        spec.type == "pon"
        and spec.actor == 0
        and spec.target == 1
        and spec.pai == "5m"
        and tuple(sorted(spec.consumed)) == ("5m", "5mr")
        for spec in specs
    )


def test_enumerate_legal_actions_allows_ankan_with_mixed_aka_family():
    player3 = PlayerState()
    player3.hand.update(["2m", "3m", "4m", "5m", "5m", "5m", "5mr", "5p", "6p", "7p", "7p", "7s", "7s", "8p"])

    gs = GameState()
    gs.players = [PlayerState(), PlayerState(), PlayerState(), player3]
    gs.actor_to_move = 3
    snap = gs.snapshot(actor=3)

    legal = enumerate_legal_actions(snap, actor=3)

    ankan_actions = [a for a in legal if a.type == "ankan"]
    assert ankan_actions
    assert any(tuple(sorted(a.consumed or [])) == ("5m", "5m", "5m", "5mr") for a in ankan_actions)


def test_reach_blocked_after_pon():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p", "5sr", "5sr"])
    player.melds = [{"type": "pon", "pai": "1p", "pai_raw": "1p", "consumed": ["1p", "1p", "1p"], "target": 2}]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=0)
    assert not any(a.type == "reach" for a in legal)


def test_reach_blocked_after_chi():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5sr", "5sr"])
    player.melds = [{"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3}]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=0)
    assert not any(a.type == "reach" for a in legal)


def test_reach_blocked_after_daiminkan():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "5sr", "5sr", "5sr"])
    player.melds = [{"type": "daiminkan", "pai": "1p", "pai_raw": "1p", "consumed": ["1p", "1p", "1p"], "target": 2}]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=0)
    assert not any(a.type == "reach" for a in legal)


def test_reach_allowed_with_only_ankan():
    player = PlayerState()
    player.hand.update(["3m", "3m", "3m", "3p", "3p", "6p", "6p", "6s", "7s", "8p", "8s"])
    player.melds = [
        {"type": "ankan", "pai": "1p", "pai_raw": "1p", "consumed": ["1p", "1p", "1p", "1p"], "target": 0},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=0)
    assert any(a.type == "reach" for a in legal)


def test_reach_allowed_on_complete_hand_if_some_discard_keeps_tenpai():
    player = PlayerState()
    player.hand.update(["4m", "4s", "5mr", "5sr", "6m", "6p", "6p", "6p", "6s", "7m", "8m", "9m", "C", "C"])

    gs = GameState()
    gs.bakaze = "S"
    gs.kyoku = 4
    gs.honba = 0
    gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
    gs.actor_to_move = 1
    gs.last_tsumo = [None, "5mr", None, None]
    gs.last_tsumo_raw = [None, "5mr", None, None]
    snap = gs.snapshot(actor=1)

    legal = enumerate_legal_actions(snap, actor=1)

    assert any(a.type == "hora" and a.target == 1 and a.pai == "5mr" for a in legal)
    assert any(a.type == "reach" and a.actor == 1 for a in legal)


def test_reach_blocked_when_already_reached():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5sr", "5sr"])
    player.reached = True

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    gs.last_tsumo = ["5sr", None, None, None]
    gs.last_tsumo_raw = ["5sr", None, None, None]
    snap = gs.snapshot(actor=0)
    snap["shanten"] = 0

    legal = enumerate_legal_actions(snap, actor=0)
    assert not any(a.type == "reach" for a in legal)


def test_chi_only_next_player():
    player0 = PlayerState()
    player0.hand.update(["2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p"])
    player2 = PlayerState()
    player2.hand.update(["2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p"])

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player0, PlayerState(), player2, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3m", "pai_raw": "3m"}
    snap0 = gs.snapshot(actor=0)
    snap2 = gs.snapshot(actor=2)

    legal0 = enumerate_legal_actions(snap0, actor=0)
    legal2 = enumerate_legal_actions(snap2, actor=2)

    assert not any(a.type == "chi" for a in legal0)
    assert any(a.type == "chi" for a in legal2)


def test_pon_blocked_when_reached():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "5sr", "5sr"])
    player.reached = True

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "1p", "pai_raw": "1p"}
    snap = gs.snapshot(actor=1)

    legal = enumerate_legal_actions(snap, actor=1)
    assert not any(a.type == "pon" for a in legal)


def test_chi_blocked_when_reached():
    player = PlayerState()
    player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "5sr", "5sr"])
    player.reached = True

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), PlayerState(), player, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3m", "pai_raw": "3m"}
    snap = gs.snapshot(actor=2)

    legal = enumerate_legal_actions(snap, actor=2)
    assert not any(a.type == "chi" for a in legal)


def test_daiminkan_blocked_when_reached():
    player = PlayerState()
    player.hand.update(["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "5sr", "5sr"])
    player.reached = True

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), PlayerState(), player, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "1m", "pai_raw": "1m"}
    snap = gs.snapshot(actor=2)

    legal = enumerate_legal_actions(snap, actor=2)
    assert not any(a.type == "daiminkan" for a in legal)


def test_ankan_legal_on_own_turn():
    player = PlayerState()
    player.hand.update(["1m", "1m", "1m", "1m", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "5sr", "5sr"])

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)

    legal = enumerate_legal_actions(snap, actor=0)
    assert any(a.type == "ankan" for a in legal)


def test_kakan_legal_on_own_turn():
    player = PlayerState()
    player.hand.update(["5mr"])
    player.melds = [
        {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)

    legal = enumerate_legal_actions(snap, actor=0)
    assert any(a.type == "kakan" for a in legal)


def test_kakan_pai_uses_aka_when_available():
    player = PlayerState()
    player.hand.update(["5mr"])
    player.melds = [
        {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    gs.actor_to_move = 0
    snap = gs.snapshot(actor=0)

    legal = enumerate_legal_actions(snap, actor=0)
    kakan_actions = [a for a in legal if a.type == "kakan"]
    assert len(kakan_actions) == 1
    assert kakan_actions[0].pai == "5mr"
