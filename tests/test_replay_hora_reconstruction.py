"""测试重现 legacy preprocess 的 IllegalLabelActionError crash。

真实 crash: event_index=608 actor=2
- label: hora target=0 (荣和 actor 0 丢的 7s)
- legal=[none]
- hand: ['1s','2s','3s','7p','8p','8s','9p','9s','W','W'] (10张)
- last_discard: actor 0 丢 7s
- melds: chi 5s (consumed [4s,6s], target=1)

根本原因分析：
- hand 只有 10 张，不是合法的 13-14 张
- 10 tiles + 1 (pai for ron) = 11 tiles < 14 tiles (minimum for valid hand)
- can_hora_from_snapshot 正确地拒绝了这个不合法的状态
- 这是 **hand reconstruction 的 bug**，不是 legal_actions 的 bug

chi 的 target=1 表示 actor 1 是 discarder，但 last_discard 是 actor 0。
chi 发生在 actor 0 丢 7s 之前。
"""

from collections import Counter

from mahjong_env.legal_actions import enumerate_legal_action_specs
from mahjong_env.scoring import can_hora_from_snapshot
from mahjong_env.state import GameState, PlayerState, apply_event


class TestHoraReconstructionCrash:
    """重现 event_index=608 actor=2 的 crash。

    核心问题：hand 只有 10 张，不是合法的 13-14 张。
    can_hora_from_snapshot 正确地拒绝了，但 crash 表明
    hand reconstruction 在某个更早的事件中丢失了 tiles。
    """

    def test_10_tile_hand_is_rejected_by_can_hora(self):
        """10 张手牌被 can_hora_from_snapshot 正确拒绝。

        10 tiles + 1 ron tile = 11 tiles < 14 tiles (minimum valid hand).
        这不是 legal_actions 的 bug，而是 hand reconstruction 的 bug。
        """
        player2 = PlayerState()
        player2.hand = Counter({"1s": 1, "2s": 1, "3s": 1, "7p": 1, "8p": 1, "8s": 1, "9p": 1, "9s": 1, "W": 2})

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "7s", "pai_raw": "7s"}
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        assert len(snap["hand"]) == 10, f"hand should have 10 tiles, got {len(snap['hand'])}"

        can_hora = can_hora_from_snapshot(
            snap,
            actor=2,
            target=0,
            pai="7s",
            is_tsumo=False,
        )
        # 正确行为：不合法的 hand 应该返回 False
        assert can_hora is False, (
            "10-tile hand is invalid (10+1=11 < 14), can_hora should return False. "
            "The crash is caused by hand RECONSTRUCTION losing tiles, not by legal_actions."
        )

    def test_legal_returns_none_for_10_tile_hand(self):
        """10 张手牌时，legal set 只有 none。"""
        player2 = PlayerState()
        player2.hand = Counter({"1s": 1, "2s": 1, "3s": 1, "7p": 1, "8p": 1, "8s": 1, "9p": 1, "9s": 1, "W": 2})

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "7s", "pai_raw": "7s"}
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        legal = enumerate_legal_action_specs(snap, actor=2)

        # 由于 hand 不合法，hora 不会被加入
        assert len(legal) == 1
        assert legal[0].type == "none"

    def test_14_tile_hand_with_tenpai_allows_hora(self):
        """14 张手牌（正常游戏状态）且听牌时，hora 应该合法。

        一个正确的 tenpai 14-tile hand：
        - 4 groups of 3 = 12 tiles
        - 1 pair = 2 tiles
        - Total = 14 tiles
        - Waits depend on structure

        例如：123s, 456s, 789s, 111p + 22p = 13 tiles (tenpai, waiting for 1p or 2p)
        但这只有 13 tiles...

        实际上，如果已经有 14 tiles，就是完整的手牌(shanten=-1)，不需要等。
        tenpai 状态通常是 13 tiles，等待 1 张。

        让我用一个 13-tile tenpai hand：
        123s, 456s, 789s, 111p = 12 tiles, + 22p = 2 tiles = 14 tiles total
        这个 hand 已经是完整的手牌（shanten=-1），不是 tenpai。

        真正的 tenpai 状态：
        - 13 tiles in hand
        - 等待 1 张来形成 14-tile winning hand

        但在 riichi 中，当有人丢牌时，你是在"反应窗口"中。
        你的手牌应该有 13 tiles（还未摸这个回合的牌）。
        如果你是 tenpai，你等待 1 张。

        但也可以是 14 tiles 的情况：你是立直状态，或者刚摸了牌。
        """
        player2 = PlayerState()
        # 13 tiles: tenpai hand waiting for 9s
        # 123s(complete), 456s(complete), 789s(complete), 11p(pair)
        # 11p + 9p = 111p (triplet) 完成
        # 等待: 1p
        # 这样只有 3*3 + 2 = 11 tiles

        # 让我重新设计：
        # 123s(complete), 456s(complete), 789s(需要7s/6s/9s), 11p(pair), 11p(额外)
        # 这也不对

        # 标准 tenpai hand: 4 groups + pair = 13 tiles waiting
        # 例如：123s, 456s, 789s, 111p, 22p = 14 tiles (complete, not tenpai)
        # 例如：123s, 456s, 78s, 111p, 22p = 13 tiles (tenpai, waiting 9s)
        player2.hand = Counter({
            "1s": 1, "2s": 1, "3s": 1,  # 123s complete
            "4s": 1, "5s": 1, "6s": 1,  # 456s complete
            "7s": 1, "8s": 1,  # 78s 等待 9s 形成 789s
            "1p": 3,  # 111p complete
            "2p": 2,  # 22p complete
        })
        # Count: 3 + 3 + 2 + 3 + 2 = 13 tiles. Waiting for 9s.

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.scores = [25000, 25000, 25000, 25000]
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "9s", "pai_raw": "9s"}  # 9s is the wait
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        assert len(snap["hand"]) == 13, f"hand should have 13 tiles, got {len(snap['hand'])}"

        can_hora = can_hora_from_snapshot(
            snap,
            actor=2,
            target=0,
            pai="9s",
            is_tsumo=False,
        )
        assert can_hora is True, (
            f"13-tile tenpai hand waiting for 9s should allow hora. got {can_hora}"
        )

        legal = enumerate_legal_action_specs(snap, actor=2)
        hora_actions = [s for s in legal if s.type == "hora"]
        assert len(hora_actions) > 0, f"hora should be legal. got {legal}"

    def test_13_tile_hand_without_tenpai_rejects_hora(self):
        """13 张手牌（还未摸牌）且不听牌时，hora 应该不合法。"""
        player2 = PlayerState()
        # 13 tiles: 还没有摸牌，不在听牌状态
        # 这个 hand 不是听牌，所以不能和
        player2.hand = Counter({
            "1s": 1, "2s": 1, "3s": 1,
            "4p": 1, "5p": 1, "6p": 1,
            "7s": 1, "8s": 1, "E": 1, "W": 1, "N": 1, "P": 1, "F": 1,
        })

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.scores = [25000, 25000, 25000, 25000]
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "E", "pai_raw": "E"}  # E is in hand but not tenpai
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        assert len(snap["hand"]) == 13

        can_hora = can_hora_from_snapshot(
            snap,
            actor=2,
            target=0,
            pai="E",
            is_tsumo=False,
        )
        # 13 tiles + 1 ron = 14 tiles but not tenpai, so can_hora should return False
        assert can_hora is False, (
            "13-tile hand not in tenpai should not allow hora. "
            "The hand is not waiting for any tile yet."
        )

    def test_houtei_ron_is_allowed_when_remaining_wall_zero(self):
        """河底荣和需要读取 snapshot.remaining_wall，而不是只看牌型。"""
        player0 = PlayerState()
        player0.hand = Counter({
            "1p": 1, "2p": 1,
            "4s": 2, "5s": 2, "6s": 2, "7s": 2,
        })
        player0.melds = [{
            "type": "chi",
            "pai": "2s",
            "pai_raw": "2s",
            "consumed": ["1s", "3s"],
            "target": 3,
        }]

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 2
        gs.honba = 1
        gs.remaining_wall = 0
        gs.players = [player0, PlayerState(), PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 1, "pai": "3p", "pai_raw": "3p"}
        gs.actor_to_move = 0

        snap = gs.snapshot(actor=0)

        assert can_hora_from_snapshot(
            snap,
            actor=0,
            target=1,
            pai="3p",
            is_tsumo=False,
        ) is True

    def test_haitei_tsumo_is_allowed_when_remaining_wall_zero(self):
        """海底自摸需要读取 snapshot.remaining_wall，而不是只看普通役种。"""
        player2 = PlayerState()
        player2.hand = Counter({
            "4m": 2,
            "5sr": 1,
            "6s": 1,
            "7p": 1,
            "7s": 1,
            "8p": 1,
            "8s": 3,
            "9p": 1,
        })
        player2.melds = [{
            "type": "chi",
            "pai": "8m",
            "pai_raw": "8m",
            "consumed": ["7m", "9m"],
            "target": 1,
        }]

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 0
        gs.remaining_wall = 0
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_tsumo = [None, None, "8s", None]
        gs.last_tsumo_raw = [None, None, "8s", None]
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        assert can_hora_from_snapshot(
            snap,
            actor=2,
            target=2,
            pai="8s",
            is_tsumo=True,
        ) is True


class TestChiAfterDiscardTimeline:
    """测试 chi 和 discard 的时间顺序问题。

    关键问题：chi 的 target=1，但 last_discard 是 actor 0。
    这说明 chi 发生在 actor 0 丢 7s 之前，chi 清除了之前的 last_discard，
    然后 actor 0 才丢了 7s。
    """

    def test_chi_clears_last_discard(self):
        """chi 会清除 last_discard。"""
        player1 = PlayerState()
        player1.hand = Counter({"4s": 1, "5s": 1, "6s": 1})

        gs = GameState()
        gs.players = [PlayerState(), player1, PlayerState(), PlayerState()]
        # 初始有 last_discard
        gs.last_discard = {"actor": 0, "pai": "7s", "pai_raw": "7s"}

        snap_before = gs.snapshot(actor=1)
        assert snap_before["last_discard"] is not None

    def test_actor_2_can_hora_after_actor_1_chi(self):
        """actor 1 chi 后，actor 0 丢 9s，actor 2 应该能荣和 9s。

        时间线：
        1. actor 0 丢 X (last_discard = X)
        2. actor 1 chi Y (last_discard = None)
        3. actor 2 pass (last_discard = None)
        4. actor 3 pass (last_discard = None)
        5. actor 0 丢 9s (last_discard = 9s)
        6. actor 1 chi ... (last_discard = None)
        7. actor 2 荣和 9s ← 这个场景
        """
        # actor 2 的手牌：13 tiles 听牌状态
        # 123s(complete), 456s(complete), 78s + 9s = 789s 等待, 11p(pair)
        # 等待: 9s
        player2 = PlayerState()
        player2.hand = Counter({
            "1s": 1, "2s": 1, "3s": 1,  # 123s complete
            "4s": 1, "5s": 1, "6s": 1,  # 456s complete
            "7s": 1, "8s": 1,  # 78s 等待 9s
            "1p": 3,  # 111p complete
            "2p": 2,  # 22p complete
        })

        # actor 1 的 meld：chi
        player1 = PlayerState()
        player1.melds = [{
            "type": "chi",
            "pai": "5s",
            "pai_raw": "5s",
            "consumed": ["4s", "6s"],
            "target": 1,
        }]

        gs = GameState()
        gs.bakaze = "S"
        gs.kyoku = 3
        gs.honba = 1
        gs.scores = [25000, 25000, 25000, 25000]
        gs.players = [PlayerState(), player1, player2, PlayerState()]
        # last_discard 是 actor 0 的 9s（chi 之后的行为）
        gs.last_discard = {"actor": 0, "pai": "9s", "pai_raw": "9s"}
        gs.actor_to_move = 2

        snap = gs.snapshot(actor=2)

        # 验证 hand 有 13 张
        assert len(snap["hand"]) == 13, f"hand should have 13 tiles, got {len(snap['hand'])}"

        # can_hora 应该返回 True
        can_hora = can_hora_from_snapshot(
            snap,
            actor=2,
            target=0,
            pai="9s",
            is_tsumo=False,
        )
        assert can_hora is True, (
            "actor 2 with 13-tile tenpai hand should be able to hora on 9s. "
            f"got can_hora={can_hora}"
        )

        # legal actions 应该包含 hora
        legal = enumerate_legal_action_specs(snap, actor=2)
        hora_actions = [s for s in legal if s.type == "hora"]
        assert len(hora_actions) > 0, f"hora should be legal. got {legal}"


def test_kakan_accepted_next_tsumo_infers_rinshan_without_explicit_flag():
    player = PlayerState()
    player.hand = Counter({"1p": 1, "6s": 1})
    player.melds = [
        {"type": "pon", "pai": "6s", "pai_raw": "6s", "consumed": ["6s", "6s"], "target": 0},
    ]

    gs = GameState()
    gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
    gs.actor_to_move = 1
    gs.last_kakan = {"actor": 1, "pai": "6s", "pai_raw": "6s", "consumed": ["6s", "6s", "6s"], "target": 0}

    apply_event(
        gs,
        {"type": "kakan_accepted", "actor": 1, "pai": "6s", "pai_raw": "6s", "consumed": ["6s", "6s", "6s"], "target": 0},
    )
    apply_event(gs, {"type": "tsumo", "actor": 1, "pai": "1p"})

    snap = gs.snapshot(actor=1)
    assert snap["rinshan_tsumo"][1] is True


def test_kan_consumes_remaining_wall_and_rinshan_draw_does_not_consume_again():
    gs = GameState()
    apply_event(gs, {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]})
    apply_event(
        gs,
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
                ["2m"] * 13,
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
    )
    assert gs.remaining_wall == 70

    apply_event(
        gs,
        {
            "type": "ankan",
            "actor": 0,
            "consumed": ["1m", "1m", "1m", "1m"],
        },
    )
    assert gs.remaining_wall == 69
    assert gs.pending_rinshan_actor == 0

    apply_event(gs, {"type": "tsumo", "actor": 0, "pai": "9m"})
    assert gs.remaining_wall == 69
    assert gs.players[0].rinshan_tsumo is True
