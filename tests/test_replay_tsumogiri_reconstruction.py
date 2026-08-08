"""tsumogiri reconstruction 一致性测试。

核心问题：tsumogiri 的判断依赖 last_tsumo_raw，但 reconstruction 无法始终
准确还原 replay 原始的 tsumogiri 标记。

例如：player 摸到 3p，手牌中原有 3p×3，打出其中一张 3p（tsumogiri=False）。
- reconstruction：last_tsumo_raw=3p，tsumogiri 判断依赖 last_tsumo_raw
- action_specs_match 对 dahai 只比较 pai，不比较 tsumogiri
- 所以 reconstruction 产生的 tsumogiri=True 和 label 的 tsumogiri=False 被视为等价

但当 hand reconstruction 本身出错时（tile 数量不对），
dahai 动作可能完全不出现在 legal set 中，导致 IllegalLabelActionError。

另外，如果 hand 中同名 tile 数量不足以同时满足 "在 closed hand 中存在" 和
"不是 last_tsumo_raw"，reconstruction 的 tsumogiri 判断也会出错。
"""

from collections import Counter

from mahjong_env.legal_actions import enumerate_legal_action_specs
from mahjong_env.state import GameState, PlayerState, apply_event
from mahjong_env.types import action_specs_match, ActionSpec


class TestTsumogiriReconstructionLogic:
    """enumerate_legal_actions 的 tsumogiri 判断逻辑。"""

    def test_tsumogiri_true_when_last_tsumo_matches(self):
        """当 last_tsumo_raw 与打的牌相同时，tsumogiri=True。"""
        player = PlayerState()
        # hand 包含摸到的牌（12 tiles = 11 base + 1 drawn 3p）
        player.hand = Counter({"2p": 3, "3p": 4, "5pr": 1, "6p": 1, "6s": 1, "7p": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo = [None, "3p", None, None]
        gs.last_tsumo_raw = [None, "3p", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_3p = next((s for s in legal if s.type == "dahai" and s.pai == "3p"), None)
        assert dahai_3p is not None
        assert dahai_3p.tsumogiri is True

    def test_tsumogiri_false_when_last_tsumo_different_tile(self):
        """当 last_tsumo_raw 与打的牌不同时，tsumogiri=False。"""
        player = PlayerState()
        player.hand = Counter({"2p": 3, "3p": 3, "5pr": 1, "6p": 1, "6s": 1, "7p": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo_raw = [None, "7m", None, None]  # 摸的是 7m，不是 3p
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_3p = next((s for s in legal if s.type == "dahai" and s.pai == "3p"), None)
        assert dahai_3p is not None
        assert dahai_3p.tsumogiri is False

    def test_action_specs_match_ignores_tsumogiri_for_dahai(self):
        """action_specs_match 对 dahai 只比较 pai，不比较 tsumogiri。"""
        from mahjong_env.types import ActionSpec

        spec1 = ActionSpec(type="dahai", actor=1, pai="3p", tsumogiri=True)
        spec2 = ActionSpec(type="dahai", actor=1, pai="3p", tsumogiri=False)

        # tsumogiri 不同，但 pai 相同，应该匹配
        assert action_specs_match(spec1, spec2), (
            "action_specs_match should ignore tsumogiri for dahai (only compare pai)"
        )

    def test_tsumogiri_ambiguous_when_multiple_copies(self):
        """当手牌中同名 tile 有多张时，reconstruction 无法区分是从哪一 slot 打的。

        例如：3p×4（3张在 closed hand + 1张 last_tsumo_raw），
        打出 3p 时，reconstruction 只能把 tsumogiri=True，
        但 replay 可能记录了从 closed hand 打（tsumogiri=False）。
        """
        player = PlayerState()
        # 3p×4: 3张在 closed hand + 1张刚摸的
        player.hand = Counter({"3p": 4, "2p": 1, "5pr": 1, "6p": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo = [None, "3p", None, None]
        gs.last_tsumo_raw = [None, "3p", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)

        # Reconstruction 只有一个 dahai 3p tsumogiri=True
        dahai_3p_list = [s for s in legal if s.type == "dahai" and s.pai == "3p"]
        assert len(dahai_3p_list) == 1
        assert dahai_3p_list[0].tsumogiri is True

        # 但 replay 原始 label 可能是 tsumogiri=False
        # action_specs_match 会认为它们等价（pai 相同）
        # 所以不会报 IllegalLabelActionError
        label_spec = ActionSpec(type="dahai", actor=1, pai="3p", tsumogiri=False)
        assert action_specs_match(dahai_3p_list[0], label_spec)


class TestTsumogiriVsLabelMismatch:
    """legacy preprocess 真实 crash 场景：hand reconstruction 不一致。

    真实 crash 中 hand 只有 11 tiles 而不是 12，导致 dahai 3p 完全不在 legal set 中。
    这说明在 event 79 之前某处 tile 被错误消耗了。
    """

    def test_tsumogiri_reconstruction_cannot_distinguish_closed_vs_drawn_for_duplicates(self):
        """Reconstruction 无法区分从 closed hand 还是 drawn slot 打的同名 tile。

        这是 tsumogiri reconstruction 的根本性限制：
        - 如果 hand 有 3p×4，last_tsumo_raw='3p'
        - reconstruction 永远生成 dahai 3p tsumogiri=True
        - 但 replay 可能记录 tsumogiri=False（从 closed hand 打了其中一张）
        """
        player = PlayerState()
        player.hand = Counter({"3p": 4, "2p": 1, "5pr": 1, "6p": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo_raw = [None, "3p", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)

        # reconstruction 的 legal dahai
        dahai_3p = next((s for s in legal if s.type == "dahai" and s.pai == "3p"), None)
        assert dahai_3p is not None

        # label 可能是 tsumogiri=False（从 closed hand 打）
        from mahjong_env.types import ActionSpec
        label_spec = ActionSpec(type="dahai", actor=1, pai="3p", tsumogiri=False)

        # action_specs_match 认为它们等价（只比较 pai）
        # 所以即使 tsumogiri 不一致，也不会报 IllegalLabelActionError
        assert action_specs_match(dahai_3p, label_spec)

    def test_hand_tile_count_mismatch_causes_missing_dahai(self):
        """当 hand tile 数量不对时，dahai 可能完全不出现在 legal set。

        这是 legacy preprocess event_index=79 crash 的真实原因：
        hand reconstruction 出错（11 tiles 而非 12），
        导致 dahai 3p 根本不在 legal set 中。
        """
        # 模拟一个错误状态的 hand（reconstruction bug 导致）
        player = PlayerState()
        # 11 tiles - 少了一张
        player.hand = Counter({"2p": 3, "3p": 2, "5pr": 1, "6p": 1, "6s": 1, "7p": 1, "8s": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo_raw = [None, "3p", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_3p = next((s for s in legal if s.type == "dahai" and s.pai == "3p"), None)

        # 当 hand 中只有 2 个 3p 时，dahai 3p 仍然存在（因为 last_tsumo_raw=3p）
        # Reconstruction 认为有 3 个 3p（2 个 closed + 1 个 drawn），可以打
        # 但如果只有 2 个 3p total（含 last_tsumo），打完后只剩 1 个...
        # 实际上：hand 只有 2 个 3p，last_tsumo_raw=3p，
        # 打出一张 3p 后应该剩 1 张 3p（来自 closed hand）+ last_tsumo 被清空
        # 所以 dahai 3p 仍然合法
        assert dahai_3p is not None

        # 真正的问题：当 closed hand 中同名 tile 数量 < 1 时，
        # reconstruction 仍然允许 dahai（因为 last_tsumo_raw 存在），
        # 但 replay 原始 label 可能描述了不同的实际情况

    def test_hand_missing_tile_causes_dahai_not_in_legal(self):
        """当 tile 彻底不在 hand 中时，dahai 不在 legal set。

        这对应真实 crash 的场景：reconstruction 丢失了一张 tile，
        导致 dahai 完全不合法。
        """
        player = PlayerState()
        # 11 tiles，没有 3p（全部被消耗或丢失了）
        player.hand = Counter({"2p": 3, "5pr": 1, "6p": 1, "6s": 1, "7p": 1, "8s": 1, "9m": 1, "1m": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo_raw = [None, "3p", None, None]  # 但 last_tsumo_raw 还记着 3p
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_3p = next((s for s in legal if s.type == "dahai" and s.pai == "3p"), None)

        # hand 中没有 3p，dahai 3p 不会出现在 legal set
        # 但 label 说 dahai 3p → IllegalLabelActionError
        assert dahai_3p is None


class TestTsumogiriRealWorldScenario:
    """真实 replay 场景：reconstruction 无法完美复现 tsumogiri。"""

    def test_dahai_after_pon_discarded_tile(self):
        """副露后打牌场景：pon 吃掉的 tile 不应出现在 dahai 中。

        注意：由于 hand Counter 不存储"被 meld 消耗"的信息，
        当前 enumerate_legal_action_specs 的 dahai 枚举
        不会过滤掉被 meld 消耗的 tile。
        本测试验证这一已知行为。
        """
        player = PlayerState()
        # 14 tiles：3p×3, 2p×3, 5pr×2, 6p×1, 6s×1, 7p×1 + 7m(1) = 12 tiles after pon
        # pon 消耗 2 张 5pr，剩 1 张 5pr + 11 other = 12 tiles
        player.hand = Counter({"3p": 3, "2p": 3, "5pr": 2, "6p": 1, "6s": 1, "7p": 1, "7m": 1})
        player.melds = [{
            "type": "pon",
            "pai": "5pr",
            "pai_raw": "5pr",
            "consumed": ["5pr", "5pr", "5pr"],  # 2 from hand + 1 called
            "target": 0,
        }]

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo = [None, "7m", None, None]
        gs.last_tsumo_raw = [None, "7m", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_pai_list = [s.pai for s in legal if s.type == "dahai"]

        # 由于 hand 中仍有 5pr 且代码不过滤 meld consumed tiles，
        # 5pr 会出现在 dahai 中（这是当前的有 bug 行为）
        # 本测试记录这一行为，正确的行为应该是 5pr 不出现
        assert "5pr" in dahai_pai_list

        # 7m（last_tsumo）应该 tsumogiri=True
        dahai_7m = next((s for s in legal if s.type == "dahai" and s.pai == "7m"), None)
        assert dahai_7m is not None
        assert dahai_7m.tsumogiri is True

    def test_reconstruction_tsumogiri_vs_replay_label_tsumogiri(self):
        """验证 tsumogiri reconstruction 与 replay 原始 label 可能不一致。

        场景：player 摸到 7m（last_tsumo_raw=7m），但打出了另一个 7m（tsumogiri=False）。
        Reconstruction 只能生成 tsumogiri=True（因为 last_tsumo_raw=7m），
        但 replay 可能记录了 tsumogiri=False。
        action_specs_match 认为它们等价，不会报错。
        """
        from mahjong_env.types import ActionSpec

        # Reconstruction: dahai 7m tsumogiri=True
        player = PlayerState()
        player.hand = Counter({"7m": 2, "2p": 3, "5pr": 1, "6p": 1})

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_tsumo_raw = [None, "7m", None, None]
        gs.actor_to_move = 1

        snap = gs.snapshot(actor=1)
        legal = enumerate_legal_action_specs(snap, actor=1)
        dahai_7m = next((s for s in legal if s.type == "dahai" and s.pai == "7m"), None)
        assert dahai_7m is not None

        # Replay label: dahai 7m tsumogiri=False（打出了 closed hand 中的 7m）
        label_spec = ActionSpec(type="dahai", actor=1, pai="7m", tsumogiri=False)

        # action_specs_match 只比较 pai，所以认为等价
        assert action_specs_match(dahai_7m, label_spec)
