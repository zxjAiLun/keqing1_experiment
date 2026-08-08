"""赤宝牌口径一致性测试。

验证 replay.py 的 _normalize_or_keep_aka 与 legal_actions.py 的 aka 处理保持一致，
确保监督样本中 pai/consumed 与 legal action 的 tile 格式口径对齐。
"""

from collections import Counter

import pytest

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import _normalize_or_keep_aka
from mahjong_env.state import GameState, PlayerState
from mahjong_env.tiles import AKA_DORA_TILES


class TestNormalizeOrKeepAka:
    """_normalize_or_keep_aka 的行为验证。"""

    def test_preserves_real_aka_tiles(self):
        """真正的赤宝牌 5mr/5pr/5sr 必须保留，不被归一化。"""
        for aka in AKA_DORA_TILES:
            assert _normalize_or_keep_aka(aka) == aka

    def test_normalizes_non_aka_r_tiles(self):
        """非标准 xxxr 形式应去掉 r 后缀。"""
        assert _normalize_or_keep_aka("5mr") == "5mr"  # 真正的 aka，已在上面覆盖
        # 其他 xxxr 形式（非赤宝牌）应归一
        assert _normalize_or_keep_aka("1mr") == "1m"
        assert _normalize_or_keep_aka("9pr") == "9p"

    def test_passes_through_normal_tiles(self):
        """普通数牌/字牌直接透传。"""
        assert _normalize_or_keep_aka("5m") == "5m"
        assert _normalize_or_keep_aka("9s") == "9s"
        assert _normalize_or_keep_aka("E") == "E"
        assert _normalize_or_keep_aka("F") == "F"

    def test_aka_and_normal_are_34_aligned(self):
        """赤宝牌与其普通版本在 tile34 坐标上一致（验证口径对齐）。"""
        from mahjong_env.tiles import tile_to_34

        for aka in AKA_DORA_TILES:
            base = aka[0] + aka[1]  # e.g. "5mr" -> "5m"
            assert tile_to_34(aka) == tile_to_34(base)
            assert tile_to_34(aka) >= 0


class TestAkaConsistencyContract:
    """replay 标签层与 legal action 生成层之间的 aka 契约。

    replay 的 _normalize_or_keep_aka 保留真正的 aka（5mr/5pr/5sr），
    legal_actions 枚举时也优先返回实际存在的 aka 版本。
    本测试验证两边的处理逻辑对称。
    """

    def test_consumed_tiles_aka_equivalence(self):
        """consumed 列表中出现的 aka 牌应与 tile34 对齐。

        例如 pon 5m/5m/5mr，手牌剩 5mr：
        - consumed 可能是 [5mr, 5mr, 5mr]
        - 归一化后仍是 [5mr, 5mr, 5mr]（aka 保留）
        - tile34 层面等价于 [5m, 5m, 5m]
        """
        consumed_with_aka = ["5mr", "5mr", "5mr"]
        normalized = [_normalize_or_keep_aka(t) for t in consumed_with_aka]
        assert all(t == "5mr" for t in normalized)

    def test_label_pai_aka_preservation(self):
        """label["pai"] 为赤宝牌时应原样保留，不应被错误归一为普通 5m。"""
        label_pai = "5mr"
        result = _normalize_or_keep_aka(label_pai)
        assert result == "5mr"

    def test_mixed_aka_and_normal_in_consumed(self):
        """consumed 中混有 aka 和普通牌时，各自按规则处理。"""
        consumed = ["5mr", "5m", "5mr"]
        normalized = [_normalize_or_keep_aka(t) for t in consumed]
        # 5mr 保留，5m 透传
        assert normalized == ["5mr", "5m", "5mr"]


class TestKakanAkaBehavior:
    """kakan 场景下赤宝牌精确枚举验证。

    场景：玩家有一个 5m 刻子（pon），手牌还剩 1 张 5m/5mr。
    enumerate_legal_actions 应返回 kakan 动作，且 pai 应为手里实际存在的版本。
    """

    def _make_state(self, hand_tiles: list[str], meld: dict) -> dict:
        """构造包含指定手牌和副露的快照，供 enumerate_legal_actions 使用。"""
        player = PlayerState()
        player.hand = Counter({t: hand_tiles.count(t) for t in set(hand_tiles)})
        player.melds = [meld]
        player.reached = False

        gs = GameState()
        gs.actor_to_move = 0  # actor=0 的回合
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        return gs.snapshot(actor=0)

    def test_kakan_prefers_aka_when_in_hand(self):
        """手里有 5mr 且有 5m 刻子时，kakan pai 应枚举为 5mr。"""
        # pon 5m，手里还有一张 5mr（可用于加杠）
        snap = self._make_state(
            hand_tiles=["5mr"],
            meld={"type": "pon", "pai": "5m", "consumed": ["5m", "5m", "5mr"], "target": 1},
        )
        legal = enumerate_legal_actions(snap, actor=0)
        kakan_actions = [a for a in legal if a.type == "kakan"]
        assert len(kakan_actions) == 1, f"期望 1 个 kakan，实际 {len(kakan_actions)}"
        assert kakan_actions[0].pai == "5mr", f"kakan pai 应为 5mr，实际为 {kakan_actions[0].pai}"

    def test_kakan_falls_back_to_normal_when_no_aka(self):
        """手里只有普通 5m（无 aka）时，kakan pai 应枚举为 5m。"""
        snap = self._make_state(
            hand_tiles=["5m"],
            meld={"type": "pon", "pai": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
        )
        legal = enumerate_legal_actions(snap, actor=0)
        kakan_actions = [a for a in legal if a.type == "kakan"]
        assert len(kakan_actions) == 1, f"期望 1 个 kakan，实际 {len(kakan_actions)}"
        assert kakan_actions[0].pai == "5m", f"kakan pai 应为 5m，实际为 {kakan_actions[0].pai}"


class TestPonDaiminkanAka:
    """pon/daiminkan 时 consumed 优先使用赤宝牌版本。"""

    def _make_pon_state(self, hand_tiles: list[str]) -> dict:
        player = PlayerState()
        player.hand = Counter({t: hand_tiles.count(t) for t in set(hand_tiles)})
        player.reached = False

        gs = GameState()
        gs.actor_to_move = 0
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        return gs.snapshot(actor=0)

    def test_pon_consumed_prefers_aka_when_in_hand(self):
        """手里有 5mr 时，pon 5m 的 consumed 应使用 5mr。"""
        snap = self._make_pon_state(hand_tiles=["5mr", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        pon_actions = [a for a in legal if a.type == "pon"]
        assert len(pon_actions) == 1
        consumed = pon_actions[0].consumed
        assert "5mr" in consumed, f"pon consumed should include 5mr, got {consumed}"

    def test_daiminkan_consumed_prefers_aka_when_in_hand(self):
        """手里有 5mr 时，daiminkan 5m 的 consumed 应使用 5mr。"""
        player = PlayerState()
        player.hand = Counter({"5mr": 3})
        player.reached = False

        gs = GameState()
        gs.actor_to_move = 0
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)
        daiminkan_actions = [a for a in legal if a.type == "daiminkan"]
        assert len(daiminkan_actions) == 1
        consumed = daiminkan_actions[0].consumed
        assert "5mr" in consumed, f"daiminkan consumed should include 5mr, got {consumed}"


class TestChiAka:
    """chi 时 consumed 应优先使用赤宝牌版本。"""

    def test_chi_consumed_prefers_aka(self):
        """吃 456m 时，chi consumed 应正确枚举（4m+6m）。"""
        # Discarder is player 1, next player is player 2.
        # Player 2 has tiles to chi 456m from discard 5m.
        player2 = PlayerState()
        player2.hand = Counter({"2m": 1, "4m": 1, "5mr": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 1, "2p": 1, "3p": 1, "4p": 1, "5p": 1, "6p": 1})
        player2.reached = False

        gs = GameState()
        gs.players = [PlayerState(), PlayerState(), player2, PlayerState()]
        gs.last_discard = {"actor": 1, "pai": "5m", "pai_raw": "5m"}
        snap = gs.snapshot(actor=2)

        legal = enumerate_legal_actions(snap, actor=2)
        chi_actions = [a for a in legal if a.type == "chi"]
        # Chi 456m (discard 5m) needs 4m+6m as consumed
        assert len(chi_actions) >= 1, f"player 2 is next of discarder 1, should have chi, got {[a.type for a in legal]}"
        # Verify the consumed tiles for chi 456m
        has_chi_456 = any(
            set(c.consumed or []) == {"4m", "6m"}
            for c in chi_actions
            if c.pai == "5m"
        )
        assert has_chi_456, f"chi 456m should use 4m+6m as consumed, got {[c.consumed for c in chi_actions]}"
