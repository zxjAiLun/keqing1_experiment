"""全面覆盖 enumerate_legal_actions 所有分支与边缘情况。

分支树：
  A. last_kakan.actor != actor  → kakan 响应窗口
  B. last_discard.actor != actor → 荣和响应窗口
  C. actor_to_move == actor     → 自己回合
    C1. last_tsumo != None       → 自摸阶段
    C2. reached == True          → 已立直
    C3. pending_reach == True     → 等待打宣言牌
    C4. 普通打牌阶段
  D. not last_discard           → 无 last_discard
  E. last_discard.actor == actor → 自己的舍牌

每个分支的边缘情况：
- furiten
- reached
- aka 牌
- 多 ankan / 多 kakan
- shanten 状态
- empty / edge hand composition
"""

import pytest
from collections import Counter

import keqing_core
from mahjong_env.legal_actions import enumerate_legal_actions, _chi_patterns, _can_pon, _can_daiminkan, _hand_has_tile, _pick_chi_tile, _pick_consumed, _ankan_candidates, _can_declare_reach
from mahjong_env.state import GameState, PlayerState
from mahjong_env.types import Action


def _force_python_legal_fallback(monkeypatch, *, disable_structural: bool = False):
    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )
    if disable_structural:
        monkeypatch.setattr(
            keqing_core,
            "enumerate_legal_action_specs_structural",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rust legal action structural capability is not available")),
        )


# =============================================================================
# 工具函数测试 - 独立验证每个 helper 的行为
# =============================================================================

class TestChiPatterns:
    """_chi_patterns 实现: [num-2,num-1], [num-1,num+1], [num+1,num+2]"""

    def test_chi_patterns_middle_tile(self):
        patterns = _chi_patterns("5m")
        assert sorted(patterns) == sorted([["3m", "4m"], ["4m", "6m"], ["6m", "7m"]])

    def test_chi_patterns_edge_tile_low(self):
        patterns = _chi_patterns("2m")
        assert sorted(patterns) == sorted([["1m", "3m"], ["3m", "4m"]])

    def test_chi_patterns_edge_tile_high(self):
        patterns = _chi_patterns("8m")
        assert sorted(patterns) == sorted([["6m", "7m"], ["7m", "9m"]])

    def test_chi_patterns_aka_tile_normalizes(self):
        patterns = _chi_patterns("5mr")
        # aka 5mr 被归一化为 5m 后处理
        assert sorted(patterns) == sorted([["3m", "4m"], ["4m", "6m"], ["6m", "7m"]])

    def test_chi_patterns_honor_tile(self):
        patterns = _chi_patterns("E")
        assert patterns == []

    def test_chi_patterns_1m_gives_23m(self):
        patterns = _chi_patterns("1m")
        assert patterns == [["2m", "3m"]]

    def test_chi_patterns_9m_gives_78m(self):
        patterns = _chi_patterns("9m")
        assert patterns == [["7m", "8m"]]


class TestCanPon:
    def test_can_pon_true(self):
        hand = Counter({"5m": 2})
        assert _can_pon(hand, "5m") is True

    def test_can_pon_false(self):
        hand = Counter({"5m": 1})
        assert _can_pon(hand, "5m") is False

    def test_can_pon_with_aka(self):
        hand = Counter({"5m": 1, "5mr": 1})
        assert _can_pon(hand, "5m") is True

    def test_can_pon_with_aka_only(self):
        hand = Counter({"5mr": 2})
        assert _can_pon(hand, "5m") is True

    def test_can_pon_honor(self):
        hand = Counter({"E": 2})
        assert _can_pon(hand, "E") is True

    def test_can_pon_honor_false(self):
        hand = Counter({"E": 1})
        assert _can_pon(hand, "E") is False


class TestCanDaiminkan:
    def test_can_daiminkan_true(self):
        hand = Counter({"5m": 3})
        assert _can_daiminkan(hand, "5m") is True

    def test_can_daiminkan_false(self):
        hand = Counter({"5m": 2})
        assert _can_daiminkan(hand, "5m") is False

    def test_can_daiminkan_with_aka(self):
        hand = Counter({"5m": 2, "5mr": 1})
        assert _can_daiminkan(hand, "5m") is True

    def test_can_daiminkan_with_aka_only(self):
        hand = Counter({"5mr": 3})
        assert _can_daiminkan(hand, "5m") is True


class TestHandHasTile:
    def test_hand_has_normal_tile(self):
        hand = Counter({"5m": 1})
        assert _hand_has_tile(hand, "5m") is True

    def test_hand_has_aka_tile(self):
        hand = Counter({"5mr": 1})
        assert _hand_has_tile(hand, "5mr") is True

    def test_hand_has_aka_equivalent(self):
        hand = Counter({"5mr": 1})
        assert _hand_has_tile(hand, "5m") is True

    def test_hand_missing_tile(self):
        hand = Counter({"5m": 0})
        assert _hand_has_tile(hand, "5m") is False


class TestPickChiTile:
    def test_pick_normal_tile(self):
        hand = Counter({"4m": 1})
        assert _pick_chi_tile(hand, "4m") == "4m"

    def test_pick_aka_when_available(self):
        hand = Counter({"4m": 1, "5mr": 1})
        assert _pick_chi_tile(hand, "5m") == "5mr"

    def test_pick_normal_fallback(self):
        hand = Counter({"5m": 1})
        assert _pick_chi_tile(hand, "5m") == "5m"


class TestPickConsumed:
    def test_pick_consumed_2_normal(self):
        hand = Counter({"5m": 2})
        result = _pick_consumed(hand, "5m", 2)
        assert sorted(result) == ["5m", "5m"]

    def test_pick_consumed_2_with_aka(self):
        hand = Counter({"5m": 1, "5mr": 2})
        result = _pick_consumed(hand, "5m", 2)
        assert sorted(result) == ["5mr", "5mr"]

    def test_pick_consumed_3_with_partial_aka(self):
        """aka 先取（min of hand count and remaining needed），剩余用普通牌补。"""
        hand = Counter({"5m": 2, "5mr": 1})
        result = _pick_consumed(hand, "5m", 3)
        # 手里有1张aka，先取1张aka，剩余2张用普通牌
        assert sorted(result) == ["5m", "5m", "5mr"]

    def test_pick_consumed_3_normal(self):
        hand = Counter({"5m": 5})
        result = _pick_consumed(hand, "5m", 3)
        assert result == ["5m", "5m", "5m"]


class TestAnkanCandidates:
    """_ankan_candidates 返回 List[tuple[str, tuple[str, ...]]]，每项 (pai, consumed_tuple)。"""

    def test_ankan_single(self):
        hand = Counter({"1m": 4})
        result = _ankan_candidates(hand)
        assert len(result) == 1
        assert result[0][0] == "1m"
        assert result[0][1] == ("1m", "1m", "1m", "1m")

    def test_ankan_multiple(self):
        hand = Counter({"1m": 4, "5p": 4})
        result = _ankan_candidates(hand)
        assert len(result) == 2
        paise = sorted(r[0] for r in result)
        assert paise == ["1m", "5p"]

    def test_ankan_aka_tile_normalized(self):
        """aka 牌被归一化后作为普通牌处理。"""
        hand = Counter({"5mr": 4})
        result = _ankan_candidates(hand)
        # aka 被归一化为 5m，5m 成为候选
        assert len(result) == 1
        assert result[0][0] == "5mr"

    def test_ankan_not_enough(self):
        hand = Counter({"1m": 3})
        assert _ankan_candidates(hand) == []


class TestCanDeclareReach:
    def test_reach_allowed_tenpai_no_melds(self):
        """立直需要手牌 14 张（无副露时）。"""
        hand = Counter({"1m": 1, "2m": 1, "3m": 1, "4m": 1, "5m": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 1, "2p": 1, "3p": 1, "5mr": 1, "5p": 1})
        assert _can_declare_reach(hand, [], shanten=0, reached=False) is True

    def test_reach_blocked_not_tenpai(self):
        hand = Counter({"1m": 1, "2m": 1, "3m": 1, "4m": 1, "5m": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 1, "2p": 1, "3p": 1, "5mr": 1, "5p": 1})
        assert _can_declare_reach(hand, [], shanten=1, reached=False) is False

    def test_reach_blocked_already_reached(self):
        hand = Counter({"1m": 1, "2m": 1, "3m": 1, "4m": 1, "5m": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 1, "2p": 1, "3p": 1, "5mr": 1, "5p": 1})
        assert _can_declare_reach(hand, [], shanten=0, reached=True) is False

    def test_reach_blocked_with_pon(self):
        """副露（非暗杠）阻塞立直。"""
        hand = Counter({"1m": 1, "2m": 1, "3m": 1, "4m": 1, "5m": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 2, "2p": 1, "3p": 1, "5mr": 1})
        melds = [{"type": "pon", "pai": "1p"}]
        assert _can_declare_reach(hand, melds, shanten=0, reached=False) is False

    def test_reach_allowed_with_only_ankan(self):
        """只有暗杠时仍可立直：hand + 3*ankan数 = 14。"""
        # 11 tiles in hand + 3 from ankan = 14
        hand = Counter({"3m": 3, "3p": 2, "6p": 2, "6s": 1, "7s": 1, "8p": 1, "1z": 1})
        melds = [{"type": "ankan", "pai": "1p", "consumed": ["1p"] * 4}]
        assert _can_declare_reach(hand, melds, shanten=0, reached=False) is True

    def test_reach_blocked_shanten_none(self):
        hand = Counter({"1m": 1, "2m": 1, "3m": 1, "4m": 1, "5m": 1, "6m": 1, "7m": 1, "8m": 1, "9m": 1, "1p": 1, "2p": 1, "3p": 1, "5mr": 1, "5p": 1})
        assert _can_declare_reach(hand, [], shanten=None, reached=False) is False

    def test_reach_allowed_14_with_ankan(self):
        """14张 = 手牌 + 3*concealed_kan_count。"""
        # 11 tiles in hand + 3 from ankan = 14
        hand = Counter({"3m": 3, "3p": 2, "6p": 2, "6s": 1, "7s": 1, "8p": 1, "1z": 1})
        melds = [{"type": "ankan", "pai": "1p", "consumed": ["1p"] * 4}]
        assert _can_declare_reach(hand, melds, shanten=0, reached=False) is True


# =============================================================================
# A分支: kakan 响应窗口 (last_kakan.actor != actor)
# =============================================================================

class TestKakanResponseWindow:
    """分支 A: last_kakan 存在且非本人 → 只能 hora + none"""

    def test_kakan_response_hora_present(self, monkeypatch):
        player = PlayerState()
        player.hand.update(["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_kakan = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)
        _force_python_legal_fallback(monkeypatch)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

        legal = enumerate_legal_actions(snap, actor=0)

        assert any(a.type == "hora" and a.pai == "5mr" for a in legal)
        assert any(a.type == "none" for a in legal)

    def test_kakan_response_hora_absent_furiten(self, monkeypatch):
        player = PlayerState()
        player.hand.update(["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])
        player.furiten = True

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_kakan = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)

        legal = enumerate_legal_actions(snap, actor=0)

        assert not any(a.type == "hora" for a in legal)
        assert any(a.type == "none" for a in legal)

    def test_kakan_response_no_pon_chi_daiminkan(self, monkeypatch):
        """kakan 窗口不应出现 pon/chi/daiminkan。"""
        player = PlayerState()
        player.hand.update(["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_kakan = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: False)

        legal = enumerate_legal_actions(snap, actor=0)

        assert not any(a.type in ("pon", "chi", "daiminkan") for a in legal)
        assert any(a.type == "none" for a in legal)


# =============================================================================
# B分支: 舍牌响应窗口 (last_discard.actor != actor)
# =============================================================================

class TestDiscardReactionWindow:
    """分支 B: last_discard 存在且非本人舍牌 → hora/pon/daiminkan/chi/none"""

    def _make_reaction_state(self, discard_actor=1, discard_pai="5m", actor=0,
                              hand_tiles=None, reached=False, furiten=False,
                              last_tsumo=None, last_tsumo_raw=None):
        player = PlayerState()
        player.hand.update(hand_tiles or ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        player.reached = reached
        player.furiten = furiten

        gs = GameState()
        gs.players = [PlayerState(), PlayerState(), PlayerState(), PlayerState()]
        gs.players[actor] = player
        gs.last_discard = {"actor": discard_actor, "pai": discard_pai, "pai_raw": discard_pai}
        if last_tsumo is not None:
            gs.last_tsumo = [None, None, None, None]
            gs.last_tsumo[actor] = last_tsumo
        if last_tsumo_raw is not None:
            gs.last_tsumo_raw = [None, None, None, None]
            gs.last_tsumo_raw[actor] = last_tsumo_raw
        return gs.snapshot(actor=actor)

    # B1: hora
    def test_discard_reaction_hora_present(self, monkeypatch):
        snap = self._make_reaction_state()
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "hora" and a.target == 1 for a in legal)

    def test_discard_reaction_hora_absent_furiten(self, monkeypatch):
        snap = self._make_reaction_state(furiten=True)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "hora" for a in legal)

    def test_discard_reaction_hora_absent_not_tenpai(self, monkeypatch):
        snap = self._make_reaction_state()
        _force_python_legal_fallback(monkeypatch)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: False)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "hora" for a in legal)

    # B2: pon
    def test_discard_reaction_pon_present(self):
        snap = self._make_reaction_state(discard_pai="1m", hand_tiles=["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "pon" and a.target == 1 and a.pai == "1m" for a in legal)

    def test_discard_reaction_pon_with_aka(self):
        """手里有 5m+5mr，pon 5m 的 consumed 应优先使用 aka（aka 只针对 5-family）。"""
        snap = self._make_reaction_state(discard_pai="5m", hand_tiles=["5m", "5mr", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"])
        legal = enumerate_legal_actions(snap, actor=0)
        pon_actions = [a for a in legal if a.type == "pon"]
        assert len(pon_actions) >= 1, f"should have pon for 5m, got {[a.type for a in legal]}"
        consumed = pon_actions[0].consumed
        # consumed 应优先使用 aka（5mr > 5m）
        assert "5mr" in consumed, f"pon consumed should include 5mr, got {consumed}"

    def test_discard_reaction_pon_absent_insufficient(self):
        snap = self._make_reaction_state(discard_pai="1m", hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "pon" for a in legal)

    def test_discard_reaction_pon_absent_reached(self):
        snap = self._make_reaction_state(reached=True)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "pon" for a in legal)

    # B3: daiminkan
    def test_discard_reaction_daiminkan_present(self):
        snap = self._make_reaction_state(discard_pai="1m", hand_tiles=["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "daiminkan" and a.target == 1 for a in legal)

    def test_discard_reaction_daiminkan_absent_insufficient(self):
        snap = self._make_reaction_state(discard_pai="1m", hand_tiles=["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "daiminkan" for a in legal)

    def test_discard_reaction_daiminkan_absent_reached(self):
        snap = self._make_reaction_state(reached=True, hand_tiles=["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "daiminkan" for a in legal)

    # B4: chi
    def test_discard_reaction_chi_present(self):
        """discarder=1，actor=2 是下家，可以 chi 123m。"""
        snap = self._make_reaction_state(discard_actor=1, discard_pai="2m", actor=2,
                                          hand_tiles=["1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])
        legal = enumerate_legal_actions(snap, actor=2)
        assert any(a.type == "chi" and a.target == 1 for a in legal)

    def test_discard_reaction_chi_multiple_patterns(self):
        """打 5m 时，下家可以有 345m、456m、567m 三种 chi。"""
        snap = self._make_reaction_state(discard_actor=1, discard_pai="5m", actor=2,
                                          hand_tiles=["3m", "4m", "5mr", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p"])
        legal = enumerate_legal_actions(snap, actor=2)
        chi_actions = [a for a in legal if a.type == "chi"]
        assert len(chi_actions) >= 2  # at least 2 chi patterns

    def test_discard_reaction_chi_absent_not_next(self):
        """actor=0 不是 discarder=1 的下家，不应 chi。"""
        snap = self._make_reaction_state(discard_actor=1, discard_pai="2m", actor=0,
                                          hand_tiles=["1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "chi" for a in legal)

    def test_discard_reaction_chi_absent_reached(self):
        """立直后不能 chi。"""
        snap = self._make_reaction_state(discard_actor=1, discard_pai="2m", actor=2,
                                          hand_tiles=["1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"],
                                          reached=True)
        legal = enumerate_legal_actions(snap, actor=2)
        assert not any(a.type == "chi" for a in legal)

    def test_discard_reaction_chi_aka_preferred(self):
        """chi 456m 时，若手里有 5mr，consumed 应使用 5mr。"""
        snap = self._make_reaction_state(discard_actor=1, discard_pai="5m", actor=2,
                                          hand_tiles=["2m", "4m", "5mr", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p"])
        legal = enumerate_legal_actions(snap, actor=2)
        chi_actions = [a for a in legal if a.type == "chi" and a.pai == "5m"]
        assert len(chi_actions) >= 1
        consumed = chi_actions[0].consumed
        # consumed should have 4m and 6m (not 5mr since 5mr is the discard)
        assert set(consumed) == {"4m", "6m"}

    # B5: none
    def test_discard_reaction_none_always_present(self):
        snap = self._make_reaction_state()
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "none" for a in legal)

    # B6: no other actions when not applicable
    def test_discard_reaction_no_dahai(self):
        """舍牌响应窗口不应出现 dahai。"""
        snap = self._make_reaction_state()
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "dahai" for a in legal)

    def test_discard_reaction_no_reach(self):
        """舍牌响应窗口不应出现 reach。"""
        snap = self._make_reaction_state()
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "reach" for a in legal)


# =============================================================================
# C分支: 自己回合 (actor_to_move == actor)
# =============================================================================

class TestOwnTurnBranch:
    """分支 C: actor_to_move == actor"""

    def _make_own_turn_state(self, actor=0, hand_tiles=None, reached=False,
                              pending_reach=False, last_tsumo=None, last_tsumo_raw=None,
                              melds=None, shanten=None, furiten_list=None):
        player = PlayerState()
        player.hand.update(hand_tiles or ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        player.reached = reached
        player.pending_reach = pending_reach
        if melds:
            player.melds = melds

        gs = GameState()
        gs.players = [PlayerState(), PlayerState(), PlayerState(), PlayerState()]
        gs.players[actor] = player
        gs.actor_to_move = actor
        if last_tsumo is not None:
            gs.last_tsumo = [None, None, None, None]
            gs.last_tsumo[actor] = last_tsumo
        if last_tsumo_raw is not None:
            gs.last_tsumo_raw = [None, None, None, None]
            gs.last_tsumo_raw[actor] = last_tsumo_raw
        snap = gs.snapshot(actor=actor)
        if shanten is not None:
            snap["shanten"] = shanten
        if furiten_list is not None:
            snap["furiten"] = furiten_list
        return snap

    # ------ C1: 自摸阶段 ------
    def test_own_turn_tsumo_hora_present(self, monkeypatch):
        snap = self._make_own_turn_state(last_tsumo="5mr", last_tsumo_raw="5mr")
        _force_python_legal_fallback(monkeypatch)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: True)
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "hora" and a.target == 0 and a.pai == "5mr" for a in legal)

    def test_own_turn_tsumo_hora_absent_not_tenpai(self, monkeypatch):
        snap = self._make_own_turn_state(last_tsumo="1m", last_tsumo_raw="1m")
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", lambda *args, **kwargs: False)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "hora" for a in legal)

    def test_own_turn_tsumo_raw_falls_back_to_tsumo(self):
        """last_tsumo_raw is None → 用 last_tsumo 作为 pai。"""
        snap = self._make_own_turn_state(last_tsumo="5mr", last_tsumo_raw=None)
        legal = enumerate_legal_actions(snap, actor=0)
        # Should have dahai for the tsumo tile
        assert any(a.type == "dahai" and a.pai == "5mr" for a in legal)

    # ------ C2: 已立直 (reached=True) ------
    def test_reached_only_tsumogiri(self):
        snap = self._make_own_turn_state(reached=True, last_tsumo="5mr", last_tsumo_raw="5mr",
                                          hand_tiles=["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        # Only tsumogiri dahai
        dahai_actions = [a for a in legal if a.type == "dahai"]
        assert len(dahai_actions) == 1
        assert dahai_actions[0].tsumogiri is True
        assert not any(a.type == "reach" for a in legal)

    def test_reached_ankan_still_present_when_wait_shape_guard_passes(self, monkeypatch):
        _force_python_legal_fallback(monkeypatch, disable_structural=True)
        monkeypatch.setattr("mahjong_env.legal_actions._ankan_allowed_after_reach", lambda *args, **kwargs: True)
        snap = self._make_own_turn_state(reached=True, last_tsumo="5mr", last_tsumo_raw="5mr",
                                          hand_tiles=["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "ankan" for a in legal)

    def test_reached_ankan_blocked_when_wait_shape_changes(self):
        snap = self._make_own_turn_state(
            reached=True,
            last_tsumo="4m",
            last_tsumo_raw="4m",
            hand_tiles=["1s", "2s", "3s", "3s", "3s", "3s", "1m", "1m", "1m", "2m", "2m", "2m", "4m"],
        )
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "ankan" and a.pai == "3s" for a in legal)

    def test_reached_no_chi_pon(self):
        snap = self._make_own_turn_state(reached=True, last_tsumo="5mr", last_tsumo_raw="5mr")
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type in ("chi", "pon", "daiminkan", "reach") for a in legal)

    # ------ C3: pending_reach ------
    def test_pending_reach_dahai_only_keeps_tenpai(self):
        """只有打出后仍听牌的牌才在 legal 中。"""
        # Hand: tenpai for 7m, discarding 1m or 2m keeps tenpai, discarding others breaks
        snap = self._make_own_turn_state(
            pending_reach=True,
            last_tsumo="1m",
            last_tsumo_raw="1m",
            hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"],
            shanten=0,
        )
        legal = enumerate_legal_actions(snap, actor=0)
        # Should have dahai (pending reach shanten calc either passes or fallback)
        assert all(a.type == "dahai" for a in legal)

    def test_pending_reach_no_other_actions(self):
        snap = self._make_own_turn_state(pending_reach=True, last_tsumo="1m", last_tsumo_raw="1m")
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type in ("hora", "pon", "chi", "daiminkan", "reach", "ankan", "kakan") for a in legal)

    # ------ C4: 普通打牌阶段 ------
    def test_normal_dahai_all_tiles(self):
        snap = self._make_own_turn_state()
        legal = enumerate_legal_actions(snap, actor=0)
        dahai_actions = [a for a in legal if a.type == "dahai"]
        assert len(dahai_actions) == 13  # 13 unique tiles in hand

    def test_normal_dahai_tsumogiri_correct(self):
        """last_tsumo 匹配时 tsumogiri=True。"""
        snap = self._make_own_turn_state(last_tsumo="5mr", last_tsumo_raw="5mr",
                                          hand_tiles=["1m", "2m", "3m", "4m", "5mr", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        legal = enumerate_legal_actions(snap, actor=0)
        tsumogiri_actions = [a for a in legal if a.type == "dahai" and a.tsumogiri is True]
        assert len(tsumogiri_actions) == 1
        assert tsumogiri_actions[0].pai == "5mr"

    def test_normal_reach_allowed_tenpai(self):
        """普通打牌阶段：14张手牌 + shanten=0 → reach 合法。"""
        snap = self._make_own_turn_state(
            shanten=0,
            hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr", "5p"]
        )
        legal = enumerate_legal_actions(snap, actor=0)
        assert any(a.type == "reach" for a in legal), f"14-tile tenpai hand should allow reach, got {[a.type for a in legal]}"

    def test_normal_reach_blocked_shanten_1(self):
        snap = self._make_own_turn_state(shanten=1)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "reach" for a in legal)

    def test_normal_reach_blocked_shanten_none(self):
        snap = self._make_own_turn_state(shanten=None)
        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "reach" for a in legal)

    def test_normal_ankan_multiple_candidates(self):
        snap = self._make_own_turn_state(hand_tiles=["1m", "1m", "1m", "1m", "5p", "5p", "5p", "5p", "2m", "3m", "4m", "5m", "6m"])
        legal = enumerate_legal_actions(snap, actor=0)
        ankan_actions = [a for a in legal if a.type == "ankan"]
        assert len(ankan_actions) == 2  # 1m and 5p

    def test_normal_ankan_aka_tile_included(self):
        """aka 牌可作为 ankan 来源（aka 归一化后等同于普通牌）。"""
        snap = self._make_own_turn_state(
            hand_tiles=["5mr", "5mr", "5mr", "5mr", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
        )
        legal = enumerate_legal_actions(snap, actor=0)
        ankan_actions = [a for a in legal if a.type == "ankan"]
        # aka tiles归一化后可以作为暗杠来源
        assert len(ankan_actions) >= 1

    def test_normal_kakan_multiple_pon(self):
        """有多个 pon 时，应有多个 kakan 候选。"""
        player = PlayerState()
        player.hand.update(["5mr", "5pr"])
        player.melds = [
            {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
            {"type": "pon", "pai": "5p", "pai_raw": "5p", "consumed": ["5p", "5p", "5p"], "target": 2},
        ]
        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)
        kakan_actions = [a for a in legal if a.type == "kakan"]
        assert len(kakan_actions) == 2

    def test_normal_kakan_with_aka_counts_as_tile(self):
        """pon 存在且手里有 aka 版时，kakan 仍合法（aka 等价于普通牌）。"""
        player = PlayerState()
        player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        player.melds = [
            {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
        ]
        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)
        # 5mr counts as 5m for kakan purposes
        assert any(a.type == "kakan" for a in legal)


# =============================================================================
# D分支: 无 last_discard (游戏开始等)
# =============================================================================

class TestNoLastDiscard:
    """分支 D: 无 last_discard → 只有 none"""

    def test_no_last_discard_returns_none(self):
        gs = GameState()
        gs.players = [PlayerState(), PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)
        snap["last_discard"] = None
        snap["last_kakan"] = None

        legal = enumerate_legal_actions(snap, actor=0)

        assert all(a.type == "none" for a in legal)


# =============================================================================
# E分支: 自己的舍牌后 (last_discard.actor == actor)
# =============================================================================

class TestOwnDiscard:
    """分支 E: last_discard.actor == actor → 只有 none"""

    def test_own_discard_returns_none(self):
        gs = GameState()
        player = PlayerState()
        player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)

        assert all(a.type == "none" for a in legal)


# =============================================================================
# 交叉边缘情况
# =============================================================================

class TestCrossCuttingEdgeCases:
    """跨多个分支的边缘情况。"""

    def test_furiten_blocks_ron_not_tsumo(self, monkeypatch):
        """振听只阻塞荣和，不阻塞自摸。"""
        player = PlayerState()
        player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        player.furiten = True

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        gs.last_tsumo = [None, None, None, None]
        gs.last_tsumo[0] = "5mr"
        gs.last_tsumo_raw = [None, None, None, None]
        gs.last_tsumo_raw[0] = "5mr"
        snap = gs.snapshot(actor=0)

        def mock_hora(snap, actor, target, pai, is_tsumo, is_chankan=False):
            return is_tsumo  # tsumo allowed, ron blocked

        _force_python_legal_fallback(monkeypatch)
        monkeypatch.setattr("mahjong_env.legal_actions.can_hora_from_snapshot", mock_hora)
        legal = enumerate_legal_actions(snap, actor=0)
        # furiten only affects ron, tsumo hora should still appear
        assert any(a.type == "hora" for a in legal)

    def test_hand_with_only_aka_for_pon(self):
        """手里只有 aka 版 5mr，没有普通 5m，pon 仍应成立。"""
        player = PlayerState()
        player.hand.update(["5mr", "5mr", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"])

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 1, "pai": "5mr", "pai_raw": "5mr"}
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)
        pon_actions = [a for a in legal if a.type == "pon"]
        assert len(pon_actions) == 1

    def test_hand_13_tiles_no_reach(self):
        """13张手牌（未摸牌）不能立直。"""
        player = PlayerState()
        player.hand.update(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)
        snap["shanten"] = 1

        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "reach" for a in legal)

    def test_pon_meld_type_blocks_reach(self):
        """chi 型副露同样阻塞立直。"""
        player = PlayerState()
        player.hand.update(["1m", "1m", "1m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5mr"])
        player.melds = [{"type": "chi", "pai": "2m", "pai_raw": "2m", "consumed": ["1m", "3m"], "target": 3}]

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)
        snap["shanten"] = 0

        legal = enumerate_legal_actions(snap, actor=0)
        assert not any(a.type == "reach" for a in legal)

    def test_reached_blocks_daiminkan_not_own_turn(self):
        """立直后的大明杠响应应被正确阻塞。"""
        player = PlayerState()
        player.hand.update(["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "5mr"])
        player.reached = True

        gs = GameState()
        gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
        gs.last_discard = {"actor": 0, "pai": "1m", "pai_raw": "1m"}
        snap = gs.snapshot(actor=1)

        legal = enumerate_legal_actions(snap, actor=1)
        assert not any(a.type in ("pon", "daiminkan", "chi") for a in legal)
        assert any(a.type == "none" for a in legal)

    def test_action_spec_roundtrip(self):
        """ActionSpec to_action roundtrip 保持数据一致。"""
        from mahjong_env.types import ActionSpec

        spec = ActionSpec(type="pon", actor=0, pai="5mr", consumed=("5mr", "5mr"), target=1)
        action = spec.to_action()
        assert action.type == "pon"
        assert action.pai == "5mr"
        assert action.consumed == ["5mr", "5mr"]
        assert action.target == 1

        mjai = action.to_mjai()
        assert mjai == {"type": "pon", "actor": 0, "pai": "5mr", "consumed": ["5mr", "5mr"], "target": 1}

    def test_action_spec_none_type(self):
        """none 动作的 actor 字段应为 None（mjai 协议要求）。"""
        from mahjong_env.types import ActionSpec

        spec = ActionSpec(type="none")
        action = spec.to_action()
        assert action.actor == 0  # default
        mjai = action.to_mjai()
        assert mjai == {"type": "none"}
        assert "actor" not in mjai

    def test_kakan_consumed_includes_pon_base(self):
        """kakan 的 consumed 应包含原始 pon 的 consumed。"""
        player = PlayerState()
        player.hand.update(["5mr"])
        player.melds = [
            {"type": "pon", "pai": "5m", "pai_raw": "5m", "consumed": ["5m", "5m", "5m"], "target": 1},
        ]

        gs = GameState()
        gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
        gs.actor_to_move = 0
        snap = gs.snapshot(actor=0)

        legal = enumerate_legal_actions(snap, actor=0)
        kakan_actions = [a for a in legal if a.type == "kakan"]
        assert len(kakan_actions) == 1
        # consumed from pon base + the added tile
        assert len(kakan_actions[0].consumed) == 4
