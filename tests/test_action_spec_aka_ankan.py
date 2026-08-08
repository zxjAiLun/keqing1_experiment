"""
测试 ankan/pon/chi/daiminkan/kakan 中赤宝牌 consumed 顺序不同时的匹配问题。

Bug 历史：
1. 最初用 Counter(left.consumed) == Counter(right.consumed) 比较，
   但 Counter({'5p':1}) != Counter({'5pr':1})（不同字符串 key），
   导致 consumed=['5p','6p'] vs ['5pr','6p'] 匹配失败。
2. 修复：引入 _normalized_tile_counter，对每个 tile 先 normalize_tile 再计数。
3. 第133行的 bug（左pai比较用了原始 left.pai 而非 left_pai）也已修复。

错误示例（event_index=568）：
- label: chi consumed=['5p','6p']  (用普通5p吃)
- legal: chi consumed=['5pr','6p'] (legal用赤5pr吃)
- hand: ['2m','2m','3s','4p','5p','5pr','5s','6p','6p','7s','7s','8s','F']
"""

import pytest
from mahjong_env.types import action_dict_to_spec, action_specs_match


class TestAkaConsumedOrderMismatch:
    """
    场景：label 的 consumed 中赤宝牌排在非赤宝后面（consumed[0] 是非赤），
    导致 canonical_meld_pai 推导出 pai='5m'；而 legal 显式设 pai='5mr'。
    两者语义等价，应匹配成功。
    """

    def test_ankan_aka_consumed_order_mismatch_label_non_aka_first(self):
        """ankan: label consumed=[5m,5m,5m,5mr] → pai='5m', legal consumed=[5mr,5m,5m,5m] → pai='5mr'"""
        label = {
            "type": "ankan",
            "actor": 3,
            "consumed": ["5m", "5m", "5m", "5mr"],
            # 无 pai 字段，canonical_meld_pai 从 consumed[0] 取
        }
        legal = {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_ankan_aka_consumed_order_mismatch_label_aka_first(self):
        """ankan: label consumed=[5mr,5m,5m,5m] → pai='5mr', legal consumed=[5m,5m,5m,5mr] → pai='5m'"""
        label = {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }
        legal = {
            "type": "ankan",
            "actor": 3,
            "consumed": ["5m", "5m", "5m", "5mr"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_pon_aka_consumed_order_mismatch(self):
        """pon: 同理，consumed 顺序不同但 Counter 相等时应匹配"""
        label = {
            "type": "pon",
            "actor": 2,
            "consumed": ["5p", "5p", "5pr"],
        }
        legal = {
            "type": "pon",
            "actor": 2,
            "pai": "5pr",
            "consumed": ["5pr", "5p", "5p"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_daiminkan_aka_consumed_order_mismatch(self):
        """daiminkan: 同理"""
        label = {
            "type": "daiminkan",
            "actor": 1,
            "consumed": ["5s", "5s", "5sr", "5s"],
        }
        legal = {
            "type": "daiminkan",
            "actor": 1,
            "pai": "5sr",
            "consumed": ["5sr", "5s", "5s", "5s"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_chi_aka_consumed_order_mismatch(self):
        """chi: 同理，consumed 顺序不同但 Counter 相等时应匹配"""
        label = {
            "type": "chi",
            "actor": 0,
            "pai": "7m",
            "consumed": ["5m", "5mr", "6m"],
        }
        legal = {
            "type": "chi",
            "actor": 0,
            "pai": "7m",
            "consumed": ["5mr", "5m", "6m"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_chi_consumed_5p_vs_5pr_counter_equality(self):
        """
        真实错误场景：hand=['2m','2m','3s','4p','5p','5pr','5s','6p','6p','7s','7s','8s','F']
        last_discard={'actor':0,'pai':'4p'}
        label: chi consumed=['5p','6p']  (用普通5p吃)
        legal:  chi consumed=['5pr','6p'] (legal用赤5pr吃)

        Counter(['5p','6p']} != Counter(['5pr','6p']}，但语义等价，应匹配。
        这是预处理错误 event_index=568 的真实场景。
        """
        label = {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "4p",
            "consumed": ["5p", "6p"],
        }
        legal = {
            "type": "chi",
            "actor": 1,
            "pai": "4p",
            "consumed": ["5pr", "6p"],
            "target": 0,
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        # 验证 _normalized_tile_counter 正确工作
        from mahjong_env.types import _normalized_tile_counter
        assert _normalized_tile_counter(label_spec.consumed) == _normalized_tile_counter(legal_spec.consumed)

        # action_specs_match 应认为两者等价
        assert action_specs_match(label_spec, legal_spec) is True

    def test_kakan_aka_consumed_order_mismatch(self):
        """kakan: 同理"""
        label = {
            "type": "kakan",
            "actor": 2,
            "consumed": ["5p", "5p", "5pr", "5p"],
        }
        legal = {
            "type": "kakan",
            "actor": 2,
            "pai": "5pr",
            "consumed": ["5pr", "5p", "5p", "5p"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_pon_5p_vs_5pr_consumed_real_case(self):
        """真实场景：hand 同时有 5p 和 5pr，label 用 5p 碰，legal 用 5pr 碰"""
        label = {
            "type": "pon",
            "actor": 2,
            "target": 0,
            "pai": "5p",
            "consumed": ["5p", "5p", "6p"],  # 实际用普通 5p
        }
        legal = {
            "type": "pon",
            "actor": 2,
            "target": 0,
            "pai": "5pr",
            "consumed": ["5pr", "5p", "6p"],  # 重建用赤 5pr
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_daiminkan_5s_vs_5sr_consumed(self):
        """daiminkan: 同理，5s vs 5sr"""
        label = {
            "type": "daiminkan",
            "actor": 3,
            "target": 1,
            "consumed": ["5s", "5s", "5sr", "5s"],
        }
        legal = {
            "type": "daiminkan",
            "actor": 3,
            "target": 1,
            "pai": "5sr",
            "consumed": ["5sr", "5s", "5s", "5s"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_multiple_aka_in_consumed(self):
        """手牌有多个赤宝牌时的 consumed 组合"""
        label = {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "consumed": ["5mr", "5m", "5pr"],
        }
        legal = {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "consumed": ["5m", "5mr", "5pr"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert action_specs_match(label_spec, legal_spec) is True

    def test_chi_all_three_suits_aka_equivalence(self):
        """chi 吃牌：验证 5mr/5pr/5sr 各自与普通牌等价"""
        for label_aka, legal_aka in [("5mr", "5m"), ("5pr", "5p"), ("5sr", "5s")]:
            label = {
                "type": "chi",
                "actor": 0,
                "target": 3,
                "pai": "7m",
                "consumed": [label_aka, "6m"],
            }
            legal = {
                "type": "chi",
                "actor": 0,
                "target": 3,
                "pai": "7m",
                "consumed": [legal_aka, "6m"],
            }
            label_spec = action_dict_to_spec(label)
            legal_spec = action_dict_to_spec(legal)
            assert action_specs_match(label_spec, legal_spec) is True, \
                f"chi with {label_aka} vs {legal_aka} should match"

    def test_ankan_pai_field_takes_precedence_over_consumed(self):
        """当 label 显式提供 pai 字段时，应使用该 pai 而非从 consumed 推导"""
        label = {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }
        legal = {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }

        label_spec = action_dict_to_spec(label)
        legal_spec = action_dict_to_spec(legal)

        assert label_spec.pai == "5mr"
        assert action_specs_match(label_spec, legal_spec) is True
