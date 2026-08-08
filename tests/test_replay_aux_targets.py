from pathlib import Path

import pytest

from mahjong_env.replay import (
    build_replay_samples_mc_return,
    read_mjai_jsonl,
)
from mahjong_env.replay_normalizer import normalize_replay_events, replay_label_matches_legal


def test_build_replay_samples_mc_return_preserves_aux_targets_after_end_kyoku():
    path = Path("artifacts/converted_mjai/closehand&atk/2017122802gm-00a9-0000-90c30e7e.mjson")
    events = list(read_mjai_jsonl(path))

    samples = build_replay_samples_mc_return(events, strict_legal_labels=True)

    assert any(sample.win_target > 0.0 for sample in samples)
    assert any(sample.dealin_target > 0.0 for sample in samples)
    assert any(abs(sample.score_delta_target) > 0.0 for sample in samples)


def test_build_replay_samples_mc_return_native_produces_samples():
    pytest.importorskip("keqing_core")
    import keqing_core

    if not keqing_core.is_available():
        pytest.skip("Rust extension not available")

    path = Path("artifacts/converted_mjai/closehand&atk/2017122802gm-00a9-0000-90c30e7e.mjson")
    events = list(read_mjai_jsonl(path))
    keqing_core.enable_rust(True)
    native_samples = build_replay_samples_mc_return(
        events,
        strict_legal_labels=True,
    )
    assert native_samples


# =============================================================================
# replay label 与 legal set 的语义匹配
# 这些回归对应外部牌谱字段口径与 battle canonical action 不一致的兼容层。
# =============================================================================

def test_label_matches_legal_allows_ankan_label_without_pai_field():
    label = {
        "type": "ankan",
        "actor": 3,
        "consumed": ["1p", "1p", "1p", "1p"],
    }
    legal_actions = [
        {
            "type": "ankan",
            "actor": 3,
            "pai": "1p",
            "consumed": ["1p", "1p", "1p", "1p"],
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_allows_hora_label_without_pai_field():
    label = {
        "type": "hora",
        "actor": 0,
        "target": 1,
        "deltas": [3900, -2900, 0, 0],
        "ura_markers": [],
    }
    legal_actions = [
        {
            "type": "hora",
            "actor": 0,
            "pai": "8m",
            "target": 1,
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_hora_pai_mismatch_allowed():
    """hora label 的 pai 与 legal 的 pai 不同，但 target 相同时应允许（label 可能缺 pai）。"""
    label = {
        "type": "hora",
        "actor": 0,
        "target": 1,
    }
    legal_actions = [
        {
            "type": "hora",
            "actor": 0,
            "pai": "5mr",
            "target": 1,
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_chi_consumed_aka_family_equivalent():
    label = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }
    legal_actions = [
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "4p",
            "consumed": ["5pr", "6p"],
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_dahai_tsumogiri_mismatch_allowed():
    label = {
        "type": "dahai",
        "actor": 1,
        "pai": "3p",
        "tsumogiri": False,
    }
    legal_actions = [
        {
            "type": "dahai",
            "actor": 1,
            "pai": "3p",
            "tsumogiri": True,
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_ankan_pai_aka_family_equivalent():
    label = {
        "type": "ankan",
        "actor": 3,
        "consumed": ["5m", "5m", "5m", "5mr"],
    }
    legal_actions = [
        {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_kakan_pai_aka_family_equivalent():
    label = {
        "type": "kakan",
        "actor": 2,
        "pai": "5p",
        "consumed": ["5p", "5p", "5p"],
    }
    legal_actions = [
        {
            "type": "kakan",
            "actor": 2,
            "pai": "5pr",
            "consumed": ["5p", "5p", "5p"],
        }
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_reach_requires_explicit_reach_action():
    label = {
        "type": "reach",
        "actor": 2,
    }
    legal_actions = [
        {"type": "reach", "actor": 2},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
    ]

    assert replay_label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_reach_does_not_match_plain_discard_set():
    label = {
        "type": "reach",
        "actor": 2,
    }
    legal_actions = [
        {"type": "hora", "actor": 2, "pai": "6s", "target": 2},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
        {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False},
    ]

    assert replay_label_matches_legal(label, legal_actions) is False


def test_normalize_replay_events_enriches_self_hora_with_haitei_flag_and_pai(monkeypatch):
    from mahjong_env.state import apply_event as real_apply_event

    def patched_apply_event(state, event):
        real_apply_event(state, event)
        if event.get("type") == "tsumo" and event.get("actor") == 1:
            state.remaining_wall = 0

    monkeypatch.setattr("mahjong_env.replay_normalizer.apply_event", patched_apply_event)

    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "hora", "actor": 1, "target": 1, "deltas": [0, 1000, -500, -500], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "6m"
    assert hora["is_haitei"] is True


def test_normalize_replay_events_enriches_discard_hora_with_houtei_flag_and_pai(monkeypatch):
    from mahjong_env.state import apply_event as real_apply_event

    def patched_apply_event(state, event):
        real_apply_event(state, event)
        if event.get("type") == "dahai" and event.get("actor") == 1:
            state.remaining_wall = 0

    monkeypatch.setattr("mahjong_env.replay_normalizer.apply_event", patched_apply_event)

    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -1000, 1000, 0], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "4m"
    assert hora["is_houtei"] is True


def test_normalize_replay_events_does_not_mark_houtei_when_remaining_wall_not_zero():
    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -1000, 1000, 0], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "4m"
    assert "is_houtei" not in hora
