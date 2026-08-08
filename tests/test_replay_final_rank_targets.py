from __future__ import annotations

import pytest

from mahjong_env.replay import build_replay_samples_mc_return


def _round_end_events() -> list[dict]:
    return [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
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
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["4p", "4p", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "5s", "5s", "5s"],
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "4s"},
        {"type": "dahai", "actor": 1, "pai": "4s", "tsumogiri": True},
        {
            "type": "ryukyoku",
            "scores": [35000, 25000, 20000, 20000],
            "tenpai_players": [],
        },
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [35000, 25000, 20000, 20000]},
    ]


def test_build_replay_samples_mc_return_fills_final_rank_targets():
    pytest.importorskip("keqing_core")

    samples = build_replay_samples_mc_return(_round_end_events(), strict_legal_labels=True)

    assert samples
    actor0 = next(sample for sample in samples if sample.actor == 0)
    assert actor0.final_rank_target == 0
    assert actor0.score_before_action == 25000
    assert actor0.final_score_delta_points_target == 10000


def test_build_replay_samples_mc_return_handles_tied_bottom_places():
    pytest.importorskip("keqing_core")

    samples = build_replay_samples_mc_return(_round_end_events(), strict_legal_labels=True)

    actor2 = next(sample for sample in samples if sample.actor == 2)
    assert actor2.final_rank_target == 2
    assert actor2.final_score_delta_points_target == -5000
