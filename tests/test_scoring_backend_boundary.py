from __future__ import annotations

import pytest

from mahjong_env import scoring as scoring_mod
from mahjong_env.state import GameState, PlayerState


class _CountingBackend:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.calls = 0

    def estimate_hand_value(self, *, tiles136, win_tile, melds, dora_indicators, config):
        self.calls += 1
        return self._delegate.estimate_hand_value(
            tiles136=tiles136,
            win_tile=win_tile,
            melds=melds,
            dora_indicators=dora_indicators,
            config=config,
        )


def _build_open_ron_state() -> GameState:
    player = PlayerState()
    player.hand.update(["2p", "2p", "3s", "4s", "5sr", "8m", "8m"])
    player.melds = [
        {"type": "chi", "pai": "3m", "pai_raw": "3m", "consumed": ["2m", "4m"], "target": 3},
        {"type": "pon", "pai": "4p", "pai_raw": "4p", "consumed": ["4p", "4p"], "target": 2},
    ]

    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.oya = 0
    gs.dora_markers = ["1m"]
    gs.scores = [25000, 25000, 25000, 25000]
    gs.players = [player, PlayerState(), PlayerState(), PlayerState()]
    return gs


def test_score_hora_uses_score_backend(monkeypatch):
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)
    monkeypatch.setattr(scoring_mod.keqing_core, "is_enabled", lambda: False)

    gs = _build_open_ron_state()
    result = scoring_mod.score_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls >= 1
    assert result.han >= 0


def test_can_hora_uses_score_backend(monkeypatch):
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)
    monkeypatch.setattr(scoring_mod.keqing_core, "is_enabled", lambda: False)

    gs = _build_open_ron_state()
    ok = scoring_mod.can_hora(gs, actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls >= 1
    assert ok is True


def test_score_hora_prefers_native_hora_truth(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_state",
        lambda *args, **kwargs: {
            "is_tsumo": False,
            "resolved_is_rinshan": False,
            "resolved_is_haitei": False,
        },
    )
    monkeypatch.setattr(
        scoring_mod,
        "_context_from_state",
        lambda state: scoring_mod._ScoreContext("E", 0, 0, 0, ["1m"], None),
    )
    monkeypatch.setattr(scoring_mod.keqing_core, "is_enabled", lambda: True)
    monkeypatch.setattr(
        scoring_mod.keqing_core,
        "prepare_hora_tile_allocation",
        lambda prepared: {
            "closed_tile_ids": [0, 1],
            "melds": [],
            "dora_ids": [2],
            "ura_ids": [],
        },
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (
            "truth",
            scoring_mod._HoraTruth(
                han=3,
                fu=30,
                yaku=["Riichi", "Tanyao"],
                yaku_details=[{"key": "Riichi", "name": "Riichi", "han": 1}],
                is_open_hand=False,
                cost={
                    "main": 3900,
                    "main_bonus": 0,
                    "additional": 0,
                    "additional_bonus": 0,
                    "kyoutaku_bonus": 0,
                    "total": 3900,
                },
                deltas=[3900, -3900, 0, 0],
                dora_count=0,
                ura_count=0,
                aka_count=0,
                backend_name="riichienv-core",
                truth_source="riichienv-core-adapter",
            ),
        ),
    )
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)

    result = scoring_mod.score_hora(_build_open_ron_state(), actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls == 0
    assert result.han == 3
    assert result.deltas == [3900, -3900, 0, 0]


def test_can_hora_prefers_native_hora_truth(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_state",
        lambda *args, **kwargs: {
            "is_tsumo": False,
            "resolved_is_rinshan": False,
            "resolved_is_haitei": False,
        },
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (
            "truth",
            scoring_mod._HoraTruth(
                han=1,
                fu=30,
                yaku=["Riichi"],
                yaku_details=[{"key": "Riichi", "name": "Riichi", "han": 1}],
                is_open_hand=False,
                cost={
                    "main": 1000,
                    "main_bonus": 0,
                    "additional": 0,
                    "additional_bonus": 0,
                    "kyoutaku_bonus": 0,
                    "total": 1000,
                },
                deltas=[1000, -1000, 0, 0],
                dora_count=0,
                ura_count=0,
                aka_count=0,
                backend_name="riichienv-core",
                truth_source="riichienv-core-adapter",
            ),
        ),
    )
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)

    ok = scoring_mod.can_hora(_build_open_ron_state(), actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls == 0
    assert ok is True


def test_can_hora_from_snapshot_prefers_native_hora_truth(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_snapshot",
        lambda *args, **kwargs: {
            "is_tsumo": False,
            "resolved_is_rinshan": False,
            "resolved_is_haitei": False,
        },
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (
            "truth",
            scoring_mod._HoraTruth(
                han=1,
                fu=30,
                yaku=["Riichi"],
                yaku_details=[{"key": "Riichi", "name": "Riichi", "han": 1}],
                is_open_hand=False,
                cost={"main": 1000, "main_bonus": 0, "additional": 0, "additional_bonus": 0, "kyoutaku_bonus": 0, "total": 1000},
                deltas=[1000, -1000, 0, 0],
                dora_count=0,
                ura_count=0,
                aka_count=0,
                backend_name="riichienv-core",
                truth_source="riichienv-core-adapter",
            ),
        ),
    )
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)

    ok = scoring_mod.can_hora_from_snapshot({}, actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls == 0
    assert ok is True


def test_score_hora_falls_back_only_for_missing_native_capability(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_state",
        lambda *args, **kwargs: {
            "closed_tiles": ["2p", "2p", "3s", "4s", "5sr", "8m", "8m", "8m"],
            "melds": [
                {"type": "chi", "opened": True, "tiles": ["2m", "3m", "4m"]},
                {"type": "pon", "opened": True, "tiles": ["4p", "4p", "4p"]},
            ],
            "bakaze": "E",
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_markers": [],
            "active_ura_markers": [],
            "reached": False,
            "ippatsu_eligible": False,
            "is_tsumo": False,
            "is_chankan": False,
            "resolved_is_rinshan": False,
            "resolved_is_haitei": False,
            "resolved_is_houtei": False,
            "pai": "8m",
            "actor": 0,
            "target": 1,
        },
    )
    monkeypatch.setattr(scoring_mod.keqing_core, "is_enabled", lambda: True)
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: ("missing_capability", None),
    )
    backend = _CountingBackend(scoring_mod._PythonMahjongBackend())
    monkeypatch.setattr(scoring_mod, "_SCORE_BACKEND", backend)

    result = scoring_mod.score_hora(_build_open_ron_state(), actor=0, target=1, pai="8m", is_tsumo=False)

    assert backend.calls >= 1
    assert result.han >= 0


def test_score_hora_propagates_unexpected_native_truth_errors(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_state",
        lambda *args, **kwargs: {"is_tsumo": False, "resolved_is_rinshan": False, "resolved_is_haitei": False},
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("native truth drift")),
    )
    monkeypatch.setattr(
        scoring_mod,
        "_estimate_hand_value_from_prepared_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    with pytest.raises(RuntimeError, match="native truth drift"):
        scoring_mod.score_hora(_build_open_ron_state(), actor=0, target=1, pai="8m", is_tsumo=False)


def test_can_hora_propagates_unexpected_native_truth_errors(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_state",
        lambda *args, **kwargs: {"is_tsumo": False, "resolved_is_rinshan": False, "resolved_is_haitei": False},
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("native truth drift")),
    )
    monkeypatch.setattr(
        scoring_mod,
        "_estimate_hand_value_from_prepared_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    with pytest.raises(RuntimeError, match="native truth drift"):
        scoring_mod.can_hora(_build_open_ron_state(), actor=0, target=1, pai="8m", is_tsumo=False)


def test_can_hora_from_snapshot_propagates_unexpected_native_truth_errors(monkeypatch):
    monkeypatch.setattr(
        scoring_mod,
        "_prepared_hora_payload_from_snapshot",
        lambda *args, **kwargs: {"is_tsumo": False, "resolved_is_rinshan": False, "resolved_is_haitei": False},
    )
    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_hora_truth_from_prepared_payload",
        lambda prepared: (_ for _ in ()).throw(RuntimeError("native truth drift")),
    )
    monkeypatch.setattr(
        scoring_mod,
        "_estimate_hand_value_from_prepared_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    with pytest.raises(RuntimeError, match="native truth drift"):
        scoring_mod.can_hora_from_snapshot({}, actor=0, target=1, pai="8m", is_tsumo=False)
