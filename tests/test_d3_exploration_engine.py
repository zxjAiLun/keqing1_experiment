from __future__ import annotations

from training.mortal.d3_exploration_engine import D3ExplorationEngine


class _FakeMortalEngine:
    is_oracle = False
    version = 1
    enable_quick_eval = False
    enable_rule_based_agari_guard = True
    enable_amp = False
    device = "cpu"

    def react_batch(self, obs, masks, invisible_obs):
        del invisible_obs
        return (
            [0 for _ in obs],
            [[1.0, 0.9, -1.0] for _ in obs],
            masks,
            [True for _ in obs],
        )

    def profile_snapshot(self):
        return {}


def test_auxiliary_context_cannot_explore_or_consume_budget() -> None:
    engine = D3ExplorationEngine(_FakeMortalEngine())
    masks = [[True, True, False], [True, True, False]]
    contexts = [
        (1, 8192, 0, 0, 0, False, False),
        (1, 8192, 0, 0, 1, False, True),
    ]

    actions, _, _, is_greedy = engine.react_batch([0, 0], masks, [None, None], contexts)

    assert actions[0] == 0
    assert actions[1] in {0, 1}
    assert is_greedy[0] is True
    assert len(engine.events) == 1
    assert engine.events[0]["context_kind"] == "primary_action"
    assert engine.summary()["counters"]["auxiliary_count"] == 1
    assert engine.summary()["counters"]["auxiliary_exploration_count"] == 0
