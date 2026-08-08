#!/usr/bin/env python3
"""Deterministic discard-only top-2 exploration wrapper for D3 generation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable


CONTRACT_ID = "D3_top2_discard_v1"
EXPLORATION_PROBABILITY = 0.25
MARGIN_THRESHOLD = 0.5
KYOKU_BUDGET = 1
HANCHAN_BUDGET = 8
DISCARD_ACTION_LIMIT = 37


def canonical_hash_u(
    generation_seed: int,
    seed_key: int,
    seat: int,
    kyoku_index: int,
    decision_index: int,
) -> tuple[str, str, float]:
    """Return the frozen canonical input, digest, and deterministic u value."""

    canonical = (
        f"{CONTRACT_ID}|{generation_seed}|{seed_key}|{seat}|"
        f"{kyoku_index}|{decision_index}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    integer = int.from_bytes(bytes.fromhex(digest[:16]), "little", signed=False)
    return canonical, digest, integer / float(1 << 64)


def _context_tuple(value: Iterable[Any]) -> tuple[int, int, int, int, int, bool, bool]:
    fields = tuple(value)
    if len(fields) != 7:
        raise ValueError(f"D3 decision context must have 7 fields, got {len(fields)}")
    (
        generation_seed,
        seed_key,
        seat,
        kyoku_index,
        decision_index,
        own_riichi,
        exploration_allowed,
    ) = fields
    return (
        int(generation_seed),
        int(seed_key),
        int(seat),
        int(kyoku_index),
        int(decision_index),
        bool(own_riichi),
        bool(exploration_allowed),
    )


class D3ExplorationEngine:
    """Wrap one MortalEngine and expose auditable D3 decisions to the runner."""

    engine_type = "mortal"
    supports_decision_context = True

    def __init__(
        self,
        base_engine: Any,
        *,
        name: str = "K0_70k",
        probability: float = EXPLORATION_PROBABILITY,
        margin_threshold: float = MARGIN_THRESHOLD,
    ) -> None:
        if probability != EXPLORATION_PROBABILITY:
            raise ValueError("D3 generation v1 only permits probability=0.25")
        if margin_threshold != MARGIN_THRESHOLD:
            raise ValueError("D3 generation v1 only permits margin_threshold=0.5")
        self.base_engine = base_engine
        self.name = name
        self.is_oracle = bool(base_engine.is_oracle)
        self.version = int(base_engine.version)
        self.enable_quick_eval = bool(base_engine.enable_quick_eval)
        self.enable_rule_based_agari_guard = bool(base_engine.enable_rule_based_agari_guard)
        self.enable_amp = bool(base_engine.enable_amp)
        self.device = base_engine.device
        self.probability = float(probability)
        self.margin_threshold = float(margin_threshold)
        self.events: list[dict[str, Any]] = []
        self.counters: Counter[str] = Counter()
        self._kyoku_counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
        self._hanchan_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
        self._lock = threading.Lock()

    def react_batch(self, obs: Any, masks: Any, invisible_obs: Any, decision_contexts: Any) -> Any:
        if len(obs) != len(decision_contexts):
            raise ValueError(
                f"D3 context length mismatch: observations={len(obs)} contexts={len(decision_contexts)}"
            )
        actions, q_values, returned_masks, is_greedy = self.base_engine.react_batch(
            obs, masks, invisible_obs
        )
        if not (len(actions) == len(q_values) == len(returned_masks) == len(is_greedy) == len(obs)):
            raise ValueError("base MortalEngine returned inconsistent batch lengths")

        output_actions = [int(action) for action in actions]
        explored_flags = [False] * len(obs)
        for index, context_value in enumerate(decision_contexts):
            context = _context_tuple(context_value)
            (
                generation_seed,
                seed_key,
                seat,
                kyoku_index,
                decision_index,
                own_riichi,
                exploration_allowed,
            ) = context
            if not exploration_allowed:
                self.counters["auxiliary_count"] += 1
                continue
            q_row = [float(value) for value in q_values[index]]
            mask_row = [bool(value) for value in returned_masks[index]]
            legal = [
                action_id
                for action_id, allowed in enumerate(mask_row)
                if allowed and math.isfinite(q_row[action_id])
            ]
            if len(legal) < 2:
                self.counters["states"] += 1
                self.counters["not_eligible_finite_action_count"] += 1
                continue

            ranked = sorted(legal, key=lambda action_id: (-q_row[action_id], action_id))
            top1_action, top2_action = ranked[:2]
            margin = q_row[top1_action] - q_row[top2_action]
            eligible = (
                not own_riichi
                and top1_action < DISCARD_ACTION_LIMIT
                and top2_action < DISCARD_ACTION_LIMIT
                and margin <= self.margin_threshold
            )
            self.counters["states"] += 1
            if not eligible:
                self.counters["not_eligible_count"] += 1
                continue

            canonical, digest, hash_u = canonical_hash_u(
                generation_seed,
                seed_key,
                seat,
                kyoku_index,
                decision_index,
            )
            kyoku_key = (generation_seed, seed_key, seat, kyoku_index)
            hanchan_key = (generation_seed, seed_key, seat)
            with self._lock:
                kyoku_count = self._kyoku_counts[kyoku_key]
                hanchan_count = self._hanchan_counts[hanchan_key]
                if hanchan_count >= HANCHAN_BUDGET:
                    reason = "hanchan_budget_exhausted"
                    selected = False
                elif kyoku_count >= KYOKU_BUDGET:
                    reason = "kyoku_budget_exhausted"
                    selected = False
                elif hash_u >= self.probability:
                    reason = "hash_rejected"
                    selected = False
                else:
                    reason = "explored"
                    selected = True
                    self._kyoku_counts[kyoku_key] += 1
                    self._hanchan_counts[hanchan_key] += 1

                actual_action = top2_action if selected else top1_action
                event = {
                    "contract_id": CONTRACT_ID,
                    "generation_seed": generation_seed,
                    "seed_key": seed_key,
                    "seat": seat,
                    "kyoku_index": kyoku_index,
                    "decision_index": decision_index,
                    "own_riichi": own_riichi,
                    "context_kind": "primary_action",
                    "exploration_allowed": exploration_allowed,
                    "top1_action": top1_action,
                    "top2_action": top2_action,
                    "top1_q": q_row[top1_action],
                    "top2_q": q_row[top2_action],
                    "margin": margin,
                    "hash_input": canonical,
                    "hash_sha256": digest,
                    "hash_u": hash_u,
                    "exploration_probability": self.probability,
                    "kyoku_exploration_count_before": kyoku_count,
                    "hanchan_exploration_count_before": hanchan_count,
                    "actual_action": actual_action,
                    "explored": selected,
                    "reason": reason,
                    "base_action": int(actions[index]),
                }
                self.events.append(event)
                self.counters["eligible_count"] += 1
                self.counters[f"{reason}_count"] += 1
                if selected:
                    output_actions[index] = actual_action
                    explored_flags[index] = True
        return output_actions, q_values, returned_masks, [
            bool(value) and not explored_flags[index]
            for index, value in enumerate(is_greedy)
        ]

    def profile_snapshot(self) -> dict[str, Any]:
        snapshot = self.base_engine.profile_snapshot()
        snapshot["d3_contract_id"] = CONTRACT_ID
        snapshot["d3_exploration_events"] = len(self.events)
        return snapshot

    def summary(self) -> dict[str, Any]:
        with self._lock:
            events = sorted(
                self.events,
                key=lambda event: (
                    int(event["generation_seed"]),
                    int(event["seed_key"]),
                    int(event["seat"]),
                    int(event["kyoku_index"]),
                    int(event["decision_index"]),
                ),
            )
            counters = dict(self.counters)
            counters.setdefault("auxiliary_exploration_count", 0)
            counters["explored_count"] = sum(bool(event["explored"]) for event in events)
            counters["event_count"] = len(events)
            counters["kyoku_count"] = len({
                (
                    event["generation_seed"],
                    event["seed_key"],
                    event["seat"],
                    event["kyoku_index"],
                )
                for event in events
            })
            counters["hanchan_count"] = len({
                (event["generation_seed"], event["seed_key"], event["seat"])
                for event in events
            })
            return {
                "contract_id": CONTRACT_ID,
                "probability": self.probability,
                "margin_threshold": self.margin_threshold,
                "kyoku_budget": KYOKU_BUDGET,
                "hanchan_budget": HANCHAN_BUDGET,
                "counters": counters,
                "events": events,
            }

    def write_audit_files(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        events_path = output_dir / "exploration_events.jsonl"
        with events_path.open("w", encoding="utf-8", newline="\n") as handle:
            for event in summary["events"]:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        summary_path = output_dir / "exploration_summary.json"
        summary_path.write_text(
            json.dumps({key: value for key, value in summary.items() if key != "events"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return {"events": events_path, "summary": summary_path}
