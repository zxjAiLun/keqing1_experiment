"""Independent per-event contract reconstruction for the D3 B250 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from training.mortal.d3_exploration_engine import (
    CONTRACT_ID, DISCARD_ACTION_LIMIT, EXPLORATION_PROBABILITY, HANCHAN_BUDGET,
    KYOKU_BUDGET, MARGIN_THRESHOLD, canonical_hash_u,
)
from training.mortal.d3_production_contract import expected_seed_keys
from training.mortal.d3_production_audit_core import DecisionSnapshot, _event_context

def audit_event_records(
    events: list[dict[str, Any]],
    snapshots: dict[tuple[int, int, int, int, int], DecisionSnapshot],
) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    kyoku_counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    hanchan_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    reason_counts: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    expected_seeds = expected_seed_keys()

    def fail(kind: str, message: str) -> None:
        violations[kind] += 1
        errors.append(message)

    for event in events:
        try:
            context = _event_context(event)
            generation_seed, seed_key, seat, kyoku_index, decision_index = context
            if context in seen:
                fail("duplicate_context", f"duplicate eligible decision context: {context}")
            seen.add(context)
            if (generation_seed, seed_key) not in expected_seeds:
                fail("seed_range", f"event seed outside B250: {context}")
            if seat not in range(4) or kyoku_index < 0 or decision_index < 0:
                fail("context", f"invalid decision context indices: {context}")
            if event.get("contract_id") != CONTRACT_ID:
                fail("contract", f"wrong contract_id: {context}")
            if not math.isclose(
                float(event.get("exploration_probability", float("nan"))),
                EXPLORATION_PROBABILITY,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                fail("contract", f"wrong exploration probability: {context}")
            if event.get("context_kind") != "primary_action":
                fail("primary_context", f"non-primary exploration event: {context}")
            if event.get("exploration_allowed") is not True:
                fail("primary_context", f"exploration_allowed is not true: {context}")
            if bool(event.get("own_riichi")):
                fail("own_riichi", f"own-riichi exploration event: {context}")

            snapshot = snapshots.get(context)
            if snapshot is None:
                fail("loader_mapping", f"eligible event has no GameplayLoader decision: {context}")
                continue
            if not snapshot.eligible:
                fail("eligibility", f"event context is not independently eligible: {context}")
            if snapshot.own_riichi:
                fail("own_riichi", f"loader marks event as own-riichi: {context}")

            top1_action = int(event["top1_action"])
            top2_action = int(event["top2_action"])
            actual_action = int(event["actual_action"])
            if not (
                0 <= top1_action < DISCARD_ACTION_LIMIT
                and 0 <= top2_action < DISCARD_ACTION_LIMIT
                and top1_action != top2_action
            ):
                fail("semantic", f"event is not distinct discard->discard: {context}")
            if top1_action not in snapshot.finite_legal_actions:
                fail("legal_finite", f"top1 is not independently legal+finite: {context}")
            if top2_action not in snapshot.finite_legal_actions:
                fail("legal_finite", f"top2 is not independently legal+finite: {context}")
            if snapshot.top1_action != top1_action or snapshot.top2_action != top2_action:
                fail(
                    "ranking",
                    f"stable top1/top2 mismatch: {context} event=({top1_action},{top2_action}) "
                    f"loader=({snapshot.top1_action},{snapshot.top2_action})",
                )

            top1_q = float(event["top1_q"])
            top2_q = float(event["top2_q"])
            margin = float(event["margin"])
            if not all(math.isfinite(value) for value in (top1_q, top2_q, margin)):
                fail("finite_q", f"event contains non-finite Q: {context}")
            if margin > MARGIN_THRESHOLD + 1e-12:
                fail("threshold", f"margin threshold violation: {context}")
            if snapshot.top1_q is None or not math.isclose(
                top1_q, snapshot.top1_q, rel_tol=1e-5, abs_tol=1e-6
            ):
                fail("q_recompute", f"top1 Q mismatch: {context}")
            if snapshot.top2_q is None or not math.isclose(
                top2_q, snapshot.top2_q, rel_tol=1e-5, abs_tol=1e-6
            ):
                fail("q_recompute", f"top2 Q mismatch: {context}")
            if snapshot.margin is None or not math.isclose(
                margin, snapshot.margin, rel_tol=1e-5, abs_tol=1e-6
            ):
                fail("q_recompute", f"margin mismatch: {context}")

            canonical, digest, hash_u = canonical_hash_u(*context)
            if event.get("hash_input") != canonical:
                fail("hash", f"canonical hash input mismatch: {context}")
            if event.get("hash_sha256") != digest:
                fail("hash", f"SHA-256 digest mismatch: {context}")
            if not math.isclose(float(event["hash_u"]), hash_u, rel_tol=0.0, abs_tol=1e-18):
                fail("hash", f"hash u mismatch: {context}")

            kyoku_key = (generation_seed, seed_key, seat, kyoku_index)
            hanchan_key = (generation_seed, seed_key, seat)
            kyoku_before = kyoku_counts[kyoku_key]
            hanchan_before = hanchan_counts[hanchan_key]
            if int(event["kyoku_exploration_count_before"]) != kyoku_before:
                fail("budget", f"kyoku budget counter mismatch: {context}")
            if int(event["hanchan_exploration_count_before"]) != hanchan_before:
                fail("budget", f"hanchan budget counter mismatch: {context}")

            if hanchan_before >= HANCHAN_BUDGET:
                expected_reason = "hanchan_budget_exhausted"
            elif kyoku_before >= KYOKU_BUDGET:
                expected_reason = "kyoku_budget_exhausted"
            elif hash_u >= EXPLORATION_PROBABILITY:
                expected_reason = "hash_rejected"
            else:
                expected_reason = "explored"
            reason = str(event["reason"])
            reason_counts[reason] += 1
            if reason != expected_reason:
                fail(
                    "budget_reason",
                    f"reason mismatch: {context} event={reason} expected={expected_reason}",
                )
            explored = bool(event["explored"])
            expected_explored = expected_reason == "explored"
            expected_action = top2_action if expected_explored else top1_action
            if explored != expected_explored:
                fail("actual_action", f"explored flag mismatch: {context}")
            if actual_action != expected_action:
                fail("actual_action", f"actual action/reason mismatch: {context}")
            if snapshot.action != actual_action:
                fail(
                    "actual_action",
                    f"GameplayLoader behavior differs from exploration event: {context} "
                    f"loader={snapshot.action} event={actual_action}",
                )
            if int(event["base_action"]) != top1_action:
                fail("base_action", f"base greedy action differs from stable top1: {context}")
            if expected_explored:
                kyoku_counts[kyoku_key] += 1
                hanchan_counts[hanchan_key] += 1
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            fail("malformed_event", f"malformed exploration event: {exc}")

    independently_eligible = {context for context, row in snapshots.items() if row.eligible}
    missing_events = sorted(independently_eligible - seen)
    extra_events = sorted(seen - independently_eligible)
    if missing_events:
        fail("event_set", f"eligible decisions missing exploration events: {missing_events[:20]}")
    if extra_events:
        fail("event_set", f"events outside independently eligible set: {extra_events[:20]}")
    if any(value > KYOKU_BUDGET for value in kyoku_counts.values()):
        fail("budget", "kyoku exploration budget exceeded")
    if any(value > HANCHAN_BUDGET for value in hanchan_counts.values()):
        fail("budget", "hanchan exploration budget exceeded")

    for reason in (
        "explored",
        "hash_rejected",
        "kyoku_budget_exhausted",
        "hanchan_budget_exhausted",
    ):
        reason_counts.setdefault(reason, 0)

    return {
        "event_count": len(events),
        "independently_eligible_count": len(independently_eligible),
        "unique_event_contexts": len(seen),
        "missing_event_count": len(missing_events),
        "extra_event_count": len(extra_events),
        "reason_counts": {key: int(value) for key, value in sorted(reason_counts.items())},
        "violation_counts": {key: int(value) for key, value in sorted(violations.items())},
        "explored_count": int(reason_counts.get("explored", 0)),
        "errors": errors,
        "passed": not errors,
    }
