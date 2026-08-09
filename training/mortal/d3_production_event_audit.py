"""Independent per-event contract reconstruction for the D3 auditor v2.

Hard gates (frozen, no replay-Q dependence):

* C. event internal contract: every exploration event is self-consistent with
  the D3_top2_discard_v1 recipe (probability, margin identity and bound,
  SHA-256 hash decision, kyoku/hanchan budget counters, reason, explored flag,
  selected action, base greedy action).
* D. independent context/action correspondence: every event maps exactly once
  to a deterministic native arena scene (reconstructed in
  ``d3_native_scene``), and the replay-observed behavior action equals the
  event's actual action.

Replay Q values, ranking, and loader eligibility are NOT hard gates in v2: the
authoritative 25h smoke proved the offline batched Q recompute is not an
interchangeable numeric oracle for generation-time arena inference. They are
reported as descriptive diagnostics (layer E) only.
"""

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


def _q_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "median": float(ordered[n // 2]),
        "p95": float(ordered[min(n - 1, int(n * 0.95))]),
        "p99": float(ordered[min(n - 1, int(n * 0.99))]),
        "max": float(ordered[-1]),
    }


def audit_event_records(
    events: list[dict[str, Any]],
    snapshots: dict[tuple[int, int, int, int, int], DecisionSnapshot],
) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    kyoku_counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    hanchan_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    reason_counts: Counter[str] = Counter()
    contract_violations: Counter[str] = Counter()
    mapping_violations: Counter[str] = Counter()
    expected_seeds = expected_seed_keys()

    d_top1: list[float] = []
    d_top2: list[float] = []
    d_margin: list[float] = []
    ranking_flips = 0
    eligibility_flips = 0
    legal_finite_disagreements = 0
    compared_q = 0

    def fail(kind: str, message: str) -> None:
        contract_violations[kind] += 1
        errors.append(message)

    def map_fail(kind: str, message: str) -> None:
        mapping_violations[kind] += 1
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

            top1_action = int(event["top1_action"])
            top2_action = int(event["top2_action"])
            actual_action = int(event["actual_action"])
            if not (
                0 <= top1_action < DISCARD_ACTION_LIMIT
                and 0 <= top2_action < DISCARD_ACTION_LIMIT
                and top1_action != top2_action
            ):
                fail("semantic", f"event is not distinct discard->discard: {context}")
            top1_q = float(event["top1_q"])
            top2_q = float(event["top2_q"])
            margin = float(event["margin"])
            if not all(math.isfinite(value) for value in (top1_q, top2_q, margin)):
                fail("finite_q", f"event contains non-finite Q: {context}")
            if not math.isclose(margin, top1_q - top2_q, rel_tol=0.0, abs_tol=1e-12):
                fail("margin_identity", f"margin != top1_q - top2_q: {context}")
            if margin > MARGIN_THRESHOLD + 1e-12:
                fail("threshold", f"margin threshold violation: {context}")

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
                fail("budget_reason", f"reason mismatch: {context}")
            explored = bool(event["explored"])
            expected_explored = expected_reason == "explored"
            expected_action = top2_action if expected_explored else top1_action
            if explored != expected_explored:
                fail("actual_action", f"explored flag mismatch: {context}")
            if actual_action != expected_action:
                fail("actual_action", f"actual action/reason mismatch: {context}")
            if int(event["base_action"]) != top1_action:
                fail("base_action", f"base greedy action differs from stable top1: {context}")
            if expected_explored:
                kyoku_counts[kyoku_key] += 1
                hanchan_counts[hanchan_key] += 1

            # ---- layer D: mapping + behavior correspondence ----
            snapshot = snapshots.get(context)
            if snapshot is None:
                map_fail(
                    "unmapped_context",
                    f"event context has no reconstructed native scene: {context}",
                )
                continue
            if snapshot.action != actual_action:
                map_fail(
                    "behavior_mismatch",
                    f"replay behavior differs from exploration event: {context} "
                    f"loader={snapshot.action} event={actual_action}",
                )

            # ---- layer E: replay-Q diagnostics (descriptive) ----
            compared_q += 1
            if snapshot.top1_q is not None:
                d_top1.append(abs(top1_q - snapshot.top1_q))
                d_top2.append(abs(top2_q - snapshot.top2_q))
            if snapshot.margin is not None:
                d_margin.append(abs(margin - snapshot.margin))
            if (snapshot.top1_action, snapshot.top2_action) != (top1_action, top2_action):
                ranking_flips += 1
            if snapshot.eligible is not True:
                eligibility_flips += 1
            if (
                top1_action not in snapshot.finite_legal_actions
                or top2_action not in snapshot.finite_legal_actions
            ):
                legal_finite_disagreements += 1
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            fail("malformed_event", f"malformed exploration event: {exc}")

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

    mapped_snapshots = {context for context in seen if context in snapshots}
    missing_events = sorted(
        {context for context in seen} - mapped_snapshots
    )
    return {
        "event_count": len(events),
        "independently_eligible_count": len(seen),
        "unique_event_contexts": len(seen),
        "missing_event_count": len(missing_events),
        "extra_event_count": 0,
        "reason_counts": {key: int(value) for key, value in sorted(reason_counts.items())},
        "contract_violations": {
            key: int(value) for key, value in sorted(contract_violations.items())
        },
        "mapping_violations": {
            key: int(value) for key, value in sorted(mapping_violations.items())
        },
        "explored_count": int(reason_counts.get("explored", 0)),
        "q_diagnostics": {
            "compared_events": compared_q,
            "abs_diff_top1_q": _q_stats(d_top1),
            "abs_diff_top2_q": _q_stats(d_top2),
            "abs_diff_margin": _q_stats(d_margin),
            "ranking_flip_count": ranking_flips,
            "eligibility_flip_count": eligibility_flips,
            "legal_finite_disagreement_count": legal_finite_disagreements,
        },
        "errors": errors,
        "passed": not contract_violations and not mapping_violations,
    }
