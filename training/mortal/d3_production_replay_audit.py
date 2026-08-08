"""GameplayLoader and K0 Q reconstruction for the D3 B250 audit."""
# ruff: noqa: E402

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys
from typing import Any

from training.mortal.d3_exploration_engine import DISCARD_ACTION_LIMIT, MARGIN_THRESHOLD
from training.mortal.d3_production_contract import DEVICE, GAMES, RANK_POINTS
from training.mortal.d3_production_audit_core import (
    DecisionSnapshot, _bucket, _phase, _quantiles, primary_row_flags,
)

def _build_decision_snapshots(
    *,
    log_manifest: dict[str, Any],
    k0_path: Path,
    mortal_root: Path,
    q_batch_size: int,
) -> tuple[dict[tuple[int, int, int, int, int], DecisionSnapshot], dict[str, Any]]:
    if q_batch_size <= 0:
        raise ValueError("--q-batch-size must be positive")
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415
    from training.mortal.audit_replay_distribution import (  # noqa: PLC0415
        action_name,
        load_checkpoint,
        load_model,
        model_q,
        records_from_game,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("production audit requires CUDA")
    device = torch.device(DEVICE)
    state = load_checkpoint(k0_path)
    brain, dqn, version = load_model(state, device)
    del state
    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=["K0_70k"],
        always_include_kan_select=True,
        augmented=False,
    )
    pts = np.asarray(RANK_POINTS, dtype=np.float64)
    snapshots: dict[tuple[int, int, int, int, int], DecisionSnapshot] = {}
    malformed: list[str] = []
    decisions_per_hanchan: list[int] = []
    phase_counts: Counter[str] = Counter()
    shanten_counts: Counter[int] = Counter()
    legal_action_counts: Counter[int] = Counter()
    action_counts: Counter[str] = Counter()
    eligible_phase: Counter[str] = Counter()
    eligible_shanten: Counter[int] = Counter()
    eligible_legal_counts: Counter[int] = Counter()
    eligible_action_counts: Counter[str] = Counter()
    gaps: list[float] = []
    eligible_gaps: list[float] = []
    behavior_legal_finite = 0
    primary_behavior_legal_finite = 0
    total_training_rows = 0
    total_primary_decisions = 0
    auxiliary_kan_select_rows = 0
    primary_greedy_agreement = 0
    all_kyoku_keys: set[tuple[int, int, int, int]] = set()

    for seed_key, row in sorted(log_manifest["rows"].items()):
        seed, key = seed_key
        path = Path(row["path"])
        k0_seat = int(row["k0_seat"])
        for kyoku_index in range(int(row["kyoku_count"])):
            all_kyoku_keys.add((seed, key, k0_seat, kyoku_index))
        try:
            loaded = loader.load_gz_log_files([str(path)])
            if len(loaded) != 1 or len(loaded[0]) != 1:
                raise ValueError(
                    f"expected exactly one K0 perspective, got outer={len(loaded)} "
                    f"inner={len(loaded[0]) if loaded else 'NA'}"
                )
            records = list(records_from_game(loaded[0][0], pts))
            if not records:
                raise ValueError("K0 perspective has zero decisions")
            q_rows: list[Any] = []
            for start in range(0, len(records), q_batch_size):
                q_rows.extend(model_q(brain, dqn, records[start : start + q_batch_size], device))
            if len(q_rows) != len(records):
                raise ValueError("parent Q row count mismatch")

            decision_index_by_kyoku: defaultdict[int, int] = defaultdict(int)
            primary_count = 0
            row_flags = primary_row_flags(record.action for record in records)
            for record, raw_q, is_primary in zip(records, q_rows, row_flags, strict=True):
                mask = np.asarray(record.mask, dtype=np.bool_)
                q = np.asarray(raw_q, dtype=np.float64)
                valid = mask & np.isfinite(q)
                finite_legal = tuple(int(value) for value in np.flatnonzero(valid))
                action = int(record.action)
                total_training_rows += 1
                if action in finite_legal:
                    behavior_legal_finite += 1

                if not is_primary:
                    auxiliary_kan_select_rows += 1
                    continue

                kyoku_index = int(record.kyoku)
                decision_index = decision_index_by_kyoku[kyoku_index]
                decision_index_by_kyoku[kyoku_index] += 1
                context = (seed, key, k0_seat, kyoku_index, decision_index)
                if context in snapshots:
                    raise ValueError(f"duplicate decision context: {context}")
                ranked = sorted(finite_legal, key=lambda candidate: (-float(q[candidate]), candidate))
                top1 = ranked[0] if ranked else None
                top2 = ranked[1] if len(ranked) >= 2 else None
                top1_q = float(q[top1]) if top1 is not None else None
                top2_q = float(q[top2]) if top2 is not None else None
                margin = top1_q - top2_q if top1_q is not None and top2_q is not None else None
                eligible = bool(
                    not record.own_riichi
                    and top1 is not None
                    and top2 is not None
                    and top1 < DISCARD_ACTION_LIMIT
                    and top2 < DISCARD_ACTION_LIMIT
                    and margin is not None
                    and margin <= MARGIN_THRESHOLD
                )
                snapshot = DecisionSnapshot(
                    context=context,
                    action=action,
                    own_riichi=bool(record.own_riichi),
                    shanten=int(record.shanten),
                    phase=_phase(kyoku_index),
                    legal_action_count=int(mask.sum()),
                    finite_legal_actions=finite_legal,
                    top1_action=top1,
                    top2_action=top2,
                    top1_q=top1_q,
                    top2_q=top2_q,
                    margin=margin,
                    eligible=eligible,
                )
                snapshots[context] = snapshot
                all_kyoku_keys.add((seed, key, k0_seat, kyoku_index))
                primary_count += 1
                total_primary_decisions += 1
                phase_counts[snapshot.phase] += 1
                shanten_counts[snapshot.shanten] += 1
                legal_action_counts[snapshot.legal_action_count] += 1
                action_counts[action_name(snapshot.action)] += 1
                if snapshot.action in finite_legal:
                    primary_behavior_legal_finite += 1
                if top1 is not None and snapshot.action == top1:
                    primary_greedy_agreement += 1
                if margin is not None:
                    gaps.append(margin)
                if eligible:
                    eligible_phase[snapshot.phase] += 1
                    eligible_shanten[snapshot.shanten] += 1
                    eligible_legal_counts[snapshot.legal_action_count] += 1
                    eligible_action_counts[action_name(snapshot.action)] += 1
                    eligible_gaps.append(float(margin))
            if primary_count <= 0:
                raise ValueError("K0 perspective has zero primary decisions")
            decisions_per_hanchan.append(primary_count)
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{path}: {exc}")

    def gap_stats(values: list[float]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)) if values else 0.0,
            "median": float(statistics.median(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }

    metrics = {
        "trainable_perspectives": len(decisions_per_hanchan),
        "total_training_rows": total_training_rows,
        "total_primary_decisions": total_primary_decisions,
        "auxiliary_kan_select_rows": auxiliary_kan_select_rows,
        "unique_decision_contexts": len(snapshots),
        "primary_decisions_per_hanchan": _quantiles(decisions_per_hanchan),
        "all_training_rows_behavior_action_legal_finite_rate": (
            behavior_legal_finite / total_training_rows if total_training_rows else 0.0
        ),
        "primary_behavior_action_legal_finite_rate": (
            primary_behavior_legal_finite / total_primary_decisions
            if total_primary_decisions
            else 0.0
        ),
        "parent_greedy_agreement_rate": (
            primary_greedy_agreement / total_primary_decisions
            if total_primary_decisions
            else 0.0
        ),
        "parent_q_gap": gap_stats(gaps),
        "eligible_parent_q_gap": gap_stats(eligible_gaps),
        "phase_counts": {key: int(value) for key, value in sorted(phase_counts.items())},
        "shanten_counts": _bucket(shanten_counts.elements()),
        "legal_action_count_distribution": _bucket(legal_action_counts.elements()),
        "action_mix": {key: int(value) for key, value in sorted(action_counts.items())},
        "eligible_phase_counts": {key: int(value) for key, value in sorted(eligible_phase.items())},
        "eligible_shanten_counts": _bucket(eligible_shanten.elements()),
        "eligible_legal_action_count_distribution": _bucket(eligible_legal_counts.elements()),
        "eligible_action_mix": {key: int(value) for key, value in sorted(eligible_action_counts.items())},
        "all_kyoku_keys": [list(key) for key in sorted(all_kyoku_keys)],
        "malformed": malformed,
        "passed": (
            not malformed
            and len(decisions_per_hanchan) == GAMES
            and total_training_rows > 0
            and total_primary_decisions > 0
            and len(snapshots) == total_primary_decisions
            and behavior_legal_finite == total_training_rows
            and primary_behavior_legal_finite == total_primary_decisions
        ),
    }
    return snapshots, metrics
