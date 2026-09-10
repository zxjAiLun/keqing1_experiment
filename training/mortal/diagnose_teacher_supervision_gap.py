#!/usr/bin/env python3
"""One-off fixed-sample diagnostic: what does the student's residual error cost?

The relabel cache stores the teacher's full Q vector per row, but the phase-1
student was trained only on the teacher's argmax hard label.  So the cached Q
is supervision we already paid for and did not use.  This script uses it to
answer a single question:

    when the student disagrees with the teacher, is it picking a near-tie
    alternative, or something the teacher scored materially worse?

Method (fixed sample, no training, no relabel):

- a deterministic sample of ~33k holdout rows per pool (evenly spaced shards,
  stride-sampled rows), all recorded in the output for reproducibility
- states with fewer than two legal actions are excluded, so agreement is only
  measured where a choice exists
- for each disagreement, the student's chosen action is ranked inside the
  teacher's ordering, and ``loss`` = teacher max Q - teacher Q at the student's
  choice
- gap-stratified error rates use within-pool quantiles of the teacher's top-2
  gap, and the severity split uses the within-pool median gap as its scale, so
  no raw Q difference is compared across pools

``loss`` is a proxy for the teacher's own valuation of the missed action.  It is
NOT a counterfactual gain, and it is not a measured win-rate cost.  Each NPZ is
decompressed once per shard and then sliced in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.four_player_native import _model_dimensions  # noqa: E402

_TOP_KEYS = ("obs", "mask", "teacher_q", "teacher_action", "pool")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-sample teacher-supervision-gap diagnostic")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards-per-pool", type=int, default=16)
    parser.add_argument("--rows-per-pool", type=int, default=33_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    return parser.parse_args()


def _build_models(*, version: int, conv_channels: int, num_blocks: int, device: torch.device, mortal_root: Path):
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from model import Brain, DQN  # noqa: PLC0415

    return Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).to(device), DQN(version=version).to(device)


def _shard_pools(dataset_dir: Path) -> dict[Path, set[int]]:
    """Which pools each holdout shard contains (only the small pool array is read)."""
    pools: dict[Path, set[int]] = {}
    for path in sorted(dataset_dir.glob("holdout_*.npz")):
        with np.load(path) as shard:
            pools[path] = {int(value) for value in np.unique(shard["pool"])}
    return pools


def _evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if count >= len(items):
        return list(items)
    index = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(index.tolist()))]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    points = np.percentile(values, [50, 75, 90, 95, 99]).tolist()
    return {
        "mean": float(values.mean()),
        "p50": float(points[0]),
        "p75": float(points[1]),
        "p90": float(points[2]),
        "p95": float(points[3]),
        "p99": float(points[4]),
        "max": float(values.max()),
    }


def _pool_report(rows: dict[str, np.ndarray], pool_name: str) -> dict[str, object]:
    """rows keys: student_rank (int), loss (float), gap (float), legal (int), agree (bool)"""
    student_rank = rows["student_rank"]
    loss = rows["loss"]
    gap = rows["gap"]
    agree = rows["agree"]
    errors = ~agree

    gap_edges = np.percentile(gap, [25, 50, 75]) if gap.size else np.array([])
    strata: list[dict[str, object]] = []
    if gap.size:
        bounds = [-np.inf, gap_edges[0], gap_edges[1], gap_edges[2], np.inf]
        for index in range(4):
            if index < 3:
                selected = (gap > bounds[index]) & (gap <= bounds[index + 1])
            else:
                selected = gap > bounds[index]
            count = int(selected.sum())
            strata.append(
                {
                    "gap_stratum": f"Q{index + 1}",
                    "gap_range": [None if np.isneginf(bounds[index]) else float(bounds[index]),
                                  None if np.isinf(bounds[index + 1]) else float(bounds[index + 1])],
                    "states": count,
                    "student_errors": int(errors[selected].sum()),
                    "student_error_rate": (float(errors[selected].mean()) if count else None),
                }
            )

    median_gap = float(np.median(gap)) if gap.size else None
    if median_gap is None:
        severity = {}
    else:
        near_tie = errors & (student_rank == 2) & (loss <= median_gap)
        material = errors & ~near_tie
        severity = {
            "scale": "within-pool median teacher top-2 gap",
            "median_top2_gap": median_gap,
            "near_tie_error_states": int(near_tie.sum()),
            "near_tie_error_share_of_errors": float(near_tie.sum() / max(1, errors.sum())),
            "material_error_states": int(material.sum()),
            "material_error_share_of_errors": float(material.sum() / max(1, errors.sum())),
            "material_error_note": "rank >= 3, or teacher's 2nd choice lost by more than the median top-2 gap",
            "errors_beyond_teacher_second_choice": int((errors & (student_rank >= 3)).sum()),
        }

    return {
        "pool": pool_name,
        "rows_analyzed": int(student_rank.size),
        "teacher_behavior_agreement": float(agree.mean()) if agree.size else None,
        "student_disagreement_count": int(errors.sum()),
        "student_choice_rank_in_teacher_order": {
            "counts": {str(rank): int((student_rank == rank).sum()) for rank in (1, 2, 3, 4) }
            | {"5+": int((student_rank >= 5).sum())},
            "note": "rank 1 == agreement; ties inside the teacher ordering resolve in the student's favour",
        },
        "loss_teacher_top1_minus_student_choice": _quantiles(loss[errors]) if errors.any() else {},
        "loss_all_rows": _quantiles(loss),
        "teacher_top2_gap": _quantiles(gap),
        "teacher_legal_q_std_per_state": _quantiles(rows["legal_std"]),
        "error_rate_by_teacher_top2_gap_stratum": strata,
        "error_severity": severity,
    }


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.resolve()
    device = torch.device(args.device)

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    pool_names = list(manifest.get("pools", {}))

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    version, conv_channels, num_blocks = _model_dimensions(state)
    mortal, dqn = _build_models(
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
        device=device, mortal_root=args.mortal_root,
    )
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    mortal.eval()
    dqn.eval()

    shard_pools = _shard_pools(dataset_dir)
    selected: dict[str, list[Path]] = {}
    for index, name in enumerate(pool_names):
        shards = [path for path in shard_pools if index in shard_pools[path]]
        if not shards:
            raise RuntimeError(f"no holdout shard contains pool {name}")
        selected[name] = _evenly_spaced(sorted(shards), int(args.shards_per_pool))

    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    sample_record: dict[str, object] = {}
    for index, name in enumerate(pool_names):
        buckets = {key: [] for key in ("student_rank", "loss", "gap", "agree", "legal_std")}
        seen = 0
        taken = 0
        quota = int(args.rows_per_pool)
        shard_summary = []
        for path in selected[name]:
            if taken >= quota:
                break
            with np.load(path) as shard:
                # Decompress each needed array exactly once, then slice in memory.
                obs = np.asarray(shard["obs"])
                mask = np.asarray(shard["mask"])
                teacher_q = np.asarray(shard["teacher_q"], dtype=np.float32)
                teacher_action = np.asarray(shard["teacher_action"], dtype=np.int64)
                pool_ids = np.asarray(shard["pool"])
            pool_rows = np.nonzero(pool_ids == index)[0]
            stride = max(1, int(round(pool_rows.size / max(1, quota - taken))))
            picked = pool_rows[::stride]
            picked = picked[: quota - taken]
            if picked.size == 0:
                continue
            shard_summary.append({"shard": path.name, "pool_rows": int(pool_rows.size), "taken": int(picked.size)})
            taken += int(picked.size)
            seen += int(pool_rows.size)

            legal = mask[picked]
            legal_count = legal.sum(axis=1)
            keep = legal_count >= 2  # agreement/loss only where a choice exists
            picked = picked[keep]
            legal = legal[keep]
            if picked.size == 0:
                continue
            q_teacher = teacher_q[picked]
            target = teacher_action[picked]
            rows = []
            student_choice = []
            for start in range(0, picked.size, int(args.batch_size)):
                stop = min(start + int(args.batch_size), picked.size)
                obs_t = torch.as_tensor(obs[picked[start:stop]], dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(legal[start:stop], dtype=torch.bool, device=device)
                with torch.inference_mode():
                    q_student = dqn(mortal(obs_t), mask_t).float().cpu().numpy()
                student_choice.append(q_student.argmax(axis=1))
            student_choice = np.concatenate(student_choice)
            rows_t = q_teacher
            legal_q = np.where(legal, rows_t, -np.inf)
            top1 = legal_q.max(axis=1)
            chosen_teacher_q = legal_q[np.arange(picked.size), student_choice]
            # rank of the student's action inside the teacher's ordering
            student_rank = 1 + (legal_q > chosen_teacher_q[:, None]).sum(axis=1)
            top2 = np.sort(legal_q, axis=1)[:, -2]
            legal_std = np.where(legal, rows_t, np.nan)
            legal_std = np.nanstd(legal_std, axis=1)

            buckets["student_rank"].append(student_rank.astype(np.int32))
            buckets["loss"].append((top1 - chosen_teacher_q).astype(np.float32))
            buckets["gap"].append((top1 - top2).astype(np.float32))
            buckets["agree"].append((student_choice == target).astype(bool))
            buckets["legal_std"].append(legal_std.astype(np.float32))
        collected[name] = {key: np.concatenate(values) if values else np.empty(0) for key, values in buckets.items()}
        sample_record[name] = {
            "shards": shard_summary,
            "rows_with_ge2_legal_actions": int(collected[name]["student_rank"].size),
            "pool_rows_seen": seen,
        }
        print(f"[{name}] sampled {collected[name]['student_rank'].size} rows with >=2 legal actions", flush=True)

    overall = {key: np.concatenate([collected[name][key] for name in pool_names]) for key in ("student_rank", "loss", "gap", "agree", "legal_std")}
    report = {
        "schema": "keqing.mortal.teacher_supervision_gap.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": int(state["steps"]),
        "dataset": str(dataset_dir),
        "sample": {
            "method": "evenly spaced holdout shards per pool, stride-sampled rows, >=2 legal actions only",
            "shards_per_pool": int(args.shards_per_pool),
            "rows_per_pool_quota": int(args.rows_per_pool),
            "detailed": sample_record,
        },
        "caveat": "loss is the teacher's own Q difference for the missed action, not a counterfactual gain; raw Q differences are only compared within a pool",
        "overall": _pool_report(overall, "overall"),
        "pools": {name: _pool_report(collected[name], name) for name in pool_names},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(key: str) -> str:
        entry = report["overall"]
        if key == "agreement":
            return f"agreement (>=2 legal)      : {entry['teacher_behavior_agreement']:.4f}"
        if key == "rank":
            counts = entry["student_choice_rank_in_teacher_order"]["counts"]
            return "student choice rank        : " + " ".join(f"{k}:{v}" for k, v in counts.items())
        if key == "loss":
            return "loss on errors p50/p90/p99 : " + "/".join(f"{entry['loss_teacher_top1_minus_student_choice'][k]:.3f}" for k in ("p50", "p90", "p99"))
        if key == "severity":
            sev = entry["error_severity"]
            return (f"error severity             : near-tie {sev['near_tie_error_states']} "
                    f"({sev['near_tie_error_share_of_errors']:.1%}) vs material {sev['material_error_states']} "
                    f"({sev['material_error_share_of_errors']:.1%}); rank>=3 {sev['errors_beyond_teacher_second_choice']}")
        return ""

    print("\n--- one-page summary (overall) ---")
    for key in ("agreement", "rank", "loss", "severity"):
        print(line(key))
    print("\n--- agreement / error rate by teacher top-2 gap stratum (within pool) ---")
    for name in pool_names:
        entry = report["pools"][name]
        strata = entry["error_rate_by_teacher_top2_gap_stratum"]
        rates = " ".join(f"{row['gap_stratum']}:{row['student_error_rate']:.3f}" for row in strata)
        print(f"{name:<4} agree={entry['teacher_behavior_agreement']:.4f} errors={entry['student_disagreement_count']:<6} {rates}")
    print("\n--- loss quantiles on errors (within pool) ---")
    for name in pool_names:
        q = report["pools"][name]["loss_teacher_top1_minus_student_choice"]
        print(f"{name:<4} p50={q['p50']:.3f} p75={q['p75']:.3f} p90={q['p90']:.3f} p99={q['p99']:.3f} max={q['max']:.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
