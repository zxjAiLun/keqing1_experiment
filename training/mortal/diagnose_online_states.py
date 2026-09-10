#!/usr/bin/env python3
"""Online-state diagnostic: how well does a student track the teacher on the
states it actually reaches in its own arena games?

Every earlier measurement used the relabel cache, i.e. the state distribution of
the *corpus* (external self-play and mixed pools).  This one replays the student's
own 1v3 arena logs, so it measures the on-policy state distribution -- the
quantity that decides whether covariate shift is a real problem for this line.

Method, per student-controlled seat in the existing logs:

- replay the hanchan with ``libriichi.state.PlayerState`` (the same state machine
  the arena uses), and snapshot ``encode_obs()`` at every state where the arena
  would consult the engine (``last_cans.can_act``);
- the student's own recorded action comes from the log, mapped to the 46-dim
  index space by ``d3_native_scene.expected_label`` (the canonical mapping from
  ``gameplay.rs``);
- run both the student and the teacher on the same replayed observation, and
  compare teacher-vs-student agreement, the teacher's valuation gap at the
  student's choice, and the student's rank inside the teacher's ordering.

No games are generated and nothing is trained.  Faithfulness is checked against
the log itself: replayed masks are compared with the recorded ``mask_bits``, and
re-inferred greedy actions are compared with the actions the log says each model
took (including the teacher's own seats, which validates the teacher path
independently).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal.d3_native_scene import expected_label, read_log_events  # noqa: E402
from training.mortal.four_player_native import _model_dimensions  # noqa: E402

# Canonical 46-dim action space (libriichi/src/dataset/gameplay.rs).
ACTION_TYPES = {
    **{index: "discard" for index in range(0, 37)},
    37: "riichi",
    **{index: "chi" for index in range(38, 41)},
    41: "pon",
    42: "kan",
    43: "hora",
    44: "ryukyoku",
    45: "pass",
}
DECISION_TYPES = ("dahai", "reach", "chi", "pon", "daiminkan", "kakan", "ankan", "hora", "ryukyoku")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online-state teacher-agreement diagnostic from existing arena logs")
    parser.add_argument("--run", action="append", required=True, help="LABEL=LOG_DIR (repeatable)")
    parser.add_argument("--student-name", required=True, help="alias in the logs that identifies the student model")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-name", default="ext_mortal")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--max-hanchans", type=int, default=0)
    parser.add_argument(
        "--teacher-integrity-hanchans",
        type=int,
        default=64,
        help="replay teacher seats for this many hanchans per run (faithfulness evidence only; 0 disables)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def _build(checkpoint: Path, device: torch.device, mortal_root: Path):
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from model import Brain, DQN  # noqa: PLC0415

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dimensions = _model_dimensions(state)
    brain = Brain(version=dimensions[0], conv_channels=dimensions[1], num_blocks=dimensions[2]).to(device)
    dqn = DQN(version=dimensions[0]).to(device)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    brain.eval()
    dqn.eval()
    return brain, dqn, dimensions


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    points = np.percentile(array, [50, 75, 90, 95, 99]).tolist()
    return {
        "mean": float(array.mean()),
        "p50": float(points[0]),
        "p90": float(points[2]),
        "p99": float(points[4]),
        "max": float(array.max()),
    }


def _mask_bits(mask: np.ndarray) -> int:
    bits = 0
    for index, value in enumerate(mask):
        if bool(value):
            bits |= 1 << index
    return bits


def _replay_hanchan(path: Path, student_seats: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Decision states per student seat, in log order."""
    from libriichi.state import PlayerState  # noqa: PLC0415

    events = read_log_events(path)
    collected: dict[int, list[dict[str, Any]]] = {seat: [] for seat in student_seats}
    for seat in student_seats:
        state = PlayerState(seat)
        kyoku = -1
        decision_index = 0
        for index, event in enumerate(events):
            if event.get("type") == "start_kyoku":
                kyoku += 1
                decision_index = 0
            state.update(json.dumps(event, ensure_ascii=False))
            if not state.last_cans.can_act:
                continue
            label = expected_label(events, index, seat, state)
            # version 4, at_kan_select=False: the main decision row of the 46-dim
            # action space (kan choices are still label 42, matching gameplay.rs).
            obs, mask = state.encode_obs(4, False)
            following = events[index + 1] if index + 1 < len(events) else None
            logged_meta = {
                "mask_bits": following.get("meta", {}).get("mask_bits"),
                "q_values": following.get("meta", {}).get("q_values"),
            } if following is not None and following.get("meta") else {}
            collected[seat].append(
                {
                    "obs": obs,
                    "mask": mask,
                    "label": label,
                    "kyoku": kyoku,
                    "decision_index": decision_index,
                    "next_type": (following or {}).get("type"),
                    "logged_mask_bits": logged_meta.get("mask_bits"),
                    "logged_q_values": logged_meta.get("q_values"),
                }
            )
            decision_index += 1
    return collected


def _infer(brain, dqn, records: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, np.ndarray]]:
    outputs: list[dict[str, np.ndarray]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        obs = torch.as_tensor(np.stack([row["obs"] for row in chunk]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.stack([row["mask"] for row in chunk]), dtype=torch.bool, device=device)
        with torch.inference_mode():
            q_out = dqn(brain(obs), mask).float()
        fill = torch.finfo(q_out.dtype).min
        masked = q_out.masked_fill(~mask, fill)
        outputs.append(
            {
                "q": q_out.cpu().numpy(),
                "argmax": masked.argmax(dim=-1).cpu().numpy(),
            }
        )
    if not outputs:
        return [{"q": np.zeros((0, 46), dtype=np.float32), "argmax": np.zeros(0, dtype=np.int64)}]
    return [
        {
            "q": np.concatenate([out["q"] for out in outputs]),
            "argmax": np.concatenate([out["argmax"] for out in outputs]),
        }
    ]


def _analyse(run_label: str, log_dir: Path, args: argparse.Namespace, student, teacher, device: torch.device) -> dict[str, Any]:
    paths = sorted(log_dir.glob("*.json.gz"))
    if int(args.max_hanchans) > 0:
        paths = paths[: int(args.max_hanchans)]

    counters = collections.Counter()
    rank_hist = collections.Counter()
    verified_rank_hist = collections.Counter()
    gap_values: list[float] = []
    verified_gap_values: list[float] = []
    gap_actual_values: list[float] = []
    per_seed: dict[int, dict[str, float]] = collections.defaultdict(lambda: {"n": 0.0, "agree": 0.0, "high_gap": 0.0})
    student_action_types = collections.Counter()
    teacher_action_types = collections.Counter()
    agree_by_action_type = collections.defaultdict(lambda: [0, 0])
    by_kyoku: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    by_phase: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])

    for hanchan_index, path in enumerate(paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        names = list(first.get("names") or [])
        student_seats = [seat for seat, name in enumerate(names) if name == args.student_name]
        teacher_seats = [seat for seat, name in enumerate(names) if name == args.teacher_name]
        if not student_seats:
            counters["hanchans_without_student_seat"] += 1
            continue
        if hanchan_index >= int(args.teacher_integrity_hanchans):
            teacher_seats = []
        seed = int(list(first.get("seed") or [0])[0])
        counters["hanchans"] += 1

        collected = _replay_hanchan(path, student_seats + [seat for seat in teacher_seats if seat not in student_seats])
        for seat, records in collected.items():
            if not records:
                continue
            outputs = _infer(student[0], student[1], records, device, int(args.batch_size))[0]
            teacher_outputs = _infer(teacher[0], teacher[1], records, device, int(args.batch_size))[0]
            is_student_seat = seat in student_seats
            is_teacher_seat = seat in teacher_seats
            for index, row in enumerate(records):
                mask = row["mask"].astype(bool)
                student_choice = int(outputs["argmax"][index])
                teacher_choice = int(teacher_outputs["argmax"][index])
                # integrity: replayed mask vs recorded mask_bits
                mask_verified = True
                if row["logged_mask_bits"] is not None:
                    counters["mask_checked"] += 1
                    if _mask_bits(mask) != int(row["logged_mask_bits"]):
                        counters["mask_mismatch"] += 1
                        mask_verified = False
                if is_student_seat:
                    counters["student_decisions"] += 1
                    if row["label"] is None:
                        counters["student_label_none"] += 1
                    else:
                        counters["student_action_matches_reinfer"] += int(student_choice == int(row["label"]))
                    # teacher evaluation of the student's choice
                    legal_q = np.where(mask, teacher_outputs["q"][index], -np.inf)
                    top1 = float(legal_q.max())
                    chosen_q = float(legal_q[student_choice])
                    teacher_rank = 1 + int((legal_q > chosen_q).sum())
                    gap = top1 - chosen_q
                    gap_values.append(gap)
                    rank_hist[teacher_rank if teacher_rank <= 4 else 5] += 1
                    agree = int(teacher_choice == student_choice)
                    counters["student_teacher_agree"] += agree
                    if mask_verified:
                        counters["student_verified_decisions"] += 1
                        counters["student_verified_agree"] += agree
                        verified_gap_values.append(gap)
                        verified_rank_hist[teacher_rank if teacher_rank <= 4 else 5] += 1
                    high_gap = int(agree == 0 and gap > 2.05)
                    counters["student_high_gap_error"] += high_gap
                    counters["student_gap_gt_1"] += int(gap > 1.0)
                    student_action_types[ACTION_TYPES.get(int(row["label"]) if row["label"] is not None else 45, "unknown")] += 1
                    teacher_action_types[ACTION_TYPES.get(teacher_choice, "unknown")] += 1
                    bucket = agree_by_action_type[ACTION_TYPES.get(teacher_choice, "unknown")]
                    bucket[0] += agree
                    bucket[1] += 1
                    if row["label"] is not None:
                        legal_actual = np.where(mask, teacher_outputs["q"][index], -np.inf)
                        gap_actual_values.append(float(legal_actual.max() - legal_actual[int(row["label"])]))
                    entry = per_seed[seed]
                    entry["n"] += 1
                    entry["agree"] += agree
                    entry["high_gap"] += high_gap
                    kyoku = int(row["kyoku"])
                    by_kyoku[kyoku][0] += agree
                    by_kyoku[kyoku][1] += 1
                    decision_index = int(row["decision_index"])
                    phase = "early(1-4)" if decision_index < 4 else ("mid(5-10)" if decision_index < 10 else "late(11+)")
                    by_phase[phase][0] += agree
                    by_phase[phase][1] += 1
                if is_teacher_seat:
                    counters["teacher_decisions"] += 1
                    if row["label"] is not None:
                        counters["teacher_action_matches_reinfer"] += int(teacher_choice == int(row["label"]))

    seeds = sorted(per_seed)
    seed_agreement = [per_seed[seed]["agree"] / per_seed[seed]["n"] for seed in seeds if per_seed[seed]["n"]]
    seed_high_gap = [per_seed[seed]["high_gap"] / per_seed[seed]["n"] for seed in seeds if per_seed[seed]["n"]]
    rng = np.random.default_rng(int(args.bootstrap_seed))

    def bootstrap(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {}
        draws = [float(array[rng.integers(0, array.size, size=array.size)].mean()) for _ in range(int(args.bootstrap_reps))]
        return {
            "seeds": int(array.size),
            "seed_mean": float(array.mean()),
            "seed_cluster_bootstrap_ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        }

    decisions = counters["student_decisions"]
    return {
        "run": run_label,
        "log_dir": str(log_dir),
        "hanchans": counters["hanchans"],
        "student_seats_decisions": decisions,
        "teacher_seats_decisions": counters["teacher_decisions"],
        "offline_comparison_anchor": {
            "note": "compare with the offline holdout figures recorded for the same student",
            "offline_full_holdout_agreement": 0.9046830135144444 if args.student_name == "student50k" else None,
        },
        "agreement": {
            "teacher_argmax_vs_student_argmax": (counters["student_teacher_agree"] / decisions) if decisions else None,
            "counts": {"agree": counters["student_teacher_agree"], "decisions": decisions},
            "mask_verified_only": (
                counters["student_verified_agree"] / counters["student_verified_decisions"]
                if counters["student_verified_decisions"]
                else None
            ),
            "mask_verified_decisions": counters["student_verified_decisions"],
        },
        "teacher_gap_at_student_choice": _quantiles(gap_values),
        "teacher_gap_at_student_choice_mask_verified_only": _quantiles(verified_gap_values),
        "student_choice_rank_mask_verified_only": {
            "counts": {str(rank): verified_rank_hist[rank] for rank in (1, 2, 3, 4, 5)},
            "share_rank_ge3": (
                sum(verified_rank_hist[r] for r in (3, 4, 5)) / counters["student_verified_decisions"]
                if counters["student_verified_decisions"]
                else None
            ),
        },
        "teacher_gap_at_student_actual_action": _quantiles(gap_actual_values),
        "student_choice_rank_in_teacher_order": {
            "counts": {str(rank): rank_hist[rank] for rank in (1, 2, 3, 4, 5)},
            "share_rank_ge3": (sum(rank_hist[r] for r in (3, 4, 5)) / decisions) if decisions else None,
        },
        "high_gap_error_definition": "disagreement with teacher gap > 2.05 (the offline median teacher top-2 gap)",
        "high_gap_error_rate": (counters["student_high_gap_error"] / decisions) if decisions else None,
        "gap_gt_1_rate": (counters["student_gap_gt_1"] / decisions) if decisions else None,
        "agreement_by_seed_cluster": bootstrap(seed_agreement),
        "high_gap_error_rate_by_seed_cluster": bootstrap(seed_high_gap),
        "agreement_by_teacher_action_type": {
            name: {"agreement": bucket[0] / bucket[1], "n": bucket[1]}
            for name, bucket in sorted(agree_by_action_type.items())
            if bucket[1]
        },
        "student_actual_action_types": dict(sorted(student_action_types.items())),
        "teacher_argmax_action_types": dict(sorted(teacher_action_types.items())),
        "agreement_by_kyoku": {
            str(kyoku): {"agreement": value[0] / value[1], "n": int(value[1])}
            for kyoku, value in sorted(by_kyoku.items())
            if value[1]
        },
        "agreement_by_in_round_phase": {
            phase: {"agreement": value[0] / value[1], "n": int(value[1])}
            for phase, value in sorted(by_phase.items())
            if value[1]
        },
        "integrity": {
            "replayed_mask_checked": counters["mask_checked"],
            "replayed_mask_mismatches": counters["mask_mismatch"],
            "replayed_mask_mismatch_rate": (
                counters["mask_mismatch"] / counters["mask_checked"] if counters["mask_checked"] else None
            ),
            "student_actions_with_label": decisions - counters["student_label_none"],
            "student_re_infer_matches_recorded_action": counters["student_action_matches_reinfer"],
            "student_re_infer_match_rate": (
                counters["student_action_matches_reinfer"] / max(1, decisions - counters["student_label_none"])
            ),
            "teacher_re_infer_matches_recorded_action": counters["teacher_action_matches_reinfer"],
            "teacher_re_infer_match_rate": (
                counters["teacher_action_matches_reinfer"] / max(1, counters["teacher_decisions"])
            ),
            "student_label_none": counters["student_label_none"],
        },
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=DIR, got {spec}")
        label, _, directory = spec.partition("=")
        runs.append((label.strip(), Path(directory.strip())))

    student = _build(args.student_checkpoint, device, args.mortal_root)
    teacher = _build(args.teacher_checkpoint, device, args.mortal_root)
    print(f"student dims={student[2]} teacher dims={teacher[2]} device={args.device}", flush=True)

    report = {
        "schema": "keqing.mortal.online_state_diagnostic.v1",
        "student_checkpoint": str(args.student_checkpoint),
        "student_name_in_logs": args.student_name,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_name_in_logs": args.teacher_name,
        "runs": {},
    }
    for label, directory in runs:
        print(f"[{label}] replaying {directory} ...", flush=True)
        report["runs"][label] = _analyse(label, directory, args, student, teacher, device)
        entry = report["runs"][label]
        print(
            f"[{label}] hanchans={entry['hanchans']} student_decisions={entry['student_seats_decisions']} "
            f"agreement={entry['agreement']['teacher_argmax_vs_student_argmax']:.4f} "
            f"high_gap_error={entry['high_gap_error_rate']:.4f} "
            f"re_infer_match={entry['integrity']['student_re_infer_match_rate']:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
