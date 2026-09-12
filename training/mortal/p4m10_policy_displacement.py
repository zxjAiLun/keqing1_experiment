#!/usr/bin/env python
"""P4-M10 fixed-panel policy displacement check (READ-ONLY, inference only).

Owner ruling (2026-09-12, second review): P4-M10 stays stopped and not promoted.
The only authorised follow-up is the previously scoped **fixed-panel displacement
check**, answering one question:

    "did the four updates merely perturb the policy, or did they visibly change
     deployment actions?"

Scope is exactly what was authorised and nothing more:

* ONE frozen panel of decisions (declared below), evaluated with every
  checkpoint so all comparisons are on identical states and identical masks;
* metrics: argmax **flip rate** (deployment semantics), **TV**, and **KL** in
  both directions (distribution semantics);
* pairs: parent -> C1, each adjacent cycle pair, and the total parent -> C4;
* NO action bucketing, NO checkpoint selection, NO training, NO arena games,
  NO budget decision (whether to continue is the owner's, not this script's).

This script never writes to a training artifact and never runs a backward pass.

Metric roles (owner ruling 2026-09-12, third review)
---------------------------------------------------
* **The result is flip / TV / KL only.** Those are the quantities that may be used
  to characterise how far the policy moved.
* **Raw action-score drift is TELEMETRY and must not be used as evidence.** The DQN
  is dueling: ``q_a = v + a_a - mean(a_legal)``, so adding a constant to every legal
  action of a state leaves both the softmax and the argmax unchanged
  (``third_party/Mortal/mortal/model.py:221``). Raw scores can move a lot while the
  policy barely moves. ``|dq|`` is still measured and written out; it just is not a
  policy-displacement argument, and no value/advantage decomposition is added.

The near-tie split reported below is a **descriptive statistic only**: the
threshold is the panel-wide max same-weight cross-batching deviation, which is not
a per-state noise bound, so it must not be used to declare a flip meaningless.

Panel substitution (registered, not rerun)
-----------------------------------------
The previously discussed panel was the P4-M4 teacher-supervision-gap fixed sample
(its fixed variables exclude single-legal-action states). The panel actually used
here is P4-M10 **cycle 1**: the parent's own on-policy sampled states. It answers
how far the updates pushed the policy inside the region the starting policy
visited, which is what this check was scoped to. It is NOT the P4-M4 panel, NOT an
independent holdout (cycle-1 states participated in C1's training), and NOT C4's
own visitation distribution.

Measurement honesty
-------------------
A batched forward is not bit-identical to the engine's own per-call batch, so
three things are measured rather than assumed:

* ``panel_integrity``  - the parent pass must reproduce the collection-time
  ``q_legal`` and ``logprob`` within the project's pre-declared P4-M10 tolerance;
* ``batched_forward_is_deterministic`` - two passes with identical weights and
  identical chunk size must be bit-identical, so the reported flip set is a
  deterministic consequence of the weight pair, not run-to-run noise;
* ``batching_sensitivity`` - the whole metric set is recomputed at other chunk
  sizes, because a flip decided by a near-tied top-2 margin can depend on it.

Only after those three does a flip count mean anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "training" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training" / "mortal"))

from p4m9_probe_onpolicy import ACTION_SPACE, _load_state, bits_to_mask
from p4m10_onpolicy_pg import RECOMPUTE_TOLERANCE, _build_modules

# The batched forward used here has a different reduction order than the
# collection-time forward, so q is only reproducible to float32 precision on a
# scale of |q| ~ 940.  The log-prob tolerance is the project's own pre-declared
# P4-M10 recompute tolerance, which P4-M10 cycle 1 itself accepted at 1.94e-03.
Q_RELATIVE_TOLERANCE = 1e-4

RUN_DIR = REPO_ROOT / "artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256"
PARENT_PATH = (
    REPO_ROOT / "artifacts/experiments/student_policy_v1/student_formal_25k/student_step_050000.pth"
)
OUT_PATH = REPO_ROOT / "artifacts/eval/p4m10_comparisons/policy_displacement.json"
MORTAL_ROOT = REPO_ROOT / "third_party/Mortal"

# The panel is P4-M10 cycle 1: 256 hanchans collected with the PARENT weights
# against 3 x ext_mortal, so it is the parent's own on-policy state distribution
# at the start of the loop.  It is used frozen for every checkpoint.
PANEL_CYCLE = 1

PAIRS = [
    ("parent", "C1"),
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("parent", "C4"),
]

FLIP_MARGIN_NOTE = (
    "descriptive only: the threshold is the panel-wide max same-weight "
    "cross-batching deviation, not a per-state noise bound; it must not be used to "
    "declare a flip meaningless"
)

METRIC_ROLES = {
    "primary_result": ["flip_rate_argmax", "tv_mean", "kl_parent_C4"],
    "telemetry_only": ["q_delta_mean_abs", "q_delta_max_abs"],
    "rule": (
        "Only flip / TV / KL may be used to characterise the policy displacement. The "
        "raw action-score drift is telemetry: the DQN is dueling "
        "(q_a = v + a_a - mean(a_legal)), so adding a constant to every legal action of "
        "a state leaves both the softmax and the argmax unchanged "
        "(third_party/Mortal/mortal/model.py:221). Raw scores can move a lot while the "
        "policy barely moves, so the two are different quantities."
    ),
}

PANEL_SUBSTITUTION = {
    "previously_discussed": (
        "the P4-M4 teacher-supervision-gap fixed sample panel, whose fixed variables "
        "exclude single-legal-action states from agreement/loss (a fixed >=2-legal panel)"
    ),
    "actually_used": (
        "P4-M10 cycle 1: exploration_allowed=True decisions sampled with the PARENT "
        "weights against 3x ext_mortal (T=1, eps=1, top_p=1)"
    ),
    "why_acceptable": (
        "It answers how far the four updates pushed the policy inside the region the "
        "starting policy actually visited, which is the question this check was scoped to."
    ),
    "boundaries": [
        "NOT the P4-M4 fixed panel that was originally discussed",
        "cycle-1 states participated in C1's training, so this is NOT an independent holdout",
        "NOT C4's own visitation distribution, and not a generalisation claim over all states",
    ],
    "rerun_required": False,
    "rerun_reason": (
        "The goal was never generalisation or strength; registering the substitution and "
        "its boundaries is sufficient and no second panel is forwarded."
    ),
}

INTERPRETATION_BOUNDARIES = {
    "near_ties": (
        "The margin split is a DESCRIPTIVE statistic: the threshold is the max "
        "same-weight cross-batching deviation over the panel, NOT a per-state noise "
        "bound, so it must not be used to declare a flip meaningless and the result must "
        "NOT be summarised as 'PG only changed near-ties' or 'no strongly preferred "
        "decision was rewritten'. The supported statement is: the flip rate is stable "
        "across batch sizes; this check does not establish the decision importance of "
        "those flips."
    ),
    "no_ratio_between_flip_and_sampling_disagreement": (
        "sample_vs_greedy_disagreement_rate is the probability that the parent's own T=1 "
        "random sample deviates from the parent's own argmax; flip_rate is the rate at "
        "which the argmax differs between two models. Different quantities: do not divide "
        "them and do not read 'deployment moved only a tenth of the exploration amplitude'."
    ),
    "not_a_budget_rule": (
        "This check answers only the size of the movement. It does not answer 'small "
        "movement means continue' or 'large movement means close'."
    ),
}

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260910
CHUNK = 2048


def legal_action_count(record: dict) -> int:
    return sum(bits_to_mask(int(record["mask_bits"])))


def load_panel(cycle_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Every decision the model actually sampled in that cycle (explore=True).

    Returns the records plus a read-only memory map of the raw fp32 observations.
    The obs file is one contiguous (n_total, 1012, 34) array, so the panel is a
    row selection over it and every checkpoint sees byte-identical inputs.
    """
    records_path = cycle_dir / "probe_records.jsonl"
    obs_path = cycle_dir / "obs_fp32.bin"

    all_records = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_records.append(json.loads(line))
    if not all_records:
        raise RuntimeError(f"no records in {records_path}")

    bytes_per_obs = int(all_records[0]["obs_bytes"])
    shape = tuple(int(value) for value in all_records[0]["obs_shape"])
    if any(int(r["obs_bytes"]) != bytes_per_obs for r in all_records):
        raise RuntimeError("obs_bytes is not uniform; panel indexing is unsafe")

    obs_map = np.memmap(obs_path, dtype=np.float32, mode="r",
                        shape=(len(all_records), *shape))

    records = [r for r in all_records if bool(r.get("explore", False))]
    if not records:
        raise RuntimeError(f"no policy decisions in {records_path}")
    rows = np.array([int(r["obs_off"]) // bytes_per_obs for r in records], dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("panel obs rows are not unique")
    return records, obs_map[rows]


def checkpoint_paths() -> dict[str, Path]:
    paths = {"parent": PARENT_PATH}
    for cycle in range(1, 5):
        paths[f"C{cycle}"] = RUN_DIR / f"C{cycle}_eval_weights.pth"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")
    return paths


def q_table_for(
    path: Path,
    *,
    records: list[dict],
    panel_obs: np.ndarray,
    device: torch.device,
    chunk_size: int = CHUNK,
) -> np.ndarray:
    """Full ACTION_SPACE-wide q for every panel decision, illegal actions = -inf."""
    version, conv_channels, num_blocks, state = _load_state(path)
    _, brain, dqn = _build_modules(
        state,
        mortal_root=MORTAL_ROOT,
        device=device,
        version=int(version),
        conv_channels=int(conv_channels),
        num_blocks=int(num_blocks),
    )
    out = np.empty((len(records), ACTION_SPACE), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(records), chunk_size):
            stop = min(start + chunk_size, len(records))
            chunk = records[start:stop]
            obs = torch.as_tensor(
                np.ascontiguousarray(panel_obs[start:stop]), device=device
            )
            mask = torch.as_tensor(
                np.array([bits_to_mask(int(r["mask_bits"])) for r in chunk]), device=device
            )
            q_out = dqn(brain(obs), mask).masked_fill(~mask, -torch.inf)
            out[start:stop] = q_out.detach().to("cpu").numpy()
    del brain, dqn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def log_softmax64(q: np.ndarray) -> np.ndarray:
    """log softmax over float64, illegal entries (-inf) stay at -inf."""
    finite = np.isfinite(q)
    shifted = np.where(finite, q - np.max(np.where(finite, q, -np.inf), axis=1, keepdims=True), -np.inf)
    exponential = np.where(finite, np.exp(shifted), 0.0)
    return np.where(finite, shifted - np.log(exponential.sum(axis=1, keepdims=True)), -np.inf)


def top_two_margin(q: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Gap between the best and second-best legal action score (0 if <2 legal)."""
    masked = np.where(legal, q.astype(np.float64), -np.inf)
    partitioned = np.partition(masked, -2, axis=1)
    margin = partitioned[:, -1] - partitioned[:, -2]
    single = np.isfinite(masked).sum(axis=1) < 2
    return np.where(single, 0.0, margin)


def legal_mask_from_records(records: list[dict]) -> np.ndarray:
    """The (N, ACTION_SPACE) legal mask — a property of the STATE, not of a model.

    Taking it from the records (rather than from which entries of a particular
    forward happened to be finite) is what makes "illegal actions never enter a
    metric" true by construction instead of by luck.
    """
    return np.array(
        [bits_to_mask(int(r["mask_bits"])) for r in records], dtype=bool
    )


def pair_metrics(
    q_a: np.ndarray, q_b: np.ndarray, legal: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-decision flip / TV / KL over the shared legal action set."""
    violations = int((~np.isfinite(q_a[legal])).sum() + (~np.isfinite(q_b[legal])).sum())
    if violations:
        raise ValueError(
            f"{violations} legal action scores are not finite; a metric computed "
            "over a partially-finite row would silently misreport the displacement"
        )
    log_pa = log_softmax64(np.where(legal, q_a, -np.inf).astype(np.float64))
    log_pb = log_softmax64(np.where(legal, q_b, -np.inf).astype(np.float64))
    pa = np.exp(log_pa)
    pb = np.exp(log_pb)

    argmax_a = np.argmax(np.where(legal, q_a, -np.inf), axis=1)
    argmax_b = np.argmax(np.where(legal, q_b, -np.inf), axis=1)

    tv = 0.5 * np.where(legal, np.abs(pa - pb), 0.0).sum(axis=1)
    # KL(A||B) is finite wherever B puts mass on everything A does; the legal set
    # is shared by construction.  Illegal entries are flattened to 0 BEFORE the
    # subtraction, otherwise -inf - -inf evaluates to nan there (harmless after
    # masking, but it emits a spurious warning).
    la = np.where(legal, log_pa, 0.0)
    lb = np.where(legal, log_pb, 0.0)
    log_ratio = la - lb
    kl_ab = (pa * log_ratio).sum(axis=1)
    kl_ba = (pb * -log_ratio).sum(axis=1)
    # Illegal entries are flattened before subtracting (-inf - -inf is nan) and
    # then marked nan so the caller's isfinite filter drops them.
    q_delta = np.where(
        legal, np.abs(np.where(legal, q_a, 0.0) - np.where(legal, q_b, 0.0)), np.nan
    )
    return {
        "flip": (argmax_a != argmax_b).astype(np.float64),
        "tv": tv,
        "kl_ab": np.maximum(kl_ab, 0.0),
        "kl_ba": np.maximum(kl_ba, 0.0),
        "max_prob_a": np.exp(log_pa).max(axis=1),
        "margin_a": top_two_margin(q_a, legal),
        "margin_b": top_two_margin(q_b, legal),
        "q_delta": q_delta,
        "argmax_a": argmax_a,
    }


def seed_clustered_ci(
    values: np.ndarray, seeds: np.ndarray, *, reps: int, seed: int
) -> list[float]:
    """Bootstrap over seed clusters (the project's resampling unit)."""
    unique, inverse = np.unique(seeds, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(reps, len(unique)))
    totals = sums[draws].sum(axis=1)
    sizes = counts[draws].sum(axis=1)
    estimates = totals / sizes
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]


def results_from_tables(
    tables: dict[str, np.ndarray],
    seeds: np.ndarray,
    q_noise_floor: float,
    legal: np.ndarray,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for a, b in PAIRS:
        metrics = pair_metrics(tables[a], tables[b], legal)
        flip_mask = metrics["flip"] > 0.5
        # DESCRIPTIVE SPLIT ONLY.  q_noise_floor is the panel-wide max
        # same-weight cross-batching deviation; it is NOT a per-state bound on
        # when a flip stops being meaningful, so this split may not be read as
        # "only near-ties changed" or "no strongly preferred decision changed".
        above_threshold = flip_mask & (
            np.minimum(metrics["margin_a"], metrics["margin_b"]) > q_noise_floor
        )
        finite_delta = metrics["q_delta"][np.isfinite(metrics["q_delta"])]
        entry: dict[str, object] = {
            "flip_rate": float(np.mean(metrics["flip"])),
            "flip_rate_ci95": seed_clustered_ci(
                metrics["flip"], seeds, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
            ),
            "tv_mean": float(np.mean(metrics["tv"])),
            "tv_ci95": seed_clustered_ci(
                metrics["tv"], seeds, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
            ),
            "kl_a_b": float(np.mean(metrics["kl_ab"])),
            "kl_a_b_ci95": seed_clustered_ci(
                metrics["kl_ab"], seeds, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
            ),
            "kl_b_a": float(np.mean(metrics["kl_ba"])),
            "kl_b_a_ci95": seed_clustered_ci(
                metrics["kl_ba"], seeds, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
            ),
            "flip_count": int(metrics["flip"].sum()),
            "decisions": len(seeds),
            "tv_max": float(np.max(metrics["tv"])),
            "flips_margin_above_cross_batch_deviation": int(above_threshold.sum()),
            "flips_margin_above_cross_batch_deviation_rate": float(
                np.mean(above_threshold)
            ),
            "flips_margin_below_cross_batch_deviation": int(
                flip_mask.sum() - above_threshold.sum()
            ),
            "flips_margin_note": FLIP_MARGIN_NOTE,
            "q_delta_mean_abs": float(finite_delta.mean()),
            "q_delta_max_abs": float(finite_delta.max()),
        }
        results[f"{a}->{b}"] = entry
    return results


def print_results(results: dict[str, dict], label: str) -> None:
    print(f"-- {label}")
    for pair, entry in results.items():
        print(f"  {pair:14s} flip={entry['flip_rate']:.4%} "
              f"({entry['flip_count']}/{entry['decisions']})  "
              f"TV={entry['tv_mean']:.5f}  KL(a||b)={entry['kl_a_b']:.6f}  "
              f"KL(b||a)={entry['kl_b_a']:.6f}  mean|dq|={entry['q_delta_mean_abs']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="smoke-test only: use the first N panel decisions (never for a reported run)",
    )
    parser.add_argument("--chunk", type=int, default=CHUNK,
                        help="forward batch size for the reported numbers")
    parser.add_argument(
        "--sensitivity-chunks", default="512,4096",
        help="comma-separated chunk sizes the whole metric set is recomputed at",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cycle_dir = RUN_DIR / f"cycle{PANEL_CYCLE}"
    records, panel_obs = load_panel(cycle_dir)
    if args.limit:
        records = records[: args.limit]
        print(f"SMOKE TEST: truncated panel to {len(records)} decisions")
    seeds = np.array([int(r["seed"]) for r in records])
    paths = checkpoint_paths()
    actions = np.array([int(r["action"]) for r in records])
    recorded_lp = np.array([float(r["logprob"]) for r in records])

    print(f"panel: {cycle_dir.name}, {len(records)} decisions, "
          f"{len(np.unique(seeds))} seeds, legal actions mean "
          f"{statistics.fmean(legal_action_count(r) for r in records):.2f}")

    tables: dict[str, np.ndarray] = {}
    for label, path in paths.items():
        tables[label] = q_table_for(
            path, records=records, panel_obs=panel_obs, device=device,
            chunk_size=args.chunk,
        )
        print(f"  forward {label:7s} <- {path.name}")

    # --- panel integrity: the parent pass must reproduce what collection recorded
    parent_q = tables["parent"]
    max_q_diff = 0.0
    mean_q_diff = 0.0
    q_scale = 0.0
    shape_mismatch = False
    for index, record in enumerate(records):
        mask = np.array(bits_to_mask(int(record["mask_bits"])))
        recomputed = parent_q[index][mask]
        recorded = np.asarray(record["q_legal"], dtype=np.float64)
        if recomputed.shape != recorded.shape:
            shape_mismatch = True
            break
        if recorded.size:
            delta = np.abs(recomputed - recorded)
            max_q_diff = max(max_q_diff, float(np.max(delta)))
            mean_q_diff += float(delta.mean())
            q_scale = max(q_scale, float(np.max(np.abs(recorded))))
    mean_q_diff /= max(1, len(records))

    log_p = log_softmax64(parent_q.astype(np.float64))
    recomputed_lp = log_p[np.arange(len(records)), actions]
    integrity: dict[str, object] = {
        "q_legal_shape_mismatch": shape_mismatch,
        "parent_q_legal_mean_abs_diff": mean_q_diff,
        "parent_q_legal_max_abs_diff": max_q_diff,
        "parent_q_legal_scale": q_scale,
        "parent_q_legal_max_rel_diff": max_q_diff / q_scale if q_scale else 0.0,
        "parent_logprob_max_abs_diff": float(np.max(np.abs(recomputed_lp - recorded_lp))),
        "logprob_tolerance": RECOMPUTE_TOLERANCE,
    }
    integrity["panel_is_faithful"] = bool(
        not shape_mismatch
        and integrity["parent_q_legal_max_rel_diff"] < Q_RELATIVE_TOLERANCE
        and integrity["parent_logprob_max_abs_diff"] < RECOMPUTE_TOLERANCE
    )
    print(f"panel integrity: q_legal max|d|={max_q_diff:.3e} (mean {mean_q_diff:.3e}, "
          f"scale {q_scale:.1f}, rel {integrity['parent_q_legal_max_rel_diff']:.2e}), "
          f"logprob max|d|={integrity['parent_logprob_max_abs_diff']:.3e} "
          f"(tol {RECOMPUTE_TOLERANCE:.0e}), faithful={integrity['panel_is_faithful']}")
    if not integrity["panel_is_faithful"]:
        raise RuntimeError("panel replay does not reproduce collection-time records")

    # Every checkpoint must be finite exactly on the legal mask: a nan or a stray
    # -inf inside the legal set would corrupt TV/KL without raising anywhere else.
    legal = legal_mask_from_records(records)
    for label, table in tables.items():
        if not np.isfinite(table[legal]).all():
            raise RuntimeError(f"{label}: non-finite score on a legal action")
        if np.isfinite(table[~legal]).any():
            raise RuntimeError(f"{label}: finite score on an illegal action")
    print(f"legal-mask integrity: all {len(tables)} checkpoints finite exactly on "
          f"the {int(legal.sum())} legal entries")

    # Same weights + same chunk size must be bit-identical, otherwise the flip set
    # below would contain run-to-run noise on top of the weight change.
    parent_repeat = q_table_for(
        paths["parent"], records=records, panel_obs=panel_obs, device=device,
        chunk_size=args.chunk,
    )
    both_finite = np.isfinite(parent_repeat) & np.isfinite(tables["parent"])
    repeat_diff = float(
        np.max(np.abs(parent_repeat[both_finite] - tables["parent"][both_finite]))
    )
    del parent_repeat
    integrity["batched_forward_repeat_max_abs_diff"] = repeat_diff
    integrity["batched_forward_is_deterministic"] = repeat_diff == 0.0
    print(f"batched forward repeat (same weights, chunk {args.chunk}): "
          f"max|d|={repeat_diff:.3e}, deterministic={integrity['batched_forward_is_deterministic']}")

    # The measured round-off between this batched pass and the collection-time
    # pass on the SAME weights is the floor below which a top-2 margin carries no
    # information.
    q_noise_floor = float(integrity["parent_q_legal_max_abs_diff"])

    results = results_from_tables(tables, seeds, q_noise_floor, legal)
    print_results(results, f"chunk {args.chunk} (reported)")

    # --- reference scale: the parent's own sample-vs-greedy disagreement.
    # The record's `is_greedy` field is only set on the eps-greedy branch, which
    # never runs at boltzmann_epsilon=1, so the disagreement is recomputed here
    # from the recorded sampled action instead of being read off that flag.
    parent_metrics = pair_metrics(tables["parent"], tables["parent"], legal)
    reference = {
        "sample_vs_greedy_disagreement_rate": float(
            np.mean(parent_metrics["argmax_a"] != actions)
        ),
        "sample_vs_greedy_disagreement_note": (
            "recomputed: recorded sampled action != argmax of the parent's own "
            "legal q. The record's is_greedy flag is uninformative here because "
            "the eps-greedy branch never runs at boltzmann_epsilon=1."
        ),
        "mean_off_argmax_mass_parent": float(np.mean(1.0 - parent_metrics["max_prob_a"])),
        "mean_off_argmax_mass_C4": float(
            np.mean(1.0 - pair_metrics(tables["C4"], tables["C4"], legal)["max_prob_a"])
        ),
    }

    # --- batching sensitivity: recompute the whole metric set at other chunks
    sensitivity: dict[str, dict] = {}
    for chunk in [
        int(value) for value in str(args.sensitivity_chunks).split(",") if value.strip()
    ]:
        if chunk == args.chunk:
            continue
        chunk_tables = {
            label: q_table_for(
                path, records=records, panel_obs=panel_obs, device=device, chunk_size=chunk
            )
            for label, path in paths.items()
        }
        sensitivity[str(chunk)] = results_from_tables(
            chunk_tables, seeds, q_noise_floor, legal
        )
        del chunk_tables
        print_results(sensitivity[str(chunk)], f"chunk {chunk} (sensitivity)")

    spread = {}
    for pair in results:
        spread[pair] = {
            metric: [
                min(entry[pair][metric] for entry in [results, *sensitivity.values()]),
                max(entry[pair][metric] for entry in [results, *sensitivity.values()]),
            ]
            for metric in ("flip_rate", "tv_mean", "kl_a_b", "kl_b_a")
        }
    sensitivity["spread"] = spread

    payload = {
        "schema": "keqing.mortal.p4m10_policy_displacement.v1",
        "read_only": True,
        "training_or_evaluation_run": False,
        "purpose": (
            "How much did the four on-policy PG updates move the policy? "
            "Fixed panel, argmax flip rate + TV + KL, parent and adjacent cycles."
        ),
        "panel": {
            "description": (
                "P4-M10 cycle 1 decisions: 256 hanchans collected with the PARENT "
                "weights against 3 x ext_mortal (T=1, eps=1, top_p=1). Frozen and "
                "shared by every checkpoint; kan-select and quick-eval bypass "
                "decisions never reach the model and are absent by construction."
            ),
            "cycle_dir": str(cycle_dir.relative_to(REPO_ROOT)),
            "decisions": len(records),
            "seeds": sorted(int(s) for s in np.unique(seeds)),
            "forward_chunk": args.chunk,
            "mean_legal_actions": float(
                statistics.fmean(legal_action_count(r) for r in records)
            ),
        },
        "semantics": {
            "flip_rate": "argmax of q over legal actions (greedy deployment policy)",
            "tv": "0.5 * sum |pi_a - pi_b| on pi = softmax(q_legal / 1) (T=1 policy)",
            "kl_a_b": "mean per-decision KL(pi_a || pi_b) in nats, not KL of the aggregate",
            "kl_b_a": "the same in the opposite direction",
        },
        "metric_roles": METRIC_ROLES,
        "panel_substitution": PANEL_SUBSTITUTION,
        "interpretation_boundaries": INTERPRETATION_BOUNDARIES,
        "panel_integrity": integrity,
        "q_noise_floor": {
            "value": q_noise_floor,
            "source": (
                "max |q_batched - q_collection_time| on the PARENT weights over the "
                "panel, i.e. the round-off of a batched forward vs the engine's own "
                "per-call batch. Recorded as a measurement-credibility number and as "
                "the descriptive split threshold, NOT as a per-state bound on how small "
                "a margin can be before a flip stops being meaningful."
            ),
            "is_per_state_bound": False,
            "may_be_used_to_discount_flips": False,
            "logprob_agreement": integrity["parent_logprob_max_abs_diff"],
            "logprob_agreement_note": (
                "same order as the P4-M10 cycle-1 recompute max_delta that the "
                "training run itself accepted (1.94e-03), so this is the known "
                "batched-forward nondeterminism and not a new discrepancy"
            ),
        },
        "reference_scale": reference,
        "results": results,
        "batching_sensitivity": sensitivity,
        "bootstrap": {
            "cluster": "seed (64 clusters)",
            "reps": BOOTSTRAP_REPS,
            "rng_seed": BOOTSTRAP_SEED,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
