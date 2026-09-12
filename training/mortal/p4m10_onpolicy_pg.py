"""P4-M10: minimal collect -> one update -> checkpoint loop (direct on-policy PG).

Authorised scope (owner ruling 2026-09-12): ONE candidate, hard50k start, 4 cycles
x 256 on-policy hanchans, each cycle = fresh collection with the current policy
followed by EXACTLY ONE policy-gradient update.  No critic, no PPO, no BC, no KL,
no entropy, no replay buffer, no opponent pool.

This module deliberately implements the smallest loop that changes the learning
mechanism.  It is not an RL framework:

    for cycle in 1..4:
        export current weights  ->  collect 256 hanchans with THOSE weights
        replay the recorded observations + one accumulated backward pass
        clip + exactly one Adam step
        save C{cycle}

Frozen first-candidate settings (pre-registered, not tunable at run time):

===========================  ==================================================
parent                       hard50k 192x40 hard-CE student
opponents                    fixed 3 x ext_mortal
architecture                 unchanged (Brain 192x40 + DQN v4)
sampling                     pi(a|s) = softmax(q_legal), T=1, eps=1, top_p=1
BN                           eval-mode statistics, autograd enabled
return                       Tenhou rank points / 135 -> [2/3, 1/3, 0, -1]
scalar baseline              0
loss                         -(1/N_h) sum_h G_h sum_{t in h} log pi(a_t|s_t)
optimizer                    fresh Adam, lr=1e-5, no weight decay, kept across cycles
gradient clip                global norm 1.0
update microbatch            512, accumulate the whole batch, one step per cycle
training                     4 cycles x 256 hanchans = max 1024 hanchans
evaluation                   final cycle-4 checkpoint only
===========================  ==================================================

Which decisions enter the loss (fixed, so "proposal" and "executed action" are
never confused again):

* every decision the CURRENT model sampled with ``exploration_allowed=True``
  enters the loss, **including** proposals that the arena resolver later denied
  (another player's higher-priority hora/pon) and including an occasional
  agari-guard rewrite - both are accounted at their sampling-time proposal;
* ``enable_quick_eval`` bypasses never reach the model and never enter the loss;
* ``exploration_allowed=False`` kan-select sub-decisions never enter the loss.

Fail-closed conditions (abort BEFORE the optimizer step): a mask/log-prob that
cannot be recomputed, a trajectory identity mismatch, a non-finite gradient, or
a missing collection record.  A claim preemption is NOT a failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.eval_metrics import resolve_rank_points
from training.mortal.p4m9_probe_onpolicy import (
    PROBE_POLICY,  # noqa: F401  (re-exported for inspection)
    ProbeEngine,
    ProbeRecorder,
    _build_modules,
    _launch_environment,
    _load_state,
    _read_obs,
    _sha256_file,
    bits_to_mask,
    parse_log_file,
    pg_loss,
)

# ---------------------------------------------------------------------------
# frozen first-candidate configuration
# ---------------------------------------------------------------------------
RANK_POINTS_PROFILE = "tenhou_reference"
RANK_POINTS_NORMALIZER = 135.0

P4M10_CONFIG: dict[str, Any] = {
    "parent": (
        "artifacts/experiments/student_policy_v1/student_formal_25k/"
        "student_step_050000.pth"
    ),
    "champion": "E:/AUbuntuProject/project/keqing1/artifacts/"
                "external_mortal_20240308_best_min.pth",
    "architecture": "unchanged (Brain 192x40 + DQN v4)",
    "sampling": {
        "boltzmann_epsilon": 1.0,
        "boltzmann_temp": 1.0,
        "top_p": 1.0,
        "stochastic_latent": False,
        "enable_amp": False,
        "dtype": "float32",
        "distribution": "pi(a|s) = softmax(q_legal(s) / T), T = 1",
    },
    "bn_mode": "eval (BatchNorm running statistics, autograd enabled)",
    "return": {
        "definition": "G_h = rank_points[rank_h - 1] / 135 for the challenger's rank",
        "rank_points_raw": [90.0, 45.0, 0.0, -135.0],
        "normalizer": RANK_POINTS_NORMALIZER,
        "rank_points_normalized": [2.0 / 3.0, 1.0 / 3.0, 0.0, -1.0],
        "rank_source": "libriichi.stat.Stat.from_log (native, as in arena evaluation)",
        "scalar_baseline": 0.0,
    },
    "loss": "-(1/N_h) * sum_h G_h * sum_{t in h} log pi(a_t|s_t)",
    "optimizer": {"kind": "Adam", "lr": 1e-5, "weight_decay": 0.0, "fresh_init": True},
    "gradient_clip": {"kind": "global_norm", "value": 1.0},
    "update_micro_batch": 512,
    "cycles": 4,
    "hanchans_per_cycle": 256,
    "seeds_per_cycle": 64,
    "splits_per_seed": 4,
    "max_hanchans": 1024,
    "evaluation": "final cycle-4 checkpoint only; training telemetry never promotes",
    "excluded_from_loss": [
        "enable_quick_eval bypass (model never sampled)",
        "exploration_allowed=False kan-select sub-decisions",
    ],
    "included_in_loss": [
        (
            "every exploration_allowed=True proposal, including claim preemptions and "
            "agari-guard rewrites, booked at its sampling-time log-prob"
        ),
    ],
}

# Pre-declared recompute tolerance.  P4-M9 measured a worst-case |delta| of
# 1.08e-03 for the same decisions replayed at a different batch shape (cuDNN
# picks different kernels), so the sampling-time log-prob is NOT bit-reproducible
# across shapes.  5e-3 keeps a ~5x margin over that measurement while still
# catching real defects (a wrong observation or mask shifts log-probs by O(1)).
RECOMPUTE_TOLERANCE = 5e-3


class P4M10ContractError(RuntimeError):
    """A fail-closed contract violation; raised before any optimizer step."""


def normalized_rank_points() -> tuple[float, float, float, float]:
    """Tenhou reference rank points scaled by the declared normalizer."""
    _profile, raw = resolve_rank_points(rank_points=None, profile=RANK_POINTS_PROFILE)
    return tuple(float(value) / RANK_POINTS_NORMALIZER for value in raw)  # type: ignore[return-value]


def assert_within_budget(cycles: int, seeds_per_cycle: int) -> None:
    """Refuse any configuration that exceeds the authorised 4 x 256 hanchans."""
    total_seeds = int(cycles) * int(seeds_per_cycle)
    allowed = int(P4M10_CONFIG["seeds_per_cycle"]) * int(P4M10_CONFIG["cycles"])
    if total_seeds > allowed:
        raise P4M10ContractError(
            f"budget exceeded: {cycles} x {seeds_per_cycle} = {total_seeds} seeds "
            f"({total_seeds * int(P4M10_CONFIG['splits_per_seed'])} hanchans) exceeds "
            f"the authorised {allowed} seeds ({P4M10_CONFIG['max_hanchans']} hanchans)"
        )


def cycle_seeds(cycle: int, *, seed_start: int, seeds_per_cycle: int) -> range:
    """Each cycle owns a disjoint, contiguous seed segment."""
    if cycle < 1:
        raise ValueError("cycle is 1-based")
    start = int(seed_start) + (int(cycle) - 1) * int(seeds_per_cycle)
    return range(start, start + int(seeds_per_cycle))


def hanchan_returns_from_logs(
    *,
    log_dir: Path,
    seeds: Sequence[int],
    seed_key: int,
    challenger_label: str,
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Per-hanchan return G_h = normalized rank points of the challenger's rank."""
    points = normalized_rank_points()
    returns: dict[tuple[int, int], float] = {}
    ranks: list[int] = []
    missing: list[str] = []
    per_seat_counts: dict[int, int] = {}
    for seed in seeds:
        for split_index, split in enumerate("abcd"):
            path = log_dir / f"{int(seed)}_{int(seed_key)}_{split}.json.gz"
            if not path.exists():
                missing.append(path.name)
                continue
            # Reuse the tested parser: it owns the native log format, the
            # challenger-seat lookup and the authoritative Stat-based rank.
            seat, _decisions, _header, _scores, rank, _terminals = parse_log_file(
                path, challenger_label
            )
            if seat < 0:
                raise P4M10ContractError(
                    f"challenger label {challenger_label!r} absent from {path.name}"
                )
            if rank is None:
                raise P4M10ContractError(f"no authoritative rank for {path.name}")
            returns[(int(seed), int(seat))] = float(points[int(rank) - 1])
            ranks.append(int(rank))
            per_seat_counts[int(split_index)] = per_seat_counts.get(int(split_index), 0) + 1
    if missing:
        raise P4M10ContractError(f"{len(missing)} arena logs missing, e.g. {missing[:3]}")
    report = {
        "hanchans": len(returns),
        "rank_counts": [sum(1 for value in ranks if value == rank) for rank in (1, 2, 3, 4)],
        "mean_return": statistics.fmean(returns.values()) if returns else None,
        "rank_points_normalized": [float(value) for value in points],
        "hanchans_per_split": per_seat_counts,
    }
    return returns, report


def run_pg_update(
    *,
    records: Sequence[dict[str, Any]],
    obs_path: Path,
    hanchan_index: dict[tuple[int, int], int],
    returns: torch.Tensor,
    n_hanchans: int,
    forward_logprobs: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    parameters: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    micro_batch: int,
    device: torch.device,
    max_clip: float,
    baseline: float = 0.0,
    recompute_tolerance: float = RECOMPUTE_TOLERANCE,
) -> dict[str, Any]:
    """One accumulated backward pass over the collected batch, then one step.

    ``forward_logprobs(obs, mask, actions)`` must return the differentiable
    log-probabilities of ``actions`` under the collection policy (that is, with
    the same weights that produced the trajectory and with BatchNorm in eval
    mode).  It is injected so the accumulation/clip/step logic is unit-testable
    without the real network.

    The recorded sampling-time log-probs are used as an integrity check only:
    the loss differentiates through ``forward_logprobs``.
    """
    if not records:
        raise P4M10ContractError("no policy records to update on")
    if micro_batch <= 0:
        raise ValueError("micro_batch must be positive")
    # A NaN sampling-time log-prob would make every later comparison false and
    # silently pass the recompute gate (`nan > tol` is False), so it is checked
    # here as well as at the call site.
    nonfinite_recorded = [
        index for index, record in enumerate(records)
        if not math.isfinite(float(record["logprob"]))
    ]
    if nonfinite_recorded:
        raise P4M10ContractError(
            f"{len(nonfinite_recorded)} recorded sampling-time log-probs are not "
            f"finite, e.g. decision {nonfinite_recorded[:3]}"
        )

    optimizer.zero_grad(set_to_none=True)
    recomputed: list[float] = []
    recorded: list[float] = []
    losses: list[float] = []
    micro_batches = 0
    finite = True

    for start in range(0, len(records), micro_batch):
        chunk = records[start:start + micro_batch]
        obs = torch.as_tensor(
            np.stack([
                _read_obs(obs_path, record["obs_off"], record["obs_bytes"],
                          record["obs_shape"])
                for record in chunk
            ]),
            device=device,
        )
        mask = torch.as_tensor(
            np.array([bits_to_mask(int(record["mask_bits"])) for record in chunk]),
            device=device,
        )
        actions = torch.as_tensor(
            [int(record["action"]) for record in chunk], device=device
        ).unsqueeze(-1)
        hanchan_ids = torch.as_tensor(
            [hanchan_index[(int(record["seed"]), int(record["seat"]))] for record in chunk],
            dtype=torch.long, device=device,
        )
        log_probs = forward_logprobs(obs, mask, actions)
        if not bool(torch.isfinite(log_probs).all()):
            finite = False
        loss = pg_loss(
            log_probs, hanchan_ids, returns, n_hanchans, baseline=baseline
        )
        # Accumulate: one step happens after the whole batch has been consumed.
        loss.backward()
        losses.append(float(loss.detach()))
        recomputed.extend(float(value) for value in log_probs.detach())
        recorded.extend(float(record["logprob"]) for record in chunk)
        micro_batches += 1

    deltas = [abs(a - b) for a, b in zip(recomputed, recorded, strict=True)]
    max_delta = max(deltas) if deltas else 0.0
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    if not math.isfinite(max_delta):
        raise P4M10ContractError(
            "log-prob recompute produced a non-finite difference"
        )
    # Accumulate the norm in float64: squaring float32 grads can overflow to inf
    # even when every gradient element is itself finite, and an infinite norm
    # would make clip_grad_norm_ scale the whole update to zero.
    grad_norm = math.sqrt(sum(
        float(parameter.grad.detach().double().pow(2).sum())
        for parameter in parameters if parameter.grad is not None
    ))
    grads_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters if parameter.grad is not None
    )
    params_with_grad = sum(1 for parameter in parameters if parameter.grad is not None)

    report: dict[str, Any] = {
        "micro_batches": micro_batches,
        "decisions_in_loss": len(records),
        "loss_sum_of_chunks": sum(losses),
        "loss_mean_of_chunks": statistics.fmean(losses) if losses else 0.0,
        "logprob_recompute": {
            "decisions_checked": len(deltas),
            "max_abs_diff": max_delta,
            "mean_abs_diff": mean_delta,
            "tolerance": recompute_tolerance,
            "within_tolerance": sum(1 for value in deltas if value <= recompute_tolerance),
        },
        "grad": {
            "params_with_grad": params_with_grad,
            "params_total": len(parameters),
            "grad_norm_preclip": grad_norm,
            "grad_finite": grads_finite,
            "sample_finite": finite,
        },
    }

    # ---- fail closed, BEFORE the step ------------------------------------
    if not finite:
        raise P4M10ContractError("non-finite log-probability in the update batch")
    if not grads_finite:
        raise P4M10ContractError("non-finite gradient in the update batch")
    if not math.isfinite(grad_norm):
        raise P4M10ContractError(
            f"non-finite gradient norm ({grad_norm}); refusing a step that would "
            "be silently scaled to zero by clipping"
        )
    if max_delta > recompute_tolerance:
        raise P4M10ContractError(
            "sampling-time log-prob not recomputable: max_abs_diff "
            f"{max_delta:.3e} exceeds the pre-declared tolerance {recompute_tolerance:.1e}"
        )
    if params_with_grad != len(parameters):
        raise P4M10ContractError(
            f"only {params_with_grad}/{len(parameters)} parameters received gradients"
        )

    # Global-norm clipping, implemented explicitly rather than via
    # ``clip_grad_norm_``: that helper measures the norm in the gradients' own
    # dtype, so a float32 overflow turns the coefficient into 1/inf = 0 and
    # silently zeroes the entire update.  Scaling by ``max_clip / norm`` with the
    # float64 norm computed above has the same semantics and cannot no-op.
    was_clipped = grad_norm > float(max_clip)
    if was_clipped:
        scale = float(max_clip) / grad_norm
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
    post_clip_norm = math.sqrt(sum(
        float(parameter.grad.detach().double().pow(2).sum())
        for parameter in parameters if parameter.grad is not None
    ))
    report["grad"]["clip_max_norm"] = float(max_clip)
    report["grad"]["was_clipped"] = was_clipped
    report["grad"]["grad_norm_postclip"] = post_clip_norm
    optimizer.step()
    report["optimizer_steps"] = 1
    return report


# ---------------------------------------------------------------------------
# per-cycle driver
# ---------------------------------------------------------------------------
def collect_cycle(
    *,
    cycle: int,
    weights_path: Path,
    seeds: Sequence[int],
    seed_key: int,
    challenger_label: str,
    champion_path: Path,
    champion_label: str,
    mortal_root: Path,
    device: torch.device,
    output_dir: Path,
    sampling_seed: int,
    version: int,
    conv_channels: int,
    num_blocks: int,
) -> dict[str, Any]:
    """Run the arena for one cycle with the exported weights of this cycle."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    obs_path = output_dir / "obs_fp32.bin"
    records_path = output_dir / "probe_records.jsonl"

    challenger_state = torch.load(
        weights_path, weights_only=True, map_location=torch.device("cpu")
    )
    challenger_version, challenger_conv, challenger_blocks = _dimensions(challenger_state)
    recorder = ProbeRecorder(obs_path)
    engine_class, brain, dqn = _build_modules(
        challenger_state, mortal_root=mortal_root, device=device,
        version=challenger_version, conv_channels=challenger_conv,
        num_blocks=challenger_blocks,
    )
    probe_engine = ProbeEngine.build(
        engine_class, recorder=recorder, options={"boltzmann_temp": 1.0}
    )
    challenger_engine = probe_engine(
        brain, dqn,
        is_oracle=False,
        version=challenger_version,
        device=device,
        stochastic_latent=False,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name=challenger_label,
        boltzmann_epsilon=float(P4M10_CONFIG["sampling"]["boltzmann_epsilon"]),
        boltzmann_temp=float(P4M10_CONFIG["sampling"]["boltzmann_temp"]),
        top_p=float(P4M10_CONFIG["sampling"]["top_p"]),
    )
    del brain, dqn

    champion_state = torch.load(
        champion_path, weights_only=True, map_location=torch.device("cpu")
    )
    champion_version, champion_conv, champion_blocks = _dimensions(champion_state)
    champion_engine_class, champion_brain, champion_dqn = _build_modules(
        champion_state, mortal_root=mortal_root, device=device,
        version=champion_version, conv_channels=champion_conv,
        num_blocks=champion_blocks,
    )
    champion_engine = champion_engine_class(
        champion_brain, champion_dqn,
        is_oracle=False,
        version=champion_version,
        device=device,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name=champion_label,
    )
    del champion_brain, champion_dqn

    from libriichi.arena import OneVsThree

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    torch.manual_seed(int(sampling_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))

    started = time.perf_counter()
    rankings = env.py_vs_py(
        challenger=challenger_engine,
        champion=champion_engine,
        seed_start=(int(seeds[0]), int(seed_key)),
        seed_count=len(seeds),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    recorder.close()

    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in recorder.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    del challenger_engine, champion_engine
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "cycle": int(cycle),
        "weights_sha256": _sha256_file(weights_path),
        "collection_seconds": seconds,
        "sampling_seed": int(sampling_seed),
        "hanchans": len(seeds) * int(P4M10_CONFIG["splits_per_seed"]),
        "records": recorder.records,
        "records_path": records_path,
        "obs_path": obs_path,
        "log_dir": log_dir,
        "arena_rankings": [int(value) for value in rankings],
        "calls": len(recorder.batch_sizes),
        "batch_sizes": {
            "min": min(recorder.batch_sizes) if recorder.batch_sizes else 0,
            "max": max(recorder.batch_sizes) if recorder.batch_sizes else 0,
            "mean": (
                statistics.fmean(recorder.batch_sizes) if recorder.batch_sizes else 0.0
            ),
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }


def _dimensions(state: dict[str, Any]) -> tuple[int, int, int]:
    from training.mortal.four_player_native import _model_dimensions

    return _model_dimensions(state)


def export_collection_weights(
    brain: Any,
    dqn: Any,
    path: Path,
    *,
    version: int = 4,
    conv_channels: int = 192,
    num_blocks: int = 40,
    cycle: int | None = None,
) -> str:
    """Write the exact weights the arena will load, in the arena's key layout.

    ``four_player_native._model_dimensions`` resolves the architecture from either
    a standard ``config`` block or a ``training_contract`` block, so the export
    carries the latter: without it the shared loader cannot rebuild Brain/DQN and
    every native arena run would fail before the first kyoku.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "training_contract": {
                "schema": "keqing.mortal.student_policy_v1",
                "student": {
                    "init": "inherited_from_parent",
                    "version": int(version),
                    "conv_channels": int(conv_channels),
                    "num_blocks": int(num_blocks),
                },
                "objective": "direct_on_policy_policy_gradient",
                "comment": (
                    "P4-M10 cycle export; only mortal + current_dqn are consumed "
                    "by the native arena. After on-policy PG these tensors are "
                    "policy logits / action scores, NOT calibrated Q estimates."
                ),
                "cycle": None if cycle is None else int(cycle),
            },
        },
        path,
    )
    return _sha256_file(path)


def build_forward_logprobs(
    brain: Any, dqn: Any, *, temperature: float = 1.0
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def forward(obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        phi = brain(obs)
        q_out = dqn(phi, mask)
        logits = (q_out / temperature).masked_fill(~mask, -torch.inf)
        return torch.log_softmax(logits, dim=-1).gather(-1, actions).squeeze(-1)

    return forward


def train_cycle(
    *,
    cycle: int,
    run_dir: Path,
    brain: Any,
    dqn: Any,
    optimizer: torch.optim.Optimizer,
    challenger_label: str,
    champion_path: Path,
    champion_label: str,
    mortal_root: Path,
    device: torch.device,
    seed_start: int,
    seed_key: int,
    seeds_per_cycle: int,
    sampling_seed: int,
    micro_batch: int,
    max_clip: float,
    version: int,
    conv_channels: int,
    num_blocks: int,
) -> dict[str, Any]:
    """Collect with the current policy, then exactly one accumulated update."""
    cycle_dir = run_dir / f"cycle{cycle}"
    seeds = cycle_seeds(cycle, seed_start=seed_start, seeds_per_cycle=seeds_per_cycle)

    weights_path = cycle_dir / "collection_weights.pth"
    weights_sha = export_collection_weights(
        brain, dqn, weights_path, version=version, conv_channels=conv_channels,
        num_blocks=num_blocks, cycle=cycle,
    )
    in_memory_sha = _parameter_digest(brain, dqn)

    cycle_report: dict[str, Any] = {
        "cycle": int(cycle),
        "seed_range": [seeds.start, seeds.stop - 1],
        "seeds": len(seeds),
        "collection_weights_sha256": weights_sha,
        "in_memory_parameters_sha256": in_memory_sha,
        "optimizer_reused": cycle > 1,
    }

    collected = collect_cycle(
        cycle=cycle, weights_path=weights_path, seeds=seeds, seed_key=seed_key,
        challenger_label=challenger_label, champion_path=champion_path,
        champion_label=champion_label, mortal_root=mortal_root, device=device,
        output_dir=cycle_dir, sampling_seed=sampling_seed, version=version,
        conv_channels=conv_channels, num_blocks=num_blocks,
    )
    records = collected["records"]
    cycle_report["collection"] = {
        key: value for key, value in collected.items()
        if key not in {"records", "records_path", "obs_path", "log_dir"}
    }

    # The exported file the arena consumed must be identical to the weights that
    # will receive the gradient: this is the trajectory-identity gate.
    if _sha256_file(weights_path) != weights_sha:
        raise P4M10ContractError("collection weights changed during collection")
    if _parameter_digest(brain, dqn) != in_memory_sha:
        raise P4M10ContractError("in-memory parameters changed during collection")

    returns_by_hanchan, return_report = hanchan_returns_from_logs(
        log_dir=collected["log_dir"], seeds=seeds, seed_key=seed_key,
        challenger_label=challenger_label,
    )
    cycle_report["returns"] = return_report

    hanchan_order = sorted({
        (int(record["seed"]), int(record["seat"]))
        for record in records
        if record["seed"] is not None and record["explore"]
    })
    hanchan_index = {key: index for index, key in enumerate(hanchan_order)}
    missing = [key for key in hanchan_order if key not in returns_by_hanchan]
    if missing:
        raise P4M10ContractError(
            f"{len(missing)} collected hanchans have no authoritative rank, e.g. {missing[:3]}"
        )
    expected_hanchans = len(seeds) * int(P4M10_CONFIG["splits_per_seed"])
    if len(hanchan_order) != expected_hanchans:
        raise P4M10ContractError(
            f"collected hanchans with decisions {len(hanchan_order)} != {expected_hanchans}"
        )

    policy_records = [record for record in records if record["explore"]]
    kan_select_records = [record for record in records if not record["explore"]]
    non_finite = sum(
        1 for record in policy_records if not math.isfinite(float(record["logprob"]))
    )
    if non_finite:
        raise P4M10ContractError(
            f"{non_finite} recorded sampling-time log-probs are not finite"
        )
    # A greedy decision would mean the engine took the argmax instead of sampling
    # from pi, so its recorded log-prob would not be an on-policy quantity.
    greedy = sum(1 for record in policy_records if record["is_greedy"])
    if greedy:
        raise P4M10ContractError(
            f"{greedy} policy decisions are greedy; the collection policy is not "
            "the sampling distribution the loss assumes"
        )

    returns = torch.tensor(
        [returns_by_hanchan[key] for key in hanchan_order],
        dtype=torch.float32, device=device,
    )
    parameters = list(brain.parameters()) + list(dqn.parameters())
    cycle_report["trainable_tensors"] = len(parameters)

    brain.eval()
    dqn.eval()
    update_report = run_pg_update(
        records=policy_records,
        obs_path=collected["obs_path"],
        hanchan_index=hanchan_index,
        returns=returns,
        n_hanchans=len(hanchan_order),
        forward_logprobs=build_forward_logprobs(
            brain, dqn, temperature=float(P4M10_CONFIG["sampling"]["boltzmann_temp"])
        ),
        parameters=parameters,
        optimizer=optimizer,
        micro_batch=int(micro_batch),
        device=device,
        max_clip=float(max_clip),
        baseline=float(P4M10_CONFIG["return"]["scalar_baseline"]),
    )
    update_report["policy_decisions_sampled"] = len(policy_records)
    update_report["kan_select_excluded"] = len(kan_select_records)
    update_report["hanchans"] = len(hanchan_order)
    cycle_report["update"] = update_report

    # Reporting only: preemptions and guard rewrites never fail the cycle.
    cycle_report["preemption_and_guard"] = {
        "batches_with_agari_guard": None,
        "note": (
            "counted by the P4-M9 reconciler; recorded for reporting and explicitly "
            "NOT a fail-close condition"
        ),
    }

    checkpoint_path = run_dir / f"C{cycle}.pth"
    output_path = run_dir / f"C{cycle}_eval_weights.pth"
    contract = {
        "schema": "keqing.mortal.student_policy_v1",
        "student": {
            "init": "inherited_from_parent",
            "version": int(version),
            "conv_channels": int(conv_channels),
            "num_blocks": int(num_blocks),
        },
        "objective": "direct_on_policy_policy_gradient",
        "parent": P4M10_CONFIG["parent"],
        "cycle": int(cycle),
    }
    torch.save(
        {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "training_contract": contract,
            "optimizer_state": optimizer.state_dict(),
            "cycle": int(cycle),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "training_contract": contract,
        },
        output_path,
    )
    cycle_report["checkpoint"] = str(checkpoint_path)
    cycle_report["eval_weights"] = str(output_path)
    cycle_report["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    return cycle_report


def _parameter_digest(brain: Any, dqn: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for module in (brain, dqn):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4-M10 minimal on-policy PG loop")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent", default=P4M10_CONFIG["parent"])
    parser.add_argument("--champion", default=P4M10_CONFIG["champion"])
    parser.add_argument("--challenger-label", default="p4m10_candidate")
    parser.add_argument("--champion-label", default="ext_mortal")
    parser.add_argument("--cycles", type=int, default=int(P4M10_CONFIG["cycles"]))
    parser.add_argument("--seeds-per-cycle", type=int,
                        default=int(P4M10_CONFIG["seeds_per_cycle"]))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--sampling-seed-base", type=int, default=20260913)
    parser.add_argument("--micro-batch", type=int,
                        default=int(P4M10_CONFIG["update_micro_batch"]))
    parser.add_argument("--learning-rate", type=float,
                        default=float(P4M10_CONFIG["optimizer"]["lr"]))
    parser.add_argument("--grad-clip", type=float,
                        default=float(P4M10_CONFIG["gradient_clip"]["value"]))
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate configuration and paths, then stop")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("CUDA required but not available")
    assert_within_budget(int(args.cycles), int(args.seeds_per_cycle))

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.parent)
    champion_path = Path(args.champion)

    result: dict[str, Any] = {
        "schema": "keqing.mortal.p4m10_onpolicy_pg.v1",
        "created_at_unix": time.time(),
        "config": P4M10_CONFIG,
        "frozen": {
            "recompute_tolerance": RECOMPUTE_TOLERANCE,
            "rank_points_profile": RANK_POINTS_PROFILE,
            "rank_points_normalizer": RANK_POINTS_NORMALIZER,
        },
        "launch_environment": _launch_environment(),
        "platform": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "sys_executable": sys.executable,
        },
        "parent": {
            "path": str(parent_path),
            "sha256": _sha256_file(parent_path),
            "sha256_before": _sha256_file(parent_path),
        },
        "champion": {"path": str(champion_path), "sha256": _sha256_file(champion_path)},
        "cycles": [],
    }

    if args.dry_run:
        (run_dir / "dry_run.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(json.dumps({
            "dry_run": True,
            "parent_sha256": result["parent"]["sha256"],
            "seed_segments": [
                [cycle_seeds(cycle, seed_start=args.seed_start,
                             seeds_per_cycle=args.seeds_per_cycle).start,
                 cycle_seeds(cycle, seed_start=args.seed_start,
                             seeds_per_cycle=args.seeds_per_cycle).stop - 1]
                for cycle in range(1, int(args.cycles) + 1)
            ],
        }, ensure_ascii=False, indent=2))
        return

    version, conv_channels, num_blocks, state = _load_state(parent_path)
    engine_class, brain, dqn = _build_modules(
        state, mortal_root=args.mortal_root, device=device,
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
    )
    del engine_class, state
    brain.eval()
    dqn.eval()
    result["architecture"] = {
        "version": int(version),
        "conv_channels": int(conv_channels),
        "num_blocks": int(num_blocks),
    }

    # Fresh Adam, created once, kept across all cycles.
    parameters = list(brain.parameters()) + list(dqn.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=float(P4M10_CONFIG["optimizer"]["weight_decay"]),
    )
    result["optimizer"] = {
        "kind": "Adam",
        "lr": float(args.learning_rate),
        "weight_decay": float(P4M10_CONFIG["optimizer"]["weight_decay"]),
        "created_once_before_cycle_1": True,
        "state_is_reused_across_cycles": True,
    }

    started = time.perf_counter()
    for cycle in range(1, int(args.cycles) + 1):
        cycle_started = time.perf_counter()
        cycle_report = train_cycle(
            cycle=cycle, run_dir=run_dir, brain=brain, dqn=dqn, optimizer=optimizer,
            challenger_label=args.challenger_label, champion_path=champion_path,
            champion_label=args.champion_label, mortal_root=args.mortal_root,
            device=device, seed_start=int(args.seed_start), seed_key=int(args.seed_key),
            seeds_per_cycle=int(args.seeds_per_cycle),
            sampling_seed=int(args.sampling_seed_base) + cycle,
            micro_batch=int(args.micro_batch), max_clip=float(args.grad_clip),
            version=int(version), conv_channels=int(conv_channels),
            num_blocks=int(num_blocks),
        )
        cycle_report["wall_seconds"] = time.perf_counter() - cycle_started
        result["cycles"].append(cycle_report)
        print(json.dumps({
            "cycle": cycle,
            "collection_seconds": round(cycle_report["collection"]["collection_seconds"], 2),
            "mean_return": cycle_report["returns"]["mean_return"],
            "loss": cycle_report["update"]["loss_sum_of_chunks"],
            "grad_norm_preclip": cycle_report["update"]["grad"]["grad_norm_preclip"],
            "grad_norm_postclip": cycle_report["update"]["grad"]["grad_norm_postclip"],
            "decisions_in_loss": cycle_report["update"]["decisions_in_loss"],
            "max_logprob_delta": cycle_report["update"]["logprob_recompute"]["max_abs_diff"],
            "checkpoint_sha256": cycle_report["checkpoint_sha256"][:16],
        }, ensure_ascii=False), flush=True)

    result["parent"]["sha256_after"] = _sha256_file(parent_path)
    result["parent"]["sha256_unchanged"] = (
        result["parent"]["sha256_after"] == result["parent"]["sha256_before"]
    )
    result["total_wall_seconds"] = time.perf_counter() - started
    result["final_checkpoint"] = str(run_dir / f"C{int(args.cycles)}.pth")
    result["evaluation_scope"] = (
        f"final cycle checkpoint only (C{int(args.cycles)}); C1..C"
        f"{int(args.cycles) - 1} must not be selected on training telemetry"
    )
    (run_dir / "p4m10_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"wrote {run_dir / 'p4m10_result.json'}")


if __name__ == "__main__":
    main()
