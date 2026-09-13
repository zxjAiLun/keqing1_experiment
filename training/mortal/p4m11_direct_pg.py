"""P4-M11: extend the direct on-policy PG loop from the frozen P4-M10 C4.

Authorised scope (owner plan 2026-09-13): ONE lineage ``hard50k -> C4 -> U32``.
C4's **full** state (model *and* Adam optimizer state at step 4) is restored and
32 further cycles of 256 on-policy hanchans are collected against the fixed
3 x external_mortal trio; each cycle is one accumulated backward pass and
exactly one Adam step.  No critic, no PPO, no BC, no KL anchor, no entropy, no
replay buffer, no opponent pool, no warmup restart, no learning-rate/temperature
search, and no change to the guard / kan-select / quick-eval semantics.

    for u in 1..32:
        export current weights -> collect 256 hanchans with THOSE weights
        replay the recorded observations -> one accumulated backward pass
        clip + exactly one Adam step              (inherited Adam step 4 -> 4+u)
        save U{u}.pth, then write the completion marker last

This module is a *thin continuation* of ``p4m10_onpolicy_pg``: collection, the
accumulated update, the reconciler and the recompute gate are imported and
reused unchanged, so the frozen P4-M10 module and its tests keep their exact
behaviour.  What is new here is only what the owner plan requires and P4-M10
does not provide:

* explicit parent **optimizer** restore with parameter-group and step validation.
  P4-M10 created a fresh Adam inside ``main``.  Restoring only the weights (for
  example from ``C4_eval_weights.pth``) and starting a fresh Adam would silently
  be a different experiment, so the step is asserted to be exactly 4 before the
  first update and 4+u after cycle u.
* a per-cycle checkpoint carrying model + optimizer + RNG + the completed cycle
  index + the next seed segment + the recipe and environment identity, committed
  by an atomic completion marker written **last**.
* strict cycle-boundary resume that never re-applies an update: every restart
  reloads the last *committed* checkpoint, so an update that was in flight when
  the process died is discarded rather than applied twice.  A completed
  collection may be reused, but only when its model identity, record
  completeness and sampling configuration still match.
* per-cycle resource snapshots plus a safe-pause path, instead of finishing a
  cycle that would exhaust the host.

Fail-closed conditions are inherited from P4-M10 and unchanged: a mask/log-prob
that cannot be recomputed, a trajectory identity mismatch, a non-finite
gradient, or a missing collection record aborts BEFORE the optimizer step.  A
claim preemption or an agari-guard rewrite is NOT a failure.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p4m9_probe_onpolicy import (
    _build_modules,
    _launch_environment,
)
from training.mortal.p4m10_onpolicy_pg import (
    P4M10_CONFIG,
    P4M10ContractError,
    _parameter_digest,
    _sha256_file,
    build_forward_logprobs,
    collect_cycle,
    cycle_seeds,
    hanchan_returns_from_logs,
    run_pg_update,
)

# ---------------------------------------------------------------------------
# frozen continuation configuration (pre-registered, not tunable at run time)
# ---------------------------------------------------------------------------
P4M11_CONFIG: dict[str, Any] = {
    "experiment": "P4-M11",
    "lineage": "hard50k -> C4 -> U32",
    "parent": (
        "artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256/C4.pth"
    ),
    "parent_sha256": (
        "6f5e5eb7148a1b364d57e8f86be7aa594a26892a7b4a0965347a2d86e0d9f79d"
    ),
    "parent_completed_cycles": 4,
    "parent_inherited_adam_step": 4,
    "champion": (
        "E:/AUbuntuProject/project/keqing1/artifacts/"
        "external_mortal_20240308_best_min.pth"
    ),
    "champion_sha256": (
        "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563"
    ),
    # A brand new lineage must start inside the pinned environment. Resume already
    # compares the saved environment with the current one, which catches "U01 was
    # right, U02 changed interpreter"; it cannot catch "U01 was already wrong",
    # because the fingerprint would simply record the wrong environment and every
    # later resume would agree with it.
    "required_interpreter": (
        "E:/AUbuntuProject/project/keqing1/.venv-win/Scripts/python.exe"
    ),
    "required_native_sha256": (
        "19bb181eaa70d0ae90417a3bd22433f6ca08d7654602f865ff3bdb102b7d9914"
    ),
    "challenger_label": "p4m11_candidate",
    "champion_label": "ext_mortal",
    "architecture": "unchanged (Brain 192x40 + DQN v4, FP32)",
    "objective": "direct_on_policy_policy_gradient",
    "sampling": dict(P4M10_CONFIG["sampling"]),
    "return": dict(P4M10_CONFIG["return"]),
    "loss": P4M10_CONFIG["loss"],
    "optimizer": {
        "kind": "Adam",
        "lr": 1e-5,
        "weight_decay": 0.0,
        "resumed_from_parent": True,
        "fresh_init": False,
    },
    "gradient_clip": {"kind": "global_norm", "value": 1.0},
    "update_micro_batch": int(P4M10_CONFIG["update_micro_batch"]),
    "cycles": 32,
    "seeds_per_cycle": int(P4M10_CONFIG["seeds_per_cycle"]),
    "splits_per_seed": int(P4M10_CONFIG["splits_per_seed"]),
    "hanchans_per_cycle": int(P4M10_CONFIG["hanchans_per_cycle"]),
    "max_hanchans": 32 * int(P4M10_CONFIG["hanchans_per_cycle"]),
    "seed_start": 730000,
    "seed_end": 732047,
    "seed_key": 8192,
    "sampling_seed_base": 2026091300,
    "expected_final_adam_step": 36,
    "evaluation": (
        "final U32 checkpoint only; U08/U16/U24 are never evaluated and never "
        "selected on training telemetry"
    ),
}

# Owner plan section 5: safe-pause thresholds.  A single breach does not pause;
# two consecutive samples at the declared interval do.
RESOURCE_LIMITS: dict[str, Any] = {
    "min_free_dedicated_vram_bytes": 700 * 1024**2,
    "min_available_ram_bytes": 1 * 1024**3,
    "min_commit_headroom_bytes": 2 * 1024**3,
    "min_target_disk_free_bytes": 64 * 1024**3,
    "consecutive_samples": 2,
    "sample_interval_seconds": 30.0,
    "max_active_training_seconds": 6 * 3600,
}

PAUSE_EXIT_CODE = 42
ZERO_DISK_RESERVE_BYTES = 284 * 1024**3

# Critical (hard-stop) thresholds.  Derived as half of the plan's safe-pause
# floors rather than chosen freely: they are reached only after the pause level
# has already been sustained, and they exist because a fusing native call cannot
# be unwound cooperatively once the host is about to run out of memory.  The
# same consecutive-sample rule applies, so ~60 s of sustained near-exhaustion
# is required before the process is killed.
CRITICAL_LIMITS: dict[str, int] = {
    "min_free_dedicated_vram_bytes": RESOURCE_LIMITS["min_free_dedicated_vram_bytes"] // 2,
    "min_available_ram_bytes": RESOURCE_LIMITS["min_available_ram_bytes"] // 2,
    "min_commit_headroom_bytes": RESOURCE_LIMITS["min_commit_headroom_bytes"] // 2,
    "min_target_disk_free_bytes": RESOURCE_LIMITS["min_target_disk_free_bytes"] // 2,
}

# A metric that is required to decide safety must be present.  A probe that is
# unavailable is NOT evidence that resources are fine, so it counts as a breach
# at runtime and as a preflight failure before the run.
REQUIRED_RUNTIME_METRICS: tuple[str, ...] = (
    "target_disk_free_bytes",
    "system_available_bytes",
    "commit_headroom_bytes",
)
REQUIRED_RUNTIME_METRICS_WITH_GPU: tuple[str, ...] = (
    *REQUIRED_RUNTIME_METRICS,
    "cuda_device_free_bytes",
)


class P4M11ContractError(P4M10ContractError):
    """A P4-M11 fail-closed contract violation."""


class P4M11ResourceStop(RuntimeError):
    """A cooperative stop between phases because resources are under pressure.

    Raised at a *safe point* (between collection and update, or before the
    checkpoint is committed), so unwinding keeps every piece of evidence and
    leaves the last committed checkpoint intact.  The hard-stop path for the
    fused native call is :func:`ResourceWatchdog.observe`, which exits the
    process after writing a marker.
    """

    def __init__(self, reason: str, *, critical: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.critical = bool(critical)


# ---------------------------------------------------------------------------
# budget and seed geometry
# ---------------------------------------------------------------------------
def assert_within_budget(cycles: int, seeds_per_cycle: int) -> None:
    """Refuse any configuration that exceeds the authorised 32 x 256 hanchans.

    Deliberately separate from ``p4m10.assert_within_budget``: that one caps the
    frozen P4-M10 budget at 4 cycles and must keep doing so.
    """
    total_seeds = int(cycles) * int(seeds_per_cycle)
    allowed = int(P4M11_CONFIG["seeds_per_cycle"]) * int(P4M11_CONFIG["cycles"])
    if total_seeds > allowed:
        raise P4M11ContractError(
            f"budget exceeded: {cycles} x {seeds_per_cycle} = {total_seeds} seeds "
            f"({total_seeds * int(P4M11_CONFIG['splits_per_seed'])} hanchans) exceeds "
            f"the authorised {allowed} seeds ({P4M11_CONFIG['max_hanchans']} hanchans)"
        )


def assert_recipe_matches_p4m10() -> None:
    """The continuation must not silently inherit a changed P4-M10 recipe.

    P4-M11 reuses P4-M10's ``collect_cycle`` and ``run_pg_update``, both of which
    read ``P4M10_CONFIG`` for the sampling/return/split constants.  If a future
    edit changes that frozen block, P4-M11 would change too without anyone
    noticing; this check makes that visible at import time.
    """
    expected = {
        "boltzmann_epsilon": 1.0,
        "boltzmann_temp": 1.0,
        "top_p": 1.0,
        "stochastic_latent": False,
        "enable_amp": False,
        "dtype": "float32",
    }
    for key, value in expected.items():
        if P4M10_CONFIG["sampling"].get(key) != value:
            raise P4M11ContractError(
                f"P4-M10 sampling[{key!r}] is {P4M10_CONFIG['sampling'].get(key)!r}, "
                f"expected {value!r}; the reused collection path would change"
            )
    if int(P4M10_CONFIG["splits_per_seed"]) != 4:
        raise P4M11ContractError("P4-M10 splits_per_seed changed; seat rotation would differ")
    if [float(v) for v in P4M10_CONFIG["return"]["rank_points_raw"]] != [90.0, 45.0, 0.0, -135.0]:
        raise P4M11ContractError("P4-M10 rank points changed; the return would differ")
    if float(P4M10_CONFIG["return"]["scalar_baseline"]) != 0.0:
        raise P4M11ContractError("P4-M10 baseline changed; the advantage would differ")


def sampling_seed_for(cycle: int) -> int:
    """Policy-RNG seed for cycle ``u``: ``2026091300 + u``.

    This seeds only the *sampling* RNG.  The mahjong game seeds are the separate
    ``seed_start`` block; the two must never be interchanged.
    """
    if cycle < 1:
        raise ValueError("cycle is 1-based")
    return int(P4M11_CONFIG["sampling_seed_base"]) + int(cycle)


def seed_segment_for(cycle: int) -> tuple[int, int]:
    """Inclusive ``[first, last]`` mahjong seed segment owned by cycle ``u``."""
    seeds = cycle_seeds(
        cycle,
        seed_start=int(P4M11_CONFIG["seed_start"]),
        seeds_per_cycle=int(P4M11_CONFIG["seeds_per_cycle"]),
    )
    return int(seeds.start), int(seeds.stop) - 1


# ---------------------------------------------------------------------------
# parent restore (model AND optimizer)
# ---------------------------------------------------------------------------
def completion_status(
    *,
    final_cycle: int,
    final_steps: Sequence[float],
    parent_unchanged: bool,
    total_cycles: int | None = None,
    expected_final_step: float | None = None,
) -> dict[str, Any]:
    """Whether the authorised endpoint has actually been reached.

    Completion is locked to U32 **at Adam step 36** from an untouched parent.
    Anything short of that cannot present itself as the final endpoint, which is
    what previously let a one-cycle run report U01 as complete.  There is no
    diagnostic exemption any more: the shortened-run mode was removed, so a run
    that is not the full U32 simply is not complete.
    """
    total = int(total_cycles if total_cycles is not None else P4M11_CONFIG["cycles"])
    want_step = float(
        expected_final_step
        if expected_final_step is not None
        else P4M11_CONFIG["expected_final_adam_step"]
    )
    steps = [float(value) for value in final_steps]
    complete = bool(
        int(final_cycle) >= total
        and steps == [want_step]
        and bool(parent_unchanged)
    )
    reason: str | None = None
    if not complete:
        reason = (
            f"stopped at U{int(final_cycle)} with Adam step {steps}; the evaluation "
            f"endpoint is U{total} at Adam step {want_step:g}"
        )
    return {
        "complete": complete,
        "reason": reason,
        "endpoint_cycle": total,
        "endpoint_step": want_step,
    }


def optimizer_steps(optimizer: torch.optim.Optimizer) -> list[float]:
    """Distinct Adam step values currently recorded in the optimizer state."""
    values = {
        float(state["step"])
        for state in optimizer.state.values()
        if "step" in state
    }
    return sorted(values)


def assert_adam_step(optimizer: torch.optim.Optimizer, expected: float) -> None:
    """Fail closed unless every parameter carries the expected Adam step."""
    steps = optimizer_steps(optimizer)
    if not steps:
        raise P4M11ContractError(
            "optimizer carries no step counter; a fresh optimizer cannot continue "
            "an inherited state"
        )
    if len(steps) != 1:
        raise P4M11ContractError(
            f"optimizer parameters disagree on the Adam step: {steps}"
        )
    if not math.isclose(steps[0], float(expected), rel_tol=0.0, abs_tol=1e-6):
        raise P4M11ContractError(
            f"inherited Adam step is {steps[0]:g}, expected {float(expected):g}; "
            "restoring only the weights with a fresh optimizer would silently be "
            "a different experiment"
        )


def build_optimizer(parameters: Sequence[torch.Tensor]) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        list(parameters),
        lr=float(P4M11_CONFIG["optimizer"]["lr"]),
        weight_decay=float(P4M11_CONFIG["optimizer"]["weight_decay"]),
    )


def restore_parent(
    parent_path: Path,
    *,
    device: torch.device,
    mortal_root: Path,
    expected_sha256: str | None = None,
    expected_cycles: int | None = None,
    expected_step: float | None = None,
) -> dict[str, Any]:
    """Restore model + Adam state from a committed P4-M10 checkpoint.

    Returns a small dict of the objects plus the provenance needed for the
    checkpoint contract.  Raises :class:`P4M11ContractError` on any mismatch —
    in particular it refuses a checkpoint that carries no optimizer state, which
    is exactly the ``*_eval_weights.pth`` failure mode the plan forbids.
    """
    from training.mortal.four_player_native import _model_dimensions

    parent_path = Path(parent_path)
    if not parent_path.exists():
        raise P4M11ContractError(f"parent checkpoint not found: {parent_path}")
    sha = _sha256_file(parent_path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise P4M11ContractError(
            f"parent sha256 is {sha}, expected {expected_sha256}; refusing to "
            "continue a different lineage"
        )

    # weights_only must stay False: the optimizer state is part of the payload
    # that makes this checkpoint resumable rather than merely evaluable.
    state = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "mortal" not in state or "current_dqn" not in state:
        raise P4M11ContractError(f"parent {parent_path} is not a student-policy checkpoint")

    version, conv_channels, num_blocks = _model_dimensions(state)
    engine_class, brain, dqn = _build_modules(
        state, mortal_root=mortal_root, device=device,
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
    )
    del engine_class
    brain.eval()
    dqn.eval()

    inherited_cycle = state.get("cycle")
    if expected_cycles is not None and int(inherited_cycle) != int(expected_cycles):
        raise P4M11ContractError(
            f"parent records cycle {inherited_cycle!r}, expected {int(expected_cycles)}"
        )

    optimizer_state = state.get("optimizer_state")
    if optimizer_state is None:
        raise P4M11ContractError(
            f"parent {parent_path.name} carries no optimizer_state; it is probably an "
            "eval-weights export. Continuing from it with a fresh Adam (or a fresh "
            "warmup) is forbidden by the P4-M11 plan."
        )

    parameters = list(brain.parameters()) + list(dqn.parameters())
    optimizer = build_optimizer(parameters)

    groups = optimizer_state.get("param_groups") or []
    if len(groups) != 1:
        raise P4M11ContractError(
            f"parent optimizer has {len(groups)} parameter groups, expected 1"
        )
    if len(groups[0].get("params", [])) != len(parameters):
        raise P4M11ContractError(
            f"parent optimizer tracks {len(groups[0].get('params', []))} tensors but "
            f"the rebuilt model has {len(parameters)}; parameter sets do not match"
        )
    # A mismatched parameter set is the failure this check exists for; load into
    # the already-built optimizer so the state lands on the right tensors.
    optimizer.load_state_dict(optimizer_state)

    if int(optimizer_state["state"].__len__()) != len(parameters):
        raise P4M11ContractError(
            f"parent optimizer state covers {len(optimizer_state['state'])} of "
            f"{len(parameters)} tensors"
        )
    assert_adam_step(
        optimizer,
        float(P4M11_CONFIG["parent_inherited_adam_step"] if expected_step is None else expected_step),
    )

    return {
        "brain": brain,
        "dqn": dqn,
        "optimizer": optimizer,
        "parameters": parameters,
        "version": int(version),
        "conv_channels": int(conv_channels),
        "num_blocks": int(num_blocks),
        "parent_path": str(parent_path),
        "parent_sha256": sha,
        "inherited_cycle": int(inherited_cycle),
        "inherited_adam_step": float(optimizer_steps(optimizer)[0]),
    }


# ---------------------------------------------------------------------------
# atomic writes and the per-cycle checkpoint
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON record and flush it to disk immediately.

    The plan requires per-cycle durability: the run must not wait for cycle 32
    before any record exists on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except (RuntimeError, AssertionError) as error:  # pragma: no cover
            state["cuda"] = None
            state["cuda_error"] = repr(error)
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def training_contract(*, cycle: int | None, parent: dict[str, Any]) -> dict[str, Any]:
    """The ``training_contract`` block shared by exports and checkpoints.

    It records the **actual** parent path and sha, rather than inheriting the
    hardcoded hard50k wording that P4-M10's frozen export carries.
    """
    return {
        "schema": "keqing.mortal.student_policy_v1",
        "student": {
            "init": "inherited_from_parent",
            "version": int(parent["version"]),
            "conv_channels": int(parent["conv_channels"]),
            "num_blocks": int(parent["num_blocks"]),
        },
        "objective": str(P4M11_CONFIG["objective"]),
        "experiment": "P4-M11",
        "lineage": str(P4M11_CONFIG["lineage"]),
        "parent": str(parent["parent_path"]),
        "parent_sha256": str(parent["parent_sha256"]),
        "opponent": str(P4M11_CONFIG["champion"]),
        "opponent_sha256": str(P4M11_CONFIG["champion_sha256"]),
        "comment": (
            "P4-M11 cycle export; only mortal + current_dqn are consumed by the "
            "native arena. After on-policy PG these tensors are policy logits / "
            "action scores, NOT calibrated Q estimates."
        ),
        "cycle": None if cycle is None else int(cycle),
    }


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> str:
    """Write a torch payload via temp file + ``os.replace``, then hash it.

    No artefact that another attempt or a resume reads back for identity may ever
    be observable in a half-written state, so every torch write goes through
    here rather than truncating the destination in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return _sha256_file(path)


def export_cycle_weights(
    brain: Any, dqn: Any, path: Path, *, parent: dict[str, Any], cycle: int
) -> str:
    """Write the exact weights the arena will load, in the arena's key layout."""
    return _save_torch_atomic(
        path,
        {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "training_contract": training_contract(cycle=cycle, parent=parent),
        },
    )


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    """Write a checkpoint via a temp file + ``os.replace``, then hash it."""
    return _save_torch_atomic(path, payload)


def commit_marker_path(run_dir: Path, cycle: int) -> Path:
    return Path(run_dir) / f"U{int(cycle)}.done.json"


def checkpoint_path(run_dir: Path, cycle: int) -> Path:
    return Path(run_dir) / f"U{int(cycle)}.pth"


def _marker_cycle_from_name(marker: Path) -> int:
    """The cycle number encoded in ``U{n}.done.json``.

    The filename is part of the contract: ``U2.done.json`` must describe cycle 2.
    A marker whose name and contents disagree means something wrote into this run
    directory that this code did not, so it is refused rather than interpreted.
    """
    name = Path(marker).name
    if not (name.startswith("U") and name.endswith(".done.json")):
        raise P4M11ContractError(f"unexpected completion marker name {name!r}")
    try:
        return int(name[1 : -len(".done.json")])
    except ValueError as error:
        raise P4M11ContractError(
            f"completion marker {name} does not encode a cycle number"
        ) from error


def committed_cycles(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Every cycle whose completion marker is present AND whose hash matches.

    A checkpoint *without* a marker was never committed — the process may have
    died between the save and the marker — so it is deliberately invisible here.
    Resume therefore reloads the last committed state and re-runs that cycle,
    which cannot double-apply an update.

    A marker that exists but is unreadable, carries no cycle number, disagrees
    with its own filename, or whose checkpoint has gone missing is NOT skipped:
    silently ignoring it and then re-collecting over that cycle would destroy the
    only evidence that something went wrong.  The committed set must also be
    contiguous from U1, so a gap or a stray marker is a hard error to be resolved
    by hand rather than a state the runner quietly accepts.
    """
    run_dir = Path(run_dir)
    committed: dict[int, dict[str, Any]] = {}
    for marker in sorted(run_dir.glob("U*.done.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise P4M11ContractError(
                f"completion marker {marker.name} is unreadable ({error}); refusing to "
                "skip it and overwrite the evidence it refers to"
            ) from error
        if not isinstance(data, dict):
            raise P4M11ContractError(
                f"completion marker {marker.name} is not a JSON object"
            )
        try:
            cycle = int(data["completed_cycles"])
        except (KeyError, TypeError, ValueError) as error:
            raise P4M11ContractError(
                f"completion marker {marker.name} has no usable completed_cycles field"
            ) from error
        named = _marker_cycle_from_name(marker)
        if cycle != named:
            raise P4M11ContractError(
                f"completion marker {marker.name} declares completed_cycles={cycle}, "
                f"but its filename says U{named}"
            )
        path = checkpoint_path(run_dir, cycle)
        if not path.exists():
            raise P4M11ContractError(
                f"completion marker {marker.name} refers to {path.name}, which is "
                "missing; refusing to skip it and re-collect over the lost evidence"
            )
        if _sha256_file(path) != str(data.get("checkpoint_sha256", "")):
            raise P4M11ContractError(
                f"checkpoint {path.name} does not match its completion marker; "
                "refusing to resume from an unverifiable state"
            )
        committed[cycle] = data

    present = sorted(committed)
    for index, cycle in enumerate(present, start=1):
        if cycle != index:
            raise P4M11ContractError(
                f"committed cycles are not contiguous from U1: expected U{index}, found "
                f"U{cycle} (present: {present})"
            )
    return committed


def environment_fingerprint(environment: dict[str, Any] | None) -> dict[str, Any]:
    """The part of the launch fingerprint that a resume must still agree on.

    Paths are recorded but not compared: the substantive identity is which
    native binaries were loaded (by content), under which interpreter and torch.
    """
    if not isinstance(environment, dict):
        return {}
    binaries = environment.get("native_binaries") or []
    return {
        "interpreter": environment.get("interpreter"),
        "python_version": environment.get("python_version"),
        "torch_version": environment.get("torch_version"),
        "cuda_available": environment.get("cuda_available"),
        "native_sha256": sorted(
            str(entry.get("sha256"))
            for entry in binaries
            if isinstance(entry, dict) and entry.get("sha256")
        ),
    }


def environment_problems(
    recorded: dict[str, Any] | None, current: dict[str, Any]
) -> list[str]:
    """Reasons the saved environment identity does not match this process."""
    saved = environment_fingerprint(recorded)
    if not saved:
        return ["checkpoint records no environment fingerprint"]
    problems: list[str] = []
    for key in ("interpreter", "python_version", "torch_version", "cuda_available"):
        if saved.get(key) != current.get(key):
            problems.append(
                f"environment[{key!r}] recorded {saved.get(key)!r} != current {current.get(key)!r}"
            )
    if saved.get("native_sha256") != current.get("native_sha256"):
        problems.append(
            "native binary set changed: recorded "
            f"{[value[:12] for value in saved.get('native_sha256') or []]} != current "
            f"{[value[:12] for value in current.get('native_sha256') or []]}"
        )
    return problems


def recipe_problems(
    recorded: dict[str, Any] | None, current: dict[str, Any]
) -> list[str]:
    """Reasons the saved frozen recipe does not match the current one."""
    if not isinstance(recorded, dict) or not recorded:
        return ["checkpoint records no recipe"]
    problems: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        if recorded.get(key) != current.get(key):
            problems.append(
                f"recipe[{key!r}] recorded {recorded.get(key)!r} != current {current.get(key)!r}"
            )
    return problems


def _normalised_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def assert_pinned_device(device: torch.device) -> None:
    """This lineage is a GPU run; there is no CPU fallback.

    ``--device`` defaults to ``cuda if available else cpu``, so an accidental CPU
    interpreter would otherwise start collecting without a word.  Requiring CUDA
    unconditionally -- rather than behind the old opt-in ``--require-cuda`` --
    removes that silent downgrade.
    """
    problems: list[str] = []
    if device.type != "cuda":
        problems.append(f"device is {device.type!r}, not 'cuda'")
    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
    if problems:
        raise P4M11ContractError(
            "refusing to train without the pinned GPU: " + "; ".join(problems)
        )


def assert_fresh_start_environment(
    *, device: torch.device, environment: dict[str, Any]
) -> None:
    """A brand new lineage starts only inside the pinned environment.

    Resume compares the environment saved in the checkpoint with the current one,
    which catches "U01 was right, U02 changed interpreter".  It cannot catch "U01
    was already wrong": the fingerprint would just record the wrong environment
    and every later resume would agree with it.  So the *first* cycle pins the
    device, the interpreter and the native binary explicitly.
    """
    problems: list[str] = []
    try:
        assert_pinned_device(device)
    except P4M11ContractError as error:
        problems.append(str(error))
    wanted_interpreter = str(P4M11_CONFIG["required_interpreter"])
    if _normalised_path(sys.executable) != _normalised_path(wanted_interpreter):
        problems.append(
            f"interpreter is {sys.executable}, expected {wanted_interpreter}"
        )
    wanted_native = str(P4M11_CONFIG["required_native_sha256"])
    native = [
        str(value)
        for value in environment_fingerprint(environment).get("native_sha256") or []
    ]
    if wanted_native not in native:
        problems.append(
            f"the pinned native binary {wanted_native[:12]}... is not loaded; found "
            f"{[value[:12] for value in native]}"
        )
    if problems:
        raise P4M11ContractError(
            "fresh start refused: the pinned training environment does not match: "
            + "; ".join(problems)
        )


def load_committed_checkpoint(
    run_dir: Path,
    cycle: int,
    *,
    device: torch.device,
    mortal_root: Path,
    expected_parent_sha256: str | None = None,
    expected_opponent_sha256: str | None = None,
    expected_recipe: dict[str, Any] | None = None,
    expected_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild model + optimizer from committed cycle ``cycle``.

    The *saved* identity is read back and compared with what this process is
    about to use.  Filling the current configuration into the return value
    instead would make the later parent comparison vacuous and would let a run
    resume with a different opponent, environment or recipe while looking
    self-consistent.
    """
    from training.mortal.four_player_native import _model_dimensions

    path = checkpoint_path(run_dir, cycle)
    if not path.exists():
        raise P4M11ContractError(f"committed checkpoint missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    # ---- identity from the checkpoint itself, never from current config ------ 
    if int(state.get("cycle", -1)) != int(cycle):
        raise P4M11ContractError(
            f"checkpoint {path.name} records cycle {state.get('cycle')!r}, expected {int(cycle)}"
        )
    if int(state.get("completed_cycles", -1)) != int(cycle):
        raise P4M11ContractError(
            f"checkpoint {path.name} records completed_cycles "
            f"{state.get('completed_cycles')!r}, expected {int(cycle)}"
        )

    recipe = expected_recipe if expected_recipe is not None else P4M11_CONFIG
    recorded_recipe = state.get("recipe")
    problems = recipe_problems(recorded_recipe, recipe)
    if problems:
        raise P4M11ContractError(
            f"checkpoint {path.name} was produced under a different frozen recipe: "
            + "; ".join(problems[:4])
        )

    current_environment = expected_environment if expected_environment is not None else _launch_environment()
    problems = environment_problems(state.get("environment"), environment_fingerprint(current_environment))
    if problems:
        raise P4M11ContractError(
            f"checkpoint {path.name} was produced in a different environment: "
            + "; ".join(problems[:4])
        )

    contract = state.get("training_contract") or {}
    # The frozen opponent is part of the resume identity, not merely of the
    # collection manifest: without this, changing the opponent refuses reuse of
    # the old rollouts but happily continues the same training lineage.
    want_opponent_sha = (
        expected_opponent_sha256 if expected_opponent_sha256 is not None
        else str(P4M11_CONFIG["champion_sha256"])
    )
    recorded_opponent_sha = state.get("opponent_sha256") or contract.get(
        "opponent_sha256"
    )
    if not recorded_opponent_sha:
        raise P4M11ContractError(
            f"checkpoint {path.name} does not record its frozen opponent"
        )
    if str(recorded_opponent_sha) != str(want_opponent_sha):
        raise P4M11ContractError(
            f"checkpoint {path.name} was trained against opponent "
            f"{recorded_opponent_sha}, expected {want_opponent_sha}"
        )
    recorded_parent_path = contract.get("parent")
    recorded_parent_sha = contract.get("parent_sha256")
    if not recorded_parent_path or not recorded_parent_sha:
        raise P4M11ContractError(
            f"checkpoint {path.name} does not record its lineage parent"
        )
    want_parent_sha = (
        expected_parent_sha256 if expected_parent_sha256 is not None
        else str(P4M11_CONFIG["parent_sha256"])
    )
    if str(recorded_parent_sha) != str(want_parent_sha):
        raise P4M11ContractError(
            f"checkpoint {path.name} belongs to lineage parent {recorded_parent_sha}, "
            f"expected {want_parent_sha}"
        )
    parent_file = Path(str(recorded_parent_path))
    if parent_file.exists() and _sha256_file(parent_file) != str(recorded_parent_sha):
        raise P4M11ContractError(
            f"the lineage parent {parent_file} has changed on disk since this "
            "checkpoint was written"
        )

    version, conv_channels, num_blocks = _model_dimensions(state)
    engine_class, brain, dqn = _build_modules(
        state, mortal_root=mortal_root, device=device,
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
    )
    del engine_class
    parameters = list(brain.parameters()) + list(dqn.parameters())
    optimizer = build_optimizer(parameters)
    optimizer_state = state.get("optimizer_state")
    if optimizer_state is None:
        raise P4M11ContractError(f"committed checkpoint {path.name} has no optimizer state")
    optimizer.load_state_dict(optimizer_state)
    if "rng" in state:
        restore_rng_state(state["rng"])
    steps = optimizer_steps(optimizer)
    recorded_step = float(
        int(P4M11_CONFIG["parent_inherited_adam_step"]) + int(cycle)
    )
    if steps != [recorded_step]:
        raise P4M11ContractError(
            f"checkpoint {path.name} should sit at Adam step {recorded_step:g} after "
            f"cycle {int(cycle)}, but the restored optimizer holds {steps}"
        )
    return {
        "brain": brain,
        "dqn": dqn,
        "optimizer": optimizer,
        "parameters": parameters,
        "version": int(version),
        "conv_channels": int(conv_channels),
        "num_blocks": int(num_blocks),
        "parent_path": str(recorded_parent_path),
        "parent_sha256": str(recorded_parent_sha),
        "inherited_cycle": int(state["cycle"]),
        "adam_step": steps[0],
        "lineage": state.get("lineage"),
    }


# ---------------------------------------------------------------------------
# collection reuse (identity + completeness + sampling configuration)
# ---------------------------------------------------------------------------
COLLECTION_MANIFEST = "collection_manifest.json"


def collection_manifest_path(cycle_dir: Path) -> Path:
    return Path(cycle_dir) / COLLECTION_MANIFEST


def collection_manifest_problems(
    manifest: dict[str, Any] | None, expected: dict[str, Any]
) -> list[str]:
    """Return the reasons a recorded collection may NOT be reused.

    Reuse is permitted only when the model identity, the record completeness and
    the sampling configuration all still match.  This returns problems instead of
    raising so the caller can retain the evidence and re-collect.
    """
    if manifest is None:
        return ["no collection manifest"]
    problems: list[str] = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            problems.append(f"{key}: manifest {manifest.get(key)!r} != expected {value!r}")
    if not manifest.get("complete"):
        problems.append("manifest is not marked complete")
    return problems


# ---------------------------------------------------------------------------
# resource sampling
# ---------------------------------------------------------------------------
def _commit_snapshot() -> dict[str, Any]:
    """Windows commit charge via ``GetPerformanceInfo`` (as in P4-M6)."""
    empty = {"commit_total_bytes": None, "commit_limit_bytes": None}
    if sys.platform != "win32":
        return empty

    class PI(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong),
        ]

    info = PI()
    info.cb = ctypes.sizeof(info)
    try:
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return empty
    except OSError:  # pragma: no cover - non-Windows ctypes
        return empty
    page = int(info.PageSize)
    total = int(info.CommitTotal) * page
    limit = int(info.CommitLimit) * page
    return {
        "commit_total_bytes": total,
        "commit_limit_bytes": limit,
        "commit_headroom_bytes": limit - total,
        "system_handle_count": int(info.HandleCount),
    }


def resource_snapshot(*, target_dir: Path, label: str | None = None) -> dict[str, Any]:
    """One resource sample: disk, RAM, commit, GPU and process handles.

    All fields are best-effort; a probe that is unavailable is recorded as
    ``None`` rather than guessed.
    """
    snapshot: dict[str, Any] = {
        "recorded_at_unix": time.time(),
        "label": label,
    }
    try:
        import psutil  # imported lazily: the trainer must run without it

        vm = psutil.virtual_memory()
        snapshot["system_available_bytes"] = int(vm.available)
        snapshot["system_memory_percent"] = float(vm.percent)
        process = psutil.Process()
        snapshot["process_rss_bytes"] = int(process.memory_info().rss)
        try:
            snapshot["process_handles"] = int(process.num_handles())
        except (AttributeError, OSError, NotImplementedError):
            snapshot["process_handles"] = None
        try:
            snapshot["process_children"] = len(process.children(recursive=True))
        except (AttributeError, OSError, NotImplementedError):
            snapshot["process_children"] = None
    except ImportError:
        snapshot.setdefault("system_available_bytes", None)
        snapshot.setdefault("system_memory_percent", None)
        snapshot.setdefault("process_rss_bytes", None)
        snapshot.setdefault("process_handles", None)
        snapshot.setdefault("process_children", None)

    snapshot.update(_commit_snapshot())

    try:
        usage = shutil.disk_usage(str(target_dir))
        snapshot["target_disk_free_bytes"] = int(usage.free)
        snapshot["target_disk_total_bytes"] = int(usage.total)
    except OSError:
        snapshot["target_disk_free_bytes"] = None
        snapshot["target_disk_total_bytes"] = None

    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            snapshot["cuda_device_free_bytes"] = int(free_bytes)
            snapshot["cuda_device_total_bytes"] = int(total_bytes)
            snapshot["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            snapshot["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
            snapshot["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        except (RuntimeError, AssertionError) as error:  # pragma: no cover
            snapshot["cuda_probe_error"] = repr(error)
    return snapshot


def _metric_breaches(
    snapshot: dict[str, Any], limits: dict[str, Any], *, scale: str
) -> list[str]:
    problems: list[str] = []

    def check(key: str, floor_key: str, label: str) -> None:
        value = snapshot.get(key)
        if value is None:
            return
        floor = int(limits[floor_key])
        if int(value) < floor:
            problems.append(
                f"{scale} {label} {int(value)} < {floor} "
                f"({int(value) / 1024**2:.0f} MiB < {floor / 1024**2:.0f} MiB)"
            )

    check("cuda_device_free_bytes", "min_free_dedicated_vram_bytes", "free dedicated VRAM")
    check("system_available_bytes", "min_available_ram_bytes", "available RAM")
    check("commit_headroom_bytes", "min_commit_headroom_bytes", "commit headroom")
    check("target_disk_free_bytes", "min_target_disk_free_bytes", "target disk free")
    return problems


def _missing_metrics(snapshot: dict[str, Any], *, require_gpu: bool) -> list[str]:
    """Safety-relevant probes that are absent from this sample.

    A missing probe is not a pass: with no reading we cannot assert that the
    host is fine, so it is treated exactly like a breach.
    """
    required = (
        REQUIRED_RUNTIME_METRICS_WITH_GPU if require_gpu else REQUIRED_RUNTIME_METRICS
    )
    return [
        f"required metric {key!r} is unavailable (probe returned nothing)"
        for key in required
        if snapshot.get(key) is None
    ]


def resource_violations(
    snapshot: dict[str, Any], *, require_gpu: bool | None = None
) -> list[str]:
    """Which safe-pause thresholds this sample breaches (possibly empty).

    ``require_gpu`` defaults to whether CUDA is actually present in this process.
    """
    if require_gpu is None:
        require_gpu = bool(torch.cuda.is_available())
    problems = _metric_breaches(snapshot, RESOURCE_LIMITS, scale="safe-pause")
    problems.extend(_missing_metrics(snapshot, require_gpu=require_gpu))
    return problems


def critical_violations(
    snapshot: dict[str, Any], *, require_gpu: bool | None = None
) -> list[str]:
    """Breaches of the tighter hard-stop thresholds (possibly empty).

    Only a *readable* value that is below the critical floor can hard-stop the
    run.  An unreadable probe is not a pass either, but it escalates to a safe
    pause rather than a hard stop: ``os._exit`` is reserved for the case where we
    can see the machine is about to fail, not for the case where we cannot see
    anything at all.
    """
    if require_gpu is None:
        require_gpu = bool(torch.cuda.is_available())
    del require_gpu  # the critical decision depends only on observed values
    return _metric_breaches(snapshot, CRITICAL_LIMITS, scale="critical")


def preflight_violations(
    snapshot: dict[str, Any], *, require_gpu: bool | None = None,
    required_disk_free_bytes: int = ZERO_DISK_RESERVE_BYTES,
) -> list[str]:
    """Startup gate (plan section 5.1) on top of the runtime floors.

    The runtime guard only protects the run once it is going; this is the
    stricter *before you start collecting* check, which additionally requires the
    full-run disk reserve to be available up front.  Every metric the decision
    depends on must be present — an empty or partial snapshot fails rather than
    passing by omission.
    """
    problems = list(resource_violations(snapshot, require_gpu=require_gpu))
    free = snapshot.get("target_disk_free_bytes")
    if free is not None and int(free) < int(required_disk_free_bytes):
        problems.append(
            f"target disk free {int(free) / 1024**3:.1f} GiB < startup reserve "
            f"{int(required_disk_free_bytes) / 1024**3:.0f} GiB"
        )
    return problems


def _existing_compute_processes() -> list[dict[str, Any]]:
    """Local processes that could compete with the run, with their command lines."""
    found: list[dict[str, Any]] = []
    try:
        import psutil
    except ImportError:
        return found
    markers = ("python", "mortal", "one_vs_three", "train_", "p4m")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            name = str(info.get("name") or "")
            joined = " ".join(info.get("cmdline") or [])
            haystack = f"{name} {joined}".lower()
            if any(marker in haystack for marker in markers):
                found.append({"pid": info["pid"], "name": name, "cmdline": joined})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def startup_survey(*, target_dir: Path) -> dict[str, Any]:
    """Everything plan section 5.1 wants recorded BEFORE collection starts."""
    survey: dict[str, Any] = {
        "snapshot": resource_snapshot(target_dir=target_dir, label="startup"),
    }
    try:
        import psutil

        swap = psutil.swap_memory()
        survey["pagefile"] = {
            "total_bytes": int(swap.total),
            "used_bytes": int(swap.used),
            "free_bytes": int(swap.free),
        }
    except ImportError:
        survey["pagefile"] = None
    survey["existing_compute_processes"] = _existing_compute_processes()
    # On WDDM the driver also lists desktop clients (CEF/ShellHost/explorer);
    # they are recorded verbatim and must NOT be read as competing learners.
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        survey["gpu_compute_apps_csv"] = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        survey["gpu_compute_apps_csv"] = None
    return survey


class ResourceGuard:
    """Sustained-pressure detector: N consecutive breaching samples trip it."""

    def __init__(self, *, limit: int | None = None) -> None:
        self.limit = int(limit or RESOURCE_LIMITS["consecutive_samples"])
        self.consecutive = 0
        self.last_problems: list[str] = []

    def observe(self, problems: list[str]) -> bool:
        self.last_problems = list(problems)
        if problems:
            self.consecutive += 1
        else:
            self.consecutive = 0
        return self.consecutive >= self.limit


class ResourceWatchdog:
    """Samples resources on a background thread for the whole of a cycle.

    The plan's thresholds only mean anything if they are evaluated *while* the
    cycle runs: a cycle takes minutes and can exhaust the host in between the
    two snapshots taken around it.  This class exists so that

    * sustained safe-pause pressure sets ``pause_requested``, which
      :meth:`check` turns into a :class:`P4M11ResourceStop` at the next safe
      point between phases; and
    * sustained *critical* pressure calls the hard-stop path, because a fused
      native call cannot be unwound cooperatively and finishing the cycle is
      explicitly not worth exhausting system memory.

    ``sample_hook`` and ``exiter`` are injectable so both paths are testable on
    CPU without touching the real host.
    """

    def __init__(
        self,
        *,
        target_dir: Path,
        interval: float | None = None,
        guard_limit: int | None = None,
        sample_log: Path | None = None,
        sample_hook: Callable[[], dict[str, Any]] | None = None,
        exiter: Callable[[int], Any] | None = None,
        on_hard_stop: Callable[[str], Any] | None = None,
        on_tick: Callable[[], Any] | None = None,
        require_gpu: bool | None = None,
    ) -> None:
        self.target_dir = Path(target_dir)
        self.sample_log = Path(sample_log) if sample_log is not None else None
        self.interval = float(
            interval if interval is not None
            else RESOURCE_LIMITS["sample_interval_seconds"]
        )
        limit = int(
            guard_limit if guard_limit is not None
            else RESOURCE_LIMITS["consecutive_samples"]
        )
        self.pause_guard = ResourceGuard(limit=limit)
        self.critical_guard = ResourceGuard(limit=limit)
        self.require_gpu = (
            bool(torch.cuda.is_available()) if require_gpu is None else bool(require_gpu)
        )
        self.samples: list[dict[str, Any]] = []
        self.pause_requested = False
        self.terminate_requested = False
        self.pause_reason: str | None = None
        self.critical_reason: str | None = None
        self.hard_stop_calls = 0
        self.tick_errors: list[str] = []
        self._sample_hook = sample_hook or (
            lambda: resource_snapshot(target_dir=self.target_dir, label="inflight")
        )
        self._exiter = exiter or os._exit
        self._on_hard_stop = on_hard_stop
        # Called once per periodic sample, never at a boundary. The cycle uses it
        # to refresh its heartbeat so a crash cannot charge the downtime.
        self._on_tick = on_tick
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- pure, thread-free logic (directly testable) ----------------------
    def observe(self, snapshot: dict[str, Any], *, advance: bool = True) -> None:
        """Fold one sample into the pause/critical state machine.

        ``advance=True`` is for the **periodic** samples: every one of them,
        healthy or not, updates both consecutive-sample counters, so a healthy
        reading really does break a run of pressure.  (Previously only a
        breaching sample reached the critical counter, so
        pressure -> healthy -> pressure still tripped the hard stop.)

        ``advance=False`` is for the phase-boundary snapshots: they are recorded
        and their problems are reported, but they must not push the counters.
        Counting them would let two readings taken moments apart around a phase
        edge masquerade as sustained pressure.
        """
        self.samples.append(snapshot)
        if self.sample_log is not None:
            append_jsonl(self.sample_log, snapshot)
        pause = resource_violations(snapshot, require_gpu=self.require_gpu)
        critical = critical_violations(snapshot, require_gpu=self.require_gpu)
        if not advance:
            return
        if self.pause_guard.observe(pause):
            self.pause_requested = True
            self.pause_reason = self.pause_reason or "; ".join(pause)
        # Always fed, so a healthy sample resets the critical run as well.
        if self.critical_guard.observe(critical):
            self.terminate_requested = True
            self.critical_reason = "; ".join(critical)
            self._hard_stop()

    def check(self) -> None:
        """Safe point: raise if the run must stop before continuing."""
        if self.terminate_requested:
            raise P4M11ResourceStop(
                self.critical_reason or "critical resource pressure", critical=True
            )
        if self.pause_requested:
            raise P4M11ResourceStop(self.pause_reason or "sustained resource pressure")

    def _hard_stop(self) -> None:
        self.hard_stop_calls += 1
        if self._on_hard_stop is not None:
            self._on_hard_stop(self.critical_reason or "critical resource pressure")
        self._exiter(PAUSE_EXIT_CODE)

    # -- background sampling ----------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._on_tick is not None:
                try:
                    self._on_tick()
                except Exception as error:  # noqa: BLE001 - bookkeeping must not kill the guard
                    self.tick_errors.append(repr(error))
            try:
                snapshot = self._sample_hook()
            except Exception as error:  # noqa: BLE001 - a probe failure is a breach
                snapshot = {"label": "inflight", "probe_error": repr(error)}
            self.observe(snapshot)
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="p4m11-resource-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval * 2))

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def record_sample(
    run_dir: Path, *, label: str, watchdog: ResourceWatchdog | None
) -> dict[str, Any]:
    """Take a labelled phase-boundary sample and persist it.

    Boundary snapshots are recorded and reported but deliberately do **not**
    advance the consecutive-sample counters: two readings taken moments apart
    around a phase edge are not evidence of sustained pressure.  Only the
    periodic in-cycle samples count towards a pause or a hard stop.
    """
    snapshot = resource_snapshot(target_dir=run_dir, label=label)
    if watchdog is not None:
        watchdog.observe(snapshot, advance=False)
    else:
        append_jsonl(Path(run_dir) / "resource_samples.jsonl", snapshot)
    return snapshot


def active_time_path(run_dir: Path) -> Path:
    return Path(run_dir) / "active_time.json"


def heartbeat_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cycle_in_progress.json"


def accumulated_active_seconds(run_dir: Path) -> float:
    """Wall time already spent on training cycles, **including interrupted ones**.

    Counting only the cycles that reached ``cycles.jsonl`` would let a loop of
    crashes and restarts spend unlimited wall time while the six-hour budget
    reported zero, so a running cycle's time is charged too.

    The running attempt is charged only up to its **last heartbeat**, which the
    in-cycle watchdog refreshes on every tick.  Charging ``now - started_unix``
    instead would bill the downtime: crash before shutdown, resume the next
    morning, and the six-hour budget reports eight hours spent.  A crash
    therefore forfeits at most one refresh interval of real work, while an
    overnight wait costs nothing.
    """
    run_dir = Path(run_dir)
    committed = 0.0
    path = active_time_path(run_dir)
    if path.exists():
        try:
            committed = float(json.loads(path.read_text(encoding="utf-8"))["active_seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            committed = 0.0
    payload = _read_heartbeat(run_dir)
    if payload is not None:
        committed = float(payload.get("active_seconds_at_start", 0.0)) + float(
            payload.get("elapsed_seconds", 0.0)
        )
    return committed


def _read_heartbeat(run_dir: Path) -> dict[str, Any] | None:
    beat = heartbeat_path(run_dir)
    if not beat.exists():
        return None
    try:
        payload = json.loads(beat.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def refresh_active_cycle(run_dir: Path) -> None:
    """Advance the running cycle's heartbeat.

    Called from the watchdog's periodic tick, so the heartbeat records how much
    time the cycle has genuinely been running rather than how long ago it
    started.  ``started_unix`` is left alone; only ``elapsed_seconds`` moves, and
    it moves monotonically.
    """
    run_dir = Path(run_dir)
    payload = _read_heartbeat(run_dir)
    if payload is None:
        return
    try:
        started = float(payload["started_unix"])
    except (KeyError, TypeError, ValueError):
        return
    now = time.time()
    payload["elapsed_seconds"] = max(
        float(payload.get("elapsed_seconds", 0.0)), max(0.0, now - started)
    )
    payload["refreshed_unix"] = now
    write_json_atomic(heartbeat_path(run_dir), payload)


def begin_active_cycle(run_dir: Path, cycle: int) -> float:
    """Commit the time already spent (including a dead attempt) and start timing.

    Returns the running total at the moment the cycle begins.
    """
    run_dir = Path(run_dir)
    already = accumulated_active_seconds(run_dir)  # charges a stale heartbeat
    now = time.time()
    write_json_atomic(
        active_time_path(run_dir),
        {"schema": "keqing.mortal.p4m11_active_time.v1", "active_seconds": already,
         "updated_at_unix": now},
    )
    write_json_atomic(
        heartbeat_path(run_dir),
        {"schema": "keqing.mortal.p4m11_cycle_heartbeat.v1", "cycle": int(cycle),
         "active_seconds_at_start": already, "started_unix": now,
         "elapsed_seconds": 0.0, "refreshed_unix": now},
    )
    return already


def end_active_cycle(run_dir: Path) -> None:
    """Stop timing the running attempt and commit its elapsed time.

    A graceful finish knows the process stayed alive for the whole cycle, so it
    charges the full wall time rather than the last heartbeat.
    """
    run_dir = Path(run_dir)
    payload = _read_heartbeat(run_dir)
    if payload is None:
        return
    try:
        total = float(payload["active_seconds_at_start"]) + max(
            0.0, time.time() - float(payload["started_unix"])
        )
    except (KeyError, TypeError, ValueError):
        return
    write_json_atomic(
        active_time_path(run_dir),
        {"schema": "keqing.mortal.p4m11_active_time.v1", "active_seconds": total,
         "updated_at_unix": time.time()},
    )
    heartbeat_path(run_dir).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# collection reuse helpers
# ---------------------------------------------------------------------------
def assert_collection_complete(
    *,
    cycle: int,
    hanchan_order: Sequence[tuple[int, int]],
    returns_by_hanchan: dict[tuple[int, int], float],
    expected_hanchans: int,
) -> None:
    """Refuse to step when the collected batch is missing hanchans.

    This runs BEFORE the update, so a short collection can never be turned into
    an optimizer step over a subset of the batch.
    """
    missing = [key for key in hanchan_order if key not in returns_by_hanchan]
    if missing:
        raise P4M11ContractError(
            f"cycle {cycle}: {len(missing)} collected hanchans have no authoritative "
            f"rank, e.g. {missing[:3]}; refusing to update on an incomplete batch"
        )
    if len(hanchan_order) != int(expected_hanchans):
        raise P4M11ContractError(
            f"cycle {cycle}: incomplete collection, {len(hanchan_order)} hanchans with "
            f"decisions != the expected {int(expected_hanchans)}; refusing to update"
        )


def read_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cycle_root(run_dir: Path, cycle: int) -> Path:
    return Path(run_dir) / f"cycle{int(cycle)}"


def cycle_attempt_dirs(run_dir: Path, cycle: int) -> list[Path]:
    """Existing attempt directories for a cycle, newest last."""
    root = cycle_root(run_dir, cycle)
    if not root.exists():
        return []
    attempts = [entry for entry in root.glob("attempt*") if entry.is_dir()]

    def index(path: Path) -> int:
        try:
            return int(path.name.removeprefix("attempt"))
        except ValueError:
            return -1

    return sorted(attempts, key=index)


def next_attempt_dir(run_dir: Path, cycle: int) -> Path:
    """A fresh, never-before-used attempt directory for this cycle.

    Re-collection must not reuse a directory: the old collector is frozen and
    opens ``obs_fp32.bin`` with ``"wb"``, which truncates it.  A failed attempt
    has to survive as evidence, so every attempt gets its own directory and an
    existing one is never opened for writing again.
    """
    existing = cycle_attempt_dirs(run_dir, cycle)
    used: set[int] = set()
    for path in existing:
        try:
            used.add(int(path.name.removeprefix("attempt")))
        except ValueError:
            continue
    index = 1
    while index in used:
        index += 1
    target = cycle_root(run_dir, cycle) / f"attempt{index}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def reusable_collection(
    attempt_dir: Path, expected: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """A previously completed collection, or ``None`` when it must be redone.

    Any doubt at all returns ``None``: re-collecting an uncommitted cycle is
    always safe (the weights are identical) whereas reusing a truncated or
    mis-identified one is not — and the fresh collection now lands in a *new*
    attempt directory, so the doubtful one is kept rather than overwritten.
    """
    attempt_dir = Path(attempt_dir)
    manifest_path = collection_manifest_path(attempt_dir)
    records_path = attempt_dir / "probe_records.jsonl"
    obs_path = attempt_dir / "obs_fp32.bin"
    if not (manifest_path.exists() and records_path.exists() and obs_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if collection_manifest_problems(manifest, expected):
        return None
    if _sha256_file(records_path) != str(manifest.get("records_sha256", "")):
        return None
    if obs_path.stat().st_size != int(manifest.get("obs_bytes", -1)):
        return None
    try:
        records = read_records_jsonl(records_path)
    except (OSError, ValueError):
        return None
    if len(records) != int(manifest.get("decision_records", -1)):
        return None
    return records


def find_reusable_attempt(
    run_dir: Path, cycle: int, expected: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]]] | None:
    """The newest attempt whose collection is still valid, if any."""
    for attempt in reversed(cycle_attempt_dirs(run_dir, cycle)):
        records = reusable_collection(attempt, expected)
        if records is not None:
            return attempt, records
    return None


# ---------------------------------------------------------------------------
# one cycle: collect (or reuse) -> one accumulated update -> committed checkpoint
# ---------------------------------------------------------------------------
def train_cycle(
    *,
    cycle: int,
    run_dir: Path,
    brain: Any,
    dqn: Any,
    optimizer: torch.optim.Optimizer,
    parent: dict[str, Any],
    challenger_label: str,
    champion_path: Path,
    champion_label: str,
    mortal_root: Path,
    device: torch.device,
    micro_batch: int,
    max_clip: float,
    watchdog: ResourceWatchdog | None = None,
    environment: dict[str, Any] | None = None,
    champion_sha256: str | None = None,
    allow_reuse: bool = True,
) -> dict[str, Any]:
    """Exactly one collection, one accumulated backward pass, one Adam step."""
    run_dir = Path(run_dir)
    root = cycle_root(run_dir, cycle)
    root.mkdir(parents=True, exist_ok=True)
    seeds = cycle_seeds(
        cycle,
        seed_start=int(P4M11_CONFIG["seed_start"]),
        seeds_per_cycle=int(P4M11_CONFIG["seeds_per_cycle"]),
    )
    seed_key = int(P4M11_CONFIG["seed_key"])
    sampling_seed = sampling_seed_for(cycle)
    expected_hanchans = len(seeds) * int(P4M11_CONFIG["splits_per_seed"])
    environment = environment if environment is not None else _launch_environment()
    fingerprint = environment_fingerprint(environment)
    champion_sha = (
        str(champion_sha256)
        if champion_sha256 is not None
        else _sha256_file(champion_path)
    )

    expected_step_before = float(
        int(P4M11_CONFIG["parent_inherited_adam_step"]) + int(cycle) - 1
    )
    assert_adam_step(optimizer, expected_step_before)

    # A deterministic staged export supplies the identity hash; each attempt then
    # gets its own copy, so the exact file an attempt consumed is preserved too.
    staged_weights = root / "collection_weights.pth"
    weights_sha = export_cycle_weights(
        brain, dqn, staged_weights, parent=parent, cycle=cycle
    )
    in_memory_sha = _parameter_digest(brain, dqn)

    report: dict[str, Any] = {
        "cycle": int(cycle),
        "seed_segment": [int(seeds.start), int(seeds.stop) - 1],
        "seeds": len(seeds),
        "sampling_seed": int(sampling_seed),
        "adam_step_before": expected_step_before,
        "collection_weights_sha256": weights_sha,
        "in_memory_parameters_sha256": in_memory_sha,
    }

    # Reuse requires the model identity, the opponent, the full sampling
    # configuration AND the native environment to all still match -- matching
    # weights alone would let a re-run silently change the experiment.
    expected_manifest = {
        "weights_sha256": weights_sha,
        "in_memory_parameters_sha256": in_memory_sha,
        "seed_start": int(seeds.start),
        "seed_stop": int(seeds.stop),
        "seed_key": int(seed_key),
        "sampling_seed": int(sampling_seed),
        "hanchans": int(expected_hanchans),
        "challenger_label": str(challenger_label),
        "champion_label": str(champion_label),
        "champion_sha256": champion_sha,
        "champion_path": str(Path(champion_path)),
        "sampling": dict(P4M11_CONFIG["sampling"]),
        "splits_per_seed": int(P4M11_CONFIG["splits_per_seed"]),
        "return_baseline": float(P4M11_CONFIG["return"]["scalar_baseline"]),
        "native_sha256": list(fingerprint.get("native_sha256") or []),
        "torch_version": fingerprint.get("torch_version"),
        "interpreter": fingerprint.get("interpreter"),
    }

    records: list[dict[str, Any]] = []
    attempt_dir: Path | None = None
    reused = False
    if allow_reuse:
        found = find_reusable_attempt(run_dir, cycle, expected_manifest)
        if found is not None:
            attempt_dir, records = found
            reused = True
    if not reused:
        # A brand new directory every time: a retained attempt is never written
        # over, because the frozen collector opens obs_fp32.bin with "wb".
        attempt_dir = next_attempt_dir(run_dir, cycle)
        weights_path = attempt_dir / "collection_weights.pth"
        shutil.copy2(staged_weights, weights_path)
        if _sha256_file(weights_path) != weights_sha:
            raise P4M11ContractError("attempt weights copy does not match the staged export")
        collected = collect_cycle(
            cycle=cycle,
            weights_path=weights_path,
            seeds=seeds,
            seed_key=seed_key,
            challenger_label=challenger_label,
            champion_path=Path(champion_path),
            champion_label=champion_label,
            mortal_root=Path(mortal_root),
            device=device,
            output_dir=attempt_dir,
            sampling_seed=int(sampling_seed),
            version=int(parent["version"]),
            conv_channels=int(parent["conv_channels"]),
            num_blocks=int(parent["num_blocks"]),
        )
        records = collected.pop("records")
        report["collection"] = {
            key: value
            for key, value in collected.items()
            if key not in {"records_path", "obs_path", "log_dir"}
        }
        # The exported file the arena consumed must still be the weights that
        # will receive the gradient: the trajectory-identity gate.
        if _sha256_file(weights_path) != weights_sha:
            raise P4M11ContractError("collection weights changed during collection")
        if _parameter_digest(brain, dqn) != in_memory_sha:
            raise P4M11ContractError("in-memory parameters changed during collection")
        obs_path = attempt_dir / "obs_fp32.bin"
        records_path = attempt_dir / "probe_records.jsonl"
        # The manifest is deliberately NOT written here. Writing it now would
        # advertise the attempt as reusable *before* the arena's authoritative
        # logs have been parsed, so an attempt whose logs are short would be
        # reported complete, fail the completeness check, and then be reused
        # again on the next start -- failing forever on the same bad attempt
        # instead of trying a fresh one. See the write after
        # assert_collection_complete() below.
    report["collection_reused"] = bool(reused)
    report["attempt_dir"] = str(attempt_dir)
    if watchdog is not None:
        watchdog.check()

    obs_path = attempt_dir / "obs_fp32.bin"
    returns_by_hanchan, return_report = hanchan_returns_from_logs(
        log_dir=attempt_dir / "logs",
        seeds=seeds,
        seed_key=seed_key,
        challenger_label=challenger_label,
    )
    report["returns"] = return_report

    policy_records = [record for record in records if record["explore"]]
    kan_select_records = [record for record in records if not record["explore"]]
    if not policy_records:
        raise P4M11ContractError(f"cycle {cycle} collected no policy decisions")
    non_finite = sum(
        1 for record in policy_records if not math.isfinite(float(record["logprob"]))
    )
    if non_finite:
        raise P4M11ContractError(
            f"{non_finite} recorded sampling-time log-probs are not finite"
        )
    greedy = sum(1 for record in policy_records if record["is_greedy"])
    if greedy:
        raise P4M11ContractError(
            f"{greedy} policy decisions are greedy; the loss assumes the sampling "
            "distribution, not the argmax"
        )

    hanchan_order = sorted({
        (int(record["seed"]), int(record["seat"]))
        for record in policy_records
        if record["seed"] is not None
    })
    hanchan_index = {key: index for index, key in enumerate(hanchan_order)}
    assert_collection_complete(
        cycle=cycle,
        hanchan_order=hanchan_order,
        returns_by_hanchan=returns_by_hanchan,
        expected_hanchans=expected_hanchans,
    )

    # Only now is the collection authoritative: the arena's own logs have been
    # parsed and every expected hanchan has a rank. A manifest therefore means
    # "authoritative log completeness PASSED", so a short-log attempt simply has
    # no manifest, is never reported reusable, and the retry lands in a new
    # attempt directory. The completion flag is written last, atomically.
    if not reused:
        records_path = attempt_dir / "probe_records.jsonl"
        write_json_atomic(
            collection_manifest_path(attempt_dir),
            {
                **expected_manifest,
                "schema": "keqing.mortal.p4m11_collection_manifest.v1",
                "experiment": "P4-M11",
                "cycle": int(cycle),
                "attempt_dir": str(attempt_dir),
                "decision_records": len(records),
                "observations": len(records),
                "records_sha256": _sha256_file(records_path),
                "obs_bytes": int((attempt_dir / "obs_fp32.bin").stat().st_size),
                "authoritative_logs_verified": True,
                "complete": True,
                "written_at_unix": time.time(),
            },
        )

    returns = torch.tensor(
        [returns_by_hanchan[key] for key in hanchan_order],
        dtype=torch.float32, device=device,
    )
    parameters = list(brain.parameters()) + list(dqn.parameters())
    report["trainable_tensors"] = len(parameters)

    # Last safe point before the irreversible part of the cycle.
    if watchdog is not None:
        watchdog.check()

    brain.eval()
    dqn.eval()
    update_report = run_pg_update(
        records=policy_records,
        obs_path=obs_path,
        hanchan_index=hanchan_index,
        returns=returns,
        n_hanchans=len(hanchan_order),
        forward_logprobs=build_forward_logprobs(
            brain, dqn, temperature=float(P4M11_CONFIG["sampling"]["boltzmann_temp"])
        ),
        parameters=parameters,
        optimizer=optimizer,
        micro_batch=int(micro_batch),
        device=device,
        max_clip=float(max_clip),
        baseline=float(P4M11_CONFIG["return"]["scalar_baseline"]),
    )
    update_report["policy_decisions_sampled"] = len(policy_records)
    update_report["kan_select_excluded"] = len(kan_select_records)
    update_report["hanchans"] = len(hanchan_order)
    report["update"] = update_report

    # Exactly one step, and the inherited Adam step must have advanced by one.
    expected_step_after = expected_step_before + 1.0
    assert_adam_step(optimizer, expected_step_after)
    report["adam_step_after"] = expected_step_after

    # Do not commit a checkpoint onto a disk that has run out of room, and keep
    # the evidence rather than half-writing one.
    if watchdog is not None:
        watchdog.check()

    # ---- commit: checkpoint first, completion marker last -------------------
    last_cycle = int(P4M11_CONFIG["cycles"])
    next_cycle = int(cycle) + 1
    payload = {
        "schema": "keqing.mortal.p4m11_direct_pg.v1",
        "experiment": "P4-M11",
        "lineage": str(P4M11_CONFIG["lineage"]),
        "mortal": brain.state_dict(),
        "current_dqn": dqn.state_dict(),
        "training_contract": training_contract(cycle=cycle, parent=parent),
        "optimizer_state": optimizer.state_dict(),
        "cycle": int(cycle),
        "completed_cycles": int(cycle),
        "next_cycle": (None if next_cycle > last_cycle else next_cycle),
        "next_seed_segment": (
            None if next_cycle > last_cycle else list(seed_segment_for(next_cycle))
        ),
        "next_sampling_seed": (
            None if next_cycle > last_cycle else sampling_seed_for(next_cycle)
        ),
        "adam_step": float(expected_step_after),
        "recipe": P4M11_CONFIG,
        "opponent_sha256": str(P4M11_CONFIG["champion_sha256"]),
        "environment": _launch_environment(),
        "rng": rng_state(),
        "written_at_unix": time.time(),
    }
    checkpoint = checkpoint_path(run_dir, cycle)
    checkpoint_sha = save_checkpoint_atomic(checkpoint, payload)
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = checkpoint_sha
    # The completion marker is NOT written here. It is the *last* durable step of
    # the cycle and is written by the caller after cycles.jsonl has been appended
    # and fsynced. Writing it here would leave "checkpoint + marker exist but
    # cycles.jsonl has no U{cycle}" as a reachable state.
    report["commit_marker"] = {
        "schema": "keqing.mortal.p4m11_cycle_commit.v1",
        "experiment": "P4-M11",
        "completed_cycles": int(cycle),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "adam_step": float(expected_step_after),
        "seed_segment": report["seed_segment"],
        "sampling_seed": int(sampling_seed),
        "decisions_in_loss": int(update_report["decisions_in_loss"]),
        "hanchans": len(hanchan_order),
        "committed_at_unix": time.time(),
    }

    # Release the per-cycle payloads: nothing big may accumulate across cycles.
    del records, policy_records, kan_select_records, returns
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def assert_recipe_arguments(args: argparse.Namespace) -> None:
    """The official entry point runs the approved recipe or it runs nothing.

    The collector reads the recipe from the frozen config, so an override that
    the CLI accepted but the config ignored would leave the executed run and the
    recipe recorded in every checkpoint disagreeing -- exactly the trap that the
    removed ``--cycles``/``--seeds-per-cycle`` flags set.  Overrides are still
    accepted on the command line, but only when they already equal the approved
    value, so the frozen recipe stays greppable and one authorised change is a
    single config edit.
    """
    problems: list[str] = []

    def compare(label: str, given: Any, approved: Any) -> None:
        if given != approved:
            problems.append(f"{label}: given {given!r} != approved {approved!r}")

    compare("--champion", str(Path(args.champion)), str(Path(P4M11_CONFIG["champion"])))
    compare("--challenger-label", args.challenger_label, P4M11_CONFIG["challenger_label"])
    compare("--champion-label", args.champion_label, P4M11_CONFIG["champion_label"])
    compare("--micro-batch", int(args.micro_batch), int(P4M11_CONFIG["update_micro_batch"]))
    compare(
        "--grad-clip",
        float(args.grad_clip),
        float(P4M11_CONFIG["gradient_clip"]["value"]),
    )
    if problems:
        raise P4M11ContractError(
            "the entry point refuses to deviate from the approved recipe: "
            + "; ".join(problems)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The official entry point is deliberately not configurable.

    ``--cycles`` and ``--seeds-per-cycle`` used to exist and could desynchronise
    the budget check from what the collector actually ran: ``--cycles 64
    --seeds-per-cycle 32`` passed a 32x64 budget check while the collector, which
    reads ``seeds_per_cycle`` from the frozen config, would have collected 16,384
    hanchans.  ``--diagnostic-cycles`` is gone too: a shortened diagnostic run
    still wrote ``U1.pth`` / ``U1.done.json`` into the official output directory,
    and ``committed_cycles()`` had no way to tell that U1 apart from a real one.
    The cycle count and seed count now come from the frozen config and nothing on
    the command line can shorten a run.
    """
    parser = argparse.ArgumentParser(
        description="P4-M11: continue direct on-policy PG from the P4-M10 C4"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path(P4M11_CONFIG["parent"]))
    parser.add_argument("--champion", type=Path, default=Path(P4M11_CONFIG["champion"]))
    parser.add_argument(
        "--challenger-label", default=str(P4M11_CONFIG["challenger_label"])
    )
    parser.add_argument("--champion-label", default=str(P4M11_CONFIG["champion_label"]))
    parser.add_argument(
        "--micro-batch", type=int, default=int(P4M11_CONFIG["update_micro_batch"])
    )
    parser.add_argument(
        "--grad-clip", type=float, default=float(P4M11_CONFIG["gradient_clip"]["value"])
    )
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--max-active-seconds",
        type=float,
        default=float(RESOURCE_LIMITS["max_active_training_seconds"]),
    )
    parser.add_argument(
        "--no-reuse-collection",
        action="store_true",
        help="re-collect even when a completed collection is still valid",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate configuration, identity and resources, then stop",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert_recipe_matches_p4m10()
    # The executed recipe and the recipe recorded in every checkpoint must be the
    # same thing, so deviations are refused rather than silently ignored.
    assert_recipe_arguments(args)

    planned_cycles = int(P4M11_CONFIG["cycles"])
    if planned_cycles < 1:
        raise P4M11ContractError("cycles must be positive")
    # The per-cycle seed count is always the frozen config value, so the budget
    # check and the collector can no longer disagree about the batch size.
    assert_within_budget(planned_cycles, int(P4M11_CONFIG["seeds_per_cycle"]))

    device = torch.device(args.device)
    # Unconditional: there is no CPU fallback for this lineage.
    assert_pinned_device(device)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.parent)
    champion_path = Path(args.champion)
    mortal_root = Path(args.mortal_root)

    if not champion_path.exists():
        raise P4M11ContractError(f"champion checkpoint not found: {champion_path}")

    environment = _launch_environment()
    champion_sha = _sha256_file(champion_path)
    if champion_sha != str(P4M11_CONFIG["champion_sha256"]):
        raise P4M11ContractError(
            f"frozen opponent {champion_path} hashes to {champion_sha}, expected "
            f"{P4M11_CONFIG['champion_sha256']}; the opponent is part of the "
            "training identity and cannot be swapped"
        )

    startup_snapshot = resource_snapshot(target_dir=run_dir, label="startup")
    survey = startup_survey(target_dir=run_dir)
    preflight = preflight_violations(startup_snapshot)
    result: dict[str, Any] = {
        "schema": "keqing.mortal.p4m11_direct_pg.v1",
        "created_at_unix": time.time(),
        "planned_cycles": int(planned_cycles),
        "authorised_cycles": int(P4M11_CONFIG["cycles"]),
        "config": P4M11_CONFIG,
        "resource_limits": RESOURCE_LIMITS,
        "critical_limits": CRITICAL_LIMITS,
        "startup_disk_reserve_bytes": ZERO_DISK_RESERVE_BYTES,
        "launch_environment": environment,
        "platform": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "sys_executable": sys.executable,
        },
        "champion": {"path": str(champion_path), "sha256": champion_sha},
        "startup_resources": startup_snapshot,
        "startup_survey": survey,
        "startup_resource_violations": resource_violations(startup_snapshot),
        "preflight_violations": preflight,
    }

    committed = committed_cycles(run_dir)
    last_committed = max(committed) if committed else 0
    result["resumed_from_cycle"] = int(last_committed)
    result["committed_cycles_on_entry"] = sorted(committed)

    if last_committed == 0:
        # A brand new lineage pins its environment before anything is collected.
        # Resume is covered by load_committed_checkpoint's environment compare.
        assert_fresh_start_environment(device=device, environment=environment)
        restored = restore_parent(
            parent_path,
            device=device,
            mortal_root=mortal_root,
            expected_sha256=str(P4M11_CONFIG["parent_sha256"]),
            expected_cycles=int(P4M11_CONFIG["parent_completed_cycles"]),
            expected_step=float(P4M11_CONFIG["parent_inherited_adam_step"]),
        )
        start_cycle = 1
        result["parent"] = {
            "path": restored["parent_path"],
            "sha256": restored["parent_sha256"],
            "sha256_before": restored["parent_sha256"],
            "inherited_cycle": restored["inherited_cycle"],
            "inherited_adam_step": restored["inherited_adam_step"],
        }
    else:
        # The checkpoint's own recorded identity is validated against this
        # process: recipe, environment, lineage parent and Adam step must all
        # still agree, or the resume is refused outright.
        restored = load_committed_checkpoint(
            run_dir,
            last_committed,
            device=device,
            mortal_root=mortal_root,
            expected_parent_sha256=str(P4M11_CONFIG["parent_sha256"]),
            expected_opponent_sha256=champion_sha,
            expected_recipe=P4M11_CONFIG,
            expected_environment=environment,
        )
        start_cycle = int(last_committed) + 1
        if restored["parent_path"] != str(parent_path):
            raise P4M11ContractError(
                f"resume was given --parent {parent_path} but the checkpoint belongs to "
                f"{restored['parent_path']}; refusing to mix lineages"
            )
        result["parent"] = {
            "path": restored["parent_path"],
            "sha256": restored["parent_sha256"],
            "sha256_before": restored["parent_sha256"],
        }

    brain = restored["brain"]
    dqn = restored["dqn"]
    optimizer = restored["optimizer"]
    parent_ref = {
        "version": restored["version"],
        "conv_channels": restored["conv_channels"],
        "num_blocks": restored["num_blocks"],
        "parent_path": restored["parent_path"],
        "parent_sha256": restored["parent_sha256"],
    }
    result["architecture"] = {
        "version": int(restored["version"]),
        "conv_channels": int(restored["conv_channels"]),
        "num_blocks": int(restored["num_blocks"]),
    }
    result["optimizer"] = {
        "kind": "Adam",
        "lr": float(P4M11_CONFIG["optimizer"]["lr"]),
        "weight_decay": float(P4M11_CONFIG["optimizer"]["weight_decay"]),
        "resumed_from_parent": True,
        "fresh_init": False,
        "adam_step_on_entry": optimizer_steps(optimizer),
        "state_is_reused_across_cycles": True,
    }

    if args.dry_run:
        result["dry_run"] = True
        result["planned_cycle_range"] = [start_cycle, int(planned_cycles)]
        result["seed_segments"] = [
            list(seed_segment_for(cycle))
            for cycle in range(start_cycle, int(planned_cycles) + 1)
        ]
        result["sampling_seeds"] = [
            sampling_seed_for(cycle)
            for cycle in range(start_cycle, int(planned_cycles) + 1)
        ]
        write_json_atomic(run_dir / "dry_run.json", result)
        print(json.dumps({
            "dry_run": True,
            "resumed_from_cycle": last_committed,
            "start_cycle": start_cycle,
            "planned_cycles": int(planned_cycles),
            "adam_step_on_entry": optimizer_steps(optimizer),
            "preflight_violations": preflight,
            "startup_resource_violations": result["startup_resource_violations"],
            "target_disk_free_gib": (
                None if startup_snapshot.get("target_disk_free_bytes") is None
                else round(startup_snapshot["target_disk_free_bytes"] / 1024**3, 1)
            ),
        }, ensure_ascii=False, indent=2), flush=True)
        return

    # Plan section 5.1: the startup check is an execution gate, not a report.
    if preflight:
        raise P4M11ContractError(
            "startup preflight failed; refusing to begin collection: "
            + "; ".join(preflight)
        )

    sample_log = run_dir / "resource_samples.jsonl"
    max_active = float(args.max_active_seconds)
    paused_reason: str | None = None
    started = time.perf_counter()

    for cycle in range(start_cycle, int(planned_cycles) + 1):
        # Interrupted attempts are charged to the budget too: only counting the
        # cycles that reached cycles.jsonl would let a crash loop run forever.
        already = accumulated_active_seconds(run_dir)
        if already >= max_active:
            paused_reason = (
                f"active training budget exhausted: {already:.0f}s >= {max_active:.0f}s"
            )
            break

        begin_active_cycle(run_dir, cycle)
        cycle_started = time.perf_counter()
        watchdog = ResourceWatchdog(
            target_dir=run_dir,
            sample_log=sample_log,
            # Refreshing the heartbeat on every periodic tick is what keeps a
            # crash from charging the downtime to the six-hour budget.
            on_tick=lambda: refresh_active_cycle(run_dir),
        )
        try:
            with watchdog:
                record_sample(run_dir, label=f"cycle{cycle}_before", watchdog=watchdog)
                watchdog.check()
                cycle_report = train_cycle(
                    cycle=cycle,
                    run_dir=run_dir,
                    brain=brain,
                    dqn=dqn,
                    optimizer=optimizer,
                    parent=parent_ref,
                    challenger_label=args.challenger_label,
                    champion_path=champion_path,
                    champion_label=args.champion_label,
                    mortal_root=mortal_root,
                    device=device,
                    micro_batch=int(args.micro_batch),
                    max_clip=float(args.grad_clip),
                    watchdog=watchdog,
                    environment=environment,
                    champion_sha256=champion_sha,
                    allow_reuse=not bool(args.no_reuse_collection),
                )
                record_sample(run_dir, label=f"cycle{cycle}_after", watchdog=watchdog)
        except P4M11ResourceStop as stop:
            # Safe point: unwound before the commit, so the last committed
            # checkpoint and every attempt directory survive untouched.
            end_active_cycle(run_dir)
            paused_reason = stop.reason
            break

        cycle_report["wall_seconds"] = time.perf_counter() - cycle_started
        end_active_cycle(run_dir)

        summary = {
            "cycle": int(cycle),
            "adam_step_after": cycle_report["adam_step_after"],
            "seed_segment": cycle_report["seed_segment"],
            "attempt_dir": cycle_report.get("attempt_dir"),
            "collection_reused": cycle_report["collection_reused"],
            "collection_seconds": cycle_report.get("collection", {}).get("collection_seconds"),
            "hanchans": cycle_report["update"]["hanchans"],
            "decisions_in_loss": cycle_report["update"]["decisions_in_loss"],
            "micro_batches": cycle_report["update"]["micro_batches"],
            "mean_return": cycle_report["returns"]["mean_return"],
            "loss": cycle_report["update"]["loss_sum_of_chunks"],
            "grad_norm_preclip": cycle_report["update"]["grad"]["grad_norm_preclip"],
            "grad_norm_postclip": cycle_report["update"]["grad"]["grad_norm_postclip"],
            "grad_finite": cycle_report["update"]["grad"]["grad_finite"],
            "max_logprob_delta": cycle_report["update"]["logprob_recompute"]["max_abs_diff"],
            "wall_seconds": cycle_report["wall_seconds"],
            "checkpoint_sha256": cycle_report["checkpoint_sha256"],
        }
        append_jsonl(run_dir / "cycles.jsonl", summary)
        # The completion marker is the LAST durable step of the cycle, and the
        # order is the contract: a marker implies this cycle's record is already
        # fsynced. Only now does the cycle become visible to committed_cycles().
        marker = dict(cycle_report["commit_marker"])
        marker["wall_seconds"] = cycle_report["wall_seconds"]
        write_json_atomic(commit_marker_path(run_dir, cycle), marker)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["resumed_from_cycle"] = int(last_committed)
    result["started_at_cycle"] = int(start_cycle)
    result["paused"] = paused_reason is not None
    result["pause_reason"] = paused_reason
    result["accumulated_active_seconds"] = accumulated_active_seconds(run_dir)
    result["session_seconds"] = time.perf_counter() - started
    result["final_resource_snapshot"] = resource_snapshot(target_dir=run_dir, label="final")

    parent_after = _sha256_file(parent_path)
    result["parent"]["sha256_after"] = parent_after
    result["parent"]["sha256_unchanged"] = parent_after == result["parent"]["sha256_before"]

    final_cycle = max(committed_cycles(run_dir), default=last_committed)
    final_steps = optimizer_steps(optimizer)
    result["final_committed_cycle"] = int(final_cycle)
    result["final_checkpoint"] = str(checkpoint_path(run_dir, final_cycle))
    result["final_adam_step"] = final_steps[0] if final_steps else None

    status = completion_status(
        final_cycle=int(final_cycle),
        final_steps=final_steps,
        parent_unchanged=bool(result["parent"]["sha256_unchanged"]),
    )
    result["complete"] = bool(status["complete"])
    result["evaluation_endpoint"] = (
        str(checkpoint_path(run_dir, int(final_cycle))) if status["complete"] else None
    )
    result["evaluation_endpoint_cycle"] = int(status["endpoint_cycle"])
    result["evaluation_endpoint_step"] = float(status["endpoint_step"])
    result["evaluation_scope"] = (
        f"final U{int(P4M11_CONFIG['cycles'])} checkpoint only; U08/U16/U24 are never "
        "evaluated"
    )
    if not status["complete"]:
        result["incomplete_reason"] = status["reason"]

    write_json_atomic(run_dir / "p4m11_result.json", result)
    if paused_reason is not None:
        write_json_atomic(
            run_dir / "PAUSED.json",
            {
                "schema": "keqing.mortal.p4m11_pause.v1",
                "reason": paused_reason,
                "completed_cycles": int(final_cycle),
                "complete": bool(result["complete"]),
                "resume_command_hint": (
                    "re-run the same command; resume starts at cycle "
                    f"{int(final_cycle) + 1} from the last committed checkpoint"
                ),
                "paused_at_unix": time.time(),
            },
        )
        print(f"SAFE PAUSE: {paused_reason}", flush=True)
        raise SystemExit(PAUSE_EXIT_CODE)

    print(f"wrote {run_dir / 'p4m11_result.json'}", flush=True)


if __name__ == "__main__":
    main()
