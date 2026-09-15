"""P4-M12: extend the direct on-policy PG loop from the P4-M11 U32 endpoint.

Owner ruling 2026-09-15: stop extending the gap probe and buy one more *fixed*
continuation window on the line that already beat K0, so that the question
"can this line still improve on top of U32?" is answered by a direct
parent-vs-child match instead of by a training curve.

This module is a *thin continuation* of ``p4m11_direct_pg``, exactly as P4-M11 is
a thin continuation of ``p4m10_onpolicy_pg``.  Nothing about the recipe changes:

    same network, same 3 x external_mortal opponent, same T=1 sampling,
    same 256 hanchans per cycle, same accumulated backward pass, exactly one
    Adam step per cycle, same lr, same loss, same global-norm clip, and the same
    parent-*optimizer* restore discipline.

What changes is only what the ruling requires:

    the parent is U32 (Adam step 36) instead of C4 (Adam step 4); 32 further
    cycles run on a *fresh* seed block; and the experiment gets its own identity,
    so the artifacts name the experiment that actually produced them.

    for u in 1..32 (lineage cycles 33..64):
        export current weights -> collect 256 hanchans with THOSE weights
        replay the recorded observations -> one accumulated backward pass
        clip + exactly one Adam step          (Adam step 36 -> 36+u)
        save U{u}.pth, then write the completion marker last

The final endpoint U64 at Adam step 68 is the only candidate.  Intermediate
checkpoints are committed normally but are never evaluated and never selected on
telemetry.  Gate C then runs the frozen bidirectional native 1v3 protocol with
U64 against **U32** (parent vs child) on a fresh evaluation seed block.

The seam is ``P4M11_CONFIG``: every P4-M11 function reads it at call time, so
rebinding it to :data:`P4M12_CONFIG` re-uses the whole validated pipeline while
labelling the artifacts truthfully.  The binding is installed by :func:`main`
and always restored, so importing this module has no effect on P4-M11's own
behaviour.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training.mortal.p4m11_direct_pg as _p4m11  # noqa: E402  (after the sys.path bootstrap)


class P4M12ContractError(RuntimeError):
    """A P4-M12 fail-closed contract violation."""


# ---------------------------------------------------------------------------
# frozen continuation configuration (pre-registered, not tunable at run time)
#
# Derived from P4-M11's frozen block by overriding identity, lineage, parent and
# seeds only.  ``changed_keys()`` below fails closed if any other key ever
# differs, so "the recipe did not change" is a checked property rather than a
# claim in a document.
# ---------------------------------------------------------------------------
_OVERRIDES: dict[str, Any] = {
    "experiment": "P4-M12",
    "schema_prefix": "keqing.mortal.p4m12",
    "result_filename": "p4m12_result.json",
    "lineage": "hard50k -> C4 -> U32 -> U64",
    # U32 is the frozen P4-M11 endpoint: the *training* checkpoint, which carries
    # model + Adam state at step 36.  The eval-weights export is deliberately NOT
    # the parent: it has no optimizer state, and resuming from it with a fresh
    # Adam would silently be a different experiment.
    "parent": (
        "artifacts/experiments/student_policy_v1/P4-M11_direct_pg_C4_plus32/U32.pth"
    ),
    "parent_sha256": (
        "511be9adba1e0a38a093cb59b498c6a7e7be031d0f5e09afb06f4b361690b9cd"
    ),
    "parent_completed_cycles": 32,
    "parent_inherited_adam_step": 36,
    # Fresh collection block.  P4-M11 owns 730000-732047 and those games are part
    # of its training identity; reusing them would make this a re-run, not a
    # continuation.  2048 seeds = 32 cycles x 64 seeds.
    "seed_start": 733000,
    "seed_end": 735047,
    "sampling_seed_base": 2026091600,
    "expected_final_adam_step": 36 + 32,
    "challenger_label": "p4m12_candidate",
    # U64 is the *model's* name -- the lineage endpoint, matching the frozen
    # ``lineage`` string above.  It is NOT a path: checkpoints are named by cycle
    # (``U1..U32``), so the endpoint file is ``U32.pth``.  The two names coincide
    # for P4-M11 (32 cycles -> U32 both ways), which is why this is stated here.
    "evaluation": (
        "final U64 (lineage endpoint, stored as U32.pth) only; intermediate "
        "cycles are committed but never evaluated and never selected on "
        "training telemetry"
    ),
}

P4M12_CONFIG: dict[str, Any] = {**copy.deepcopy(_p4m11.P4M11_CONFIG), **_OVERRIDES}

# The only keys a continuation is allowed to change.  Pinned as an *equality* so
# that both a forgotten override and an accidental recipe edit are caught: a
# continuation that silently inherits a changed lr, loss, opponent or clip is
# exactly the failure this check exists for.
CHANGED_KEYS: frozenset[str] = frozenset(_OVERRIDES)


def changed_keys() -> set[str]:
    """Keys whose value differs between the frozen P4-M11 config and P4-M12's."""
    keys = set(_p4m11.P4M11_CONFIG) | set(P4M12_CONFIG)
    return {k for k in keys if _p4m11.P4M11_CONFIG.get(k) != P4M12_CONFIG.get(k)}


def assert_only_identity_changed() -> None:
    """Fail closed unless P4-M12 differs from P4-M11 in the intended keys only."""
    changed = changed_keys()
    unexpected = sorted(changed - CHANGED_KEYS)
    if unexpected:
        raise P4M12ContractError(
            f"P4-M12 would change {unexpected} relative to the frozen P4-M11 recipe; "
            "a continuation may only change "
            f"{sorted(CHANGED_KEYS)}. Either the recipe drifted or the override is wrong."
        )
    missing = sorted(CHANGED_KEYS - changed)
    if missing:
        raise P4M12ContractError(
            f"P4-M12 declares {missing} as changed but the values are identical to "
            "P4-M11's; the continuation would not actually be a continuation."
        )
    # Inherited blocks must be *identical objects by value*, which the loop above
    # already proves; this only makes the intent explicit for the three blocks the
    # reused collector reads.
    for block in ("sampling", "return", "loss", "optimizer", "gradient_clip"):
        if P4M12_CONFIG[block] != _p4m11.P4M11_CONFIG[block]:
            raise P4M12ContractError(f"P4-M12 changed the inherited {block!r} block")


# Import-time: cheap, pure, and it makes a drifted continuation fail immediately
# rather than after the first cycle has been collected.
assert_only_identity_changed()


def install_config() -> dict[str, Any]:
    """Bind P4-M11's machinery to the P4-M12 configuration.

    Returns the previous binding so the caller can restore it.  Nothing else in
    the process is mutated, and no state is written.
    """
    assert_only_identity_changed()
    previous = _p4m11.P4M11_CONFIG
    _p4m11.P4M11_CONFIG = P4M12_CONFIG
    return previous


def restore_config(previous: dict[str, Any]) -> None:
    """Undo :func:`install_config`."""
    _p4m11.P4M11_CONFIG = previous


def summary() -> dict[str, Any]:
    """The fields a run record should print before any GPU work happens."""
    return {
        "experiment": P4M12_CONFIG["experiment"],
        "lineage": P4M12_CONFIG["lineage"],
        "parent": P4M12_CONFIG["parent"],
        "parent_sha256": P4M12_CONFIG["parent_sha256"],
        "parent_completed_cycles": P4M12_CONFIG["parent_completed_cycles"],
        "parent_inherited_adam_step": P4M12_CONFIG["parent_inherited_adam_step"],
        "cycles": P4M12_CONFIG["cycles"],
        "hanchans": P4M12_CONFIG["max_hanchans"],
        "seed_block": [P4M12_CONFIG["seed_start"], P4M12_CONFIG["seed_end"]],
        "expected_final_adam_step": P4M12_CONFIG["expected_final_adam_step"],
        "champion": P4M12_CONFIG["champion"],
        "challenger_label": P4M12_CONFIG["challenger_label"],
        "changed_keys": sorted(CHANGED_KEYS),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run the P4-M12 continuation through the validated P4-M11 pipeline.

    Every flag of ``p4m11_direct_pg`` still applies (``--output-dir`` is the only
    required one, and ``--dry-run`` validates identity and resume without doing
    any GPU work).  Their defaults now resolve against the P4-M12 config, so the
    default parent is U32 and the default opponent is the frozen external Mortal.
    """
    previous = install_config()
    try:
        _p4m11.main(argv)
    finally:
        restore_config(previous)


if __name__ == "__main__":
    main()
