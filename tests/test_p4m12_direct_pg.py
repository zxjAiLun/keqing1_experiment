"""P4-M12 identity and continuation invariants.

These are pure, CPU-only checks: no CUDA, no model, no GPU work.  They exist
because P4-M12 reuses the validated P4-M11 pipeline through a config rebind, and
that seam is only safe if the *only* thing it changes is identity plus seeds.

Not in ``pyproject.toml``'s ``python_files`` allowlist: the repository's existing
convention is that training-recipe test modules (``test_p4m11_direct_pg.py``,
``test_p4m10_onpolicy_pg.py``, ...) run when invoked explicitly rather than in the
default suite.  P4-M12 follows it.  Adding this file to the allowlist is a
one-line change if that convention is ever revised.
"""

from __future__ import annotations

import collections
import copy
import pathlib
import re

import pytest

from training.mortal import p4m11_direct_pg as p4m11
from training.mortal import p4m12_direct_pg as p4m12

P4M11_SOURCE = pathlib.Path(p4m11.__file__).read_text(encoding="utf-8")

# Every quoted artifact name that starts with ``p4m11``.  The earlier version of
# this check looked only for ``keqing.mortal.p4m11``, so it could not see
# ``"p4m11_result.json"`` -- and the P4-M12 run's summary really did land in a
# file called ``p4m11_result.json``.  Pinning the whole set means a new literal
# fails here instead of surviving to a run directory.
ARTIFACT_LITERAL = re.compile(r'"(?:keqing\.mortal\.)?p4m11[^"]*"')
# The one match that is not an artifact identity: a thread name shows up in
# thread dumps, never in an artifact, so it is not required to be config-driven.
NON_ARTIFACT_ALLOWED = frozenset({'"p4m11-resource-watchdog"'})

# Seed blocks that already belong to another experiment.  Training and evaluation
# blocks must be disjoint: reusing P4-M11's games would make this a re-run rather
# than a continuation, and reusing an evaluation block would break the "fresh
# evaluation seeds" requirement.
P4M11_TRAINING_SEEDS = (730000, 732047)
P4M11_GATE_SEEDS = {
    "gateA": (740000, 740255),
    "gateB": (741000, 741255),
}
P4M12_GATE_SEEDS = {
    "gateC (U64 vs U32)": (742000, 742255),
    "gateD (U64 vs external)": (743000, 743255),
}


@pytest.fixture
def installed():
    """Bind P4-M11's machinery to the P4-M12 config, then always restore it.

    Without the restore, a rebind would leak into every later test in the same
    process and silently re-label another module's work.
    """
    previous = p4m12.install_config()
    try:
        yield p4m12.P4M12_CONFIG
    finally:
        p4m12.restore_config(previous)


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


# --------------------------------------------------------------------------
# 1. the seam changes identity and seeds, and nothing else
# --------------------------------------------------------------------------


def test_only_the_authorised_keys_change():
    assert p4m12.changed_keys() == set(p4m12.CHANGED_KEYS)


def test_every_change_is_declared_and_every_declaration_is_real():
    """Both directions: no undeclared drift, and no no-op "override"."""
    p4m12.assert_only_identity_changed()  # must not raise

    # an undeclared recipe edit is refused
    previous = p4m12.P4M12_CONFIG
    try:
        drifted = copy.deepcopy(previous)
        drifted["optimizer"] = {**drifted["optimizer"], "lr": 3e-5}
        p4m12.P4M12_CONFIG = drifted
        with pytest.raises(p4m12.P4M12ContractError, match="would change"):
            p4m12.assert_only_identity_changed()

        # a declared-but-unreal change (a forgotten seed override) is refused too,
        # because then this would not be a continuation at all
        stale = copy.deepcopy(previous)
        stale["seed_start"] = p4m11.P4M11_CONFIG["seed_start"]
        p4m12.P4M12_CONFIG = stale
        with pytest.raises(p4m12.P4M12ContractError, match="identical to"):
            p4m12.assert_only_identity_changed()
    finally:
        p4m12.P4M12_CONFIG = previous


def test_the_inherited_recipe_is_bit_for_bit_p4m11s():
    """The blocks the reused collector reads must be the same objects by value."""
    for block in ("sampling", "return", "loss", "optimizer", "gradient_clip"):
        assert p4m12.P4M12_CONFIG[block] == p4m11.P4M11_CONFIG[block], block


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cycles", 32),
        ("seeds_per_cycle", 64),
        ("splits_per_seed", 4),
        ("hanchans_per_cycle", 256),
        ("max_hanchans", 8192),
        ("seed_key", 8192),
        ("update_micro_batch", 512),
        ("objective", "direct_on_policy_policy_gradient"),
    ],
)
def test_the_scale_of_the_run_is_unchanged(key, expected):
    assert p4m12.P4M12_CONFIG[key] == expected
    assert p4m11.P4M11_CONFIG[key] == expected


def test_sampling_and_return_constants_are_unchanged():
    cfg = p4m12.P4M12_CONFIG
    assert cfg["sampling"]["boltzmann_temp"] == 1.0
    assert cfg["sampling"]["boltzmann_epsilon"] == 1.0
    assert cfg["sampling"]["top_p"] == 1.0
    assert cfg["sampling"]["enable_amp"] is False
    assert cfg["return"]["scalar_baseline"] == 0.0
    assert list(cfg["return"]["rank_points_raw"]) == [90.0, 45.0, 0.0, -135.0]
    assert cfg["optimizer"]["lr"] == 1e-5
    assert cfg["gradient_clip"]["value"] == 1.0
    assert cfg["champion_sha256"] == p4m11.P4M11_CONFIG["champion_sha256"]


# --------------------------------------------------------------------------
# 2. parent identity and the endpoint
# --------------------------------------------------------------------------


def test_parent_is_the_p4m11_training_endpoint_not_its_export():
    assert p4m12.P4M12_CONFIG["parent"].endswith("U32.pth")
    assert "eval_weights" not in p4m12.P4M12_CONFIG["parent"]
    assert p4m12.P4M12_CONFIG["parent_sha256"] == (
        "511be9adba1e0a38a093cb59b498c6a7e7be031d0f5e09afb06f4b361690b9cd"
    )
    assert p4m12.P4M12_CONFIG["parent_completed_cycles"] == 32
    assert p4m12.P4M12_CONFIG["parent_inherited_adam_step"] == 36


def test_the_endpoint_is_the_parent_step_plus_the_cycles():
    cfg = p4m12.P4M12_CONFIG
    assert cfg["expected_final_adam_step"] == (
        cfg["parent_inherited_adam_step"] + cfg["cycles"]
    )
    assert cfg["expected_final_adam_step"] == 68


def test_completion_is_locked_to_u64_at_step_68(installed):
    done = p4m11.completion_status(final_cycle=32, final_steps=[68.0], parent_unchanged=True)
    assert done["complete"] is True
    assert done["endpoint_cycle"] == 32 and done["endpoint_step"] == 68.0

    short = p4m11.completion_status(final_cycle=31, final_steps=[67.0], parent_unchanged=True)
    assert short["complete"] is False
    assert short["reason"]

    wrong_step = p4m11.completion_status(final_cycle=32, final_steps=[36.0], parent_unchanged=True)
    assert wrong_step["complete"] is False

    parent_moved = p4m11.completion_status(final_cycle=32, final_steps=[68.0], parent_unchanged=False)
    assert parent_moved["complete"] is False


# --------------------------------------------------------------------------
# 3. fresh seeds
# --------------------------------------------------------------------------


def test_the_seed_block_is_fresh_and_the_right_size():
    start, end = p4m12.P4M12_CONFIG["seed_start"], p4m12.P4M12_CONFIG["seed_end"]
    assert end - start + 1 == 2048
    assert (end - start + 1) // p4m12.P4M12_CONFIG["seeds_per_cycle"] == 32
    assert p4m12.P4M12_CONFIG["sampling_seed_base"] != p4m11.P4M11_CONFIG["sampling_seed_base"]


def test_the_seed_block_overlaps_no_other_experiment():
    training = (p4m12.P4M12_CONFIG["seed_start"], p4m12.P4M12_CONFIG["seed_end"])
    assert not overlaps(training, P4M11_TRAINING_SEEDS), "reusing P4-M11's training games"
    for name, block in {**P4M11_GATE_SEEDS, **P4M12_GATE_SEEDS}.items():
        assert not overlaps(training, block), f"training block overlaps {name}"


def test_the_evaluation_blocks_are_fresh_too():
    training = (p4m12.P4M12_CONFIG["seed_start"], p4m12.P4M12_CONFIG["seed_end"])
    for name, block in P4M12_GATE_SEEDS.items():
        assert not overlaps(block, P4M11_TRAINING_SEEDS), name
        assert not overlaps(block, training), name
        for other_name, other in P4M11_GATE_SEEDS.items():
            assert not overlaps(block, other), f"{name} overlaps {other_name}"
    assert not overlaps(P4M12_GATE_SEEDS["gateC (U64 vs U32)"],
                        P4M12_GATE_SEEDS["gateD (U64 vs external)"])


def test_derived_seed_segments_and_sampling_seeds(installed):
    assert p4m11.seed_segment_for(1) == (733000, 733063)
    assert p4m11.seed_segment_for(2) == (733064, 733127)
    assert p4m11.seed_segment_for(32) == (734984, 735047)
    assert p4m11.sampling_seed_for(1) == 2026091601
    assert p4m11.sampling_seed_for(32) == 2026091632


def test_the_budget_guard_accepts_the_authorised_scale(installed):
    p4m11.assert_within_budget(32, 64)  # must not raise
    with pytest.raises(p4m11.P4M11ContractError, match="budget exceeded"):
        p4m11.assert_within_budget(64, 64)


# --------------------------------------------------------------------------
# 4. artifacts name P4-M12, and the rebind does not leak
# --------------------------------------------------------------------------


def test_identity_is_config_driven_rather_than_a_literal():
    """P4-M11's module must not hard-code an artifact name anywhere.

    A continuation writes its own artifacts through P4-M11's code; if a schema
    prefix or an output filename were a literal, a P4-M12 run would emit
    artifacts claiming to be P4-M11 ones.  The schema prefix was covered from the
    start; the result filename was not, and P4-M12's summary was written to
    ``p4m11_result.json`` because of it.

    This asserts a *count*, not set membership.  The first version of this test
    collected the literals into a set, and a negative control that reintroduced
    ``"p4m11_result.json"`` at the call site stayed green -- because the config
    line already contributed that name to the set.  One occurrence per name, on
    the config line, is the property that actually matters.
    """
    counts = collections.Counter(
        match.group(0) for match in ARTIFACT_LITERAL.finditer(P4M11_SOURCE)
    )
    assert set(counts) - NON_ARTIFACT_ALLOWED == {
        '"keqing.mortal.p4m11"',
        '"p4m11_candidate"',
        '"p4m11_result.json"',
    }, counts
    for literal, key in (
        ('"keqing.mortal.p4m11"', '"schema_prefix"'),
        ('"p4m11_candidate"', '"challenger_label"'),
        ('"p4m11_result.json"', '"result_filename"'),
    ):
        assert counts[literal] == 1, (literal, counts[literal])
        line = next(
            candidate for candidate in P4M11_SOURCE.splitlines()
            if literal in candidate
        )
        assert key in line, (literal, line.strip())


def test_every_artifact_name_literal_is_overridden_by_the_continuation():
    """Each pinned literal must have a matching override, or the run lies.

    Deliberately *not* the ``installed`` fixture: once installed the two names
    are the same dict object and the inequality below would be vacuous.
    """
    assert p4m11.P4M11_CONFIG is not p4m12.P4M12_CONFIG
    pairs = {
        '"keqing.mortal.p4m11"': ("schema_prefix", "keqing.mortal.p4m12"),
        '"p4m11_candidate"': ("challenger_label", "p4m12_candidate"),
        '"p4m11_result.json"': ("result_filename", "p4m12_result.json"),
    }
    for literal, (key, expected) in pairs.items():
        assert literal in P4M11_SOURCE, literal
        assert p4m12.P4M12_CONFIG[key] == expected, key
        assert p4m12.P4M12_CONFIG[key] != p4m11.P4M11_CONFIG[key], key


def test_training_contract_carries_the_p4m12_identity(installed):
    parent = {
        "version": 4,
        "conv_channels": 192,
        "num_blocks": 40,
        "parent_path": p4m12.P4M12_CONFIG["parent"],
        "parent_sha256": p4m12.P4M12_CONFIG["parent_sha256"],
    }
    contract = p4m11.training_contract(cycle=1, parent=parent)
    assert contract["experiment"] == "P4-M12"
    assert contract["lineage"] == "hard50k -> C4 -> U32 -> U64"
    assert contract["schema"] == "keqing.mortal.student_policy_v1"
    assert "P4-M12" in contract["comment"]
    assert contract["parent_sha256"] == p4m12.P4M12_CONFIG["parent_sha256"]
    assert contract["cycle"] == 1


def test_installing_and_restoring_config_does_not_leak():
    before = p4m11.P4M11_CONFIG
    previous = p4m12.install_config()
    try:
        assert p4m11.P4M11_CONFIG is p4m12.P4M12_CONFIG
        assert p4m11.P4M11_CONFIG["experiment"] == "P4-M12"
    finally:
        p4m12.restore_config(previous)
    assert p4m11.P4M11_CONFIG is before
    assert p4m11.P4M11_CONFIG["experiment"] == "P4-M11"
    assert p4m11.P4M11_CONFIG["expected_final_adam_step"] == 36


def test_importing_p4m12_has_no_side_effect_on_p4m11():
    """Import alone must never rebind; only main()/install_config() may."""
    assert p4m11.P4M11_CONFIG is not p4m12.P4M12_CONFIG
    assert p4m11.P4M11_CONFIG["experiment"] == "P4-M11"
    assert p4m12.P4M12_CONFIG["experiment"] == "P4-M12"
