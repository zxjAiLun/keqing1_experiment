"""Frozen constants and lineage helpers for D3 continuation shards.

Continuation is the same D3_top2_discard_v1 generation scaled to 23 further
B250 shards (1800250..1805999), NOT a new experiment. The first-B250 runner
(2cc12b4) stays frozen and reproducible; this module parameterizes only the
shard identity (seed range, output directory, confirmation token).

Lineage is three-layered:

    D3 semantic anchor        2cc12b4   (generation contract frozen)
    first-gate auditor        cf9bb86
    continuation governance   67f2ccb
    continuation runner       <HEAD at execution time>

A continuation preflight must prove 67f2ccb is an ancestor of HEAD, the D3
semantic files are unchanged since 2cc12b4, and the authoritative runtime
anchors (protocol / native / patch / models / Mortal) still match.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from training.mortal.d3_exploration_engine import CONTRACT_ID
from training.mortal.d3_production_contract import (
    AMP,
    AUTHORITATIVE_MORTAL_COMMIT,
    AUTHORITATIVE_NATIVE_BINARY_SHA256,
    AUTHORITATIVE_NATIVE_PATCH_SHA256,
    AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
    D3_SEMANTIC_PATHS,
    DEVICE,
    EXPECTED_BRANCH,
    RANK_POINTS,
    REQUIRED_LABELS,
    REPO_ROOT,
    SEAT_MODE,
    SEED_KEY,
    git_text,
)

CONTINUATION_SCHEMA = "keqing.mortal.d3_continuation_shard_protocol.v1"
CONTINUATION_AUDIT_SCHEMA = "keqing.mortal.d3_continuation_shard_audit.v2"
SHARD_COUNT = 23
GAMES_PER_SHARD = 250
PRODUCTION_SEED_START = 1_800_000

D3_SEMANTIC_ANCHOR = "2cc12b46f81850da11c6e669d1c54b039476b440"
FIRST_GATE_AUDITOR = "cf9bb86a40e1e52e24deea8d5b2af8ab12e1a63b"
CONTINUATION_GOVERNANCE = "67f2ccb96fe932abc5c2c4b889ad396d4f584823"

CONTINUATION_IMPLEMENTATION_PATHS = (
    "training/mortal/d3_continuation_contract.py",
    "training/mortal/d3_continuation_preflight.py",
    "training/mortal/run_d3_continuation_shard_2026_08.py",
    "training/mortal/audit_d3_continuation_shard_2026_08.py",
)
# Files whose change would alter the frozen D3 generation semantics.
CONTINUATION_SEMANTIC_PATHS = (
    "training/mortal/d3_exploration_engine.py",
    "training/mortal/patches/libriichi_d3_decision_context.patch",
    "training/mortal/d3_production_contract.py",
)

DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/experiments/model_pool_2026_07/"
    "D3_uncertainty_guided_exploration_2026_08/generation_continuation"
)


def shard_seed_start(shard_index: int) -> int:
    if not 1 <= shard_index <= SHARD_COUNT:
        raise ValueError(f"shard index must be in 1..{SHARD_COUNT}, got {shard_index}")
    return PRODUCTION_SEED_START + shard_index * GAMES_PER_SHARD


def shard_seed_end_exclusive(shard_index: int) -> int:
    return shard_seed_start(shard_index) + GAMES_PER_SHARD


def shard_dir_name(shard_index: int) -> str:
    start = shard_seed_start(shard_index)
    end_inclusive = shard_seed_end_exclusive(shard_index) - 1
    return f"shard_{shard_index:03d}_{start}_{end_inclusive}"


def shard_confirmation_token(shard_index: int) -> str:
    start = shard_seed_start(shard_index)
    end_inclusive = shard_seed_end_exclusive(shard_index) - 1
    return f"D3_CONTINUE_SHARD_{shard_index:03d}_{start}_{end_inclusive}_SINGLE_SHOT"


def shard_seed_keys(shard_index: int) -> set[tuple[int, int]]:
    return {
        (seed, SEED_KEY)
        for seed in range(shard_seed_start(shard_index), shard_seed_end_exclusive(shard_index))
    }


def shard_output_dir(shard_index: int) -> Path:
    return DEFAULT_OUTPUT_ROOT / shard_dir_name(shard_index)


def continuation_lineage(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Mechanically verified continuation lineage.

    Hard requirements:
      * current branch == main, worktree clean (via project_lineage)
      * CONTINUATION_GOVERNANCE is a Git ancestor of HEAD
      * D3 semantic files unchanged since D3_SEMANTIC_ANCHOR
      * authoritative anchors still referenced by the frozen contract
    """
    repo_root = repo_root.resolve()
    governance_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CONTINUATION_GOVERNANCE, "HEAD"],
        cwd=repo_root,
        capture_output=True,
    ).returncode == 0
    semantic_diff = git_text(
        repo_root,
        "diff",
        "--name-only",
        D3_SEMANTIC_ANCHOR,
        "--",
        *CONTINUATION_SEMANTIC_PATHS,
    ).splitlines()
    errors: list[str] = []
    if not governance_ancestor:
        errors.append(
            "continuation governance commit is not an ancestor of HEAD: "
            f"{CONTINUATION_GOVERNANCE}"
        )
    if semantic_diff:
        errors.append(
            f"D3 semantic files changed since {D3_SEMANTIC_ANCHOR}: {semantic_diff}"
        )
    return {
        "branch": git_text(repo_root, "branch", "--show-current"),
        "head": git_text(repo_root, "rev-parse", "HEAD"),
        "d3_semantic_anchor": D3_SEMANTIC_ANCHOR,
        "first_gate_auditor": FIRST_GATE_AUDITOR,
        "continuation_governance": CONTINUATION_GOVERNANCE,
        "governance_is_ancestor": governance_ancestor,
        "semantic_paths": list(CONTINUATION_SEMANTIC_PATHS),
        "semantic_diff_paths": semantic_diff,
        "errors": errors,
        "passed": not errors,
    }
