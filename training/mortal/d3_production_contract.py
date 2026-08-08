#!/usr/bin/env python3
"""Frozen constants and lineage helpers for the first D3 production B250 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from training.mortal.d3_exploration_engine import CONTRACT_ID

REPO_ROOT = Path(__file__).resolve().parents[2]

GATE_ID = "D3_first_B250_production_gate_2026_08"
PRODUCTION_SCHEMA = "keqing.mortal.d3_production_gate_protocol.v1"
AUDIT_SCHEMA = "keqing.mortal.d3_production_gate_audit.v1"
SEED_START = 1_800_000
GAMES = 250
SEED_END_EXCLUSIVE = SEED_START + GAMES
SEED_KEY = 8192
NATIVE_BATCH_GAMES = 250
SEAT_MODE = "random"
DEVICE = "cuda"
AMP = False
RANK_POINTS = (90.0, 45.0, 0.0, -135.0)
REQUIRED_LABELS = ("K0_70k", "V2_74000", "V3_74000", "ext_mortal")
SMOKE_SEEDS = frozenset(range(1_799_000, 1_799_025))
EXPECTED_BRANCH = "main"
CONFIRMATION_TOKEN = "D3_B250_1800000_1800249_SINGLE_SHOT"

AUTHORITATIVE_SMOKE_PROJECT_COMMIT = "7eb48f1310e917cfb2f1f45445e546b4e92d1a89"
AUTHORITATIVE_MORTAL_COMMIT = "813859fc8110ea178f56f009994bc4f1b9fee645"
AUTHORITATIVE_NATIVE_PATCH_SHA256 = (
    "fa7ff1e12687c3ba7ec4d5ea47902ded2c263f7695e0da195e0daabf437ffed1"
)
AUTHORITATIVE_NATIVE_BINARY_SHA256 = (
    "19bb181eaa70d0ae90417a3bd22433f6ca08d7654602f865ff3bdb102b7d9914"
)
# Legacy training lineage that the new repo acknowledges but does not re-join
# as Git ancestry. These record where the authoritative smoke actually ran and
# where the frozen old training tip lived, so provenance survives the migration.
# They are historical records only; they are never required to be Git
# ancestors of the current keqing1_experiment/main HEAD.
LEGACY_TRAINING_SOURCE_COMMIT = "6ff580cb"
# Migration lineage in keqing1_experiment proper (real Git ancestry).
MIGRATION_CONTENT_COMMIT = "8e3b58f50c08c3c9ad795ea63c1af44e3b5ed11b"
TRAINING_TRANSFER_ANCHOR = "74a3154d0c543b805a75e679ab93c74f2afbefaf"
D3_SEMANTIC_PATHS = (
    "training/mortal/d3_exploration_engine.py",
    "training/mortal/patches/libriichi_d3_decision_context.patch",
)
PRODUCTION_IMPLEMENTATION_PATHS = (
    "training/mortal/d3_production_contract.py",
    "training/mortal/d3_production_preflight.py",
    "training/mortal/run_d3_exploration_production_2026_08.py",
    "training/mortal/d3_production_audit_core.py",
    "training/mortal/d3_production_event_audit.py",
    "training/mortal/d3_production_replay_audit.py",
    "training/mortal/d3_production_lineage_audit.py",
    "training/mortal/d3_production_report.py",
    "training/mortal/audit_d3_exploration_production_2026_08.py",
)
DEFAULT_SMOKE_PROTOCOL = Path(
    "artifacts/experiments/model_pool_2026_07/"
    "D3_uncertainty_guided_exploration_2026_08/generation_smoke/"
    "authoritative_run_a/protocol.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/experiments/model_pool_2026_07/"
    "D3_uncertainty_guided_exploration_2026_08/generation_production/"
    "shard_000_1800000_1800249"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when the frozen production contract or lineage does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_text(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def expected_seed_keys() -> set[tuple[int, int]]:
    return {(SEED_START + offset, SEED_KEY) for offset in range(GAMES)}


def ignored_path_audit(paths: dict[str, Path], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Record whether relevant local artifacts are ignored or external to the project tree."""

    repo_root = repo_root.resolve()
    rows: dict[str, dict[str, Any]] = {}
    for label, original in paths.items():
        path = original.resolve()
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            rows[label] = {
                "path": str(path),
                "external_to_repo": True,
                "ignored": None,
                "ignore_rule": None,
            }
            continue
        completed = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", str(relative)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        rows[label] = {
            "path": str(path),
            "relative_path": str(relative),
            "external_to_repo": False,
            "ignored": completed.returncode == 0,
            "ignore_rule": completed.stdout.strip() or None,
        }
    return rows


def assert_empty_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ContractError(f"production output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise ContractError(
            f"production output must be absent or empty; delete the whole shard before retry: {path}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def validate_authoritative_smoke_protocol(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        protocol.get("schema") == "keqing.mortal.d3_generation_smoke_protocol.v1",
        "authoritative smoke protocol schema mismatch",
    )
    require(protocol.get("contract_id") == CONTRACT_ID, "authoritative smoke contract mismatch")
    require(protocol.get("seed_start") == 1_799_000, "authoritative smoke seed_start mismatch")
    require(protocol.get("seed_end_exclusive") == 1_799_025, "authoritative smoke seed_end mismatch")
    require(protocol.get("seed_key") == SEED_KEY, "authoritative smoke seed_key mismatch")
    require(protocol.get("games") == 25, "authoritative smoke games mismatch")
    require(protocol.get("native_batch_games") == 25, "authoritative smoke batch mismatch")
    require(protocol.get("seat_mode") == SEAT_MODE, "authoritative smoke seat mode mismatch")
    require(protocol.get("device") == DEVICE, "authoritative smoke device mismatch")
    require(protocol.get("amp") is AMP, "authoritative smoke AMP mismatch")
    require(protocol.get("project_git_dirty") is False, "authoritative smoke project was dirty")
    require(protocol.get("mortal_source_dirty") is False, "authoritative smoke Mortal was dirty")
    require(
        protocol.get("project_git_commit") == AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
        "authoritative smoke project commit mismatch",
    )
    require(
        protocol.get("mortal_source_commit") == AUTHORITATIVE_MORTAL_COMMIT,
        "authoritative smoke Mortal commit mismatch",
    )
    require(
        protocol.get("d3_native_patch_sha256") == AUTHORITATIVE_NATIVE_PATCH_SHA256,
        "authoritative smoke native patch mismatch",
    )
    require(
        protocol.get("loaded_libriichi_sha256") == AUTHORITATIVE_NATIVE_BINARY_SHA256,
        "authoritative smoke native binary mismatch",
    )
    require(protocol.get("native_build_profile") == "release", "authoritative smoke build is not release")
    models = protocol.get("models")
    require(isinstance(models, dict), "authoritative smoke model manifest missing")
    if isinstance(models, dict):
        require(set(models) == set(REQUIRED_LABELS), "authoritative smoke model labels mismatch")
        for label in REQUIRED_LABELS:
            row = models.get(label)
            require(isinstance(row, dict), f"authoritative smoke model row missing: {label}")
            if isinstance(row, dict):
                require(bool(row.get("path")), f"authoritative smoke model path missing: {label}")
                require(_valid_sha256(row.get("sha256")), f"authoritative smoke model SHA invalid: {label}")
    require(
        protocol.get("engine_order") == list(REQUIRED_LABELS),
        "authoritative smoke engine order mismatch",
    )
    return errors


def load_authoritative_smoke_protocol(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"authoritative smoke protocol not found: {path}")
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError(f"authoritative smoke protocol is not an object: {path}")
    errors = validate_authoritative_smoke_protocol(protocol)
    if errors:
        raise ContractError("; ".join(errors))
    return protocol, sha256_file(path)


def parse_model_specs(
    specs: Iterable[str] | None,
    *,
    authoritative_models: dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    supplied: dict[str, Path] = {}
    if specs:
        for spec in specs:
            if "=" not in spec:
                raise ContractError(f"model spec must be LABEL=PATH, got {spec!r}")
            label, raw_path = spec.split("=", 1)
            label = label.strip()
            raw_path = raw_path.strip()
            if not label or not raw_path or label in supplied:
                raise ContractError(f"invalid or duplicate model spec: {spec!r}")
            supplied[label] = Path(raw_path)
        if set(supplied) != set(REQUIRED_LABELS):
            raise ContractError(
                f"production requires exactly {list(REQUIRED_LABELS)}, got {sorted(supplied)}"
            )
    else:
        supplied = {
            label: Path(str(authoritative_models[label]["path"])) for label in REQUIRED_LABELS
        }

    resolved: dict[str, Path] = {}
    for label in REQUIRED_LABELS:
        path = supplied[label]
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise ContractError(f"checkpoint not found for {label}: {path}")
        actual_sha = sha256_file(path)
        expected_sha = str(authoritative_models[label]["sha256"])
        if actual_sha != expected_sha:
            raise ContractError(
                f"checkpoint SHA differs from authoritative smoke for {label}: "
                f"expected={expected_sha} actual={actual_sha} path={path}"
            )
        resolved[label] = path
    return resolved


def implementation_manifest(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, str]]:
    repo_root = repo_root.resolve()
    manifest: dict[str, dict[str, str]] = {}
    for relative in PRODUCTION_IMPLEMENTATION_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise ContractError(f"production implementation file missing: {path}")
        manifest[relative] = {"path": str(path), "sha256": sha256_file(path)}
    return manifest


def _project_lineage_facts(
    *,
    branch: str,
    commit: str,
    dirty_entries: list[str],
    transfer_anchor_is_ancestor: bool,
    semantic_diff_paths: list[str],
) -> dict[str, Any]:
    """Pure lineage decision logic, isolated from git so tests can inject facts.

    The authoritative smoke project commit (``AUTHORITATIVE_SMOKE_PROJECT_COMMIT``)
    is deliberately NOT required to be a Git ancestor here: the experiment repo
    inherited the old ``mortal-training-next`` mainline by content migration, not
    by chaining the old commit SHAs. That commit is retained as historical
    provenance (``authoritative_smoke_commit``) but the mechanically enforced
    lineage is the new repo's transfer anchor.
    """
    errors: list[str] = []
    if branch != EXPECTED_BRANCH:
        errors.append(f"project branch must be {EXPECTED_BRANCH}, got {branch!r}")
    if dirty_entries:
        errors.append(f"project worktree is dirty: {dirty_entries[:20]}")
    if not transfer_anchor_is_ancestor:
        errors.append("current HEAD is not a descendant of the training transfer anchor")
    if semantic_diff_paths:
        errors.append(f"D3 semantic paths changed since transfer anchor: {semantic_diff_paths}")
    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
        "transfer_anchor": TRAINING_TRANSFER_ANCHOR,
        "transfer_anchor_is_ancestor": transfer_anchor_is_ancestor,
        "legacy_training_source_commit": LEGACY_TRAINING_SOURCE_COMMIT,
        "authoritative_smoke_commit": AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
        "migration_content_commit": MIGRATION_CONTENT_COMMIT,
        "semantic_paths": list(D3_SEMANTIC_PATHS),
        "semantic_diff_paths": semantic_diff_paths,
        "errors": errors,
        "passed": not errors,
    }


def project_lineage(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    branch = git_text(repo_root, "branch", "--show-current")
    commit = git_text(repo_root, "rev-parse", "HEAD")
    dirty_entries = git_text(repo_root, "status", "--porcelain").splitlines()
    transfer_anchor_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TRAINING_TRANSFER_ANCHOR, "HEAD"],
        cwd=repo_root,
        capture_output=True,
    ).returncode == 0
    semantic_diff = git_text(
        repo_root,
        "diff",
        "--name-only",
        TRAINING_TRANSFER_ANCHOR,
        "--",
        *D3_SEMANTIC_PATHS,
    ).splitlines()
    return _project_lineage_facts(
        branch=branch,
        commit=commit,
        dirty_entries=dirty_entries,
        transfer_anchor_is_ancestor=transfer_anchor_is_ancestor,
        semantic_diff_paths=semantic_diff,
    )


def mortal_lineage(native_root: Path) -> dict[str, Any]:
    native_root = native_root.resolve()
    commit = git_text(native_root, "rev-parse", "HEAD")
    parent = git_text(native_root, "rev-parse", "HEAD^")
    dirty_entries = git_text(native_root, "status", "--porcelain").splitlines()
    errors: list[str] = []
    if commit != AUTHORITATIVE_MORTAL_COMMIT:
        errors.append(
            f"nested Mortal commit mismatch: expected={AUTHORITATIVE_MORTAL_COMMIT} actual={commit}"
        )
    if dirty_entries:
        errors.append(f"nested Mortal worktree is dirty: {dirty_entries[:20]}")
    return {
        "root": str(native_root),
        "commit": commit,
        "parent_commit": parent,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
        "errors": errors,
        "passed": not errors,
    }


def archive_mortal_lineage(native_root: Path, output_dir: Path) -> dict[str, Any]:
    """Persist the local-only native commit without depending on a remote ref."""

    output_dir.mkdir(parents=True, exist_ok=False)
    patch_path = output_dir / f"mortal_{AUTHORITATIVE_MORTAL_COMMIT[:7]}.patch"
    patch_path.write_bytes(git_bytes(native_root, "format-patch", "-1", "--stdout", "HEAD"))
    show_path = output_dir / "mortal_commit_show.txt"
    show_path.write_text(
        git_text(native_root, "show", "--format=fuller", "--stat", "HEAD") + "\n",
        encoding="utf-8",
    )
    return {
        "format_patch": str(patch_path),
        "format_patch_sha256": sha256_file(patch_path),
        "commit_show": str(show_path),
        "commit_show_sha256": sha256_file(show_path),
    }
