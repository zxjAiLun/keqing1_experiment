#!/usr/bin/env python3
"""Fail-closed C1 launcher with a fully implemented, authorization-only branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support both ``python -m ...`` and direct execution by path.
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from training.mortal.prepare_c1_training_2026_08 import (
    IMPLEMENTATION_SOURCES,
    K0_PARENT_SHA256,
    REPO_ROOT,
    ROUTES,
    SEEDS,
    future_training_argv,
    inspect_parent,
    load_toml,
    sha256_file,
)

DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/"
    "training_implementation_2026_08/training_manifest.json"
)
DEFAULT_PREFLIGHT = DEFAULT_MANIFEST.parent / "preflight/training_preflight.json"
TRAINING_AUTHORIZED = False
APPROVED_IMPLEMENTATION_COMMIT = None
AUTHORIZED_PREFLIGHT_SHA256 = None
AUTHORIZED_MANIFEST_SHA256 = None
TRAINING_AUTHORIZATION_NOTE = "not authorized"


class AuthorizationError(RuntimeError):
    """Raised when a launch-time authorization or provenance guard fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--confirmation-token")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorizationError(f"C1 training manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorizationError(f"C1 training preflight is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_run(manifest: dict[str, Any], route: str, seed: int) -> dict[str, Any]:
    for run in manifest.get("runs", []):
        if run.get("route") == route and int(run.get("seed")) == seed:
            return run
    raise AuthorizationError(f"C1 run is not present in the frozen manifest: {route}/{seed}")


def assert_frozen_run_matrix(manifest: dict[str, Any]) -> None:
    expected = {(route, seed) for route in ROUTES for seed in SEEDS}
    actual = {(str(run.get("route")), int(run.get("seed"))) for run in manifest.get("runs", [])}
    if actual != expected or len(manifest.get("runs", [])) != len(expected):
        raise AuthorizationError(f"C1 manifest run matrix mismatch: expected={expected}, actual={actual}")


def confirmation_token(*, route: str, seed: int, implementation_commit: str, preflight_sha256: str) -> str:
    if not implementation_commit or not preflight_sha256:
        raise AuthorizationError("confirmation token cannot be derived from empty authorization bindings")
    return f"C1_TRAIN_{route}_{seed}_{implementation_commit[:12]}_{preflight_sha256[:12]}"


def _git_blob_oid(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def current_implementation_sources() -> list[dict[str, str]]:
    records = []
    for relative_path in IMPLEMENTATION_SOURCES:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise AuthorizationError(f"implementation source is missing: {path}")
        records.append(
            {
                "path": relative_path,
                "content_sha256": sha256_file(path),
                "git_blob_oid": _git_blob_oid(path),
            }
        )
    return records


def assert_no_training_outputs(run_dir: Path) -> None:
    forbidden = [run_dir / "mortal.pth", run_dir / "mortal_best.pth"]
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        forbidden.extend(checkpoint_dir.glob("mortal_*.pth"))
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise AuthorizationError(f"training output/checkpoint already exists: {existing}")


def validate_authorized_launch(
    *,
    route: str,
    seed: int,
    supplied_token: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    if TRAINING_AUTHORIZED is not True:
        raise AuthorizationError(
            "C1 training is not authorized: TRAINING_AUTHORIZED=false; "
            f"authorization note={TRAINING_AUTHORIZATION_NOTE}"
        )
    if not APPROVED_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("approved implementation commit is not bound")
    if not AUTHORIZED_PREFLIGHT_SHA256:
        raise AuthorizationError("authorized preflight SHA is not bound")
    if not AUTHORIZED_MANIFEST_SHA256:
        raise AuthorizationError("authorized manifest SHA is not bound")
    expected_token = confirmation_token(
        route=route,
        seed=seed,
        implementation_commit=APPROVED_IMPLEMENTATION_COMMIT,
        preflight_sha256=AUTHORIZED_PREFLIGHT_SHA256,
    )
    if supplied_token != expected_token:
        raise AuthorizationError("confirmation token mismatch")

    preflight_path = DEFAULT_PREFLIGHT.resolve()
    if sha256_file(preflight_path) != AUTHORIZED_PREFLIGHT_SHA256:
        raise AuthorizationError("authorized preflight SHA mismatch")
    preflight = load_preflight(preflight_path)
    if preflight.get("passed") is not True:
        raise AuthorizationError("preflight is not passed")
    if preflight.get("optimizer_steps") != 0 or preflight.get("new_checkpoints") != 0:
        raise AuthorizationError("preflight records execution")
    if preflight.get("git", {}).get("commit") != APPROVED_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("preflight implementation commit mismatch")
    if preflight.get("checks", {}).get("formal_config_equals_smoke_config") is not True:
        raise AuthorizationError("preflight does not bind formal and smoke config")

    manifest_path = DEFAULT_MANIFEST.resolve()
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != AUTHORIZED_MANIFEST_SHA256:
        raise AuthorizationError("authorized manifest SHA mismatch")
    if preflight.get("manifest_sha256") != actual_manifest_sha256:
        raise AuthorizationError("preflight manifest SHA mismatch")
    manifest = load_manifest(manifest_path)
    if manifest.get("implementation_commit") != APPROVED_IMPLEMENTATION_COMMIT:
        raise AuthorizationError("manifest implementation commit mismatch")
    if manifest.get("status") != "prepared_not_authorized":
        raise AuthorizationError("manifest status is not prepared_not_authorized")
    if manifest.get("new_training_runs") != 6:
        raise AuthorizationError("manifest new_training_runs is not six")
    if manifest.get("training_authorized") is not False or manifest.get("evaluation_authorized") is not False:
        raise AuthorizationError("manifest authorization flags are not false")
    assert_frozen_run_matrix(manifest)
    run = selected_run(manifest, route, seed)

    config_path = Path(run["cql_off_config"]).resolve()
    if not config_path.is_file():
        raise AuthorizationError(f"formal CQL_OFF config is missing: {config_path}")
    config_sha256 = sha256_file(config_path)
    if config_sha256 != run.get("cql_off_config_sha256"):
        raise AuthorizationError("formal CQL_OFF config SHA mismatch")
    if run.get("formal_training_config_sha256") != config_sha256:
        raise AuthorizationError("formal training config SHA mismatch")
    if run.get("smoke_config_sha256") != config_sha256 or run.get("exact_same_config") is not True:
        raise AuthorizationError("formal training and smoke config are not exactly the same")
    config = load_toml(config_path)
    if float(config.get("cql", {}).get("min_q_weight", -1.0)) != 0.0:
        raise AuthorizationError("formal CQL_OFF config is not CQL zero")
    if config.get("objective", {}).get("mode") != "behavior_action_mc":
        raise AuthorizationError("formal objective mismatch")
    if config.get("reward", {}).get("mode") != "final_rank_mc":
        raise AuthorizationError("formal reward mismatch")
    if "allow_legacy_data_replay" in config:
        raise AuthorizationError("legacy replay override is present")

    parent_record = manifest.get("parent", {})
    parent_path = Path(parent_record.get("path", "")).resolve()
    if parent_record.get("sha256") != K0_PARENT_SHA256:
        raise AuthorizationError("manifest K0 parent SHA mismatch")
    parent = inspect_parent(parent_path)
    if parent["sha256"] != K0_PARENT_SHA256 or parent["digest"] != parent_record.get("digest"):
        raise AuthorizationError("K0 parent or preserved Adam digest mismatch")

    run_dir = Path(run["run_output_dir"]).resolve()
    for key, expected in {
        "state_file": str((run_dir / "mortal.pth").resolve()),
        "best_state_file": str((run_dir / "mortal_best.pth").resolve()),
        "tensorboard_dir": str((run_dir / "tb_mortal").resolve()),
    }.items():
        if config.get("control", {}).get(key) != expected:
            raise AuthorizationError(f"formal output path mismatch: {key}")
    assert_no_training_outputs(run_dir)

    if manifest.get("implementation_sources") != current_implementation_sources():
        raise AuthorizationError("implementation source content/blob binding mismatch")

    expected_argv = future_training_argv(
        config_path=config_path,
        parent_path=parent_path,
        seed=seed,
        run_dir=run_dir,
    )
    if run.get("future_training_argv") != expected_argv:
        raise AuthorizationError("future training argv differs from the frozen argv")
    if run.get("future_training_command") != shlex.join(expected_argv):
        raise AuthorizationError("human-readable command differs from authoritative argv")
    if "--allow-legacy-data-replay" in expected_argv:
        raise AuthorizationError("legacy replay override is present in argv")
    return manifest, preflight, run, expected_argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        manifest = load_manifest()
        run = selected_run(manifest, args.route, args.seed)
        command_argv = run.get("future_training_argv")
        if not isinstance(command_argv, list):
            raise SystemExit("manifest does not contain authoritative future_training_argv")
        if args.print_command:
            print(shlex.join(command_argv))
        else:
            print(
                json.dumps(
                    {
                        "experiment_id": manifest.get("experiment_id"),
                        "route": args.route,
                        "seed": args.seed,
                        "training_authorized": TRAINING_AUTHORIZED,
                        "command_available": True,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    try:
        _, _, run, command_argv = validate_authorized_launch(
            route=args.route,
            seed=args.seed,
            supplied_token=args.confirmation_token,
        )
    except (AuthorizationError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.print_command:
        print(shlex.join(command_argv))
    subprocess.run(command_argv, cwd=REPO_ROOT, check=True, shell=False)
    print(
        json.dumps(
            {
                "route": args.route,
                "seed": args.seed,
                "executed": True,
                "future_training_argv": run["future_training_argv"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    main()
