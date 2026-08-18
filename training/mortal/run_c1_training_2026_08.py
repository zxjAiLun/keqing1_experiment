#!/usr/bin/env python3
"""Expose the frozen C1 command without allowing unauthorized execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts/experiments/C1_corpus_cql_interaction_2026_08/"
    "training_implementation_2026_08/training_manifest.json"
)
ROUTES = ("M0_CQL_OFF", "D1_CQL_OFF")
SEEDS = (20260806, 20260807, 20260808)
TRAINING_AUTHORIZED = False
APPROVED_IMPLEMENTATION_COMMIT = None
AUTHORIZED_PREFLIGHT_SHA256 = None
TRAINING_AUTHORIZATION_NOTE = "not authorized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def load_manifest() -> dict[str, Any]:
    if not DEFAULT_MANIFEST.is_file():
        raise SystemExit(f"C1 training manifest is missing: {DEFAULT_MANIFEST}")
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def selected_run(manifest: dict[str, Any], route: str, seed: int) -> dict[str, Any]:
    for run in manifest.get("runs", []):
        if run.get("route") == route and int(run.get("seed")) == seed:
            return run
    raise SystemExit(f"C1 run is not present in the frozen manifest: {route}/{seed}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and not TRAINING_AUTHORIZED:
        raise SystemExit(
            "C1 training is not authorized: TRAINING_AUTHORIZED=false; "
            f"authorization note={TRAINING_AUTHORIZATION_NOTE}"
        )
    manifest = load_manifest()
    run = selected_run(manifest, args.route, args.seed)
    if args.print_command:
        print(run["future_training_command"])
    if args.execute:
        raise SystemExit("C1 training launcher reached an unreachable authorization branch")
    if not args.print_command:
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


if __name__ == "__main__":
    main()
