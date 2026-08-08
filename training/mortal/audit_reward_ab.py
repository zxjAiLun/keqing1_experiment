#!/usr/bin/env python3
"""Audit matched reward A/B checkpoints and their shared training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments/model_pool_2026_07/reward_ab_2026_07_epoch2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="20260718,20260719,20260720")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    groups = ["F_final_rank_mc_weights_only", "G_mortal_grp_delta_pt_weights_only"]
    rows: list[dict] = []
    for group in groups:
        for seed in seeds:
            run_dir = args.root / group / f"seed_{seed}"
            state_path = run_dir / "mortal.pth"
            archive_path = run_dir / "checkpoints" / "mortal_72000.pth"
            if not state_path.exists() or not archive_path.exists():
                raise SystemExit(f"missing final checkpoint for {group}/{seed}")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            contract = state.get("training_contract", {})
            with (run_dir / "config.toml").open("rb") as handle:
                run_config = tomllib.load(handle)
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "steps": int(state.get("steps", -1)),
                    "archive_exists": archive_path.exists(),
                    "reward_mode": contract.get("reward_mode"),
                    "grp_sha256": contract.get("reward", {}).get("grp", {}).get("sha256"),
                    "file_count": contract.get("dataset", {}).get("file_count"),
                    "file_index_sha256": contract.get("dataset", {}).get("file_index_sha256"),
                    "manifest_sha256": contract.get("dataset", {}).get("manifest_sha256"),
                    "num_epochs": contract.get("dataset", {}).get("num_epochs", run_config["dataset"]["num_epochs"]),
                    "data_seed": state.get("data_stream", {}).get("data_seed"),
                    "batches_consumed": state.get("data_stream", {}).get("batches_consumed"),
                    "initialization_mode": contract.get("initialization", {}).get("mode"),
                    "parent_sha256": contract.get("initialization", {}).get("parent_sha256"),
                    "git_commit": contract.get("git_commit"),
                    "git_dirty": contract.get("git_dirty"),
                }
            )

    if any(row["steps"] != 72000 for row in rows):
        raise SystemExit("one or more A/B checkpoints did not reach 72000")
    if {row["reward_mode"] for row in rows} != {"final_rank_mc", "mortal_grp_delta_pt"}:
        raise SystemExit("unexpected reward modes")
    invariant_keys = (
        "file_count", "file_index_sha256", "manifest_sha256", "num_epochs",
        "initialization_mode", "parent_sha256", "git_commit", "git_dirty",
    )
    for key in invariant_keys:
        if len({row[key] for row in rows}) != 1:
            raise SystemExit(f"matched contract invariant failed: {key}")
    if rows and rows[0]["git_dirty"] is not False:
        raise SystemExit("training contracts must record git_dirty=false")
    for seed in seeds:
        pair = [row for row in rows if row["seed"] == seed]
        if len(pair) != 2 or pair[0]["data_seed"] != pair[1]["data_seed"]:
            raise SystemExit(f"matched data seed invariant failed: {seed}")

    report = {
        "schema": "keqing.mortal.reward_ab_audit.v1",
        "root": str(args.root.resolve()),
        "matched_contract": True,
        "runs": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
