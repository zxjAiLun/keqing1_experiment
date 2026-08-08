#!/usr/bin/env python3
"""Validate the frozen M0/D1 contract and run six zero-step CUDA smokes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from prepare_d1_training_2026_07 import (  # noqa: E402
    REPO_ROOT,
    SEED_VALUES,
    sha256_file,
    tensor_digest,
)


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value["control"].pop(key, None)
    for key in ("globs", "file_index", "player_names_files"):
        value["dataset"].pop(key, None)
    value.pop("experiment", None)
    return value


def load_file_list(path: Path) -> list[str]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 6000:
        raise ValueError(f"expected 6000 file paths in {path}")
    if len(set(str(value) for value in values)) != 6000:
        raise ValueError(f"duplicate file paths in {path}")
    for value in values:
        path_value = Path(str(value))
        path_value = path_value if path_value.is_absolute() else REPO_ROOT / path_value
        if not path_value.is_file():
            raise FileNotFoundError(path_value)
    return [str(value) for value in values]


def run_preview(*, config: Path, seed: int, output: Path, repeat: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    preview_script = REPO_ROOT / "scripts/mortal/preview_dataloader_batches_2026_07.py"
    for index in range(repeat):
        path = output.parent / f"{output.stem}_repeat_{index}.json"
        command = [
            sys.executable,
            str(preview_script),
            "--config",
            str(config),
            "--data-seed",
            str(seed),
            "--batch-count",
            "3",
            "--output",
            str(path),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        reports.append(load_json(path))
    first = reports[0]["batches"]
    if any(report["batches"] != first for report in reports[1:]):
        raise RuntimeError(f"dataloader batch hashes are not deterministic: {config}")
    output.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reports


def run_zero_smoke(*, config: Path, parent: Path, seed: int, output: Path) -> dict[str, Any]:
    script = REPO_ROOT / "scripts/mortal/zero_step_mortal_smoke_2026_07.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--parent",
            str(parent),
            "--data-seed",
            str(seed),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return load_json(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D1_project_owned_population_2026_07/training_prep_2026_07/training_manifest.json",
    )
    parser.add_argument("--skip-zero-step", action="store_true", help="validate contract only; never use for formal start")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    contract = load_json(manifest_path)
    if contract.get("status") != "prepared_not_started":
        raise ValueError(f"unexpected training preparation status: {contract.get('status')}")

    git = git_info()
    frozen_git = contract["git"]
    if git["branch"] != "codex/mortal-training-next" or git["dirty"]:
        raise SystemExit(f"preflight requires clean codex/mortal-training-next, got {git}")
    if frozen_git["branch"] != git["branch"] or frozen_git["commit"] != git["commit"] or frozen_git["dirty"]:
        raise SystemExit(f"training manifest was not generated from this clean commit: frozen={frozen_git}, current={git}")

    parent = Path(contract["protocol"]["parent_checkpoint"]).resolve()
    if not parent.is_file() or sha256_file(parent) != contract["protocol"]["parent_sha256"]:
        raise SystemExit("parent checkpoint SHA256 does not match frozen contract")
    state = torch.load(parent, weights_only=False, map_location="cpu")
    if int(state.get("steps", -1)) != 70000:
        raise SystemExit("parent checkpoint step is not 70000")
    for key in ("mortal", "current_dqn", "aux_net", "optimizer"):
        if key not in state:
            raise SystemExit(f"parent checkpoint missing {key}")
    parent_digest = contract["protocol"]["parent_tensor_digest"]
    actual_parent_digest = {
        "checkpoint_sha256": sha256_file(parent),
        "mortal_sha256": tensor_digest(state["mortal"]),
        "current_dqn_sha256": tensor_digest(state["current_dqn"]),
        "aux_net_sha256": tensor_digest(state["aux_net"]),
        "optimizer_sha256": tensor_digest(state["optimizer"]),
        "optimizer_state_count": len(state["optimizer"]["state"]),
        "steps": int(state["steps"]),
    }
    if actual_parent_digest != parent_digest:
        raise SystemExit(f"parent tensor/optimizer digest mismatch: {actual_parent_digest} != {parent_digest}")
    del state

    datasets = contract["datasets"]
    dataset_hashes: dict[str, list[str]] = {}
    for route, dataset in datasets.items():
        index_path = Path(dataset["file_index"]).resolve()
        if sha256_file(index_path) != dataset["file_index_sha256"]:
            raise SystemExit(f"{route} file index SHA mismatch")
        files = load_file_list(index_path)
        manifest_path_value = Path(dataset["content_manifest"]["path"]).resolve()
        if sha256_file(manifest_path_value) != dataset["content_manifest_sha256"]:
            raise SystemExit(f"{route} content manifest SHA mismatch")
        content = load_json(manifest_path_value)
        if int(content["file_count"]) != 6000 or len(content["rows"]) != 6000:
            raise SystemExit(f"{route} content manifest count mismatch")
        dataset_hashes[route] = [row["canonical_hanchan_sha256"] for row in content["rows"]]
        labels = dataset["trainable_label"]
        if route == "D1_variant":
            audit_path = Path(dataset["dataset_audit"]).resolve()
            generation_path = Path(dataset["generation_manifest"]).resolve()
            if sha256_file(audit_path) != dataset["dataset_audit_sha256"] or not load_json(audit_path)["summary"]["passed"]:
                raise SystemExit("D1 aggregate audit is not the frozen passing audit")
            if sha256_file(generation_path) != dataset["generation_manifest_sha256"]:
                raise SystemExit("D1 generation manifest SHA mismatch")
        if not labels:
            raise SystemExit(f"empty trainable label for {route}")
        if len(files) != 6000:
            raise SystemExit(f"{route} file count mismatch")
    if set(dataset_hashes["M0_control"]) & set(dataset_hashes["D1_variant"]):
        raise SystemExit("M0 and D1 content manifests unexpectedly share canonical hanchans")

    configs = contract["configs"]
    if len(configs) != 6:
        raise SystemExit(f"expected six configs, got {len(configs)}")
    normalized: list[dict[str, Any]] = []
    config_paths: dict[tuple[str, int], Path] = {}
    for item in configs:
        config_path = Path(item["config"]).resolve()
        if sha256_file(config_path) != item["config_sha256"]:
            raise SystemExit(f"config SHA mismatch: {config_path}")
        import tomllib

        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if config["objective"]["mode"] != "behavior_action_mc" or config["reward"]["mode"] != "final_rank_mc":
            raise SystemExit(f"objective/reward contract mismatch: {config_path}")
        route = str(item["route"])
        label = str(item["label"])
        expected = datasets[route]
        if label != expected["trainable_label"]:
            raise SystemExit(f"label mismatch for {config_path}")
        if Path(config["dataset"]["file_index"]).resolve() != Path(expected["file_index"]).resolve():
            raise SystemExit(f"file index path mismatch for {config_path}")
        if config["control"]["enable_amp"]:
            raise SystemExit("AMP must be disabled")
        normalized.append(normalize_config(config))
        config_paths[(route, int(item["seed"]))] = config_path
    if any(value != normalized[0] for value in normalized[1:]):
        raise SystemExit("M0/D1 configs differ outside the explicit route/path/label metadata")

    prep_dir = manifest_path.parent
    preview_reports: dict[str, Any] = {}
    for route, seed in (("M0_control", SEED_VALUES[0]), ("D1_variant", SEED_VALUES[0])):
        preview_path = prep_dir / "preflight" / f"{route.lower()}_batch_preview.json"
        repeats = run_preview(config=config_paths[(route, seed)], seed=seed, output=preview_path, repeat=2)
        preview_reports[route] = repeats[0]
    if preview_reports["M0_control"]["batches"][0]["sha256"] == preview_reports["D1_variant"]["batches"][0]["sha256"]:
        raise SystemExit("M0 and D1 first batch hashes unexpectedly match")

    smoke_reports: list[dict[str, Any]] = []
    if not args.skip_zero_step:
        for item in configs:
            route = str(item["route"])
            seed = int(item["seed"])
            config_path = config_paths[(route, seed)]
            run_dir = Path(item["run_dir"]).resolve()
            for state_file in (run_dir / "mortal.pth", run_dir / "mortal_best.pth"):
                if state_file.exists():
                    raise SystemExit(f"formal state file already exists before smoke: {state_file}")
            output = prep_dir / "preflight" / f"zero_step_{route.lower()}_{seed}.json"
            smoke = run_zero_smoke(config=config_path, parent=parent, seed=seed, output=output)
            if not smoke.get("finite") or smoke.get("optimizer_step_performed") or smoke.get("state_file_written"):
                raise SystemExit(f"zero-step smoke contract failed: {output}")
            smoke_reports.append(smoke)

    report = {
        "schema": "keqing.mortal.d1_training_preflight.v1",
        "passed": True,
        "manifest": str(manifest_path),
        "git": git,
        "parent_digest": actual_parent_digest,
        "preview": preview_reports,
        "zero_step_smokes": smoke_reports,
        "checks": {
            "clean_training_branch": True,
            "parent_step_70000": True,
            "parent_weights_and_optimizer_present": True,
            "m0_d1_counts_and_digests": True,
            "m0_d1_content_disjoint": True,
            "config_equality_outside_route_paths": True,
            "within_arm_batch_determinism": True,
            "cross_arm_batch_hashes_differ": True,
            "six_zero_step_cuda_smokes": args.skip_zero_step is False,
        },
    }
    output_path = prep_dir / "preflight" / "preflight_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(output_path), "smokes": len(smoke_reports)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
