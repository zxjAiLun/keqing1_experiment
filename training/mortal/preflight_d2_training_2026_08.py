#!/usr/bin/env python3
"""Validate D2's one-perspective-per-hanchan contract before training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260806, 20260807, 20260808)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_info():
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD"), "dirty": bool(status)}


def file_list(path: Path) -> list[str]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    values = payload.get("file_list") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"invalid file index: {path}")
    resolved = [str((Path(str(value)) if Path(str(value)).is_absolute() else REPO_ROOT / str(value)).resolve()) for value in values]
    if len(resolved) != len(set(resolved)) or any(not Path(value).is_file() for value in resolved):
        raise ValueError(f"file index has duplicates or missing files: {path}")
    return resolved


def normalized_config(config: dict):
    value = copy.deepcopy(config)
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value["control"].pop(key, None)
    value["experiment"].pop("training_seed", None)
    return value


def run_preview(config: Path, seed: int, output: Path) -> dict:
    script = REPO_ROOT / "scripts/mortal/preview_dataloader_batches_2026_07.py"
    reports = []
    for index in range(2):
        repeat_path = output.parent / f"{output.stem}_repeat_{index}.json"
        subprocess.run(
            [sys.executable, str(script), "--config", str(config), "--data-seed", str(seed), "--batch-count", "3", "--output", str(repeat_path)],
            cwd=REPO_ROOT,
            check=True,
        )
        reports.append(load_json(repeat_path))
    if reports[0]["batches"] != reports[1]["batches"]:
        raise SystemExit(f"D2 dataloader is not deterministic: {config}")
    output.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reports[0]


def run_zero_step(config: Path, parent: Path, seed: int, output: Path) -> dict:
    script = REPO_ROOT / "scripts/mortal/zero_step_mortal_smoke_2026_07.py"
    subprocess.run(
        [sys.executable, str(script), "--config", str(config), "--parent", str(parent), "--data-seed", str(seed), "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
    )
    report = load_json(output)
    if not report.get("finite") or report.get("optimizer_step_performed") or report.get("state_file_written"):
        raise SystemExit(f"D2 zero-step contract failed: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D2_project_owned_descendant_view_mix_2026_08/training_prep_2026_08/training_manifest.json",
    )
    parser.add_argument("--skip-zero-step", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    contract = load_json(manifest_path)
    if contract.get("status") != "prepared_not_started":
        raise ValueError(f"unexpected D2 status: {contract.get('status')}")
    git = git_info()
    if git["branch"] != "codex/mortal-training-next" or git["dirty"]:
        raise SystemExit(f"D2 preflight requires a clean training branch: {git}")
    if contract["git"]["commit"] != git["commit"] or contract["git"]["dirty"]:
        raise SystemExit(f"D2 manifest was not generated from the current clean commit: {contract['git']} vs {git}")

    parent = Path(contract["protocol"]["parent_checkpoint"]).resolve()
    if not parent.is_file() or sha256_file(parent) != contract["protocol"]["parent_sha256"]:
        raise SystemExit("D2 parent SHA mismatch")
    parent_state = torch.load(parent, weights_only=False, map_location="cpu")
    if int(parent_state.get("steps", -1)) != 70000 or "optimizer" not in parent_state:
        raise SystemExit("D2 parent must be a step-70000 checkpoint with Adam state")
    del parent_state

    dataset = contract["dataset"]
    d2_index = Path(dataset["file_index"]).resolve()
    v2_index = Path(dataset["v2_file_index"]).resolve()
    v3_index = Path(dataset["v3_file_index"]).resolve()
    d2_files = file_list(d2_index)
    v2_files = file_list(v2_index)
    v3_files = file_list(v3_index)
    if len(d2_files) != 6000 or len(v2_files) != 3000 or len(v3_files) != 3000:
        raise SystemExit("D2 index counts are not 6000/3000/3000")
    if set(v2_files) & set(v3_files) or set(v2_files) | set(v3_files) != set(d2_files):
        raise SystemExit("D2 V2/V3 indexes are not a disjoint partition of the D2 index")
    mapping_path = Path(contract["view_assignment"]["mapping"]).resolve()
    mapping = load_json(mapping_path)
    normalized_mapping = {str(Path(str(key)).resolve()): str(value) for key, value in mapping.items()}
    if set(normalized_mapping) != set(d2_files):
        raise SystemExit("D2 per-file mapping does not cover exactly the D2 index")
    counts = {label: sum(value == label for value in normalized_mapping.values()) for label in ("V2_74000", "V3_74000")}
    if counts != {"V2_74000": 3000, "V3_74000": 3000}:
        raise SystemExit(f"D2 mapping counts are wrong: {counts}")
    if sha256_file(mapping_path) != contract["view_assignment"]["mapping_sha256"]:
        raise SystemExit("D2 mapping SHA mismatch")

    prep_dir = manifest_path.parent
    config_paths = []
    configs = []
    for item in contract["configs"]:
        seed = int(item["seed"])
        if seed not in SEEDS or seed in [entry["seed"] for entry in configs]:
            raise SystemExit(f"unexpected/duplicate D2 seed: {seed}")
        config_path = Path(item["config"]).resolve()
        if sha256_file(config_path) != item["config_sha256"]:
            raise SystemExit(f"D2 config SHA mismatch: {config_path}")
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if config["objective"]["mode"] != "behavior_action_mc" or config["reward"]["mode"] != "final_rank_mc":
            raise SystemExit("D2 objective/reward mismatch")
        if Path(config["dataset"]["file_index"]).resolve() != d2_index or Path(config["dataset"]["player_names_by_file"]).resolve() != mapping_path:
            raise SystemExit(f"D2 config dataset path mismatch: {config_path}")
        if config["control"]["enable_amp"]:
            raise SystemExit("D2 requires AMP=false")
        config_paths.append((seed, config_path))
        configs.append({"seed": seed, "config": config_path, "normalized": normalized_config(config)})
    if len(configs) != 3 or any(item["normalized"] != configs[0]["normalized"] for item in configs[1:]):
        raise SystemExit("D2 configs differ outside per-run state paths and seed metadata")

    previews = {}
    smokes = []
    for seed, config_path in config_paths:
        previews[str(seed)] = run_preview(config_path, seed, prep_dir / "preflight" / f"d2_batch_preview_{seed}.json")
        if not args.skip_zero_step:
            smokes.append(run_zero_step(config_path, parent, seed, prep_dir / "preflight" / f"zero_step_d2_{seed}.json"))

    report = {
        "schema": "keqing.mortal.d2_training_preflight.v1",
        "passed": True,
        "manifest": str(manifest_path),
        "git": git,
        "checks": {
            "clean_training_branch": True,
            "parent_step_70000_and_optimizer": True,
            "d2_index_6000": True,
            "v2_v3_partition_3000_each": True,
            "single_perspective_mapping_exact": True,
            "mapping_sha": True,
            "config_equality": True,
            "within_seed_batch_determinism": True,
            "three_zero_step_cuda_smokes": not args.skip_zero_step,
        },
        "counts": {"d2_files": 6000, **counts},
        "previews": previews,
        "zero_step_smokes": smokes,
    }
    output = prep_dir / "preflight" / "preflight_d2_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(output), "zero_step_smokes": len(smokes)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
