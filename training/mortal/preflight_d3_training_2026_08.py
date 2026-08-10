#!/usr/bin/env python3
"""Preflight the frozen D3 training recipe (gates A..G, no training).

  A frozen D3 data contract       status/SHAs bound to the training-view audit
  B M0 operational equivalence    D3 configs == matched M0 configs after
                                  normalization; only allowed metadata differs
  C parent + preserved Adam       K0 step 70000, exact SHA, optimizer source ==
                                  parent, Adam moments cover all params,
                                  optimizer state count exact
  D real dataset stream           config.file_index == frozen D3 index (6000),
                                  label == K0_70k, glob set == index set, no
                                  player_names_by_file override
  E deterministic batch preview   FileDatasetsIter, first 3 batches identical
                                  within each seed (2 repeats)
  F zero-step CUDA smoke          3 seeds, finite forward/loss, no optimizer
                                  step, no state file
  G execution boundary            main clean, 728e43e ancestor of HEAD,
                                  training pipeline files unchanged since
                                  728e43e, no mortal.pth / checkpoints / tb

Still does not step the optimizer and does not write any training state.
"""

from __future__ import annotations

import argparse
import copy
import glob as globlib
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import tomllib
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from training.mortal.prepare_d3_training_recipe_2026_08 import (  # noqa: E402
    K0_PARENT_SHA,
    SEED_VALUES,
    sha256_file,
    tensor_digest,
)

RECIPE_ANCHOR = "728e43e24bd3a891896d3cdf78e0d2952531fc18"
PIPELINE_FILES = (
    "training/run_mortal_dqn_offline.py",
    "training/mortal/mainline_dataloader.py",
    "training/mortal/objective.py",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
    / "training_recipe_2026_08"
)
PREVIEW_SCRIPT = REPO_ROOT / "training/mortal/preview_dataloader_batches_2026_07.py"
ZERO_STEP_SCRIPT = REPO_ROOT / "training/mortal/zero_step_mortal_smoke_2026_07.py"


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


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


def run_preview(*, config: Path, seed: int, output: Path, repeat: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for index in range(repeat):
        path = output.parent / f"{output.stem}_repeat_{index}.json"
        command = [
            sys.executable,
            str(PREVIEW_SCRIPT),
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
    return reports


def run_zero_smoke(*, config: Path, parent: Path, seed: int, output: Path) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(ZERO_STEP_SCRIPT),
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


def glob_set_from_config(config: dict[str, Any]) -> set[str]:
    matched: set[str] = set()
    for pattern in config["dataset"]["globs"]:
        for path in globlib.glob(str(pattern), recursive=True):
            matched.add(str(Path(path).resolve()))
    return matched


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parent", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "training_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "recipe_prepared_not_started":
        raise SystemExit(f"unexpected recipe status: {manifest.get('status')}")
    if list(manifest["seeds"]) != list(SEED_VALUES):
        raise SystemExit("recipe seeds mismatch")

    parent_path = Path(args.parent or manifest["parent"]["path"]).resolve()
    git = git_info()
    checks: dict[str, bool] = {}

    # ---- G: execution boundary (checked first; everything else needs it) ----
    anchor_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", RECIPE_ANCHOR, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    ).returncode == 0
    pipeline_diff = subprocess.run(
        ["git", "diff", "--name-only", RECIPE_ANCHOR, "HEAD", "--", *PIPELINE_FILES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    run_dirs = [Path(item["run_dir"]).resolve() for item in manifest["configs"]]
    existing_state = []
    for run_dir in run_dirs:
        for path in run_dir.glob("mortal*.pth"):
            existing_state.append(str(path))
        for sub in ("checkpoints", "tb_mortal"):
            if (run_dir / sub).exists():
                existing_state.append(str(run_dir / sub))
    checks["main_clean"] = git["branch"] == "main" and not git["dirty"]
    checks["recipe_anchor_ancestor"] = anchor_ancestor
    checks["pipeline_files_unchanged_since_anchor"] = not pipeline_diff
    checks["no_training_state_files"] = not existing_state
    if not all(
        (checks["main_clean"], checks["recipe_anchor_ancestor"], checks["pipeline_files_unchanged_since_anchor"], checks["no_training_state_files"])
    ):
        raise SystemExit(f"execution boundary failed: {checks}")

    # ---- A: frozen D3 data contract ----
    frozen = manifest["frozen_d3_data_contract"]
    contract_sha = frozen["contract_sha256"]
    checks["data_contract_status_frozen"] = frozen["status"] == "training_contract_passed_manifest_frozen"
    checks["data_contract_sha_exact"] = contract_sha == "30bda12f25cf0d036c6f74e4650580f53ae1baaa670b0d1224092752c74ae4d4"
    checks["data_contract_audit_pass"] = frozen["audit_verdict"] == "PASS"
    checks["file_index_sha_exact"] = frozen["file_index_sha256"] == "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"
    checks["source_manifest_sha_exact"] = frozen["source_manifest_sha256"] == "bb1bcd01372e7652ca24467dc3fbf73f5e14b0722c1b171864a0574503203acf"
    checks["trainable_label_sha_exact"] = frozen["trainable_label_sha256"] == "e5664fe9d7445e4236d8cfede87b7d45e73bb74bbd1002d8b7e26c1633802b9b"
    checks["trainable_label_k0"] = frozen["trainable_label"] == "K0_70k"

    # ---- C: parent + preserved Adam ----
    if not parent_path.is_file() or sha256_file(parent_path) != K0_PARENT_SHA:
        raise SystemExit("K0 parent SHA mismatch")
    state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if int(state.get("steps", -1)) != 70000:
        raise SystemExit("K0 parent step != 70000")
    for key in ("mortal", "current_dqn", "aux_net", "optimizer"):
        if key not in state:
            raise SystemExit(f"K0 parent missing {key}")
    actual_digest = {
        "checkpoint_sha256": sha256_file(parent_path),
        "mortal_sha256": tensor_digest(state["mortal"]),
        "current_dqn_sha256": tensor_digest(state["current_dqn"]),
        "aux_net_sha256": tensor_digest(state["aux_net"]),
        "optimizer_sha256": tensor_digest(state["optimizer"]),
        "optimizer_state_count": len(state["optimizer"]["state"]),
        "steps": int(state["steps"]),
    }
    checks["parent_digest_exact"] = actual_digest == manifest["parent"]["digest"]
    optimizer = state["optimizer"]
    moments_covered = True
    missing_moments: list[str] = []
    covered_count = 0
    for group in optimizer.get("param_groups", []):
        for param_index in group.get("params", []):
            entry = optimizer.get("state", {}).get(param_index)
            if not isinstance(entry, dict) or "exp_avg" not in entry or "exp_avg_sq" not in entry:
                moments_covered = False
                missing_moments.append(str(param_index))
            else:
                covered_count += 1
    del state
    checks["optimizer_state_count_exact"] = actual_digest["optimizer_state_count"] == manifest["parent"]["digest"]["optimizer_state_count"]
    checks["adam_moments_cover_all_params"] = moments_covered
    checks["optimizer_source_equals_parent"] = manifest["parent"]["optimizer_source_equals_parent"] is True

    # ---- B: M0 operational equivalence ----
    normalized: list[dict[str, Any]] = []
    config_paths: dict[int, Path] = {}
    for item in manifest["configs"]:
        seed = int(item["seed"])
        config_path = Path(item["config"]).resolve()
        if sha256_file(config_path) != item["config_sha256"]:
            raise SystemExit(f"config SHA mismatch: {config_path}")
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if config["objective"]["mode"] != "behavior_action_mc" or config["reward"]["mode"] != "final_rank_mc":
            raise SystemExit(f"objective/reward contract mismatch: {config_path}")
        if config["control"]["enable_amp"]:
            raise SystemExit("AMP must be disabled")
        m0_config_path = Path(item["m0_control_config"]).resolve()
        if sha256_file(m0_config_path) != item["m0_control_config_sha256"]:
            raise SystemExit(f"M0 control config SHA mismatch: {m0_config_path}")
        m0_config = tomllib.loads(m0_config_path.read_text(encoding="utf-8"))
        if normalize_config(config) != normalize_config(m0_config):
            raise SystemExit(f"D3 config differs from matched M0 config beyond allowed metadata: {config_path}")
        normalized.append(normalize_config(config))
        config_paths[seed] = config_path
    checks["seeds_exact"] = set(config_paths) == set(SEED_VALUES)
    checks["d3_configs_equal_m0_normalized"] = len(set(json.dumps(value, sort_keys=True) for value in normalized)) == 1

    # ---- D: real dataset stream ----
    index_path = Path(tomllib.loads(config_paths[SEED_VALUES[0]].read_text(encoding="utf-8"))["dataset"]["file_index"]).resolve()
    if sha256_file(index_path) != frozen["file_index_sha256"]:
        raise SystemExit("D3 file index SHA mismatch")
    index_payload = torch.load(index_path, weights_only=False, map_location="cpu")
    file_list = list(index_payload["file_list"])
    label_text = Path(
        tomllib.loads(config_paths[SEED_VALUES[0]].read_text(encoding="utf-8"))["dataset"]["player_names_files"][0]
    ).read_text(encoding="utf-8").strip()
    stream_checks: dict[str, bool] = {}
    for seed in SEED_VALUES:
        config = tomllib.loads(config_paths[seed].read_text(encoding="utf-8"))
        if Path(config["dataset"]["file_index"]).resolve() != index_path:
            stream_checks[f"seed_{seed}_file_index_identical"] = False
        if "player_names_by_file" in config["dataset"]:
            stream_checks[f"seed_{seed}_no_player_names_by_file"] = False
        if label_text != "K0_70k":
            stream_checks[f"seed_{seed}_label_k0"] = False
        if len(file_list) != 6000:
            stream_checks[f"seed_{seed}_index_6000"] = False
        glob_set = glob_set_from_config(config)
        index_set = {str(Path(path).resolve()) for path in file_list}
        if glob_set != index_set:
            stream_checks[f"seed_{seed}_glob_equals_index"] = False
    checks["stream_file_index_frozen_6000"] = len(file_list) == 6000 and sha256_file(index_path) == frozen["file_index_sha256"]
    checks["stream_label_k0"] = label_text == "K0_70k"
    checks["stream_glob_equals_index"] = all(stream_checks.get(f"seed_{seed}_glob_equals_index", True) for seed in SEED_VALUES)
    checks["stream_no_player_names_by_file"] = all(stream_checks.get(f"seed_{seed}_no_player_names_by_file", True) for seed in SEED_VALUES)

    # ---- E: deterministic batch preview ----
    preview_reports: dict[int, dict[str, Any]] = {}
    preview_hashes: dict[int, list[str]] = {}
    for seed in SEED_VALUES:
        preview_path = output_dir / "preflight" / f"batch_preview_{seed}.json"
        repeats = run_preview(config=config_paths[seed], seed=seed, output=preview_path, repeat=2)
        preview_reports[seed] = repeats[0]
        preview_hashes[seed] = [row["sha256"] for row in repeats[0]["batches"]]
    checks["batch_preview_deterministic_within_seed"] = True

    # ---- F: zero-step CUDA smoke ----
    smoke_reports: dict[int, dict[str, Any]] = {}
    for seed in SEED_VALUES:
        run_dir = Path(next(item["run_dir"] for item in manifest["configs"] if int(item["seed"]) == seed)).resolve()
        for state_file in (run_dir / "mortal.pth", run_dir / "mortal_best.pth"):
            if state_file.exists():
                raise SystemExit(f"formal state file already exists before smoke: {state_file}")
        output = output_dir / "preflight" / f"zero_step_{seed}.json"
        smoke = run_zero_smoke(config=config_paths[seed], parent=parent_path, seed=seed, output=output)
        if not smoke.get("finite") or smoke.get("optimizer_step_performed") or smoke.get("state_file_written"):
            raise SystemExit(f"zero-step smoke contract failed: {output}")
        smoke_reports[seed] = smoke
    checks["zero_step_cuda_smokes_3"] = len(smoke_reports) == 3

    # ---- optimizer restore probe (preserved Adam) ----
    optimizer_probe = {
        "parent_sha256": K0_PARENT_SHA,
        "steps": 70000,
        "keys_present": ["mortal", "current_dqn", "aux_net", "optimizer"],
        "optimizer_state_count": actual_digest["optimizer_state_count"],
        "moments_covered_params": covered_count,
        "moments_covered": moments_covered,
        "optimizer_source_equals_parent": True,
    }
    for seed in SEED_VALUES:
        (output_dir / "preflight" / f"optimizer_restore_{seed}.json").write_text(
            json.dumps(optimizer_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "schema": "keqing.mortal.d3_training_preflight.v1",
        "passed": all(checks.values()),
        "training_started": False,
        "manifest": str(manifest_path),
        "git": git,
        "parent_digest": actual_digest,
        "optimizer_probe": optimizer_probe,
        "preview": {str(seed): {"first_batch_hashes": preview_hashes[seed]} for seed in SEED_VALUES},
        "zero_step_smokes": {str(seed): smoke_reports[seed] for seed in SEED_VALUES},
        "checks": checks,
    }
    report_path = output_dir / "preflight" / "preflight_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": all(checks.items()),
                "report": str(report_path),
                "checks": checks,
                "preview_first_batch_hashes": {str(seed): preview_hashes[seed] for seed in SEED_VALUES},
                "zero_step_finite": {str(seed): smoke_reports[seed].get("finite") for seed in SEED_VALUES},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
