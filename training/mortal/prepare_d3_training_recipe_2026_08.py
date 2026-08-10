#!/usr/bin/env python3
"""Prepare the frozen D3 training recipe (3 configs derived from M0 control).

D3 does NO recipe search: the only experiment variable is the frozen D3 replay
corpus. Training-side recipe is inherited verbatim from the M0/D1/D2 matched
operational recipe. The three D3 configs are derived from the frozen M0_control
configs (same seeds 20260806/07/08) with ONLY these differences:

  control.state_file / best_state_file / tensorboard_dir
  dataset.globs / file_index / player_names_files
  experiment.route / trainable_label / provenance metadata

Everything else is byte-identical after normalization. No training, no
checkpoint, no optimizer step in this script.
"""

from __future__ import annotations

import argparse
import copy
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

SEED_VALUES = (20260806, 20260807, 20260808)
D3_EXP_ROOT = (
    REPO_ROOT
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
)
DEFAULT_TRAINING_CONTRACT_DIR = D3_EXP_ROOT / "training_contract_2026_08"
DEFAULT_OUTPUT_DIR = D3_EXP_ROOT / "training_recipe_2026_08"
DEFAULT_PARENT = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\models\K0_70k\mortal_default_70k_promoted_candidate.pth"
)
DEFAULT_M0_ROOT = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D1_project_owned_population_2026_07\training_prep_2026_07\M0_control"
)
DEFAULT_D1_MANIFEST = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D1_project_owned_population_2026_07\training_prep_2026_07\training_manifest.json"
)
DEFAULT_D1_EVAL_PROTOCOL = Path(
    r"E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07"
    r"\D1_project_owned_population_2026_07\eval_b250_1000h_2026_08\protocol.json"
)
K0_PARENT_SHA = "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"
D3_LABEL = "K0_70k"
D3_CONTRACT_SHA = "30bda12f25cf0d036c6f74e4650580f53ae1baaa670b0d1224092752c74ae4d4"
D3_INDEX_SHA = "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"
D3_SOURCE_MANIFEST_SHA = "bb1bcd01372e7652ca24467dc3fbf73f5e14b0722c1b171864a0574503203acf"
D3_LABEL_SHA = "e5664fe9d7445e4236d8cfede87b7d45e73bb74bbd1002d8b7e26c1633802b9b"

M0_CHECKPOINT_SHA = {
    20260806: "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
    20260807: "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
    20260808: "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(prefix.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, dict):
            for key in sorted(item, key=lambda value: repr(value)):
                visit(key, prefix + ".key")
                visit(item[key], prefix + f"[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, prefix + f"[{index}]")
            return
        digest.update(prefix.encode())
        digest.update(repr(item).encode())

    visit(value, "root")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_d3_config(
    m0_config: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
    file_index: Path,
    data_globs: list[str],
    label_file: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(m0_config)
    control = config["control"]
    control["state_file"] = str((output_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((output_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((output_dir / "tb_mortal").resolve())
    dataset = config["dataset"]
    dataset["globs"] = data_globs
    dataset["file_index"] = str(file_index.resolve())
    dataset["player_names_files"] = [str(label_file.resolve())]
    config["experiment"] = {
        "route": "D3_variant",
        "trainable_label": D3_LABEL,
        "training_seed": seed,
        "parent_steps": 70000,
        "reward_mode": "final_rank_mc",
        **provenance,
    }
    return config


def main(argv: list[str] | None = None) -> None:
    import toml  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-contract-dir", type=Path, default=DEFAULT_TRAINING_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--d1-manifest", type=Path, default=DEFAULT_D1_MANIFEST)
    parser.add_argument("--d1-eval-protocol", type=Path, default=DEFAULT_D1_EVAL_PROTOCOL)
    args = parser.parse_args(argv)

    contract_dir = args.training_contract_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_path = args.parent.resolve()
    m0_root = args.m0_root.resolve()
    d1_manifest = load_json(args.d1_manifest.resolve())
    d1_eval = load_json(args.d1_eval_protocol.resolve())

    # ---- frozen D3 data contract ----
    contract = load_json(contract_dir / "d3_training_data_contract.json")
    audit = load_json(contract_dir / "d3_training_contract_audit.json")
    if contract.get("status") != "training_contract_passed_manifest_frozen":
        raise SystemExit("D3 training data contract is not frozen-passed")
    if sha256_file(contract_dir / "d3_training_data_contract.json") != D3_CONTRACT_SHA:
        raise SystemExit("D3 training data contract SHA mismatch")
    if audit.get("gate", {}).get("verdict") != "PASS":
        raise SystemExit("D3 training contract audit is not PASS")
    if sha256_file(contract_dir / "file_index_d3_k0.pth") != D3_INDEX_SHA:
        raise SystemExit("D3 file index SHA mismatch")
    if sha256_file(contract_dir / "d3_6000h_training_source_manifest.json") != D3_SOURCE_MANIFEST_SHA:
        raise SystemExit("D3 source manifest SHA mismatch")
    if sha256_file(contract_dir / "trainable_label.txt") != D3_LABEL_SHA:
        raise SystemExit("D3 trainable label SHA mismatch")

    # ---- M0 controls: configs bound to the D1 manifest, checkpoints bound to
    # the D1 eval protocol (never retrained) ----
    m0_config_paths: dict[int, Path] = {}
    m0_configs: dict[int, dict[str, Any]] = {}
    m0_checkpoints: dict[int, Path] = {}
    d1_m0_config_sha = {
        int(item["seed"]): str(item["config_sha256"])
        for item in d1_manifest["configs"]
        if item["route"] == "M0_control"
    }
    for seed in SEED_VALUES:
        config_path = m0_root / f"seed_{seed}" / "config.toml"
        if not config_path.is_file():
            raise SystemExit(f"M0 control config missing: {config_path}")
        if sha256_file(config_path) != d1_m0_config_sha[seed]:
            raise SystemExit(f"M0 control config SHA does not match D1 manifest for {seed}")
        m0_config_paths[seed] = config_path
        m0_configs[seed] = tomllib.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = m0_root / f"seed_{seed}" / "checkpoints" / "mortal_72000.pth"
        if not checkpoint.is_file():
            raise SystemExit(f"M0 control checkpoint missing: {checkpoint}")
        if sha256_file(checkpoint) != M0_CHECKPOINT_SHA[seed]:
            raise SystemExit(f"M0 control checkpoint SHA mismatch for {seed}")
        eval_sha = d1_eval["models"][f"M0_{seed}"]
        if eval_sha != M0_CHECKPOINT_SHA[seed]:
            raise SystemExit(f"D1 eval protocol M0 SHA mismatch for {seed}")
        state = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if int(state.get("steps", -1)) != 72000:
            raise SystemExit(f"M0 control checkpoint step != 72000 for {seed}")
        del state
        m0_checkpoints[seed] = checkpoint

    # ---- parent K0: full checkpoint, preserved Adam ----
    if sha256_file(parent_path) != K0_PARENT_SHA:
        raise SystemExit("K0 parent SHA mismatch")
    parent_state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if int(parent_state.get("steps", -1)) != 70000:
        raise SystemExit("K0 parent step != 70000")
    for key in ("mortal", "current_dqn", "aux_net", "optimizer"):
        if key not in parent_state:
            raise SystemExit(f"K0 parent missing {key}")
    reference_digest = d1_manifest["protocol"]["parent_tensor_digest"]
    parent_digest = {
        "checkpoint_sha256": sha256_file(parent_path),
        "mortal_sha256": tensor_digest(parent_state["mortal"]),
        "current_dqn_sha256": tensor_digest(parent_state["current_dqn"]),
        "aux_net_sha256": tensor_digest(parent_state["aux_net"]),
        "optimizer_sha256": tensor_digest(parent_state["optimizer"]),
        "optimizer_state_count": len(parent_state["optimizer"]["state"]),
        "steps": int(parent_state["steps"]),
    }
    if parent_digest != reference_digest:
        raise SystemExit("K0 parent digest differs from the frozen D1/M0 reference")
    parent_optimizer = parent_state["optimizer"]
    moments_covered = True
    missing_moments: list[str] = []
    for group in parent_optimizer.get("param_groups", []):
        for param_index in group.get("params", []):
            entry = parent_optimizer.get("state", {}).get(param_index)
            if not isinstance(entry, dict) or "exp_avg" not in entry or "exp_avg_sq" not in entry:
                moments_covered = False
                missing_moments.append(str(param_index))
    del parent_state

    # ---- build D3 configs ----
    file_index = (contract_dir / "file_index_d3_k0.pth").resolve()
    label_file = (contract_dir / "trainable_label.txt").resolve()
    data_globs = [
        str(
            (
                D3_EXP_ROOT
                / "generation_production/shard_000_1800000_1800249/logs/*.json.gz"
            ).resolve()
        ),
        str((D3_EXP_ROOT / "generation_continuation/shard_*/logs/*.json.gz").resolve()),
    ]
    configs: list[dict[str, Any]] = []
    for seed in SEED_VALUES:
        run_dir = output_dir / f"seed_{seed}"
        config = build_d3_config(
            m0_configs[seed],
            seed=seed,
            output_dir=run_dir,
            file_index=file_index,
            data_globs=data_globs,
            label_file=label_file,
            provenance={
                "data_contract_sha256": D3_CONTRACT_SHA,
                "source_manifest_sha256": D3_SOURCE_MANIFEST_SHA,
                "file_index_sha256": D3_INDEX_SHA,
                "trainable_label_sha256": D3_LABEL_SHA,
            },
        )
        config_path = run_dir / "config.toml"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(toml.dumps(config), encoding="utf-8")
        configs.append(
            {
                "route": "D3_variant",
                "seed": seed,
                "label": D3_LABEL,
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "run_dir": str(run_dir.resolve()),
                "m0_control_config": str(m0_config_paths[seed]),
                "m0_control_config_sha256": sha256_file(m0_config_paths[seed]),
                "m0_control_checkpoint": str(m0_checkpoints[seed]),
                "m0_control_checkpoint_sha256": M0_CHECKPOINT_SHA[seed],
                "file_index_sha256": D3_INDEX_SHA,
            }
        )

    m0_sample = m0_configs[SEED_VALUES[0]]
    recipe_values = {
        "control": {
            "version": m0_sample["control"]["version"],
            "batch_size": m0_sample["control"]["batch_size"],
            "opt_step_every": m0_sample["control"]["opt_step_every"],
            "save_every": m0_sample["control"]["save_every"],
            "enable_amp": m0_sample["control"]["enable_amp"],
            "device": "cuda",
        },
        "dataset": {
            "file_batch_size": m0_sample["dataset"]["file_batch_size"],
            "reserve_ratio": m0_sample["dataset"]["reserve_ratio"],
            "num_workers": 0,
            "num_epochs": m0_sample["dataset"]["num_epochs"],
            "enable_augmentation": m0_sample["dataset"]["enable_augmentation"],
            "augmented_first": m0_sample["dataset"]["augmented_first"],
        },
        "env": {"gamma": m0_sample["env"]["gamma"], "pts": m0_sample["env"]["pts"]},
        "reward": {"mode": m0_sample["reward"]["mode"]},
        "objective": {"mode": m0_sample["objective"]["mode"]},
        "resnet": m0_sample["resnet"],
        "cql": m0_sample["cql"],
        "aux": m0_sample["aux"],
        "freeze_bn": m0_sample["freeze_bn"],
        "optim": {
            "eps": m0_sample["optim"]["eps"],
            "betas": m0_sample["optim"]["betas"],
            "weight_decay": m0_sample["optim"]["weight_decay"],
            "max_grad_norm": m0_sample["optim"]["max_grad_norm"],
            "scheduler": m0_sample["optim"]["scheduler"],
        },
    }
    command_template = (
        "& $bundlePython .\\training\\run_mortal_dqn_offline.py "
        "--config <seed-config> --mortal-root .\\third_party\\Mortal "
        "--target-steps 72000 --device cuda --seed <SEED> --data-seed <SEED> "
        "--initialize-from <K0-parent> --initialize-optimizer-from <same-K0-parent> "
        "--initial-steps 70000 --num-workers 0 "
        "--archive-steps 70001,70010,70100,70500,71000,72000 "
        "--archive-dir <seed-run-dir>\\checkpoints --log-every 50"
    )
    manifest = {
        "schema": "keqing.mortal.d3_training_recipe.v1",
        "experiment_arm": "D3_variant_only",
        "status": "recipe_prepared_not_started",
        "git": git_info(),
        "seeds": list(SEED_VALUES),
        "parent": {
            "path": str(parent_path),
            "sha256": K0_PARENT_SHA,
            "digest": parent_digest,
            "optimizer_moments_covered": moments_covered,
            "optimizer_missing_moments": missing_moments,
            "optimizer_source_equals_parent": True,
        },
        "frozen_d3_data_contract": {
            "status": contract["status"],
            "contract_sha256": D3_CONTRACT_SHA,
            "audit_verdict": audit["gate"]["verdict"],
            "file_index_sha256": D3_INDEX_SHA,
            "source_manifest_sha256": D3_SOURCE_MANIFEST_SHA,
            "trainable_label_sha256": D3_LABEL_SHA,
            "trainable_label": D3_LABEL,
        },
        "m0_controls": {
            str(seed): {
                "config_sha256": d1_m0_config_sha[seed],
                "checkpoint_sha256": M0_CHECKPOINT_SHA[seed],
                "checkpoint_path": str(m0_checkpoints[seed]),
                "d1_eval_protocol_sha256": sha256_file(args.d1_eval_protocol.resolve()),
            }
            for seed in SEED_VALUES
        },
        "configs": configs,
        "recipe_values": recipe_values,
        "pairing": {
            "model_seed_equals_data_seed": True,
            "initial_steps": 70000,
            "target_steps": 72000,
            "preserved_adam_from_parent": True,
            "fresh_scheduler": True,
            "fresh_scaler": True,
            "fresh_data_stream": True,
            "amp": False,
            "num_workers": 0,
        },
        "promotion_eval_protocol": {
            "seed_starts": {str(seed): start for seed, start in zip(SEED_VALUES, (1700000, 1710000, 1720000), strict=True)},
            "games_per_seed": 1000,
            "shards_per_seed": 4,
            "games_per_shard": 250,
            "seed_key": 8192,
            "seat_mode": "random",
            "amp": False,
            "rank_points": [90, 45, 0, -135],
            "lineup_per_seed": ["D3_seed_72k", "matched_M0_seed_72k", "K0_70k", "ext_mortal"],
            "primary_comparison": "D3 - matched M0",
            "secondary_comparison": "D3 - K0",
            "data_route_promotion_criteria": [
                "3/3 training-seed mean direction > 0",
                "equal-seed hierarchical bootstrap 95% CI lower bound > 0",
            ],
            "k1_checkpoint_lineage_criteria": [
                "D3 data-route promotion PASS",
                "D3-K0 equal-seed hierarchical 95% CI lower bound > 0",
                "no hard integrity / legality / runtime regression",
            ],
            "note": "D3 route promoted with K1 still null is a legal outcome",
        },
        "training_command_template": command_template,
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "training_manifest_sha256": sha256_file(manifest_path),
                "configs": {
                    str(seed): next(
                        item["config_sha256"] for item in configs if item["seed"] == seed
                    )
                    for seed in SEED_VALUES
                },
                "m0_control_checkpoints": M0_CHECKPOINT_SHA,
                "k0_parent_sha256": K0_PARENT_SHA,
                "k0_optimizer_state_count": parent_digest["optimizer_state_count"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
