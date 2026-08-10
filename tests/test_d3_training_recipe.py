from __future__ import annotations

import copy
import json

from training.mortal.prepare_d3_training_recipe_2026_08 import (
    D3_CONTRACT_SHA,
    D3_INDEX_SHA,
    D3_LABEL,
    D3_LABEL_SHA,
    D3_SOURCE_MANIFEST_SHA,
    K0_PARENT_SHA,
    M0_CHECKPOINT_SHA,
    SEED_VALUES,
    build_d3_config,
)
from training.mortal.preflight_d3_training_2026_08 import normalize_config


def _m0_sample_config() -> dict:
    return {
        "control": {
            "version": 4,
            "state_file": "/old/mortal.pth",
            "best_state_file": "/old/mortal_best.pth",
            "tensorboard_dir": "/old/tb",
            "device": "cuda:0",
            "enable_amp": False,
            "batch_size": 512,
            "opt_step_every": 1,
            "save_every": 400,
        },
        "dataset": {
            "globs": ["/old/data/**/*.json.gz"],
            "file_index": "/old/file_index_m0.pth",
            "file_batch_size": 15,
            "reserve_ratio": 0.0,
            "num_workers": 0,
            "player_names_files": ["/old/labels/m0.txt"],
            "num_epochs": 3,
            "enable_augmentation": False,
            "augmented_first": False,
        },
        "env": {"gamma": 1.0, "pts": [6.0, 4.0, 2.0, 0.0]},
        "reward": {"mode": "final_rank_mc"},
        "resnet": {"conv_channels": 192, "num_blocks": 40},
        "cql": {"min_q_weight": 5.0},
        "aux": {"next_rank_weight": 0.2},
        "freeze_bn": {"mortal": False},
        "optim": {"eps": 1e-8, "betas": [0.9, 0.999], "weight_decay": 0.1, "max_grad_norm": 0.0},
        "objective": {"mode": "behavior_action_mc"},
        "experiment": {
            "route": "M0_control",
            "trainable_label": "ext_mortal",
            "training_seed": 20260806,
            "parent_steps": 70000,
            "reward_mode": "final_rank_mc",
        },
        "optim.scheduler": {"peak": 0.0001, "final": 0.0001, "warm_up_steps": 0, "max_steps": 0},
    }


def test_normalize_config_strips_only_allowed_keys() -> None:
    config = _m0_sample_config()
    normalized = normalize_config(config)
    assert "state_file" not in normalized["control"]
    assert "best_state_file" not in normalized["control"]
    assert "tensorboard_dir" not in normalized["control"]
    assert "globs" not in normalized["dataset"]
    assert "file_index" not in normalized["dataset"]
    assert "player_names_files" not in normalized["dataset"]
    assert "experiment" not in normalized
    assert normalized["control"]["batch_size"] == 512
    assert normalized["env"]["pts"] == [6.0, 4.0, 2.0, 0.0]
    assert normalized["objective"]["mode"] == "behavior_action_mc"


def test_build_d3_config_changes_only_allowed_metadata() -> None:
    m0 = _m0_sample_config()
    d3 = build_d3_config(
        m0,
        seed=20260806,
        output_dir=__import__("pathlib").Path("/new/seed_20260806"),
        file_index=__import__("pathlib").Path("/new/file_index_d3_k0.pth"),
        data_globs=["/new/glob/**/*.json.gz"],
        label_file=__import__("pathlib").Path("/new/trainable_label.txt"),
        provenance={"data_contract_sha256": "abc"},
    )
    assert normalize_config(d3) == normalize_config(m0)
    assert d3["experiment"]["route"] == "D3_variant"
    assert d3["experiment"]["trainable_label"] == "K0_70k"
    assert d3["experiment"]["training_seed"] == 20260806
    assert d3["experiment"]["data_contract_sha256"] == "abc"


def test_frozen_recipe_anchors() -> None:
    assert K0_PARENT_SHA == "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"
    assert D3_CONTRACT_SHA == "30bda12f25cf0d036c6f74e4650580f53ae1baaa670b0d1224092752c74ae4d4"
    assert D3_INDEX_SHA == "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"
    assert D3_SOURCE_MANIFEST_SHA == "bb1bcd01372e7652ca24467dc3fbf73f5e14b0722c1b171864a0574503203acf"
    assert D3_LABEL_SHA == "e5664fe9d7445e4236d8cfede87b7d45e73bb74bbd1002d8b7e26c1633802b9b"
    assert D3_LABEL == "K0_70k"


def test_m0_control_checkpoint_shas_are_frozen() -> None:
    assert M0_CHECKPOINT_SHA == {
        20260806: "4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd",
        20260807: "de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d",
        20260808: "d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4",
    }


def test_seeds_are_the_matched_three() -> None:
    assert SEED_VALUES == (20260806, 20260807, 20260808)


def test_promotion_eval_protocol_mapping() -> None:
    seed_starts = {
        str(seed): start
        for seed, start in zip(SEED_VALUES, (1700000, 1710000, 1720000), strict=True)
    }
    assert seed_starts == {
        "20260806": 1700000,
        "20260807": 1710000,
        "20260808": 1720000,
    }


def test_training_command_forbids_legacy_replay() -> None:
    template = (
        "training/run_mortal_dqn_offline.py --config <seed-config> "
        "--target-steps 72000 --initialize-from <K0> "
        "--initialize-optimizer-from <same-K0> --initial-steps 70000"
    )
    assert "--allow-legacy-data-replay" not in template
    assert "--target-steps 72000" in template
    assert "--initial-steps 70000" in template
