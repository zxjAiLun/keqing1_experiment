#!/usr/bin/env python3
"""Run one CUDA forward/loss pass without stepping or writing a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import tomllib

import torch
from torch import optim
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PYTHON = REPO_ROOT / "third_party" / "Mortal" / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")

    config_path = args.config.resolve()
    os.environ["MORTAL_CFG"] = str(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    from config import config as mortal_config  # noqa: PLC0415
    from lr_scheduler import LinearWarmUpCosineAnnealingLR  # noqa: PLC0415
    from model import AuxNet, Brain, DQN  # noqa: PLC0415
    from training.mortal.mainline_dataloader import FileDatasetsIter  # noqa: PLC0415
    from training.mortal.objective import compute_objective_losses, objective_contract_from_config  # noqa: PLC0415
    from training.run_mortal_dqn_offline import (  # noqa: PLC0415
        _optimizer_group_metadata,
        _optimizer_param_groups,
        _validate_preserved_optimizer,
    )

    if mortal_config != config:
        raise ValueError("MORTAL_CFG config mismatch")
    parent_path = args.parent.resolve()
    state = torch.load(parent_path, weights_only=True, map_location="cpu")
    if int(state.get("steps", -1)) != 70000:
        raise ValueError(f"parent must be step 70000, got {state.get('steps')}")
    if "optimizer" not in state:
        raise ValueError("parent has no optimizer state")

    device = torch.device("cuda")
    version = int(config["control"]["version"])
    mortal = Brain(version=version, **config["resnet"]).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    aux_net.load_state_dict(state["aux_net"])
    mortal.freeze_bn(bool(config["freeze_bn"]["mortal"]))
    optimizer = optim.AdamW(
        _optimizer_param_groups((mortal, dqn, aux_net), weight_decay=float(config["optim"]["weight_decay"])),
        lr=1,
        weight_decay=0,
        betas=tuple(float(value) for value in config["optim"]["betas"]),
        eps=float(config["optim"]["eps"]),
    )
    LinearWarmUpCosineAnnealingLR(optimizer, **config["optim"]["scheduler"])
    fresh_groups = _optimizer_group_metadata(optimizer)
    optimizer.load_state_dict(state["optimizer"])
    _validate_preserved_optimizer(
        optimizer,
        fresh_groups=fresh_groups,
        expected_parameter_tensors=sum(len(group["params"]) for group in optimizer.param_groups),
    )
    del state

    random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)
    index_path = Path(str(config["dataset"]["file_index"])).resolve()
    payload = torch.load(index_path, weights_only=False, map_location="cpu")
    files = list(payload["file_list"] if isinstance(payload, dict) else payload)
    labels: set[str] = set()
    for label_file in config["dataset"]["player_names_files"]:
        labels.update(line.strip() for line in Path(str(label_file)).read_text(encoding="utf-8").splitlines() if line.strip())
    mapping = None
    mapping_path = config["dataset"].get("player_names_by_file")
    if mapping_path:
        payload = json.loads(Path(str(mapping_path)).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"player_names_by_file must be a JSON object: {mapping_path}")
        mapping = {str(Path(str(key)).resolve()): str(value) for key, value in payload.items()}
        indexed = {str(Path(str(value)).resolve()) for value in files}
        if indexed != set(mapping):
            raise ValueError("player_names_by_file must cover exactly the smoke file index")
    dataset = FileDatasetsIter(
        version=version,
        file_list=files,
        pts=config["env"]["pts"],
        file_batch_size=int(config["dataset"]["file_batch_size"]),
        reserve_ratio=float(config["dataset"]["reserve_ratio"]),
        player_names=sorted(labels),
        num_epochs=int(config["dataset"]["num_epochs"]),
        enable_augmentation=bool(config["dataset"]["enable_augmentation"]),
        augmented_first=bool(config["dataset"]["augmented_first"]),
        player_names_by_file=mapping,
    )
    batch = next(
        iter(
            DataLoader(
                dataset=dataset,
                batch_size=int(config["control"]["batch_size"]),
                drop_last=True,
                num_workers=0,
                pin_memory=False,
            )
        )
    )
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    obs = obs.to(dtype=torch.float32, device=device)
    actions = actions.to(dtype=torch.int64, device=device)
    masks = masks.to(dtype=torch.bool, device=device)
    steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
    kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
    player_ranks = player_ranks.to(dtype=torch.int64, device=device)
    if not bool(masks[torch.arange(obs.shape[0], device=device), actions].all().item()):
        raise RuntimeError("first batch contains an illegal behavior action")
    q_target_mc = (float(config["env"]["gamma"]) ** steps_to_done * kyoku_rewards).to(torch.float32)
    mortal.eval()
    dqn.eval()
    aux_net.eval()
    with torch.inference_mode():
        phi = mortal(obs)
        q_out = dqn(phi, masks)
        (next_rank_logits,) = aux_net(phi)
        losses = compute_objective_losses(
            q_out=q_out,
            masks=masks,
            actions=actions,
            q_target_mc=q_target_mc,
            next_rank_logits=next_rank_logits,
            player_ranks=player_ranks,
            mode=objective_contract_from_config(config)["mode"],
            cql_weight=float(config["cql"]["min_q_weight"]),
            aux_weight=float(config["aux"]["next_rank_weight"]),
        )
    finite = all(bool(value.isfinite().all().item()) for value in losses.values() if isinstance(value, torch.Tensor))
    if not finite:
        raise RuntimeError("zero-step smoke produced a non-finite loss or diagnostic")
    report = {
        "schema": "keqing.mortal.zero_step_smoke.v1",
        "config": str(config_path),
        "parent": str(parent_path),
        "device": torch.cuda.get_device_name(device),
        "data_seed": args.data_seed,
        "samples": int(obs.shape[0]),
        "objective": objective_contract_from_config(config),
        "finite": finite,
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items() if value.ndim == 0},
        "diagnostics": {key: float(value.detach().to(torch.float32).mean().cpu()) for key, value in losses.items() if value.ndim > 0},
        "optimizer_step_performed": False,
        "state_file_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
