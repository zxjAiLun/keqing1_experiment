#!/usr/bin/env python3
"""K0 State-Action Training-Signal Mechanism Audit runner (prereg v1).

Read-only diagnostic audit.  The implementation phase is limited to
unit/synthetic tests and checkpoint-only model/Q smoke.  The formal runner
must refuse to touch the real four-route corpus until a separate
authorization-only commit enables FORMAL_RUN_AUTHORIZED.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import riichi
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "third_party" / "Mortal" / "mortal") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Mortal" / "mortal"))

from training.mortal.audit_cross_corpus_mechanisms_2026_08 import (
    D1_PREP,
    D2_DATASET,
    D3_INDEX,
    DATA_ROOT,
    K0_MODEL,
    sha256,
)
from training.mortal.audit_replay_distribution import load_checkpoint

# ---------------------------------------------------------------- prereg gate
PREREG_COMMIT = "7bee592c7c1d00614ca1f5083032dc16b1665d36"
PREREG_FILE = (
    REPO
    / "training/docs/mortal/experiments_zh"
    / "2026-08_K0状态动作训练信号机制审计_预注册设计.md"
)
PREREG_FILE_SHA256 = "1e27e97e6efb509eba80299f644507fe025e4e66375183155f7190a76c639a9d"
OUTPUT_ROOT = DATA_ROOT / "mortal/authoritative/K0_decision_signal_audit_2026_08"
K0_REPR_OUTPUT = DATA_ROOT / "mortal/authoritative/K0_representation_audit_2026_08"
FORMAL_RUN_AUTHORIZED = False
RUN_AUTHORIZATION_NOTE = "implementation + synthetic/unit tests authorized; formal corpus run NOT authorized"

FROZEN_INPUT_SHA256 = {
    "k0_checkpoint": (K0_MODEL, "6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0"),
    "m0_index": (D1_PREP / "file_index_m0.pth", "755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2"),
    "d1_index": (D1_PREP / "file_index_d1.pth", "e357bdb00d5bf3cd7e0afa6960ee43af656421cfed381a3320f6b83ac56087f0"),
    "d2_index": (D2_DATASET / "file_index_d2.pth", "9c86b29204e86df0be8a5d3b0c4e211c010b07bd52e8e491fcdfc7e79e104bb1"),
    "d3_index": (D3_INDEX, "174122d9ff12365bc37331364ea2372c7a80bf382de039a3298da2fa5a8201f4"),
    "d2_mapping": (D2_DATASET / "player_names_by_file.json", "23b6d5a1589c4b3cba731332896dd33d3f23d50e8b987cb56de0789fdeca5970"),
    "riichi_extension": (Path(riichi.__file__).resolve(), "da687ececbae8c803c99fe58fb8f66d0e4b9e762eb2bb7257a2115c57e5dd82b"),
    "k0_repr_report": (K0_REPR_OUTPUT / "report.json", "6298e1e97668549702cd2c6294222b00a858162d0317411e109bbfc4da1cf1a2"),
    "k0_projection_matrix": (K0_REPR_OUTPUT / "projection_matrix.npy", "ba8d57a4d58f6d9db5c3590aeab3ae034edd1fa4ed00ebd81a5de4264eb87ca9"),
    "m0_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/M0/manifest.json", "077ffba14530135a622c3e5740945554222a23b68cc0b59ae27b3e794188e530"),
    "d1_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D1/manifest.json", "479a133b83a4596eaf7969bf2e9bf0ab268c2fe981087094861e6859e5ad571b"),
    "d2_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D2/manifest.json", "d7b0853a44b794f89968cc37a0e37d668aa1158e5b1fb5481df8b16ba93d1546"),
    "d3_route_manifest": (K0_REPR_OUTPUT / "route_artifacts/D3/manifest.json", "a8f48ff1ad444c899ea6f21391f02e337a54aa9ba39dbbba43094b60de3086fc"),
    "m0_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/M0/canonical_metadata.npz", "fad453d504149782fc2fe4950ceb25fe81530d2dcf40b3c49aef1694b880c3c6"),
    "d1_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D1/canonical_metadata.npz", "4dada3ec9223d42e6bd9655bc7eec90d404336852ba2348fdc0425d822a05759"),
    "d2_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D2/canonical_metadata.npz", "efa4a5e34164a41198e78e7e1ba7948dd3784301eaf75d6d3465b04093b0a003"),
    "d3_canonical_metadata": (K0_REPR_OUTPUT / "route_artifacts/D3/canonical_metadata.npz", "6d96fcebdaf9b4a4cf23babd00c231335fdb9b2d8089ffc477a2f83c6ea4c22b"),
}


def check_preregistration() -> dict[str, str | bool]:
    actual = sha256(PREREG_FILE)
    return {
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_file": str(PREREG_FILE),
        "preregistration_file_sha256": actual,
        "preregistration_sha_matches": actual == PREREG_FILE_SHA256,
    }


def git_worktree_metadata() -> dict[str, object]:
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        cwd=REPO,
    ).strip().splitlines()
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip(),
        "git_worktree_clean": not any(line.strip() for line in git_status),
        "git_worktree_status": git_status,
    }


def formal_preflight(device: torch.device, out_root: Path) -> dict[str, object]:
    """Fail-closed Gate A for the eventual formal run.

    This function only checks files/provenance; it does not open replay corpus.
    """
    checks: dict[str, bool] = {}
    for key, (path, expected) in FROZEN_INPUT_SHA256.items():
        checks[key] = path.is_file() and sha256(path) == expected
    worktree = git_worktree_metadata()
    checks["git_worktree_clean"] = bool(worktree["git_worktree_clean"])
    checks["device_is_cuda0"] = bool(device.type == "cuda" and getattr(device, "index", -1) == 0)
    checks["torch_cuda_available"] = bool(torch.cuda.is_available())
    checks["output_dir_absent_or_empty"] = not out_root.exists() or not any(out_root.iterdir())
    gate_a = {
        key: value
        for key, value in checks.items()
        if key not in {"git_worktree_clean", "device_is_cuda0", "torch_cuda_available", "output_dir_absent_or_empty"}
    }
    all_checks = all(checks.values())
    return {
        "checks": checks,
        "gate_a": gate_a,
        "gate_a_pass": all(gate_a.values()),
        "all_pass": all_checks,
        "worktree": worktree,
        "formal_run_authorized": FORMAL_RUN_AUTHORIZED,
    }


def checkpoint_smoke(device: torch.device) -> dict[str, object]:
    """Checkpoint-only phi/Q shape smoke. No corpus file is opened."""
    from model import Brain, obs_shape

    state = load_checkpoint(K0_MODEL)
    config = state["config"]
    version = int(config["control"].get("version", 4))
    brain = Brain(version=version, **config["resnet"]).to(device).eval()
    brain.load_state_dict(state["mortal"])
    shape = obs_shape(version)
    obs = torch.zeros(8, *shape, device=device)
    with torch.inference_mode():
        phi = brain(obs)
    phi_np = phi.detach().cpu().numpy()
    result = {
        "checkpoint_sha256": sha256(K0_MODEL),
        "version": version,
        "obs_shape": list(shape),
        "phi_ndim": int(phi_np.ndim),
        "phi_shape": list(phi_np.shape),
        "phi_dim": int(phi_np.shape[1]) if phi_np.ndim == 2 else None,
        "phi_finite_fraction": float(np.isfinite(phi_np).mean()),
        "smoke_pass": bool(
            phi_np.ndim == 2
            and phi_np.shape[1] == 1024
            and float(np.isfinite(phi_np).mean()) == 1.0
        ),
    }
    if not result["smoke_pass"]:
        raise RuntimeError(f"checkpoint smoke failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-prereg", action="store_true", help="verify prereg document SHA")
    parser.add_argument("--checkpoint-smoke", action="store_true", help="run checkpoint-only phi smoke")
    parser.add_argument("--formal-preflight", action="store_true", help="run provenance preflight without corpus access")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.check_prereg:
        result = check_preregistration()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["preregistration_sha_matches"]:
            raise SystemExit(1)
        return

    if args.checkpoint_smoke:
        device = torch.device(args.device)
        result = checkpoint_smoke(device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.formal_preflight:
        device = torch.device(args.device)
        result = formal_preflight(device, OUTPUT_ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not result["all_pass"] or not FORMAL_RUN_AUTHORIZED:
            raise SystemExit(1)
        return

    parser.print_help()
    raise SystemExit(
        "Formal corpus audit is NOT AUTHORIZED in this phase. "
        "Use --check-prereg, --checkpoint-smoke, or --formal-preflight."
    )


if __name__ == "__main__":
    main()
