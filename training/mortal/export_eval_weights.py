#!/usr/bin/env python3
"""Repackage a training checkpoint into the arena's eval-weights layout.

Why this exists
---------------
The native arena loads a checkpoint with ``torch.load(..., weights_only=True)``.
A *training* checkpoint cannot be loaded that way: its ``training_contract``
carries ``torch.torch_version.TorchVersion``, which the strict unpickler refuses
with ``Unsupported global: GLOBAL torch.torch_version.TorchVersion``.  Trainers in
this repo therefore write a separate eval-weights file next to the checkpoint
(P4-M10 wrote ``C{cycle}_eval_weights.pth``; P4-M11 did not, which is why its
endpoint had no arena-loadable representation until this tool was run).

This is **pure repackaging, not a new model**: the export is the source minus
``optimizer_state`` and ``cycle``, i.e. exactly the relation between P4-M10's
``C4.pth`` and ``C4_eval_weights.pth``.  Identity is verified tensor-by-tensor and
the real loader path is exercised before anything is written.

    python training/mortal/export_eval_weights.py \
        --checkpoint artifacts/experiments/.../U32.pth \
        --expect-source-sha256 511be9ad... \
        --label p4m11_u32
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The arena consumes exactly these keys (see four_player_native._load_engine).
EVAL_KEYS = ("mortal", "current_dqn", "training_contract")
STUDENT_POLICY_SCHEMA = "keqing.mortal.student_policy_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_identity(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    """Bitwise equality of two state dicts, with the first difference named."""
    import torch  # noqa: PLC0415

    if set(a) != set(b):
        return False, f"key sets differ: {sorted(set(a) ^ set(b))}"
    for name in sorted(a):
        ta, tb = a[name], b[name]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            return False, f"{name}: shape/dtype {ta.shape}/{ta.dtype} vs {tb.shape}/{tb.dtype}"
        if not torch.equal(ta, tb):
            return False, f"{name}: values differ (max |delta| {float((ta - tb).abs().max()):g})"
    return True, f"{len(a)} tensors identical"


def export_eval_weights(
    *,
    checkpoint: Path,
    out: Path | None = None,
    expect_source_sha256: str | None = None,
    label: str | None = None,
    mortal_root: Path = _REPO_ROOT / "third_party" / "Mortal",
    device: str | None = None,
    verify_loader: bool = True,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    target = out or checkpoint.with_name(checkpoint.stem + "_eval_weights.pth")
    source_sha = sha256(checkpoint)
    if expect_source_sha256 and source_sha != expect_source_sha256:
        raise SystemExit(
            f"source sha256 {source_sha} != expected {expect_source_sha256}; refusing to export"
        )
    print("source            :", checkpoint.name)
    print("source sha256     :", source_sha)

    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    contract = state.get("training_contract")
    if not isinstance(contract, dict) or contract.get("schema") != STUDENT_POLICY_SCHEMA:
        raise SystemExit(
            f"checkpoint does not carry the {STUDENT_POLICY_SCHEMA} contract; this tool only "
            "repackages that layout"
        )
    missing = [key for key in EVAL_KEYS if key not in state]
    if missing:
        raise SystemExit(f"checkpoint is missing {missing}")

    payload = {key: state[key] for key in EVAL_KEYS}

    # Verify before writing: the export must be the source's own tensors.
    for key in ("mortal", "current_dqn"):
        ok, detail = state_identity(state[key], payload[key])
        print(f"identity {key:<14}: {ok} ({detail})")
        if not ok:
            raise SystemExit(f"{key} does not round-trip")
    if payload["training_contract"] != contract:
        raise SystemExit("contract changed in transit")

    if target.exists():
        existing = torch.load(target, weights_only=False, map_location="cpu")
        ok, detail = state_identity(existing["mortal"], payload["mortal"])
        print("existing identity :", ok, detail)
        if not ok:
            raise SystemExit("an existing export disagrees with the frozen source; refusing to touch it")
    else:
        tmp = target.with_name(target.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)
        print("wrote             :", target)

    export_sha = sha256(target)
    print("export sha256     :", export_sha)
    print("export bytes      :", target.stat().st_size)

    loader_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verify_loader:
        from training.mortal import four_player_native as fpn  # noqa: PLC0415

        engine = fpn._load_engine(
            label=label or checkpoint.stem,
            state_file=target,
            mortal_root=mortal_root,
            device=loader_device,
            enable_amp=False,
            enable_profile=False,
        )
        print(f"arena loader      : ACCEPTED on {loader_device} (FP32)")
        del engine
        if loader_device == "cuda":
            torch.cuda.empty_cache()
    else:
        print("arena loader      : SKIPPED (--no-verify-loader)")

    return {
        "source": {"path": str(checkpoint), "sha256": source_sha},
        "export": {
            "path": str(target),
            "sha256": export_sha,
            "bytes": target.stat().st_size,
            "keys": list(EVAL_KEYS),
        },
        "transform": "pure repackaging: source minus optimizer_state and cycle",
        "identity_verified": True,
        "loader_verified": bool(verify_loader),
        "loader_device": loader_device if verify_loader else None,
        "adam_step": state.get("adam_step"),
        "contract_schema": contract.get("schema"),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <checkpoint>_eval_weights.pth")
    parser.add_argument("--expect-source-sha256", default=None)
    parser.add_argument("--label", default=None, help="agent label handed to the arena loader")
    parser.add_argument("--mortal-root", type=Path, default=_REPO_ROOT / "third_party" / "Mortal")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-verify-loader", action="store_true")
    parser.add_argument("--provenance", type=Path, default=None, help="default: <out>.provenance.json")
    parser.add_argument(
        "--schema",
        default="keqing.mortal.eval_weights_export.v1",
        help="provenance schema name (P4-M11's historical file used keqing.mortal.p4m11_eval_weights_export.v1)",
    )
    parser.add_argument("--no-provenance", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    record = export_eval_weights(
        checkpoint=args.checkpoint,
        out=args.out,
        expect_source_sha256=args.expect_source_sha256,
        label=args.label,
        mortal_root=args.mortal_root,
        device=args.device,
        verify_loader=not args.no_verify_loader,
    )
    if not args.no_provenance:
        target = args.provenance or Path(record["export"]["path"]).with_name(
            Path(record["export"]["path"]).name + ".provenance.json"
        )
        payload = {"schema": args.schema, **record}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("provenance        :", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
