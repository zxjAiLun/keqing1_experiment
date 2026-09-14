#!/usr/bin/env python3
"""Publish a checkpoint as a NEW immutable authoritative bundle under a data root.

``mortal/authoritative/<existing bundle>`` is an immutable, SHA-verified asset
bundle with its own manifest, so a new candidate must never be dropped into it.
This creates a sibling bundle following the same convention:

    <data-root>/mortal/authoritative/<bundle-id>/
        manifest.json                          # keqing.mortal.authoritative_asset_bundle.v1
        models/<family>/<basename>

The copy is byte-identical and SHA-256 verified, and nothing existing is
modified: an existing bundle whose content differs is refused, never rewritten.

    python training/mortal/publish_authoritative_bundle.py \
        --data-root E:/AUbuntuProject/keqing-data \
        --bundle P4M11_U32_2026_09 --family P4M11_U32 \
        --source artifacts/experiments/.../U32_eval_weights.pth \
        --expect-source-sha256 3703c943...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

BUNDLE_SCHEMA = "keqing.mortal.authoritative_asset_bundle.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_provenance(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` where VALUE is JSON (so objects/lists work)."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--provenance must be KEY=JSON, got: {pair}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--provenance must be KEY=JSON, got: {pair}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def publish(
    *,
    data_root: Path,
    bundle_id: str,
    family: str,
    source: Path,
    expect_source_sha256: str | None = None,
    basename: str | None = None,
    description: str = "",
    provenance: dict[str, Any] | None = None,
    source_paths: dict[str, str] | None = None,
    check_resolver: Path | None = None,
    force_manifest: bool = False,
) -> dict[str, Any]:
    basename = basename or source.name
    bundle = data_root / "mortal" / "authoritative" / bundle_id
    target = bundle / "models" / family / basename

    source_sha = sha256(source)
    if expect_source_sha256 and source_sha != expect_source_sha256:
        raise SystemExit(f"source sha256 {source_sha} != expected {expect_source_sha256}")

    if target.exists():
        existing_sha = sha256(target)
        if existing_sha != source_sha:
            raise SystemExit(
                f"{target} already exists with sha256 {existing_sha} != {source_sha}; refusing to "
                "rewrite a published bundle"
            )
        print("model file        : already published, byte-identical")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied = sha256(target)
        if copied != source_sha:
            raise SystemExit(f"copy is not byte-identical: {copied} != {source_sha}")
        print("model file        :", target)

    manifest_path = bundle / "manifest.json"
    if manifest_path.exists() and not force_manifest:
        # A published bundle is immutable: re-running with different provenance
        # text must not silently replace the manifest that was recorded.
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_sha = ((recorded.get("artifacts") or {}).get(family) or {}).get("sha256")
        if recorded_sha == source_sha:
            print("bundle            :", bundle)
            print("manifest          : already present for this content; not rewritten")
            print("                      (pass --force-manifest to replace it)")
            return recorded
        raise SystemExit(
            f"{manifest_path} records sha256 {recorded_sha} for {family!r}, not {source_sha}; "
            "refusing to overwrite a published bundle's manifest"
        )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle": bundle_id,
        "description": description,
        "source_repo": "keqing1_experiment",
        "source_paths": dict(source_paths or {family: str(source)}),
        "provenance": dict(provenance or {}),
        "artifacts": {
            family: {
                "path": f"models/{family}/{basename}",
                "sha256": source_sha,
                "bytes": target.stat().st_size,
            }
        },
        "copied_at_unix": int(time.time()),
        "copy_verification": "SHA-256 verified source == copy for the single model file",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("bundle            :", bundle)
    print("sha256            :", source_sha)
    print("bytes             :", target.stat().st_size)

    if check_resolver is not None:
        # The consumer must be able to find it through its own resolver.
        sys.path.insert(0, str(check_resolver))
        from inference.bot_registry import _search_authoritative_checkpoint  # noqa: PLC0415

        found = _search_authoritative_checkpoint(Path(family) / basename)
        print("resolver          :", found)
        if found is None or Path(found).resolve() != target.resolve():
            raise SystemExit("the workbench authoritative resolver did not find the new bundle")
        print("resolver          : OK (exactly one match, no ambiguity)")

    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bundle", required=True, help="bundle id, e.g. P4M11_U32_2026_09")
    parser.add_argument("--family", required=True, help="model family directory, e.g. P4M11_U32")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-source-sha256", default=None)
    parser.add_argument("--basename", default=None, help="default: the source file name")
    parser.add_argument("--description", default="")
    parser.add_argument("--provenance", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="additional recorded source path",
    )
    parser.add_argument(
        "--check-resolver",
        type=Path,
        default=None,
        help="path to the consumer repo's src/ to verify its resolver finds the bundle",
    )
    parser.add_argument(
        "--force-manifest",
        action="store_true",
        help="replace an existing manifest for the same content (default: leave it alone)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_paths = {args.family: str(args.source)}
    for pair in args.source_path:
        if "=" not in pair:
            raise SystemExit(f"--source-path must be KEY=PATH, got: {pair}")
        key, value = pair.split("=", 1)
        source_paths[key.strip()] = value.strip()
    publish(
        data_root=args.data_root,
        bundle_id=args.bundle,
        family=args.family,
        source=args.source,
        expect_source_sha256=args.expect_source_sha256,
        basename=args.basename,
        description=args.description,
        provenance=parse_provenance(args.provenance),
        source_paths=source_paths,
        check_resolver=args.check_resolver,
        force_manifest=args.force_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
