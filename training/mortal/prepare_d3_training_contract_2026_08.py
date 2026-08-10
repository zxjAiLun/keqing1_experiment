#!/usr/bin/env python3
"""Prepare the frozen D3 6000h training source manifest / file index / contract.

Pure file/SHA preparation (no CUDA, no model, no training). The source set is
NOT taken from a bare directory glob: it is bound to the already-closed
generation provenance (aggregate audit verdict PASS + 24-row shard manifest),
then materialized as:

  d3_6000h_training_source_manifest.json / .tsv
  file_index_d3_k0.pth            (existing file-index contract)
  trainable_label.txt             (exactly "K0_70k")
  d3_training_data_contract.json  (frozen data/target contract; status filled
                                   by the audit on PASS)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from training.mortal.d3_continuation_contract import (  # noqa: E402
    shard_dir_name,
    shard_seed_end_exclusive,
    shard_seed_start,
)
from training.mortal.d3_production_audit_core import _canonical_log_hash, _log_key, _read_log  # noqa: E402

TRAINING_LABEL = "K0_70k"
ENV_PTS = [6, 4, 2, 0]
REWARD_MODE = "final_rank_mc"
OBJECTIVE_MODE = "behavior_action_mc"
VALUE_STATISTIC = "behavior_action_q"
PREFERENCE_LOSS = "existing_cql"

D3_EXP_ROOT = (
    REPO_ROOT
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
)
B250_DIR = D3_EXP_ROOT / "generation_production/shard_000_1800000_1800249"
CONT_DIR = D3_EXP_ROOT / "generation_continuation"
AGGREGATE_DIR = D3_EXP_ROOT / "generation_aggregate"
DEFAULT_OUTPUT_DIR = D3_EXP_ROOT / "training_contract_2026_08"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def shard_logs(shard_index: int) -> list[Path]:
    if shard_index == 0:
        return sorted(B250_DIR.glob("logs/*.json.gz"))
    return sorted((CONT_DIR / shard_dir_name(shard_index)).glob("logs/*.json.gz"))


def load_aggregate_provenance(aggregate_dir: Path) -> dict[str, Any]:
    audit = json.loads(
        (aggregate_dir / "d3_generation_6000h_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("gate", {}).get("verdict") != "PASS":
        raise ValueError("aggregate audit is not PASS; source set cannot be bound")
    manifest = json.loads((aggregate_dir / "shard_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("shards")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError(f"aggregate shard manifest must have 24 rows, got {len(rows) if rows else 0}")
    if any(row.get("verdict") != "PASS" for row in rows):
        raise ValueError("not all aggregate shard rows are PASS")
    return {"audit": audit, "shard_manifest": manifest}


def build_source_manifest(aggregate_dir: Path) -> dict[str, Any]:
    provenance = load_aggregate_provenance(aggregate_dir)
    shard_rows = {int(row["shard_index"]): row for row in provenance["shard_manifest"]["shards"]}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    canonical_seen: set[str] = set()
    for shard_index in range(24):
        shard_row = shard_rows[shard_index]
        for log_path in shard_logs(shard_index):
            events = _read_log(log_path)
            key = _log_key(events, log_path)
            seed, seed_key = key
            if key in seen:
                raise ValueError(f"duplicate seed/key: {key}")
            seen.add(key)
            names = events[0].get("names")
            if not isinstance(names, list) or names.count(TRAINING_LABEL) != 1:
                raise ValueError(f"log has != 1 {TRAINING_LABEL}: {log_path.name}")
            canonical = _canonical_log_hash(events)
            if canonical in canonical_seen:
                raise ValueError(f"duplicate canonical hanchan: {canonical}")
            canonical_seen.add(canonical)
            rows.append(
                {
                    "seed": seed,
                    "seed_key": seed_key,
                    "source_shard": shard_index,
                    "path": relative_path(log_path),
                    "compressed_sha256": sha256_file(log_path),
                    "canonical_hanchan_sha256": canonical,
                    "k0_seat": names.index(TRAINING_LABEL),
                    "source_protocol_sha256": shard_row.get("generation_protocol_sha256"),
                    "source_final_audit_sha256": shard_row.get("audit_v2_json_sha256"),
                }
            )
    rows.sort(key=lambda row: (row["seed"], row["seed_key"]))
    seeds = [(row["seed"], row["seed_key"]) for row in rows]
    if len(rows) != 6000:
        raise ValueError(f"expected 6000 rows, got {len(rows)}")
    if seeds != list(zip(range(1_800_000, 1_806_000), [8192] * 6000, strict=True)):
        raise ValueError("manifest seed set is not exactly 1800000..1805999 with key 8192")
    return {
        "schema": "keqing.mortal.d3_6000h_training_source_manifest.v1",
        "trainable_label": TRAINING_LABEL,
        "aggregate_audit_sha256": sha256_file(aggregate_dir / "d3_generation_6000h_audit.json"),
        "shard_manifest_sha256": sha256_file(aggregate_dir / "shard_manifest.json"),
        "file_count": len(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=AGGREGATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_source_manifest(args.aggregate_dir.resolve())
    json_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    json_path = output_dir / "d3_6000h_training_source_manifest.json"
    json_path.write_bytes(json_bytes)
    tsv_path = output_dir / "d3_6000h_training_source_manifest.tsv"
    tsv_lines = [
        "\t".join(
            [
                "seed",
                "seed_key",
                "source_shard",
                "path",
                "compressed_sha256",
                "canonical_hanchan_sha256",
                "k0_seat",
                "source_protocol_sha256",
                "source_final_audit_sha256",
            ]
        )
    ]
    for row in manifest["rows"]:
        tsv_lines.append(
            "\t".join(
                [
                    str(row["seed"]),
                    str(row["seed_key"]),
                    str(row["source_shard"]),
                    row["path"],
                    row["compressed_sha256"],
                    row["canonical_hanchan_sha256"],
                    str(row["k0_seat"]),
                    row["source_protocol_sha256"],
                    row["source_final_audit_sha256"],
                ]
            )
        )
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    file_list = [
        str((REPO_ROOT / row["path"]).resolve())
        for row in manifest["rows"]
    ]
    index_path = output_dir / "file_index_d3_k0.pth"
    torch.save({"file_list": file_list}, index_path)

    label_path = output_dir / "trainable_label.txt"
    label_path.write_text(TRAINING_LABEL + "\n", encoding="utf-8")

    contract = {
        "schema": "keqing.mortal.d3_training_data_contract.v1",
        "trainable_label": TRAINING_LABEL,
        "trainable_views_per_hanchan": 1,
        "source": {
            "hanchans": 6000,
            "seeds": "1800000..1805999",
            "seed_key": 8192,
            "source_manifest": json_path.name,
            "source_manifest_sha256": hashlib.sha256(json_bytes).hexdigest(),
            "file_index": index_path.name,
            "trainable_label_file": label_path.name,
        },
        "canonical_loader_view": {
            "oracle": False,
            "player_names": [TRAINING_LABEL],
            "augmented": False,
            "excludes": None,
            "always_include_kan_select": "mainline default (not overridden)",
        },
        "reward": {
            "mode": REWARD_MODE,
            "env_pts": ENV_PTS,
            "centered_targets": {
                "rank1": float(ENV_PTS[0] - sum(ENV_PTS) / len(ENV_PTS)),
                "rank2": float(ENV_PTS[1] - sum(ENV_PTS) / len(ENV_PTS)),
                "rank3": float(ENV_PTS[2] - sum(ENV_PTS) / len(ENV_PTS)),
                "rank4": float(ENV_PTS[3] - sum(ENV_PTS) / len(ENV_PTS)),
            },
            "target_formula": "pts[final_rank] - pts.mean()",
        },
        "objective": {
            "mode": OBJECTIVE_MODE,
            "value_statistic": VALUE_STATISTIC,
            "preference_loss": PREFERENCE_LOSS,
            "forbidden": [
                "top2 preference bonus",
                "event-Q regression target",
                "margin weighting",
                "importance sampling correction",
                "exploration weighting",
                "NAGA/reviewer labels",
                "filtering explored rows",
                "reverting actual_action to base_action/top1",
            ],
        },
        "generation_rank_points_are_not_training_targets": True,
        "status": "prepared_manifest_frozen_audit_pending",
    }
    contract_path = output_dir / "d3_training_data_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": contract["status"],
                "source_manifest_json_sha256": hashlib.sha256(json_bytes).hexdigest(),
                "source_manifest_tsv_sha256": sha256_file(tsv_path),
                "file_index_sha256": sha256_file(index_path),
                "trainable_label_sha256": sha256_file(label_path),
                "contract_json_sha256": sha256_file(contract_path),
                "file_count": len(manifest["rows"]),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
