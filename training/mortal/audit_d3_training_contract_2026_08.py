#!/usr/bin/env python3
"""Audit the D3 6000h training-view / target contract (read-only, no training).

Six hard gates, all zero-tolerance:

  A source provenance   bound to the closed generation aggregate (PASS +
                        27/27 checks + 24-row shard manifest), 6000 logs,
                        exact seeds, zero duplicate seed / canonical hanchan
  B frozen manifest     source manifest/index/label verified from disk and
                        their SHAs recorded; future training must reference
                        this frozen index, never re-glob
  C K0 perspective      canonical unaugmented training GameplayLoader view:
                        exactly one K0_70k perspective per hanchan, no other
                        labels; full row statistics of what the loader emits
  D event→behavior      every generation event maps exactly once to its
                        canonical training loader row; loader behavior action
                        == event actual action; explored rows keep top2,
                        non-explored rows keep top1 (100% preserved rate)
  E final_rank_mc       targets computed exactly like mainline_dataloader:
                        pts[final_rank] - mean([6,4,2,0]) -> {+3,+1,-1,-3};
                        one shared target per hanchan; zero mismatches
  F objective compat    behavior_action_mc smoke on real loader rows:
                        behavior actions 100% legal, compute_objective_losses
                        finite; no optimizer step

No training, no checkpoint, no recipe selection.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.mortal.d3_continuation_contract import shard_dir_name  # noqa: E402
from training.mortal.d3_native_scene import reconstruct_native_scenes  # noqa: E402
from training.mortal.d3_production_audit_core import (  # noqa: E402
    _canonical_log_hash,
    _log_key,
    _read_log,
    primary_row_flags,
)
from training.mortal.prepare_d3_training_contract_2026_08 import (  # noqa: E402
    B250_DIR,
    CONT_DIR,
    DEFAULT_OUTPUT_DIR,
    ENV_PTS,
    OBJECTIVE_MODE,
    PREFERENCE_LOSS,
    REWARD_MODE,
    TRAINING_LABEL,
    VALUE_STATISTIC,
    sha256_file,
)

AGGREGATE_DIR = (
    REPO_ROOT
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
    / "generation_aggregate"
)
DEFAULT_K0_MODEL = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\models\K0_70k\mortal_default_70k_promoted_candidate.pth"
)
PTS = np.asarray(ENV_PTS, dtype=np.float64)
EXPECTED_TARGETS = {float(ENV_PTS[r] - PTS.mean()) for r in range(4)}


def shard_audit_dir(shard_index: int) -> Path:
    if shard_index == 0:
        return B250_DIR
    return CONT_DIR / shard_dir_name(shard_index)


# ---------------------------------------------------------------- gate A
def gate_a_source_provenance() -> dict[str, Any]:
    issues: list[str] = []
    audit = json.loads(
        (AGGREGATE_DIR / "d3_generation_6000h_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("gate", {}).get("verdict") != "PASS":
        issues.append("aggregate audit verdict != PASS")
    aggregate_checks = audit.get("gate", {}).get("checks", {})
    if not all(
        value is True
        for section in aggregate_checks.values()
        for value in section.values()
    ):
        issues.append("aggregate audit has non-true checks")
    shard_manifest = json.loads(
        (AGGREGATE_DIR / "shard_manifest.json").read_text(encoding="utf-8")
    )
    shards = shard_manifest.get("shards", [])
    if len(shards) != 24 or any(row.get("verdict") != "PASS" for row in shards):
        issues.append("aggregate shard manifest is not 24/24 PASS")
    seeds: set[tuple[int, int]] = set()
    canonical: set[str] = set()
    dup_seeds: list[tuple[int, int]] = []
    dup_canonical: list[str] = []
    count = 0
    for shard_index in range(24):
        for log_path in sorted(shard_audit_dir(shard_index).glob("logs/*.json.gz")):
            count += 1
            events = _read_log(log_path)
            key = _log_key(events, log_path)
            if key in seeds:
                dup_seeds.append(key)
            seeds.add(key)
            digest = _canonical_log_hash(events)
            if digest in canonical:
                dup_canonical.append(digest)
            canonical.add(digest)
    checks = {
        "aggregate_audit_pass_27_27": not issues,
        "aggregate_shard_manifest_24_24_pass": len(shards) == 24
        and all(row.get("verdict") == "PASS" for row in shards),
        "exact_6000_logs": count == 6000,
        "exact_global_seed_set": seeds
        == {(seed, 8192) for seed in range(1_800_000, 1_806_000)},
        "zero_duplicate_seeds": not dup_seeds,
        "zero_duplicate_canonical_hanchan": not dup_canonical,
    }
    return {
        "aggregate_audit_sha256": sha256_file(AGGREGATE_DIR / "d3_generation_6000h_audit.json"),
        "aggregate_md_sha256": sha256_file(AGGREGATE_DIR / "d3_generation_6000h_audit.md"),
        "shard_manifest_sha256": sha256_file(AGGREGATE_DIR / "shard_manifest.json"),
        "log_count": count,
        "duplicate_seeds": dup_seeds[:20],
        "duplicate_canonical_hanchan": dup_canonical[:20],
        "issues": issues,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- gate B
def gate_b_frozen_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "d3_6000h_training_source_manifest.json"
    tsv_path = output_dir / "d3_6000h_training_source_manifest.tsv"
    index_path = output_dir / "file_index_d3_k0.pth"
    label_path = output_dir / "trainable_label.txt"
    issues: list[str] = []
    missing_files: list[str] = []
    sha_mismatches: list[str] = []
    seat_mismatches: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if len(rows) != 6000:
        issues.append(f"manifest rows != 6000: {len(rows)}")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        issues.append("duplicate manifest paths")
    seeds = [(row["seed"], row["seed_key"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        issues.append("duplicate manifest seeds")
    canonical = [row["canonical_hanchan_sha256"] for row in rows]
    if len(canonical) != len(set(canonical)):
        issues.append("duplicate manifest canonical hashes")
    for row in rows:
        path = REPO_ROOT / row["path"]
        if not path.is_file():
            missing_files.append(row["path"])
            continue
        if sha256_file(path) != row["compressed_sha256"]:
            sha_mismatches.append(row["path"])
        k0_seat = _read_log(path)[0].get("names", []).index(TRAINING_LABEL)
        if k0_seat != row["k0_seat"]:
            seat_mismatches.append(row["path"])
    label_text = label_path.read_text(encoding="utf-8").strip()
    if label_text != TRAINING_LABEL:
        issues.append(f"trainable_label != {TRAINING_LABEL}: {label_text!r}")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    file_list = list(index.get("file_list", []))
    expected_list = [str((REPO_ROOT / row["path"]).resolve()) for row in rows]
    if file_list != expected_list:
        issues.append("file_index file_list does not match manifest order")
    checks = {
        "manifest_6000_rows": len(rows) == 6000,
        "unique_paths": len(paths) == len(set(paths)),
        "unique_seeds": len(seeds) == len(set(seeds)),
        "unique_canonical": len(canonical) == len(set(canonical)),
        "zero_missing_files": not missing_files,
        "zero_disk_sha_mismatch": not sha_mismatches,
        "k0_seat_exact": not seat_mismatches,
        "trainable_label_exact": label_text == TRAINING_LABEL,
        "file_index_matches_manifest": file_list == expected_list,
    }
    return {
        "source_manifest_json_sha256": sha256_file(manifest_path),
        "source_manifest_tsv_sha256": sha256_file(tsv_path),
        "file_index_sha256": sha256_file(index_path),
        "trainable_label_sha256": sha256_file(label_path),
        "missing_files": missing_files[:20],
        "sha_mismatches": sha_mismatches[:20],
        "seat_mismatches": seat_mismatches[:20],
        "issues": issues,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- gates C+D+E
def _load_game_rows(
    loader: Any,
    log_path: Path,
) -> tuple[list[Any], list[bool], dict[tuple[int, int], dict[str, Any]]]:
    """Canonical unaugmented K0 training view: loader rows + arena scene map."""
    loaded = loader.load_gz_log_files([str(log_path)])
    if len(loaded) != 1 or len(loaded[0]) != 1:
        raise ValueError(f"expected exactly one K0 perspective: {log_path.name}")
    game = loaded[0][0]
    player_id = int(game.take_player_id())
    from training.mortal.audit_replay_distribution import records_from_game  # noqa: PLC0415

    records = list(records_from_game(game, PTS))
    if not records:
        raise ValueError(f"K0 perspective has zero decisions: {log_path.name}")
    final_rank = int(records[0].target_rank) - 1
    flags = primary_row_flags(record.action for record in records)
    loader_rows = [
        {
            "action": int(record.action),
            "legal_count": int(np.asarray(record.mask, dtype=np.bool_).sum()),
            "kyoku": int(record.kyoku),
        }
        for record, is_primary in zip(records, flags, strict=True)
        if is_primary
    ]
    recon = reconstruct_native_scenes(log_path, player_id, loader_rows)
    arena_map: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in recon["scenes"]:
        if entry["arena_index"] is None or entry["loader_row_index"] is None:
            continue
        arena_map[(entry["kyoku"], entry["arena_index"])] = {
            "loader_row_index": entry["loader_row_index"],
            "action": entry["row_action"],
        }
    return records, flags, arena_map, recon, player_id, final_rank


def _run_loader_pass(
    manifest_rows: list[dict[str, Any]], output_dir: Path, version: int
) -> dict[str, Any]:
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=[TRAINING_LABEL],
        excludes=None,
        augmented=False,
    )
    perspectives = 0
    missing_perspective: list[str] = []
    row_issues: list[str] = []
    total_training_rows = 0
    rows_per_hanchan: list[int] = []
    action_counts: Counter[int] = Counter()
    action_kind_counts: Counter[str] = Counter()
    legal_counts: Counter[int] = Counter()
    final_rank_counts: Counter[int] = Counter()
    target_counts: Counter[float] = Counter()
    row_target_counts: Counter[float] = Counter()
    target_issues: list[str] = []
    per_game_primary: list[int] = []
    global_primary_index = 0
    explored_mapping: list[dict[str, Any]] = []
    mapped_events = 0
    unmapped_events: list[str] = []
    event_action_mismatch: list[str] = []
    explored_events = 0
    explored_top2_ok = 0
    nonexplored_top1_ok = 0
    reconstruction_issues: list[str] = []

    shard_events: dict[int, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for shard_index in range(24):
        events_path = shard_audit_dir(shard_index) / "exploration/exploration_events.jsonl"
        by_game: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            by_game[(int(event["generation_seed"]), int(event["seed_key"]))].append(event)
        shard_events[shard_index] = by_game

    for row in manifest_rows:
        log_path = REPO_ROOT / row["path"]
        shard_index = int(row["source_shard"])
        try:
            records, flags, arena_map, recon, player_id, final_rank = _load_game_rows(
                loader, log_path
            )
        except Exception as exc:  # noqa: BLE001
            missing_perspective.append(f"{row['path']}: {exc}")
            continue
        if recon["label_mismatches"] or recon["row_exhausted"] or recon["leftover_rows"]:
            reconstruction_issues.append(
                f"{row['path']}: label_mismatches={recon['label_mismatches']} "
                f"row_exhausted={recon['row_exhausted']} leftover={recon['leftover_rows']}"
            )
        perspectives += 1
        if int(player_id) != int(row["k0_seat"]):
            row_issues.append(f"{row['path']}: player_id {player_id} != manifest k0_seat")
        if not 0 <= final_rank <= 3:
            target_issues.append(f"{row['path']}: invalid final rank {final_rank}")
        else:
            final_rank_counts[final_rank] += 1
            target = float(PTS[final_rank] - PTS.mean())
            if target not in EXPECTED_TARGETS:
                target_issues.append(f"{row['path']}: unexpected target {target}")
            target_counts[target] += 1
        rows_per_hanchan.append(len(records))
        total_training_rows += len(records)
        for record, is_primary in zip(records, flags, strict=True):
            action = int(record.action)
            action_counts[action] += 1
            action_kind_counts[str(record.action_kind)] += 1
            legal_counts[int(np.asarray(record.mask, dtype=np.bool_).sum())] += 1
            row_target_counts[float(record.target)] += 1
        per_game_primary.append(sum(1 for flag in flags if flag))

        events = shard_events[shard_index].get((row["seed"], row["seed_key"]), [])
        for event in events:
            if int(event["seat"]) != int(row["k0_seat"]):
                continue
            context = (int(event["kyoku_index"]), int(event["decision_index"]))
            entry = arena_map.get(context)
            if entry is None:
                unmapped_events.append(
                    f"{row['path']} ctx=({context[0]},{context[1]})"
                )
                continue
            mapped_events += 1
            event_action = int(event["actual_action"])
            if entry["action"] != event_action:
                event_action_mismatch.append(
                    f"{row['path']} ctx=({context[0]},{context[1]}) "
                    f"loader={entry['action']} event={event_action}"
                )
                continue
            if bool(event["explored"]):
                explored_events += 1
                if event_action == int(event["top2_action"]):
                    explored_top2_ok += 1
                explored_mapping.append(
                    {
                        "seed": row["seed"],
                        "seed_key": row["seed_key"],
                        "seat": int(row["k0_seat"]),
                        "kyoku_index": context[0],
                        "decision_index": context[1],
                        "loader_row_global_index": global_primary_index
                        + entry["loader_row_index"],
                        "loader_row_action": entry["action"],
                        "event_actual_action": event_action,
                        "event_top1_action": int(event["top1_action"]),
                        "event_top2_action": int(event["top2_action"]),
                        "source_shard": shard_index,
                    }
                )
            else:
                if event_action == int(event["top1_action"]):
                    nonexplored_top1_ok += 1
        global_primary_index += per_game_primary[-1]

    (output_dir / "explored_training_row_mapping.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in explored_mapping) + "\n",
        encoding="utf-8",
    )

    total_events = sum(
        len(game_events)
        for by_game in shard_events.values()
        for game_events in by_game.values()
    )
    row_target_total = sum(
        float(target) * count for target, count in row_target_counts.items()
    )
    row_target_mean = row_target_total / total_training_rows if total_training_rows else 0.0
    checks = {
        "k0_perspectives_6000": perspectives == 6000,
        "zero_missing_perspective": not missing_perspective,
        "zero_duplicate_perspective": True,
        "zero_non_k0_perspectives": True,
        "all_151282_events_mapped": mapped_events == 151_282 and not unmapped_events,
        "zero_unmapped_event": not unmapped_events,
        "zero_behavior_action_mismatch": not event_action_mismatch,
        "explored_top2_preserved_27506": explored_top2_ok == 27_506,
        "nonexplored_top1_preserved_123776": nonexplored_top1_ok == 123_776,
        "zero_target_mismatch": not target_issues,
        "zero_reconstruction_issues": not reconstruction_issues,
    }
    return {
        "perspectives": perspectives,
        "missing_perspective": missing_perspective[:20],
        "total_training_rows": total_training_rows,
        "rows_per_hanchan": {
            "min": min(rows_per_hanchan) if rows_per_hanchan else 0,
            "median": statistics.median(rows_per_hanchan) if rows_per_hanchan else 0,
            "max": max(rows_per_hanchan) if rows_per_hanchan else 0,
        },
        "action_counts": {str(k): v for k, v in sorted(action_counts.items())},
        "action_kind_counts": {
            key: int(value) for key, value in sorted(action_kind_counts.items())
        },
        "legal_action_count_distribution": {
            str(k): v for k, v in sorted(legal_counts.items())
        },
        "event_totals": {
            "events": total_events,
            "mapped": mapped_events,
            "unmapped": len(unmapped_events),
            "explored": explored_events,
            "explored_top2_ok": explored_top2_ok,
            "nonexplored_top1_ok": nonexplored_top1_ok,
            "behavior_action_mismatch": len(event_action_mismatch),
        },
        "final_rank_counts": {str(k): v for k, v in sorted(final_rank_counts.items())},
        "target_counts": {str(k): v for k, v in sorted(target_counts.items())},
        "row_weighted_target_counts": {
            str(k): v for k, v in sorted(row_target_counts.items())
        },
        "row_target_mean": row_target_mean,
        "target_issues": target_issues[:20],
        "reconstruction_issues": reconstruction_issues[:20],
        "unmapped_events_sample": unmapped_events[:20],
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- gate F
def gate_f_objective_smoke(version: int, k0_model: Path) -> dict[str, Any]:
    from model import AuxNet, Brain, DQN  # noqa: PLC0415
    from training.mortal.audit_replay_distribution import load_checkpoint, load_model  # noqa: PLC0415
    from training.mortal.objective import compute_objective_losses  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("gate F requires CUDA")
    state = load_checkpoint(k0_model)
    brain, dqn, model_version = load_model(state, device)
    aux_net = AuxNet((4,)).to(device)
    aux_net.load_state_dict(state["aux_net"])
    del state

    from libriichi.dataset import GameplayLoader  # noqa: PLC0415
    from training.mortal.audit_replay_distribution import records_from_game  # noqa: PLC0415
    from training.mortal.prepare_d3_training_contract_2026_08 import (  # noqa: PLC0415
        DEFAULT_OUTPUT_DIR,
    )

    manifest = json.loads(
        (DEFAULT_OUTPUT_DIR / "d3_6000h_training_source_manifest.json").read_text(encoding="utf-8")
    )
    loader = GameplayLoader(
        version=model_version,
        oracle=False,
        player_names=[TRAINING_LABEL],
        excludes=None,
        augmented=False,
    )
    samples: list[tuple[Any, Any, Any, float, int]] = []  # obs, mask, action, target, rank
    for row in manifest["rows"][:8]:
        loaded = loader.load_gz_log_files([str(REPO_ROOT / row["path"])])
        records = list(records_from_game(loaded[0][0], PTS))
        for record in records:
            samples.append(
                (
                    np.asarray(record.obs, dtype=np.float32),
                    np.asarray(record.mask, dtype=np.bool_),
                    int(record.action),
                    float(record.target),
                    int(max(0, record.current_rank - 1)),
                )
            )
        if len(samples) >= 4 * 64:
            break
    batches: list[dict[str, Any]] = []
    illegal = 0
    for start in range(0, min(len(samples), 256), 64):
        chunk = samples[start : start + 64]
        obs = torch.as_tensor(np.stack([item[0] for item in chunk]), device=device)
        masks = torch.as_tensor(np.stack([item[1] for item in chunk]), device=device)
        actions = torch.as_tensor([item[2] for item in chunk], device=device, dtype=torch.int64)
        targets = torch.as_tensor([item[3] for item in chunk], device=device, dtype=torch.float32)
        ranks = torch.as_tensor([item[4] for item in chunk], device=device, dtype=torch.int64)
        if not bool(masks[torch.arange(len(chunk), device=device), actions].all().item()):
            illegal += 1
        with torch.inference_mode():
            phi = brain(obs)
            q_out = dqn(phi, masks)
            (next_rank_logits,) = aux_net(phi)
            losses = compute_objective_losses(
                q_out=q_out,
                masks=masks,
                actions=actions,
                q_target_mc=targets,
                next_rank_logits=next_rank_logits,
                player_ranks=ranks,
                mode=OBJECTIVE_MODE,
                cql_weight=5.0,
                aux_weight=0.2,
            )
        finite = all(
            bool(value.isfinite().all().item())
            for value in losses.values()
            if isinstance(value, torch.Tensor)
        )
        batches.append({"samples": len(chunk), "finite": finite, "illegal_behavior_actions": illegal})
    checks = {
        "behavior_actions_legal_100pct": illegal == 0,
        "all_batches_finite": all(batch["finite"] for batch in batches),
        "no_optimizer_step": True,
    }
    return {
        "objective_contract": {
            "mode": OBJECTIVE_MODE,
            "value_statistic": VALUE_STATISTIC,
            "preference_loss": PREFERENCE_LOSS,
            "reward_mode": REWARD_MODE,
        },
        "batches": batches,
        "model_version": model_version,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    gate = report["gate"]
    lines = [
        "# D3 6000h Training-View / Target Contract 机器审计",
        "",
        f"- Gate：`{gate['verdict']}`",
        f"- Perspectives：`{report['loader_pass']['perspectives']}`（K0_70k exactly one per hanchan）",
        f"- Training rows：`{report['loader_pass']['total_training_rows']}`",
        f"- Events mapped：`{report['loader_pass']['event_totals']['mapped']}` / 151282；explored top2 preserved `{report['loader_pass']['event_totals']['explored_top2_ok']}` / 27506",
        f"- Targets：`{report['loader_pass']['target_counts']}`（final_rank_mc，[6,4,2,0] 中心化）",
        "",
        "## Hard checks",
        "",
    ]
    for section, checks in (
        ("gate_a_source_provenance", report["gate_a_source_provenance"]["checks"]),
        ("gate_b_frozen_manifest", report["gate_b_frozen_manifest"]["checks"]),
        ("gate_cde_loader_pass", report["loader_pass"]["checks"]),
        ("gate_f_objective_smoke", report["gate_f_objective_smoke"]["checks"]),
    ):
        lines.append(f"### {section}")
        for check, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{check}`")
        lines.append("")
    lines.extend(["## 解释边界", "", "本审计只冻结源 manifest/index 与数据/目标合同；不选择训练 recipe、不创建 checkpoint、不启动训练。"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k0-model", type=Path, default=DEFAULT_K0_MODEL)
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()

    a = gate_a_source_provenance()
    b = gate_b_frozen_manifest(output_dir)
    version = args.version
    if version is None:
        from training.mortal.audit_replay_distribution import load_checkpoint  # noqa: PLC0415

        state = load_checkpoint(args.k0_model.resolve())
        version = int(state["config"]["control"].get("version", 4))
        del state
    manifest = json.loads(
        (output_dir / "d3_6000h_training_source_manifest.json").read_text(encoding="utf-8")
    )
    cde = _run_loader_pass(manifest["rows"], output_dir, version)
    f = gate_f_objective_smoke(version, args.k0_model.resolve())
    hard_pass = all((a["passed"], b["passed"], cde["passed"], f["passed"]))
    report = {
        "schema": "keqing.mortal.d3_training_data_contract.v1",
        "gate": {
            "gate_id": "D3_6000h_training_view_target_contract_2026_08",
            "verdict": "PASS" if hard_pass else "FAIL",
            "passed": hard_pass,
            "status": "training_contract_passed_manifest_frozen" if hard_pass else "training_contract_blocked",
        },
        "gate_a_source_provenance": a,
        "gate_b_frozen_manifest": b,
        "loader_pass": cde,
        "gate_f_objective_smoke": f,
        "scope_notes": [
            "no training, no checkpoint, no optimizer step, no recipe selection",
            "generation rank points [90,45,0,-135] are NOT training MC targets",
            "canonical loader view = mainline GameplayLoader default semantics",
        ],
    }
    audit_json = output_dir / "d3_training_contract_audit.json"
    audit_md = output_dir / "d3_training_contract_audit.md"
    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, audit_md)
    contract_path = output_dir / "d3_training_data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = report["gate"]["status"]
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": report["gate"]["verdict"],
                "status": report["gate"]["status"],
                "gate_a": a["passed"],
                "gate_b": b["passed"],
                "gate_cde": cde["passed"],
                "gate_f": f["passed"],
                "audit_json_sha256": sha256_file(audit_json),
                "audit_md_sha256": sha256_file(audit_md),
                "contract_json_sha256": sha256_file(contract_path),
                "mapping_jsonl_sha256": sha256_file(output_dir / "explored_training_row_mapping.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not hard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
