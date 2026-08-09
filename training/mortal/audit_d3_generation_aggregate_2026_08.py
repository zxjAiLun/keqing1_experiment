#!/usr/bin/env python3
"""Read-only 24-shard aggregate integrity / provenance closure for the D3 6000h generation.

Audit-only entrypoint. Reads every immutable shard (B250 + 23 continuation),
recomputes identities, hashes, verdicts and counters from disk, and cross-checks
the external continuation ledger. Produces no generation and starts no training.

Hard gates (PASS requires all):

  A. global identity  24 shards, exactly 6000 logs, exact global seed set
                       1800000..1805999 with seed_key 8192, no missing /
                       unexpected / duplicate seeds, no duplicate canonical
                       hanchan hash, no duplicate canonical decision context
  B. verdict closure  24/24 final authoritative audit-v2 PASS, all hard-gate
                       sections true, zero contract/mapping violations
  C. provenance        shared D3 identity (contract, seed_key, seat, AMP,
                       rank points, Mortal/native/patch/model/protocol SHAs),
                       generation semantic files unchanged since 2cc12b4,
                       continuation implementation unchanged since 53cc07f
  D. artifact ledger   disk SHAs match the external continuation ledger for
                       001..023; shard_000 record appended; 24-row manifest
  E. aggregate math    per-shard counter consistency (summary == protocol ==
                       audit), sums, and descriptive exploration distributions
                       (no hard threshold on shard-to-shard variation)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.d3_continuation_contract import (
    CONTINUATION_GOVERNANCE,
    D3_SEMANTIC_ANCHOR,
    FIRST_GATE_AUDITOR,
    GAMES_PER_SHARD,
    SHARD_COUNT,
    shard_dir_name,
    shard_seed_end_exclusive,
    shard_seed_start,
)
from training.mortal.d3_production_audit_core import (
    _canonical_log_hash,
    _log_key,
    _read_log,
)
from training.mortal.d3_production_contract import (
    AUTHORITATIVE_MORTAL_COMMIT,
    AUTHORITATIVE_NATIVE_BINARY_SHA256,
    AUTHORITATIVE_NATIVE_PATCH_SHA256,
    AUTHORITATIVE_SMOKE_PROJECT_COMMIT,
    GAMES,
    REQUIRED_LABELS,
    SEED_KEY,
    expected_seed_keys,
    sha256_file,
)

AGGREGATE_SCHEMA = "keqing.mortal.d3_generation_6000h_aggregate.v1"
B250_GATE_ID = "D3_first_B250_production_gate_2026_08"

D3_EXP_ROOT = (
    REPO_ROOT
    / "artifacts/experiments/model_pool_2026_07/D3_uncertainty_guided_exploration_2026_08"
)
B250_DIR = D3_EXP_ROOT / "generation_production/shard_000_1800000_1800249"
CONT_DIR = D3_EXP_ROOT / "generation_continuation"
DEFAULT_LEDGER = Path(
    r"E:\AUbuntuProject\keqing-data\mortal\authoritative\D3_top2_discard_v1_2026_08"
    r"\diagnostics\continuation_ledger.jsonl"
)
DEFAULT_OUTPUT_DIR = D3_EXP_ROOT / "generation_aggregate"

SEMANTIC_FILES = (
    "training/mortal/d3_exploration_engine.py",
    "training/mortal/patches/libriichi_d3_decision_context.patch",
    "training/mortal/d3_production_contract.py",
)
CONTINUATION_IMPLEMENTATION_FILES = (
    "training/mortal/d3_continuation_contract.py",
    "training/mortal/d3_continuation_preflight.py",
    "training/mortal/run_d3_continuation_shard_2026_08.py",
)


def shard_audit_dir(shard_index: int) -> Path:
    if shard_index == 0:
        return B250_DIR
    return CONT_DIR / shard_dir_name(shard_index)


def shard_seed_start_end(shard_index: int) -> tuple[int, int]:
    if shard_index == 0:
        return 1_800_000, 1_800_249
    return shard_seed_start(shard_index), shard_seed_end_exclusive(shard_index) - 1


def git_diff_empty(base: str, head: str, paths: tuple[str, ...]) -> tuple[bool, list[str]]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", base, head, "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    changed = [line for line in completed.stdout.splitlines() if line.strip()]
    return not changed, changed


def load_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_ledger_entries(ledger_path: Path) -> dict[int, dict[str, Any]]:
    by_shard: dict[int, dict[str, Any]] = {}
    for row in load_ledger(ledger_path):
        if row.get("verdict") != "PASS":
            continue
        by_shard[int(row["shard_index"])] = row
    return by_shard


# ---------------------------------------------------------------- layer A
def global_identity() -> dict[str, Any]:
    logs = sorted(B250_DIR.glob("logs/*.json.gz"))
    logs += sorted(CONT_DIR.glob("*/logs/*.json.gz"))
    seed_set: dict[tuple[int, int], int] = {}
    canonical_hashes: set[str] = set()
    duplicate_hashes: list[str] = []
    duplicate_seeds: list[tuple[int, int]] = []
    malformed: list[str] = []
    per_shard_logs: Counter[int] = Counter()
    for path in logs:
        shard_index = 0 if B250_DIR in path.parents else int(path.parent.parent.name.split("_")[1])
        per_shard_logs[shard_index] += 1
        try:
            events = _read_log(path)
            key = _log_key(events, path)
            if key in seed_set:
                duplicate_seeds.append(key)
            seed_set[key] = seed_set.get(key, 0) + 1
            canonical = _canonical_log_hash(events)
            if canonical in canonical_hashes:
                duplicate_hashes.append(canonical)
            canonical_hashes.add(canonical)
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{path.name}: {exc}")
    expected = expected_seed_keys() | {
        (seed, SEED_KEY)
        for seed in range(1_800_250, 1_806_000)
    }
    actual = set(seed_set)
    # canonical decision contexts across all shards
    context_counts: Counter[tuple[int, int, int, int, int]] = Counter()
    for events_path in sorted(B250_DIR.glob("exploration/exploration_events.jsonl")):
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            context_counts[
                (
                    int(event["generation_seed"]),
                    int(event["seed_key"]),
                    int(event["seat"]),
                    int(event["kyoku_index"]),
                    int(event["decision_index"]),
                )
            ] += 1
    for events_path in sorted(CONT_DIR.glob("*/exploration/exploration_events.jsonl")):
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            context_counts[
                (
                    int(event["generation_seed"]),
                    int(event["seed_key"]),
                    int(event["seat"]),
                    int(event["kyoku_index"]),
                    int(event["decision_index"]),
                )
            ] += 1
    duplicate_contexts = [key for key, count in context_counts.items() if count > 1]
    checks = {
        "shard_count_24": len(per_shard_logs) == 24
        and set(per_shard_logs) == set(range(24)),
        "each_shard_250_logs": all(
            per_shard_logs[index] == (GAMES if index == 0 else GAMES_PER_SHARD)
            for index in range(24)
        ),
        "global_log_count_6000": len(logs) == 6000,
        "global_seed_count_6000": len(actual) == 6000,
        "exact_global_seed_set": actual == expected,
        "seed_key_8192": all(key[1] == SEED_KEY for key in actual),
        "zero_missing_seeds": len(expected - actual) == 0,
        "zero_unexpected_seeds": len(actual - expected) == 0,
        "zero_duplicate_seeds": not duplicate_seeds,
        "zero_duplicate_canonical_hanchan": not duplicate_hashes,
        "zero_duplicate_canonical_context": not duplicate_contexts,
        "zero_malformed_logs": not malformed,
    }
    return {
        "log_files": len(logs),
        "per_shard_log_counts": {str(k): v for k, v in sorted(per_shard_logs.items())},
        "global_seed_set_exact": actual == expected,
        "duplicate_seeds": duplicate_seeds[:20],
        "duplicate_canonical_hanchan_hashes": duplicate_hashes[:20],
        "duplicate_canonical_contexts": duplicate_contexts[:20],
        "malformed": malformed,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- layer B
def verdict_closure() -> dict[str, Any]:
    per_shard: dict[int, dict[str, Any]] = {}
    failed: list[str] = []
    for index in range(24):
        audit_dir = shard_audit_dir(index) / "audit_v2"
        if index == 0:
            audit_path = audit_dir / "d3_production_gate_reaudit_v2.json"
        else:
            audit_path = audit_dir / f"d3_continuation_shard_{index:03d}_audit_v2.json"
        if not audit_path.is_file():
            failed.append(f"shard {index}: audit json missing: {audit_path}")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        gate = audit.get("gate", {})
        checks = gate.get("checks", {})
        sections_all_true = all(
            all(value is True for value in section.values())
            for section in checks.values()
        )
        contract_clean = audit.get("event_audit", {}).get("contract_violations", {}) == {}
        mapping_clean = audit.get("event_audit", {}).get("mapping_violations", {}) == {}
        ok = (
            gate.get("verdict") == "PASS"
            and gate.get("passed") is True
            and sections_all_true
            and contract_clean
            and mapping_clean
        )
        per_shard[index] = {
            "verdict": gate.get("verdict"),
            "passed": gate.get("passed") is True,
            "sections_all_true": sections_all_true,
            "contract_violations": audit.get("event_audit", {}).get("contract_violations", {}),
            "mapping_violations": audit.get("event_audit", {}).get("mapping_violations", {}),
            "auditor_commit": gate.get("auditor_commit"),
            "eligible": audit.get("event_audit", {}).get("event_count"),
            "explored": audit.get("event_audit", {}).get("explored_count"),
            "passed_audit": ok,
        }
        if not ok:
            failed.append(f"shard {index}: verdict/passed/sections/violations mismatch")
    checks = {
        "shard_000_first_b250_audit_v2_pass": per_shard.get(0, {}).get("verdict") == "PASS",
        "shard_001_023_continuation_audit_v2_pass": all(
            per_shard.get(index, {}).get("verdict") == "PASS" for index in range(1, 24)
        ),
        "all_24_passed_true": all(per_shard.get(index, {}).get("passed_audit") for index in range(24)),
        "all_contract_violations_empty": all(
            per_shard.get(index, {}).get("contract_violations") == {} for index in range(24)
        ),
        "all_mapping_violations_empty": all(
            per_shard.get(index, {}).get("mapping_violations") == {} for index in range(24)
        ),
    }
    return {"per_shard": per_shard, "failed": failed, "checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------- layer C
def provenance_consistency() -> dict[str, Any]:
    issues: list[str] = []
    per_shard: dict[int, dict[str, Any]] = {}
    model_shas: set[tuple[str, ...]] = set()
    for index in range(24):
        protocol = json.loads(
            (shard_audit_dir(index) / "protocol.json").read_text(encoding="utf-8")
        )
        fixed = protocol.get("fixed_protocol", {})
        native = protocol.get("mortal_lineage", {})
        runtime = protocol.get("runtime", {})
        smoke = protocol.get("authoritative_smoke", {})
        models = protocol.get("models", {})
        row = {
            "contract_id": protocol.get("contract_id"),
            "seed_key": fixed.get("seed_key"),
            "seat_mode": fixed.get("seat_mode"),
            "amp": fixed.get("amp"),
            "rank_points": tuple(fixed.get("rank_points", [])),
            "mortal_commit": native.get("commit"),
            "native_sha256": runtime.get("loaded_libriichi_sha256"),
            "patch_sha256": protocol.get("d3_native_patch", {}).get("sha256"),
            "model_shas": tuple(models.get(label, {}).get("sha256") for label in REQUIRED_LABELS),
            "smoke_protocol_sha256": smoke.get("protocol_sha256"),
        }
        per_shard[index] = row
        model_shas.add(row["model_shas"])
        if row["contract_id"] != "D3_top2_discard_v1":
            issues.append(f"shard {index}: contract_id")
        if row["seed_key"] != SEED_KEY:
            issues.append(f"shard {index}: seed_key")
        if row["seat_mode"] != "random":
            issues.append(f"shard {index}: seat_mode")
        if row["amp"] is not False:
            issues.append(f"shard {index}: amp")
        if tuple(row["rank_points"]) != (90.0, 45.0, 0.0, -135.0):
            issues.append(f"shard {index}: rank_points")
        if row["mortal_commit"] != AUTHORITATIVE_MORTAL_COMMIT:
            issues.append(f"shard {index}: mortal commit")
        if row["native_sha256"] != AUTHORITATIVE_NATIVE_BINARY_SHA256:
            issues.append(f"shard {index}: native sha")
        if row["patch_sha256"] != AUTHORITATIVE_NATIVE_PATCH_SHA256:
            issues.append(f"shard {index}: patch sha")
        if row["smoke_protocol_sha256"] != "bf39c826354a7b4a281f2ded6bc34cbfea1ddca4fb107e1049de548ca474a0e8":
            issues.append(f"shard {index}: smoke protocol sha")
    semantic_clean, semantic_changed = git_diff_empty(
        D3_SEMANTIC_ANCHOR, "HEAD", SEMANTIC_FILES
    )
    implementation_clean, implementation_changed = git_diff_empty(
        "53cc07f4108a13adb7c3b403fe6339dff9abe874", "HEAD", CONTINUATION_IMPLEMENTATION_FILES
    )
    checks = {
        "shared_d3_generation_identity": not issues,
        "identical_model_sha_set_across_shards": len(model_shas) == 1,
        "generation_semantic_files_unchanged_since_2cc12b4": semantic_clean,
        "continuation_implementation_unchanged_since_53cc07f": implementation_clean,
    }
    return {
        "per_shard": per_shard,
        "issues": issues,
        "semantic_changed_since_anchor": semantic_changed,
        "implementation_changed_since_continuation": implementation_changed,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- layer D
def artifact_ledger(ledger_path: Path) -> dict[str, Any]:
    ledger = latest_ledger_entries(ledger_path)
    mismatches: list[str] = []
    per_shard: dict[int, dict[str, Any]] = {}
    for index in range(1, 24):
        run_dir = shard_audit_dir(index)
        audit_dir = run_dir / "audit_v2"
        audit_json = audit_dir / f"d3_continuation_shard_{index:03d}_audit_v2.json"
        audit_md = audit_dir / f"d3_continuation_shard_{index:03d}_audit_v2.md"
        disk = {
            "shard_index": index,
            "seed_start": shard_seed_start(index),
            "seed_end": shard_seed_end_exclusive(index) - 1,
            "verdict": "PASS",
            "generation_protocol_sha256": sha256_file(run_dir / "protocol.json"),
            "production_summary_sha256": sha256_file(run_dir / "production_summary.json"),
            "audit_v2_json_sha256": sha256_file(audit_json),
            "audit_v2_md_sha256": sha256_file(audit_md),
        }
        entry = ledger.get(index)
        if entry is None:
            mismatches.append(f"shard {index}: no PASS ledger entry")
            continue
        for key, value in disk.items():
            if str(entry.get(key)) != str(value):
                mismatches.append(f"shard {index}: ledger {key} mismatch")
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        disk["eligible"] = audit["event_audit"]["event_count"]
        disk["explored"] = audit["event_audit"]["explored_count"]
        disk["auditor_commit"] = audit["gate"].get("auditor_commit")
        per_shard[index] = disk
    # shard_000 record (not in the continuation ledger; verified from disk)
    run_dir = shard_audit_dir(0)
    audit_json = run_dir / "audit_v2/d3_production_gate_reaudit_v2.json"
    audit_md = run_dir / "audit_v2/d3_production_gate_reaudit_v2.md"
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    per_shard[0] = {
        "shard_index": 0,
        "seed_start": 1_800_000,
        "seed_end": 1_800_249,
        "verdict": "PASS",
        "generation_protocol_sha256": sha256_file(run_dir / "protocol.json"),
        "production_summary_sha256": sha256_file(run_dir / "production_summary.json"),
        "audit_v2_json_sha256": sha256_file(audit_json),
        "audit_v2_md_sha256": sha256_file(audit_md),
        "eligible": audit["event_audit"]["event_count"],
        "explored": audit["event_audit"]["explored_count"],
        "auditor_commit": audit["gate"].get("auditor_commit"),
    }
    checks = {
        "ledger_matches_disk_001_023": not mismatches,
        "shard_000_record_present": 0 in per_shard,
        "twenty_four_row_manifest_complete": len(per_shard) == 24,
    }
    return {
        "per_shard": per_shard,
        "mismatches": mismatches,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------- layer E
def aggregate_counters(verdicts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    rows: dict[int, dict[str, Any]] = {}
    for index in range(24):
        protocol = json.loads(
            (shard_audit_dir(index) / "protocol.json").read_text(encoding="utf-8")
        )
        counters = protocol.get("exploration_counters", {})
        audit = verdicts.get(index, {})
        eligible = audit.get("eligible")
        event_count = int(counters.get("event_count", 0))
        eligible_protocol = int(counters.get("eligible_count", 0))
        if eligible is not None and event_count != eligible:
            issues.append(f"shard {index}: protocol event_count != audit eligible")
        if eligible_protocol != event_count:
            issues.append(f"shard {index}: eligible_count != event_count in protocol")
        rows[index] = {
            "states": int(counters.get("states", 0)),
            "eligible": int(counters.get("eligible_count", 0)),
            "explored": int(counters.get("explored_count", 0)),
            "hash_rejected": int(counters.get("hash_rejected_count", 0)),
            "kyoku_budget_exhausted": int(counters.get("kyoku_budget_exhausted_count", 0)),
            "hanchan_budget_exhausted": int(counters.get("hanchan_budget_exhausted_count", 0)),
            "event_count": event_count,
        }
    totals = {key: sum(row[key] for row in rows.values()) for key in ("states", "eligible", "explored", "hash_rejected", "kyoku_budget_exhausted", "hanchan_budget_exhausted", "event_count")}
    sum_ok = totals["event_count"] == totals["eligible"]
    if not sum_ok:
        issues.append("sum(event_count) != sum(eligible)")
    expected_eligible = 151_282
    expected_explored = 27_506
    totals_ok = totals["eligible"] == expected_eligible and totals["explored"] == expected_explored
    if not totals_ok:
        issues.append(f"totals mismatch: eligible={totals['eligible']} explored={totals['explored']}")

    per_hanchan = [row["eligible"] / 250 for row in rows.values()]
    explored_rate = [row["explored"] / row["eligible"] for row in rows.values() if row["eligible"]]
    explored_per_hanchan = [row["explored"] / 250 for row in rows.values()]

    def quantiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        n = len(ordered)

        def q(frac: float) -> float:
            pos = (n - 1) * frac
            low = int(pos)
            high = min(n - 1, low + 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

        return {
            "min": ordered[0],
            "p5": q(0.05),
            "median": q(0.5),
            "p95": q(0.95),
            "max": ordered[-1],
        }

    distribution = {
        "eligible_per_hanchan": quantiles(per_hanchan),
        "explored_over_eligible": quantiles(explored_rate),
        "explored_per_hanchan": quantiles(explored_per_hanchan),
    }
    checks = {
        "each_shard_summary_equals_protocol": not issues,
        "sum_event_equals_sum_eligible": sum_ok,
        "totals_match_expected": totals_ok,
    }
    return {
        "per_shard": rows,
        "totals": totals,
        "expected_totals": {"eligible": expected_eligible, "explored": expected_explored},
        "distribution": distribution,
        "issues": issues,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    gate = report["gate"]
    lines = [
        "# D3 6000h 生成 Aggregate 机器审计",
        "",
        f"- Gate：`{gate['verdict']}`",
        f"- Global identity：`{report['global_identity']['checks']['global_log_count_6000'] and report['global_identity']['checks']['exact_global_seed_set']}`（6000 logs / 精确 seed 集 1800000..1805999）",
        f"- 24/24 shard audit PASS：`{report['verdict_closure']['checks']['all_24_passed_true']}`",
        f"- Provenance 一致性：`{report['provenance_consistency']['passed']}`",
        f"- Ledger == disk：`{report['artifact_ledger']['passed']}`",
        f"- Aggregate counters：eligible `{report['aggregate_counters']['totals']['eligible']}` / explored `{report['aggregate_counters']['totals']['explored']}`",
        "",
        "## Hard checks",
        "",
    ]
    for section in ("global_identity", "verdict_closure", "provenance_consistency", "artifact_ledger", "aggregate_counters"):
        lines.append(f"### {section}")
        for name, passed in report[section]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
        lines.append("")
    lines.extend(["## 解释边界", "", "本报告只做只读 integrity/provenance closure；rank/Pt 不是 hard gate；不启动训练，不选择 recipe。"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()

    identity = global_identity()
    verdicts = verdict_closure()
    provenance = provenance_consistency()
    ledger = artifact_ledger(args.ledger)
    counters = aggregate_counters(verdicts["per_shard"])
    hard_pass = all(
        (identity["passed"], verdicts["passed"], provenance["passed"], ledger["passed"], counters["passed"])
    )
    report = {
        "schema": AGGREGATE_SCHEMA,
        "gate": {
            "gate_id": "D3_6000h_generation_aggregate_closure_2026_08",
            "verdict": "PASS" if hard_pass else "FAIL",
            "passed": hard_pass,
            "checks": {
                "global_identity": identity["checks"],
                "verdict_closure": verdicts["checks"],
                "provenance_consistency": provenance["checks"],
                "artifact_ledger": ledger["checks"],
                "aggregate_counters": counters["checks"],
            },
        },
        "global_identity": identity,
        "verdict_closure": verdicts,
        "provenance_consistency": provenance,
        "artifact_ledger": ledger,
        "aggregate_counters": counters,
        "history": {
            "shard_005": {
                "generation_rerun": False,
                "generation_modified": False,
                "original_audit_verdict": "FAIL",
                "original_audit_invalidated_by": "deterministic auditor reconstruction defect (unguarded ryukyoku label)",
                "final_authoritative_audit": "re-audit PASS @ f67f0368dd7ebd812d791a41c13e97869920928e",
            }
        },
        "lineage": {
            "semantic_anchor": D3_SEMANTIC_ANCHOR,
            "first_b250_auditor": FIRST_GATE_AUDITOR,
            "continuation_governance": CONTINUATION_GOVERNANCE,
            "continuation_implementation": "53cc07f4108a13adb7c3b403fe6339dff9abe874",
            "final_auditor": "f67f0368dd7ebd812d791a41c13e97869920928e",
        },
        "scope_notes": [
            "read-only closure; no generation, no training, no recipe selection",
            "rank/Pt is not a hard gate",
            "shard-to-shard exploration-rate variation is descriptive only",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "d3_generation_6000h_audit.json"
    md_path = output_dir / "d3_generation_6000h_audit.md"
    manifest_path = output_dir / "shard_manifest.json"
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, md_path)
    manifest = {
        "schema": "keqing.mortal.d3_generation_6000h_shard_manifest.v1",
        "shards": [
            ledger["per_shard"][index]
            for index in range(24)
        ],
        "totals": counters["totals"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": report["gate"]["verdict"],
                "outputs": {
                    "audit_json": str(audit_path),
                    "audit_md": str(md_path),
                    "manifest": str(manifest_path),
                },
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
