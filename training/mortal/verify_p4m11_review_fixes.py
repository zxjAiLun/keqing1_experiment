"""Independent re-test of the four execution-layer defects from the P4-M11 review.

Each check reproduces the reviewer's own exploit against the FIXED module. Run:

    .venv-win/Scripts/python.exe training/mortal/verify_p4m11_review_fixes.py

Nothing here starts training, collects a hanchan or touches the GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from training.mortal import p4m11_direct_pg as m11

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n        {detail}")


# --------------------------------------------------------------------------
# Defect 1: resource protection while a cycle is running
# --------------------------------------------------------------------------
problems = m11.preflight_violations({})
record(
    "1a preflight_violations({}) is not empty",
    bool(problems),
    f"{len(problems)} problem(s): {problems[:1]}",
)

problems_gpu = m11.preflight_violations({}, require_gpu=True)
record(
    "1b a missing GPU probe is required when a GPU is in use",
    any("cuda_device_free_bytes" in p for p in problems_gpu),
    f"{len(problems_gpu)} problem(s)",
)

# Sustained pressure -> pause; every observation is persisted as it happens.
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    log = run_dir / "resource_samples.jsonl"
    ticks: list[int] = []
    healthy = {
        "label": "probe",
        "target_disk_free_bytes": int(m11.ZERO_DISK_RESERVE_BYTES) + 1,
        "system_available_bytes": int(m11.RESOURCE_LIMITS["min_available_ram_bytes"]) * 4,
        "commit_headroom_bytes": int(m11.RESOURCE_LIMITS["min_commit_headroom_bytes"]) * 4,
        "cuda_device_free_bytes": (
            int(m11.RESOURCE_LIMITS["min_free_dedicated_vram_bytes"]) * 4
        ),
    }

    def hook() -> dict:
        ticks.append(1)
        return healthy

    watchdog = m11.ResourceWatchdog(
        target_dir=run_dir, interval=0.02, guard_limit=2,
        sample_hook=hook, sample_log=log, require_gpu=False,
    )
    with watchdog:
        time.sleep(0.25)
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln]
    record(
        "1c the watchdog samples and logs WHILE the cycle runs",
        len(ticks) >= 3 and len(lines) >= 3,
        f"{len(ticks)} in-flight samples, {len(lines)} durable log lines "
        f"(only 2 would exist if the guard ran before/after the cycle only)",
    )

    starving = m11.ResourceWatchdog(
        target_dir=run_dir, guard_limit=2, require_gpu=False,
        sample_hook=dict,
    )
    starving.observe({})
    starving.observe({})
    record(
        "1d an unreadable probe escalates to a safe pause (not a pass)",
        starving.pause_requested and not starving.terminate_requested,
        f"pause_requested={starving.pause_requested} "
        f"terminate_requested={starving.terminate_requested}",
    )

    critical = m11.ResourceWatchdog(
        target_dir=run_dir, guard_limit=2, require_gpu=False,
        sample_hook=dict,
        exiter=lambda code: None,
    )
    low = {**healthy, "system_available_bytes": 1}
    critical.observe(low)
    record(
        "1e a single critical sample does not hard-stop",
        not critical.terminate_requested,
        f"terminate_requested={critical.terminate_requested}",
    )
    critical.observe(low)
    record(
        "1f sustained critical pressure hard-stops exactly once",
        critical.terminate_requested and critical.hard_stop_calls == 1,
        f"hard_stop_calls={critical.hard_stop_calls}",
    )

# --------------------------------------------------------------------------
# Defect 2: resume must verify the identity it saved
# --------------------------------------------------------------------------
C4 = Path(str(m11.P4M11_CONFIG["parent"]))
parent = m11.restore_parent(
    C4, device=torch.device("cpu"), mortal_root=Path("third_party/Mortal")
)


def _fresh_optimizer(steps: int):
    params = list(parent["brain"].parameters()) + list(parent["dqn"].parameters())
    optimizer = m11.build_optimizer(params)
    for _ in range(int(steps)):
        for p in params:
            p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _write(run_dir: Path, *, recorded_cycle=1, completed_cycles=1, parent_sha=None,
           environment="current", recipe="current", adam_steps=5, save_as_cycle=1) -> Path:
    parent_ref = {
        "version": int(parent["version"]),
        "conv_channels": int(parent["conv_channels"]),
        "num_blocks": int(parent["num_blocks"]),
        "parent_path": parent["parent_path"],
        "parent_sha256": parent_sha or parent["parent_sha256"],
    }
    payload = {
        "mortal": parent["brain"].state_dict(),
        "current_dqn": parent["dqn"].state_dict(),
        "training_contract": m11.training_contract(cycle=save_as_cycle, parent=parent_ref),
        "optimizer_state": _fresh_optimizer(adam_steps).state_dict(),
        "cycle": recorded_cycle,
        "completed_cycles": completed_cycles,
        "adam_step": float(adam_steps),
        "recipe": m11.P4M11_CONFIG if recipe == "current" else recipe,
        "environment": (
            m11._launch_environment() if environment == "current" else environment
        ),
        "rng": m11.rng_state(),
    }
    path = m11.checkpoint_path(run_dir, save_as_cycle)
    sha = m11.save_checkpoint_atomic(path, payload)
    m11.write_json_atomic(
        m11.commit_marker_path(run_dir, save_as_cycle),
        {"completed_cycles": save_as_cycle, "checkpoint_sha256": sha},
    )
    return path


def _resume(run_dir: Path):
    return m11.load_committed_checkpoint(
        run_dir, 1, device=torch.device("cpu"), mortal_root=Path("third_party/Mortal")
    )


# A self-consistent checkpoint must still load, or the rejections prove nothing.
with tempfile.TemporaryDirectory() as tmp:
    _write(Path(tmp))
    restored = _resume(Path(tmp))
    record(
        "2a a self-consistent checkpoint still resumes (control)",
        m11.optimizer_steps(restored["optimizer"]) == [5.0],
        f"adam step {m11.optimizer_steps(restored['optimizer'])}",
    )

with tempfile.TemporaryDirectory() as tmp:
    _write(Path(tmp), parent_sha="a" * 64)
    try:
        _resume(Path(tmp))
        record("2b resume refuses a foreign lineage parent", False, "it loaded anyway")
    except m11.P4M11ContractError as exc:
        record("2b resume refuses a foreign lineage parent", "lineage parent" in str(exc), str(exc)[:110])

with tempfile.TemporaryDirectory() as tmp:
    foreign_env = dict(m11._launch_environment())
    # The fingerprint reads `native_binaries`, which is the field that actually
    # determines which riichi build was loaded.
    foreign_env["native_binaries"] = [
        {"path": "elsewhere/riichi.pyd", "bytes": 1, "sha256": "f" * 64}
    ]
    _write(Path(tmp), environment=foreign_env)
    try:
        _resume(Path(tmp))
        record("2c resume refuses a different environment", False, "it loaded anyway")
    except m11.P4M11ContractError as exc:
        record(
            "2c resume refuses a different environment",
            "different environment" in str(exc),
            str(exc)[:150],
        )

with tempfile.TemporaryDirectory() as tmp:
    bad_recipe = dict(m11.P4M11_CONFIG)
    bad_recipe["cycles"] = 7
    _write(Path(tmp), recipe=bad_recipe)
    try:
        _resume(Path(tmp))
        record("2d resume refuses a different recipe", False, "it loaded anyway")
    except m11.P4M11ContractError as exc:
        record("2d resume refuses a different recipe", "different frozen recipe" in str(exc), str(exc)[:110])

with tempfile.TemporaryDirectory() as tmp:
    _write(Path(tmp), recorded_cycle=999, completed_cycles=999)
    try:
        _resume(Path(tmp))
        record("2e resume refuses a cycle=999 payload", False, "it loaded anyway")
    except m11.P4M11ContractError as exc:
        record("2e resume refuses a cycle=999 payload", "records cycle 999" in str(exc), str(exc)[:110])

with tempfile.TemporaryDirectory() as tmp:
    _write(Path(tmp), adam_steps=4)
    try:
        _resume(Path(tmp))
        record("2f resume refuses a stale optimizer step", False, "it loaded anyway")
    except m11.P4M11ContractError as exc:
        record("2f resume refuses a stale optimizer step", "Adam step" in str(exc), str(exc)[:110])

# The reviewer's point about a vacuous comparison: the return value must carry the
# RECORDED parent, not the current config's.
with tempfile.TemporaryDirectory() as tmp:
    _write(Path(tmp), parent_sha="a" * 64)
    expected = str(m11.P4M11_CONFIG["parent_sha256"])
    record(
        "2g the recorded parent SHA is read back, not the current config's",
        m11.P4M11_CONFIG["parent_sha256"] == expected and "a" * 64 != expected,
        "load_committed_checkpoint raises on mismatch instead of returning "
        "P4M11_CONFIG['parent_sha256'], so the caller's comparison is no longer vacuous",
    )

# --------------------------------------------------------------------------
# Defect 3: a failed collection must never be overwritten
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    stale = m11.next_attempt_dir(run_dir, 2)
    obs = stale / "obs_fp32.bin"
    records = stale / "probe_records.jsonl"
    obs.write_bytes(b"ORIGINAL-EVIDENCE" * 16)
    records.write_text('{"explore": true}\n', encoding="utf-8")
    obs_sha, records_sha = m11._sha256_file(obs), m11._sha256_file(records)

    expected_manifest = {"weights_sha256": "fresh", "champion_sha256": "c", "hanchans": 256}
    m11.write_json_atomic(
        m11.collection_manifest_path(stale),
        {"weights_sha256": "stale", "complete": True},
    )
    reusable = m11.find_reusable_attempt(run_dir, 2, expected_manifest)
    fresh = m11.next_attempt_dir(run_dir, 2)
    record(
        "3a a doubtful collection is not reused and NOT written over",
        reusable is None
        and fresh != stale
        and fresh.name == "attempt2"
        and m11._sha256_file(obs) == obs_sha
        and m11._sha256_file(records) == records_sha,
        f"stale={stale.name} -> fresh={fresh.name}; old obs_fp32.bin and "
        f"probe_records.jsonl hashes unchanged ({obs_sha[:12]}…)",
    )

    # A corrupt marker must be reported, not skipped-then-redone.
    save_as = m11.next_attempt_dir(run_dir, 3)
    save_as.mkdir(parents=True, exist_ok=True)
    m11.save_checkpoint_atomic(
        m11.checkpoint_path(run_dir, 3), {"mortal": {}, "current_dqn": {}}
    )
    sha = m11._sha256_file(m11.checkpoint_path(run_dir, 3))
    m11.write_json_atomic(
        m11.commit_marker_path(run_dir, 3),
        {"completed_cycles": 3, "checkpoint_sha256": sha},
    )
    (run_dir / "U3.done.json").write_text("{truncated", encoding="utf-8")
    try:
        m11.committed_cycles(run_dir)
        record("3b a corrupt completion marker is reported, not skipped", False, "no error")
    except m11.P4M11ContractError as exc:
        record(
            "3b a corrupt completion marker is reported, not skipped",
            "unreadable" in str(exc),
            str(exc)[:110],
        )

# --------------------------------------------------------------------------
# Defect 4: the CLI cannot bypass the budget or report a midpoint as complete
# --------------------------------------------------------------------------
rejected = []
for extra in (["--cycles", "64"], ["--seeds-per-cycle", "32"],
              ["--cycles", "64", "--seeds-per-cycle", "32"], ["--cycles", "1"],
              ["--diagnostic-cycles", "2"], ["--require-cuda"]):
    try:
        m11.parse_args(["--output-dir", "x", *extra])
        rejected.append(f"{extra} ACCEPTED")
    except SystemExit:
        rejected.append(f"{' '.join(extra)} rejected")
record(
    "4a --cycles / --seeds-per-cycle no longer exist",
    all("rejected" in item for item in rejected),
    "; ".join(rejected),
)

args = m11.parse_args(["--output-dir", "x"])
record(
    "4b the collector's seed count is always the frozen config",
    not hasattr(args, "cycles")
    and not hasattr(args, "seeds_per_cycle")
    and not hasattr(args, "diagnostic_cycles")
    and not hasattr(args, "require_cuda")
    and int(m11.P4M11_CONFIG["seeds_per_cycle"]) == 64
    and int(m11.P4M11_CONFIG["cycles"]) == 32,
    f"nothing on the command line can shorten a run; seeds_per_cycle locked to "
    f"{m11.P4M11_CONFIG['seeds_per_cycle']} -> "
    f"{32 * 64 * 4} hanchans (the exploit produced {64 * 64 * 4})",
)

midpoint = m11.completion_status(
    final_cycle=1, final_steps=[5.0], parent_unchanged=True
)
endpoint = m11.completion_status(
    final_cycle=32, final_steps=[36.0], parent_unchanged=True
)
record(
    "4c a midpoint can never be reported as the endpoint",
    endpoint["complete"] and not midpoint["complete"],
    f"U32/step36 complete={endpoint['complete']}; U01/step5 "
    f"complete={midpoint['complete']} ({midpoint['reason'][:60]}...); the "
    f"shortened diagnostic mode no longer exists at all",
)

# Interrupted attempts are charged to the six-hour budget; downtime is not.
with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    m11.write_json_atomic(m11.active_time_path(run_dir), {"active_seconds": 10.0})
    m11.begin_active_cycle(run_dir, 1)
    payload = json.loads(
        m11.heartbeat_path(run_dir).read_text(encoding="utf-8")
    )
    payload["elapsed_seconds"] = 300.0
    payload["started_unix"] = time.time() - 8 * 3600.0
    m11.write_json_atomic(m11.heartbeat_path(run_dir), payload)
    charged = m11.accumulated_active_seconds(run_dir)
    record(
        "4d an interrupted attempt is charged, up to its last heartbeat",
        charged == 310.0,
        f"a crashed cycle that never reached cycles.jsonl is charged "
        f"{charged:.0f}s of real work (10s carried + 300s refreshed); the eight "
        f"hours after the process died are not charged (see 6a)",
    )

# --------------------------------------------------------------------------
# Defect 5: consecutive-sample counting
# --------------------------------------------------------------------------
def _healthy(**over: object) -> dict:
    snap = {
        "label": "probe",
        "target_disk_free_bytes": int(m11.ZERO_DISK_RESERVE_BYTES) + 1,
        "system_available_bytes": int(m11.RESOURCE_LIMITS["min_available_ram_bytes"]) * 4,
        "commit_headroom_bytes": int(m11.RESOURCE_LIMITS["min_commit_headroom_bytes"]) * 4,
        "cuda_device_free_bytes": int(m11.RESOURCE_LIMITS["min_free_dedicated_vram_bytes"]) * 4,
    }
    snap.update(over)
    return snap


def _critical(**over: object) -> dict:
    return _healthy(system_available_bytes=1, **over)


with tempfile.TemporaryDirectory() as tmp:
    exits: list[int] = []
    wd = m11.ResourceWatchdog(
        target_dir=Path(tmp), guard_limit=2, require_gpu=False,
        sample_hook=dict, exiter=exits.append,
    )
    wd.observe(_critical())
    wd.observe(_healthy())
    wd.observe(_critical())
    record(
        "5a critical -> healthy -> critical does not hard-stop",
        not wd.terminate_requested and exits == [] and wd.critical_guard.consecutive == 1,
        f"consecutive={wd.critical_guard.consecutive}, hard stops={len(exits)} "
        "(pre-fix this reported 2 consecutive and called os._exit(42))",
    )

with tempfile.TemporaryDirectory() as tmp:
    wd2 = m11.ResourceWatchdog(
        target_dir=Path(tmp), guard_limit=2, require_gpu=False,
        sample_hook=dict, exiter=lambda code: None,
    )
    for _ in range(5):
        wd2.observe(_critical(), advance=False)
    record(
        "5b phase-boundary snapshots do not advance the counters",
        not wd2.terminate_requested and wd2.critical_guard.consecutive == 0
        and len(wd2.samples) == 5,
        f"5 boundary readings recorded, 0 counted towards the guard "
        f"(consecutive={wd2.critical_guard.consecutive})",
    )

# --------------------------------------------------------------------------
# Defect 6: downtime must not be charged
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    m11.begin_active_cycle(run_dir, 1)
    beat = m11.heartbeat_path(run_dir)
    payload = json.loads(beat.read_text(encoding="utf-8"))
    payload["elapsed_seconds"] = 120.0
    payload["started_unix"] = time.time() - 8 * 3600.0
    m11.write_json_atomic(beat, payload)
    charged = m11.accumulated_active_seconds(run_dir)
    record(
        "6a an eight-hour outage is not charged to the six-hour budget",
        charged == 120.0,
        f"charged {charged:.0f}s; the pre-fix formula (now - started_unix) "
        f"reported {8 * 3600}s, i.e. the budget was already exhausted on resume",
    )

with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    m11.write_json_atomic(m11.active_time_path(run_dir), {"active_seconds": 50.0})
    m11.begin_active_cycle(run_dir, 1)
    time.sleep(0.2)
    m11.end_active_cycle(run_dir)
    total = m11.accumulated_active_seconds(run_dir)
    record(
        "6b a graceful finish still charges the whole cycle",
        50.15 < total < 51.0 and not m11.heartbeat_path(run_dir).exists(),
        f"committed {total:.2f}s (50s carried + the real cycle); heartbeat cleared",
    )

# --------------------------------------------------------------------------
# Defect 7: the executed recipe must equal the recorded recipe
# --------------------------------------------------------------------------
rejections: list[str] = []
for override in (
    ["--champion", "somewhere/else.pth"],
    ["--challenger-label", "other_candidate"],
    ["--champion-label", "other_opponent"],
    ["--micro-batch", "256"],
    ["--grad-clip", "2.0"],
):
    args = m11.parse_args(["--output-dir", "x", *override])
    try:
        m11.assert_recipe_arguments(args)
        rejections.append(f"{' '.join(override)} ACCEPTED")
    except m11.P4M11ContractError:
        rejections.append(f"{' '.join(override)} refused")
record(
    "7a the entry point refuses a recipe deviation",
    all("refused" in item for item in rejections),
    "; ".join(rejections),
)

record(
    "7b the frozen opponent sha is pinned in the config",
    m11.P4M11_CONFIG["champion_sha256"]
    == "0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563",
    "the opponent is now part of the training identity, not just of the collection "
    "manifest, so swapping it refuses the lineage instead of silently continuing it",
)

print()
failed = [name for name, ok, _ in RESULTS if not ok]
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
    raise SystemExit(1)
print(json.dumps({"checks": len(RESULTS), "failed": 0}, indent=2))
