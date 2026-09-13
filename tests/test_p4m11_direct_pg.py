"""Focused contract tests for the P4-M11 direct-PG continuation.

These cover exactly the increments the P4-M11 plan asks a reviewer to focus on:

* restoring the parent **optimizer** (not just the weights) and validating the
  inherited Adam step;
* cycle bookkeeping that cannot apply the same update twice;
* refusing to step when a collection is incomplete;
* accumulating the whole batch into a single gradient/step;
* a save -> restore round trip;
* the parent file being left untouched;
* the P4-M10 module and its frozen budget keeping their exact behaviour.

They deliberately do NOT try to re-prove training correctness by counting
contract tests: the plan is explicit that a green suite is not evidence that
32 cycles of collected data are scientifically meaningful.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal import p4m10_onpolicy_pg as m10
from training.mortal import p4m11_direct_pg as m11
from training.mortal.p4m9_probe_onpolicy import bits_to_mask

C4 = REPO_ROOT / "artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256/C4.pth"
C4_EVAL = (
    REPO_ROOT
    / "artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256/C4_eval_weights.pth"
)
MORTAL_ROOT = REPO_ROOT / "third_party/Mortal"

pytestmark = pytest.mark.skipif(
    not C4.exists(), reason="P4-M10 C4 checkpoint is not present"
)


@pytest.fixture(scope="module")
def cpu_parent() -> dict:
    """The real C4 restored on CPU, once per module (it is a 130 MB load)."""
    return m11.restore_parent(
        C4,
        device=torch.device("cpu"),
        mortal_root=MORTAL_ROOT,
        expected_sha256=str(m11.P4M11_CONFIG["parent_sha256"]),
        expected_cycles=4,
        expected_step=4.0,
    )


# ---------------------------------------------------------------------------
# parent / optimizer restore
# ---------------------------------------------------------------------------
def test_restore_parent_inherits_adam_step_four(cpu_parent: dict) -> None:
    assert cpu_parent["inherited_cycle"] == 4
    assert cpu_parent["inherited_adam_step"] == 4.0
    assert m11.optimizer_steps(cpu_parent["optimizer"]) == [4.0]
    assert len(cpu_parent["parameters"]) == 409
    group = cpu_parent["optimizer"].param_groups[0]
    assert math.isclose(float(group["lr"]), 1e-5, rel_tol=0.0, abs_tol=0.0)
    assert float(group["weight_decay"]) == 0.0
    assert (int(cpu_parent["version"]), int(cpu_parent["conv_channels"]),
            int(cpu_parent["num_blocks"])) == (4, 192, 40)


def test_restore_matches_the_pinned_parent_sha(cpu_parent: dict) -> None:
    assert cpu_parent["parent_sha256"] == m11.P4M11_CONFIG["parent_sha256"]
    assert cpu_parent["parent_sha256"] == (
        "6f5e5eb7148a1b364d57e8f86be7aa594a26892a7b4a0965347a2d86e0d9f79d"
    )


def test_restore_refuses_wrong_parent_sha() -> None:
    with pytest.raises(m11.P4M11ContractError, match="refusing to continue"):
        m11.restore_parent(
            C4, device=torch.device("cpu"), mortal_root=MORTAL_ROOT,
            expected_sha256="0" * 64,
        )


def test_restore_refuses_a_wrong_inherited_cycle() -> None:
    with pytest.raises(m11.P4M11ContractError, match="records cycle"):
        m11.restore_parent(
            C4, device=torch.device("cpu"), mortal_root=MORTAL_ROOT,
            expected_cycles=7,
        )


def test_restore_refuses_eval_weights_without_optimizer_state() -> None:
    """The plan's forbidden shortcut: eval weights + a fresh Adam."""
    assert C4_EVAL.exists(), "expected the P4-M10 eval-weights export to exist"
    with pytest.raises(m11.P4M11ContractError, match="no optimizer_state"):
        m11.restore_parent(C4_EVAL, device=torch.device("cpu"), mortal_root=MORTAL_ROOT)


def test_assert_adam_step_rejects_a_fresh_optimizer(cpu_parent: dict) -> None:
    fresh = m11.build_optimizer(cpu_parent["parameters"])
    with pytest.raises(m11.P4M11ContractError, match="no step counter"):
        m11.assert_adam_step(fresh, 4.0)


def test_assert_adam_step_rejects_the_wrong_step(cpu_parent: dict) -> None:
    with pytest.raises(m11.P4M11ContractError, match="inherited Adam step is 4"):
        m11.assert_adam_step(cpu_parent["optimizer"], 5.0)


def test_build_optimizer_reuses_the_declared_hyperparameters() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(3))]
    optimizer = m11.build_optimizer(parameters)
    group = optimizer.param_groups[0]
    assert math.isclose(float(group["lr"]), 1e-5, rel_tol=0.0, abs_tol=0.0)
    assert float(group["weight_decay"]) == 0.0
    assert tuple(group["betas"]) == (0.9, 0.999)


# ---------------------------------------------------------------------------
# budget and seed geometry
# ---------------------------------------------------------------------------
def test_budget_allows_exactly_thirty_two_cycles() -> None:
    m11.assert_within_budget(32, 64)
    with pytest.raises(m11.P4M11ContractError, match="budget exceeded"):
        m11.assert_within_budget(33, 64)
    with pytest.raises(m11.P4M11ContractError, match="budget exceeded"):
        m11.assert_within_budget(32, 128)


def test_p4m10_frozen_budget_is_unchanged() -> None:
    """Extending P4-M11 must not have loosened the frozen P4-M10 budget."""
    assert int(m10.P4M10_CONFIG["cycles"]) == 4
    assert int(m10.P4M10_CONFIG["max_hanchans"]) == 1024
    m10.assert_within_budget(4, 64)
    with pytest.raises(m10.P4M10ContractError, match="budget exceeded"):
        m10.assert_within_budget(5, 64)


def test_seed_geometry_matches_the_plan() -> None:
    assert m11.seed_segment_for(1) == (730000, 730063)
    assert m11.seed_segment_for(32) == (731984, 732047)
    # Contiguous and non-overlapping across the whole run.
    segments = [m11.seed_segment_for(cycle) for cycle in range(1, 33)]
    for (_, last), (first_next, _) in pairwise(segments):
        assert last + 1 == first_next
    assert segments[0][0] == 730000
    assert segments[-1][1] == 732047


def test_sampling_seeds_are_distinct_from_the_game_seeds() -> None:
    first = m11.sampling_seed_for(1)
    last = m11.sampling_seed_for(32)
    assert (first, last) == (2026091301, 2026091332)
    game = set(range(730000, 732048))
    assert not (set(range(first, last + 1)) & game)


def test_recipe_guard_passes_and_is_pinned_to_p4m10() -> None:
    m11.assert_recipe_matches_p4m10()
    assert m11.P4M11_CONFIG["seeds_per_cycle"] == m10.P4M10_CONFIG["seeds_per_cycle"]
    assert m11.P4M11_CONFIG["splits_per_seed"] == m10.P4M10_CONFIG["splits_per_seed"]
    assert int(m11.P4M11_CONFIG["update_micro_batch"]) == 512
    assert float(m11.P4M11_CONFIG["gradient_clip"]["value"]) == 1.0
    assert m11.P4M11_CONFIG["loss"] == m10.P4M10_CONFIG["loss"]


# ---------------------------------------------------------------------------
# cycle bookkeeping: an update can never be applied twice
# ---------------------------------------------------------------------------
def test_uncommitted_checkpoint_is_invisible_to_resume(tmp_path: Path) -> None:
    """A checkpoint without its completion marker was never committed."""
    payload = {"hello": "world"}
    m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, 1), payload)
    assert m11.committed_cycles(tmp_path) == {}

    sha = m11._sha256_file(m11.checkpoint_path(tmp_path, 1))
    m11.write_json_atomic(
        m11.commit_marker_path(tmp_path, 1),
        {"completed_cycles": 1, "checkpoint_sha256": sha},
    )
    committed = m11.committed_cycles(tmp_path)
    assert sorted(committed) == [1]

    # A later, uncommitted checkpoint for cycle 2 stays invisible.
    m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, 2), payload)
    assert sorted(m11.committed_cycles(tmp_path)) == [1]


def test_tampered_checkpoint_is_refused(tmp_path: Path) -> None:
    m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, 3), {"a": 1})
    sha = m11._sha256_file(m11.checkpoint_path(tmp_path, 3))
    m11.write_json_atomic(
        m11.commit_marker_path(tmp_path, 3),
        {"completed_cycles": 3, "checkpoint_sha256": sha},
    )
    # Rewrite the checkpoint so it no longer matches its marker.
    m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, 3), {"a": 2})
    with pytest.raises(m11.P4M11ContractError, match="does not match its completion marker"):
        m11.committed_cycles(tmp_path)


def test_resume_reloads_the_last_committed_step_not_the_in_flight_one(
    tmp_path: Path, cpu_parent: dict
) -> None:
    """The core no-double-apply property.

    Cycle 1 commits at Adam step 5.  Then cycle 2's update is performed in
    memory but never committed.  A resume must come back at step 5, and running
    cycle 2 again must land on step 6 -- not 7.
    """
    brain = cpu_parent["brain"]
    dqn = cpu_parent["dqn"]
    optimizer = cpu_parent["optimizer"]
    parent_ref = {
        "version": int(cpu_parent["version"]),
        "conv_channels": int(cpu_parent["conv_channels"]),
        "num_blocks": int(cpu_parent["num_blocks"]),
        "parent_path": cpu_parent["parent_path"],
        "parent_sha256": cpu_parent["parent_sha256"],
    }

    def commit(cycle: int) -> None:
        payload = {
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
            "training_contract": m11.training_contract(cycle=cycle, parent=parent_ref),
            "optimizer_state": optimizer.state_dict(),
            "cycle": int(cycle),
            "rng": m11.rng_state(),
        }
        sha = m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, cycle), payload)
        m11.write_json_atomic(
            m11.commit_marker_path(tmp_path, cycle),
            {"completed_cycles": int(cycle), "checkpoint_sha256": sha},
        )

    def one_step() -> None:
        for parameter in cpu_parent["parameters"]:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert m11.optimizer_steps(optimizer) == [4.0]

    one_step()                      # cycle 1's update
    assert m11.optimizer_steps(optimizer) == [5.0]
    commit(1)

    one_step()                      # cycle 2's update, in flight
    assert m11.optimizer_steps(optimizer) == [6.0]

    committed = m11.committed_cycles(tmp_path)
    assert sorted(committed) == [1]

    resumed = m11.load_committed_checkpoint(
        tmp_path, max(committed), device=torch.device("cpu"), mortal_root=MORTAL_ROOT
    )
    assert m11.optimizer_steps(resumed["optimizer"]) == [5.0]

    # Re-running cycle 2 from the committed state applies exactly one more step.
    for parameter in resumed["parameters"]:
        parameter.grad = torch.ones_like(parameter)
    resumed["optimizer"].step()
    assert m11.optimizer_steps(resumed["optimizer"]) == [6.0]


def test_training_contract_records_the_actual_parent(cpu_parent: dict) -> None:
    parent_ref = {
        "version": int(cpu_parent["version"]),
        "conv_channels": int(cpu_parent["conv_channels"]),
        "num_blocks": int(cpu_parent["num_blocks"]),
        "parent_path": cpu_parent["parent_path"],
        "parent_sha256": cpu_parent["parent_sha256"],
    }
    contract = m11.training_contract(cycle=7, parent=parent_ref)
    assert contract["experiment"] == "P4-M11"
    assert contract["parent"].endswith("C4.pth")
    assert contract["parent_sha256"] == m11.P4M11_CONFIG["parent_sha256"]
    assert contract["cycle"] == 7
    # The hardcoded hard50k parent must NOT be inherited from P4-M10.
    assert "student_formal_25k" not in json.dumps(contract)


def test_export_cycle_weights_is_loadable_and_hashes_stably(
    tmp_path: Path, cpu_parent: dict
) -> None:
    parent_ref = {
        "version": int(cpu_parent["version"]),
        "conv_channels": int(cpu_parent["conv_channels"]),
        "num_blocks": int(cpu_parent["num_blocks"]),
        "parent_path": cpu_parent["parent_path"],
        "parent_sha256": cpu_parent["parent_sha256"],
    }
    path = tmp_path / "collection_weights.pth"
    first = m11.export_cycle_weights(
        cpu_parent["brain"], cpu_parent["dqn"], path, parent=parent_ref, cycle=1
    )
    second = m11.export_cycle_weights(
        cpu_parent["brain"], cpu_parent["dqn"], path, parent=parent_ref, cycle=1
    )
    assert first == second
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert set(state) == {"mortal", "current_dqn", "training_contract"}
    assert "optimizer_state" not in state


# ---------------------------------------------------------------------------
# incomplete collections must not become a step
# ---------------------------------------------------------------------------
def test_incomplete_collection_is_refused() -> None:
    order = [(730000, 0), (730000, 1), (730001, 0)]
    returns = {(730000, 0): 0.0, (730000, 1): 1.0, (730001, 0): 0.5}
    m11.assert_collection_complete(
        cycle=1, hanchan_order=order, returns_by_hanchan=returns, expected_hanchans=3
    )
    with pytest.raises(m11.P4M11ContractError, match="incomplete collection"):
        m11.assert_collection_complete(
            cycle=1, hanchan_order=order, returns_by_hanchan=returns, expected_hanchans=256
        )


def test_collection_missing_a_rank_is_refused() -> None:
    order = [(730000, 0), (730000, 1)]
    with pytest.raises(m11.P4M11ContractError, match="no authoritative rank"):
        m11.assert_collection_complete(
            cycle=1,
            hanchan_order=order,
            returns_by_hanchan={(730000, 0): 0.0},
            expected_hanchans=2,
        )


def test_collection_manifest_gates_reuse() -> None:
    expected = {"weights_sha256": "abc", "hanchans": 256, "sampling_seed": 2026091301}
    good = {**expected, "complete": True}
    assert m11.collection_manifest_problems(good, expected) == []
    assert m11.collection_manifest_problems(None, expected) == ["no collection manifest"]
    assert m11.collection_manifest_problems({**good, "weights_sha256": "zzz"}, expected)
    assert m11.collection_manifest_problems({**good, "complete": False}, expected)


def test_reusable_collection_requires_intact_files(tmp_path: Path) -> None:
    cycle_dir = tmp_path / "cycle1"
    cycle_dir.mkdir()
    records = [{"explore": True, "logprob": -1.0}]
    records_path = cycle_dir / "probe_records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    obs_path = cycle_dir / "obs_fp32.bin"
    obs_path.write_bytes(b"\x00" * 8)
    expected = {
        "weights_sha256": "abc",
        "in_memory_parameters_sha256": "def",
        "seed_start": 730000,
        "seed_stop": 730064,
        "seed_key": 8192,
        "sampling_seed": 2026091301,
        "hanchans": 256,
    }
    manifest = {
        **expected,
        "decision_records": len(records),
        "records_sha256": m11._sha256_file(records_path),
        "obs_bytes": obs_path.stat().st_size,
        "complete": True,
    }

    assert m11.reusable_collection(cycle_dir, expected) is None  # no manifest yet

    m11.write_json_atomic(m11.collection_manifest_path(cycle_dir), manifest)
    assert m11.reusable_collection(cycle_dir, expected) == records

    # A sampling-config change invalidates the collection.
    changed = {**expected, "sampling_seed": 2026091302}
    assert m11.reusable_collection(cycle_dir, changed) is None

    # Truncating the records file invalidates it too.
    records_path.write_text("{}\n", encoding="utf-8")
    assert m11.reusable_collection(cycle_dir, expected) is None

    # So does a short observation blob.
    records_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    m11.write_json_atomic(
        m11.collection_manifest_path(cycle_dir),
        {**manifest, "records_sha256": m11._sha256_file(records_path)},
    )
    obs_path.write_bytes(b"\x00" * 4)
    assert m11.reusable_collection(cycle_dir, expected) is None


# ---------------------------------------------------------------------------
# the accumulated update itself
# ---------------------------------------------------------------------------
ACTION_SPACE = 46


def _tiny_setup(*, n_records: int, n_hanchans: int) -> dict:
    """A 2-feature linear scorer plus records/obs that exercise run_pg_update."""
    torch.manual_seed(20260913)

    class Scorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2, ACTION_SPACE) * 0.05)

        def forward(self, obs, mask, actions):  # type: ignore[no-untyped-def]
            logits = obs @ self.weight
            logits = logits.masked_fill(~mask, -torch.inf)
            return torch.log_softmax(logits, dim=-1).gather(-1, actions).squeeze(-1)

    scorer = Scorer()
    obs_payload = torch.randn(n_records, 2, dtype=torch.float32).numpy().tobytes()
    records = []
    for index in range(n_records):
        mask_row = [False] * ACTION_SPACE
        mask_row[index % ACTION_SPACE] = True
        mask_row[(index + 1) % ACTION_SPACE] = True
        records.append({
            "obs_off": index * 2 * 4,
            "obs_bytes": 2 * 4,
            "obs_shape": [2],
            "mask_bits": int(sum(1 << i for i, on in enumerate(mask_row) if on)),
            "action": index % ACTION_SPACE,
            "seed": 730000 + (index % n_hanchans),
            "seat": index % 4,
            "logprob": 0.0,      # replaced below with the true value
        })
    return {
        "scorer": scorer,
        "obs_payload": obs_payload,
        "records": records,
        "hanchan_order": sorted({(r["seed"], r["seat"]) for r in records}),
        "n_hanchans": n_hanchans,
    }


def _write_obs(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "obs_fp32.bin"
    path.write_bytes(payload)
    return path


def _forward(scorer: torch.nn.Module):
    def forward(obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return scorer(obs, mask, actions)

    return forward


def _prepared_update(tmp_path: Path, *, n_records: int) -> dict:
    """Records carrying their true sampling-time log-probs, plus matching obs."""
    setup = _tiny_setup(n_records=n_records, n_hanchans=n_records)
    obs_path = _write_obs(tmp_path, setup["obs_payload"])
    records = [dict(record) for record in setup["records"]]
    scorer = setup["scorer"]
    obs = torch.stack([
        torch.frombuffer(
            bytearray(setup["obs_payload"][r["obs_off"]:r["obs_off"] + r["obs_bytes"]]),
            dtype=torch.float32,
        ).clone()
        for r in records
    ])
    mask = torch.as_tensor([bits_to_mask(r["mask_bits"]) for r in records])
    actions = torch.as_tensor([r["action"] for r in records]).unsqueeze(-1)
    with torch.no_grad():
        true_logprobs = scorer(obs, mask, actions)
    for record, value in zip(records, true_logprobs, strict=True):
        record["logprob"] = float(value)
    order = sorted({(int(r["seed"]), int(r["seat"])) for r in records})
    return {
        "records": records,
        "obs_path": obs_path,
        "scorer": scorer,
        "parameters": list(scorer.parameters()),
        "hanchan_index": {key: index for index, key in enumerate(order)},
        "returns": torch.tensor([1.0] * len(order), dtype=torch.float32),
        "n_hanchans": len(order),
    }


def _run_prepared(prepared: dict, *, micro_batch: int) -> dict:
    optimizer = torch.optim.Adam(prepared["parameters"], lr=1e-5, weight_decay=0.0)
    return m10.run_pg_update(
        records=prepared["records"],
        obs_path=prepared["obs_path"],
        hanchan_index=prepared["hanchan_index"],
        returns=prepared["returns"],
        n_hanchans=prepared["n_hanchans"],
        forward_logprobs=_forward(prepared["scorer"]),
        parameters=prepared["parameters"],
        optimizer=optimizer,
        micro_batch=micro_batch,
        device=torch.device("cpu"),
        max_clip=1e9,
    )


def test_accumulated_update_takes_exactly_one_step_and_matches_a_single_batch(
    tmp_path: Path,
) -> None:
    """Micro-batching must not change the gradient: the whole batch accumulates."""

    def run(micro_batch: int) -> tuple[torch.Tensor, dict]:
        # _tiny_setup reseeds, so both calls start from identical weights/obs.
        prepared = _prepared_update(tmp_path, n_records=6)
        report = _run_prepared(prepared, micro_batch=micro_batch)
        grads = torch.cat([
            parameter.grad.detach().reshape(-1)
            for parameter in prepared["parameters"]
        ])
        return grads, report

    grads_chunked, report_chunked = run(2)
    grads_single, report_single = run(6)

    assert report_chunked["micro_batches"] == 3
    assert report_single["micro_batches"] == 1
    assert report_chunked["optimizer_steps"] == 1
    assert report_single["optimizer_steps"] == 1
    assert report_chunked["decisions_in_loss"] == 6
    assert torch.allclose(grads_chunked, grads_single, atol=1e-6)


def test_update_refuses_a_non_finite_recorded_logprob(tmp_path: Path) -> None:
    prepared = _prepared_update(tmp_path, n_records=4)
    prepared["records"][0]["logprob"] = float("nan")
    with pytest.raises(m10.P4M10ContractError, match="not finite"):
        _run_prepared(prepared, micro_batch=2)


def test_update_refuses_unrecomputable_logprobs(tmp_path: Path) -> None:
    prepared = _prepared_update(tmp_path, n_records=4)
    prepared["records"][0]["logprob"] = -12345.0  # far outside the 5e-3 tolerance
    with pytest.raises(m10.P4M10ContractError, match="not recomputable"):
        _run_prepared(prepared, micro_batch=2)


# ---------------------------------------------------------------------------
# resource lifecycle
# ---------------------------------------------------------------------------
def test_resource_violations_use_the_declared_thresholds() -> None:
    limits = m11.RESOURCE_LIMITS
    healthy = {
        "cuda_device_free_bytes": int(limits["min_free_dedicated_vram_bytes"]) + 1,
        "system_available_bytes": int(limits["min_available_ram_bytes"]) + 1,
        "commit_headroom_bytes": int(limits["min_commit_headroom_bytes"]) + 1,
        "target_disk_free_bytes": int(limits["min_target_disk_free_bytes"]) + 1,
    }
    assert m11.resource_violations(healthy) == []

    starving = dict(healthy)
    starving["cuda_device_free_bytes"] = int(limits["min_free_dedicated_vram_bytes"]) - 1
    starving["target_disk_free_bytes"] = int(limits["min_target_disk_free_bytes"]) - 1
    problems = m11.resource_violations(starving)
    assert len(problems) == 2
    assert any("VRAM" in problem for problem in problems)
    assert any("disk" in problem for problem in problems)


def test_resource_violations_ignore_unavailable_probes() -> None:
    assert m11.resource_violations({"cuda_device_free_bytes": None}) == []


def test_preflight_requires_the_full_run_disk_reserve() -> None:
    reserve = int(m11.ZERO_DISK_RESERVE_BYTES)
    assert reserve == 284 * 1024**3
    # The 64 GiB runtime floor is not enough to start a full run.
    roomy = {"target_disk_free_bytes": reserve + 1}
    assert m11.preflight_violations(roomy) == []
    tight = {"target_disk_free_bytes": reserve - 1}
    assert m11.preflight_violations(tight) == [
        problem for problem in m11.preflight_violations(tight) if "startup reserve" in problem
    ]
    assert m11.preflight_violations(tight)
    assert m11.preflight_violations(
        {"target_disk_free_bytes": 100 * 1024**3}
    ), "below the reserve must be refused even though it is above the 64 GiB floor"


def test_preflight_also_reports_the_runtime_floors() -> None:
    problems = m11.preflight_violations(
        {
            "target_disk_free_bytes": int(m11.ZERO_DISK_RESERVE_BYTES) + 1,
            "commit_headroom_bytes": 1,
        }
    )
    assert any("commit headroom" in problem for problem in problems)


def test_startup_survey_records_the_environment(tmp_path: Path) -> None:
    survey = m11.startup_survey(target_dir=tmp_path)
    assert "snapshot" in survey
    assert isinstance(survey["existing_compute_processes"], list)
    assert survey["existing_compute_processes"], "pytest itself should be visible"
    assert any("python" in entry["name"].lower() for entry in survey["existing_compute_processes"])


def test_resource_guard_requires_sustained_pressure() -> None:
    guard = m11.ResourceGuard(limit=2)
    assert guard.observe(["low"]) is False
    assert guard.observe(["low"]) is True
    assert guard.observe([]) is False
    assert guard.consecutive == 0
    assert m11.RESOURCE_LIMITS["consecutive_samples"] == 2
    assert float(m11.RESOURCE_LIMITS["sample_interval_seconds"]) == 30.0


def test_resource_snapshot_reports_disk_and_ram(tmp_path: Path) -> None:
    snapshot = m11.resource_snapshot(target_dir=tmp_path, label="unit-test")
    assert snapshot["label"] == "unit-test"
    assert snapshot["target_disk_free_bytes"] is not None
    assert snapshot["target_disk_free_bytes"] > 0


def test_accumulated_active_seconds_sums_the_jsonl(tmp_path: Path) -> None:
    assert m11.accumulated_active_seconds(tmp_path) == 0.0
    for seconds in (10.0, 5.5):
        m11.append_jsonl(tmp_path / "cycles.jsonl", {"wall_seconds": seconds})
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(15.5)
    # Garbage lines must not break the budget accounting.
    with (tmp_path / "cycles.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(15.5)


def test_atomic_writes_leave_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.json"
    m11.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert list(target.parent.glob("*.tmp")) == []

    checkpoint = tmp_path / "nested" / "U1.pth"
    m11.save_checkpoint_atomic(checkpoint, {"b": 2})
    assert list(checkpoint.parent.glob("*.tmp")) == []
    assert torch.load(checkpoint, map_location="cpu", weights_only=False) == {"b": 2}


def test_parent_file_is_never_modified_by_a_restore(cpu_parent: dict) -> None:
    before = m11._sha256_file(C4)
    m11.restore_parent(C4, device=torch.device("cpu"), mortal_root=MORTAL_ROOT)
    assert m11._sha256_file(C4) == before
    assert before == m11.P4M11_CONFIG["parent_sha256"]
