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
import time
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
            "completed_cycles": int(cycle),
            "adam_step": float(4 + int(cycle)),
            "recipe": m11.P4M11_CONFIG,
            "environment": m11._launch_environment(),
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


def test_resource_violations_treat_a_missing_probe_as_a_breach() -> None:
    """With no reading we cannot assert the host is fine, so it is not a pass."""
    problems = m11.resource_violations({}, require_gpu=False)
    assert problems, "an empty snapshot must not read as a healthy host"
    assert any("unavailable" in problem for problem in problems)
    # The GPU metric is only required when a GPU is actually in play.
    assert not any("cuda_device_free_bytes" in problem for problem in problems)
    assert any(
        "cuda_device_free_bytes" in problem
        for problem in m11.resource_violations({}, require_gpu=True)
    )


def test_preflight_requires_the_full_run_disk_reserve() -> None:
    reserve = int(m11.ZERO_DISK_RESERVE_BYTES)
    assert reserve == 284 * 1024**3
    healthy = {
        "target_disk_free_bytes": reserve + 1,
        "system_available_bytes": int(m11.RESOURCE_LIMITS["min_available_ram_bytes"]) + 1,
        "commit_headroom_bytes": int(m11.RESOURCE_LIMITS["min_commit_headroom_bytes"]) + 1,
    }
    assert m11.preflight_violations(healthy, require_gpu=False) == []
    tight = {**healthy, "target_disk_free_bytes": 100 * 1024**3}
    problems = m11.preflight_violations(tight, require_gpu=False)
    assert any("startup reserve" in problem for problem in problems)


def test_preflight_rejects_an_empty_snapshot() -> None:
    """The reviewer's check: `preflight_violations({})` must not be empty."""
    assert m11.preflight_violations({}, require_gpu=False) != []
    assert m11.preflight_violations({}, require_gpu=True) != []


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


def test_accumulated_active_seconds_counts_and_commits_attempt_time(
    tmp_path: Path,
) -> None:
    assert m11.accumulated_active_seconds(tmp_path) == 0.0

    # A previously committed total is carried forward.
    m11.write_json_atomic(m11.active_time_path(tmp_path), {"active_seconds": 10.0})
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(10.0)

    # An attempt that died is charged from its heartbeat even though it never
    # reached cycles.jsonl.
    m11.write_json_atomic(
        m11.heartbeat_path(tmp_path),
        {"cycle": 3, "active_seconds_at_start": 10.0, "started_unix": time.time() - 120.0},
    )
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(130.0, abs=5.0)

    # Starting a new cycle folds the dead attempt's time in permanently.
    base = m11.begin_active_cycle(tmp_path, 4)
    assert base == pytest.approx(130.0, abs=5.0)
    m11.write_json_atomic(m11.active_time_path(tmp_path), {"active_seconds": base})
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(base, abs=5.0)

    # A graceful finish commits the wall time and clears the heartbeat.
    m11.end_active_cycle(tmp_path)
    assert not m11.heartbeat_path(tmp_path).exists()
    assert m11.accumulated_active_seconds(tmp_path) == pytest.approx(base, abs=5.0)


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


# ===========================================================================
# Defect 1: resource protection must run WHILE a cycle is running
# ===========================================================================
def _healthy_snapshot(**overrides: object) -> dict:
    snapshot = {
        "label": "healthy",
        "target_disk_free_bytes": int(m11.ZERO_DISK_RESERVE_BYTES) + 1,
        "system_available_bytes": int(m11.RESOURCE_LIMITS["min_available_ram_bytes"]) * 4,
        "commit_headroom_bytes": int(m11.RESOURCE_LIMITS["min_commit_headroom_bytes"]) * 4,
        "cuda_device_free_bytes": (
            int(m11.RESOURCE_LIMITS["min_free_dedicated_vram_bytes"]) * 4
        ),
    }
    snapshot.update(overrides)
    return snapshot


def _pause_only_snapshot(**overrides: object) -> dict:
    """Below a safe-pause floor but above the hard-stop floor."""
    ram = (
        int(m11.RESOURCE_LIMITS["min_available_ram_bytes"])
        + int(m11.CRITICAL_LIMITS["min_available_ram_bytes"])
    ) // 2
    assert int(m11.CRITICAL_LIMITS["min_available_ram_bytes"]) < ram < int(
        m11.RESOURCE_LIMITS["min_available_ram_bytes"]
    )
    return _healthy_snapshot(label="pause-only", system_available_bytes=ram, **overrides)


def _critical_snapshot(**overrides: object) -> dict:
    return _healthy_snapshot(
        label="critical",
        system_available_bytes=int(m11.CRITICAL_LIMITS["min_available_ram_bytes"]) // 2,
        **overrides,
    )


def test_watchdog_needs_sustained_pressure_before_pausing() -> None:
    watchdog = m11.ResourceWatchdog(
        target_dir=Path("."), guard_limit=2, require_gpu=False,
        sample_hook=lambda: _healthy_snapshot(),
    )
    watchdog.observe(_healthy_snapshot())
    assert watchdog.pause_requested is False
    watchdog.check()  # no-op while healthy

    watchdog.observe(_pause_only_snapshot())
    assert watchdog.pause_requested is False, "one sample must not trip the guard"
    watchdog.observe(_pause_only_snapshot())
    assert watchdog.pause_requested is True
    with pytest.raises(m11.P4M11ResourceStop) as caught:
        watchdog.check()
    assert caught.value.critical is False
    assert "safe-pause" in caught.value.reason

    # A healthy sample resets the run of breaches.
    reset = m11.ResourceWatchdog(target_dir=Path("."), guard_limit=2, require_gpu=False)
    reset.observe(_pause_only_snapshot())
    reset.observe(_healthy_snapshot())
    reset.observe(_pause_only_snapshot())
    assert reset.pause_requested is False


def test_watchdog_hard_stops_on_critical_pressure() -> None:
    exits: list[int] = []
    reasons: list[str] = []
    watchdog = m11.ResourceWatchdog(
        target_dir=Path("."), guard_limit=2, require_gpu=False,
        sample_hook=lambda: _critical_snapshot(),
        exiter=exits.append,
        on_hard_stop=reasons.append,
    )
    watchdog.observe(_critical_snapshot())
    assert exits == [], "a single critical sample must not kill the process"
    watchdog.observe(_critical_snapshot())
    assert watchdog.terminate_requested is True
    assert exits == [m11.PAUSE_EXIT_CODE]
    assert watchdog.hard_stop_calls == 1
    assert reasons and "critical" in reasons[0]
    with pytest.raises(m11.P4M11ResourceStop) as caught:
        watchdog.check()
    assert caught.value.critical is True


def test_watchdog_treats_missing_metrics_as_pressure() -> None:
    """A dead probe cannot be read as a healthy host.

    It escalates to a safe pause, never to a hard stop: an unreadable probe is
    not evidence that the machine is about to fail.
    """

    def blank() -> dict:
        return {}

    watchdog = m11.ResourceWatchdog(
        target_dir=Path("."), guard_limit=2, require_gpu=False, sample_hook=blank,
    )
    watchdog.observe({})
    watchdog.observe({})
    assert watchdog.pause_requested is True
    assert watchdog.terminate_requested is False
    assert m11.critical_violations({}, require_gpu=False) == []
    with pytest.raises(m11.P4M11ResourceStop):
        watchdog.check()


def test_watchdog_samples_in_flight_and_persists_them(tmp_path: Path) -> None:
    """The guard must be evaluated while the cycle runs, not only around it."""
    log = tmp_path / "resource_samples.jsonl"
    ticks: list[int] = []

    def hook() -> dict:
        ticks.append(1)
        return _healthy_snapshot(label="inflight")

    watchdog = m11.ResourceWatchdog(
        target_dir=tmp_path, interval=0.02, guard_limit=2,
        sample_hook=hook, sample_log=log, require_gpu=False,
    )
    with watchdog:
        time.sleep(0.25)
    # The background thread sampled repeatedly, well beyond the two samples that
    # the pre/post-cycle snapshots alone would have produced.
    assert len(ticks) >= 3
    assert len(watchdog.samples) >= 3
    assert log.exists()
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) >= 3
    assert json.loads(lines[0])["label"] == "inflight"
    # stop() must join the thread rather than leaving it behind.
    assert watchdog._thread is None


def test_watchdog_survives_a_failing_probe_as_pressure(tmp_path: Path) -> None:
    def hook() -> dict:
        raise OSError("probe unavailable")

    watchdog = m11.ResourceWatchdog(
        target_dir=tmp_path, interval=0.02, guard_limit=2,
        sample_hook=hook, exiter=lambda code: None, require_gpu=False,
    )
    with watchdog:
        time.sleep(0.2)
    assert watchdog.pause_requested is True
    # A dead probe pauses the run; it never kills it outright.
    assert watchdog.hard_stop_calls == 0


# ===========================================================================
# Defect 2: a resume must verify the identity it saves
# ===========================================================================
def _stepped_optimizer(cpu_parent: dict, total_steps: int):
    """A fresh optimizer advanced to an absolute Adam step.

    Deliberately not derived from the module-scoped optimizer, whose step other
    tests advance; the identity check compares the step number, so the fixture
    must be independent of test ordering.
    """
    parameters = list(cpu_parent["brain"].parameters()) + list(cpu_parent["dqn"].parameters())
    optimizer = m11.build_optimizer(parameters)
    for _ in range(int(total_steps)):
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _write_identity_checkpoint(
    cpu_parent: dict,
    run_dir: Path,
    cycle: int,
    *,
    recipe: object = "default",
    environment: object = "default",
    parent_sha256: str | None = None,
    parent_path: str | None = None,
    recorded_cycle: int | None = None,
    completed_cycles: int | None = None,
    adam_step: float | None = None,
    optimizer: object = None,
) -> Path:
    """Write a self-consistent committed checkpoint at ``cycle``.

    ``recorded_cycle`` forges what the payload *claims*, which is what a resume
    must catch when it disagrees with the file it was asked to load.
    """
    payload_cycle = int(cycle if recorded_cycle is None else recorded_cycle)
    parent_ref = {
        "version": int(cpu_parent["version"]),
        "conv_channels": int(cpu_parent["conv_channels"]),
        "num_blocks": int(cpu_parent["num_blocks"]),
        "parent_path": parent_path or cpu_parent["parent_path"],
        "parent_sha256": parent_sha256 or cpu_parent["parent_sha256"],
    }
    if optimizer is None:
        optimizer = _stepped_optimizer(cpu_parent, 4 + payload_cycle)
    payload = {
        "mortal": cpu_parent["brain"].state_dict(),
        "current_dqn": cpu_parent["dqn"].state_dict(),
        "training_contract": m11.training_contract(cycle=cycle, parent=parent_ref),
        "optimizer_state": optimizer.state_dict(),
        "cycle": payload_cycle,
        "completed_cycles": int(
            payload_cycle if completed_cycles is None else completed_cycles
        ),
        "adam_step": float(
            4 + payload_cycle if adam_step is None else adam_step
        ),
        "recipe": m11.P4M11_CONFIG if recipe == "default" else recipe,
        "environment": (
            m11._launch_environment() if environment == "default" else environment
        ),
        "rng": m11.rng_state(),
    }
    sha = m11.save_checkpoint_atomic(m11.checkpoint_path(run_dir, cycle), payload)
    m11.write_json_atomic(
        m11.commit_marker_path(run_dir, cycle),
        {"completed_cycles": int(cycle), "checkpoint_sha256": sha},
    )
    return m11.checkpoint_path(run_dir, cycle)


def test_load_committed_checkpoint_accepts_a_matching_identity(
    tmp_path: Path, cpu_parent: dict
) -> None:
    _write_identity_checkpoint(cpu_parent, tmp_path, 1)
    restored = m11.load_committed_checkpoint(
        tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
    )
    assert m11.optimizer_steps(restored["optimizer"]) == [5.0]
    assert restored["parent_sha256"] == cpu_parent["parent_sha256"]


def test_load_committed_checkpoint_refuses_a_foreign_recipe(
    tmp_path: Path, cpu_parent: dict
) -> None:
    foreign = dict(m11.P4M11_CONFIG)
    foreign["cycles"] = 12
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, recipe=foreign)
    with pytest.raises(m11.P4M11ContractError, match="different frozen recipe"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_a_foreign_environment(
    tmp_path: Path, cpu_parent: dict
) -> None:
    foreign = dict(m11._launch_environment())
    foreign["native_binaries"] = [
        {"path": "elsewhere/riichi.pyd", "bytes": 1, "sha256": "f" * 64}
    ]
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, environment=foreign)
    with pytest.raises(m11.P4M11ContractError, match="different environment"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_a_missing_environment(
    tmp_path: Path, cpu_parent: dict
) -> None:
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, environment={})
    with pytest.raises(m11.P4M11ContractError, match="different environment"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_a_foreign_parent(
    tmp_path: Path, cpu_parent: dict
) -> None:
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, parent_sha256="a" * 64)
    with pytest.raises(m11.P4M11ContractError, match="lineage parent"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_the_wrong_cycle(
    tmp_path: Path, cpu_parent: dict
) -> None:
    """The reviewer's `cycle=999` case: it must not load as if it were cycle 1."""
    _write_identity_checkpoint(
        cpu_parent, tmp_path, 1, recorded_cycle=999, completed_cycles=999
    )
    with pytest.raises(m11.P4M11ContractError, match="records cycle 999"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_mismatched_completed_cycles(
    tmp_path: Path, cpu_parent: dict
) -> None:
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, completed_cycles=0)
    with pytest.raises(m11.P4M11ContractError, match="records completed_cycles 0"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_load_committed_checkpoint_refuses_a_changed_native_binary(
    tmp_path: Path, cpu_parent: dict
) -> None:
    """Forge the field the fingerprint actually reads: ``native_binaries``.

    Forging a top-level ``native_sha256`` key would prove nothing, because the
    fingerprint only ever reads ``native_binaries``; this is the directory the
    checkpoint really stores, so this is the comparison that must bite.
    """
    environment = dict(m11._launch_environment())
    environment["native_binaries"] = [
        {"path": "elsewhere/riichi.pyd", "bytes": 1, "sha256": "f" * 64}
    ]
    _write_identity_checkpoint(cpu_parent, tmp_path, 1, environment=environment)
    with pytest.raises(m11.P4M11ContractError, match="different environment"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_environment_fingerprint_reads_native_binaries_only() -> None:
    """Guard against a vacuous environment comparison."""
    environment = m11._launch_environment()
    fingerprint = m11.environment_fingerprint(environment)
    assert fingerprint["native_sha256"] == sorted(
        entry["sha256"] for entry in environment["native_binaries"]
    )
    # A forged top-level key is ignored, which is why the test above forges
    # native_binaries and not this one.
    decoy = dict(environment)
    decoy["native_sha256"] = ["0" * 64]
    assert m11.environment_fingerprint(decoy) == fingerprint


def test_load_committed_checkpoint_refuses_a_wrong_adam_step(
    tmp_path: Path, cpu_parent: dict
) -> None:
    # Claims cycle 1 (so step 5 is expected) but carries an un-advanced optimizer.
    _write_identity_checkpoint(
        cpu_parent, tmp_path, 1, optimizer=_stepped_optimizer(cpu_parent, 4)
    )
    with pytest.raises(m11.P4M11ContractError, match="should sit at Adam step"):
        m11.load_committed_checkpoint(
            tmp_path, 1, device=torch.device("cpu"), mortal_root=MORTAL_ROOT
        )


def test_environment_problems_detects_a_changed_native_binary() -> None:
    recorded = m11._launch_environment()
    current = m11.environment_fingerprint(recorded)
    assert m11.environment_problems(recorded, current) == []

    changed = dict(current)
    changed["native_sha256"] = ["0" * 64]
    assert any("native binary set changed" in p for p in m11.environment_problems(recorded, changed))
    assert m11.environment_problems(None, current)


def test_reuse_requires_the_champion_and_the_full_sampling_config() -> None:
    base = {
        "weights_sha256": "w",
        "champion_sha256": "c",
        "sampling": {"boltzmann_temp": 1.0},
        "hanchans": 256,
    }
    manifest = {**base, "complete": True}
    assert m11.collection_manifest_problems(manifest, base) == []
    # A different opponent invalidates the collection even though the weights match.
    other = {**base, "champion_sha256": "z"}
    assert m11.collection_manifest_problems(manifest, other)
    # So does a changed sampling configuration.
    resampled = {**base, "sampling": {"boltzmann_temp": 2.0}}
    assert m11.collection_manifest_problems(manifest, resampled)


# ===========================================================================
# Defect 3: a failed collection must survive, never be overwritten
# ===========================================================================
def test_every_identity_artefact_is_written_atomically(tmp_path: Path) -> None:
    """No file another attempt or a resume trusts may be truncated in place."""
    source = Path(m11.__file__).read_text(encoding="utf-8")
    # The only bare torch.save lives inside the atomic helper itself.
    assert source.count("torch.save(") == 1
    assert "_save_torch_atomic" in source.rsplit("def _save_torch_atomic", 1)[-1]

    target = tmp_path / "nested" / "weights.pth"
    sha = m11._save_torch_atomic(target, {"a": torch.ones(3)})
    assert target.exists()
    assert m11._sha256_file(target) == sha
    assert not (tmp_path / "nested" / "weights.pth.tmp").exists()


def test_next_attempt_dir_never_reuses_a_directory(tmp_path: Path) -> None:
    first = m11.next_attempt_dir(tmp_path, 1)
    assert first.name == "attempt1"
    second = m11.next_attempt_dir(tmp_path, 1)
    assert second.name == "attempt2"
    assert first != second
    assert [p.name for p in m11.cycle_attempt_dirs(tmp_path, 1)] == ["attempt1", "attempt2"]


def test_recollection_preserves_the_previous_attempt(tmp_path: Path) -> None:
    """A doubtful collection must survive a re-collection byte for byte."""
    cycle = 2
    stale = m11.next_attempt_dir(tmp_path, cycle)
    obs = stale / "obs_fp32.bin"
    records = stale / "probe_records.jsonl"
    obs.write_bytes(b"ORIGINAL-OBSERVATIONS" * 8)
    records.write_text('{"explore": true}\n', encoding="utf-8")
    obs_before = m11._sha256_file(obs)
    records_before = m11._sha256_file(records)
    # The manifest does not match (different weights), so it cannot be reused.
    m11.write_json_atomic(
        m11.collection_manifest_path(stale),
        {"weights_sha256": "stale", "complete": True},
    )

    expected = {"weights_sha256": "fresh", "hanchans": 256}
    assert m11.find_reusable_attempt(tmp_path, cycle, expected) is None

    fresh = m11.next_attempt_dir(tmp_path, cycle)
    assert fresh != stale
    assert fresh.name == "attempt2"
    # The old attempt is untouched: the frozen collector was never pointed at it.
    assert m11._sha256_file(obs) == obs_before
    assert m11._sha256_file(records) == records_before


def test_find_reusable_attempt_prefers_the_newest_valid_one(tmp_path: Path) -> None:
    cycle = 1
    expected = {"weights_sha256": "w", "hanchans": 4}
    for payload, name in ((b"old", "attempt1"), (b"new", "attempt2")):
        attempt = tmp_path / f"cycle{cycle}" / name
        attempt.mkdir(parents=True)
        (attempt / "obs_fp32.bin").write_bytes(payload)
        (attempt / "probe_records.jsonl").write_text(
            json.dumps({"explore": True}) + "\n", encoding="utf-8"
        )
        m11.write_json_atomic(
            m11.collection_manifest_path(attempt),
            {
                **expected,
                "complete": True,
                "decision_records": 1,
                "records_sha256": m11._sha256_file(attempt / "probe_records.jsonl"),
                "obs_bytes": (attempt / "obs_fp32.bin").stat().st_size,
            },
        )
    found = m11.find_reusable_attempt(tmp_path, cycle, expected)
    assert found is not None
    attempt, records = found
    assert attempt.name == "attempt2"
    assert records == [{"explore": True}]


def test_committed_cycles_refuses_a_corrupt_marker(tmp_path: Path) -> None:
    m11.save_checkpoint_atomic(m11.checkpoint_path(tmp_path, 1), {"a": 1})
    sha = m11._sha256_file(m11.checkpoint_path(tmp_path, 1))
    m11.write_json_atomic(
        m11.commit_marker_path(tmp_path, 1),
        {"completed_cycles": 1, "checkpoint_sha256": sha},
    )
    assert sorted(m11.committed_cycles(tmp_path)) == [1]

    # Corruption must be reported, not silently skipped so the cycle is redone
    # on top of the evidence it refers to.
    (tmp_path / "U1.done.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(m11.P4M11ContractError, match="unreadable"):
        m11.committed_cycles(tmp_path)

    m11.write_json_atomic(
        m11.commit_marker_path(tmp_path, 1),
        {"checkpoint_sha256": sha},
    )
    with pytest.raises(m11.P4M11ContractError, match="completed_cycles"):
        m11.committed_cycles(tmp_path)


def test_committed_cycles_refuses_a_missing_checkpoint(tmp_path: Path) -> None:
    m11.write_json_atomic(
        m11.commit_marker_path(tmp_path, 1),
        {"completed_cycles": 1, "checkpoint_sha256": "0" * 64},
    )
    with pytest.raises(m11.P4M11ContractError, match="missing"):
        m11.committed_cycles(tmp_path)


# ===========================================================================
# Defect 4: the official entry point cannot bypass the budget
# ===========================================================================
def test_official_cli_rejects_the_budget_bypass_flags(tmp_path: Path) -> None:
    """`--cycles 64 --seeds-per-cycle 32` used to pass the budget check."""
    for extra in (
        ["--cycles", "64"],
        ["--cycles", "1"],
        ["--seeds-per-cycle", "32"],
        ["--cycles", "64", "--seeds-per-cycle", "32"],
    ):
        with pytest.raises(SystemExit):
            m11.parse_args(["--output-dir", str(tmp_path), *extra])

    args = m11.parse_args(["--output-dir", str(tmp_path)])
    assert not hasattr(args, "cycles")
    assert not hasattr(args, "seeds_per_cycle")
    assert args.diagnostic_cycles is None


def test_budget_geometry_is_frozen_to_the_config() -> None:
    assert int(m11.P4M11_CONFIG["cycles"]) == 32
    assert int(m11.P4M11_CONFIG["seeds_per_cycle"]) == 64
    assert int(m11.P4M11_CONFIG["splits_per_seed"]) == 4
    assert int(m11.P4M11_CONFIG["max_hanchans"]) == 8192
    # The honest total is 32*64*4 = 8192; the old exploit produced 16,384.
    assert (
        int(m11.P4M11_CONFIG["cycles"])
        * int(m11.P4M11_CONFIG["seeds_per_cycle"])
        * int(m11.P4M11_CONFIG["splits_per_seed"])
        == int(m11.P4M11_CONFIG["max_hanchans"])
    )
    with pytest.raises(m11.P4M11ContractError, match="budget exceeded"):
        # A diagnostic run is bounded by the same authorised ceiling.
        m11.assert_within_budget(64, int(m11.P4M11_CONFIG["seeds_per_cycle"]))


def test_completion_requires_u32_and_adam_step_36() -> None:
    endpoint = int(m11.P4M11_CONFIG["cycles"])
    final_step = float(m11.P4M11_CONFIG["expected_final_adam_step"])
    assert (endpoint, final_step) == (32, 36.0)

    ok = m11.completion_status(
        final_cycle=32, final_steps=[36.0], parent_unchanged=True, diagnostic=False
    )
    assert ok["complete"] is True
    assert ok["reason"] is None

    # A one-cycle run must never present U01 as the endpoint.
    early = m11.completion_status(
        final_cycle=1, final_steps=[5.0], parent_unchanged=True, diagnostic=False
    )
    assert early["complete"] is False
    assert "U32" in early["reason"]
    assert "U1 " in early["reason"]

    assert not m11.completion_status(
        final_cycle=32, final_steps=[35.0], parent_unchanged=True, diagnostic=False
    )["complete"]
    assert not m11.completion_status(
        final_cycle=32, final_steps=[36.0], parent_unchanged=False, diagnostic=False
    )["complete"]
    # A diagnostic run is never an endpoint, even at U32/step 36.
    diagnostic = m11.completion_status(
        final_cycle=32, final_steps=[36.0], parent_unchanged=True, diagnostic=True
    )
    assert diagnostic["complete"] is False
    assert "diagnostic" in diagnostic["reason"]
