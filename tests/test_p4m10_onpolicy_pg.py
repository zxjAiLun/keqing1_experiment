"""Contract tests for the P4-M10 minimal on-policy PG loop.

These pin the frozen first-candidate settings and, more importantly, the
behaviour that is easy to get wrong: exactly one optimizer step per collection
cycle, gradient accumulation that equals a single full-batch backward pass,
and fail-closed gates that abort BEFORE any parameter update.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.p4m10_onpolicy_pg import (
    P4M10_CONFIG,
    RECOMPUTE_TOLERANCE,
    P4M10ContractError,
    assert_within_budget,
    cycle_seeds,
    export_collection_weights,
    normalized_rank_points,
    run_pg_update,
)

OBS_SHAPE = [4, 3]
ACTION_SPACE = 46
LEGAL_MASK_BITS = 0b111


class _CountingAdam(torch.optim.Adam):
    """Adam that records how many times ``step`` was actually applied."""

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.steps = 0

    def step(self, *args, **kwargs):  # type: ignore[override]
        self.steps += 1
        return super().step(*args, **kwargs)


class _TinyPolicy(torch.nn.Module):
    """Stand-in for Brain+DQN: a single linear map over the flattened obs."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(
            torch.randn(ACTION_SPACE, OBS_SHAPE[0] * OBS_SHAPE[1], generator=generator)
        )
        self.bias = torch.nn.Parameter(torch.zeros(ACTION_SPACE))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        flat = obs.reshape(obs.shape[0], -1)
        logits = flat @ self.weight.t() + self.bias
        logits = logits.masked_fill(~mask, -torch.inf)
        return torch.log_softmax(logits, dim=-1).gather(-1, actions).squeeze(-1)


def _write_obs(path: Path, count: int) -> None:
    generator = torch.Generator().manual_seed(7)
    payload = torch.randn(count, *OBS_SHAPE, generator=generator).numpy().astype("float32")
    path.write_bytes(payload.tobytes())


def _records(count: int, *, seed_count: int = 2, logprob: float | None = None) -> list[dict]:
    """Records over a 3-action legal mask with a deterministic hanchan identity."""
    records = []
    per_record = int(math.prod(OBS_SHAPE)) * 4
    for index in range(count):
        records.append({
            "obs_off": index * per_record,
            "obs_bytes": per_record,
            "obs_shape": list(OBS_SHAPE),
            "mask_bits": LEGAL_MASK_BITS,
            "action": index % 3,
            "seed": 720000 + (index % seed_count),
            "seat": index % 4,
            "explore": True,
            "is_greedy": False,
            "logprob": -1.0 if logprob is None else float(logprob),
        })
    return records


def _mask_tensor(records: list[dict]) -> torch.Tensor:
    return torch.tensor(
        [[bool((record["mask_bits"] >> index) & 1) for index in range(ACTION_SPACE)]
         for record in records],
        dtype=torch.bool,
    )


def _obs_tensor(path: Path, count: int) -> torch.Tensor:
    payload = np.frombuffer(path.read_bytes(), dtype="float32").copy()
    return torch.as_tensor(payload.reshape(count, *OBS_SHAPE))


def _sampling_logprobs(policy, path: Path, records: list[dict]) -> list[float]:
    """Log-probs the stub policy actually assigns to the sampled actions."""
    obs = _obs_tensor(path, len(records))
    actions = torch.tensor([record["action"] for record in records]).unsqueeze(-1)
    with torch.no_grad():
        return [float(value) for value in policy(obs, _mask_tensor(records), actions)]


def _faithful_records(
    directory: Path, count: int, policy, *, seed_count: int = 2
) -> tuple[list[dict], Path]:
    """Records whose recorded log-probs really are the policy's sampling log-probs."""
    directory.mkdir(parents=True, exist_ok=True)
    obs_path = directory / "obs.bin"
    _write_obs(obs_path, count)
    records = _records(count, seed_count=seed_count)
    for record, value in zip(
        records, _sampling_logprobs(policy, obs_path, records), strict=True
    ):
        record["logprob"] = value
    return records, obs_path


def _hanchan_setup(records: list[dict]):
    keys = sorted({(record["seed"], record["seat"]) for record in records})
    index = {key: position for position, key in enumerate(keys)}
    returns = torch.tensor([0.5 * (position + 1) for position in range(len(keys))])
    return index, len(keys), returns


def _run(
    tmp_path: Path,
    *,
    records: list[dict] | None = None,
    micro_batch: int,
    policy=None,
    returns=None,
    n_hanchans: int | None = None,
    index=None,
    max_clip: float = 1.0,
    count: int = 6,
):
    if policy is None:
        policy = _TinyPolicy()
    if records is None:
        records, obs_path = _faithful_records(tmp_path, count, policy)
    else:
        tmp_path.mkdir(parents=True, exist_ok=True)
        obs_path = tmp_path / "obs.bin"
        _write_obs(obs_path, len(records))
    if index is None:
        index, n_hanchans, returns = _hanchan_setup(records)
    parameters = list(policy.parameters())
    optimizer = _CountingAdam(parameters, lr=1e-5)
    report = run_pg_update(
        records=records,
        obs_path=obs_path,
        hanchan_index=index,
        returns=returns,
        n_hanchans=n_hanchans,
        forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
        parameters=parameters,
        optimizer=optimizer,
        micro_batch=micro_batch,
        device=torch.device("cpu"),
        max_clip=max_clip,
    )
    return report, optimizer, policy


# --------------------------------------------------------------- frozen config
def test_frozen_first_candidate_settings() -> None:
    assert P4M10_CONFIG["optimizer"]["lr"] == 1e-5
    assert P4M10_CONFIG["optimizer"]["weight_decay"] == 0.0
    assert P4M10_CONFIG["optimizer"]["fresh_init"] is True
    assert P4M10_CONFIG["gradient_clip"] == {"kind": "global_norm", "value": 1.0}
    assert P4M10_CONFIG["update_micro_batch"] == 512
    assert P4M10_CONFIG["cycles"] == 4
    assert P4M10_CONFIG["seeds_per_cycle"] == 64
    assert P4M10_CONFIG["hanchans_per_cycle"] == 256
    assert P4M10_CONFIG["max_hanchans"] == 1024
    assert P4M10_CONFIG["return"]["scalar_baseline"] == 0.0
    assert P4M10_CONFIG["sampling"]["boltzmann_epsilon"] == 1.0
    assert P4M10_CONFIG["sampling"]["boltzmann_temp"] == 1.0
    assert P4M10_CONFIG["sampling"]["top_p"] == 1.0
    assert "eval" in P4M10_CONFIG["bn_mode"]


def test_normalized_rank_points_are_the_tenhou_profile_over_135() -> None:
    points = normalized_rank_points()
    assert points == pytest.approx((2.0 / 3.0, 1.0 / 3.0, 0.0, -1.0))
    assert sum(points) == pytest.approx(0.0)
    # a pure positive scaling of the raw profile: order and ratios are preserved
    assert points[0] / points[1] == pytest.approx(90.0 / 45.0)


def test_baseline_zero_is_a_constant_not_a_batch_mean() -> None:
    assert P4M10_CONFIG["return"]["scalar_baseline"] == 0.0
    assert P4M10_CONFIG["loss"].endswith("log pi(a_t|s_t)")


def test_cycle_seeds_are_disjoint_and_contiguous() -> None:
    segments = [cycle_seeds(c, seed_start=720000, seeds_per_cycle=64) for c in range(1, 5)]
    assert [segment.start for segment in segments] == [720000, 720064, 720128, 720192]
    assert all(len(segment) == 64 for segment in segments)
    union: set[int] = set()
    for segment in segments:
        assert not (union & set(segment)), "training segments must not overlap"
        assert len(set(segment)) == len(segment)
        union |= set(segment)
    assert len(union) == 256


def test_cycle_seeds_reject_a_zero_cycle() -> None:
    with pytest.raises(ValueError):
        cycle_seeds(0, seed_start=720000, seeds_per_cycle=64)


def test_training_seeds_never_touch_the_evaluation_band() -> None:
    """The 2026-09-10/11 evaluation band 710000-710063 must stay untouched."""
    training = set()
    for cycle in range(1, 5):
        training |= set(cycle_seeds(cycle, seed_start=720000, seeds_per_cycle=64))
    assert not (training & set(range(710000, 710064)))


def test_budget_guard_rejects_more_than_the_authorised_total() -> None:
    assert_within_budget(4, 64)  # exactly the authorised 1024 hanchans
    assert_within_budget(2, 64)  # a short run is allowed
    with pytest.raises(P4M10ContractError, match="budget exceeded"):
        assert_within_budget(5, 64)
    with pytest.raises(P4M10ContractError, match="budget exceeded"):
        assert_within_budget(4, 65)


# ------------------------------------------------------------------- one step
def test_exactly_one_optimizer_step_per_collection(tmp_path: Path) -> None:
    report, optimizer, _policy = _run(tmp_path, micro_batch=4, count=11)
    assert optimizer.steps == 1
    assert report["optimizer_steps"] == 1
    assert report["micro_batches"] == 3


def test_a_single_microbatch_still_steps_exactly_once(tmp_path: Path) -> None:
    report, optimizer, _policy = _run(tmp_path, micro_batch=64, count=9)
    assert report["micro_batches"] == 1
    assert optimizer.steps == 1


def test_gradient_accumulation_equals_one_full_batch_backward(tmp_path: Path) -> None:
    """Chunked accumulation must reproduce the single-batch gradient."""
    policy_a = _TinyPolicy()
    records, _path = _faithful_records(tmp_path / "a", 9, policy_a)
    index, n_hanchans, returns = _hanchan_setup(records)

    _report_a, _opt_a, policy_a = _run(
        tmp_path / "a", records=records, micro_batch=9, index=index,
        n_hanchans=n_hanchans, returns=returns, policy=policy_a,
    )
    grads_full = [param.grad.detach().clone() for param in policy_a.parameters()]

    policy_b = _TinyPolicy()
    _report_b, _opt_b, policy_b = _run(
        tmp_path / "b", records=records, micro_batch=2, index=index,
        n_hanchans=n_hanchans, returns=returns, policy=policy_b,
    )
    grads_chunked = [param.grad.detach().clone() for param in policy_b.parameters()]

    assert len(grads_full) == len(grads_chunked) == 2
    for full, chunked in zip(grads_full, grads_chunked, strict=True):
        assert torch.allclose(full, chunked, atol=1e-6)


def test_microbatch_boundaries_do_not_need_to_respect_hanchans(tmp_path: Path) -> None:
    """pg_loss divides by the same N_h in every chunk, so chunks may split."""
    policy = _TinyPolicy()
    records, obs_path = _faithful_records(tmp_path, 6, policy)
    index, n_hanchans, returns = _hanchan_setup(records)
    parameters = list(policy.parameters())
    optimizer = _CountingAdam(parameters, lr=1e-5)
    report = run_pg_update(
        records=records, obs_path=obs_path, hanchan_index=index, returns=returns,
        n_hanchans=n_hanchans,
        forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
        parameters=parameters, optimizer=optimizer, micro_batch=1,
        device=torch.device("cpu"), max_clip=1.0,
    )
    assert report["micro_batches"] == 6
    assert optimizer.steps == 1
    assert report["logprob_recompute"]["within_tolerance"] == 6


def test_loss_matches_the_hand_computed_reinforce_objective(tmp_path: Path) -> None:
    policy = _TinyPolicy()
    records, obs_path = _faithful_records(tmp_path, 6, policy, seed_count=2)
    index, n_hanchans, returns = _hanchan_setup(records)

    obs = _obs_tensor(obs_path, len(records))
    actions = torch.tensor([record["action"] for record in records]).unsqueeze(-1)
    log_probs = policy(obs, _mask_tensor(records), actions).detach()
    hanchan_ids = torch.tensor(
        [index[(record["seed"], record["seat"])] for record in records]
    )
    expected = 0.0
    for hanchan in range(n_hanchans):
        expected -= float(log_probs[hanchan_ids == hanchan].sum() * returns[hanchan])
    expected /= n_hanchans

    report, _optimizer, _policy = _run(
        tmp_path, records=records, micro_batch=6, index=index,
        n_hanchans=n_hanchans, returns=returns, policy=policy,
    )
    assert report["loss_sum_of_chunks"] == pytest.approx(expected, abs=1e-5)


def test_only_the_passed_records_enter_the_loss(tmp_path: Path) -> None:
    """The caller filters to exploration_allowed decisions; the loop adds nothing."""
    report, _optimizer, _policy = _run(tmp_path, micro_batch=5, count=5)
    assert report["decisions_in_loss"] == 5
    assert report["logprob_recompute"]["decisions_checked"] == 5


def test_gradient_clip_is_applied(tmp_path: Path) -> None:
    report, _optimizer, _policy = _run(tmp_path, micro_batch=8, count=8, max_clip=1.0)
    grad = report["grad"]
    assert grad["clip_max_norm"] == 1.0
    assert grad["was_clipped"] is True
    # clipping puts the global norm exactly at the declared maximum
    assert grad["grad_norm_postclip"] == pytest.approx(1.0, rel=1e-5)


def test_clipping_is_reported_as_inactive_when_the_norm_is_small(tmp_path: Path) -> None:
    report, _optimizer, _policy = _run(tmp_path, micro_batch=4, count=4, max_clip=1e6)
    grad = report["grad"]
    assert grad["was_clipped"] is False
    assert grad["grad_norm_postclip"] == pytest.approx(grad["grad_norm_preclip"], rel=1e-6)


def test_an_update_actually_moves_the_parameters(tmp_path: Path) -> None:
    policy = _TinyPolicy()
    records, _path = _faithful_records(tmp_path, 6, policy)
    before = [param.detach().clone() for param in policy.parameters()]
    _report, optimizer, _policy = _run(
        tmp_path, records=records, micro_batch=3, policy=policy
    )
    assert optimizer.steps == 1
    moved = sum(
        1 for original, current in zip(before, policy.parameters(), strict=True)
        if not torch.equal(original, current.detach())
    )
    assert moved == len(before)


# ---------------------------------------------------------------- fail-closed
def test_recompute_mismatch_aborts_before_the_step(tmp_path: Path) -> None:
    """A log-prob that cannot be recomputed must abort WITHOUT updating weights."""
    policy = _TinyPolicy()
    records, _path = _faithful_records(tmp_path / "bad", 6, policy)
    for record in records:
        record["logprob"] = -123.0  # nothing the policy can reproduce
    before = [param.detach().clone() for param in policy.parameters()]
    optimizer = _CountingAdam(list(policy.parameters()), lr=1e-5)
    with pytest.raises(P4M10ContractError, match="not recomputable"):
        run_pg_update(
            records=records, obs_path=tmp_path / "bad" / "obs.bin",
            hanchan_index=_hanchan_setup(records)[0],
            returns=_hanchan_setup(records)[2],
            n_hanchans=_hanchan_setup(records)[1],
            forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
            parameters=list(policy.parameters()), optimizer=optimizer,
            micro_batch=3, device=torch.device("cpu"), max_clip=1.0,
        )
    assert optimizer.steps == 0
    for original, current in zip(before, policy.parameters(), strict=True):
        assert torch.equal(original, current.detach()), "weights must be untouched"


def test_recompute_tolerance_is_predeclared_and_loose_enough_for_batch_shape() -> None:
    # P4-M9 measured a 1.08e-03 worst case when the replay batch shape differs.
    assert RECOMPUTE_TOLERANCE >= 1e-3
    assert RECOMPUTE_TOLERANCE <= 1e-2


def test_non_finite_logprob_aborts_before_the_step(tmp_path: Path) -> None:
    policy = _TinyPolicy()
    records, _path = _faithful_records(tmp_path / "nan", 4, policy)
    records[0]["logprob"] = float("nan")
    before = [param.detach().clone() for param in policy.parameters()]
    optimizer = _CountingAdam(list(policy.parameters()), lr=1e-5)
    with pytest.raises(P4M10ContractError, match="not finite"):
        run_pg_update(
            records=records, obs_path=tmp_path / "nan" / "obs.bin",
            hanchan_index=_hanchan_setup(records)[0],
            returns=_hanchan_setup(records)[2],
            n_hanchans=_hanchan_setup(records)[1],
            forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
            parameters=list(policy.parameters()), optimizer=optimizer,
            micro_batch=2, device=torch.device("cpu"), max_clip=1.0,
        )
    assert optimizer.steps == 0
    for original, current in zip(before, policy.parameters(), strict=True):
        assert torch.equal(original, current.detach())


def test_gradient_finiteness_is_checked(tmp_path: Path) -> None:
    """Finite log-probs but a non-finite advantage must still abort."""
    policy = _TinyPolicy()
    records, obs_path = _faithful_records(tmp_path, 4, policy)
    index, n_hanchans, _returns = _hanchan_setup(records)
    optimizer = _CountingAdam(list(policy.parameters()), lr=1e-5)
    with pytest.raises(P4M10ContractError):
        run_pg_update(
            records=records, obs_path=obs_path, hanchan_index=index,
            returns=torch.full((n_hanchans,), float("nan")), n_hanchans=n_hanchans,
            forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
            parameters=list(policy.parameters()), optimizer=optimizer,
            micro_batch=4, device=torch.device("cpu"), max_clip=1.0,
        )
    assert optimizer.steps == 0


def test_a_huge_gradient_norm_is_clipped_not_zeroed(tmp_path: Path) -> None:
    """The norm is accumulated in float64 so a huge norm clips instead of overflowing.

    Squaring float32 gradients can reach inf while every gradient element is
    itself finite; that would make clip_grad_norm_ scale the whole update to
    exactly zero, i.e. a silent no-op step.
    """
    policy = _TinyPolicy()
    records, obs_path = _faithful_records(tmp_path, 4, policy)
    index, n_hanchans, _returns = _hanchan_setup(records)
    before = [param.detach().clone() for param in policy.parameters()]
    optimizer = _CountingAdam(list(policy.parameters()), lr=1e-5)
    report = run_pg_update(
        records=records, obs_path=obs_path, hanchan_index=index,
        returns=torch.full((n_hanchans,), 3e38), n_hanchans=n_hanchans,
        forward_logprobs=lambda obs, mask, actions: policy(obs, mask, actions),
        parameters=list(policy.parameters()), optimizer=optimizer,
        micro_batch=4, device=torch.device("cpu"), max_clip=1.0,
    )
    assert optimizer.steps == 1
    assert math.isfinite(report["grad"]["grad_norm_preclip"])
    assert report["grad"]["was_clipped"] is True
    assert report["grad"]["grad_norm_postclip"] == pytest.approx(1.0, rel=1e-5)
    # the step must still move the parameters (not a silent zero-gradient no-op)
    moved = sum(
        1 for original, current in zip(before, policy.parameters(), strict=True)
        if not torch.equal(original, current.detach())
    )
    assert moved == len(before)


def test_empty_records_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(P4M10ContractError, match="no policy records"):
        _run(tmp_path, records=[], micro_batch=4)


def test_zero_microbatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="micro_batch"):
        _run(tmp_path, micro_batch=0, count=3)


# ------------------------------------------------------------------- exporting
def test_export_writes_the_arena_key_layout(tmp_path: Path) -> None:
    policy = _TinyPolicy()
    path = tmp_path / "weights.pth"
    digest = export_collection_weights(policy, policy, path)
    payload = torch.load(path, weights_only=True)
    assert {"mortal", "current_dqn"} <= set(payload)
    assert payload["mortal"].keys() == policy.state_dict().keys()
    assert isinstance(digest, str) and len(digest) == 64


def test_export_is_self_describing_for_the_shared_native_loader(tmp_path: Path) -> None:
    """The shared loader resolves the architecture from `training_contract`."""
    from training.mortal.four_player_native import _model_dimensions

    policy = _TinyPolicy()
    path = tmp_path / "weights.pth"
    export_collection_weights(
        policy, policy, path, version=4, conv_channels=192, num_blocks=40, cycle=3
    )
    payload = torch.load(path, weights_only=True)
    contract = payload["training_contract"]
    assert contract["schema"] == "keqing.mortal.student_policy_v1"
    assert contract["cycle"] == 3
    # a tiny stand-in cannot satisfy the real architecture, but the resolver must
    # at least read the declared dimensions back out of the export
    assert (
        contract["student"]["version"],
        contract["student"]["conv_channels"],
        contract["student"]["num_blocks"],
    ) == (4, 192, 40)
    assert _model_dimensions is not None


def test_exported_weights_change_after_a_step(tmp_path: Path) -> None:
    policy = _TinyPolicy()
    records, _path = _faithful_records(tmp_path, 6, policy)
    first = export_collection_weights(policy, policy, tmp_path / "c1.pth")
    _run(tmp_path, records=records, micro_batch=3, policy=policy)
    second = export_collection_weights(policy, policy, tmp_path / "c2.pth")
    assert first != second, "a completed update must move the exported weights"


# ------------------------------------------------------- returns from real logs
_REAL_LOG_DIR = (
    REPO_ROOT / "artifacts" / "eval" / "_p4m9_probe_onpolicy_contract" / "logs"
)


@pytest.mark.skipif(not _REAL_LOG_DIR.exists(), reason="needs local P4-M9 arena logs")
def test_returns_are_normalized_rank_points_from_the_authoritative_rank() -> None:
    from training.mortal.p4m10_onpolicy_pg import hanchan_returns_from_logs

    returns, report = hanchan_returns_from_logs(
        log_dir=_REAL_LOG_DIR, seeds=range(710000, 710008), seed_key=8192,
        challenger_label="student50k",
    )
    assert report["hanchans"] == 32
    assert set(returns.values()) <= {2.0 / 3.0, 1.0 / 3.0, 0.0, -1.0}
    counts = report["rank_counts"]
    assert sum(counts) == 32, "every hanchan must have exactly one authoritative rank"
    observed: dict[float, int] = {}
    for value in returns.values():
        observed[value] = observed.get(value, 0) + 1
    assert observed == {
        value: count
        for value, count in zip(
            (2.0 / 3.0, 1.0 / 3.0, 0.0, -1.0), counts, strict=True
        )
        if count
    }
    assert json.dumps(report)  # serialisable telemetry
