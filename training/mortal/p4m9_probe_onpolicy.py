#!/usr/bin/env python3
"""P4-M9: on-policy sampling/update contract + resource probe.

Scope (owner-authorized, 2026-09-12): **contract verification and resource
measurement only**.  This script

* collects exactly **one native batch** (default 8 seeds = 32 hanchans) with the
  challenger seat sampling from its own policy,
* verifies that sampling / execution / recomputation / backward are mutually
  consistent, and
* measures what one batch of trajectory collection plus a real policy-gradient
  backward actually costs.

It does **not** train a candidate, does **not** overwrite the parent checkpoint,
does **not** start a collect--update loop, and introduces **no** learned critic,
PPO clipping, entropy term, KL anchor, replay buffer, BC mixing or opponent pool.
The single optimizer step it performs runs on a throwaway in-memory copy of the
weights, purely to prove the gradient path is real.

Frozen sampling contract for this probe
---------------------------------------
``boltzmann_epsilon=1`` (draw the action from the distribution; *not* a uniform
random action), ``boltzmann_temp=1``, ``top_p=1``, FP32, ``stochastic_latent=False``.
Under those settings ``MortalEngine._react_batch`` reduces to

    pi(a|s) = softmax(q_legal(s) / T)          with T = 1

because ``epsilon=1`` makes every ``is_greedy`` flag False and ``top_p >= 1``
makes ``sample_top_p`` return ``Categorical(logits=logits).sample()`` where
``logits`` is ``q`` masked to ``-inf`` on illegal actions.

Known post-sampling rewrite (must be *detected*, not fixed here)
---------------------------------------------------------------
``enable_rule_based_agari_guard=True`` (which is what the production loader sets)
lets the Rust agent replace a model-chosen agari (action 43) with the argmax of Q
over the other actions when the rule-based agari check disagrees
(``third_party/Mortal/libriichi/src/agent/mortal.rs:384``).  The executed action
is then *not* the sampled action, and the substitute is greedy and Q-dependent,
so it is neither a fixed environment mapping nor safely filterable by the
sampled action.  This probe only detects and reports such cases.

Observability used
------------------
An engine that exposes the attribute ``supports_decision_context = True`` is
called as ``react_batch(states, masks, invisible_states, contexts)`` where
``contexts`` carries ``(generation_seed, seed_key, seat, kyoku_index,
decision_index, own_riichi, exploration_allowed)``
(``third_party/Mortal/libriichi/src/agent/defs.rs``).  ``exploration_allowed``
is False exactly for the kan-select sub-decision, which is therefore
identifiable and excluded from the policy update.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mortal.eval_metrics import resolve_rank_points

# ---------------------------------------------------------------------------
# action space (third_party/Mortal/libriichi/src/consts.rs, ACTION_SPACE = 46)
# ---------------------------------------------------------------------------
ACTION_SPACE = 46
ACTION_DISCARD_MAX = 36  # 0..=36 -> discard tile index
ACTION_RIICHI = 37
ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH = 38, 39, 40
ACTION_PON = 41
ACTION_KAN = 42
ACTION_AGARI = 43
ACTION_RYUKYOKU = 44
ACTION_PASS = 45

# tile index == discard action index; order from libriichi/src/tile.rs
TILE_STRINGS = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
]
TILE_INDEX = {name: index for index, name in enumerate(TILE_STRINGS)}

AKA_TO_NORMAL = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}

DECISION_EVENT_TYPES = (
    "dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan",
)

# Frozen probe policy.  Everything the eventual candidate must reproduce.
PROBE_POLICY: dict[str, Any] = {
    "boltzmann_epsilon": 1.0,
    "boltzmann_temp": 1.0,
    "top_p": 1.0,
    "stochastic_latent": False,
    "enable_amp": False,
    "dtype": "float32",
    "distribution": "pi(a|s) = softmax(q_legal(s) / T), T = 1",
    "excluded": "forced actions, quick-eval bypasses, kan-select sub-decisions, "
                "post-sampling rewrites (agari guard)",
}

# Declared numeric tolerances for "the sampling-time log-prob can be recomputed".
# Replaying the *same* batch shape only has to absorb fp32 log-softmax rounding;
# a different batch shape additionally changes cuDNN kernel/accumulation order,
# which the owner explicitly allows ("allow float noise from batch shape").
LOGPROB_SAME_SHAPE_TOL = 1e-5
LOGPROB_BATCH_SHAPE_TOL = 1e-3


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_p4m9_probe_contract.py)
# ---------------------------------------------------------------------------
def deaka(pai: str) -> str:
    """Map an aka tile string to its normal tile (5mr -> 5m)."""
    return AKA_TO_NORMAL.get(pai, pai)


def tile_index(pai: str) -> int:
    return TILE_INDEX[pai]


def mask_to_bits(mask: Any) -> int:
    """Replicates MortalBatchAgent::gen_meta bit packing (bit i <=> mask[i])."""
    bits = 0
    for index, flag in enumerate(mask):
        if bool(flag):
            bits |= 1 << index
    return bits


def bits_to_mask(bits: int) -> list[bool]:
    return [bool((bits >> index) & 1) for index in range(ACTION_SPACE)]


def compact_legal(q_row: Any, mask_row: Any) -> list[float]:
    """Legal-only q values in ascending action index order (log `q_values`)."""
    return [
        float(value)
        for index, value in enumerate(q_row)
        if bool(list(mask_row)[index])
    ]


def legal_logits(q_row: Any, mask_row: Any, temperature: float = 1.0) -> np.ndarray:
    """q_legal / T with -inf on illegal actions (exact sampler logits)."""
    q = np.asarray(q_row, dtype=np.float64)
    mask = np.asarray([bool(value) for value in mask_row], dtype=bool)
    logits = q / float(temperature)
    logits = np.where(mask, logits, -np.inf)
    return logits


def sampling_log_prob(
    q_row: Any, mask_row: Any, action: int, temperature: float = 1.0
) -> float:
    """log pi(action|s) under the frozen sampling contract."""
    logits = legal_logits(q_row, mask_row, temperature)
    legal = np.isfinite(logits)
    if not bool(legal[int(action)]):
        return float("-inf")
    values = logits[legal]
    max_value = float(np.max(values))
    log_sum_exp = max_value + float(np.log(np.sum(np.exp(values - max_value))))
    return float(logits[int(action)] - log_sum_exp)


def chi_action_index(pai: str, consumed: Any) -> int | None:
    """Classify a logged chi into the 38/39/40 action index."""
    if not consumed or len(consumed) != 2:
        return None
    called = tile_index(deaka(pai))
    taken = sorted(tile_index(deaka(tile)) for tile in consumed)
    # chi_low: consumed = pai+1, pai+2 ; chi_mid: pai-1, pai+1 ; chi_high: pai-2, pai-1
    if taken == [called + 1, called + 2]:
        return ACTION_CHI_LOW
    if taken == [called - 1, called + 1]:
        return ACTION_CHI_MID
    if taken == [called - 2, called - 1]:
        return ACTION_CHI_HIGH
    return None


def event_to_action(event: dict[str, Any]) -> int | None:
    """Translate a logged decision event into the action index it represents.

    Returns None when the event does not determine a unique action index.
    """
    kind = event.get("type")
    if kind == "dahai":
        pai = event.get("pai")
        return tile_index(pai) if pai in TILE_INDEX else None
    if kind == "reach":
        return ACTION_RIICHI
    if kind == "chi":
        return chi_action_index(str(event.get("pai")), event.get("consumed"))
    if kind == "pon":
        return ACTION_PON
    if kind in ("daiminkan", "ankan", "kakan"):
        return ACTION_KAN
    if kind == "hora":
        return ACTION_AGARI
    if kind == "ryukyoku":
        return ACTION_RYUKYOKU
    if kind == "none":
        return ACTION_PASS
    return None


def compatible_with_action(action: int, event_type: str) -> bool:
    """Whether a sampled action index could have produced this logged event."""
    if 0 <= action <= ACTION_DISCARD_MAX:
        return event_type == "dahai"
    if action == ACTION_RIICHI:
        return event_type == "reach"
    if action in (ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH):
        return event_type == "chi"
    if action == ACTION_PON:
        return event_type == "pon"
    if action == ACTION_KAN:
        return event_type in ("daiminkan", "ankan", "kakan")
    return False


def q_values_match(left: Any, right: Any, rel_tol: float = 1e-6) -> bool:
    """Compare two legal-only q lists at float32 resolution."""
    if left is None or right is None or len(left) != len(right):
        return False
    for a, b in zip(left, right):
        a32 = float(np.float32(a))
        b32 = float(np.float32(b))
        if a32 == b32:
            continue
        if abs(a32 - b32) > rel_tol * max(1.0, abs(a32), abs(b32)):
            return False
    return True


def hanchan_return(scores: Any, init_score: int = 25000, scale: float = 1000.0) -> float:
    """Probe-only placeholder return; NOT the frozen rank-point definition."""
    return float((float(scores) - float(init_score)) / float(scale))


def pg_loss(
    log_probs: torch.Tensor,
    hanchan_ids: torch.Tensor,
    hanchan_returns: torch.Tensor,
    n_hanchans: int,
    baseline: float = 0.0,
) -> torch.Tensor:
    """REINFORCE loss, per-hanchan mean, log-prob summed within a hanchan.

        L = -(1/N) * sum_h (G_h - b) * sum_{t in h} log pi(a_t|s_t)

    ``baseline=0`` is the frozen first-version probe setting: nothing in the
    baseline may depend on the current trajectory's own return.
    """
    if n_hanchans <= 0:
        raise ValueError("n_hanchans must be positive")
    advantage = hanchan_returns[hanchan_ids] - float(baseline)
    per_hanchan = torch.zeros(
        n_hanchans, dtype=log_probs.dtype, device=log_probs.device
    ).index_add(0, hanchan_ids, advantage * log_probs)
    return -per_hanchan.sum() / float(n_hanchans)


CLAIM_ACTIONS = (
    ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH, ACTION_PON, ACTION_KAN,
)


def decision_matches(record: dict[str, Any], logged: dict[str, Any]) -> bool:
    """Identify a probe decision with a logged decision event.

    The logged ``meta`` is produced from the same ``q`` tensor and mask the probe
    recorded, so ``(mask_bits, q_values)`` is an exact identity for the decision.
    Matching on identity - instead of assuming a strict 1:1 event order - keeps a
    single non-executed action from shifting every later comparison.
    """
    meta = logged.get("meta") or {}
    if int(meta.get("mask_bits", -1)) != int(record["mask_bits"]):
        return False
    return q_values_match(meta.get("q_values"), record["q_legal"])


def reconcile_kyoku(
    probe_records: list[dict[str, Any]],
    log_decision_events: list[dict[str, Any]],
    *,
    terminal: dict[str, Any] | None = None,
    challenger_seat: int | None = None,
) -> dict[str, Any]:
    """Align one challenger seat's probe records against its logged events.

    Reported classes:

    ``aligned``                 sampled action == executed action (identity-checked);
    ``sampled_pass``            action 45; ``Event::None`` is never logged;
    ``sampled_agari``           action 43 with no executed event; the kyoku should
                                have been resolved in this seat's favour;
    ``sampled_ryukyoku``        action 44; logged without decision meta;
    ``executed_action_differs`` the environment executed a *different* action than
                                the one sampled (post-sampling rewrite);
    ``agari_guard_suspected``   a sampled agari whose kyoku did not resolve for
                                this seat - with ``agari_guard`` enabled the Rust
                                agent replaces action 43 by the argmax alternative;
    ``claim_not_executed``      a sampled chi/pon/kan that never executed, usually
                                because a higher-priority claim resolved the same
                                discard first;
    ``mismatches``              structural problems that must stay zero.
    """
    terminal = terminal or {}
    report: dict[str, Any] = {
        "policy_decisions": 0,
        "kan_select_states": 0,
        "aligned": 0,
        "sampled_pass": 0,
        "sampled_agari": 0,
        "sampled_ryukyoku": 0,
        "executed_action_differs": [],
        "agari_guard_suspected": [],
        "claim_not_executed": [],
        "mismatches": [],
        "kan_select_checks": 0,
        "kan_select_mismatches": [],
        "unmatched_log_events": 0,
    }
    ordered = sorted(probe_records, key=lambda item: item["dec"])
    pointer = 0
    pending_kan_select: dict[str, Any] | None = None
    for index, record in enumerate(ordered):
        if not record["explore"]:
            report["kan_select_states"] += 1
            pending_kan_select = record
            continue
        report["policy_decisions"] += 1
        action = int(record["action"])

        # Neither `Event::None` (pass) nor `Event::Ryukyoku` is logged with
        # decision meta, so neither may consume a log event here: matching one
        # by identity would silently shift the whole alignment.
        if action == ACTION_PASS:
            report["sampled_pass"] += 1
            pending_kan_select = None
            continue
        if action == ACTION_RYUKYOKU:
            report["sampled_ryukyoku"] += 1
            pending_kan_select = None
            continue

        if pointer < len(log_decision_events) and decision_matches(
            record, log_decision_events[pointer]
        ):
            logged = log_decision_events[pointer]
            pointer += 1
            executed = logged.get("observed_action")
            if executed is not None and int(executed) == action:
                report["aligned"] += 1
            else:
                report["executed_action_differs"].append({
                    "decision_index": int(record["dec"]),
                    "sampled_action": action,
                    "executed_action": executed,
                    "log_event_type": logged["type"],
                    "sampled_logprob": float(record["logprob"]),
                })
            meta = logged.get("meta") or {}
            kan_meta = meta.get("kan_select")
            if isinstance(kan_meta, dict):
                report["kan_select_checks"] += 1
                if pending_kan_select is None or not (
                    int(kan_meta.get("mask_bits", -1))
                    == int(pending_kan_select["mask_bits"])
                    and q_values_match(
                        kan_meta.get("q_values"), pending_kan_select["q_legal"]
                    )
                ):
                    report["kan_select_mismatches"].append({
                        "decision_index": int(record["dec"]),
                        "had_probe_kan_select_record": pending_kan_select is not None,
                    })
            pending_kan_select = None
            continue

        # No executed event carries this decision's identity.
        hora_actors = [actor for actor in (terminal.get("hora_actors") or []) if actor is not None]
        seat_won = (
            challenger_seat is not None
            and any(int(actor) == int(challenger_seat) for actor in hora_actors)
        )
        if action == ACTION_AGARI:
            report["sampled_agari"] += 1
            if not seat_won:
                report["agari_guard_suspected"].append({
                    "decision_index": int(record["dec"]),
                    "sampled_action": action,
                    "sampled_logprob": float(record["logprob"]),
                    "kyoku_winners": [int(actor) for actor in hora_actors],
                    "kyoku_ended_by": terminal.get("ended_by"),
                    "challenger_seat": (
                        int(challenger_seat) if challenger_seat is not None else None
                    ),
                })
            continue
        if action in CLAIM_ACTIONS:
            report["claim_not_executed"].append({
                "decision_index": int(record["dec"]),
                "sampled_action": action,
                "sampled_logprob": float(record["logprob"]),
                "kyoku_ended_by": terminal.get("ended_by"),
                "kyoku_winners": [int(actor) for actor in hora_actors],
                # The kyoku's final discard, NOT necessarily the tile the claim
                # targeted; it is recorded to make the priority story checkable.
                "kyoku_last_discard_actor": terminal.get("last_discard_actor"),
            })
            continue
        # A discard must always execute; if it did not, the alignment itself is
        # suspect and must not be silent.
        report["mismatches"].append({
            "kind": "discard_without_executed_event",
            "decision_index": int(record["dec"]),
            "sampled_action": action,
            "next_log_event_type": (
                log_decision_events[pointer]["type"]
                if pointer < len(log_decision_events) else None
            ),
            "next_probe_decision_action": (
                int(ordered[index + 1]["action"]) if index + 1 < len(ordered) else None
            ),
        })
        pending_kan_select = None

    report["unmatched_log_events"] = len(log_decision_events) - pointer
    report["sampled_action_equals_executed_action"] = (
        not report["executed_action_differs"]
        and not report["agari_guard_suspected"]
        and not report["claim_not_executed"]
    )
    report["alignment_complete"] = (
        not report["mismatches"] and report["unmatched_log_events"] == 0
    )
    return report


# ---------------------------------------------------------------------------
# recorder + probe engine
# ---------------------------------------------------------------------------
class ProbeRecorder:
    """Records every decision the challenger engine is asked to act on."""

    def __init__(self, obs_path: Path) -> None:
        self.obs_path = obs_path
        self.obs_handle = obs_path.open("wb")
        self.obs_offset = 0
        self.records: list[dict[str, Any]] = []
        self.call_index = 0
        self.batch_sizes: list[int] = []
        self.per_call_seconds: list[float] = []
        self.write_seconds = 0.0
        self._call_marker: tuple[int, float] = (0, 0.0)

    def close(self) -> None:
        self.obs_handle.close()

    def begin_call(
        self, obs: Any, masks: Any, contexts: Any
    ) -> tuple[int, int]:
        self._call_marker = (len(self.records), time.perf_counter())
        self.call_index += 1
        count = len(obs)
        self.batch_sizes.append(count)
        write_started = time.perf_counter()
        for element in range(count):
            array = np.ascontiguousarray(obs[element], dtype=np.float32)
            payload = array.tobytes()
            self.obs_handle.write(payload)
            self.records.append({
                "i": len(self.records),
                "call": self.call_index - 1,
                "elem": element,
                "obs_off": self.obs_offset,
                "obs_bytes": len(payload),
                "obs_shape": list(array.shape),
                "mask_row": [bool(value) for value in masks[element]],
                "context": (
                    list(contexts[element]) if contexts is not None else None
                ),
                "call_seconds": None,
            })
            self.obs_offset += len(payload)
        self.write_seconds += time.perf_counter() - write_started
        return self.call_index - 1, count

    def end_call(self, result: Any, temperature: float) -> None:
        actions, q_values, _masks, is_greedy = result
        start_index, call_started = self._call_marker
        count = len(actions)
        for element in range(count):
            record = self.records[start_index + element]
            mask_row = record["mask_row"]
            q_row = q_values[element]
            action = int(actions[element])
            if record["context"] is not None:
                (
                    generation_seed,
                    seed_key,
                    seat,
                    kyoku_index,
                    decision_index,
                    own_riichi,
                    exploration_allowed,
                ) = record["context"]
                record["seed"] = int(generation_seed)
                record["seed_key"] = int(seed_key)
                record["seat"] = int(seat)
                record["kyoku"] = int(kyoku_index)
                record["dec"] = int(decision_index)
                record["own_riichi"] = bool(own_riichi)
                record["explore"] = bool(exploration_allowed)
            else:
                record["seed"] = None
                record["seed_key"] = None
                record["seat"] = None
                record["kyoku"] = None
                record["dec"] = None
                record["own_riichi"] = None
                record["explore"] = None
            record["action"] = action
            record["is_greedy"] = bool(is_greedy[element])
            record["mask_bits"] = mask_to_bits(mask_row)
            record["q_legal"] = compact_legal(q_row, mask_row)
            record["logprob"] = sampling_log_prob(q_row, mask_row, action, temperature)
            record["call_seconds"] = time.perf_counter() - call_started
        self.per_call_seconds.append(time.perf_counter() - call_started)
        for element in range(count):
            del self.records[start_index + element]["mask_row"]


class ProbeEngine:
    """Builds a ``MortalEngine`` subclass that records decisions and opts in to
    ``DecisionContext``.

    Defined as a factory so the third-party package is only imported at call
    time (``third_party/Mortal/mortal`` shadowing rules).
    """

    @staticmethod
    def build(base_class: Any, *, recorder: ProbeRecorder, options: dict[str, Any]) -> Any:
        class _ProbeEngine(base_class):  # type: ignore[misc, valid-type]
            supports_decision_context = True

            def react_batch(self, obs, masks, invisible_obs, contexts=None):
                recorder.begin_call(obs, masks, contexts)
                result = super().react_batch(obs, masks, invisible_obs)
                recorder.end_call(result, float(options["boltzmann_temp"]))
                return result

        return _ProbeEngine


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _launch_environment() -> dict[str, Any]:
    """Fingerprint the environment AT LAUNCH TIME (never reconstructed later)."""
    import libriichi

    libriichi_file = Path(libriichi.__file__).resolve()
    site_packages = libriichi_file.parent.parent
    binaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in (site_packages, libriichi_file.parent):
        for pattern in ("riichi*.pyd", "riichi*.so"):
            for path in sorted(root.glob(pattern)):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                binaries.append({
                    "path": key,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
    arena_names: list[str] = []
    try:
        from libriichi import arena as _arena
        arena_names = sorted(
            name for name in dir(_arena) if not name.startswith("_")
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        arena_names = [f"<import failed: {error}>"]
    return {
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "libriichi_module_file": str(libriichi_file),
        "native_binaries": binaries,
        "arena_exports": arena_names,
        "recorded_at_launch": True,
    }


def _load_state(state_file: Path) -> tuple[int, int, int, dict[str, Any]]:
    from training.mortal.four_player_native import _model_dimensions

    state = torch.load(state_file, weights_only=True, map_location=torch.device("cpu"))
    version, conv_channels, num_blocks = _model_dimensions(state)
    return version, conv_channels, num_blocks, state


def _build_modules(
    state: dict[str, Any],
    *,
    mortal_root: Path,
    device: torch.device,
    version: int,
    conv_channels: int,
    num_blocks: int,
) -> tuple[Any, Any, Any]:
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from engine import MortalEngine
    from model import DQN, Brain

    brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
    dqn = DQN(version=version)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine, brain.to(device).eval(), dqn.to(device).eval()


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def collect_probe_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of what the recorder saw at sampling time."""
    greedy = sum(1 for record in records if record["is_greedy"])
    by_action: dict[str, int] = {}
    for record in records:
        key = str(record["action"])
        by_action[key] = by_action.get(key, 0) + 1
    return {
        "decisions_recorded": len(records),
        "policy_states": sum(1 for record in records if record["explore"]),
        "kan_select_states": sum(1 for record in records if not record["explore"]),
        "is_greedy_true": greedy,
        "sampled_agari": by_action.get(str(ACTION_AGARI), 0),
        "sampled_pass": by_action.get(str(ACTION_PASS), 0),
        "non_finite_logprobs": sum(
            1 for record in records if not math.isfinite(float(record["logprob"]))
        ),
        "distinct_seeds": len({record["seed"] for record in records}),
        "distinct_hanchans": len(
            {(record["seed"], record["seat"]) for record in records}
        ),
        "action_histogram": dict(sorted(by_action.items(), key=lambda kv: int(kv[0]))),
    }


def challenger_rank_from_log(log_text: str, challenger_seat: int) -> int | None:
    """Authoritative per-hanchan rank, via the same native Stat the arena uses."""
    try:
        from libriichi.stat import Stat
    except ImportError:  # pragma: no cover - environment dependent
        return None
    stat = Stat.from_log(log_text, challenger_seat)
    counts = []
    for rank in range(1, 5):
        value = getattr(stat, f"rank_{rank}")
        counts.append(int(value() if callable(value) else value))
    if sum(counts) != 1:
        return None
    return counts.index(1) + 1


def parse_log_file(
    path: Path, challenger_label: str
) -> tuple[int, dict[int, list[dict[str, Any]]], dict[str, Any], list[int], int | None]:
    """Return (challenger_seat, decisions_by_kyoku, header, final_scores, rank).

    ``final_scores`` is reconstructed by accumulating the ``deltas`` of the
    logged ``hora`` / ``ryukyoku`` events, because ``end_game`` in these logs
    carries no scores.
    """
    challenger_seat = -1
    kyoku_index = -1
    decisions: dict[int, list[dict[str, Any]]] = {}
    terminals: dict[int, dict[str, Any]] = {}
    forced_no_meta = 0
    final_scores = [25000, 25000, 25000, 25000]
    header: dict[str, Any] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        log_text = handle.read()
    for line in log_text.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "start_game":
            names = event.get("names") or []
            header = dict(event)
            if challenger_label in names:
                challenger_seat = names.index(challenger_label)
            continue
        if kind == "start_kyoku":
            kyoku_index += 1
            decisions.setdefault(kyoku_index, [])
            terminals[kyoku_index] = {
                "hora_actors": [],
                "ended_by": None,
                "last_discard_actor": None,
            }
            continue
        if kind == "dahai" and kyoku_index >= 0:
            # Tracked for the terminal record only; the event still has to fall
            # through to the decision collection below.
            terminals.setdefault(kyoku_index, {})["last_discard_actor"] = event.get("actor")
        if kind in ("hora", "ryukyoku"):
            deltas = event.get("deltas") or []
            if len(deltas) == 4:
                final_scores = [
                    total + int(delta) for total, delta in zip(final_scores, deltas)
                ]
            if kyoku_index >= 0:
                entry = terminals.setdefault(kyoku_index, {})
                entry["ended_by"] = kind
                if kind == "hora":
                    # My terminal record is the collected flag, not the last event.
                    entry.setdefault("hora_actors", []).append(event.get("actor"))
            continue
        if kind == "end_game":
            continue
        if kyoku_index < 0 or kind not in DECISION_EVENT_TYPES:
            continue
        if int(event.get("actor", -1)) != challenger_seat:
            continue
        meta = event.get("meta")
        if not meta:
            # quick-eval bypass: the model was never consulted.
            forced_no_meta += 1
            continue
        decisions[kyoku_index].append({
            "type": kind,
            "meta": meta,
            "observed_action": event_to_action(event),
        })
    header["forced_no_meta_decisions"] = forced_no_meta
    rank = (
        challenger_rank_from_log(log_text, challenger_seat)
        if challenger_seat >= 0
        else None
    )
    return challenger_seat, decisions, header, final_scores, rank, terminals


def check_sampling_vs_execution(
    *,
    log_dir: Path,
    records: list[dict[str, Any]],
    challenger_label: str,
    seed_key: int,
    seed_start: int,
    seed_count: int,
) -> dict[str, Any]:
    by_hanchan: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        if record["seed"] is None:
            continue
        by_hanchan.setdefault((record["seed"], record["seat"]), []).append(record)

    totals = {
        "hanchans_compared": 0,
        "policy_decisions": 0,
        "aligned": 0,
        "mismatches": [],
        "post_sampling_rewrites": [],
        "sampled_agari": 0,
        "sampled_ryukyoku": 0,
        "sampled_pass": 0,
        "kan_select_states": 0,
        "kan_select_checks": 0,
        "kan_select_mismatches": [],
        "executed_action_differs": [],
        "executed_action_differs_count": 0,
        "agari_guard_suspected": [],
        "agari_guard_suspected_count": 0,
        "claim_not_executed": [],
        "claim_not_executed_count": 0,
        "unmatched_log_events": 0,
        "forced_decisions_not_offered_to_model": 0,
        "decision_index_gaps_lower_bound": 0,
    }
    per_hanchan: list[dict[str, Any]] = []
    seen_seats: dict[int, int] = {}
    seat_matches_split: bool | None = None

    for seed in range(seed_start, seed_start + seed_count):
        for split_index, split in enumerate("abcd"):
            path = log_dir / f"{seed}_{seed_key}_{split}.json.gz"
            if not path.exists():
                raise FileNotFoundError(f"missing arena log: {path}")
            seat, decisions, header, final_scores, rank, terminals = parse_log_file(
                path, challenger_label
            )
            if seat < 0:
                raise RuntimeError(f"challenger label {challenger_label} not in {path.name}")
            seen_seats[split_index] = seat
            seat_matches_split = (
                seat == split_index if seat_matches_split is None
                else seat_matches_split and seat == split_index
            )
            hanchan_records = by_hanchan.get((seed, seat), [])
            if not hanchan_records:
                continue
            totals["hanchans_compared"] += 1
            totals["forced_decisions_not_offered_to_model"] += int(
                header.get("forced_no_meta_decisions", 0)
            )
            honchan_report = {"seed": seed, "split": split, "seat": seat}
            merged = {
                "policy_decisions": 0, "kan_select_states": 0, "aligned": 0,
                "sampled_agari": 0, "sampled_ryukyoku": 0, "sampled_pass": 0,
                "unmatched_log_events": 0, "kan_select_checks": 0,
                "executed_action_differs_count": 0,
                "agari_guard_suspected_count": 0,
                "claim_not_executed_count": 0,
            }
            for kyoku_index in sorted({record["kyoku"] for record in hanchan_records}):
                kyoku_records = [
                    record for record in hanchan_records
                    if record["kyoku"] == kyoku_index
                ]
                report = reconcile_kyoku(
                    kyoku_records,
                    decisions.get(kyoku_index, []),
                    terminal=terminals.get(kyoku_index),
                    challenger_seat=seat,
                )
                merged["executed_action_differs_count"] += len(
                    report["executed_action_differs"]
                )
                merged["agari_guard_suspected_count"] += len(
                    report["agari_guard_suspected"]
                )
                merged["claim_not_executed_count"] += len(report["claim_not_executed"])
                for key in merged:
                    if key in report:
                        merged[key] += report[key]
                totals["mismatches"].extend(
                    {"seed": seed, "split": split, **item} for item in report["mismatches"]
                )
                totals["kan_select_mismatches"].extend(
                    {"seed": seed, "split": split, **item}
                    for item in report["kan_select_mismatches"]
                )
                totals["executed_action_differs"].extend(
                    {"seed": seed, "split": split, **item}
                    for item in report["executed_action_differs"]
                )
                totals["agari_guard_suspected"].extend(
                    {"seed": seed, "split": split, **item}
                    for item in report["agari_guard_suspected"]
                )
                totals["claim_not_executed"].extend(
                    {"seed": seed, "split": split, **item}
                    for item in report["claim_not_executed"]
                )
                # decision_index gaps: a forced (quick-eval) decision consumed an
                # index without being offered to the model.  Kan-select states
                # share their decision_index with the action record.
                all_indexes = {record["dec"] for record in kyoku_records}
                expected = max(all_indexes, default=-1) + 1
                # A forced decision that is the last one in its kyoku leaves no
                # gap behind, so this counts a LOWER BOUND; the authoritative
                # count is forced_decisions_not_offered_to_model above.
                if expected > len(all_indexes):
                    totals["decision_index_gaps_lower_bound"] += (
                        expected - len(all_indexes)
                    )
            honchan_report.update(merged)
            honchan_report["final_scores"] = final_scores
            honchan_report["challenger_rank"] = rank
            per_hanchan.append(honchan_report)
            for key in (
                "policy_decisions", "aligned", "sampled_agari", "sampled_ryukyoku",
                "sampled_pass", "kan_select_states", "unmatched_log_events",
                "kan_select_checks", "executed_action_differs_count",
                "agari_guard_suspected_count", "claim_not_executed_count",
            ):
                totals[key] += merged[key]

    totals["challenger_seats_per_split"] = seen_seats
    totals["challenger_seat_equals_split_index"] = seat_matches_split
    totals["per_hanchan"] = per_hanchan
    # Agari / ryukyoku / pass decisions end or skip the kyoku and are logged
    # without decision meta (hora/ryukyoku) or not logged at all (none), so they
    # consume no aligned log event.  Every other policy decision must align.
    totals["policy_decisions_accounted_for"] = (
        totals["aligned"]
        + totals["sampled_agari"]
        + totals["sampled_ryukyoku"]
        + totals["sampled_pass"]
        + totals["executed_action_differs_count"]
        + totals["claim_not_executed_count"]
    )
    totals["decision_class_accounting_ok"] = (
        totals["policy_decisions_accounted_for"] == totals["policy_decisions"]
    )
    # Alignment integrity: every decision is explained and no logged event is left
    # over.  This is what certifies that the comparison itself is trustworthy.
    totals["alignment_complete"] = (
        totals["decision_class_accounting_ok"]
        and not totals["mismatches"]
        and not totals["kan_select_mismatches"]
        and totals["unmatched_log_events"] == 0
    )
    # The stronger assertion the owner asked about: was the sampled action the
    # action that actually executed?
    totals["sampled_action_equals_executed_action"] = (
        totals["alignment_complete"]
        and totals["executed_action_differs_count"] == 0
        and totals["agari_guard_suspected_count"] == 0
        and totals["claim_not_executed_count"] == 0
    )
    return totals


def _read_obs(path: Path, offset: int, nbytes: int, shape: list[int]) -> np.ndarray:
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(nbytes)
    return np.frombuffer(payload, dtype=np.float32).reshape(shape)


def check_logprob_recompute(
    *,
    obs_path: Path,
    records: list[dict[str, Any]],
    brain: Any,
    dqn: Any,
    device: torch.device,
    temperature: float,
    chunk: int,
    same_shape_tolerance: float = LOGPROB_SAME_SHAPE_TOL,
    batch_shape_tolerance: float = LOGPROB_BATCH_SHAPE_TOL,
) -> dict[str, Any]:
    """Recompute log pi(a|s) on the un-updated weights in update mode.

    "Update mode" means: module in ``eval()`` (BN uses running statistics, so the
    policy is per-sample deterministic and independent of batch composition) but
    **autograd enabled** and outside ``torch.inference_mode()``.

    Two replays are reported:

    * ``same_batch_shape`` - regenerate each decision in the batch group it was
      originally sampled in.  Only fp32 log-softmax rounding may differ.
    * ``fixed_chunk`` - replay in fixed-size chunks, i.e. a different batch shape.
      Any residual difference here is the batch-shape float sensitivity, which
      the owner explicitly allows and which is *not* a contract violation.
    """
    policy = [record for record in records if record["explore"]]
    bn_modules = [m for m in brain.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    saved = [
        (module.running_mean.detach().clone(), module.running_var.detach().clone(),
         int(module.num_batches_tracked))
        for module in bn_modules
    ]

    def _measure(groups: list[list[dict[str, Any]]], tolerance: float) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "forward_calls": len(groups),
            "batch_sizes": sorted({len(group) for group in groups}),
            "decisions_checked": 0,
            "max_abs_diff": 0.0,
            "max_abs_q_diff": 0.0,
            "within_tolerance": 0,
            "tolerance": tolerance,
            "violations": [],
        }
        for group in groups:
            if not group:
                continue
            obs = torch.as_tensor(
                np.stack([
                    _read_obs(obs_path, record["obs_off"], record["obs_bytes"],
                              record["obs_shape"])
                    for record in group
                ]),
                device=device,
            )
            mask = torch.as_tensor(
                np.array([bits_to_mask(record["mask_bits"]) for record in group]),
                device=device,
            )
            actions = torch.as_tensor(
                [record["action"] for record in group], device=device
            ).unsqueeze(-1)
            with torch.enable_grad():
                phi = brain(obs)
                q_out = dqn(phi, mask)
                logits = (q_out / temperature).masked_fill(~mask, -torch.inf)
                recomputed = torch.log_softmax(logits, dim=-1).gather(
                    -1, actions
                ).squeeze(-1)
            values = recomputed.detach().cpu().tolist()
            q_rows = q_out.detach().cpu().numpy()
            masks = mask.detach().cpu().numpy()
            for position, record in enumerate(group):
                diff = abs(float(values[position]) - float(record["logprob"]))
                stats["decisions_checked"] += 1
                stats["max_abs_diff"] = max(stats["max_abs_diff"], diff)
                if diff <= tolerance:
                    stats["within_tolerance"] += 1
                elif len(stats["violations"]) < 20:
                    stats["violations"].append({
                        "i": record["i"],
                        "sampled": record["logprob"],
                        "recomputed": float(values[position]),
                        "abs_diff": diff,
                    })
                legal = masks[position]
                recomputed_legal = [
                    float(value) for value, flag in zip(q_rows[position], legal) if flag
                ]
                recorded_legal = record["q_legal"]
                if len(recomputed_legal) == len(recorded_legal):
                    for left, right in zip(recomputed_legal, recorded_legal):
                        stats["max_abs_q_diff"] = max(
                            stats["max_abs_q_diff"], abs(left - float(right))
                        )
        stats["pass"] = (
            stats["decisions_checked"] > 0
            and stats["within_tolerance"] == stats["decisions_checked"]
        )
        return stats

    by_call: dict[int, list[dict[str, Any]]] = {}
    for record in policy:
        by_call.setdefault(int(record["call"]), []).append(record)
    same_shape = _measure(
        [by_call[key] for key in sorted(by_call)], same_shape_tolerance
    )
    fixed_chunk = _measure(
        [policy[start:start + chunk] for start in range(0, len(policy), chunk)],
        batch_shape_tolerance,
    )

    after = [
        (module.running_mean.detach().clone(), module.running_var.detach().clone(),
         int(module.num_batches_tracked))
        for module in bn_modules
    ]
    return {
        "mode": "eval() + torch.enable_grad(), outside inference_mode",
        "chain_rule_checked": (
            "log pi = log_softmax(q_legal) gathered at the recorded action, "
            "then truncated back to q"
        ),
        "same_batch_shape": same_shape,
        "fixed_chunk": fixed_chunk,
        "bn_running_stats_unchanged": all(
            torch.equal(left[0], right[0])
            and torch.equal(left[1], right[1])
            and left[2] == right[2]
            for left, right in zip(saved, after)
        ),
        "recomputable": bool(same_shape["pass"]),
        "batch_shape_sensitivity_max_abs_diff": fixed_chunk["max_abs_diff"],
    }


def check_bn_mode_sensitivity(
    *,
    obs_path: Path,
    records: list[dict[str, Any]],
    brain: Any,
    dqn: Any,
    device: torch.device,
    temperature: float,
    limit: int,
) -> dict[str, Any]:
    """Diagnostic: how much does train()-mode BN move the sampled log-prob?"""
    batch = [record for record in records if record["explore"]][:limit]
    if not batch:
        return {"decisions_checked": 0}
    obs = torch.as_tensor(
        np.stack([
            _read_obs(obs_path, record["obs_off"], record["obs_bytes"], record["obs_shape"])
            for record in batch
        ]),
        device=device,
    )
    mask = torch.as_tensor(
        np.array([bits_to_mask(record["mask_bits"]) for record in batch]), device=device
    )
    actions = torch.as_tensor(
        [record["action"] for record in batch], device=device
    ).unsqueeze(-1)

    def _log_probs() -> list[float]:
        with torch.enable_grad():
            phi = brain(obs)
            q_out = dqn(phi, mask)
            logits = (q_out / temperature).masked_fill(~mask, -torch.inf)
            values = torch.log_softmax(logits, dim=-1).gather(-1, actions).squeeze(-1)
        return values.detach().cpu().tolist()

    sampled = [record["logprob"] for record in batch]
    bn_modules = [m for m in brain.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    saved = [
        (module.running_mean.detach().clone(), module.running_var.detach().clone(),
         int(module.num_batches_tracked))
        for module in bn_modules
    ]

    def _restore() -> None:
        for module, (mean, var, tracked) in zip(bn_modules, saved):
            with torch.no_grad():
                module.running_mean.copy_(mean)
                module.running_var.copy_(var)
                module.num_batches_tracked.fill_(tracked)

    brain.eval()
    eval_values = _log_probs()
    brain.train()
    train_values = _log_probs()
    _restore()
    freeze_bn_supported = hasattr(brain, "freeze_bn")
    frozen_values = None
    if freeze_bn_supported:
        # train() with BN kept in eval: parameters are grad-capable while the
        # normalisation statistics remain the ones used at sampling time.
        brain.freeze_bn(True)
        frozen_values = _log_probs()
    brain.freeze_bn(False) if freeze_bn_supported else None
    _restore()
    brain.eval()
    return {
        "decisions_checked": len(batch),
        "max_abs_diff_eval_vs_sampled": max(
            abs(a - b) for a, b in zip(eval_values, sampled)
        ),
        "max_abs_diff_train_vs_sampled": max(
            abs(a - b) for a, b in zip(train_values, sampled)
        ),
        "max_abs_diff_train_vs_eval": max(
            abs(a - b) for a, b in zip(train_values, eval_values)
        ),
        "freeze_bn_supported": bool(freeze_bn_supported),
        "max_abs_diff_frozen_bn_vs_eval": (
            max(abs(a - b) for a, b in zip(frozen_values, eval_values))
            if frozen_values is not None
            else None
        ),
        "interpretation": (
            "eval() (and train()+freeze_bn(True)) reproduce the sampling-time "
            "log-prob; plain train() does not, because BatchNorm1d then uses "
            "minibatch statistics that depend on batch composition."
        ),
    }


def check_gradient_path(
    *,
    obs_path: Path,
    records: list[dict[str, Any]],
    state: dict[str, Any],
    mortal_root: Path,
    device: torch.device,
    version: int,
    conv_channels: int,
    num_blocks: int,
    temperature: float,
    micro_batches: list[int],
    hanchan_returns: dict[tuple[int, int], float],
    baseline: float,
) -> dict[str, Any]:
    """One optimizer step on a throwaway copy; measures the real backward cost."""
    hanchan_order = sorted({(record["seed"], record["seat"]) for record in records})
    hanchan_index = {key: index for index, key in enumerate(hanchan_order)}
    returns = torch.tensor(
        [hanchan_returns.get(key, 0.0) for key in hanchan_order],
        dtype=torch.float32, device=device,
    )
    policy_records = [record for record in records if record["explore"]]

    report: dict[str, Any] = {
        "baseline": baseline,
        "hanchans": len(hanchan_order),
        "policy_decisions": len(policy_records),
        "loss_definition": "-(1/N_h) * sum_h (G_h - b) * sum_{t in h} log pi(a_t|s_t)",
        "one_batch_one_update": True,
        "micro_batch_measurements": [],
    }

    for micro in micro_batches:
        _engine_class, brain, dqn = _build_modules(
            state, mortal_root=mortal_root, device=device,
            version=version, conv_channels=conv_channels, num_blocks=num_blocks,
        )
        parameters = list(brain.parameters()) + list(dqn.parameters())
        optimizer = torch.optim.Adam(parameters, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        bn_before = [
            (module.running_mean.detach().clone(),
             module.running_var.detach().clone(),
             int(module.num_batches_tracked))
            for module in brain.modules() if isinstance(module, torch.nn.BatchNorm1d)
        ]
        n_micro = 0
        for start in range(0, len(policy_records), micro):
            batch = policy_records[start:start + micro]
            obs = torch.as_tensor(
                np.stack([
                    _read_obs(obs_path, record["obs_off"], record["obs_bytes"],
                              record["obs_shape"])
                    for record in batch
                ]),
                device=device,
            )
            mask = torch.as_tensor(
                np.array([bits_to_mask(record["mask_bits"]) for record in batch]),
                device=device,
            )
            actions = torch.as_tensor(
                [record["action"] for record in batch], device=device
            ).unsqueeze(-1)
            hanchan_ids = torch.as_tensor(
                [hanchan_index[(record["seed"], record["seat"])] for record in batch],
                dtype=torch.long, device=device,
            )
            phi = brain(obs)
            q_out = dqn(phi, mask)
            logits = (q_out / temperature).masked_fill(~mask, -torch.inf)
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions).squeeze(-1)
            loss = pg_loss(
                log_probs, hanchan_ids, returns, len(hanchan_order), baseline=baseline
            )
            # Accumulate the gradient over micro-batches; the optimizer step
            # happens exactly once, after the whole batch has been consumed.
            # Summing per-chunk losses is exact because pg_loss divides by the
            # same N_hanchans in every chunk and is linear in the log-probs.
            loss.backward()
            n_micro += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_backward_seconds = time.perf_counter() - started
        grad_norms = {
            "params_with_grad": sum(
                1 for parameter in parameters if parameter.grad is not None
            ),
            "params_total": len(parameters),
            "grad_finite": all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters if parameter.grad is not None
            ),
            "grad_absmax": max(
                (float(parameter.grad.abs().max()) for parameter in parameters
                 if parameter.grad is not None),
                default=0.0,
            ),
            "grad_l2": math.sqrt(sum(
                float(parameter.grad.detach().pow(2).sum())
                for parameter in parameters if parameter.grad is not None
            )),
            "nonzero_grad_tensors": sum(
                1 for parameter in parameters
                if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            ),
        }
        bn_after = [
            (module.running_mean.detach().clone(),
             module.running_var.detach().clone(),
             int(module.num_batches_tracked))
            for module in brain.modules() if isinstance(module, torch.nn.BatchNorm1d)
        ]
        bn_unchanged = all(
            torch.equal(left[0], right[0])
            and torch.equal(left[1], right[1])
            and left[2] == right[2]
            for left, right in zip(bn_before, bn_after)
        )
        before_step = [
            parameter.detach().clone() for parameter in parameters
        ]
        step_started = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        changed = sum(
            1 for parameter, snapshot in zip(parameters, before_step)
            if not torch.equal(parameter.detach(), snapshot)
        )
        peak_vram = (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        )
        report["micro_batch_measurements"].append({
            "micro_batch": micro,
            "micro_batches": n_micro,
            "forward_backward_seconds": forward_backward_seconds,
            "seconds_per_decision": forward_backward_seconds / max(1, len(policy_records)),
            "optimizer_step_seconds": step_seconds,
            "peak_vram_bytes": peak_vram,
            "bn_running_stats_unchanged": bn_unchanged,
            "params_changed_by_step": changed,
            **grad_norms,
        })
        del brain, dqn, optimizer, parameters
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["gradient_path_ok"] = all(
        item["params_with_grad"] == item["params_total"]
        and item["grad_finite"]
        and item["nonzero_grad_tensors"] > 0
        and item["bn_running_stats_unchanged"]
        and item["params_changed_by_step"] > 0
        for item in report["micro_batch_measurements"]
    )
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P4-M9 on-policy sampling/update contract + resource probe"
    )
    parser.add_argument(
        "--challenger", default=None, help="LABEL=CHECKPOINT (default: student50k)"
    )
    parser.add_argument("--champion", default=None, help="LABEL=CHECKPOINT (default: ext_mortal)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8, help="seeds; each seed = 4 hanchans")
    parser.add_argument("--seed-start", type=int, default=710000)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--recompute-chunk", type=int, default=256)
    parser.add_argument("--bn-check-limit", type=int, default=512)
    parser.add_argument("--micro-batches", default="128,256,512")
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--skip-arena", action="store_true", help="reuse existing records")
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260912,
        help=(
            "seed for the process-global torch RNG that Categorical.sample() draws "
            "from; required for a reproducible collection run"
        ),
    )
    return parser.parse_args()


def _parse_model_spec(value: str | None, default_label: str, default_path: Path) -> tuple[str, Path]:
    if value is None:
        return default_label, default_path
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    challenger_label, challenger_path = _parse_model_spec(
        args.challenger,
        "student50k",
        Path("artifacts/experiments/student_policy_v1/student_formal_25k/student_step_050000.pth"),
    )
    champion_label, champion_path = _parse_model_spec(
        args.champion,
        "ext_mortal",
        Path("E:/AUbuntuProject/project/keqing1/artifacts/external_mortal_20240308_best_min.pth"),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    obs_path = output_dir / "obs_fp32.bin"
    records_path = output_dir / "probe_records.jsonl"

    version, conv_channels, num_blocks, challenger_state = _load_state(challenger_path)
    (
        champion_version,
        champion_conv_channels,
        champion_num_blocks,
        champion_state,
    ) = _load_state(champion_path)

    result: dict[str, Any] = {
        "schema": "keqing.mortal.p4m9_onpolicy_probe.v1",
        "created_at_unix": time.time(),
        "policy": PROBE_POLICY,
        "scope": (
            "contract verification + resource measurement only; no training, no candidate, "
            "no collect-update loop, no critic/PPO/BC"
        ),
        "challenger": {
            "label": challenger_label,
            "path": str(challenger_path),
            "sha256": _sha256_file(challenger_path),
            "version": version,
            "conv_channels": conv_channels,
            "num_blocks": num_blocks,
        },
        "champion": {
            "label": champion_label,
            "path": str(champion_path),
            "sha256": _sha256_file(champion_path),
        },
        "launch_environment": _launch_environment(),
        "seed_start": int(args.seed_start),
        "seed_key": int(args.seed_key),
        "seeds": int(args.seeds),
        "hanchans": int(args.seeds) * 4,
    }

    if args.skip_arena:
        if not records_path.exists():
            raise SystemExit("--skip-arena requires an existing probe_records.jsonl")
        records = [json.loads(line) for line in records_path.open(encoding="utf-8")]
        result["collection"] = {"reused_records": len(records)}
    else:
        MortalEngine, brain, dqn = _build_modules(
            challenger_state, mortal_root=args.mortal_root, device=device,
            version=version, conv_channels=conv_channels, num_blocks=num_blocks,
        )
        recorder = ProbeRecorder(obs_path)
        probe_class = ProbeEngine.build(MortalEngine, recorder=recorder, options=PROBE_POLICY)
        challenger_engine = probe_class(
            brain, dqn,
            is_oracle=False,
            version=version,
            device=device,
            stochastic_latent=False,
            enable_amp=False,
            enable_quick_eval=True,
            enable_rule_based_agari_guard=True,
            name=challenger_label,
            boltzmann_epsilon=float(PROBE_POLICY["boltzmann_epsilon"]),
            boltzmann_temp=float(PROBE_POLICY["boltzmann_temp"]),
            top_p=float(PROBE_POLICY["top_p"]),
        )
        _, champion_brain, champion_dqn = _build_modules(
            champion_state, mortal_root=args.mortal_root, device=device,
            version=champion_version, conv_channels=champion_conv_channels,
            num_blocks=champion_num_blocks,
        )
        champion_engine = MortalEngine(
            champion_brain, champion_dqn,
            is_oracle=False,
            version=champion_version,
            device=device,
            enable_quick_eval=True,
            enable_rule_based_agari_guard=True,
            name=champion_label,
        )
        del brain, dqn

        from libriichi.arena import OneVsThree

        env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        # The policy's Categorical.sample() consumes the process-global torch RNG
        # (the arena seeds only its own deal RNG), so the collection run is only
        # reproducible if this RNG is seeded explicitly.
        sampling_seed = int(args.sampling_seed)
        torch.manual_seed(sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed)
        result["policy_sampling_rng"] = {
            "source": "process-global torch RNG consumed by Categorical.sample()",
            "seed": sampling_seed,
            "seeded_before_collection": True,
            "note": (
                "without this seeding two otherwise identical collection runs produce "
                "different games, because the sampled actions differ"
            ),
        }
        collection_started = time.perf_counter()
        rankings = env.py_vs_py(
            challenger=challenger_engine,
            champion=champion_engine,
            seed_start=(int(args.seed_start), int(args.seed_key)),
            seed_count=int(args.seeds),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        collection_seconds = time.perf_counter() - collection_started
        recorder.close()
        records = recorder.records
        with records_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        result["collection"] = {
            "seconds": collection_seconds,
            "hanchans": int(args.seeds) * 4,
            "hanchans_per_second": (int(args.seeds) * 4) / max(collection_seconds, 1e-9),
            "rankings": [int(value) for value in rankings],
            "calls": len(recorder.batch_sizes),
            "batch_sizes": {
                "min": min(recorder.batch_sizes) if recorder.batch_sizes else 0,
                "max": max(recorder.batch_sizes) if recorder.batch_sizes else 0,
                "mean": (
                    sum(recorder.batch_sizes) / len(recorder.batch_sizes)
                    if recorder.batch_sizes else 0.0
                ),
            },
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
            ),
            "obs_write_seconds_inside_react_batch": recorder.write_seconds,
            "note": (
                "collection wall time includes writing the raw observation of every "
                "decision to disk inside react_batch"
            ),
        }
        del challenger_engine, champion_engine
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["sampling"] = collect_probe_records(records)
    obs_bytes = obs_path.stat().st_size
    record_bytes = records_path.stat().st_size
    hanchans = len({(record["seed"], record["seat"]) for record in records})
    decisions = result["sampling"]["decisions_recorded"]
    per_decision = obs_bytes / max(1, decisions)
    decisions_per_hanchan = decisions / max(1, hanchans)
    result["trajectory_cost"] = {
        "obs_fp32_bytes": obs_bytes,
        "metadata_jsonl_bytes": record_bytes,
        "bytes_per_decision_obs": per_decision,
        "metadata_bytes_per_decision": record_bytes / max(1, decisions),
        "decisions_per_hanchan": decisions_per_hanchan,
        "hanchans_measured": hanchans,
        "projected_obs_bytes_per_1024_hanchans": (
            per_decision * decisions_per_hanchan * 1024
        ),
        "projected_metadata_bytes_per_1024_hanchans": (
            (record_bytes / max(1, decisions)) * decisions_per_hanchan * 1024
        ),
        "observation_shape": records[0]["obs_shape"] if records else None,
        "note": (
            "observation is stored raw fp32; masks/actions/contexts/log-probs are "
            "metadata. The same observation is re-encodable from the arena log, which "
            "would trade this storage for a replay implementation."
        ),
    }

    result["check_sampling_vs_execution"] = check_sampling_vs_execution(
        log_dir=log_dir,
        records=records,
        challenger_label=challenger_label,
        seed_key=int(args.seed_key),
        seed_start=int(args.seed_start),
        seed_count=int(args.seeds),
    )

    update_engine_class, update_brain, update_dqn = _build_modules(
        challenger_state, mortal_root=args.mortal_root, device=device,
        version=version, conv_channels=conv_channels, num_blocks=num_blocks,
    )
    del update_engine_class

    result["check_logprob_recompute"] = check_logprob_recompute(
        obs_path=obs_path, records=records, brain=update_brain, dqn=update_dqn,
        device=device,
        temperature=float(PROBE_POLICY["boltzmann_temp"]),
        chunk=int(args.recompute_chunk),
    )
    result["check_bn_mode_sensitivity"] = check_bn_mode_sensitivity(
        obs_path=obs_path, records=records, brain=update_brain, dqn=update_dqn,
        device=device, temperature=float(PROBE_POLICY["boltzmann_temp"]),
        limit=int(args.bn_check_limit),
    )

    # The PG return is the frozen rank-point value of the challenger's own
    # terminal rank in that hanchan (same definition the arena evaluations use).
    rank_profile, rank_points = resolve_rank_points(
        rank_points=None, profile="tenhou_reference"
    )
    hanchan_returns: dict[tuple[int, int], float] = {}
    for item in result["check_sampling_vs_execution"]["per_hanchan"]:
        rank = item.get("challenger_rank")
        if rank is not None:
            hanchan_returns[(item["seed"], item["seat"])] = float(
                rank_points[int(rank) - 1]
            )
    result["return_definition"] = {
        "definition": "G_h = rank_points[rank_h - 1] for the challenger's rank in hanchan h",
        "rank_points_profile": rank_profile,
        "rank_points": [float(value) for value in rank_points],
        "baseline": float(args.baseline),
        "rank_source": "libriichi.stat.Stat.from_log (native, same as arena evaluations)",
        "placeholder_hanchan_return_used_for_records_only": (
            "hanchan_return() is a score-based helper kept for reference; it is NOT "
            "the return used by the probe loss"
        ),
    }
    result["check_gradient_path"] = check_gradient_path(
        obs_path=obs_path, records=records, state=challenger_state,
        mortal_root=args.mortal_root, device=device, version=version,
        conv_channels=conv_channels, num_blocks=num_blocks,
        temperature=float(PROBE_POLICY["boltzmann_temp"]),
        micro_batches=[int(item) for item in str(args.micro_batches).split(",") if item],
        hanchan_returns=hanchan_returns, baseline=float(args.baseline),
    )

    # The probe must not have touched the loaded challenger weights: the update
    # copy is a separate in-memory module (and the parent file is never written).
    result["parent_checkpoint_untouched"] = {
        "path": str(challenger_path),
        "sha256_after": _sha256_file(challenger_path),
        "sha256_unchanged": _sha256_file(challenger_path)
        == result["challenger"]["sha256"],
        "update_used_separate_in_memory_copy": True,
    }
    (output_dir / "probe_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "alignment_complete": result["check_sampling_vs_execution"]["alignment_complete"],
        "sampled_action_equals_executed_action": result["check_sampling_vs_execution"][
            "sampled_action_equals_executed_action"
        ],
        "logprob_recomputable": result["check_logprob_recompute"]["recomputable"],
        "gradient_path_ok": result["check_gradient_path"]["gradient_path_ok"],
        "collection_seconds": result["collection"].get("seconds"),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
