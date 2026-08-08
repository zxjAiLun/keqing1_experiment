from __future__ import annotations

import argparse
from collections import deque
import gc
import importlib
import json
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from mahjong_env.replay import (
    IllegalLabelActionError,
    build_replay_samples_mc_return,
    read_mjai_jsonl,
)
from mahjong_env.action_space import action_to_idx, build_legal_mask
from mahjong_env.event_history import compute_event_history
from training.state_features import encode
from keqing_core.cache_schema import (
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_OPPORTUNITY_DIM,
    KEQINGV4_RULE_CONTEXT_DIM,
)

_SUIT_PERMS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_SUITS = ("m", "p", "s")
_PAI_RE = re.compile(r'"([1-9])([mps])r?"')
_V3_CACHE_CLEAR_EVERY = 8
_WORKER_PROCESSED_FILES = 0
_WORKER_ENCODE_FNS: dict[str, object] = {}
_WORKER_ENCODE_TIMED_FNS: dict[str, object] = {}
_WORKER_CLEAR_CACHE_HOOKS: dict[str, object] = {}


class PreprocessBuildError(RuntimeError):
    pass


def _permute_mjson_text(text: str, perm: tuple) -> str:
    src_to_dst = {_SUITS[src]: _SUITS[dst] for dst, src in enumerate(perm)}

    def replace_pai(m: re.Match) -> str:
        num, suit = m.group(1), m.group(2)
        new_suit = src_to_dst[suit]
        full = m.group(0)
        if full.endswith('r"'):
            return f'"{num}{new_suit}r"'
        return f'"{num}{new_suit}"'

    return _PAI_RE.sub(replace_pai, text)


def _parse_mjai_jsonl_text(text: str) -> List[Dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _get_encode_fn(module_name: str):
    fn = _WORKER_ENCODE_FNS.get(module_name)
    if fn is None:
        fn = importlib.import_module(module_name).encode
        _WORKER_ENCODE_FNS[module_name] = fn
    return fn


def _get_encode_timed_fn(module_name: str):
    fn = _WORKER_ENCODE_TIMED_FNS.get(module_name)
    if fn is not None or module_name in _WORKER_ENCODE_TIMED_FNS:
        return fn
    try:
        fn = importlib.import_module(module_name).encode_with_timings
    except Exception:
        fn = None
    _WORKER_ENCODE_TIMED_FNS[module_name] = fn
    return fn


def _get_clear_progress_caches_hook(module_name: str):
    hook = _WORKER_CLEAR_CACHE_HOOKS.get(module_name)
    if hook is not None or module_name in _WORKER_CLEAR_CACHE_HOOKS:
        return hook
    if module_name != "training.state_features":
        _WORKER_CLEAR_CACHE_HOOKS[module_name] = None
        return None
    try:
        hook = importlib.import_module("mahjong_env.progress_oracle").clear_progress_caches
    except Exception:
        hook = None
    _WORKER_CLEAR_CACHE_HOOKS[module_name] = hook
    return hook


class BasePreprocessAdapter:
    extra_field_names: tuple[str, ...] = ()

    def init_rows(self) -> Dict[str, list]:
        return {}

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        del sample, action_idx
        return {}

    def permute_result_extras(self, result: Dict[str, np.ndarray], perm: tuple) -> Dict[str, np.ndarray]:
        del perm
        return {
            name: result[name]
            for name in self.extra_field_names
            if name in result
        }

    def finalize_result_extras(self, rows: Dict[str, list]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for key in self.extra_field_names:
            result[key] = np.array(rows[key], dtype=object)
        return result


class KeqingV4PreprocessAdapter(BasePreprocessAdapter):
    extra_field_names = (
        "score_delta_target",
        "win_target",
        "dealin_target",
        "pts_given_win_target",
        "pts_given_dealin_target",
        "opp_tenpai_target",
        "final_rank_target",
        "final_score_delta_points_target",
        "event_history",
        "v4_opportunity",
        "v4_discard_summary",
        "v4_call_summary",
        "v4_special_summary",
        "rule_context",
    )

    def init_rows(self) -> Dict[str, list]:
        return {
            "score_delta_target": [],
            "win_target": [],
            "dealin_target": [],
            "pts_given_win_target": [],
            "pts_given_dealin_target": [],
            "opp_tenpai_target": [],
            "final_rank_target": [],
            "final_score_delta_points_target": [],
            "event_history": [],
            "v4_opportunity": [],
            "v4_discard_summary": [],
            "v4_call_summary": [],
            "v4_special_summary": [],
            "rule_context": [],
        }

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        del action_idx
        from keqingv4.preprocess_features import build_typed_action_summaries

        if not getattr(sample, "events", None):
            raise PreprocessBuildError(
                "keqingv4 preprocess sample is missing normalized replay events required for event_history"
            )
        event_history = compute_event_history(sample.events, int(sample.event_index))
        if event_history.shape != (KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM):
            raise PreprocessBuildError(
                "keqingv4 preprocess event_history shape drift: "
                f"expected {(KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM)}, got {event_history.shape}"
            )

        discard_summary, call_summary, special_summary, v4_opportunity = build_typed_action_summaries(
            sample.state,
            sample.actor,
            sample.legal_actions,
        )
        return {
            "score_delta_target": float(np.clip(sample.score_delta_target, -1.0, 1.0)),
            "win_target": float(sample.win_target),
            "dealin_target": float(sample.dealin_target),
            "pts_given_win_target": float(getattr(sample, "pts_given_win_target", 0.0)),
            "pts_given_dealin_target": float(getattr(sample, "pts_given_dealin_target", 0.0)),
            "opp_tenpai_target": np.asarray(
                getattr(sample, "opp_tenpai_target", (0.0, 0.0, 0.0)),
                dtype=np.float32,
            ).reshape(3),
            "final_rank_target": int(getattr(sample, "final_rank_target", 0)),
            "final_score_delta_points_target": int(getattr(sample, "final_score_delta_points_target", 0)),
            "event_history": event_history,
            "v4_opportunity": np.asarray(v4_opportunity, dtype=np.uint8).reshape(KEQINGV4_OPPORTUNITY_DIM),
            "v4_discard_summary": discard_summary.astype(np.float16),
            "v4_call_summary": call_summary.astype(np.float16),
            "v4_special_summary": special_summary.astype(np.float16),
            "rule_context": np.zeros((KEQINGV4_RULE_CONTEXT_DIM,), dtype=np.float32),
        }

    def finalize_result_extras(self, rows: Dict[str, list]) -> Dict[str, np.ndarray]:
        return {
            "score_delta_target": np.array(rows["score_delta_target"], dtype=np.float16),
            "win_target": np.array(rows["win_target"], dtype=np.float16),
            "dealin_target": np.array(rows["dealin_target"], dtype=np.float16),
            "pts_given_win_target": np.array(rows["pts_given_win_target"], dtype=np.float32),
            "pts_given_dealin_target": np.array(rows["pts_given_dealin_target"], dtype=np.float32),
            "opp_tenpai_target": np.stack(rows["opp_tenpai_target"]).astype(np.float32),
            "final_rank_target": np.array(rows["final_rank_target"], dtype=np.int8),
            "final_score_delta_points_target": np.array(rows["final_score_delta_points_target"], dtype=np.int32),
            "event_history": np.stack(rows["event_history"]).astype(np.int16),
            "v4_opportunity": np.stack(rows["v4_opportunity"]).astype(np.uint8),
            "v4_discard_summary": np.stack(rows["v4_discard_summary"]).astype(np.float16),
            "v4_call_summary": np.stack(rows["v4_call_summary"]).astype(np.float16),
            "v4_special_summary": np.stack(rows["v4_special_summary"]).astype(np.float16),
            "rule_context": np.stack(rows["rule_context"]).astype(np.float32),
        }


class Xmodel2PreprocessAdapter(BasePreprocessAdapter):
    extra_field_names = (
        "score_delta_target",
        "win_target",
        "dealin_target",
        "pts_given_win_target",
        "pts_given_dealin_target",
        "opp_tenpai_target",
        "final_rank_target",
        "final_score_delta_points_target",
    )

    def init_rows(self) -> Dict[str, list]:
        return {
            "score_delta_target": [],
            "win_target": [],
            "dealin_target": [],
            "pts_given_win_target": [],
            "pts_given_dealin_target": [],
            "opp_tenpai_target": [],
            "final_rank_target": [],
            "final_score_delta_points_target": [],
        }

    def sample_extras(self, sample, action_idx: int) -> Dict[str, object]:
        del action_idx
        return {
            "score_delta_target": float(np.clip(sample.score_delta_target, -1.0, 1.0)),
            "win_target": float(sample.win_target),
            "dealin_target": float(sample.dealin_target),
            "pts_given_win_target": float(getattr(sample, "pts_given_win_target", 0.0)),
            "pts_given_dealin_target": float(getattr(sample, "pts_given_dealin_target", 0.0)),
            "opp_tenpai_target": np.asarray(
                getattr(sample, "opp_tenpai_target", (0.0, 0.0, 0.0)),
                dtype=np.float32,
            ).reshape(3),
            "final_rank_target": int(getattr(sample, "final_rank_target", 0)),
            "final_score_delta_points_target": int(getattr(sample, "final_score_delta_points_target", 0)),
        }

    def finalize_result_extras(self, rows: Dict[str, list]) -> Dict[str, np.ndarray]:
        return {
            "score_delta_target": np.array(rows["score_delta_target"], dtype=np.float32),
            "win_target": np.array(rows["win_target"], dtype=np.float32),
            "dealin_target": np.array(rows["dealin_target"], dtype=np.float32),
            "pts_given_win_target": np.array(rows["pts_given_win_target"], dtype=np.float32),
            "pts_given_dealin_target": np.array(rows["pts_given_dealin_target"], dtype=np.float32),
            "opp_tenpai_target": np.stack(rows["opp_tenpai_target"]).astype(np.float32),
            "final_rank_target": np.array(rows["final_rank_target"], dtype=np.int8),
            "final_score_delta_points_target": np.array(rows["final_score_delta_points_target"], dtype=np.int32),
        }


def create_preprocess_adapter(name: str) -> BasePreprocessAdapter:
    normalized = name.strip().lower()
    if normalized == "base":
        return BasePreprocessAdapter()
    if normalized == "keqingv4_aux":
        return KeqingV4PreprocessAdapter()
    if normalized == "xmodel2_aux":
        return Xmodel2PreprocessAdapter()
    raise ValueError(f"unknown preprocess adapter: {name}")


def events_to_cached_arrays(
    events,
    *,
    actor_name_filter=None,
    adapter: Optional[BasePreprocessAdapter] = None,
    value_strategy: str = "mc_return",
    encode_module: str = "training.state_features",
) -> Optional[Dict[str, np.ndarray]]:
    """Public helper for event-stream -> cache-array conversion.

    This is the shared entrypoint that selfplay and offline preprocessing should
    both use when exporting the base cache schema. Adapter-specific cache shapes
    still belong to training.preprocess / training.cached_dataset rather than
    ad hoc exporters in selfplay.
    """
    chosen_adapter = adapter or BasePreprocessAdapter()
    encode_fn = _get_encode_fn(encode_module)
    encode_timed_fn = _get_encode_timed_fn(encode_module)
    return _parse_events_to_arrays(
        events,
        actor_name_filter=actor_name_filter,
        adapter=chosen_adapter,
        value_strategy=value_strategy,
        encode_fn=encode_fn,
        encode_timed_fn=encode_timed_fn,
    )


def save_events_to_cache_file(
    out_path: Path,
    events,
    *,
    actor_name_filter=None,
    adapter: Optional[BasePreprocessAdapter] = None,
    adapter_name: str = "base",
    value_strategy: str = "mc_return",
    encode_module: str = "training.state_features",
    compress_output: bool = True,
) -> bool:
    chosen_adapter = adapter or create_preprocess_adapter(adapter_name)
    try:
        result = events_to_cached_arrays(
            events,
            actor_name_filter=actor_name_filter,
            adapter=chosen_adapter,
            value_strategy=value_strategy,
            encode_module=encode_module,
        )
    except PreprocessBuildError:
        return False
    if result is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress_output:
        np.savez_compressed(out_path, **result)
    else:
        np.savez(out_path, **result)
    return True


def _parse_events_to_arrays(
    events,
    *,
    actor_name_filter=None,
    adapter: BasePreprocessAdapter,
    value_strategy: str = "mc_return",
    encode_fn=encode,
    encode_timed_fn=None,
) -> Optional[Dict[str, np.ndarray]]:
    rows = {
        "tile_feat": [],
        "scalar": [],
        "mask": [],
        "action_idx": [],
        "value": [],
    }
    rows.update(adapter.init_rows())

    timings = {
        "sample_build_s": 0.0,
        "encode_s": 0.0,
        "tracker_s": 0.0,
        "progress_s": 0.0,
        "fill_s": 0.0,
        "standard_shanten_s": 0.0,
        "waits_s": 0.0,
        "ukeire_improvement_s": 0.0,
        "shape_s": 0.0,
        "standard_shanten_calls": 0.0,
        "waits_calls": 0.0,
        "standard_shanten_hits": 0.0,
        "standard_shanten_misses": 0.0,
        "waits_hits": 0.0,
        "waits_misses": 0.0,
    }

    try:
        t_build0 = time.perf_counter()
        if value_strategy != "mc_return":
            raise PreprocessBuildError(f"unsupported value_strategy: {value_strategy}")
        samples = build_replay_samples_mc_return(
            events,
            actor_name_filter=actor_name_filter,
        )
        timings["sample_build_s"] = time.perf_counter() - t_build0
    except IllegalLabelActionError:
        raise
    except Exception as exc:
        raise PreprocessBuildError(str(exc)) from exc

    t_encode0 = time.perf_counter()
    for s in samples:
        try:
            if encode_timed_fn is not None:
                tile_feat, scalar, encode_timings = encode_timed_fn(s.state, s.actor)
                if isinstance(encode_timings, dict):
                    timings["tracker_s"] += float(encode_timings.get("tracker_s", 0.0))
                    timings["progress_s"] += float(encode_timings.get("progress_s", 0.0))
                    timings["fill_s"] += float(encode_timings.get("fill_s", 0.0))
                    timings["standard_shanten_s"] += float(encode_timings.get("standard_shanten_s", 0.0))
                    timings["waits_s"] += float(encode_timings.get("waits_s", 0.0))
                    timings["ukeire_improvement_s"] += float(encode_timings.get("ukeire_improvement_s", 0.0))
                    timings["shape_s"] += float(encode_timings.get("shape_s", 0.0))
                    timings["standard_shanten_calls"] += float(encode_timings.get("standard_shanten_calls", 0.0))
                    timings["waits_calls"] += float(encode_timings.get("waits_calls", 0.0))
                    timings["standard_shanten_hits"] += float(encode_timings.get("standard_shanten_hits", 0.0))
                    timings["standard_shanten_misses"] += float(encode_timings.get("standard_shanten_misses", 0.0))
                    timings["waits_hits"] += float(encode_timings.get("waits_hits", 0.0))
                    timings["waits_misses"] += float(encode_timings.get("waits_misses", 0.0))
            else:
                tile_feat, scalar = encode_fn(s.state, s.actor)
            mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
            action_idx = action_to_idx(s.label_action)
            if mask[action_idx] == 0:
                continue
            value = float(np.clip(s.value_target, -1.0, 1.0))
            rows["tile_feat"].append(tile_feat)
            rows["scalar"].append(scalar)
            rows["mask"].append(mask)
            rows["action_idx"].append(action_idx)
            rows["value"].append(value)
            for key, val in adapter.sample_extras(s, action_idx).items():
                rows[key].append(val)
        except Exception:
            continue
    timings["encode_s"] = time.perf_counter() - t_encode0

    if not rows["tile_feat"]:
        return None

    result: Dict[str, np.ndarray] = {
        "tile_feat": np.stack(rows["tile_feat"]).astype(np.float16),
        "scalar": np.stack(rows["scalar"]).astype(np.float16),
        "mask": np.stack(rows["mask"]).astype(np.uint8),
        "action_idx": np.array(rows["action_idx"], dtype=np.int16),
        "value": np.array(rows["value"], dtype=np.float16),
    }
    result.update(adapter.finalize_result_extras(rows))
    result["_timings"] = timings
    return result


def _concat_results(results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(results[0].keys())
    return {key: np.concatenate([r[key] for r in results], axis=0) for key in keys}


def process_file(args: Tuple) -> Tuple[str, int, Dict[str, float], str, str]:
    global _WORKER_PROCESSED_FILES
    src_path, out_path, augment, actor_name_filter, adapter, value_strategy, encode_module, compress_output = args
    clear_progress_caches = _get_clear_progress_caches_hook(encode_module)
    timings = {
        "read_s": 0.0,
        "base_s": 0.0,
        "sample_build_s": 0.0,
        "encode_s": 0.0,
        "tracker_s": 0.0,
        "progress_s": 0.0,
        "fill_s": 0.0,
        "standard_shanten_s": 0.0,
        "waits_s": 0.0,
        "ukeire_improvement_s": 0.0,
        "shape_s": 0.0,
        "standard_shanten_calls": 0.0,
        "waits_calls": 0.0,
        "standard_shanten_hits": 0.0,
        "standard_shanten_misses": 0.0,
        "waits_hits": 0.0,
        "waits_misses": 0.0,
        "augment_s": 0.0,
        "save_s": 0.0,
        "total_s": 0.0,
    }
    t0 = time.perf_counter()
    if out_path.exists():
        timings["total_s"] = time.perf_counter() - t0
        return ("skip", 0, timings, str(src_path), "")
    try:
        t_read0 = time.perf_counter()
        text = src_path.read_text(encoding="utf-8")
        events = _parse_mjai_jsonl_text(text)
        timings["read_s"] = time.perf_counter() - t_read0
    except Exception:
        timings["total_s"] = time.perf_counter() - t0
        return ("error", 0, timings, str(src_path), "read_text_or_parse_failed")

    all_results: List[Dict[str, np.ndarray]] = []
    t_base0 = time.perf_counter()
    try:
        result = events_to_cached_arrays(
            events,
            actor_name_filter=actor_name_filter,
            adapter=adapter,
            value_strategy=value_strategy,
            encode_module=encode_module,
        )
    except IllegalLabelActionError as exc:
        timings["base_s"] = time.perf_counter() - t_base0
        timings["total_s"] = time.perf_counter() - t0
        return ("error", 0, timings, str(src_path), f"IllegalLabelActionError: {exc}")
    except PreprocessBuildError:
        timings["base_s"] = time.perf_counter() - t_base0
        timings["total_s"] = time.perf_counter() - t0
        return ("error", 0, timings, str(src_path), "PreprocessBuildError")
    if result is not None:
        extra_timings = result.pop("_timings", None)
        if isinstance(extra_timings, dict):
            timings["sample_build_s"] += float(extra_timings.get("sample_build_s", 0.0))
            timings["encode_s"] += float(extra_timings.get("encode_s", 0.0))
            timings["tracker_s"] += float(extra_timings.get("tracker_s", 0.0))
            timings["progress_s"] += float(extra_timings.get("progress_s", 0.0))
            timings["fill_s"] += float(extra_timings.get("fill_s", 0.0))
            timings["standard_shanten_s"] += float(extra_timings.get("standard_shanten_s", 0.0))
            timings["waits_s"] += float(extra_timings.get("waits_s", 0.0))
            timings["ukeire_improvement_s"] += float(extra_timings.get("ukeire_improvement_s", 0.0))
            timings["shape_s"] += float(extra_timings.get("shape_s", 0.0))
            timings["standard_shanten_calls"] += float(extra_timings.get("standard_shanten_calls", 0.0))
            timings["waits_calls"] += float(extra_timings.get("waits_calls", 0.0))
            timings["standard_shanten_hits"] += float(extra_timings.get("standard_shanten_hits", 0.0))
            timings["standard_shanten_misses"] += float(extra_timings.get("standard_shanten_misses", 0.0))
            timings["waits_hits"] += float(extra_timings.get("waits_hits", 0.0))
            timings["waits_misses"] += float(extra_timings.get("waits_misses", 0.0))
    timings["base_s"] = time.perf_counter() - t_base0
    if result is not None:
        all_results.append(result)

    if augment:
        t_aug0 = time.perf_counter()
        for perm in _SUIT_PERMS[1:]:
            permuted_text = _permute_mjson_text(text, perm)
            try:
                perm_events = _parse_mjai_jsonl_text(permuted_text)
                result = _parse_events_to_arrays(
                    perm_events,
                    actor_name_filter=actor_name_filter,
                    adapter=adapter,
                    value_strategy=value_strategy,
                    encode_fn=_get_encode_fn(encode_module),
                )
                if result is not None:
                    extra_timings = result.pop("_timings", None)
                    if isinstance(extra_timings, dict):
                        timings["sample_build_s"] += float(extra_timings.get("sample_build_s", 0.0))
                        timings["encode_s"] += float(extra_timings.get("encode_s", 0.0))
                        timings["tracker_s"] += float(extra_timings.get("tracker_s", 0.0))
                        timings["progress_s"] += float(extra_timings.get("progress_s", 0.0))
                        timings["fill_s"] += float(extra_timings.get("fill_s", 0.0))
                        timings["standard_shanten_s"] += float(extra_timings.get("standard_shanten_s", 0.0))
                        timings["waits_s"] += float(extra_timings.get("waits_s", 0.0))
                        timings["ukeire_improvement_s"] += float(extra_timings.get("ukeire_improvement_s", 0.0))
                        timings["shape_s"] += float(extra_timings.get("shape_s", 0.0))
                        timings["standard_shanten_calls"] += float(extra_timings.get("standard_shanten_calls", 0.0))
                        timings["waits_calls"] += float(extra_timings.get("waits_calls", 0.0))
                        timings["standard_shanten_hits"] += float(extra_timings.get("standard_shanten_hits", 0.0))
                        timings["standard_shanten_misses"] += float(extra_timings.get("standard_shanten_misses", 0.0))
                        timings["waits_hits"] += float(extra_timings.get("waits_hits", 0.0))
                        timings["waits_misses"] += float(extra_timings.get("waits_misses", 0.0))
                    result.update(adapter.permute_result_extras(result, perm))
                    all_results.append(result)
            except PreprocessBuildError:
                continue
            except Exception:
                pass
        timings["augment_s"] = time.perf_counter() - t_aug0

    if not all_results:
        status = ("empty", 0)
    else:
        merged = _concat_results(all_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t_save0 = time.perf_counter()
        if compress_output:
            np.savez_compressed(out_path, **merged)
        else:
            np.savez(out_path, **merged)
        timings["save_s"] = time.perf_counter() - t_save0
        status = ("ok", len(merged["tile_feat"]))

    _WORKER_PROCESSED_FILES += 1
    if (
        clear_progress_caches is not None
        and _WORKER_PROCESSED_FILES % _V3_CACHE_CLEAR_EVERY == 0
    ):
        clear_progress_caches()
        gc.collect()
    timings["total_s"] = time.perf_counter() - t0
    return status[0], status[1], timings, str(src_path), ""


def run_preprocess(
    *,
    default_output_dir: str,
    adapter: BasePreprocessAdapter,
    default_value_strategy: str = "mc_return",
    encode_module: str = "training.state_features",
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--maxtasksperchild", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--eta-window", type=int, default=None)
    parser.add_argument("--actor_name_filter", nargs="+", type=str, default=None)
    parser.add_argument("--compress-output", dest="compress_output", action="store_true")
    parser.add_argument("--no-compress-output", dest="compress_output", action="store_false")
    parser.set_defaults(compress_output=None)
    parser.add_argument(
        "--value-strategy",
        type=str,
        default=None,
        choices=["mc_return"],
    )
    args = parser.parse_args()

    cfg: dict = {}
    data_dirs: List[str] = []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        data_dirs = cfg.get("data_dirs", [])
    if args.data_dirs:
        data_dirs = args.data_dirs

    if not data_dirs:
        print("错误：请通过 --data_dirs 或 --config 指定数据目录")
        sys.exit(1)

    output_dir_value = args.output_dir or cfg.get("output_dir", default_output_dir)
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else int(cfg.get("workers", max(1, cpu_count() - 2)))
    maxtasksperchild = (
        args.maxtasksperchild
        if args.maxtasksperchild is not None
        else cfg.get("maxtasksperchild", 64)
    )
    augment = args.augment if args.augment is not None else bool(cfg.get("augment", False))
    progress_every = (
        args.progress_every
        if args.progress_every is not None
        else int(cfg.get("progress_every", 20))
    )
    eta_window = (
        args.eta_window
        if args.eta_window is not None
        else int(cfg.get("eta_window", 200))
    )
    compress_output = (
        args.compress_output
        if args.compress_output is not None
        else bool(cfg.get("compress_output", True))
    )
    actor_name_filter_raw = args.actor_name_filter if args.actor_name_filter is not None else cfg.get("actor_name_filter")
    actor_name_filter = set(actor_name_filter_raw) if actor_name_filter_raw else None
    value_strategy = args.value_strategy or cfg.get("value_strategy", default_value_strategy)

    tasks = []
    for data_dir in data_dirs:
        src_dir = Path(data_dir)
        ds_name = src_dir.name
        out_ds_dir = output_dir / ds_name
        for mjson_file in sorted(src_dir.glob("*.mjson")):
            out_file = out_ds_dir / (mjson_file.stem + ".npz")
            tasks.append((
                mjson_file,
                out_file,
                augment,
                actor_name_filter,
                adapter,
                value_strategy,
                encode_module,
                compress_output,
            ))

    total = len(tasks)
    filter_info = f"，actor_filter={list(actor_name_filter)}" if actor_name_filter else ""
    print(
        f"共 {total} 个文件，使用 {workers} 个进程，augment={augment} "
        f"value_strategy={value_strategy} maxtasksperchild={maxtasksperchild} "
        f"compress_output={compress_output}{filter_info}"
    )

    done = skipped = errors = empty = total_samples = 0
    wall_t0 = time.perf_counter()
    recent_files: deque[tuple[bool, float]] = deque(maxlen=max(1, eta_window))
    failure_log_path = output_dir / "preprocess_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()
    timing_sums = {
        "read_s": 0.0,
        "base_s": 0.0,
        "sample_build_s": 0.0,
        "encode_s": 0.0,
        "tracker_s": 0.0,
        "progress_s": 0.0,
        "fill_s": 0.0,
        "standard_shanten_s": 0.0,
        "waits_s": 0.0,
        "ukeire_improvement_s": 0.0,
        "shape_s": 0.0,
        "standard_shanten_calls": 0.0,
        "waits_calls": 0.0,
        "standard_shanten_hits": 0.0,
        "standard_shanten_misses": 0.0,
        "waits_hits": 0.0,
        "waits_misses": 0.0,
        "augment_s": 0.0,
        "save_s": 0.0,
        "total_s": 0.0,
    }
    with Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        for status, n_samples, timings, src_path_str, detail in pool.imap_unordered(process_file, tasks, chunksize=4):
            done += 1
            recent_files.append((status != "skip", float(timings.get("total_s", 0.0))))
            for key in timing_sums:
                timing_sums[key] += float(timings.get(key, 0.0))
            if status == "ok":
                total_samples += n_samples
            elif status == "skip":
                skipped += 1
            elif status == "empty":
                empty += 1
                with failure_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"status": status, "path": src_path_str, "detail": detail}, ensure_ascii=False) + "\n")
            elif status == "error":
                errors += 1
                with failure_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"status": status, "path": src_path_str, "detail": detail}, ensure_ascii=False) + "\n")
            if done % progress_every == 0 or done == total:
                elapsed_s = time.perf_counter() - wall_t0
                remaining_files = max(0, total - done)
                recent_work = [cost for worked, cost in recent_files if worked]
                recent_work_ratio = (
                    sum(1 for worked, _cost in recent_files if worked) / max(1, len(recent_files))
                )
                recent_avg_work_s = (
                    sum(recent_work) / max(1, len(recent_work))
                )
                eta_s = remaining_files * recent_work_ratio * recent_avg_work_s
                print(
                    f"  [{done}/{total}] 跳过={skipped} 空={empty} 错误={errors} 样本={total_samples:,} "
                    f"| 已运行={elapsed_s:.0f}s"
                    f" 预计剩余={eta_s:.0f}s"
                )

    print(
        f"\n完成！总样本数: {total_samples:,}，跳过: {skipped}，空: {empty}，错误: {errors}"
    )
    if empty or errors:
        print(f"失败明细已写入: {failure_log_path}")


__all__ = [
    "BasePreprocessAdapter",
    "create_preprocess_adapter",
    "KeqingV4PreprocessAdapter",
    "Xmodel2PreprocessAdapter",
    "events_to_cached_arrays",
    "save_events_to_cache_file",
    "run_preprocess",
]
