"""Optional Rust acceleration for Keqing core helpers.

This package prefers a package-local ``keqing_core._native`` extension.
In source-development mode the Python package under ``src/`` shadows the
installed wheel package, so we also probe installed site-packages for the
compiled submodule and load it explicitly when present.
"""

from __future__ import annotations

from importlib import machinery as _importlib_machinery
from importlib import util as _importlib_util
from pathlib import Path as _Path
import numpy as _np
import json as _json
import site as _site
import sys as _sys
import warnings as _warnings

from riichienv import calculate_shanten as _riichienv_shanten

from .cache_schema import (
    KEQINGV4_SUMMARY_DIM,
    XMODEL1_SCHEMA_NAME as _PY_XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION as _PY_XMODEL1_SCHEMA_VERSION,
)

_USE_RUST = False
_RUST_AVAILABLE = False
_RUST_FUNC = None
_RUST_SHANTEN_NORMAL = None
_RUST_SHANTEN_ALL = None
_RUST_SHANTEN_MANY = None
_RUST_REQUIRED_TILES = None
_RUST_DRAW_DELTAS = None
_RUST_DISCARD_DELTAS = None
_RUST_BUILD_136_POOL_ENTRIES = None
_RUST_SUMMARIZE_3N1 = None
_RUST_SUMMARIZE_ONE_SHANTEN_DRAW_METRICS = None
_RUST_SUMMARIZE_3N2_CANDIDATES = None
_RUST_SUMMARIZE_BEST_3N2_CANDIDATE = None
_RUST_BUILD_XMODEL1_DISCARD_RECORDS = None
_RUST_BUILD_KEQINGV4_CACHED_RECORDS = None
_RUST_BUILD_REPLAY_DECISION_RECORDS_MC_RETURN_JSON = None
_RUST_NATIVE_SCHEMA_INFO_JSON = None
_RUST_ACTION_IDENTITY_JSON = None
_RUST_DECODE_ACTION_ID_JSON = None
_RUST_MJAI_EVENTS_FOR_ACTION_JSON = None
_RUST_RESOLVE_TERMINAL_ACTION_JSON = None
_RUST_XMODEL1_SCHEMA_INFO = None
_RUST_VALIDATE_XMODEL1_DISCARD_RECORD = None
_RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON = None
_RUST_REPLAY_STATE_SNAPSHOT_JSON = None
_RUST_VALIDATE_REPLAY_STATE_SNAPSHOT_JSON = None
_RUST_BUILD_KEQINGRL_ACTION_FEATURES = None
_RUST_BUILD_KEQINGRL_ACTION_FEATURES_TYPED = None
_RUST_KEQINGRL_ACTION_FEATURE_DIM = None
_RUST_FIXED_SEED_EVAL_GATE_JSON = None
_RUST_ENUMERATE_LEGAL_ACTION_SPECS_STRUCTURAL_JSON = None
_RUST_ENUMERATE_PUBLIC_LEGAL_ACTION_SPECS_JSON = None
_RUST_ENUMERATE_HORA_CANDIDATES_JSON = None
_RUST_CAN_HORA_SHAPE_FROM_SNAPSHOT_JSON = None
_RUST_PREPARE_HORA_EVALUATION_FROM_SNAPSHOT_JSON = None
_RUST_COMPUTE_HORA_DELTAS_JSON = None
_RUST_PREPARE_HORA_TILE_ALLOCATION_JSON = None
_RUST_BUILD_HORA_RESULT_PAYLOAD_JSON = None
_RUST_EVALUATE_HORA_FROM_PREPARED_JSON = None
_RUST_EVALUATE_HORA_TRUTH_FROM_PREPARED_JSON = None
_RUST_BUILD_KEQINGV4_DISCARD_SUMMARY_JSON = None
_RUST_BUILD_KEQINGV4_CALL_SUMMARY_JSON = None
_RUST_BUILD_KEQINGV4_SPECIAL_SUMMARY_JSON = None
_RUST_BUILD_KEQINGV4_TYPED_SUMMARIES_JSON = None
_RUST_BUILD_KEQINGV4_CONTINUATION_SCENARIOS_JSON = None
_RUST_SCORE_KEQINGV4_CONTINUATION_SCENARIO_JSON = None
_RUST_AGGREGATE_KEQINGV4_CONTINUATION_SCORES_JSON = None
_RUST_PROJECT_KEQINGV4_CALL_SNAPSHOT_JSON = None
_RUST_PROJECT_KEQINGV4_DISCARD_SNAPSHOT_JSON = None
_RUST_PROJECT_KEQINGV4_RINSHAN_DRAW_SNAPSHOT_JSON = None
_RUST_ENUMERATE_KEQINGV4_POST_MELD_DISCARDS_JSON = None
_RUST_ENUMERATE_KEQINGV4_LIVE_DRAW_WEIGHTS_JSON = None
_RUST_ENUMERATE_KEQINGV4_REACH_DISCARDS_JSON = None
_RUST_PROJECT_KEQINGV4_REACH_SNAPSHOT_JSON = None
_RUST_CHOOSE_RULEBASE_ACTION_JSON = None
_RUST_SCORE_RULEBASE_ACTIONS_JSON = None
_RUST_IMPORT_ERROR = None
_RUST_XMODEL1_SCHEMA_MISMATCH = None


def _json_default(value):
    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, (_np.integer, _np.floating)):
        return value.item()
    if isinstance(value, _Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _is_native_extension(path: _Path) -> bool:
    if not path.is_file():
        return False
    return any(str(path).endswith(suffix) for suffix in _importlib_machinery.EXTENSION_SUFFIXES)


def _candidate_native_paths() -> list[_Path]:
    package_dir = _Path(__file__).resolve().parent
    candidates = sorted(
        path for path in package_dir.glob("_native*") if _is_native_extension(path)
    )
    loaded_ext = globals().get("_rust_ext")
    loaded_path = getattr(loaded_ext, "__file__", None) if loaded_ext is not None else None
    if loaded_path is not None:
        candidate_path = _Path(loaded_path)
        if _is_native_extension(candidate_path):
            candidates.append(candidate_path)
    search_roots = []
    try:
        search_roots.extend(_site.getsitepackages())
    except AttributeError:
        pass
    try:
        search_roots.append(_site.getusersitepackages())
    except AttributeError:
        pass
    for root in search_roots:
        package_root = _Path(root) / "keqing_core"
        try:
            if not package_root.exists():
                continue
            discovered = sorted(
                path
                for path in package_root.glob("_native*")
                if _is_native_extension(path)
            )
        except OSError:
            # A locked or inaccessible user-site package (stat 或 glob 失败)
            # 都不能阻止项目本地 native runtime 被发现。
            continue
        candidates.extend(discovered)
    deduped: list[_Path] = []
    seen: set[_Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_native_module():
    module_name = f"{__name__}._native"
    first_error = None
    for candidate in _candidate_native_paths():
        spec = _importlib_util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = _importlib_util.module_from_spec(spec)
        try:
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except ImportError as exc:
            _sys.modules.pop(module_name, None)
            if first_error is None:
                first_error = exc
            continue
        setattr(_sys.modules[__name__], "_native", module)
        return module, None

    try:
        import importlib as _importlib
        fallback = _importlib.import_module("_native")
        inner = getattr(fallback, "_native", None)
        if inner is not None and hasattr(inner, "counts34_to_ids_py"):
            _sys.modules[module_name] = inner
            setattr(_sys.modules[__name__], "_native", inner)
            return inner, None
        if hasattr(fallback, "counts34_to_ids_py"):
            _sys.modules[module_name] = fallback
            setattr(_sys.modules[__name__], "_native", fallback)
            return fallback, None
    except ImportError as exc:
        if first_error is None:
            first_error = exc

    return None, first_error


def _disable_stale_xmodel1_native_if_needed() -> None:
    global _RUST_BUILD_XMODEL1_DISCARD_RECORDS
    global _RUST_XMODEL1_SCHEMA_INFO
    global _RUST_VALIDATE_XMODEL1_DISCARD_RECORD
    global _RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON
    global _RUST_XMODEL1_SCHEMA_MISMATCH
    global _RUST_VALIDATE_REPLAY_STATE_SNAPSHOT_JSON

    if (
        _PY_XMODEL1_SCHEMA_NAME is None
        or _PY_XMODEL1_SCHEMA_VERSION is None
        or _RUST_XMODEL1_SCHEMA_INFO is None
    ):
        return
    try:
        native_name, native_version, *_ = _RUST_XMODEL1_SCHEMA_INFO()
    except Exception as exc:
        _warnings.warn(
            "keqing_core native extension exposed xmodel1 schema metadata but it "
            f"could not be queried cleanly: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    native_name = str(native_name)
    native_version = int(native_version)
    if (
        native_name == _PY_XMODEL1_SCHEMA_NAME
        and native_version == _PY_XMODEL1_SCHEMA_VERSION
    ):
        return
    native_path = getattr(_rust_ext, "__file__", "<unknown>")
    _RUST_XMODEL1_SCHEMA_MISMATCH = (
        "Rust Xmodel1 native capability is unavailable because the loaded "
        f"keqing_core extension at {native_path} exposes {native_name}@{native_version}, "
        f"but the source tree requires {_PY_XMODEL1_SCHEMA_NAME}@{_PY_XMODEL1_SCHEMA_VERSION}. "
        "Rebuild and reinstall rust/keqing_core before running xmodel1 preprocess "
        "or runtime export."
    )
    _warnings.warn(_RUST_XMODEL1_SCHEMA_MISMATCH, RuntimeWarning, stacklevel=2)
    _RUST_BUILD_XMODEL1_DISCARD_RECORDS = None
    _RUST_XMODEL1_SCHEMA_INFO = None
    _RUST_VALIDATE_XMODEL1_DISCARD_RECORD = None
    _RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON = None


_rust_ext, _RUST_IMPORT_ERROR = _load_native_module()
if _rust_ext is not None and hasattr(_rust_ext, "counts34_to_ids_py"):
    _RUST_AVAILABLE = True
    _RUST_FUNC = _rust_ext.counts34_to_ids_py
    _RUST_SHANTEN_NORMAL = getattr(_rust_ext, "calc_shanten_normal_py", None)
    _RUST_SHANTEN_ALL = getattr(_rust_ext, "calc_shanten_all_py", None)
    _RUST_SHANTEN_MANY = getattr(_rust_ext, "standard_shanten_many_py", None)
    _RUST_REQUIRED_TILES = getattr(_rust_ext, "calc_required_tiles_py", None)
    _RUST_DRAW_DELTAS = getattr(_rust_ext, "calc_draw_deltas_py", None)
    _RUST_DISCARD_DELTAS = getattr(_rust_ext, "calc_discard_deltas_py", None)
    _RUST_BUILD_136_POOL_ENTRIES = getattr(_rust_ext, "build_136_pool_entries_py", None)
    _RUST_SUMMARIZE_3N1 = getattr(_rust_ext, "summarize_3n1_py", None)
    _RUST_SUMMARIZE_ONE_SHANTEN_DRAW_METRICS = getattr(
        _rust_ext, "summarize_one_shanten_draw_metrics_py", None
    )
    _RUST_SUMMARIZE_3N2_CANDIDATES = getattr(_rust_ext, "summarize_3n2_candidates_py", None)
    _RUST_SUMMARIZE_BEST_3N2_CANDIDATE = getattr(_rust_ext, "summarize_best_3n2_candidate_py", None)
    _RUST_BUILD_XMODEL1_DISCARD_RECORDS = getattr(_rust_ext, "build_xmodel1_discard_records_py", None)
    _RUST_BUILD_KEQINGV4_CACHED_RECORDS = getattr(_rust_ext, "build_keqingv4_cached_records_py", None)
    _RUST_BUILD_REPLAY_DECISION_RECORDS_MC_RETURN_JSON = getattr(
        _rust_ext, "build_replay_decision_records_mc_return_json_py", None
    )
    _RUST_NATIVE_SCHEMA_INFO_JSON = getattr(_rust_ext, "native_schema_info_json_py", None)
    _RUST_ACTION_IDENTITY_JSON = getattr(_rust_ext, "action_identity_json_py", None)
    _RUST_DECODE_ACTION_ID_JSON = getattr(_rust_ext, "decode_action_id_json_py", None)
    _RUST_MJAI_EVENTS_FOR_ACTION_JSON = getattr(_rust_ext, "mjai_events_for_action_json_py", None)
    _RUST_RESOLVE_TERMINAL_ACTION_JSON = getattr(_rust_ext, "resolve_terminal_action_json_py", None)
    _RUST_XMODEL1_SCHEMA_INFO = getattr(_rust_ext, "xmodel1_schema_info_py", None)
    _RUST_VALIDATE_XMODEL1_DISCARD_RECORD = getattr(_rust_ext, "validate_xmodel1_discard_record_py", None)
    _RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON = getattr(
        _rust_ext, "build_xmodel1_runtime_tensors_json_py", None
    )
    _RUST_REPLAY_STATE_SNAPSHOT_JSON = getattr(_rust_ext, "replay_state_snapshot_json_py", None)
    _RUST_VALIDATE_REPLAY_STATE_SNAPSHOT_JSON = getattr(_rust_ext, "validate_replay_state_snapshot_json_py", None)
    _RUST_BUILD_KEQINGRL_ACTION_FEATURES = getattr(_rust_ext, "build_keqingrl_action_features_py", None)
    _RUST_BUILD_KEQINGRL_ACTION_FEATURES_TYPED = getattr(_rust_ext, "build_keqingrl_action_features_typed_py", None)
    _RUST_KEQINGRL_ACTION_FEATURE_DIM = getattr(_rust_ext, "keqingrl_action_feature_dim_py", None)
    _RUST_FIXED_SEED_EVAL_GATE_JSON = getattr(_rust_ext, "fixed_seed_eval_gate_json_py", None)
    _RUST_ENUMERATE_LEGAL_ACTION_SPECS_STRUCTURAL_JSON = getattr(
        _rust_ext, "enumerate_legal_action_specs_structural_json_py", None
    )
    _RUST_ENUMERATE_PUBLIC_LEGAL_ACTION_SPECS_JSON = getattr(
        _rust_ext, "enumerate_public_legal_action_specs_json_py", None
    )
    _RUST_ENUMERATE_HORA_CANDIDATES_JSON = getattr(
        _rust_ext, "enumerate_hora_candidates_json_py", None
    )
    _RUST_CAN_HORA_SHAPE_FROM_SNAPSHOT_JSON = getattr(
        _rust_ext, "can_hora_shape_from_snapshot_json_py", None
    )
    _RUST_PREPARE_HORA_EVALUATION_FROM_SNAPSHOT_JSON = getattr(
        _rust_ext, "prepare_hora_evaluation_from_snapshot_json_py", None
    )
    _RUST_COMPUTE_HORA_DELTAS_JSON = getattr(
        _rust_ext, "compute_hora_deltas_json_py", None
    )
    _RUST_PREPARE_HORA_TILE_ALLOCATION_JSON = getattr(
        _rust_ext, "prepare_hora_tile_allocation_json_py", None
    )
    _RUST_BUILD_HORA_RESULT_PAYLOAD_JSON = getattr(
        _rust_ext, "build_hora_result_payload_json_py", None
    )
    _RUST_EVALUATE_HORA_FROM_PREPARED_JSON = getattr(
        _rust_ext, "evaluate_hora_from_prepared_json_py", None
    )
    _RUST_EVALUATE_HORA_TRUTH_FROM_PREPARED_JSON = getattr(
        _rust_ext, "evaluate_hora_truth_from_prepared_json_py", None
    )
    _RUST_BUILD_KEQINGV4_DISCARD_SUMMARY_JSON = getattr(
        _rust_ext, "build_keqingv4_discard_summary_json_py", None
    )
    _RUST_BUILD_KEQINGV4_CALL_SUMMARY_JSON = getattr(
        _rust_ext, "build_keqingv4_call_summary_json_py", None
    )
    _RUST_BUILD_KEQINGV4_SPECIAL_SUMMARY_JSON = getattr(
        _rust_ext, "build_keqingv4_special_summary_json_py", None
    )
    _RUST_BUILD_KEQINGV4_TYPED_SUMMARIES_JSON = getattr(
        _rust_ext, "build_keqingv4_typed_summaries_json_py", None
    )
    _RUST_BUILD_KEQINGV4_CONTINUATION_SCENARIOS_JSON = getattr(
        _rust_ext, "build_keqingv4_continuation_scenarios_json_py", None
    )
    _RUST_SCORE_KEQINGV4_CONTINUATION_SCENARIO_JSON = getattr(
        _rust_ext, "score_keqingv4_continuation_scenario_json_py", None
    )
    _RUST_AGGREGATE_KEQINGV4_CONTINUATION_SCORES_JSON = getattr(
        _rust_ext, "aggregate_keqingv4_continuation_scores_json_py", None
    )
    _RUST_PROJECT_KEQINGV4_CALL_SNAPSHOT_JSON = getattr(
        _rust_ext, "project_keqingv4_call_snapshot_json_py", None
    )
    _RUST_PROJECT_KEQINGV4_DISCARD_SNAPSHOT_JSON = getattr(
        _rust_ext, "project_keqingv4_discard_snapshot_json_py", None
    )
    _RUST_PROJECT_KEQINGV4_RINSHAN_DRAW_SNAPSHOT_JSON = getattr(
        _rust_ext, "project_keqingv4_rinshan_draw_snapshot_json_py", None
    )
    _RUST_ENUMERATE_KEQINGV4_POST_MELD_DISCARDS_JSON = getattr(
        _rust_ext, "enumerate_keqingv4_post_meld_discards_json_py", None
    )
    _RUST_ENUMERATE_KEQINGV4_LIVE_DRAW_WEIGHTS_JSON = getattr(
        _rust_ext, "enumerate_keqingv4_live_draw_weights_json_py", None
    )
    _RUST_ENUMERATE_KEQINGV4_REACH_DISCARDS_JSON = getattr(
        _rust_ext, "enumerate_keqingv4_reach_discards_json_py", None
    )
    _RUST_PROJECT_KEQINGV4_REACH_SNAPSHOT_JSON = getattr(
        _rust_ext, "project_keqingv4_reach_snapshot_json_py", None
    )
    _RUST_CHOOSE_RULEBASE_ACTION_JSON = getattr(
        _rust_ext, "choose_rulebase_action_json_py", None
    )
    _RUST_SCORE_RULEBASE_ACTIONS_JSON = getattr(
        _rust_ext, "score_rulebase_actions_json_py", None
    )
    _USE_RUST = True
    _disable_stale_xmodel1_native_if_needed()


def counts34_to_ids(counts34):
    if _USE_RUST and _RUST_AVAILABLE and _RUST_FUNC is not None:
        return _rust_counts34_to_ids(counts34)
    return _python_counts34_to_ids(counts34)


def _rust_counts34_to_ids(counts34):
    return tuple(_RUST_FUNC(list(counts34)))


def _python_counts34_to_ids(counts34):
    total = sum(counts34)
    if total == 0:
        return ()
    ids = []
    for t34, cnt in enumerate(counts34):
        if cnt > 0:
            for copy_idx in range(cnt):
                ids.append(t34 * 4 + copy_idx)
    return tuple(ids)


def calc_standard_shanten(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_ALL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_ALL(list(counts34), total_tiles // 3))
    ids = counts34_to_ids(counts34)
    return int(_riichienv_shanten(ids))


def calc_shanten_normal(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_NORMAL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_NORMAL(list(counts34), total_tiles // 3))
    return calc_standard_shanten(counts34)


def calc_shanten_all(counts34):
    if not counts34:
        return 8
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_ALL is not None:
        total_tiles = sum(int(v) for v in counts34)
        return int(_RUST_SHANTEN_ALL(list(counts34), total_tiles // 3))
    return calc_standard_shanten(counts34)


def standard_shanten_many(counts34_list):
    if _USE_RUST and _RUST_AVAILABLE and _RUST_SHANTEN_MANY is not None:
        payload = [list(counts34) for counts34 in counts34_list]
        return tuple(int(v) for v in _RUST_SHANTEN_MANY(payload))
    return tuple(calc_standard_shanten(counts34) for counts34 in counts34_list)


def calc_required_tiles(counts34, visible_counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_REQUIRED_TILES is not None):
        raise RuntimeError("Rust required_tiles capability is not available")
    return tuple(
        (int(tile34), int(live_count))
        for tile34, live_count in _RUST_REQUIRED_TILES(list(counts34), list(visible_counts34), int(len_div3))
    )


def calc_draw_deltas(counts34, visible_counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_DRAW_DELTAS is not None):
        raise RuntimeError("Rust draw_deltas capability is not available")
    return tuple(
        (int(tile34), int(live_count), int(shanten_diff))
        for tile34, live_count, shanten_diff in _RUST_DRAW_DELTAS(list(counts34), list(visible_counts34), int(len_div3))
    )


def calc_discard_deltas(counts34, len_div3):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_DISCARD_DELTAS is not None):
        raise RuntimeError("Rust discard_deltas capability is not available")
    return tuple(
        (int(tile34), int(shanten_diff))
        for tile34, shanten_diff in _RUST_DISCARD_DELTAS(list(counts34), int(len_div3))
    )


def build_136_pool_entries(tiles):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_BUILD_136_POOL_ENTRIES is not None):
        raise RuntimeError("Rust 136 pool builder is not available")
    return tuple(
        (str(tile), tuple(int(v) for v in ids))
        for tile, ids in _RUST_BUILD_136_POOL_ENTRIES(list(tiles))
    )


def summarize_3n1(counts34, visible_counts34):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_3N1 is not None):
        raise RuntimeError("Rust 3n+1 summary is not available")
    item = _RUST_SUMMARIZE_3N1(list(counts34), list(visible_counts34))
    return (
        int(item[0]),
        int(item[1]),
        tuple(bool(v) for v in item[2]),
        int(item[3]),
        int(item[4]),
        int(item[5]),
        tuple(bool(v) for v in item[6]),
    )


def summarize_one_shanten_draw_metrics(counts34, visible_counts34):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_SUMMARIZE_ONE_SHANTEN_DRAW_METRICS is not None
    ):
        raise RuntimeError("Rust one-shanten draw metrics are not available")
    item = _RUST_SUMMARIZE_ONE_SHANTEN_DRAW_METRICS(list(counts34), list(visible_counts34))
    return int(item[0]), int(item[1])


def summarize_3n2_candidates(counts34, visible_counts34, summarize_fn):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_3N2_CANDIDATES is not None):
        raise RuntimeError("Rust 3n+2 candidate summaries are not available")
    return tuple(
        (
            int(discard_tile34),
            tuple(int(value) for value in after_counts34),
            int(shanten),
            int(waits_count),
            int(ukeire_type_count),
            int(ukeire_live_count),
            int(good_shape_ukeire_type_count),
            int(good_shape_ukeire_live_count),
            int(improvement_type_count),
            int(improvement_live_count),
        )
        for (
            discard_tile34,
            after_counts34,
            shanten,
            waits_count,
            ukeire_type_count,
            ukeire_live_count,
            good_shape_ukeire_type_count,
            good_shape_ukeire_live_count,
            improvement_type_count,
            improvement_live_count,
        ) in _RUST_SUMMARIZE_3N2_CANDIDATES(list(counts34), list(visible_counts34), summarize_fn)
    )


def summarize_best_3n2_candidate(counts34, visible_counts34, summarize_fn):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_BEST_3N2_CANDIDATE is not None):
        raise RuntimeError("Rust best 3n+2 candidate summary is not available")
    item = _RUST_SUMMARIZE_BEST_3N2_CANDIDATE(list(counts34), list(visible_counts34), summarize_fn)
    if item is None:
        return None
    return (
        int(item[0]),
        tuple(int(value) for value in item[1]),
        int(item[2]),
        int(item[3]),
        int(item[4]),
        int(item[5]),
        int(item[6]),
        int(item[7]),
        int(item[8]),
        int(item[9]),
    )


def _first_exported_xmodel1_npz(output_dir: str) -> _Path | None:
    root = _Path(output_dir)
    if not root.exists():
        return None
    for path in sorted(root.rglob("*.npz")):
        return path
    return None


def _rust_xmodel1_export_looks_complete(output_dir: str) -> bool:
    sample = _first_exported_xmodel1_npz(output_dir)
    if sample is None:
        return False
    try:
        with _np.load(sample, allow_pickle=False) as data:
            if "state_tile_feat" not in data or "state_scalar" not in data:
                return False
            state_tile_feat = data["state_tile_feat"]
            state_scalar = data["state_scalar"]
            if state_tile_feat.size == 0 or state_scalar.size == 0:
                return False
            return bool(_np.any(state_tile_feat != 0)) and bool(_np.any(state_scalar != 0))
    except Exception:
        return False


def build_xmodel1_discard_records(
    *,
    data_dirs=None,
    output_dir: str = "processed_xmodel1",
    smoke: bool = False,
    limit_files: int | None = None,
    progress_every: int | None = None,
    jobs: int | None = None,
    resume: bool | None = None,
):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_BUILD_XMODEL1_DISCARD_RECORDS is not None):
        if _RUST_XMODEL1_SCHEMA_MISMATCH is not None:
            raise RuntimeError(_RUST_XMODEL1_SCHEMA_MISMATCH)
        raise RuntimeError("Rust Xmodel1 discard export is not available")
    data_dirs = list(data_dirs or [])
    try:
        result = _RUST_BUILD_XMODEL1_DISCARD_RECORDS(
            data_dirs,
            str(output_dir),
            bool(smoke),
            None if limit_files is None else int(limit_files),
            None if progress_every is None else int(progress_every),
            None if jobs is None else int(jobs),
            None if resume is None else bool(resume),
        )
    except TypeError:
        try:
            result = _RUST_BUILD_XMODEL1_DISCARD_RECORDS(
                data_dirs,
                str(output_dir),
                bool(smoke),
                None if limit_files is None else int(limit_files),
            )
        except TypeError:
            result = _RUST_BUILD_XMODEL1_DISCARD_RECORDS(data_dirs, str(output_dir), bool(smoke))
    produced_npz = bool(result[2]) if isinstance(result, tuple) and len(result) >= 3 else False
    if produced_npz and not _rust_xmodel1_export_looks_complete(output_dir):
        raise RuntimeError("Rust Xmodel1 full export did not produce complete v2 caches")
    return result


def build_keqingv4_cached_records(*, data_dirs=None, output_dir: str = "processed_v4", smoke: bool = False):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_BUILD_KEQINGV4_CACHED_RECORDS is not None):
        raise RuntimeError("Rust keqingv4 cached export is not available")
    data_dirs = list(data_dirs or [])
    return _RUST_BUILD_KEQINGV4_CACHED_RECORDS(data_dirs, str(output_dir), bool(smoke))


def build_replay_decision_records_mc_return(events):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_REPLAY_DECISION_RECORDS_MC_RETURN_JSON is not None
    ):
        raise RuntimeError("Rust replay sample builder capability is not available")
    payload = _json.dumps(events, ensure_ascii=False)
    return _json.loads(_RUST_BUILD_REPLAY_DECISION_RECORDS_MC_RETURN_JSON(payload))


def native_schema_info():
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_NATIVE_SCHEMA_INFO_JSON is not None):
        raise RuntimeError("Rust native schema capability is not available")
    return _json.loads(_RUST_NATIVE_SCHEMA_INFO_JSON())


def require_native_schema(
    *,
    schema_name: str = "keqingrl_native_boundary",
    schema_version: int = 1,
    action_identity_version: int = 1,
    legal_enumeration_version: int = 1,
    terminal_resolver_version: int = 1,
):
    info = native_schema_info()
    expected = {
        "schema_name": schema_name,
        "schema_version": int(schema_version),
        "action_identity_version": int(action_identity_version),
        "legal_enumeration_version": int(legal_enumeration_version),
        "terminal_resolver_version": int(terminal_resolver_version),
        "runtime_fallback_policy": "fail_closed_no_silent_python_fallback",
    }
    for key, expected_value in expected.items():
        actual = info.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"Rust native schema mismatch for {key}: {actual!r}; expected {expected_value!r}"
            )
    capabilities = dict(info.get("capabilities") or {})
    required_capabilities = {
        "action_identity_v1": "supported",
        "action_id_decode_v1": "supported",
        "mjai_event_expansion_v1": "supported",
        "legal_enumeration_v1": "supported",
        "terminal_resolver_v1": "supported",
        "state_apply_validation_v1": "supported",
        "action_feature_parity_v1": "supported",
        "fixed_seed_eval_gate_v1": "supported",
        "typed_action_feature_api_v1": "supported",
        "nuki": "unsupported",
    }
    for key, expected_value in required_capabilities.items():
        actual = capabilities.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"Rust native capability mismatch for {key}: {actual!r}; expected {expected_value!r}"
            )
    return info


def action_identity(action: dict):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_ACTION_IDENTITY_JSON is not None):
        raise RuntimeError("Rust action identity capability is not available")
    return _json.loads(
        _RUST_ACTION_IDENTITY_JSON(_json.dumps(action, ensure_ascii=False, default=_json_default))
    )


def decode_action_id(action_id: int):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_DECODE_ACTION_ID_JSON is not None):
        raise RuntimeError("Rust action id decode capability is not available")
    return _json.loads(_RUST_DECODE_ACTION_ID_JSON(int(action_id)))


def mjai_events_for_action(action: dict, *, actor: int | None = None):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_MJAI_EVENTS_FOR_ACTION_JSON is not None):
        raise RuntimeError("Rust MJAI action expansion capability is not available")
    return _json.loads(
        _RUST_MJAI_EVENTS_FOR_ACTION_JSON(
            _json.dumps(action, ensure_ascii=False, default=_json_default),
            None if actor is None else int(actor),
        )
    )


def resolve_terminal_action(state_snapshot, actor: int, legal_actions: list[dict], forced_action_types):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_RESOLVE_TERMINAL_ACTION_JSON is not None):
        raise RuntimeError("Rust terminal resolver capability is not available")
    payload = _RUST_RESOLVE_TERMINAL_ACTION_JSON(
        _json.dumps(state_snapshot, ensure_ascii=False, default=_json_default),
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False, default=_json_default),
        _json.dumps(list(forced_action_types), ensure_ascii=False, default=_json_default),
    )
    return None if payload is None else _json.loads(payload)


def xmodel1_schema_info():
    if _USE_RUST and _RUST_AVAILABLE and _RUST_XMODEL1_SCHEMA_INFO is not None:
        name, version, max_candidates, candidate_dim, flag_dim = _RUST_XMODEL1_SCHEMA_INFO()
        return (
            str(name),
            int(version),
            int(max_candidates),
            int(candidate_dim),
            int(flag_dim),
        )
    from .cache_schema import (
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_SCHEMA_NAME,
        XMODEL1_SCHEMA_VERSION,
    )

    return (
        XMODEL1_SCHEMA_NAME,
        XMODEL1_SCHEMA_VERSION,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
    )


def validate_xmodel1_discard_record(chosen_candidate_idx, candidate_mask, candidate_tile_id):
    if _USE_RUST and _RUST_AVAILABLE and _RUST_VALIDATE_XMODEL1_DISCARD_RECORD is not None:
        return bool(
            _RUST_VALIDATE_XMODEL1_DISCARD_RECORD(
                int(chosen_candidate_idx),
                list(candidate_mask),
                list(candidate_tile_id),
            )
        )
    from .cache_schema import XMODEL1_MAX_CANDIDATES

    if len(candidate_mask) != XMODEL1_MAX_CANDIDATES or len(candidate_tile_id) != XMODEL1_MAX_CANDIDATES:
        raise ValueError("candidate arrays must match XMODEL1_MAX_CANDIDATES")
    chosen_candidate_idx = int(chosen_candidate_idx)
    if not (0 <= chosen_candidate_idx < XMODEL1_MAX_CANDIDATES):
        raise ValueError("chosen_candidate_idx out of range")
    if int(candidate_mask[chosen_candidate_idx]) != 1:
        raise ValueError("chosen_candidate_idx must point to an active candidate")

    for idx, (mask_value, tile_id) in enumerate(zip(candidate_mask, candidate_tile_id)):
        mask_value = int(mask_value)
        tile_id = int(tile_id)
        if mask_value == 0 and tile_id != -1:
            raise ValueError(f"padding candidate at index {idx} must use tile_id=-1")
        if mask_value == 1 and not (0 <= tile_id < 34):
            raise ValueError(f"active candidate at index {idx} must use tile34 in [0, 33]")
    return True


def build_xmodel1_runtime_tensors(snapshot: dict, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON is not None
    ):
        if _RUST_XMODEL1_SCHEMA_MISMATCH is not None:
            raise RuntimeError(_RUST_XMODEL1_SCHEMA_MISMATCH)
        raise RuntimeError("Rust xmodel1 runtime tensor capability is not available")
    return _json.loads(
        _RUST_BUILD_XMODEL1_RUNTIME_TENSORS_JSON(
            _json.dumps(snapshot, ensure_ascii=False, default=_json_default),
            int(actor),
            _json.dumps(legal_actions, ensure_ascii=False, default=_json_default),
        )
    )


def replay_state_snapshot(events, actor: int):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_REPLAY_STATE_SNAPSHOT_JSON is not None):
        raise RuntimeError("Rust replay state snapshot capability is not available")
    payload = _json.dumps(events, ensure_ascii=False)
    return _json.loads(_RUST_REPLAY_STATE_SNAPSHOT_JSON(payload, int(actor)))


def validate_replay_state_snapshot(events, actor: int, expected_snapshot: dict | None = None):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_VALIDATE_REPLAY_STATE_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust replay state snapshot validation capability is not available")
    expected_payload = None
    if expected_snapshot is not None:
        expected_payload = _json.dumps(expected_snapshot, ensure_ascii=False, default=_json_default)
    return _json.loads(
        _RUST_VALIDATE_REPLAY_STATE_SNAPSHOT_JSON(
            _json.dumps(events, ensure_ascii=False, default=_json_default),
            int(actor),
            expected_payload,
        )
    )


def build_keqingrl_action_features(snapshot: dict, actions: list[dict], remaining_wall: int | float):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_BUILD_KEQINGRL_ACTION_FEATURES is not None):
        raise RuntimeError("Rust KeqingRL action feature capability is not available")
    return _RUST_BUILD_KEQINGRL_ACTION_FEATURES(
        _json.dumps(snapshot, ensure_ascii=False, default=_json_default),
        _json.dumps(actions, ensure_ascii=False, default=_json_default),
        float(remaining_wall),
    )


def build_keqingrl_action_features_typed(
    hand_counts34,
    visible_counts34,
    action_types,
    tiles,
    flags,
    remaining_wall: int | float,
):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_BUILD_KEQINGRL_ACTION_FEATURES_TYPED is not None):
        raise RuntimeError("Rust KeqingRL typed action feature capability is not available")
    return _RUST_BUILD_KEQINGRL_ACTION_FEATURES_TYPED(
        [int(value) for value in hand_counts34],
        [int(value) for value in visible_counts34],
        [int(value) for value in action_types],
        [int(value) for value in tiles],
        [int(value) for value in flags],
        float(remaining_wall),
    )


def keqingrl_action_feature_dim() -> int:
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_KEQINGRL_ACTION_FEATURE_DIM is not None):
        raise RuntimeError("Rust KeqingRL action feature dim capability is not available")
    return int(_RUST_KEQINGRL_ACTION_FEATURE_DIM())


def fixed_seed_eval_gate(report: dict):
    if not (_USE_RUST and _RUST_AVAILABLE and _RUST_FIXED_SEED_EVAL_GATE_JSON is not None):
        raise RuntimeError("Rust fixed-seed eval gate capability is not available")
    return _json.loads(
        _RUST_FIXED_SEED_EVAL_GATE_JSON(
            _json.dumps(report, ensure_ascii=False, default=_json_default)
        )
    )


def enumerate_legal_action_specs_structural(state_snapshot, actor: int):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_LEGAL_ACTION_SPECS_STRUCTURAL_JSON is not None
    ):
        raise RuntimeError("Rust legal action structural capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False)
    return _json.loads(_RUST_ENUMERATE_LEGAL_ACTION_SPECS_STRUCTURAL_JSON(payload, int(actor)))


def enumerate_public_legal_action_specs(state_snapshot, actor: int):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_PUBLIC_LEGAL_ACTION_SPECS_JSON is not None
    ):
        raise RuntimeError("Rust public legal action capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False)
    return _json.loads(_RUST_ENUMERATE_PUBLIC_LEGAL_ACTION_SPECS_JSON(payload, int(actor)))


def choose_rulebase_action(state_snapshot, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_CHOOSE_RULEBASE_ACTION_JSON is not None
    ):
        raise RuntimeError("Rust rulebase capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False, default=_json_default)
    chosen = _RUST_CHOOSE_RULEBASE_ACTION_JSON(
        payload,
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False, default=_json_default),
    )
    return None if chosen is None else _json.loads(chosen)


def score_rulebase_actions(state_snapshot, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_SCORE_RULEBASE_ACTIONS_JSON is not None
    ):
        raise RuntimeError("Rust rulebase scoring capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False, default=_json_default)
    scored = _RUST_SCORE_RULEBASE_ACTIONS_JSON(
        payload,
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False, default=_json_default),
    )
    return _json.loads(scored)


def enumerate_hora_candidates(state_snapshot, actor: int):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_HORA_CANDIDATES_JSON is not None
    ):
        raise RuntimeError("Rust hora candidate capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False)
    return _json.loads(_RUST_ENUMERATE_HORA_CANDIDATES_JSON(payload, int(actor)))


def can_hora_shape_from_snapshot(state_snapshot, actor: int, pai: str, is_tsumo: bool):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_CAN_HORA_SHAPE_FROM_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust hora shape capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False)
    return bool(
        _RUST_CAN_HORA_SHAPE_FROM_SNAPSHOT_JSON(
            payload,
            int(actor),
            str(pai),
            bool(is_tsumo),
        )
    )


def prepare_hora_evaluation_from_snapshot(
    state_snapshot,
    actor: int,
    pai: str,
    is_tsumo: bool,
    *,
    is_chankan: bool = False,
    is_rinshan=None,
    is_haitei=None,
    is_houtei=None,
):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PREPARE_HORA_EVALUATION_FROM_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust hora evaluation preparation capability is not available")
    payload = _json.dumps(state_snapshot, ensure_ascii=False)
    return _json.loads(
        _RUST_PREPARE_HORA_EVALUATION_FROM_SNAPSHOT_JSON(
            payload,
            int(actor),
            str(pai),
            bool(is_tsumo),
            bool(is_chankan),
            is_rinshan,
            is_haitei,
            is_houtei,
        )
    )


def compute_hora_deltas(oya: int, actor: int, target: int, is_tsumo: bool, cost):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_COMPUTE_HORA_DELTAS_JSON is not None
    ):
        raise RuntimeError("Rust hora delta capability is not available")
    payload = _json.dumps(cost, ensure_ascii=False)
    return _json.loads(
        _RUST_COMPUTE_HORA_DELTAS_JSON(
            int(oya),
            int(actor),
            int(target),
            bool(is_tsumo),
            payload,
        )
    )


def prepare_hora_tile_allocation(prepared):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PREPARE_HORA_TILE_ALLOCATION_JSON is not None
    ):
        raise RuntimeError("Rust hora tile allocation capability is not available")
    payload = _json.dumps(prepared, ensure_ascii=False)
    return _json.loads(_RUST_PREPARE_HORA_TILE_ALLOCATION_JSON(payload))


def build_hora_result_payload(
    *,
    han: int,
    fu: int,
    is_open_hand: bool,
    yaku_names,
    base_yaku_details,
    dora_count: int,
    ura_count: int,
    aka_count: int,
    cost,
    deltas,
):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_HORA_RESULT_PAYLOAD_JSON is not None
    ):
        raise RuntimeError("Rust hora result payload capability is not available")
    return _json.loads(
        _RUST_BUILD_HORA_RESULT_PAYLOAD_JSON(
            int(han),
            int(fu),
            bool(is_open_hand),
            _json.dumps(list(yaku_names), ensure_ascii=False),
            _json.dumps(base_yaku_details, ensure_ascii=False),
            int(dora_count),
            int(ura_count),
            int(aka_count),
            _json.dumps(cost, ensure_ascii=False),
            _json.dumps(list(deltas), ensure_ascii=False),
        )
    )


def evaluate_hora_from_prepared(prepared):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_EVALUATE_HORA_FROM_PREPARED_JSON is not None
    ):
        raise RuntimeError("Rust hora truth capability is not available")
    payload = _json.dumps(prepared, ensure_ascii=False)
    return _json.loads(_RUST_EVALUATE_HORA_FROM_PREPARED_JSON(payload))


def evaluate_hora_truth_from_prepared(prepared):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_EVALUATE_HORA_TRUTH_FROM_PREPARED_JSON is not None
    ):
        raise RuntimeError("Rust hora truth capability is not available")
    payload = _json.dumps(prepared, ensure_ascii=False)
    return _json.loads(_RUST_EVALUATE_HORA_TRUTH_FROM_PREPARED_JSON(payload))


def build_keqingv4_discard_summary(snapshot: dict, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_KEQINGV4_DISCARD_SUMMARY_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 discard summary capability is not available")
    flat = _RUST_BUILD_KEQINGV4_DISCARD_SUMMARY_JSON(
        _json.dumps(snapshot, ensure_ascii=False),
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False),
    )
    return _np.asarray(flat, dtype=_np.float32).reshape(34, KEQINGV4_SUMMARY_DIM)


def build_keqingv4_call_summary(snapshot: dict, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_KEQINGV4_CALL_SUMMARY_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 call summary capability is not available")
    flat = _RUST_BUILD_KEQINGV4_CALL_SUMMARY_JSON(
        _json.dumps(snapshot, ensure_ascii=False),
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False),
    )
    return _np.asarray(flat, dtype=_np.float32).reshape(8, KEQINGV4_SUMMARY_DIM)


def build_keqingv4_special_summary(snapshot: dict, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_KEQINGV4_SPECIAL_SUMMARY_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 special summary capability is not available")
    flat = _RUST_BUILD_KEQINGV4_SPECIAL_SUMMARY_JSON(
        _json.dumps(snapshot, ensure_ascii=False),
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False),
    )
    return _np.asarray(flat, dtype=_np.float32).reshape(3, KEQINGV4_SUMMARY_DIM)


def build_keqingv4_typed_summaries(snapshot: dict, actor: int, legal_actions: list[dict]):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_KEQINGV4_TYPED_SUMMARIES_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 typed summaries capability is not available")
    discard_flat, call_flat, special_flat = _RUST_BUILD_KEQINGV4_TYPED_SUMMARIES_JSON(
        _json.dumps(snapshot, ensure_ascii=False),
        int(actor),
        _json.dumps(legal_actions, ensure_ascii=False),
    )
    return (
        _np.asarray(discard_flat, dtype=_np.float32).reshape(34, KEQINGV4_SUMMARY_DIM),
        _np.asarray(call_flat, dtype=_np.float32).reshape(8, KEQINGV4_SUMMARY_DIM),
        _np.asarray(special_flat, dtype=_np.float32).reshape(3, KEQINGV4_SUMMARY_DIM),
    )


def resolve_keqingv4_continuation_scenarios(snapshot: dict, actor: int, action: dict):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_BUILD_KEQINGV4_CONTINUATION_SCENARIOS_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 continuation scenario capability is not available")
    return _json.loads(
        _RUST_BUILD_KEQINGV4_CONTINUATION_SCENARIOS_JSON(
            _json.dumps(snapshot, ensure_ascii=False),
            int(actor),
            _json.dumps(action, ensure_ascii=False),
        )
    )


def score_keqingv4_continuation_scenario(
    continuation_kind: str,
    policy_logits,
    legal_actions: list[dict],
    *,
    value: float,
    score_delta: float,
    win_prob: float,
    dealin_prob: float,
    rank_pt_value: float,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
    rank_pt_lambda: float,
):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_SCORE_KEQINGV4_CONTINUATION_SCENARIO_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 continuation scoring capability is not available")
    policy_logits_payload = _json.dumps([float(v) for v in policy_logits], ensure_ascii=False)
    return _json.loads(
        _RUST_SCORE_KEQINGV4_CONTINUATION_SCENARIO_JSON(
            str(continuation_kind),
            policy_logits_payload,
            _json.dumps(legal_actions, ensure_ascii=False),
            float(value),
            float(score_delta),
            float(win_prob),
            float(dealin_prob),
            float(rank_pt_value),
            float(beam_lambda),
            float(score_delta_lambda),
            float(win_prob_lambda),
            float(dealin_prob_lambda),
            float(rank_pt_lambda),
        )
    )


def aggregate_keqingv4_continuation_scores(
    root_policy_logits,
    action: dict,
    scenario_scores: list[dict],
):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_AGGREGATE_KEQINGV4_CONTINUATION_SCORES_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 continuation aggregation capability is not available")
    policy_logits_payload = _json.dumps([float(v) for v in root_policy_logits], ensure_ascii=False)
    return _json.loads(
        _RUST_AGGREGATE_KEQINGV4_CONTINUATION_SCORES_JSON(
            policy_logits_payload,
            _json.dumps(action, ensure_ascii=False),
            _json.dumps(scenario_scores, ensure_ascii=False),
        )
    )


def project_keqingv4_call_snapshot(snapshot: dict, actor: int, action: dict):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PROJECT_KEQINGV4_CALL_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 call snapshot projection capability is not available")
    payload = _RUST_PROJECT_KEQINGV4_CALL_SNAPSHOT_JSON(
        _json.dumps(snapshot, ensure_ascii=False),
        int(actor),
        _json.dumps(action, ensure_ascii=False),
    )
    if payload is None:
        return None
    return _json.loads(payload)


def project_keqingv4_discard_snapshot(snapshot: dict, actor: int, pai: str):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PROJECT_KEQINGV4_DISCARD_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 discard snapshot projection capability is not available")
    return _json.loads(
        _RUST_PROJECT_KEQINGV4_DISCARD_SNAPSHOT_JSON(
            _json.dumps(snapshot, ensure_ascii=False),
            int(actor),
            str(pai),
        )
    )


def project_keqingv4_rinshan_draw_snapshot(snapshot: dict, actor: int, pai: str):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PROJECT_KEQINGV4_RINSHAN_DRAW_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 rinshan draw snapshot projection capability is not available")
    return _json.loads(
        _RUST_PROJECT_KEQINGV4_RINSHAN_DRAW_SNAPSHOT_JSON(
            _json.dumps(snapshot, ensure_ascii=False),
            int(actor),
            str(pai),
        )
    )


def enumerate_keqingv4_post_meld_discards(snapshot: dict, actor: int):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_KEQINGV4_POST_MELD_DISCARDS_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 post-meld discard enumeration capability is not available")
    return _json.loads(
        _RUST_ENUMERATE_KEQINGV4_POST_MELD_DISCARDS_JSON(
            _json.dumps(snapshot, ensure_ascii=False),
            int(actor),
        )
    )


def enumerate_keqingv4_live_draw_weights(snapshot: dict):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_KEQINGV4_LIVE_DRAW_WEIGHTS_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 live draw weight enumeration capability is not available")
    return [
        (str(tile), int(weight))
        for tile, weight in _json.loads(
            _RUST_ENUMERATE_KEQINGV4_LIVE_DRAW_WEIGHTS_JSON(
                _json.dumps(snapshot, ensure_ascii=False),
            )
        )
    ]


def enumerate_keqingv4_reach_discards(snapshot: dict, actor: int):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_ENUMERATE_KEQINGV4_REACH_DISCARDS_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 reach discard enumeration capability is not available")
    return [
        (str(pai), bool(tsumogiri))
        for pai, tsumogiri in _json.loads(
            _RUST_ENUMERATE_KEQINGV4_REACH_DISCARDS_JSON(
                _json.dumps(snapshot, ensure_ascii=False),
                int(actor),
            )
        )
    ]


def project_keqingv4_reach_snapshot(snapshot: dict, actor: int, pai: str):
    if not (
        _USE_RUST
        and _RUST_AVAILABLE
        and _RUST_PROJECT_KEQINGV4_REACH_SNAPSHOT_JSON is not None
    ):
        raise RuntimeError("Rust keqingv4 reach snapshot projection capability is not available")
    return _json.loads(
        _RUST_PROJECT_KEQINGV4_REACH_SNAPSHOT_JSON(
            _json.dumps(snapshot, ensure_ascii=False),
            int(actor),
            str(pai),
        )
    )


def resolve_keqingv4_post_meld_followup(snapshot: dict, actor: int, action: dict):
    projected = project_keqingv4_call_snapshot(snapshot, actor, action)
    if projected is None:
        return None, []
    return projected, enumerate_keqingv4_post_meld_discards(projected, actor)


def resolve_keqingv4_rinshan_followup(snapshot: dict, actor: int, pai: str):
    projected = project_keqingv4_rinshan_draw_snapshot(snapshot, actor, pai)
    return projected, enumerate_legal_action_specs_structural(projected, actor)


def resolve_keqingv4_reach_followup(snapshot: dict, actor: int, pai: str):
    projected = project_keqingv4_reach_snapshot(snapshot, actor, pai)
    return projected, enumerate_legal_action_specs_structural(projected, actor)


def is_missing_rust_capability_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "capability is not available" in str(exc)


def is_available():
    return _RUST_AVAILABLE


def is_enabled():
    return _USE_RUST and _RUST_AVAILABLE


def has_3n2_candidate_summaries():
    return _USE_RUST and _RUST_AVAILABLE and _RUST_SUMMARIZE_3N2_CANDIDATES is not None


def enable_rust(enable=True):
    global _USE_RUST
    if enable and not _RUST_AVAILABLE:
        detail = f": {_RUST_IMPORT_ERROR}" if _RUST_IMPORT_ERROR else ""
        _warnings.warn(
            f"Rust library not available{detail}",
            RuntimeWarning,
        )
    _USE_RUST = bool(enable) and _RUST_AVAILABLE


__all__ = [
    "counts34_to_ids",
    "calc_shanten_normal",
    "calc_standard_shanten",
    "calc_shanten_all",
    "standard_shanten_many",
    "calc_required_tiles",
    "calc_draw_deltas",
    "calc_discard_deltas",
    "build_136_pool_entries",
    "summarize_3n1",
    "summarize_one_shanten_draw_metrics",
    "summarize_3n2_candidates",
    "summarize_best_3n2_candidate",
    "build_xmodel1_discard_records",
    "build_keqingv4_cached_records",
    "build_replay_decision_records_mc_return",
    "native_schema_info",
    "require_native_schema",
    "action_identity",
    "decode_action_id",
    "mjai_events_for_action",
    "resolve_terminal_action",
    "xmodel1_schema_info",
    "validate_xmodel1_discard_record",
    "build_xmodel1_runtime_tensors",
    "replay_state_snapshot",
    "validate_replay_state_snapshot",
    "build_keqingrl_action_features",
    "build_keqingrl_action_features_typed",
    "keqingrl_action_feature_dim",
    "fixed_seed_eval_gate",
    "enumerate_legal_action_specs_structural",
    "enumerate_public_legal_action_specs",
    "choose_rulebase_action",
    "score_rulebase_actions",
    "enumerate_hora_candidates",
    "can_hora_shape_from_snapshot",
    "prepare_hora_evaluation_from_snapshot",
    "compute_hora_deltas",
    "prepare_hora_tile_allocation",
    "build_hora_result_payload",
    "evaluate_hora_from_prepared",
    "evaluate_hora_truth_from_prepared",
    "build_keqingv4_discard_summary",
    "build_keqingv4_call_summary",
    "build_keqingv4_special_summary",
    "build_keqingv4_typed_summaries",
    "resolve_keqingv4_continuation_scenarios",
    "score_keqingv4_continuation_scenario",
    "aggregate_keqingv4_continuation_scores",
    "project_keqingv4_call_snapshot",
    "project_keqingv4_discard_snapshot",
    "project_keqingv4_rinshan_draw_snapshot",
    "enumerate_keqingv4_post_meld_discards",
    "enumerate_keqingv4_live_draw_weights",
    "enumerate_keqingv4_reach_discards",
    "project_keqingv4_reach_snapshot",
    "resolve_keqingv4_post_meld_followup",
    "resolve_keqingv4_rinshan_followup",
    "resolve_keqingv4_reach_followup",
    "is_missing_rust_capability_error",
    "is_available",
    "is_enabled",
    "has_3n2_candidate_summaries",
    "enable_rust",
]
