//! PyO3 Python bindings for keqing_core._native
//!
//! This module provides Python bindings for the keqing_core Rust library.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::action_features::{
    action_feature_dim, build_keqingrl_action_features, build_keqingrl_action_features_from_parts,
};
use crate::action_identity::{action_identity_json, decode_action_id, mjai_events_for_action};
use crate::boundary_contract::native_schema_info;
use crate::continuation_scenarios::build_keqingv4_continuation_scenarios;
use crate::continuation_scoring::{aggregate_continuation_scores, score_continuation_scenario};
use crate::counts::TILE_COUNT;
use crate::eval_gate::fixed_seed_eval_gate;
use crate::event_apply::{replay_state_snapshot, validate_replay_state_snapshot};
use crate::hora_truth::{
    evaluate_hora_truth_from_prepared, legacy_error_payload, legacy_payload_from_truth,
};
use crate::keqingv4_export::build_keqingv4_cached_records_with_options;
use crate::keqingv4_summary::{
    build_keqingv4_call_summary, build_keqingv4_discard_summary, build_keqingv4_special_summary,
    enumerate_keqingv4_live_draw_weights, enumerate_keqingv4_post_meld_discards,
    enumerate_keqingv4_reach_discards, project_keqingv4_call_snapshot,
    project_keqingv4_discard_snapshot, project_keqingv4_reach_snapshot,
    project_keqingv4_rinshan_draw_snapshot,
};
use crate::legal_actions::{
    can_hora_shape_from_snapshot, enumerate_hora_candidates,
    enumerate_legal_action_specs_structural, enumerate_public_legal_action_specs,
    prepare_hora_evaluation_from_snapshot,
};
use crate::progress_batch::{
    summarize_3n2_candidates_py_impl, summarize_best_3n2_candidate_py_impl,
};
use crate::progress_delta::{calc_discard_deltas, calc_draw_deltas, calc_required_tiles};
use crate::progress_summary::{summarize_3n1, summarize_one_shanten_draw_metrics};
use crate::replay_samples::build_replay_decision_records_mc_return;
use crate::rulebase::{choose_rulebase_action, score_rulebase_actions};
use crate::score_rules::{
    build_hora_result_payload, compute_hora_deltas, prepare_hora_tile_allocation,
};
use crate::scoring_pool::build_136_pool_entries;
use crate::shanten_table::{calc_shanten_all, calc_shanten_normal, ensure_init};
use crate::standard::counts34_to_ids;
use crate::terminal_resolver::resolve_terminal_action;
use crate::xmodel1_export::{
    build_xmodel1_runtime_tensors, validate_xmodel1_discard_record, xmodel1_schema_info,
};
use crate::xmodel1_schema::{
    XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION,
};

#[pyfunction]
fn counts34_to_ids_py(counts34: &Bound<'_, PyList>) -> PyResult<Vec<u16>> {
    let mut counts = [0i32; TILE_COUNT];

    for (i, item) in counts34.iter().enumerate() {
        if i >= TILE_COUNT {
            break;
        }
        counts[i] = item.extract::<i32>()?;
    }

    Ok(counts34_to_ids(&counts))
}

fn extract_counts34_array(seq: &Bound<'_, PyList>) -> PyResult<[u8; TILE_COUNT]> {
    let mut counts = [0u8; TILE_COUNT];
    for (i, item) in seq.iter().enumerate() {
        if i >= TILE_COUNT {
            break;
        }
        counts[i] = item.extract::<u8>()?;
    }
    Ok(counts)
}

#[pyfunction]
fn calc_shanten_normal_py(counts34: &Bound<'_, PyList>, len_div3: u8) -> PyResult<i32> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    Ok(calc_shanten_normal(&counts, len_div3) as i32)
}

#[pyfunction]
fn calc_shanten_all_py(counts34: &Bound<'_, PyList>, len_div3: u8) -> PyResult<i32> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    Ok(calc_shanten_all(&counts, len_div3) as i32)
}

#[pyfunction]
fn standard_shanten_many_py(
    _py: Python<'_>,
    counts_list: &Bound<'_, PyList>,
) -> PyResult<Vec<i32>> {
    let mut out = Vec::with_capacity(counts_list.len());
    ensure_init();

    for item in counts_list.iter() {
        let counts34 = item.downcast::<PyList>()?;
        let counts = extract_counts34_array(&counts34)?;
        let total_tiles: u8 = counts.iter().sum();
        out.push(calc_shanten_all(&counts, total_tiles / 3) as i32);
    }

    Ok(out)
}

#[pyfunction]
fn calc_required_tiles_py(
    counts34: &Bound<'_, PyList>,
    visible_counts34: &Bound<'_, PyList>,
    len_div3: u8,
) -> PyResult<Vec<(u8, u8)>> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    let visible = extract_counts34_array(visible_counts34)?;
    Ok(calc_required_tiles(&counts, &visible, len_div3)
        .into_iter()
        .map(|item| (item.tile34, item.live_count))
        .collect())
}

#[pyfunction]
fn calc_draw_deltas_py(
    counts34: &Bound<'_, PyList>,
    visible_counts34: &Bound<'_, PyList>,
    len_div3: u8,
) -> PyResult<Vec<(u8, u8, i32)>> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    let visible = extract_counts34_array(visible_counts34)?;
    Ok(calc_draw_deltas(&counts, &visible, len_div3)
        .into_iter()
        .map(|item| (item.tile34, item.live_count, item.shanten_diff as i32))
        .collect())
}

#[pyfunction]
fn calc_discard_deltas_py(counts34: &Bound<'_, PyList>, len_div3: u8) -> PyResult<Vec<(u8, i32)>> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    Ok(calc_discard_deltas(&counts, len_div3)
        .into_iter()
        .map(|item| (item.tile34, item.shanten_diff as i32))
        .collect())
}

#[pyfunction]
fn build_136_pool_entries_py(tiles: Vec<String>) -> PyResult<Vec<(String, Vec<u8>)>> {
    Ok(build_136_pool_entries(&tiles))
}

#[pyfunction]
fn summarize_3n1_py(
    counts34: &Bound<'_, PyList>,
    visible_counts34: &Bound<'_, PyList>,
) -> PyResult<(i32, i32, Vec<bool>, i32, i32, i32, Vec<bool>)> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    let visible = extract_counts34_array(visible_counts34)?;
    let summary = summarize_3n1(&counts, &visible);
    Ok((
        summary.shanten as i32,
        summary.waits_count as i32,
        summary.waits_tiles.into_iter().collect(),
        summary.tehai_count as i32,
        summary.ukeire_type_count as i32,
        summary.ukeire_live_count as i32,
        summary.ukeire_tiles.into_iter().collect(),
    ))
}

#[pyfunction]
fn summarize_one_shanten_draw_metrics_py(
    counts34: &Bound<'_, PyList>,
    visible_counts34: &Bound<'_, PyList>,
) -> PyResult<(i32, i32)> {
    ensure_init();
    let counts = extract_counts34_array(counts34)?;
    let visible = extract_counts34_array(visible_counts34)?;
    let (good_shape_live, improvement_live) = summarize_one_shanten_draw_metrics(&counts, &visible);
    Ok((good_shape_live as i32, improvement_live as i32))
}

#[pyfunction]
fn summarize_3n2_candidates_py(
    py: Python<'_>,
    counts34: &Bound<'_, PyAny>,
    visible_counts34: &Bound<'_, PyAny>,
    summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u8, Vec<u8>, i32, i32, i32, i32, i32, i32, i32, i32)>> {
    summarize_3n2_candidates_py_impl(py, counts34, visible_counts34, summarize_fn)
}

#[pyfunction]
fn summarize_best_3n2_candidate_py(
    py: Python<'_>,
    counts34: &Bound<'_, PyAny>,
    visible_counts34: &Bound<'_, PyAny>,
    summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Option<(u8, Vec<u8>, i32, i32, i32, i32, i32, i32, i32, i32)>> {
    summarize_best_3n2_candidate_py_impl(py, counts34, visible_counts34, summarize_fn)
}

#[pyfunction]
#[pyo3(signature = (data_dirs, output_dir, smoke, limit_files=None, progress_every=None, jobs=None, resume=None))]
fn build_xmodel1_discard_records_py(
    data_dirs: Vec<String>,
    output_dir: String,
    smoke: bool,
    limit_files: Option<usize>,
    progress_every: Option<usize>,
    jobs: Option<usize>,
    resume: Option<bool>,
) -> PyResult<(usize, String, bool)> {
    match crate::xmodel1_export::build_xmodel1_discard_records_with_options(
        &data_dirs,
        &output_dir,
        crate::xmodel1_export::ExportRunOptions {
            smoke,
            resume: resume.unwrap_or(true),
            progress_every: progress_every.unwrap_or(0),
            jobs: jobs.unwrap_or(0),
            limit_files: limit_files.unwrap_or(0),
        },
    ) {
        Ok((count, manifest_path, produced_npz)) => Ok((count, manifest_path, produced_npz)),
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}

#[pyfunction]
fn build_keqingv4_cached_records_py(
    data_dirs: Vec<String>,
    output_dir: String,
    smoke: bool,
) -> PyResult<(usize, String, bool)> {
    let options = crate::xmodel1_export::ExportRunOptions {
        smoke,
        resume: false,
        progress_every: 0,
        jobs: 1,
        limit_files: 0,
    };
    match build_keqingv4_cached_records_with_options(&data_dirs, &output_dir, options) {
        Ok((count, manifest_path, produced_npz)) => Ok((count, manifest_path, produced_npz)),
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}

#[pyfunction]
fn build_replay_decision_records_mc_return_json_py(events_json: &str) -> PyResult<String> {
    let events: Vec<serde_json::Value> = serde_json::from_str(events_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let records =
        build_replay_decision_records_mc_return(&events).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&records).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn native_schema_info_json_py() -> PyResult<String> {
    serde_json::to_string(&native_schema_info())
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn action_identity_json_py(action_json: &str) -> PyResult<String> {
    let action: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let identity = action_identity_json(&action).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&identity).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn decode_action_id_json_py(action_id: u64) -> PyResult<String> {
    let decoded = decode_action_id(action_id).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&decoded).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (action_json, actor=None))]
fn mjai_events_for_action_json_py(action_json: &str, actor: Option<usize>) -> PyResult<String> {
    let action: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let events = mjai_events_for_action(&action, actor).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&events).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn resolve_terminal_action_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
    forced_action_types_json: &str,
) -> PyResult<Option<String>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let forced_action_types: Vec<String> = serde_json::from_str(forced_action_types_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let resolved = resolve_terminal_action(&snapshot, actor, &legal_actions, &forced_action_types)
        .map_err(PyRuntimeError::new_err)?;
    resolved
        .map(|value| {
            serde_json::to_string(&value).map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
        .transpose()
}

#[pyfunction]
fn build_keqingv4_discard_summary_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<Vec<f32>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Ok(build_keqingv4_discard_summary(
        &snapshot,
        actor,
        &legal_actions,
    ))
}

#[pyfunction]
fn build_keqingv4_call_summary_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<Vec<f32>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Ok(build_keqingv4_call_summary(
        &snapshot,
        actor,
        &legal_actions,
    ))
}

#[pyfunction]
fn build_keqingv4_special_summary_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<Vec<f32>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Ok(build_keqingv4_special_summary(
        &snapshot,
        actor,
        &legal_actions,
    ))
}

#[pyfunction]
fn build_keqingv4_typed_summaries_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Ok((
        build_keqingv4_discard_summary(&snapshot, actor, &legal_actions),
        build_keqingv4_call_summary(&snapshot, actor, &legal_actions),
        build_keqingv4_special_summary(&snapshot, actor, &legal_actions),
    ))
}

#[pyfunction]
fn build_keqingv4_continuation_scenarios_json_py(
    snapshot_json: &str,
    actor: usize,
    action_json: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let action: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let scenarios = build_keqingv4_continuation_scenarios(&snapshot, actor, &action);
    serde_json::to_string(&scenarios).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    continuation_kind,
    policy_logits_json,
    legal_actions_json,
    value,
    score_delta,
    win_prob,
    dealin_prob,
    rank_pt_value,
    beam_lambda,
    score_delta_lambda,
    win_prob_lambda,
    dealin_prob_lambda,
    rank_pt_lambda
))]
fn score_keqingv4_continuation_scenario_json_py(
    continuation_kind: &str,
    policy_logits_json: &str,
    legal_actions_json: &str,
    value: f32,
    score_delta: f32,
    win_prob: f32,
    dealin_prob: f32,
    rank_pt_value: f32,
    beam_lambda: f32,
    score_delta_lambda: f32,
    win_prob_lambda: f32,
    dealin_prob_lambda: f32,
    rank_pt_lambda: f32,
) -> PyResult<String> {
    let policy_logits: Vec<f32> = serde_json::from_str(policy_logits_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = score_continuation_scenario(
        continuation_kind,
        &policy_logits,
        &legal_actions,
        value,
        score_delta,
        win_prob,
        dealin_prob,
        rank_pt_value,
        beam_lambda,
        score_delta_lambda,
        win_prob_lambda,
        dealin_prob_lambda,
        rank_pt_lambda,
    );
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn aggregate_keqingv4_continuation_scores_json_py(
    root_policy_logits_json: &str,
    action_json: &str,
    scenario_scores_json: &str,
) -> PyResult<String> {
    let root_policy_logits: Vec<f32> = serde_json::from_str(root_policy_logits_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let action: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let scenario_scores: Vec<serde_json::Value> = serde_json::from_str(scenario_scores_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let parsed = scenario_scores
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = aggregate_continuation_scores(&root_policy_logits, &action, &parsed);
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn project_keqingv4_call_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    action_json: &str,
) -> PyResult<Option<String>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let action: serde_json::Value = serde_json::from_str(action_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    match project_keqingv4_call_snapshot(&snapshot, actor, &action) {
        Some(projected) => serde_json::to_string(&projected)
            .map(Some)
            .map_err(|err| PyRuntimeError::new_err(err.to_string())),
        None => Ok(None),
    }
}

#[pyfunction]
fn project_keqingv4_discard_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    pai: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let projected = project_keqingv4_discard_snapshot(&snapshot, actor, pai);
    serde_json::to_string(&projected).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn project_keqingv4_rinshan_draw_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    pai: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let projected = project_keqingv4_rinshan_draw_snapshot(&snapshot, actor, pai);
    serde_json::to_string(&projected).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_keqingv4_post_meld_discards_json_py(
    snapshot_json: &str,
    actor: usize,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions = enumerate_keqingv4_post_meld_discards(&snapshot, actor);
    serde_json::to_string(&actions).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_keqingv4_live_draw_weights_json_py(snapshot_json: &str) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let weights = enumerate_keqingv4_live_draw_weights(&snapshot);
    serde_json::to_string(&weights).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_keqingv4_reach_discards_json_py(
    snapshot_json: &str,
    actor: usize,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions = enumerate_keqingv4_reach_discards(&snapshot, actor);
    serde_json::to_string(&actions).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn project_keqingv4_reach_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    pai: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let projected = project_keqingv4_reach_snapshot(&snapshot, actor, pai);
    serde_json::to_string(&projected).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn xmodel1_schema_info_py() -> PyResult<(String, u32, usize, usize, usize)> {
    let (name, version, max_candidates, candidate_dim, flag_dim) = xmodel1_schema_info();
    Ok((
        name.to_string(),
        version,
        max_candidates,
        candidate_dim,
        flag_dim,
    ))
}

#[pyfunction]
fn validate_xmodel1_discard_record_py(
    chosen_candidate_idx: i16,
    candidate_mask: Vec<u8>,
    candidate_tile_id: Vec<i16>,
) -> PyResult<bool> {
    validate_xmodel1_discard_record(chosen_candidate_idx, &candidate_mask, &candidate_tile_id)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(true)
}

#[pyfunction]
fn replay_state_snapshot_json_py(events_json: &str, actor: usize) -> PyResult<String> {
    let events: Vec<serde_json::Value> = serde_json::from_str(events_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    replay_state_snapshot(&events, actor).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (events_json, actor, expected_snapshot_json=None))]
fn validate_replay_state_snapshot_json_py(
    events_json: &str,
    actor: usize,
    expected_snapshot_json: Option<&str>,
) -> PyResult<String> {
    let events: Vec<serde_json::Value> = serde_json::from_str(events_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let expected_snapshot = expected_snapshot_json
        .map(|payload| serde_json::from_str::<serde_json::Value>(payload))
        .transpose()
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    validate_replay_state_snapshot(&events, actor, expected_snapshot.as_ref())
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn build_keqingrl_action_features_py(
    snapshot_json: &str,
    actions_json: &str,
    remaining_wall: f32,
) -> PyResult<Vec<Vec<f32>>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions: Vec<serde_json::Value> = serde_json::from_str(actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    build_keqingrl_action_features(&snapshot, &actions, remaining_wall)
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn build_keqingrl_action_features_typed_py(
    hand_counts34: Vec<u8>,
    visible_counts34: Vec<u8>,
    action_types: Vec<u8>,
    tiles: Vec<i16>,
    flags: Vec<u32>,
    remaining_wall: f32,
) -> PyResult<Vec<Vec<f32>>> {
    build_keqingrl_action_features_from_parts(
        &hand_counts34,
        &visible_counts34,
        &action_types,
        &tiles,
        &flags,
        remaining_wall,
    )
    .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn keqingrl_action_feature_dim_py() -> PyResult<usize> {
    Ok(action_feature_dim())
}

#[pyfunction]
fn fixed_seed_eval_gate_json_py(report_json: &str) -> PyResult<String> {
    let report: serde_json::Value = serde_json::from_str(report_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let result = fixed_seed_eval_gate(&report).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn build_xmodel1_runtime_tensors_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = build_xmodel1_runtime_tensors(&snapshot, actor, &legal_actions)
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_legal_action_specs_structural_json_py(
    snapshot_json: &str,
    actor: usize,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions = enumerate_legal_action_specs_structural(&snapshot, actor)
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&actions).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_public_legal_action_specs_json_py(
    snapshot_json: &str,
    actor: usize,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions =
        enumerate_public_legal_action_specs(&snapshot, actor).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&actions).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn enumerate_hora_candidates_json_py(snapshot_json: &str, actor: usize) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let actions = enumerate_hora_candidates(&snapshot, actor).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&actions).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn can_hora_shape_from_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    pai: &str,
    is_tsumo: bool,
) -> PyResult<bool> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    can_hora_shape_from_snapshot(&snapshot, actor, pai, is_tsumo).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (snapshot_json, actor, pai, is_tsumo, is_chankan, is_rinshan=None, is_haitei=None, is_houtei=None))]
fn prepare_hora_evaluation_from_snapshot_json_py(
    snapshot_json: &str,
    actor: usize,
    pai: &str,
    is_tsumo: bool,
    is_chankan: bool,
    is_rinshan: Option<bool>,
    is_haitei: Option<bool>,
    is_houtei: Option<bool>,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = prepare_hora_evaluation_from_snapshot(
        &snapshot, actor, pai, is_tsumo, is_chankan, is_rinshan, is_haitei, is_houtei,
    )
    .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn compute_hora_deltas_json_py(
    oya: usize,
    actor: usize,
    target: usize,
    is_tsumo: bool,
    cost_json: &str,
) -> PyResult<String> {
    let cost: serde_json::Value = serde_json::from_str(cost_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let deltas = compute_hora_deltas(oya, actor, target, is_tsumo, &cost)
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&deltas).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn prepare_hora_tile_allocation_json_py(prepared_json: &str) -> PyResult<String> {
    let prepared: serde_json::Value = serde_json::from_str(prepared_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = prepare_hora_tile_allocation(&prepared).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn build_hora_result_payload_json_py(
    han: i32,
    fu: i32,
    is_open_hand: bool,
    yaku_names_json: &str,
    base_yaku_details_json: &str,
    dora_count: i32,
    ura_count: i32,
    aka_count: i32,
    cost_json: &str,
    deltas_json: &str,
) -> PyResult<String> {
    let yaku_names: Vec<String> = serde_json::from_str(yaku_names_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let base_yaku_details: serde_json::Value = serde_json::from_str(base_yaku_details_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let cost: serde_json::Value = serde_json::from_str(cost_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let deltas: Vec<i32> = serde_json::from_str(deltas_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = build_hora_result_payload(
        han,
        fu,
        is_open_hand,
        &yaku_names,
        &base_yaku_details,
        dora_count,
        ura_count,
        aka_count,
        &cost,
        &deltas,
    )
    .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn evaluate_hora_from_prepared_json_py(prepared_json: &str) -> PyResult<String> {
    let prepared: serde_json::Value = serde_json::from_str(prepared_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let payload = match evaluate_hora_truth_from_prepared(&prepared) {
        Ok(truth) => legacy_payload_from_truth(&truth),
        Err(err) if err == "no cost" => legacy_error_payload(&err),
        Err(err) => return Err(PyRuntimeError::new_err(err)),
    };
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn evaluate_hora_truth_from_prepared_json_py(prepared_json: &str) -> PyResult<String> {
    let prepared: serde_json::Value = serde_json::from_str(prepared_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let truth = evaluate_hora_truth_from_prepared(&prepared).map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&truth).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn choose_rulebase_action_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<Option<String>> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let chosen = choose_rulebase_action(&snapshot, actor, &legal_actions)
        .map_err(PyRuntimeError::new_err)?;
    chosen
        .map(|value| {
            serde_json::to_string(&value).map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
        .transpose()
}

#[pyfunction]
fn score_rulebase_actions_json_py(
    snapshot_json: &str,
    actor: usize,
    legal_actions_json: &str,
) -> PyResult<String> {
    let snapshot: serde_json::Value = serde_json::from_str(snapshot_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let legal_actions: Vec<serde_json::Value> = serde_json::from_str(legal_actions_json)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let scored = score_rulebase_actions(&snapshot, actor, &legal_actions)
        .map_err(PyRuntimeError::new_err)?;
    serde_json::to_string(&scored).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pymodule]
pub fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    ensure_init();
    m.add_function(wrap_pyfunction!(counts34_to_ids_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_shanten_normal_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_shanten_all_py, m)?)?;
    m.add_function(wrap_pyfunction!(standard_shanten_many_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_required_tiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_draw_deltas_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_discard_deltas_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_136_pool_entries_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_3n1_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_one_shanten_draw_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_3n2_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_best_3n2_candidate_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_xmodel1_discard_records_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_keqingv4_cached_records_py, m)?)?;
    m.add_function(wrap_pyfunction!(native_schema_info_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(action_identity_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_action_id_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(mjai_events_for_action_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_terminal_action_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_replay_decision_records_mc_return_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(build_keqingv4_discard_summary_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_keqingv4_call_summary_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_keqingv4_special_summary_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_keqingv4_typed_summaries_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_keqingv4_continuation_scenarios_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        score_keqingv4_continuation_scenario_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggregate_keqingv4_continuation_scores_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(project_keqingv4_call_snapshot_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        project_keqingv4_discard_snapshot_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        project_keqingv4_rinshan_draw_snapshot_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        enumerate_keqingv4_post_meld_discards_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        enumerate_keqingv4_live_draw_weights_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        enumerate_keqingv4_reach_discards_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        project_keqingv4_reach_snapshot_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(xmodel1_schema_info_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_xmodel1_discard_record_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_xmodel1_runtime_tensors_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(replay_state_snapshot_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_replay_state_snapshot_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_keqingrl_action_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_keqingrl_action_features_typed_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(keqingrl_action_feature_dim_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_seed_eval_gate_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        enumerate_legal_action_specs_structural_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        enumerate_public_legal_action_specs_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(enumerate_hora_candidates_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(can_hora_shape_from_snapshot_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        prepare_hora_evaluation_from_snapshot_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_hora_deltas_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_hora_tile_allocation_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_hora_result_payload_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_hora_from_prepared_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        evaluate_hora_truth_from_prepared_json_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(choose_rulebase_action_json_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_rulebase_actions_json_py, m)?)?;
    m.add("TILE_COUNT", TILE_COUNT)?;
    m.add("XMODEL1_SCHEMA_NAME", XMODEL1_SCHEMA_NAME)?;
    m.add("XMODEL1_SCHEMA_VERSION", XMODEL1_SCHEMA_VERSION)?;
    m.add("XMODEL1_MAX_CANDIDATES", XMODEL1_MAX_CANDIDATES)?;
    m.add(
        "XMODEL1_CANDIDATE_FEATURE_DIM",
        XMODEL1_CANDIDATE_FEATURE_DIM,
    )?;
    m.add("XMODEL1_CANDIDATE_FLAG_DIM", XMODEL1_CANDIDATE_FLAG_DIM)?;
    Ok(())
}
