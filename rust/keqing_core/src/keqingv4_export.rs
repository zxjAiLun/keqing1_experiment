use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use half::f16;
use serde::Serialize;
use serde_json::Value;
use zip::ZipWriter;

use crate::export_common::{
    collect_mjson_files, finalize_temp_npz, output_npz_path, print_export_progress, temp_npz_path,
    write_json_manifest, write_npy_f16, write_npy_f32, write_npy_i16, write_npy_i32, write_npy_i8,
    write_npy_u8,
};
use crate::keqingv4_summary::{
    build_keqingv4_call_summary, build_keqingv4_discard_summary, build_keqingv4_special_summary,
};
use crate::replay_export_core as replay_core;
use crate::snapshot::snapshot_for_actor;
use crate::state_core::GameStateCore;
use crate::xmodel1_export::{
    action_idx_from_action, encode_state_features_for_actor, ExportRunOptions,
};

const KEQINGV4_SCHEMA_NAME: &str = "keqingv4";
const KEQINGV4_SCHEMA_VERSION: u32 = 7;
const KEQINGV4_SUMMARY_DIM: usize = 28;
const KEQINGV4_CALL_SUMMARY_SLOTS: usize = 8;
const KEQINGV4_SPECIAL_SUMMARY_SLOTS: usize = 3;
const KEQINGV4_OPPORTUNITY_DIM: usize = 3;

const STATE_TILE_CHANNELS: usize = 57;
const STATE_SCALAR_DIM: usize = 56;
const ACTION_SPACE: usize = 45;
const SCORE_NORM: f32 = 30000.0;
const MC_RETURN_GAMMA: f32 = 0.99;

type RoundState = replay_core::RoundState;

fn visible_counts_for_decision(round_state: &RoundState, actor: usize) -> [u8; 34] {
    round_state.feature_tracker.players[actor].visible_counts34
}

#[derive(Debug, Clone, Serialize)]
struct ExportManifest<'a> {
    schema_name: &'a str,
    schema_version: u32,
    summary_dim: usize,
    call_summary_slots: usize,
    special_summary_slots: usize,
    opportunity_dim: usize,
    file_count: usize,
    exported_file_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    shard_file_counts: BTreeMap<String, usize>,
    export_mode: &'a str,
    used_python_semantics: bool,
    files: Vec<String>,
}

#[derive(Debug, Clone)]
struct ExportJob {
    src_path: PathBuf,
    out_path: PathBuf,
}

#[derive(Debug, Clone)]
struct JobResult {
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    exported_file_count: usize,
    shard: String,
    source: String,
}

#[derive(Debug, Clone)]
struct KeqingV4Record {
    actor: usize,
    tile_feat: Vec<u16>,
    scalar: Vec<u16>,
    mask: Vec<u8>,
    action_idx: i16,
    value_target: f32,
    score_delta_target: f32,
    win_target: f32,
    dealin_target: f32,
    pts_given_win_target: f32,
    pts_given_dealin_target: f32,
    opp_tenpai_target: [f32; 3],
    score_before_action: i32,
    final_score_delta_points_target: i32,
    final_rank_target: u8,
    event_history: [[i16; replay_core::EVENT_HISTORY_FEATURE_DIM]; replay_core::EVENT_HISTORY_LEN],
    v4_opportunity: [u8; KEQINGV4_OPPORTUNITY_DIM],
    discard_summary: Vec<u16>,
    call_summary: Vec<u16>,
    special_summary: Vec<u16>,
}

fn tie_break_order(game_start_oya: usize, actor: usize) -> usize {
    (actor + 4 - game_start_oya) % 4
}

fn final_rank_targets(scores: &[i32; 4], game_start_oya: usize) -> [u8; 4] {
    let mut ordered = [0usize, 1, 2, 3];
    ordered.sort_by(|lhs, rhs| {
        scores[*rhs].cmp(&scores[*lhs]).then_with(|| {
            tie_break_order(game_start_oya, *lhs).cmp(&tie_break_order(game_start_oya, *rhs))
        })
    });
    let mut out = [0u8; 4];
    for (rank, actor) in ordered.iter().enumerate() {
        out[*actor] = rank as u8;
    }
    out
}

fn load_events(path: &Path) -> Result<Vec<Value>, String> {
    let raw = path.to_string_lossy().to_string();
    replay_core::normalize_replay_events(&raw)
}

fn build_legal_mask(legal_actions: &[Value]) -> Vec<u8> {
    let mut mask = vec![0u8; ACTION_SPACE];
    for action in legal_actions {
        let idx = action_idx_from_action(action);
        if idx >= 0 && (idx as usize) < ACTION_SPACE {
            mask[idx as usize] = 1;
        }
    }
    mask
}

fn f16_bits_vec(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| f16::from_f32(*v).to_bits()).collect()
}

fn build_v4_opportunity(legal_actions: &[Value]) -> [u8; KEQINGV4_OPPORTUNITY_DIM] {
    let mut out = [0u8; KEQINGV4_OPPORTUNITY_DIM];
    for action in legal_actions {
        let action_type = action
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match action_type {
            "reach" => out[0] = 1,
            "hora" => out[1] = 1,
            "chi" | "pon" | "daiminkan" | "ankan" | "kakan" | "none" => out[2] = 1,
            _ => {}
        }
    }
    out
}

fn encode_record(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    legal_actions: &[Value],
    chosen_action: &Value,
    event_history: [[i16; replay_core::EVENT_HISTORY_FEATURE_DIM]; replay_core::EVENT_HISTORY_LEN],
) -> Result<KeqingV4Record, String> {
    let (tile_feat, scalar) = encode_state_features_for_actor(round_state, actor)?;
    let mask = build_legal_mask(legal_actions);
    let action_idx = action_idx_from_action(chosen_action);
    if action_idx < 0 || action_idx as usize >= ACTION_SPACE {
        return Err(format!(
            "invalid action_idx from chosen_action: {action_idx}"
        ));
    }
    if mask[action_idx as usize] == 0 {
        return Err(format!(
            "chosen action idx {} is not legal in current legal set",
            action_idx
        ));
    }

    let snapshot = snapshot_for_actor(core_state, actor);
    let snapshot_json = serde_json::to_value(&snapshot)
        .map_err(|err| format!("failed to serialize snapshot for actor {actor}: {err}"))?;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let opp_tenpai_target =
        replay_core::compute_opp_tenpai_target(round_state, actor, &before_visible34);

    let discard_summary = build_keqingv4_discard_summary(&snapshot_json, actor, legal_actions);
    let call_summary = build_keqingv4_call_summary(&snapshot_json, actor, legal_actions);
    let special_summary = build_keqingv4_special_summary(&snapshot_json, actor, legal_actions);
    let v4_opportunity = build_v4_opportunity(legal_actions);

    Ok(KeqingV4Record {
        actor,
        tile_feat,
        scalar,
        mask,
        action_idx,
        value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        opp_tenpai_target,
        score_before_action: round_state.scores[actor],
        final_score_delta_points_target: 0,
        final_rank_target: 0,
        event_history,
        v4_opportunity,
        discard_summary: f16_bits_vec(&discard_summary),
        call_summary: f16_bits_vec(&call_summary),
        special_summary: f16_bits_vec(&special_summary),
    })
}

fn apply_round_target_updates(
    records: &mut [KeqingV4Record],
    updates: &[replay_core::RoundTargetUpdate],
) {
    replay_core::apply_round_target_updates(records, updates, |record, update| {
        record.score_delta_target = update.score_delta_target;
        record.value_target = update.global_value_target;
        record.win_target = update.win_target;
        record.dealin_target = update.dealin_target;
        record.pts_given_win_target = update.pts_given_win_target;
        record.pts_given_dealin_target = update.pts_given_dealin_target;
    });
}

fn collect_records_from_mjson(path: &Path) -> Result<Vec<KeqingV4Record>, String> {
    let events = load_events(path)?;
    let events_slice: &[Value] = events.as_slice();
    let mut records = replay_core::drive_export_records(
        &events,
        SCORE_NORM,
        MC_RETURN_GAMMA,
        |ctx| {
            let Some(decision) = replay_core::build_structural_actor_chosen_action_context(ctx)?
            else {
                return Ok(None);
            };
            let event_history = replay_core::compute_event_history(
                events_slice,
                decision.decision.event.event_index,
            );
            Ok(encode_record(
                decision.decision.event.state,
                decision.decision.event.core_state,
                decision.decision.event.actor,
                &decision.decision.legal_actions,
                &decision.chosen_action,
                event_history,
            )
            .ok())
        },
        |_ctx| Ok(None),
        |ctx| {
            let reaction = replay_core::build_reaction_none_context(ctx);
            let event_history =
                replay_core::compute_event_history(events_slice, reaction.reaction.event_index);
            Ok(encode_record(
                reaction.reaction.state,
                reaction.reaction.core_state,
                reaction.reaction.actor,
                reaction.reaction.legal_actions,
                &reaction.chosen_action,
                event_history,
            )
            .ok())
        },
        apply_round_target_updates,
        |_| Ok(()),
    )?;

    let mut terminal_round_state = RoundState::default();
    for event in events_slice {
        replay_core::apply_event_round_state(&mut terminal_round_state, event);
    }
    let game_start_oya = terminal_round_state.game_start_oya.clamp(0, 3) as usize;
    let final_rank_targets = final_rank_targets(&terminal_round_state.scores, game_start_oya);
    for record in &mut records {
        if record.actor < 4 {
            record.final_score_delta_points_target =
                terminal_round_state.scores[record.actor] - record.score_before_action;
            record.final_rank_target = final_rank_targets[record.actor];
        } else {
            record.final_score_delta_points_target = 0;
            record.final_rank_target = 0;
        }
    }

    Ok(records)
}

fn write_npz(path: &Path, records: &[KeqingV4Record]) -> Result<(), String> {
    let temp_path = temp_npz_path(path);
    let file = fs::File::create(&temp_path)
        .map_err(|err| format!("failed to create temp npz {}: {err}", temp_path.display()))?;
    let mut zip = ZipWriter::new(file);
    let n = records.len();

    let mut tile_feat = Vec::with_capacity(n * STATE_TILE_CHANNELS * 34);
    let mut scalar = Vec::with_capacity(n * STATE_SCALAR_DIM);
    let mut mask = Vec::with_capacity(n * ACTION_SPACE);
    let mut action_idx = Vec::with_capacity(n);
    let mut value = Vec::with_capacity(n);
    let mut score_delta = Vec::with_capacity(n);
    let mut win = Vec::with_capacity(n);
    let mut dealin = Vec::with_capacity(n);
    let mut pts_given_win = Vec::with_capacity(n);
    let mut pts_given_dealin = Vec::with_capacity(n);
    let mut opp_tenpai = Vec::with_capacity(n * 3);
    let mut final_rank = Vec::with_capacity(n);
    let mut final_score_delta_points = Vec::with_capacity(n);
    let mut event_history = Vec::with_capacity(
        n * replay_core::EVENT_HISTORY_LEN * replay_core::EVENT_HISTORY_FEATURE_DIM,
    );
    let mut v4_opportunity = Vec::with_capacity(n * KEQINGV4_OPPORTUNITY_DIM);
    let mut discard_summary = Vec::with_capacity(n * 34 * KEQINGV4_SUMMARY_DIM);
    let mut call_summary =
        Vec::with_capacity(n * KEQINGV4_CALL_SUMMARY_SLOTS * KEQINGV4_SUMMARY_DIM);
    let mut special_summary =
        Vec::with_capacity(n * KEQINGV4_SPECIAL_SUMMARY_SLOTS * KEQINGV4_SUMMARY_DIM);

    for record in records {
        tile_feat.extend_from_slice(&record.tile_feat);
        scalar.extend_from_slice(&record.scalar);
        mask.extend_from_slice(&record.mask);
        action_idx.push(record.action_idx);
        value.push(f16::from_f32(record.value_target).to_bits());
        score_delta.push(f16::from_f32(record.score_delta_target).to_bits());
        win.push(f16::from_f32(record.win_target).to_bits());
        dealin.push(f16::from_f32(record.dealin_target).to_bits());
        pts_given_win.push(record.pts_given_win_target);
        pts_given_dealin.push(record.pts_given_dealin_target);
        opp_tenpai.extend_from_slice(&record.opp_tenpai_target);
        final_rank.push(record.final_rank_target as i8);
        final_score_delta_points.push(record.final_score_delta_points_target);
        for row in record.event_history.iter() {
            event_history.extend_from_slice(row);
        }
        v4_opportunity.extend_from_slice(&record.v4_opportunity);
        discard_summary.extend_from_slice(&record.discard_summary);
        call_summary.extend_from_slice(&record.call_summary);
        special_summary.extend_from_slice(&record.special_summary);
    }

    write_npy_f16(
        &mut zip,
        "tile_feat.npy",
        &[n, STATE_TILE_CHANNELS, 34],
        &tile_feat,
    )?;
    write_npy_f16(&mut zip, "scalar.npy", &[n, STATE_SCALAR_DIM], &scalar)?;
    write_npy_u8(&mut zip, "mask.npy", &[n, ACTION_SPACE], &mask)?;
    write_npy_i16(&mut zip, "action_idx.npy", &[n], &action_idx)?;
    write_npy_f16(&mut zip, "value.npy", &[n], &value)?;
    write_npy_f16(&mut zip, "score_delta_target.npy", &[n], &score_delta)?;
    write_npy_f16(&mut zip, "win_target.npy", &[n], &win)?;
    write_npy_f16(&mut zip, "dealin_target.npy", &[n], &dealin)?;
    write_npy_f32(&mut zip, "pts_given_win_target.npy", &[n], &pts_given_win)?;
    write_npy_f32(
        &mut zip,
        "pts_given_dealin_target.npy",
        &[n],
        &pts_given_dealin,
    )?;
    write_npy_f32(&mut zip, "opp_tenpai_target.npy", &[n, 3], &opp_tenpai)?;
    write_npy_i8(&mut zip, "final_rank_target.npy", &[n], &final_rank)?;
    write_npy_i32(
        &mut zip,
        "final_score_delta_points_target.npy",
        &[n],
        &final_score_delta_points,
    )?;
    write_npy_i16(
        &mut zip,
        "event_history.npy",
        &[
            n,
            replay_core::EVENT_HISTORY_LEN,
            replay_core::EVENT_HISTORY_FEATURE_DIM,
        ],
        &event_history,
    )?;
    write_npy_u8(
        &mut zip,
        "v4_opportunity.npy",
        &[n, KEQINGV4_OPPORTUNITY_DIM],
        &v4_opportunity,
    )?;
    write_npy_f16(
        &mut zip,
        "v4_discard_summary.npy",
        &[n, 34, KEQINGV4_SUMMARY_DIM],
        &discard_summary,
    )?;
    write_npy_f16(
        &mut zip,
        "v4_call_summary.npy",
        &[n, KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM],
        &call_summary,
    )?;
    write_npy_f16(
        &mut zip,
        "v4_special_summary.npy",
        &[n, KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM],
        &special_summary,
    )?;

    finalize_temp_npz(zip, &temp_path, path, false)
}

fn process_job(job: &ExportJob, resume: bool) -> Result<JobResult, String> {
    if resume && job.out_path.exists() {
        return Ok(JobResult {
            processed_file_count: 0,
            skipped_existing_file_count: 1,
            exported_file_count: 0,
            shard: job
                .src_path
                .parent()
                .and_then(|parent| parent.file_name())
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            source: job.src_path.to_string_lossy().to_string(),
        });
    }

    let records = collect_records_from_mjson(&job.src_path)?;
    if !records.is_empty() {
        write_npz(&job.out_path, &records)?;
    }

    Ok(JobResult {
        processed_file_count: usize::from(!records.is_empty()),
        skipped_existing_file_count: 0,
        exported_file_count: usize::from(!records.is_empty()),
        shard: job
            .src_path
            .parent()
            .and_then(|parent| parent.file_name())
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        source: job.src_path.to_string_lossy().to_string(),
    })
}

fn write_manifest(
    output_dir: &Path,
    file_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    exported_file_count: usize,
    shard_file_counts: BTreeMap<String, usize>,
    files: Vec<String>,
) -> Result<String, String> {
    let manifest = ExportManifest {
        schema_name: KEQINGV4_SCHEMA_NAME,
        schema_version: KEQINGV4_SCHEMA_VERSION,
        summary_dim: KEQINGV4_SUMMARY_DIM,
        call_summary_slots: KEQINGV4_CALL_SUMMARY_SLOTS,
        special_summary_slots: KEQINGV4_SPECIAL_SUMMARY_SLOTS,
        opportunity_dim: KEQINGV4_OPPORTUNITY_DIM,
        file_count,
        exported_file_count,
        processed_file_count,
        skipped_existing_file_count,
        shard_file_counts,
        export_mode: "rust_semantic_core",
        used_python_semantics: false,
        files,
    };
    write_json_manifest(output_dir, "keqingv4_export_manifest.json", &manifest)
}

pub fn build_keqingv4_cached_records_with_options(
    data_dirs: &[String],
    output_dir: &str,
    options: ExportRunOptions,
) -> Result<(usize, String, bool), String> {
    let output_dir = PathBuf::from(output_dir);
    fs::create_dir_all(&output_dir).map_err(|err| {
        format!(
            "failed to create output dir {}: {err}",
            output_dir.display()
        )
    })?;

    let mut jobs: Vec<ExportJob> = Vec::new();
    for raw_file in collect_mjson_files(data_dirs, options.smoke)? {
        let src_path = PathBuf::from(&raw_file);
        let out_path = output_npz_path(&output_dir, &raw_file);
        if let Some(out_shard) = out_path.parent() {
            fs::create_dir_all(out_shard).map_err(|err| {
                format!("failed to create shard dir {}: {err}", out_shard.display())
            })?;
        }
        jobs.push(ExportJob { src_path, out_path });
    }

    let total = jobs.len();
    let mut processed_file_count = 0usize;
    let mut skipped_existing_file_count = 0usize;
    let mut exported_file_count = 0usize;
    let mut shard_file_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut exported_sources: Vec<String> = Vec::new();
    let started = Instant::now();

    let jobs_count = if options.jobs == 0 { 1 } else { options.jobs };
    let (tx, rx) = mpsc::channel::<Result<JobResult, String>>();
    let work_queue = std::sync::Arc::new(std::sync::Mutex::new(jobs));
    let mut handles = Vec::new();
    for _ in 0..jobs_count {
        let tx = tx.clone();
        let queue = work_queue.clone();
        let resume = options.resume;
        handles.push(thread::spawn(move || loop {
            let maybe_job = {
                let mut queue = queue.lock().expect("work queue poisoned");
                queue.pop()
            };
            let Some(job) = maybe_job else {
                break;
            };
            let result = process_job(&job, resume);
            let _ = tx.send(result);
        }));
    }
    drop(tx);

    let mut done = 0usize;
    while let Ok(result) = rx.recv() {
        done += 1;
        match result {
            Ok(result) => {
                processed_file_count += result.processed_file_count;
                skipped_existing_file_count += result.skipped_existing_file_count;
                exported_file_count += result.exported_file_count;
                if result.processed_file_count > 0 {
                    *shard_file_counts.entry(result.shard).or_insert(0) += 1;
                    exported_sources.push(result.source);
                }
            }
            Err(err) => return Err(err),
        }
        if done % options.progress_every.max(1) == 0 || done == total {
            print_export_progress(
                "keqingv4",
                &started,
                done,
                total,
                processed_file_count,
                skipped_existing_file_count,
                exported_file_count,
                &std::collections::VecDeque::new(),
                jobs_count,
            );
        }
    }
    for handle in handles {
        let _ = handle.join();
    }

    exported_sources.sort();
    let manifest_path = write_manifest(
        &output_dir,
        total,
        processed_file_count,
        skipped_existing_file_count,
        exported_file_count,
        shard_file_counts,
        exported_sources,
    )?;
    Ok((total, manifest_path, exported_file_count > 0))
}
