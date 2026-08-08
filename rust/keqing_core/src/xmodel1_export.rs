//! Rust-side Xmodel1 discard export.
//!
//! This module now produces real candidate-centric `.npz` exports instead of
//! placeholder smoke tensors. The implementation intentionally mirrors the
//! current Python preprocessing contract closely enough for training to start,
//! while staying focused on the discard-only Xmodel1 schema.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use half::f16;
use serde::Serialize;
use serde_json::{json, Value};
use zip::ZipWriter;

use crate::export_common::{
    collect_mjson_files, finalize_temp_npz, output_npz_path, print_export_progress,
    read_npy_first_dim_from_zip, temp_npz_path, write_json_manifest, write_npy_f16, write_npy_f32,
    write_npy_i16, write_npy_i32, write_npy_i8, write_npy_u8, write_npy_unicode_scalar,
};
use crate::keqingv4_summary::project_keqingv4_call_snapshot;
use crate::progress_summary::{summarize_3n1, summarize_like_python, Summary3n1};
use crate::replay_export_core as replay_core;
use crate::replay_export_core::{normalize_tile_repr, strip_aka, tile34_from_pai};
use crate::replay_samples::public_legal_actions_for_snapshot;
use crate::shanten_table::ensure_init;
use crate::snapshot::snapshot_for_actor;
use crate::state_core::{GameStateCore, SnapshotCore};
use crate::value_proxy::{
    confirmed_han_floor as shared_confirmed_han_floor, tenpai_value_proxy_norm,
};
use crate::xmodel1_schema::{
    validate_candidate_mask_and_choice, XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES, XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    XMODEL1_SPECIAL_TYPE_ANKAN, XMODEL1_SPECIAL_TYPE_CHI_HIGH, XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID, XMODEL1_SPECIAL_TYPE_DAIMINKAN, XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_SPECIAL_TYPE_HORA, XMODEL1_SPECIAL_TYPE_KAKAN, XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON, XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_RYUKYOKU,
};

const XMODEL1_STATE_TILE_CHANNELS: usize = 57;
const XMODEL1_STATE_SCALAR_DIM: usize = 56;
const XMODEL1_RULE_CONTEXT_DIM: usize = 6;
const XMODEL1_SAMPLE_TYPE_DISCARD: i8 = 0;
const XMODEL1_SAMPLE_TYPE_RIICHI: i8 = 1;
const XMODEL1_SAMPLE_TYPE_CALL: i8 = 2;
const XMODEL1_SAMPLE_TYPE_HORA: i8 = 3;
const MC_RETURN_GAMMA: f32 = 0.99;
const SCORE_NORM: f32 = 30000.0;
const TILE_KIND_COUNT: usize = 34;
const DEEP_SHANTEN_FAST_PATH_CUTOFF: i8 = 3;
const INTERRUPTED_ERROR_PREFIX: &str = "xmodel1 preprocess interrupted";

#[derive(Debug, Clone, Copy)]
enum ExportStage {
    Normalize,
    RecordDrive,
    DiscardCandidateAnalysis,
    SpecialSummary,
    NpzWrite,
}

#[derive(Debug, Clone, Copy, Default)]
struct ExportStageTimings {
    normalize_s: f64,
    record_drive_s: f64,
    discard_candidate_analysis_s: f64,
    special_summary_s: f64,
    npz_write_s: f64,
}

impl ExportStageTimings {
    fn add_stage(&mut self, stage: ExportStage, elapsed_s: f64) {
        match stage {
            ExportStage::Normalize => self.normalize_s += elapsed_s,
            ExportStage::RecordDrive => self.record_drive_s += elapsed_s,
            ExportStage::DiscardCandidateAnalysis => {
                self.discard_candidate_analysis_s += elapsed_s;
            }
            ExportStage::SpecialSummary => self.special_summary_s += elapsed_s,
            ExportStage::NpzWrite => self.npz_write_s += elapsed_s,
        }
    }

    fn accumulated_s(&self) -> f64 {
        self.normalize_s
            + self.record_drive_s
            + self.discard_candidate_analysis_s
            + self.special_summary_s
            + self.npz_write_s
    }
}

impl AddAssign for ExportStageTimings {
    fn add_assign(&mut self, rhs: Self) {
        self.normalize_s += rhs.normalize_s;
        self.record_drive_s += rhs.record_drive_s;
        self.discard_candidate_analysis_s += rhs.discard_candidate_analysis_s;
        self.special_summary_s += rhs.special_summary_s;
        self.npz_write_s += rhs.npz_write_s;
    }
}

static EXPORT_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static EXPORT_CANCELLED: AtomicBool = AtomicBool::new(false);
static EXPORT_CANCEL_HANDLER_INSTALL: OnceLock<Result<(), String>> = OnceLock::new();
static EXPORT_TEST_FILE_SLEEP_MS: OnceLock<u64> = OnceLock::new();

thread_local! {
    static ACTIVE_EXPORT_TIMINGS: RefCell<Option<ExportStageTimings>> = const { RefCell::new(None) };
}

fn export_profile_enabled() -> bool {
    *EXPORT_PROFILE_ENABLED.get_or_init(|| {
        std::env::var_os("XMODEL1_EXPORT_PROFILE")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

fn export_test_file_sleep_ms() -> u64 {
    *EXPORT_TEST_FILE_SLEEP_MS.get_or_init(|| {
        std::env::var("XMODEL1_EXPORT_TEST_SLEEP_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

fn with_export_stage_timing<T>(stage: ExportStage, f: impl FnOnce() -> T) -> T {
    if !export_profile_enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let elapsed_s = t0.elapsed().as_secs_f64();
    ACTIVE_EXPORT_TIMINGS.with(|cell| {
        if let Some(timings) = cell.borrow_mut().as_mut() {
            timings.add_stage(stage, elapsed_s);
        }
    });
    out
}

fn begin_export_profile_scope() {
    if export_profile_enabled() {
        ACTIVE_EXPORT_TIMINGS.with(|cell| {
            *cell.borrow_mut() = Some(ExportStageTimings::default());
        });
    }
}

fn finish_export_profile_scope() -> ExportStageTimings {
    if !export_profile_enabled() {
        return ExportStageTimings::default();
    }
    ACTIVE_EXPORT_TIMINGS
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_default()
}

fn ensure_cancel_handler_installed() -> Result<(), String> {
    EXPORT_CANCEL_HANDLER_INSTALL
        .get_or_init(|| {
            ctrlc::set_handler(|| {
                EXPORT_CANCELLED.store(true, Ordering::SeqCst);
            })
            .map_err(|err| format!("failed to install xmodel1 preprocess signal handler: {err}"))
        })
        .clone()
}

fn reset_cancel_flag() {
    EXPORT_CANCELLED.store(false, Ordering::SeqCst);
}

fn cancel_requested() -> bool {
    EXPORT_CANCELLED.load(Ordering::SeqCst)
}

fn interrupted_error(context: &str) -> String {
    format!("{INTERRUPTED_ERROR_PREFIX}: {context}")
}

fn is_interrupted_error(err: &str) -> bool {
    err.starts_with(INTERRUPTED_ERROR_PREFIX)
}

fn poll_cancel(context: &str) -> Result<(), String> {
    if cancel_requested() {
        return Err(interrupted_error(context));
    }
    Ok(())
}

pub fn xmodel1_schema_info() -> (&'static str, u32, usize, usize, usize) {
    (
        XMODEL1_SCHEMA_NAME,
        XMODEL1_SCHEMA_VERSION,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
    )
}

pub fn validate_xmodel1_discard_record(
    chosen_candidate_idx: i16,
    candidate_mask: &[u8],
    candidate_tile_id: &[i16],
) -> Result<(), String> {
    validate_candidate_mask_and_choice(chosen_candidate_idx, candidate_mask, candidate_tile_id)
}

fn f16_bits_to_f32_vec(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|value| f16::from_bits(*value).to_f32())
        .collect()
}

fn default_rule_context() -> [f32; XMODEL1_RULE_CONTEXT_DIM] {
    [1.0, 0.5, 0.0, -1.5, 0.0, 1.0]
}

fn runtime_history_summary_from_snapshot(
    snapshot: &Value,
) -> Result<[f32; replay_core::HISTORY_SUMMARY_DIM], String> {
    let mut out = [0.0f32; replay_core::HISTORY_SUMMARY_DIM];
    let Some(values) = snapshot.get("history_summary") else {
        return Ok(out);
    };
    let array = values
        .as_array()
        .ok_or_else(|| "xmodel1 runtime snapshot history_summary must be an array".to_string())?;
    if array.len() != replay_core::HISTORY_SUMMARY_DIM {
        return Err(format!(
            "xmodel1 runtime snapshot history_summary len {} != {}",
            array.len(),
            replay_core::HISTORY_SUMMARY_DIM
        ));
    }
    for (idx, value) in array.iter().enumerate() {
        out[idx] = value.as_f64().ok_or_else(|| {
            format!("xmodel1 runtime snapshot history_summary[{idx}] must be numeric")
        })? as f32;
    }
    Ok(out)
}

fn runtime_apply_hand_count_delta(
    tracker: &mut replay_core::PlayerRoundTracker,
    tile34: usize,
    delta: i32,
) {
    let before = tracker.hand_counts34[tile34] as i32;
    let after = (before + delta).clamp(0, 4);
    tracker.pair_count = tracker.pair_count + usize::from(after >= 2) - usize::from(before >= 2);
    tracker.ankoutsu_count =
        tracker.ankoutsu_count + usize::from(after >= 3) - usize::from(before >= 3);
    tracker.hand_counts34[tile34] = after as u8;
}

fn runtime_apply_overall_tile_delta(
    tracker: &mut replay_core::PlayerRoundTracker,
    tile: &str,
    delta: i32,
) {
    let Some(tile34) = tile34_from_pai(tile) else {
        return;
    };
    let suit = match tile34 {
        0..=8 => 0usize,
        9..=17 => 1usize,
        18..=26 => 2usize,
        _ => 3usize,
    };
    if delta >= 0 {
        tracker.suit_counts[suit] = tracker.suit_counts[suit].saturating_add(delta as usize);
    } else {
        tracker.suit_counts[suit] = tracker.suit_counts[suit].saturating_sub((-delta) as usize);
    }
    if replay_core::tile_is_aka(tile) {
        if delta >= 0 {
            tracker.aka_counts[0] = tracker.aka_counts[0].saturating_add(delta as usize);
        } else {
            tracker.aka_counts[0] = tracker.aka_counts[0].saturating_sub((-delta) as usize);
        }
        match strip_aka(tile).as_str() {
            "5m" => {
                if delta >= 0 {
                    tracker.aka_counts[1] = tracker.aka_counts[1].saturating_add(delta as usize);
                } else {
                    tracker.aka_counts[1] = tracker.aka_counts[1].saturating_sub((-delta) as usize);
                }
            }
            "5p" => {
                if delta >= 0 {
                    tracker.aka_counts[2] = tracker.aka_counts[2].saturating_add(delta as usize);
                } else {
                    tracker.aka_counts[2] = tracker.aka_counts[2].saturating_sub((-delta) as usize);
                }
            }
            "5s" => {
                if delta >= 0 {
                    tracker.aka_counts[3] = tracker.aka_counts[3].saturating_add(delta as usize);
                } else {
                    tracker.aka_counts[3] = tracker.aka_counts[3].saturating_sub((-delta) as usize);
                }
            }
            _ => {}
        }
    }
}

fn runtime_parse_melds(values: &[Value]) -> Vec<replay_core::MeldInfo> {
    values
        .iter()
        .map(|meld| replay_core::MeldInfo {
            kind: meld
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            pai: meld
                .get("pai")
                .and_then(Value::as_str)
                .map(normalize_tile_repr),
            consumed: meld
                .get("consumed")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_tile_repr)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        })
        .collect()
}

fn runtime_build_round_state_from_snapshot(
    snapshot: &SnapshotCore,
    actor: usize,
) -> Result<RoundState, String> {
    if actor >= 4 {
        return Err(format!("xmodel1 runtime actor out of range: {actor}"));
    }
    if snapshot.hand.is_empty() {
        return Err(format!(
            "xmodel1 runtime snapshot has empty hand for actor {actor}"
        ));
    }
    let mut round_state = RoundState::default();
    round_state.bakaze = snapshot.bakaze.clone();
    round_state.kyoku = snapshot.kyoku as i8;
    round_state.honba = snapshot.honba as i8;
    round_state.kyotaku = snapshot.kyotaku as i8;
    round_state.oya = snapshot.oya as i8;
    for (idx, score) in snapshot.scores.iter().take(4).enumerate() {
        round_state.scores[idx] = *score;
    }
    round_state.dora_markers = snapshot
        .dora_markers
        .iter()
        .map(|tile| normalize_tile_repr(tile))
        .collect();
    round_state.last_tsumo = snapshot
        .last_tsumo
        .iter()
        .map(|tile| tile.as_ref().map(|value| normalize_tile_repr(value)))
        .collect();
    round_state.remaining_wall = snapshot.remaining_wall.unwrap_or(70).max(0);
    round_state.pending_rinshan_actor = snapshot.pending_rinshan_actor;
    round_state.in_game = true;
    round_state.round_step_index = snapshot
        .discards
        .iter()
        .map(|discards| discards.len())
        .sum::<usize>() as i32;
    round_state.round_terminal_finalized = false;

    let mut parsed_melds: Vec<Vec<replay_core::MeldInfo>> = Vec::with_capacity(4);
    for pid in 0..4 {
        let discards = snapshot.discards.get(pid).cloned().unwrap_or_default();
        let meld_values = snapshot.melds.get(pid).cloned().unwrap_or_default();
        let melds = runtime_parse_melds(&meld_values);
        parsed_melds.push(melds.clone());
        round_state.players[pid] = replay_core::PlayerState {
            discards: discards
                .iter()
                .filter_map(|discard| {
                    let tile = normalize_tile_repr(&discard.pai);
                    let tile34 = tile34_from_pai(&tile)?;
                    Some(replay_core::DiscardInfo {
                        tile34,
                        tsumogiri: discard.tsumogiri,
                        reach_declared: discard.reach_declared,
                    })
                })
                .collect(),
            melds,
            reached: snapshot.reached.get(pid).copied().unwrap_or(false),
            pending_reach: snapshot.pending_reach.get(pid).copied().unwrap_or(false),
            furiten: snapshot.furiten.get(pid).copied().unwrap_or(false),
        };
    }

    let mut actor_tracker = replay_core::PlayerRoundTracker::default();
    actor_tracker.hand_tiles = snapshot
        .hand
        .iter()
        .map(|tile| normalize_tile_repr(tile))
        .collect();
    if actor_tracker.hand_tiles.len() % 3 == 1 {
        if let Some(tsumo_pai) = snapshot.tsumo_pai.as_deref() {
            actor_tracker
                .hand_tiles
                .push(normalize_tile_repr(tsumo_pai));
        }
    }
    for tile in actor_tracker.hand_tiles.clone() {
        if let Some(tile34) = tile34_from_pai(&tile).map(|value| value as usize) {
            runtime_apply_hand_count_delta(&mut actor_tracker, tile34, 1);
        }
        runtime_apply_overall_tile_delta(&mut actor_tracker, &tile, 1);
    }
    for meld in &parsed_melds[actor] {
        for tile in meld.consumed.iter().chain(meld.pai.iter()) {
            actor_tracker.meld_tiles.push(tile.clone());
            if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
                actor_tracker.meld_counts34[tile34] =
                    actor_tracker.meld_counts34[tile34].saturating_add(1);
            }
            runtime_apply_overall_tile_delta(&mut actor_tracker, tile, 1);
        }
    }
    actor_tracker.discards_count = round_state.players[actor].discards.len();
    actor_tracker.meld_count = round_state.players[actor].melds.len();

    let mut visible_counts34 = [0u8; TILE_KIND_COUNT];
    for tile in &actor_tracker.hand_tiles {
        if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
            visible_counts34[tile34] = visible_counts34[tile34].saturating_add(1);
        }
    }
    for melds in &parsed_melds {
        for meld in melds {
            for tile in meld.consumed.iter().chain(meld.pai.iter()) {
                if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
                    visible_counts34[tile34] = visible_counts34[tile34].saturating_add(1);
                }
            }
        }
    }
    for discards in &snapshot.discards {
        for discard in discards {
            let tile = normalize_tile_repr(&discard.pai);
            if let Some(tile34) = tile34_from_pai(&tile).map(|value| value as usize) {
                visible_counts34[tile34] = visible_counts34[tile34].saturating_add(1);
            }
        }
    }
    for marker in &round_state.dora_markers {
        if let Some(tile34) = tile34_from_pai(marker).map(|value| value as usize) {
            visible_counts34[tile34] = visible_counts34[tile34].saturating_add(1);
        }
    }
    actor_tracker.visible_counts34 = visible_counts34;
    round_state.feature_tracker.players[actor] = actor_tracker;
    Ok(round_state)
}

pub fn build_xmodel1_runtime_tensors(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Result<Value, String> {
    let parsed_snapshot: SnapshotCore = serde_json::from_value(snapshot.clone())
        .map_err(|err| format!("failed to parse xmodel1 runtime snapshot: {err}"))?;
    let round_state = runtime_build_round_state_from_snapshot(&parsed_snapshot, actor)?;
    let history_summary = runtime_history_summary_from_snapshot(snapshot)?;
    let decision_ctx = DecisionAnalysisContext::new(&round_state, actor, true);

    let mut candidate_feat = vec![0u16; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_flags = vec![0u8; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM];

    for (slot, analysis) in decision_ctx.candidate_analyses.iter().enumerate() {
        let feat_offset = slot * XMODEL1_CANDIDATE_FEATURE_DIM;
        for (idx, value) in analysis.feat.iter().enumerate() {
            candidate_feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        let flags_offset = slot * XMODEL1_CANDIDATE_FLAG_DIM;
        candidate_flags[flags_offset..flags_offset + XMODEL1_CANDIDATE_FLAG_DIM]
            .copy_from_slice(&analysis.flags);
        candidate_tile_id[slot] = analysis.tile34;
        candidate_mask[slot] = 1;
    }

    let response = encode_response_candidate_arrays(
        snapshot,
        actor,
        legal_actions,
        &json!({"type":"dahai"}),
        None,
    )?;

    Ok(json!({
        "candidate_feat": f16_bits_to_f32_vec(&candidate_feat),
        "candidate_tile_id": candidate_tile_id,
        "candidate_mask": candidate_mask,
        "candidate_flags": candidate_flags,
        "response_action_idx": response.action_idx,
        "response_action_mask": response.mask,
        "response_post_candidate_feat": f16_bits_to_f32_vec(&response.post_candidate_feat),
        "response_post_candidate_tile_id": response.post_candidate_tile_id,
        "response_post_candidate_mask": response.post_candidate_mask,
        "response_post_candidate_flags": response.post_candidate_flags,
        "response_post_candidate_quality_score": response.post_candidate_quality,
        "response_post_candidate_hard_bad_flag": response.post_candidate_hard_bad,
        "response_teacher_discard_idx": response.teacher_discard_idx,
        "response_human_discard_idx": response.human_discard_idx,
        "history_summary": history_summary,
        "rule_context": default_rule_context(),
    }))
}

#[derive(Debug, Clone, Serialize)]
struct ExportManifest<'a> {
    schema_name: &'a str,
    schema_version: u32,
    max_candidates: usize,
    candidate_feature_dim: usize,
    candidate_flag_dim: usize,
    file_count: usize,
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    shard_file_counts: BTreeMap<String, usize>,
    shard_sample_counts: BTreeMap<String, usize>,
    used_fallback: bool,
    export_mode: &'a str,
    files: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ExportRunOptions {
    pub smoke: bool,
    pub resume: bool,
    pub progress_every: usize,
    pub jobs: usize,
    pub limit_files: usize,
}

type RoundState = replay_core::RoundState;

#[derive(Debug, Clone)]
struct FullRecord {
    state_tile_feat: Vec<u16>,
    state_scalar: Vec<u16>,
    candidate_feat: Vec<u16>,
    candidate_tile_id: Vec<i16>,
    candidate_mask: Vec<u8>,
    candidate_flags: Vec<u8>,
    chosen_candidate_idx: i16,
    candidate_quality: Vec<f32>,
    candidate_rank: Vec<i8>,
    candidate_hard_bad: Vec<u8>,
    response_action_idx: Vec<i16>,
    response_action_mask: Vec<u8>,
    chosen_response_action_idx: i16,
    response_post_candidate_feat: Vec<u16>,
    response_post_candidate_tile_id: Vec<i16>,
    response_post_candidate_mask: Vec<u8>,
    response_post_candidate_flags: Vec<u8>,
    response_post_candidate_quality: Vec<f32>,
    response_post_candidate_hard_bad: Vec<u8>,
    response_teacher_discard_idx: Vec<i16>,
    response_human_discard_idx: Vec<i16>,
    action_idx_target: i16,
    global_value_target: f32,
    score_delta_target: f32,
    win_target: f32,
    dealin_target: f32,
    pts_given_win_target: f32,
    pts_given_dealin_target: f32,
    // Stage 2: 决策时刻对手 tenpai 标签 (1.0 = 对手当前向听 ≤ 0)。
    // 顺序: [(actor+1)%4, (actor+2)%4, (actor+3)%4] —— 与 Python reference
    // `mahjong_env.replay._compute_opp_tenpai_target` 完全一致。
    opp_tenpai_target: [f32; 3],
    history_summary: [f32; replay_core::HISTORY_SUMMARY_DIM],
    offense_quality_target: f32,
    sample_type: i8,
    actor: i8,
    score_before_action: i32,
    final_score_delta_points_target: i32,
    final_rank_target: u8,
    event_index: i32,
    kyoku: i8,
    honba: i8,
    is_open_hand: u8,
    rule_context: [f32; XMODEL1_RULE_CONTEXT_DIM],
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

fn last_scores_from_events(events: &[Value]) -> Option<[i32; 4]> {
    let mut last: Option<[i32; 4]> = None;
    for event in events {
        let Some(scores) = event.get("scores").and_then(Value::as_array) else {
            continue;
        };
        if scores.len() < 4 {
            continue;
        }
        let mut out = [0i32; 4];
        for (idx, value) in scores.iter().take(4).enumerate() {
            out[idx] = value.as_i64().unwrap_or(0) as i32;
        }
        last = Some(out);
    }
    last
}

fn game_start_oya_from_events(events: &[Value]) -> usize {
    for event in events {
        if event.get("type").and_then(Value::as_str) != Some("start_kyoku") {
            continue;
        }
        return event
            .get("oya")
            .and_then(Value::as_i64)
            .map(|value| value.clamp(0, 3) as usize)
            .unwrap_or(0);
    }
    0
}

fn apply_final_game_targets(records: &mut [FullRecord], events: &[Value]) {
    let Some(final_scores) = last_scores_from_events(events) else {
        return;
    };
    let game_start_oya = game_start_oya_from_events(events);
    let rank_targets = final_rank_targets(&final_scores, game_start_oya);
    for record in records {
        let actor = record.actor.clamp(0, 3) as usize;
        record.final_score_delta_points_target = final_scores[actor] - record.score_before_action;
        record.final_rank_target = rank_targets[actor];
    }
}

#[derive(Debug, Clone, Copy)]
struct CandidateMetrics {
    quality: f32,
    rank_bucket: i8,
    hard_bad: u8,
}

#[derive(Debug, Clone, Copy)]
struct AfterStateStructureMetrics {
    yakuhai_pair_preserved: f32,
    dual_yakuhai_pair_value: f32,
    tanyao_path: f32,
    flush_path: f32,
    after_dora_count: f32,
    after_aka_count: f32,
    yakuhai_triplet_count: f32,
    confirmed_han_floor: f32,
}

#[derive(Debug, Clone, Copy)]
struct AfterStateValueMetrics {
    after_max_hand_value_norm: f32,
    hand_value_survives: f32,
}

#[derive(Debug, Clone)]
struct DiscardCandidateAnalysis {
    discard_tile: String,
    tile34: i16,
    feat: [f32; XMODEL1_CANDIDATE_FEATURE_DIM],
    flags: [u8; XMODEL1_CANDIDATE_FLAG_DIM],
    metrics: CandidateMetrics,
}

#[derive(Debug, Clone)]
struct DecisionAnalysisContext {
    before_counts34: [u8; TILE_KIND_COUNT],
    before_visible34: [u8; TILE_KIND_COUNT],
    before_progress: Summary3n1,
    before_shanten_raw: i8,
    before_decision_shanten: i8,
    before_decision_waits_count: u8,
    before_waits_live: usize,
    yakuhai: [bool; TILE_KIND_COUNT],
    dora_flags: [bool; TILE_KIND_COUNT],
    before_tanyao: f32,
    any_other_reached: bool,
    is_open_hand: bool,
    hand_aka_total: usize,
    risk_table: [[f32; 3]; TILE_KIND_COUNT],
    candidate_analyses: Vec<DiscardCandidateAnalysis>,
}

#[derive(Debug, Clone)]
struct SpecialCandidateArrays {
    feat: Vec<u16>,
    type_id: Vec<i16>,
    mask: Vec<u8>,
    quality: Vec<f32>,
    rank: Vec<i8>,
    hard_bad: Vec<u8>,
    chosen_idx: i16,
}

#[derive(Debug, Clone)]
struct ResponseCandidateArrays {
    action_idx: Vec<i16>,
    mask: Vec<u8>,
    chosen_idx: i16,
    post_candidate_feat: Vec<u16>,
    post_candidate_tile_id: Vec<i16>,
    post_candidate_mask: Vec<u8>,
    post_candidate_flags: Vec<u8>,
    post_candidate_quality: Vec<f32>,
    post_candidate_hard_bad: Vec<u8>,
    teacher_discard_idx: Vec<i16>,
    human_discard_idx: Vec<i16>,
}

#[derive(Debug, Clone, Copy)]
struct SpecialHandSummary {
    shanten: i8,
    tenpai: f32,
    waits_count: u8,
    waits_live_norm: f32,
    round_progress: f32,
    score_gap: f32,
    threat_proxy_any_reached: f32,
    threat_by_opponent: [f32; 3],
    is_open: f32,
    best_discard_quality_norm: f32,
    current_han_floor_norm: f32,
    current_dora_count_norm: f32,
    current_max_value_norm: f32,
    current_hand_value_survives: f32,
}

pub(crate) fn action_idx_from_action(action: &Value) -> i16 {
    let et = action.get("type").and_then(Value::as_str).unwrap_or("none");
    match et {
        "dahai" => action
            .get("pai")
            .and_then(Value::as_str)
            .and_then(tile34_from_pai)
            .unwrap_or(0),
        "reach" => 34,
        "chi" => {
            let pai = action.get("pai").and_then(Value::as_str).unwrap_or("");
            let consumed = action
                .get("consumed")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut ranks: Vec<i32> = consumed
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|tile| {
                    strip_aka(tile)
                        .chars()
                        .next()
                        .and_then(|ch| ch.to_digit(10))
                        .map(|v| v as i32)
                })
                .collect();
            ranks.sort();
            let pai_rank = strip_aka(pai)
                .chars()
                .next()
                .and_then(|ch| ch.to_digit(10))
                .map(|v| v as i32)
                .unwrap_or(0);
            if ranks.len() >= 2 {
                if pai_rank < ranks[0] {
                    35
                } else if pai_rank < ranks[1] {
                    36
                } else {
                    37
                }
            } else {
                35
            }
        }
        "pon" => 38,
        "daiminkan" => 39,
        "ankan" => 40,
        "kakan" => 41,
        "hora" => 42,
        "ryukyoku" => 43,
        _ => 44,
    }
}

pub(crate) fn encode_state_features_for_actor(
    round_state: &replay_core::RoundState,
    actor: usize,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let before_counts34 = round_state.feature_tracker.players[actor].hand_counts34;
    let before_visible34 = visible_counts_for_decision(round_state, actor);
    let before_progress = analyze_progress_like_python(&before_counts34, &before_visible34);
    let tile = encode_state_tile_features(round_state, actor, &before_progress, &before_visible34)?;
    let scalar =
        encode_state_scalar_features(round_state, actor, &before_progress, &before_visible34)?;
    Ok((tile, scalar))
}

fn pair_taatsu_metrics_from_counts(counts: &[u8; TILE_KIND_COUNT]) -> (usize, usize) {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count();
    let mut taatsu_count = 0usize;
    for base in [0usize, 9, 18] {
        let suit = &counts[base..base + 9];
        for i in 0..8 {
            if suit[i] > 0 && suit[i + 1] > 0 {
                taatsu_count += 1;
            }
        }
        for i in 0..7 {
            if suit[i] > 0 && suit[i + 2] > 0 {
                taatsu_count += 1;
            }
        }
    }
    (pair_count, taatsu_count)
}

fn tanyao_path_from_counts(counts: &[u8; TILE_KIND_COUNT]) -> f32 {
    f32::from(counts.iter().enumerate().all(|(idx, count)| {
        *count == 0
            || !matches!(
                idx,
                0 | 8 | 9 | 17 | 18 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33
            )
    }))
}

fn flush_path_from_counts(counts: &[u8; TILE_KIND_COUNT]) -> f32 {
    let suit_presence = [
        counts[0..9].iter().sum::<u8>() > 0,
        counts[9..18].iter().sum::<u8>() > 0,
        counts[18..27].iter().sum::<u8>() > 0,
    ];
    f32::from(suit_presence.iter().filter(|flag| **flag).count() <= 1)
}

fn pinfu_like_path_from_counts(
    counts: &[u8; TILE_KIND_COUNT],
    yakuhai: &[bool; TILE_KIND_COUNT],
    is_open_hand: bool,
) -> f32 {
    if is_open_hand || counts.iter().any(|count| *count >= 3) {
        return 0.0;
    }
    let pair_tiles: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(idx, count)| (*count >= 2).then_some(idx))
        .collect();
    f32::from(pair_tiles.len() == 1 && !yakuhai[pair_tiles[0]])
}

fn iipeikou_like_path_from_counts(counts: &[u8; TILE_KIND_COUNT], is_open_hand: bool) -> f32 {
    if is_open_hand {
        return 0.0;
    }
    for base in [0usize, 9, 18] {
        let suit = &counts[base..base + 9];
        for idx in 0..7 {
            if suit[idx].min(suit[idx + 1]).min(suit[idx + 2]) >= 2 {
                return 1.0;
            }
        }
    }
    0.0
}

fn per_opponent_dealin_risks(
    round_state: &RoundState,
    actor: usize,
    tile34: usize,
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> [f32; 3] {
    let live_factor = ((4.0 - visible_counts34[tile34] as f32) / 4.0).clamp(0.0, 1.0);
    let mut risks = [0.0; 3];
    for rel in 1..=3 {
        let opp = (actor + rel) % 4;
        let open_pressure = f32::from(
            round_state.players[opp]
                .melds
                .iter()
                .any(|meld| meld.kind != "ankan"),
        );
        let mut base = if round_state.players[opp].reached {
            1.0
        } else {
            0.2 + 0.15 * open_pressure
        };
        if tile34 >= 27 {
            base *= 0.9;
        }
        risks[rel - 1] = (live_factor * base).clamp(0.0, 1.0);
    }
    risks
}

fn tile_risk_table(
    round_state: &RoundState,
    actor: usize,
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> [[f32; 3]; TILE_KIND_COUNT] {
    let mut table = [[0.0; 3]; TILE_KIND_COUNT];
    for tile34 in 0..TILE_KIND_COUNT {
        table[tile34] = per_opponent_dealin_risks(round_state, actor, tile34, visible_counts34);
    }
    table
}

fn pair_taatsu_ankoutsu_metrics_from_counts(
    counts: &[u8; TILE_KIND_COUNT],
) -> (usize, usize, usize) {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count();
    let ankoutsu_count = counts.iter().filter(|&&count| count >= 3).count();
    let (_, taatsu_count) = pair_taatsu_metrics_from_counts(counts);
    (pair_count, taatsu_count, ankoutsu_count)
}

fn analyze_progress_like_python(
    hand_counts34: &[u8; TILE_KIND_COUNT],
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> Summary3n1 {
    summarize_like_python(hand_counts34, visible_counts34)
}

fn visible_counts_for_decision(round_state: &RoundState, actor: usize) -> [u8; TILE_KIND_COUNT] {
    round_state.feature_tracker.players[actor].visible_counts34
}

fn chiitoi_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    let pair_count = counts.iter().filter(|&&count| count >= 2).count() as i32;
    let distinct_count = counts.iter().filter(|&&count| count > 0).count() as i32;
    6 - pair_count + (7 - distinct_count).max(0)
}

fn kokushi_shanten(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    const YAOCHUU: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let unique_count = YAOCHUU.iter().filter(|&&tile| counts[tile] > 0).count() as i32;
    let has_pair = YAOCHUU.iter().any(|&tile| counts[tile] >= 2);
    13 - unique_count - i32::from(has_pair)
}

fn is_suited_sequence_start(tile34: usize) -> bool {
    tile34 < 27 && (tile34 % 9) <= 6
}

fn is_complete_regular_counts_for_waits(
    counts: &mut [u8; TILE_KIND_COUNT],
    melds_needed: u8,
    need_pair: bool,
) -> bool {
    let Some(first) = counts.iter().position(|&count| count > 0) else {
        return melds_needed == 0 && !need_pair;
    };
    if need_pair && counts[first] >= 2 {
        counts[first] -= 2;
        if is_complete_regular_counts_for_waits(counts, melds_needed, false) {
            counts[first] += 2;
            return true;
        }
        counts[first] += 2;
    }
    if melds_needed > 0 && counts[first] >= 3 {
        counts[first] -= 3;
        if is_complete_regular_counts_for_waits(counts, melds_needed - 1, need_pair) {
            counts[first] += 3;
            return true;
        }
        counts[first] += 3;
    }
    if melds_needed > 0
        && is_suited_sequence_start(first)
        && counts[first + 1] > 0
        && counts[first + 2] > 0
    {
        counts[first] -= 1;
        counts[first + 1] -= 1;
        counts[first + 2] -= 1;
        if is_complete_regular_counts_for_waits(counts, melds_needed - 1, need_pair) {
            counts[first] += 1;
            counts[first + 1] += 1;
            counts[first + 2] += 1;
            return true;
        }
        counts[first] += 1;
        counts[first + 1] += 1;
        counts[first + 2] += 1;
    }
    false
}

fn find_regular_waits_tiles(counts34: &[u8; TILE_KIND_COUNT]) -> [bool; TILE_KIND_COUNT] {
    let mut waits = [false; TILE_KIND_COUNT];
    let tile_count: usize = counts34.iter().map(|&count| count as usize).sum();
    if tile_count % 3 != 1 {
        return waits;
    }
    let melds_needed = (((tile_count + 1) - 2) / 3) as u8;
    for tile34 in 0..TILE_KIND_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        waits[tile34] = is_complete_regular_counts_for_waits(&mut work, melds_needed, true);
    }
    waits
}

fn find_special_waits_tiles(counts34: &[u8; TILE_KIND_COUNT]) -> (i32, [bool; TILE_KIND_COUNT]) {
    let chiitoi = chiitoi_shanten(counts34);
    let kokushi = kokushi_shanten(counts34);
    let special_shanten = chiitoi.min(kokushi);
    let mut waits = [false; TILE_KIND_COUNT];
    if special_shanten != 0 {
        return (special_shanten, waits);
    }
    for tile34 in 0..TILE_KIND_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        if (chiitoi == 0 && chiitoi_shanten(&work) == -1)
            || (kokushi == 0 && kokushi_shanten(&work) == -1)
        {
            waits[tile34] = true;
        }
    }
    (special_shanten, waits)
}

fn calc_shanten_waits_like_python(
    counts34: &[u8; TILE_KIND_COUNT],
    has_melds: bool,
) -> (i8, u8, [bool; TILE_KIND_COUNT]) {
    let tile_count: u8 = counts34.iter().sum();
    let regular_shanten = crate::shanten_table::calc_shanten_all(counts34, tile_count / 3);
    let mut shanten = regular_shanten;
    let mut waits = if regular_shanten == 0 && tile_count % 3 == 1 {
        find_regular_waits_tiles(counts34)
    } else {
        [false; TILE_KIND_COUNT]
    };
    if !has_melds {
        let (special_shanten, special_waits) = find_special_waits_tiles(counts34);
        if special_shanten < shanten as i32 {
            shanten = special_shanten as i8;
            waits = if special_shanten == 0 {
                special_waits
            } else {
                [false; TILE_KIND_COUNT]
            };
        } else if special_shanten == shanten as i32 && shanten == 0 {
            for tile34 in 0..TILE_KIND_COUNT {
                waits[tile34] |= special_waits[tile34];
            }
        }
    }
    let waits_count = waits.iter().filter(|flag| **flag).count() as u8;
    (shanten, waits_count, waits)
}

fn score_rank(scores: &[i32; 4], actor_score: i32) -> usize {
    scores.iter().filter(|&&score| score > actor_score).count()
}

fn dora_next(marker: &str) -> Option<&'static str> {
    match strip_aka(marker).as_str() {
        "1m" => Some("2m"),
        "2m" => Some("3m"),
        "3m" => Some("4m"),
        "4m" => Some("5m"),
        "5m" => Some("6m"),
        "6m" => Some("7m"),
        "7m" => Some("8m"),
        "8m" => Some("9m"),
        "9m" => Some("1m"),
        "1p" => Some("2p"),
        "2p" => Some("3p"),
        "3p" => Some("4p"),
        "4p" => Some("5p"),
        "5p" => Some("6p"),
        "6p" => Some("7p"),
        "7p" => Some("8p"),
        "8p" => Some("9p"),
        "9p" => Some("1p"),
        "1s" => Some("2s"),
        "2s" => Some("3s"),
        "3s" => Some("4s"),
        "4s" => Some("5s"),
        "5s" => Some("6s"),
        "6s" => Some("7s"),
        "7s" => Some("8s"),
        "8s" => Some("9s"),
        "9s" => Some("1s"),
        "E" => Some("S"),
        "S" => Some("W"),
        "W" => Some("N"),
        "N" => Some("E"),
        "P" => Some("F"),
        "F" => Some("C"),
        "C" => Some("P"),
        _ => None,
    }
}

fn yakuhai_tiles(bakaze: &str, actor: i8, oya: i8) -> [bool; TILE_KIND_COUNT] {
    let mut flags = [false; TILE_KIND_COUNT];
    flags[31] = true;
    flags[32] = true;
    flags[33] = true;
    match bakaze {
        "E" => flags[27] = true,
        "S" => flags[28] = true,
        "W" => flags[29] = true,
        "N" => flags[30] = true,
        _ => {}
    }
    let jikaze = 27 + ((actor - oya).rem_euclid(4) as usize);
    if jikaze < TILE_KIND_COUNT {
        flags[jikaze] = true;
    }
    flags
}

fn summarize_after_discard(
    counts34: &[u8; TILE_KIND_COUNT],
    visible_counts34: &[u8; TILE_KIND_COUNT],
) -> Summary3n1 {
    summarize_3n1(counts34, visible_counts34)
}

fn sorted_candidate_tiles(hand_tiles: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tile in hand_tiles {
        let normalized = normalize_tile_repr(tile);
        if !out.iter().any(|value| value == &normalized) {
            out.push(normalized);
        }
    }
    out.sort_by(|lhs, rhs| {
        let lhs_tile34 = tile34_from_pai(&strip_aka(lhs)).unwrap_or(99);
        let rhs_tile34 = tile34_from_pai(&strip_aka(rhs)).unwrap_or(99);
        lhs_tile34
            .cmp(&rhs_tile34)
            .then_with(|| strip_aka(lhs).cmp(&strip_aka(rhs)))
            .then_with(|| lhs.cmp(rhs))
    });
    out
}

fn candidate_metrics(
    after_shanten: i8,
    after_max_hand_value_norm: f32,
    good_shape_ukeire_live_norm: f32,
    improvement_live_norm: f32,
    confirmed_han_floor_norm: f32,
    break_tenpai: u8,
    break_meld_structure: u8,
    yaku_break_any: bool,
    mean_risk: f32,
    discard_dead: f32,
) -> CandidateMetrics {
    let quality = 1.5 * f32::from(after_shanten == 0) - 0.8 * f32::from(after_shanten)
        + 0.5 * after_max_hand_value_norm
        + 0.2 * good_shape_ukeire_live_norm
        + 0.15 * improvement_live_norm
        + 0.15 * confirmed_han_floor_norm
        - 1.2 * f32::from(break_tenpai)
        - 0.5 * f32::from(yaku_break_any)
        - 0.7 * mean_risk
        - 0.15 * discard_dead;
    let hard_bad = u8::from(break_tenpai == 1 || break_meld_structure == 1 || mean_risk > 0.75);
    let rank_bucket = if hard_bad == 1 {
        0
    } else if quality >= 1.0 {
        3
    } else if quality >= 0.0 {
        2
    } else {
        1
    };
    CandidateMetrics {
        quality,
        rank_bucket,
        hard_bad,
    }
}

fn dora_target_flags(round_state: &RoundState) -> [bool; TILE_KIND_COUNT] {
    let mut flags = [false; TILE_KIND_COUNT];
    for marker in &round_state.dora_markers {
        if let Some(tile34) = dora_next(marker)
            .and_then(tile34_from_pai)
            .map(|value| value as usize)
        {
            if tile34 < TILE_KIND_COUNT {
                flags[tile34] = true;
            }
        }
    }
    flags
}

fn after_state_structure_metrics(
    after_counts34: &[u8; TILE_KIND_COUNT],
    yakuhai: &[bool; TILE_KIND_COUNT],
    dora_flags: &[bool; TILE_KIND_COUNT],
    after_aka_total: usize,
) -> AfterStateStructureMetrics {
    let yakuhai_pair_preserved = f32::from(
        yakuhai
            .iter()
            .enumerate()
            .any(|(idx, flag)| *flag && after_counts34[idx] >= 2),
    );
    let dual_yakuhai_pair_value = f32::from(
        yakuhai
            .iter()
            .enumerate()
            .filter(|(idx, flag)| **flag && after_counts34[*idx] >= 2)
            .count()
            >= 2,
    );
    let tanyao_path = tanyao_path_from_counts(after_counts34);
    let flush_path = flush_path_from_counts(after_counts34);
    let after_aka_count = after_aka_total as f32;
    let after_dora_count = after_counts34
        .iter()
        .enumerate()
        .filter(|(idx, _)| dora_flags[*idx])
        .map(|(_, count)| *count as usize)
        .sum::<usize>() as f32
        + after_aka_count;
    let yakuhai_triplet_count = yakuhai
        .iter()
        .enumerate()
        .filter(|(idx, flag)| **flag && after_counts34[*idx] >= 3)
        .count() as f32;
    let confirmed_han_floor =
        shared_confirmed_han_floor(yakuhai_triplet_count, tanyao_path, after_dora_count);
    AfterStateStructureMetrics {
        yakuhai_pair_preserved,
        dual_yakuhai_pair_value,
        tanyao_path,
        flush_path,
        after_dora_count,
        after_aka_count,
        yakuhai_triplet_count,
        confirmed_han_floor,
    }
}

fn exact_after_state_value_metrics(
    structure: &AfterStateStructureMetrics,
    after_waits_count: u8,
    after_waits_live: usize,
    is_open_hand: bool,
) -> AfterStateValueMetrics {
    let after_max_hand_value_norm = tenpai_value_proxy_norm(
        structure.confirmed_han_floor
            + 0.5 * structure.after_dora_count
            + 0.35 * structure.after_aka_count,
        after_waits_count,
        after_waits_live,
        after_waits_count > 0,
    );
    let hand_value_survives = f32::from(!is_open_hand || structure.confirmed_han_floor > 0.0);
    AfterStateValueMetrics {
        after_max_hand_value_norm,
        hand_value_survives,
    }
}

fn baseline_after_state_value_metrics(
    structure: &AfterStateStructureMetrics,
    is_open_hand: bool,
) -> AfterStateValueMetrics {
    let after_max_hand_value_norm = tenpai_value_proxy_norm(
        structure.confirmed_han_floor
            + 0.5 * structure.after_dora_count
            + 0.35 * structure.after_aka_count,
        0,
        0,
        false,
    );
    let hand_value_survives = f32::from(!is_open_hand || structure.confirmed_han_floor > 0.0);
    AfterStateValueMetrics {
        after_max_hand_value_norm,
        hand_value_survives,
    }
}

fn deep_shanten_after_state_value_metrics(
    structure: &AfterStateStructureMetrics,
    ukeire_live_count: usize,
    pair_count: usize,
    taatsu_count: usize,
    ankoutsu_count: usize,
    is_open_hand: bool,
) -> AfterStateValueMetrics {
    let structure_density = ((pair_count + taatsu_count + ankoutsu_count) as f32 / 10.0).min(1.0);
    let ukeire_live_norm = (ukeire_live_count as f32 / 136.0).min(1.0);
    let dora_norm = (structure.after_dora_count / 4.0).min(1.0);
    let confirmed_han_floor_norm = (structure.confirmed_han_floor / 8.0).min(1.0);
    let after_max_hand_value_norm = (0.35 * ukeire_live_norm
        + 0.22 * structure_density
        + 0.15 * dora_norm
        + 0.1 * structure.yakuhai_pair_preserved
        + 0.08 * structure.flush_path
        + 0.1 * confirmed_han_floor_norm)
        .clamp(0.0, 1.0);
    let hand_value_survives = f32::from(
        !is_open_hand
            || structure.confirmed_han_floor > 0.0
            || structure.flush_path > 0.5
            || structure.yakuhai_pair_preserved > 0.5
            || structure.after_dora_count > 0.0,
    );
    AfterStateValueMetrics {
        after_max_hand_value_norm,
        hand_value_survives,
    }
}

fn deep_shanten_candidate_metrics(
    after_shanten: i8,
    after_max_hand_value_norm: f32,
    confirmed_han_floor_norm: f32,
    ukeire_live_count: usize,
    pair_count: usize,
    taatsu_count: usize,
    ankoutsu_count: usize,
    flush_path: f32,
    yakuhai_pair_preserved: f32,
    break_tenpai: u8,
    break_meld_structure: u8,
    yaku_break_any: bool,
    mean_risk: f32,
    discard_dead: f32,
) -> CandidateMetrics {
    let structure_density = ((pair_count + taatsu_count + ankoutsu_count) as f32 / 10.0).min(1.0);
    let ukeire_live_norm = (ukeire_live_count as f32 / 136.0).min(1.0);
    let quality = -0.55 * f32::from(after_shanten)
        + 0.95 * ukeire_live_norm
        + 0.5 * structure_density
        + 0.22 * flush_path
        + 0.18 * yakuhai_pair_preserved
        + 0.35 * after_max_hand_value_norm
        + 0.12 * confirmed_han_floor_norm
        - 1.2 * f32::from(break_tenpai)
        - 0.35 * f32::from(break_meld_structure)
        - 0.45 * f32::from(yaku_break_any)
        - 0.65 * mean_risk
        - 0.15 * discard_dead;
    let hard_bad = u8::from(
        break_tenpai == 1 || break_meld_structure == 1 || mean_risk > 0.75 || after_shanten >= 5,
    );
    let rank_bucket = if hard_bad == 1 {
        0
    } else if quality >= 1.0 {
        3
    } else if quality >= 0.0 {
        2
    } else {
        1
    };
    CandidateMetrics {
        quality,
        rank_bucket,
        hard_bad,
    }
}

fn hand_aka_count(hand_tiles: &[String]) -> usize {
    hand_tiles
        .iter()
        .filter(|tile| replay_core::tile_is_aka(tile))
        .count()
}

impl DecisionAnalysisContext {
    fn new(round_state: &RoundState, actor: usize, include_candidate_analyses: bool) -> Self {
        let tracker = &round_state.feature_tracker.players[actor];
        let before_counts34 = tracker.hand_counts34;
        let before_visible34 = visible_counts_for_decision(round_state, actor);
        let before_progress = analyze_progress_like_python(&before_counts34, &before_visible34);
        let before_shanten_raw = before_progress.shanten;
        let before_decision_shanten = before_progress.shanten;
        let before_decision_waits_count = before_progress.waits_count;
        let before_waits_live: usize = before_progress
            .waits_tiles
            .iter()
            .enumerate()
            .filter(|(_, flag)| **flag)
            .map(|(tile34, _)| usize::from(4u8.saturating_sub(before_visible34[tile34])))
            .sum();
        let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
        let is_open_hand = !round_state.players[actor].melds.is_empty();
        let before_tanyao = tanyao_path_from_counts(&before_counts34);
        let mut ctx = Self {
            before_counts34,
            before_visible34,
            before_progress,
            before_shanten_raw,
            before_decision_shanten,
            before_decision_waits_count,
            before_waits_live,
            yakuhai,
            dora_flags: dora_target_flags(round_state),
            before_tanyao,
            any_other_reached: round_state
                .players
                .iter()
                .enumerate()
                .any(|(pid, player)| pid != actor && player.reached),
            is_open_hand,
            hand_aka_total: hand_aka_count(&tracker.hand_tiles),
            risk_table: tile_risk_table(round_state, actor, &before_visible34),
            candidate_analyses: Vec::new(),
        };
        if include_candidate_analyses {
            let candidate_tiles = sorted_candidate_tiles(&tracker.hand_tiles);
            ctx.candidate_analyses =
                with_export_stage_timing(ExportStage::DiscardCandidateAnalysis, || {
                    candidate_tiles
                        .iter()
                        .take(XMODEL1_MAX_CANDIDATES)
                        .filter_map(|discard_tile| analyze_discard_candidate(discard_tile, &ctx))
                        .collect()
                });
        }
        ctx
    }

    fn best_discard_quality_for_legal_actions(&self, legal_actions: &[Value]) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for action in legal_actions {
            if action.get("type").and_then(Value::as_str) != Some("dahai") {
                continue;
            }
            let Some(pai) = action.get("pai").and_then(Value::as_str) else {
                continue;
            };
            let normalized = normalize_tile_repr(pai);
            if let Some(analysis) = self
                .candidate_analyses
                .iter()
                .find(|analysis| analysis.discard_tile == normalized)
            {
                best = best.max(analysis.metrics.quality);
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    }
}

fn analyze_discard_candidate(
    discard_tile: &str,
    ctx: &DecisionAnalysisContext,
) -> Option<DiscardCandidateAnalysis> {
    let tile34 = tile34_from_pai(discard_tile)?;
    let discard_idx = tile34 as usize;
    let mut after_counts34 = ctx.before_counts34;
    if after_counts34[discard_idx] == 0 {
        return None;
    }
    after_counts34[discard_idx] -= 1;

    let after_progress = summarize_after_discard(&after_counts34, &ctx.before_visible34);
    let after_shanten_raw = after_progress.shanten;
    let after_waits_count_raw = after_progress.waits_count;
    let after_waits_live: usize = after_progress
        .waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(ctx.before_visible34[tile34])))
        .sum();
    let (pair_count, taatsu_count, ankoutsu_count) =
        pair_taatsu_ankoutsu_metrics_from_counts(&after_counts34);
    let break_tenpai = u8::from(ctx.before_shanten_raw == 0 && after_shanten_raw > 0);
    let break_best_wait =
        u8::from(ctx.before_shanten_raw == 0 && after_waits_live < ctx.before_waits_live);
    let break_meld_structure =
        u8::from(ctx.before_shanten_raw <= 1 && after_shanten_raw > ctx.before_shanten_raw);
    let is_aka = u8::from(replay_core::tile_is_aka(discard_tile));
    let after_aka_total = ctx.hand_aka_total.saturating_sub(is_aka as usize);
    let structure = after_state_structure_metrics(
        &after_counts34,
        &ctx.yakuhai,
        &ctx.dora_flags,
        after_aka_total,
    );
    let yaku_break_tanyao = u8::from(ctx.before_tanyao > 0.5 && structure.tanyao_path < 0.5);
    let yaku_break_pinfu = 0;
    let yaku_break_iipeikou = 0;
    let after_risks = ctx.risk_table[discard_idx];
    let discard_dead = f32::from(ctx.before_visible34[discard_idx] >= 3);
    let wait_density = if after_waits_count_raw > 0 {
        (after_waits_live as f32 / (4.0 * after_waits_count_raw as f32).max(1.0)).min(1.0)
    } else {
        0.0
    };
    let ukeire_live_norm = (after_progress.ukeire_live_count as f32 / 136.0).min(1.0);
    let good_shape_ukeire_live_norm = if after_shanten_raw <= 1 {
        ukeire_live_norm
    } else {
        0.0
    };
    let improvement_live_norm = if after_shanten_raw > 0 {
        ukeire_live_norm
    } else {
        0.0
    };
    let structure_density = ((pair_count + taatsu_count) as f32 / 8.0).min(1.0);
    let is_dora = u8::from(ctx.dora_flags[discard_idx] || is_aka == 1);
    let is_yakuhai = u8::from(ctx.yakuhai[discard_idx]);
    let confirmed_han_floor_norm = (structure.confirmed_han_floor / 8.0).min(1.0);
    let value_metrics = if after_shanten_raw >= DEEP_SHANTEN_FAST_PATH_CUTOFF {
        deep_shanten_after_state_value_metrics(
            &structure,
            after_progress.ukeire_live_count as usize,
            pair_count,
            taatsu_count,
            ankoutsu_count,
            ctx.is_open_hand,
        )
    } else {
        exact_after_state_value_metrics(
            &structure,
            after_waits_count_raw,
            after_waits_live,
            ctx.is_open_hand,
        )
    };
    let metrics = if after_shanten_raw >= DEEP_SHANTEN_FAST_PATH_CUTOFF {
        deep_shanten_candidate_metrics(
            after_shanten_raw,
            value_metrics.after_max_hand_value_norm,
            confirmed_han_floor_norm,
            after_progress.ukeire_live_count as usize,
            pair_count,
            taatsu_count,
            ankoutsu_count,
            structure.flush_path,
            structure.yakuhai_pair_preserved,
            break_tenpai,
            break_meld_structure,
            yaku_break_tanyao == 1 || yaku_break_pinfu == 1 || yaku_break_iipeikou == 1,
            (after_risks[0] + after_risks[1] + after_risks[2]) / 3.0,
            discard_dead,
        )
    } else {
        candidate_metrics(
            after_shanten_raw,
            value_metrics.after_max_hand_value_norm,
            good_shape_ukeire_live_norm,
            improvement_live_norm,
            confirmed_han_floor_norm,
            break_tenpai,
            break_meld_structure,
            yaku_break_tanyao == 1 || yaku_break_pinfu == 1 || yaku_break_iipeikou == 1,
            (after_risks[0] + after_risks[1] + after_risks[2]) / 3.0,
            discard_dead,
        )
    };
    let feat = [
        f32::from(after_shanten_raw) / 8.0,
        f32::from(after_shanten_raw == 0),
        after_waits_count_raw as f32 / 34.0,
        (after_progress.ukeire_live_count as f32 / 136.0).min(1.0),
        value_metrics.after_max_hand_value_norm,
        (((after_shanten_raw - ctx.before_shanten_raw) as f32) / 4.0).clamp(-1.0, 1.0),
        (structure.after_dora_count / 4.0).min(1.0),
        after_risks[0],
        after_risks[1],
        after_risks[2],
        wait_density,
        pair_count as f32 / 7.0,
        taatsu_count as f32 / 6.0,
        structure_density,
        value_metrics.hand_value_survives,
        structure.yakuhai_pair_preserved,
        structure.tanyao_path,
        structure.flush_path,
        discard_dead,
        good_shape_ukeire_live_norm,
        improvement_live_norm,
        confirmed_han_floor_norm,
    ];
    let flags = [
        break_tenpai,
        break_best_wait,
        break_meld_structure,
        yaku_break_tanyao,
        yaku_break_pinfu,
        yaku_break_iipeikou,
        is_dora,
        is_yakuhai,
    ];

    Some(DiscardCandidateAnalysis {
        discard_tile: normalize_tile_repr(discard_tile),
        tile34,
        feat,
        flags,
        metrics,
    })
}

fn special_call_family_score(action: &Value) -> f32 {
    let consumed_len = action
        .get("consumed")
        .and_then(Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0) as f32;
    let bonus = match action.get("type").and_then(Value::as_str).unwrap_or("none") {
        "pon" => 0.2,
        "daiminkan" | "ankan" | "kakan" => 0.1,
        _ => 0.0,
    };
    consumed_len + bonus
}

fn chi_special_type(action: &Value) -> i16 {
    let pai_rank = action
        .get("pai")
        .and_then(Value::as_str)
        .map(normalize_tile_repr)
        .and_then(|tile| tile.chars().next())
        .and_then(|ch| ch.to_digit(10))
        .unwrap_or(0);
    let mut consumed_ranks: Vec<u32> = action
        .get("consumed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .filter_map(|tile| tile.chars().next().and_then(|ch| ch.to_digit(10)))
        .collect();
    consumed_ranks.sort_unstable();
    if consumed_ranks.len() < 2 || pai_rank < consumed_ranks[0] {
        return XMODEL1_SPECIAL_TYPE_CHI_LOW;
    }
    if pai_rank < consumed_ranks[1] {
        return XMODEL1_SPECIAL_TYPE_CHI_MID;
    }
    XMODEL1_SPECIAL_TYPE_CHI_HIGH
}

fn special_type_from_action(action: &Value, include_terminal_actions: bool) -> Option<i16> {
    match action.get("type").and_then(Value::as_str).unwrap_or("none") {
        "chi" => Some(chi_special_type(action)),
        "pon" => Some(XMODEL1_SPECIAL_TYPE_PON),
        "daiminkan" => Some(XMODEL1_SPECIAL_TYPE_DAIMINKAN),
        "ankan" => Some(XMODEL1_SPECIAL_TYPE_ANKAN),
        "kakan" => Some(XMODEL1_SPECIAL_TYPE_KAKAN),
        "hora" if include_terminal_actions => Some(XMODEL1_SPECIAL_TYPE_HORA),
        "ryukyoku" if include_terminal_actions => Some(XMODEL1_SPECIAL_TYPE_RYUKYOKU),
        "none" => Some(XMODEL1_SPECIAL_TYPE_NONE),
        _ => None,
    }
}

fn is_response_action(action: &Value) -> bool {
    matches!(
        action.get("type").and_then(Value::as_str).unwrap_or("none"),
        "reach" | "hora" | "chi" | "pon" | "daiminkan" | "ankan" | "kakan" | "ryukyoku" | "none"
    )
}

fn response_action_requires_post_discard(action: &Value) -> bool {
    matches!(
        action.get("type").and_then(Value::as_str).unwrap_or("none"),
        "reach" | "chi" | "pon" | "daiminkan"
    )
}

fn response_action_sort_key(action: &Value) -> (i16, String, Vec<String>) {
    let mut consumed = action
        .get("consumed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .collect::<Vec<_>>();
    consumed.sort_unstable();
    (
        action_idx_from_action(action),
        action
            .get("pai")
            .and_then(Value::as_str)
            .map(normalize_tile_repr)
            .unwrap_or_default(),
        consumed,
    )
}

fn normalized_consumed_tiles(action: &Value) -> Vec<String> {
    let mut consumed = action
        .get("consumed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .collect::<Vec<_>>();
    consumed.sort_unstable();
    consumed
}

fn response_actions_equivalent(left: &Value, right: &Value) -> bool {
    let left_type = left.get("type").and_then(Value::as_str).unwrap_or("none");
    let right_type = right.get("type").and_then(Value::as_str).unwrap_or("none");
    if left_type != right_type {
        return false;
    }
    if matches!(left_type, "none" | "ryukyoku" | "reach") {
        return true;
    }
    if left_type == "hora" {
        let left_target = left.get("target").and_then(Value::as_i64);
        let right_target = right.get("target").and_then(Value::as_i64);
        if left_target != right_target {
            return false;
        }
        let left_pai = left
            .get("pai")
            .and_then(Value::as_str)
            .map(normalize_tile_repr);
        let right_pai = right
            .get("pai")
            .and_then(Value::as_str)
            .map(normalize_tile_repr);
        return left_pai == right_pai || left_pai.is_none() || right_pai.is_none();
    }
    let left_target = left.get("target").and_then(Value::as_i64);
    let right_target = right.get("target").and_then(Value::as_i64);
    if left_target != right_target {
        return false;
    }
    let left_pai = left
        .get("pai")
        .and_then(Value::as_str)
        .map(normalize_tile_repr);
    let right_pai = right
        .get("pai")
        .and_then(Value::as_str)
        .map(normalize_tile_repr);
    if left_pai != right_pai {
        return false;
    }
    normalized_consumed_tiles(left) == normalized_consumed_tiles(right)
}

fn human_response_post_discard_event(
    events: &[Value],
    event_index: i32,
    actor: usize,
    chosen_action: &Value,
) -> Option<Value> {
    let chosen_type = chosen_action
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("none");
    if !matches!(chosen_type, "reach" | "chi" | "pon" | "daiminkan") {
        return None;
    }
    let start = usize::try_from(event_index).ok()?.saturating_add(1);
    for event in events.iter().skip(start) {
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        if matches!(et, "reach_accepted" | "dora" | "new_dora" | "tsumo") {
            continue;
        }
        if et == "dahai" && replay_core::value_usize(event, "actor") == Some(actor) {
            return Some(event.clone());
        }
        return None;
    }
    None
}

fn empty_response_candidate_arrays() -> ResponseCandidateArrays {
    ResponseCandidateArrays {
        action_idx: vec![-1i16; XMODEL1_MAX_RESPONSE_CANDIDATES],
        mask: vec![0u8; XMODEL1_MAX_RESPONSE_CANDIDATES],
        chosen_idx: -1,
        post_candidate_feat: vec![
            0u16;
            XMODEL1_MAX_RESPONSE_CANDIDATES
                * XMODEL1_MAX_CANDIDATES
                * XMODEL1_CANDIDATE_FEATURE_DIM
        ],
        post_candidate_tile_id: vec![
            -1i16;
            XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES
        ],
        post_candidate_mask: vec![0u8; XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES],
        post_candidate_flags: vec![
            0u8;
            XMODEL1_MAX_RESPONSE_CANDIDATES
                * XMODEL1_MAX_CANDIDATES
                * XMODEL1_CANDIDATE_FLAG_DIM
        ],
        post_candidate_quality: vec![
            0.0f32;
            XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES
        ],
        post_candidate_hard_bad: vec![
            0u8;
            XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES
        ],
        teacher_discard_idx: vec![-1i16; XMODEL1_MAX_RESPONSE_CANDIDATES],
        human_discard_idx: vec![-1i16; XMODEL1_MAX_RESPONSE_CANDIDATES],
    }
}

fn discard_actions_from_legal_actions(legal_actions: &[Value]) -> Vec<Value> {
    legal_actions
        .iter()
        .filter(|action| action.get("type").and_then(Value::as_str) == Some("dahai"))
        .cloned()
        .collect()
}

fn populate_response_post_discard_arrays(
    out: &mut ResponseCandidateArrays,
    slot: usize,
    snapshot_value: &Value,
    actor: usize,
    discard_actions: &[Value],
    human_post_discard: Option<&Value>,
    context: &str,
) -> Result<(), String> {
    let snapshot_core: SnapshotCore =
        serde_json::from_value(snapshot_value.clone()).map_err(|err| {
            format!("failed to parse {context} response snapshot for actor {actor}: {err}")
        })?;
    let round_state = runtime_build_round_state_from_snapshot(&snapshot_core, actor)?;
    let decision_ctx = DecisionAnalysisContext::new(&round_state, actor, true);
    let mut best_teacher_idx = -1i16;
    let mut best_teacher_quality = f32::NEG_INFINITY;
    for (candidate_slot, discard_action) in discard_actions
        .iter()
        .take(XMODEL1_MAX_CANDIDATES)
        .enumerate()
    {
        let Some(discard_tile) = discard_action.get("pai").and_then(Value::as_str) else {
            continue;
        };
        let Some(analysis) = analyze_discard_candidate(discard_tile, &decision_ctx) else {
            continue;
        };
        let flat_index = slot * XMODEL1_MAX_CANDIDATES + candidate_slot;
        let feat_offset = flat_index * XMODEL1_CANDIDATE_FEATURE_DIM;
        for (idx, value) in analysis.feat.iter().enumerate() {
            out.post_candidate_feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        let flags_offset = flat_index * XMODEL1_CANDIDATE_FLAG_DIM;
        out.post_candidate_flags[flags_offset..flags_offset + XMODEL1_CANDIDATE_FLAG_DIM]
            .copy_from_slice(&analysis.flags);
        out.post_candidate_tile_id[flat_index] = analysis.tile34;
        out.post_candidate_mask[flat_index] = 1;
        out.post_candidate_quality[flat_index] = analysis.metrics.quality;
        out.post_candidate_hard_bad[flat_index] = analysis.metrics.hard_bad;
        if let Some(human_action) = human_post_discard {
            if action_idx_from_action(human_action) == action_idx_from_action(discard_action) {
                out.human_discard_idx[slot] = candidate_slot as i16;
            }
        }
        if analysis.metrics.quality > best_teacher_quality {
            best_teacher_quality = analysis.metrics.quality;
            best_teacher_idx = candidate_slot as i16;
        }
    }
    out.teacher_discard_idx[slot] = best_teacher_idx;
    Ok(())
}

fn encode_response_candidate_arrays(
    snapshot_value: &Value,
    actor: usize,
    legal_actions: &[Value],
    chosen_action: &Value,
    human_post_discard: Option<&Value>,
) -> Result<ResponseCandidateArrays, String> {
    let mut out = empty_response_candidate_arrays();
    let mut response_actions = legal_actions
        .iter()
        .filter(|action| is_response_action(action))
        .cloned()
        .collect::<Vec<_>>();
    if is_response_action(chosen_action)
        && !response_actions
            .iter()
            .any(|action| response_actions_equivalent(chosen_action, action))
    {
        response_actions.push(chosen_action.clone());
    }
    response_actions.sort_by_key(response_action_sort_key);
    for (slot, action) in response_actions
        .iter()
        .take(XMODEL1_MAX_RESPONSE_CANDIDATES)
        .enumerate()
    {
        let action_type = action.get("type").and_then(Value::as_str).unwrap_or("none");
        out.action_idx[slot] = action_idx_from_action(action);
        out.mask[slot] = 1;
        if response_actions_equivalent(chosen_action, action) {
            out.chosen_idx = slot as i16;
        }
        if !response_action_requires_post_discard(action) {
            continue;
        }
        let human_for_slot = if response_actions_equivalent(chosen_action, action) {
            human_post_discard
        } else {
            None
        };
        if action_type == "reach" {
            let discard_actions = discard_actions_from_legal_actions(legal_actions);
            populate_response_post_discard_arrays(
                &mut out,
                slot,
                snapshot_value,
                actor,
                &discard_actions,
                human_for_slot,
                "reach",
            )?;
            continue;
        }
        let projected_snapshot = match action_type {
            "chi" | "pon" | "daiminkan" => {
                project_keqingv4_call_snapshot(snapshot_value, actor, action)
            }
            _ => None,
        };
        let Some(projected_snapshot) = projected_snapshot else {
            continue;
        };
        let projected_legal_actions =
            public_legal_actions_for_snapshot(&projected_snapshot, actor)?;
        let discard_actions = discard_actions_from_legal_actions(&projected_legal_actions);
        populate_response_post_discard_arrays(
            &mut out,
            slot,
            &projected_snapshot,
            actor,
            &discard_actions,
            human_for_slot,
            action_type,
        )?;
    }
    Ok(out)
}

fn special_action_bonus_metrics(
    action: &Value,
    yakuhai: &[bool; TILE_KIND_COUNT],
    dora_flags: &[bool; TILE_KIND_COUNT],
) -> (f32, f32, f32, f32) {
    let mut action_tile_ids: Vec<usize> = Vec::new();
    if let Some(consumed) = action.get("consumed").and_then(Value::as_array) {
        for tile in consumed.iter().filter_map(Value::as_str) {
            if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
                action_tile_ids.push(tile34);
            }
        }
    }
    if let Some(tile) = action.get("pai").and_then(Value::as_str) {
        if let Some(tile34) = tile34_from_pai(tile).map(|value| value as usize) {
            action_tile_ids.push(tile34);
        }
    }
    let action_dora_bonus = (action_tile_ids
        .iter()
        .filter(|tile34| dora_flags[**tile34])
        .count() as f32
        + action_tile_ids
            .iter()
            .filter(|tile34| {
                action
                    .get("consumed")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items.iter().filter_map(Value::as_str).any(|tile| {
                            replay_core::tile_is_aka(tile)
                                && tile34_from_pai(tile).map(|v| v as usize) == Some(**tile34)
                        })
                    })
                    .unwrap_or(false)
            })
            .count() as f32)
        / 4.0;
    let action_yakuhai_bonus = action_tile_ids
        .iter()
        .filter(|tile34| yakuhai[**tile34])
        .count() as f32
        / 4.0;
    let action_type = action.get("type").and_then(Value::as_str).unwrap_or("none");
    let call_breaks_closed = f32::from(matches!(action_type, "chi" | "pon" | "daiminkan"));
    let kan_rinshan_bonus = f32::from(matches!(action_type, "daiminkan" | "ankan" | "kakan"));
    (
        action_dora_bonus.min(1.0),
        action_yakuhai_bonus.min(1.0),
        call_breaks_closed,
        kan_rinshan_bonus,
    )
}

fn special_exposure_factor(special_type: i16, ankan_preserves_tenpai: f32) -> f32 {
    match special_type {
        XMODEL1_SPECIAL_TYPE_CHI_LOW
        | XMODEL1_SPECIAL_TYPE_CHI_MID
        | XMODEL1_SPECIAL_TYPE_CHI_HIGH => 0.65,
        XMODEL1_SPECIAL_TYPE_PON => 0.55,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN => 0.8,
        XMODEL1_SPECIAL_TYPE_KAKAN => 0.7,
        XMODEL1_SPECIAL_TYPE_ANKAN => {
            if ankan_preserves_tenpai > 0.5 {
                0.2
            } else {
                0.4
            }
        }
        XMODEL1_SPECIAL_TYPE_REACH | XMODEL1_SPECIAL_TYPE_DAMA => 0.35,
        _ => 0.0,
    }
}

fn build_special_hand_summary(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    decision_ctx: &DecisionAnalysisContext,
) -> SpecialHandSummary {
    let tracker = &round_state.feature_tracker.players[actor];
    let actor_score = round_state.scores.get(actor).copied().unwrap_or(25000) as f32;
    let mean_score = round_state
        .scores
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / 4.0;
    let score_gap = ((actor_score - mean_score) / SCORE_NORM).clamp(-1.0, 1.0);
    let threat_by_opponent = decision_ctx.risk_table[27];
    let current_structure = after_state_structure_metrics(
        &tracker.hand_counts34,
        &decision_ctx.yakuhai,
        &decision_ctx.dora_flags,
        hand_aka_count(&tracker.hand_tiles),
    );
    let current_value = if decision_ctx.before_progress.shanten >= DEEP_SHANTEN_FAST_PATH_CUTOFF {
        let (pair_count, taatsu_count, ankoutsu_count) =
            pair_taatsu_ankoutsu_metrics_from_counts(&tracker.hand_counts34);
        deep_shanten_after_state_value_metrics(
            &current_structure,
            decision_ctx.before_progress.ukeire_live_count as usize,
            pair_count,
            taatsu_count,
            ankoutsu_count,
            decision_ctx.is_open_hand,
        )
    } else {
        baseline_after_state_value_metrics(&current_structure, decision_ctx.is_open_hand)
    };
    let best_discard_quality = decision_ctx.best_discard_quality_for_legal_actions(legal_actions);
    let total_discards: usize = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum();
    SpecialHandSummary {
        shanten: decision_ctx.before_decision_shanten,
        tenpai: f32::from(decision_ctx.before_decision_shanten == 0),
        waits_count: decision_ctx.before_decision_waits_count,
        waits_live_norm: (decision_ctx.before_waits_live as f32 / 136.0).min(1.0),
        round_progress: (total_discards as f32 / 60.0).min(1.0),
        score_gap,
        threat_proxy_any_reached: f32::from(
            round_state
                .players
                .iter()
                .enumerate()
                .any(|(pid, player)| pid != actor && player.reached),
        ),
        threat_by_opponent,
        is_open: f32::from(decision_ctx.is_open_hand),
        best_discard_quality_norm: (best_discard_quality / 3.0).min(1.0),
        current_han_floor_norm: (current_structure.confirmed_han_floor / 8.0).min(1.0),
        current_dora_count_norm: (current_structure.after_dora_count / 4.0).min(1.0),
        current_max_value_norm: current_value.after_max_hand_value_norm,
        current_hand_value_survives: current_value.hand_value_survives,
    }
}

fn special_action_after_value(
    summary: &SpecialHandSummary,
    special_type: i16,
    action: &Value,
    yakuhai: &[bool; TILE_KIND_COUNT],
    dora_flags: &[bool; TILE_KIND_COUNT],
) -> f32 {
    let (action_dora_bonus, action_yakuhai_bonus, _call_breaks_closed, _kan_rinshan_bonus) =
        special_action_bonus_metrics(action, yakuhai, dora_flags);
    match special_type {
        XMODEL1_SPECIAL_TYPE_REACH | XMODEL1_SPECIAL_TYPE_DAMA => {
            let reach_bonus = if special_type == XMODEL1_SPECIAL_TYPE_REACH {
                0.15
            } else {
                0.0
            };
            (summary.current_max_value_norm + reach_bonus + 0.2 * action_dora_bonus).min(1.0)
        }
        XMODEL1_SPECIAL_TYPE_CHI_LOW
        | XMODEL1_SPECIAL_TYPE_CHI_MID
        | XMODEL1_SPECIAL_TYPE_CHI_HIGH
        | XMODEL1_SPECIAL_TYPE_PON
        | XMODEL1_SPECIAL_TYPE_DAIMINKAN
        | XMODEL1_SPECIAL_TYPE_ANKAN
        | XMODEL1_SPECIAL_TYPE_KAKAN => (0.6 * summary.best_discard_quality_norm
            + 0.25 * action_dora_bonus
            + 0.25 * action_yakuhai_bonus)
            .min(1.0),
        XMODEL1_SPECIAL_TYPE_HORA => summary.current_max_value_norm.max(0.8),
        _ => 0.0,
    }
}

#[allow(unused_assignments)]
fn encode_special_candidate_arrays_with_legal_actions(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    chosen_action: &Value,
    decision_ctx: &DecisionAnalysisContext,
) -> SpecialCandidateArrays {
    let summary = with_export_stage_timing(ExportStage::SpecialSummary, || {
        build_special_hand_summary(round_state, actor, legal_actions, decision_ctx)
    });
    let yakuhai = decision_ctx.yakuhai;
    let dora_flags = decision_ctx.dora_flags;

    let mut feat =
        vec![0u16; XMODEL1_MAX_SPECIAL_CANDIDATES * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM];
    let mut type_id = vec![-1i16; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut mask = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut quality = vec![0.0f32; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut rank = vec![0i8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut hard_bad = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut chosen_idx = -1i16;

    let mut grouped: Vec<Option<Value>> = vec![None; XMODEL1_MAX_SPECIAL_CANDIDATES];
    if legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("reach"))
    {
        grouped[XMODEL1_SPECIAL_TYPE_REACH as usize] = Some(json!({"type":"reach"}));
        grouped[XMODEL1_SPECIAL_TYPE_DAMA as usize] = Some(json!({"type":"dama"}));
    }
    for action in legal_actions {
        let Some(special_type) = special_type_from_action(action, true) else {
            continue;
        };
        if matches!(
            special_type,
            XMODEL1_SPECIAL_TYPE_REACH | XMODEL1_SPECIAL_TYPE_DAMA
        ) {
            continue;
        }
        let replace = grouped[special_type as usize]
            .as_ref()
            .map(|prev| special_call_family_score(action) > special_call_family_score(prev))
            .unwrap_or(true);
        if replace {
            grouped[special_type as usize] = Some(action.clone());
        }
    }
    if let Some(chosen_special_type) = special_type_from_action(chosen_action, true) {
        if !matches!(
            chosen_special_type,
            XMODEL1_SPECIAL_TYPE_REACH | XMODEL1_SPECIAL_TYPE_DAMA
        ) && grouped[chosen_special_type as usize].is_none()
        {
            grouped[chosen_special_type as usize] = Some(chosen_action.clone());
        }
    }
    let has_call_special = [
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_KAKAN,
    ]
    .iter()
    .any(|special_type| grouped[*special_type as usize].is_some());
    if has_call_special {
        grouped[XMODEL1_SPECIAL_TYPE_NONE as usize] = grouped[XMODEL1_SPECIAL_TYPE_NONE as usize]
            .clone()
            .or_else(|| Some(json!({"type":"none"})));
    }

    let order = [
        XMODEL1_SPECIAL_TYPE_REACH,
        XMODEL1_SPECIAL_TYPE_DAMA,
        XMODEL1_SPECIAL_TYPE_HORA,
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_KAKAN,
        XMODEL1_SPECIAL_TYPE_RYUKYOKU,
        XMODEL1_SPECIAL_TYPE_NONE,
    ];
    for (slot, special_type) in order
        .iter()
        .take(XMODEL1_MAX_SPECIAL_CANDIDATES)
        .enumerate()
    {
        let Some(action) = grouped[*special_type as usize].as_ref() else {
            continue;
        };
        let special_type = *special_type;
        type_id[slot] = special_type;
        mask[slot] = 1;

        let (action_dora_bonus, action_yakuhai_bonus, _call_breaks_closed, _kan_rinshan_bonus) =
            special_action_bonus_metrics(action, &yakuhai, &dora_flags);
        let ankan_preserves_tenpai =
            f32::from(special_type == XMODEL1_SPECIAL_TYPE_ANKAN && summary.tenpai > 0.5);
        let after_value_norm =
            special_action_after_value(&summary, special_type, action, &yakuhai, &dora_flags);
        let (speed_gain_norm, value_loss_norm) = match special_type {
            XMODEL1_SPECIAL_TYPE_REACH => (0.4, 0.05),
            XMODEL1_SPECIAL_TYPE_DAMA => (0.0, 0.0),
            XMODEL1_SPECIAL_TYPE_CHI_LOW
            | XMODEL1_SPECIAL_TYPE_CHI_MID
            | XMODEL1_SPECIAL_TYPE_CHI_HIGH => (0.35, 0.35),
            XMODEL1_SPECIAL_TYPE_PON => (0.45, 0.2),
            XMODEL1_SPECIAL_TYPE_DAIMINKAN | XMODEL1_SPECIAL_TYPE_KAKAN => (0.3, 0.45),
            XMODEL1_SPECIAL_TYPE_ANKAN => (0.3, 0.2),
            XMODEL1_SPECIAL_TYPE_HORA => (1.0, 0.0),
            _ => (0.0, 0.0),
        };
        let hand_value_survives = if matches!(
            special_type,
            XMODEL1_SPECIAL_TYPE_HORA | XMODEL1_SPECIAL_TYPE_NONE | XMODEL1_SPECIAL_TYPE_RYUKYOKU
        ) {
            1.0
        } else {
            f32::from(
                summary.current_hand_value_survives > 0.0
                    || after_value_norm > 0.0
                    || summary.current_han_floor_norm > 0.0,
            )
        };
        let exposure = special_exposure_factor(special_type, ankan_preserves_tenpai);
        let risk_proxy = summary
            .threat_by_opponent
            .map(|value| (exposure * value).min(1.0));
        let feature_values = [
            summary.shanten as f32 / 8.0,
            summary.tenpai,
            summary.waits_count as f32 / 34.0,
            summary.waits_live_norm,
            summary.round_progress,
            summary.score_gap.clamp(-1.0, 1.0),
            summary.threat_proxy_any_reached,
            summary.is_open,
            after_value_norm,
            speed_gain_norm,
            value_loss_norm,
            action_dora_bonus,
            action_yakuhai_bonus,
            risk_proxy[0],
            risk_proxy[1],
            risk_proxy[2],
            hand_value_survives,
            summary.current_han_floor_norm,
            summary.current_dora_count_norm,
        ];
        let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
        for (idx, value) in feature_values.iter().enumerate() {
            feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }

        let candidate_quality = 0.6 * after_value_norm + 0.35 * speed_gain_norm
            - 0.4 * value_loss_norm
            - 0.35 * ((risk_proxy[0] + risk_proxy[1] + risk_proxy[2]) / 3.0)
            + 0.15 * action_dora_bonus
            + 0.15 * action_yakuhai_bonus
            + f32::from(special_type == XMODEL1_SPECIAL_TYPE_HORA);
        quality[slot] = candidate_quality;
        let candidate_hard_bad = u8::from(
            (special_type == XMODEL1_SPECIAL_TYPE_REACH
                && summary.threat_proxy_any_reached > 0.5
                && summary.waits_live_norm < 0.03)
                || (matches!(
                    special_type,
                    XMODEL1_SPECIAL_TYPE_CHI_LOW
                        | XMODEL1_SPECIAL_TYPE_CHI_MID
                        | XMODEL1_SPECIAL_TYPE_CHI_HIGH
                        | XMODEL1_SPECIAL_TYPE_DAIMINKAN
                        | XMODEL1_SPECIAL_TYPE_ANKAN
                        | XMODEL1_SPECIAL_TYPE_KAKAN
                ) && value_loss_norm >= 0.35
                    && summary.current_han_floor_norm <= 0.0),
        );
        hard_bad[slot] = candidate_hard_bad;
        rank[slot] = if candidate_hard_bad == 1 {
            0
        } else if candidate_quality >= 1.0 {
            3
        } else if candidate_quality >= 0.25 {
            2
        } else {
            1
        };

        let chosen_type = chosen_action
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("none");
        if (chosen_type == "reach" && special_type == XMODEL1_SPECIAL_TYPE_REACH)
            || (chosen_type == "dahai"
                && special_type == XMODEL1_SPECIAL_TYPE_DAMA
                && grouped[XMODEL1_SPECIAL_TYPE_REACH as usize].is_some())
            || (chosen_type == "hora" && special_type == XMODEL1_SPECIAL_TYPE_HORA)
            || (chosen_type == "ryukyoku" && special_type == XMODEL1_SPECIAL_TYPE_RYUKYOKU)
            || (chosen_type == "pon" && special_type == XMODEL1_SPECIAL_TYPE_PON)
            || (chosen_type == "chi"
                && special_type == special_type_from_action(chosen_action, true).unwrap_or(-1))
            || (chosen_type == "daiminkan" && special_type == XMODEL1_SPECIAL_TYPE_DAIMINKAN)
            || (chosen_type == "ankan" && special_type == XMODEL1_SPECIAL_TYPE_ANKAN)
            || (chosen_type == "kakan" && special_type == XMODEL1_SPECIAL_TYPE_KAKAN)
            || (chosen_type == "none" && special_type == XMODEL1_SPECIAL_TYPE_NONE)
        {
            chosen_idx = slot as i16;
        }
    }

    SpecialCandidateArrays {
        feat,
        type_id,
        mask,
        quality,
        rank,
        hard_bad,
        chosen_idx,
    }
}

#[allow(dead_code)]
fn encode_special_candidate_arrays(
    round_state: &RoundState,
    actor: usize,
    legal_actions: &[Value],
    before_decision_shanten: i8,
    before_waits_count: u8,
    before_waits_live: usize,
    best_discard_quality: f32,
) -> SpecialCandidateArrays {
    let mut feat =
        vec![0u16; XMODEL1_MAX_SPECIAL_CANDIDATES * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM];
    let mut type_id = vec![-1i16; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut mask = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut quality = vec![0.0f32; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut rank = vec![0i8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut hard_bad = vec![0u8; XMODEL1_MAX_SPECIAL_CANDIDATES];
    let mut chosen_idx = -1i16;

    let actor_score = round_state.scores.get(actor).copied().unwrap_or(25000) as f32;
    let mean_score = round_state
        .scores
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / 4.0;
    let score_gap = ((actor_score - mean_score) / SCORE_NORM).clamp(-1.0, 1.0);
    let threat_proxy = f32::from(
        round_state
            .players
            .iter()
            .enumerate()
            .any(|(pid, player)| pid != actor && player.reached),
    );
    let total_discards: usize = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum();
    let round_progress = (total_discards as f32 / 60.0).min(1.0);
    let has_any_meld = !round_state.players[actor].melds.is_empty();
    let has_open_meld = round_state.players[actor]
        .melds
        .iter()
        .any(|meld| !matches!(meld.kind.as_str(), "ankan"));
    let is_open = f32::from(has_any_meld);
    let tracker = &round_state.feature_tracker.players[actor];
    let yakuhai = yakuhai_tiles(&round_state.bakaze, actor as i8, round_state.oya);
    let dora_flags = dora_target_flags(round_state);
    let current_structure = after_state_structure_metrics(
        &tracker.hand_counts34,
        &yakuhai,
        &dora_flags,
        round_state.feature_tracker.players[actor].aka_counts[0],
    );
    let current_dora_count = current_structure.after_dora_count;
    let current_yakuhai_triplet_count = current_structure.yakuhai_triplet_count;
    let current_han_floor = current_structure.confirmed_han_floor;
    let reach_available = before_decision_shanten <= 0
        && !has_open_meld
        && !round_state.players[actor].reached
        && !round_state.players[actor].pending_reach
        && round_state.scores[actor] >= 1000;

    if reach_available {
        let candidates = [
            (
                XMODEL1_SPECIAL_TYPE_REACH,
                0.5f32,
                before_waits_live as f32 / 20.0,
                -0.2f32,
                0.0f32,
                ((current_han_floor + current_dora_count) / 8.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                (best_discard_quality / 3.0).clamp(0.0, 1.0),
                1.0f32,
                1.2 * f32::from(before_decision_shanten == 0)
                    + 0.15 * before_waits_live as f32
                    + 0.5 * score_gap
                    - 0.6 * threat_proxy
                    + 0.3 * (1.0 - round_progress)
                    + 0.15 * ((current_han_floor + current_dora_count) / 8.0).min(1.0),
            ),
            (
                XMODEL1_SPECIAL_TYPE_DAMA,
                0.0f32,
                best_discard_quality,
                0.0f32,
                0.0f32,
                (current_dora_count / 10.0).min(1.0),
                (current_han_floor / 8.0).min(1.0),
                (best_discard_quality / 3.0).clamp(0.0, 1.0),
                1.0f32,
                0.8 * f32::from(before_decision_shanten == 0)
                    + 0.35 * best_discard_quality
                    + 0.25 * score_gap
                    + 0.15 * threat_proxy
                    + 0.1 * (current_han_floor / 8.0).min(1.0),
            ),
        ];
        for (
            slot,
            (
                special_type,
                speed_gain,
                retain_value,
                value_loss,
                han_floor,
                action_dora_bonus,
                action_yakuhai_bonus,
                context_bonus,
                hand_value_survives,
                candidate_quality,
            ),
        ) in candidates.iter().enumerate()
        {
            type_id[slot] = *special_type;
            mask[slot] = 1;
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap,
                threat_proxy,
                is_open,
                (*speed_gain).clamp(0.0, 1.0),
                (*retain_value / 3.0).clamp(-1.0, 1.0),
                (*value_loss).clamp(0.0, 1.0),
                (*han_floor).clamp(0.0, 1.0),
                (*action_dora_bonus).clamp(0.0, 1.0),
                (*action_yakuhai_bonus).clamp(0.0, 1.0),
                (*context_bonus).clamp(0.0, 1.0),
                (*hand_value_survives).clamp(0.0, 1.0),
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::from(best_discard_quality > 0.0),
            ];
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] = *candidate_quality;
            let candidate_hard_bad = u8::from(
                *special_type == XMODEL1_SPECIAL_TYPE_REACH
                    && threat_proxy > 0.5
                    && before_waits_live < 4,
            );
            hard_bad[slot] = candidate_hard_bad;
            rank[slot] = if candidate_hard_bad == 1 {
                0
            } else if *candidate_quality >= 1.0 {
                3
            } else if *candidate_quality >= 0.25 {
                2
            } else {
                1
            };
        }
        chosen_idx = 1;
    }

    if legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("hora"))
    {
        let slot = mask
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(XMODEL1_MAX_SPECIAL_CANDIDATES);
        if slot < XMODEL1_MAX_SPECIAL_CANDIDATES {
            type_id[slot] = XMODEL1_SPECIAL_TYPE_HORA;
            mask[slot] = 1;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap.clamp(-1.0, 1.0),
                threat_proxy,
                is_open,
                1.0,
                ((current_han_floor + current_dora_count) / 8.0).min(1.0) / 3.0,
                0.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_han_floor / 8.0).min(1.0),
                1.0,
                1.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                f32::from(best_discard_quality > 0.0),
            ];
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] = 3.0 + ((current_han_floor + current_dora_count) / 8.0).min(1.0);
            rank[slot] = 3;
        }
    }

    if legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("ryukyoku"))
    {
        let slot = mask
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(XMODEL1_MAX_SPECIAL_CANDIDATES);
        if slot < XMODEL1_MAX_SPECIAL_CANDIDATES {
            type_id[slot] = XMODEL1_SPECIAL_TYPE_RYUKYOKU;
            mask[slot] = 1;
            let feature_values = [
                before_decision_shanten as f32 / 8.0,
                f32::from(before_decision_shanten == 0),
                before_waits_count as f32 / 34.0,
                (before_waits_live as f32 / 20.0).min(1.0),
                round_progress,
                score_gap.clamp(-1.0, 1.0),
                threat_proxy,
                is_open,
                0.1,
                (before_decision_shanten as f32 / 8.0).min(1.0) / 3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (1.0 - round_progress).clamp(0.0, 1.0),
                1.0,
                (current_han_floor / 8.0).min(1.0),
                (current_dora_count / 10.0).min(1.0),
                (current_yakuhai_triplet_count / 3.0).min(1.0),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                f32::from(best_discard_quality > 0.0),
            ];
            let feat_offset = slot * XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM;
            for (idx, value) in feature_values.iter().enumerate() {
                feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
            }
            quality[slot] =
                0.15 + 0.25 * (1.0 - round_progress).clamp(0.0, 1.0) + 0.2 * threat_proxy;
            rank[slot] = 1;
        }
    }

    SpecialCandidateArrays {
        feat,
        type_id,
        mask,
        quality,
        rank,
        hard_bad,
        chosen_idx,
    }
}

fn encode_special_sample_record(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    _legal_actions: &[Value],
    chosen_action: &Value,
    sample_type: i8,
    event_index: i32,
    history_summary: [f32; replay_core::HISTORY_SUMMARY_DIM],
    human_post_discard: Option<&Value>,
) -> Result<FullRecord, String> {
    let response_snapshot = serde_json::to_value(snapshot_for_actor(core_state, actor))
        .map_err(|err| format!("failed to serialize response snapshot for actor {actor}: {err}"))?;
    let response_legal_actions = public_legal_actions_for_snapshot(&response_snapshot, actor)?;
    let include_candidate_analyses = response_legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) == Some("dahai"));
    let decision_ctx = DecisionAnalysisContext::new(round_state, actor, include_candidate_analyses);
    let state_tile_feat = encode_state_tile_features(
        round_state,
        actor,
        &decision_ctx.before_progress,
        &decision_ctx.before_visible34,
    )?;
    let state_scalar = encode_state_scalar_features(
        round_state,
        actor,
        &decision_ctx.before_progress,
        &decision_ctx.before_visible34,
    )?;
    let mut candidate_feat = vec![0u16; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_flags = vec![0u8; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM];
    let mut candidate_quality = vec![0.0f32; XMODEL1_MAX_CANDIDATES];
    let mut candidate_rank = vec![0i8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_hard_bad = vec![0u8; XMODEL1_MAX_CANDIDATES];
    for (slot, analysis) in decision_ctx.candidate_analyses.iter().enumerate() {
        let feat_offset = slot * XMODEL1_CANDIDATE_FEATURE_DIM;
        for (idx, value) in analysis.feat.iter().enumerate() {
            candidate_feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        let flags_offset = slot * XMODEL1_CANDIDATE_FLAG_DIM;
        candidate_flags[flags_offset..flags_offset + XMODEL1_CANDIDATE_FLAG_DIM]
            .copy_from_slice(&analysis.flags);
        candidate_tile_id[slot] = analysis.tile34;
        candidate_mask[slot] = 1;
        candidate_quality[slot] = analysis.metrics.quality;
        candidate_rank[slot] = analysis.metrics.rank_bucket;
        candidate_hard_bad[slot] = analysis.metrics.hard_bad;
    }
    let action_idx_target = action_idx_from_action(chosen_action);
    let response = encode_response_candidate_arrays(
        &response_snapshot,
        actor,
        &response_legal_actions,
        chosen_action,
        human_post_discard,
    )?;
    let offense_quality_target = if response.chosen_idx >= 0 {
        let chosen_slot = response.chosen_idx as usize;
        let teacher_idx = response.teacher_discard_idx[chosen_slot];
        if teacher_idx >= 0 {
            response.post_candidate_quality
                [chosen_slot * XMODEL1_MAX_CANDIDATES + teacher_idx as usize]
        } else {
            0.0
        }
    } else {
        0.0
    };
    let opp_tenpai_target =
        replay_core::compute_opp_tenpai_target(round_state, actor, &decision_ctx.before_visible34);
    Ok(FullRecord {
        state_tile_feat,
        state_scalar,
        candidate_feat,
        candidate_tile_id,
        candidate_mask,
        candidate_flags,
        chosen_candidate_idx: -1,
        candidate_quality,
        candidate_rank,
        candidate_hard_bad,
        response_action_idx: response.action_idx,
        response_action_mask: response.mask,
        chosen_response_action_idx: response.chosen_idx,
        response_post_candidate_feat: response.post_candidate_feat,
        response_post_candidate_tile_id: response.post_candidate_tile_id,
        response_post_candidate_mask: response.post_candidate_mask,
        response_post_candidate_flags: response.post_candidate_flags,
        response_post_candidate_quality: response.post_candidate_quality,
        response_post_candidate_hard_bad: response.post_candidate_hard_bad,
        response_teacher_discard_idx: response.teacher_discard_idx,
        response_human_discard_idx: response.human_discard_idx,
        action_idx_target,
        global_value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        opp_tenpai_target,
        history_summary,
        offense_quality_target,
        sample_type,
        actor: actor as i8,
        score_before_action: round_state.scores[actor],
        final_score_delta_points_target: 0,
        final_rank_target: 0,
        event_index,
        kyoku: round_state.kyoku,
        honba: round_state.honba,
        is_open_hand: u8::from(decision_ctx.is_open_hand),
        rule_context: default_rule_context(),
    })
}

fn encode_candidate_features(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    chosen_tile: &str,
    event_index: i32,
    history_summary: [f32; replay_core::HISTORY_SUMMARY_DIM],
) -> Result<FullRecord, String> {
    let decision_ctx = DecisionAnalysisContext::new(round_state, actor, true);
    let state_tile_feat = encode_state_tile_features(
        round_state,
        actor,
        &decision_ctx.before_progress,
        &decision_ctx.before_visible34,
    )?;
    let state_scalar = encode_state_scalar_features(
        round_state,
        actor,
        &decision_ctx.before_progress,
        &decision_ctx.before_visible34,
    )?;

    let mut candidate_feat = vec![0u16; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM];
    let mut candidate_tile_id = vec![-1i16; XMODEL1_MAX_CANDIDATES];
    let mut candidate_mask = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_flags = vec![0u8; XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM];
    let mut candidate_quality = vec![0.0f32; XMODEL1_MAX_CANDIDATES];
    let mut candidate_rank = vec![0i8; XMODEL1_MAX_CANDIDATES];
    let mut candidate_hard_bad = vec![0u8; XMODEL1_MAX_CANDIDATES];
    let mut chosen_candidate_idx: Option<i16> = None;

    let chosen_normalized = normalize_tile_repr(chosen_tile);
    for (slot, analysis) in decision_ctx.candidate_analyses.iter().enumerate() {
        let feat_offset = slot * XMODEL1_CANDIDATE_FEATURE_DIM;
        for (idx, value) in analysis.feat.iter().enumerate() {
            candidate_feat[feat_offset + idx] = f16::from_f32(*value).to_bits();
        }
        let flags_offset = slot * XMODEL1_CANDIDATE_FLAG_DIM;
        candidate_flags[flags_offset..flags_offset + XMODEL1_CANDIDATE_FLAG_DIM]
            .copy_from_slice(&analysis.flags);
        candidate_tile_id[slot] = analysis.tile34;
        candidate_mask[slot] = 1;
        candidate_quality[slot] = analysis.metrics.quality;
        candidate_rank[slot] = analysis.metrics.rank_bucket;
        candidate_hard_bad[slot] = analysis.metrics.hard_bad;
        if chosen_normalized == analysis.discard_tile {
            chosen_candidate_idx = Some(slot as i16);
        }
    }

    let chosen_candidate_idx = chosen_candidate_idx.ok_or_else(|| {
        format!(
            "failed to map chosen discard into candidate set: actor={} event_index={} chosen_tile={}",
            actor, event_index, chosen_tile
        )
    })?;
    validate_xmodel1_discard_record(chosen_candidate_idx, &candidate_mask, &candidate_tile_id)?;
    let chosen_quality = candidate_quality[chosen_candidate_idx as usize];
    let snapshot = snapshot_for_actor(core_state, actor);
    let snapshot_value = serde_json::to_value(&snapshot)
        .map_err(|err| format!("failed to serialize discard snapshot for actor {actor}: {err}"))?;
    let legal_actions = public_legal_actions_for_snapshot(&snapshot_value, actor)?;
    let response = encode_response_candidate_arrays(
        &snapshot_value,
        actor,
        &legal_actions,
        &json!({"type":"dahai", "pai": chosen_tile}),
        None,
    )?;

    Ok(FullRecord {
        state_tile_feat,
        state_scalar,
        candidate_feat,
        candidate_tile_id,
        candidate_mask,
        candidate_flags,
        chosen_candidate_idx,
        candidate_quality,
        candidate_rank,
        candidate_hard_bad,
        response_action_idx: response.action_idx,
        response_action_mask: response.mask,
        chosen_response_action_idx: response.chosen_idx,
        response_post_candidate_feat: response.post_candidate_feat,
        response_post_candidate_tile_id: response.post_candidate_tile_id,
        response_post_candidate_mask: response.post_candidate_mask,
        response_post_candidate_flags: response.post_candidate_flags,
        response_post_candidate_quality: response.post_candidate_quality,
        response_post_candidate_hard_bad: response.post_candidate_hard_bad,
        response_teacher_discard_idx: response.teacher_discard_idx,
        response_human_discard_idx: response.human_discard_idx,
        action_idx_target: tile34_from_pai(chosen_tile).unwrap_or(0),
        global_value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        opp_tenpai_target: replay_core::compute_opp_tenpai_target(
            round_state,
            actor,
            &decision_ctx.before_visible34,
        ),
        history_summary,
        offense_quality_target: chosen_quality,
        sample_type: XMODEL1_SAMPLE_TYPE_DISCARD,
        actor: actor as i8,
        score_before_action: round_state.scores[actor],
        final_score_delta_points_target: 0,
        final_rank_target: 0,
        event_index,
        kyoku: round_state.kyoku,
        honba: round_state.honba,
        is_open_hand: u8::from(decision_ctx.is_open_hand),
        rule_context: default_rule_context(),
    })
}

fn encode_state_tile_features(
    round_state: &RoundState,
    actor: usize,
    progress: &Summary3n1,
    visible_counts: &[u8; TILE_KIND_COUNT],
) -> Result<Vec<u16>, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let (_decision_shanten, _decision_waits_count, decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &tracker.hand_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let mut tile_feat = vec![0f32; XMODEL1_STATE_TILE_CHANNELS * TILE_KIND_COUNT];
    let discards = &round_state.players;
    let other_pids: Vec<usize> = (0..4).filter(|&pid| pid != actor).collect();
    let mut latest_riichi_pid: Option<usize> = None;
    let mut latest_riichi_turn: i32 = -1;
    let mut opponent_meld_counts = [0.0f32; 3];

    for tile34 in 0..TILE_KIND_COUNT {
        let count = tracker.hand_counts34[tile34].min(4);
        for plane in 0..count as usize {
            tile_feat[(plane * TILE_KIND_COUNT) + tile34] = 1.0;
        }
    }

    for (meld_slot, meld) in round_state.players[actor].melds.iter().take(4).enumerate() {
        let channel = 4 + meld_slot;
        for tile in meld.consumed.iter().chain(meld.pai.iter()) {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                tile_feat[channel * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (slot, pid) in other_pids.iter().enumerate() {
        let channel_base = 8 + slot * 2;
        let pid_discards = &discards[*pid].discards;
        for (turn, discard) in pid_discards.iter().enumerate() {
            let idx = discard.tile34 as usize;
            if discard.reach_declared {
                tile_feat[channel_base * TILE_KIND_COUNT + idx] = 1.0;
                if (turn as i32) > latest_riichi_turn {
                    latest_riichi_turn = turn as i32;
                    latest_riichi_pid = Some(*pid);
                }
            }
            if !discard.tsumogiri {
                tile_feat[(channel_base + 1) * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (slot, pid) in other_pids.iter().enumerate() {
        let pid_melds = &round_state.players[*pid].melds;
        opponent_meld_counts[slot] = pid_melds.len() as f32 / 4.0;
        for meld in pid_melds {
            for tile in meld.consumed.iter().chain(meld.pai.iter()) {
                if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                    tile_feat[(14 + slot) * TILE_KIND_COUNT + idx] = 1.0;
                }
            }
        }
    }

    for discard in &round_state.players[actor].discards {
        tile_feat[17 * TILE_KIND_COUNT + discard.tile34 as usize] = 1.0;
    }

    for marker in &round_state.dora_markers {
        if let Some(actual) = dora_next(marker)
            .and_then(tile34_from_pai)
            .map(|value| value as usize)
        {
            tile_feat[18 * TILE_KIND_COUNT + actual] += 1.0;
        }
    }

    for (idx, wait) in decision_waits_tiles.iter().enumerate() {
        if *wait {
            tile_feat[19 * TILE_KIND_COUNT + idx] = 1.0;
        }
    }

    let mut slot_for_pid = [0usize; 4];
    for (slot, pid) in other_pids.iter().enumerate() {
        slot_for_pid[*pid] = slot;
    }
    slot_for_pid[actor] = 3;
    for pid in 0..4 {
        let slot = slot_for_pid[pid];
        for (turn, discard) in round_state.players[pid].discards.iter().enumerate() {
            let seg = (turn / 3).min(7);
            let channel = 20 + slot * 8 + seg;
            tile_feat[channel * TILE_KIND_COUNT + discard.tile34 as usize] = 1.0;
        }
    }

    if let Some(latest_riichi_pid) = latest_riichi_pid {
        let mut safe_tiles = [false; TILE_KIND_COUNT];
        for pid in 0..4 {
            if pid == latest_riichi_pid {
                continue;
            }
            for discard in &round_state.players[pid].discards {
                safe_tiles[discard.tile34 as usize] = true;
            }
        }
        for (idx, flag) in safe_tiles.iter().enumerate() {
            if *flag {
                tile_feat[53 * TILE_KIND_COUNT + idx] = 1.0;
            }
        }
    }

    for (idx, flag) in progress.ukeire_tiles.iter().enumerate() {
        if *flag {
            tile_feat[54 * TILE_KIND_COUNT + idx] = 1.0;
        }
    }

    for idx in 0..TILE_KIND_COUNT {
        tile_feat[56 * TILE_KIND_COUNT + idx] = (visible_counts[idx] as f32 / 4.0).clamp(0.0, 1.0);
    }

    Ok(tile_feat
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect())
}

fn encode_state_scalar_features(
    round_state: &RoundState,
    actor: usize,
    progress: &Summary3n1,
    visible_counts: &[u8; TILE_KIND_COUNT],
) -> Result<Vec<u16>, String> {
    let tracker = &round_state.feature_tracker.players[actor];
    let (_decision_shanten, _decision_waits_count, decision_waits_tiles) =
        calc_shanten_waits_like_python(
            &tracker.hand_counts34,
            !round_state.players[actor].melds.is_empty(),
        );
    let actor_score = round_state.scores[actor];
    let other_pids: Vec<usize> = (0..4).filter(|&pid| pid != actor).collect();
    let reached: Vec<bool> = round_state
        .players
        .iter()
        .map(|player| player.reached)
        .collect();
    let mut scalar = vec![0f32; XMODEL1_STATE_SCALAR_DIM];

    let mut all34_list: Vec<usize> = Vec::new();
    for (tile34, &count) in tracker.hand_counts34.iter().enumerate() {
        all34_list.extend(std::iter::repeat(tile34).take(count as usize));
    }
    for (tile34, &count) in tracker.meld_counts34.iter().enumerate() {
        all34_list.extend(std::iter::repeat(tile34).take(count as usize));
    }
    let total_tiles = all34_list.len();
    let all34_set: std::collections::BTreeSet<usize> = all34_list.iter().copied().collect();
    let yaochuu_tiles = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let yaochuu_set: std::collections::BTreeSet<usize> = yaochuu_tiles.into_iter().collect();
    let man_cnt = tracker.suit_counts[0];
    let pin_cnt = tracker.suit_counts[1];
    let sou_cnt = tracker.suit_counts[2];
    let honor_cnt = tracker.suit_counts[3];
    let yaochuu_cnt = all34_list
        .iter()
        .filter(|tile| yaochuu_set.contains(tile))
        .count();
    let has_tanyao = !all34_set.is_empty() && all34_set.is_disjoint(&yaochuu_set);
    let suupai_yaochuu_cnt = all34_list
        .iter()
        .filter(|tile| matches!(**tile, 0 | 8 | 9 | 17 | 18 | 26))
        .count();
    let is_open = !round_state.players[actor].melds.is_empty();
    let has_junchan =
        !is_open && honor_cnt == 0 && total_tiles > 0 && suupai_yaochuu_cnt * 3 >= total_tiles;
    let has_chanta = total_tiles > 0 && yaochuu_cnt * 3 >= total_tiles;
    let suit_counts_non_zero: Vec<usize> = [man_cnt, pin_cnt, sou_cnt]
        .into_iter()
        .filter(|count| *count > 0)
        .collect();
    let has_honitsu = suit_counts_non_zero.len() == 1 && total_tiles > 0;
    let has_chinitsu = suit_counts_non_zero.len() == 1 && honor_cnt == 0 && total_tiles > 0;
    let (_, taatsu_count, _ank_unused) =
        pair_taatsu_ankoutsu_metrics_from_counts(&tracker.hand_counts34);
    let pair_count = tracker.pair_count;
    let ankoutsu_cnt = tracker.ankoutsu_count;
    let chiitoi = if is_open {
        6
    } else {
        chiitoi_shanten(&tracker.hand_counts34)
    };
    let kokushi = if is_open {
        13
    } else {
        kokushi_shanten(&tracker.hand_counts34)
    };
    let mut estimated_wall_draws = round_state
        .players
        .iter()
        .map(|player| player.discards.len())
        .sum::<usize>();
    let mut chi_pon_calls = 0usize;
    let mut kan_calls = 0usize;
    let mut opponent_meld_counts = [0.0f32; 3];
    for (slot, pid) in other_pids.iter().enumerate() {
        let melds = &round_state.players[*pid].melds;
        opponent_meld_counts[slot] = melds.len() as f32 / 4.0;
        for meld in melds {
            if matches!(meld.kind.as_str(), "chi" | "pon") {
                chi_pon_calls += 1;
            }
            if matches!(meld.kind.as_str(), "daiminkan" | "ankan" | "kakan") {
                kan_calls += 1;
            }
        }
    }
    estimated_wall_draws = estimated_wall_draws.saturating_sub(chi_pon_calls) + kan_calls;
    let tiles_left = (70i32 - estimated_wall_draws as i32).max(0);
    let top_gap = (*round_state.scores.iter().max().unwrap_or(&actor_score) - actor_score).max(0);
    let bottom_gap =
        (actor_score - *round_state.scores.iter().min().unwrap_or(&actor_score)).max(0);
    let riichi_threat_count = reached
        .iter()
        .enumerate()
        .filter(|(pid, flag)| *pid != actor && **flag)
        .count();
    let is_all_last_like = f32::from(
        matches!(round_state.bakaze.as_str(), "W" | "N")
            || (round_state.bakaze == "S" && round_state.kyoku >= 4),
    );
    let tenpai_live_tile_count: usize = decision_waits_tiles
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(tile34, _)| usize::from(4u8.saturating_sub(visible_counts[tile34])))
        .sum();
    let wait_hiddenness_ratio = if progress.waits_count > 0 {
        tenpai_live_tile_count as f32 / (4.0 * progress.waits_count as f32).max(1.0)
    } else {
        0.0
    };

    let dora_count: usize = round_state
        .dora_markers
        .iter()
        .filter_map(|marker| dora_next(marker))
        .filter_map(tile34_from_pai)
        .map(|tile34| {
            let tile34 = tile34 as usize;
            all34_list.iter().filter(|&&value| value == tile34).count()
        })
        .sum();
    let jikaze = ((actor as i8 - round_state.oya).rem_euclid(4)) as usize;

    scalar[0] = f32::from(round_state.bakaze == "E");
    scalar[1] = f32::from(round_state.bakaze == "S");
    scalar[2] = f32::from(round_state.bakaze == "W");
    scalar[3] = round_state.kyoku as f32 / 4.0;
    scalar[4] = round_state.honba as f32 / 8.0;
    scalar[5] = round_state.kyotaku as f32 / 8.0;
    scalar[6] = actor_score as f32 / 50000.0;
    scalar[7] = score_rank(&round_state.scores, actor_score) as f32 / 3.0;
    scalar[8] = progress.shanten as f32 / 8.0;
    scalar[9] = progress.waits_count as f32 / 34.0;
    scalar[10] = round_state.players[actor].discards.len() as f32 / 18.0;
    scalar[11] = round_state.players[actor].melds.len() as f32 / 4.0;

    let all_pids_by_slot = [actor, other_pids[0], other_pids[1], other_pids[2]];
    for (slot, pid) in all_pids_by_slot.into_iter().enumerate() {
        scalar[12 + slot] = f32::from(reached[pid]);
    }

    let aka_total = tracker.aka_counts[0];
    let aka_m = tracker.aka_counts[1];
    let aka_p = tracker.aka_counts[2];
    let aka_s = tracker.aka_counts[3];
    scalar[16] = aka_total as f32 / 4.0;
    scalar[17] = f32::from(has_tanyao);
    scalar[18] = f32::from(aka_m > 0);
    scalar[19] = f32::from(aka_p > 0);
    scalar[20] = f32::from(aka_s > 0);
    scalar[21] = opponent_meld_counts[0];
    scalar[22] = opponent_meld_counts[1];
    scalar[23] = opponent_meld_counts[2];
    scalar[24] = (round_state.scores[other_pids[0]] - actor_score) as f32 / 30000.0;
    scalar[25] = (round_state.scores[other_pids[1]] - actor_score) as f32 / 30000.0;
    scalar[26] = (round_state.scores[other_pids[2]] - actor_score) as f32 / 30000.0;
    scalar[27] = jikaze as f32 / 3.0;
    scalar[28] = ((dora_count + aka_total) as f32 / 10.0).min(1.0);

    let bakaze_34 = match round_state.bakaze.as_str() {
        "E" => 27usize,
        "S" => 28,
        "W" => 29,
        "N" => 30,
        _ => 27,
    };
    let jikaze_34 = 27 + jikaze;
    let mut confirmed_han = 0usize;
    let mut honor_koutsu = [false; TILE_KIND_COUNT];
    for meld in &round_state.players[actor].melds {
        if !matches!(meld.kind.as_str(), "pon" | "daiminkan" | "ankan" | "kakan") {
            continue;
        }
        if let Some(tile34) = meld
            .pai
            .as_ref()
            .and_then(|tile| tile34_from_pai(tile))
            .map(|value| value as usize)
        {
            if tile34 >= 27 {
                honor_koutsu[tile34] = true;
            }
        }
    }
    for (tile34, &count) in tracker.hand_counts34.iter().enumerate() {
        if tile34 >= 27 && count >= 3 {
            honor_koutsu[tile34] = true;
        }
    }
    if honor_koutsu[bakaze_34] {
        confirmed_han += 1;
    }
    if jikaze_34 != bakaze_34 && honor_koutsu[jikaze_34] {
        confirmed_han += 1;
    }
    for tile34 in [31usize, 32, 33] {
        if honor_koutsu[tile34] {
            confirmed_han += 1;
        }
    }
    if has_tanyao {
        confirmed_han += 1;
    }
    if has_chinitsu {
        confirmed_han += if is_open { 5 } else { 6 };
    } else if has_honitsu {
        confirmed_han += if is_open { 2 } else { 3 };
    }
    confirmed_han += dora_count + aka_total;
    scalar[29] = (confirmed_han as f32 / 8.0).min(1.0);

    if total_tiles > 0 {
        scalar[30] = man_cnt as f32 / total_tiles as f32;
        scalar[31] = pin_cnt as f32 / total_tiles as f32;
        scalar[32] = sou_cnt as f32 / total_tiles as f32;
        scalar[33] = honor_cnt as f32 / total_tiles as f32;
    }
    scalar[34] = f32::from(has_junchan);
    scalar[35] = f32::from(has_chanta);
    scalar[36] = f32::from(has_honitsu);
    scalar[37] = f32::from(has_chinitsu);
    scalar[38] = (progress.ukeire_live_count as f32 / 34.0).min(1.0);
    scalar[39] = 0.0;
    scalar[40] = (tenpai_live_tile_count as f32 / 136.0).min(1.0);
    scalar[41] = pair_count as f32 / 7.0;
    scalar[42] = taatsu_count as f32 / 6.0;
    scalar[43] = ankoutsu_cnt as f32 / 4.0;
    scalar[44] = f32::from(round_state.players[actor].furiten);
    scalar[45] = riichi_threat_count as f32 / 3.0;
    scalar[46] = chiitoi as f32 / 6.0;
    scalar[47] = kokushi as f32 / 13.0;
    scalar[48] = f32::from(actor as i8 == round_state.oya);
    scalar[49] = is_all_last_like;
    scalar[50] = top_gap as f32 / 30000.0;
    scalar[51] = bottom_gap as f32 / 30000.0;
    scalar[52] = tiles_left as f32 / 70.0;
    scalar[53] = 0.0;
    scalar[54] = 0.0;
    scalar[55] = wait_hiddenness_ratio.clamp(0.0, 1.0);

    Ok(scalar
        .into_iter()
        .map(|value| f16::from_f32(value).to_bits())
        .collect())
}

fn existing_export_sample_count(path: &Path) -> Result<Option<usize>, String> {
    let sample_count = read_npy_first_dim_from_zip(path, "state_tile_feat.npy")?;
    let Some(sample_count) = sample_count else {
        return Ok(None);
    };
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open npz {}: {err}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|err| format!("failed to open npz {}: {err}", path.display()))?;
    for required_member in ["schema_name.npy", "schema_version.npy"] {
        if zip.by_name(required_member).is_err() {
            return Err(format!(
                "existing export missing required member {required_member} in {}",
                path.display()
            ));
        }
    }
    Ok(Some(sample_count))
}

#[derive(Debug)]
struct FileExportResult {
    source_file: String,
    ds_name: String,
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    produced_npz: bool,
    elapsed_s: f64,
    timings: ExportStageTimings,
}

fn resolved_export_jobs(requested_jobs: usize, total_files: usize) -> usize {
    if total_files <= 1 {
        return total_files.max(1);
    }
    let auto_jobs = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let jobs = if requested_jobs == 0 {
        auto_jobs
    } else {
        requested_jobs
    };
    jobs.clamp(1, total_files)
}

fn process_export_file(
    file: &str,
    output_path: &Path,
    resume: bool,
) -> Result<FileExportResult, String> {
    let file_t0 = Instant::now();
    begin_export_profile_scope();
    let input_path = Path::new(file);
    let ds_name = input_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    let out_file = output_npz_path(output_path, file);
    poll_cancel(&format!("before exporting {}", input_path.display()))?;
    let test_sleep_ms = export_test_file_sleep_ms();
    if test_sleep_ms > 0 {
        thread::sleep(Duration::from_millis(test_sleep_ms));
        poll_cancel(&format!("before exporting {}", input_path.display()))?;
    }

    if resume {
        match existing_export_sample_count(&out_file) {
            Ok(Some(existing_sample_count)) => {
                return Ok(FileExportResult {
                    source_file: file.to_string(),
                    ds_name,
                    exported_file_count: 1,
                    exported_sample_count: existing_sample_count,
                    processed_file_count: 0,
                    skipped_existing_file_count: 1,
                    produced_npz: true,
                    elapsed_s: file_t0.elapsed().as_secs_f64(),
                    timings: finish_export_profile_scope(),
                });
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!(
                    "[xmodel1 preprocess] ignoring corrupt existing export {}: {}",
                    out_file.display(),
                    err
                );
                if out_file.exists() {
                    fs::remove_file(&out_file).map_err(|remove_err| {
                        format!(
                            "failed to remove corrupt export {} after probe error ({err}): {remove_err}",
                            out_file.display()
                        )
                    })?;
                }
            }
        }
    }

    let records = collect_records_from_mjson(file)
        .map_err(|err| format!("failed to export {file}: {err}"))?;
    if !records.is_empty() {
        let out_dir = out_file.parent().unwrap_or(output_path);
        fs::create_dir_all(out_dir).map_err(|err| {
            format!(
                "failed to create dataset output dir {}: {err}",
                out_dir.display()
            )
        })?;
        with_export_stage_timing(ExportStage::NpzWrite, || {
            write_full_npz(&out_file, &records)
        })
        .map_err(|err| format!("failed to write export for {file}: {err}"))?;
        return Ok(FileExportResult {
            source_file: file.to_string(),
            ds_name,
            exported_file_count: 1,
            exported_sample_count: records.len(),
            processed_file_count: 1,
            skipped_existing_file_count: 0,
            produced_npz: true,
            elapsed_s: file_t0.elapsed().as_secs_f64(),
            timings: finish_export_profile_scope(),
        });
    }

    Ok(FileExportResult {
        source_file: file.to_string(),
        ds_name,
        exported_file_count: 0,
        exported_sample_count: 0,
        processed_file_count: 0,
        skipped_existing_file_count: 0,
        produced_npz: false,
        elapsed_s: file_t0.elapsed().as_secs_f64(),
        timings: finish_export_profile_scope(),
    })
}

fn apply_round_target_updates(
    records: &mut [FullRecord],
    updates: &[replay_core::RoundTargetUpdate],
) {
    replay_core::apply_round_target_updates(records, updates, |record, update| {
        record.score_delta_target = update.score_delta_target;
        record.global_value_target = update.global_value_target;
        record.win_target = update.win_target;
        record.dealin_target = update.dealin_target;
        record.pts_given_win_target = update.pts_given_win_target;
        record.pts_given_dealin_target = update.pts_given_dealin_target;
    });
}

fn collect_records_from_mjson(path: &str) -> Result<Vec<FullRecord>, String> {
    ensure_init();
    poll_cancel(&format!("before normalizing {path}"))?;
    let events = with_export_stage_timing(ExportStage::Normalize, || {
        replay_core::normalize_replay_events(path)
    })?;
    // v3: history_summary 统一由 normalized events 计算并贯穿 preprocess/runtime/parity。
    let events_slice: &[Value] = events.as_slice();
    let mut records = with_export_stage_timing(ExportStage::RecordDrive, || {
        replay_core::drive_export_records(
            &events,
            SCORE_NORM,
            MC_RETURN_GAMMA,
            |ctx| {
                let Some(decision) = replay_core::build_special_actor_chosen_action_context(ctx)?
                else {
                    return Ok(None);
                };
                let sample_type = match decision.decision.event.et {
                    "reach" => XMODEL1_SAMPLE_TYPE_RIICHI,
                    "hora" => XMODEL1_SAMPLE_TYPE_HORA,
                    _ => XMODEL1_SAMPLE_TYPE_CALL,
                };
                let history_summary = replay_core::compute_history_summary(
                    events_slice,
                    decision.decision.event.event_index,
                    decision.decision.event.actor,
                );
                let human_post_discard = human_response_post_discard_event(
                    events_slice,
                    decision.decision.event.event_index,
                    decision.decision.event.actor,
                    &decision.chosen_action,
                );
                encode_special_sample_record(
                    decision.decision.event.state,
                    decision.decision.event.core_state,
                    decision.decision.event.actor,
                    &decision.decision.legal_actions,
                    &decision.chosen_action,
                    sample_type,
                    decision.decision.event.event_index,
                    history_summary,
                    human_post_discard.as_ref(),
                )
                .map(Some)
            },
            |ctx| {
                let Some(discard) = replay_core::build_discard_decision_context(ctx) else {
                    return Ok(None);
                };
                let history_summary = replay_core::compute_history_summary(
                    events_slice,
                    discard.event.event_index,
                    discard.event.actor,
                );
                encode_candidate_features(
                    discard.event.state,
                    discard.event.core_state,
                    discard.event.actor,
                    &discard.chosen_tile,
                    discard.event.event_index,
                    history_summary,
                )
                .map(Some)
            },
            |ctx| {
                let reaction = replay_core::build_reaction_none_context(ctx);
                let history_summary = replay_core::compute_history_summary(
                    events_slice,
                    reaction.reaction.event_index,
                    reaction.reaction.actor,
                );
                encode_special_sample_record(
                    reaction.reaction.state,
                    reaction.reaction.core_state,
                    reaction.reaction.actor,
                    reaction.reaction.legal_actions,
                    &reaction.chosen_action,
                    2,
                    reaction.reaction.event_index,
                    history_summary,
                    None,
                )
                .map(Some)
            },
            apply_round_target_updates,
            |event_index| poll_cancel(&format!("while scanning {path} at event {event_index}")),
        )
    })?;
    apply_final_game_targets(&mut records, events_slice);
    Ok(records)
}

fn write_full_npz(path: &Path, records: &[FullRecord]) -> Result<(), String> {
    let temp_path = temp_npz_path(path);
    let write_result = (|| {
        let file = fs::File::create(&temp_path)
            .map_err(|err| format!("failed to create temp npz {}: {err}", temp_path.display()))?;
        let mut zip = ZipWriter::new(file);
        let n = records.len();

        let mut state_tile_feat =
            Vec::with_capacity(n * XMODEL1_STATE_TILE_CHANNELS * TILE_KIND_COUNT);
        let mut state_scalar = Vec::with_capacity(n * XMODEL1_STATE_SCALAR_DIM);
        let mut candidate_feat =
            Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FEATURE_DIM);
        let mut candidate_tile_id = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
        let mut candidate_mask = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
        let mut candidate_flags =
            Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES * XMODEL1_CANDIDATE_FLAG_DIM);
        let mut chosen_candidate_idx = Vec::with_capacity(n);
        let mut candidate_quality = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
        let mut candidate_hard_bad = Vec::with_capacity(n * XMODEL1_MAX_CANDIDATES);
        let mut response_action_idx = Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES);
        let mut response_action_mask = Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES);
        let mut chosen_response_action_idx = Vec::with_capacity(n);
        let mut response_post_candidate_feat = Vec::with_capacity(
            n * XMODEL1_MAX_RESPONSE_CANDIDATES
                * XMODEL1_MAX_CANDIDATES
                * XMODEL1_CANDIDATE_FEATURE_DIM,
        );
        let mut response_post_candidate_tile_id =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES);
        let mut response_post_candidate_mask =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES);
        let mut response_post_candidate_flags = Vec::with_capacity(
            n * XMODEL1_MAX_RESPONSE_CANDIDATES
                * XMODEL1_MAX_CANDIDATES
                * XMODEL1_CANDIDATE_FLAG_DIM,
        );
        let mut response_post_candidate_quality =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES);
        let mut response_post_candidate_hard_bad =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES * XMODEL1_MAX_CANDIDATES);
        let mut response_teacher_discard_idx =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES);
        let mut response_human_discard_idx =
            Vec::with_capacity(n * XMODEL1_MAX_RESPONSE_CANDIDATES);
        let mut action_idx_target = Vec::with_capacity(n);
        let mut win_target = Vec::with_capacity(n);
        let mut dealin_target = Vec::with_capacity(n);
        let mut pts_given_win_target = Vec::with_capacity(n);
        let mut pts_given_dealin_target = Vec::with_capacity(n);
        let mut opp_tenpai_target: Vec<f32> = Vec::with_capacity(n * 3);
        let mut final_rank_target = Vec::with_capacity(n);
        let mut final_score_delta_points_target = Vec::with_capacity(n);
        let mut history_summary = Vec::with_capacity(n * replay_core::HISTORY_SUMMARY_DIM);
        let mut rule_context = Vec::with_capacity(n * XMODEL1_RULE_CONTEXT_DIM);
        let mut sample_type = Vec::with_capacity(n);
        let mut actor = Vec::with_capacity(n);
        let mut event_index = Vec::with_capacity(n);
        let mut kyoku = Vec::with_capacity(n);
        let mut honba = Vec::with_capacity(n);
        let mut is_open_hand = Vec::with_capacity(n);

        for (idx, record) in records.iter().enumerate() {
            if idx % 256 == 0 {
                poll_cancel(&format!("while flattening {}", path.display()))?;
            }
            state_tile_feat.extend_from_slice(&record.state_tile_feat);
            state_scalar.extend_from_slice(&record.state_scalar);
            candidate_feat.extend_from_slice(&record.candidate_feat);
            candidate_tile_id.extend_from_slice(&record.candidate_tile_id);
            candidate_mask.extend_from_slice(&record.candidate_mask);
            candidate_flags.extend_from_slice(&record.candidate_flags);
            chosen_candidate_idx.push(record.chosen_candidate_idx);
            candidate_quality.extend_from_slice(&record.candidate_quality);
            candidate_hard_bad.extend_from_slice(&record.candidate_hard_bad);
            response_action_idx.extend_from_slice(&record.response_action_idx);
            response_action_mask.extend_from_slice(&record.response_action_mask);
            chosen_response_action_idx.push(record.chosen_response_action_idx);
            response_post_candidate_feat.extend_from_slice(&record.response_post_candidate_feat);
            response_post_candidate_tile_id
                .extend_from_slice(&record.response_post_candidate_tile_id);
            response_post_candidate_mask.extend_from_slice(&record.response_post_candidate_mask);
            response_post_candidate_flags.extend_from_slice(&record.response_post_candidate_flags);
            response_post_candidate_quality
                .extend_from_slice(&record.response_post_candidate_quality);
            response_post_candidate_hard_bad
                .extend_from_slice(&record.response_post_candidate_hard_bad);
            response_teacher_discard_idx.extend_from_slice(&record.response_teacher_discard_idx);
            response_human_discard_idx.extend_from_slice(&record.response_human_discard_idx);
            action_idx_target.push(record.action_idx_target);
            win_target.push(record.win_target);
            dealin_target.push(record.dealin_target);
            pts_given_win_target.push(record.pts_given_win_target);
            pts_given_dealin_target.push(record.pts_given_dealin_target);
            opp_tenpai_target.extend_from_slice(&record.opp_tenpai_target);
            final_rank_target.push(record.final_rank_target as i8);
            final_score_delta_points_target.push(record.final_score_delta_points_target);
            for value in record.history_summary.iter() {
                history_summary.push(f16::from_f32(*value).to_bits());
            }
            for value in record.rule_context.iter() {
                rule_context.push(*value);
            }
            sample_type.push(record.sample_type);
            actor.push(record.actor);
            event_index.push(record.event_index);
            kyoku.push(record.kyoku);
            honba.push(record.honba);
            is_open_hand.push(record.is_open_hand);
        }

        poll_cancel(&format!("before finalizing {}", path.display()))?;
        write_npy_unicode_scalar(&mut zip, "schema_name.npy", XMODEL1_SCHEMA_NAME)?;
        write_npy_i32(
            &mut zip,
            "schema_version.npy",
            &[],
            &[XMODEL1_SCHEMA_VERSION as i32],
        )?;
        write_npy_f16(
            &mut zip,
            "state_tile_feat.npy",
            &[n, XMODEL1_STATE_TILE_CHANNELS, TILE_KIND_COUNT],
            &state_tile_feat,
        )?;
        write_npy_f16(
            &mut zip,
            "state_scalar.npy",
            &[n, XMODEL1_STATE_SCALAR_DIM],
            &state_scalar,
        )?;
        write_npy_f16(
            &mut zip,
            "candidate_feat.npy",
            &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM],
            &candidate_feat,
        )?;
        write_npy_i16(
            &mut zip,
            "candidate_tile_id.npy",
            &[n, XMODEL1_MAX_CANDIDATES],
            &candidate_tile_id,
        )?;
        write_npy_u8(
            &mut zip,
            "candidate_mask.npy",
            &[n, XMODEL1_MAX_CANDIDATES],
            &candidate_mask,
        )?;
        write_npy_u8(
            &mut zip,
            "candidate_flags.npy",
            &[n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM],
            &candidate_flags,
        )?;
        write_npy_i16(
            &mut zip,
            "chosen_candidate_idx.npy",
            &[n],
            &chosen_candidate_idx,
        )?;
        write_npy_f32(
            &mut zip,
            "candidate_quality_score.npy",
            &[n, XMODEL1_MAX_CANDIDATES],
            &candidate_quality,
        )?;
        write_npy_u8(
            &mut zip,
            "candidate_hard_bad_flag.npy",
            &[n, XMODEL1_MAX_CANDIDATES],
            &candidate_hard_bad,
        )?;
        write_npy_i16(
            &mut zip,
            "response_action_idx.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES],
            &response_action_idx,
        )?;
        write_npy_u8(
            &mut zip,
            "response_action_mask.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES],
            &response_action_mask,
        )?;
        write_npy_i16(
            &mut zip,
            "chosen_response_action_idx.npy",
            &[n],
            &chosen_response_action_idx,
        )?;
        write_npy_f16(
            &mut zip,
            "response_post_candidate_feat.npy",
            &[
                n,
                XMODEL1_MAX_RESPONSE_CANDIDATES,
                XMODEL1_MAX_CANDIDATES,
                XMODEL1_CANDIDATE_FEATURE_DIM,
            ],
            &response_post_candidate_feat,
        )?;
        write_npy_i16(
            &mut zip,
            "response_post_candidate_tile_id.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES],
            &response_post_candidate_tile_id,
        )?;
        write_npy_u8(
            &mut zip,
            "response_post_candidate_mask.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES],
            &response_post_candidate_mask,
        )?;
        write_npy_u8(
            &mut zip,
            "response_post_candidate_flags.npy",
            &[
                n,
                XMODEL1_MAX_RESPONSE_CANDIDATES,
                XMODEL1_MAX_CANDIDATES,
                XMODEL1_CANDIDATE_FLAG_DIM,
            ],
            &response_post_candidate_flags,
        )?;
        write_npy_f32(
            &mut zip,
            "response_post_candidate_quality_score.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES],
            &response_post_candidate_quality,
        )?;
        write_npy_u8(
            &mut zip,
            "response_post_candidate_hard_bad_flag.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES],
            &response_post_candidate_hard_bad,
        )?;
        write_npy_i16(
            &mut zip,
            "response_teacher_discard_idx.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES],
            &response_teacher_discard_idx,
        )?;
        write_npy_i16(
            &mut zip,
            "response_human_discard_idx.npy",
            &[n, XMODEL1_MAX_RESPONSE_CANDIDATES],
            &response_human_discard_idx,
        )?;
        write_npy_i16(&mut zip, "action_idx_target.npy", &[n], &action_idx_target)?;
        write_npy_f32(&mut zip, "win_target.npy", &[n], &win_target)?;
        write_npy_f32(&mut zip, "dealin_target.npy", &[n], &dealin_target)?;
        write_npy_f32(
            &mut zip,
            "pts_given_win_target.npy",
            &[n],
            &pts_given_win_target,
        )?;
        write_npy_f32(
            &mut zip,
            "pts_given_dealin_target.npy",
            &[n],
            &pts_given_dealin_target,
        )?;
        write_npy_f32(
            &mut zip,
            "opp_tenpai_target.npy",
            &[n, 3],
            &opp_tenpai_target,
        )?;
        write_npy_i8(&mut zip, "final_rank_target.npy", &[n], &final_rank_target)?;
        write_npy_i32(
            &mut zip,
            "final_score_delta_points_target.npy",
            &[n],
            &final_score_delta_points_target,
        )?;
        write_npy_f16(
            &mut zip,
            "history_summary.npy",
            &[n, replay_core::HISTORY_SUMMARY_DIM],
            &history_summary,
        )?;
        write_npy_f32(
            &mut zip,
            "rule_context.npy",
            &[n, XMODEL1_RULE_CONTEXT_DIM],
            &rule_context,
        )?;
        write_npy_i8(&mut zip, "sample_type.npy", &[n], &sample_type)?;
        write_npy_i8(&mut zip, "actor.npy", &[n], &actor)?;
        write_npy_i32(&mut zip, "event_index.npy", &[n], &event_index)?;
        write_npy_i8(&mut zip, "kyoku.npy", &[n], &kyoku)?;
        write_npy_i8(&mut zip, "honba.npy", &[n], &honba)?;
        write_npy_u8(&mut zip, "is_open_hand.npy", &[n], &is_open_hand)?;
        finalize_temp_npz(zip, &temp_path, path, true)
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }
    Ok(())
}

fn write_manifest(
    output_dir: &str,
    files: &[String],
    exported_file_count: usize,
    exported_sample_count: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    shard_file_counts: &BTreeMap<String, usize>,
    shard_sample_counts: &BTreeMap<String, usize>,
    used_fallback: bool,
    export_mode: &'static str,
) -> Result<String, String> {
    let output_path = Path::new(output_dir);
    let manifest = ExportManifest {
        schema_name: XMODEL1_SCHEMA_NAME,
        schema_version: XMODEL1_SCHEMA_VERSION,
        max_candidates: XMODEL1_MAX_CANDIDATES,
        candidate_feature_dim: XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim: XMODEL1_CANDIDATE_FLAG_DIM,
        file_count: files.len(),
        exported_file_count,
        exported_sample_count,
        processed_file_count,
        skipped_existing_file_count,
        shard_file_counts: shard_file_counts.clone(),
        shard_sample_counts: shard_sample_counts.clone(),
        used_fallback,
        export_mode,
        files: files.to_vec(),
    };
    write_json_manifest(output_path, "xmodel1_export_manifest.json", &manifest)
}

pub fn build_xmodel1_discard_records(
    data_dirs: &[String],
    output_dir: &str,
    smoke: bool,
) -> Result<(usize, String, bool), String> {
    build_xmodel1_discard_records_with_options(
        data_dirs,
        output_dir,
        ExportRunOptions {
            smoke,
            resume: false,
            progress_every: 0,
            jobs: 1,
            limit_files: 0,
        },
    )
}

pub fn build_xmodel1_discard_records_with_options(
    data_dirs: &[String],
    output_dir: &str,
    options: ExportRunOptions,
) -> Result<(usize, String, bool), String> {
    ensure_cancel_handler_installed()?;
    reset_cancel_flag();
    let mut files = collect_mjson_files(data_dirs, options.smoke)?;
    if options.limit_files > 0 {
        files.truncate(options.limit_files);
    }
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path).map_err(|err| {
        format!(
            "failed to create xmodel1 output dir {}: {err}",
            output_path.display()
        )
    })?;
    let start = Instant::now();
    let jobs_count = resolved_export_jobs(options.jobs, files.len());
    let mut produced_npz = false;
    let mut exported_file_count = 0usize;
    let mut exported_sample_count = 0usize;
    let mut processed_file_count = 0usize;
    let mut skipped_existing_file_count = 0usize;
    let mut shard_file_counts = BTreeMap::<String, usize>::new();
    let mut shard_sample_counts = BTreeMap::<String, usize>::new();
    let progress_every = options.progress_every.max(1);
    let mut recent_file_seconds = VecDeque::with_capacity(32);
    let mut accumulated_timings = ExportStageTimings::default();

    let (tx, rx) = mpsc::channel::<Result<FileExportResult, String>>();
    let work_queue = Arc::new(Mutex::new(files.clone()));
    let mut handles = Vec::new();
    for _ in 0..jobs_count {
        let tx = tx.clone();
        let queue = work_queue.clone();
        let output_dir = output_path.to_path_buf();
        let resume = options.resume;
        handles.push(thread::spawn(move || loop {
            if cancel_requested() {
                break;
            }
            let maybe_file = {
                let mut queue = queue.lock().expect("work queue poisoned");
                queue.pop()
            };
            let Some(file) = maybe_file else {
                break;
            };
            if cancel_requested() {
                break;
            }
            let result = process_export_file(&file, &output_dir, resume);
            let should_stop = matches!(&result, Err(err) if is_interrupted_error(err));
            let _ = tx.send(result);
            if should_stop {
                break;
            }
        }));
    }
    drop(tx);

    let mut done = 0usize;
    let mut first_error: Option<String> = None;
    let mut interrupted = false;
    while let Ok(result) = rx.recv() {
        done += 1;
        match result {
            Ok(result) => {
                produced_npz |= result.produced_npz;
                exported_file_count += result.exported_file_count;
                exported_sample_count += result.exported_sample_count;
                processed_file_count += result.processed_file_count;
                skipped_existing_file_count += result.skipped_existing_file_count;
                accumulated_timings += result.timings;
                if result.exported_file_count > 0 {
                    *shard_file_counts.entry(result.ds_name.clone()).or_insert(0) +=
                        result.exported_file_count;
                    *shard_sample_counts.entry(result.ds_name).or_insert(0) +=
                        result.exported_sample_count;
                }
                if export_profile_enabled() {
                    let per_sample_ms = if result.exported_sample_count > 0 {
                        1000.0 * result.elapsed_s / result.exported_sample_count as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[xmodel1 preprocess][profile] file={} samples={} total={:.3}s sample_ms={:.3} normalize={:.3}s record_drive={:.3}s discard_candidate_analysis={:.3}s special_summary={:.3}s npz_write={:.3}s",
                        result.source_file,
                        result.exported_sample_count,
                        result.elapsed_s,
                        per_sample_ms,
                        result.timings.normalize_s,
                        result.timings.record_drive_s,
                        result.timings.discard_candidate_analysis_s,
                        result.timings.special_summary_s,
                        result.timings.npz_write_s,
                    );
                }
                recent_file_seconds.push_back(result.elapsed_s);
                while recent_file_seconds.len() > 32 {
                    recent_file_seconds.pop_front();
                }
            }
            Err(err) => {
                if is_interrupted_error(&err) {
                    interrupted = true;
                    EXPORT_CANCELLED.store(true, Ordering::SeqCst);
                } else {
                    first_error = Some(err);
                    EXPORT_CANCELLED.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
        if options.progress_every > 0 && ((done % progress_every == 0) || (done == files.len())) {
            print_export_progress(
                "xmodel1",
                &start,
                done,
                files.len(),
                processed_file_count,
                skipped_existing_file_count,
                exported_sample_count,
                &recent_file_seconds,
                jobs_count,
            );
        }
    }
    for handle in handles {
        let _ = handle.join();
    }
    if export_profile_enabled() {
        let elapsed = start.elapsed().as_secs_f64();
        let avg_file_s = if done > 0 { elapsed / done as f64 } else { 0.0 };
        let avg_sample_ms = if exported_sample_count > 0 {
            1000.0 * elapsed / exported_sample_count as f64
        } else {
            0.0
        };
        eprintln!(
            "[xmodel1 preprocess][profile] total_wall={:.3}s files={} processed_files={} exported_files={} samples={} avg_file={:.3}s avg_sample={:.3}ms accumulated_stage={:.3}s normalize={:.3}s record_drive={:.3}s discard_candidate_analysis={:.3}s special_summary={:.3}s npz_write={:.3}s",
            elapsed,
            files.len(),
            processed_file_count,
            exported_file_count,
            exported_sample_count,
            avg_file_s,
            avg_sample_ms,
            accumulated_timings.accumulated_s(),
            accumulated_timings.normalize_s,
            accumulated_timings.record_drive_s,
            accumulated_timings.discard_candidate_analysis_s,
            accumulated_timings.special_summary_s,
            accumulated_timings.npz_write_s,
        );
    }
    let export_mode = if options.smoke {
        "rust_full_npz_smoke_export"
    } else {
        "rust_full_npz_export"
    };
    let manifest_path = write_manifest(
        output_dir,
        &files,
        exported_file_count,
        exported_sample_count,
        processed_file_count,
        skipped_existing_file_count,
        &shard_file_counts,
        &shard_sample_counts,
        false,
        export_mode,
    )?;
    if let Some(err) = first_error {
        return Err(err);
    }
    if interrupted {
        return Err(format!(
            "{}; manifest={manifest_path}; processed_files={processed_file_count}; skipped_existing_files={skipped_existing_file_count}; exported_files={exported_file_count}",
            interrupted_error("safe partial manifest written")
        ));
    }
    Ok((files.len(), manifest_path, produced_npz))
}
