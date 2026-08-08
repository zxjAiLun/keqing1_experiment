use serde::Serialize;
use serde_json::{json, Value};

use crate::event_apply::apply_event as apply_event_core;
use crate::hora_truth::evaluate_hora_truth_from_prepared;
use crate::legal_actions::{
    enumerate_hora_candidates, enumerate_legal_action_specs_structural,
    prepare_hora_evaluation_from_snapshot,
};
use crate::replay_export_core::{
    self as replay_core, PendingRoundRecord, RoundState, RoundTargetUpdate,
};
use crate::snapshot::snapshot_for_actor;
use crate::state_core::GameStateCore;

#[derive(Debug, Clone, Serialize)]
pub struct ReplayDecisionRecord {
    pub state: Value,
    pub actor: usize,
    pub label_action: Value,
    pub legal_actions: Vec<Value>,
    pub value_target: f32,
    pub score_delta_target: f32,
    pub win_target: f32,
    pub dealin_target: f32,
    pub pts_given_win_target: f32,
    pub pts_given_dealin_target: f32,
    pub ryukyoku_tenpai_target: f32,
    pub opp_tenpai_target: [f32; 3],
    pub score_before_action: i32,
    pub final_score_delta_points_target: i32,
    pub final_rank_target: u8,
    pub event_index: i32,
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

fn next_meaningful_event<'a>(events: &'a [Value], event_index: usize) -> Option<&'a Value> {
    for event in events.iter().skip(event_index + 1) {
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        if !matches!(et, "reach_accepted" | "dora" | "new_dora") {
            return Some(event);
        }
    }
    None
}

fn sample_snapshot_value(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    hora_event: Option<&Value>,
) -> Result<Value, String> {
    let snapshot = snapshot_for_actor(core_state, actor);
    let mut value = serde_json::to_value(&snapshot)
        .map_err(|err| format!("failed to serialize replay sample snapshot: {err}"))?;
    if let Some(event) = hora_event {
        if event
            .get("is_haitei")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            value["_hora_is_haitei"] = json!(true);
        }
        if event
            .get("is_houtei")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            value["_hora_is_houtei"] = json!(true);
        }
        if event
            .get("is_rinshan")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            value["_hora_is_rinshan"] = json!(true);
        }
        if event
            .get("is_chankan")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            value["_hora_is_chankan"] = json!(true);
        }
    }
    if let Some(player) = round_state.feature_tracker.players.get(actor) {
        value["feature_tracker"] = json!({
            "actor": actor,
            "hand_tiles": player.hand_tiles,
            "meld_tiles": player.meld_tiles,
            "hand_counts34": player.hand_counts34.to_vec(),
            "meld_counts34": player.meld_counts34.to_vec(),
            "visible_counts34": player.visible_counts34.to_vec(),
            "discards_count": player.discards_count,
            "meld_count": player.meld_count,
            "pair_count": player.pair_count,
            "ankoutsu_count": player.ankoutsu_count,
            "suit_counts": player.suit_counts.to_vec(),
            "aka_counts": player.aka_counts.to_vec(),
        });
    }
    Ok(value)
}

fn resolved_label_action(round_state: &RoundState, event: &Value, actor: usize) -> Option<Value> {
    let mut label = replay_core::resolve_label_action(round_state, event)?;
    if label.get("actor").is_none() {
        label["actor"] = json!(actor);
    }
    Some(label)
}

fn candidate_flag_bool(candidate: &Value, key: &str) -> Option<bool> {
    candidate.get(key).and_then(Value::as_bool)
}

fn can_hora_from_snapshot_candidate(
    state_snapshot: &Value,
    actor: usize,
    target: usize,
    pai: &str,
    is_tsumo: bool,
    candidate: &Value,
) -> Result<bool, String> {
    let prepared = prepare_hora_evaluation_from_snapshot(
        state_snapshot,
        actor,
        pai,
        is_tsumo,
        candidate_flag_bool(candidate, "is_chankan").unwrap_or(false),
        candidate_flag_bool(candidate, "is_rinshan"),
        candidate_flag_bool(candidate, "is_haitei"),
        candidate_flag_bool(candidate, "is_houtei"),
    )?;
    match evaluate_hora_truth_from_prepared(&prepared) {
        Ok(_) => Ok(true),
        Err(message) if message == "no cost" => Ok(false),
        Err(message) => Err(format!(
            "failed to evaluate hora truth for actor={actor} target={target} pai={pai}: {message}"
        )),
    }
}

pub(crate) fn public_legal_actions_for_snapshot(
    state_snapshot: &Value,
    actor: usize,
) -> Result<Vec<Value>, String> {
    let mut legal = Vec::new();
    for candidate in enumerate_hora_candidates(state_snapshot, actor)? {
        let target = candidate
            .get("target")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| "invalid hora candidate target".to_string())?;
        let pai = candidate
            .get("pai")
            .and_then(Value::as_str)
            .ok_or_else(|| "invalid hora candidate pai".to_string())?;
        let is_tsumo = candidate
            .get("is_tsumo")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if can_hora_from_snapshot_candidate(
            state_snapshot,
            actor,
            target,
            pai,
            is_tsumo,
            &candidate,
        )? {
            legal.push(json!({
                "type": "hora",
                "actor": actor,
                "target": target,
                "pai": pai,
            }));
        }
    }
    legal.extend(enumerate_legal_action_specs_structural(
        state_snapshot,
        actor,
    )?);
    Ok(legal)
}

fn actor_visible_counts(
    round_state: &RoundState,
    actor: usize,
) -> [u8; replay_core::TILE_KIND_COUNT] {
    round_state
        .feature_tracker
        .players
        .get(actor)
        .map(|player| player.visible_counts34)
        .unwrap_or([0; replay_core::TILE_KIND_COUNT])
}

fn build_actor_record(
    sample_round_state: &RoundState,
    sample_core_state: &GameStateCore,
    event: &Value,
    actor: usize,
    event_index: i32,
) -> Result<Option<ReplayDecisionRecord>, String> {
    let hora_event = (event.get("type").and_then(Value::as_str) == Some("hora")).then_some(event);
    let state = sample_snapshot_value(sample_round_state, sample_core_state, actor, hora_event)?;
    let hand_visible = state
        .get("hand")
        .and_then(Value::as_array)
        .map(|hand| !hand.is_empty())
        .unwrap_or(false);
    if !hand_visible {
        return Ok(None);
    }
    let legal_actions = public_legal_actions_for_snapshot(&state, actor)?;
    if legal_actions.is_empty() {
        return Ok(None);
    }
    let Some(label_action) = resolved_label_action(sample_round_state, event, actor) else {
        return Ok(None);
    };
    Ok(Some(ReplayDecisionRecord {
        state,
        actor,
        label_action,
        legal_actions,
        value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        ryukyoku_tenpai_target: 0.0,
        opp_tenpai_target: replay_core::compute_opp_tenpai_target(
            sample_round_state,
            actor,
            &actor_visible_counts(sample_round_state, actor),
        ),
        score_before_action: sample_round_state.scores[actor],
        final_score_delta_points_target: 0,
        final_rank_target: 0,
        event_index,
    }))
}

fn build_reaction_none_record(
    round_state: &RoundState,
    core_state: &GameStateCore,
    actor: usize,
    event_index: i32,
) -> Result<Option<ReplayDecisionRecord>, String> {
    let state = sample_snapshot_value(round_state, core_state, actor, None)?;
    let hand_visible = state
        .get("hand")
        .and_then(Value::as_array)
        .map(|hand| !hand.is_empty())
        .unwrap_or(false);
    if !hand_visible {
        return Ok(None);
    }
    let legal_actions = public_legal_actions_for_snapshot(&state, actor)?;
    let has_non_none = legal_actions
        .iter()
        .any(|action| action.get("type").and_then(Value::as_str) != Some("none"));
    if !has_non_none {
        return Ok(None);
    }
    Ok(Some(ReplayDecisionRecord {
        state,
        actor,
        label_action: json!({"type": "none", "actor": actor}),
        legal_actions,
        value_target: 0.0,
        score_delta_target: 0.0,
        win_target: 0.0,
        dealin_target: 0.0,
        pts_given_win_target: 0.0,
        pts_given_dealin_target: 0.0,
        ryukyoku_tenpai_target: 0.0,
        opp_tenpai_target: replay_core::compute_opp_tenpai_target(
            round_state,
            actor,
            &actor_visible_counts(round_state, actor),
        ),
        score_before_action: round_state.scores[actor],
        final_score_delta_points_target: 0,
        final_rank_target: 0,
        event_index,
    }))
}

fn push_record(
    records: &mut Vec<ReplayDecisionRecord>,
    pending_records: &mut Vec<PendingRoundRecord>,
    round_state: &mut RoundState,
    record: ReplayDecisionRecord,
) {
    let actor = record.actor;
    let record_index = records.len();
    records.push(record);
    pending_records.push(PendingRoundRecord {
        record_index,
        round_step_index: round_state.round_step_index,
        actor: actor as i8,
    });
    round_state.round_step_index += 1;
}

fn apply_round_target_updates(records: &mut [ReplayDecisionRecord], updates: &[RoundTargetUpdate]) {
    for update in updates {
        if let Some(record) = records.get_mut(update.record_index) {
            record.value_target = update.global_value_target;
            record.score_delta_target = update.score_delta_target;
            record.win_target = update.win_target;
            record.dealin_target = update.dealin_target;
            record.pts_given_win_target = update.pts_given_win_target;
            record.pts_given_dealin_target = update.pts_given_dealin_target;
            record.ryukyoku_tenpai_target = update.ryukyoku_tenpai_target;
        }
    }
}

pub fn build_replay_decision_records_mc_return(
    events: &[Value],
) -> Result<Vec<ReplayDecisionRecord>, String> {
    let mut round_state = RoundState::default();
    let mut core_state = GameStateCore::default();
    let mut records = Vec::<ReplayDecisionRecord>::new();
    let mut pending_records = Vec::<PendingRoundRecord>::new();
    let mut multi_ron_round_state: Option<RoundState> = None;
    let mut multi_ron_core_state: Option<GameStateCore> = None;

    for (event_index, event) in events.iter().enumerate() {
        let event_index_i32 = event_index as i32;
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        let actor = replay_core::value_usize(event, "actor");
        let next_ev = next_meaningful_event(events, event_index);

        replay_core::begin_round_event(&mut round_state, &mut pending_records, et);
        if et == "start_kyoku" {
            multi_ron_round_state = None;
            multi_ron_core_state = None;
        }

        if et == "hora"
            && multi_ron_round_state.is_none()
            && next_ev
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str)
                == Some("hora")
        {
            multi_ron_round_state = Some(round_state.clone());
            multi_ron_core_state = Some(core_state.clone());
        }

        let collect_sample = actor
            .map(|actor| {
                replay_core::is_structural_actor_decision_event(et)
                    && round_state.in_game
                    && !(et == "dahai" && round_state.players[actor].reached)
            })
            .unwrap_or(false);

        if collect_sample {
            let actor = actor.unwrap_or(0);
            let sample_round_state = if et == "hora" {
                multi_ron_round_state.as_ref().unwrap_or(&round_state)
            } else {
                &round_state
            };
            let sample_core_state = if et == "hora" {
                multi_ron_core_state.as_ref().unwrap_or(&core_state)
            } else {
                &core_state
            };
            if let Some(record) = build_actor_record(
                sample_round_state,
                sample_core_state,
                event,
                actor,
                event_index_i32,
            )? {
                push_record(&mut records, &mut pending_records, &mut round_state, record);
            }
        }

        replay_core::apply_event_round_state(&mut round_state, event);
        apply_event_core(&mut core_state, event).map_err(|err| {
            format!("failed to apply replay sample event at {event_index}: {err}")
        })?;

        if replay_core::is_discard_followed_by_tsumo(et, next_ev) && round_state.in_game {
            let discarder = actor.unwrap_or(usize::MAX);
            for reaction_actor in 0..4 {
                if reaction_actor == discarder {
                    continue;
                }
                if let Some(record) = build_reaction_none_record(
                    &round_state,
                    &core_state,
                    reaction_actor,
                    event_index_i32,
                )? {
                    push_record(&mut records, &mut pending_records, &mut round_state, record);
                }
            }
        }

        if let Some(updates) = replay_core::maybe_finalize_round_targets(
            &mut round_state,
            &pending_records,
            et,
            event,
            30000.0,
            0.99,
        ) {
            apply_round_target_updates(&mut records, &updates);
        }
        replay_core::reset_after_end_kyoku(&mut round_state, &mut pending_records, et);

        if et == "hora"
            && next_ev
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str)
                != Some("hora")
        {
            multi_ron_round_state = None;
            multi_ron_core_state = None;
        }
    }

    if let Some(updates) = replay_core::finalize_remaining_round_targets(
        &mut round_state,
        &pending_records,
        30000.0,
        0.99,
    ) {
        apply_round_target_updates(&mut records, &updates);
    }

    let game_start_oya = round_state.game_start_oya.clamp(0, 3) as usize;
    let final_rank_targets = final_rank_targets(&round_state.scores, game_start_oya);
    for record in &mut records {
        if record.actor < 4 {
            record.final_score_delta_points_target =
                round_state.scores[record.actor] - record.score_before_action;
            record.final_rank_target = final_rank_targets[record.actor];
        } else {
            record.final_score_delta_points_target = 0;
            record.final_rank_target = 0;
        }
    }

    Ok(records)
}
