use serde_json::{json, Value};

use crate::legal_actions::{
    calc_shanten_waits_like_python, can_hora_from_snapshot_candidate, counts34_from_hand,
    tile_to_34,
};
use crate::state_core::{GameStateCore, PlayerStateCore};
use crate::types::{DiscardEntry, LastDiscard};

fn normalize_or_keep_aka(tile: &str) -> String {
    match tile {
        "5mr" | "5pr" | "5sr" => tile.to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

fn get_usize(event: &Value, key: &str) -> Result<usize, String> {
    event
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing or invalid usize field: {}", key))
}

fn get_i32(event: &Value, key: &str) -> Result<i32, String> {
    event
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| format!("missing or invalid i32 field: {}", key))
}

fn get_str<'a>(event: &'a Value, key: &str) -> Result<&'a str, String> {
    event
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid string field: {}", key))
}

fn remove_tile(
    hand: &mut std::collections::BTreeMap<String, u8>,
    tile: &str,
) -> Result<(), String> {
    match hand.get_mut(tile) {
        Some(count) if *count > 1 => {
            *count -= 1;
            Ok(())
        }
        Some(_) => {
            hand.remove(tile);
            Ok(())
        }
        None => Err(format!("missing tile in hand: {}", tile)),
    }
}

fn consume_wall_for_kan(state: &mut GameStateCore) {
    if let Some(remaining_wall) = state.remaining_wall.as_mut() {
        *remaining_wall = (*remaining_wall).max(1) - 1;
    }
}

fn tile_family_key(tile: &str) -> String {
    match tile {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

fn refresh_furiten(player: &mut PlayerStateCore) {
    player.furiten = player.sutehai_furiten || player.riichi_furiten || player.doujun_furiten;
}

fn actor_waits_on_tile(state: &GameStateCore, actor: usize, pai: &str) -> bool {
    let Some(tile34) = tile_to_34(pai) else {
        return false;
    };
    let player = &state.players[actor];
    let counts34 = counts34_from_hand(&player.hand);
    let (_, _, waits) = calc_shanten_waits_like_python(&counts34, !player.melds.is_empty());
    waits[tile34]
}

fn recompute_sutehai_furiten_after_discard(state: &mut GameStateCore, actor: usize) {
    let player = &state.players[actor];
    let counts34 = counts34_from_hand(&player.hand);
    let (_, _, waits) = calc_shanten_waits_like_python(&counts34, !player.melds.is_empty());
    let discarded_wait = player.discards.iter().any(|discard| {
        tile_to_34(&discard.pai)
            .map(|tile34| waits[tile34])
            .unwrap_or(false)
    });

    let player = &mut state.players[actor];
    player.sutehai_furiten = discarded_wait;
    if player.reached && discarded_wait {
        player.riichi_furiten = true;
    }
    player.doujun_furiten = false;
    refresh_furiten(player);
}

fn mark_same_cycle_furiten(state: &mut GameStateCore, actor: usize) {
    let player = &mut state.players[actor];
    if player.reached {
        player.riichi_furiten = true;
    } else {
        player.doujun_furiten = true;
    }
    refresh_furiten(player);
}

fn can_current_ron_from_state(
    state: &GameStateCore,
    actor: usize,
    target: usize,
    pai: &str,
    is_chankan: bool,
) -> bool {
    let Ok(state_snapshot) =
        serde_json::to_value(crate::snapshot::snapshot_for_actor(state, actor))
    else {
        return false;
    };
    let mut candidate = json!({
        "target": target,
        "pai": pai,
        "is_tsumo": false,
    });
    if is_chankan {
        candidate["is_chankan"] = Value::Bool(true);
    }
    can_hora_from_snapshot_candidate(&state_snapshot, actor, target, pai, false, &candidate)
        .unwrap_or(false)
}

fn mark_no_yaku_furiten_for_discard_waiters(
    state: &mut GameStateCore,
    discarder: usize,
    pai: &str,
) {
    let mut actors_to_mark = Vec::new();
    for actor in 0..state.players.len() {
        if actor == discarder || state.players[actor].furiten {
            continue;
        }
        if !actor_waits_on_tile(state, actor, pai) {
            continue;
        }
        if !can_current_ron_from_state(state, actor, discarder, pai, false) {
            actors_to_mark.push(actor);
        }
    }
    for actor in actors_to_mark {
        mark_same_cycle_furiten(state, actor);
    }
}

fn mark_missed_discard_hora_if_waiting(
    state: &mut GameStateCore,
    actor: usize,
) -> Result<(), String> {
    let Some(last_discard) = state.last_discard.clone() else {
        return Ok(());
    };
    if last_discard.actor == actor || state.players[actor].furiten {
        return Ok(());
    }
    let pai = last_discard.pai_raw.as_str();
    if actor_waits_on_tile(state, actor, pai) {
        mark_same_cycle_furiten(state, actor);
    }
    Ok(())
}

fn mark_missed_chankan_if_waiting(state: &mut GameStateCore, actor: usize) -> Result<(), String> {
    let Some(last_kakan) = state.last_kakan.clone() else {
        return Ok(());
    };
    let target = last_kakan
        .get("actor")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| "invalid last_kakan.actor".to_string())?;
    if target == actor || state.players[actor].furiten {
        return Ok(());
    }
    let pai = last_kakan
        .get("pai_raw")
        .or_else(|| last_kakan.get("pai"))
        .and_then(Value::as_str)
        .ok_or_else(|| "invalid last_kakan.pai".to_string())?;
    if actor_waits_on_tile(state, actor, pai)
        && can_current_ron_from_state(state, actor, target, pai, true)
    {
        mark_same_cycle_furiten(state, actor);
    }
    Ok(())
}

fn mark_all_missed_responses(state: &mut GameStateCore) -> Result<(), String> {
    if let Some(last_discard) = state.last_discard.clone() {
        let discarder = last_discard.actor;
        for offset in 1..4 {
            mark_missed_discard_hora_if_waiting(state, (discarder + offset) % 4)?;
        }
        return Ok(());
    }
    if let Some(last_kakan) = state.last_kakan.clone() {
        let actor = last_kakan
            .get("actor")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| "invalid last_kakan.actor".to_string())?;
        for offset in 1..4 {
            mark_missed_chankan_if_waiting(state, (actor + offset) % 4)?;
        }
    }
    Ok(())
}

fn upgrade_pon_meld_to_kakan(player: &mut PlayerStateCore, pai: &str, added_tile: &str) {
    let pai_family = tile_family_key(pai);
    for meld in player.melds.iter_mut() {
        if meld.get("type").and_then(Value::as_str) != Some("pon") {
            continue;
        }
        let Some(meld_pai) = meld.get("pai").and_then(Value::as_str) else {
            continue;
        };
        if tile_family_key(meld_pai) != pai_family {
            continue;
        }

        let mut consumed = meld
            .get("consumed")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let called_tile = normalize_or_keep_aka(
            meld.get("pai_raw")
                .and_then(Value::as_str)
                .or_else(|| meld.get("pai").and_then(Value::as_str))
                .unwrap_or(pai),
        );
        consumed.push(Value::String(called_tile));
        consumed.push(Value::String(added_tile.to_string()));
        let target = meld.get("target").cloned().unwrap_or(Value::Null);
        *meld = serde_json::json!({
            "type": "kakan",
            "pai": pai,
            "pai_raw": pai,
            "consumed": consumed,
            "target": target,
        });
        return;
    }

    player.melds.push(serde_json::json!({
        "type": "kakan",
        "pai": pai,
        "pai_raw": pai,
        "consumed": [pai, pai, pai, pai],
        "target": Value::Null,
    }));
}

pub fn apply_event(state: &mut GameStateCore, event: &Value) -> Result<(), String> {
    let et = get_str(event, "type")?;
    match et {
        "start_game" => {
            state.in_game = true;
            Ok(())
        }
        "start_kyoku" => {
            state.bakaze = get_str(event, "bakaze")?.to_string();
            state.kyoku = get_i32(event, "kyoku")?;
            state.honba = get_i32(event, "honba")?;
            state.kyotaku = get_i32(event, "kyotaku")?;
            state.oya = get_usize(event, "oya")?;
            state.scores = event
                .get("scores")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing scores".to_string())?
                .iter()
                .map(|v| v.as_i64().unwrap_or(25000) as i32)
                .collect();
            state.dora_markers = vec![normalize_or_keep_aka(get_str(event, "dora_marker")?)];
            state.ura_dora_markers.clear();
            state.last_discard = None;
            state.last_kakan = None;
            state.actor_to_move = Some(state.oya);
            state.last_tsumo = vec![None, None, None, None];
            state.last_tsumo_raw = vec![None, None, None, None];
            state.remaining_wall = Some(70);
            state.pending_rinshan_actor = None;
            state.ryukyoku_tenpai_players.clear();
            state.players = vec![
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
                PlayerStateCore::default(),
            ];
            let tehais = event
                .get("tehais")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing tehais".to_string())?;
            for (pid, tehai) in tehais.iter().enumerate().take(4) {
                let tiles = tehai
                    .as_array()
                    .ok_or_else(|| "invalid tehai".to_string())?;
                for tile_value in tiles {
                    if let Some(tile) = tile_value.as_str() {
                        if tile != "?" {
                            let key = normalize_or_keep_aka(tile);
                            *state.players[pid].hand.entry(key).or_insert(0) += 1;
                        }
                    }
                }
            }
            Ok(())
        }
        "tsumo" => {
            let actor = get_usize(event, "actor")?;
            let pai = get_str(event, "pai")?;
            let is_rinshan = event
                .get("rinshan")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || state.pending_rinshan_actor == Some(actor);
            if pai != "?" {
                let key = normalize_or_keep_aka(pai);
                *state.players[actor].hand.entry(key.clone()).or_insert(0) += 1;
                state.last_tsumo[actor] = Some(key);
                state.last_tsumo_raw[actor] = Some(pai.to_string());
                if !is_rinshan {
                    if let Some(remaining_wall) = state.remaining_wall.as_mut() {
                        *remaining_wall = (*remaining_wall).max(1) - 1;
                    }
                }
            } else {
                state.last_tsumo[actor] = None;
                state.last_tsumo_raw[actor] = None;
            }
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.pending_rinshan_actor = None;
            state.players[actor].doujun_furiten = false;
            refresh_furiten(&mut state.players[actor]);
            state.players[actor].rinshan_tsumo = is_rinshan;
            Ok(())
        }
        "dahai" => {
            let actor = get_usize(event, "actor")?;
            let pai_raw = get_str(event, "pai")?;
            if pai_raw == "?" {
                return Err("unsupported dahai pai='?' in Rust state core slice".to_string());
            }
            let tsumogiri = event
                .get("tsumogiri")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let tile_key = normalize_or_keep_aka(pai_raw);
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                remove_tile(&mut state.players[actor].hand, &tile_key)?;
            }
            let reach_declared = state.players[actor].pending_reach;
            state.players[actor].discards.push(DiscardEntry {
                pai: tile_key.clone(),
                tsumogiri,
                reach_declared,
            });
            state.last_discard = Some(LastDiscard {
                actor,
                pai: tile_key,
                pai_raw: pai_raw.to_string(),
            });
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.actor_to_move = Some((actor + 1) % 4);
            state.players[actor].ippatsu_eligible = false;
            state.players[actor].rinshan_tsumo = false;
            if state.players[actor].pending_reach {
                state.players[actor].reached = true;
                state.players[actor].pending_reach = false;
            }
            recompute_sutehai_furiten_after_discard(state, actor);
            mark_no_yaku_furiten_for_discard_waiters(state, actor, pai_raw);
            Ok(())
        }
        "pon" | "chi" | "daiminkan" => {
            let actor = get_usize(event, "actor")?;
            mark_missed_discard_hora_if_waiting(state, actor)?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let consumed_values = event
                .get("consumed")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing consumed".to_string())?;
            let mut consumed = Vec::new();
            for value in consumed_values {
                let tile = value
                    .as_str()
                    .ok_or_else(|| "invalid consumed tile".to_string())?;
                let key = normalize_or_keep_aka(tile);
                if !event
                    .get("skip_hand_update")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    remove_tile(&mut state.players[actor].hand, &key)?;
                }
                consumed.push(Value::String(key));
            }
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let pai_raw =
                normalize_or_keep_aka(event.get("pai_raw").and_then(Value::as_str).unwrap_or(&pai));
            let target = event.get("target").cloned().unwrap_or(Value::Null);
            let meld = serde_json::json!({
                "type": et,
                "pai": pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": target,
            });
            state.players[actor].melds.push(meld);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.actor_to_move = Some(actor);
            if et == "daiminkan" {
                consume_wall_for_kan(state);
                state.pending_rinshan_actor = Some(actor);
            }
            Ok(())
        }
        "ankan" => {
            let actor = get_usize(event, "actor")?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .ok_or_else(|| "missing consumed for ankan".to_string())?
                .iter()
                .filter_map(Value::as_str)
                .map(normalize_or_keep_aka)
                .collect::<Vec<_>>();
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                for tile in &consumed {
                    remove_tile(&mut state.players[actor].hand, tile)?;
                }
            }
            let ankan_pai = event
                .get("pai")
                .and_then(Value::as_str)
                .map(normalize_or_keep_aka)
                .or_else(|| consumed.first().cloned())
                .ok_or_else(|| "missing pai/consumed for ankan".to_string())?;
            let pai_raw = normalize_or_keep_aka(
                event
                    .get("pai_raw")
                    .and_then(Value::as_str)
                    .unwrap_or(&ankan_pai),
            );
            state.players[actor].melds.push(serde_json::json!({
                "type": "ankan",
                "pai": ankan_pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": actor,
            }));
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            state.ryukyoku_tenpai_players.clear();
            consume_wall_for_kan(state);
            state.pending_rinshan_actor = Some(actor);
            Ok(())
        }
        "kakan" => {
            let actor = get_usize(event, "actor")?;
            for player in state.players.iter_mut() {
                player.ippatsu_eligible = false;
            }
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let pai_raw =
                normalize_or_keep_aka(event.get("pai_raw").and_then(Value::as_str).unwrap_or(&pai));
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .map(|tiles| {
                    tiles
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_or_keep_aka)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let target = event.get("target").cloned().unwrap_or(Value::Null);
            state.last_kakan = Some(serde_json::json!({
                "actor": actor,
                "pai": pai,
                "pai_raw": pai_raw,
                "consumed": consumed,
                "target": target,
            }));
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            let mut actors_to_mark = Vec::new();
            for opponent in 0..state.players.len() {
                if opponent == actor || state.players[opponent].furiten {
                    continue;
                }
                if actor_waits_on_tile(state, opponent, &pai)
                    && !can_current_ron_from_state(state, opponent, actor, &pai, true)
                {
                    actors_to_mark.push(opponent);
                }
            }
            for opponent in actors_to_mark {
                mark_same_cycle_furiten(state, opponent);
            }
            Ok(())
        }
        "kakan_accepted" => {
            let actor = get_usize(event, "actor")?;
            let pai = normalize_or_keep_aka(get_str(event, "pai")?);
            let consumed = event
                .get("consumed")
                .and_then(Value::as_array)
                .map(|tiles| {
                    tiles
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_or_keep_aka)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut added_tile = pai.clone();
            if state.players[actor]
                .hand
                .get(&added_tile)
                .copied()
                .unwrap_or(0)
                == 0
            {
                if let Some(tile) = consumed
                    .iter()
                    .find(|tile| state.players[actor].hand.get(*tile).copied().unwrap_or(0) > 0)
                {
                    added_tile = tile.clone();
                }
            }
            if !event
                .get("skip_hand_update")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                remove_tile(&mut state.players[actor].hand, &added_tile)?;
            }
            upgrade_pon_meld_to_kakan(&mut state.players[actor], &pai, &added_tile);
            state.actor_to_move = Some(actor);
            state.last_discard = None;
            state.last_kakan = None;
            state.last_tsumo[actor] = None;
            state.last_tsumo_raw[actor] = None;
            consume_wall_for_kan(state);
            state.pending_rinshan_actor = Some(actor);
            Ok(())
        }
        "reach" => {
            let actor = get_usize(event, "actor")?;
            state.players[actor].pending_reach = true;
            Ok(())
        }
        "reach_accepted" => {
            let actor = get_usize(event, "actor")?;
            state.players[actor].ippatsu_eligible = true;
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                state.scores = scores
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(25000) as i32)
                    .collect();
            }
            if let Some(kyotaku) = event.get("kyotaku").and_then(Value::as_i64) {
                state.kyotaku = kyotaku as i32;
            }
            Ok(())
        }
        "none" => {
            if let Some(actor) = event
                .get("actor")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
            {
                if state
                    .last_discard
                    .as_ref()
                    .is_some_and(|last_discard| last_discard.actor != actor)
                {
                    mark_missed_discard_hora_if_waiting(state, actor)?;
                    let discarder = state.last_discard.as_ref().map(|last| last.actor);
                    if let Some(discarder) = discarder {
                        let next_actor = (actor + 1) % 4;
                        if next_actor == discarder {
                            state.last_discard = None;
                            state.actor_to_move = Some((discarder + 1) % 4);
                        } else {
                            state.actor_to_move = Some(next_actor);
                        }
                    }
                } else if state
                    .last_kakan
                    .as_ref()
                    .and_then(|last_kakan| last_kakan.get("actor").and_then(Value::as_u64))
                    .is_some_and(|kan_actor| kan_actor as usize != actor)
                {
                    mark_missed_chankan_if_waiting(state, actor)?;
                    if let Some(kan_actor) = state
                        .last_kakan
                        .as_ref()
                        .and_then(|last_kakan| last_kakan.get("actor").and_then(Value::as_u64))
                    {
                        state.actor_to_move = Some(kan_actor as usize);
                    }
                }
                return Ok(());
            }

            mark_all_missed_responses(state)?;
            if let Some(last_discard) = state.last_discard.take() {
                state.actor_to_move = Some((last_discard.actor + 1) % 4);
            } else if let Some(last_kakan) = state.last_kakan.take() {
                if let Some(actor) = last_kakan.get("actor").and_then(Value::as_u64) {
                    state.actor_to_move = Some(actor as usize);
                }
            }
            Ok(())
        }
        "hora" | "ryukyoku" => {
            state.pending_rinshan_actor = None;
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                state.scores = scores
                    .iter()
                    .map(|v| v.as_i64().unwrap_or(25000) as i32)
                    .collect();
            }
            if let Some(honba) = event
                .get("state_honba")
                .or_else(|| event.get("honba"))
                .and_then(Value::as_i64)
            {
                state.honba = honba as i32;
            }
            if let Some(kyotaku) = event
                .get("state_kyotaku")
                .or_else(|| event.get("kyotaku"))
                .and_then(Value::as_i64)
            {
                state.kyotaku = kyotaku as i32;
            }
            if let Some(oya) = event.get("oya").and_then(Value::as_u64) {
                state.oya = oya as usize;
            }
            if let Some(ura_markers) = event.get("ura_dora_markers").and_then(Value::as_array) {
                state.ura_dora_markers = ura_markers
                    .iter()
                    .filter_map(Value::as_str)
                    .map(normalize_or_keep_aka)
                    .collect();
            }
            if let Some(tenpai_players) = event.get("tenpai_players").and_then(Value::as_array) {
                state.ryukyoku_tenpai_players = tenpai_players
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as usize)
                    .collect();
            } else if et == "hora" {
                state.ryukyoku_tenpai_players.clear();
            }
            state.actor_to_move = None;
            state.last_discard = None;
            state.last_kakan = None;
            Ok(())
        }
        "end_kyoku" | "end_game" => {
            state.actor_to_move = None;
            state.last_discard = None;
            state.last_kakan = None;
            state.pending_rinshan_actor = None;
            state.ryukyoku_tenpai_players.clear();
            Ok(())
        }
        "dora" => {
            let marker = normalize_or_keep_aka(get_str(event, "dora_marker")?);
            state.dora_markers.push(marker);
            Ok(())
        }
        _ => Err(format!(
            "unsupported event type in Rust state core slice: {}",
            et
        )),
    }
}

pub fn replay_state_snapshot_value(events: &[Value], actor: usize) -> Result<Value, String> {
    let mut state = GameStateCore::default();
    for event in events {
        apply_event(&mut state, event)?;
    }
    serde_json::to_value(crate::snapshot::snapshot_for_actor(&state, actor))
        .map_err(|err| err.to_string())
}

pub fn replay_state_snapshot(events: &[Value], actor: usize) -> Result<String, String> {
    let snapshot = replay_state_snapshot_value(events, actor)?;
    serde_json::to_string(&snapshot).map_err(|err| err.to_string())
}

pub fn validate_replay_state_snapshot(
    events: &[Value],
    actor: usize,
    expected_snapshot: Option<&Value>,
) -> Result<String, String> {
    let rust_snapshot = replay_state_snapshot_value(events, actor)?;
    let matches_expected = expected_snapshot.map(|expected| expected == &rust_snapshot);
    let payload = json!({
        "ok": matches_expected.unwrap_or(true),
        "actor": actor,
        "event_count": events.len(),
        "rust_snapshot": rust_snapshot,
        "expected_snapshot": expected_snapshot,
        "mismatch": matches_expected == Some(false),
    });
    serde_json::to_string(&payload).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::legal_actions::enumerate_public_legal_action_specs;

    use super::{replay_state_snapshot, replay_state_snapshot_value};

    #[test]
    fn basic_start_tsumo_dahai_snapshot_works() {
        let events = vec![
            json!({"type":"start_game","names":["p0","p1","p2","p3"]}),
            json!({
                "type":"start_kyoku",
                "bakaze":"E",
                "kyoku":1,
                "honba":0,
                "kyotaku":0,
                "oya":0,
                "scores":[25000,25000,25000,25000],
                "dora_marker":"1m",
                "tehais":[
                    ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
                    ["1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m"],
                    ["2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m"],
                    ["3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m"]
                ]
            }),
            json!({"type":"tsumo","actor":0,"pai":"5p"}),
            json!({"type":"dahai","actor":0,"pai":"5p","tsumogiri":true}),
        ];
        let snapshot = replay_state_snapshot(&events, 0).unwrap();
        assert!(snapshot.contains("\"actor\":0"));
        assert!(snapshot.contains("\"last_discard\""));
        assert!(snapshot.contains("\"pai\":\"5p\""));
    }

    fn ron_pass_fixture_events() -> Vec<serde_json::Value> {
        vec![
            json!({"type":"start_game","names":["p0","p1","p2","p3"]}),
            json!({
                "type":"start_kyoku",
                "bakaze":"E",
                "kyoku":1,
                "honba":0,
                "kyotaku":0,
                "oya":1,
                "scores":[25000,25000,25000,25000],
                "dora_marker":"1s",
                "tehais":[
                    ["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],
                    ["1m","2m","3m","4m","5m","6m","7m","8m","9m","P","1p","2p","3p"],
                    ["1m","2m","3m","4p","5p","6p","7s","8s","9s","P","P","5m","5m"],
                    ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]
                ]
            }),
            json!({"type":"tsumo","actor":1,"pai":"9s"}),
            json!({"type":"dahai","actor":1,"pai":"P","tsumogiri":false}),
        ]
    }

    #[test]
    fn actor_scoped_none_marks_only_that_actor_doujun_furiten() {
        let mut events = ron_pass_fixture_events();
        let before_pass = replay_state_snapshot_value(&events, 2).unwrap();
        let before_legal = enumerate_public_legal_action_specs(&before_pass, 2).unwrap();
        assert!(before_legal
            .iter()
            .any(|action| action.get("type").and_then(serde_json::Value::as_str) == Some("hora")));

        events.push(json!({"type":"none","actor":2}));
        let after_pass = replay_state_snapshot_value(&events, 2).unwrap();
        assert_eq!(after_pass["doujun_furiten"][2], true);
        assert_eq!(after_pass["furiten"][2], true);
        assert_eq!(after_pass["furiten"][3], false);

        let after_legal = enumerate_public_legal_action_specs(&after_pass, 2).unwrap();
        assert!(!after_legal
            .iter()
            .any(|action| action.get("type").and_then(serde_json::Value::as_str) == Some("hora")));
    }

    #[test]
    fn own_tsumo_clears_actor_scoped_doujun_furiten() {
        let mut events = ron_pass_fixture_events();
        events.push(json!({"type":"none","actor":2}));
        events.push(json!({"type":"none","actor":3}));
        events.push(json!({"type":"none","actor":0}));
        events.push(json!({"type":"none"}));
        events.push(json!({"type":"tsumo","actor":2,"pai":"1s"}));

        let snapshot = replay_state_snapshot_value(&events, 2).unwrap();
        assert_eq!(snapshot["doujun_furiten"][2], false);
        assert_eq!(snapshot["furiten"][2], false);
    }
}
