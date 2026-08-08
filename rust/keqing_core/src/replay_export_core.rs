use std::fs;

use serde_json::{json, Value};

use crate::event_apply::apply_event as apply_event_core;
use crate::legal_actions::enumerate_legal_action_specs_structural;
use crate::progress_summary::summarize_like_python;
use crate::snapshot::snapshot_for_actor;
use crate::state_core::GameStateCore;

pub const TILE_KIND_COUNT: usize = 34;
pub const EVENT_HISTORY_LEN: usize = 48;
pub const EVENT_HISTORY_FEATURE_DIM: usize = 5;
pub const HISTORY_SUMMARY_DIM: usize = 20;
pub const EVENT_MAX_TURN_IDX: i16 = 15;
pub const EVENT_TYPE_PAD: i16 = 0;
pub const EVENT_TYPE_TSUMO: i16 = 1;
pub const EVENT_TYPE_DAHAI: i16 = 2;
pub const EVENT_TYPE_PON: i16 = 3;
pub const EVENT_TYPE_CHI: i16 = 4;
pub const EVENT_TYPE_DAIMINKAN: i16 = 5;
pub const EVENT_TYPE_ANKAN: i16 = 6;
pub const EVENT_TYPE_KAKAN: i16 = 7;
pub const EVENT_TYPE_REACH: i16 = 8;
pub const EVENT_TYPE_DORA: i16 = 9;
pub const EVENT_TYPE_HORA: i16 = 10;
pub const EVENT_TYPE_RYUKYOKU: i16 = 11;
pub const EVENT_TYPE_UNKNOWN: i16 = 12;
pub const EVENT_NO_ACTOR: i16 = 4;
pub const EVENT_NO_TILE: i16 = -1;

#[derive(Debug, Clone, Default)]
pub struct DiscardInfo {
    pub tile34: i16,
    pub tsumogiri: bool,
    pub reach_declared: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MeldInfo {
    pub kind: String,
    pub pai: Option<String>,
    pub consumed: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PlayerState {
    pub discards: Vec<DiscardInfo>,
    pub melds: Vec<MeldInfo>,
    pub reached: bool,
    pub pending_reach: bool,
    pub furiten: bool,
}

#[derive(Debug, Clone)]
pub struct PlayerRoundTracker {
    pub hand_tiles: Vec<String>,
    pub meld_tiles: Vec<String>,
    pub hand_counts34: [u8; TILE_KIND_COUNT],
    pub meld_counts34: [u8; TILE_KIND_COUNT],
    pub visible_counts34: [u8; TILE_KIND_COUNT],
    pub discards_count: usize,
    pub meld_count: usize,
    pub pair_count: usize,
    pub ankoutsu_count: usize,
    pub suit_counts: [usize; 4],
    pub aka_counts: [usize; 4],
}

impl Default for PlayerRoundTracker {
    fn default() -> Self {
        Self {
            hand_tiles: Vec::new(),
            meld_tiles: Vec::new(),
            hand_counts34: [0; TILE_KIND_COUNT],
            meld_counts34: [0; TILE_KIND_COUNT],
            visible_counts34: [0; TILE_KIND_COUNT],
            discards_count: 0,
            meld_count: 0,
            pair_count: 0,
            ankoutsu_count: 0,
            suit_counts: [0; 4],
            aka_counts: [0; 4],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoundFeatureTracker {
    pub players: Vec<PlayerRoundTracker>,
}

impl Default for RoundFeatureTracker {
    fn default() -> Self {
        Self {
            players: vec![
                PlayerRoundTracker::default(),
                PlayerRoundTracker::default(),
                PlayerRoundTracker::default(),
                PlayerRoundTracker::default(),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoundState {
    pub bakaze: String,
    pub kyoku: i8,
    pub honba: i8,
    pub kyotaku: i8,
    pub oya: i8,
    pub game_start_oya: i8,
    pub scores: [i32; 4],
    pub dora_markers: Vec<String>,
    pub players: Vec<PlayerState>,
    pub feature_tracker: RoundFeatureTracker,
    pub last_tsumo: Vec<Option<String>>,
    pub remaining_wall: i32,
    pub pending_rinshan_actor: Option<usize>,
    pub in_game: bool,
    pub round_step_index: i32,
    pub round_terminal_finalized: bool,
}

impl Default for RoundState {
    fn default() -> Self {
        Self {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            game_start_oya: -1,
            scores: [25000, 25000, 25000, 25000],
            dora_markers: Vec::new(),
            players: vec![
                PlayerState::default(),
                PlayerState::default(),
                PlayerState::default(),
                PlayerState::default(),
            ],
            feature_tracker: RoundFeatureTracker::default(),
            last_tsumo: vec![None, None, None, None],
            remaining_wall: 70,
            pending_rinshan_actor: None,
            in_game: false,
            round_step_index: 0,
            round_terminal_finalized: false,
        }
    }
}

pub fn normalize_tile_repr(tile: &str) -> String {
    match tile {
        "0m" => "5mr".to_string(),
        "0p" => "5pr".to_string(),
        "0s" => "5sr".to_string(),
        _ => tile.to_string(),
    }
}

pub fn strip_aka(tile: &str) -> String {
    if tile == "5mr" || tile == "0m" {
        return "5m".to_string();
    }
    if tile == "5pr" || tile == "0p" {
        return "5p".to_string();
    }
    if tile == "5sr" || tile == "0s" {
        return "5s".to_string();
    }
    if tile.ends_with('r') {
        return tile.trim_end_matches('r').to_string();
    }
    tile.to_string()
}

pub fn tile34_from_pai(pai: &str) -> Option<i16> {
    let normalized = normalize_tile_repr(pai);
    let pai = normalized.as_str();
    let mut chars = pai.chars();
    let first = chars.next()?;
    if let Some(suit) = chars.next() {
        if matches!(suit, 'm' | 'p' | 's') {
            let rank = if first == '0' {
                5
            } else {
                first.to_digit(10)? as i16
            };
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return None,
            };
            return Some(base + rank - 1);
        }
    }
    match pai {
        "E" => Some(27),
        "S" => Some(28),
        "W" => Some(29),
        "N" => Some(30),
        "P" => Some(31),
        "F" => Some(32),
        "C" => Some(33),
        _ => None,
    }
}

pub fn tile_is_aka(tile: &str) -> bool {
    matches!(tile, "5mr" | "5pr" | "5sr" | "0m" | "0p" | "0s") || tile.ends_with('r')
}

fn suit_bucket_from_tile34(tile34: i16) -> usize {
    match tile34 {
        0..=8 => 0,
        9..=17 => 1,
        18..=26 => 2,
        _ => 3,
    }
}

fn apply_hand_count_delta(player: &mut PlayerRoundTracker, tile34: usize, delta: i32) {
    let before = player.hand_counts34[tile34] as i32;
    let after = (before + delta).clamp(0, 4);
    player.pair_count = player.pair_count + usize::from(after >= 2) - usize::from(before >= 2);
    player.ankoutsu_count =
        player.ankoutsu_count + usize::from(after >= 3) - usize::from(before >= 3);
    player.hand_counts34[tile34] = after as u8;
}

fn apply_overall_tile_delta(player: &mut PlayerRoundTracker, tile: &str, delta: i32) {
    let Some(tile34) = tile34_from_pai(tile) else {
        return;
    };
    let suit = suit_bucket_from_tile34(tile34);
    if delta >= 0 {
        player.suit_counts[suit] = player.suit_counts[suit].saturating_add(delta as usize);
    } else {
        player.suit_counts[suit] = player.suit_counts[suit].saturating_sub((-delta) as usize);
    }
    if tile_is_aka(tile) {
        if delta >= 0 {
            player.aka_counts[0] = player.aka_counts[0].saturating_add(delta as usize);
        } else {
            player.aka_counts[0] = player.aka_counts[0].saturating_sub((-delta) as usize);
        }
        match strip_aka(tile).as_str() {
            "5m" => {
                if delta >= 0 {
                    player.aka_counts[1] = player.aka_counts[1].saturating_add(delta as usize);
                } else {
                    player.aka_counts[1] = player.aka_counts[1].saturating_sub((-delta) as usize);
                }
            }
            "5p" => {
                if delta >= 0 {
                    player.aka_counts[2] = player.aka_counts[2].saturating_add(delta as usize);
                } else {
                    player.aka_counts[2] = player.aka_counts[2].saturating_sub((-delta) as usize);
                }
            }
            "5s" => {
                if delta >= 0 {
                    player.aka_counts[3] = player.aka_counts[3].saturating_add(delta as usize);
                } else {
                    player.aka_counts[3] = player.aka_counts[3].saturating_sub((-delta) as usize);
                }
            }
            _ => {}
        }
    }
}

fn remove_first_tile_exact(tiles: &mut Vec<String>, tile: &str) -> bool {
    if let Some(pos) = tiles.iter().position(|value| value == tile) {
        tiles.remove(pos);
        return true;
    }
    let stripped = strip_aka(tile);
    if let Some(pos) = tiles.iter().position(|value| strip_aka(value) == stripped) {
        tiles.remove(pos);
        return true;
    }
    false
}

impl RoundFeatureTracker {
    pub fn from_start_kyoku(tehais: &[Vec<String>], dora_markers: &[String]) -> Self {
        let mut tracker = Self::default();
        for actor in 0..4 {
            let player = &mut tracker.players[actor];
            player.hand_tiles = tehais.get(actor).cloned().unwrap_or_default();
            for tile in &player.hand_tiles.clone() {
                if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                    apply_hand_count_delta(player, idx, 1);
                    player.visible_counts34[idx] = player.visible_counts34[idx].saturating_add(1);
                    apply_overall_tile_delta(player, tile, 1);
                }
            }
        }
        for marker in dora_markers {
            if let Some(idx) = tile34_from_pai(marker).map(|value| value as usize) {
                for player in &mut tracker.players {
                    player.visible_counts34[idx] = player.visible_counts34[idx].saturating_add(1);
                }
            }
        }
        tracker
    }

    pub fn on_tsumo(&mut self, actor: usize, pai: &str) {
        let Some(idx) = tile34_from_pai(pai).map(|value| value as usize) else {
            return;
        };
        let player = &mut self.players[actor];
        player.hand_tiles.push(pai.to_string());
        apply_hand_count_delta(player, idx, 1);
        apply_overall_tile_delta(player, pai, 1);
        player.visible_counts34[idx] = player.visible_counts34[idx].saturating_add(1);
    }

    pub fn on_dahai(&mut self, actor: usize, pai: &str) {
        let Some(idx) = tile34_from_pai(pai).map(|value| value as usize) else {
            return;
        };
        let player = &mut self.players[actor];
        remove_first_tile_exact(&mut player.hand_tiles, pai);
        apply_hand_count_delta(player, idx, -1);
        apply_overall_tile_delta(player, pai, -1);
        player.discards_count += 1;
        for p in &mut self.players {
            p.visible_counts34[idx] = p.visible_counts34[idx].saturating_add(1);
        }
    }

    pub fn on_open_meld(&mut self, actor: usize, consumed: &[String], pai: Option<&str>) {
        let player = &mut self.players[actor];
        for tile in consumed {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                apply_hand_count_delta(player, idx, -1);
                player.meld_counts34[idx] = player.meld_counts34[idx].saturating_add(1);
            }
            remove_first_tile_exact(&mut player.hand_tiles, tile);
        }
        player.meld_count += 1;

        let mut meld_tiles = consumed.to_vec();
        if let Some(pai) = pai {
            meld_tiles.push(pai.to_string());
            if let Some(idx) = tile34_from_pai(pai).map(|value| value as usize) {
                player.meld_counts34[idx] = player.meld_counts34[idx].saturating_add(1);
            }
            apply_overall_tile_delta(player, pai, 1);
        }
        player.meld_tiles.extend(meld_tiles.iter().cloned());
        for tile in &meld_tiles {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                for p in &mut self.players {
                    p.visible_counts34[idx] = p.visible_counts34[idx].saturating_add(1);
                }
            }
        }
    }

    pub fn on_ankan(&mut self, actor: usize, consumed: &[String], pai: Option<&str>) {
        let player = &mut self.players[actor];
        for tile in consumed {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                apply_hand_count_delta(player, idx, -1);
                player.meld_counts34[idx] = player.meld_counts34[idx].saturating_add(1);
            }
            remove_first_tile_exact(&mut player.hand_tiles, tile);
        }
        player.meld_count += 1;

        let mut meld_tiles = consumed.to_vec();
        if let Some(pai) = pai {
            meld_tiles.push(pai.to_string());
            if let Some(idx) = tile34_from_pai(pai).map(|value| value as usize) {
                player.meld_counts34[idx] = player.meld_counts34[idx].saturating_add(1);
            }
        }
        player.meld_tiles.extend(meld_tiles.iter().cloned());
        for tile in &meld_tiles {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                for p in &mut self.players {
                    p.visible_counts34[idx] = p.visible_counts34[idx].saturating_add(1);
                }
            }
        }
    }

    pub fn on_kakan_accepted(&mut self, actor: usize, added_tile: &str, pai: Option<&str>) {
        let player = &mut self.players[actor];
        if let Some(idx) = tile34_from_pai(added_tile).map(|value| value as usize) {
            apply_hand_count_delta(player, idx, -1);
            player.meld_counts34[idx] = player.meld_counts34[idx].saturating_add(1);
        }
        remove_first_tile_exact(&mut player.hand_tiles, added_tile);
        player.meld_tiles.push(added_tile.to_string());
        for tile in [Some(added_tile), pai].into_iter().flatten() {
            if let Some(idx) = tile34_from_pai(tile).map(|value| value as usize) {
                for p in &mut self.players {
                    p.visible_counts34[idx] = p.visible_counts34[idx].saturating_add(1);
                }
            }
        }
    }

    pub fn on_dora(&mut self, marker: &str) {
        if let Some(idx) = tile34_from_pai(marker).map(|value| value as usize) {
            for player in &mut self.players {
                player.visible_counts34[idx] = player.visible_counts34[idx].saturating_add(1);
            }
        }
    }
}

pub fn start_new_round(state: &mut RoundState, event: &Value) {
    state.bakaze = value_string(event, "bakaze").unwrap_or_else(|| "E".to_string());
    state.kyoku = value_i8(event, "kyoku", 1);
    state.honba = value_i8(event, "honba", 0);
    state.kyotaku = value_i8(event, "kyotaku", 0);
    state.oya = value_i8(event, "oya", 0);
    if state.game_start_oya < 0 {
        state.game_start_oya = state.oya;
    }
    if let Some(scores) = event.get("scores").and_then(Value::as_array) {
        for (idx, score) in scores.iter().take(4).enumerate() {
            state.scores[idx] = score.as_i64().unwrap_or(25000) as i32;
        }
    }
    state.dora_markers = value_string(event, "dora_marker").into_iter().collect();
    state.players = vec![
        PlayerState::default(),
        PlayerState::default(),
        PlayerState::default(),
        PlayerState::default(),
    ];
    let tehais: Vec<Vec<String>> = event
        .get("tehais")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|tehai| {
                    tehai
                        .as_array()
                        .map(|tiles| {
                            tiles
                                .iter()
                                .filter_map(Value::as_str)
                                .filter(|tile| *tile != "?")
                                .map(normalize_tile_repr)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
    state.feature_tracker = RoundFeatureTracker::from_start_kyoku(&tehais, &state.dora_markers);
    state.last_tsumo = vec![None, None, None, None];
    state.remaining_wall = 70;
    state.pending_rinshan_actor = None;
    state.in_game = true;
    state.round_step_index = 0;
    state.round_terminal_finalized = false;
}

pub fn apply_event_round_state(state: &mut RoundState, event: &Value) {
    let et = event.get("type").and_then(Value::as_str).unwrap_or("");
    match et {
        "start_game" => {
            state.in_game = true;
            state.game_start_oya = -1;
        }
        "start_kyoku" => start_new_round(state, event),
        "tsumo" => {
            let Some(actor) = value_usize(event, "actor") else {
                return;
            };
            let is_rinshan = event
                .get("rinshan")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || state.pending_rinshan_actor == Some(actor);
            if let Some(pai) = value_string(event, "pai") {
                state.feature_tracker.on_tsumo(actor, &pai);
                state.last_tsumo[actor] = Some(pai);
                if !is_rinshan {
                    state.remaining_wall = (state.remaining_wall - 1).max(0);
                }
            } else {
                state.last_tsumo[actor] = None;
            }
            state.pending_rinshan_actor = None;
        }
        "dahai" => {
            let Some(actor) = value_usize(event, "actor") else {
                return;
            };
            let tsumogiri = event
                .get("tsumogiri")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let pai = if let Some(pai) = value_string(event, "pai") {
                if pai == "?" {
                    if tsumogiri {
                        state.last_tsumo[actor]
                            .clone()
                            .unwrap_or_else(|| "?".to_string())
                    } else {
                        "?".to_string()
                    }
                } else {
                    pai
                }
            } else if tsumogiri {
                state.last_tsumo[actor]
                    .clone()
                    .unwrap_or_else(|| "?".to_string())
            } else {
                "?".to_string()
            };
            if pai != "?" {
                state.feature_tracker.on_dahai(actor, &pai);
            }
            let reach_declared = state.players[actor].pending_reach;
            if let Some(tile34) = tile34_from_pai(&pai) {
                state.players[actor].discards.push(DiscardInfo {
                    tile34,
                    tsumogiri,
                    reach_declared,
                });
            }
            state.last_tsumo[actor] = None;
            if state.players[actor].pending_reach {
                state.players[actor].reached = true;
                state.players[actor].pending_reach = false;
            }
        }
        "chi" | "pon" | "daiminkan" => {
            let Some(actor) = value_usize(event, "actor") else {
                return;
            };
            let consumed = value_tile_list(event, "consumed");
            let pai = value_string(event, "pai");
            state
                .feature_tracker
                .on_open_meld(actor, &consumed, pai.as_deref());
            state.players[actor].melds.push(MeldInfo {
                kind: et.to_string(),
                pai: pai.clone(),
                consumed,
            });
            if et == "daiminkan" {
                state.remaining_wall = (state.remaining_wall - 1).max(0);
                state.pending_rinshan_actor = Some(actor);
            }
            state.last_tsumo[actor] = None;
        }
        "ankan" => {
            let Some(actor) = value_usize(event, "actor") else {
                return;
            };
            let consumed = value_tile_list(event, "consumed");
            let pai = value_string(event, "pai").or_else(|| consumed.first().cloned());
            state
                .feature_tracker
                .on_ankan(actor, &consumed, pai.as_deref());
            state.players[actor].melds.push(MeldInfo {
                kind: "ankan".to_string(),
                pai: pai.clone(),
                consumed,
            });
            state.last_tsumo[actor] = None;
            state.remaining_wall = (state.remaining_wall - 1).max(0);
            state.pending_rinshan_actor = Some(actor);
        }
        "kakan" => {
            if let Some(actor) = value_usize(event, "actor") {
                state.last_tsumo[actor] = None;
            }
        }
        "kakan_accepted" => {
            let Some(actor) = value_usize(event, "actor") else {
                return;
            };
            let pai = value_string(event, "pai");
            let consumed = value_tile_list(event, "consumed");
            let mut added_tile = pai.clone().unwrap_or_default();
            if added_tile.is_empty() {
                added_tile = consumed.first().cloned().unwrap_or_default();
            }
            state
                .feature_tracker
                .on_kakan_accepted(actor, &added_tile, pai.as_deref());
            if let Some(existing) = state.players[actor].melds.iter_mut().find(|meld| {
                meld.kind == "pon"
                    && meld.pai.as_ref().and_then(|tile| tile34_from_pai(tile))
                        == pai.as_ref().and_then(|tile| tile34_from_pai(tile))
            }) {
                existing.kind = "kakan".to_string();
                existing.consumed.push(added_tile.clone());
                if let Some(pai) = &pai {
                    existing.consumed.push(pai.clone());
                    existing.pai = Some(pai.clone());
                }
            } else {
                state.players[actor].melds.push(MeldInfo {
                    kind: "kakan".to_string(),
                    pai: pai.clone(),
                    consumed,
                });
            }
            state.last_tsumo[actor] = None;
            state.remaining_wall = (state.remaining_wall - 1).max(0);
            state.pending_rinshan_actor = Some(actor);
        }
        "reach" => {
            if let Some(actor) = value_usize(event, "actor") {
                state.players[actor].pending_reach = true;
            }
        }
        "reach_accepted" => {
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                for (idx, score) in scores.iter().take(4).enumerate() {
                    state.scores[idx] = score.as_i64().unwrap_or(state.scores[idx] as i64) as i32;
                }
            }
            if let Some(kyotaku) = event.get("kyotaku").and_then(Value::as_i64) {
                state.kyotaku = kyotaku as i8;
            }
        }
        "hora" | "ryukyoku" => {
            if let Some(scores) = event.get("scores").and_then(Value::as_array) {
                for (idx, score) in scores.iter().take(4).enumerate() {
                    state.scores[idx] = score.as_i64().unwrap_or(state.scores[idx] as i64) as i32;
                }
            }
            if let Some(honba) = event.get("honba").and_then(Value::as_i64) {
                state.honba = honba as i8;
            }
            if let Some(kyotaku) = event.get("kyotaku").and_then(Value::as_i64) {
                state.kyotaku = kyotaku as i8;
            }
            state.pending_rinshan_actor = None;
        }
        "end_kyoku" | "end_game" => {
            state.pending_rinshan_actor = None;
        }
        "dora" | "new_dora" => {
            if let Some(marker) = value_string(event, "dora_marker") {
                state.dora_markers.push(marker.clone());
                state.feature_tracker.on_dora(&marker);
            }
        }
        _ => {}
    }
}

pub fn action_event_should_advance_step(state: &RoundState, event: &Value) -> bool {
    let et = event.get("type").and_then(Value::as_str).unwrap_or("");
    let Some(actor) = value_usize(event, "actor") else {
        return false;
    };
    matches!(
        et,
        "dahai" | "chi" | "pon" | "daiminkan" | "ankan" | "kakan" | "reach" | "hora" | "ryukyoku"
    ) && state.in_game
        && !(et == "dahai" && state.players[actor].reached)
        && !state.feature_tracker.players[actor].hand_tiles.is_empty()
}

pub fn resolve_discard_tile(state: &RoundState, event: &Value) -> Option<String> {
    let actor = value_usize(event, "actor")?;
    let tsumogiri = event
        .get("tsumogiri")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let chosen_tile = if let Some(pai) = value_string(event, "pai") {
        if pai == "?" {
            if tsumogiri {
                state.last_tsumo[actor]
                    .clone()
                    .unwrap_or_else(|| "?".to_string())
            } else {
                "?".to_string()
            }
        } else {
            pai
        }
    } else if tsumogiri {
        state.last_tsumo[actor]
            .clone()
            .unwrap_or_else(|| "?".to_string())
    } else {
        "?".to_string()
    };
    Some(chosen_tile)
}

pub fn resolve_label_action(state: &RoundState, event: &Value) -> Option<Value> {
    let et = event.get("type").and_then(Value::as_str).unwrap_or("");
    if et != "dahai" {
        return Some(event.clone());
    }
    let actor = value_usize(event, "actor")?;
    let tsumogiri = event
        .get("tsumogiri")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let chosen_tile = resolve_discard_tile(state, event)?;
    if chosen_tile == "?" {
        return None;
    }
    Some(json!({
        "type": "dahai",
        "actor": actor,
        "pai": chosen_tile,
        "tsumogiri": tsumogiri,
    }))
}

pub fn action_none() -> Value {
    json!({"type": "none"})
}

pub fn is_structural_actor_decision_event(et: &str) -> bool {
    matches!(
        et,
        "dahai" | "chi" | "pon" | "daiminkan" | "ankan" | "kakan" | "reach" | "hora" | "ryukyoku"
    )
}

pub fn is_special_sampling_event(et: &str) -> bool {
    matches!(
        et,
        "reach" | "chi" | "pon" | "daiminkan" | "ankan" | "kakan" | "hora"
    )
}

pub fn compute_opp_tenpai_target(
    round_state: &RoundState,
    actor: usize,
    visible_counts: &[u8; TILE_KIND_COUNT],
) -> [f32; 3] {
    let mut result = [0.0f32; 3];
    let num_players = round_state.feature_tracker.players.len();
    if num_players < 4 {
        return result;
    }
    for rel in 0..3 {
        let opp_idx = (actor + 1 + rel) % 4;
        let opp_counts = round_state.feature_tracker.players[opp_idx].hand_counts34;
        let tile_sum: u8 = opp_counts.iter().sum();
        if tile_sum == 0 {
            continue;
        }
        let summary = summarize_like_python(&opp_counts, visible_counts);
        if summary.shanten <= 0 {
            result[rel] = 1.0;
        }
    }
    result
}

pub fn event_type_id_from_str(et: &str) -> i16 {
    match et {
        "tsumo" => EVENT_TYPE_TSUMO,
        "dahai" => EVENT_TYPE_DAHAI,
        "pon" => EVENT_TYPE_PON,
        "chi" => EVENT_TYPE_CHI,
        "daiminkan" => EVENT_TYPE_DAIMINKAN,
        "ankan" => EVENT_TYPE_ANKAN,
        "kakan" => EVENT_TYPE_KAKAN,
        "reach" => EVENT_TYPE_REACH,
        "dora" => EVENT_TYPE_DORA,
        "hora" => EVENT_TYPE_HORA,
        "ryukyoku" => EVENT_TYPE_RYUKYOKU,
        _ => EVENT_TYPE_UNKNOWN,
    }
}

pub fn event_tile_id_from_event(event: &Value, et: &str) -> i16 {
    match et {
        "tsumo" | "dahai" | "pon" | "chi" | "daiminkan" | "ankan" | "kakan" | "dora" => event
            .get("pai")
            .and_then(Value::as_str)
            .and_then(tile34_from_pai)
            .unwrap_or(EVENT_NO_TILE),
        _ => EVENT_NO_TILE,
    }
}

pub fn compute_event_history(
    all_events: &[Value],
    event_index: i32,
) -> [[i16; EVENT_HISTORY_FEATURE_DIM]; EVENT_HISTORY_LEN] {
    let mut out = [[0i16; EVENT_HISTORY_FEATURE_DIM]; EVENT_HISTORY_LEN];
    for token in &mut out {
        token[0] = EVENT_NO_ACTOR;
        token[1] = EVENT_TYPE_PAD;
        token[2] = EVENT_NO_TILE;
        token[3] = 0;
        token[4] = 0;
    }
    if event_index <= 0 {
        return out;
    }
    let end = (event_index as usize).min(all_events.len());
    if end == 0 {
        return out;
    }
    let mut kyoku_start: usize = 0;
    for i in (0..end).rev() {
        let et = all_events[i]
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if et == "start_kyoku" {
            kyoku_start = i + 1;
            break;
        }
    }
    if kyoku_start >= end {
        return out;
    }
    let slice_start = end.saturating_sub(EVENT_HISTORY_LEN).max(kyoku_start);
    let mut dahai_count_so_far: i32 = 0;
    for item in all_events.iter().take(slice_start).skip(kyoku_start) {
        let et = item.get("type").and_then(Value::as_str).unwrap_or("");
        if et == "dahai" {
            dahai_count_so_far += 1;
        }
    }
    let token_count = end - slice_start;
    let pad_len = EVENT_HISTORY_LEN - token_count;
    for (offset, idx) in (slice_start..end).enumerate() {
        let event = &all_events[idx];
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        let actor_raw = event.get("actor").and_then(Value::as_i64);
        let actor = match actor_raw {
            Some(a) if (0..=3).contains(&a) => a as i16,
            _ => EVENT_NO_ACTOR,
        };
        let event_type = event_type_id_from_str(et);
        let tile_id = event_tile_id_from_event(event, et);
        let turn_idx = ((dahai_count_so_far / 4).max(0) as i16).min(EVENT_MAX_TURN_IDX);
        let is_tedashi = if et == "dahai" {
            let tsumogiri = event
                .get("tsumogiri")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if tsumogiri {
                0
            } else {
                1
            }
        } else {
            0
        };
        let slot = pad_len + offset;
        out[slot][0] = actor;
        out[slot][1] = event_type;
        out[slot][2] = tile_id;
        out[slot][3] = turn_idx;
        out[slot][4] = is_tedashi;
        if et == "dahai" {
            dahai_count_so_far += 1;
        }
    }
    out
}

fn meaningful_event_kind(event_type: &str) -> Option<&'static str> {
    match event_type {
        "dahai" => Some("discard"),
        "chi" | "pon" => Some("call"),
        "daiminkan" | "ankan" | "kakan_accepted" => Some("kan"),
        "reach" => Some("riichi"),
        _ => None,
    }
}

fn recency_norm(distance: Option<usize>) -> f32 {
    let Some(distance) = distance else {
        return 0.0;
    };
    if distance == 0 {
        return 0.0;
    }
    (1.0 - (distance.min(16) as f32 / 16.0)).max(0.0)
}

pub fn compute_history_summary(
    all_events: &[Value],
    event_index: i32,
    actor: usize,
) -> [f32; HISTORY_SUMMARY_DIM] {
    let mut out = [0.0f32; HISTORY_SUMMARY_DIM];
    if event_index <= 0 {
        return out;
    }
    let end = (event_index as usize).min(all_events.len());
    if end == 0 {
        return out;
    }
    let mut kyoku_start: usize = 0;
    for i in (0..end).rev() {
        let et = all_events[i]
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if et == "start_kyoku" {
            kyoku_start = i + 1;
            break;
        }
    }
    if kyoku_start >= end {
        return out;
    }

    let mut meaningful_count = 0usize;
    let mut discard_count = 0usize;
    let mut call_count = 0usize;
    let mut kan_count = 0usize;
    let mut riichi_count = 0usize;
    let mut last_seen = [[None; 4]; 4];

    for event in all_events.iter().take(end).skip(kyoku_start) {
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        let Some(kind) = meaningful_event_kind(et) else {
            continue;
        };
        meaningful_count += 1;
        match kind {
            "discard" => discard_count += 1,
            "call" => call_count += 1,
            "kan" => kan_count += 1,
            "riichi" => riichi_count += 1,
            _ => {}
        }
        let who = event
            .get("actor")
            .and_then(Value::as_i64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value < 4);
        let Some(who) = who else {
            continue;
        };
        let kind_idx = match kind {
            "discard" => 0,
            "call" => 1,
            "kan" => 2,
            "riichi" => 3,
            _ => continue,
        };
        last_seen[who][kind_idx] = Some(meaningful_count);
    }

    if meaningful_count == 0 {
        return out;
    }

    out[0] = (discard_count as f32 / 60.0).min(1.0);
    out[1] = (call_count as f32 / 16.0).min(1.0);
    out[2] = (kan_count as f32 / 8.0).min(1.0);
    out[3] = (riichi_count as f32 / 4.0).min(1.0);

    let mut offset = 4usize;
    for rel in 0..4usize {
        let who = (actor + rel) % 4;
        for kind_idx in 0..4usize {
            let distance = last_seen[who][kind_idx].map(|pos| meaningful_count - pos + 1);
            out[offset] = recency_norm(distance);
            offset += 1;
        }
    }
    out
}

pub fn is_discard_followed_by_tsumo(et: &str, next_ev: Option<&Value>) -> bool {
    et == "dahai"
        && next_ev
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            == Some("tsumo")
}

#[derive(Debug, Clone)]
pub struct ReactionOpportunity {
    pub actor: usize,
    pub legal_actions: Vec<Value>,
}

pub fn enumerate_reaction_none_opportunities(
    core_state: &GameStateCore,
    discarder: usize,
) -> Result<Vec<ReactionOpportunity>, String> {
    let mut out = Vec::new();
    for actor in 0..4 {
        if actor == discarder {
            continue;
        }
        let snapshot = snapshot_for_actor(core_state, actor);
        if snapshot.hand.is_empty() {
            continue;
        }
        let snapshot_value = serde_json::to_value(&snapshot).map_err(|err| {
            format!("failed to serialize reaction snapshot for actor {actor}: {err}")
        })?;
        let legal_actions = enumerate_legal_action_specs_structural(&snapshot_value, actor)?;
        let has_non_none = legal_actions
            .iter()
            .any(|action| action.get("type").and_then(Value::as_str) != Some("none"));
        if !has_non_none {
            continue;
        }
        out.push(ReactionOpportunity {
            actor,
            legal_actions,
        });
    }
    Ok(out)
}

pub fn enumerate_actor_legal_actions(
    core_state: &GameStateCore,
    actor: usize,
) -> Result<Vec<Value>, String> {
    let snapshot = snapshot_for_actor(core_state, actor);
    let snapshot_value = serde_json::to_value(&snapshot)
        .map_err(|err| format!("failed to serialize snapshot for actor {actor}: {err}"))?;
    enumerate_legal_action_specs_structural(&snapshot_value, actor)
}

pub fn push_record_with_pending<R>(
    records: &mut Vec<R>,
    pending_records: &mut Vec<PendingRoundRecord>,
    round_step_index: i32,
    actor: usize,
    record: R,
) {
    let record_index = records.len();
    records.push(record);
    push_pending_record(pending_records, record_index, round_step_index, actor);
}

pub fn collect_actor_legal_record<R, F>(
    core_state: &GameStateCore,
    records: &mut Vec<R>,
    pending_records: &mut Vec<PendingRoundRecord>,
    round_step_index: i32,
    actor: usize,
    build_record: F,
) -> Result<(), String>
where
    F: FnOnce(&[Value]) -> Result<Option<R>, String>,
{
    let legal_actions = enumerate_actor_legal_actions(core_state, actor)?;
    if legal_actions.is_empty() {
        return Ok(());
    }
    if let Some(record) = build_record(&legal_actions)? {
        push_record_with_pending(records, pending_records, round_step_index, actor, record);
    }
    Ok(())
}

pub fn collect_reaction_none_records<R, F>(
    core_state: &GameStateCore,
    et: &str,
    next_ev: Option<&Value>,
    actor: Option<usize>,
    records: &mut Vec<R>,
    pending_records: &mut Vec<PendingRoundRecord>,
    round_step_index: i32,
    mut build_record: F,
) -> Result<(), String>
where
    F: FnMut(usize, &[Value]) -> Result<Option<R>, String>,
{
    if !is_discard_followed_by_tsumo(et, next_ev) {
        return Ok(());
    }
    let discarder = actor.unwrap_or(usize::MAX);
    for opportunity in enumerate_reaction_none_opportunities(core_state, discarder)? {
        if let Some(record) = build_record(opportunity.actor, &opportunity.legal_actions)? {
            push_record_with_pending(
                records,
                pending_records,
                round_step_index,
                opportunity.actor,
                record,
            );
        }
    }
    Ok(())
}

pub fn begin_round_event(
    state: &mut RoundState,
    pending_records: &mut Vec<PendingRoundRecord>,
    et: &str,
) {
    if et == "start_kyoku" {
        pending_records.clear();
        state.round_step_index = 0;
        state.round_terminal_finalized = false;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PendingRoundRecord {
    pub record_index: usize,
    pub round_step_index: i32,
    pub actor: i8,
}

pub fn push_pending_record(
    pending_records: &mut Vec<PendingRoundRecord>,
    record_index: usize,
    round_step_index: i32,
    actor: usize,
) {
    pending_records.push(PendingRoundRecord {
        record_index,
        round_step_index,
        actor: actor as i8,
    });
}

#[derive(Debug, Clone, Copy)]
pub struct RoundTargetUpdate {
    pub record_index: usize,
    pub score_delta_target: f32,
    pub global_value_target: f32,
    pub win_target: f32,
    pub dealin_target: f32,
    pub pts_given_win_target: f32,
    pub pts_given_dealin_target: f32,
    pub ryukyoku_tenpai_target: f32,
}

pub fn finalize_round_targets(
    state: &mut RoundState,
    pending_records: &[PendingRoundRecord],
    terminal_event: Option<&Value>,
    score_norm: f32,
    mc_return_gamma: f32,
) -> Vec<RoundTargetUpdate> {
    if pending_records.is_empty() {
        return Vec::new();
    }
    let mut score_deltas: Option<[f32; 4]> = None;
    let mut hora_actor: Option<i8> = None;
    let mut hora_target: Option<i8> = None;
    let mut ryukyoku_tenpai_players = [false; 4];
    if let Some(event) = terminal_event {
        if let Some(deltas) = event
            .get("deltas")
            .or_else(|| event.get("score_delta"))
            .and_then(Value::as_array)
        {
            if deltas.len() >= 4 {
                let mut out = [0.0f32; 4];
                for (idx, delta) in deltas.iter().take(4).enumerate() {
                    out[idx] = delta.as_f64().unwrap_or(0.0) as f32 / score_norm;
                }
                score_deltas = Some(out);
            }
        }
        if event.get("type").and_then(Value::as_str) == Some("hora") {
            hora_actor = event
                .get("actor")
                .and_then(Value::as_i64)
                .map(|value| value as i8);
            hora_target = event
                .get("target")
                .and_then(Value::as_i64)
                .map(|value| value as i8);
        } else if event.get("type").and_then(Value::as_str) == Some("ryukyoku") {
            if let Some(players) = event.get("tenpai_players").and_then(Value::as_array) {
                for player in players {
                    if let Some(idx) = player.as_u64().map(|value| value as usize) {
                        if idx < 4 {
                            ryukyoku_tenpai_players[idx] = true;
                        }
                    }
                }
            }
        }
    }

    let last_step = state.round_step_index.saturating_sub(1);
    let mut updates = Vec::with_capacity(pending_records.len());
    for pending in pending_records {
        let actor = pending.actor.clamp(0, 3) as usize;
        let (score_delta_target, global_value_target) = if let Some(score_deltas) = score_deltas {
            let steps_remaining = (last_step - pending.round_step_index).max(0) as f32;
            let score_delta_target = score_deltas[actor];
            (
                score_delta_target,
                score_delta_target * mc_return_gamma.powf(steps_remaining),
            )
        } else {
            (0.0, 0.0)
        };
        let actor_i8 = pending.actor;
        let win_target = f32::from(hora_actor == Some(actor_i8));
        let dealin_target =
            f32::from(hora_target == Some(actor_i8) && hora_actor != Some(actor_i8));
        updates.push(RoundTargetUpdate {
            record_index: pending.record_index,
            score_delta_target,
            global_value_target,
            win_target,
            dealin_target,
            pts_given_win_target: if win_target > 0.0 {
                score_delta_target.max(0.0)
            } else {
                0.0
            },
            pts_given_dealin_target: if dealin_target > 0.0 {
                (-score_delta_target).max(0.0)
            } else {
                0.0
            },
            ryukyoku_tenpai_target: if ryukyoku_tenpai_players[actor] {
                1.0
            } else {
                0.0
            },
        });
    }
    state.round_terminal_finalized = true;
    updates
}

pub fn maybe_finalize_round_targets(
    state: &mut RoundState,
    pending_records: &[PendingRoundRecord],
    et: &str,
    event: &Value,
    score_norm: f32,
    mc_return_gamma: f32,
) -> Option<Vec<RoundTargetUpdate>> {
    if matches!(et, "hora" | "ryukyoku") {
        return Some(finalize_round_targets(
            state,
            pending_records,
            Some(event),
            score_norm,
            mc_return_gamma,
        ));
    }
    if et == "end_kyoku" && !state.round_terminal_finalized {
        return Some(finalize_round_targets(
            state,
            pending_records,
            None,
            score_norm,
            mc_return_gamma,
        ));
    }
    None
}

pub fn reset_after_end_kyoku(
    state: &mut RoundState,
    pending_records: &mut Vec<PendingRoundRecord>,
    et: &str,
) {
    if et == "end_kyoku" {
        pending_records.clear();
        state.round_step_index = 0;
        state.round_terminal_finalized = false;
    }
}

pub fn finalize_remaining_round_targets(
    state: &mut RoundState,
    pending_records: &[PendingRoundRecord],
    score_norm: f32,
    mc_return_gamma: f32,
) -> Option<Vec<RoundTargetUpdate>> {
    if pending_records.is_empty() || state.round_terminal_finalized {
        return None;
    }
    Some(finalize_round_targets(
        state,
        pending_records,
        None,
        score_norm,
        mc_return_gamma,
    ))
}

pub struct ExportEventContext<'a> {
    pub state: &'a RoundState,
    pub core_state: &'a GameStateCore,
    pub actor: usize,
    pub event: &'a Value,
    pub et: &'a str,
    pub event_index: i32,
}

pub struct ActorDecisionContext<'a> {
    pub event: ExportEventContext<'a>,
    pub legal_actions: Vec<Value>,
}

pub struct DiscardDecisionContext<'a> {
    pub event: ExportEventContext<'a>,
    pub chosen_tile: String,
}

pub struct ActorChosenActionContext<'a> {
    pub decision: ActorDecisionContext<'a>,
    pub chosen_action: Value,
}

#[derive(Clone, Copy)]
pub struct ReactionRecordContext<'a> {
    pub state: &'a RoundState,
    pub core_state: &'a GameStateCore,
    pub actor: usize,
    pub legal_actions: &'a [Value],
    pub event_index: i32,
}

pub struct ReactionChosenActionContext<'a> {
    pub reaction: ReactionRecordContext<'a>,
    pub chosen_action: Value,
}

pub fn build_actor_decision_context(
    event: ExportEventContext<'_>,
) -> Result<Option<ActorDecisionContext<'_>>, String> {
    let legal_actions = enumerate_actor_legal_actions(event.core_state, event.actor)?;
    if legal_actions.is_empty() {
        return Ok(None);
    }
    Ok(Some(ActorDecisionContext {
        event,
        legal_actions,
    }))
}

pub fn build_discard_decision_context(
    event: ExportEventContext<'_>,
) -> Option<DiscardDecisionContext<'_>> {
    if event.et != "dahai" {
        return None;
    }
    let chosen_tile = resolve_discard_tile(event.state, event.event)?;
    if chosen_tile == "?"
        || event.state.feature_tracker.players[event.actor]
            .hand_tiles
            .is_empty()
    {
        return None;
    }
    Some(DiscardDecisionContext { event, chosen_tile })
}

pub fn build_actor_chosen_action_context<F>(
    event: ExportEventContext<'_>,
    resolve_action: F,
) -> Result<Option<ActorChosenActionContext<'_>>, String>
where
    F: FnOnce(&RoundState, &Value) -> Option<Value>,
{
    let Some(chosen_action) = resolve_action(event.state, event.event) else {
        return Ok(None);
    };
    let Some(decision) = build_actor_decision_context(event)? else {
        return Ok(None);
    };
    Ok(Some(ActorChosenActionContext {
        decision,
        chosen_action,
    }))
}

pub fn build_structural_actor_chosen_action_context(
    event: ExportEventContext<'_>,
) -> Result<Option<ActorChosenActionContext<'_>>, String> {
    if !is_structural_actor_decision_event(event.et) {
        return Ok(None);
    }
    build_actor_chosen_action_context(event, resolve_label_action)
}

pub fn build_special_actor_decision_context(
    event: ExportEventContext<'_>,
) -> Result<Option<ActorDecisionContext<'_>>, String> {
    if !is_special_sampling_event(event.et) {
        return Ok(None);
    }
    build_actor_decision_context(event)
}

pub fn build_special_actor_chosen_action_context(
    event: ExportEventContext<'_>,
) -> Result<Option<ActorChosenActionContext<'_>>, String> {
    if !is_special_sampling_event(event.et) {
        return Ok(None);
    }
    build_actor_chosen_action_context(event, |_state, event| Some(event.clone()))
}

pub fn build_reaction_none_context(
    reaction: ReactionRecordContext<'_>,
) -> ReactionChosenActionContext<'_> {
    ReactionChosenActionContext {
        reaction,
        chosen_action: action_none(),
    }
}

pub fn apply_round_target_updates<R, F>(
    records: &mut [R],
    updates: &[RoundTargetUpdate],
    mut apply: F,
) where
    F: FnMut(&mut R, &RoundTargetUpdate),
{
    for update in updates {
        if let Some(record) = records.get_mut(update.record_index) {
            apply(record, update);
        }
    }
}

pub fn drive_export_records<R, FActor, FStep, FReaction, FApply, FPoll>(
    events: &[Value],
    score_norm: f32,
    mc_return_gamma: f32,
    mut on_actor_event: FActor,
    mut on_step_event: FStep,
    mut on_reaction_none: FReaction,
    mut apply_updates: FApply,
    mut poll_continue: FPoll,
) -> Result<Vec<R>, String>
where
    FActor: FnMut(ExportEventContext<'_>) -> Result<Option<R>, String>,
    FStep: FnMut(ExportEventContext<'_>) -> Result<Option<R>, String>,
    FReaction: FnMut(ReactionRecordContext<'_>) -> Result<Option<R>, String>,
    FApply: FnMut(&mut [R], &[RoundTargetUpdate]),
    FPoll: FnMut(usize) -> Result<(), String>,
{
    let mut state = RoundState::default();
    let mut core_state = GameStateCore::default();
    let mut records: Vec<R> = Vec::new();
    let mut pending_records: Vec<PendingRoundRecord> = Vec::new();

    for (event_index, event) in events.iter().enumerate() {
        if event_index % 64 == 0 {
            poll_continue(event_index)?;
        }
        let event_index_i32 = event_index as i32;
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        let next_ev = events.get(event_index + 1);

        begin_round_event(&mut state, &mut pending_records, et);

        let should_advance_step = action_event_should_advance_step(&state, event);
        let actor = value_usize(event, "actor");

        if let Some(actor) = actor {
            let ctx = ExportEventContext {
                state: &state,
                core_state: &core_state,
                actor,
                event,
                et,
                event_index: event_index_i32,
            };
            if let Some(record) = on_actor_event(ctx)? {
                push_record_with_pending(
                    &mut records,
                    &mut pending_records,
                    state.round_step_index,
                    actor,
                    record,
                );
            }
        }

        if should_advance_step {
            if let Some(actor) = actor {
                let ctx = ExportEventContext {
                    state: &state,
                    core_state: &core_state,
                    actor,
                    event,
                    et,
                    event_index: event_index_i32,
                };
                if let Some(record) = on_step_event(ctx)? {
                    push_record_with_pending(
                        &mut records,
                        &mut pending_records,
                        state.round_step_index,
                        actor,
                        record,
                    );
                }
            }
            state.round_step_index += 1;
        }

        apply_event_round_state(&mut state, event);
        apply_event_core(&mut core_state, event)
            .map_err(|err| format!("failed to apply core event at {event_index}: {err}"))?;

        collect_reaction_none_records(
            &core_state,
            et,
            next_ev,
            actor,
            &mut records,
            &mut pending_records,
            state.round_step_index,
            |reaction_actor, legal_actions| {
                on_reaction_none(ReactionRecordContext {
                    state: &state,
                    core_state: &core_state,
                    actor: reaction_actor,
                    legal_actions,
                    event_index: event_index_i32,
                })
            },
        )?;

        if let Some(updates) = maybe_finalize_round_targets(
            &mut state,
            &pending_records,
            et,
            event,
            score_norm,
            mc_return_gamma,
        ) {
            apply_updates(&mut records, &updates);
        }
        reset_after_end_kyoku(&mut state, &mut pending_records, et);
    }

    if let Some(updates) =
        finalize_remaining_round_targets(&mut state, &pending_records, score_norm, mc_return_gamma)
    {
        apply_updates(&mut records, &updates);
    }
    poll_continue(events.len())?;

    Ok(records)
}

pub fn value_i8(value: &Value, key: &str, default: i8) -> i8 {
    value
        .get(key)
        .and_then(Value::as_i64)
        .map(|item| item as i8)
        .unwrap_or(default)
}

pub fn value_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|item| item as usize)
}

pub fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(normalize_tile_repr)
}

pub fn value_tile_list(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter(|tile| *tile != "?")
                .map(normalize_tile_repr)
                .collect()
        })
        .unwrap_or_default()
}

pub fn normalize_event(event: &Value) -> Value {
    let mut out = event.clone();
    if let Some(pai) = out.get("pai").and_then(Value::as_str) {
        out["pai"] = json!(normalize_tile_repr(pai));
    }
    if let Some(pai_raw) = out.get("pai_raw").and_then(Value::as_str) {
        out["pai_raw"] = json!(normalize_tile_repr(pai_raw));
    }
    if let Some(marker) = out.get("dora_marker").and_then(Value::as_str) {
        out["dora_marker"] = json!(normalize_tile_repr(marker));
    }
    if let Some(consumed) = out.get("consumed").and_then(Value::as_array) {
        out["consumed"] = json!(consumed
            .iter()
            .filter_map(Value::as_str)
            .map(normalize_tile_repr)
            .collect::<Vec<_>>());
    }
    if let Some(markers) = out.get("ura_dora_markers").and_then(Value::as_array) {
        out["ura_dora_markers"] = json!(markers
            .iter()
            .filter_map(Value::as_str)
            .map(normalize_tile_repr)
            .collect::<Vec<_>>());
    }
    if let Some(tehais) = out.get("tehais").and_then(Value::as_array) {
        let normalized: Vec<Vec<String>> = tehais
            .iter()
            .map(|tehai| {
                tehai
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .filter(|tile| *tile != "?")
                            .map(normalize_tile_repr)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect();
        out["tehais"] = json!(normalized);
    }
    out
}

pub fn normalize_replay_events(path: &str) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut raw_events = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        raw_events.push(
            serde_json::from_str::<Value>(line)
                .map_err(|err| format!("failed to parse JSON line in {path}: {err}"))?,
        );
    }

    let mut normalized = Vec::new();
    let mut pending_kakan: Option<Value> = None;
    for raw_event in raw_events {
        let event = normalize_event(&raw_event);
        let et = event.get("type").and_then(Value::as_str).unwrap_or("");
        if pending_kakan.is_some() && et != "hora" && et != "kakan_accepted" {
            let pending = pending_kakan.take().unwrap();
            normalized.push(json!({
                "type": "kakan_accepted",
                "actor": pending.get("actor").cloned().unwrap_or(json!(0)),
                "pai": pending.get("pai").cloned().unwrap_or(json!("")),
                "pai_raw": pending.get("pai_raw").cloned().unwrap_or_else(|| pending.get("pai").cloned().unwrap_or(json!(""))),
                "consumed": pending.get("consumed").cloned().unwrap_or_else(|| json!([])),
                "target": pending.get("target").cloned().unwrap_or(Value::Null),
            }));
        }
        normalized.push(event.clone());
        match et {
            "kakan" => pending_kakan = Some(event),
            "kakan_accepted" => pending_kakan = None,
            "reach_accepted" | "dora" | "new_dora" | "hora" => {}
            _ => pending_kakan = None,
        }
    }
    if let Some(pending) = pending_kakan.take() {
        normalized.push(json!({
            "type": "kakan_accepted",
            "actor": pending.get("actor").cloned().unwrap_or(json!(0)),
            "pai": pending.get("pai").cloned().unwrap_or(json!("")),
            "pai_raw": pending.get("pai_raw").cloned().unwrap_or_else(|| pending.get("pai").cloned().unwrap_or(json!(""))),
            "consumed": pending.get("consumed").cloned().unwrap_or_else(|| json!([])),
            "target": pending.get("target").cloned().unwrap_or(Value::Null),
        }));
    }
    Ok(normalized)
}
