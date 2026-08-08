use serde_json::Value;

use crate::counts::TILE_COUNT;
use crate::future_truth::{future_truth_metrics, FutureTruthMetrics};
use crate::legal_actions::reach_discard_candidates;
use crate::progress_summary::{
    summarize_like_python, summarize_one_shanten_draw_metrics,
    summarize_one_shanten_tenpai_pressure,
};
use crate::value_proxy::{
    confirmed_han_floor as shared_confirmed_han_floor, tenpai_value_proxy_norm,
};

fn normalize_tile_repr(tile: &str) -> String {
    match tile {
        "0m" => "5mr".to_string(),
        "0p" => "5pr".to_string(),
        "0s" => "5sr".to_string(),
        _ => tile.to_string(),
    }
}

pub(crate) fn strip_aka(tile: &str) -> String {
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

fn tile34_from_pai(pai: &str) -> Option<usize> {
    let normalized = normalize_tile_repr(pai);
    let pai = normalized.as_str();
    let mut chars = pai.chars();
    let first = chars.next()?;
    if let Some(suit) = chars.next() {
        if matches!(suit, 'm' | 'p' | 's') {
            let rank = if first == '0' {
                5
            } else {
                first.to_digit(10)? as usize
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

fn collect_combined_hand(snapshot: &Value) -> Vec<String> {
    let mut hand: Vec<String> = snapshot
        .get("hand")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .collect();
    if let Some(tsumo) = snapshot.get("tsumo_pai").and_then(Value::as_str) {
        hand.push(normalize_tile_repr(tsumo));
    }
    hand
}

fn terminal_honor_unique_count(hand_tiles: &[String]) -> usize {
    let mut seen = [false; TILE_COUNT];
    let mut count = 0usize;
    for tile in hand_tiles {
        if let Some(idx) = tile34_from_pai(tile) {
            if matches!(
                idx,
                0 | 8 | 9 | 17 | 18 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33
            ) && !seen[idx]
            {
                seen[idx] = true;
                count += 1;
            }
        }
    }
    count
}

fn actor_melds(snapshot: &Value, actor: usize) -> Vec<Value> {
    snapshot
        .get("melds")
        .and_then(Value::as_array)
        .and_then(|groups| groups.get(actor))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn collect_actor_all_tiles(after_hand: &[String], snapshot: &Value, actor: usize) -> Vec<String> {
    let mut tiles = after_hand.to_vec();
    for meld in actor_melds(snapshot, actor) {
        if let Some(consumed) = meld.get("consumed").and_then(Value::as_array) {
            for tile in consumed.iter().filter_map(Value::as_str) {
                tiles.push(normalize_tile_repr(tile));
            }
        }
        if let Some(pai) = meld.get("pai").and_then(Value::as_str) {
            tiles.push(normalize_tile_repr(pai));
        }
    }
    tiles
}

pub(crate) fn counts34_from_tiles(tiles: &[String]) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for tile in tiles {
        if let Some(idx) = tile34_from_pai(tile) {
            counts[idx] = counts[idx].saturating_add(1);
        }
    }
    counts
}

fn visible_counts34_from_snapshot(snapshot: &Value) -> [u8; TILE_COUNT] {
    let mut visible = [0u8; TILE_COUNT];
    for tile in collect_combined_hand(snapshot) {
        if let Some(idx) = tile34_from_pai(&tile) {
            visible[idx] = visible[idx].saturating_add(1);
        }
    }
    if let Some(meld_groups) = snapshot.get("melds").and_then(Value::as_array) {
        for meld_group in meld_groups {
            if let Some(melds) = meld_group.as_array() {
                for meld in melds {
                    if let Some(consumed) = meld.get("consumed").and_then(Value::as_array) {
                        for tile in consumed.iter().filter_map(Value::as_str) {
                            if let Some(idx) = tile34_from_pai(tile) {
                                visible[idx] = visible[idx].saturating_add(1);
                            }
                        }
                    }
                    if let Some(pai) = meld.get("pai").and_then(Value::as_str) {
                        if let Some(idx) = tile34_from_pai(pai) {
                            visible[idx] = visible[idx].saturating_add(1);
                        }
                    }
                }
            }
        }
    }
    if let Some(discard_groups) = snapshot.get("discards").and_then(Value::as_array) {
        for discard_group in discard_groups {
            if let Some(discards) = discard_group.as_array() {
                for discard in discards {
                    let pai_opt = if let Some(obj) = discard.as_object() {
                        obj.get("pai").and_then(Value::as_str)
                    } else {
                        discard.as_str()
                    };
                    if let Some(pai) = pai_opt {
                        if let Some(idx) = tile34_from_pai(pai) {
                            visible[idx] = visible[idx].saturating_add(1);
                        }
                    }
                }
            }
        }
    }
    if let Some(markers) = snapshot.get("dora_markers").and_then(Value::as_array) {
        for marker in markers.iter().filter_map(Value::as_str) {
            if let Some(idx) = tile34_from_pai(marker) {
                visible[idx] = visible[idx].saturating_add(1);
            }
        }
    }
    visible
}

fn remove_tile_once(tiles: &[String], target: &str) -> Vec<String> {
    let norm_target = strip_aka(target);
    let mut removed = false;
    let mut out = Vec::with_capacity(tiles.len().saturating_sub(1));
    for tile in tiles {
        if !removed && strip_aka(tile) == norm_target {
            removed = true;
            continue;
        }
        out.push(tile.clone());
    }
    out
}

fn pair_taatsu_metrics(counts34: &[u8; TILE_COUNT]) -> (u8, u8) {
    let pair_count = counts34.iter().filter(|&&count| count >= 2).count() as u8;
    let mut taatsu_count = 0u8;
    for base in [0usize, 9, 18] {
        let suit = &counts34[base..base + 9];
        for idx in 0..8 {
            if suit[idx] > 0 && suit[idx + 1] > 0 {
                taatsu_count = taatsu_count.saturating_add(1);
            }
        }
        for idx in 0..7 {
            if suit[idx] > 0 && suit[idx + 2] > 0 {
                taatsu_count = taatsu_count.saturating_add(1);
            }
        }
    }
    (pair_count, taatsu_count)
}

fn bakaze_tile(snapshot: &Value) -> usize {
    match snapshot
        .get("bakaze")
        .and_then(Value::as_str)
        .unwrap_or("E")
    {
        "E" => 27,
        "S" => 28,
        "W" => 29,
        "N" => 30,
        _ => 27,
    }
}

fn yakuhai_pair_flag(counts34: &[u8; TILE_COUNT], snapshot: &Value, actor: usize) -> f32 {
    let oya = snapshot
        .get("oya")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as usize;
    let jikaze = 27 + ((actor + 4 - (oya % 4)) % 4);
    let mut ids = vec![31usize, 32, 33, bakaze_tile(snapshot), jikaze];
    ids.sort_unstable();
    ids.dedup();
    if ids
        .iter()
        .any(|&idx| idx < TILE_COUNT && counts34[idx] >= 2)
    {
        1.0
    } else {
        0.0
    }
}

fn opponent_reach_flag(snapshot: &Value, actor: usize) -> f32 {
    let reached = snapshot
        .get("reached")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (idx, flag) in reached.iter().enumerate() {
        if idx != actor && flag.as_bool().unwrap_or(false) {
            return 1.0;
        }
    }
    0.0
}

fn open_hand_flag(snapshot: &Value, actor: usize) -> f32 {
    for meld in actor_melds(snapshot, actor) {
        if meld.get("type").and_then(Value::as_str) != Some("ankan") {
            return 1.0;
        }
    }
    0.0
}

fn dora_from_marker_idx(marker_idx: usize) -> Option<usize> {
    match marker_idx {
        0..=8 => Some(if marker_idx == 8 { 0 } else { marker_idx + 1 }),
        9..=17 => Some(if marker_idx == 17 { 9 } else { marker_idx + 1 }),
        18..=26 => Some(if marker_idx == 26 { 18 } else { marker_idx + 1 }),
        27..=30 => Some(if marker_idx == 30 { 27 } else { marker_idx + 1 }),
        31..=33 => Some(if marker_idx == 33 { 31 } else { marker_idx + 1 }),
        _ => None,
    }
}

fn after_state_bonus_metrics(
    snapshot: &Value,
    actor: usize,
    after_hand: &[String],
) -> (f32, f32, f32, f32, f32, f32) {
    let all_tiles = collect_actor_all_tiles(after_hand, snapshot, actor);
    let counts34 = counts34_from_tiles(&all_tiles);
    let mut dora_count = 0f32;
    if let Some(markers) = snapshot.get("dora_markers").and_then(Value::as_array) {
        for marker in markers.iter().filter_map(Value::as_str) {
            if let Some(marker_idx) = tile34_from_pai(marker).and_then(dora_from_marker_idx) {
                dora_count += f32::from(counts34[marker_idx]);
            }
        }
    }
    let aka_count = all_tiles.iter().filter(|tile| tile.ends_with('r')).count() as f32;
    let terminal_honor_unique = terminal_honor_unique_count(&all_tiles) as f32;
    let meld_count = actor_melds(snapshot, actor).len() as f32;
    let remaining_wall = snapshot
        .get("remaining_wall")
        .and_then(Value::as_i64)
        .unwrap_or(70)
        .max(0) as f32;
    let tanyao_keep = if terminal_honor_unique <= 0.0 {
        1.0
    } else {
        0.0
    };
    (
        dora_count / 10.0,
        aka_count / 3.0,
        terminal_honor_unique / 13.0,
        meld_count / 4.0,
        (remaining_wall / 70.0).min(1.0),
        tanyao_keep,
    )
}

fn yakuhai_triplet_flag(counts34: &[u8; TILE_COUNT], snapshot: &Value, actor: usize) -> f32 {
    let oya = snapshot
        .get("oya")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as usize;
    let jikaze = 27 + ((actor + 4 - (oya % 4)) % 4);
    let mut ids = vec![31usize, 32, 33, bakaze_tile(snapshot), jikaze];
    ids.sort_unstable();
    ids.dedup();
    if ids
        .iter()
        .any(|&idx| idx < TILE_COUNT && counts34[idx] >= 3)
    {
        1.0
    } else {
        0.0
    }
}

fn chiitoi_path_flag(counts34: &[u8; TILE_COUNT], open_hand_flag: f32) -> f32 {
    if open_hand_flag > 0.5 {
        return 0.0;
    }
    let pair_count = counts34.iter().filter(|&&count| count >= 2).count();
    let unique_count = counts34.iter().filter(|&&count| count > 0).count();
    if pair_count >= 6 && unique_count >= 7 {
        1.0
    } else {
        0.0
    }
}

fn iipeiko_path_flag(counts34: &[u8; TILE_COUNT], open_hand_flag: f32) -> f32 {
    if open_hand_flag > 0.5 {
        return 0.0;
    }
    for base in [0usize, 9, 18] {
        let suit = &counts34[base..base + 9];
        for idx in 0..7 {
            if suit[idx] >= 2 && suit[idx + 1] >= 2 && suit[idx + 2] >= 2 {
                return 1.0;
            }
        }
    }
    0.0
}

fn pinfu_like_path_flag(
    counts34: &[u8; TILE_COUNT],
    snapshot: &Value,
    actor: usize,
    open_hand_flag: f32,
) -> f32 {
    if open_hand_flag > 0.5 {
        return 0.0;
    }
    let oya = snapshot
        .get("oya")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as usize;
    let jikaze = 27 + ((actor + 4 - (oya % 4)) % 4);
    let mut yakuhai_ids = vec![31usize, 32, 33, bakaze_tile(snapshot), jikaze];
    yakuhai_ids.sort_unstable();
    yakuhai_ids.dedup();
    let non_value_pair =
        (0..TILE_COUNT).any(|idx| counts34[idx] >= 2 && !yakuhai_ids.contains(&idx));
    if !non_value_pair {
        return 0.0;
    }
    let mut sequence_heads = 0usize;
    for base in [0usize, 9, 18] {
        let suit = &counts34[base..base + 9];
        for idx in 0..7 {
            if suit[idx] > 0 && suit[idx + 1] > 0 && suit[idx + 2] > 0 {
                sequence_heads += 1;
            }
        }
    }
    if sequence_heads >= 3 {
        1.0
    } else {
        0.0
    }
}

fn common_yaku_path_metrics(
    counts34: &[u8; TILE_COUNT],
    snapshot: &Value,
    actor: usize,
    open_hand_flag: f32,
) -> (f32, f32, f32, f32, f32) {
    let tanyao_keep = if counts34.iter().enumerate().any(|(idx, &count)| {
        count > 0
            && matches!(
                idx,
                0 | 8 | 9 | 17 | 18 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33
            )
    }) {
        0.0
    } else {
        1.0
    };
    let yakuhai_pair = yakuhai_pair_flag(counts34, snapshot, actor);
    let chiitoi_path = chiitoi_path_flag(counts34, open_hand_flag);
    let iipeiko_path = iipeiko_path_flag(counts34, open_hand_flag);
    let pinfu_like_path = pinfu_like_path_flag(counts34, snapshot, actor, open_hand_flag);
    (
        tanyao_keep,
        yakuhai_pair,
        chiitoi_path,
        iipeiko_path,
        pinfu_like_path,
    )
}

pub(crate) fn progress_key(summary: &crate::progress_summary::Summary3n1) -> (i32, i32, i32, i32) {
    (
        -(summary.shanten as i32),
        summary.ukeire_live_count as i32,
        summary.ukeire_type_count as i32,
        summary.waits_count as i32,
    )
}

pub(crate) fn hand_tiles_from_snapshot(snapshot: &Value) -> Vec<String> {
    snapshot
        .get("hand")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn snapshot_with_after_hand(
    snapshot: &Value,
    actor: usize,
    after_hand: &[String],
) -> Value {
    let mut next_snapshot = snapshot.clone();
    if let Some(obj) = next_snapshot.as_object_mut() {
        obj.insert(
            "hand".to_string(),
            Value::Array(after_hand.iter().cloned().map(Value::String).collect()),
        );
        obj.insert("tsumo_pai".to_string(), Value::Null);
        obj.insert("last_discard".to_string(), Value::Null);
        obj.insert("last_kakan".to_string(), Value::Null);
        obj.insert("actor_to_move".to_string(), Value::from(actor as i64));
        if let Some(last_tsumo) = obj.get_mut("last_tsumo").and_then(Value::as_array_mut) {
            if actor < last_tsumo.len() {
                last_tsumo[actor] = Value::Null;
            }
        }
        if let Some(last_tsumo_raw) = obj.get_mut("last_tsumo_raw").and_then(Value::as_array_mut) {
            if actor < last_tsumo_raw.len() {
                last_tsumo_raw[actor] = Value::Null;
            }
        }
    }
    next_snapshot
}
type ExactTruthMetrics = FutureTruthMetrics;

fn value_proxy_metrics(
    summary: &crate::progress_summary::Summary3n1,
    exact_tenpai_metrics: ExactTruthMetrics,
    exact_one_shanten_metrics: ExactTruthMetrics,
    good_shape_live: u8,
    improvement_live: u8,
    tenpai_live: u8,
    best_tenpai_waits_live: u8,
    dora_norm: f32,
    aka_norm: f32,
    open_hand_flag: f32,
    yakuhai_pair: f32,
    yakuhai_triplet: f32,
    current_tanyao_keep: f32,
    after_tanyao_keep: f32,
    current_yakuhai_pair: f32,
    current_chiitoi_path: f32,
    after_chiitoi_path: f32,
    current_iipeiko_path: f32,
    after_iipeiko_path: f32,
    current_pinfu_like_path: f32,
    after_pinfu_like_path: f32,
) -> (f32, f32, f32, f32) {
    let tenpai_flag = if summary.shanten == 0 { 1.0 } else { 0.0 };
    let confirmed_han_floor = shared_confirmed_han_floor(
        yakuhai_triplet,
        after_tanyao_keep,
        dora_norm * 10.0 + aka_norm * 3.0,
    );
    let tenpai_value_proxy = tenpai_value_proxy_norm(
        confirmed_han_floor,
        summary.waits_count,
        summary.ukeire_live_count as usize,
        summary.shanten == 0,
    );
    let one_shanten_tenpai_pressure = if summary.shanten == 1 {
        (tenpai_live as f32 / 34.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let one_shanten_shape_bonus = if summary.shanten == 1 {
        ((good_shape_live as f32 / 34.0) * 0.6
            + (improvement_live as f32 / 34.0) * 0.4
            + (best_tenpai_waits_live.min(12) as f32 / 12.0) * 0.35)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let pre_tenpai_value_proxy = (1.0 - open_hand_flag)
        * one_shanten_tenpai_pressure
        * (0.65 * (confirmed_han_floor / 8.0) + 0.35 * one_shanten_shape_bonus).clamp(0.0, 1.0);
    let non_tenpai_value_proxy = exact_one_shanten_metrics
        .max_hand_value_proxy
        .max(pre_tenpai_value_proxy);
    let max_hand_value_proxy = exact_tenpai_metrics
        .max_hand_value_proxy
        .max(tenpai_flag * tenpai_value_proxy)
        .max(non_tenpai_value_proxy);
    let reach_bonus =
        0.12 + 0.05 * after_pinfu_like_path + 0.05 * after_iipeiko_path + 0.04 * after_chiitoi_path;
    let reach_hand_value_proxy = exact_tenpai_metrics
        .reach_hand_value_proxy
        .max(exact_one_shanten_metrics.reach_hand_value_proxy)
        .max(
            (tenpai_flag * (1.0 - open_hand_flag) * (max_hand_value_proxy + reach_bonus).min(1.0))
                .max(
                    non_tenpai_value_proxy
                        * (0.55
                            + 0.20 * after_pinfu_like_path
                            + 0.15 * after_iipeiko_path
                            + 0.10 * after_chiitoi_path)
                            .clamp(0.0, 1.0),
                ),
        );
    let after_tanyao_truth = after_tanyao_keep
        .max(exact_tenpai_metrics.tanyao_route)
        .max(exact_one_shanten_metrics.tanyao_route);
    let after_yakuhai_truth = yakuhai_pair
        .max(exact_tenpai_metrics.yakuhai_route)
        .max(exact_one_shanten_metrics.yakuhai_route);
    let after_chiitoi_truth = after_chiitoi_path
        .max(exact_tenpai_metrics.chiitoi_route)
        .max(exact_one_shanten_metrics.chiitoi_route);
    let after_iipeiko_truth = after_iipeiko_path
        .max(exact_tenpai_metrics.iipeiko_route)
        .max(exact_one_shanten_metrics.iipeiko_route);
    let after_pinfu_truth = after_pinfu_like_path
        .max(exact_tenpai_metrics.pinfu_route)
        .max(exact_one_shanten_metrics.pinfu_route);
    let yaku_break_flag = if current_tanyao_keep > after_tanyao_truth
        || current_yakuhai_pair > after_yakuhai_truth
        || current_chiitoi_path > after_chiitoi_truth
        || current_iipeiko_path > after_iipeiko_truth
        || current_pinfu_like_path > after_pinfu_truth
    {
        1.0
    } else {
        0.0
    };
    let closed_route_bonus = after_pinfu_like_path
        .max(after_iipeiko_path)
        .max(after_chiitoi_path)
        .max(exact_tenpai_metrics.pinfu_route)
        .max(exact_tenpai_metrics.iipeiko_route)
        .max(exact_tenpai_metrics.chiitoi_route)
        .max(exact_one_shanten_metrics.pinfu_route)
        .max(exact_one_shanten_metrics.iipeiko_route)
        .max(exact_one_shanten_metrics.chiitoi_route);
    let closed_value_proxy = (1.0 - open_hand_flag)
        * (0.45 * max_hand_value_proxy
            + 0.25 * reach_hand_value_proxy
            + 0.2 * (confirmed_han_floor / 8.0)
            + 0.1 * closed_route_bonus
            + 0.1 * non_tenpai_value_proxy)
            .min(1.0);
    (
        max_hand_value_proxy,
        reach_hand_value_proxy,
        yaku_break_flag,
        closed_value_proxy,
    )
}

fn shape_value_metrics(
    counts34: &[u8; TILE_COUNT],
    pair_count: u8,
    open_hand_flag: f32,
    yakuhai_triplet: f32,
    exact_yakuhai_route: f32,
) -> (f32, f32, f32, f32) {
    let suit_counts = [
        counts34[0..9].iter().map(|&v| v as usize).sum::<usize>(),
        counts34[9..18].iter().map(|&v| v as usize).sum::<usize>(),
        counts34[18..27].iter().map(|&v| v as usize).sum::<usize>(),
    ];
    let honor_count = counts34[27..34].iter().map(|&v| v as usize).sum::<usize>();
    let total_tiles = (suit_counts.iter().sum::<usize>() + honor_count).max(1) as f32;
    let max_suit = *suit_counts.iter().max().unwrap_or(&0) as f32;
    let honitsu_tendency = (max_suit + honor_count as f32) / total_tiles;
    let chinitsu_tendency = if honor_count > 0 {
        0.0
    } else {
        max_suit / (suit_counts.iter().sum::<usize>().max(1) as f32)
    };
    let chiitoi_tendency = (1.0 - open_hand_flag) * ((pair_count as f32) / 7.0).min(1.0);
    let yakuhai_value_proxy = (0.6 * yakuhai_triplet + 0.4 * chiitoi_tendency)
        .max(exact_yakuhai_route)
        .min(1.0);
    (
        honitsu_tendency,
        chinitsu_tendency,
        chiitoi_tendency,
        yakuhai_value_proxy,
    )
}

fn summary_vector(
    after_hand: &[String],
    visible_counts34: &[u8; TILE_COUNT],
    snapshot: &Value,
    actor: usize,
    current_shanten: i8,
    open_hand_flag_value: f32,
    current_tanyao_keep: f32,
    current_yakuhai_pair: f32,
    current_chiitoi_path: f32,
    current_iipeiko_path: f32,
    current_pinfu_like_path: f32,
) -> [f32; 28] {
    let counts34 = counts34_from_tiles(after_hand);
    let summary = summarize_like_python(&counts34, visible_counts34);
    let (good_shape_live, improvement_live) =
        summarize_one_shanten_draw_metrics(&counts34, visible_counts34);
    let (tenpai_live, best_tenpai_waits_live) =
        summarize_one_shanten_tenpai_pressure(&counts34, visible_counts34);
    let future_metrics = future_truth_metrics(
        snapshot,
        actor,
        after_hand,
        &summary,
        visible_counts34,
        open_hand_flag_value,
    );
    let exact_tenpai_metrics = if summary.shanten == 0 {
        future_metrics
    } else {
        ExactTruthMetrics::default()
    };
    let exact_one_shanten_metrics = if matches!(summary.shanten, 1 | 2) {
        future_metrics
    } else {
        ExactTruthMetrics::default()
    };
    let (pair_count, taatsu_count) = pair_taatsu_metrics(&counts34);
    let delta_shanten = ((current_shanten - summary.shanten) as f32 / 4.0).clamp(-1.0, 1.0);
    let (dora_norm, aka_norm, terminal_norm, meld_norm, wall_norm, _tanyao_keep_bonus) =
        after_state_bonus_metrics(snapshot, actor, after_hand);
    let (
        after_tanyao_keep,
        yakuhai_pair,
        after_chiitoi_path,
        after_iipeiko_path,
        after_pinfu_like_path,
    ) = common_yaku_path_metrics(&counts34, snapshot, actor, open_hand_flag_value);
    let yakuhai_triplet = yakuhai_triplet_flag(&counts34, snapshot, actor);
    let (max_hand_value_proxy, reach_hand_value_proxy, yaku_break_flag, closed_value_proxy) =
        value_proxy_metrics(
            &summary,
            exact_tenpai_metrics,
            exact_one_shanten_metrics,
            good_shape_live,
            improvement_live,
            tenpai_live,
            best_tenpai_waits_live,
            dora_norm,
            aka_norm,
            open_hand_flag_value,
            yakuhai_pair,
            yakuhai_triplet,
            current_tanyao_keep,
            after_tanyao_keep,
            current_yakuhai_pair,
            current_chiitoi_path,
            after_chiitoi_path,
            current_iipeiko_path,
            after_iipeiko_path,
            current_pinfu_like_path,
            after_pinfu_like_path,
        );
    let (honitsu_tendency, chinitsu_tendency, chiitoi_tendency, yakuhai_value_proxy) =
        shape_value_metrics(
            &counts34,
            pair_count,
            open_hand_flag_value,
            yakuhai_triplet,
            exact_tenpai_metrics
                .yakuhai_route
                .max(exact_one_shanten_metrics.yakuhai_route),
        );
    [
        summary.shanten as f32 / 8.0,
        if summary.shanten == 0 { 1.0 } else { 0.0 },
        summary.waits_count as f32 / 34.0,
        summary.ukeire_type_count as f32 / 34.0,
        summary.ukeire_live_count as f32 / 34.0,
        good_shape_live as f32 / 34.0,
        improvement_live as f32 / 34.0,
        pair_count as f32 / 7.0,
        taatsu_count as f32 / 6.0,
        yakuhai_pair,
        opponent_reach_flag(snapshot, actor),
        open_hand_flag_value,
        delta_shanten,
        1.0,
        dora_norm,
        aka_norm,
        terminal_norm,
        meld_norm,
        wall_norm,
        after_tanyao_keep,
        max_hand_value_proxy,
        reach_hand_value_proxy,
        yaku_break_flag,
        closed_value_proxy,
        honitsu_tendency,
        chinitsu_tendency,
        chiitoi_tendency,
        yakuhai_value_proxy,
    ]
}

pub fn build_keqingv4_discard_summary(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Vec<f32> {
    let ctx = current_summary_context(snapshot, actor);

    let mut out = vec![0.0f32; 34 * 28];
    let mut seen = [false; TILE_COUNT];
    for action in legal_actions {
        if action.get("type").and_then(Value::as_str) != Some("dahai") {
            continue;
        }
        let Some(pai) = action.get("pai").and_then(Value::as_str) else {
            continue;
        };
        let Some(tile34) = tile34_from_pai(pai) else {
            continue;
        };
        if seen[tile34] {
            continue;
        }
        seen[tile34] = true;
        let after_hand = remove_tile_once(&ctx.hand, pai);
        let vec = summary_vector(
            &after_hand,
            &ctx.visible_counts34,
            snapshot,
            actor,
            ctx.current_summary.shanten,
            ctx.current_open,
            ctx.current_tanyao_keep,
            ctx.current_yakuhai_pair,
            ctx.current_chiitoi_path,
            ctx.current_iipeiko_path,
            ctx.current_pinfu_like_path,
        );
        let offset = tile34 * 28;
        out[offset..offset + 28].copy_from_slice(&vec);
    }
    out
}

fn summary_score(vec: &[f32; 28]) -> f32 {
    -100.0 * vec[0]
        + 12.0 * vec[1]
        + 10.0 * vec[4]
        + 3.0 * vec[2]
        + 1.0 * vec[8]
        + 0.5 * vec[7]
        + 2.0 * vec[14]
        + 1.0 * vec[19]
        + 3.0 * vec[20]
        + 2.0 * vec[21]
        - 1.5 * vec[22]
        + 1.0 * vec[23]
        + 0.8 * vec[24]
        + 0.8 * vec[25]
        + 0.6 * vec[26]
        + 1.2 * vec[27]
}

fn call_action_slot(action: &Value) -> Option<usize> {
    let action_type = action.get("type").and_then(Value::as_str)?;
    match action_type {
        "none" => Some(7),
        "pon" => Some(3),
        "daiminkan" => Some(4),
        "ankan" => Some(5),
        "kakan" => Some(6),
        "chi" => {
            let pai_rank = action
                .get("pai")
                .and_then(Value::as_str)
                .and_then(|pai| strip_aka(pai).chars().next())
                .and_then(|ch| ch.to_digit(10))
                .unwrap_or(0) as i32;
            let mut ranks: Vec<i32> = action
                .get("consumed")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter_map(|tile| {
                    strip_aka(tile)
                        .chars()
                        .next()
                        .and_then(|ch| ch.to_digit(10))
                })
                .map(|v| v as i32)
                .collect();
            ranks.sort_unstable();
            Some(if ranks.len() == 2 {
                if pai_rank < ranks[0] {
                    0
                } else if pai_rank < ranks[1] {
                    1
                } else {
                    2
                }
            } else {
                0
            })
        }
        _ => None,
    }
}

struct CurrentSummaryContext {
    hand: Vec<String>,
    visible_counts34: [u8; TILE_COUNT],
    current_summary: crate::progress_summary::Summary3n1,
    current_open: f32,
    current_vec: [f32; 28],
    current_tanyao_keep: f32,
    current_yakuhai_pair: f32,
    current_chiitoi_path: f32,
    current_iipeiko_path: f32,
    current_pinfu_like_path: f32,
}

fn current_summary_context(snapshot: &Value, actor: usize) -> CurrentSummaryContext {
    let hand = collect_combined_hand(snapshot);
    let counts34 = counts34_from_tiles(&hand);
    let visible_counts34 = visible_counts34_from_snapshot(snapshot);
    let current_summary = summarize_like_python(&counts34, &visible_counts34);
    let current_open = open_hand_flag(snapshot, actor);
    let current_all_tiles = collect_actor_all_tiles(&hand, snapshot, actor);
    let current_counts34_all = counts34_from_tiles(&current_all_tiles);
    let (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = common_yaku_path_metrics(&current_counts34_all, snapshot, actor, current_open);
    let current_vec = summary_vector(
        &hand,
        &visible_counts34,
        snapshot,
        actor,
        current_summary.shanten,
        current_open,
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    );
    CurrentSummaryContext {
        hand,
        visible_counts34,
        current_summary,
        current_open,
        current_vec,
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    }
}

fn project_call_state(
    snapshot: &Value,
    actor: usize,
    action: &Value,
) -> Option<(Vec<String>, Vec<Value>, f32, bool)> {
    let action_type = action.get("type").and_then(Value::as_str)?;
    let mut remove_tiles: Vec<String> = action
        .get("consumed")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(normalize_tile_repr)
        .collect();
    let actor_melds_current = actor_melds(snapshot, actor);
    let current_open = open_hand_flag(snapshot, actor);
    let pai = action
        .get("pai")
        .and_then(Value::as_str)
        .map(normalize_tile_repr);
    if action_type == "kakan" {
        let pai = pai.clone()?;
        remove_tiles = vec![pai];
    }
    let hand = collect_combined_hand(snapshot);
    let after_hand = {
        let mut out = hand.clone();
        for tile in &remove_tiles {
            out = remove_tile_once(&out, tile);
        }
        out
    };
    if after_hand.len() >= hand.len() {
        return None;
    }

    match action_type {
        "chi" | "pon" | "daiminkan" => {
            let mut melds = actor_melds_current.clone();
            let consumed_json: Vec<Value> = action
                .get("consumed")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            melds.push(serde_json::json!({
                "type": action_type,
                "pai": pai,
                "consumed": consumed_json,
                "target": action.get("target").cloned().unwrap_or(Value::Null),
            }));
            let open_flag = 1.0;
            let needs_rinshan = action_type == "daiminkan";
            Some((after_hand, melds, open_flag, needs_rinshan))
        }
        "ankan" => {
            let mut melds = actor_melds_current.clone();
            let consumed_json: Vec<Value> = action
                .get("consumed")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            melds.push(serde_json::json!({
                "type": "ankan",
                "pai": pai.clone().or_else(|| remove_tiles.first().cloned()),
                "consumed": consumed_json,
                "target": actor,
            }));
            Some((after_hand, melds, current_open, true))
        }
        "kakan" => {
            let mut melds = Vec::new();
            let pai_norm = pai?;
            let mut upgraded = false;
            for meld in actor_melds_current {
                let mut next_meld = meld;
                if !upgraded
                    && next_meld.get("type").and_then(Value::as_str) == Some("pon")
                    && next_meld.get("pai").and_then(Value::as_str).map(strip_aka)
                        == Some(strip_aka(&pai_norm))
                {
                    let mut consumed: Vec<Value> = next_meld
                        .get("consumed")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    consumed.push(Value::String(pai_norm.clone()));
                    next_meld["type"] = Value::String("kakan".to_string());
                    next_meld["consumed"] = Value::Array(consumed);
                    next_meld["pai"] = Value::String(strip_aka(&pai_norm));
                    upgraded = true;
                }
                melds.push(next_meld);
            }
            if !upgraded {
                melds.push(serde_json::json!({
                    "type": "kakan",
                    "pai": strip_aka(&pai_norm),
                    "consumed": [pai_norm.clone()],
                    "target": Value::Null,
                }));
            }
            Some((after_hand, melds, current_open, true))
        }
        _ => None,
    }
}

pub fn project_keqingv4_call_snapshot(
    snapshot: &Value,
    actor: usize,
    action: &Value,
) -> Option<Value> {
    let (projected_hand, projected_melds, _projected_open, _needs_rinshan) =
        project_call_state(snapshot, actor, action)?;
    let mut next_snapshot = snapshot.clone();
    if let Some(obj) = next_snapshot.as_object_mut() {
        obj.insert(
            "hand".to_string(),
            Value::Array(projected_hand.into_iter().map(Value::String).collect()),
        );
        if let Some(meld_groups) = obj.get_mut("melds").and_then(Value::as_array_mut) {
            if actor < meld_groups.len() {
                meld_groups[actor] = Value::Array(projected_melds);
            }
        }
        obj.insert("last_discard".to_string(), Value::Null);
        obj.insert("last_kakan".to_string(), Value::Null);
        obj.insert("tsumo_pai".to_string(), Value::Null);
        obj.insert("actor_to_move".to_string(), Value::from(actor as i64));
    }
    Some(next_snapshot)
}

pub fn project_keqingv4_discard_snapshot(snapshot: &Value, actor: usize, pai: &str) -> Value {
    let norm_pai = normalize_tile_repr(pai);
    let mut removed = false;
    let current_hand = snapshot
        .get("hand")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let new_hand: Vec<Value> = current_hand
        .into_iter()
        .filter(|tile| {
            if removed {
                return true;
            }
            let tile_str = tile.as_str().unwrap_or_default();
            if strip_aka(tile_str) == strip_aka(&norm_pai) {
                removed = true;
                return false;
            }
            true
        })
        .collect();

    let mut next_snapshot = snapshot.clone();
    if let Some(obj) = next_snapshot.as_object_mut() {
        obj.insert("hand".to_string(), Value::Array(new_hand));
        if let Some(discard_groups) = obj.get_mut("discards").and_then(Value::as_array_mut) {
            if actor < discard_groups.len() {
                let mut next_discards = discard_groups[actor]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                next_discards.push(Value::String(pai.to_string()));
                discard_groups[actor] = Value::Array(next_discards);
            }
        }
        obj.insert("tsumo_pai".to_string(), Value::Null);
    }
    next_snapshot
}

pub fn project_keqingv4_rinshan_draw_snapshot(snapshot: &Value, actor: usize, pai: &str) -> Value {
    let mut next_snapshot = snapshot.clone();
    if let Some(obj) = next_snapshot.as_object_mut() {
        let mut hand = obj
            .get("hand")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        hand.push(Value::String(pai.to_string()));
        obj.insert("hand".to_string(), Value::Array(hand));
        obj.insert("tsumo_pai".to_string(), Value::String(pai.to_string()));

        let norm_pai = strip_aka(pai);
        let mut last_tsumo = obj
            .get("last_tsumo")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::Null; 4]);
        while last_tsumo.len() <= actor {
            last_tsumo.push(Value::Null);
        }
        last_tsumo[actor] = Value::String(norm_pai);
        obj.insert("last_tsumo".to_string(), Value::Array(last_tsumo));

        let mut last_tsumo_raw = obj
            .get("last_tsumo_raw")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::Null; 4]);
        while last_tsumo_raw.len() <= actor {
            last_tsumo_raw.push(Value::Null);
        }
        last_tsumo_raw[actor] = Value::String(pai.to_string());
        obj.insert("last_tsumo_raw".to_string(), Value::Array(last_tsumo_raw));

        obj.insert("actor_to_move".to_string(), Value::from(actor as i64));
        obj.insert("last_discard".to_string(), Value::Null);
        obj.insert("last_kakan".to_string(), Value::Null);
    }
    next_snapshot
}

pub fn enumerate_keqingv4_post_meld_discards(snapshot: &Value, actor: usize) -> Vec<Value> {
    let hand = snapshot
        .get("hand")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut seen = std::collections::BTreeSet::<String>::new();
    let mut actions = Vec::new();
    for tile in hand {
        let Some(tile_str) = tile.as_str() else {
            continue;
        };
        let key = strip_aka(tile_str);
        if !seen.insert(key) {
            continue;
        }
        actions.push(serde_json::json!({
            "type": "dahai",
            "actor": actor,
            "pai": tile_str,
            "tsumogiri": false,
        }));
    }
    actions
}

pub fn enumerate_keqingv4_live_draw_weights(snapshot: &Value) -> Vec<(String, u8)> {
    let visible = visible_counts34_from_snapshot(snapshot);
    let mut out = Vec::new();
    for tile34 in 0..TILE_COUNT {
        let live = 4u8.saturating_sub(visible[tile34]);
        if live > 0 {
            out.push((TILE34_STRINGS[tile34].to_string(), live));
        }
    }
    out
}

fn best_discard_summary_from_snapshot(snapshot: &Value, actor: usize) -> [f32; 28] {
    let ctx = current_summary_context(snapshot, actor);
    let legal_actions = enumerate_keqingv4_post_meld_discards(snapshot, actor);
    if legal_actions.is_empty() {
        return ctx.current_vec;
    }

    let mut best_vec: Option<[f32; 28]> = None;
    let mut best_score = f32::NEG_INFINITY;
    for action in &legal_actions {
        let Some(pai) = action.get("pai").and_then(Value::as_str) else {
            continue;
        };
        let after_hand = remove_tile_once(&ctx.hand, pai);
        let vec = summary_vector(
            &after_hand,
            &ctx.visible_counts34,
            snapshot,
            actor,
            ctx.current_summary.shanten,
            ctx.current_open,
            ctx.current_tanyao_keep,
            ctx.current_yakuhai_pair,
            ctx.current_chiitoi_path,
            ctx.current_iipeiko_path,
            ctx.current_pinfu_like_path,
        );
        let score = summary_score(&vec);
        if score > best_score {
            best_score = score;
            best_vec = Some(vec);
        }
    }

    best_vec.unwrap_or(ctx.current_vec)
}

fn projected_call_summary_vector(
    snapshot: &Value,
    actor: usize,
    action: &Value,
) -> Option<[f32; 28]> {
    let (_projected_hand, _projected_melds, _projected_open, needs_rinshan) =
        project_call_state(snapshot, actor, action)?;
    let projected_snapshot = project_keqingv4_call_snapshot(snapshot, actor, action)?;

    if !needs_rinshan {
        return Some(best_discard_summary_from_snapshot(
            &projected_snapshot,
            actor,
        ));
    }

    let draw_weights = enumerate_keqingv4_live_draw_weights(&projected_snapshot);
    if draw_weights.is_empty() {
        return Some(current_summary_context(&projected_snapshot, actor).current_vec);
    }

    let mut acc = [0.0f32; 28];
    let mut total_weight = 0.0f32;
    for (draw_tile, live) in draw_weights {
        if live == 0 {
            continue;
        }
        let draw_snapshot =
            project_keqingv4_rinshan_draw_snapshot(&projected_snapshot, actor, &draw_tile);
        let vec = best_discard_summary_from_snapshot(&draw_snapshot, actor);
        let weight = f32::from(live);
        for idx in 0..28 {
            acc[idx] += weight * vec[idx];
        }
        total_weight += weight;
    }

    if total_weight <= 0.0 {
        return Some(current_summary_context(&projected_snapshot, actor).current_vec);
    }
    for value in &mut acc {
        *value /= total_weight;
    }
    Some(acc)
}

pub(crate) const TILE34_STRINGS: [&str; TILE_COUNT] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p",
    "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P", "F",
    "C",
];

pub fn build_keqingv4_call_summary(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Vec<f32> {
    let ctx = current_summary_context(snapshot, actor);
    let mut out = vec![0.0f32; 8 * 28];
    for action in legal_actions {
        let Some(action_type) = action.get("type").and_then(Value::as_str) else {
            continue;
        };
        let Some(slot) = call_action_slot(action) else {
            continue;
        };
        let offset = slot * 28;
        if action_type == "none" {
            out[offset..offset + 28].copy_from_slice(&ctx.current_vec);
            continue;
        }
        let Some(vec) = projected_call_summary_vector(snapshot, actor, action) else {
            out[offset..offset + 28].copy_from_slice(&ctx.current_vec);
            continue;
        };
        out[offset..offset + 28].copy_from_slice(&vec);
    }
    out
}

fn resolve_reach_tsumo_tile(snapshot: &Value, actor: usize) -> (Option<String>, Option<String>) {
    if let Some(tsumo) = snapshot.get("tsumo_pai").and_then(Value::as_str) {
        return (Some(strip_aka(tsumo)), Some(tsumo.to_string()));
    }
    let last_tsumo = snapshot
        .get("last_tsumo")
        .and_then(Value::as_array)
        .and_then(|items| items.get(actor))
        .and_then(Value::as_str)
        .map(strip_aka);
    let last_tsumo_raw = snapshot
        .get("last_tsumo_raw")
        .and_then(Value::as_array)
        .and_then(|items| items.get(actor))
        .and_then(Value::as_str)
        .map(|value| value.to_string());
    match (last_tsumo, last_tsumo_raw) {
        (Some(norm), Some(raw)) => (Some(norm), Some(raw)),
        (Some(norm), None) => (Some(norm.clone()), Some(norm)),
        (None, Some(raw)) => (Some(strip_aka(&raw)), Some(raw)),
        (None, None) => (None, None),
    }
}

pub fn enumerate_keqingv4_reach_discards(snapshot: &Value, actor: usize) -> Vec<(String, bool)> {
    let hand_tiles = snapshot
        .get("hand")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let mut hand = std::collections::BTreeMap::<String, u8>::new();
    for tile in hand_tiles {
        let normalized = strip_aka(tile);
        *hand.entry(normalized).or_insert(0) += 1;
    }
    let concealed_tile_count: u8 = hand.values().copied().sum();
    if concealed_tile_count % 3 == 1 {
        if let Some(tsumo) = snapshot.get("tsumo_pai").and_then(Value::as_str) {
            let normalized = strip_aka(tsumo);
            *hand.entry(normalized).or_insert(0) += 1;
        }
    }
    if hand.is_empty() {
        return Vec::new();
    }
    let (last_tsumo, last_tsumo_raw) = resolve_reach_tsumo_tile(snapshot, actor);
    reach_discard_candidates(&hand, last_tsumo.as_deref(), last_tsumo_raw.as_deref())
}

pub fn project_keqingv4_reach_snapshot(snapshot: &Value, actor: usize, pai: &str) -> Value {
    let mut next_snapshot = project_keqingv4_discard_snapshot(snapshot, actor, pai);
    if let Some(obj) = next_snapshot.as_object_mut() {
        let mut reached = obj
            .get("reached")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::Bool(false); 4]);
        while reached.len() <= actor {
            reached.push(Value::Bool(false));
        }
        reached[actor] = Value::Bool(true);
        obj.insert("reached".to_string(), Value::Array(reached));

        let mut pending_reach = obj
            .get("pending_reach")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::Bool(false); 4]);
        while pending_reach.len() <= actor {
            pending_reach.push(Value::Bool(false));
        }
        pending_reach[actor] = Value::Bool(false);
        obj.insert("pending_reach".to_string(), Value::Array(pending_reach));
    }
    next_snapshot
}

fn best_reach_summary_vector(
    snapshot: &Value,
    actor: usize,
    visible_counts34: &[u8; TILE_COUNT],
    current_shanten: i8,
) -> Option<[f32; 28]> {
    let hand = collect_combined_hand(snapshot);
    if hand.is_empty() {
        return None;
    }
    let candidates = enumerate_keqingv4_reach_discards(snapshot, actor)
        .into_iter()
        .map(|(pai, _tsumogiri)| pai)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }
    let mut best_vec: Option<[f32; 28]> = None;
    let ctx = current_summary_context(snapshot, actor);
    let mut best_score = f32::NEG_INFINITY;
    for tile in &candidates {
        let after_hand = remove_tile_once(&hand, tile);
        let vec = summary_vector(
            &after_hand,
            visible_counts34,
            snapshot,
            actor,
            current_shanten,
            0.0,
            ctx.current_tanyao_keep,
            ctx.current_yakuhai_pair,
            ctx.current_chiitoi_path,
            ctx.current_iipeiko_path,
            ctx.current_pinfu_like_path,
        );
        let score = summary_score(&vec);
        if score > best_score {
            best_score = score;
            best_vec = Some(vec);
        }
    }
    best_vec
}

fn build_hora_summary_vector(
    snapshot: &Value,
    actor: usize,
    current_vector: &[f32; 28],
    action: &Value,
) -> [f32; 28] {
    let mut vec = *current_vector;
    let target = action
        .get("target")
        .and_then(Value::as_i64)
        .unwrap_or(actor as i64) as usize;
    let is_tsumo = if target == actor { 1.0 } else { 0.0 };
    let is_ron = 1.0 - is_tsumo;
    let rinshan = if action
        .get("is_rinshan")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || snapshot
            .get("_hora_is_rinshan")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };
    let chankan = if action
        .get("is_chankan")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || snapshot
            .get("_hora_is_chankan")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };
    let haitei = if action
        .get("is_haitei")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || snapshot
            .get("_hora_is_haitei")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };
    let houtei = if action
        .get("is_houtei")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || snapshot
            .get("_hora_is_houtei")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };
    vec[0] = -0.125;
    vec[1] = 1.0;
    vec[2] = is_tsumo;
    vec[3] = is_ron;
    vec[4] = rinshan;
    vec[5] = chankan;
    vec[6] = haitei;
    vec[7] = houtei;
    vec[12] = 1.0;
    vec[13] = 1.0;
    vec
}

fn build_ryukyoku_summary_vector(hand_tiles: &[String], current_vector: &[f32; 28]) -> [f32; 28] {
    let mut vec = *current_vector;
    let unique_yaochu = terminal_honor_unique_count(hand_tiles) as f32;
    vec[2] = unique_yaochu / 13.0;
    vec[3] = if unique_yaochu >= 9.0 { 1.0 } else { 0.0 };
    vec[4] = unique_yaochu / 9.0;
    vec[12] = if unique_yaochu >= 9.0 { 1.0 } else { 0.0 };
    vec[13] = 1.0;
    vec
}

pub fn build_keqingv4_special_summary(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Vec<f32> {
    let ctx = current_summary_context(snapshot, actor);
    let mut out = vec![0.0f32; 3 * 28];
    for action in legal_actions {
        match action.get("type").and_then(Value::as_str) {
            Some("reach") => {
                let vec = best_reach_summary_vector(
                    snapshot,
                    actor,
                    &ctx.visible_counts34,
                    ctx.current_summary.shanten,
                )
                .unwrap_or(ctx.current_vec);
                out[0..28].copy_from_slice(&vec);
            }
            Some("hora") => {
                let vec = build_hora_summary_vector(snapshot, actor, &ctx.current_vec, action);
                out[28..56].copy_from_slice(&vec);
            }
            Some("ryukyoku") => {
                let vec = build_ryukyoku_summary_vector(&ctx.hand, &ctx.current_vec);
                out[56..84].copy_from_slice(&vec);
            }
            _ => {}
        }
    }
    out
}
