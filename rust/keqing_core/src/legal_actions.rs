use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::counts::TILE_COUNT;
use crate::hora_truth::evaluate_hora_truth_from_prepared;
use crate::shanten_table::{calc_shanten_all, ensure_init};

pub(crate) fn normalize_tile(tile: &str) -> String {
    match tile {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

pub(crate) fn tile_to_34(tile: &str) -> Option<usize> {
    let normalized = normalize_tile(tile);
    let mut chars = normalized.chars();
    let first = chars.next()?;
    if let Some(suit) = chars.next() {
        if matches!(suit, 'm' | 'p' | 's') {
            let rank = first.to_digit(10)? as usize;
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return None,
            };
            return Some(base + rank - 1);
        }
    }
    match normalized.as_str() {
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

fn meld_tile_sort_key(tile: &str) -> (u8, u8, u8) {
    let normalized = normalize_tile(tile);
    let suit = normalized.chars().last().unwrap_or('z');
    if matches!(suit, 'm' | 'p' | 's') {
        let suit_idx = match suit {
            'm' => 0,
            'p' => 1,
            's' => 2,
            _ => 3,
        };
        let rank = normalized
            .chars()
            .next()
            .and_then(|ch| ch.to_digit(10))
            .unwrap_or(9) as u8;
        let aka_rank = if tile.ends_with('r') { 0 } else { 1 };
        return (suit_idx, rank, aka_rank);
    }
    let honor_digit = match normalized.as_str() {
        "E" => 1,
        "S" => 2,
        "W" => 3,
        "N" => 4,
        "P" => 5,
        "F" => 6,
        "C" => 7,
        _ => 9,
    };
    let aka_rank = if tile.ends_with('r') { 0 } else { 1 };
    (3, honor_digit, aka_rank)
}

fn scoring_meld_tiles(meld: &Value) -> Vec<String> {
    let mut consumed: Vec<String> = meld
        .get("consumed")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let called_tile = meld
        .get("pai_raw")
        .or_else(|| meld.get("pai"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let meld_type = meld.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(meld_type, "ankan" | "kakan") {
        if let Some(called_tile) = called_tile {
            consumed.push(called_tile);
        }
    }
    consumed.sort_by_key(|tile| meld_tile_sort_key(tile));
    consumed
}

fn hand_list_from_counter(hand: &BTreeMap<String, u8>) -> Vec<String> {
    let mut tiles = Vec::new();
    for (tile, count) in hand {
        for _ in 0..*count {
            tiles.push(tile.clone());
        }
    }
    tiles
}

fn get_actor_list_bool(snapshot: &Value, key: &str, actor: usize) -> bool {
    snapshot
        .get(key)
        .and_then(Value::as_array)
        .and_then(|items| items.get(actor))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn get_optional_tile_from_actor_list(snapshot: &Value, key: &str, actor: usize) -> Option<String> {
    snapshot
        .get(key)
        .and_then(Value::as_array)
        .and_then(|items| items.get(actor))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn hand_from_snapshot(snapshot: &Value) -> Result<BTreeMap<String, u8>, String> {
    let mut hand = BTreeMap::new();
    let tiles = snapshot
        .get("hand")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing hand in state snapshot".to_string())?;
    for tile in tiles.iter().filter_map(Value::as_str) {
        *hand.entry(tile.to_string()).or_insert(0) += 1;
    }
    Ok(hand)
}

fn hand_count_for_normalized(hand: &BTreeMap<String, u8>, tile: &str) -> u8 {
    let normalized = normalize_tile(tile);
    let mut count = *hand.get(&normalized).unwrap_or(&0);
    let aka_tile = format!("{normalized}r");
    if matches!(aka_tile.as_str(), "5mr" | "5pr" | "5sr") {
        count += hand.get(&aka_tile).copied().unwrap_or(0);
    }
    count
}

fn can_pon(hand: &BTreeMap<String, u8>, tile: &str) -> bool {
    let normalized = normalize_tile(tile);
    if normalized.is_empty() {
        return false;
    }
    if normalized.len() == 1 {
        return hand.get(&normalized).copied().unwrap_or(0) >= 2;
    }
    hand_count_for_normalized(hand, &normalized) >= 2
}

fn can_daiminkan(hand: &BTreeMap<String, u8>, tile: &str) -> bool {
    let normalized = normalize_tile(tile);
    if normalized.is_empty() {
        return false;
    }
    if normalized.len() == 1 {
        return hand.get(&normalized).copied().unwrap_or(0) >= 3;
    }
    hand_count_for_normalized(hand, &normalized) >= 3
}

fn hand_has_tile(hand: &BTreeMap<String, u8>, tile: &str) -> bool {
    if hand.get(tile).copied().unwrap_or(0) >= 1 {
        return true;
    }
    let normalized = normalize_tile(tile);
    if normalized.len() == 2 && normalized.starts_with('5') {
        let aka = format!("{normalized}r");
        return hand.get(&aka).copied().unwrap_or(0) >= 1;
    }
    false
}

fn pick_chi_tile(hand: &BTreeMap<String, u8>, tile: &str) -> String {
    let normalized = normalize_tile(tile);
    if normalized.len() == 2 && normalized.starts_with('5') {
        let aka = format!("{normalized}r");
        if matches!(aka.as_str(), "5mr" | "5pr" | "5sr")
            && hand.get(&aka).copied().unwrap_or(0) >= 1
        {
            return aka;
        }
    }
    normalized
}

fn pick_consumed(hand: &BTreeMap<String, u8>, normalized: &str, n: usize) -> Vec<String> {
    let normalized = normalize_tile(normalized);
    let mut result = Vec::new();
    if normalized.len() == 2 && normalized.starts_with('5') {
        let aka = format!("{normalized}r");
        if matches!(aka.as_str(), "5mr" | "5pr" | "5sr") {
            let aka_count = hand.get(&aka).copied().unwrap_or(0) as usize;
            for _ in 0..aka_count.min(n) {
                result.push(aka.clone());
            }
        }
    }
    while result.len() < n {
        result.push(normalized.clone());
    }
    result
}

fn chi_patterns(tile: &str) -> Vec<[String; 2]> {
    let tile = normalize_tile(tile);
    if tile.len() != 2 {
        return Vec::new();
    }
    let mut chars = tile.chars();
    let Some(first) = chars.next() else {
        return Vec::new();
    };
    let Some(suit) = chars.next() else {
        return Vec::new();
    };
    if !matches!(suit, 'm' | 'p' | 's') {
        return Vec::new();
    }
    let Some(num) = first.to_digit(10).map(|v| v as i32) else {
        return Vec::new();
    };
    let mut patterns = Vec::new();
    for (a, b) in [(num - 2, num - 1), (num - 1, num + 1), (num + 1, num + 2)] {
        if (1..=9).contains(&a) && (1..=9).contains(&b) {
            patterns.push([format!("{a}{suit}"), format!("{b}{suit}")]);
        }
    }
    patterns
}

fn chi_kuikae_forbidden_tiles(pai: &str, consumed: &[Value]) -> BTreeSet<String> {
    let pai_norm = normalize_tile(pai);
    let mut forbidden = BTreeSet::new();
    forbidden.insert(pai_norm.clone());
    if pai_norm.len() != 2 {
        return forbidden;
    }
    let suit = pai_norm.chars().nth(1).unwrap_or('z');
    if !matches!(suit, 'm' | 'p' | 's') || consumed.len() != 2 {
        return forbidden;
    }
    let mut tiles = vec![pai_norm.clone()];
    for value in consumed {
        let Some(tile) = value.as_str() else {
            return forbidden;
        };
        let normalized = normalize_tile(tile);
        if normalized.len() != 2 || normalized.chars().nth(1) != Some(suit) {
            return forbidden;
        }
        tiles.push(normalized);
    }
    let mut ranks = Vec::with_capacity(3);
    for tile in &tiles {
        let Some(rank) = tile.chars().next().and_then(|ch| ch.to_digit(10)) else {
            return forbidden;
        };
        ranks.push(rank as i32);
    }
    ranks.sort_unstable();
    if ranks[1] != ranks[0] + 1 || ranks[2] != ranks[1] + 1 {
        return forbidden;
    }
    let Some(called_rank) = pai_norm.chars().next().and_then(|ch| ch.to_digit(10)) else {
        return forbidden;
    };
    let called_rank = called_rank as i32;
    if called_rank == ranks[0] {
        let outside_rank = ranks[2] + 1;
        if outside_rank <= 9 {
            forbidden.insert(format!("{outside_rank}{suit}"));
        }
    } else if called_rank == ranks[2] {
        let outside_rank = ranks[0] - 1;
        if outside_rank >= 1 {
            forbidden.insert(format!("{outside_rank}{suit}"));
        }
    }
    forbidden
}

fn post_open_meld_forbidden_discard_tiles(
    state_snapshot: &Value,
    actor: usize,
    melds: &[Value],
) -> BTreeSet<String> {
    let actor_to_move = state_snapshot
        .get("actor_to_move")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    if actor_to_move != Some(actor) {
        return BTreeSet::new();
    }
    if state_snapshot
        .get("last_discard")
        .filter(|value| !value.is_null())
        .is_some()
        || state_snapshot
            .get("last_kakan")
            .filter(|value| !value.is_null())
            .is_some()
        || get_optional_tile_from_actor_list(state_snapshot, "last_tsumo", actor).is_some()
        || get_optional_tile_from_actor_list(state_snapshot, "last_tsumo_raw", actor).is_some()
    {
        return BTreeSet::new();
    }
    let Some(last_meld) = melds.last() else {
        return BTreeSet::new();
    };
    let Some(meld_type) = last_meld.get("type").and_then(Value::as_str) else {
        return BTreeSet::new();
    };
    if !matches!(meld_type, "chi" | "pon") {
        return BTreeSet::new();
    }
    let Some(pai) = last_meld
        .get("pai_raw")
        .or_else(|| last_meld.get("pai"))
        .and_then(Value::as_str)
    else {
        return BTreeSet::new();
    };
    if meld_type == "chi" {
        let consumed = last_meld
            .get("consumed")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        return chi_kuikae_forbidden_tiles(pai, consumed);
    }
    BTreeSet::from([normalize_tile(pai)])
}

fn ankan_candidates(hand: &BTreeMap<String, u8>) -> Vec<(String, Vec<String>)> {
    let mut result = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for tile in hand.keys() {
        let normalized = normalize_tile(tile);
        if !seen.insert(normalized.clone()) {
            continue;
        }
        let consumed = pick_consumed(hand, &normalized, 4);
        let mut valid = true;
        let mut needed = BTreeMap::<String, u8>::new();
        for tile in &consumed {
            *needed.entry(tile.clone()).or_insert(0) += 1;
        }
        for (tile, count) in needed {
            if hand.get(&tile).copied().unwrap_or(0) < count {
                valid = false;
                break;
            }
        }
        if valid && consumed.len() == 4 {
            result.push((consumed[0].clone(), consumed));
        }
    }
    result
}

pub(crate) fn counts34_from_hand(hand: &BTreeMap<String, u8>) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for (tile, count) in hand {
        if let Some(tile34) = tile_to_34(tile) {
            counts[tile34] = counts[tile34].saturating_add(*count);
        }
    }
    counts
}

fn counts34_from_snapshot_hand_tiles(
    state_snapshot: &Value,
    actor: usize,
    pai: &str,
    is_tsumo: bool,
) -> Result<[u8; TILE_COUNT], String> {
    let hand_tiles = state_snapshot
        .get("hand")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing hand in state snapshot".to_string())?;
    let mut counts = [0u8; TILE_COUNT];
    for tile in hand_tiles.iter().filter_map(Value::as_str) {
        if let Some(tile34) = tile_to_34(tile) {
            counts[tile34] = counts[tile34].saturating_add(1);
        }
    }
    if !is_tsumo {
        let Some(tile34) = tile_to_34(pai) else {
            return Err(format!("invalid hora tile: {pai}"));
        };
        counts[tile34] = counts[tile34].saturating_add(1);
    } else {
        let last_tsumo = get_optional_tile_from_actor_list(state_snapshot, "last_tsumo", actor);
        let last_tsumo_raw =
            get_optional_tile_from_actor_list(state_snapshot, "last_tsumo_raw", actor);
        let expected = last_tsumo_raw
            .as_deref()
            .or(last_tsumo.as_deref())
            .unwrap_or(pai);
        if normalize_tile(expected) != normalize_tile(pai) {
            return Err("hora tile does not match current tsumo tile".to_string());
        }
    }
    Ok(counts)
}

fn counts34_from_hand_list(hand: &[String]) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for tile in hand {
        if let Some(tile34) = tile_to_34(tile) {
            counts[tile34] = counts[tile34].saturating_add(1);
        }
    }
    counts
}

fn is_suited_sequence_start_for_waits(tile34: usize) -> bool {
    tile34 < 27 && (tile34 % 9) <= 6
}

fn is_complete_regular_counts_for_waits(
    counts: &mut [u8; TILE_COUNT],
    melds_needed: u8,
    need_pair: bool,
) -> bool {
    let first = counts.iter().position(|&cnt| cnt > 0);
    let Some(first) = first else {
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
        && is_suited_sequence_start_for_waits(first)
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

fn find_regular_waits_tiles(counts34: &[u8; TILE_COUNT]) -> [bool; TILE_COUNT] {
    let mut waits = [false; TILE_COUNT];
    let tile_count: usize = counts34.iter().map(|&count| count as usize).sum();
    if tile_count % 3 != 1 {
        return waits;
    }
    let melds_needed = (((tile_count + 1) - 2) / 3) as u8;
    for tile34 in 0..TILE_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        waits[tile34] = is_complete_regular_counts_for_waits(&mut work, melds_needed, true);
    }
    waits
}

fn find_special_waits_tiles(counts34: &[u8; TILE_COUNT]) -> (i32, [bool; TILE_COUNT]) {
    let chiitoi = crate::shanten_table::calc_shanten_chitoi(counts34) as i32;
    let kokushi = crate::shanten_table::calc_shanten_kokushi(counts34) as i32;
    let special_shanten = chiitoi.min(kokushi);
    let mut waits = [false; TILE_COUNT];
    if special_shanten != 0 {
        return (special_shanten, waits);
    }
    for tile34 in 0..TILE_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        if (chiitoi == 0 && crate::shanten_table::calc_shanten_chitoi(&work) == -1)
            || (kokushi == 0 && crate::shanten_table::calc_shanten_kokushi(&work) == -1)
        {
            waits[tile34] = true;
        }
    }
    (special_shanten, waits)
}

pub(crate) fn calc_shanten_waits_like_python(
    counts34: &[u8; TILE_COUNT],
    has_melds: bool,
) -> (i8, u8, [bool; TILE_COUNT]) {
    let tile_count: u8 = counts34.iter().sum();
    let regular_shanten = calc_shanten_all(counts34, tile_count / 3);
    let mut shanten = regular_shanten;
    let mut waits = if regular_shanten == 0 && tile_count % 3 == 1 {
        find_regular_waits_tiles(counts34)
    } else {
        [false; TILE_COUNT]
    };
    if !has_melds {
        let (special_shanten, special_waits) = find_special_waits_tiles(counts34);
        if special_shanten < shanten as i32 {
            shanten = special_shanten as i8;
            waits = if special_shanten == 0 {
                special_waits
            } else {
                [false; TILE_COUNT]
            };
        } else if special_shanten == shanten as i32 && shanten == 0 {
            for tile34 in 0..TILE_COUNT {
                waits[tile34] |= special_waits[tile34];
            }
        }
    }
    let waits_count = waits.iter().filter(|flag| **flag).count() as u8;
    (shanten, waits_count, waits)
}

fn remove_tile_once(hand: &mut Vec<String>, tile: &str) -> bool {
    if let Some(index) = hand.iter().position(|existing| existing == tile) {
        hand.remove(index);
        return true;
    }
    let normalized = normalize_tile(tile);
    if normalized != tile {
        if let Some(index) = hand.iter().position(|existing| existing == &normalized) {
            hand.remove(index);
            return true;
        }
    }
    false
}

fn ankan_allowed_after_reach(
    hand: &BTreeMap<String, u8>,
    melds: &[Value],
    consumed: &[String],
    last_tsumo: Option<&str>,
    last_tsumo_raw: Option<&str>,
) -> bool {
    let mut full_hand = hand_list_from_counter(hand);
    if last_tsumo.is_some() && full_hand.len() % 3 == 1 {
        full_hand.push(last_tsumo_raw.unwrap_or(last_tsumo.unwrap()).to_string());
    }

    let mut before_hand = full_hand.clone();
    if let Some(draw_tile) = last_tsumo_raw.or(last_tsumo) {
        remove_tile_once(&mut before_hand, draw_tile);
    }
    let before_counts = counts34_from_hand_list(&before_hand);
    let (before_shanten, _, before_waits) =
        calc_shanten_waits_like_python(&before_counts, !melds.is_empty());
    if before_shanten != 0 {
        return false;
    }

    let mut after_hand = full_hand;
    for tile in consumed {
        if !remove_tile_once(&mut after_hand, tile) {
            return false;
        }
    }
    let after_counts = counts34_from_hand_list(&after_hand);
    let (after_shanten, _, after_waits) = calc_shanten_waits_like_python(&after_counts, true);
    after_shanten == 0 && before_waits == after_waits
}

pub(crate) fn reach_discard_candidates(
    hand: &BTreeMap<String, u8>,
    last_tsumo: Option<&str>,
    last_tsumo_raw: Option<&str>,
) -> Vec<(String, bool)> {
    ensure_init();
    let mut candidates = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut counts34 = counts34_from_hand(hand);

    for tile in hand.keys() {
        let tsumogiri = last_tsumo == Some(tile.as_str());
        let pai_out = if tsumogiri {
            last_tsumo_raw.unwrap_or(tile.as_str()).to_string()
        } else {
            tile.clone()
        };
        let Some(tile34) = tile_to_34(tile) else {
            continue;
        };
        if counts34[tile34] == 0 {
            continue;
        }
        counts34[tile34] -= 1;
        let total_tiles: u8 = counts34.iter().sum();
        if calc_shanten_all(&counts34, total_tiles / 3) == 0 {
            let key = (pai_out.clone(), tsumogiri);
            if seen.insert(key.clone()) {
                candidates.push(key);
            }
        }
        counts34[tile34] += 1;
    }

    candidates
}

fn can_declare_reach(
    hand: &BTreeMap<String, u8>,
    melds: &[Value],
    reached: bool,
    last_tsumo: Option<&str>,
    last_tsumo_raw: Option<&str>,
) -> bool {
    if reached {
        return false;
    }
    if melds
        .iter()
        .any(|meld| meld.get("type").and_then(Value::as_str) != Some("ankan"))
    {
        return false;
    }
    let concealed_kan_count = melds
        .iter()
        .filter(|meld| meld.get("type").and_then(Value::as_str) == Some("ankan"))
        .count();
    let hand_len: usize = hand.values().map(|count| *count as usize).sum();
    if hand_len + 3 * concealed_kan_count != 14 {
        return false;
    }
    !reach_discard_candidates(hand, last_tsumo, last_tsumo_raw).is_empty()
}

fn action_none() -> Value {
    json!({"type": "none"})
}

fn structural_reaction_specs(
    hand: &BTreeMap<String, u8>,
    actor: usize,
    reached: bool,
    discarder: usize,
    tile_raw: &str,
    calls_allowed: bool,
) -> Vec<Value> {
    let tile_norm = normalize_tile(tile_raw);
    let mut legal = Vec::new();
    if !reached && calls_allowed {
        if can_pon(hand, tile_raw) {
            legal.push(json!({
                "type": "pon",
                "actor": actor,
                "target": discarder,
                "pai": tile_raw,
                "consumed": pick_consumed(hand, &tile_norm, 2),
            }));
        }
        if can_daiminkan(hand, tile_raw) {
            legal.push(json!({
                "type": "daiminkan",
                "actor": actor,
                "target": discarder,
                "pai": tile_raw,
                "consumed": pick_consumed(hand, &tile_norm, 3),
            }));
        }
        let next_player = (discarder + 1) % 4;
        if actor == next_player {
            for pattern in chi_patterns(tile_raw) {
                if hand_has_tile(hand, &pattern[0]) && hand_has_tile(hand, &pattern[1]) {
                    legal.push(json!({
                        "type": "chi",
                        "actor": actor,
                        "target": discarder,
                        "pai": tile_raw,
                        "consumed": [
                            pick_chi_tile(hand, &pattern[0]),
                            pick_chi_tile(hand, &pattern[1]),
                        ],
                    }));
                }
            }
        }
    }
    legal.push(action_none());
    legal
}

pub fn enumerate_legal_action_specs_structural(
    state_snapshot: &Value,
    actor: usize,
) -> Result<Vec<Value>, String> {
    let hand = hand_from_snapshot(state_snapshot)?;
    let last_discard = state_snapshot
        .get("last_discard")
        .filter(|value| !value.is_null());
    let last_kakan = state_snapshot
        .get("last_kakan")
        .filter(|value| !value.is_null());
    let actor_to_move = state_snapshot
        .get("actor_to_move")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    let last_tsumo = get_optional_tile_from_actor_list(state_snapshot, "last_tsumo", actor);
    let last_tsumo_raw = get_optional_tile_from_actor_list(state_snapshot, "last_tsumo_raw", actor);
    let reached = get_actor_list_bool(state_snapshot, "reached", actor);

    if let Some(last_kakan) = last_kakan {
        if last_kakan
            .get("actor")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            != Some(actor)
        {
            return Ok(vec![action_none()]);
        }
    }

    if let Some(last_discard) = last_discard {
        let discarder = last_discard
            .get("actor")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| "invalid last_discard.actor".to_string())?;
        if discarder != actor {
            let tile_raw = last_discard
                .get("pai_raw")
                .or_else(|| last_discard.get("pai"))
                .and_then(Value::as_str)
                .ok_or_else(|| "invalid last_discard.pai".to_string())?;
            let calls_allowed = state_snapshot
                .get("remaining_wall")
                .and_then(Value::as_i64)
                .map(|remaining_wall| remaining_wall > 0)
                .unwrap_or(true);
            return Ok(structural_reaction_specs(
                &hand,
                actor,
                reached,
                discarder,
                tile_raw,
                calls_allowed,
            ));
        }
    }

    if actor_to_move == Some(actor) {
        let kan_allowed = state_snapshot
            .get("remaining_wall")
            .and_then(Value::as_i64)
            .map(|remaining_wall| remaining_wall > 0)
            .unwrap_or(true);
        let pending_reach = get_actor_list_bool(state_snapshot, "pending_reach", actor);
        let melds = state_snapshot
            .get("melds")
            .and_then(Value::as_array)
            .and_then(|all| all.get(actor))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let forbidden_discards =
            post_open_meld_forbidden_discard_tiles(state_snapshot, actor, &melds);
        if reached {
            let mut legal = Vec::new();
            if let Some(pai) = last_tsumo_raw.as_deref().or(last_tsumo.as_deref()) {
                legal.push(json!({
                    "type": "dahai",
                    "actor": actor,
                    "pai": pai,
                    "tsumogiri": true,
                }));
            }
            if kan_allowed {
                for (pai, consumed) in ankan_candidates(&hand) {
                    if ankan_allowed_after_reach(
                        &hand,
                        &melds,
                        &consumed,
                        last_tsumo.as_deref(),
                        last_tsumo_raw.as_deref(),
                    ) {
                        legal.push(json!({
                            "type": "ankan",
                            "actor": actor,
                            "pai": pai,
                            "consumed": consumed,
                        }));
                    }
                }
            }
            return Ok(legal);
        }
        if pending_reach {
            let candidates =
                reach_discard_candidates(&hand, last_tsumo.as_deref(), last_tsumo_raw.as_deref());
            let mut legal = Vec::new();
            if !candidates.is_empty() {
                for (pai_out, tsumogiri) in candidates {
                    legal.push(json!({
                        "type": "dahai",
                        "actor": actor,
                        "pai": pai_out,
                        "tsumogiri": tsumogiri,
                    }));
                }
            } else {
                for tile in hand.keys() {
                    let tsumogiri = last_tsumo.as_deref() == Some(tile.as_str());
                    let pai_out = if tsumogiri {
                        last_tsumo_raw
                            .as_deref()
                            .unwrap_or(tile.as_str())
                            .to_string()
                    } else {
                        tile.clone()
                    };
                    legal.push(json!({
                        "type": "dahai",
                        "actor": actor,
                        "pai": pai_out,
                        "tsumogiri": tsumogiri,
                    }));
                }
            }
            return Ok(legal);
        }

        let mut legal = Vec::new();
        for tile in hand.keys() {
            let tsumogiri = last_tsumo.as_deref() == Some(tile.as_str());
            let pai_out = if tsumogiri {
                last_tsumo_raw
                    .as_deref()
                    .unwrap_or(tile.as_str())
                    .to_string()
            } else {
                tile.clone()
            };
            if forbidden_discards.contains(&normalize_tile(&pai_out)) {
                continue;
            }
            legal.push(json!({
                "type": "dahai",
                "actor": actor,
                "pai": pai_out,
                "tsumogiri": tsumogiri,
            }));
        }
        if can_declare_reach(
            &hand,
            &melds,
            reached,
            last_tsumo.as_deref(),
            last_tsumo_raw.as_deref(),
        ) {
            legal.push(json!({
                "type": "reach",
                "actor": actor,
            }));
        }
        if kan_allowed {
            for (pai, consumed) in ankan_candidates(&hand) {
                legal.push(json!({
                    "type": "ankan",
                    "actor": actor,
                    "pai": pai,
                    "consumed": consumed,
                }));
            }
            for meld in melds {
                if meld.get("type").and_then(Value::as_str) != Some("pon") {
                    continue;
                }
                let Some(meld_pai) = meld.get("pai").and_then(Value::as_str) else {
                    continue;
                };
                let meld_norm = normalize_tile(meld_pai);
                if hand_has_tile(&hand, &meld_norm) {
                    let mut consumed = meld
                        .get("consumed")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    consumed.push(Value::String(meld_pai.to_string()));
                    legal.push(json!({
                        "type": "kakan",
                        "actor": actor,
                        "pai": pick_chi_tile(&hand, &meld_norm),
                        "consumed": consumed,
                    }));
                }
            }
        }
        return Ok(legal);
    }

    if last_discard.is_none() {
        return Ok(vec![action_none()]);
    }

    let last_discard = last_discard.unwrap();
    let discarder = last_discard
        .get("actor")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| "invalid last_discard.actor".to_string())?;
    if discarder == actor {
        return Ok(vec![action_none()]);
    }
    let tile_raw = last_discard
        .get("pai_raw")
        .or_else(|| last_discard.get("pai"))
        .and_then(Value::as_str)
        .ok_or_else(|| "invalid last_discard.pai".to_string())?;
    let calls_allowed = state_snapshot
        .get("remaining_wall")
        .and_then(Value::as_i64)
        .map(|remaining_wall| remaining_wall > 0)
        .unwrap_or(true);
    Ok(structural_reaction_specs(
        &hand,
        actor,
        reached,
        discarder,
        tile_raw,
        calls_allowed,
    ))
}

pub fn enumerate_hora_candidates(
    state_snapshot: &Value,
    actor: usize,
) -> Result<Vec<Value>, String> {
    let last_discard = state_snapshot
        .get("last_discard")
        .filter(|value| !value.is_null());
    let last_kakan = state_snapshot
        .get("last_kakan")
        .filter(|value| !value.is_null());
    let actor_to_move = state_snapshot
        .get("actor_to_move")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    let last_tsumo = get_optional_tile_from_actor_list(state_snapshot, "last_tsumo", actor);
    let last_tsumo_raw = get_optional_tile_from_actor_list(state_snapshot, "last_tsumo_raw", actor);
    let actor_furiten = get_actor_list_bool(state_snapshot, "furiten", actor);
    let hora_is_haitei = state_snapshot
        .get("_hora_is_haitei")
        .cloned()
        .unwrap_or(Value::Null);
    let hora_is_houtei = state_snapshot
        .get("_hora_is_houtei")
        .cloned()
        .unwrap_or(Value::Null);
    let hora_is_rinshan = state_snapshot
        .get("_hora_is_rinshan")
        .cloned()
        .unwrap_or(Value::Null);
    let hora_is_chankan = state_snapshot
        .get("_hora_is_chankan")
        .cloned()
        .unwrap_or(Value::Null);

    if let Some(last_kakan) = last_kakan {
        if last_kakan
            .get("actor")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            != Some(actor)
            && !actor_furiten
        {
            let target = last_kakan
                .get("actor")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| "invalid last_kakan.actor".to_string())?;
            let pai = last_kakan
                .get("pai_raw")
                .or_else(|| last_kakan.get("pai"))
                .and_then(Value::as_str)
                .ok_or_else(|| "invalid last_kakan.pai".to_string())?;
            let is_chankan = if hora_is_chankan.is_null() {
                Value::Bool(true)
            } else {
                hora_is_chankan
            };
            return Ok(vec![json!({
                "target": target,
                "pai": pai,
                "is_tsumo": false,
                "is_chankan": is_chankan,
            })]);
        }
        return Ok(Vec::new());
    }

    if let Some(last_discard) = last_discard {
        let discarder = last_discard
            .get("actor")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| "invalid last_discard.actor".to_string())?;
        if discarder != actor && !actor_furiten {
            let pai = last_discard
                .get("pai_raw")
                .or_else(|| last_discard.get("pai"))
                .and_then(Value::as_str)
                .ok_or_else(|| "invalid last_discard.pai".to_string())?;
            return Ok(vec![json!({
                "target": discarder,
                "pai": pai,
                "is_tsumo": false,
                "is_houtei": hora_is_houtei,
            })]);
        }
        return Ok(Vec::new());
    }

    if actor_to_move == Some(actor) {
        if let Some(pai) = last_tsumo_raw.as_deref().or(last_tsumo.as_deref()) {
            return Ok(vec![json!({
                "target": actor,
                "pai": pai,
                "is_tsumo": true,
                "is_rinshan": hora_is_rinshan,
                "is_haitei": hora_is_haitei,
            })]);
        }
    }

    Ok(Vec::new())
}

fn candidate_flag_bool(candidate: &Value, key: &str) -> Option<bool> {
    candidate.get(key).and_then(Value::as_bool)
}

pub(crate) fn can_hora_from_snapshot_candidate(
    state_snapshot: &Value,
    actor: usize,
    target: usize,
    pai: &str,
    is_tsumo: bool,
    candidate: &Value,
) -> Result<bool, String> {
    if !can_hora_shape_from_snapshot(state_snapshot, actor, pai, is_tsumo)? {
        return Ok(false);
    }
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
        Err(message)
            if message.contains("failed to allocate tile id")
                || message.contains("failed to find win tile id") =>
        {
            Ok(false)
        }
        Err(message) => Err(format!(
            "failed to evaluate hora truth for actor={actor} target={target} pai={pai}: {message}"
        )),
    }
}

pub fn enumerate_public_legal_action_specs(
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

pub fn can_hora_shape_from_snapshot(
    state_snapshot: &Value,
    actor: usize,
    pai: &str,
    is_tsumo: bool,
) -> Result<bool, String> {
    ensure_init();
    let counts34 = counts34_from_snapshot_hand_tiles(state_snapshot, actor, pai, is_tsumo)?;
    let meld_count = state_snapshot
        .get("melds")
        .and_then(Value::as_array)
        .and_then(|all| all.get(actor))
        .and_then(Value::as_array)
        .map(|melds| melds.len())
        .unwrap_or(0);
    let tile_count: u8 = counts34.iter().sum();
    if usize::from(tile_count) + 3 * meld_count != 14 {
        return Ok(false);
    }
    let shanten = calc_shanten_all(&counts34, tile_count / 3);
    Ok(shanten == -1)
}

pub fn prepare_hora_evaluation_from_snapshot(
    state_snapshot: &Value,
    actor: usize,
    pai: &str,
    is_tsumo: bool,
    is_chankan: bool,
    is_rinshan: Option<bool>,
    is_haitei: Option<bool>,
    is_houtei: Option<bool>,
) -> Result<Value, String> {
    let hand_tiles = state_snapshot
        .get("hand")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing hand in state snapshot".to_string())?
        .iter()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let melds = state_snapshot
        .get("melds")
        .and_then(Value::as_array)
        .and_then(|all| all.get(actor))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let reached = get_actor_list_bool(state_snapshot, "reached", actor);
    let ippatsu = get_actor_list_bool(state_snapshot, "ippatsu_eligible", actor);
    let rinshan_tsumo = get_actor_list_bool(state_snapshot, "rinshan_tsumo", actor);
    let remaining_wall = state_snapshot.get("remaining_wall").and_then(Value::as_i64);

    let resolved_is_rinshan = is_rinshan.unwrap_or(rinshan_tsumo);
    let resolved_is_haitei = is_haitei.unwrap_or(is_tsumo && remaining_wall == Some(0));
    let resolved_is_houtei = is_houtei.unwrap_or(!is_tsumo && remaining_wall == Some(0));
    let active_ura_markers = if reached {
        state_snapshot
            .get("ura_dora_markers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let target = if is_tsumo {
        actor
    } else if is_chankan {
        state_snapshot
            .get("last_kakan")
            .and_then(Value::as_object)
            .and_then(|payload| payload.get("actor"))
            .and_then(Value::as_u64)
            .ok_or_else(|| "missing last_kakan.actor for chankan".to_string())
            .and_then(|raw| {
                usize::try_from(raw).map_err(|_| format!("last_kakan.actor out of range: {raw}"))
            })?
    } else {
        state_snapshot
            .get("last_discard")
            .and_then(Value::as_object)
            .and_then(|payload| payload.get("actor"))
            .and_then(Value::as_u64)
            .ok_or_else(|| "missing last_discard.actor for ron".to_string())
            .and_then(|raw| {
                usize::try_from(raw).map_err(|_| format!("last_discard.actor out of range: {raw}"))
            })?
    };

    let mut closed_tiles = hand_tiles;
    if !is_tsumo {
        closed_tiles.push(pai.to_string());
    }
    let meld_payload = melds
        .iter()
        .map(|meld| {
            let meld_type = meld.get("type").and_then(Value::as_str).unwrap_or("");
            json!({
                "type": meld_type,
                "opened": matches!(meld_type, "chi" | "pon" | "daiminkan" | "kakan"),
                "tiles": scoring_meld_tiles(meld),
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "closed_tiles": closed_tiles,
        "melds": meld_payload,
        "bakaze": state_snapshot.get("bakaze").cloned().unwrap_or(Value::String("E".to_string())),
        "honba": state_snapshot.get("honba").cloned().unwrap_or(Value::from(0)),
        "kyotaku": state_snapshot.get("kyotaku").cloned().unwrap_or(Value::from(0)),
        "oya": state_snapshot.get("oya").cloned().unwrap_or(Value::from(0)),
        "dora_markers": state_snapshot.get("dora_markers").cloned().unwrap_or(Value::Array(Vec::new())),
        "active_ura_markers": active_ura_markers,
        "reached": reached,
        "ippatsu_eligible": ippatsu,
        "is_tsumo": is_tsumo,
        "is_chankan": is_chankan,
        "resolved_is_rinshan": resolved_is_rinshan,
        "resolved_is_haitei": resolved_is_haitei,
        "resolved_is_houtei": resolved_is_houtei,
        "pai": pai,
        "actor": actor,
        "target": target,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::enumerate_legal_action_specs_structural;

    fn discard_pais(actions: &[serde_json::Value]) -> Vec<String> {
        actions
            .iter()
            .filter(|action| {
                action.get("type").and_then(serde_json::Value::as_str) == Some("dahai")
            })
            .filter_map(|action| action.get("pai").and_then(serde_json::Value::as_str))
            .map(ToOwned::to_owned)
            .collect()
    }

    #[test]
    fn post_chi_low_called_tile_filters_suji_kuikae() {
        let snapshot = json!({
            "actor": 2,
            "hand": ["4p", "5m"],
            "melds": [
                [],
                [],
                [{"type": "chi", "pai": "1p", "pai_raw": "1p", "consumed": ["2p", "3p"], "target": 1}],
                []
            ],
            "reached": [false, false, false, false],
            "pending_reach": [false, false, false, false],
            "actor_to_move": 2,
            "last_discard": null,
            "last_kakan": null,
            "last_tsumo": [null, null, null, null],
            "last_tsumo_raw": [null, null, null, null]
        });

        let actions = enumerate_legal_action_specs_structural(&snapshot, 2).unwrap();

        assert_eq!(discard_pais(&actions), vec!["5m"]);
    }

    #[test]
    fn post_chi_middle_called_tile_keeps_outside_suji() {
        let snapshot = json!({
            "actor": 1,
            "hand": ["7m", "9m", "5s"],
            "melds": [
                [],
                [{"type": "chi", "pai": "7m", "pai_raw": "7m", "consumed": ["6m", "8m"], "target": 0}],
                [],
                []
            ],
            "reached": [false, false, false, false],
            "pending_reach": [false, false, false, false],
            "actor_to_move": 1,
            "last_discard": null,
            "last_kakan": null,
            "last_tsumo": [null, null, null, null],
            "last_tsumo_raw": [null, null, null, null]
        });

        let actions = enumerate_legal_action_specs_structural(&snapshot, 1).unwrap();

        assert_eq!(discard_pais(&actions), vec!["5s", "9m"]);
    }

    #[test]
    fn houtei_response_blocks_calls() {
        let snapshot = json!({
            "actor": 2,
            "hand": ["7s", "8s", "1m", "2m", "3m", "4p", "5p"],
            "melds": [[], [], [], []],
            "reached": [false, false, false, false],
            "pending_reach": [false, false, false, false],
            "actor_to_move": 2,
            "last_discard": {"actor": 1, "pai": "6s", "pai_raw": "6s"},
            "last_kakan": null,
            "last_tsumo": [null, null, null, null],
            "last_tsumo_raw": [null, null, null, null],
            "remaining_wall": 0
        });

        let actions = enumerate_legal_action_specs_structural(&snapshot, 2).unwrap();

        assert_eq!(actions, vec![json!({"type": "none"})]);
    }

    #[test]
    fn haitei_self_turn_blocks_ankan_and_kakan() {
        let snapshot = json!({
            "actor": 0,
            "hand": ["1m", "2m", "3m", "4m", "5p", "8s", "9s", "9s", "9s", "9s"],
            "melds": [
                [{"type": "pon", "pai": "5p", "pai_raw": "5p", "consumed": ["5p", "5p"], "target": 1}],
                [],
                [],
                []
            ],
            "reached": [false, false, false, false],
            "pending_reach": [false, false, false, false],
            "actor_to_move": 0,
            "last_discard": null,
            "last_kakan": null,
            "last_tsumo": ["9s", null, null, null],
            "last_tsumo_raw": ["9s", null, null, null],
            "remaining_wall": 0
        });

        let actions = enumerate_legal_action_specs_structural(&snapshot, 0).unwrap();

        assert!(actions.iter().all(|action| action["type"] != "ankan"));
        assert!(actions.iter().all(|action| action["type"] != "kakan"));
        assert!(actions.iter().any(|action| action["type"] == "dahai"));
    }
}
