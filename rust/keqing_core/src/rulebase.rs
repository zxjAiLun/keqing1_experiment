use serde_json::{json, Value};

use crate::counts::TILE_COUNT;
use crate::replay_export_core::{normalize_tile_repr, strip_aka, tile34_from_pai};
use crate::shanten_table::{calc_shanten_all, ensure_init};

const EAST: usize = 27;
const WHITE: usize = 31;
const GREEN: usize = 32;
const RED: usize = 33;

#[derive(Debug, Clone)]
struct MeldView {
    kind: String,
    pai: Option<String>,
    consumed: Vec<String>,
}

#[derive(Debug, Clone)]
struct RuleContext {
    actor: usize,
    hand_tiles: Vec<String>,
    self_melds: Vec<MeldView>,
    opponent_discards: Vec<Vec<usize>>,
    visible_counts34: [u8; TILE_COUNT],
    current_shanten: i8,
    yakuhai_tile_types: [bool; TILE_COUNT],
    self_reached: bool,
    last_tsumo: Option<String>,
}

#[derive(Debug, Clone)]
struct DiscardEvaluation {
    action: Value,
    shanten: i8,
    ukeire: u8,
    safe_against_riichi_count: u8,
}

#[derive(Debug, Clone)]
struct CallEvaluation {
    action: Value,
    shanten: i8,
    ukeire: u8,
}

pub fn choose_rulebase_action(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Result<Option<Value>, String> {
    let scored = score_rulebase_actions(snapshot, actor, legal_actions)?;
    if scored.is_empty() {
        return Ok(None);
    }
    let best_index = best_scored_action_index(&scored).unwrap_or(0);
    Ok(legal_actions.get(best_index).cloned())
}

pub fn score_rulebase_actions(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Result<Vec<Value>, String> {
    if legal_actions.is_empty() {
        return Ok(Vec::new());
    }

    let chosen = choose_rulebase_action_impl(snapshot, actor, legal_actions)?;
    let chosen_index = chosen
        .as_ref()
        .and_then(|chosen_action| {
            legal_actions
                .iter()
                .position(|action| action == chosen_action)
        })
        .unwrap_or(0);

    Ok(legal_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let action_type_name = action_type(action).unwrap_or_else(|| "unknown".to_string());
            let selected = index == chosen_index;
            let raw_score = if selected {
                0.0
            } else {
                base_rulebase_penalty(&action_type_name) - (index as f64 * 0.001)
            };
            let priority = if selected {
                1000
            } else {
                base_rulebase_priority(&action_type_name)
            };
            json!({
                "action_index": index,
                "raw_score": raw_score,
                "priority": priority,
                "tie_break_rank": -(index as i64),
                "components": {
                    "selected_by_rulebase": selected,
                    "action_type": action_type_name,
                    "legal_order": index,
                }
            })
        })
        .collect())
}

fn choose_rulebase_action_impl(
    snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
) -> Result<Option<Value>, String> {
    if legal_actions.is_empty() {
        return Ok(None);
    }

    if let Some(action) = legal_actions
        .iter()
        .find(|action| matches!(action_type(action).as_deref(), Some("hora")))
    {
        return Ok(Some(action.clone()));
    }

    if let Some(action) = legal_actions
        .iter()
        .find(|action| matches!(action_type(action).as_deref(), Some("reach")))
    {
        return Ok(Some(action.clone()));
    }

    let context = RuleContext::new(snapshot, actor)?;

    if context.self_reached {
        if let Some(action) = choose_post_riichi_discard(&context, legal_actions) {
            return Ok(Some(action));
        }
    }

    if context.any_opponent_riichi(snapshot) && context.current_shanten >= 2 {
        if let Some(action) = choose_betaori_discard(&context, snapshot, legal_actions) {
            return Ok(Some(action));
        }
    }

    if let Some(action) = choose_kan_action(&context, legal_actions) {
        return Ok(Some(action));
    }

    if let Some(action) = choose_yakuhai_pon_action(&context, legal_actions) {
        return Ok(Some(action));
    }

    if context.has_secured_yakuhai() {
        if let Some(action) = choose_improving_open_meld_action(&context, legal_actions) {
            return Ok(Some(action));
        }

        if let Some(action) = choose_daiminkan_action(&context, legal_actions) {
            return Ok(Some(action));
        }
    }

    Ok(choose_best_discard(&context, snapshot, legal_actions)
        .or_else(|| find_action_by_type(legal_actions, "none"))
        .or_else(|| legal_actions.first().cloned()))
}

fn best_scored_action_index(scored: &[Value]) -> Option<usize> {
    let mut best: Option<(usize, f64, i64, i64)> = None;
    for (index, entry) in scored.iter().enumerate() {
        let raw_score = entry
            .get("raw_score")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        let priority = entry
            .get("priority")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let tie_break_rank = entry
            .get("tie_break_rank")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| -(index as i64));
        let candidate = (index, raw_score, priority, tie_break_rank);
        if best
            .as_ref()
            .is_none_or(|current| compare_scored_tuple(&candidate, current).is_gt())
        {
            best = Some(candidate);
        }
    }
    best.map(|item| item.0)
}

fn compare_scored_tuple(
    lhs: &(usize, f64, i64, i64),
    rhs: &(usize, f64, i64, i64),
) -> std::cmp::Ordering {
    lhs.1
        .partial_cmp(&rhs.1)
        .unwrap_or(std::cmp::Ordering::Less)
        .then_with(|| lhs.2.cmp(&rhs.2))
        .then_with(|| lhs.3.cmp(&rhs.3))
        .then_with(|| rhs.0.cmp(&lhs.0))
}

fn base_rulebase_penalty(action_type_name: &str) -> f64 {
    match action_type_name {
        "hora" => -1.0,
        "reach" => -2.0,
        "dahai" => -3.0,
        "ankan" | "kakan" | "daiminkan" => -4.0,
        "pon" | "chi" => -5.0,
        "none" => -6.0,
        _ => -8.0,
    }
}

fn base_rulebase_priority(action_type_name: &str) -> i64 {
    match action_type_name {
        "hora" => 90,
        "reach" => 80,
        "dahai" => 70,
        "ankan" | "kakan" | "daiminkan" => 60,
        "pon" | "chi" => 50,
        "none" => 10,
        _ => 0,
    }
}

impl RuleContext {
    fn new(snapshot: &Value, actor: usize) -> Result<Self, String> {
        ensure_init();
        let hand_tiles = snapshot
            .get("hand")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(normalize_tile_repr)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let self_melds = snapshot
            .get("melds")
            .and_then(Value::as_array)
            .and_then(|groups| groups.get(actor))
            .and_then(Value::as_array)
            .map(|items| parse_melds(items))
            .unwrap_or_default();
        let opponent_discards = parse_opponent_discards(snapshot);
        let visible_counts34 = collect_visible_counts34(snapshot, actor);
        let hand_counts = counts34_from_tiles(&hand_tiles);
        let current_shanten = shanten_from_counts(&hand_counts);
        let yakuhai_tile_types = yakuhai_tile_types(snapshot, actor);
        let self_reached = snapshot
            .get("reached")
            .and_then(Value::as_array)
            .and_then(|items| items.get(actor))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let last_tsumo = snapshot
            .get("last_tsumo")
            .and_then(Value::as_array)
            .and_then(|items| items.get(actor))
            .and_then(Value::as_str)
            .map(normalize_tile_repr);

        Ok(Self {
            actor,
            hand_tiles,
            self_melds,
            opponent_discards,
            visible_counts34,
            current_shanten,
            yakuhai_tile_types,
            self_reached,
            last_tsumo,
        })
    }

    fn any_opponent_riichi(&self, snapshot: &Value) -> bool {
        snapshot
            .get("reached")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .any(|(player, item)| player != self.actor && item.as_bool().unwrap_or(false))
            })
            .unwrap_or(false)
    }

    fn safe_against_riichi_count(&self, snapshot: &Value, tile34: usize) -> u8 {
        snapshot
            .get("reached")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .filter(|(player, item)| {
                        *player != self.actor && item.as_bool().unwrap_or(false)
                    })
                    .filter(|(player, _)| self.opponent_discards[*player].contains(&tile34))
                    .count() as u8
            })
            .unwrap_or(0)
    }

    fn has_secured_yakuhai(&self) -> bool {
        if self.self_melds.iter().any(|meld| {
            let tile34 = meld
                .pai
                .as_ref()
                .and_then(|tile| tile34_from_pai(tile))
                .map(|idx| idx as usize)
                .or_else(|| {
                    meld.consumed
                        .first()
                        .and_then(|tile| tile34_from_pai(tile))
                        .map(|idx| idx as usize)
                });
            tile34.is_some_and(|idx| {
                self.yakuhai_tile_types[idx]
                    && matches!(meld.kind.as_str(), "pon" | "daiminkan" | "ankan" | "kakan")
            })
        }) {
            return true;
        }

        let counts = counts34_from_tiles(&self.hand_tiles);
        self.yakuhai_tile_types
            .iter()
            .enumerate()
            .any(|(tile34, is_yakuhai)| *is_yakuhai && counts[tile34] >= 3)
    }

    fn is_yakuhai_tile(&self, tile34: usize) -> bool {
        self.yakuhai_tile_types[tile34]
    }
}

fn choose_post_riichi_discard(context: &RuleContext, legal_actions: &[Value]) -> Option<Value> {
    let last_tsumo = context.last_tsumo.as_ref()?;
    legal_actions.iter().find_map(|action| {
        if action_type(action).as_deref() != Some("dahai") {
            return None;
        }
        let pai = action.get("pai").and_then(Value::as_str)?;
        let tsumogiri = action
            .get("tsumogiri")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        (tsumogiri && same_tile_family(pai, last_tsumo)).then(|| action.clone())
    })
}

fn choose_betaori_discard(
    context: &RuleContext,
    snapshot: &Value,
    legal_actions: &[Value],
) -> Option<Value> {
    let candidates = discard_evaluations(context, snapshot, legal_actions)
        .into_iter()
        .filter(|evaluation| evaluation.safe_against_riichi_count > 0)
        .collect::<Vec<_>>();
    choose_best_discard_eval(candidates, true).map(|evaluation| evaluation.action)
}

fn choose_kan_action(context: &RuleContext, legal_actions: &[Value]) -> Option<Value> {
    legal_actions
        .iter()
        .filter(|action| matches!(action_type(action).as_deref(), Some("ankan" | "kakan")))
        .filter_map(|action| {
            let new_hand = hand_after_call_action(&context.hand_tiles, action);
            let new_shanten = shanten_from_tiles(&new_hand);
            (new_shanten <= context.current_shanten).then(|| action.clone())
        })
        .next()
}

fn choose_yakuhai_pon_action(context: &RuleContext, legal_actions: &[Value]) -> Option<Value> {
    let candidates = legal_actions
        .iter()
        .filter(|action| action_type(action).as_deref() == Some("pon"))
        .filter(|action| {
            action
                .get("pai")
                .and_then(Value::as_str)
                .and_then(tile34_from_pai)
                .map(|idx| context.is_yakuhai_tile(idx as usize))
                .unwrap_or(false)
        })
        .filter_map(|action| {
            let new_hand = hand_after_call_action(&context.hand_tiles, action);
            let new_shanten = shanten_from_tiles(&new_hand);
            (new_shanten < context.current_shanten).then(|| CallEvaluation {
                action: action.clone(),
                shanten: new_shanten,
                ukeire: calculate_ukeire(&new_hand, &context.visible_counts34),
            })
        })
        .collect::<Vec<_>>();
    choose_best_call_eval(candidates).map(|evaluation| evaluation.action)
}

fn choose_improving_open_meld_action(
    context: &RuleContext,
    legal_actions: &[Value],
) -> Option<Value> {
    let candidates = legal_actions
        .iter()
        .filter(|action| matches!(action_type(action).as_deref(), Some("chi" | "pon")))
        .filter_map(|action| {
            let new_hand = hand_after_call_action(&context.hand_tiles, action);
            let new_shanten = shanten_from_tiles(&new_hand);
            (new_shanten < context.current_shanten).then(|| CallEvaluation {
                action: action.clone(),
                shanten: new_shanten,
                ukeire: calculate_ukeire(&new_hand, &context.visible_counts34),
            })
        })
        .collect::<Vec<_>>();
    choose_best_call_eval(candidates).map(|evaluation| evaluation.action)
}

fn choose_daiminkan_action(context: &RuleContext, legal_actions: &[Value]) -> Option<Value> {
    legal_actions
        .iter()
        .filter(|action| action_type(action).as_deref() == Some("daiminkan"))
        .find_map(|action| {
            let new_hand = hand_after_call_action(&context.hand_tiles, action);
            let new_shanten = shanten_from_tiles(&new_hand);
            (new_shanten <= context.current_shanten).then(|| action.clone())
        })
}

fn choose_best_discard(
    context: &RuleContext,
    snapshot: &Value,
    legal_actions: &[Value],
) -> Option<Value> {
    choose_best_discard_eval(discard_evaluations(context, snapshot, legal_actions), false)
        .map(|evaluation| evaluation.action)
}

fn discard_evaluations(
    context: &RuleContext,
    snapshot: &Value,
    legal_actions: &[Value],
) -> Vec<DiscardEvaluation> {
    legal_actions
        .iter()
        .filter(|action| action_type(action).as_deref() == Some("dahai"))
        .filter_map(|action| {
            let pai = action.get("pai").and_then(Value::as_str)?;
            let new_hand = remove_one_tile(&context.hand_tiles, pai)?;
            let tile34 = tile34_from_pai(&normalize_tile_repr(pai))? as usize;
            Some(DiscardEvaluation {
                action: action.clone(),
                shanten: shanten_from_tiles(&new_hand),
                ukeire: calculate_ukeire(&new_hand, &context.visible_counts34),
                safe_against_riichi_count: context.safe_against_riichi_count(snapshot, tile34),
            })
        })
        .collect()
}

fn choose_best_discard_eval(
    evaluations: Vec<DiscardEvaluation>,
    prioritize_safety: bool,
) -> Option<DiscardEvaluation> {
    let mut best: Option<DiscardEvaluation> = None;
    for evaluation in evaluations {
        match &best {
            None => best = Some(evaluation),
            Some(current) => {
                if compare_discard_eval(&evaluation, current, prioritize_safety).is_lt() {
                    best = Some(evaluation);
                }
            }
        }
    }
    best
}

fn compare_discard_eval(
    lhs: &DiscardEvaluation,
    rhs: &DiscardEvaluation,
    prioritize_safety: bool,
) -> std::cmp::Ordering {
    if prioritize_safety {
        rhs.safe_against_riichi_count
            .cmp(&lhs.safe_against_riichi_count)
            .then_with(|| lhs.shanten.cmp(&rhs.shanten))
            .then_with(|| rhs.ukeire.cmp(&lhs.ukeire))
    } else {
        lhs.shanten
            .cmp(&rhs.shanten)
            .then_with(|| rhs.ukeire.cmp(&lhs.ukeire))
            .then_with(|| {
                rhs.safe_against_riichi_count
                    .cmp(&lhs.safe_against_riichi_count)
            })
    }
}

fn choose_best_call_eval(evaluations: Vec<CallEvaluation>) -> Option<CallEvaluation> {
    let mut best: Option<CallEvaluation> = None;
    for evaluation in evaluations {
        match &best {
            None => best = Some(evaluation),
            Some(current) => {
                let ordering = evaluation
                    .shanten
                    .cmp(&current.shanten)
                    .then_with(|| current.ukeire.cmp(&evaluation.ukeire));
                if ordering.is_lt() {
                    best = Some(evaluation);
                }
            }
        }
    }
    best
}

fn hand_after_call_action(hand: &[String], action: &Value) -> Vec<String> {
    let action_type = action_type(action).unwrap_or_default();
    let mut new_hand = hand.to_vec();
    match action_type.as_str() {
        "ankan" | "chi" | "pon" | "daiminkan" => {
            for tile in action
                .get("consumed")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                if let Some(next) = remove_one_tile(&new_hand, tile) {
                    new_hand = next;
                }
            }
        }
        "kakan" => {
            if let Some(tile) = action.get("pai").and_then(Value::as_str) {
                if let Some(next) = remove_one_tile(&new_hand, tile) {
                    new_hand = next;
                }
            }
        }
        _ => {}
    }
    new_hand
}

fn calculate_ukeire(hand: &[String], visible_counts34: &[u8; TILE_COUNT]) -> u8 {
    let current_counts = counts34_from_tiles(hand);
    let current_shanten = shanten_from_counts(&current_counts);
    let mut ukeire = 0u8;

    for tile34 in 0..TILE_COUNT {
        if current_counts[tile34] >= 4 {
            continue;
        }
        let mut test_counts = current_counts;
        test_counts[tile34] += 1;
        let new_shanten = shanten_from_counts(&test_counts);
        if new_shanten < current_shanten {
            let seen = visible_counts34[tile34].saturating_add(current_counts[tile34]);
            ukeire = ukeire.saturating_add(4u8.saturating_sub(seen));
        }
    }

    ukeire
}

fn parse_opponent_discards(snapshot: &Value) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if let Some(groups) = snapshot.get("discards").and_then(Value::as_array) {
        for (player, group) in groups.iter().enumerate().take(4) {
            if let Some(items) = group.as_array() {
                for discard in items {
                    let pai = discard
                        .get("pai")
                        .and_then(Value::as_str)
                        .or_else(|| discard.as_str());
                    if let Some(tile34) = pai
                        .map(normalize_tile_repr)
                        .and_then(|tile| tile34_from_pai(&tile))
                    {
                        result[player].push(tile34 as usize);
                    }
                }
            }
        }
    }
    result
}

fn collect_visible_counts34(snapshot: &Value, actor: usize) -> [u8; TILE_COUNT] {
    let mut visible = [0u8; TILE_COUNT];

    for tile in snapshot
        .get("dora_markers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if let Some(tile34) = tile34_from_pai(&normalize_tile_repr(tile)) {
            visible[tile34 as usize] = visible[tile34 as usize].saturating_add(1);
        }
    }

    if let Some(groups) = snapshot.get("discards").and_then(Value::as_array) {
        for (player, group) in groups.iter().enumerate().take(4) {
            if player == actor {
                continue;
            }
            for discard in group.as_array().into_iter().flatten() {
                let pai = discard
                    .get("pai")
                    .and_then(Value::as_str)
                    .or_else(|| discard.as_str());
                if let Some(tile34) = pai
                    .map(normalize_tile_repr)
                    .and_then(|tile| tile34_from_pai(&tile))
                {
                    visible[tile34 as usize] = visible[tile34 as usize].saturating_add(1);
                }
            }
        }
    }

    if let Some(groups) = snapshot.get("melds").and_then(Value::as_array) {
        for group in groups {
            for meld in group.as_array().into_iter().flatten() {
                for tile in meld
                    .get("consumed")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    if let Some(tile34) = tile34_from_pai(&normalize_tile_repr(tile)) {
                        visible[tile34 as usize] = visible[tile34 as usize].saturating_add(1);
                    }
                }
                if let Some(tile) = meld.get("pai").and_then(Value::as_str) {
                    if let Some(tile34) = tile34_from_pai(&normalize_tile_repr(tile)) {
                        visible[tile34 as usize] = visible[tile34 as usize].saturating_add(1);
                    }
                }
            }
        }
    }

    visible
}

fn counts34_from_tiles(tiles: &[String]) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for tile in tiles {
        if let Some(tile34) = tile34_from_pai(tile) {
            let idx = tile34 as usize;
            counts[idx] = counts[idx].saturating_add(1);
        }
    }
    counts
}

fn shanten_from_tiles(tiles: &[String]) -> i8 {
    let counts = counts34_from_tiles(tiles);
    shanten_from_counts(&counts)
}

fn shanten_from_counts(counts: &[u8; TILE_COUNT]) -> i8 {
    let total_tiles: u8 = counts.iter().sum();
    calc_shanten_all(counts, total_tiles / 3)
}

fn parse_melds(values: &[Value]) -> Vec<MeldView> {
    values
        .iter()
        .filter_map(|value| {
            let kind = value.get("type").and_then(Value::as_str)?.to_string();
            let pai = value
                .get("pai")
                .and_then(Value::as_str)
                .map(normalize_tile_repr);
            let consumed = value
                .get("consumed")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(normalize_tile_repr)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Some(MeldView {
                kind,
                pai,
                consumed,
            })
        })
        .collect()
}

fn yakuhai_tile_types(snapshot: &Value, actor: usize) -> [bool; TILE_COUNT] {
    let mut flags = [false; TILE_COUNT];
    flags[WHITE] = true;
    flags[GREEN] = true;
    flags[RED] = true;

    match snapshot
        .get("bakaze")
        .and_then(Value::as_str)
        .unwrap_or("E")
    {
        "E" => flags[EAST] = true,
        "S" => flags[28] = true,
        "W" => flags[29] = true,
        "N" => flags[30] = true,
        _ => {}
    }

    let oya = snapshot
        .get("oya")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .clamp(0, 3) as usize;
    let seat_wind = EAST + ((actor + 4 - oya) % 4);
    flags[seat_wind] = true;
    flags
}

fn remove_one_tile(hand: &[String], tile: &str) -> Option<Vec<String>> {
    let target = tile34_from_pai(&normalize_tile_repr(tile))? as usize;
    let mut removed = false;
    let mut out = Vec::with_capacity(hand.len().saturating_sub(1));
    for candidate in hand {
        let candidate_tile34 = tile34_from_pai(candidate).map(|idx| idx as usize);
        if !removed && candidate_tile34 == Some(target) {
            removed = true;
            continue;
        }
        out.push(candidate.clone());
    }
    removed.then_some(out)
}

fn same_tile_family(lhs: &str, rhs: &str) -> bool {
    strip_aka(&normalize_tile_repr(lhs)) == strip_aka(&normalize_tile_repr(rhs))
}

fn action_type(action: &Value) -> Option<String> {
    action
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn find_action_by_type(legal_actions: &[Value], expected_type: &str) -> Option<Value> {
    legal_actions.iter().find_map(|action| {
        (action_type(action).as_deref() == Some(expected_type)).then(|| action.clone())
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{choose_rulebase_action, score_rulebase_actions};

    #[test]
    fn chooses_safe_tile_against_riichi() {
        let snapshot = json!({
            "bakaze": "E",
            "oya": 0,
            "hand": ["1m", "9p", "1p", "2p", "4s", "5s", "7s", "E", "S", "W", "N", "P", "F", "C"],
            "melds": [[], [], [], []],
            "discards": [[], [{"pai": "1m"}], [], []],
            "dora_markers": ["3m"],
            "reached": [false, true, false, false],
            "last_tsumo": ["C", null, null, null],
        });
        let legal_actions = vec![
            json!({"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": false}),
            json!({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": false}),
        ];
        let chosen = choose_rulebase_action(&snapshot, 0, &legal_actions)
            .expect("rulebase choose action should succeed")
            .expect("rulebase should choose a discard");
        assert_eq!(chosen["type"], "dahai");
        assert_eq!(chosen["pai"], "1m");
    }

    #[test]
    fn declines_open_chi_without_secured_yakuhai() {
        let snapshot = json!({
            "bakaze": "E",
            "oya": 0,
            "hand": ["2m", "3m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4s", "5s", "6s", "9s"],
            "melds": [[], [], [], []],
            "discards": [[], [], [], []],
            "dora_markers": ["3m"],
            "reached": [false, false, false, false],
            "last_tsumo": [null, null, null, null],
        });
        let legal_actions = vec![
            json!({"type": "chi", "actor": 0, "pai": "4m", "consumed": ["2m", "3m"], "target": 1}),
            json!({"type": "none", "actor": 0}),
        ];
        let chosen = choose_rulebase_action(&snapshot, 0, &legal_actions)
            .expect("rulebase choose action should succeed")
            .expect("rulebase should choose pass");
        assert_eq!(chosen["type"], "none");
    }

    #[test]
    fn rulebase_scores_argmax_matches_choose_action() {
        let snapshot = json!({
            "bakaze": "E",
            "oya": 0,
            "hand": ["1m", "9p", "1p", "2p", "4s", "5s", "7s", "E", "S", "W", "N", "P", "F", "C"],
            "melds": [[], [], [], []],
            "discards": [[], [{"pai": "1m"}], [], []],
            "dora_markers": ["3m"],
            "reached": [false, true, false, false],
            "last_tsumo": ["C", null, null, null],
        });
        let legal_actions = vec![
            json!({"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": false}),
            json!({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": false}),
        ];
        let chosen = choose_rulebase_action(&snapshot, 0, &legal_actions)
            .expect("choose should succeed")
            .expect("chosen action");
        let scored =
            score_rulebase_actions(&snapshot, 0, &legal_actions).expect("score should succeed");
        let best = scored
            .iter()
            .max_by(|lhs, rhs| {
                let lhs_score = lhs["raw_score"].as_f64().unwrap();
                let rhs_score = rhs["raw_score"].as_f64().unwrap();
                lhs_score.partial_cmp(&rhs_score).unwrap()
            })
            .unwrap();
        let best_index = best["action_index"].as_u64().unwrap() as usize;
        assert_eq!(legal_actions[best_index], chosen);
    }
}
