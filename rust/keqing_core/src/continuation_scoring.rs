use serde::Deserialize;
use serde_json::{json, Value};

use crate::keqingv4_summary::strip_aka;

const ACTION_SPACE: usize = 45;
const REACH_IDX: usize = 34;
const CHI_LOW_IDX: usize = 35;
const CHI_MID_IDX: usize = 36;
const CHI_HIGH_IDX: usize = 37;
const PON_IDX: usize = 38;
const DAIMINKAN_IDX: usize = 39;
const ANKAN_IDX: usize = 40;
const KAKAN_IDX: usize = 41;
const HORA_IDX: usize = 42;
const RYUKYOKU_IDX: usize = 43;
const NONE_IDX: usize = 44;

fn tile_name_to_idx(tile: &str) -> Option<usize> {
    let normalized = strip_aka(tile);
    match normalized.as_str() {
        "1m" => Some(0),
        "2m" => Some(1),
        "3m" => Some(2),
        "4m" => Some(3),
        "5m" => Some(4),
        "6m" => Some(5),
        "7m" => Some(6),
        "8m" => Some(7),
        "9m" => Some(8),
        "1p" => Some(9),
        "2p" => Some(10),
        "3p" => Some(11),
        "4p" => Some(12),
        "5p" => Some(13),
        "6p" => Some(14),
        "7p" => Some(15),
        "8p" => Some(16),
        "9p" => Some(17),
        "1s" => Some(18),
        "2s" => Some(19),
        "3s" => Some(20),
        "4s" => Some(21),
        "5s" => Some(22),
        "6s" => Some(23),
        "7s" => Some(24),
        "8s" => Some(25),
        "9s" => Some(26),
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

fn tile_rank(tile: &str) -> i32 {
    let normalized = strip_aka(tile);
    if normalized.len() == 2 {
        let mut chars = normalized.chars();
        let rank = chars.next().unwrap_or_default();
        let suit = chars.next().unwrap_or_default();
        if matches!(suit, 'm' | 'p' | 's') {
            return rank.to_digit(10).map(|value| value as i32).unwrap_or(-1);
        }
    }
    -1
}

fn chi_type_idx(pai: &str, consumed: &[Value]) -> usize {
    let pai_rank = tile_rank(pai);
    let mut ranks = consumed
        .iter()
        .filter_map(Value::as_str)
        .map(tile_rank)
        .collect::<Vec<_>>();
    ranks.sort_unstable();
    if ranks.len() < 2 {
        return CHI_LOW_IDX;
    }
    let lo = ranks[0];
    let hi = ranks[1];
    if pai_rank < lo {
        CHI_LOW_IDX
    } else if pai_rank < hi {
        CHI_MID_IDX
    } else {
        CHI_HIGH_IDX
    }
}

fn action_to_idx(action: &Value) -> usize {
    let action_type = action.get("type").and_then(Value::as_str).unwrap_or("none");
    match action_type {
        "dahai" => action
            .get("pai")
            .and_then(Value::as_str)
            .and_then(tile_name_to_idx)
            .unwrap_or(0),
        "reach" => REACH_IDX,
        "chi" => {
            let pai = action
                .get("pai")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let empty = Vec::<Value>::new();
            let consumed = action
                .get("consumed")
                .and_then(Value::as_array)
                .unwrap_or(&empty);
            chi_type_idx(pai, consumed)
        }
        "pon" => PON_IDX,
        "daiminkan" => DAIMINKAN_IDX,
        "ankan" => ANKAN_IDX,
        "kakan" => KAKAN_IDX,
        "hora" => HORA_IDX,
        "ryukyoku" => RYUKYOKU_IDX,
        _ => NONE_IDX,
    }
}

fn aux_bonus(
    score_delta: f32,
    win_prob: f32,
    dealin_prob: f32,
    score_delta_lambda: f32,
    win_prob_lambda: f32,
    dealin_prob_lambda: f32,
) -> f32 {
    score_delta_lambda * score_delta + win_prob_lambda * win_prob - dealin_prob_lambda * dealin_prob
}

fn placement_bonus(rank_pt_value: f32, rank_pt_lambda: f32) -> f32 {
    rank_pt_lambda * rank_pt_value
}

fn legal_score(
    policy_logits: &[f32],
    action: &Value,
    value: f32,
    style_lambda: f32,
    aux_bonus: f32,
) -> f32 {
    let idx = action_to_idx(action);
    let mut score = policy_logits.get(idx).copied().unwrap_or(-1e9);
    let action_type = action.get("type").and_then(Value::as_str).unwrap_or("none");
    if action_type != "none" {
        score += style_lambda * value;
        score += aux_bonus;
    }
    score
}

pub(crate) fn score_continuation_scenario(
    continuation_kind: &str,
    policy_logits: &[f32],
    legal_actions: &[Value],
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
) -> Value {
    debug_assert!(policy_logits.len() >= ACTION_SPACE);
    let scenario_aux_bonus = aux_bonus(
        score_delta,
        win_prob,
        dealin_prob,
        score_delta_lambda,
        win_prob_lambda,
        dealin_prob_lambda,
    );
    let scenario_placement_bonus = placement_bonus(rank_pt_value, rank_pt_lambda);
    if matches!(continuation_kind, "reach_declaration" | "state_value") {
        return json!({
            "best_action": Value::Null,
            "score": beam_lambda * value + scenario_aux_bonus + scenario_placement_bonus,
        });
    }

    if legal_actions.is_empty() {
        return json!({
            "best_action": Value::Null,
            "score": -1e18_f32,
        });
    }

    let mut best_action = legal_actions[0].clone();
    let mut best_score = legal_score(policy_logits, &best_action, value, 0.0, scenario_aux_bonus)
        + scenario_placement_bonus;
    for action in legal_actions.iter().skip(1) {
        let score = legal_score(policy_logits, action, value, 0.0, scenario_aux_bonus)
            + scenario_placement_bonus;
        if score > best_score {
            best_score = score;
            best_action = action.clone();
        }
    }

    json!({
        "best_action": best_action,
        "score": best_score,
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScenarioScoreInput {
    score: f32,
    #[serde(default)]
    weight: Option<f32>,
    #[serde(default)]
    continuation_kind: Option<String>,
    #[serde(default)]
    declaration_action: Option<Value>,
}

pub(crate) fn aggregate_continuation_scores(
    root_policy_logits: &[f32],
    action: &Value,
    scenario_scores: &[ScenarioScoreInput],
) -> Value {
    debug_assert!(root_policy_logits.len() >= ACTION_SPACE);
    let action_idx = action_to_idx(action);
    let root_action_logit = root_policy_logits.get(action_idx).copied().unwrap_or(-1e9);
    if scenario_scores.is_empty() {
        return json!({
            "final_score": root_action_logit,
            "meta": {},
        });
    }

    let action_type = action
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if action_type == "reach" {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_meta = json!({});
        for scenario in scenario_scores {
            let _continuation_kind = scenario.continuation_kind.as_deref().unwrap_or_default();
            let declaration_logit = scenario
                .declaration_action
                .as_ref()
                .map(|item| {
                    let idx = action_to_idx(item);
                    root_policy_logits.get(idx).copied().unwrap_or(-1e9)
                })
                .unwrap_or(0.0);
            let total_scenario_score = scenario.score + declaration_logit;
            if total_scenario_score > best_score {
                best_score = total_scenario_score;
                best_meta = match scenario.declaration_action.clone() {
                    Some(declaration_action) => json!({"reach_discard": declaration_action}),
                    None => json!({}),
                };
            }
        }
        return json!({
            "final_score": root_action_logit + best_score,
            "meta": best_meta,
        });
    }

    let total_weight: f32 = scenario_scores
        .iter()
        .map(|scenario| scenario.weight.unwrap_or(1.0).max(0.0))
        .sum();
    let continuation_score = if total_weight <= 0.0 {
        scenario_scores
            .iter()
            .map(|scenario| scenario.score)
            .fold(f32::NEG_INFINITY, f32::max)
    } else {
        scenario_scores
            .iter()
            .map(|scenario| scenario.weight.unwrap_or(1.0).max(0.0) * scenario.score)
            .sum::<f32>()
            / total_weight
    };
    json!({
        "final_score": root_action_logit + continuation_score,
        "meta": {},
    })
}

#[cfg(test)]
mod tests {
    use super::{
        aggregate_continuation_scores, score_continuation_scenario, ScenarioScoreInput,
        ACTION_SPACE, DAIMINKAN_IDX, REACH_IDX,
    };
    use serde_json::json;

    #[test]
    fn score_continuation_scenario_matches_followup_compare_semantics() {
        let logits = vec![-1e9; ACTION_SPACE];
        let mut logits = logits;
        logits[8] = 1.25;
        logits[4] = -0.5;
        let legal_actions = vec![
            json!({"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}),
            json!({"type":"dahai","actor":0,"pai":"9m","tsumogiri":false}),
        ];
        let scored = score_continuation_scenario(
            "post_meld_followup",
            &logits,
            &legal_actions,
            0.0,
            0.4,
            0.2,
            0.1,
            0.0,
            1.0,
            0.5,
            0.25,
            0.75,
            0.0,
        );
        assert_eq!(scored["best_action"]["pai"], json!("9m"));
        let expected = 1.25 + 0.5 * 0.4 + 0.25 * 0.2 - 0.75 * 0.1;
        assert!((scored["score"].as_f64().unwrap() as f32 - expected).abs() < 1e-6);
    }

    #[test]
    fn aggregate_continuation_scores_matches_reach_and_weighted_paths() {
        let mut logits = vec![-1e9; ACTION_SPACE];
        logits[REACH_IDX] = 0.8;
        logits[3] = 0.1;
        let aggregated_reach = aggregate_continuation_scores(
            &logits,
            &json!({"type":"reach","actor":0}),
            &[ScenarioScoreInput {
                score: 0.4,
                weight: Some(1.0),
                continuation_kind: Some("reach_declaration".to_string()),
                declaration_action: Some(
                    json!({"type":"dahai","actor":0,"pai":"4m","tsumogiri":false}),
                ),
            }],
        );
        assert_eq!(
            aggregated_reach["meta"]["reach_discard"]["pai"],
            json!("4m")
        );
        assert!((aggregated_reach["final_score"].as_f64().unwrap() as f32 - 1.3).abs() < 1e-6);

        let aggregated_weighted = aggregate_continuation_scores(
            &logits,
            &json!({"type":"daiminkan","actor":0,"pai":"5m","consumed":["5m","5m","5m"]}),
            &[
                ScenarioScoreInput {
                    score: 1.0,
                    weight: Some(2.0),
                    continuation_kind: Some("rinshan_followup".to_string()),
                    declaration_action: None,
                },
                ScenarioScoreInput {
                    score: -0.5,
                    weight: Some(1.0),
                    continuation_kind: Some("rinshan_followup".to_string()),
                    declaration_action: None,
                },
            ],
        );
        let expected = logits[DAIMINKAN_IDX] + (2.0 * 1.0 + 1.0 * -0.5) / 3.0;
        assert!(
            (aggregated_weighted["final_score"].as_f64().unwrap() as f32 - expected).abs() < 1e-6
        );
    }
}
