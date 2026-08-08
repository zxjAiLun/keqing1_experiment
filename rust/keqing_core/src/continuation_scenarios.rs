use serde_json::{json, Value};

use crate::keqingv4_summary::{
    enumerate_keqingv4_live_draw_weights, enumerate_keqingv4_post_meld_discards,
    enumerate_keqingv4_reach_discards, project_keqingv4_call_snapshot,
    project_keqingv4_reach_snapshot, project_keqingv4_rinshan_draw_snapshot,
};
use crate::legal_actions::enumerate_legal_action_specs_structural;

pub fn build_keqingv4_continuation_scenarios(
    snapshot: &Value,
    actor: usize,
    action: &Value,
) -> Vec<Value> {
    let action_type = action
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match action_type {
        "chi" | "pon" => build_post_meld_scenarios(snapshot, actor, action),
        "daiminkan" | "ankan" | "kakan" => build_kan_scenarios(snapshot, actor, action),
        "reach" => build_reach_scenarios(snapshot, actor),
        _ => Vec::new(),
    }
}

fn structural_legal_actions(snapshot: &Value, actor: usize) -> Vec<Value> {
    enumerate_legal_action_specs_structural(snapshot, actor).unwrap_or_default()
}

fn build_post_meld_scenarios(snapshot: &Value, actor: usize, action: &Value) -> Vec<Value> {
    let Some(projected_snapshot) = project_keqingv4_call_snapshot(snapshot, actor, action) else {
        return Vec::new();
    };
    let legal_actions = enumerate_keqingv4_post_meld_discards(&projected_snapshot, actor);
    vec![json!({
        "projected_snapshot": projected_snapshot,
        "legal_actions": legal_actions,
        "weight": 1.0,
        "continuation_kind": "post_meld_followup",
        "declaration_action": Value::Null,
    })]
}

fn build_kan_scenarios(snapshot: &Value, actor: usize, action: &Value) -> Vec<Value> {
    let Some(post_meld_snapshot) = project_keqingv4_call_snapshot(snapshot, actor, action) else {
        return Vec::new();
    };
    let mut scenarios = Vec::new();
    for (pai, weight) in enumerate_keqingv4_live_draw_weights(&post_meld_snapshot) {
        let projected_snapshot =
            project_keqingv4_rinshan_draw_snapshot(&post_meld_snapshot, actor, &pai);
        let legal_actions = structural_legal_actions(&projected_snapshot, actor);
        scenarios.push(json!({
            "projected_snapshot": projected_snapshot,
            "legal_actions": legal_actions,
            "weight": weight as f32,
            "continuation_kind": "rinshan_followup",
            "declaration_action": Value::Null,
        }));
    }
    if scenarios.is_empty() {
        scenarios.push(json!({
            "projected_snapshot": post_meld_snapshot,
            "legal_actions": structural_legal_actions(&post_meld_snapshot, actor),
            "weight": 1.0,
            "continuation_kind": "state_value",
            "declaration_action": Value::Null,
        }));
    }
    scenarios
}

fn build_reach_scenarios(snapshot: &Value, actor: usize) -> Vec<Value> {
    let mut scenarios = Vec::new();
    for (pai, tsumogiri) in enumerate_keqingv4_reach_discards(snapshot, actor) {
        let projected_snapshot = project_keqingv4_reach_snapshot(snapshot, actor, &pai);
        let legal_actions = structural_legal_actions(&projected_snapshot, actor);
        scenarios.push(json!({
            "projected_snapshot": projected_snapshot,
            "legal_actions": legal_actions,
            "weight": 1.0,
            "continuation_kind": "reach_declaration",
            "declaration_action": {
                "type": "dahai",
                "actor": actor,
                "pai": pai,
                "tsumogiri": tsumogiri,
            },
        }));
    }
    if scenarios.is_empty() {
        let mut projected_snapshot = snapshot.clone();
        if let Some(obj) = projected_snapshot.as_object_mut() {
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
        }
        let legal_actions = structural_legal_actions(&projected_snapshot, actor);
        scenarios.push(json!({
            "projected_snapshot": projected_snapshot,
            "legal_actions": legal_actions,
            "weight": 1.0,
            "continuation_kind": "state_value",
            "declaration_action": Value::Null,
        }));
    }
    scenarios
}
