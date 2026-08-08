use serde_json::{json, Value};

pub fn resolve_terminal_action(
    _state_snapshot: &Value,
    actor: usize,
    legal_actions: &[Value],
    forced_action_types: &[String],
) -> Result<Option<Value>, String> {
    for (index, action) in legal_actions.iter().enumerate() {
        let Some(reason) = terminal_reason(action, actor) else {
            continue;
        };
        if forced_action_types.iter().any(|item| {
            item.eq_ignore_ascii_case(&reason)
                || item.eq_ignore_ascii_case(action_type_name(&reason))
        }) {
            return Ok(Some(json!({
                "action_index": index,
                "action": action,
                "terminal_reason": reason,
            })));
        }
    }
    Ok(None)
}

fn terminal_reason(action: &Value, actor: usize) -> Option<String> {
    match action.get("type").and_then(Value::as_str).unwrap_or("none") {
        "ryukyoku" => Some("ryukyoku".to_string()),
        "hora" => {
            let target = action
                .get("target")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            if target.is_none() || target == Some(actor) {
                Some("tsumo".to_string())
            } else {
                Some("ron".to_string())
            }
        }
        _ => None,
    }
}

fn action_type_name(reason: &str) -> &'static str {
    match reason {
        "tsumo" => "TSUMO",
        "ron" => "RON",
        "ryukyoku" => "RYUKYOKU",
        _ => "UNKNOWN",
    }
}
