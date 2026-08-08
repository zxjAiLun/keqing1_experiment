use serde_json::{json, Value};

pub fn fixed_seed_eval_gate(report: &Value) -> Result<Value, String> {
    let illegal_action_rate = f64_field(report, "illegal_action_rate")?;
    let fallback_rate = f64_field(report, "fallback_rate")?;
    let forced_terminal_missed = i64_field(report, "forced_terminal_missed")?;
    let fourth_rate = f64_field(report, "fourth_rate")?;
    let deal_in_rate = f64_field(report, "deal_in_rate")?;
    let max_fourth_rate = f64_field(report, "max_fourth_rate")?;
    let max_deal_in_rate = f64_field(report, "max_deal_in_rate")?;
    let terminal_reason_count = report
        .get("terminal_reason_count")
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal_reason_count is required".to_string())?;

    let mut failure_reasons = Vec::new();
    if illegal_action_rate > 0.0 {
        failure_reasons.push(format!("illegal_action_rate > 0: {illegal_action_rate}"));
    }
    if fallback_rate > 0.0 {
        failure_reasons.push(format!("fallback_rate > 0: {fallback_rate}"));
    }
    if forced_terminal_missed > 0 {
        failure_reasons.push(format!(
            "forced_terminal_missed > 0: {forced_terminal_missed}"
        ));
    }
    if terminal_reason_count.is_empty() {
        failure_reasons.push("terminal_reason_count is empty".to_string());
    }
    if fourth_rate > max_fourth_rate {
        failure_reasons.push(format!(
            "fourth_rate {fourth_rate} exceeded {max_fourth_rate}"
        ));
    }
    if deal_in_rate > max_deal_in_rate {
        failure_reasons.push(format!(
            "deal_in_rate {deal_in_rate} exceeded {max_deal_in_rate}"
        ));
    }

    Ok(json!({
        "passed": failure_reasons.is_empty(),
        "failure_reasons": failure_reasons,
    }))
}

fn f64_field(payload: &Value, key: &str) -> Result<f64, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{key} is required"))
}

fn i64_field(payload: &Value, key: &str) -> Result<i64, String> {
    payload
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("{key} is required"))
}
