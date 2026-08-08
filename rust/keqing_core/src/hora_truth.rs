use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::native_scoring::{evaluate_native_hora, YakuDetail};
use crate::score_rules::compute_hora_deltas;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoraTruth {
    pub han: i32,
    pub fu: i32,
    pub yaku: Vec<String>,
    pub yaku_details: Vec<YakuDetail>,
    pub dora_count: i32,
    pub ura_count: i32,
    pub aka_count: i32,
    pub cost: Value,
    pub deltas: Vec<i32>,
    pub is_open_hand: bool,
    pub backend_name: String,
    pub truth_source: String,
}

fn prepared_actor(prepared: &Value) -> Result<usize, String> {
    prepared
        .get("actor")
        .or_else(|| prepared.get("target"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing actor".to_string())
        .and_then(|raw| usize::try_from(raw).map_err(|_| format!("actor out of range: {raw}")))
}

fn prepared_target(prepared: &Value, actor: usize, is_tsumo: bool) -> Result<usize, String> {
    if is_tsumo {
        return Ok(actor);
    }
    let target = prepared
        .get("target")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing target".to_string())
        .and_then(|raw| usize::try_from(raw).map_err(|_| format!("target out of range: {raw}")))?;
    if target == actor {
        return Err("missing ron target".to_string());
    }
    Ok(target)
}

fn build_yaku_details(
    base_yaku_details: &[YakuDetail],
    dora_count: i32,
    ura_count: i32,
    aka_count: i32,
) -> Vec<YakuDetail> {
    let mut yaku_details = base_yaku_details.to_vec();
    if dora_count > 0 {
        yaku_details.push(YakuDetail::new("Dora", dora_count));
    }
    if ura_count > 0 {
        yaku_details.push(YakuDetail::new("Ura Dora", ura_count));
    }
    if aka_count > 0 {
        yaku_details.push(YakuDetail::new("Aka Dora", aka_count));
    }
    yaku_details
}

fn build_yaku_names(
    base_yaku_details: &[YakuDetail],
    dora_count: i32,
    ura_count: i32,
    aka_count: i32,
) -> Vec<String> {
    let mut yaku = base_yaku_details
        .iter()
        .map(|detail| detail.name.clone())
        .collect::<Vec<_>>();
    if dora_count > 0 {
        yaku.push("Dora".to_string());
    }
    if ura_count > 0 {
        yaku.push("Ura Dora".to_string());
    }
    if aka_count > 0 {
        yaku.push("Aka Dora".to_string());
    }
    yaku
}

pub fn evaluate_hora_truth_from_prepared(prepared: &Value) -> Result<HoraTruth, String> {
    let outcome = evaluate_native_hora(prepared)?.ok_or_else(|| "no cost".to_string())?;
    let actor = prepared_actor(prepared)?;
    let is_tsumo = prepared
        .get("is_tsumo")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let oya = prepared
        .get("oya")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing oya".to_string())
        .and_then(|raw| usize::try_from(raw).map_err(|_| format!("oya out of range: {raw}")))?;
    let target = prepared_target(prepared, actor, is_tsumo)?;
    let deltas = compute_hora_deltas(oya, actor, target, is_tsumo, &outcome.cost)?;
    let yaku = build_yaku_names(
        &outcome.base_yaku_details,
        outcome.dora_count,
        outcome.ura_count,
        outcome.aka_count,
    );
    let yaku_details = build_yaku_details(
        &outcome.base_yaku_details,
        outcome.dora_count,
        outcome.ura_count,
        outcome.aka_count,
    );

    Ok(HoraTruth {
        han: outcome.han,
        fu: outcome.fu,
        yaku,
        yaku_details,
        dora_count: outcome.dora_count,
        ura_count: outcome.ura_count,
        aka_count: outcome.aka_count,
        cost: outcome.cost,
        deltas,
        is_open_hand: outcome.is_open_hand,
        backend_name: "riichienv-core".to_string(),
        truth_source: "riichienv-core-adapter".to_string(),
    })
}

pub fn legacy_payload_from_truth(truth: &HoraTruth) -> Value {
    let base_yaku_details = truth
        .yaku_details
        .iter()
        .filter(|detail| !matches!(detail.name.as_str(), "Dora" | "Ura Dora" | "Aka Dora"))
        .map(|detail| {
            json!({
                "key": detail.key,
                "name": detail.name,
                "han": detail.han,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "error": Value::Null,
        "han": truth.han,
        "fu": truth.fu,
        "yaku": truth.yaku,
        "base_yaku_details": base_yaku_details,
        "is_open_hand": truth.is_open_hand,
        "cost": truth.cost,
        "dora_count": truth.dora_count,
        "ura_count": truth.ura_count,
        "aka_count": truth.aka_count,
    })
}

pub fn legacy_error_payload(message: &str) -> Value {
    json!({
        "error": message,
        "cost": Value::Null,
    })
}
