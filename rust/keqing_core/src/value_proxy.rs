use serde_json::Value;

pub fn confirmed_han_floor(
    yakuhai_triplet_count: f32,
    tanyao_path: f32,
    dora_aka_count: f32,
) -> f32 {
    (yakuhai_triplet_count + tanyao_path + dora_aka_count).min(8.0)
}

pub fn tenpai_value_proxy_norm(
    base_value_proxy: f32,
    waits_count: u8,
    waits_live: usize,
    is_tenpai: bool,
) -> f32 {
    let mut value_proxy = base_value_proxy;
    if is_tenpai {
        value_proxy +=
            0.5 + (waits_count.min(5) as f32) * 0.15 + (waits_live.min(12) as f32) * 0.03;
    }
    (value_proxy / 8.0).min(1.0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HoraTruthYakuFlags {
    pub tanyao: f32,
    pub yakuhai: f32,
    pub chiitoi: f32,
    pub iipeiko: f32,
    pub pinfu: f32,
}

pub fn hora_truth_value_proxy_from_parts(han: i32, cost_total: i64) -> f32 {
    let han = han.max(0) as f32;
    let cost_total = cost_total.max(0) as f32;
    let han_norm = (han.min(8.0) / 8.0).clamp(0.0, 1.0);
    let cost_norm = (cost_total / 12000.0).clamp(0.0, 1.0);
    (0.65 * han_norm + 0.35 * cost_norm).clamp(0.0, 1.0)
}

pub fn hora_truth_yaku_flags_from_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> HoraTruthYakuFlags {
    let mut flags = HoraTruthYakuFlags::default();
    for name in names {
        match name {
            "Tanyao" => flags.tanyao = 1.0,
            "Chiitoitsu" => flags.chiitoi = 1.0,
            "Iipeikou" => flags.iipeiko = 1.0,
            "Pinfu" => flags.pinfu = 1.0,
            _ if name.starts_with("Yakuhai (") => flags.yakuhai = 1.0,
            _ => {}
        }
    }
    flags
}

pub fn hora_truth_value_proxy(payload: &Value) -> Option<f32> {
    if payload.get("error").is_some_and(|value| !value.is_null()) {
        return None;
    }
    let han = payload.get("han").and_then(Value::as_i64).unwrap_or(0) as i32;
    let cost_total = payload
        .get("cost")
        .and_then(|value| value.get("total"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    Some(hora_truth_value_proxy_from_parts(han, cost_total))
}

pub fn hora_truth_yaku_flags(payload: &Value) -> Option<HoraTruthYakuFlags> {
    if payload.get("error").is_some_and(|value| !value.is_null()) {
        return None;
    }
    Some(hora_truth_yaku_flags_from_names(
        payload
            .get("yaku")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn confirmed_han_floor_clamps_to_eight() {
        assert_eq!(confirmed_han_floor(3.0, 1.0, 6.0), 8.0);
    }

    #[test]
    fn tenpai_value_proxy_norm_adds_wait_bonus_only_in_tenpai() {
        let tenpai = tenpai_value_proxy_norm(2.0, 4, 9, true);
        let noten = tenpai_value_proxy_norm(2.0, 4, 9, false);
        assert!(tenpai > noten);
        assert!(tenpai <= 1.0);
    }

    #[test]
    fn hora_truth_value_proxy_uses_han_and_cost() {
        let payload = json!({
            "error": Value::Null,
            "han": 3,
            "cost": {"total": 5200},
        });
        let value = hora_truth_value_proxy(&payload).unwrap();
        assert!(value > 0.0);
        assert!(value < 1.0);
        assert_eq!(value, hora_truth_value_proxy_from_parts(3, 5200));
    }

    #[test]
    fn hora_truth_yaku_flags_detect_supported_routes() {
        let payload = json!({
            "error": Value::Null,
            "yaku": ["Tanyao", "Pinfu", "Yakuhai (haku)", "Chiitoitsu"],
        });
        let flags = hora_truth_yaku_flags(&payload).unwrap();
        assert_eq!(
            flags,
            HoraTruthYakuFlags {
                tanyao: 1.0,
                yakuhai: 1.0,
                chiitoi: 1.0,
                iipeiko: 0.0,
                pinfu: 1.0,
            }
        );
    }
}
