use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::types::{Conditions, Meld as RiichiMeld, MeldType, Wind};
use riichienv_core::yaku::{get_yaku_by_id, ID_AKADORA, ID_DORA, ID_URADORA};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::score_rules::prepare_hora_tile_allocation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YakuDetail {
    pub key: String,
    pub name: String,
    pub han: i32,
}

impl YakuDetail {
    pub fn new(name: &str, han: i32) -> Self {
        Self {
            key: name.to_string(),
            name: name.to_string(),
            han,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeHoraOutcome {
    pub han: i32,
    pub fu: i32,
    pub yaku: Vec<String>,
    pub base_yaku_details: Vec<YakuDetail>,
    pub is_open_hand: bool,
    pub cost: Value,
    pub dora_count: i32,
    pub ura_count: i32,
    pub aka_count: i32,
}

fn parse_wind(value: &str) -> Wind {
    match value {
        "S" => Wind::South,
        "W" => Wind::West,
        "N" => Wind::North,
        _ => Wind::East,
    }
}

fn seat_wind(oya: usize, actor: usize) -> Wind {
    match (actor + 4usize).wrapping_sub(oya) % 4 {
        1 => Wind::South,
        2 => Wind::West,
        3 => Wind::North,
        _ => Wind::East,
    }
}

fn riichi_meld_type(meld_type: &str) -> Result<MeldType, String> {
    match meld_type {
        "chi" => Ok(MeldType::Chi),
        "pon" => Ok(MeldType::Pon),
        "daiminkan" => Ok(MeldType::Daiminkan),
        "ankan" => Ok(MeldType::Ankan),
        "kakan" => Ok(MeldType::Kakan),
        other => Err(format!("unsupported meld type: {other}")),
    }
}

fn tile_ids_from_value(value: &Value, key: &str) -> Result<Vec<u8>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {key}"))?
        .iter()
        .map(|tile| {
            let raw = tile
                .as_u64()
                .ok_or_else(|| format!("{key} must contain integers"))?;
            u8::try_from(raw).map_err(|_| format!("{key} value out of range: {raw}"))
        })
        .collect()
}

fn parse_allocation_melds(allocation: &Value) -> Result<Vec<RiichiMeld>, String> {
    allocation
        .get("melds")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing melds".to_string())?
        .iter()
        .map(|meld| {
            let meld_type = meld
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "meld missing type".to_string())?;
            let tile_ids = tile_ids_from_value(meld, "tile_ids")?;
            let called_tile = meld
                .get("opened")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then(|| tile_ids.last().copied())
                .flatten();
            Ok(RiichiMeld::new(
                riichi_meld_type(meld_type)?,
                tile_ids,
                meld.get("opened").and_then(Value::as_bool).unwrap_or(false),
                -1,
                called_tile,
            ))
        })
        .collect()
}

fn next_dora_tile(tile34: u8) -> u8 {
    match tile34 {
        0..=8 => (tile34 + 1) % 9,
        9..=17 => 9 + (tile34 - 9 + 1) % 9,
        18..=26 => 18 + (tile34 - 18 + 1) % 9,
        27..=30 => 27 + (tile34 - 27 + 1) % 4,
        31..=33 => 31 + (tile34 - 31 + 1) % 3,
        _ => tile34,
    }
}

fn dora_count(tile_ids: &[u8], indicators: &[u8]) -> u8 {
    let mut count = 0u8;
    for indicator in indicators {
        let target = next_dora_tile(indicator / 4);
        count = count.saturating_add(
            tile_ids
                .iter()
                .filter(|tile_id| (**tile_id / 4) == target)
                .count() as u8,
        );
    }
    count
}

fn aka_count(tile_ids: &[u8]) -> u8 {
    tile_ids
        .iter()
        .filter(|tile_id| matches!(**tile_id, 16 | 52 | 88))
        .count() as u8
}

fn yaku_han(id: u32, is_open_hand: bool) -> i32 {
    match id {
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 30 => 1,
        15 | 16 | 17 => {
            if is_open_hand {
                1
            } else {
                2
            }
        }
        18 => 2,
        19 | 20 | 21 | 22 | 23 | 24 | 25 => 2,
        26 => {
            if is_open_hand {
                2
            } else {
                3
            }
        }
        27 => {
            if is_open_hand {
                2
            } else {
                3
            }
        }
        28 => 3,
        29 => {
            if is_open_hand {
                5
            } else {
                6
            }
        }
        35..=45 => 13,
        47..=50 => 26,
        _ => 0,
    }
}

fn display_name(id: u32, fallback: String) -> String {
    match id {
        38 => "Suu Ankou".to_string(),
        39 => "Tsuuiisou".to_string(),
        40 => "Ryuuiisou".to_string(),
        43 => "Shou Suushii".to_string(),
        44 => "Suu Kantsu".to_string(),
        49 => "Kokushi Musou Juusanmen Matchi".to_string(),
        47 => "Junsei Chuuren Poutou".to_string(),
        48 => "Suu Ankou Tanki".to_string(),
        50 => "Dai Suushii".to_string(),
        _ => fallback,
    }
}

fn base_yaku_details(yaku_ids: &[u32], is_open_hand: bool) -> Vec<YakuDetail> {
    yaku_ids
        .iter()
        .filter(|id| !matches!(**id, ID_DORA | ID_AKADORA | ID_URADORA))
        .filter_map(|id| {
            let yaku = get_yaku_by_id(*id)?;
            let name = display_name(*id, yaku.name_en);
            Some(YakuDetail {
                key: name.clone(),
                name,
                han: yaku_han(*id, is_open_hand),
            })
        })
        .collect()
}

fn yaku_names(yaku_ids: &[u32]) -> Vec<String> {
    yaku_ids
        .iter()
        .filter_map(|id| get_yaku_by_id(*id).map(|yaku| display_name(*id, yaku.name_en)))
        .collect()
}

fn yaku_level(han: u32, fu: u32) -> String {
    if han >= 13 {
        let count = han / 13;
        return if count <= 1 {
            "yakuman".to_string()
        } else {
            format!("{count}x yakuman")
        };
    }
    if han >= 5 || (han == 4 && fu >= 40) || (han == 3 && fu >= 70) {
        return "mangan".to_string();
    }
    match han {
        11 | 12 => "sanbaiman".to_string(),
        8..=10 => "baiman".to_string(),
        6 | 7 => "haneman".to_string(),
        _ => String::new(),
    }
}

fn build_cost(
    result: &riichienv_core::types::WinResult,
    is_tsumo: bool,
    actor_is_oya: bool,
    kyotaku: i64,
) -> Value {
    let kyoutaku_bonus = kyotaku.saturating_mul(1000);
    if is_tsumo {
        let main = if actor_is_oya {
            i64::from(result.tsumo_agari_ko)
        } else {
            i64::from(result.tsumo_agari_oya)
        };
        let additional = if actor_is_oya {
            0
        } else {
            i64::from(result.tsumo_agari_ko)
        };
        let total = if actor_is_oya {
            main.saturating_mul(3).saturating_add(kyoutaku_bonus)
        } else {
            main.saturating_add(additional.saturating_mul(2))
                .saturating_add(kyoutaku_bonus)
        };
        json!({
            "main": main,
            "main_bonus": 0,
            "additional": if actor_is_oya { main } else { additional },
            "additional_bonus": 0,
            "kyoutaku_bonus": kyoutaku_bonus,
            "total": total,
            "yaku_level": yaku_level(result.han, result.fu),
        })
    } else {
        let main = i64::from(result.ron_agari);
        json!({
            "main": main,
            "main_bonus": 0,
            "additional": 0,
            "additional_bonus": 0,
            "kyoutaku_bonus": kyoutaku_bonus,
            "total": main.saturating_add(kyoutaku_bonus),
            "yaku_level": yaku_level(result.han, result.fu),
        })
    }
}

pub fn evaluate_native_hora(prepared: &Value) -> Result<Option<NativeHoraOutcome>, String> {
    let allocation = prepare_hora_tile_allocation(prepared)?;
    let closed_tile_ids = tile_ids_from_value(&allocation, "closed_tile_ids")?;
    let win_tile = allocation
        .get("win_tile")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing win_tile".to_string())
        .and_then(|raw| u8::try_from(raw).map_err(|_| format!("win_tile out of range: {raw}")))?;
    let melds = parse_allocation_melds(&allocation)?;
    let dora_ids = tile_ids_from_value(&allocation, "dora_ids")?;
    let ura_ids = tile_ids_from_value(&allocation, "ura_ids")?;

    let actor = prepared
        .get("actor")
        .or_else(|| prepared.get("target"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing actor".to_string())
        .and_then(|raw| usize::try_from(raw).map_err(|_| format!("actor out of range: {raw}")))?;
    let oya = prepared
        .get("oya")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing oya".to_string())
        .and_then(|raw| usize::try_from(raw).map_err(|_| format!("oya out of range: {raw}")))?;
    let bakaze = prepared
        .get("bakaze")
        .and_then(Value::as_str)
        .unwrap_or("E");
    let kyotaku = prepared.get("kyotaku").and_then(Value::as_i64).unwrap_or(0);
    let honba = prepared.get("honba").and_then(Value::as_u64).unwrap_or(0);
    let is_tsumo = prepared
        .get("is_tsumo")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let conditions = Conditions {
        tsumo: is_tsumo,
        riichi: prepared
            .get("reached")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        double_riichi: false,
        ippatsu: prepared
            .get("ippatsu_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        haitei: prepared
            .get("resolved_is_haitei")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        houtei: prepared
            .get("resolved_is_houtei")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        rinshan: prepared
            .get("resolved_is_rinshan")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        player_wind: seat_wind(oya, actor),
        round_wind: parse_wind(bakaze),
        chankan: prepared
            .get("is_chankan")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        tsumo_first_turn: false,
        riichi_sticks: u32::try_from(kyotaku.max(0)).unwrap_or(0),
        honba: u32::try_from(honba).unwrap_or(0),
        kita_count: 0,
        is_sanma: false,
        num_players: 4,
    };

    let mut all_tile_ids = closed_tile_ids.clone();
    for meld in &melds {
        all_tile_ids.extend(meld.tiles.iter().copied());
    }
    let dora_count_value = dora_count(&all_tile_ids, &dora_ids);
    let ura_count_value = dora_count(&all_tile_ids, &ura_ids);
    let aka_count_value = aka_count(&all_tile_ids);

    let evaluator = HandEvaluator::new(closed_tile_ids, melds.clone());
    let result = evaluator.calc(win_tile, dora_ids, ura_ids, Some(conditions));
    if !result.is_win {
        return Ok(None);
    }

    let is_open_hand = melds.iter().any(|meld| meld.opened);
    let yaku_ids = result.yaku.clone();
    let cost = build_cost(&result, is_tsumo, actor == oya, kyotaku);
    let fu = if result.yakuman && result.fu == 0 && yaku_ids.contains(&48) {
        50
    } else {
        result.fu
    };

    Ok(Some(NativeHoraOutcome {
        han: i32::try_from(result.han).unwrap_or(i32::MAX),
        fu: i32::try_from(fu).unwrap_or(i32::MAX),
        yaku: yaku_names(&yaku_ids),
        base_yaku_details: base_yaku_details(&yaku_ids, is_open_hand),
        is_open_hand,
        cost,
        dora_count: i32::from(dora_count_value),
        ura_count: i32::from(ura_count_value),
        aka_count: i32::from(aka_count_value),
    }))
}
