use std::collections::BTreeMap;

use serde_json::{json, Value};

fn normalize_tile(tile: &str) -> String {
    match tile {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}

fn take_tile_id(pool: &mut BTreeMap<String, Vec<u8>>, tile: &str) -> Result<u8, String> {
    if let Some(exact) = pool.get_mut(tile) {
        if !exact.is_empty() {
            return Ok(exact.remove(0));
        }
    }

    let normalized = normalize_tile(tile);
    if let Some(fallback) = pool.get_mut(&normalized) {
        if !fallback.is_empty() {
            return Ok(fallback.remove(0));
        }
    }

    if normalized.len() == 2 {
        let bytes = normalized.as_bytes();
        if bytes[0] == b'5' && matches!(bytes[1] as char, 'm' | 'p' | 's') {
            let aka = format!("{normalized}r");
            if let Some(aka_pool) = pool.get_mut(&aka) {
                if !aka_pool.is_empty() {
                    return Ok(aka_pool.remove(0));
                }
            }
        }
    }

    Err(format!("failed to allocate tile id for {tile} from pool"))
}

fn find_win_tile_id(hand_tile_ids: &[u8], hand_tiles: &[String], pai: &str) -> Result<u8, String> {
    for (tile_id, tile) in hand_tile_ids.iter().zip(hand_tiles.iter()).rev() {
        if tile == pai {
            return Ok(*tile_id);
        }
    }
    let pai_norm = normalize_tile(pai);
    for (tile_id, tile) in hand_tile_ids.iter().zip(hand_tiles.iter()).rev() {
        if normalize_tile(tile) == pai_norm {
            return Ok(*tile_id);
        }
    }
    Err(format!("failed to find win tile id for {pai}"))
}

pub fn compute_hora_deltas(
    oya: usize,
    actor: usize,
    target: usize,
    is_tsumo: bool,
    cost: &Value,
) -> Result<Vec<i32>, String> {
    let mut deltas = vec![0i32; 4];
    let main = cost.get("main").and_then(Value::as_i64).unwrap_or(0) as i32;
    let main_bonus = cost.get("main_bonus").and_then(Value::as_i64).unwrap_or(0) as i32;
    let additional = cost.get("additional").and_then(Value::as_i64).unwrap_or(0) as i32;
    let additional_bonus = cost
        .get("additional_bonus")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let kyoutaku_bonus = cost
        .get("kyoutaku_bonus")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;

    if actor >= 4 || target >= 4 || oya >= 4 {
        return Err("actor/target/oya out of range".to_string());
    }

    if is_tsumo {
        if actor == oya {
            let payment = main + main_bonus;
            for pid in 0..4 {
                if pid == actor {
                    continue;
                }
                deltas[pid] -= payment;
                deltas[actor] += payment;
            }
        } else {
            let dealer_payment = main + main_bonus;
            let non_dealer_payment = additional + additional_bonus;
            for pid in 0..4 {
                if pid == actor {
                    continue;
                }
                let payment = if pid == oya {
                    dealer_payment
                } else {
                    non_dealer_payment
                };
                deltas[pid] -= payment;
                deltas[actor] += payment;
            }
        }
        deltas[actor] += kyoutaku_bonus;
    } else {
        let payment = main + main_bonus + kyoutaku_bonus;
        deltas[target] -= payment;
        deltas[actor] += payment;
    }

    Ok(deltas)
}

pub fn prepare_hora_tile_allocation(prepared: &Value) -> Result<Value, String> {
    let closed_tiles = prepared
        .get("closed_tiles")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing closed_tiles".to_string())?
        .iter()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let melds = prepared
        .get("melds")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let dora_markers = prepared
        .get("dora_markers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let ura_markers = prepared
        .get("active_ura_markers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let pai = prepared
        .get("pai")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing pai".to_string())?;

    let mut all_tiles = closed_tiles.clone();
    for meld in &melds {
        let tiles = meld
            .get("tiles")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for tile in tiles {
            if let Some(tile) = tile.as_str() {
                all_tiles.push(tile.to_string());
            }
        }
    }
    for marker in &dora_markers {
        if let Some(marker) = marker.as_str() {
            all_tiles.push(marker.to_string());
        }
    }
    for marker in &ura_markers {
        if let Some(marker) = marker.as_str() {
            all_tiles.push(marker.to_string());
        }
    }

    let mut pool: BTreeMap<String, Vec<u8>> =
        crate::scoring_pool::build_136_pool_entries(&all_tiles)
            .into_iter()
            .collect();

    let closed_tile_ids = closed_tiles
        .iter()
        .map(|tile| take_tile_id(&mut pool, tile))
        .collect::<Result<Vec<_>, _>>()?;
    let win_tile = find_win_tile_id(&closed_tile_ids, &closed_tiles, pai)?;

    let meld_payload = melds
        .iter()
        .map(|meld| {
            let tiles = meld
                .get("tiles")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let tile_ids = tiles
                .iter()
                .filter_map(Value::as_str)
                .map(|tile| take_tile_id(&mut pool, tile))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!({
                "type": meld.get("type").cloned().unwrap_or(Value::Null),
                "opened": meld.get("opened").cloned().unwrap_or(Value::Bool(false)),
                "tile_ids": tile_ids,
            }))
        })
        .collect::<Result<Vec<Value>, String>>()?;

    let dora_ids = dora_markers
        .iter()
        .filter_map(Value::as_str)
        .map(|marker| take_tile_id(&mut pool, marker))
        .collect::<Result<Vec<_>, _>>()?;
    let ura_ids = ura_markers
        .iter()
        .filter_map(Value::as_str)
        .map(|marker| take_tile_id(&mut pool, marker))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(json!({
        "closed_tile_ids": closed_tile_ids,
        "win_tile": win_tile,
        "melds": meld_payload,
        "dora_ids": dora_ids,
        "ura_ids": ura_ids,
    }))
}

pub fn build_hora_result_payload(
    han: i32,
    fu: i32,
    is_open_hand: bool,
    yaku_names: &[String],
    base_yaku_details: &Value,
    dora_count: i32,
    ura_count: i32,
    aka_count: i32,
    cost: &Value,
    deltas: &[i32],
) -> Result<Value, String> {
    let mut yaku_details = base_yaku_details
        .as_array()
        .cloned()
        .ok_or_else(|| "base_yaku_details must be an array".to_string())?;

    if dora_count > 0 {
        yaku_details.push(json!({"key": "Dora", "name": "Dora", "han": dora_count}));
    }
    if ura_count > 0 {
        yaku_details.push(json!({"key": "Ura Dora", "name": "Ura Dora", "han": ura_count}));
    }
    if aka_count > 0 {
        yaku_details.push(json!({"key": "Aka Dora", "name": "Aka Dora", "han": aka_count}));
    }

    Ok(json!({
        "han": han,
        "fu": fu,
        "yaku": yaku_names,
        "yaku_details": yaku_details,
        "is_open_hand": is_open_hand,
        "cost": cost,
        "deltas": deltas,
    }))
}
