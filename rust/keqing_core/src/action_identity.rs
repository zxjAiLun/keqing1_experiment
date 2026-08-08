use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const ACTION_FLAG_TSUMOGIRI: u32 = 1 << 0;
pub const ACTION_FLAG_REACH: u32 = 1 << 1;
const TILE_SLOT_BASE: u64 = 35;
const FROM_WHO_BASE: u64 = 5;
const FLAGS_BASE: u64 = 256;
const MAX_CONSUMED: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionIdentity {
    pub version: u32,
    pub action_type: u8,
    pub action_type_name: String,
    pub actor: Option<usize>,
    pub tile: Option<u8>,
    pub consumed: Vec<u8>,
    pub from_who: Option<usize>,
    pub flags: u32,
    pub canonical_key: String,
    pub action_id: u64,
    pub supported: bool,
    pub unsupported_reason: Option<String>,
}

pub fn action_identity_from_value(action: &Value) -> Result<ActionIdentity, String> {
    let action_type = parse_action_type(action)?;
    let action_type_name = action_type_name(action_type).to_string();
    let actor = optional_usize(action, "actor")?;
    let tile = optional_tile(action)?;
    let consumed = consumed_tiles(action)?;
    let from_who = optional_usize(action, "from_who")?.or(optional_usize(action, "target")?);
    let flags = parse_flags(action, action_type);
    let supported = action_type != 10;
    let unsupported_reason = if supported {
        None
    } else {
        Some("NUKI is explicitly unsupported by ActionIdentity v1".to_string())
    };
    let canonical_key = make_canonical_key(action_type, tile, &consumed, from_who, flags);
    let action_id = encode_action_id(action_type, tile, &consumed, from_who, flags);
    Ok(ActionIdentity {
        version: 1,
        action_type,
        action_type_name,
        actor,
        tile,
        consumed,
        from_who,
        flags,
        canonical_key,
        action_id,
        supported,
        unsupported_reason,
    })
}

pub fn action_identity_json(action: &Value) -> Result<Value, String> {
    serde_json::to_value(action_identity_from_value(action)?).map_err(|err| err.to_string())
}

pub fn decode_action_id(action_id: u64) -> Result<Value, String> {
    let mut work = action_id;
    let flags = (work % FLAGS_BASE) as u32;
    work /= FLAGS_BASE;
    let from_who = decode_from_who_slot(work % FROM_WHO_BASE)?;
    work /= FROM_WHO_BASE;

    let mut consumed_rev = Vec::new();
    for _ in 0..MAX_CONSUMED {
        let tile = decode_tile_slot(work % TILE_SLOT_BASE)?;
        work /= TILE_SLOT_BASE;
        if let Some(tile) = tile {
            consumed_rev.push(tile);
        }
    }
    let tile = decode_tile_slot(work % TILE_SLOT_BASE)?;
    work /= TILE_SLOT_BASE;
    let action_type =
        u8::try_from(work).map_err(|_| format!("decoded action type out of range: {work}"))?;
    if action_type > 11 {
        return Err(format!("decoded action type out of range: {action_type}"));
    }
    consumed_rev.reverse();
    Ok(json!({
        "action_type": action_type,
        "tile": tile,
        "consumed": consumed_rev,
        "from_who": from_who,
        "flags": flags,
    }))
}

pub fn mjai_events_for_action(
    action: &Value,
    actor_override: Option<usize>,
) -> Result<Vec<Value>, String> {
    let identity = action_identity_from_value(action)?;
    if !identity.supported {
        return Err(identity
            .unsupported_reason
            .unwrap_or_else(|| "unsupported action".to_string()));
    }
    let actor = actor_override.or(identity.actor).unwrap_or(0);
    match identity.action_type {
        0 => Ok(vec![dahai_event(actor, identity.tile, identity.flags)?]),
        1 => Ok(vec![
            json!({"type": "reach", "actor": actor}),
            dahai_event(actor, identity.tile, identity.flags)?,
        ]),
        2 | 3 => {
            let mut payload = json!({"type": "hora", "actor": actor});
            if let Some(tile) = identity.tile {
                payload["pai"] = Value::String(tile_id_to_name(tile)?);
            }
            if identity.action_type == 3 {
                if let Some(from_who) = identity.from_who {
                    payload["target"] = Value::from(from_who);
                }
            }
            Ok(vec![payload])
        }
        4 | 5 | 6 | 7 | 8 => {
            let action_type = match identity.action_type {
                4 => "chi",
                5 => "pon",
                6 => "daiminkan",
                7 => "ankan",
                8 => "kakan",
                _ => unreachable!(),
            };
            let mut payload = json!({"type": action_type, "actor": actor});
            if let Some(tile) = identity.tile {
                payload["pai"] = Value::String(tile_id_to_name(tile)?);
            }
            if !identity.consumed.is_empty() {
                payload["consumed"] = Value::Array(
                    identity
                        .consumed
                        .iter()
                        .map(|tile| tile_id_to_name(*tile).map(Value::String))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            if matches!(identity.action_type, 4 | 5 | 6) {
                if let Some(from_who) = identity.from_who {
                    payload["target"] = Value::from(from_who);
                }
            }
            Ok(vec![payload])
        }
        9 => Ok(vec![json!({"type": "none"})]),
        11 => Ok(vec![json!({"type": "ryukyoku", "actor": actor})]),
        _ => Err(format!(
            "unsupported action type for MJAI expansion: {}",
            identity.action_type
        )),
    }
}

fn parse_action_type(action: &Value) -> Result<u8, String> {
    if let Some(raw) = action.get("action_type").and_then(Value::as_u64) {
        return u8::try_from(raw).map_err(|_| format!("action_type out of range: {raw}"));
    }
    let raw_type = action
        .get("action_type_name")
        .or_else(|| action.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("none");
    match raw_type {
        "DISCARD" | "dahai" => Ok(0),
        "REACH_DISCARD" | "reach_discard" => Ok(1),
        "TSUMO" => Ok(2),
        "RON" => Ok(3),
        "CHI" | "chi" => Ok(4),
        "PON" | "pon" => Ok(5),
        "DAIMINKAN" | "daiminkan" => Ok(6),
        "ANKAN" | "ankan" => Ok(7),
        "KAKAN" | "kakan" => Ok(8),
        "PASS" | "none" => Ok(9),
        "NUKI" | "nuki" => Ok(10),
        "RYUKYOKU" | "ryukyoku" => Ok(11),
        "hora" => {
            let actor = optional_usize(action, "actor")?;
            let target = optional_usize(action, "target")?;
            if target.is_none() || actor.is_some() && target == actor {
                Ok(2)
            } else {
                Ok(3)
            }
        }
        other => Err(format!("unsupported action type for identity: {other}")),
    }
}

fn action_type_name(action_type: u8) -> &'static str {
    match action_type {
        0 => "DISCARD",
        1 => "REACH_DISCARD",
        2 => "TSUMO",
        3 => "RON",
        4 => "CHI",
        5 => "PON",
        6 => "DAIMINKAN",
        7 => "ANKAN",
        8 => "KAKAN",
        9 => "PASS",
        10 => "NUKI",
        11 => "RYUKYOKU",
        _ => "UNKNOWN",
    }
}

fn optional_usize(action: &Value, key: &str) -> Result<Option<usize>, String> {
    match action.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| format!("{key} must be an unsigned integer"))
            .and_then(|raw| usize::try_from(raw).map_err(|_| format!("{key} out of range: {raw}")))
            .map(Some),
    }
}

fn optional_tile(action: &Value) -> Result<Option<u8>, String> {
    if let Some(raw) = action.get("tile") {
        if raw.is_null() {
            return Ok(None);
        }
        return raw
            .as_u64()
            .ok_or_else(|| "tile must be an unsigned integer".to_string())
            .and_then(|value| tile_u8(value, "tile"))
            .map(Some);
    }
    if let Some(raw) = action.get("pai").and_then(Value::as_str) {
        return tile_name_to_id(raw).map(Some);
    }
    Ok(None)
}

fn consumed_tiles(action: &Value) -> Result<Vec<u8>, String> {
    let Some(items) = action.get("consumed").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        if let Some(value) = item.as_u64() {
            result.push(tile_u8(value, "consumed tile")?);
        } else if let Some(value) = item.as_str() {
            result.push(tile_name_to_id(value)?);
        } else {
            return Err("consumed tiles must be integers or tile names".to_string());
        }
    }
    Ok(result)
}

fn parse_flags(action: &Value, action_type: u8) -> u32 {
    if let Some(raw) = action.get("flags").and_then(Value::as_u64) {
        return raw as u32;
    }
    let mut flags = 0;
    if action
        .get("tsumogiri")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        flags |= ACTION_FLAG_TSUMOGIRI;
    }
    if action_type == 1
        || action
            .get("reach")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        flags |= ACTION_FLAG_REACH;
    }
    flags
}

fn make_canonical_key(
    action_type: u8,
    tile: Option<u8>,
    consumed: &[u8],
    from_who: Option<usize>,
    flags: u32,
) -> String {
    let consumed_text = consumed
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}|tile={}|consumed={}|from={}|flags={}",
        action_type,
        tile.map(i32::from).unwrap_or(-1),
        consumed_text,
        from_who.map(|value| value as i32).unwrap_or(-1),
        flags
    )
}

fn encode_action_id(
    action_type: u8,
    tile: Option<u8>,
    consumed: &[u8],
    from_who: Option<usize>,
    flags: u32,
) -> u64 {
    let mut encoded = u64::from(action_type);
    encoded = encoded * TILE_SLOT_BASE + tile_slot(tile);
    for slot in 0..MAX_CONSUMED {
        encoded = encoded * TILE_SLOT_BASE + tile_slot(consumed.get(slot).copied());
    }
    encoded = encoded * FROM_WHO_BASE + from_who_slot(from_who);
    encoded * FLAGS_BASE + u64::from(flags)
}

fn decode_tile_slot(slot: u64) -> Result<Option<u8>, String> {
    if slot == 0 {
        return Ok(None);
    }
    let tile = slot - 1;
    tile_u8(tile, "tile slot").map(Some)
}

fn decode_from_who_slot(slot: u64) -> Result<Option<usize>, String> {
    if slot == 0 {
        return Ok(None);
    }
    let from_who = slot - 1;
    if from_who <= 3 {
        Ok(Some(from_who as usize))
    } else {
        Err(format!("from_who slot out of range: {slot}"))
    }
}

fn dahai_event(actor: usize, tile: Option<u8>, flags: u32) -> Result<Value, String> {
    let tile = tile.ok_or_else(|| "dahai action requires tile".to_string())?;
    Ok(json!({
        "type": "dahai",
        "actor": actor,
        "pai": tile_id_to_name(tile)?,
        "tsumogiri": flags & ACTION_FLAG_TSUMOGIRI != 0,
    }))
}

fn tile_slot(tile: Option<u8>) -> u64 {
    tile.map(|value| u64::from(value) + 1).unwrap_or(0)
}

fn from_who_slot(from_who: Option<usize>) -> u64 {
    from_who.map(|value| value as u64 + 1).unwrap_or(0)
}

fn tile_u8(value: u64, field: &str) -> Result<u8, String> {
    if value < 34 {
        Ok(value as u8)
    } else {
        Err(format!("{field} must be in [0, 33], got {value}"))
    }
}

fn tile_name_to_id(tile: &str) -> Result<u8, String> {
    let normalized = normalize_tile(tile);
    if normalized.len() == 2 {
        let mut chars = normalized.chars();
        let rank = chars
            .next()
            .and_then(|ch| ch.to_digit(10))
            .ok_or_else(|| format!("invalid tile name: {tile}"))?;
        let suit = chars
            .next()
            .ok_or_else(|| format!("invalid tile name: {tile}"))?;
        if (1..=9).contains(&rank) {
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return Err(format!("invalid tile suit: {tile}")),
            };
            return Ok((base + rank - 1) as u8);
        }
    }
    match normalized.as_str() {
        "E" => Ok(27),
        "S" => Ok(28),
        "W" => Ok(29),
        "N" => Ok(30),
        "P" => Ok(31),
        "F" => Ok(32),
        "C" => Ok(33),
        _ => Err(format!("invalid tile name: {tile}")),
    }
}

fn tile_id_to_name(tile: u8) -> Result<String, String> {
    if tile < 27 {
        let suit = match tile / 9 {
            0 => 'm',
            1 => 'p',
            2 => 's',
            _ => unreachable!(),
        };
        return Ok(format!("{}{}", tile % 9 + 1, suit));
    }
    match tile {
        27 => Ok("E".to_string()),
        28 => Ok("S".to_string()),
        29 => Ok("W".to_string()),
        30 => Ok("N".to_string()),
        31 => Ok("P".to_string()),
        32 => Ok("F".to_string()),
        33 => Ok("C".to_string()),
        _ => Err(format!("tile id out of range: {tile}")),
    }
}

fn normalize_tile(tile: &str) -> String {
    match tile {
        "0m" | "5mr" => "5m".to_string(),
        "0p" | "5pr" => "5p".to_string(),
        "0s" | "5sr" => "5s".to_string(),
        _ if tile.ends_with('r') => tile.trim_end_matches('r').to_string(),
        _ => tile.to_string(),
    }
}
