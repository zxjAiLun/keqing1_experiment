use serde_json::Value;

use crate::action_identity::{action_identity_from_value, ACTION_FLAG_TSUMOGIRI};

const ACTION_TYPE_MAX: f32 = 11.0;
const ACTION_FEATURE_DIM: usize = 8;

pub fn build_keqingrl_action_features(
    snapshot: &Value,
    actions: &[Value],
    remaining_wall: f32,
) -> Result<Vec<Vec<f32>>, String> {
    let tracker = FeatureTrackerSlice::from_snapshot(snapshot)?;
    actions
        .iter()
        .map(|action| action_feature_row(&tracker, action, remaining_wall))
        .collect()
}

pub fn build_keqingrl_action_features_from_parts(
    hand_counts34: &[u8],
    visible_counts34: &[u8],
    action_types: &[u8],
    tiles: &[i16],
    flags: &[u32],
    remaining_wall: f32,
) -> Result<Vec<Vec<f32>>, String> {
    let action_count = action_types.len();
    if hand_counts34.len() != 34 {
        return Err("hand_counts34 must have 34 entries".to_string());
    }
    if visible_counts34.len() != 34 {
        return Err("visible_counts34 must have 34 entries".to_string());
    }
    if tiles.len() != action_count || flags.len() != action_count {
        return Err("action_types, tiles, and flags must have the same length".to_string());
    }
    let tracker = FeatureTrackerSlice {
        hand_counts34: hand_counts34.to_vec(),
        visible_counts34: visible_counts34.to_vec(),
    };
    let mut rows = Vec::with_capacity(action_count);
    for index in 0..action_count {
        rows.push(action_feature_row_from_parts(
            &tracker,
            action_types[index],
            tiles[index],
            flags[index],
            remaining_wall,
        )?);
    }
    Ok(rows)
}

fn action_feature_row_from_parts(
    tracker: &FeatureTrackerSlice,
    action_type: u8,
    tile: i16,
    flags: u32,
    remaining_wall: f32,
) -> Result<Vec<f32>, String> {
    if action_type > 11 {
        return Err(format!("action_type must be in [0, 11], got {action_type}"));
    }
    let tile = if tile < 0 {
        None
    } else if tile < 34 {
        Some(tile as usize)
    } else {
        return Err(format!("tile must be -1 or in [0, 33], got {tile}"));
    };
    Ok(action_feature_row_from_decoded(
        tracker,
        action_type,
        tile,
        flags,
        remaining_wall,
    ))
}

pub fn action_feature_dim() -> usize {
    ACTION_FEATURE_DIM
}

fn action_feature_row(
    tracker: &FeatureTrackerSlice,
    action: &Value,
    remaining_wall: f32,
) -> Result<Vec<f32>, String> {
    let identity = action_identity_from_value(action)?;
    Ok(action_feature_row_from_decoded(
        tracker,
        identity.action_type,
        identity.tile.map(usize::from),
        identity.flags,
        remaining_wall,
    ))
}

fn action_feature_row_from_decoded(
    tracker: &FeatureTrackerSlice,
    action_type: u8,
    tile: Option<usize>,
    flags: u32,
    remaining_wall: f32,
) -> Vec<f32> {
    let tile_norm = tile.map(|value| value as f32 / 33.0).unwrap_or(0.0);
    let hand_count = tile
        .and_then(|value| tracker.hand_counts34.get(value).copied())
        .map(|value| value as f32 / 4.0)
        .unwrap_or(0.0);
    let visible_count = tile
        .and_then(|value| tracker.visible_counts34.get(value).copied())
        .map(|value| value as f32 / 4.0)
        .unwrap_or(0.0);
    let is_honor = tile.map(|value| value >= 27).unwrap_or(false);
    let is_terminal = tile
        .map(|value| value < 27 && (value % 9 == 0 || value % 9 == 8))
        .unwrap_or(false);
    vec![
        f32::from(action_type) / ACTION_TYPE_MAX,
        tile_norm,
        if flags & ACTION_FLAG_TSUMOGIRI != 0 {
            1.0
        } else {
            0.0
        },
        hand_count,
        visible_count,
        if is_honor { 1.0 } else { 0.0 },
        if is_terminal { 1.0 } else { 0.0 },
        remaining_wall / 70.0,
    ]
}

struct FeatureTrackerSlice {
    hand_counts34: Vec<u8>,
    visible_counts34: Vec<u8>,
}

impl FeatureTrackerSlice {
    fn from_snapshot(snapshot: &Value) -> Result<Self, String> {
        if let Some(feature_tracker) = snapshot.get("feature_tracker") {
            return Ok(Self {
                hand_counts34: read_counts34(feature_tracker, "hand_counts34")?,
                visible_counts34: read_counts34(feature_tracker, "visible_counts34")?,
            });
        }

        let actor = snapshot
            .get("actor")
            .and_then(Value::as_u64)
            .ok_or_else(|| "snapshot.actor is required for action features".to_string())?
            as usize;
        let mut hand_counts34 = vec![0u8; 34];
        for tile in snapshot
            .get("hand")
            .and_then(Value::as_array)
            .ok_or_else(|| "snapshot.hand is required for action features".to_string())?
        {
            if let Some(tile_name) = tile.as_str() {
                increment_count(&mut hand_counts34, tile_name);
            }
        }

        let mut visible_counts34 = hand_counts34.clone();
        if let Some(meld_groups) = snapshot.get("melds").and_then(Value::as_array) {
            for meld_group in meld_groups {
                let Some(melds) = meld_group.as_array() else {
                    continue;
                };
                for meld in melds {
                    if let Some(consumed) = meld.get("consumed").and_then(Value::as_array) {
                        for tile in consumed {
                            if let Some(tile_name) = tile.as_str() {
                                increment_count(&mut visible_counts34, tile_name);
                            }
                        }
                    }
                    if let Some(tile_name) = meld.get("pai").and_then(Value::as_str) {
                        increment_count(&mut visible_counts34, tile_name);
                    }
                }
            }
        }
        if let Some(discard_groups) = snapshot.get("discards").and_then(Value::as_array) {
            for discard_group in discard_groups {
                let Some(discards) = discard_group.as_array() else {
                    continue;
                };
                for discard in discards {
                    let tile_name = discard
                        .get("pai")
                        .and_then(Value::as_str)
                        .or_else(|| discard.as_str());
                    if let Some(tile_name) = tile_name {
                        increment_count(&mut visible_counts34, tile_name);
                    }
                }
            }
        }
        if let Some(dora_markers) = snapshot.get("dora_markers").and_then(Value::as_array) {
            for marker in dora_markers {
                if let Some(tile_name) = marker.as_str() {
                    increment_count(&mut visible_counts34, tile_name);
                }
            }
        }
        if let Some(tile_name) = snapshot.get("tsumo_pai").and_then(Value::as_str) {
            increment_count(&mut visible_counts34, tile_name);
        } else if actor < 4 {
            if let Some(last_tsumo) = snapshot.get("last_tsumo").and_then(Value::as_array) {
                if let Some(tile_name) = last_tsumo.get(actor).and_then(Value::as_str) {
                    increment_count(&mut visible_counts34, tile_name);
                }
            }
        }

        Ok(Self {
            hand_counts34,
            visible_counts34,
        })
    }
}

fn read_counts34(container: &Value, key: &str) -> Result<Vec<u8>, String> {
    let values = container
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("feature_tracker.{key} is required"))?;
    if values.len() != 34 {
        return Err(format!("feature_tracker.{key} must have 34 entries"));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|item| item as u8)
                .ok_or_else(|| format!("feature_tracker.{key} contains non-integer count"))
        })
        .collect()
}

fn increment_count(counts: &mut [u8], tile_name: &str) {
    if let Some(tile34) = tile_name_to_34(tile_name) {
        counts[tile34] = counts[tile34].saturating_add(1);
    }
}

fn tile_name_to_34(tile_name: &str) -> Option<usize> {
    let normalized = match tile_name {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        value if value.ends_with('r') => value.trim_end_matches('r').to_string(),
        value => value.to_string(),
    };
    match normalized.as_str() {
        "E" => Some(27),
        "S" => Some(28),
        "W" => Some(29),
        "N" => Some(30),
        "P" => Some(31),
        "F" => Some(32),
        "C" => Some(33),
        _ => {
            let mut chars = normalized.chars();
            let rank = chars.next()?.to_digit(10)? as usize;
            let suit = chars.next()?;
            if !(1..=9).contains(&rank) {
                return None;
            }
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return None,
            };
            Some(base + rank - 1)
        }
    }
}
