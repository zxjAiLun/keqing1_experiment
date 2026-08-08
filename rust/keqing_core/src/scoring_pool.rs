//! Helpers for scoring-side tile pool construction.

use std::collections::BTreeMap;

use crate::counts::TILE_COUNT;

fn parse_tile34(tile: &str) -> Option<(usize, bool)> {
    match tile {
        "E" => Some((27, false)),
        "S" => Some((28, false)),
        "W" => Some((29, false)),
        "N" => Some((30, false)),
        "P" => Some((31, false)),
        "F" => Some((32, false)),
        "C" => Some((33, false)),
        "5mr" => Some((4, true)),
        "5pr" => Some((13, true)),
        "5sr" => Some((22, true)),
        _ => {
            let bytes = tile.as_bytes();
            if bytes.len() != 2 {
                return None;
            }
            let suit = bytes[1] as char;
            let digit = (bytes[0] as char).to_digit(10)? as usize;
            if !(1..=9).contains(&digit) {
                return None;
            }
            let base = match suit {
                'm' => 0,
                'p' => 9,
                's' => 18,
                _ => return None,
            };
            Some((base + digit - 1, false))
        }
    }
}

fn tile_key(tile34: usize, aka: bool) -> String {
    match tile34 {
        27 => "E".to_string(),
        28 => "S".to_string(),
        29 => "W".to_string(),
        30 => "N".to_string(),
        31 => "P".to_string(),
        32 => "F".to_string(),
        33 => "C".to_string(),
        _ => {
            let suit = match tile34 / 9 {
                0 => 'm',
                1 => 'p',
                _ => 's',
            };
            let digit = (tile34 % 9) + 1;
            if aka && matches!(tile34, 4 | 13 | 22) {
                format!("{digit}{suit}r")
            } else {
                format!("{digit}{suit}")
            }
        }
    }
}

pub fn build_136_pool_entries(tiles: &[String]) -> Vec<(String, Vec<u8>)> {
    let mut plain_counts = [0u8; TILE_COUNT];
    let mut aka_counts = [0u8; TILE_COUNT];

    for tile in tiles {
        let Some((tile34, aka)) = parse_tile34(tile) else {
            continue;
        };
        if aka {
            aka_counts[tile34] = aka_counts[tile34].saturating_add(1);
        } else {
            plain_counts[tile34] = plain_counts[tile34].saturating_add(1);
        }
    }

    for &tile34 in &[4usize, 13usize, 22usize] {
        if plain_counts[tile34] >= 4 && aka_counts[tile34] == 0 {
            plain_counts[tile34] -= 1;
            aka_counts[tile34] = 1;
        }
    }

    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for tile34 in 0..TILE_COUNT {
        let base = (tile34 * 4) as u8;
        if aka_counts[tile34] > 0 {
            out.insert(tile_key(tile34, true), vec![base]);
        }
        let plain_count = plain_counts[tile34] as usize;
        if plain_count == 0 {
            continue;
        }
        let start = if matches!(tile34, 4 | 13 | 22) { 1 } else { 0 };
        let ids: Vec<u8> = (0..plain_count)
            .map(|offset| base + start + offset as u8)
            .collect();
        out.insert(tile_key(tile34, false), ids);
    }

    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_136_pool_entries_keeps_explicit_red_fives() {
        let tiles = vec![
            "5mr".to_string(),
            "5m".to_string(),
            "5m".to_string(),
            "5m".to_string(),
        ];
        let pool = build_136_pool_entries(&tiles);
        let map: BTreeMap<_, _> = pool.into_iter().collect();
        assert_eq!(map.get("5mr"), Some(&vec![16]));
        assert_eq!(map.get("5m"), Some(&vec![17, 18, 19]));
    }

    #[test]
    fn build_136_pool_entries_promotes_one_plain_five_when_all_four_are_plain() {
        let tiles = vec![
            "5m".to_string(),
            "5m".to_string(),
            "5m".to_string(),
            "5m".to_string(),
        ];
        let pool = build_136_pool_entries(&tiles);
        let map: BTreeMap<_, _> = pool.into_iter().collect();
        assert_eq!(map.get("5mr"), Some(&vec![16]));
        assert_eq!(map.get("5m"), Some(&vec![17, 18, 19]));
    }
}
