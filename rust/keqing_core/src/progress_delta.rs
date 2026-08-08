//! Delta-oriented helpers for draw/discard and required-tile analysis.

use crate::counts::TILE_COUNT;
use crate::shanten_table::calc_shanten_all;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredTile {
    pub tile34: u8,
    pub live_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawDelta {
    pub tile34: u8,
    pub live_count: u8,
    pub shanten_diff: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscardDelta {
    pub tile34: u8,
    pub shanten_diff: i8,
}

#[must_use]
pub fn calc_required_tiles(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
    len_div3: u8,
) -> Vec<RequiredTile> {
    let shanten = calc_shanten_all(counts34, len_div3);
    let mut work = *counts34;
    let mut out = Vec::new();

    for tile34 in 0..TILE_COUNT {
        let live_count = 4u8.saturating_sub(visible_counts34[tile34]);
        if live_count == 0 {
            continue;
        }
        if work[tile34] >= 4 {
            continue;
        }
        work[tile34] += 1;
        let after = calc_shanten_all(&work, len_div3);
        work[tile34] -= 1;
        if after < shanten {
            out.push(RequiredTile {
                tile34: tile34 as u8,
                live_count,
            });
        }
    }

    out
}

#[must_use]
pub fn calc_draw_deltas(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
    len_div3: u8,
) -> Vec<DrawDelta> {
    let shanten = calc_shanten_all(counts34, len_div3);
    let mut work = *counts34;
    let mut out = Vec::new();

    for tile34 in 0..TILE_COUNT {
        let live_count = 4u8.saturating_sub(visible_counts34[tile34]);
        if live_count == 0 {
            continue;
        }
        if work[tile34] >= 4 {
            continue;
        }
        work[tile34] += 1;
        let after = calc_shanten_all(&work, len_div3);
        work[tile34] -= 1;
        out.push(DrawDelta {
            tile34: tile34 as u8,
            live_count,
            shanten_diff: after - shanten,
        });
    }

    out
}

#[must_use]
pub fn calc_discard_deltas(counts34: &[u8; TILE_COUNT], len_div3: u8) -> Vec<DiscardDelta> {
    let shanten = calc_shanten_all(counts34, len_div3);
    let mut work = *counts34;
    let mut out = Vec::new();

    for tile34 in 0..TILE_COUNT {
        if work[tile34] == 0 {
            continue;
        }
        work[tile34] -= 1;
        let after = calc_shanten_all(&work, len_div3);
        work[tile34] += 1;
        out.push(DiscardDelta {
            tile34: tile34 as u8,
            shanten_diff: after - shanten,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(entries: &[(usize, u8)]) -> [u8; 34] {
        let mut tiles = [0u8; 34];
        for &(idx, count) in entries {
            tiles[idx] = count;
        }
        tiles
    }

    #[test]
    fn required_tiles_contains_wait_for_tenpai_hand() {
        let counts = hand(&[
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (12, 1),
            (13, 2),
        ]);
        let visible = counts;
        let required = calc_required_tiles(&counts, &visible, 4);
        assert!(!required.is_empty());
    }

    #[test]
    fn discard_deltas_are_emitted_for_present_tiles_only() {
        let counts = hand(&[(0, 1), (1, 1), (2, 1), (27, 2)]);
        let deltas = calc_discard_deltas(&counts, 1);
        assert_eq!(deltas.len(), 4);
        assert!(deltas
            .iter()
            .all(|item| matches!(item.tile34, 0 | 1 | 2 | 27)));
    }
}
