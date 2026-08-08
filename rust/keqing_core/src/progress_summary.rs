//! Summary-oriented helpers for 3n+1 progress analysis.

use crate::counts::TILE_COUNT;
use crate::progress_delta::{calc_discard_deltas, calc_draw_deltas, calc_required_tiles};
use crate::shanten_table::{calc_shanten_all, calc_shanten_normal};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary3n1 {
    pub shanten: i8,
    pub waits_count: u8,
    pub waits_tiles: [bool; TILE_COUNT],
    pub tehai_count: u8,
    pub ukeire_type_count: u8,
    pub ukeire_live_count: u8,
    pub ukeire_tiles: [bool; TILE_COUNT],
}

fn candidate_progress_key(summary: &Summary3n1) -> (i32, i32, i32, i32) {
    (
        -(summary.shanten as i32),
        summary.ukeire_live_count as i32,
        summary.ukeire_type_count as i32,
        summary.waits_count as i32,
    )
}

fn best_after_draw_summary(
    counts14: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> Option<Summary3n1> {
    let mut best_summary: Option<Summary3n1> = None;
    let mut best_key: Option<(i32, i32, i32, i32)> = None;
    for discard34 in select_candidate_discards_3n2(counts14) {
        let mut after_counts = *counts14;
        after_counts[discard34] = after_counts[discard34].saturating_sub(1);
        let summary = summarize_3n1(&after_counts, visible_counts34);
        let key = candidate_progress_key(&summary);
        if best_key.as_ref().is_none_or(|current| key > *current) {
            best_key = Some(key);
            best_summary = Some(summary);
        }
    }
    best_summary
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OneShantenDrawAnalysis {
    good_shape_live: u8,
    improvement_live: u8,
    tenpai_live: u8,
    best_tenpai_waits_live: u8,
}

fn tile_in_obvious_meld(counts34: &[u8; TILE_COUNT], tile34: usize) -> bool {
    let cnt = counts34[tile34];
    if cnt == 0 {
        return false;
    }
    if cnt >= 3 {
        return true;
    }
    if tile34 >= 27 {
        return false;
    }
    let pos = tile34 % 9;
    let base = tile34 - pos;
    if pos >= 2 && counts34[base + pos - 1] > 0 && counts34[base + pos - 2] > 0 {
        return true;
    }
    if (1..=7).contains(&pos) && counts34[base + pos - 1] > 0 && counts34[base + pos + 1] > 0 {
        return true;
    }
    if pos <= 6 && counts34[base + pos + 1] > 0 && counts34[base + pos + 2] > 0 {
        return true;
    }
    false
}

fn select_candidate_discards_3n2(counts34: &[u8; TILE_COUNT]) -> Vec<usize> {
    let mut preferred = Vec::new();
    let mut fallback = Vec::new();
    let mut seen = [false; TILE_COUNT];
    for (tile34, &cnt) in counts34.iter().enumerate() {
        if cnt == 0 || seen[tile34] {
            continue;
        }
        seen[tile34] = true;
        fallback.push(tile34);
        if !tile_in_obvious_meld(counts34, tile34) {
            preferred.push(tile34);
        }
    }
    let candidate_discards = if preferred.is_empty() {
        fallback.clone()
    } else {
        preferred
    };
    if candidate_discards.is_empty() {
        return candidate_discards;
    }
    let len_div3 = counts34.iter().sum::<u8>() / 3;
    let current_shanten = calc_shanten_all(counts34, len_div3);
    let discard_deltas = calc_discard_deltas(counts34, len_div3);
    let mut best_after_shanten: Option<i8> = None;
    let mut selected = Vec::new();
    for discard34 in candidate_discards {
        let Some(delta) = discard_deltas
            .iter()
            .find(|item| item.tile34 as usize == discard34)
        else {
            continue;
        };
        let after_shanten = current_shanten + delta.shanten_diff;
        match best_after_shanten {
            None => {
                best_after_shanten = Some(after_shanten);
                selected.push(discard34);
            }
            Some(best) if after_shanten < best => {
                best_after_shanten = Some(after_shanten);
                selected.clear();
                selected.push(discard34);
            }
            Some(best) if after_shanten == best => selected.push(discard34),
            Some(_) => {}
        }
    }
    if selected.is_empty() {
        fallback
    } else {
        selected
    }
}

fn is_suited_sequence_start(tile34: usize) -> bool {
    tile34 < 27 && (tile34 % 9) <= 6
}

fn is_complete_regular_counts(
    counts: &mut [u8; TILE_COUNT],
    melds_needed: u8,
    need_pair: bool,
) -> bool {
    let first = counts.iter().position(|&cnt| cnt > 0);
    let Some(first) = first else {
        return melds_needed == 0 && !need_pair;
    };

    if need_pair && counts[first] >= 2 {
        counts[first] -= 2;
        if is_complete_regular_counts(counts, melds_needed, false) {
            counts[first] += 2;
            return true;
        }
        counts[first] += 2;
    }

    if melds_needed > 0 && counts[first] >= 3 {
        counts[first] -= 3;
        if is_complete_regular_counts(counts, melds_needed - 1, need_pair) {
            counts[first] += 3;
            return true;
        }
        counts[first] += 3;
    }

    if melds_needed > 0
        && is_suited_sequence_start(first)
        && counts[first + 1] > 0
        && counts[first + 2] > 0
    {
        counts[first] -= 1;
        counts[first + 1] -= 1;
        counts[first + 2] -= 1;
        if is_complete_regular_counts(counts, melds_needed - 1, need_pair) {
            counts[first] += 1;
            counts[first + 1] += 1;
            counts[first + 2] += 1;
            return true;
        }
        counts[first] += 1;
        counts[first + 1] += 1;
        counts[first + 2] += 1;
    }

    false
}

fn find_regular_waits(counts34: &[u8; TILE_COUNT]) -> [bool; TILE_COUNT] {
    let mut waits = [false; TILE_COUNT];
    let tile_count: usize = counts34.iter().map(|&v| usize::from(v)).sum();
    if tile_count % 3 != 1 {
        return waits;
    }

    let melds_needed = (((tile_count + 1) - 2) / 3) as u8;
    for tile34 in 0..TILE_COUNT {
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut work = *counts34;
        work[tile34] += 1;
        waits[tile34] = is_complete_regular_counts(&mut work, melds_needed, true);
    }
    waits
}

pub fn summarize_3n1(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> Summary3n1 {
    let tehai_count: u8 = counts34.iter().sum();
    if tehai_count == 0 {
        return Summary3n1 {
            shanten: 8,
            waits_count: 0,
            waits_tiles: [false; TILE_COUNT],
            tehai_count: 0,
            ukeire_type_count: 0,
            ukeire_live_count: 0,
            ukeire_tiles: [false; TILE_COUNT],
        };
    }

    let len_div3 = tehai_count / 3;
    let shanten = calc_shanten_all(counts34, len_div3);
    let regular_shanten = calc_shanten_normal(counts34, len_div3);
    let waits_tiles = if shanten <= 1 && regular_shanten == shanten {
        find_regular_waits(counts34)
    } else {
        [false; TILE_COUNT]
    };
    let waits_count = waits_tiles.iter().filter(|&&flag| flag).count() as u8;

    let mut ukeire_tiles = [false; TILE_COUNT];
    if shanten > 0 {
        if shanten > 2 {
            for item in calc_required_tiles(counts34, visible_counts34, len_div3) {
                ukeire_tiles[item.tile34 as usize] = true;
            }
        } else {
            for draw in calc_draw_deltas(counts34, visible_counts34, len_div3) {
                if draw.shanten_diff < 0 {
                    ukeire_tiles[draw.tile34 as usize] = true;
                    continue;
                }
                let mut counts_3n2 = *counts34;
                counts_3n2[draw.tile34 as usize] += 1;
                let discard_deltas = calc_discard_deltas(&counts_3n2, len_div3);
                if discard_deltas
                    .iter()
                    .any(|discard| draw.shanten_diff + discard.shanten_diff < 0)
                {
                    ukeire_tiles[draw.tile34 as usize] = true;
                }
            }
        }
    }

    let mut ukeire_type_count = 0u8;
    let mut ukeire_live_count = 0u8;
    for (tile34, &flag) in ukeire_tiles.iter().enumerate() {
        if !flag {
            continue;
        }
        ukeire_type_count += 1;
        ukeire_live_count =
            ukeire_live_count.saturating_add(4u8.saturating_sub(visible_counts34[tile34]));
    }

    Summary3n1 {
        shanten,
        waits_count,
        waits_tiles,
        tehai_count,
        ukeire_type_count,
        ukeire_live_count,
        ukeire_tiles,
    }
}

pub fn summarize_like_python(
    hand_counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> Summary3n1 {
    let tile_count: u8 = hand_counts34.iter().sum();
    match tile_count % 3 {
        1 => summarize_3n1(hand_counts34, visible_counts34),
        2 => {
            let mut best_summary: Option<Summary3n1> = None;
            let mut best_key: Option<(i32, i32, i32, i32)> = None;
            for discard34 in select_candidate_discards_3n2(hand_counts34) {
                let mut after_counts = *hand_counts34;
                after_counts[discard34] -= 1;
                let summary = summarize_3n1(&after_counts, visible_counts34);
                let key = candidate_progress_key(&summary);
                if best_key.as_ref().is_none_or(|current| key > *current) {
                    best_key = Some(key);
                    best_summary = Some(summary);
                }
            }

            best_summary.unwrap_or_else(|| summarize_3n1(hand_counts34, visible_counts34))
        }
        _ => Summary3n1 {
            shanten: 8,
            waits_count: 0,
            waits_tiles: [false; TILE_COUNT],
            tehai_count: tile_count,
            ukeire_type_count: 0,
            ukeire_live_count: 0,
            ukeire_tiles: [false; TILE_COUNT],
        },
    }
}

fn analyze_one_shanten_draw_metrics(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> OneShantenDrawAnalysis {
    let current = summarize_3n1(counts34, visible_counts34);
    if current.shanten != 1 {
        return OneShantenDrawAnalysis::default();
    }

    let len_div3 = counts34.iter().sum::<u8>() / 3;
    let current_key = candidate_progress_key(&current);
    let mut analysis = OneShantenDrawAnalysis::default();

    for draw in calc_draw_deltas(counts34, visible_counts34, len_div3) {
        if draw.live_count == 0 {
            continue;
        }
        let tile34 = draw.tile34 as usize;
        if counts34[tile34] >= 4 {
            continue;
        }
        let mut counts14 = *counts34;
        counts14[tile34] = counts14[tile34].saturating_add(1);
        let Some(after_best) = best_after_draw_summary(&counts14, visible_counts34) else {
            continue;
        };
        if after_best.shanten == 0 {
            analysis.tenpai_live = analysis.tenpai_live.saturating_add(draw.live_count);
            analysis.best_tenpai_waits_live = analysis
                .best_tenpai_waits_live
                .max(after_best.ukeire_live_count);
        }
        if after_best.shanten == 0 && after_best.ukeire_live_count > 4 {
            analysis.good_shape_live = analysis.good_shape_live.saturating_add(draw.live_count);
            continue;
        }
        if after_best.shanten == current.shanten
            && candidate_progress_key(&after_best) > current_key
        {
            analysis.improvement_live = analysis.improvement_live.saturating_add(draw.live_count);
        }
    }

    analysis
}

pub fn summarize_one_shanten_draw_metrics(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> (u8, u8) {
    let analysis = analyze_one_shanten_draw_metrics(counts34, visible_counts34);
    (analysis.good_shape_live, analysis.improvement_live)
}

pub fn summarize_one_shanten_tenpai_pressure(
    counts34: &[u8; TILE_COUNT],
    visible_counts34: &[u8; TILE_COUNT],
) -> (u8, u8) {
    let analysis = analyze_one_shanten_draw_metrics(counts34, visible_counts34);
    (analysis.tenpai_live, analysis.best_tenpai_waits_live)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(entries: &[(usize, u8)]) -> [u8; TILE_COUNT] {
        let mut tiles = [0u8; TILE_COUNT];
        for &(idx, count) in entries {
            tiles[idx] = count;
        }
        tiles
    }

    #[test]
    fn summarize_3n1_reports_waits_for_tenpai_hand() {
        let counts = hand(&[
            (4, 1),
            (5, 1),
            (17, 3),
            (21, 3),
            (22, 1),
            (23, 1),
            (24, 1),
            (32, 2),
        ]);
        let summary = summarize_3n1(&counts, &counts);
        assert_eq!(summary.shanten, 0);
        assert!(summary.waits_count > 0);
    }

    #[test]
    fn summarize_3n1_reports_ukeire_for_non_tenpai_hand() {
        let counts = hand(&[
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (9, 1),
            (10, 1),
            (12, 1),
            (13, 1),
            (18, 1),
        ]);
        let summary = summarize_3n1(&counts, &counts);
        assert!(summary.shanten > 0);
        assert!(summary.ukeire_type_count > 0);
    }

    #[test]
    fn summarize_like_python_uses_best_after_discard_for_14_tile_hand() {
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
            (9, 2),
            (10, 1),
            (11, 1),
            (22, 1),
        ]);
        let summary = summarize_like_python(&counts, &counts);
        assert_eq!(summary.tehai_count, 13);
        assert!(summary.shanten >= 0);
    }

    #[test]
    fn summarize_one_shanten_draw_metrics_detects_positive_shape_or_improvement() {
        let counts = hand(&[
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (9, 1),
            (10, 1),
            (15, 1),
            (24, 2),
        ]);
        let visible = counts;
        let (good_shape_live, improvement_live) =
            summarize_one_shanten_draw_metrics(&counts, &visible);
        assert!(good_shape_live > 0 || improvement_live > 0);
    }

    #[test]
    fn summarize_one_shanten_tenpai_pressure_detects_tenpai_draws() {
        let counts = hand(&[
            (0, 1),
            (1, 1),
            (3, 1),
            (11, 2),
            (12, 1),
            (13, 2),
            (20, 1),
            (21, 1),
            (22, 1),
            (24, 2),
        ]);
        let visible = counts;
        let (tenpai_live, best_tenpai_waits_live) =
            summarize_one_shanten_tenpai_pressure(&counts, &visible);
        assert!(tenpai_live > 0);
        assert!(best_tenpai_waits_live <= 34);
    }
}
