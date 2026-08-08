//! Batch helpers for progress-analysis candidate expansion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::counts::TILE_COUNT;
use crate::progress_delta::calc_discard_deltas;
use crate::progress_summary::summarize_3n1;
use crate::shanten_table::calc_shanten_all;

type CandidateProgressPy = (u8, Vec<u8>, i32, i32, i32, i32, i32, i32, i32, i32);

fn extract_counts34(seq: &Bound<'_, PyAny>, label: &str) -> PyResult<[i32; TILE_COUNT]> {
    let values: Vec<i32> = seq.extract()?;
    if values.len() != TILE_COUNT {
        return Err(PyValueError::new_err(format!(
            "{label} must contain exactly {TILE_COUNT} items, got {}",
            values.len()
        )));
    }
    let mut counts = [0i32; TILE_COUNT];
    for (idx, value) in values.into_iter().enumerate() {
        counts[idx] = value;
    }
    Ok(counts)
}

fn tile_in_obvious_meld(counts34: &[i32; TILE_COUNT], tile34: usize) -> bool {
    let cnt = counts34[tile34];
    if cnt <= 0 {
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

fn candidate_discards_no_meld_break(counts34: &[i32; TILE_COUNT]) -> Vec<usize> {
    let mut preferred = Vec::new();
    let mut fallback = Vec::new();
    let mut seen = [false; TILE_COUNT];

    for (tile34, &cnt) in counts34.iter().enumerate() {
        if cnt <= 0 || seen[tile34] {
            continue;
        }
        seen[tile34] = true;
        fallback.push(tile34);
        if !tile_in_obvious_meld(counts34, tile34) {
            preferred.push(tile34);
        }
    }

    if preferred.is_empty() {
        fallback
    } else {
        preferred
    }
}

fn to_u8_counts34(counts34: &[i32; TILE_COUNT]) -> [u8; TILE_COUNT] {
    let mut out = [0u8; TILE_COUNT];
    for (idx, value) in counts34.iter().enumerate() {
        out[idx] = (*value).max(0) as u8;
    }
    out
}

fn select_candidate_discards_3n2(counts34: &[i32; TILE_COUNT]) -> Vec<usize> {
    let candidate_discards = candidate_discards_no_meld_break(counts34);
    if candidate_discards.is_empty() {
        return candidate_discards;
    }

    let counts_u8 = to_u8_counts34(counts34);
    let len_div3 = counts_u8.iter().sum::<u8>() / 3;
    let current_shanten = calc_shanten_all(&counts_u8, len_div3);
    let discard_deltas = calc_discard_deltas(&counts_u8, len_div3);
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
            Some(best) if after_shanten == best => {
                selected.push(discard34);
            }
            Some(_) => {}
        }
    }

    if selected.is_empty() {
        candidate_discards_no_meld_break(counts34)
    } else {
        selected
    }
}

pub fn summarize_3n2_candidates_py_impl(
    _py: Python<'_>,
    counts34_seq: &Bound<'_, PyAny>,
    visible_counts34_seq: &Bound<'_, PyAny>,
    _summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<CandidateProgressPy>> {
    let counts34 = extract_counts34(counts34_seq, "counts34")?;
    let visible_counts34 = extract_counts34(visible_counts34_seq, "visible_counts34")?;
    let mut out = Vec::new();
    let visible_counts_u8 = to_u8_counts34(&visible_counts34);

    for discard34 in select_candidate_discards_3n2(&counts34) {
        let mut after_counts34 = counts34;
        after_counts34[discard34] -= 1;
        let after_counts_u8 = to_u8_counts34(&after_counts34);
        let summary = summarize_3n1(&after_counts_u8, &visible_counts_u8);

        out.push((
            discard34 as u8,
            after_counts34.iter().map(|&value| value as u8).collect(),
            summary.shanten as i32,
            summary.waits_count as i32,
            summary.ukeire_type_count as i32,
            summary.ukeire_live_count as i32,
            0,
            0,
            0,
            0,
        ));
    }

    Ok(out)
}

fn candidate_progress_key(item: &CandidateProgressPy) -> (i32, i32, i32, i32, i32, i32) {
    (-item.2, item.5, item.4, item.3, 0, 0)
}

pub fn summarize_best_3n2_candidate_py_impl(
    py: Python<'_>,
    counts34_seq: &Bound<'_, PyAny>,
    visible_counts34_seq: &Bound<'_, PyAny>,
    summarize_fn: &Bound<'_, PyAny>,
) -> PyResult<Option<CandidateProgressPy>> {
    let candidates =
        summarize_3n2_candidates_py_impl(py, counts34_seq, visible_counts34_seq, summarize_fn)?;
    let mut best: Option<CandidateProgressPy> = None;
    let mut best_key: Option<(i32, i32, i32, i32, i32, i32)> = None;

    for candidate in candidates {
        let key = candidate_progress_key(&candidate);
        if best_key.as_ref().is_none_or(|current| key > *current) {
            best_key = Some(key);
            best = Some(candidate);
        }
    }

    Ok(best)
}
