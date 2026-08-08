//! Xmodel1 schema constants.
//!
//! This module defines the Rust-side constants for the candidate-centric Xmodel1
//! cache protocol. The first implementation phase keeps the module deliberately
//! small so the Python and Rust layers can converge on the same field contract
//! before the export logic is filled in.

pub const XMODEL1_SCHEMA_NAME: &str = "xmodel1_discard_v6";
pub const XMODEL1_SCHEMA_VERSION: u32 = 6;
pub const XMODEL1_MAX_CANDIDATES: usize = 14;
pub const XMODEL1_CANDIDATE_FEATURE_DIM: usize = 22;
pub const XMODEL1_CANDIDATE_FLAG_DIM: usize = 8;
pub const XMODEL1_MAX_SPECIAL_CANDIDATES: usize = 12;
pub const XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM: usize = 19;
pub const XMODEL1_HISTORY_SUMMARY_DIM: usize = 20;
pub const XMODEL1_MAX_RESPONSE_CANDIDATES: usize = 8;

pub const XMODEL1_SPECIAL_TYPE_REACH: i16 = 0;
pub const XMODEL1_SPECIAL_TYPE_DAMA: i16 = 1;
pub const XMODEL1_SPECIAL_TYPE_HORA: i16 = 2;
pub const XMODEL1_SPECIAL_TYPE_CHI_LOW: i16 = 3;
pub const XMODEL1_SPECIAL_TYPE_CHI_MID: i16 = 4;
pub const XMODEL1_SPECIAL_TYPE_CHI_HIGH: i16 = 5;
pub const XMODEL1_SPECIAL_TYPE_PON: i16 = 6;
pub const XMODEL1_SPECIAL_TYPE_DAIMINKAN: i16 = 7;
pub const XMODEL1_SPECIAL_TYPE_ANKAN: i16 = 8;
pub const XMODEL1_SPECIAL_TYPE_KAKAN: i16 = 9;
pub const XMODEL1_SPECIAL_TYPE_RYUKYOKU: i16 = 10;
pub const XMODEL1_SPECIAL_TYPE_NONE: i16 = 11;

pub const XMODEL1_SPECIAL_TYPE_CHI: i16 = XMODEL1_SPECIAL_TYPE_CHI_LOW;
pub const XMODEL1_SPECIAL_TYPE_KAN: i16 = XMODEL1_SPECIAL_TYPE_DAIMINKAN;

pub fn validate_candidate_mask_and_choice(
    chosen_candidate_idx: i16,
    candidate_mask: &[u8],
    candidate_tile_id: &[i16],
) -> Result<(), String> {
    if candidate_mask.len() != XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "candidate_mask length {} != {}",
            candidate_mask.len(),
            XMODEL1_MAX_CANDIDATES
        ));
    }
    if candidate_tile_id.len() != XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "candidate_tile_id length {} != {}",
            candidate_tile_id.len(),
            XMODEL1_MAX_CANDIDATES
        ));
    }
    let idx = usize::try_from(chosen_candidate_idx)
        .map_err(|_| format!("chosen_candidate_idx {} is negative", chosen_candidate_idx))?;
    if idx >= XMODEL1_MAX_CANDIDATES {
        return Err(format!(
            "chosen_candidate_idx {} out of range for {} candidates",
            chosen_candidate_idx, XMODEL1_MAX_CANDIDATES
        ));
    }
    if candidate_mask[idx] == 0 {
        return Err(format!(
            "chosen_candidate_idx {} points to masked candidate",
            chosen_candidate_idx
        ));
    }
    for i in 0..XMODEL1_MAX_CANDIDATES {
        let masked = candidate_mask[i] == 0;
        if masked && candidate_tile_id[i] != -1 {
            return Err(format!(
                "padding candidate {} must use tile_id=-1, got {}",
                i, candidate_tile_id[i]
            ));
        }
        if !masked && !(0..=33).contains(&candidate_tile_id[i]) {
            return Err(format!(
                "active candidate {} has invalid tile_id {}",
                i, candidate_tile_id[i]
            ));
        }
    }
    Ok(())
}
