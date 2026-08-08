//! Standard shanten support helpers shared across native entrypoints.

use crate::counts::{Counts34, TILE_COUNT};

/// Precomputed tile to 136 ID mapping.
/// For each tile type (0-33) and count (0-4), precompute the 136 IDs.
static COUNT34_TO_136: [[u16; 5]; TILE_COUNT] = precompute_lookup();

const fn precompute_lookup() -> [[u16; 5]; TILE_COUNT] {
    let mut result = [[0u16; 5]; TILE_COUNT];
    let mut t = 0;
    while t < TILE_COUNT {
        let mut cnt = 0;
        while cnt < 5 {
            result[t][cnt] = ((t * 4) + cnt) as u16;
            cnt += 1;
        }
        t += 1;
    }
    result
}

/// Convert 34-tile counts to 136-tile IDs.
#[inline]
pub fn counts34_to_ids(counts34: &[i32; TILE_COUNT]) -> Vec<u16> {
    let total: usize = counts34.iter().map(|&c| c as usize).sum();
    let mut ids = Vec::with_capacity(total);

    for t in 0..TILE_COUNT {
        let cnt = counts34[t] as usize;
        if cnt > 0 && cnt < 5 {
            ids.extend_from_slice(&COUNT34_TO_136[t][0..cnt]);
        }
    }

    ids
}

impl Counts34 {
    pub fn to_ids(&self) -> Vec<u16> {
        counts34_to_ids(self.as_array())
    }
}

impl From<&Counts34> for Vec<u16> {
    fn from(counts: &Counts34) -> Self {
        counts.to_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counts34_to_ids() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 2;
        counts[1] = 1;
        counts[2] = 1;

        let ids = counts34_to_ids(&counts);
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn test_counts34_empty() {
        let counts = [0i32; TILE_COUNT];
        let ids = counts34_to_ids(&counts);
        assert_eq!(ids.len(), 0);
    }
}
