//! Alignment tests for keqing_core shanten module.
//!
//! This module tests that Rust implementation matches Python's
//! _counts34_to_tile136_ids function output exactly.

#[cfg(test)]
mod tests {
    use _native::shanten::{counts34_to_ids, TILE_COUNT};

    /// Reference Python implementation for comparison.
    fn python_counts34_to_ids(counts34: &[i32; TILE_COUNT]) -> Vec<u16> {
        let total: usize = counts34.iter().map(|&c| c as usize).sum();
        let mut ids = Vec::with_capacity(total);

        for t in 0..TILE_COUNT {
            let cnt = counts34[t] as usize;
            if cnt > 0 && cnt < 5 {
                for copy_idx in 0..cnt {
                    ids.push((t * 4 + copy_idx) as u16);
                }
            }
        }

        ids
    }

    #[test]
    fn test_empty_hand() {
        let counts = [0i32; TILE_COUNT];
        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);
        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 0);
    }

    #[test]
    fn test_single_tile() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 1; // 1m x 1

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 1);
        assert_eq!(rust_result[0], 0); // 1m = tile 0, copy 0
    }

    #[test]
    fn test_two_tiles_same() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 2; // 1m x 2

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 2);
        assert_eq!(rust_result[0], 0); // 1m, copy 0
        assert_eq!(rust_result[1], 1); // 1m, copy 1
    }

    #[test]
    fn test_three_tiles_same() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 3; // 1m x 3

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 3);
        assert_eq!(rust_result[0], 0);
        assert_eq!(rust_result[1], 1);
        assert_eq!(rust_result[2], 2);
    }

    #[test]
    fn test_four_tiles_same() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 4; // 1m x 4

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 4);
    }

    #[test]
    fn test_sequence_tiles() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 1; // 1m
        counts[1] = 1; // 2m
        counts[2] = 1; // 3m

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 3);
        assert_eq!(rust_result, vec![0, 4, 8]); // 1m(0), 2m(4), 3m(8)
    }

    #[test]
    fn test_honor_tiles() {
        let mut counts = [0i32; TILE_COUNT];
        counts[27] = 1; // E (east wind)
        counts[28] = 1; // S (south wind)
        counts[33] = 1; // G (green dragon)

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 3);
    }

    #[test]
    fn test_all_suits() {
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 1; // 1m
        counts[9] = 1; // 1p
        counts[18] = 1; // 1s
        counts[27] = 1; // E

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 4);
    }

    #[test]
    fn test_full_hand() {
        // A typical 13-tile hand: 1123456789m + 1p + EE
        let mut counts = [0i32; TILE_COUNT];
        counts[0] = 2; // 1m x 2
        counts[1] = 1; // 2m x 1
        counts[2] = 1; // 3m x 1
        counts[3] = 1; // 4m x 1
        counts[4] = 1; // 5m x 1
        counts[5] = 1; // 6m x 1
        counts[6] = 1; // 7m x 1
        counts[7] = 1; // 8m x 1
        counts[8] = 1; // 9m x 1
        counts[9] = 1; // 1p x 1
        counts[27] = 2; // E x 2
                        // Total: 2+1+1+1+1+1+1+1+1+1+2 = 13 tiles

        let rust_result = counts34_to_ids(&counts);
        let python_result = python_counts34_to_ids(&counts);

        assert_eq!(rust_result, python_result);
        assert_eq!(rust_result.len(), 13);
    }

    #[test]
    fn test_random_hands_alignment() {
        // Test with 100 random hands to ensure alignment
        let mut rng = 42u64;

        for seed in 0..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);

            let mut counts = [0i32; TILE_COUNT];
            let mut remaining = [4i32; TILE_COUNT];
            let mut tile_count = 0;

            while tile_count < 13 {
                let idx = ((rng >> 16) as usize) % 34;
                if remaining[idx] > 0 {
                    counts[idx] += 1;
                    remaining[idx] -= 1;
                    tile_count += 1;
                }
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            }

            let rust_result = counts34_to_ids(&counts);
            let python_result = python_counts34_to_ids(&counts);

            assert_eq!(
                rust_result, python_result,
                "Mismatch at seed {}: {:?} vs {:?}",
                seed, rust_result, python_result
            );
        }
    }
}
