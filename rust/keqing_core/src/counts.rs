//! Count-based tile primitives used by native Keqing helpers.

/// Tile counts in 34-tile format (0-8: manzu, 9-17: pinzu, 18-26: souzu, 27-33: honors)
pub const TILE_COUNT: usize = 34;

/// Tile counts in 34-tile format.
#[derive(Debug, Clone, Copy)]
pub struct Counts34([i32; TILE_COUNT]);

impl Counts34 {
    pub fn new() -> Self {
        Self([0; TILE_COUNT])
    }

    pub fn from_array(arr: &[i32; TILE_COUNT]) -> Self {
        Self(*arr)
    }

    pub fn as_array(&self) -> &[i32; TILE_COUNT] {
        &self.0
    }
}

impl Default for Counts34 {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&[i32; TILE_COUNT]> for Counts34 {
    fn from(arr: &[i32; TILE_COUNT]) -> Self {
        Self::from_array(arr)
    }
}
