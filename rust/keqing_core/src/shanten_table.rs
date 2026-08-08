//! Table-driven shanten calculation adapted for local keqing_core use.
//!
//! The table layout and DP merge strategy follow the same approach used by
//! Mortal's libriichi crate, which is itself a Rust port of tomohxx's
//! shanten-number-calculator.

use std::io::Read;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

const JIHAI_TABLE_SIZE: usize = 78_032;
const SUHAI_TABLE_SIZE: usize = 1_940_777;

static JIHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_jihai.bin.gz"),
        JIHAI_TABLE_SIZE,
    )
});
static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = Vec::new();
    gz.read_to_end(&mut raw).unwrap();

    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);
    ret
}

pub fn ensure_init() {
    assert_eq!(JIHAI_TABLE.len(), JIHAI_TABLE_SIZE);
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();

    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
        }
        lhs[j] = sht;
    }

    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

fn add_jihai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = JIHAI_TABLE.get(index).copied().unwrap_or_default();

    let j = m + 5;
    let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
    }
    lhs[j] = sht;
}

#[must_use]
pub fn calc_shanten_normal(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, sum_tiles(&tiles[9..18]), len_div3);
    add_suhai(&mut ret, sum_tiles(&tiles[18..27]), len_div3);
    add_jihai(&mut ret, sum_tiles(&tiles[27..]), len_div3);

    (ret[5 + len_div3] as i8) - 1
}

#[must_use]
pub fn calc_shanten_chitoi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        if c >= 2 {
            pairs += 1;
        }
    });

    let reduct = 7_u8.saturating_sub(kinds) as i8;
    7 - pairs + reduct - 1
}

#[must_use]
pub fn calc_shanten_kokushi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;

    for &idx in &[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33] {
        let count = tiles[idx];
        if count > 0 {
            kinds += 1;
            if count >= 2 {
                pairs += 1;
            }
        }
    }

    let reduct = (pairs > 0) as i8;
    14 - kinds - reduct - 1
}

#[must_use]
pub fn calc_shanten_all(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let mut shanten = calc_shanten_normal(tiles, len_div3);
    if shanten <= 0 || len_div3 < 4 {
        return shanten;
    }

    shanten = shanten.min(calc_shanten_chitoi(tiles));
    if shanten > 0 {
        shanten.min(calc_shanten_kokushi(tiles))
    } else {
        shanten
    }
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
    fn normal_examples() {
        ensure_init();
        assert_eq!(
            calc_shanten_all(&hand(&[(0, 4), (11, 3), (19, 3), (30, 3)]), 4),
            1
        );
        assert_eq!(
            calc_shanten_all(
                &hand(&[
                    (0, 1),
                    (3, 1),
                    (6, 1),
                    (10, 1),
                    (13, 1),
                    (16, 1),
                    (20, 1),
                    (27, 1),
                    (28, 1),
                    (29, 1),
                    (30, 1),
                    (31, 1),
                    (32, 1)
                ]),
                4
            ),
            6
        );
    }

    #[test]
    fn chitoi_and_kokushi_examples() {
        ensure_init();
        let chitoi = hand(&[(0, 2), (1, 2), (2, 2), (9, 2), (10, 2), (18, 2), (27, 1)]);
        assert_eq!(calc_shanten_chitoi(&chitoi), 0);

        let kokushi = hand(&[
            (0, 1),
            (8, 1),
            (9, 1),
            (17, 1),
            (18, 1),
            (26, 1),
            (27, 1),
            (28, 1),
            (29, 1),
            (30, 1),
            (31, 1),
            (32, 1),
            (33, 1),
        ]);
        assert_eq!(calc_shanten_kokushi(&kokushi), 0);
    }
}
