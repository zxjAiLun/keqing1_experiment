# keqing_core provenance notes

This crate now contains a table-driven shanten implementation and embedded
table assets under `src/data/`.

## Source lineage

- The shanten table layout and DP merge strategy follow the approach used by
  `third_party/Mortal/libriichi/src/algo/shanten.rs`.
- Mortal documents that its shanten implementation is a Rust port of
  `tomohxx/shanten-number`, and that the original source lineage is GPL/LGPL
  related. See:
  - `third_party/Mortal/docs/src/ref/meta.md`
  - `third_party/Mortal/libriichi/Cargo.toml`
- The embedded table assets currently vendored into this crate:
  - `src/data/shanten_jihai.bin.gz`
  - `src/data/shanten_suhai.bin.gz`
  were copied from:
  - `third_party/Mortal/libriichi/src/algo/data/`

## Licensing decision for this crate

Because the current implementation and embedded assets are derived from that
lineage, `rust/keqing_core` is treated conservatively as:

- `AGPL-3.0-or-later`

This is intentionally narrower than the top-level repository and applies to
this crate/component specifically.

## Future cleanup option

If the project later replaces these assets/implementation with an
independently generated table pipeline or a separately licensed source, this
notice and the crate license should be revisited in the same change.
