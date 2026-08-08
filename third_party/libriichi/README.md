# libriichi Python shims

`libriichi` is the compiled `riichi` extension built from the vendored
`third_party/Mortal` Rust crate.  These tiny re-export modules give it the
`libriichi.*` package surface that Mortal tooling imports.

- Build the extension: `cargo build --manifest-path third_party/Mortal/Cargo.toml -p libriichi --lib --release`
- Install: copy `third_party/Mortal/target/release/riichi.dll` to
  `site-packages/riichi.pyd` and this directory to `site-packages/libriichi/`.

`scripts/setup-dev.ps1` does both.
