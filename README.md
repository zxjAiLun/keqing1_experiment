# Keqing1 Experiment (keqing-mortal)

Training, self-play, and evaluation for the Mortal-based Riichi Mahjong stack.

This repository was split from `keqing1` at commit `b714e5c` (initial split)
and carries the Mortal training lineage from `keqing1`
`codex/mortal-training-next` @ `6ff580cb`, transferred at `74a3154`.
That commit is the baseline for all training work; the old branch is a
frozen reference. The Workbench repository is `keqing1-workbench`; runtime
data shared between them lives in `KEQING_DATA_ROOT` (defaults to the shared
`keqing-data` directory beside the project folder).

## Layout

- `training/` — Mortal training, self-play, evaluation, research notes, and
  runbooks (`mortal/` holds the per-run scripts).
- `src/keqing_core/` — shared core (shanten/legal-action primitives, cache
  schema), built into the `keqing_core` wheel via `rust/keqing_core`.
- `src/mahjong_env/` — Mahjong semantics shared with the Workbench repo.
  The initial split intentionally duplicates `mahjong_env`; no shared package
  is introduced until independent evolution demonstrates a real maintenance
  cost.
- `rust/keqing_core/` — Rust core; `build.py` produces the `keqing_core`
  wheel that the Workbench repo consumes from `keqing-data/runtime/keqing_core`.
- `third_party/Mortal/` — upstream Mortal runtime (git-ignored; `target/`
  build output excluded).
- `third_party/libriichi/` — Python shims for the compiled `riichi`
  extension; the extension itself is built from the vendored Mortal crate.
- `tests/` — training/core test suite (Mortal-adjacent plus `mahjong_env`).

## Quick start (Windows)

```powershell
.\scripts\setup-dev.ps1
cd training
python run_mortal_dqn_offline.py --help
```

`scripts/setup-dev.ps1`:

1. creates the venv and installs Python dependencies;
2. builds the `libriichi` runtime from the vendored Mortal crate (cargo);
3. builds and installs the `keqing_core` wheel, then publishes a copy to
   `KEQING_DATA_ROOT/runtime/keqing_core` for the Workbench repo.

`scripts/setup-dev.sh` is the Linux equivalent (no wheel publish step yet).

## Known issue / backlog

The pre-split arena scripts cross-checked native rank counts against the
platform rank-system ledger; that check was removed with the ownership split.
See `docs/backlog_removed_rank_crosscheck.md`.
