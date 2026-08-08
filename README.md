# Keqing Mortal

Training, self-play, and evaluation for the Mortal-based Riichi Mahjong stack.

This repository was split from `keqing1` at commit `b714e5c` (initial split).
The Workbench repository is `keqing-workbench`; runtime data shared between
them lives in `KEQING_DATA_ROOT` (defaults to the shared `keqing-data`
directory beside the project folder).

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
  wheel that the Workbench repo installs.
- `third_party/Mortal/` — upstream Mortal runtime (git-ignored, copied from
  the keqing1 workspace; `target/` build output excluded).
- `tests/` — training/core test suite (Mortal-adjacent plus `mahjong_env`).

## Quick start (Windows)

```powershell
.\scripts\setup-dev.ps1          # venv + deps + keqing_core wheel + libriichi runtime
cd training
python run_mortal_dqn_offline.py --help
```

`scripts/setup-dev.sh` is the Linux equivalent.

The setup script copies the compiled `libriichi`/`riichi` runtime bits from a
reference environment (default: the sibling `keqing1` venv, `..\keqing1\.venv-win`)
when they are missing; pass `-ReferenceVenv` to point elsewhere.

## Known issue / backlog

The pre-split arena scripts cross-checked native rank counts against the
platform rank-system ledger; that check was removed with the ownership split.
See `docs/backlog_removed_rank_crosscheck.md`.
