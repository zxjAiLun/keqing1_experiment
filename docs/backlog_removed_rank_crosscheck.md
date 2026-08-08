# Removed: arena rank-counts cross-check against the platform ledger

## What changed and when

During the phase-3 ownership split (`keqing1` @ `b714e5c`), the ladder/report
publication machinery moved from `training/mortal/` to `workbench/replay/`, and
the arena scripts (`four_player_native.py`, `selfplay_native.py`) stopped
building platform account reports.

This removed one defensive check from the arena scripts: after a run, the
platform rank-system ledger counted placements and the arena compared those
counts against its own native rank counts, raising `RuntimeError` on mismatch.

## Why it was acceptable to drop

The cross-check depended on the platform account report, which is now owned by
the Workbench repo.  Keeping it would re-introduce a training -> workbench
import, which the split explicitly forbids.

## What to watch

- Arena rank counts now come from a single source (`libriichi.stat`) instead
  of two independent ones.
- If evaluation statistics ever look anomalous (e.g. wrong placement counts
  per model family), this redundancy is gone.  Reintroduce a training-side
  sanity check without the platform ledger if it proves valuable.
