# Local runtime data

This directory is intentionally Git-ignored except for this guide. Set
`KEQING_DATA_ROOT` to override it; otherwise the repository uses this path as
the default Workbench data base.

```text
data/
  models/        Mortal checkpoints, grouped by model name
  datasets/      training inputs
  runs/          training outputs
  ladder/        registry-adjacent snapshots and reports
  participants/  account/model/match ledger files
  replays/       uploaded and generated replay data
  captures/      Play-with-you captures
  logs/          local runtime logs
```

Existing `artifacts/` content is deliberately not copied or deleted by this
layout change. Participant, replay, ladder, and Play-with-you log state now
resolve below this data base by default; model checkpoint, dataset, and
training-run consumers continue to use their legacy paths until each is
explicitly migrated.
