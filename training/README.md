# Training

This directory owns Mortal training and research work:

- `mortal/` — self-play, preparation, evaluation, and training utilities;
- the legacy training helpers at this directory root;
- `configs/`, `docs/`, `plans/`, and `reports/` that describe those runs.

Mutable datasets, checkpoints, run outputs, and logs belong below the
repository `data/` root, not beside this source code.  The shared Mortal
runtime remains in `src/` for this first migration step because Workbench
still imports it in-process.

Training writes eval results (logs, `detailed_stats`, `metrics.json`). Ladder
snapshots and platform account reports are built by Workbench
(`workbench/replay/`) from those results; arena scripts no longer produce
`platform_accounts/` themselves.

## Running from this directory

Entry scripts self-bootstrap their import paths (`training/` package, the
repository root, and `third_party/Mortal` where needed), so you can work from
this directory without returning to the repository root:

```bash
# from training/
uv run python run_mortal_dqn_offline.py --config ... --target-steps ...
uv run python mortal/four_player_native.py --model LABEL=CHECKPOINT ...
uv run python mortal/selfplay_native.py --model CHECKPOINT ...
uv run python mortal/run_grp_training.py ...
uv run python mortal/publish_ladder_snapshot.py --registry ...
```

`training/` scripts resolve `src/` packages through the editable install
created by `uv sync`; keep using `uv run` (or the project venv) rather than a
bare system python.

CLI defaults that are relative paths (`--mortal-root third_party/Mortal`,
`--output-dir artifacts/...`) still resolve against the repository root as
they did before the split; pass explicit paths when launching from here.
