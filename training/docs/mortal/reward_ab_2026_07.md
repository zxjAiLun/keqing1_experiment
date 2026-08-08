# Reward Semantics A/B

## Purpose

The next experiment isolates reward semantics before changing CQL, learning rate, auxiliary loss, or network architecture.

The two primary groups are:

| Group | Reward | Initialization |
| --- | --- | --- |
| F | `final_rank_mc` | 70k weights, fresh Adam, fresh data stream |
| G | `mortal_grp_delta_pt` | 70k weights, fresh Adam, fresh data stream |

Both groups must use the same 6,000-hanchan file index, `num_epochs=2`, matched data/model seeds, `gamma=1.0`, CQL weight `5.0`, next-rank weight `0.2`, and constant learning rate `1e-4`. The first comparison stops at step 72,000. Three matched seed pairs are required before extending any run to 74,000.

## Implementation Status

- `training/mortal/mainline_dataloader.py` now supports `mortal_grp_delta_pt` by reusing Mortal's `GRP` and `RewardCalculator`.
- `training/mortal/test_reward_adapter.py` verifies upstream/project parity, telescoping, and data identity.
- `training/mortal/preflight_reward_distribution.py` reports reward mean, standard deviation, quantiles, nonzero rate, absolute delta-Pt quantiles, and per-hanchan absolute movement before training.
- `training/mortal/prepare_reward_ab.py` prepares the six matched-seed configs and a shared file index.
- `training/mortal/prepare_grp_v1.py` creates the independent GRP train/validation/holdout split, and `training/mortal/run_grp_training.py` trains the project-owned GRP checkpoint.
- `training/mortal/evaluate_grp_checkpoint.py` evaluates validation and holdout without changing checkpoint selection.
- `scripts/run_mortal_dqn_offline.py` supports `--initialize-optimizer-from`, which preserves only Adam moments while keeping scheduler and data stream fresh.
- Checkpoints now record reward, GRP hash, file-index hash, dataset manifest hash, initialization mode, project commit, Mortal revision, and libriichi revision.

## GRP Checkpoint

The workspace did not contain an upstream GRP checkpoint, so the project-owned `keqing_grp_v1` was trained from an independent 2,000-hanchan corpus. Its validation/holdout results and reward preflight are recorded in [`grp_v1_2026_07.md`](grp_v1_2026_07.md). All G runs use the same frozen checkpoint SHA256 recorded in `manifest.json`. A first attempt with `num_epochs=1` ended normally at step 71,832 and is retained separately as a configuration audit; the formal retry is `reward_ab_2026_07_epoch2`.

## Commands

Adapter correctness test:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python training/mortal/test_reward_adapter.py `
  --output artifacts/experiments/model_pool_2026_07/reward_adapter_test.json
```

Prepare or refresh the matched A/B matrix with the frozen project GRP checkpoint:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python training/mortal/prepare_reward_ab.py `
  --grp-checkpoint artifacts/experiments/model_pool_2026_07/keqing_grp_v1/keqing_grp_v1_best.pth
```

The command only writes configs and manifests. It does not start training.

## 72000 Checkpoint Audit

The formal retry completed all six checkpoints in `reward_ab_2026_07_epoch2`:

- F: `final_rank_mc`, seeds `20260718/19/20`.
- G: `mortal_grp_delta_pt`, seeds `20260718/19/20`.
- Every run reached step `72000`, consumed the same 6,000-file index for two epochs, and used the same 70k parent.
- Every G run uses the frozen `keqing_grp_v1` SHA256 recorded in the audit.
- The contract audit remains a local-only artifact at `artifacts/experiments/model_pool_2026_07/reward_ab_2026_07_epoch2/reward_ab_audit.json`; raw audit artifacts are intentionally not uploaded.

## 1000-Hanchan Matched Evaluation

The three matched-seed native random-seat evaluations were extended from 250 to a fixed 1,000 hanchans each, keeping the lineup `70k + ext_mortal + F + G`. The existing 250 logs were resumed, not regenerated:

| Pair | F avg rank | G avg rank | F avg Pt | G avg Pt | G-F Pt |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260718` | 2.551 | 2.460 | -2.745 | +2.880 | +5.625 |
| `20260719` | 2.535 | 2.533 | -1.665 | -1.845 | -0.180 |
| `20260720` | 2.546 | 2.547 | -3.600 | -3.465 | +0.135 |

The paired statistic is calculated per hanchan as `delta_pt = Pt(G) - Pt(F)`. Over 3,000 paired hanchans:

- Mean paired delta: `+1.86 Pt/局`.
- Hanchan-cluster bootstrap 95% CI: `[-3.17, +6.86] Pt/局`.
- G finished ahead in `50.67%` of hanchans.
- Training-seed means: `[+5.625, -0.180, +0.135]`; positive seeds `2/3`.
- One-sided three-seed exact sign-test p-value: `0.500` for `2/3` positive seeds.

Pooled behavior remains close: G agari `20.49%` vs F `20.40%`, houjuu `12.22%` vs `12.18%`, fuuro `25.53%` vs `24.93%`, and riichi `18.96%` vs `19.70%`. The earlier lower-houjuu signal did not remain stable after extension.

Conclusion: `mortal_grp_delta_pt` remains a valid implemented reward alternative, but it does **not** pass the current evidence threshold for promotion to the default research reward. The arena-level CI includes zero and the training-seed result is mixed. Do not start Adam-preserved, LR, CQL, or architecture variants yet. The next controlled step is to add 2-3 new matched F/G training seeds, or explicitly stop the reward hypothesis and return to data/optimizer diagnostics.

The complete paired report is [`reward_ab_eval_1000h_summary.md`](../../reports/mortal/reward_ab_2026_07_epoch2/reward_ab_eval_1000h_summary.md), with machine-readable output in the adjacent JSON file. The raw 250/1000-hanchan evaluation artifacts remain local-only under `artifacts/` for auditability.

## Epoch3 Seed Expansion And Six-Seed Result

The planned follow-up added three new matched pairs without changing the recipe:

- F/G seeds `20260721/22/23`, 70k weights-only warm start, fresh Adam, target step `72000`.
- The same frozen GRP checkpoint, 6,000-file index, two epochs, and clean training commit were used for all six new checkpoints.
- All six new contracts recorded `git_dirty=false`; the checkpoint audit passed.
- Each new pair was evaluated for exactly 1,000 native random-seat hanchans with non-overlapping seed ranges `923000/924000/925000`.

The new paired differences were:

| Pair | F avg Pt | G avg Pt | G-F Pt |
| --- | ---: | ---: | ---: |
| `20260721` | -6.210 | +1.890 | +8.100 |
| `20260722` | +0.585 | -2.475 | -3.060 |
| `20260723` | -3.825 | -1.035 | +2.790 |

Combining epoch2 and epoch3 gives six training-seed pairs and 6,000 paired hanchans:

- Seed means: `[+5.625, -0.180, +0.135, +8.100, -3.060, +2.790]`.
- Mean of seed means: `+2.235 Pt/局`; median: `+1.463 Pt/局`.
- Positive non-tie seeds: `4/6`; exact one-sided sign-test p-value: `0.34375`.
- Pooled hanchan bootstrap 95% CI: `[-1.25, +5.85] Pt/局`.
- Equal-seed hierarchical bootstrap 95% CI: `[-2.37, +6.81] Pt/局`.
- G finished ahead in `50.58%` of paired hanchans.

Conclusion: the GRP reward still shows a positive central tendency, but the additional seeds do not establish a stable recipe improvement. Keep `final_rank_mc` as the conservative operational default, without claiming that it has been proven superior. Do not start more GRP seeds, Adam/LR/CQL variants, or promote a G checkpoint. The six-seed machine-readable summary is local-only at `artifacts/experiments/model_pool_2026_07/reward_ab_2026_07_epoch3/eval_1000h/summary_six_seeds/reward_ab_eval_1000h_summary.json`.

## Checkpoint Drift Audit

After closing the reward hypothesis, a pure analysis pass compared the 12 F/G checkpoints at step `72000` against the same 70k parent on a deterministic probe of `4096` states sampled from `128` arena hanchans. These logs are outside the 6000-file offline training index. The audit artifact is local-only at `artifacts/experiments/model_pool_2026_07/checkpoint_drift_audit/checkpoint_drift_audit.json` and the implementation is `training/mortal/audit_checkpoint_drift.py`.

The main findings are:

- Greedy action changes versus 70k are `7.7%` to `9.1%`, so the 2000 offline updates produce a measurable but not wholesale policy shift.
- Brain relative parameter L2 is approximately `2.86%` for every run; DQN is `11.4%` to `11.9%`.
- AuxNet drift is `16.4%` to `18.2%` for F and `11.1%` to `12.0%` for G.
- Mean absolute Q drift is `2.10` to `2.56` for F and `3.06` to `3.41` for G. Mean margins also increase, approximately `+1.62` to `+1.99` for F and `+2.04` to `+2.37` for G.
- Drift is not isolated to one late-game slice. Early/middle/late greedy-change rates are similar. Rank-4 and large-behind states have larger F Q drift than rank-1/ahead states, while G Q drift is comparatively flat.
- States after the target player's riichi have very low greedy-change rates (`about 0.6%` to `0.8%`) and lower Q drift than the full probe, so the current audit does not support an `after-riichi`-specific optimizer claim.

This separates the next questions. The checkpoints are not nearly identical, but the action-policy change is much smaller than the parameter/Q movement. That is consistent with value-scale and representation drift under the offline target, not proof that fresh Adam is the cause. The small six-seed sample also cannot support a drift-performance correlation. Therefore the next experiment, if run, must be one matched `final_rank_mc` fresh-Adam versus preserved-Adam comparison with all other variables fixed; no reward change is bundled into it. If that comparison remains seed-sensitive, stop local reward/optimizer tuning and move to a new data or project-owned lineage rather than opening another recipe grid.
