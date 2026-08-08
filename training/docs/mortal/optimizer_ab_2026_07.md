# Fresh Adam vs Preserved Adam

## Scope

This phase tests only optimizer-state transfer during a controlled continuation from the 70k parent:

- reward: `final_rank_mc`
- parent: `mortal_default_70k_promoted_candidate.pth`, step `70000`
- target: step `72000`
- groups: fresh Adam versus preserved Adam moments
- training seeds: `20260724` through `20260729`
- evaluation: 1000 hanchans per seed, native four-model random-seat lineup
- lineup: `70k`, external Mortal, the matched fresh checkpoint, and the matched preserved checkpoint
- rank points: `[90, 45, 0, -135]`

The six matched A/B pairs were trained from the same parent, data stream, model seed, and training configuration. The only intended variable was whether Adam state was initialized empty or restored from the legacy 70k parent. All six training contracts recorded `git_dirty=false`.

## Six-Seed Result

`delta_pt` is calculated per complete hanchan as `Pt(preserved) - Pt(fresh)`.

| Training seed | Fresh avg Pt | Preserved avg Pt | Preserved - fresh | Preserved ahead rate |
| ---: | ---: | ---: | ---: | ---: |
| 20260724 | -0.315 | +4.230 | +4.545 | 50.9% |
| 20260725 | -4.140 | -2.475 | +1.665 | 50.7% |
| 20260726 | -7.335 | -1.800 | +5.535 | 51.4% |
| 20260727 | -7.515 | -3.555 | +3.960 | 50.1% |
| 20260728 | -5.715 | -1.575 | +4.140 | 51.7% |
| 20260729 | -5.445 | -2.340 | +3.105 | 51.6% |

All six seed-level averages favor preserved Adam.

- Mean of seed means: `+3.825 Pt/局`.
- Median of seed means: `+4.050 Pt/局`.
- Pooled complete-hanchan bootstrap 95% CI: `[+0.285, +7.343] Pt/局`.
- Equal-seed hierarchical bootstrap 95% CI: `[+0.285, +7.342] Pt/局`.
- Exact one-sided seed-direction sign test: `6/6`, `p=0.015625`.

The hanchan interval measures arena uncertainty conditional on these checkpoints. The hierarchical interval additionally resamples the six training seeds equally. Both intervals are now positive under this fixed six-seed experiment, while the sign test provides a consistent direction across all training seeds.

## Against the 70k Anchor

The same 6000 hanchans also allow a paired comparison against the 70k seat in each lineup. A negative rank delta means the candidate finished better than 70k.

| Training seed | Preserved - 70k Pt | Fresh - 70k Pt | Preserved - 70k rank | Fresh - 70k rank |
| ---: | ---: | ---: | ---: | ---: |
| 20260724 | +9.855 | +5.310 | -0.115 | -0.060 |
| 20260725 | +3.060 | +1.395 | -0.044 | -0.037 |
| 20260726 | +0.630 | -4.905 | +0.044 | +0.095 |
| 20260727 | -2.115 | -6.075 | +0.065 | +0.093 |
| 20260728 | +1.710 | -2.430 | +0.006 | +0.070 |
| 20260729 | -1.035 | -4.140 | +0.037 | +0.084 |

Pooled preserved-minus-70k is `+2.018 Pt/局`, with hanchan bootstrap CI `[-1.508, +5.407]` and equal-seed hierarchical CI `[-2.618, +6.848]`. Pooled fresh-minus-70k is `-1.808 Pt/局`, with hanchan bootstrap CI `[-5.445, +1.680]` and equal-seed hierarchical CI `[-6.315, +3.015]`.

This supports the narrower conclusion that preserved Adam reduces 70k continuation degradation relative to fresh Adam. It does not establish that a particular preserved checkpoint is a stable replacement for 70k.

## Decision

Promote preserved Adam as the default optimizer initialization for **70k legacy continuation** experiments. This is a recipe promotion only:

- future 70k continuation runs restore the 70k Adam state by default;
- `final_rank_mc`, LR, CQL, architecture, data index, and target-step settings remain unchanged;
- no preserved checkpoint is promoted to serving/default status from this A/B;
- no additional optimizer-seed expansion is planned.

The result does not justify another local optimizer grid. The next research phase should return to project-owned lineage and data distribution, using the preserved-Adam continuation recipe as the fixed operational baseline where 70k continuation is still required.

## Behavior Readout

Preserved Adam generally recovers part of the continuation loss seen with fresh Adam, but the change is not a single monotonic behavior shift. Across the six lineups, agari, houjuu, fuuro, riichi, after-riichi, and after-fuuro metrics remain seed-dependent. These are diagnostic readouts, not evidence that preserved Adam discovered a new playing style.

## Reproducibility

The six-seed machine-generated summary is kept locally at:

`artifacts/experiments/model_pool_2026_07/optimizer_ab_2026_07_replication/eval_1000h/summary_six_seed/optimizer_ab_eval_1000h_summary.json`

and:

`artifacts/experiments/model_pool_2026_07/optimizer_ab_2026_07_replication/eval_1000h/summary_six_seed/optimizer_ab_eval_1000h_summary.md`

The raw logs and checkpoints remain local artifacts and are intentionally not added to Git.
