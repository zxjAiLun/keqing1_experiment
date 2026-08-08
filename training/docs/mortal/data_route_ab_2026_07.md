# M0 Mixed vs S0 Pure Selfplay Route A/B

## Question

This experiment tests whether the retained mixed replay ecology should be
replaced by a pure `ext_mortal` selfplay corpus. It is a data-distribution
comparison, not a reviewer-label experiment and not a checkpoint promotion.

- `M0`: existing audited 6,000-hanchan mixed corpus, training perspective
  `ext_mortal`.
- `S0`: 6,000 new independent native `ext_mortal` selfplay hanchans, one
  randomly assigned `train_ext` perspective per hanchan.
- Both routes: 70k weights plus the same 70k Adam state, fresh scheduler,
  scaler, data stream and RNG, `final_rank_mc`, target step 72,000.
- Matched training seeds: `20260731`, `20260801`, `20260802`.
- Evaluation: 1,000 native random-seat hanchans per seed with
  `70k / ext_mortal / M_candidate / S_candidate` and rank points
  `[90,45,0,-135]`.

The S0 corpus audit found 6,000 files, 6,000 unique seeds, 6,000 canonical
unique hanchans, zero malformed logs, zero state-only duplicates and exactly
1,500 training perspectives in each seat. Its action support was close to M0:
70k greedy agreement was `89.1566%`, mean behavior Q rank `1.137`, and mean
greedy-minus-behavior Q regret `0.0601`. The main change was opponent ecology
and outcome calibration, not obvious new action novelty.

## Paired Results

All differences are computed per complete hanchan and then averaged. No seat
is treated as an independent sample. The full machine-readable report is the
local artifact:

`artifacts/experiments/model_pool_2026_07/data_route_ab_2026_07/summary/data_route_ab_summary.md`

| Training seed | S-M Pt | S-70k Pt | M-70k Pt |
| ---: | ---: | ---: | ---: |
| 20260731 | +0.810 | +7.020 | +6.210 |
| 20260801 | -4.410 | -0.990 | +3.420 |
| 20260802 | -3.375 | -3.690 | -0.315 |
| **seed mean** | **-2.325** | **+0.780** | **+3.105** |

The equal-seed hierarchical 95% intervals are:

- S-M: `[-7.815, +3.315]` Pt;
- S-70k: `[-6.045, +7.906]` Pt;
- M-70k: `[-2.821, +8.880]` Pt.

S-M is positive in `1/3` training seeds; the one-sided exact sign-test p-value
is `0.875`. These results do not support replacing M0 with pure selfplay.

## Behavior Readout

Across the 3,000 evaluated hanchans:

| Model | Avg Pt | Avg rank | Agari | Houjuu | Fuuro | Riichi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 70k | -3.405 | 2.541 | 21.723% | 13.497% | 29.010% | 18.403% |
| ext_mortal | +6.330 | 2.417 | 21.882% | 11.765% | 29.505% | 18.523% |
| M0 candidate | -0.300 | 2.502 | 21.255% | 12.974% | 28.343% | 20.436% |
| S0 candidate | -2.625 | 2.540 | 20.685% | 12.727% | 24.944% | 20.258% |

S0 has lower fuuro and slightly lower agari than M0, with a small reduction in
houjuu. Its after-riichi readout is close to M0, so this is not evidence of a
specific successful defensive mechanism. The main observable style shift is a
large reduction in fuuro rather than a strength improvement.

## Decision

1. Do not promote S0 or the pure-selfplay route.
2. Do not generate another 6,000-hanchan pure `ext_mortal` pool solely to
   repeat this comparison.
3. Keep M0 and the preserved-Adam continuation contract as the operational
   reference for the next controlled study.
4. Freeze this route result and move the next research design toward a new
   project-owned lineage or a deliberately different data ecology. Do not
   reopen reviewer/teacher CE, GRP, LR or CQL grids based on this result.

The result does not prove that pure selfplay can never help. It shows that,
under the current 6,000-hanchan size, 70k warm-start, 2,000-step continuation
and final-rank MC objective, pure `ext_mortal` selfplay did not produce a
stable advantage over the existing mixed corpus.
