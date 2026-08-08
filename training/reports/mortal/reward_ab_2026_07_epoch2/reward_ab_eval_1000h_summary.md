# Reward Semantics A/B: 1000-Hanchan Screening

This is a matched-seed native random-seat screening evaluation.
All rows are reported as separate F/G results; no two-way aggregate is used.

## Per Pair

| Pair | F avg rank | G avg rank | F avg Pt | G avg Pt | G-F Pt | G-F agari | G-F houjuu | G-F fuuro | G-F riichi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_G_20260718 | 2.551 | 2.460 | -2.75 | 2.88 | +5.62 | +0.21pp | -0.44pp | +0.46pp | -0.60pp |
| F_G_20260719 | 2.535 | 2.533 | -1.67 | -1.84 | -0.18 | -0.24pp | -0.10pp | +0.27pp | -1.46pp |
| F_G_20260720 | 2.546 | 2.547 | -3.60 | -3.46 | +0.14 | +0.28pp | +0.64pp | +1.08pp | -0.14pp |

## Paired Hanchan Differential

`delta_pt = Pt(G) - Pt(F)` and `delta_rank = rank(F) - rank(G)`. Bootstrap resamples complete hanchans, not individual seats.

| Scope | Games | Mean delta Pt | Median delta Pt | Mean delta rank | G ahead | 95% CI for mean delta Pt | Rank-rate diff (1st/2nd/3rd/4th) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F_G_20260718 | 1000 | +5.62 | +45.00 | +0.091 | 51.20% | [-2.83, +13.90] | ['+3.80pp', '-0.20pp', '-1.90pp', '-1.70pp'] |
| F_G_20260719 | 1000 | -0.18 | -45.00 | +0.002 | 49.60% | [-8.73, +8.23] | ['+2.10pp', '-3.70pp', '+1.30pp', '+0.30pp'] |
| F_G_20260720 | 1000 | +0.14 | +45.00 | -0.001 | 51.20% | [-8.37, +8.91] | ['-0.40pp', '+0.50pp', '+0.10pp', '-0.20pp'] |
| pooled_hanchans | 3000 | +1.86 | +45.00 | +0.031 | 50.67% | [-3.17, +6.86] | ['+1.83pp', '-1.13pp', '-0.17pp', '-0.53pp'] |

## Training-Seed View

- Seed-level mean delta Pt: `[5.62, -0.18, 0.14]`.
- Positive non-tie seed count: `2/3`; one-sided sign-test p-value under the zero-direction null: `0.5000`.
- The hanchan bootstrap CI measures arena uncertainty conditional on these checkpoints; it does not remove the separate training-seed uncertainty.

## Pooled Auxiliary View

| Model | Games | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | Rank counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 70k | 3000 | 2.509 | -1.32 | 22.08% | 12.79% | 29.60% | 18.44% | [761, 730, 729, 780] |
| ext_mortal | 3000 | 2.433 | 4.80 | 21.60% | 12.35% | 28.90% | 19.28% | [816, 758, 736, 690] |
| F | 3000 | 2.544 | -2.67 | 20.40% | 12.18% | 24.93% | 19.70% | [684, 773, 770, 773] |
| G | 3000 | 2.513 | -0.81 | 20.49% | 12.22% | 25.53% | 18.96% | [739, 739, 765, 757] |

## Reading

- G is ahead of F on average rank Pt in 2/3 matched pairs.
- This is a direction check; reward promotion still requires interpreting both the paired hanchan CI and the separate training-seed uncertainty.
- The 70k and ext_mortal rows are controls for this lineup, not a claim that this screen replaces the final model-pool league.
