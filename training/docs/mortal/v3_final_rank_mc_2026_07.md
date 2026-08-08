# V3 Final-Rank MC Mainline Result

## Scope

V3 was trained from the 70k anchor to steps 74,000 on the existing 6,000-hanchan native corpus. The training seat is the renamed external reference `ext_mortal`; no teacher CE, reviewer label, risk gate, or new replay pool was added.

The new training reward is `final_rank_mc`: centered final-rank return `[+3,+1,-1,-3]` is assigned to every decision in the hanchan. The 72,000 and 74,000 checkpoints both carry this training contract and loaded successfully.

## 250h Screen

The screen used `ext_mortal`, `70k`, `V2@72000`, and `V3@72000`, with 250 random-seat hanchans, CUDA, `seed_start=990000`, `seed_key=8192`, and `[90,45,0,-135]` rank points.

| Model | Avg rank | Avg rank pt | Agari | Houjuu | Fuuro | Riichi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ext_mortal` | 2.500 | -0.54 | 21.56% | 13.31% | 30.11% | 18.42% |
| `70k` | 2.608 | -9.72 | 21.14% | 14.33% | 29.77% | 18.80% |
| `V2@72000` | 2.372 | +11.70 | 23.15% | 12.97% | 26.63% | 20.61% |
| `V3@72000` | 2.520 | -1.44 | 21.44% | 12.97% | 29.69% | 17.59% |

V3 did not show catastrophic behavior and continued to the 74,000 checkpoint. The screen was not used as a promotion result because 250 hanchans are too small and the model-pool composition affects the ranking distribution.

## Balanced 1000h League

Five random-seat lineups were run for 200 hanchans each. Each family received exactly 1,000 seat-hanchans. The protocol used CUDA, `seed_key=8192`, seeds `991000..991999`, and Tenhou rank points `[90,45,0,-135]`.

| Model family | Games | Avg rank | Avg rank pt | Avg game delta | Agari | Houjuu | Fuuro | Riichi | Rank counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ext_mortal` | 1000 | 2.449 | +3.47 | +462.5 | 21.87% | 12.55% | 30.04% | 18.36% | 267/254/242/237 |
| `70k` | 1000 | 2.484 | +1.35 | +114.9 | 21.90% | 12.65% | 29.60% | 18.72% | 257/245/255/243 |
| `V3@74000` | 1000 | 2.527 | -1.76 | -217.6 | 21.55% | 13.58% | 28.35% | 18.86% | 242/245/257/256 |
| `V2@74000` | 1000 | 2.540 | -3.06 | -359.8 | 20.00% | 12.22% | 23.40% | 19.39% | 234/256/246/264 |

The final league does not promote V3 over the 70k anchor. V3 is slightly ahead of V2 in this balanced run, with more wins and calls than V2, but it also has a higher deal-in rate and remains below both the external reference and 70k on average rank and rank points.

The result is not evidence that `final_rank_mc` is useless. It shows that this reward change produces a viable, distinct student policy, but the current data volume and objective do not yet convert that policy into a stronger model than the 70k reference.

## Platform Account Report

The league generated 1,000 compressed hanchan logs, a combined log directory, per-account ledger, rating curve, and Pt/R summaries under:

`artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/league_1000h_balanced/`

Because some lineups intentionally duplicated a model family with `_a`/`_b` labels, the raw platform report contains multiple account identities for a family. `league_summary.json` is the family-level aggregate and is the primary comparison for this experiment.

## Decision

- Keep `70k` as the current local trained-student reference.
- Keep `V2@74000` and `V3@74000` as archived research checkpoints; do not promote either as the default GUI candidate.
- Keep `ext_mortal` as the external strength reference and synthetic-data actor; it is not a project-generated V-series model.
- Do not generate a new replay pool automatically after every promotion. Reuse the existing native corpus unless a new hypothesis changes the training distribution or objective.
- The next research decision should target either a better reward/target formulation or a controlled multi-style population experiment, with the same final balanced league as the promotion gate.

## Artifacts

- V3 config and reward preflight: `artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/`
- 250h screen: `artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/eval_250h_native_screen/`
- Final league: `artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/league_1000h_balanced/`
