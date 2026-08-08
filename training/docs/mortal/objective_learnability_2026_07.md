# Objective Learnability Audit (2026-07)

## Scope

This was an analysis-only audit of the retained 6,000-hanchan M0 mixed corpus
and the 6,000-hanchan S0 pure `ext_mortal` corpus. It did not update model
parameters, create a checkpoint, or change the training default.

The audit reconstructed the `final_rank_mc` target used by the offline runner,
calibrated the 70k parent Q values on deterministic decision samples, compared
the 72k candidates with the parent on a shared probe, and measured the relative
DQN/CQL/next-rank gradient signals on one fixed small batch per checkpoint.

## Results

| Route | Decisions | Target mean | Target std | Parent Q/target Pearson | Linear explained variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 mixed | 938,158 | 0.2254 | 2.1884 | 0.2593 | 6.3% |
| S0 pure `ext_mortal` | 964,091 | 0.0813 | 2.2449 | 0.1841 | 3.4% |

The target distribution is centered near zero but has a standard deviation of
about 2.2 Pt. On the sampled decisions, the 70k parent has strict OLS R^2 of
about 6.7% for M0 and 3.4% for S0; the identity-Q explained-variance ratios
are reported separately by the audit. This is a learnability warning: the
target is not a clean local action-value label, even though the corpus is
complete and non-duplicated.

The target variability is not limited to late decisions. M0 target standard
deviation ranges from 2.183 in early decisions to 2.200 late; S0 ranges from
2.244 to 2.246. The audit therefore does not support a narrow late-game-only
data explanation.

On a shared 482-state legal-action probe, both candidates changed the parent
greedy action on about 8.3--8.5% of states. Most Q movement was a common offset
or scale movement rather than centered action preference movement:

| Candidate | Greedy change | Raw Q abs delta | Centered advantage abs delta | Abs common offset | Parent/candidate margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0@72000 | 8.5% | 1.8021 | 0.8719 | 1.6134 | 2.2823 / 4.1230 |
| S0@72000 | 8.3% | 1.7234 | 0.8989 | 1.5136 | 2.2823 / 4.3835 |

This points to Q calibration and action-margin expansion as a material part of
the continuation drift. It is not evidence that either route is stronger.

The small gradient diagnostic also showed different objective interactions:

- M0: weighted CQL backbone norm `16.31` versus DQN `20.93`, with DQN/CQL
  backbone cosine `+0.72`.
- S0: weighted CQL backbone norm `11.84` versus DQN `9.35`, with DQN/CQL
  backbone cosine `-0.34`.
- The next-rank auxiliary signal was smaller on the shared backbone than the
  weighted DQN/CQL signals in both checkpoints.

These gradient values are diagnostic, not a significance test: they use one
fixed 32-sample batch per checkpoint. They do show that the two data routes
arrive at materially different optimization geometry under the same nominal
recipe.

## Decision

1. Close the pure-selfplay expansion. The audit does not justify generating
   another S0 pool.
2. Keep `final_rank_mc` as the operational reward and preserved Adam as the
   70k legacy-continuation initialization. Neither is re-promoted by this
   audit.
3. Do not open LR, CQL, GRP, teacher-CE, or another broad data-route grid.
4. Treat the current bottleneck as objective/representation learnability:
   final-rank MC carries high conditional variance, while Q drift is dominated
   partly by common offset and margin changes.

The next experiment should be one pre-registered, project-owned objective
variant that separates action preference from scalar Q calibration, with the
same M0 corpus, parent, preserved Adam contract, and matched seeds. It should
be compared against the current `final_rank_mc` control before any new data
pool is generated. This audit alone does not promote a candidate checkpoint.

## Reproduction

The full local artifacts are written to:

`artifacts/experiments/model_pool_2026_07/objective_learnability_audit_2026_07/`

The analysis entry point is
[`audit_objective_learnability.py`](../../training/mortal/audit_objective_learnability.py).
It uses batched native log loading and requires CUDA for the full run on the
Windows environment.
