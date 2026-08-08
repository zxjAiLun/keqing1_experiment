# Replay Data Distribution Audit 2026-07

## Scope

This was an analysis-only audit of the retained V3 `final_rank_mc` replay index. It used the frozen 70k parent only for Q-support measurements. No replay, file index, checkpoint, or training state was modified.

Inputs:

- `artifacts/experiments/model_pool_2026_07/V3_final_rank_mc_warmstart_2026_07/file_index.pth`
- `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`
- V3 `final_rank_mc` config with centered rank values `+3, +1, -1, -3`

The machine-readable result is [data_distribution_audit.json](../../artifacts/experiments/model_pool_2026_07/data_distribution_audit/data_distribution_audit.json) and the generated report is [data_distribution_audit.md](../../artifacts/experiments/model_pool_2026_07/data_distribution_audit/data_distribution_audit.md). These artifact links are local-workspace references; the raw audit artifacts are intentionally not Git-tracked.

## Corpus Integrity

- `6000` trainable hanchan perspectives were loaded successfully.
- Total decision samples: `938,158`.
- Malformed files: `0`.
- Exact duplicate rate for `(obs, legal mask, behavior action)`: `0 / 938,158`.
- Decision-count ESS across hanchans: `5573.9 / 6000`.
- Decision-count Gini: `0.1536`.
- The top 10% longest hanchans contribute `15.03%` of decisions.

The corpus is therefore not dominated by a small number of long games, and the retained files are not accidental exact replay duplicates.

## What The Corpus Actually Is

This index is not pure external-model selfplay. It contains three mixed pools, each with `2000` hanchans:

- `v4_70k_t1_v0b_2000h`
- `v4_70k_v1_80k_2000h`
- `v4_v0b_v1_t1_2000h`

The `ext_mortal` perspective appears in all `6000` files with near-uniform random seat allocation. Its observed hanchan results are:

- 1st: `1632 / 6000 = 27.20%`
- 2nd: `1636 / 6000 = 27.27%`
- 3rd: `1509 / 6000 = 25.15%`
- 4th: `1223 / 6000 = 20.38%`

This is consistent with `ext_mortal` being stronger than the other models in this particular mixed pool, but it is not by itself a promotion result. It also means the final-rank target distribution is not an independent uniform label source.

## Decision Distribution

The main action mix is:

- discard: `77.84%`
- pass: `16.37%`
- pon: `1.80%`
- agari: `1.49%`
- reach: `1.25%`
- chi/kan/ryukyoku: the remaining `1.25%`

Phase coverage is `359,960` east, `338,574` south, and `239,624` extension decisions. The target perspective is in riichi state for `60,025 / 938,158 = 6.40%` of decisions. These numbers describe a normal offline decision stream; they do not indicate that a rare action class is being oversampled.

## 70k Behavior Support

Against the frozen 70k parent:

- behavior action legal rate: `100%`
- ext_mortal action equals 70k greedy action: `89.07%`
- mean behavior-action Q rank: `1.138`
- mean greedy-minus-behavior Q difference: `0.0604`
- mean legal-action Q absolute scale: `4.006`

The important conclusion is not that the 70k policy is wrong. It is that most of this corpus is close to the 70k policy under the parent model's own Q ordering. About `10.93%` of decisions are behavior/parent greedy disagreements, but the average internal Q gap is small. This is a limited but measurable new-policy signal, not a clean distribution shift.

The Q difference is an internal support diagnostic, not ground-truth regret. It must not be used as an automatic relabeling or reweighting rule.

## Target Conflict

`final_rank_mc` assigns one centered terminal target to every decision in a hanchan. High target standard deviation inside a state bucket therefore means that similar-looking decisions occur in hanchans with different eventual ranks; it does not prove that a sample is mislabeled.

The broad, well-populated buckets show target standard deviations around `2.0` to `2.3`, especially for rank-four and post-riichi states. This is expected for a terminal outcome target and identifies where target variance may limit continuation learning. It does not justify adding teacher CE, risk gating, or manual reviewer labels.

Opponent-riichi count is intentionally absent: the current `GameplayLoader` does not expose a reliable global per-decision opponent-riichi field in this audit path.

## Decision

The 6000-hanchan corpus is technically healthy and large enough for controlled baseline experiments, but the audit does not support another local reward/optimizer sweep:

1. The data is not concentrated or duplicated.
2. The data is behaviorally close to the 70k parent for most decisions.
3. The terminal target has substantial within-state outcome variance.
4. Preserved Adam already solved the measured continuation-initialization problem; it did not produce a checkpoint that reliably exceeded 70k.

The next research phase should therefore change the data lineage rather than add another continuation knob. The next experiment should pre-register a small, controlled project-owned lineage comparison with the same parent/checkpoint/evaluation contract and only one data variable:

- pure `ext_mortal` selfplay data with independent seeds and no seat-rotation duplicates;
- the current mixed model-pool data as the matched reference;
- the existing native four-model random-seat arena as the primary strength readout, with final avg pt/rank and behavior statistics;
- the platform-style per-account Pt/R ledger retained for long-run model-pool tracking.

Do not add GRP, teacher CE, NAGA labels, reviewer correction, LR/CQL sweeps, or automatic sample weights in that comparison. If pure selfplay still produces a model close to 70k rather than a clear improvement, the next question is data target quality or scratch-lineage optimization, not another small loss adjustment.

## Reproducibility

The audit implementation is [audit_replay_distribution.py](../../training/mortal/audit_replay_distribution.py). It supports CUDA preflight, bounded file batches, progress output, exact duplicate hashing, and frozen-parent Q support. The full run used CUDA, `file_batch_size=5`, `q_batch_size=512`, and `require_cuda`.
