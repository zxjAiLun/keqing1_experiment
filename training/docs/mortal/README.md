# Mortal research notes index

## Document layout (adopted 2026-09-13)

Two layers, split by **responsibility** (not by file extension):

| Content | Lives in |
| --- | --- |
| Code / experiment tooling | `training/mortal/` |
| Human research conclusion, design, retrospective | `training/docs/mortal/YYYY-MM/` |
| Checkpoints, logs, JSON, obs, run data, generated evidence | `artifacts/` |

**Canonical research documents**

- New canonical reports live in `YYYY-MM/` (e.g. `2026-09/`).
- Filename: `YYYY-MM-DD_<experiment>_<topic>.md`; the date is the date the
  experiment/conclusion was formed, and must match the `date` field.
- Every `YYYY-MM/` document starts with this front-matter:

  ```yaml
  ---
  experiment: P4-M10
  date: 2026-09-12
  last_updated: 2026-09-13
  status: closed_not_supported
  ---
  ```

  `date` is fixed once the conclusion is formed; later substantive revisions
  bump `last_updated` only. Editing a document must never move the experiment
  to a later date.

**Evidence**

- `artifacts/` may contain **machine-generated Markdown** (`detailed_stats.md`,
  shard audits, auto-generated `RESULT.md`, ...). That is legitimate evidence
  and stays where it is.
- Artifact Markdown is **not** the canonical research record. Human research
  conclusions must not take their authoritative home under `artifacts/`
  (`ADJUDICATION.md`, `POLICY_DISPLACEMENT.md`, ... are supporting evidence and
  are reached from the canonical report's "Evidence" section).
- The registry splits the two: `report_paths` = human conclusion,
  `artifact_paths` = evidence.

**Legacy**

- Documents outside `YYYY-MM/` (`experiments_zh/`, the flat `2026-09-07_*` and
  `*_2026_0X.md` files) are legacy. They remain at their historical paths until
  touched for substantive reasons; they are not bulk-renamed, bulk-moved, or
  bulk-backfilled with front-matter.
- `README.md`, `研发总览_当前.md` and `research_registry.json` are living
  documents and keep their names; the overview carries `Last updated:` instead.

## Current execution plan (2026-09)

## Current execution plan (2026-09)

- [P4-M11：direct PG 学习预算、K0 实用门与 external EVE 门](2026-09/2026-09-13_P4-M11_directPG学习预算与K0替代验收方案.md)：C4 完整状态续训32×256半庄；固定U32先过K0门，再挑战external门。计划不代表训练已启动；消费替代与EVE研究结论分开登记。

## Timeline

| Date | Experiment | Result |
| --- | --- | --- |
| 2026-07 | V2/V3 recipe era | transition to controlled-training diagnostics |
| 2026-08 | D1/D2/D3 | controlled-training diagnostics chapter closed |
| 2026-09-07 | route review | mainline shift to external-teacher student |
| 2026-09-10 | P4-M2 | hard50k student completed |
| 2026-09-11 | P4-M3 | soft distillation not supported |
| 2026-09-12 | P4-M9 | on-policy contract usable; log-prob and gradient contracts pass |
| 2026-09-12 | P4-M10 | direct PG: small real policy movement, no deployment gain (`not_supported`) |
| 2026-09-13 | P4-M11 | direct PG learning-budget plan (two-level gate), not started |

Canonical reports for the 2026-09 P4 entries live in `2026-09/`; earlier
chapters keep their historical paths.

Chronological research history of the Mortal training line (keqing1
succession). The transition chapter this directory documents first is:

> **2026-07 中旬：从 V2/V3 recipe 迭代转向 controlled training diagnostics**

## Transition chapter (mid-July 2026)

| Document | Date | Role in the research history |
| --- | --- | --- |
| `v3_final_rank_mc_2026_07.md` | ~7/17-18 | Last "change one major recipe and observe" experiment. V3 switched from V2's `terminal_rank` to `final_rank_mc` on the same 6,000h corpus and the same 70k anchor. Result: a distinct policy, but not stronger than the 70k anchor - which is what ended single-recipe iteration. |
| `grp_v1_2026_07.md` | ~7/19 | The start of the controlled-diagnostics era. `prepare_grp_v1.py` built a frozen, reproducible reward source: an independent 2,000h corpus (outside the formal F/G corpus), one frozen checkpoint shared by all G seeds (no per-seed retraining). GRP v1 was infrastructure, not a "make GRP stronger" effort. |
| `reward_ab_2026_07.md` | 7/19+ | The first matched controlled experiment: F = `final_rank_mc` vs G = `mortal_grp_delta_pt`, with everything else locked (same 6,000h data, same 70k parent, fresh Adam, matched seeds, same data stream, `gamma=1`, CQL `5.0`, next-rank `0.2`, LR `1e-4`). |
| `optimizer_ab_2026_07.md` | 7/22+ | The second isolated training factor: fresh Adam vs preserved Adam continuation. |

The intentional constraint at this point: isolate one training-contract
dimension per experiment (reward semantics first; no CQL / learning-rate /
auxiliary-loss / architecture changes until reward is settled).

## Later chapters

- 7/26: pure-selfplay data route
- 7/28: objective learnability audit
- 7/29: D1 data route registration
- 8月: D1/D2/D3 data design and the uncertainty-exploration generation gate
  (see `experiments_zh/`, `d2_descendant_view_mix_2026_08.md`,
  `research_registry.json`)
- 8月: **D3 first production B250 gate PASS** (seeds `1800000..1800249`,
  generation `2cc12b4` frozen, auditor v2 `cf9bb86`, 74/74 hard checks,
  6304/6304 native-scene correspondence). Remaining 5750h and training are
  NOT authorized until continuation governance. audit-v1 verdict is superseded
  as invalid (reproduced on the authoritative 25h smoke); see
  `experiments_zh/2026-08_D3生产B250Gate_结果报告.md` and the D3 `gate_record`
  in `research_registry.json`.
- 8月: **D3 6000h generation COMPLETE** — 24/24 shards PASS, aggregate
  closure 27/27 (`35f9cea`), 151282 eligible / 27506 explored, zero duplicate
  seeds/hanchans/contexts. Generation integrity CLOSED; see
  `experiments_zh/2026-08_D3_6000h生成闭环_结果报告.md`.
- 8月: **D3 full controlled-experiment closure** — 3×70k→72k training
  complete, 3×1000h matched evaluation complete, and the preregistered
  promotion gate **FAILED mechanically** (D3−M0 0/3 seed means positive,
  hierarchical CI95 [-12.060,+2.070]): D3 data route NOT PROMOTED, no
  checkpoint promoted, K1 null. Summary statistics went through an
  auditor-side ReachAccepted repair (`8b36ee8`; v1 `a6b1d4b` invalidated);
  D1/D2 historical summaries corrected to the same fixed reconstruction
  (direction unchanged). See
  `experiments_zh/2026-08_D3不确定性探索数据路线_最终结果报告.md` and the D3
  `promotion_summary` / D1/D2 `historical_summary_correction` in
  `research_registry.json`. This closes the D1/D2/D3 controlled-training
  diagnostics chapter.
