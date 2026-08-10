# Mortal research notes index

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
  seeds/hanchans/contexts. Generation integrity CLOSED; the D3 data route is
  produced but NOT promoted over M0; no trained checkpoint exists (K1 still
  null). Training-view/target contract audit and training-recipe governance
  still required before any training. See
  `experiments_zh/2026-08_D3_6000h生成闭环_结果报告.md` and the D3
  `aggregate_record` in `research_registry.json`.
