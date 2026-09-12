---
experiment: P4-M9
date: 2026-09-12
last_updated: 2026-09-12
status: closed
---

# P4-M9：on-policy 契约与资源探针

**这是 P4-M9 的权威人读文档。** `artifacts/eval/_p4m9_probe_onpolicy_contract*/` 下的
`RESULT.md` / `probe_result.json` / `probe_records.jsonl` 是证据层，见第 6 节。

**一句话结论**：**strict proposal→executed identity 不成立，但"采样 proposal"的
on-policy 契约可用；结构对齐 / log-prob / 梯度三项契约通过。**

**授权范围（只验证不训练）**：只做契约验证与资源测量；不训练、不产出候选、
不覆盖 parent、不建立"采集—更新"循环、不用 critic/PPO/BC/entropy/KL/replay/opponent-pool。
为测量梯度路径与成本，在**一次性副本**上做了 forward/backward 与**单个** optimizer step；
parent 权重文件未被触碰。

## 1. 为什么要做这个探针
P4-M6 把 256×54 容量线关闭后，下一候选是「直接在策略上做 on-policy 更新」。
在花任何训练预算之前，必须先回答四个问题，且只回答这四个问题：

1. **采样与执行是否一致**（模型采的动作真的是环境执行的动作吗）？
2. **log-prob 是否可重算**（能否用同权重、同环境重算出采样时的 log-prob）？
3. **梯度路径是否正确**（能否得到有限、非零、且不污染 BN 的梯度）？
4. **真实成本是多少**？

本探针**不回答** RL 路线图，也不设计修复。
## 2. 探针与记录方式
`training/mortal/p4m9_probe_onpolicy.py` + `tests/test_p4m9_probe_contract.py`（34 项，含负向对照；ruff clean）。
`ProbeEngine` 是 `MortalEngine` 的子类（`supports_decision_context = True`），
拦截 `MortalEngine._react_batch`，逐决策记录：
`mask_bits` / `q_legal` / `obs_off` / `context` / `logprob`，并把 fp32 obs 原样写入 `obs_fp32.bin`。

**样本**：3 次独立采集，每次 **8 seed × 4 分片 = 32 半庄**，seed 710000–710007，
`seed_key=8192`，`--batch-seeds 8`。被交换策略 `student50k`（`student_step_050000.pth`，
sha256 `fda20413…`，v4 / 192×40）；对手 `ext_mortal`（`0a88ddad…`）。
## 3. 逐决策对齐：身份匹配而不是位置匹配
对齐用 `reconcile_kyoku`，按 `(mask_bits, q_values)` 的**身份**把决策匹配到日志事件
（`decision_matches`），而不是按位置 —— 这样一个未被执行的动作不会让后续比较整体错位。
分类包括：`aligned` / `sampled_pass` / `sampled_agari` / `sampled_ryukyoku` /
`executed_action_differs` / `agari_guard_suspected` / `claim_not_executed` /
`mismatches` / `unmatched_log_events` / `kan_select_states` /
`forced_decisions_not_offered_to_model`。

**3 次采集合计**（15,249 个策略决策）：`aligned 12414`、`sampled_agari 235`、
`sampled_pass 2595`、`sampled_ryukyoku 0`、`executed_action_differs 0`、
`agari_guard_suspected 0`、`claim_not_executed 5`、`mismatches 0`、
`unmatched_log_events 0`、`forced_decisions_not_offered_to_model 1167`（7.65%）。
账目精确（12414+235+2595+0+5 = 15249），三次 `alignment_complete: true`。
## 4. 四个答案
**① 采样与执行是否一致？**
严格意义**不一致**：`sampled_action_equals_executed_action = false`（三次均为 false）。
但这不是契约失败 —— 模型采样的是一次**动作 proposal**，另一个玩家抢先和牌／碰走该张牌，
是环境的多玩家裁决（规则优先级），不是模型动作被改写。

- **claim 被抢先 5 例（5/15249 = 0.033%）**：逐一核过。例：`710005/b dec11` 采样碰，
  该弃牌被他人荣和；`710002/d dec7` 采样吃，但 seat 0 先碰了 seat 2 的 4p（**碰优先于吃**）。
- **引擎改写模型动作 = 0**（保留数据中未出现）。
- **235 次采样 agari 全部被兑现**（含 3 次双响）。

**② log-prob 是否可重算？** **通过**。同批次形状下最大绝对差 **4.81e-07**，q 逐位相同；
固定 chunk 下最大 **1.08e-03**。

**③ 梯度路径是否正确？** **通过**。409/409 参数梯度有限且非零；BN running stats 未变；
一次 Adam step 移动 409 个参数；parent sha256 未变。**`eval()` 与
`train()+freeze_bn(True)` 都能以差 0.0 复现采样 log-prob，而朴素 `train()` 会把
log-prob 上移最多 1.4168 nats** —— 这正是 P4-M10 规定 BN 必须用 eval-mode 统计的原因。

**④ 真实成本？** 32 半庄采集 69.6 s（重复测 110.7 s / 240.3 s）；
逐决策 obs 137,632 B。
## 5. 收口判定
| 项 | 判定 |
| --- | --- |
| strict proposal→executed identity | **不成立**（但非契约失败） |
| 结构对齐 / log-prob / 梯度契约 | **通过** |
| claim preemption | OBSERVED 5/15249 —— 环境裁决，不过滤、不阻塞 |
| agari guard | 0/235 保留（机制存活，发生就记录，不关闭、不重设计） |
| quick-eval bypass | 1167/15249 —— 模型从未采样，排除在 PG loss 之外 |
| kan_select | 0 观察到 —— 未覆盖角落，不阻塞 |

**负责人（2026-09-12）批准的措辞**：“strict proposal→executed identity does not hold,
but the sampled-proposal on-policy contract is usable; structural alignment /
log-prob / gradient contracts pass.”

**批准的 next step**：探针已足够，可以进入一个真实的 policy-gradient 候选 → P4-M10。

## 6. 证据与产物引用（证据层，非权威）
| 内容 | 路径 |
| --- | --- |
| 探针脚本 / 单测 | `training/mortal/p4m9_probe_onpolicy.py` / `tests/test_p4m9_probe_contract.py` |
| 探针产物（3 次采集） | `artifacts/eval/_p4m9_probe_onpolicy_contract/`、`_sample2/`、`_sample3/` |
| 每次采集内容 | `RESULT.md`、`probe_result.json`、`probe_records.jsonl`、`obs_fp32.bin`、`logs/` |
| 历史生成产物（降级，不再权威） | 上述各目录的 `RESULT.md` |

后继：P4-M10 direct PG → `2026-09-12_P4-M10_direct_onpolicy_pg.md`。
