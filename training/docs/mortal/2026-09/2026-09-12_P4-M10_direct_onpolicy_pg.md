---
experiment: P4-M10
date: 2026-09-12
last_updated: 2026-09-13
status: closed_not_supported
---

# P4-M10：hard50k → 直接 on-policy policy gradient（4 cycle × 256 半庄）

**这是 P4-M10 的唯一权威人读文档。** 产出的 JSON／权重／日志／逐决策记录都是证据，
按第 10 节引用；`artifacts/` 下遗留的 `RESULT.md`、`ADJUDICATION.md`、
`POLICY_DISPLACEMENT.md` 降级为历史生成产物，**不再承担权威结论**。

**一句话结论**：四次真实的 `collect → 一次 PG update → checkpoint` 闭环全部跑通，
候选 C4 确实小幅移动了策略，但**没有获得部署收益支持**（`not_supported`）；
机制有效性仍未解决。

<!-- 第 1 节起见下 -->

## 1. 研究问题

在此之前的 P4-M9 探针确认了：采样用的 log-prob 可以重算、梯度路径正确、
逐决策记录与执行日志能结构化对齐。**剩下的问题是**：在一个真实的最小闭环里，
用当前策略自己采集的数据做一次策略梯度更新，到底能不能推动策略、能不能带来部署收益。

**这不是**"证明 direct PG 有效"，也不是"排除 direct PG"。本轮购买的是一次
**闭环可行性 + 单候选端点强度**的检验。

## 2. 前置：P4-M9 契约结论（本轮的前提）

- 严格"proposal → executed"动作恒等**不成立**，但**按采样 proposal 记账的 on-policy 契约可用**。
- 结构对齐、log-prob 重算、梯度路径三项契约**通过**。
- claim 抢先（他人更高优先级和了/碰）是环境的多方结算行为，**不过滤、不阻塞训练**；
  这些 proposal 按 sampling-time log-prob 计入 loss。
- agari guard 机制在实现层是活的，但**本轮未直接观测原生 guard 触发次数**（见第 6 节）。
- 真正 fail-close 的只有四类：重算失败、轨迹身份不一致、梯度非有限、采集记录缺失。

## 3. 冻结配方（预注册，运行期不可调）

| 项 | 值 |
| --- | --- |
| 起点 | hard50k（`student_step_050000.pth`，sha256 `fda20413…`） |
| 对手 | 固定 `3 × ext_mortal` |
| 架构 | 不变（Brain 192×40 + DQN v4） |
| 采样 | `softmax(q_legal)`，T=1、ε=1、top_p=1、FP32、无 stochastic latent |
| BN | **eval-mode 统计 + autograd 开**（绝不退回普通 `train()` 反向） |
| 回报 | Tenhou rank points `/135` → `[2/3, 1/3, 0, −1]`，标量 baseline **0** |
| loss | `−(1/N_h) Σ_h G_h Σ_t log π(a_t\|s_t)` |
| 优化器 | fresh Adam，lr=1e-5，wd=0，**只建一次并跨 cycle 保留状态** |
| 梯度裁剪 | global norm **1.0** |
| update microbatch | 512，整批累积，**每 cycle 恰好一次 step** |
| 预算 | 4 cycle × 256 半庄（上限 1024） |
| 评测 | **只评 cycle-4 端点**，且只评 C4 |
| 采集种子 | 720000–720255（四段各 64 seed × 四座轮换 = 256 半庄/cycle） |

**哪些决策进入 loss**：每一个由当前模型采样、且 `exploration_allowed=True` 的 proposal，
**包括 claim 抢先与 agari-guard 改写**，按 sampling-time log-prob 记账；
`enable_quick_eval` 绕过从未采样、不进 loss；`exploration_allowed=False` 的 kan-select 不进 loss。

**关键实现纪律**：实现一个最小的 `collect → one-update → checkpoint` 循环，
**不要实现 RL 框架**。不加 critic / PPO / BC / KL / entropy / replay / opponent pool。

## 4. 资源与成本观测
- 采集成本**由固定开销主导**：8 半庄与 32 半庄都约 213 s，而 256 半庄为 366–467 s。
- 四条 cycle 训练总墙钟 **1886.9 s**（约半小时），峰值显存 **2.17–2.24 GB**。
- 逐决策观测 **137,632 B/decision** → 1024 半庄约 **23.0 GB** obs + 约 98 MB 元数据。
- 单 cycle obs 约 6 GB；评测峰值显存 665.7 MB (A) / 531.0 MB (B)。

## 5. 四轮训练结果

训练种子 **720000–720255**（4 段各 64），`seed_key=8192`，采样种子 20260914–20260917。

| cycle | 种子 | 采集 s | 入 loss 决策 | microbatch | 名次计数 (1/2/3/4) | 平均 returnᵗ | lossᵗ | maxΔ | grad preclip→post |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 720000–720063 | 365.9 | 38,057 | 75 | 59/56/54/87 | −0.1133 | −4.132 | 1.94e-03 | 20.38 → 1.0 |
| 2 | 720064–720127 | 369.4 | 37,314 | 73 | 45/59/68/84 | −0.1341 | −4.623 | 1.69e-03 | 15.01 → 1.0 |
| 3 | 720128–720191 | 466.5 | 38,810 | 76 | 56/56/55/89 | −0.1289 | −3.376 | 1.59e-03 | 15.73 → 1.0 |
| 4 | 720192–720255 | 405.7 | 39,261 | 77 | 49/67/50/90 | −0.1367 | −4.578 | 1.69e-03 | 15.84 → 1.0 |

**契约全部为真**：每 cycle 恰好 1 次 optimizer step；每 cycle 409/409 参数梯度有限非零；
**100% 决策的 log-prob 重算落在预声明 5e-3 容差内**（38057/38057、37314/37314、38810/38810、39261/39261）；
导出权重在采集前后 sha256 不变；采集期间内存参数摘要不变；父模型 `fda20413…` sha256 未变；
优化器只创建一次并在 cycle 2–4 复用；`kan_select` 排除路径出现 1/1/0/0 次。

## 6. 训练期契约检查（全部通过）
- 每 cycle **恰好 1 次** optimizer step；四段各自 **409/409** 参数拿到有限梯度。
- 全部决策 log-prob 重算落在预声明 **5e-3** 内：38057/38057、37314/37314、38810/38810、39261/39261。
- 导出权重在采集前后 **sha256 一致**；内存参数摘要采集期间未变；父模型 `fda20413…` `sha256_unchanged = True`。
- optimizer **只建一次**并跨 cycle 复用（`created_once_before_cycle_1` + `optimizer_reused = True`）。
- `kan_select` 排除路径本轮触发 **1/1/0/0**（P4-M9 曾为 0）→ 路径被覆盖，但样本极少。

## 7. Guard 证据边界（2026-09-12 只读补记）
运行期 `p4m10_result.json` 的 `preemption_and_guard.batches_with_agari_guard` **四个 cycle 全为 `null`**
——该字段在运行时**根本没有被计算过**；运行期注释声称"已由 P4-M9 reconciler 计数"并不成立。
用**现成 reconciler 对保留记录做只读补记**（`p4m10_guard_backfill.py`；未修改训练产物、未重跑任何一局）。

**证据边界（最终措辞）**：
> 发现**一例明确的采样—执行动作改写**，以及**一例疑似 guard 事件**；
> **现有记录未直接观测原生 guard 触发次数。**

- **明确的一例**（`executed_action_differs`）：cycle 3 `seed 720188 / split a / dec 14`，采样 agari
  （log-prob −0.0225），身份匹配到 `dahai`、执行动作 = 弃和 1 → **确实发生过一次改写**，
  结合已知实现与 guard 行为相符。
- **疑似的一例**（`agari_guard_suspected`）：cycle 2 `seed 720092 / split b / dec 17`，采样 agari
  （log-prob −0.6157），该 kyoku 由 seat 0 和了。该分类器只判定"采样了和牌但本座位未成为赢家"
  （`p4m9_probe_onpolicy.py:406`），**不读取原生 guard 触发标记** → 属疑似。
- **两类证据不得相加**，**不得写成「确认触发 2 次」或给出触发率**。

## 8. 强度评测（预注册，只评 C4 末端）
评测环境为 P4-M8 恢复过的 V1 环境，历史 seed 段 **710000–710063**，两个方向各 256 半庄，
部署策略为 **greedy**（`boltzmann_epsilon=0`），全程 `reused_seed_count=0`、`ranks_match_native=true`。
| run | challenger | 名次计数 | avg_rank | avg_rank_pt | CI95 |
| --- | --- | --- | --- | --- | --- |
| hard50k solo vs 3×ext（历史） | `student50k` | 60/66/49/81 | 2.5898 | −10.0195 | [−19.3359, −1.0547] |
| **C4 solo vs 3×ext** | `p4m10_C4` | 56/68/50/82 | 2.6172 | **−11.6016** | [−20.7422, −2.4609] |
| ext solo vs 3×hard50k（历史） | `ext_mortal` | 73/70/57/56 | 2.3750 | **+8.4375** | [−0.0044, +16.3477] |
| **ext solo vs 3×C4** | `ext_mortal` | 74/54/61/67 | 2.4727 | **+0.1758** | [−9.3208, +9.6680] |

**同种子配对比较**（64 个种子簇、5000 次 bootstrap、seed 20260910）：

- 方向 A = C4 − hard50k = **−1.5820 pt**，CI95 [−8.6133, +5.4492]（逐种子为正 12/28，符号检验 p 0.8275）；avg_rank +0.0273，CI95 [−0.0508, +0.1094]。
- 方向 B = ext_vs_3C4 − ext_vs_3hard50k = **−8.2617 pt**，CI95 [−17.9297, +1.2305]（逐种子为正 22/49，符号检验 p 0.8042）→ **candidate benefit = +8.2617 pt**，CI95 [−1.2305, +17.9297]。

**裁决：`not_supported`。** 方向 A 的点估计为**负**（"两方向都为正"这一条直接不成立）；两个 CI95 均跨 0；两方向**符号相反**（典型的无清晰效应）。不追加局数、不调 lr/T/baseline；C1–C3 仍未评测、不可选。

**量纲警告**：方向 A 是**每座位**量，方向 B 是**三座整体**量，两者**不得并列**；镜像的 +8.26 pt 按零和折算到每个学生座位约 **+2.75 pt**，不得读成"每座位 +8.26 pt"。

结论只能是：**没有足够证据晋级 C4，也没有足够证据说 C4 更弱**。

## 9. 策略位移（只读固定面板检查，2026-09-13 补��）
问题：四次 update 到底是"轻微扰动"还是"已经明显改变部署动作"？脚本 `p4m10_policy_displacement.py`（只读推理，5 个 checkpoint 各自重算一遍前向）。

**主结果（只有这三项）**——面板 = cycle 1 的 38,057 个 parent on-policy 决策：

| 指标 | hard50k → C4 |
| --- | --- |
| greedy 动作改变（flip） | **367 / 38,057 = 0.964%**（CI95 0.861–1.074%）|
| 平均 TV | **0.00880** |
| KL(parent ‖ C4) | **0.001083** nats |

跨 batch 大小 512/2048/4096 重算，flip 数最多相差**一个决策**，TV/KL 只在小数点后 4–5 位变动 → 该位移**不是普通批处理数值波动**。

**准确结论**：四次 PG update 造成了**可重复观测的、总体较小的策略位移**。这排除了一个此前真实存在的疑问——“代码虽然 backward/step 了但策略根本没动”；但**没有**证明约 1% 的 flip 是太少、还是恰好含少量高价值决策。

**不成立的表述（已撤回）**：
- **不得用 raw action-score 变化证明“移动很多”**。DQN 是 dueling 形式，同状态所有合法动作同时平移常数时 softmax/argmax 都不变，因此 raw `\|Δq\|` 与 policy displacement 不是同一个量；`\|Δq\|` 已降级为 telemetry。
- **不得说“只改变 near-ties / 未改写强偏好决策”**。253/367 只是**描述统计**，统一阈值不是逐状态噪声界，且有 114 次翻转越过了它。
- **不得把 0.964% 与 9.58% 相除**再解释成“部署只移动了探索幅度的十分之一”：一个是两个模型 argmax 的差异，另一个是单个随机策略偏离自身 argmax 的概率，含义不同。

**面板替换登记**：实际面板是 **P4-M10 cycle 1 的 38,057 个 parent on-policy 决策**，**不是**最初讨论的 P4-M4 固定面板（fixed ≥2 legal）。cycle-1 状态参与过 C1 训练，因此**不是独立 holdout**；也**不是 C4 自己的访问分布**。本次目标本就不是泛化或强度评估，登记边界即足；不重跑旧面板。

**其它可信度检查**：父权重复现采集记录（`q_legal` 相对偏差 7.6e-05、`logprob` 2.09e-03 vs 阈值 5e-3）；同权重同 chunk 下前向 **bit-exact**；5 个 checkpoint 在 **338,563 个合法动作格**上恰好有限（非法动作结构上进不了指标）。

## 10. 最终定位（2026-09-12 负责人裁定）

> **一次小幅、真实的策略更新试验，未获得部署收益支持；不是充分学习后的失败，也不是无效更新。**

展开为四句，**彼此不得互相推导**：

- **工程上**：`collect → on-policy sample → backward → Adam step → checkpoint → greedy evaluation` 闭环已经**真实工作**。
- **学习上**：四次 update **确实移动了 policy**，但 measured intervention 较温和。
- **强度上**：C4 **没有**获得预注册的 deployment improvement support。
- **研究上**：direct PG 的**机制有效性仍 unresolved**。

**收口**：P4-M10 保持 `not_supported`、停止且不晋级；保留 C1–C4，不挑中间权重、不追加评测；
不自动续训、不因本次结果跳到 PPO/critic、不预建后续实验、不预先决定 `12×256`。
续训 PG 现在是一个**合理的新投资选项**，但不是现有结果推出的“下一步修复”。
今后若继续 PG，购买的应是**一次更有学习机会、但收益未知的独立预算**。

**语义变化**：PG 之后 C4 的 DQN 输出是 **policy logits / action scores**，**不再是 calibrated Q estimate**；
argmax 部署没问题，但不可再做「Q gap = 教师估值差」的解释。

## 11. 证据与产物引用（证据层，非权威）

| 内容 | 路径 |
| --- | --- |
| 训练循环实现 / 测试 | `training/mortal/p4m10_onpolicy_pg.py` / `tests/test_p4m10_onpolicy_pg.py` |
| guard 只读补记工具 | `training/mortal/p4m10_guard_backfill.py` |
| 位移检查工具 / 测试 | `training/mortal/p4m10_policy_displacement.py` / `tests/test_p4m10_policy_displacement.py` |
| 训练产物（权重/records/obs/logs） | `artifacts/experiments/student_policy_v1/P4-M10_onpolicy_pg_4x256/` |
| 结果 JSON | `.../p4m10_result.json`、`.../guard_reconciliation_backfill.json` |
| 双向评测 | `artifacts/eval/ovt_p4m10_C4_vs_3ext/`、`artifacts/eval/ovt_3p4m10_C4_vs_ext/` |
| 比较与位移 JSON | `artifacts/eval/p4m10_comparisons/` 下的 `summary.json`、`solo_vs_ext_c4_minus_hard50k.json`、`ext_vs_trio_delta.json`、`policy_displacement.json` |
| 历史生成产物（降级，不再权威） | `.../P4-M10_onpolicy_pg_4x256/RESULT.md`、`p4m10_comparisons/{ADJUDICATION,POLICY_DISPLACEMENT}.md` |

前置：P4-M9 on-policy 契约与资源探针 → `2026-09-12_P4-M9_onpolicy_contract_probe.md`。
后继：P4-M11 学习预算与验收方案 → `2026-09-13_P4-M11_directPG学习预算与K0替代验收方案.md`（尚未启动）。
