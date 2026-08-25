# O2 在线自对弈 Continuation 方向筛选实验 最终结果报告

> **实验 ID**: `O2_keqing_online_continuation_pilot_2026_08`  
> **代码提交**: [`cb9368426d6d068c62a8abc00866944bddbeafc9`](https://github.com/zxjAiLun/keqing1_experiment/commit/cb9368426d6d068c62a8abc00866944bddbeafc9)  
> **正式训练记录**: `artifacts/experiments/O2_keqing_online_continuation_pilot_2026_08/training/training_completion.json`  
> **训练记录 SHA-256**: `794eb7b27b4d9513847b4715b9de6c622148186b6ac25a8221d2a0ba73c6640a`  
> **Checkpoint SHA-256**: `c1acfdc66d73d7ebb9050b31cc7d69d1250f2ca1699f82161e2b0468206489ad`  
> **正式评测清单**: `artifacts/experiments/O2_keqing_online_continuation_pilot_2026_08/evaluation/evaluation_manifest.json`  
> **评测清单 SHA-256**: `4b55e5c06a58e7eb532aff1086ed209feb251f056adabc4531e8395b57c269a7`  
> **正式裁决总结**: `artifacts/experiments/O2_keqing_online_continuation_pilot_2026_08/evaluation/o2_summary.json`  
> **裁决总结 SHA-256**: `613e9d9bdcb451e465a76dfcf19332eeaf1ec41eb87f80358331974f38ecb9b4`  
> **最终裁决**: **`not_promising`**（方向筛选未发现正向信号，不启动 O3 复验，Recipe 未晋级，Checkpoint 未晋级，K1 保持为 `null`）

---

## 1. 实验目标与科学定位

本实验旨在对在线强化学习 continuation 路线进行小规模方向筛选（Direction Screening Pilot）：
- **基座母体**: `K0_70k` (`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`)
- **核心机制**: 采用持续刷新的 trainee replay 语料、Project-owned `final_rank_mc` 目标函数（`[+3, +1, -1, -3]`）、在线关闭 CQL（`cql_weight=0.0`）与冻结 Mortal BatchNorm。
- **训练规格**: 严格执行 16 个 refresh cycles，每 cycle 训练 25 步（总计 400 optimizer steps / 204,800 rows），步数推进至 70,400。
- **定位原则**: 仅作为探索性筛选实验，**无论结果如何均不直接晋级 K1**。

---

## 2. 训练执行与 Hard Gates 闭环

训练过程在 CUDA 上完整闭环，通过了全部 14 项严格 Hard Gates：
1. `parent_verified`: `True`
2. `production_loader_used`: `True`（生产 `FileDatasetsIter` 消费）
3. `exact_16_cycles`: `True`
4. `exact_400_optimizer_steps`: `True`（70000 → 70400）
5. `exact_204800_rows_consumed`: `True`（204,800 rows）
6. `no_replay_identity_reuse`: `True`（2,048 局独立对局身份）
7. `online_cql_disabled`: `True`（有效 CQL 权重为 0.0）
8. `final_rank_mc_verified`: `True`
9. `gradients_finite`: `True`
10. `parameters_finite`: `True`（Mortal, DQN, AuxNet 全网络参数有限）
11. `parameters_changed_from_k0`: `True`（Mortal, DQN, AuxNet 全网络权重更新）
12. `bn_frozen`: `True`
13. `final_checkpoint_70400_created`: `True`
14. `resume_state_consistent`: `True`

---

## 3. 评测方案与统计检验结果

### 3.1 评测协议
- **Lineup**: `[K0_70k, ext_mortal, M0_CURRENT_20260807, O2_70400]`
- **规模**: 4 个 Shards 共 1,000 局四人对局（对局 ID `2100000..2100999`，随机座位，`seed_key=8192`）
- **计分档位**: 天凤标准顺位点（`[90.0, 45.0, 0.0, -135.0]` pt）
- **统计方法**: 5,000 次常规配对 Bootstrap（seed=20260903）

### 3.2 检验数据与置信区间
1. **相对母体基座差异 $X = \text{Pt}(O2) - \text{Pt}(K0)$**:
   - **Mean Difference**: **`-6.615 pt`**
   - **95% Bootstrap CI**: **`[-15.4350, +2.2061] pt`**
2. **相对同源对照组差异 $Y = \text{Pt}(O2) - \text{Pt}(M0)$**:
   - **Mean Difference**: **`-8.460 pt`**
   - **95% Bootstrap CI**: **`[-17.1450, +0.1811] pt`**

---

## 4. 结论与下一步

1. **科学结论**:
   - 精确的 400-step 在线 continuation 机制（trainee replay + no-CQL）在本次 pilot 筛选中未观测到任何正向收益信号（$X$ 与 $Y$ 均值均为负），判定为 **`not_promising`**。
   - 由于两个 95% 置信区间均跨零，不能宣称在线训练造成了统计显著的退化，但明确证伪了该配置下存在值得推进多 seed 复验的正向信号。
2. **决策与路线收口**:
   - 不启动 O3 三 seed 确认实验。
   - `recipe_promotion = false`, `checkpoint_promotion = false`, `K1 = null`。
   - 终止继续微调 O2 超参数或延长 continuation 步数；下一阶段转向真正的 policy-improvement 机制探索（`P1_project_owned_policy_improvement_target_feasibility_2026_09`）。
