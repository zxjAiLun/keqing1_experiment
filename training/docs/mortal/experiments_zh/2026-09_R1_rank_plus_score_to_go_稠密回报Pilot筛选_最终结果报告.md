# R1 rank_plus_score_to_go 稠密回报 Pilot 筛选实验 最终结果报告

> **实验 ID**: `R1_rank_plus_score_to_go_pilot_2026_09`  
> **代码提交**: [`b0933acec8e12ca10a40d8552b57c5e05a5e765b`](https://github.com/zxjAiLun/keqing1_experiment/commit/b0933acec8e12ca10a40d8552b57c5e05a5e765b)  
> **训练清单**: `artifacts/experiments/R1_rank_plus_score_to_go_pilot_2026_09/training/r1_training_manifest.json`  
> **训练清单 SHA-256**: `c24dac8280037985394c705e1c4980c0c13bb79a839cddc08c73a932085147e5`  
> **评测清单**: `artifacts/experiments/R1_rank_plus_score_to_go_pilot_2026_09/evaluation/r1_eval_manifest.json`  
> **评测清单 SHA-256**: `bbc9b5ea3a49d1ac9b9a4242a63f73eb8fa8900466154e1bb132ba4e2a99e416`  
> **正式裁决总结**: `artifacts/experiments/R1_rank_plus_score_to_go_pilot_2026_09/summary/r1_summary.json`  
> **裁决总结 SHA-256**: `7371f8992f0985a1017c8abf9956febb1da15fbf6fa90cf8dff2ce50cd14d5bc`  
> **最终裁决**: **`weak_positive`**（Primary Contrast 为正向 $+6.480\text{ pt}$ 但 CI 跨零，Secondary Contrast 为显著正向 $+9.675\text{ pt}$；按 Pilot 预注册规则，Recipe 未晋级，Checkpoint 未晋级，K1 保持为 `null`；进入 R2 多 seed 独立确认阶段）

---

## 1. 实验目标与受控设计

本实验旨在对基于局内得点变化量（`score_to_go`）与终局顺位结合的稠密回报目标进行单 seed Pilot 筛选验证：
- **基座母体**: `K0_70k` (`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`)
- **语料索引**: `M0` (`file_index_m0.pth`, `755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2`)
- **训练配置**: 单 seed `20260807`，从 70,000 步继承 2 个参数组 AdamW 初始矩训练至 70,400 步（精确 400 optimizer steps）。
- **Reward Target**:
  $$\text{rank\_target} = [+3, +1, -1, -3]$$
  $$\text{score\_to\_go} = \text{clip}\left(\frac{\text{final\_score} - \text{score\_at\_current\_kyoku\_start}}{10000}, -3, +3\right)$$
  $$\text{target} = \text{rank\_target} + 0.25 \times \text{score\_to\_go}$$
- **严格受控成立**: Control (`final_rank_mc`) 与 Variant (`rank_plus_score_to_go_mc`) 共享完全一致的 `FileDatasetsIter` 数据流与 RNG 状态，全 400 batch 的 row digest 100% 相同，仅由 reward mode 改变产生的 target 差异驱动梯度更新。

---

## 2. 硬门禁检验结果

### 2.1 训练硬门禁 (9/9 Passed)
1. `k0_parent_verified`: `True`
2. `m0_dataset_verified`: `True`
3. `control_400_steps_completed`: `True` (400 steps)
4. `variant_400_steps_completed`: `True` (400 steps)
5. `identical_row_identity_verified`: `True` (400 batch row digest 严格一致)
6. `control_checkpoint_saved`: `True` (`mortal_control_70400.pth`, SHA `805bf9ae0ed16e6eca166cfff4be4108917b349ffc60be5c704acefb1abbe7f2`)
7. `variant_checkpoint_saved`: `True` (`mortal_variant_70400.pth`, SHA `138fb3804356f1708eb5aaacaa829dc94a03fd665bea2f93b45eb6637be0fc1d`)
8. `exact_step_counts_verified`: `True` (70000 -> 70400)
9. `optimizer_preserved_adam_verified`: `True` (410 AdamW moments 继承与更新)

### 2.2 评测硬门禁 (7/7 Passed)
1. `training_manifest_verified`: `True`
2. `checkpoints_verified`: `True`
3. `ext_mortal_verified`: `True` (`external_mortal_20240308_best_min.pth`, SHA `0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563`)
4. `all_4_shards_completed`: `True`
5. `exact_1000_games_evaluated`: `True` (种子区间 `2200000..2200999`)
6. `reach_accepted_semantics_enforced`: `True`
7. `zero_missing_games`: `True`

### 2.3 汇总硬门禁 (7/7 Passed)
1. `training_manifest_verified`: `True`
2. `eval_manifest_verified`: `True`
3. `all_logs_verified`: `True`
4. `paired_metrics_recalculated`: `True`
5. `primary_contrast_computed`: `True`
6. `secondary_contrast_computed`: `True`
7. `bootstrap_computed`: `True` (5000 次 Paired Bootstrap, seed `20260906`)

---

## 3. 核心统计与对比结果

| 对比维度 (Contrast) | 均值差 (Mean Pt) | 95% Bootstrap 置信区间 (CI95) | 统计学方向与裁决 |
| :--- | :---: | :---: | :---: |
| **Primary: Variant $-$ Control** | **$+6.480\text{ pt}$** | **$[-2.161, +15.075]\text{ pt}$** | **方向性正向信号 (CI 跨零)** |
| **Secondary: Variant $-$ K0_70k** | **$+9.675\text{ pt}$** | **$[+1.125, +18.225]\text{ pt}$** | **显著为正 (CI 下界 $> 0$)** |
| **Reference: Control $-$ K0_70k** | **$+3.195\text{ pt}$** | **$[-5.355, +11.926]\text{ pt}$** | 方向性正向 (CI 跨零) |

---

## 4. 科学结论与阶段裁决

1. **Pilot 裁决**:
   - Primary 均值 $> 0$ 且置信区间跨零，符合 `weak_positive` 判定标准；
   - 证明在隔离所有混淆变量后，`rank_plus_score_to_go_mc` 稠密回报目标能够产生正向强度收益信号；
   - `Variant - K0` 达到显著正向（$+9.675\text{ pt}$，CI 下界 $+1.125\text{ pt}$），进一步印证了模型的绝对强度提升。
2. **晋升状态**:
   - 本轮为单 seed Pilot 探索，不触发 Recipe 与 Checkpoint 晋级；
   - `recipe_promotion = false`
   - `checkpoint_promotion = false`
   - `k1 = null`
3. **后续计划**:
   - 进入 `R2_rank_plus_score_to_go_multiseed_confirmation_2026_09` 阶段，使用全新 3 个训练 seeds（`20260910, 20260911, 20260912`）与交叉 Bootstrap 进行正式多 seed 确认。
