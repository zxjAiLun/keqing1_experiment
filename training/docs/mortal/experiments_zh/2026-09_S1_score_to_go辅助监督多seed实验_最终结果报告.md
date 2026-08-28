# S1 score_to_go 辅助监督多 seed 实验 — 最终结果报告

> 实验身份：`S1_score_to_go_auxiliary_multiseed_2026_09`
> 状态：**CLOSED / not_supported / K1 = null**
> 本报告为不可变最终结果记录，一经提交不覆盖。

---

## 1. 实验定位与预注册假设

S1 是 R2 终止 `rank_plus_score_to_go` 主 Q-target 路线后，对 score-to-go 信号的最后一次检验：将其降级为**独立辅助回归头**，主 Q target 严格保持 `final_rank_mc` 不变。预注册四态裁决：

- `promotion_supported`：Primary（AuxVariant − R2 Control）3/3 seed 均值为正且 CI95 下界 > 0，**且** Absolute（AuxVariant − K0_70k）同样满足；
- `auxiliary_effect_only`：仅 Primary 满足；
- `not_supported`：其余情况；
- 任一 gate 失败：`no_verdict_gates_failed`。

### 1.1 冻结契约

| 项 | 值 |
| :--- | :--- |
| Parent model | `K0_70k`（SHA `6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`） |
| 数据 | M0 mixed replay corpus（`file_index_m0.pth`，SHA `755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2`） |
| Objective | `behavior_action_mc` / `behavior_action_q`（不变） |
| Trainable 视角 | 仅 `["ext_mortal"]` |
| 主 Q target | `final_rank_mc`（centered [+3,+1,−1,−3]），**不含 score 分量**（`excluded_from_q_target=true`） |
| 辅助头 | `ScoreToGoHead = Linear(1024, 1, bias=False)`，target = clip(diff/10000, −3, +3)/3 ∈ [−1,+1]，loss = 0.2 × (0.5 × MSE) 仅更新 Brain+Head；DQN/AuxNet 梯度路径与推理结构不变 |
| Control | R2 冻结 Control checkpoints（同 seeds，SHA 三向绑定，不重训） |
| 训练 seeds | 20260910 / 20260911 / 20260912（与 R2 Control 相同 seed/stream） |
| 步数 | 70000 → 70400，精确 400 optimizer steps |
| Optimizer | 严格恢复 410 个 parent Adam moments + fresh head 参数组（LR 1e-4, wd 0），三组 [165, 245, 1] 共 411 states |
| Row identity | 每 seed 全 400 batch reward 外字段滚动 SHA256 必须逐字节等于 R2 Control 冻结 digest |
| 梯度路由验证 | 确定性前 8 行 probe 在 full-batch forward 之前运行：Brain/Head 参数梯度非零 finite，DQN/AuxNet 为 None/0，验证后立即释放计算图并恢复 `.grad` 状态 |
| 评测 | 每 seed 1000 局四人同桌（`four_player_native --seat-mode=random`），公共随机数 seeds 2400000..2400999，seed key 8192，4 shards × 250 |
| 阵容 | `K0_70k, ext_mortal, R2_Control_seed_s, S1_AuxVariant_seed_s` |
| 统计 | Crossed bootstrap（seed 轴 × game-id 轴，5000 reps，seed 20260920），Primary/Absolute 共享重采样索引 |

---

## 2. 执行记录

### 2.1 训练（10/10 hard gates PASS）

3 个 AuxVariant 训练全部完成。gates：`k0_parent_verified`、`m0_dataset_verified`、`r2_control_checkpoints_verified`、`all_3_seeds_completed`、`all_3_variant_checkpoints_saved`、`all_seeds_row_identity_matches_r2_control`、`main_q_target_final_rank_mc_verified`（每批 Q target ∈ {+3,+1,−1,−3}，runner 实测）、`score_loss_routed_to_brain_and_head_only`（首批 8 行 probe autograd 实测）、`optimizer_410_parent_moments_plus_fresh_head_group`（结构性验证）、`exact_step_counts_verified`（400 steps）。

Row identity 三 seed 全部精确匹配冻结 R2 Control digest：

| Seed | digest（= R2 Control） |
| :--- | :--- |
| 20260910 | `2e1e41ad31487fa19953d6e1cd1cd777c76d2229592ffdaf2a62943b7c30c013` |
| 20260911 | `e503dc4043b21ae05382172a9071b358dc504bd55713b8c40373a6c93dd569c9` |
| 20260912 | `fdeb82219f8e0b7bd9e306949377f07e0f0e2424d505751fe1a648149f8b3fba` |

三个 AuxVariant checkpoint（SHA256，training manifest ↔ eval manifest ↔ disk 三向绑定校验通过，均可由正式 `four_player_native` 加载）：

| Checkpoint | SHA256 |
| :--- | :--- |
| `mortal_aux_variant_70400_seed_20260910.pth` | `3e427e48b72a68f9ae3bebdd3c01bf8816cde5b92c9d68b7a8b4deb73b5f7c94` |
| `mortal_aux_variant_70400_seed_20260911.pth` | `6fd0c316a943d33da5d989e63290bf72a33c17098433cac2cc6ec008bf27bf3c` |
| `mortal_aux_variant_70400_seed_20260912.pth` | `6f2cf74fd4a3680ec690636d82a856155aa9a29aacaf0475101b614ab92ea4ca` |

冻结复用的 R2 Control checkpoint：`fe7fecf8…`（20260910）、`8b3cad1a…`（20260911）、`cee3fd1e…`（20260912），与 R2 训练 manifest 完全一致，未重训。

训练 manifest SHA256：`c02414346a23f6636325338a2bad500ad1114adf30321562c360ea010d04af4a`。

辅助头确实在学习：`score_aux_loss` 从 ~0.11 降至 ~0.05–0.10。

### 2.2 评测（7/7 hard gates PASS）

3 panels × 1000 局 = 3000 局完整对局。gates：`training_manifest_verified`、`all_checkpoints_verified`、`ext_mortal_verified`、`all_3_panels_completed`、`exact_3000_games_evaluated`、`reach_accepted_semantics_enforced`、`zero_missing_games`。每个 panel 1000 份日志 game ID 连续唯一覆盖 `2400000..2400999`，全部 `end_game` 完整结束，Training → Evaluation manifest 哈希链通过。

评测 manifest SHA256：`f89bd842970178b5557b83184e05625c0d460ad864531adc9631141dfce9b65d`。

### 2.3 汇总（7/7 hard gates PASS）

gates：`training_manifest_verified`、`eval_manifest_verified`、`all_3000_logs_verified`、`paired_metrics_recalculated`、`crossed_bootstrap_computed`、`primary_contrast_evaluated`、`absolute_contrast_evaluated`。

`s1_summary.json` SHA256：`e6d6f3ddcc54b068d805abff417911535a9ca4591068461b724aae22249b8ce4`。

---

## 3. 正式统计结果

### 3.1 Primary：AuxVariant − R2 Control（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **+2.025** |
| 20260911 | **+0.180** |
| 20260912 | **+2.205** |
| **总体（grand mean）** | **+1.470** |

Crossed-bootstrap CI95：**[−5.880, +8.655]**（跨零）。
`all_seed_means_positive = true`；`ci_lower_positive = false`。

### 3.2 Absolute：AuxVariant − K0_70k（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **−4.545** |
| 20260911 | **−5.670** |
| 20260912 | **−6.750** |
| **总体（grand mean）** | **−5.655** |

Crossed-bootstrap CI95：**[−12.706, +1.290]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.3 机械裁决

```text
verdict               not_supported
recipe_promotion      false
checkpoint_promotion  false
K1                    null（canonical seed 20260911 备录，未晋升）
```

Primary 3/3 正向但 CI 下界 ≤ 0（`primary_pass = false`），Absolute 0/3 正向，落入 `not_supported`。

---

## 4. 科学结论

1. **Primary 是一致的方向性信号但未达统计确定。** AuxVariant 相对同 seed/stream 的 R2 Control 三 seed 全正（[+2.025, +0.180, +2.205]，总体 +1.47 pt），但 crossed-bootstrap CI95 [−5.880, +8.655] 跨零，不能确认辅助目标有效。
2. **Absolute 三 seed 全负（−4.545 / −5.670 / −6.750，总体 −5.655 pt）**，不存在超越 K0 的晋级价值。
3. **描述性诊断（非晋级证据）**：同一批评测推算 Control−K0 约为 [−6.57, −5.85, −8.955] pt；AuxVariant 的 Absolute 均值高于该基准，与 Primary 正向信号一致，提示 auxiliary 可能部分抵消 continuation 退化。该推算仅作描述性参考，不构成任何晋级依据。
4. **主 Q target 与推理结构未被破坏。** 主 Q target 逐批验证为 {+3,+1,−1,−3}，DQN/AuxNet 梯度路由实测为零，因此本结果只否证「该辅助头在该权重下带来可确认强度收益」的假设，不否证 `final_rank_mc` operational 配置。

## 5. 终止事项（停止规则）

- 不追加对局；
- 不调整 auxiliary weight；
- 不更换 head 结构；
- 不挑选 seed；
- **不继续 score-to-go 主目标或辅助目标路线。**

score-to-go 方向（含主 Q-target 与 auxiliary 两种用法）至此全部按预注册规则正式关闭。

## 6. 后续方向

下一科研方向转向 **targeted high-signal decision filtering**；具体实验规范由研究决策方另行下达，`next_experiment = null / not_selected`，未预先登记任何候选。

## 7. Artifact 索引

| Artifact | 路径 | SHA256 |
| :--- | :--- | :--- |
| 训练 manifest | `artifacts/experiments/S1_score_to_go_auxiliary_multiseed_2026_09/training/s1_training_manifest.json` | `c02414346a23f6636325338a2bad500ad1114adf30321562c360ea010d04af4a` |
| 3 AuxVariant checkpoints | `…/training/mortal_aux_variant_70400_seed_*.pth` | 见 2.1 |
| 评测 manifest | `…/evaluation/s1_eval_manifest.json` | `f89bd842970178b5557b83184e05625c0d460ad864531adc9631141dfce9b65d` |
| 3000 原始日志 | `…/evaluation/panel_seed_*/shard_*/logs/` | — |
| 汇总 | `…/summary/s1_summary.json` | `e6d6f3ddcc54b068d805abff417911535a9ca4591068461b724aae22249b8ce4` |
