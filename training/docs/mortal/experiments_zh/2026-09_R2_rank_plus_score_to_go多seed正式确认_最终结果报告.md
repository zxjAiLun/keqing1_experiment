# R2 rank_plus_score_to_go 多 seed 正式确认实验 — 最终结果报告

> 实验身份：`R2_rank_plus_score_to_go_multiseed_confirmation_2026_09`
> 状态：**CLOSED / not_supported / K1 = null**
> 本报告为不可变最终结果记录，一经提交不覆盖。

---

## 1. 实验定位与预注册假设

R2 是 R1 pilot（`weak_positive`）的正式多 seed 确认实验。预注册假设：若 `rank_plus_score_to_go_mc` 稠密回报目标具有真实的强度收益，则在三个全新训练 seeds 上：

- Primary（Variant − Control）每 seed 均值全为正，且 crossed-bootstrap CI95 下界 > 0；
- Absolute（Variant − K0_70k）每 seed 均值全为正，且 CI95 下界 > 0。

两式同时满足才判定 `promotion_supported`；仅 Primary 满足判定 `reward_effect_only`；否则 `not_supported`。

### 1.1 冻结契约（与 R1 最终版完全一致的 reward-only 协议）

| 项 | 值 |
| :--- | :--- |
| Parent model | `K0_70k`（SHA `6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`） |
| 数据 | M0 mixed replay corpus（`file_index_m0.pth`，SHA `755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2`） |
| Objective | `behavior_action_mc` / `behavior_action_q`（与 M0 operational 完全一致） |
| Trainable 视角 | 仅 `["ext_mortal"]` |
| Control reward | `final_rank_mc`（centered [+3,+1,−1,−3]） |
| Variant reward | `rank_plus_score_to_go_mc`：rank_target + 0.25 × clip((final_score − kyoku_start_score)/10000, −3, +3) |
| 训练 seeds | 20260910 / 20260911 / 20260912（全新，与 R1 的 20260807 无重叠） |
| 步数 | 70000 → 70400，每条件精确 400 optimizer steps，preserved Adam moments |
| Batch/LR/CQL | 512 / 1e-4 / CQL min_q 5.0，aux 0.2，γ=1.0 |
| Row identity | 每 seed Control/Variant 全 400 batch 的 reward 外字段（obs, actions, masks, steps_to_done, player_ranks）滚动 SHA256 必须逐字节一致 |
| 评测 | 每 seed 1000 局四人同桌（`four_player_native --seat-mode=random`），公共随机数 seeds 2300000..2300999，seed key 8192，4 shards × 250 |
| 阵容 | `K0_70k, ext_mortal, Control_70400, Variant_70400` |
| 统计 | Crossed bootstrap（seed 轴 × game-id 轴，5000 reps，seed 20260910），Primary/Absolute 共享重采样索引 |

---

## 2. 执行记录

### 2.1 训练（7/7 hard gates PASS）

3 seeds × 2 条件 = 6 次 400-step 训练全部完成。每 seed 的全 400 batch row digest Control/Variant 严格一致：

| Seed | Control digest = Variant digest | identical |
| :--- | :--- | :---: |
| 20260910 | `2e1e41ad31487fa19953d6e1cd1cd777c76d2229592ffdaf2a62943b7c30c013` | ✓ |
| 20260911 | `e503dc4043b21ae05382172a9071b358dc504bd55713b8c40373a6c93dd569c9` | ✓ |
| 20260912 | `fdeb82219f8e0b7bd9e306949377f07e0f0e2424d505751fe1a648149f8b3fba` | ✓ |

六个 checkpoint（SHA256，已做 training manifest ↔ eval manifest ↔ disk 三向绑定校验）：

| Checkpoint | SHA256 |
| :--- | :--- |
| `mortal_control_70400_seed_20260910.pth` | `fe7fecf86d8e99b107ee974ab73ad7f590d8986ce493b1165241dc33cbc9df92` |
| `mortal_variant_70400_seed_20260910.pth` | `0f067f5eeea0307b29b42e1eb7f4e4d5bc461253efe690be253dece906f8d559` |
| `mortal_control_70400_seed_20260911.pth` | `8b3cad1a0a5ab43c529f7f8ce7c6c8c017f079f578e9ba3db28b4168ab9ed849` |
| `mortal_variant_70400_seed_20260911.pth` | `f2948892a0ed0db294f9fceb97294159f4fda1f9951baba3c4616dc46e67b9df` |
| `mortal_control_70400_seed_20260912.pth` | `cee3fd1e780f8d0b1bf13d550e90084802ec1cf22ca9ceffe38f003d92b1eeb2` |
| `mortal_variant_70400_seed_20260912.pth` | `756323a028ae89cd3503a82d0e0fbc47ec43a842958092e2e8ec8bfda8e8a0d6` |

训练 manifest SHA256：`dddc217210abd7016434cadae9015303c4ac207fec3c3eee345c183653969e92`。

### 2.2 评测（7/7 hard gates PASS）

3 panels × 1000 局 = 3000 局完整对局。每个 panel 的 1000 份日志经 fail-closed 校验：文件名 `{game_id}_8192_{a/b/c/d}.json.gz`、`start_game.seed=[game_id, 8192]` 与文件名一致、阵容精确匹配、`end_game` 完整结束、`reach_accepted` 结构语义正确、game ID 连续唯一覆盖 `2300000..2300999`。日志发现采用 glob 而非硬编码 `_0` 后缀（`four_player_native --seat-mode=random` 实际生成 a/b/c/d 后缀）。

评测 manifest SHA256：`0e92985d2dc7926fc268f249ab304f0e2b3adf105da13cc4edd368aa2063d403`。

### 2.3 汇总（7/7 hard gates PASS）

`r2_summary.json` SHA256：`7d6793960e92788320649c25b07ac90f7b96f80634406edf5b9119ebdfb69962`。

---

## 3. 正式统计结果

### 3.1 Primary：Variant − Control（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **+2.475** |
| 20260911 | **−3.600** |
| 20260912 | **+0.630** |
| **总体（grand mean）** | **−0.165** |

Crossed-bootstrap CI95：**[−7.620, +7.560]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.2 Absolute：Variant − K0_70k（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **+3.600** |
| 20260911 | **−6.840** |
| 20260912 | **+4.185** |
| **总体（grand mean）** | **+0.315** |

Crossed-bootstrap CI95：**[−9.121, +9.405]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.3 机械裁决

```text
verdict               not_supported
recipe_promotion      false
checkpoint_promotion  false
K1                    null
```

---

## 4. 科学结论

1. **R1 的 `weak_positive` 未能在三个 fresh seeds 中复现。** R1 pilot 观测到的 Primary +6.480 pt（CI 跨零）在 R2 中收缩为 −0.165 pt，方向不一致（seed 20260911 为负），CI 宽且跨零。R1 的正向信号更符合单 seed 抽样波动而非真实收益。
2. **Primary 总体接近零且 seed 方向分裂**（[+2.475, −3.600, +0.630]），不能支持 reward shaping 有效的结论。
3. **Absolute 同样接近零（+0.315）、CI 很宽（[−9.121, +9.405]）、seed 方向分裂**，不存在任何 checkpoint 晋级依据；尤其不得挑选表现较好的 seed（如 20260910）或其 checkpoint。
4. **不能宣称该回报必然有害。** 准确结论是：在当前 400-step continuation、0.25 权重、±3 clip 协议下，**没有可复现的收益证据**。

## 5. 终止事项（预注册停止规则）

- 不追加评测；
- 不做 score-to-go 权重 sweep；
- 不挑选表现较好的 seed；
- 不晋级 20260911（或任何 seed）的 checkpoint；
- **正式终止 rank_plus_score_to_go 作为主 Q-target 的路线。**

## 6. 后续路线建议

保持 `final_rank_mc` 主目标与 `behavior_action_mc` objective 不变，把 score-to-go 转移为**独立的 auxiliary representation signal**（不进入主 Q target）。该方向需建立新的 preregistered experiment ID，用以检验稠密信号是否具有辅助表示价值，同时避免再次污染主 Q target。

## 7. Artifact 索引

| Artifact | 路径 |
| :--- | :--- |
| 训练 manifest | `artifacts/experiments/R2_rank_plus_score_to_go_multiseed_confirmation_2026_09/training/r2_training_manifest.json` |
| 6 checkpoints | `artifacts/experiments/R2_rank_plus_score_to_go_multiseed_confirmation_2026_09/training/mortal_{control,variant}_70400_seed_*.pth` |
| 评测 manifest | `artifacts/experiments/R2_rank_plus_score_to_go_multiseed_confirmation_2026_09/evaluation/r2_eval_manifest.json` |
| 3000 原始日志 | `artifacts/experiments/R2_rank_plus_score_to_go_multiseed_confirmation_2026_09/evaluation/panel_seed_*/shard_*/logs/` |
| 汇总 | `artifacts/experiments/R2_rank_plus_score_to_go_multiseed_confirmation_2026_09/summary/r2_summary.json` |
