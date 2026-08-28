# F2 无重采样 disagreement 温和加权 Pilot — 最终结果报告

> 实验身份：`F2_disagreement_weighting_without_resampling_pilot_2026_09`
> 状态：**CLOSED / not_supported / K1 = null**
> 本报告为不可变最终结果记录，一经提交不覆盖。

---

## 1. 实验定位与预注册假设

F2 是 F1（disagreement-priority 带放回重采样，显著有害）之后的温和对照：在同一 disagreement 判定与同一 base stream 上，移除重采样，改为**无重采样的 loss 行加权**——每批 512 行全部使用一次，disagreement 行权重 2.0、其余 1.0，按 batch 权重和归一化，同一权重应用于逐行 value、CQL、next-rank loss。

预注册三态裁决（adaptive pilot，无论结果如何均不直接晋升 K1）：

- `reweighting_promising`：Primary（WeightedVariant − R2 Control）与 Absolute（WeightedVariant − K0）均 3/3 seed 均值为正且 CI95 下界 > 0；
- `control_recovery_only`：仅 Primary 满足；
- `not_supported`：Primary 未满足。

### 1.1 冻结契约

| 项 | 值 |
| :--- | :--- |
| Parent model | `K0_70k`（SHA `6c0e7005…49e0`） |
| 数据 | M0 mixed replay corpus；与 R2 Control 完全相同 base stream（base_row_digest 逐一绑定） |
| Objective / reward | `behavior_action_mc` / `final_rank_mc`（不变） |
| Control | R2 冻结 Control checkpoints（seeds 20260910/11/12，SHA 三向绑定，不重训） |
| 加权 | 无重采样、无丢弃、无复制；权重 {2.0, 1.0}；weighted_mean(x) = Σ(w·x_row)/Σw；同一权重应用于 value/CQL/next-rank 三项逐行 loss |
| 冻结 scorer | 与 F1 完全相同（eval、float32、无 AMP、参数 bit-exact、不注册 optimizer；actions != argmax legal Q；无 Q-margin 阈值） |
| Optimizer | preserved 410 K0 Adam moments（两组 [165,245]，无新参数组） |
| 评测 | 每 seed 1000 局四人同桌，game IDs 2600000..2600999，seed key 8192，4 shards × 250，seat-mode random |
| 统计 | Crossed bootstrap（5000 reps，seed 20261001），Primary/Absolute 共享重采样索引 |
| 禁止 | 权重网格、margin threshold、任何新 target |

## 2. 执行记录

### 2.1 训练（12/12 hard gates PASS）

三 seed 全部完成；`base_row_digest` 三 seed 精确匹配冻结 R2 Control digest（`2e1e41ad…` / `e503dc40…` / `fdeb8221…`）；每批 rows_used=512（每行恰好一次）；主 Q target 逐批验证 ∈ {+3,+1,−1,−3}；scorer 参数前后 bit-exact。

| Seed | disagree min/max（每批） | rate 均值 |
| :--- | :--- | :--- |
| 20260910 | 34 / 80 | 0.1089 |
| 20260911 | 38 / 74 | 0.1104 |
| 20260912 | 34 / 79 | 0.1100 |

（与 F1 完全一致，佐证同一判定同一 stream。）

训练 manifest SHA256：`dc8c16318e953bfe71c7b0b71fec62b1fff6c3e09c2deac00e6f0b09b00b741e`。

三个 WeightedVariant checkpoint（SHA256，三向绑定校验通过）：

| Checkpoint | SHA256 |
| :--- | :--- |
| `mortal_weighted_variant_70400_seed_20260910.pth` | `c116e19b6f13b2100c7218a1c9b30b39ac2f061555a0210e9b60854d0967af29` |
| `mortal_weighted_variant_70400_seed_20260911.pth` | `e1c014f0d68eb3220325024e3c521f74fd6c2c4d1279c1b0e4e760050f9f8e2e` |
| `mortal_weighted_variant_70400_seed_20260912.pth` | `a5c41a2504950a66dfc0401affd87ea30a4cec75eb6a06ccd41e7b25f219e5a3` |

### 2.2 评测（7/7 hard gates PASS）

3 panels × 1000 局 = 3000 局完整对局，game ID 2600000..2600999 连续唯一无缺失，单次运行无中断。评测 manifest SHA256：`1d11188f461778f3bef76b844f555c659e2631e39517d9a6a39320fed3db058c`。

### 2.3 汇总（7/7 hard gates PASS）

`f2_summary.json` SHA256：`201f5b4c7f06016c1802e2fbe9dd9feb1eb5928cd56d683680371bb6453bf4e7`。

## 3. 正式统计结果

### 3.1 Primary：WeightedVariant − R2 Control（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **−4.410** |
| 20260911 | **−2.295** |
| 20260912 | **−0.405** |
| **总体（grand mean）** | **−2.370** |

Crossed-bootstrap CI95：**[−9.405, +4.576]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.2 Absolute：WeightedVariant − K0_70k（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **+5.040** |
| 20260911 | **+0.135** |
| 20260912 | **+1.305** |
| **总体（grand mean）** | **+2.160** |

Crossed-bootstrap CI95：**[−5.265, +9.601]**（跨零）。
`all_seed_means_positive = true`；`ci_lower_positive = false`。

### 3.3 机械裁决

```text
verdict               not_supported
recipe_promotion      false
checkpoint_promotion  false
K1                    null（adaptive pilot：任何裁决均不直接晋升）
```

## 4. 科学结论（证据边界）

1. **本实验支持的结论只有一条：2× disagreement 加权没有改善 R2 Control。** Primary 3/3 负向（−4.410/−2.295/−0.405）、CI 跨零，未通过预注册 Primary 判据。
2. **F1→F2 的 Primary 回升（−12.03 → −2.37）与「重采样伤害」假设一致**：在同为 2× effective emphasis 的两实验间，移除带放回重采样后重度负效应基本消失。但这只是跨实验描述性对照，两个实验的对照条件不同，**不能完成机制归因**——无法区分「重复行破坏梯度多样性」与「行分布偏移」等候选解释，也未对机制做任何直接测量。
3. **Absolute 3/3 正向（+5.04/+0.135/+1.305，CI 跨零）不构成任何晋级证据**，且按预注册判据（Primary 全过才可进入 `control_recovery_only`）甚至不改变裁决状态。
4. **不支持的推断**：温和加权本身无害/有益、disagreement 行信号质量、以及任何关于 F1 负效应来源的因果结论。

## 5. 终止事项（停止规则）

- F2 身份下不做权重网格（含 0.5/1.5/3× 等）、不做 margin threshold、不做比例 sweep；
- 不挑选 seed、不追加对局、不晋级任何 checkpoint；
- 后续 disagreement 路线研究建立新 preregistered experiment ID（F3 反向 0.5× downweighting pilot 已按此规则建立，为该路线最后一个实验）。

## 6. 后续路线

F3（`F3_disagreement_downweighting_pilot_2026_09`）：同一判定、同一 base stream、同一无重采样协议，把 disagreement 行权重改为 **0.5**（agreement 1.0）。若 F3 裁决不是 `downweighting_promising`，永久关闭 K0-disagreement weighting/filtering 路线，不再做权重、比例或 margin sweep。

## 7. Artifact 索引

| Artifact | 路径 | SHA256 |
| :--- | :--- | :--- |
| 训练 manifest | `artifacts/experiments/F2_disagreement_weighting_without_resampling_pilot_2026_09/training/f2_training_manifest.json` | `dc8c16318e953bfe71c7b0b71fec62b1fff6c3e09c2deac00e6f0b09b00b741e` |
| 3 WeightedVariant checkpoints | `…/training/mortal_weighted_variant_70400_seed_*.pth` | 见 2.1 |
| 评测 manifest | `…/evaluation/f2_eval_manifest.json` | `1d11188f461778f3bef76b844f555c659e2631e39517d9a6a39320fed3db058c` |
| 3000 原始日志 | `…/evaluation/panel_seed_*/shard_*/logs/` | — |
| 汇总 | `…/summary/f2_summary.json` | `201f5b4c7f06016c1802e2fbe9dd9feb1eb5928cd56d683680371bb6453bf4e7` |
