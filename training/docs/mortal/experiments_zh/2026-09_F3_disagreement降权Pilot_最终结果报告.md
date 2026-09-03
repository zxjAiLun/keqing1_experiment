# F3 disagreement 降权 Pilot — 最终结果报告

> 实验身份：`F3_disagreement_downweighting_pilot_2026_09`
> 状态：**CLOSED / not_supported / K1 = null**
> 本报告为不可变最终结果记录，一经提交不覆盖。

---

## 1. 实验定位与预注册假设

F3 是 F1/F2 之后的最后一个 K0-disagreement 对照：保持同一 scorer、同一 base stream 和无重采样协议，只把 disagreement 行的 loss 权重降为 0.5，agreement 行保持 1.0。它检验的不是新的数据、reward 或模型结构，而是“当前 K0 不同意的行为行是否应当被抑制”。

预注册三态裁决（adaptive pilot，任何结果均不直接晋升 K1）：

- `downweighting_promising`：Primary（DownweightedVariant − R2 Control）与 Absolute（DownweightedVariant − K0）均 3/3 seed 均值为正且 CI95 下界 > 0；
- `control_recovery_only`：仅 Primary 满足；
- `not_supported`：Primary 未满足。

### 1.1 冻结契约

| 项 | 值 |
| :--- | :--- |
| Parent model | `K0_70k`（SHA `6c0e7005…49e0`） |
| 数据 | M0 mixed replay corpus；与 R2 Control 完全相同 base stream（base_row_digest 逐一绑定） |
| Objective / reward | `behavior_action_mc` / `final_rank_mc`（不变） |
| Control | R2 冻结 Control checkpoints（seeds 20260910/11/12，SHA 三向绑定，不重训） |
| 加权 | 无重采样、无丢弃、无复制；权重 {0.5, 1.0}；weighted_mean(x) = Σ(w·x_row)/Σw；同一权重应用于 value/CQL/next-rank 三项逐行 loss |
| 冻结 scorer | 与 F1/F2 完全相同（eval、float32、无 AMP、参数 bit-exact、不注册 optimizer；actions != argmax legal Q；无 Q-margin 阈值） |
| Optimizer | preserved 410 K0 Adam moments（两组 [165,245]，无新参数组） |
| 训练 | seeds 20260910/11/12；70000 → 70400；每 seed 精确 400 optimizer steps；batch 512 |
| 评测 | 每 seed 1000 局四人同桌，game IDs 2700000..2700999，seed key 8192，4 shards × 250，seat-mode random |
| 统计 | Crossed bootstrap（5000 reps，seed 20261002），Primary/Absolute 共享重采样索引 |
| 停止规则 | 若不是 `downweighting_promising`，永久关闭 K0-disagreement weighting/filtering 路线，不做权重、比例或 margin sweep |

## 2. 执行记录

### 2.1 训练（12/12 hard gates PASS）

三 seed 全部完成；`base_row_digest` 三 seed 精确匹配冻结 R2 Control digest（`2e1e41ad…` / `e503dc40…` / `fdeb8221…`）；每批 rows_used=512（每行恰好一次）；权重仅为 0.5/1.0；主 Q target 逐批验证 ∈ {+3,+1,−1,−3}；scorer 参数前后 bit-exact。

| Seed | disagree min/max（每批） | rate 均值 |
| :--- | :--- | :--- |
| 20260910 | 34 / 80 | 0.1089 |
| 20260911 | 38 / 74 | 0.1104 |
| 20260912 | 34 / 79 | 0.1100 |

训练 manifest SHA256：`4aea776bd5a1a248ca13a7ededfdcaedf353bb184fc861b0916c403fd54e0d3e`。

三个 DownweightedVariant checkpoint（SHA256，manifest ↔ eval manifest ↔ disk 三向绑定校验通过）：

| Checkpoint | SHA256 |
| :--- | :--- |
| `mortal_downweighted_variant_70400_seed_20260910.pth` | `7e1f999abd37ec036b5eaeb6f301fb25b259fd3d263dcbb64fee76720a7a0930` |
| `mortal_downweighted_variant_70400_seed_20260911.pth` | `3446b198a680713a3c1a91a2fbc459dc4bfd33cfa9ab411108003f4f3b922e5e` |
| `mortal_downweighted_variant_70400_seed_20260912.pth` | `fe495c2e507a6309b6633ae583d90ec4988a9b96cd31a5062b32ea6140c8d01b` |

过程备注：2026-09-01 的一次运行完成三 seed 后，因预期 hard-gate 键仍误用 F2 名称 `weights_2_1_normalized_all_batches`，在写 manifest 前 fail-closed；未据 checkpoint 补造 manifest。修正为 F3 的 `weights_0_5_1_normalized_all_batches` 并加入真实 manifest 组装回归测试后，2026-09-03 从空实验目录完成本次正式运行。

### 2.2 评测（7/7 hard gates PASS）

3 panels × 1000 局 = 3000 局完整对局；每个 panel 的 game ID 均连续唯一覆盖 2700000..2700999，无缺失。评测 manifest SHA256：`1ec865143d93f578a74090482aeff88ef124a4d2ab6ed97587d48f037af5f994`。

### 2.3 汇总（7/7 hard gates PASS）

`f3_summary.json` SHA256：`ed563b33f7a176f21e6a3615a5f1633548b39f66ce77a83d8f64c6f09d6022a0`。

## 3. 正式统计结果

### 3.1 Primary：DownweightedVariant − R2 Control（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **−1.350** |
| 20260911 | **+3.690** |
| 20260912 | **+6.750** |
| **总体（grand mean）** | **+3.030** |

Crossed-bootstrap CI95：**[−5.145, +11.235]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.2 Absolute：DownweightedVariant − K0_70k（Pt/半庄）

| Seed | 均值 |
| :--- | ---: |
| 20260910 | **+0.270** |
| 20260911 | **−4.275** |
| 20260912 | **−2.610** |
| **总体（grand mean）** | **−2.205** |

Crossed-bootstrap CI95：**[−9.810, +5.282]**（跨零）。
`all_seed_means_positive = false`；`ci_lower_positive = false`。

### 3.3 机械裁决

```text
verdict               not_supported
recipe_promotion      false
checkpoint_promotion  false
K1                    null
```

## 4. 科学结论（证据边界）

1. **0.5× disagreement 降权没有形成稳定收益证据。** Primary 总体均值虽为 +3.030 pt，但 seed 20260910 为负且 CI95 跨零，未满足预注册 Primary 判据。
2. **F3 没有超越 K0 的证据。** Absolute 总体为 −2.205 pt，2/3 seed 为负且 CI95 跨零；不能把 Primary 的正中心值解释为 K1 候选。
3. **F1/F2/F3 共同否证的是当前三种具体协议，不支持扩大结论。** F1 的带放回 priority 明显有害；F2 的 2× 加权未改善 Control；F3 的 0.5× 降权也不稳定。这不能证明所有 high-signal 方法无效，但足以按预注册规则关闭基于该 K0 disagreement 判定的采样、加权和过滤路线。
4. **不支持的推断：** 挑选 20260912 seed、追加对局后重新裁决、把权重调成其他数值、加入 Q-margin 后沿用 F3 身份，或声称 disagreement 行普遍“好”或“坏”。

## 5. 终止事项（停止规则）

- 不重跑 F3，不追加对局，不挑选 seed，不晋级任何 checkpoint；
- 不做其他 disagreement 权重、比例、阈值或 margin sweep；
- 永久关闭 K0-disagreement weighting/filtering 路线；
- K1 保持 `null`，下一个实验回到 `null / not_selected`。

## 6. 后续路线

下一阶段进行全新的 K1 route reset。具体实验身份与冻结契约尚未选择；本次收口不预注册新候选，也不启动训练。

## 7. Artifact 索引

| Artifact | 路径 | SHA256 |
| :--- | :--- | :--- |
| 训练 manifest | `artifacts/experiments/F3_disagreement_downweighting_pilot_2026_09/training/f3_training_manifest.json` | `4aea776bd5a1a248ca13a7ededfdcaedf353bb184fc861b0916c403fd54e0d3e` |
| 3 DownweightedVariant checkpoints | `…/training/mortal_downweighted_variant_70400_seed_*.pth` | 见 2.1 |
| 评测 manifest | `…/evaluation/f3_eval_manifest.json` | `1ec865143d93f578a74090482aeff88ef124a4d2ab6ed97587d48f037af5f994` |
| 3000 原始日志 | `…/evaluation/panel_seed_*/shard_*/logs/` | — |
| 汇总 | `…/summary/f3_summary.json` | `ed563b33f7a176f21e6a3615a5f1633548b39f66ce77a83d8f64c6f09d6022a0` |
