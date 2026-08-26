# P3 局末决策点 Counterfactual 信号密度筛选 最终结果报告

> **实验 ID**: `P3_late_decision_counterfactual_signal_density_2026_09`  
> **代码提交**: [`97bbe8456a127afe4e9fe0277aa57707afdba11a`](https://github.com/zxjAiLun/keqing1_experiment/commit/97bbe8456a127afe4e9fe0277aa57707afdba11a)  
> **Panel 清单**: `artifacts/experiments/P3_late_decision_counterfactual_signal_density_2026_09/counterfactual_panel/counterfactual_panel_manifest.json`  
> **Panel 清单 SHA-256**: `63046a060e8aad9bc5d4487d242cd0890d6609831998c9cf299bc03a01a59851`  
> **正式裁决总结**: `artifacts/experiments/P3_late_decision_counterfactual_signal_density_2026_09/summary/p3_summary.json`  
> **裁决总结 SHA-256**: `243cbf30ff8f9626bd7c702616d1a8270f82fcceb4844252338d9a830c4815ee`  
> **最终裁决**: **`counterfactual_targets_insufficiently_dense`**（目标信号密度不足，正式终止“单次 Top1/Top2 分支 + 终局回报标签”路线；Recipe 未晋级，Checkpoint 未晋级，K1 保持为 `null`）

---

## 1. 实验目标与科学定位

本实验旨在对基于确定性 Seed-Replay 的 Counterfactual Rollout 生成机制进行**局末决策点干预下的信号密度评估**：
- **基座母体**: `K0_70k` (`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`)
- **样本规模**: 128 局（种子区间 `3100000..3100127`，`seed_key = 8192`），评估席位固定为席位 0 (Split `a`)。
- **干预机制**: 在每局基线运行中捕获**最后一个**合法非立直两选舍牌决策点，分别强制执行冻结的 Top1 与 Top2 动作完成独立 Rollout。
- **定位原则**: 纯只读数据生成与质量分析实验，**不训练模型、不创建 Checkpoint，K1 保持为 null**。

---

## 2. 硬门禁检验结果

### 2.1 Panel 生成门禁 (9/9 Passed)
1. `k0_parent_verified`: `True`
2. `exact_128_pairs_generated`: `True` (128 对 / 256 局完整半庄)
3. `seeds_strictly_contiguous`: `True` (`3100000..3100127`)
4. `focal_seat_verified`: `True` (Seat 0)
5. `all_target_contexts_intervened_exactly_once`: `True`
6. `all_prefixes_exact_matched`: `True` (干预前事件流在排除 transient wall-clock 耗时 `eval_time_ns` 后规范化 JSONL 序列 100% 逐字吻合)
7. `all_first_divergences_verified_dahai`: `True` (首次差异事件严格为目标席位 Top1 vs Top2 舍牌)
8. `all_branches_completed_end_game`: `True` (全部以 `end_game` 结束)
9. `scores_and_ranks_valid`: `True`

### 2.2 Summary 原始日志校验门禁 (9/9 Passed)
1. `manifest_verified`: `True`
2. `k0_parent_verified`: `True`
3. `exact_128_pairs_analyzed`: `True`
4. `seeds_contiguous`: `True`
5. `all_branch_logs_verified`: `True` (逐一复核 256 份 `.json.gz` 日志)
6. `canonical_content_hashes_verified`: `True` (排除 transient 耗时后的规范化内容哈希一致)
7. `independent_metrics_recalculated_match`: `True` (独立从原始日志重算指标与 Manifest 严格一致)
8. `p2_comparison_verified`: `True` (P2 权威摘要文件存在且哈希严格校验为 `014726b14a3dd1e9f40fc0db8cc98becb00cf2788febc9e9677adc5bbdbb3f4a`)
9. `bootstrap_computed`: `True` (5000 次两样本 Bootstrap，seed=20260905)

---

## 3. 目标质量与 P3 vs P2 信号密度对比

### 3.1 核心分布与差值统计 (P3 128 对)
- **最终素点差 $\Delta S = S(\text{top2}) - S(\text{top1})$**:
  - 非零对数: **49 对 (38.28%)**（平局 79 对 / 61.72%）；
  - Top2 素点更高: 24 对 (18.75%)，Top1 素点更高: 25 对 (19.53%)；
  - 均值: **`-321.88 点`** (95% CI: `[-925.78, +234.39] 点`)。
- **天凤段位点顺位差 $\Delta R = R(\text{top2}) - R(\text{top1})$**:
  - 非零对数: **15 对 (11.72%)**（平局 113 对 / 88.28%）；
  - Top2 顺位更优: 5 对 (3.91%)，Top1 顺位更优: 10 对 (7.81%)；
  - 均值: **`-2.8125 pt`** (95% CI: `[-8.7891, +3.1641] pt`)。

### 3.2 P3 vs P2 信号密度对比 (两样本 Bootstrap CI95，Seed 20260905)
| 指标 | P2 (首个决策点) | P3 (末位决策点) | $\Delta = \text{P3} - \text{P2}$ | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **素点非零率** | 27.34% (35/128) | **38.28% (49/128)** | **+10.94%** | **`[-0.78%, +22.66%]`** |
| **顺位非零率** | 16.41% (21/128) | **11.72% (15/128)** | **-4.69%** | **`[-13.28%, +3.91%]`** |

---

## 4. 科学结论与路线裁决

1. **预注册门禁核验**:
   - $\text{P3 score non-zero rate} \ge 40\%$：未满足 (`38.28% < 40%`)；
   - $\text{CI}_{95}^{\text{lower}}(\Delta \text{score non-zero rate}) > 0$：未满足 (`CI 下界为 -0.78% \le 0`)。
2. **科学定论**:
   - 证明了确定性 Seed-Replay 具备出色的工程可复现性；
   - 但无论是开局早期还是局末决策点，“单次 Top1/Top2 分支 + 终局回报标签”机制所产生的非零监督信号均过于稀疏，且未能在统计上产生显著密度跃升；
   - 按照预注册规则，**正式终止 counterfactual-target 单步终局回报路线**。
