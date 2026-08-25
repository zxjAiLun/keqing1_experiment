# P2 确定性 Seed-Replay Counterfactual 目标质量评测 最终结果报告

> **实验 ID**: `P2_seed_replay_counterfactual_target_quality_2026_09`  
> **代码提交**: [`698e783ad1a1adbfbea9de255bc5f6a700432f86`](https://github.com/zxjAiLun/keqing1_experiment/commit/698e783ad1a1adbfbea9de255bc5f6a700432f86)  
> **Panel 清单**: `artifacts/experiments/P2_seed_replay_counterfactual_target_quality_2026_09/counterfactual_panel/counterfactual_panel_manifest.json`  
> **Panel 清单 SHA-256**: `103d98bf0c500a9b1d894e2c7b1a945a8269aefbdbac1d9ae37dc775c7026823`  
> **正式裁决总结**: `artifacts/experiments/P2_seed_replay_counterfactual_target_quality_2026_09/summary/p2_summary.json`  
> **裁决总结 SHA-256**: `014726b14a3dd1e9f40fc0db8cc98becb00cf2788febc9e9677adc5bbdbb3f4a`  
> **最终裁决**: **`counterfactual_target_quality_evaluated`**（目标质量可行但信号稀疏；Recipe 未晋级，Checkpoint 未晋级，K1 保持为 `null`）

---

## 1. 实验目标与科学定位

本实验旨在对基于确定性 Seed-Replay 的 Counterfactual Rollout 生成机制进行**样本级目标质量与信号密度评估**：
- **基座母体**: `K0_70k` (`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`)
- **样本规模**: 128 局（种子区间 `3000000..3000127`，`seed_key = 8192`），评估席位固定为席位 0 (Split `a`)。
- **干预机制**: 在每局基线运行中捕获首个合法非立直两选舍牌决策点，分别强制执行冻结的 Top1 与 Top2 动作完成独立 Rollout。
- **定位原则**: 纯只读数据生成与质量分析实验，**不训练模型、不创建 Checkpoint，K1 保持为 null**。

---

## 2. 硬门禁检验结果

### 2.1 Panel 生成门禁 (9/9 Passed)
1. `k0_parent_verified`: `True`
2. `exact_128_pairs_generated`: `True` (128 对 / 256 局完整半庄)
3. `seeds_strictly_contiguous`: `True` (`3000000..3000127`)
4. `focal_seat_verified`: `True` (Seat 0)
5. `all_target_contexts_intervened_exactly_once`: `True`
6. `all_prefixes_exact_matched`: `True` (干预前事件流 100% 逐字匹配)
7. `all_first_divergences_verified_dahai`: `True` (首次差异事件严格为目标席位 Top1 vs Top2 舍牌)
8. `all_branches_completed_end_game`: `True` (全部以 `end_game` 结束)
9. `scores_and_ranks_valid`: `True`

### 2.2 Summary 原始日志校验门禁 (8/8 Passed)
1. `manifest_verified`: `True`
2. `k0_parent_verified`: `True`
3. `exact_128_pairs_analyzed`: `True`
4. `seeds_contiguous`: `True`
5. `all_branch_logs_verified`: `True` (逐一复核 256 份 `.json.gz` 日志)
6. `canonical_content_hashes_verified`: `True` (排除 transient 耗时后的规范化内容哈希一致)
7. `independent_metrics_recalculated_match`: `True` (独立从原始日志重算指标与 Manifest 严格一致)
8. `bootstrap_computed`: `True` (5000 次配对 Bootstrap，seed=20260904)

---

## 3. 目标质量与差值分布统计

### 3.1 天凤段位点顺位差 $\Delta R = R(\text{top2}) - R(\text{top1})$
- **总配对数**: 128 对
- **平局对数 (Tie, $\Delta R = 0$)**: **107 对 (83.59%)**（单步次优决策在首个决策点绝大多数未传播至最终顺位差异）
- **非零对数 (Non-zero, $\Delta R \neq 0$)**: **21 对 (16.41%)**
  - **Top2 顺位更优 ($\Delta R > 0$)**: 9 对 (7.03%)
  - **Top1 顺位更优 ($\Delta R < 0$)**: 12 对 (9.38%)
- **全样本均值 (Mean Difference)**: **`-6.328 pt`**
- **95% Bootstrap CI**: **`[-14.4141, +1.4062] pt`**（跨零且均值为负）

### 3.2 最终素点差 $\Delta S = S(\text{top2}) - S(\text{top1})$
- **总配对数**: 128 对
- **平局对数 ($\Delta S = 0$)**: **93 对 (72.66%)**
- **非零对数 ($\Delta S \neq 0$)**: **35 对 (27.34%)**
  - **Top2 素点更高 ($\Delta S > 0$)**: 17 对 (13.28%)
  - **Top1 素点更高 ($\Delta S < 0$)**: 18 对 (14.06%)
- **全样本均值 (Mean Difference)**: **`-857.03 点`**
- **95% Bootstrap CI**: **`[-2372.68, +548.55] 点`**（跨零且均值为负）

### 3.3 决策边际（Margin）子组分解
- **Tight Margin ($\le 0.5$)**: 23 对 (17.97%)，顺位差非零率 0.0%，均值 `0.0 pt` (CI: `[0.0, 0.0]`)。
- **Wide Margin ($> 0.5$)**: 105 对 (82.03%)，顺位差非零率 20.00% (21/105)，均值 `-7.714 pt` (CI: `[-18.0, +1.7143]`)。

---

## 4. 科学结论与下一步

1. **工程可行性**:
   - 证明了基于确定性 Seed-Replay 机制无需修改 Rust/PyO3 即可稳定生成成对、可复现的 Counterfactual Rollout 样本。
2. **目标质量与稀疏性**:
   - 当前“首个决策点 + 单次终局回报”机制标签信号极其稀疏（顺位非零率 16.41%，素点非零率 27.34%）；
   - Top1 与 Top2 方向差异均未达到统计确定性，不能宣称 Top1 或 Top2 显著更强。
3. **下一步路线**:
   - 针对开局早期决策被后续漫长对局稀释的问题，进行最后的无训练密度筛选实验：`P3_late_decision_counterfactual_signal_density_2026_09`（改为在每局最后一个非立直两选舍牌点进行干预，评估素点非零率能否达到 40% 以上并产生显著密度跃升）。
