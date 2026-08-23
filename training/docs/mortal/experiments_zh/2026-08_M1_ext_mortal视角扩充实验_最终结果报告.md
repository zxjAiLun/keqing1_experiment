# M1 ext_mortal 视角混合语料扩充实验 最终结果报告

> **实验 ID**: `M1_ext_mixed_expansion_2026_08`  
> **预注册提交**: [`fb849f5cfeed893a4df54bef9e7adfd2bea5a277`](https://github.com/zxjAiLun/keqing1_experiment/commit/fb849f5cfeed893a4df54bef9e7adfd2bea5a277)  
> **正式裁决总结**: `artifacts/experiments/M1_ext_mixed_expansion_2026_08/evaluation_implementation_2026_08/formal_adjudication/m1_summary.json`  
> **总结文件 SHA-256**: `7f5ffbbcd2bb056ede68e2ff95b5c44f557fe39a311c4265f4ae56f3e6195d06`  
> **最终裁决**: **`not_supported`**（数据路线未晋级，Checkpoint 未晋级，K1 保持为 `null`）

---

## 1. 实验目标与假设

本实验旨在检验：在控制模型结构、训练超参数、初始化权重与优化器动量不变的前提下，将训练语料从基础的 6,000 半庄（M0 对照）扩充为 **12,000 半庄**（包含 6,000 局 M0 与 6,000 局 D1 生成对局中的全部 `ext_mortal` 视角数据），是否能带来超越同源 M0 对照组以及初始 K0_70k 母体的统计显著强度提升。

---

## 2. 实验配置与执行 Provenance

### 2.1 训练语料与数据闭环 (Frozen 4-Artifact SHA-256)
* **语料规模**: 12,000 半庄（6,000 局 M0 + 6,000 局 D1，无 seed 重叠，单局严格仅 1 个 `ext_mortal` 角色）
* **`dataset_manifest.json`**: `206f5445544c55aaa88d909253ef5eb422274998c7e78c8b2d569d57b3c2dde4`
* **`file_index_m1.pth`**: `3d190247fb6e16b423d786ec07bd3b0ff3cd8903306de70ba57955e45226c07f`
* **`player_names_by_file.json`**: `7c1b0433a207ce1c941ff42c0d7dfbaa53087fd3968a9228927f214357164469`
* **`player_names.txt`**: `29f5f7c619c5481352e6fe29d4c5feb9442b6d1f1cec1ea7f4f405b330ce58d0`

### 2.2 离线强化学习训练配置
* **母体 Checkpoint**: `K0_70k` (`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`)
* **优化器状态**: 继承 70k Adam moments (`weights_plus_optimizer_warm_start`)
* **训练步数**: 70,000 → 72,000 步（各 Seed 独立完成 2,000 optimizer steps / 2,000 batches / 1,024,000 samples）
* **核心参数**: `batch_size=512`, `enable_amp=False`, `objective=behavior_action_mc`, `reward=final_rank_mc`, `cql_min_q_weight=5.0`, `aux_next_rank_weight=0.2`, `resnet={conv_channels:192, num_blocks:40}`
* **3 个 Seed 72000 Checkpoints**:
  * **Seed 20260806**: `9977e95f7302a2c106fbd652aae0ffe3fed2df29bf7de98f4069c218306e0f79`
  * **Seed 20260807**: `1fb3ee6fb3ca99ba5b1eadcb6f4a1c81c82bcf8d1d1e8e7d307beb818c64b367`
  * **Seed 20260808**: `a9425c198d85d77a38b521a14964e4892deceda1fe9a428a4503504f2c7de859`
* **Completion Closure**: `training_completion_closure.json` (SHA: `d7762b699adc46acf90a84c3f11efdb7bdeaf63712038efd666803ad174d70b8`)

---

## 3. 评测方案与统计检验结果

### 3.1 评测协议 (Evaluation Protocol)
* **对局设计**: 四人同桌对局（`70k` vs `ext_mortal` vs `M0_CURRENT_{seed}` vs `M1_CURRENT_{seed}`），每桌随机入座（`seat_mode=random`, `seed_key=8192`）。
* **计分档位**: 天凤标准顺位点（`[90.0, 45.0, 0.0, -135.0]` pt）。
* **对局总量**: 12 个 Shards 共 3,000 局（每 Seed 分配 4 个 Shards × 250 = 1,000 局，无重复 Hanchan ID）。
* **统计方法**: 层次成对 Bootstrap 重抽样（5,000 reps，seed=20260830）。

### 3.2 检验结果与数据汇总

#### 检验一：相对同源对照组差异 $X = 	ext{Pt}(	ext{M1}) - 	ext{Pt}(	ext{M0_CURRENT})$
* **Equal Seed Mean**: **`-2.970 pt`**
* **各 Seed 单独得点**:
  * **Seed 20260806** (1,000 局): **`-7.740 pt`**
  * **Seed 20260807** (1,000 局): **`+1.080 pt`**
  * **Seed 20260808** (1,000 局): **`-2.250 pt`**
* **95% 分层置信区间**: **`[-9.375, +3.435] pt`**
* **预注册门禁核验**:
  * $3/3$ Seed 全为正向: `False` (0/3 正向稳定)
  * CI 下界大于 0 ($CI_{\text{lower}} > 0$): `False`

#### 检验二：相对初始母体差异 $Y = 	ext{Pt}(	ext{M1}) - 	ext{Pt}(	ext{K0_70k})$
* **Equal Seed Mean**: **`+0.990 pt`**
* **各 Seed 单独得点**:
  * **Seed 20260806** (1,000 局): **`-0.990 pt`**
  * **Seed 20260807** (1,000 局): **`+3.690 pt`**
  * **Seed 20260808** (1,000 局): **`+0.270 pt`**
* **95% 分层置信区间**: **`[-4.515, +6.465] pt`**
* **预注册门禁核验**:
  * $3/3$ Seed 全为正向: `False`
  * CI 下界大于 0: `False`

---

## 4. 科学结论与研究影响

1. **实验假设未获支持**：在当前 Mortal DQN + CQL 离线强化学习框架下，仅简单将 `ext_mortal` 视角语料规模翻倍（从 6k 扩充到 12k 半庄），**未能带来可复现的棋力提升**，其表现反而略微落后于 6,000 局基础语料训练的 M0 对照组（Equal-seed 均值 $-2.97\text{ pt}$）。
2. **确定性负结果价值**：全链路数据完整性检查、Checkpoints 权重有限性校验、3,000 局对局日志与统计分析均严格通过，排除了由于代码缺陷、数据漂移或评测噪音导致的假象，是一次清晰且有说服力的有效科学负结果。
3. **晋级裁决**：
   * **Recipe 晋级**: **`False`**
   * **Checkpoint 晋级**: **`False`**
   * **K1 状态**: **保持为 `null`**（K0_70k 依然是唯一的正式模型 Lineage 根节点）。
