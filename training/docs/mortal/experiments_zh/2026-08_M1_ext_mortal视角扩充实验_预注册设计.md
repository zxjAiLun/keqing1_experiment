# M1 ext_mortal 视角混合语料扩充实验：预注册设计

> 实验状态：**预注册实现完成（implemented_not_started）；包含数据契约、训练启动器、完成验证器、独立评测与判定规则；暂未准备正式数据，暂未启动训练与评测。**

## 1. 路线重置背景与研究目的

- **实验 ID**: `M1_ext_mixed_expansion_2026_08`
- **路线重置理由**:
  1. C1/C2/C3 针对 D1 数据路线与 CQL 关闭机制的探究已完全收口：C3 明确证实 `D1_CQL_OFF` 对比 `K0_70k` 和 `M0_CURRENT` 均显著为负，D1/CQL_OFF 分支永久终止；
  2. 在全库历史中，**M0 路线**（混合生态下的 ext_mortal 视角）是唯一曾出现 3/3 seed 正向对 K0 的路线，且 `ext_mortal` 是现有冻结生态中最强的行为策略来源；
  3. D1 牌谱生成阶段共包含 6,000 局四人生态对局，其中每局均包含 1 个 `ext_mortal` 视角，但历史上 D1/D2 仅提取了 K0 与后代模型的视角进行训练，其高质量的 `ext_mortal` 视角从未被利用；
  4. M1 路线不引入任何新生成的牌谱，不扫描超参数或优化器，而是将 M0 的 6,000 局 ext_mortal 视角与 D1 牌谱中的 6,000 局 ext_mortal 视角合并，构建 **12,000 局规模的 ext_mortal 混合生态训练语料**。
- **实验目标**: 检验 12,000 局 M1 语料训练的 checkpoint 是否在绝对对局强度上同时优于：
  1. 正式 operational continuation control **`M0_CURRENT`**
  2. 正式 lineage 根模型 **`K0_70k`**
- **晋级目标**: 若统计条件全量满足，将预先指定的 canonical checkpoint `M1_CURRENT_20260807` 晋级为正式 **`K1`**；若任一条件失败，判定为 `not_supported`，不晋级任何 recipe/checkpoint，K1 保持 `null`。

## 2. 数据变量与严格验证要求

### 2.1 Control 与 Variant 定义
- **Control**: `M0_operational_control`（6,000 hanchans，trainable perspective = `ext_mortal`）
- **Variant (M1)**:
  $$\text{M1 Corpus} = \text{M0 全部 6,000 局} + \text{D1 Generation 全部 6,000 局的 ext\_mortal 视角}$$
  总计：**12,000 个独立半庄，12,000 个 trainable perspectives**。

### 2.2 严格数据验证要求
- **零修改**: M0 原始 `file_index_m0.pth` 与数据文件保持严格只读，不得修改；
- **单一视角**: 确认 D1 每局中恰好存在 1 个 `ext_mortal` 角色；
- **隔离性**: 绝对不使用 D1 的 K0 视角或 D2 的 V2/V3 视角分配；
- **零重叠**: M0 与 D1 两部分牌谱的 canonical hanchan ID 严格零重叠；
- **唯一性**: 12,000 个牌谱、视角与 loader 条目完整唯一；
- **行为契约**: 行为动作全部合法，训练 target 严格为 $\{-3, -1, +1, +3\}$。

## 3. 训练配置与实验矩阵

每个 training seed 训练 1 个 M1 checkpoint（共 3 个 checkpoints）：

```text
Parent checkpoint:       K0_70k (mortal_default_70k_promoted_candidate.pth)
Training steps:          70,000 → 72,000 (2,000 steps)
Training seeds:          20260806 / 20260807 / 20260808
Optimizer:               preserved K0 Adam (保留 70k checkpoint 的 Adam 矩)
Scheduler / Scaler:      fresh
Batch size:              512
Opt step every:          1
Device:                  cuda:0
AMP 混合精度:            false
Objective:               behavior_action_mc
Reward:                  final_rank_mc
CQL min_q_weight:        5.0
Aux next-rank weight:    0.2
Freeze BN:               mortal: false
Archive steps:           70001, 70010, 70100, 70500, 71000, 72000
```

> **说明**: 除 `file_index` 与训练语料规模扩大为 12,000 局外，其余所有训练配置与超参数与 `M0_CURRENT` 严格完全一致。已有的 `M0_CURRENT_<seed>` 作为 control 直接复用，不重新训练。

## 4. 独立评测设计 (Fresh Evaluation)

固定 4-player 牌桌模型顺序：

```text
[0] 70k (K0_70k promoted candidate, SHA256: 6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0)
[1] ext_mortal (External Mortal 20240308, SHA256: 0a88ddad649804d085491b5397d895f596b0e55f30632c549ea145bb44786563)
[2] M0_CURRENT_<seed> (M0 control 72k checkpoint, SHA256 matches D1_PREP)
[3] M1_CURRENT_<seed> (M1 variant 72k checkpoint)
```

### 评测分片与参数
```text
总对局量 (TOTAL_GAMES):       3000
分片总数 (TOTAL_SHARDS):      12 (4 shards / seed)
每分片局数 (GAMES_PER_SHARD): 250
随机座次密钥 (SEED_KEY):      8192
座次模式 (SEAT_MODE):         random
执行设备 (DEVICE):            cuda
AMP 混合精度:                 false (--no-amp)
顺位点数契约 (RANK_POINTS):    [+90, +45, 0, -135]
```

### 独立对局 ID 区间
- **Seed 20260806**: hanchan 1930000..1930999 (`shard_00..03`)
- **Seed 20260807**: hanchan 1940000..1940999 (`shard_04..07`)
- **Seed 20260808**: hanchan 1950000..1950999 (`shard_08..11`)

## 5. 统计口径与晋级判定规则

在每个 seed 的每个同桌半庄中，定义两组配对差值：

$$x = \text{Pt}(\text{M1}) - \text{Pt}(\text{M0\_CURRENT})$$
$$y = \text{Pt}(\text{M1}) - \text{Pt}(\text{K0\_70k})$$

### Bootstrap 协议
- **抽样方法**: Equal-seed hierarchical paired bootstrap
  - Outer: 有放回重抽 3 个 training seeds；
  - Inner: 针对抽中的每个 seed，有放回重抽 1,000 局对局；
  - $x$ 与 $y$ 严格共享相同的 outer 与 inner 抽样索引。
- **Reps**: 5000
- **Seed**: 20260830
- **置信区间**: 95%

### 判定逻辑
必须**同时**满足以下全部条件：
1. **$x$ vs M0_CURRENT**:
   - 3/3 seed means > 0
   - Hierarchical CI95 下限 > 0
2. **$y$ vs K0_70k**:
   - 3/3 seed means > 0
   - Hierarchical CI95 下限 > 0
3. **Hard Gates**: 全部 baseline 及 M1 checkpoints 存在且 SHA 校验通过；全部 12 shards (3,000 unique logs) 解析完整。

输出：
```text
verdict              promotion_supported
recipe_promotion     true
checkpoint_promotion true
K1                   M1_CURRENT_20260807
```

若任一统计条件不满足：
```text
verdict              not_supported
recipe_promotion     false
checkpoint_promotion false
K1                   null
```
