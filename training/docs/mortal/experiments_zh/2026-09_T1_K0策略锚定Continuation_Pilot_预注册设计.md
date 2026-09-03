# T1 K0 策略锚定 Continuation Pilot：预注册设计（FROZEN v1）

## 1. 状态与授权边界

```text
EXPERIMENT ID          T1_k0_policy_anchor_continuation_pilot_2026_09
PREREGISTRATION        FROZEN v1
CATEGORY               policy_stability_intervention
PARENT                 K0_70k
CONTROL                frozen R2 final_rank_mc Control, seeds 20260910/11/12
VARIANT                Control + K0 legal-policy KL anchor
TRAINING                authorized after implementation/tests pass
EVALUATION              authorized after training hard gates pass
DIRECT K1 PROMOTION     forbidden (adaptive pilot)
```

本预注册冻结一次完整的“训练 → 机器评测 → 自动裁决”实验。实现与运行均限定在 Windows 原生 CUDA 环境；不得使用 WSL，不创建新 Python 环境，不下载新的 PyTorch/CUDA 包。

## 2. 背景

历史 objective-learnability 审计在共享 482-state panel 上观察到，M0@72000 与 S0@72000 相对 K0 的 greedy action change 约为 8.5% / 8.3%；Q 变化中包含显著 centered-action-preference 与 margin movement。R2、S1、F2、F3 的结果又显示，多种 continuation Variant 偶尔能相对 Control 回升，但没有形成稳定超越 K0 的证据。

F1/F2/F3 已永久关闭的是 **按 K0/behavior disagreement 对训练行重采样或加权** 的路线。T1 不读取 behavior disagreement，不改变任何训练行权重，不筛选、不删除、不复制训练行；它检验完全不同的机制：是否可通过对冻结 K0 合法动作分布的 proximal anchor，抑制 continuation 对已有策略结构的破坏。

早期 KeqingRL teacher-CE 探针也不等同于 T1：那些探针让 KeqingRL consumer 模仿另一个 Mortal teacher，并以 top-K movement 为目标；T1 的 Mortal-native student 从 K0 本身初始化，anchor 在相同 eval-mode policy 上初始为零，其目标是抵抗后续漂移，而不是注入新的 hard/soft action label。

## 3. 可证伪假设

> 在 K0@70000、M0 mixed replay、preserved Adam、`behavior_action_mc + final_rank_mc + CQL 5.0 + next-rank aux 0.2` 与 400-step continuation 全部固定时，增加一个温和、全行等权的冻结 K0 legal-policy KL anchor，会在三个 matched seeds 上同时降低 policy drift，并使 Variant 相对 R2 Control、相对 K0 都获得可复现正收益。

单纯“更接近 K0”不等于更强；机制门和两层强度门必须分别报告。

## 4. 唯一干预变量

Control objective 保持不变：

```text
L_base = L_final_rank_mc + 5.0 * L_CQL + 0.2 * L_next_rank
```

Variant：

```text
P_K0      = softmax(center_legal(Q_K0) / 1.0)
P_current = softmax(center_legal(Q_current_eval) / 1.0)
L_anchor  = mean_rows KL(P_K0 || P_current)
L_T1      = L_base + 0.7867009210376891 * L_anchor
```

冻结细节：

- 只对合法动作计算；非法动作不贡献概率或梯度。
- `center_legal` 减去每行合法 Q 均值；目标不依赖 common Q offset。
- `temperature=1.0`，方向固定为 `KL(P_K0 || P_current)`。
- base forward 使用原始 `train()` / BN active 路径。
- anchor 的 current forward 临时使用 `eval()`，避免把 train-mode BN batch statistics 与 K0 eval policy 混为同一漂移；该额外 forward 不更新 BN running buffers。
- 冻结 K0 scorer 始终 `eval()`、float32、无 AMP、无 optimizer registration，训练前后参数与 buffers bit-exact。
- 每个原始 512-row batch 的全部行在 base objective 和 anchor 中各使用一次；没有 disagreement/margin 判定，没有行权重。

## 5. λ 只读定标（已完成并冻结）

λ 不是通过 T1 game evaluation 或 Variant 训练网格选取。定标对三个已冻结 R2 Control@70400 各读取第一个 matched M0 batch，只计算当前 base objective 与 anchor 在 Brain+DQN 参数上的 gradient norm：

```text
target anchor/base gradient ratio = 0.10
lambda_seed = 0.10 * ||g_base|| / ||g_anchor||
lambda       = median(lambda_seed_20260910/11/12)
```

| seed | base grad norm | anchor grad norm | seed λ |
| --- | ---: | ---: | ---: |
| 20260910 | 9.4284705853 | 1.1984822101 | 0.7867009210 |
| 20260911 | 7.2548951231 | 1.3629657449 | 0.5322874144 |
| 20260912 | 10.3041055221 | 1.1111745006 | 0.9273165931 |

冻结结果：

```text
selected lambda        0.7867009210376891
calibration hard gates 9/9 PASS
artifact               artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/calibration/t1_lambda_calibration.json
artifact SHA256        ab6b3f105fd69321265f181c32017918ff1790678d6114d552276f933e64ec27
optimizer steps        0
game outcomes read     0
```

## 6. 冻结训练合同

- K0 SHA256：`6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0`。
- M0 index SHA256：`755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2`。
- seeds：`20260910 / 20260911 / 20260912`。
- 每 seed：`70000 → 70400`，恰好 400 optimizer steps，batch 512，LR `1e-4`。
- AdamW 从 K0 恢复两组 `[165,245]`、共 410 个 moments；不添加参数组。
- Control 复用 R2 冻结 checkpoint，不重复训练。
- Variant 的 base-row SHA 必须逐 seed 精确匹配 R2 Control frozen row identity。
- 主 Q target 的所有值必须属于 `{+3,+1,-1,-3}`。
- 训练脚本必须逐 step 记录 base loss、anchor KL、weighted anchor loss 和 total loss，任何非有限值 fail closed。

## 7. 冻结机制审计

每个 seed 重建相同 M0 deterministic stream，跳过用于训练的前 400 batches，在 held-out batches `401..416` 上审计 8192 行。K0、R2 Control 和 T1 Variant 全部使用 eval-mode inference。

三个正式机制指标：

1. `mean KL(candidate || K0 policy definition)`：实际计算冻结方向 `KL(P_K0 || P_candidate)`；Variant 必须小于 Control。
2. `greedy disagreement rate to K0`：Variant 必须小于 Control。
3. `centered advantage RMSE to K0`：Variant 必须小于 Control。

三个指标必须在 3/3 seed 上全部严格改善，才有 `mechanism_pass=true`。`mean_abs_margin_delta_to_k0` 只作 descriptive cross-check，不进入机器投票。

机制审计失败不取消强度评测：只要训练 provenance/integrity gates 通过，仍完成全部 3000 局，避免以中间指标选择是否评测。

## 8. 冻结机器评测

每个训练 seed 建立一个 1000-hanchan CRN panel：

```text
[K0_70k, ext_mortal, R2_Control_seed_s, T1_AnchorVariant_seed_s]
```

- 共 3 panels × 1000 = 3000 局。
- game IDs：每个 panel 都严格覆盖 `2800000..2800999`。
- 每 panel 4 shards × 250。
- seed key：`8192`。
- `seat-mode=random`。
- 必须按 ReachAccepted `-1000` 语义重算终局分数与顺位。
- Primary/Absolute 使用同一 crossed-bootstrap resampling indices。
- crossed bootstrap：5000 reps，seed `20261003`，training-seed axis × shared game-ID axis。

正式对比：

```text
Primary  = T1 AnchorVariant - R2 Control
Absolute = T1 AnchorVariant - K0
```

每层 strength pass 均要求：

- 3/3 seed means `> 0`；
- crossed-bootstrap CI95 lower bound `> 0`。

## 9. 自动裁决

在所有 provenance/integrity gates 通过后，只允许：

- `anchor_promising`：mechanism pass，Primary pass，Absolute pass。只允许开启 fresh-seed T2 confirmation；本轮不直接产生 K1。
- `stability_only`：mechanism pass，Primary pass，Absolute fail。说明能修复 Control degradation，但未证明超过 K0；不晋级。
- `mechanism_not_supported`：机制三指标任一 seed/指标未全部改善；无论 strength 中心值如何均不以该机制继续。
- `not_supported`：mechanism pass，但 Primary 未通过。
- `no_verdict_gates_failed`：任一 hard provenance/integrity gate 失败。

所有 verdict 下：`recipe_promotion=false`、`checkpoint_promotion=false`、`K1=null`。T1 是 adaptive pilot，不得直接晋级。

## 10. 反自欺与停止规则

- 不做 λ、temperature、KL 方向、continuation 长度或评测局数 sweep。
- 不追加对局，不挑 seed，不以 canonical seed 单独晋级。
- 不根据 game result 修改机制 panel、metric 或 held-out batch 范围。
- 不重新打开 F1/F2/F3 disagreement weighting/filtering。
- 不同时修改 CQL、reward、score-to-go、optimizer、数据路线或模型结构。
- 不把低 KL、低 greedy flip、低 centered RMSE 单独解释为更强。
- 若不是 `anchor_promising`，T1 当轮关闭；任何新方向必须建立新 ID。

## 11. 执行顺序

1. 定标实现与 artifact 完整性验证。
2. preregistration/registry freeze。
3. 训练实现与聚焦测试通过。
4. Windows CUDA variant-only 3-seed training。
5. held-out policy-drift audit。
6. Windows CUDA 3×1000 fresh evaluation。
7. crossed bootstrap + machine adjudication。
8. 写入不可变最终报告并更新 registry；不自动 push。
