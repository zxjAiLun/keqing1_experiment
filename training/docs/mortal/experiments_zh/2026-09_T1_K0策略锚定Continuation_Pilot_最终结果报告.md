# T1 K0 合法动作策略锚定 Continuation Pilot：最终结果报告

> 实验身份：`T1_k0_policy_anchor_continuation_pilot_2026_09`  
> 状态：**CLOSED / not_supported / K1 = null**  
> 本报告记录冻结 preregistration、Windows CUDA 训练、3000 局机器评测与自动统计裁决的完整闭环；artifact 只读保存。

## 1. 结论先行

T1 的机制假设得到支持：在三个 training seed 上，冻结 K0 的 legal-policy KL anchor 都同时降低了 KL drift、greedy disagreement 和 centered-advantage RMSE。但这没有转化为对局强度收益：

- Primary（T1 AnchorVariant − R2 Control）三个 seed 均为负，grand mean **−6.855 pt**，crossed-bootstrap CI95 **[−13.860, +0.270]**；
- Absolute（T1 AnchorVariant − K0）三个 seed 均为负，grand mean **−3.090 pt**，CI95 **[−10.215, +4.261]**。

因此 Primary 与 Absolute 都未通过预注册 strength gate。机械裁决为 `not_supported`；`recipe_promotion=false`、`checkpoint_promotion=false`，K1 继续为 `null`。按停止规则关闭本 T1 身份，不追加对局、不挑 seed、不做 λ/temperature/长度 sweep。

## 2. 冻结契约与执行

T1 从 K0_70k 初始化，以冻结的 R2 Control checkpoint 作为对照；唯一干预是对合法动作 centered Q 的 `KL(P_K0 || P_current)`，temperature=1.0，selected `λ=0.7867009210376891`。训练继续使用 M0 mixed replay、`behavior_action_mc + final_rank_mc + CQL 5.0 + next-rank aux 0.2`、preserved K0 Adam、400 steps、batch 512、LR `1e-4`。每行只使用一次；不读取 behavior disagreement、reward 或评测结果。

λ 定标为只读过程，9/9 gates PASS，optimizer steps=0，game outcomes read=0。定标 artifact SHA256 为 `ab6b3f105fd69321265f181c32017918ff1790678d6114d552276f933e64ec27`。

执行限定为 Windows 原生 CUDA，使用 `E:\AUbuntuProject\project\keqing1\.venv-win\Scripts\python.exe`，设备为 RTX 4060 Laptop GPU；未使用 WSL，也未下载或创建环境。训练与评测实现提交包括 `673f9bc`、`1af260a`、`c62a569`、`b7f5795`、`04a2bb6`。

## 3. 训练与机制审计

三个 seed（`20260910/20260911/20260912`）均完成 `70000 → 70400` 的 400 optimizer steps。training manifest 的 16 个 hard gates 全部通过：父模型、M0、R2 checkpoint、定标、row identity、逐行使用、有限值、BN/scorer 模式、bit-exact scorer、Q target、preserved optimizer、步数和机制审计均已验证。

held-out panel 跳过训练使用的 400 batches，审计 batches 401–416，每 seed 8192 rows。下表为 `control → T1 variant`；三项正式指标在三个 seed 上均严格下降，故 `mechanism_pass=true`。

| seed | KL to K0 | greedy disagreement | centered-advantage RMSE | descriptive margin |
| --- | ---: | ---: | ---: | ---: |
| 20260910 | 0.167939 → **0.041498** | 0.080444 → **0.078735** | 0.912390 → **0.608032** | 1.288978 → **0.562717** |
| 20260911 | 0.178892 → **0.045466** | 0.093872 → **0.087891** | 0.922109 → **0.585898** | 1.354783 → **0.562401** |
| 20260912 | 0.162605 → **0.040407** | 0.083130 → **0.080933** | 0.934361 → **0.563218** | 1.282874 → **0.541766** |

training manifest SHA256：`35127a28a00a044cbf33330a4033ba2669f98803b1f815fd0f85e1adf39000f2`。三个 AnchorVariant checkpoint 与评测 manifest 做了三向 SHA 绑定：

| seed | checkpoint SHA256 |
| --- | --- |
| 20260910 | `de090dd404d1aa67a8c5c2606deaef3de991055c0c3d14df9ba0ce634905d283` |
| 20260911 | `b3446671c43e2c5f63c63bf874bf55e9f15dc5f971b6a6a208b3e53c61f6f5c5` |
| 20260912 | `b564130383ee829ca93fe0e26ce40fc0582690ec5e93aa2bec767a0406bd3c91` |

## 4. 机器评测与统计

评测使用三个 matched panels，每 panel 1000 局、4 shards × 250，lineup 为 `[K0_70k, ext_mortal, R2_Control_seed_s, T1_AnchorVariant_seed_s]`。game IDs 每 panel 严格覆盖 `2800000..2800999`，seed key `8192`，`seat-mode=random`，按 `reach_accepted` 的 −1000 语义重算终局分数。评测 7/7 hard gates 全部通过，完整验证 3000 个日志。

第一次评测进程在 1200 局处退出；恢复时逐文件验证已有日志、模型 lineup、start_game seed、end_game 与完整 50 局批次边界，再沿用 native `--resume` 完成剩余对局。既有 1200 局没有被重写。评测 manifest SHA256：`06a6487c0cbcdec1b3602af0333030a71cbb1c4c8c409c31577681508a42465b`。

### 4.1 Primary：T1 AnchorVariant − R2 Control（pt/半庄）

| seed | 均值 |
| --- | ---: |
| 20260910 | **−7.380** |
| 20260911 | **−5.670** |
| 20260912 | **−7.515** |
| **grand mean** | **−6.855** |

Crossed-bootstrap（5000 reps，seed `20261003`）CI95：**[−13.860, +0.270]**。三个 seed 均值均非正，CI 下界不为正。

### 4.2 Absolute：T1 AnchorVariant − K0_70k（pt/半庄）

| seed | 均值 |
| --- | ---: |
| 20260910 | **−1.440** |
| 20260911 | **−4.815** |
| 20260912 | **−3.015** |
| **grand mean** | **−3.090** |

共享抽样索引的 crossed-bootstrap CI95：**[−10.215, +4.261]**。三个 seed 均值均非正，CI 下界不为正。

summary 的 8 个 hard gates（training/eval manifest、3000 日志、paired metrics、crossed bootstrap、Primary、Absolute、mechanism）全部通过。summary SHA256：`2686035e149292393ed49d82c3ed172354e5aa8ddba60953d4c1b1fd33c09edc`。

## 5. 证据边界与停止事项

本实验支持“该强度与训练长度下的 legal-policy anchor 能抑制 K0 策略漂移”，不支持“抑制漂移会提升对局强度”。由于 Primary 未通过，不能把它命名为 `stability_only`；该状态只适用于 Primary 通过而 Absolute 未通过的情况。当前结果也不能外推到其他 λ、temperature、anchor 方向、训练长度、数据路线或更广泛的 policy regularization。

停止事项：不追加对局、不挑选 seed、不重跑 T1、不做参数或长度 sweep、不晋级任何 recipe/checkpoint，K1 保持 `null`。后续新方向必须建立新的实验 ID；当前 registry 的 `next_experiment` 回到 `null / not_selected`。

## 6. Artifact 索引

| Artifact | 路径 | SHA256 |
| --- | --- | --- |
| λ calibration | `artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/calibration/t1_lambda_calibration.json` | `ab6b3f105fd69321265f181c32017918ff1790678d6114d552276f933e64ec27` |
| training manifest | `artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/training/t1_training_manifest.json` | `35127a28a00a044cbf33330a4033ba2669f98803b1f815fd0f85e1adf39000f2` |
| evaluation manifest | `artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/evaluation/t1_eval_manifest.json` | `06a6487c0cbcdec1b3602af0333030a71cbb1c4c8c409c31577681508a42465b` |
| summary | `artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/summary/t1_summary.json` | `2686035e149292393ed49d82c3ed172354e5aa8ddba60953d4c1b1fd33c09edc` |
| raw logs | `artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09/evaluation/panel_seed_*/shard_*/logs/` | 3000 verified |

机器可读裁决以 `t1_summary.json` 为准；本报告不修改任何训练 checkpoint 或原始日志。
