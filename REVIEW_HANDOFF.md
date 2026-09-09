# REVIEW_HANDOFF — P4-M1 现成语料 external 重标注 + Mortal-native 学生（第一阶段）

日期：2026-09-09。交付人：本地 agent。状态：**第一阶段实现/测试/正式标签缓存全部完成，正式学生长训练未启动，等待 review。**

路线依据：[`training/docs/mortal/2026-09-07_现成语料与早期路线复盘_主线改向.md`](training/docs/mortal/2026-09-07_现成语料与早期路线复盘_主线改向.md) 第 5 节：现成状态 → 本地 external 重标注 → 随机初始化 Mortal-native 学生 → 纯合法动作策略训练。未加入 MC/GRP/辅助头/K0 锚定；未重建审计平台；未动原始语料（三池 18,000 场全部只读）。

## Commits

| 角色 | commit |
| --- | --- |
| base（交接起点） | `d4d27aa` |
| relabel 管线 + 学生训练器 + 13 测试 | `a95884a` |
| 分相计时 + manifest 行数 + trainer 不解压数行 | `c0c8b23` |
| zlib-1 分片压缩（资源 smoke 发现） | `a47f6ab` |
| launcher + 精确 pool 行数 | `5438367`（head） |

## 关键文件

| 文件 | 职责 |
| --- | --- |
| `training/mortal/relabel_ext_teacher.py` | 全视角重标注：`GameplayLoader(oracle=False, player_names=None)` 取四家视角（信息边界=仅当前玩家可见 obs (1012,34)），冻结 external checkpoint 纯贪心重答，46 维 hard label + 诊断 Q。npz 分片 + manifest（SHA 绑定源文件与教师 checkpoint；`shard_rows` 行数；分相 timing；快照 resume 含 stale-tail 清理） |
| `training/mortal/train_student_policy.py` | 学生训练器：随机初始化 Brain(192×40)+DQN v4，纯 masked-Q CE（教师贪心 hard label），AdamW + 线性 warmup；行数从 manifest 读（fail-closed）；shard 流 cursor+RNG+optimizer/scheduler 完整恢复；内置 holdout CE/agreement |
| `scripts/mortal/run_full_relabel.ps1` | 正式 relabel launcher：显式 2048/512/100 冻结参数、自动展开 D3 24 shard、启动前三检查（目录/18000 计数/GPU 独占）、`-Resume` |
| `tests/test_student_policy_pipeline.py` | 13 个针对性测试（见下） |
| `artifacts/experiments/student_policy_v1/`（gitignored） | 正式标签缓存 + 运行产物 |

## 正式标签缓存（一次性，已生成并验证）

`artifacts/experiments/student_policy_v1/labels_full_18000h`：

| 项 | 值 |
| --- | --- |
| 半庄 | 18,000（S0 6000 + D3 6000 + V2 6000，按比赛身份，原始日志只读） |
| 决策行 | **11,487,044**（train 10,340,197 / holdout 1,146,847；holdout 10.03%，按 hanchan SHA-256 划分） |
| teacher-behavior agreement | **94.72%**（train/holdout 间一致：94.72/94.68） |
| 分片 | 5,796 npz（zlib-1，23GB） |
| 完整性验证 | `incomplete=None`；抽样 12 分片 23,379 行教师+行为标签 100% 在 legal mask 内；`shard_rows` 与分片逐个精确一致；总和 == totals；源文件 SHA 全记录 |
| 生成耗时 | 两段共约 8h（首轮 5.6h 到 68% 中断 + 续跑 2.8h），中间经历一次手动停止→manifest 快照续跑 |

三池行数：S0 3,856,181 / D3 3,872,813 / V2 3,758,214。

**标签语义（重点 review 项）**：教师标签 = external 在该状态（当前玩家可见信息）上的**纯贪心**动作。与日志行为动作的差异有两类：(a) S0 中 ~0.04% 是生成期 arena `rule_based_agari_guard` 见逃（日志记录守卫后动作，标签是纯贪心 hora）——测试 `test_s0_relabel_agreement` 强制 S0 mismatch 只能是 hora 翻转签名；(b) D3/V2 混战桌非 ext 座 ~13% 不一致——这正是重标注的价值（教师对弱模型状态重新作答）。此语义已按改向文档第 4 节"external 在任意已有状态上重新回答"实现。

## 实际配置（冻结）

relabel：`--rows-per-shard 2048 --inference-batch 512 --manifest-snapshot-every 100`，fp32 无 AMP，GPU 独占。

- `inference_batch=1024` 被实测否决：8GB WDDM 卡上有确定性显存悬崖（同数据 inference 36.2s vs 512 的 3.6s，两次完全复现）。
- zlib-6→zlib-1：每 2048 行分片写 2.38s→0.85s，文件 2.7→4.2MiB（13→23GB/全量，E 盘 616GiB 空闲）。
- 资源 smoke（60 场，正式参数）：peak RSS 2.49/15.6 GiB、peak VRAM 5779/8188 MiB（含桌面共享）、463 rows/s。
- 分相占比（smoke）：parse 43% / write 29% / inference 21%——CPU 瓶颈，GPU 低占用。

student：`192×40 v4、batch 512、lr 1e-4、AdamW wd 0.1、warmup 200、fp32`。

## 测试结果

`tests/test_student_policy_pipeline.py` **13 passed**：
- S0 relabel agreement ≥0.99 且 mismatch 必须是 agari-guard hora 翻转（真实语料+真实教师）
- 流 resume 精确性（cursor 重放逐位一致）、行越界 fail
- masked CE 参考实现对齐、非法动作 inf 损失
- LR warmup 回归（首版曾 warmup 结束突跳 lr=1.0 发散）
- manifest 行数读取（shard_rows/flushed fallback/缺失拒绝）
- 信息边界（oracle=False obs shape (1012,34) 与 oracle shape 不同）

## 短跑指标

**pilot 300 场（88,418 行）**：2000 步 train CE 1.75→0.09、agreement 0.38→0.97；holdout step 500 触底 0.708 后过拟合（预期，小包容量耗尽）。恢复验证：2100 步 cursor 与逐 shard 模拟逐位一致。

**正式缓存 sanity（11.49M 行）**：1200 步（61 万行暴露）：

| steps | holdout CE | holdout agreement |
|---|---|---|
| 400 | 0.787 | 0.738 |
| 800 | 0.608 | 0.776 |
| 1200 | 0.584 | 0.788 |

**无过拟合迹象，持续收敛**。train CE 0.563 / agreement 0.794 @1200。吞吐 ~1,100 rows/s（0.47s/step 含 holdout 评估）。

## 正式训练预算（提案，待 review 决策）

按实测 1,100 rows/s（0.47s/step）：

| 方案 | 步数 | 行暴露 | 耗时 | 说明 |
|---|---|---|---|---|
| 最小正式 | 25,000 | 12.8M（≈1.24 epoch） | ~3.3h | 略超一个 epoch 的暴露量 |
| 建议 | 50,000 | 25.6M（≈2.5 epoch） | ~6.5h | pilot 过拟合点在大包下显著推迟 |
| 上限 | 100,000 | 51.2M（≈5 epoch） | ~13h | 需看 holdout 曲线决定是否值得 |

建议以 holdout agreement 为读点指标（每 2000 步评估一次，checkpoint 每 2000 步），先跑 50,000 步（~6.5h，过夜），按曲线再议是否延长。磁盘：checkpoint ~130MB/个 × 保留读点。

## 已知问题与限制

1. **CUDA conv backward 非确定性**：断点恢复后权重轨迹与不间断运行相差 ~1e-3 量级（两次 fresh run 也不同）。流 cursor/RNG/optimizer/scheduler/scaler 状态恢复是精确的。与 Mortal 主线 runner 同等保证水平。
2. holdout 划分按 hanchan SHA-256，**未按池分层**（实测 train/holdout agreement 一致，但 S0/D3/V2 在 holdout 中的比例未显式控制）。
3. 杠选择行（`always_include_kan_select=True`）与行为动作同语义重标注（教师贪心含杠后选牌）。
4. V2 的 `platform_accounts` 是同 6,000 场重命名副本，未计入（勘察确认）；三池合计 18,000 场无重复 canonical game。
5. relabel 的 `pool_stats.rows` 首轮（12,300 场）用的是近似计数（最后一场×4），续跑段为精确计数；manifest `files`/`shard_rows`/totals 均精确，`pool_stats.rows` 仅 S0/D3 段有 ~1% 近似（已在 head commit 修复后续运行）。

## 需要 review 决策的事项

1. **训练预算**：25k / 50k / 100k 步（或按 epoch 定义），读点间隔。
2. **holdout 划分**是否需要改为按池分层（当前 SHA 哈希，实测无偏）。
3. **标签语义确认**：纯贪心（无 agari guard）是否符合"教师重新作答"的预期——S0 内 ~0.04% 守卫翻转行,学生会被教成"all-last 末位不放弃和牌"。这是文档语义的自然结果，但值得显式确认。
4. **学生容量**：192×40（K0 家族）vs 教师同容量 256×54。当前按改向文档"网络容量按现有 GPU 吞吐定一个配置"选了 192×40（也是 K0 容量，便于历史对照）。
5. 后续阶段（学生自身状态采集）不在本 handoff 范围。
