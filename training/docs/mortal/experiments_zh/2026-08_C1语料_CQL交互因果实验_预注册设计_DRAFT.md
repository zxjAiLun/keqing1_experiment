# C1 语料 × CQL 交互因果实验：预注册设计 DRAFT

> **状态：`preregistration_draft`；训练未授权。**
>
> 本文件只记录 feasibility audit 结论和待 review 的 preregistration draft。
> 它不是正式冻结的 preregistration，不是 training recipe，不创建 registry entry，
> 也不授权训练、生成或 checkpoint 产出。

## 1. 治理状态与边界

### 1.1 候选身份

| 字段 | draft 值 |
| --- | --- |
| candidate experiment ID | `C1_corpus_cql_interaction_2026_08` |
| category | causal mechanism experiment |
| parent | `K0_70k` at step `70000` |
| current status | `preregistration_draft` |
| training status | `training_not_authorized` |
| formal registry status | 未注册；`research_registry.json.current_state.next_experiment` 仍为 `null` |
| model promotion status | 不适用；不产生 K1 |

在当前仓库、`research_registry.json`、已有 Mortal docs 和已检查的 artifact namespace 中，
没有发现 `C1_corpus_cql_interaction_2026_08` 冲突。本结论只允许使用该候选名称，
不把它改名为 `C2`、`C3`，也不把它加入 registry。

### 1.2 本轮禁止事项

- 不启动训练、replay generation 或 evaluator run。
- 不创建 C1 artifact root、checkpoint、config 或 experiment ID 的正式实例。
- 不修改 D1、D2、D3 artifact、冻结的 K0 decision-signal audit 或既有 prereg。
- 不扫描 CQL weight，不做 reward、optimizer、LR、architecture 或 view-ratio sweep。
- 不因为 feasibility 数值改变唯一 intervention。
- `FORMAL_RUN_AUTHORIZED` 继续保持 `False`。

本 draft 的来源仓库基线是 `2ef09390e4a96ce5d66a24b3346144f4b2135c43`。
基线当时 `origin/main` 一致；工作树另有一个未跟踪的 `1.md`，不属于本实验 draft。

## 2. 背景与因果问题

上一轮 `K0_DECISION_SIGNAL_AUDIT_2026_08` 已正式 closure，权威结果为：

```text
support_signal                  false
objective_gradient_shift_signal true
machine_readout                 objective_gradient_shift_signal
authoritative                   true
```

正式阳性集中在 value/CQL gradient relationship 和 centered CQL preference pressure；
G1 total-gradient direction 未通过。该结果不证明 CQL 普遍有害，也不证明
`cql_weight=0` 会更强。

本实验的 primary causal question 是：

> 对固定的 K0@70k legacy continuation contract，D1 相对 M0 的 continuation penalty，
> 是否依赖当前 CQL preference contribution？

形式化为 corpus type 与 objective intervention 的 interaction，而不是总体比较
“CQL=0 是否更强”。唯一候选 intervention 是 continuation 阶段把 CQL contribution
weight 设为零。

## 3. 为什么选择 M0 与 D1

primary corpus 只包含：

```text
M0_operational_control
D1_project_owned_k0_view
```

D1 是第一轮最小 causal factorial 中更干净的 stress case：

- D1 的 parent-greedy agreement 约为 `99.9894%`，接近 K0-like/on-policy continuation。
- M0 的对应 agreement 约为 `89%`。
- corrected D1-M0 三个 training-seed means 为 `[-7.425, -10.980, -7.155]` Pt/半庄。
- corrected pooled mean 为 `-8.520` Pt/半庄。
- corrected equal-seed hierarchical CI95 为 `[-14.025, -3.225]`，三个 seed 均为负。

D2 和 D3 不进入本轮 training factorial，也不进入 primary machine vote：

- D2 在同一 D1 hanchans 上改变 trainable perspective，是 descendant-policy view 变量，
  不是新的最小 corpus 对照。
- D3 改变 generation behavior policy，额外混入 top-2 exploration behavior variable。

D2/D3 只能作为历史或 descriptive context。C1 不能把 D2/D3 的结果并入 primary
interaction，也不能把 C1 改写为 D3 continuation 或 model promotion experiment。

Factor A 的科学含义是 **D1 full replay lineage vs M0 full replay lineage**，不是单独的
K0-greedy behavior factor。D1 相对 M0 同时改变完整 replay lineage、state distribution、
opponent ecology、behavior distribution 和 final-rank target distribution。即使 C1
通过 interaction gate，也只能说明 CQL continuation contribution 与这一整套
corpus-lineage contrast 发生 interaction，不能把结果偷换成“K0-greedy behavior 本身
与 CQL 发生了因果 interaction”。

## 4. 候选 2×2 设计

### 4.1 Factor A：corpus

| level | 定义 |
| --- | --- |
| `M0` | `M0_operational_control`，既有 mixed replay operational corpus |
| `D1` | `D1_project_owned_k0_view`，6000 个 project-owned K0-view hanchans |

### 4.2 Factor B：continuation objective

| level | 定义 |
| --- | --- |
| `CURRENT` | `behavior_action_mc` value target + 当前 CQL implementation，`min_q_weight=5.0` |
| `CQL_OFF` | 保持所有其它项不变，仅将 CQL contribution weight 设为 `0.0` |

候选 cell 为：

| corpus | `CURRENT` | `CQL_OFF` |
| --- | --- | --- |
| M0 | historical current control cell | 新训练 `M0 × CQL_OFF` |
| D1 | historical current cell | 新训练 `D1 × CQL_OFF` |

每个新 cell 使用 training seeds `20260806`、`20260807`、`20260808`。最终是否可以
合法复用两个 historical CURRENT cells，见第 7 节；在 compatibility gate 未通过前，
不能把上述表格当作已授权的六次训练指令。

## 5. CURRENT objective 的实际 contract

### 5.1 已核对的 production/current 值

已核对 D1/M0 frozen training manifest、六个 checkpoint 内嵌 config、data exposure
和当前 `training/mortal/objective.py`：

```text
objective.mode              = behavior_action_mc
reward.mode                 = final_rank_mc
cql.min_q_weight            = 5.0
aux.next_rank_weight        = 0.2
env.gamma                   = 1.0
control.batch_size          = 512
control.enable_amp          = false
resnet.conv_channels        = 192
resnet.num_blocks           = 40
optim.betas                 = [0.9, 0.999]
optim.eps                   = 1e-8
optim.weight_decay          = 0.1
optim.max_grad_norm         = 0.0
optim.scheduler             = peak=1e-4, final=1e-4, warm_up_steps=0, max_steps=0
```

训练数据的 reward rank points 是 `[6, 4, 2, 0]`，不能与 evaluation 的 rank points
`[90, 45, 0, -135]` 混写；两者在各自 contract 中分别固定。

### 5.2 value、CQL、aux 的数学定义

在 `training/mortal/objective.py` 中，`behavior_action_mc` 使用：

```text
value_prediction = Q(s, a_behavior)
value_loss       = 0.5 * MSE(value_prediction, q_target_mc)
cql_loss         = mean(logsumexp(Q(s, legal_actions)))
                   - mean(Q(s, a_behavior))
aux_loss         = cross_entropy(next_rank_logits, player_ranks)
total_loss       = value_loss + cql_weight * cql_loss + aux_weight * aux_loss
```

DQN 输出已将非法动作设为 `-inf`，因此 CQL 的 `logsumexp` 只看 legal actions。
`q_target_mc` 仍是 `final_rank_mc` 的 behavior-action MC target；CQL 不改变 reward、
target 或 action label。

### 5.3 CQL_OFF 是否只有一个变量

候选 `CQL_OFF` 只改变：

```text
config["cql"]["min_q_weight"]: 5.0 -> 0.0
```

其语义是：

```text
total_loss_current = value_loss + 5.0 * cql_loss + 0.2 * aux_loss
total_loss_cql_off = value_loss + 0.0 * cql_loss + 0.2 * aux_loss
```

当前 objective implementation 不会因为 weight 为零而进入另一个 objective mode；
它仍计算 raw `cql_loss`，仍执行 finite diagnostics，且 raw CQL telemetry 可以保留。
`cql_loss` 不得通过 detached/替代 loss 或其它路径进入 optimizer gradient。

这意味着 CQL_OFF 是单一、概念明确的 CQL ablation，而不是 CQL grid。正式实施前
必须由新 preflight 检查：

- config 中唯一 scientific diff 是 `min_q_weight=0.0`；
- objective mode、reward、aux、loader、optimizer、scheduler、scaler 和 data stream 不变；
- raw CQL loss 仍 finite 且只作为 descriptive telemetry；
- 不发生额外 conditional code path 或 optimizer step 差异。

本轮不修改 objective 或 runner，不添加 CQL_OFF config，也不执行 zero-step run。

## 6. 固定变量 contract

除 corpus 和 CQL intervention 外，候选 contract 固定如下：

| 组件 | 固定值 |
| --- | --- |
| parent | `K0_70k` checkpoint，step `70000` |
| continuation | `70000 -> 72000`，共 2000 optimizer steps |
| architecture | Brain + DQN + Aux，192 channels / 40 blocks，动作契约不变 |
| value objective | `behavior_action_mc` |
| reward | `final_rank_mc` |
| value target | behavior-action MC target |
| aux | next-rank head 与 `aux_weight=0.2` 保持不变 |
| optimizer initialization | 从同一 K0 parent 恢复 preserved Adam moments |
| scheduler | fresh，保持 operational continuation recipe |
| scaler | fresh |
| data stream | fresh；`data_seed=training_seed`；不恢复历史 data cursor |
| batch | 512；`num_workers=0` |
| AMP | 与 historical M0/D1 一致，`false` |
| device | formal training 需要 `cuda:0`，不允许 CPU fallback |
| RNG | 每个 cell 使用对应固定 training seed，不按中间结果挑选 |
| archive | 候选仍固定 `70001, 70010, 70100, 70500, 71000, 72000` |

preserved Adam 是本实验的固定 continuation state，不是本实验要消融的变量。CQL_OFF
只移除 step `70001..72000` 期间产生的新的 CQL gradient contribution；它不移除已经
存在于 `K0@70k` weights 或 `K0@70k` Adam `m/v` moments 中的 0→70k historical CQL
influence。因此 causal claim 必须限定为：

> under the frozen K0@70k legacy state, turning off new CQL gradient contribution during
> the 70k→72k continuation changes the D1-vs-M0 penalty.

任何 implementation 发现额外 side effect 时，本轮设计失效并回到 review；不得现场修复
后继续使用同一 C1 draft。

## 7. Historical CURRENT cells 复用 feasibility

### 7.1 已核对的 parent、manifest 与 checkpoint

本地实际 artifact root 为：

```text
/media/bailan/DISK/AUbuntuProject/project/keqing1/
  artifacts/experiments/model_pool_2026_07/
  D1_project_owned_population_2026_07/
  training_prep_2026_07/
```

共享 authoritative K0 parent 为：

```text
/media/bailan/DISK/AUbuntuProject/keqing-data/mortal/authoritative/
  D3_top2_discard_v1_2026_08/models/K0_70k/
  mortal_default_70k_promoted_candidate.pth
```

| artifact | SHA256 |
| --- | --- |
| K0 parent checkpoint，step 70000 | `6c0e70058644e02671440ddf7dd2b41c637ae7c2132c9154595593ab690d49e0` |
| D1 training manifest | `9b91897084fc93c5658283239b9136a6a1644b060dbb14a2a159a1c8529ce126` |
| historical B250 protocol | `8ef6e7bb512e2a6b6d0dbaa1749c51fdc74ab1a66dc4dfbfe920367e8be29988` |
| corrected D1 summary v2 | `a20d9435fe757593b520b78c466e8a3ec482de2defa4352a72ec7daf92706524` |
| D1 reconstruction equivalence | `da10477c82fa612a6d4189e195a2c4e3afa2c1dfd8f00349a6e71b49978e0895` |
| corrected D1 paired rows CSV | `896817ba6883f9ece24cf713c56dfb772a6c8572d1845529912249fdac1739dd` |
| M0 file index | `755b1d5976e3837402eec708d160ede081605e2fcda37d9acdb1436d8a72fce2` |
| D1 file index | `e357bdb00d5bf3cd7e0afa6960ee43af656421cfed381a3320f6b83ac56087f0` |
| M0 content manifest | `5d842d356364d5b06c537c6f4448f4d8a03e99f8009c98521b8a1b7acd4588b5` |
| D1 content manifest | `d4b4f5df75a89dcc07ed4bd3dc6ec06d92b051968364cc9d28c9d5657188f446` |
| D1 dataset audit | `c41cce7b4354cc91e1f982d739855736e5e28399a9774e97da786037514cb69d` |
| D1 generation manifest | `efe9b4cbb2501c3ec8ac7fad0422c918903100cd4dbc8585f9be43b0709a91bc` |

历史 manifest 记录 `git_dirty=false`、training commit
`90d148aedbcb905aa36615775462f8e2eece080b`、parent step 70000、target step 72000、
preserved optimizer、fresh scheduler/scaler/data stream，且六个 checkpoint 内嵌
metadata 都报告相同 contract。每个 seed 均消费 2000 batches / 1,024,000 samples，
数据 file count 为 6000。

### 7.2 六个 historical CURRENT cell 的精确冻结信息

| cell | config SHA256 | `mortal_72000.pth` SHA256 |
| --- | --- | --- |
| M0 / 20260806 | `89d8a9947402d12aaa879c561f1729f2c671358ac8639e723209308834f05d93` | `4a6a5dd1eb55d8d207d7689b02c4682146c2a0cc70eaef554e6cfa869804dbdd` |
| M0 / 20260807 | `e97b4b5dc09d32cffe8068007e0cdbe012cdb3b68cf9955bafe377f76febfe55` | `de7f6da7c0c07b89d658554050f2112f09fd9c021247104d5db44228db04823d` |
| M0 / 20260808 | `0302c11ad71f19fad38e0e6e7db696898d338cfc66475e9e3a9e58a11cd4b694` | `d2d0b0b6cdc86423ecbef852d34edc785e6efdcaaaf425e05988d7ff472d46c4` |
| D1 / 20260806 | `08d35c0aaf2a2db4a5039cb5242cf02a6d53a1bd5c6a8c042d2f29eda8f0a0dd` | `9425109b2562eb48a86ca7b3a250738b5691503f9156f29bc50a2b20e7a922aa` |
| D1 / 20260807 | `655339eebbacf8e9fd85de820c612515c9840025b4dbc1c7dfa4bb3ec03841b3` | `e2718ee8d572071b8d46d04beaf5f2aa6d90ad847762254f80648de9639a0b3d` |
| D1 / 20260808 | `4baf422cad7711f85f85b456c4c4329c0fd4d874df09b51506b0e03821b12d2e` | `985a3e532ef13cd7fab945c92839a941390fd9f7cc5dc0e177d4d4182a116f41` |

六个 config 中实际核对到的 `min_q_weight` 都是 `5.0`，objective 都是
`behavior_action_mc`，reward 都是 `final_rank_mc`。六个 checkpoint 都存在且
`steps=72000`；每个 checkpoint 的 initialization 都是同一个 parent SHA、
`optimizer=preserved`、`scheduler/scaler/data_stream=fresh`。

### 7.3 完整 loader compatibility gate（2026-08-18）

已完成 M0/D1 × 三个 training seed 的完整 2000-batch loader equivalence audit。每个
stream 均比较 `1,024,000` samples、canonical dtype、shape、contiguous CPU bytes 和
sample order，并记录 source/config/file-index/labels provenance、ordered SHA256、
first mismatch batch/tensor。D1 / 20260808 的 historical 与 current stream 分别在
独立 child process 中完成，之后由只读 finalizer 合并；没有把两个大 stream 留在同一
Python allocator 中。

最终 feasibility report 为：

```text
path:
/media/bailan/DISK/AUbuntuProject/project/keqing1_experiment/
  artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/
  loader_compatibility.json

sha256:
80d78702e36c7770e6d781ccb651ad088c5cbe84b42d21e1a6c19a7609903df7
```

六组结果如下；每行的 historical/current ordered SHA 完全相同，`exact_match=true`，
`first_mismatch_batch=null`，`first_mismatch_tensor=null`：

| route / seed | historical/current ordered SHA256 | batches | samples |
| --- | --- | ---: | ---: |
| M0 / 20260806 | `c111c3b1fe223bfc42a52507226963b093c17be792e9197ef0d3686f5b794b3f` | 2000 | 1,024,000 |
| M0 / 20260807 | `6d418d89a23509293d69cf359de91e99f23b210dcfc6570ce2b4ac8d95ffb2a0` | 2000 | 1,024,000 |
| M0 / 20260808 | `6f35793b3d18f9bb5325ce57232b0ad02ee23e3b3cb68927c3aaef6737d604ed` | 2000 | 1,024,000 |
| D1 / 20260806 | `f7e8c46436b069206583b0c5151c3a4be7c6019ade054a100d0950990dea823f` | 2000 | 1,024,000 |
| D1 / 20260807 | `1b68076ec2683d60af28a1aa9b8724d049f568e0c97738ae3c47f1cfca475d35` | 2000 | 1,024,000 |
| D1 / 20260808 | `74bb248b0bf17d192b1ebad986e7ff7f56e387ddf3e605bd0db765cc552b05ce` | 2000 | 1,024,000 |

**结论：`historical CURRENT training reuse = APPROVED`。**

因此 C1 若通过后续 preregistration freeze、registry、implementation/preflight 和
authorization-only ordering，所需的新 continuation trainings 固定为：

```text
M0 × CQL_OFF：3 seeds
D1 × CQL_OFF：3 seeds
new trainings = 6
```

这不改变当前 governance 状态：本轮仍未 freeze、未 registration、未 authorization，
也没有启动任何训练。

## 8. Evaluation feasibility 与 matched DID

### 8.1 历史 evaluation contract

历史 D1/M0 B250 protocol 的固定值为：

```text
evaluator commit       = feb2ad675a7576b437c674b4e65e9735df16a83e
native batch           = 250
games per training seed= 1000
shards per seed        = 4
seat mode              = random
seed key               = 8192
seed starts            = 1700000, 1710000, 1720000
device                 = cuda
AMP                    = false
rank points            = [90, 45, 0, -135]
```

历史 raw evaluation 每个 seed 有 1000 个完整 hanchan，文件前缀分别覆盖
`1700000..1700999`、`1710000..1710999`、`1720000..1720999`，且每个 raw log 同时包含
`70k`、`ext_mortal`、对应的 `M0_seed` 和 `D1_seed` 四个玩家。这样历史
`D1_CURRENT - M0_CURRENT` 是同一 hanchan 内的 paired difference。

ReachAccepted Repair 1 后没有重跑或修改 raw evaluation。corrected D1 summary v2
完成了：

- 3000 个 raw logs、12000 个 player-rank 的 native `Stat` 等价核对；
- reconstructed ranks 与 `detailed_stats`、`metrics` 的 rank counts 全部一致；
- `mean(delta) == mean(pt_a) - mean(pt_b)` 的代数不变量通过；
- v1 summary 因遗漏 `reach_accepted -1000` 已 invalidated。

因此历史 raw logs、current checkpoint 和 corrected summary 可以作为 immutable
historical reference，不能被覆盖或重算为旧口径。

### 8.2 C1 的统一 fresh evaluation

历史 D1/M0 evaluation logs 只作为 historical provenance/descriptive reference，
不进入 C1 primary interaction statistic。C1 evaluation 全部在一个新冻结的 Linux/CUDA
runtime 中 fresh 运行，不复用历史 current evaluation logs。

每个 training seed 运行两个独立的四人 lineup。CURRENT lineup 为：

```text
70k                = K0 parent，SHA 6c0e7005...
ext_mortal         = authoritative external model，SHA 0a88ddad...
M0_CURRENT_seed    = historical current M0 checkpoint，在新 runtime fresh evaluation
D1_CURRENT_seed    = historical current D1 checkpoint，在新 runtime fresh evaluation
```

CURRENT model specification 顺序固定为
`[70k, ext_mortal, M0_CURRENT_seed, D1_CURRENT_seed]`。CQL_OFF lineup 为：

```text
70k                = 同一 K0 parent
ext_mortal         = 同一 authoritative external model
M0_CQL_OFF_seed    = 新的 M0 CQL_OFF checkpoint
D1_CQL_OFF_seed    = 新的 D1 CQL_OFF checkpoint
```

CQL_OFF model specification 顺序固定为
`[70k, ext_mortal, M0_CQL_OFF_seed, D1_CQL_OFF_seed]`。不能仅保持标签集合而改变
engine 参数顺序，因为 random-seat 的 common-random block 需要同时固定 lineup order。

每个 objective condition、每个 training seed 都评估 1000 个完整 hanchans；总量为：

```text
3 seeds × 2 objective conditions × 1000 hanchans = 6000 evaluation hanchans
```

使用与历史完全相同的 seed starts、seed key、random-seat、B250 shard、rank points、
native evaluator source 和 opponent population。CURRENT/OFF 使用完全相同的 evaluation
hanchan identities、seed block、seed key、lineup order 和 random-seat contract；每个
condition 的 raw log 必须在同一 hanchan 内同时包含对应的 M0 与 D1，形成 within-cell
paired gap。

fresh current gap 与 fresh off gap 通过相同的 hanchan seed block、相同的 lineup ecology、
相同的 seat/random contract 对齐，但不应声称是同一局轨迹：两个 model pair 改变后，
牌局 action trajectory 可能不同。若 runtime、seed block 或 opponent ecology 不同，
则连 common-random block 的解释都不能成立。

### 8.3 runtime mismatch blocker

当前本地可用的 authoritative asset bundle `manifest.json` 记录了一个 Windows
runtime reference：

```text
available runtime reference = Windows authoritative runtime
native file                  = riichi.pyd
native SHA256                = 19bb181eaa70d0ae90417a3bd22433f6ca08d7654602f865ff3bdb102b7d9914
bundle Mortal revision       = 813859fc8110ea178f56f009994bc4f1b9fee645
```

但是 D1/M0 historical B250 `protocol.json` 只记录
`mortal_revision=0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`，没有记录 native extension
SHA；因此上面的 bundle runtime 不能被错误地当作 D1 evaluation 的 exact native
provenance。当前 formal Linux/CUDA environment 的冻结要求是 `riichi.so` SHA
`da687ececbae8c803c99fe58fb8f66d0e4b9e762eb2bb7257a2115c57e5dd82b`。这两个 native
binary 不是 byte-identical，而且 historical D1 native SHA 本身未被 protocol 冻结；
当前 artifact 没有证明跨平台 native arena 在固定 seed 下的行为和随机流完全等价。

因此 C1 freeze 时固定采用统一 fresh Linux evaluation；不再保留“若 Windows runtime
可用则复用 historical evaluation”的 future branch。historical current checkpoint
是否进入新 CURRENT lineup，只取决于第 7 节 training reuse gate；historical current
raw logs 永不进入 C1 primary statistic。

## 9. Primary estimand

对每个 matched training seed `s` 和 evaluation hanchan identity `h`，先在同一
fresh-runtime evaluation block 内形成 row-of-pairs：

```text
d_current[s,h]
  = Pt(D1_CURRENT, s, h) - Pt(M0_CURRENT, s, h)

d_off[s,h]
  = Pt(D1_CQL_OFF, s, h) - Pt(M0_CQL_OFF, s, h)

interaction_row[s,h]
  = d_off[s,h] - d_current[s,h]
```

`CURRENT` 与 `CQL_OFF` 的 row-of-pairs 必须共享：

```text
training seed
evaluation hanchan identity
seed_key
lineup order
random-seat contract
native runtime
```

每个 training seed 的 primary point estimate 是：

```text
interaction_seed[s]
  = mean_h interaction_row[s,h]
```

primary estimand 是三个 `interaction_seed[s]` 的 equal-seed mean：

```text
I = mean_s interaction_seed[s]
```

符号解释固定为：

```text
interaction > 0
  => 移除 CQL 后，D1 相对 M0 的 continuation penalty 变小
```

primary 不使用四个 cell 的绝对 Pt，也不使用单独的 `CQL_OFF - CURRENT` 总体提升。
以下只允许作为 secondary/descriptive，不能使用历史 evaluation log 替代 fresh
`d_current[s,h]`：

```text
M0_CQL_OFF - K0
D1_CQL_OFF - K0
M0_CQL_OFF - M0_CURRENT
D1_CQL_OFF - D1_CURRENT
```

绝对 result 不能覆盖 primary interaction verdict。

## 10. Candidate machine adjudication

本节是 candidate rule，尚未 freeze。正式 freeze 时必须同时冻结 artifact gate、
runtime gate 和失败标签；bootstrap 的 `B` 与 seed 在本 draft 中已经预先指定。

候选 primary rule：

```text
interaction_supported iff

  all three matched training-seed interaction point estimates > 0

  AND

  equal-seed hierarchical bootstrap 95% CI lower bound > 0

  AND

  training provenance / evaluation pairing / runtime gates all PASS
```

否则：

```text
interaction_not_confirmed
```

若 training、provenance、loader compatibility 或 evaluation runtime gate 失败：

```text
no_verdict_gates_failed
```

不得用总体中心均值为正替代三 seed direction gate。bootstrap 先对已经形成的
`interaction_row[s,h]` 做 resampling，不得分别 bootstrap `d_current` 和 `d_off` 后再
相减。候选实现沿用 Mortal corrected summarizer 的 complete-hanchan unit 和
equal-seed hierarchical resampling：每个 replicate 从三个 training seed identity
等概率有放回抽取三个 seed，再在每个被抽取的 seed 内对其 `interaction_row[s,h]`
按 hanchan 有放回抽取 1000 行，最后平均三个 seed means。

```text
bootstrap_reps = 5000
bootstrap_seed = 20260818
```

## 11. Power / scale feasibility

本节只用已冻结的 D1-M0 historical rows 做设计方差 proxy，不改变 intervention、seed
数量或 evaluation scale。

从 corrected `d1_b250_eval_1000h_rows.csv` 读取的 `D1-M0` hanchan-level dispersion：

| training seed | hanchans | mean Pt | sample SD Pt | mean SE Pt |
| ---: | ---: | ---: | ---: | ---: |
| 20260806 | 1000 | -7.425 | 144.968 | 4.584 |
| 20260807 | 1000 | -10.980 | 136.281 | 4.310 |
| 20260808 | 1000 | -7.155 | 138.372 | 4.376 |

pooled hanchan sample SD 为 `139.887` Pt，pooled mean SE 为 `2.554` Pt；历史
equal-seed hierarchical CI 为 `[-14.025, -3.225]`。若保守地假设 current 与 CQL_OFF
gap 的 hanchan noise 独立，interaction 的 hanchan SD proxy 为约 `197.830` Pt，
三个 seed、每 seed 1000h 的 pooled mean SE proxy 约 `3.612` Pt，尚未计入 model/seed
heterogeneity。

据此，`3 training seeds × 1000h` 是检测大约 8 Pt 以上、方向稳定的 interaction 的
合理最低规模；对小于数 Pt 的机制效应明显没有充足保证。3 seeds 不能提供强的泛化
power claim，但与当前治理风格一致，也不应在看到 historical variance 后现场扩大到
6 seeds 或改变 1000h。若 review 认为预期 effect 小于该可辨识范围，应在 freeze 前
明确标记为 design limitation，而不是事后改变 intervention 或 primary gate。

## 12. 允许与禁止的科学结论

### 12.1 若 `interaction_supported`

最多允许得出：

> 在固定的 K0@70k legacy state、M0/D1 full corpus-lineage contrast、preserved-Adam、
> 2000-step continuation、`behavior_action_mc + final_rank_mc` contract 下，
> continuation 阶段新的 CQL gradient contribution causally participates in the
> D1-vs-M0 penalty。

仍然不允许得出：

- CQL 在所有 corpus、任务或训练阶段都不好。
- `cql_weight=0` 是新的 operational recipe。
- CQL 应永久删除，或应改用某个其它 weight。
- CQL_OFF checkpoint 自动晋级 K1 或 model pool。
- D1 的全部 underperformance 都由 CQL 单独造成。

recipe promotion 必须另开 confirmation experiment；机制确认与 operational recipe
promotion 不合并。

### 12.2 若未通过

`interaction_not_confirmed` 只表示本设计下没有通过预先候选的 interaction gate，
不证明 CQL 无 interaction，也不证明 D1/M0 等价。若是 provenance/runtime/loader gate
失败，则使用 `no_verdict_gates_failed`，不得把 gate failure 当成科学阴性结果。

## 13. 未来 freeze 前的最小 gate 清单

以下是 review 后才可实施的 checklist，不是本轮执行命令：

- 完成 feasibility audit 并解析 historical CURRENT training reuse 的唯一 PASS/FAIL 结论。
- 完成 draft repair review 后，先正式 freeze C1 preregistration。
- preregistration freeze 后再注册 registry entry；draft 本身不注册。
- 固定 CQL_OFF config、config SHA 和 objective-side diff；不允许新增其它 candidate。
- 完成 M0/D1 loader compatibility preflight；不通过则 fresh full 2×2 training。
- 明确新 CQL_OFF training artifact root、seed、parent SHA、optimizer state digest 和
  `70000 -> 72000` completion proof。
- 冻结统一 fresh Linux evaluation runtime、native SHA、Mortal revision、evaluator commit、
  seed blocks、B250 shards、opponent model SHA 和 rank points。
- 对 CURRENT/OFF 两个独立四人 lineup 使用相同 hanchan identities；不得把历史 evaluation
  logs 混入 C1 primary statistic。
- 冻结 complete-hanchan interaction row schema、bootstrap `B`、seed 和 adjudication rule。
- 完成 implementation/preflight review 后，才提交 authorization-only training commit。
- authorization-only commit 永远发生在 prereg freeze 与 registry registration 之后；本 draft
  本身永不触发 optimizer step。

## 14. 本轮交付声明

```text
TRAINING NOT STARTED
GENERATION NOT STARTED
OPTIMIZER STEPS = 0
NEW CHECKPOINTS = 0
K1 = null
D1/D2/D3 unchanged
C1 NOT YET FROZEN / NOT AUTHORIZED
```

本轮新增并验证：

```text
training/mortal/audit_c1_loader_compatibility_2026_08.py
tests/test_c1_loader_compatibility.py
```

审计 artifact 写入非-authoritative feasibility root：

```text
/media/bailan/DISK/AUbuntuProject/project/keqing1_experiment/
  artifacts/experiments/C1_corpus_cql_interaction_2026_08_feasibility/
```

`research_registry.json`、frozen K0 prereg、D1/D2/D3 artifacts 和 authoritative output
root 均未修改；本轮没有启动训练、生成或 evaluation。
