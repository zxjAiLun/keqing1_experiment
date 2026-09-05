# T1 Windows 评测恢复记录

2026-09-05 恢复时，原评测句柄 61289 已失效，系统中未发现对应 evaluator 或 four_player_native Python 进程。现有评测目录保留 1200 局：seed 20260910 四个完整 shard 共 1000 局，seed 20260911 的 shard_000 已落盘 200 局。尚无最终 evaluation manifest 或 summary，因此没有强度结论。

恢复没有修改冻结的模型、训练、评测 game IDs、seed key、随机座位、50-game inference batch 或统计判据。训练不重跑。T1 evaluator 新增显式 `--resume`：

- 在启动新对局前，检查全部已有日志的文件名、start_game seed、四模型名称、end_game，以及每 shard 的唯一连续前缀。
- 只接受完整 50 局批次，避免从批内任意位置恢复改变 inference batching。
- 已完成 shard 同时验证 metrics 的协议与模型路径；未完成 shard 使用底层已有的 native `--resume`。
- 将既有日志 SHA256 与 training manifest SHA256 写入独立恢复凭据；运行后验证已保留日志内容没有改变。
- 训练 manifest、K0、外部模型与各 seed checkpoint 继续执行既有验证；最终仍须验证完整 3000 局才能输出 eval manifest。
- 子进程诊断持久化到每 shard 的 execution.log。Windows 原生后台脚本顺序运行评测与汇总；最终报告与 registry 收口仍需人工代理审计。

恢复校验的聚焦测试为 22 passed，Ruff 与 git diff --check 通过。中断原因目前未确认；进程终止不能仅凭日志空窗推断为 GPU 故障。
