#!/usr/bin/env python3
"""Build the tracked artifact index for the P4 student line.

Why this exists: the line's data, checkpoints, logs and result JSON all live in
``artifacts/`` which is gitignored, and they are split across two roots
(``artifacts/experiments/student_policy_v1`` for caches/training/diagnostics and
``artifacts/eval`` for arena evaluations).  Without a tracked index there is no
way to tell from the repository which files a result came from.  This script
regenerates ``training/docs/mortal/experiments_zh/2026-09_P4_学生主线_产物索引.json``:
one entry per artifact with size (and sha256 for irreplaceable files, or a
sorted (name:size) listing digest for directories).

Run from the repository root:

    python training/mortal/build_student_line_artifact_index.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SP = "artifacts/experiments/student_policy_v1"
EV = "artifacts/eval"
OUT = ROOT / "training/docs/mortal/experiments_zh/2026-09_P4_学生主线_产物索引.json"

ENTRIES: list[tuple[str, str, str, str, bool, str]] = [
    # (experiment, kind, path, description, hash, dir pattern)
    # ---- P4-M1: external relabel label cache -------------------------------
    ("P4-M1", "dir", f"{SP}/labels_full_18000h",
     "正式标签缓存：5,796 个 npz 分片 + manifest.json（18,000 半庄 / 11,487,044 行 / teacher-behavior 94.72%）", True, "*.npz"),
    ("P4-M1", "file", f"{SP}/labels_full_18000h/manifest.json",
     "cache manifest：源文件 SHA-256、逐分片行数、train/holdout split、分相 timing、totals", True, ""),
    ("P4-M1", "dir", f"{SP}/labels_pilot_300h",
     "pilot 标签缓存（150 半庄，早期管线 smoke；非正式）", False, "*.npz"),
    ("P4-M1", "file", f"{SP}/relabel_full.log", "正式 relabel 首段日志（跑到 12,300 半庄时手动停止）", False, ""),
    ("P4-M1", "file", f"{SP}/relabel_full_resume.log", "正式 relabel 续跑日志（12,300 → 18,000，含分相 timing）", True, ""),
    # ---- P4-M2: hard-label student + arena evaluation ----------------------
    ("P4-M2", "dir", f"{SP}/student_formal_25k",
     "硬标签学生正式运行目录（student.pth 50k、student_step_050000.pth 归档、history.json、各段训练日志、tb_student、training_contract.json）", False, "*"),
    ("P4-M2", "file", f"{SP}/student_formal_25k/student_step_050000.pth",
     "★ 50k 学生权重归档（当前唯一保留的学生权重；25k/36k/46k 已被周期保存覆盖）", True, ""),
    ("P4-M2", "file", f"{SP}/student_formal_25k/student_step_050000.archive.json", "归档记录（steps / sha256 / 覆盖说明）", False, ""),
    ("P4-M2", "file", f"{SP}/student_formal_25k/history.json", "26 个 holdout 读点（overall + S0/D3/V2）", True, ""),
    ("P4-M2", "file", f"{SP}/student_formal_25k/full_holdout_50k.json", "全量 holdout（1,146,847 行）结果：CE 0.2420 / agreement 0.9047", False, ""),
    ("P4-M2", "dir", f"{SP}/student_sampling_sanity", "采样修复后的 2 步端到端 sanity（非正式）", False, "*"),
    ("P4-M2", "dir", f"{SP}/student_full_cache_sanity", "大缓存 1200 步 sanity（非正式）", False, "*"),
    ("P4-M2", "dir", f"{SP}/student_pilot_run", "pilot 300h 短训（最早一轮，非正式）", False, "*"),
    ("P4-M2", "dir", f"{EV}/ovt_student50k_vs_3k0", "★ 1v3：student50k 单挑 vs 3×K0_70k（256 半庄，seed 700100-700163）", False, "*"),
    ("P4-M2", "dir", f"{EV}/ovt_3student50k_vs_k0", "★ 1v3 反向：K0_70k 单挑 vs 3×student50k", False, "*"),
    ("P4-M2", "dir", f"{EV}/ovt_student50k_vs_3ext", "★ 1v3：student50k 单挑 vs 3×ext_mortal（seed 710000-710063）", False, "*"),
    ("P4-M2", "dir", f"{EV}/ovt_3student50k_vs_ext", "★ 1v3 反向：ext_mortal 单挑 vs 3×student50k", False, "*"),
    ("P4-M2", "dir", f"{EV}/statreport_hard50k_A", "行为表型报告（student50k 单挑座位；detailed_stats.json/md）", False, "*"),
    ("P4-M2", "dir", f"{EV}/statreport_hard50k_B", "行为表型报告（student50k 三座方向）", False, "*"),
    ("P4-M2", "dir", f"{EV}/student50k_smoke", "8 局四引擎冒烟（仅加载链路验证；其“偏保守”结论已撤回）", False, "*"),
    ("P4-M2", "dir", f"{EV}/one_vs_three_speedtest", "1v3 吞吐测速（32 半庄，对照四引擎模式 3.2 局/分钟）", False, "*"),
    # ---- P4-M3: T=1 pure-KL distillation -----------------------------------
    ("P4-M3", "dir", f"{SP}/student_distill_t1_5k",
     "★ T=1 纯 KL 蒸馏阶段目录（distill_recipe.json、distill_state.pth、三个 stage 权重、distill_history.json、tb_distill）", False, "*"),
    ("P4-M3", "file", f"{SP}/student_distill_t1_5k/student_distill_step_005000.pth", "★ 蒸馏端点（5000 步）权重", True, ""),
    ("P4-M3", "file", f"{SP}/student_distill_t1_5k/student_distill_step_002500.pth", "蒸馏 2500 步 stage 权重", True, ""),
    ("P4-M3", "file", f"{SP}/student_distill_t1_5k/student_distill_step_000000.pth", "蒸馏 0 步 stage 权重（应等于父 50k 模型权重）", True, ""),
    ("P4-M3", "file", f"{SP}/student_distill_t1_5k/distill_recipe.json", "蒸馏配方身份（父 sha256 / T / lr / warmup / batch / seed / 预算）", True, ""),
    ("P4-M3", "file", f"{SP}/student_distill_t1_5k/distill_history.json", "蒸馏监测轨迹（KL + hard agreement，overall/分池）", True, ""),
    ("P4-M3", "dir", f"{EV}/ovt_distill5k_vs_3ext", "★ 1v3：distill5k 单挑 vs 3×ext（同 seed 段，用于配对比较）", False, "*"),
    ("P4-M3", "dir", f"{EV}/ovt_3distill5k_vs_ext", "★ 1v3 反向：ext 单挑 vs 3×distill5k", False, "*"),
    ("P4-M3", "file", f"{EV}/distill_5k.err.log", "蒸馏训练日志（logging INFO 走 stderr；stdout 为空为预期）", True, ""),
    ("P4-M3", "file", f"{SP}/gap_diag_distill_step_000000.json", "stage 0 的固定样本监督诊断（父基线）", False, ""),
    ("P4-M3", "file", f"{SP}/gap_diag_distill_step_002500.json", "stage 2500 的固定样本监督诊断", False, ""),
    ("P4-M3", "file", f"{SP}/gap_diag_distill_step_005000.json", "stage 5000 的固定样本监督诊断", False, ""),
    # ---- P4-M4: fixed-sample supervision-gap diagnostic --------------------
    ("P4-M4", "file", f"{SP}/teacher_supervision_gap_50k.json",
     "★ 固定样本（75,855 行 ≥2 合法动作）监督缺口诊断：agreement 0.9007、rank/gap/loss 分布", True, ""),
    ("P4-M4", "file", f"{EV}/gap_diag.stdout.log", "该诊断的运行日志（第一次运行；分组命名随后在 48f484b 修正）", False, ""),
    # ---- P4-M5: online-state diagnostic -----------------------------------
    ("P4-M5", "file", f"{SP}/online_state_diag_hard50k.json",
     "★ 在线状态诊断（student50k）：forced-discard 修正后 agreement 0.9006/0.9023", True, ""),
    ("P4-M5", "file", f"{SP}/online_state_diag_distill5k.json",
     "★ 在线状态诊断（distill5k）：0.9134/0.9126", True, ""),
    # ---- P4-M6: capacity candidate (resource probe + first 10k segment) ----
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_probe/microbatch128.json",
     "资源探针 microbatch128：稳定 672.6 rows/s，WDDM 最低余量 1604 MiB（PASS）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_probe/microbatch256.json",
     "资源探针 microbatch256：WDDM 最低余量 116 MiB、系统内存 92.3%、提交内存贴近上限（REJECT）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_probe/p4_m6_resource_probe.py",
     "资源探针脚本（真实 trainer 模型/数据/损失路径，不保存权重）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_probe/run_capacity_10k_monitored.py",
     "10k 首段的带资源监控运行包装脚本", False, ""),
    ("P4-M6", "dir", f"{SP}/P4-M6_capacity_256x54_hardce_10k",
     "容量候选首段训练目录（256×54、microbatch128×accum4、纯 hard 教师贪心 CE，10k optimizer updates）", False, "*"),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_000000.pth",
     "★ 0 步 stage 权重（fresh 256×54 初始化）", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_002000.pth", "★ 2k 步 stage 权重", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_005000.pth", "★ 5k 步 stage 权重", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_010000.pth",
     "★ 10k 步 stage 权重（首段端点；尚未做 1v3 强度评测）", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/history.json",
     "5 个 holdout 读点（2k/4k/6k/8k/10k；截至 10k agreement 0.8653）", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/resource_summary.json",
     "10k 运行资源峰值（wall 9567 s、GPU 峰值 6801/8188 MiB、系统内存峰值 98.7%）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/train.stderr.log", "10k 首段训练日志", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_015000.pth",
     "★ 15k 步 stage 权重（续训段）", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_020000.pth",
     "★ 20k 步 stage 权重（续训端点；尚未做 1v3 强度评测）", True, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_probe/run_capacity_resume_20k_monitored.py",
     "续训到 20k 的监控运行包装脚本（含时间序列与安全暂停）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/resource_timeseries_10k_to_20k.jsonl",
     "★ 续训资源时间序列（每 10s 一行，1150 行）", False, ""),
    ("P4-M6", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/resource_summary_10k_to_20k.json",
     "续训资源汇总（wall 11624 s、未触发暂停）", False, ""),
    # ---- P4-M7: eval-only strength screen -------------------------------
    ("P4-M7", "file", f"{SP}/P4-M6_capacity_256x54_hardce_10k/student_step_020000_eval_weights.pth",
     "★ P4-M7 eval-only 权重副本（仅去除训练状态；模型张量与 20k checkpoint 逐张量一致）", True, ""),
    ("P4-M7", "dir", f"{EV}/ovt_capacity256x54_20k_vs_3ext",
     "★ 256×54@20k 单挑 vs 3×external（256 半庄，seed 710000–710063）", False, "*"),
    ("P4-M7", "dir", f"{EV}/ovt_3capacity256x54_20k_vs_ext",
     "★ external 单挑 vs 3×256×54@20k（256 半庄，seed 710000–710063）", False, "*"),
    ("P4-M7", "dir", f"{EV}/p4m7_comparisons",
     "★ P4-M7 paired comparison 与预注册裁决 summary", False, "*"),
    ("P4-M7", "file", f"{EV}/p4m7_comparisons/ENVIRONMENT_PROVENANCE.md",
     "★ P4-M7 原生评测环境溯源与可比性判定（未确认与基线一致）", False, ""),
    # ---- P4-M8: baseline evaluation entry-point reproduction check --------
    ("P4-M8", "dir", f"{EV}/_envcheck_hard50k_vs_3ext_710000_710007",
     "★ P4-M8 基线评测入口复现检查（同环境 32 半庄，逐事件一致 REPRODUCED）", False, "*"),
    ("P4-M8", "file", f"{EV}/_envcheck_hard50k_vs_3ext_710000_710007/RESULT.md",
     "★ P4-M8 结果记录（授权范围、实际命令、环境指纹、判定与不可外推声明）", False, ""),
    ("P4-M8", "file", f"{EV}/_envcheck_hard50k_vs_3ext_710000_710007/launch_environment.json",
     "启动时环境指纹（解释器/PYTHONPATH/原生二进制 sha256）", False, ""),
    ("P4-M8", "file", f"{EV}/_envcheck_hard50k_vs_3ext_710000_710007/comparison_vs_historical_baseline.json",
     "★ 逐事件比较结果（32/32 文件、33,362 条记录、0 处游戏字段不一致）", False, ""),
    # ---- P4-M9: on-policy contract + resource probe --------------------------
    ("P4-M9", "dir", f"{EV}/_p4m9_probe_onpolicy_contract",
     "★ P4-M9 on-policy 契约+资源探针（32 半庄采集，契约未完全通过）", False, "*"),
    ("P4-M9", "file", f"{EV}/_p4m9_probe_onpolicy_contract/RESULT.md",
     "★ P4-M9 结果记录（四个问题、分歧类别、成本、不可外推声明）", False, ""),
    ("P4-M9", "file", f"{EV}/_p4m9_probe_onpolicy_contract/probe_result.json",
     "★ 四项检查结果（对齐/重算/BN/梯度）+ 成本 + 启动环境指纹", False, ""),
    ("P4-M9", "file", f"{EV}/_p4m9_probe_onpolicy_contract/probe_records.jsonl",
     "逐决策记录（mask_bits/q_legal/采样动作/log-prob，obs 偏移索引）", False, ""),
    ("P4-M9", "file", f"{EV}/_p4m9_probe_onpolicy_contract_sample2/probe_result.json",
     "第 2 次重采的四项检查（独立性/稳定性复核）", False, ""),
    ("P4-M9", "file", f"{EV}/_p4m9_probe_onpolicy_contract_sample3/probe_result.json",
     "第 3 次重采的四项检查（独立性/稳定性复核）", False, ""),
    # ---- P4-M10: first direct on-policy PG candidate -------------------------
    ("P4-M10", "dir", f"{SP}/P4-M10_onpolicy_pg_4x256",
     "★ P4-M10 首候选：hard50k → 直接 on-policy PG（4 cycle × 256 半庄，只评 C4；not_supported）", False, "*"),
    ("P4-M10", "file", f"{SP}/P4-M10_onpolicy_pg_4x256/RESULT.md",
     "★ P4-M10 训练记录（冻结配方、每 cycle 契约检查、成本、只评 C4 与不可外推声明）", False, ""),
    ("P4-M10", "file", f"{SP}/P4-M10_onpolicy_pg_4x256/p4m10_result.json",
     "★ 四 cycle 逐轮结果（采集/更新/重算门限/梯度/checkpoint 与解释器指纹）", False, ""),
    ("P4-M10", "file", f"{SP}/P4-M10_onpolicy_pg_4x256/C4_eval_weights.pth",
     "被评测的最终候选权重（652 张量与 C4.pth 逐一相同）", True, ""),
    ("P4-M10", "dir", f"{EV}/ovt_p4m10_C4_vs_3ext",
     "★ C4 单挑 vs 3×ext_mortal（256 半庄，greedy 部署策略）", False, "logs/metrics.json"),
    ("P4-M10", "dir", f"{EV}/ovt_3p4m10_C4_vs_ext",
     "★ ext_mortal 单挑 vs 3×C4（256 半庄，greedy 部署策略）", False, "logs/metrics.json"),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/summary.json",
     "★ P4-M10 裁决（candidate benefit 双向符号、规则判定、follow-up 禁令）", False, ""),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/ADJUDICATION.md",
     "★ P4-M10 裁决文档（预注册规则、原始结果、允许/不允许结论）", False, ""),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/solo_vs_ext_c4_minus_hard50k.json",
     "方向 A same-seed paired 差值（C4 − hard50k）", False, ""),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/ext_vs_trio_delta.json",
     "方向 B same-seed paired 差值（ext 面对 3×C4 − 3×hard50k）", False, ""),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/policy_displacement.json",
     "★ 固定面板策略位移（flip/TV/KL，parent 与相邻 cycle）——只读检查", False, ""),
    ("P4-M10", "file", f"{EV}/p4m10_comparisons/POLICY_DISPLACEMENT.md",
     "★ 策略位移检查说明与读数边界", False, ""),
    ("P4-M10", "file", f"{SP}/P4-M10_onpolicy_pg_4x256/guard_reconciliation_backfill.json",
     "★ agari guard / claim 抢先的只读补记（运行期字段为 null，本文件补出真实计数）", False, ""),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dir_listing_digest(path: Path, pattern: str) -> dict[str, object]:
    digest = hashlib.sha256()
    files = 0
    total = 0
    for item in sorted(path.glob(pattern)):
        if item.is_file():
            size = item.stat().st_size
            digest.update(f"{item.name}:{size}\n".encode())
            files += 1
            total += size
    return {"files": files, "bytes": total, "listing_sha256": digest.hexdigest()}


def main() -> None:
    entries = []
    for experiment, kind, relative, description, want_hash, pattern in ENTRIES:
        path = ROOT / relative
        entry: dict[str, object] = {
            "experiment": experiment,
            "kind": kind,
            "path": relative,
            "description": description,
            "exists": path.exists(),
        }
        if path.exists():
            if kind == "file":
                entry["bytes"] = path.stat().st_size
                if want_hash:
                    entry["sha256"] = sha256_file(path)
            else:
                entry.update(dir_listing_digest(path, pattern or "*"))
        entries.append(entry)

    index = {
        "schema": "keqing.mortal.student_line_artifact_index.v1",
        "created_at": "2026-09-11",
        "line": "P4 学生主线（Mortal-native 学生：现成语料 → external 重标注 → 纯策略训练 → 实战评测）",
        "generator": "training/mortal/build_student_line_artifact_index.py",
        "report": "training/docs/mortal/experiments_zh/2026-09_P4_学生主线_50k训练_1v3双向评测_在线诊断_最终结果报告.md",
        "artifact_roots": {
            "training_and_diagnostics": "artifacts/experiments/student_policy_v1/",
            "arena_evaluation": "artifacts/eval/",
            "note": "本线产物分散在两个根下；新增实验请使用 P4-Mx 前缀目录。artifacts/ 被 gitignore，本索引是仓库内可追踪的唯一清单。",
        },
        "conventions": [
            "file 条目：bytes；★ 条目另含 sha256（不可再生或高价值产物）。",
            "dir 条目：files / bytes / listing_sha256 = 对 (文件名:大小) 排序列表的 SHA-256，用于低成本漂移检测，不是内容哈希。",
            "要重算索引：python training/mortal/build_student_line_artifact_index.py",
        ],
        "entries": entries,
    }
    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [e["path"] for e in entries if not e["exists"]]
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} entries, {len(missing)} missing")
    for path in missing:
        print("  MISSING:", path)


if __name__ == "__main__":
    main()
