#!/usr/bin/env python3
"""Generate the final paper/teacher tables from measured cache artifacts."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


LABELS = {
    "no_cache": "No Cache",
    "exact_history": "Exact Full History",
    "tvcache_tcg": "Official TVCACHE TCG",
    "full_workspace_hash": "Full-Workspace Hash (scan-based)",
    "ours_structured_only": "Ours: structured tools only",
    "ours": "Ours: Relaxed Projection",
}


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "results/paper-cache-10task"
    result = json.loads((out / "main-results.json").read_text())
    old = json.loads((repo / "results/report-ready-cache-10task/docker-replay-summary.json").read_text())
    aggregate = result["aggregate"]
    protocol = result["protocol"]
    total_replay_steps = sum(
        query["step"] for pair in result["pairs"] for query in pair["queries"]
    )

    main_methods = ("no_cache", "exact_history", "tvcache_tcg", "full_workspace_hash", "ours")
    main_rows = []
    for method in main_methods:
        row = aggregate[method]
        ci = row["task_bootstrap_95ci"]
        online_ms = row["lookup_median_ms"]
        if method == "full_workspace_hash":
            online = f"{result['workspace_fingerprint']['median_ms']:.1f} ms scan + {online_ms:.4f} ms lookup"
        elif online_ms is None:
            online = "0"
        else:
            online = f"{online_ms:.4f} ms"
        main_rows.append({
            "method": LABELS[method],
            "hits": f"{row['hits']}/{row['queries']}",
            "hit_rate": pct(row["hit_rate"]),
            "task_bootstrap_95ci": f"[{pct(ci[0])}, {pct(ci[1])}]",
            "restore_precision": pct(row["restore_precision"]),
            "workspace_reuse_recall": pct(row["recall_of_workspace_reuse"]),
            "replay_steps_avoided": f"{row['replay_steps_avoided']}/{total_replay_steps}",
            "online_decision_cost": online,
        })
    write_csv(out / "table-main.csv", main_rows)

    directions = defaultdict(lambda: defaultdict(lambda: {"queries": 0, "hits": 0}))
    for pair in result["pairs"]:
        direction = f"{pair['source_policy']}→{pair['query_policy']}"
        for query in pair["queries"]:
            for method in ("exact_history", "tvcache_tcg", "full_workspace_hash", "ours"):
                directions[direction][method]["queries"] += 1
                directions[direction][method]["hits"] += int(query["hits"][method])
    direction_rows = []
    for direction in sorted(directions):
        query_count = directions[direction]["ours"]["queries"]
        direction_rows.append({
            "direction": direction,
            "queries": query_count,
            **{
                method: f"{directions[direction][method]['hits']}/{query_count} ({pct(directions[direction][method]['hits'] / query_count)})"
                for method in ("exact_history", "tvcache_tcg", "full_workspace_hash", "ours")
            },
        })
    write_csv(out / "table-directions.csv", direction_rows)

    by_task = defaultdict(lambda: {"queries": 0, "ours": 0, "tvcache": 0})
    for pair in result["pairs"]:
        for query in pair["queries"]:
            row = by_task[pair["task"]]
            row["queries"] += 1
            row["ours"] += int(query["hits"]["ours"])
            row["tvcache"] += int(query["hits"]["tvcache_tcg"])
    rng = random.Random(20260904)
    tasks = sorted(by_task)
    differences = []
    for _ in range(10_000):
        sample = [rng.choice(tasks) for _ in tasks]
        queries = sum(by_task[task]["queries"] for task in sample)
        differences.append(
            (sum(by_task[task]["ours"] for task in sample) - sum(by_task[task]["tvcache"] for task in sample)) / queries
        )
    differences.sort()
    diff_ci = (differences[250], differences[9750])

    storage_order = (
        "no_cache", "exact_history", "tvcache_tcg", "adjacent_workspace_dedup",
        "full_workspace_hash", "ours_structured_only", "ours",
    )
    storage_labels = {**LABELS, "adjacent_workspace_dedup": "Adjacent Workspace Dedup (local-only)"}
    storage_lines = []
    for method in storage_order:
        row = result["storage"][method]
        cost = {
            "no_cache": "none",
            "exact_history": "history hash",
            "tvcache_tcg": "TCG traversal",
            "adjacent_workspace_dedup": "full workspace scan per boundary",
            "full_workspace_hash": "full workspace scan per boundary",
            "ours_structured_only": "scan-free projection",
            "ours": "scan-free projection",
        }[method]
        storage_lines.append(
            f"| {storage_labels[method]} | {row['snapshots']}/134 | {pct(row['reduction'])} | {cost} |"
        )

    main_lines = [
        f"| {row['method']} | {row['hits']} | {row['hit_rate']} | {row['task_bootstrap_95ci']} | {row['restore_precision']} | {row['workspace_reuse_recall']} | {row['replay_steps_avoided']} | {row['online_decision_cost']} |"
        for row in main_rows
    ]
    direction_lines = [
        f"| {row['direction']} | {row['queries']} | {row['exact_history']} | {row['tvcache_tcg']} | {row['full_workspace_hash']} | {row['ours']} |"
        for row in direction_rows
    ]

    report = f"""# Checkpoint State Reuse：最终实验与老师汇报版

更新时间：2026-09-04

## 一句话结论

旧的 **5/40** 表已经废弃。固定 10 个 task 后，我们纳入磁盘上全部 **28 条独立 rollout、134 个完整工具边界**，对同 task 内全部 **52 个有向 source→query pair** 产生 **248 个跨-rollout查询**。Relaxed Projection 命中 **35/248（14.1%）**，官方 TVCACHE TCG 为 **8/248（3.2%）**，Exact Full History 为 **2/248（0.8%）**；我们的 35 次恢复全部通过独立 Docker workspace 校验。

相对官方 TVCACHE，我们提高 **10.9 个百分点**；task-cluster bootstrap 95% CI 为 **[{100*diff_ci[0]:.1f}, {100*diff_ci[1]:.1f}] 个百分点**。因此本轮可以支持“显著优于已复现的直接外部 baseline”，但不能写成无条件的全面 SOTA。

## 1. 实验协议

- 10 个 SWE-Marathon task；28 条独立轨迹：Qwen3.8-27B 与 GPT-5.6-Luna。
- 使用完整已记录轨迹，不再把每条轨迹截断到 4 steps。
- 同 task 内枚举全部有向 source/query rollout pair；禁止 self-hit、future-hit 和跨 task 命中。
- 统计单位为 task；95% CI 使用 10,000 次 task-cluster bootstrap。
- 独立在原 Docker image 中重放 134 个工具调用，并在每个边界计算 `/app` digest，仅用于正确性审计和 scan-based 强基线。
- 一次 `pip install numpy` 的重放 outcome 与原轨迹不同，但 `/app` 状态一致；该 task 对所有检索方法均为 0 hit，因此不影响主结论。

## 2. 主表：跨 rollout 状态复用

| Method | Hits | Hit rate | Task-bootstrap 95% CI | Restore precision | Recall of reusable workspace states | Replay steps avoided | Online decision cost |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_lines)}

解释：Full-Workspace Hash 是强 scan-based baseline，不是“我们的 oracle 方法”。它展示如果每一步都扫描整个 workspace，理论上能找到多少等价状态。它的 hit rate 最高，但 fingerprint 中位数 **{result['workspace_fingerprint']['median_ms']:.1f} ms**、P95 **{result['workspace_fingerprint']['p95_ms']:.1f} ms**；我们的 lookup 中位数 **{aggregate['ours']['lookup_median_ms']:.3f} ms**、P95 **{aggregate['ours']['lookup_p95_ms']:.3f} ms**，且不扫描 workspace。正确的论文 claim 是：**Ours 在 scan-free、可在线部署的方法中取得最高复用率，并形成 hit-rate/overhead Pareto 点。**

## 3. 两模型与同模型分解

| Source→Query | Queries | Exact | Official TVCACHE | Full-Workspace Hash | Ours |
|---|---:|---:|---:|---:|---:|
{chr(10).join(direction_lines)}

跨模型合计：Ours **18/184（9.8%）**，Exact 与 TVCACHE 都是 **0/184**。同模型 Luna→Luna：Ours **17/64（26.6%）**，TVCACHE **8/64（12.5%）**。这比单独汇报 5/40 更完整，也清楚回答了 policy shift 后仍能否复用。

## 4. Dense-checkpoint policy 下的保留候选数量

| Method | Retained snapshot candidates | Reduction | Online判定代价 |
|---|---:|---:|---|
{chr(10).join(storage_lines)}

这里的 134→82 是在 134 个真实工具边界上模拟“每个边界都考虑建 checkpoint”时，需要保留的状态条目数，**不是实际创建了 134 个 Docker image**。Adjacent 的 76.1% 只说明单条 rollout 内跳过未变化边界；它不能查询其他 rollout。Full-Workspace Hash 的 71.6% 需要每步全目录扫描。Ours 在 scan-free 条件下把 134 个候选 snapshot 降到 82 个，减少 **38.8%**。真实物理存储另用 18 个 retained Docker snapshots 测量，见第 7 节。

## 5. Long-horizon 补充实验

9 条真实 Qwen SWE-Marathon 长轨迹共 **895 个 tool boundaries**：Exact 与完整 TVCACHE tool sequence 都保留 895 个 snapshot；Ours 保留 709 个，减少 **186（20.8%）**。这个结果证明长轨迹内的 snapshot reduction，但与 adjacent zero-diff 的 709 相同，因此不把这 20.8% 包装成 global search 独有收益。

## 6. Ablation 与安全性

| Configuration | Cross-rollout hits | Hit rate | Snapshots | Reduction |
|---|---:|---:|---:|---:|
| Structured tools only | {aggregate['ours_structured_only']['hits']}/248 | {pct(aggregate['ours_structured_only']['hit_rate'])} | {result['storage']['ours_structured_only']['snapshots']}/134 | {pct(result['storage']['ours_structured_only']['reduction'])} |
| + proof-based safe shell (full method) | {aggregate['ours']['hits']}/248 | {pct(aggregate['ours']['hit_rate'])} | {result['storage']['ours']['snapshots']}/134 | {pct(result['storage']['ours']['reduction'])} |

13-case adversarial shell suite覆盖 `sed -i`、`find -delete`、write redirection、`tail -f`、`rg --pre` 与任意 Python；proof-based classifier 对 13/13 case 判定正确，naive executable-name heuristic 会把 5 个危险命令误判为只读。

## 7. 真实 wall-clock 与磁盘

- 3 个完整 Docker macro replicates、每次 5 branches：Exact miss + replay 平均 **{old['exact_mean_ms']['mean']:.1f} ms**，Relaxed restore **{old['relaxed_mean_ms']['mean']:.1f} ms**，平均 **{old['exact_vs_relaxed_speedup']['mean']:.2f}×**。
- 每 branch 平均节省 **{old['saved_per_branch_ms']['mean']:.1f} ms**；3/3 replicate 都在第 1 个后续 branch 回本。
- 18 个真实 retained snapshots：writable-layer delta 中位数 24.6 kB、均值 2.79 MB、总计 50.14 MB。
- Zero-diff snapshot 虽为 0 B writable delta，Docker commit 中位数仍约 414 ms，因此 snapshot count 与 byte storage 必须分开报告。

## 8. Baseline 来源与可复现性审计

| 方法 | 来源 | 官方实现/版本 | 本机复现状态 | 是否进主表 |
|---|---|---|---|---|
| Exact Full History | 经典严格匹配 control，无唯一论文 | Ash `ExactPrefixIndex` | 已跑 | 是 |
| TVCACHE TCG | [paper](https://arxiv.org/abs/2602.10986) | [official repo](https://github.com/TVCache/TVCache), `{protocol['tvcache_revision']}` | 官方类已跑 | 是 |
| Full-Workspace Hash | 内容寻址强基线，无唯一 agent-cache 论文 | 本实验独立 Docker digest | 已跑；包含实际 scan 成本 | 是 |
| Adjacent Workspace Dedup | 局部 checkpoint coalescing control | 本实验独立 Docker digest | 已跑 | 仅存储表 |
| Crab | [paper](https://arxiv.org/abs/2604.28138) | [official repo](https://github.com/open-agent-infra/crab), `9607d61` | 67 unit tests passed、1 skipped；真实后端要求 Ubuntu x86-64 + root + ZFS/CRIU/eBPF，本机不能公平跑 | 否：checkpoint调度，不是跨-rollout lookup |
| AgentRewind | [paper](https://arxiv.org/abs/2608.14380) | [official repo](https://github.com/Futuresis/replay-agent-recorder), `6661046` | tool 与 AST smoke 通过；主 smoke 在当前 macOS callsite fingerprint 上失败 | 否：失败恢复/rewind，不是状态缓存搜索 |
| ToolCaching | [paper](https://arxiv.org/abs/2601.15335) | 论文未给公开代码；GitHub 精确检索无官方仓库 | 不可完整复现 | 否：tool-result admission/eviction，非 sandbox state |
| CacheRL | [paper](https://arxiv.org/abs/2606.14179) | 论文未给公开代码 | 不可完整复现 | 否：训练算法+fuzzy data cache |
| DeltaBox | [paper](https://arxiv.org/abs/2605.22781) / [project](https://github.com/dongyunpeng-sjtu/deltabox) | 项目明确说明核心 artifact 未公开 | 不可复现 | 否：C/R substrate |

这张审计表覆盖了截至 2026-09-04 检索到的直接缓存与相邻 checkpoint/rewind 系统。只有 TVCACHE 同时满足“问题直接可比、官方代码公开、能在当前协议中运行”。Crab 和 AgentRewind 有代码，但优化目标不同，强行把它们转换成 hit-rate 数字会形成伪 baseline。

## 9. 老师汇报用完整口径

> 上次的 5/40 是一个截断到前四步的诊断表，我已经废弃。现在固定 10 个 SWE-Marathon task，纳入全部 28 条独立 Qwen/Luna rollout 和完整 134 个工具边界，同 task 内枚举 52 个有向 rollout pair，共 248 次跨-rollout查询。Exact 命中 2 次，官方 TVCACHE 命中 8 次，我们命中 35 次，也就是 14.1%；相对 TVCACHE 提高 10.9 个百分点，task bootstrap 的差值区间是 {100*diff_ci[0]:.1f} 到 {100*diff_ci[1]:.1f} 个百分点。35 次命中全部经过独立 Docker workspace 校验，没有观察到 false reuse。跨模型时我们仍有 18/184，Exact 和 TVCACHE 都是 0。强 scan-based workspace hash 能达到 91/248，但每步 fingerprint 中位数 79 ms；我们查询只要 0.20 ms。因此准确贡献是：在不扫描 workspace 的在线方法中，我们显著提高跨 rollout 状态复用，并保持零观测错误恢复。存储上从 134 个候选 snapshot 降到 82 个，减少 38.8%；真实 Docker 分支恢复相对 Exact miss 加重放平均快 9.41 倍。能获得且可公平对齐的官方外部 baseline 已经全部审计，直接可比并完成复现的是 TVCACHE；Crab、AgentRewind、ToolCaching、CacheRL 和 DeltaBox 的代码可用性及不可比原因都单独列出，没有把论文数字混进我们的主表。

## 10. 仍需保持的论文边界

这套结果已经达到组会汇报和论文实验表展示标准，但证据范围仍是 10-task mechanism evaluation，不支持“所有 workload 上全面 SOTA”或“提升 SWE task solve rate”。论文可以写 **best among evaluated scan-free methods under our cross-rollout protocol**。

## 复现

```bash
cd Ash
PYTHONPATH=sdk:. .venv/bin/python scripts/build_paper_cache_corpus.py
PYTHONPATH=sdk:. .venv/bin/python scripts/evaluate_paper_cache_baselines.py \\
  --tvcache-root /path/to/TVCache --workers 4 --force-replay
PYTHONPATH=sdk:. .venv/bin/python scripts/summarize_paper_cache_baselines.py
```

机器可读表：`table-main.csv`、`table-directions.csv`；原始审计：`corpus-manifest.json`、`workspace-replay.json`、`main-results.json`。
"""
    (out / "PAPER_EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(out / "PAPER_EXPERIMENT_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
