# Qwen 四方法实验运行指南

只使用 **`exp/qwen-experiments`** 分支。通过 `--dataset` 和 `--method` 选择三个数据集 × 四种方法，共 12 种实验。所有命令在仓库根目录的 Bash 中运行。

## 1. 安装一次

需要 Python 3.12+、Go 1.25+、可用的 AgentENV 快照服务，以及能够访问任务镜像仓库的网络。AgentENV 需另外部署；已有服务器可直接复用。完整 Ash 仓库是运行依赖，不能只复制 `experiments/qwen/`。

```bash
git clone --single-branch --branch exp/qwen-experiments https://github.com/dreamyang-liu/Ash.git
cd Ash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r experiments/qwen/requirements.txt
pip install ./sdk
export PYTHONPATH=.:sdk
export LITELLM_LOCAL_MODEL_COST_MAP=True
(cd runtime && go build -o ash-runtime .)
```

准备私有接口和密钥。以下均为占位配置，替换为自己的值；不要把密钥提交到 Git。

```bash
export QWEN_BASE_URL=http://your-private-model-endpoint
read -rs -p 'Qwen API key: ' QWEN_API_KEY; echo
export QWEN_API_KEY
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_KEY_FILE=/path/to/aenv-key
```

## 2. 设置任务目录和评测范围

数据集和环境镜像需单独准备，仓库中包含的是实验代码及 HARD-51 选择清单。任务目录须保留 `task.toml`、题目、环境配置和官方验证脚本；仅下载原始样本表不够。

| 数据集 | `--dataset` | 任务来源与准备要求 |
|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | 使用已准备的 `v2/tasks`；清单为仓库内 `selection.json`。已有服务器路径：`/opt/ash-validation/src/SWE-bench_Pro-v2-20260925/v2/tasks` |
| TerminalBench 2.1 | `terminalbench21` | [官方任务](https://github.com/harbor-framework/terminal-bench-2-1)及自己的评测清单 |
| DeepSWE | `deepswe` | [官方任务](https://github.com/datacurve-ai/deep-swe)及自己的评测清单 |

清单是任务子目录名的 JSON 数组，例如 `["largest-eigenval"]` 表示只跑一个 TerminalBench 任务。正式评测须明确列出整个评测集合，四种方法使用相同清单。

以下配置在当前 Bash 会话中执行一次；先把路径替换成实际路径。输出目录放在仓库外。

```bash
export PRO_TASKS=/data/swebenchpro/v2/tasks
export TB_TASKS=/data/terminal-bench-2-1/tasks
export DEEP_TASKS=/data/deep-swe/tasks
export TB_SELECTION=/data/cohorts/terminalbench21.json
export DEEP_SELECTION=/data/cohorts/deepswe.json
export RUNS=/data/runs/qwen

COMMON=(--runtime-bin "$PWD/runtime/ash-runtime" --api-key-file "$AENV_KEY_FILE" --workers 16)
PRO=(--dataset swebenchpro-v2-hard --tasks-dir "$PRO_TASKS" --selection "$PWD/experiments/qwen/swebenchpro-v2-hard/selection.json")
TB=(--dataset terminalbench21 --tasks-dir "$TB_TASKS" --selection "$TB_SELECTION")
DEEP=(--dataset deepswe --tasks-dir "$DEEP_TASKS" --selection "$DEEP_SELECTION")
```

新终端需要重新激活虚拟环境并设置这些环境变量和 Bash 数组。

## 3. 选择 12 种实验中的一种

每个数据集先完成 **baseline**，再运行该数据集的 SPROUT、BPO、Shepherd；后三者用 `--baseline-root` 复用同一条首轮轨迹。

默认：mini-swe-agent、`qwen3.8-27b`、推理强度 `high`、assistant-turn 分支、每条 rollout 最多 300 次模型调用、输出上限 64000。baseline 无论成功或失败均跑 4 条；分支方法首条成功则跳过，失败则含首条最多 8 次。

### SWE-bench-Pro v2 HARD-51：实验 1–4

```bash
# 1. baseline
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/swebenchpro-v2-hard-baseline"
# 2. SPROUT
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-sprout"
# 3. entropy-based BPO
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-bpo"
# 4. Shepherd
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-shepherd"
```

### TerminalBench 2.1：实验 5–8

```bash
# 5. baseline
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/terminalbench21-baseline"
# 6. SPROUT
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-sprout"
# 7. entropy-based BPO
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-bpo"
# 8. Shepherd
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-shepherd"
```

### DeepSWE：实验 9–12

```bash
# 9. baseline
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/deepswe-baseline"
# 10. SPROUT
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-sprout"
# 11. entropy-based BPO
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-bpo"
# 12. Shepherd
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-shepherd"
```

每条命令默认并发 16；同时运行多条命令时并发会叠加。把 `COMMON` 中的 `--workers` 调低即可限制资源。

## 4. 可选：一个命令运行一个数据集的四种方法

这与上面的分开运行是两种启动方式，选择其中一种即可。`all` 按任务先执行四条 baseline，再依次执行三个分支方法。

```bash
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method all --output "$RUNS/swebenchpro-v2-hard-all"
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method all --output "$RUNS/terminalbench21-all"
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method all --output "$RUNS/deepswe-all"
```

## 5. 预检与小任务验证

在第 3 或第 4 节的命令末尾加 `--plan`，只校验任务文件并打印执行计划，不调用模型、不创建 VM。预检不能替代网络、镜像和官方验证器的实测。

下面两条命令各选择一个任务，并使用小预算验证四种方法的执行链路：

```bash
python -m experiments.qwen.run --dataset terminalbench21 --method all --tasks-dir "$TB_TASKS" --task largest-eigenval "${COMMON[@]}" --workers 1 --max-turns 2 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/terminalbench21-smoke"
python -m experiments.qwen.run --dataset deepswe --method all --tasks-dir "$DEEP_TASKS" --task superjson-error-stack-serialization "${COMMON[@]}" --workers 1 --max-turns 3 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/deepswe-smoke"
```

小预算仍执行四条 baseline；每个分支方法的总预算缩小到 2。小任务结果只用于联调，不代表完整评测成绩。

## 6. 输出五方法、四指标

如果使用第 3 节的独立运行方式，完成后选择数据集并汇总：

```bash
DS=swebenchpro-v2-hard  # 或 terminalbench21、deepswe
python -m experiments.qwen.compare \
  --baseline "$RUNS/$DS-baseline" \
  --sprout "$RUNS/$DS-sprout" \
  --bpo "$RUNS/$DS-bpo" \
  --shepherd "$RUNS/$DS-shepherd" \
  --output "$RUNS/$DS-comparison"
```

输出 `comparison.csv` 和 `comparison.json`：五行是 pass@1、pass@4（baseline）、BPO、Shepherd、SPROUT；四项指标是 Resolve Rate、Recovery、Steps、Tokens (M)。

如果使用第 4 节的 `all` 方式，用下面的命令生成包含五行 `comparison` 的 JSON：

```bash
DS=swebenchpro-v2-hard  # 或 terminalbench21、deepswe
python -m experiments.qwen.report "$RUNS/$DS-all" --method all --save "$RUNS/$DS-all/comparison.json"
```

## 7. 续跑与 checkpoint 保留

- 完全相同配置、相同代码提交下，重复原命令会复用已完成阶段。存在未完成阶段时会拒绝覆盖；保留原输出后，用新目录重试，并保留失败消耗记录。
- 本入口只识别自己生成的 manifest 和阶段记录。服务器历史控制器的旧运行目录不能直接作为本入口的 `--baseline-root`。
- 首条失败会保存 `retained-checkpoints.json`、日志及精确会话前缀引用；恢复还依赖 AgentENV 后端的快照对象。不要只复制清单后就删除后端存储。
- 实验包不执行快照清理。磁盘快照不保留 RAM、运行中进程或 tmpfs；依赖这些状态的任务需要完整快照支持及单独标注的实验。
- BPO、Shepherd 使用推理阶段策略适配。预算、统计和恢复限制详见 [实验协议](experiments/qwen/PROTOCOL.md)。
