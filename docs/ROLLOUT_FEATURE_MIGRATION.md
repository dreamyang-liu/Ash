# Rollout 功能迁移验收清单

本文是将 `haixin/rollout-interface` 的功能迁移到 `dev-latest` 持久化任务
架构时使用的唯一验收清单。迁移以外部可观察行为为准，不机械复制原来的
`swebench/rollout_groups` 目录。只有同时明确了持久化架构中的实现位置和
验证证据，一项能力才算迁移完成。

状态含义：

- **已迁移**：持久化架构中的实现保留了要求的行为，并有回归测试。
- **已替代**：`dev-latest` 已有语义更强的持久化机制，因此
  有意不复制旧实现。
- **部分完成**：部分语义已经具备，但仍有明确列出的缺口。
- **待完成**：仍缺实现或有决定性的验证尚未完成。
- **外部事项**：由部署策略或配套仓库负责，不属于本仓库实现。

## 上下文中断后如何继续

本文是迁移范围和进度的事实来源。发生会话中断或上下文压缩后：

1. 在重新分析单个提交前，先完整阅读本文。
2. 确认目标分支仍然基于 `origin/dev-latest`，不要静默更换基线。
3. 检查工作区，但不得擅自暂存、丢弃或覆盖已有改动。
4. 从“按顺序执行的剩余工作”中第一项未完成任务继续。
5. 只有在本文中同时写明持久化架构中的实现位置和测试证据后，才能更新状态。
6. 如果 `dev-latest` 已经提供了更强的实现，应标为“已替代”并记录语义对应
   关系，而不是复制旧代码。

原 `rollout-interface` 分支的提交是行为功能清单；下表是这些功能的权威
核销记录。除非旧分支的功能范围本身发生变化，否则无需重新逐提交猜测迁移
范围。

## 按顺序执行的剩余工作

- [x] **持久化运行统计与导出**：权威的终态统计与 rollout 结果在同一次
  数据库事务中持久化，再由独立导出器确定、幂等地重建 JSONL 文件。v2
  轨迹可以保留精确 token 统计；v3 只有清洗后的消息，不能伪造 token 统计。
  测试重复导出、稳定排序、
  服务重启和并发导出。实现位于 `rl_driver.profiling`；迁移相关全量回归
  236 passed、43 skipped（缺少隔离 PostgreSQL）。
- [x] **补齐 SWE-rebench patch 语义**：执行前持久化仓库原有的 untracked
  文件基线，从 agent patch 中排除这些文件，并在 durable result 中保留
  patch hash、大小、文件数、行数和 verifier 耗时。root 在 agent 启动前
  采集基线，retry/branch 从来源 attempt 的持久化事件继承；grade job
  同时冻结 workdir/base commit/baseline，并使用独立 Git index 提取 patch，
  不污染评分 worktree。迁移相关回归 236 passed、43 skipped（缺少隔离
  PostgreSQL）。
- [x] **闭合 checkpoint/branch 协议链路**：v3 将 N-1 个 Miles sample slots
  延迟消费，策略从 Run Store recovery point 创建 child；恢复点同时校验
  AgentENV snapshot、Claude native prefix 和 Miles SessionTree model position，
  导出保留 parent/child lineage 与 sample-slot 身份。Ash 专项 83 passed。
- [x] **Miles v3 importer**：同一 `AshRolloutFn` 可切换 v2/v3；v3 对清洗后的
  消息重新 tokenize，重建 assistant-only loss mask 和渲染后的 branch token
  前缀，禁止 rollout log-prob/跳过 actor forward。Miles 官方镜像专项
  66 passed；真实 optimizer step 仍由下方全链路验收覆盖。
- [x] **PostgreSQL 持久化语义测试**：在隔离的 PostgreSQL 实例中运行当前跳过的
  Run Store 测试，覆盖并发 claim、lease fencing、取消、崩溃恢复和服务
  重启。不得使用生产数据库。2026-09-15 在临时数据库 schema 上运行
  `runstore/tests/test_store.py` 与 `rl_driver/tests/test_http_postgres.py`，
  17 passed。
- [x] **真实全链路验收**：运行 Claude → Miles SessionTree → AgentENV
  checkpoint/fork → parent/child trajectory 导入 → GRPO/Megatron optimizer
  step → 推理权重更新。2026-09-15 的 Qwen3.8-27B 单步运行返回 2 个
  sample、1 个 branch、4 次模型调用和 1 次工具调用；actor forward 和
  backward 完成，`grad_norm=15.815799713134766`，推理权重版本从 1 更新到 2。
- [x] **公开仓库最终审计**：核销表中每一行，更新陈旧 README/配置，并移除
  个人工作站路径、集群或镜像仓库信息、kubeconfig 和凭据。

## 按旧分支提交核销

| 原提交 | 需要保留的行为 | 持久化架构中的实现位置 | 状态与证据 |
| --- | --- | --- | --- |
| `d623166`, `9ad0a68`, `77952a9`, `7ed8c6d` | 与策略无关、严格且幂等的 rollout-group wire contract | `rl_driver.protocol`、`rl_driver.server`、`rl_driver.ledger`、`rl_driver.driver` | **已迁移**：driver、协议和 HTTP 测试 |
| `793ffdd`, `e232f1f` | 整个 group 共用一个有限 wall-time deadline，执行端能够感知 | adapter deadline、`harness.rollout.remaining_timeout`、Run Store worker dispatch | **已迁移**：rollout control 和 adapter 测试 |
| `e1f9a71`, `65f3073` | 已分配的 sample slot 都会执行并返回终态叶子 | execution group 与 Run Store job | **已替代**：可重启的 job/attempt 模型替代进程内 strategy runner |
| `63c6685`, `d25842b`, `7988319`, `6c98ea7` | agent 的模型调用经过 Miles SessionTree，并导出精确模型记录 | v2 `rl_driver.miles`、`RolloutControls`、Miles session endpoint | **已迁移/已验证**：精确 token/span/version 测试及 2026-09-15 真实 E2E 均通过 |
| `b2b38dd` | model/tool 次数限制可以有限或不设上限，但 wall time 必须有限 | `RolloutBudget`、v3 `max_turns`、`RolloutControls` | **已迁移**：null 和有限预算测试 |
| `ca7cac6`, `8a7f23c`, `0b12b2a` | 可部署、由 AgentENV 支撑的 rollout 服务 | Run Store API/worker、harness orchestrator、RL driver HTTP 服务 | **已替代**：可持久化、可恢复的多进程服务；部署文档仍需最终审查 |
| `ccfa1df` | 明确 sandbox/checkpoint 的所有权和清理责任 | `ResourceLedger`、checkpoint interceptors、Run Store recovery index 和 worker 生命周期 | **已替代**：生命周期和 checkpoint 测试 |
| `0238b23`, `60ba5ae` | 在完整 tool result 后建立 checkpoint；恢复环境及匹配的 native/model prefix；导出 parent/child lineage | `harness.checkpointing`、`runstore.index`、`runstore.native`、`/v1/jobs/{id}/branch` | **已替代/已完成**：持久化恢复点测试及真实 Miles SessionTree branch E2E 均通过；parent 1 次工具调用、child 0 次 |
| `cc02a73` | 可信 template/snapshot 查询和 digest-pinned OCI 准入 | `rl_driver.environment_catalog.EnvironmentResolver`、harness template builder | **已迁移**：两套 adapter 和 template 测试 |
| `317382b` | Claude 原生执行框架、Ash MCP 工具、SWE-rebench 任务/奖励路径和运行统计输出 | 原生 `claude-code` 槽位、Run Store 评分/结果、`rl_driver.profiling` | **已迁移**：执行框架已由更强的原生实现替代；运行统计与 SWE-rebench patch 元数据均已持久化并有回归测试 |
| `6041971` | 常量大小的实时 call/token/time 进度 | gateway → rollout controls → child progress file → worker heartbeat → driver aggregation | **已迁移**：rollout、worker、watchdog 和 adapter 测试 |

## 跨组件接口与语义核销

| 能力 | 持久化架构中的实现位置 | 状态 | 验证证据或剩余缺口 |
| --- | --- | --- | --- |
| prompt-group 身份、sample slots、`max_samples` 和 `minimum_returned_samples` | `rl_driver.protocol`、`rl_driver.message_protocol`、adapters | **已迁移** | adapter 和协议测试 |
| fresh prompt 的结构化映射 | `rl_driver.miles._native_prompt`、native slot 配置 | **v2 Claude 已迁移** | system/developer preamble 使用 Claude preset `append`；最终 user 是 query；assistant/tool 历史只有作为已验证 native prefix 才能恢复 |
| prompt-token 对齐关系 | v2 trajectory 的 `prompt_token_alignment`、Miles importer | **已迁移** | Claude 声明 `harness_rendered`；文本 Codex 保持 `request_exact`；未知值 fail closed |
| 精确 token IDs、generated spans、log-probs 和实际 weight version | Miles SessionTree records 和 v2 trajectory export | **已迁移** | `test_miles.py` 和 Miles importer 测试 |
| 去 hint 的 message-only trajectory 与 Ash 侧 reward | v3 adapter、`runstore.message_export`、`runstore.grading` | **已保留** | message/export/grading 测试；Miles v3 importer 已接入统一 `AshRolloutFn` |
| 运行中 job 的取消 | driver cancel intent、Run Store worker/watchdog、SDK stream、sandbox teardown | **已迁移** | DELETE 仅确认取消意图；worker 完成清理前 GET 保持 `running/cancelling`，之后才成为 `cancelled` |
| 实时进度 | child progress receipt、worker heartbeat、driver aggregation | **已迁移** | 常量大小进度包含 group calls 和当前 trajectory token counters |
| SWE-rebench task/environment 精确绑定 | `rl_driver.tasks.resolve_task`、冻结的 grader dataset | **已迁移** | 部署侧 task catalog 在入队前绑定 `task_id`、不可变 `environment_ref`、仓库预检和同 ID grader |
| rollout 前的 base commit 和 `/testbed` 准备 | `Orchestrator._prepare_repository` | **已迁移** | sandbox 创建后、agent 启动前，以封闭 schema 校验真实 Git HEAD，并创建或验证 `/testbed`；不接受调用方任意 setup command |
| 感知 baseline untracked 文件的 patch 捕获 | `harness.orchestrator`、Run Store events/worker、`rl_driver.driver`、`runstore.swerebench` | **已迁移** | root 基线在 agent 前持久化，branch/retry 继承；SWE-rebench grade 冻结仓库身份并用隔离 index 导出 binary patch |
| hidden verifier、官方 parser 和二值 reward | final snapshot 上的隔离 grade job | **已迁移** | 强制使用冻结 dataset checksum、parser checksum、hidden patch、expected tests 和 network policy |
| verifier 耗时与 patch/reward 元数据 | 持久化评分结果与 `rl_driver.profiling` | **已迁移** | 保存 verifier duration、patch SHA-256、字符数、文件数、变更行数、新增路径和 reward |
| 可恢复的运行统计及 JSONL | driver 账本中随结果保存的 `profiling_records`、`rl_driver.profiling` 导出器 | **已迁移** | v2 使用精确 token/span；v3 缺少权威 token 的字段为 null；测试覆盖重复导出、稳定排序、重启和并发导出器 |
| 存储监督、离线 GC 和 VM 轮换 | 部署工具 | **外部事项** | 除非形成 backend-neutral lifecycle API，否则不进入公开库 |
| v3 trajectory 导入和 GRPO 消费 | Miles 仓库 | **已迁移** | cleaned-message tokenize、assistant-only mask、lineage 和 actor-forward 约束已通过专项；真实 GRPO 完成 backward 与 optimizer step |
| 真实全链路验收 | Ash + Miles + AgentENV + Megatron | **已完成** | 2026-09-15 完成 `Claude → Miles SessionTree → AgentENV checkpoint/fork → trajectory import → GRPO optimizer step → SGLang weight 1→2` |

## 明确不迁移的行为

旧版“固定选择第一个 checkpoint”的策略不是基础设施能力。持久化架构只
提供 recovery points 和 branch 机制，由训练或搜索策略决定消费哪个点。

旧 SWE-bench agent 的 `_nudge` 行为也不迁移：native agent 的自然 stop 默认
就是终止，除非显式 harness policy 要求继续。

公开仓库的代码和文档不得包含个人工作站路径、集群名、镜像仓库凭据、
kubeconfig 或其他个人部署配置。
