# Miles rollout driver

## 去 hint 的消息训练（v3）

`POST /rollout-groups` 现在也接收 `protocol_version: "ash-rollout-v3"`；
v2 的请求和客户端保持兼容。Miles 使用同一个
`miles.rollout.ash.rollout_fn.AshRolloutFn`，通过
`--ash-rollout-protocol-version ash-rollout-v3` 发送 `task_id`、可信
`environment_ref`、prompt、sample slots、采样参数以及可选的 Miles Session
Server 地址。v3 不发送 prompt token IDs；Ash 返回去 hint 的结构化消息，
Miles 使用当前训练模型的 tokenizer/chat template 重建 token 和 loss mask。

Ash 按 `image_resources` 选择资源，将镜像交给已有 worker 的环境/runtime
准备流程。Prime 数据集的 `prime/primeintellect/...` 名称在 Ash 解析到其
上游 `docker.io/swerebenchv2/...` 镜像。需要配置现有 worker profile 的
microVM、runtime 和推理访问方式，无需由 Miles 准备目录。

`tasks[task_id]` 必须存在并绑定可信环境；其中 `grade` 可选。配置 grader 时，
评分结果随消息轨迹的 `reward` 返回；未配置时返回 `reward: null`，由 Miles
沿用现有 reward hook 评分。
新增 `swe-rebench-v2` grader：`dataset_path`/`dataset_sha256` 固定任务行；
`parser_path` 指向 Prime/SWE-rebench 的日志解析文件，`grader_revision`
为该文件的 `sha256:<digest>`。评分在独立恢复的 snapshot 中进行，只在
评分时恢复测试文件并应用 test_patch，全部 FAIL_TO_PASS/PASS_TO_PASS
通过才返回 resolved=true；基础设施错误不会伪装成零分。

worker 从实际 native history 导出消息，保留工具调用/结果和分支来源。
工具调用沿用 OpenAI wire format：`function.arguments` 是 JSON 字符串；Miles
导入时仅在 chat-template 渲染副本中解码成对象，持久化的原始消息不改写。
v3 分支 continuation prompt 在执行前包上保留标记
`<ash_training_hint>...</ash_training_hint>`；导出只从 user/system/developer
消息删除这些注入内容，assistant/tool 原文不做全局替换。旧的未标记 hint
不能自动可靠删除；不完整工具记录、压缩历史和不支持的内容会明确失败。
未配置 `miles.branch_policy` 时默认仍执行独立样本。配置可信的
`module:callable` 后，第一个 slot 作为 root，其余已由 Miles 分配的 slots
延迟消费；policy 可返回 root/branch/skip，branch 复用 Run Store recovery
point，同时恢复 Claude 原生 prefix、AgentENV snapshot 和经 Miles 验证的
SessionTree 模型位置。`rl_driver.policy.first_available_recovery` 只用于机制
验收，不是正式选点算法或训练策略。

Policy 对每个 deferred slot 只能做三种决定：`root` 将其作为新的
独立 rollout；`branch` 必须指定同组的 `source_sample_slot_id`、可用
`point_id` 和受限的 `overrides`；`skip` 则释放该 slot，不产生
trajectory。协议已能通过 `minimum_returned_samples < max_samples` 表达
部分返回，Ash 在达到下限时可返回 `early_stopped`。当前 Miles GRPO
训练适配器仍要求固定组大小：它把下限设为 `max_samples`，并拒绝
`actual_samples < max_samples`。因此现阶段 `skip` 用于让无法分支的 group
可终止而不是继续训练；支持 `K < N` 训练是 Miles 侧的后续实现项。

v3 原生采样支持 temperature、top_p、top_k、文本 stop 和输出长度；
其他字段明确拒绝。Responses 使用 max_output_tokens，Messages 使用
max_tokens/stop_sequences。v2 的 raw-token detokenization 开关不属于
消息版协议。模型端必须支持 native
agent 使用的 Responses 或 Messages，但无需返回训练 token records。

v3 的次数限制统一为请求顶层的 `max_turns`（正整数），按每条轨迹生效：

```json
{
  "max_turns": 64,
  "budgets": {"max_wall_time_seconds": 3600.0}
}
```

一个 turn 是一次模型调用；该响应触发的工具调用不额外占 turn。
group size=8 时每条都有64 turns，driver 不按组大小分配或摊薄。
工具调用仍统计但没有独立次数上限。v3 不再接受 `max_model_calls` /
`max_tool_calls`，旧字段会明确报错。组 deadline、actor timeout 和
输出 token 上限保持独立。
此调整不影响下文原 v2 的 group call-budget 接口。

v3 到达 turns 或执行时间上限后，先停止 agent，等待工具结束并保存最终
snapshot，再提交原有 grader。返回轨迹标记 `status: "truncated"`，
`stop_reason` 为 `max_turns_reached` 或 `timeout`；reward 来自真实评分。
Run Store 的 job 可以成功产出一个截断 episode，不会把它伪装成 agent
自然完成。失败路径从 journal 的 `session.ref` 恢复原生会话身份；
截断的最后一条未写完整 JSON 记录不进入训练，原始日志保留。

`budgets.max_wall_time_seconds` 约束执行阶段；请求的
`finalization_timeout_seconds`（默认1800）为最终快照、消息导出和评分
留出额外时间。子进程有120秒终止收尾窗口，lease/失联检查继续生效。
执行不确定、无法保存安全快照或原生记录无法构成有效轨迹仍会明确失败，
不将基础设施错误变成零分训练样本。

Miles 对去 hint 消息重新 tokenize 并计算 logprobs；这不等于恢复带 hint
采样时的行为概率。不要给这条训练路径启用 use-rollout-logprobs、跳过
actor forward、TIS 或旧 rollout token replay。完整数据准备示例位于
Miles 的 `examples/swe-rebench-ash/`。Ash `rl_driver.client.Client`
仍是 v2 客户端；Miles 的统一 `AshRolloutClient` 根据请求的
`protocol_version` 校验 v2/v3 响应。

## 原 v2 接口

该服务直接接收原分支的 **`RolloutGroupRequest` / `ash-rollout-v2`**，默认
监听 **127.0.0.1:11001**。Miles 不需要构造 RunSpec；driver 内部将一个
prompt group 转成多个 RunSpec，通过 HTTP 提交给 **127.0.0.1:18110** 的
Run Store。实际执行仍由独立 worker / Orchestrator / native slot 完成。

```text
Miles POST /rollout-groups (ash-rollout-v2)
    → driver: environment lookup + sample allocation + RunSpec translation
    → Run Store :18110 → PostgreSQL → independent workers
    → Orchestrator → Codex/Claude + Ash tools
                     └→ inference gateway → configured model / Miles session
    → recorded session tokens + optional official grade
    → driver GET /rollout-groups/{id} (ash-rollout-v2)
```

协议类型和环境目录复用自 `haixin/rollout-interface` 的提交
`cc02a738b7833e8582440817636767eb7b5605f2`，位于 `protocol.py` 和
`environment_catalog.py`。driver 本身不启动 agent、VM 或训练。

## 启动与配置

Run Store 和 workers 需要已单独启动。在 Ash 根目录运行：

```bash
python3.12 -m pip install -r rl_driver/requirements.txt
export PYTHONPATH=.:sdk
export ASH_RUNSTORE_TOKEN='<existing-runstore-token>'
python3.12 -m rl_driver --config rl_driver/config.example.json
```

替换示例 `miles.environment_catalog.environments[*].spawn_ref`，使其指向
已有的 runtime-ready image/template/snapshot，并确认 `miles.profile` 是
Run Store 已配置的 worker profile。资源规格在 `miles.resources` 中明确
配置。环境请求必须精确匹配 kind/id/revision/resource_profile；未知项在
入队前拒绝。镜像准备继续使用现有执行层，本版不新增动态 OCI builder。

示例在本机监听，`driver_token_env: null` 允许原 Miles 调用方无需新增
Authorization header；也可设为 `ASH_RL_DRIVER_TOKEN`，此时调用方必须携带
对应 Bearer token。Run Store 认证始终保留。`--host`、`--port` 可以修改
监听设置。ledger 相对路径以配置文件目录为基准，默认保存到
`Ash/runs/rl-driver/groups.sqlite3`。

如果推理端需要认证，`miles.api_key_env` 指定 **worker 环境**中的变量名，
并在 worker profile 的 `worker_env` 中以 `$env` 引用传入。模型地址来自
请求，因此执行端不会沿用其他旧 route 的凭据。`run_defaults.routes_file`
仍可提供显式 namespace-flattening 设置，但模型地址由当前请求决定。

## Miles 请求与结果

`group.example.json` 使用原请求格式；示例 token IDs 仅用于展示，真实请求
应由 Miles 生成。模型和环境地址也需替换为实际部署值。

```python
import json
from rl_driver.client import Client

client = Client("http://127.0.0.1:11001")
try:
    with open("rl_driver/group.example.json") as file:
        group_id = client.submit(json.load(file))
    result = client.wait(group_id, timeout_s=1800, interval_s=2)
    print(result["status"], result["trajectories"])
    # After persisting/consuming the result:
    client.release(group_id)
finally:
    client.close()
```

| 原请求字段 | 当前实际处理 |
| --- | --- |
| rollout_job_id / prompt_group_id / rollout_id | 原样保存；外部返回原 ID，内部 ID 与幂等键独立派生 |
| sample_slots / max_samples | 从分配名额中选择前 max_samples 个，各自提交独立 RunSpec |
| task_id | 写入任务上下文；可查部署侧 tasks 映射中的评分模板 |
| environment_ref | 通过可信目录解析为 sandbox_image，并设置对应 CPU/内存 |
| prompt | 文本或单条 user message → RunSpec.prompt |
| prompt_token_ids | 原样保存；训练导出使用推理端记录的真实完整输入，而不重新 tokenize 文本 |
| model / model_endpoint | 设置 RunSpec.model，并由执行侧 gateway 实际路由 |
| session_server_endpoint | worker 创建 Miles session，模型请求路由至其 session 路径，结束时保存 records 后释放 session |
| sampling_params | 原样交给执行契约；支持项在真正发往模型的请求上设置 |
| budgets | group 墙钟 deadline；各 sample 分配的 model/tool admission caps 总和不超过原预算 |
| expected_weight_version / return_rollout_logprobs | 导出时检查观测版本并按需返回 logprobs |
| minimum_returned_samples | 有效 token 轨迹不足时返回 failed，满足下限但少于全部时可返回 early_stopped |

POST 返回原 v2 的 queued/running/terminal acknowledgement；GET 返回
`RolloutGroupResult`，包括 `actual_samples`、`trajectories`、`consumed_budget`。
每条轨迹保留原 `sample_slot_id`。默认采用无分支的独立 sample 策略；
`search_branches=0`，不会暗中用固定 checkpoint 策略替代训练算法。

## 执行端支持范围

这次适配包含 `harness/rollout.py` 的 opt-in 执行控制，并挂到现有
Orchestrator、gateway 和 HTTP tool pipeline：

- 当前 native slot 支持 `codex`（Responses）和 `claude-code`（Messages）。
  `model_endpoint` 或 session 下的 `/v1/responses`、`/v1/messages` 必须真的
  支持相应协议。**只有 Chat Completions 的 Miles Session Server 仍需要
  native API 协议桥**；本次没有声称实现该桥。请求转换不会自动解决它。
- 支持 temperature、top_p、max_new_tokens/max_tokens/max_output_tokens。
  输出长度按 native API 映射。其他采样参数明确在入队前拒绝，不静默忽略。
- 原协议允许的任意多消息历史，目前不能直接灌入 native session；本版
  接受文本和单条 user message，其他历史在入队前明确拒绝。
- 每个 sample 的模型调用和 Ash 工具调用在执行前计数并限制；SDK 的上游
  重试也消耗模型调用额度。为避免 worker attempt 重置 group 额度，v2
  计划的 `max_infra_retries=0`。driver 的幂等 HTTP 重试不会重跑 actor。
- group deadline 从提交起计算，含队列等待。过期后不再提交工作，并请求
  取消排队 job；worker 内的 timer 停止 agent，新模型/工具调用也被拒绝。
  已进行的工具操作仍按现有 drain/清理规则收尾，不承诺瞬时停止所有副作用。

## 训练数据与评分

worker 将 Miles session 的原始 records 写入 Run Store 轨迹事件后再释放
session。driver 用真实 input_ids、output_token_logprobs 和
accumulated_token_ids 导出 generated_spans；原协议校验器验证前缀和位置。
缺失 observed weight_version 不会被填成 expected_weight_version，版本
不匹配也拒绝导出。

**接收原 Miles 请求已经实现；真实训练是否可运行仍取决于实际推理部署。**
仅能返回普通文本、没有 session token records 的执行可以正常跑完，但
v2 GET 会返回 `failed` 和明确的导出原因，绝不编造 token 或伪称训练样本
已完成。普通执行结果始终可从 `/miles-executions/{rollout_job_id}` 查询。

可在部署配置中设置 `miles.tasks[task_id].grade`，其内容沿用内部 grade
JobSpec 模板并省略 snapshot_id。driver 按已有规则从 actor 最终可用恢复点
选取 snapshot，独立排队评分，然后将官方 resolved 布尔值转换为 reward。
没有评分模板时 reward 为 null，评分仍可由 Miles 自己负责。task_id 本身
不会自动安装仓库、建立测试环境或推断评分配置。

## 路径、重启与兼容性

| 路径 | 用途 |
| --- | --- |
| `GET /rollout-environments` | 原 v2 环境发现，隐藏 spawn_ref 和凭据 |
| `POST /rollout-groups` | 原 RolloutGroupRequest |
| `GET /rollout-groups/{id}` | 原 RolloutGroupResult |
| `DELETE /rollout-groups/{id}` | 原 v2 轻量确认；活跃 group 请求取消，后台仍跟踪执行清理 |
| `GET /miles-executions/{id}` | 内部 job/attempt/状态与普通结果 |
| `/execution-groups` | 原自定义格式的内部调试入口，不是 Miles wire API |

同一 group ID/请求重复提交返回原计划；不会因为目录、模型默认值或时间
变化重新规划。结果导出后保存在 driver ledger，重启后无需保留活跃
Session Server 对象。DELETE 保留持久幂等记录和执行数据，不删除 snapshot。
原自定义调用方迁移到 `ExecutionClient` 和 `/execution-groups`；详见
[EXECUTION.md](EXECUTION.md)。`Client` 默认已指向 Miles v2。

## Profiling 导出

终态 trajectory 的 profiling 记录与 v2/v3 result 一起保存在 driver
ledger 中，而不是在任务完成时直接向 JSONL 追加。这保证 driver 在任意位置
崩溃后都可以从持久化结果重新导出，不会出现“result 已保存但统计丢失”或
重复追加同一 trajectory 的情况。

```bash
PYTHONPATH=.:sdk python3.12 -m rl_driver.profiling \
  --ledger runs/rl-driver/groups.sqlite3 \
  --output runs/profile.jsonl
```

exporter 按 `(rollout_job_id, sample_slot_id, branch_id)` 稳定排序、去重，并以
原子替换方式重写目标文件，因此同一个 ledger 可以安全重复导出。v2 记录直接
使用 Miles SessionTree 返回的精确 token IDs 和 generated spans。v3 只返回
去 hint 的消息，没有权威 token 对齐信息，因此 token 统计字段明确为 `null`，
不会通过文本长度或重新 tokenize 猜测；model/tool calls、耗时、reward、状态
和身份等有权威来源的字段仍会保留。

## 验证

```bash
PYTHONPATH=.:sdk python3.12 -m pytest \
  rl_driver/tests/test_driver.py rl_driver/tests/test_miles.py \
  harness/tests/test_rollout_contract.py -q
```

设置独立的 `ASH_RUNSTORE_TEST_DSN` 后，`rl_driver/tests/test_http_postgres.py`
还验证真实 TCP → Run Store API → PostgreSQL → Orchestrator → 测试用
native/session endpoints → 原 v2 trajectory。模型、工具、snapshot 验证器
使用受控 fixture；没有运行 GPU 训练或真实 VM。未设置 DSN 时这些测试会
明确 skip。原始请求 fixture 来自 pinned haixin 分支的 HTTP e2e 测试。
