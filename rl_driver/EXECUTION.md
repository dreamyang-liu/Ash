> 内部执行调试接口。Miles 入口已由 [README.md](README.md) 中的原 v2 协议接管；下文自定义格式只用于 `/execution-groups`。

# RL driver

独立的 RL 调度入口，默认监听 **127.0.0.1:11001**，通过 HTTP 连接
**127.0.0.1:18110** 的 Run Store。实际 agent、工具、VM、checkpoint 和
grader 仍由已有 worker / Orchestrator 执行。

```text
RL caller / trainer
        │ POST /execution-groups
        ▼
rl_driver :11001        group/sample/job 映射、提交、轮询、结果聚合
        │ HTTP
        ▼
runstore :18110         PostgreSQL jobs / attempts / events / recovery points
        ▲
        │ claim / heartbeat / result
independent workers    RunSpec → Orchestrator → Codex / Claude → AgentENV
                       GradeSpec → official verifier
```

## 首版边界

本版协议为 **`ash-runstore-driver-v1`**。保留原 rollout 服务的端口和
`/execution-groups` 路径；请求内容直接描述 RunSpec/branch 和可选 GradeSpec，
结果是执行记录及轨迹查询接口。

**这不是原分支的 `ash-rollout-v2` token 轨迹协议。** 旧 Miles 请求会返回
422，避免把普通消息当成可训练 token 样本。当前执行层还未提供完整真实
token IDs、generated spans、logprobs、observed weight version；后续 Miles
adapter 应将推理端记录与本 driver 返回的 job/attempt/sample 身份关联。

- 支持一次提交多个独立 sample，或从指定 Run Store recovery point 提交分支。
- 可在 actor 成功后自动提交独立 GradeSpec，绑定最终完整工具边界的可用
  recovery point。无法证明最终边界时报告失败，不用较早 snapshot 评分。
- 不选择分支算法、不计算 advantage/loss mask、不切换训练权重。
- `RunSpec.timeout_s` 等现有参数正常传递。没有添加 group model/tool-call
  硬限额、旧版 `sampling_params` 或动态 `model_endpoint` 参数；模型路由由
 现有 worker profile / RunSpec gateway 配置决定。
- SQLite 只保存 driver 的提交意图、group/sample/job 映射、结果副本和消费
  收据。执行任务的权威状态、lease 和重试仍在 Run Store PostgreSQL。
- 一个 ledger 只允许一个 driver 进程。可有多个 HTTP 调用方；执行并发由
  Run Store workers 决定。保留 ledger 与 Run Store DB/关联 artifacts。

## 启动

从 Ash 根目录运行；Run Store 与 workers 需要已单独启动。

```bash
python3.12 -m pip install -r rl_driver/requirements.txt
export PYTHONPATH=.:sdk
# 使用运行中 Run Store 的控制面 token：
export ASH_RUNSTORE_TOKEN='<existing-runstore-token>'
# caller 与 driver 共用此 driver token：
export ASH_RL_DRIVER_TOKEN='<driver-token>'
python3.12 -m rl_driver --config rl_driver/config.example.json
```

默认端口为 11001；可用 `--host` / `--port` 修改。配置里的 ledger 相对路径
以**配置文件所在目录**为基准；示例保存到 `Ash/runs/rl-driver/groups.sqlite3`。
两种 token 仅从环境读取，不保存在 ledger 或配置内容中。

启动 driver 不会启动 Run Store、worker、模型服务器或训练。关闭 driver
也不会停止已提交的任务；重启后继续查询相同 job IDs。

## 提交普通 rollout

`execution-group.example.json` 是完整请求模板。替换 prepared image、model 和
worker profile 后提交；profile 使用 Run Store 里的名称。

```python
import json
import os
from rl_driver.client import ExecutionClient as Client

client = Client("http://127.0.0.1:11001", os.environ["ASH_RL_DRIVER_TOKEN"])
try:
    with open("rl_driver/execution-group.example.json") as file:
        group_id = client.submit(json.load(file))
    result = client.wait(group_id, timeout_s=3600, interval_s=2)
    print(result["status"], result["samples"])
    # only after the caller has durably consumed the result:
    if result["ready"]:
        client.release(group_id)
finally:
    client.close()
```

`rollout_job_id` 是 group 的幂等身份：相同 ID/请求重复提交返回原 group；
同 ID/不同请求返回 409。各 sample 必须有不同 `sample_slot_id`。每个 sample
包含 `run` 或 `branch` 之一；`run` 是 `JobSpec(kind="rollout", spec=RunSpec)`。
正常运行的基础设施重试由 JobSpec.max_infra_retries 和现有 worker 管理。

group 的 `context` 可存 experiment、训练 step 等信息。普通 rollout 会将
这些信息放入 Run Store job 的 `context.rl_driver`；branch 保持源 JobSpec
上下文，新的 group/sample 关联保存在 driver ledger。

## 评分和分支

在 sample 内添加 `grade`，其结构是 grade JobSpec 模板，`spec` 使用现有
GradeSpec 字段，**省略 snapshot_id**：

```json
{
  "profile": "grade",
  "spec": {
    "benchmark": "swebench-verified",
    "instance_id": "sympy__sympy-13091",
    "dataset_path": "/worker/frozen-tasks.json",
    "dataset_sha256": "REPLACE_WITH_DATASET_SHA256",
    "grader_revision": "3.0.15",
    "timeout_s": 600
  }
}
```

dataset 路径及 backend 等配置必须在 **worker** 上有效；profile 可以提供
`grade_defaults`。driver 从 actor 最终 attempt 的工具记录与 recovery points
选择最终 snapshot，固定提交后重试 HTTP 不会另选 snapshot。
`resolved: false` 是评分任务成功完成的有效结果，不会触发重跑 actor。

分支先查询恢复点，再用一个新的 group ID 提交：

```python
parent = client.get("experiment-1-group-0")
actor_job = parent["samples"][0]["actor"]["job_id"]
points = client.recovery_points("experiment-1-group-0", "sample-0")
point = next(p for p in points if p["available"])
client.submit({
    "rollout_job_id": "experiment-1-group-0-branch-1",
    "prompt_group_id": "task-0",
    "samples": [{
        "sample_slot_id": "branch-1",
        "branch": {
            "job_id": actor_job,
            "point_id": point["id"],
            "overrides": {"prompt": "Continue from this state; try another fix."}
        }
    }]
})
```

恢复点由调用方的 RL 策略选择。Run Store 在提交及执行时验证原生前缀与
snapshot。driver 不直接恢复 VM、不复制原生会话。原有 branch 限制仍有效：
只能改 prompt、model、timeout_s、budget_usd，保留环境、slot 和工具配置。

## 接口与状态

所有接口要求 `Authorization: Bearer <ASH_RL_DRIVER_TOKEN>`。

| 接口 | 行为 |
| --- | --- |
| `GET /health` | 协议版本、driver 角色及 token export 能力 |
| `POST /execution-groups` | 校验并保存 group；返回 202 和 ID/状态 |
| `GET /execution-groups/{id}` | 完整 group 状态和已取得的 actor/grade 结果 |
| `GET /execution-groups/{id}/result` | 同上，可重复读取 |
| `GET /execution-groups/{id}/samples/{sample}/events?after=N` | 代理 Run Store 分页轨迹；下一页传最后一个 seq |
| `GET /execution-groups/{id}/samples/{sample}/recovery-points` | 返回关联 attempt 的恢复点和可用性 |
| `DELETE /execution-groups/{id}` | 活跃 group 请求取消；终态 group 确认消费 |
| `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` | 关联丢失回包的已知 job；见下文 |

group 状态：`queued`、`running`、`quarantined`、`cancelling`、`completed`、
`failed`、`cancelled`。只有 `ready: true` 才是终态；有一个 sample 的 actor
或 grader 失败，group 就为 failed，其他完成结果仍保留。quarantined 是
需要处理的执行状态，不是假装完成；`Client.wait` 会返回它供调用方判断。

每个 actor/grade 结果包含 job_id、attempt_id、执行 state/phase、result/error。
轨迹接口使用这个 attempt_id，避免将不同 attempt 的事件混到一起。结果
中的 artifact 路径属于执行服务所在主机，不是 driver 可直接下载的文件。

## 重启、取消和消费

- HTTP 提交前保存意图；断线/丢失回包后，用稳定幂等键重新查询提交结果。
  driver 不因执行失败自建另一个 actor attempt。
- DELETE 活跃 group 返回 202/cancelling。尚未提交的 sample 停止提交，
  已排队 job 请求取消；running job 继续跟踪到终态。Run Store 目前没有
  running-job 中断 API，因此本版没有即时中断保证。
- 若取消时还缺少一次提交的回包，driver 不会为“查找结果”重发可能创建新
  job 的 POST。状态保留对应 idempotency_key。操作方在 Run Store 查到该
  key 的 job 后，可以调用 `client.reconcile_submission(group, sample,
  phase="actor", job_id=...)`。driver 先向 Run Store 验证 key，再恢复跟踪；
  此操作不会创建新任务。`phase` 也可以是 `grade`。
- DELETE 终态 group 记录 `acknowledged_at`，**不删除**执行数据、snapshot
  或幂等记录。调用方应在持久保存/消费结果后确认，并按 group/sample 身份
  去重；driver 的收据不能替代 trainer 自身的消费事务。
- 没有自动 TTL/GC。保留 group 历史便于重启和排查；ledger 绑定 Run Store
  URL，防止将已有映射意外指向另一个控制面。

## 验证

快速测试不创建服务或 VM：

```bash
PYTHONPATH=.:sdk python3.12 -m pytest rl_driver/tests/test_driver.py -q
```

真实 TCP + PostgreSQL 测试使用独立测试数据库，每个测试创建唯一 schema：

```bash
ASH_RUNSTORE_TEST_DSN='host=127.0.0.1 port=TEST_PORT user=postgres dbname=postgres' \
  PYTHONPATH=.:sdk python3.12 -m pytest rl_driver/tests -q
```

测试覆盖 driver → Run Store API → PostgreSQL、受控 worker claim/result、
grade 最终 snapshot、branch 提交、重启和消费收据。它们不运行模型、官方
verifier 或真实 VM；原生恢复/执行正确性由现有 Run Store 测试另行覆盖。
没有 DSN 时 PostgreSQL 测试会明确 skip，不能把 skip 当作接线通过。
