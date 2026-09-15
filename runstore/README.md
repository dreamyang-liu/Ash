# Run Store v1

An opt-in, authenticated HTTP control plane backed by PostgreSQL. One worker
manager claims jobs with `FOR UPDATE SKIP LOCKED` and supervises concurrent attempts using the existing
`harness.orchestrator.run.Orchestrator`. Existing eval commands and historical
results are unchanged. There is deliberately no history-import command.

## Boundaries

- One trusted operator, one host, shared durable artifact directory and database.
  This is not a multi-tenant or multi-host scheduler. Keep the HTTP listener local
  or behind an authenticated private proxy; clients can submit executable tasks.
- Rollouts use owned AgentENV microVMs and HTTP tool transport. Workers assign
  unique run IDs, journals and neutral per-attempt working directories. Live
  sessions, existing sandbox bindings, custom MCP processes and `codex-cli` are
  rejected. Codex SDK and Claude Code are the supported native adapters.
- Tool calls/results and physical captures retain tool granularity. Public
  recovery points require a **closed model response, all its tool results, an
  exact snapshot and a checksum-pinned native prefix**. A response with two tools
  exposes its end, not the halfway point. Missing/unsafe captures are unavailable.
- Disk-only snapshots do not restore processes, memory, extra mounts or external
  side effects. A native prefix is not a process-memory checkpoint.
- PostgreSQL holds requests, frozen attempt payloads, status, journal events, tool responses,
  prefix nodes and recovery metadata. Native logs, patch/test artifacts and VM
  layers stay on disk. Prefix references use a shared file plus byte offset/hash;
  they do not copy the whole transcript once per tool.
- Native compaction/rollback or an unproven boundary fails closed in v1: tool
  records remain queryable, but later messages may not expose recovery points.
  Earlier checksum-valid prefix views remain available. This does not promise
  every native log/version is resumable at every message.
- No automatic tool-response cache execution, relaxed reuse, KV reuse, GC or
  historical migration. Retain native logs, referenced outputs and snapshot layers
  for as long as their recovery points are needed. Back up DB and artifacts together.

## Start

Provision a **dedicated PostgreSQL database with persistent storage**. Do not use
an ephemeral/tmpfs test database for actual runs. Run these from the Ash checkout:

```bash
python3.12 -m pip install -r runstore/requirements.txt
export PYTHONPATH="$PWD:$PWD/sdk"
export ASH_RUNSTORE_DSN='host=127.0.0.1 dbname=ash_runstore user=ash_runstore'
export ASH_RUNSTORE_TOKEN="$(openssl rand -hex 32)"
python3.12 -m runstore init --config /path/to/worker-config.json
python3.12 -m runstore serve --config /path/to/worker-config.json
```

Use `.pgpass` or your normal secret manager for the database password. The API
defaults to `127.0.0.1:18110`. `config.example.json` shows this machine's paths;
adjust them before use. Copy the Codex profile's backend/runtime defaults into
the Claude profile when using the same environment. The grade profile accepts
those settings in `GradeSpec.backend` instead.

Start one worker manager, with the same DSN and config:

```bash
python3.12 -m runstore worker --config /path/to/worker-config.json --concurrency 24
```

`--concurrency` defaults to 1. One manager maintains a single claim loop and at
most N active attempt supervisors, filling a free slot when one finishes. Each
attempt still runs in its own child process; no new broker or in-memory task queue
replaces PostgreSQL. You do not need to launch N worker CLI processes. Additional
managers on the same host remain fenced by the existing DB locks and leases.
`--once` still claims at most one new job, even with concurrency greater than 1.
Concurrent cold starts of the same content-addressed image may both stage the
runtime, but only one template build wins the alias; the other waits for that
build to become ready. Its unused staging snapshot remains until repository GC
because AgentENV does not expose snapshot deletion. Prebuilding images/templates
still avoids duplicate staging work. If the winning build reports failure,
waiters fail explicitly; if it never settles, they fail at the configured build
timeout. Avoiding duplicate staging and automatic takeover require an
AgentENV-native create-or-join operation and are not implemented here.
The controller can use Python 3.12 while `profile.python` selects Python 3.11 for the
Codex SDK child. Install `openai-codex` in the Codex interpreter and
`claude-agent-sdk` in the Claude interpreter, plus each interpreter's normal Ash
dependencies. Codex's experimental dynamic-tool/native-history integration was
validated with SDK/bundled CLI **0.147.0**; revalidate before upgrading it.

Owned rollout VMs use `max(explicit microvm.sandbox_ttl, ceil(RunSpec.timeout_s)
+ 600)` seconds of VM lifetime. For example, a 1,800-second rollout gets at least
2,400 seconds, and a longer explicit TTL is preserved. The common Orchestrator
applies this for both HTTP and stdio transport, including new runs from snapshots.
This extends only the VM lease, not the actor's execution budget. Externally owned
sessions/servers remain their owner's responsibility; generic pool defaults do not change.

The exposed `shell` tool defaults to 300 seconds when no timeout is supplied.
It currently has no separate fixed upper clamp on an explicit timeout such as
900 or 1,800 seconds. A longer tool timeout does not extend the rollout budget;
the outer supervisor can stop the run first, with in-flight work drained/cleaned up.
GatewayBackend's 360-second timeout is HTTP read inactivity, not a total
tool-call limit; the runtime sends heartbeat bytes every 10 seconds while alive.
Custom-tool manifests have a separate 600-second maximum. `web_fetch` caps its
timeout at 60 seconds and `wait_for_events` at 300 seconds.

Profile `env` is explicitly forwarded to the agent; `worker_env` is only for the
worker-side gateway. Credential values are not accepted in requests/profiles:
use `{"$env": "HOST_VARIABLE"}`. Backend credentials and worker-only secret
variables are blanked in the agent environment. Do not put secrets in prompts,
tool commands or opaque config strings: arbitrary text cannot be safely inferred
to be a secret. Agent-auth credentials explicitly named in `env` are an exception
to agent isolation, necessary for a direct provider connection.

For Codex using Bedrock, submit its normal `extra.config_overrides` selecting
`amazon-bedrock`; the example below demonstrates that route. For a Responses
gateway, set `use_gateway` and `routes_file` on RunSpec and put the upstream key
reference in `worker_env`. The orchestrator supplies Codex a custom gateway
provider and short-lived slot token. Gateway route files use `api_key_env`;
namespace flattening remains opt-in for providers that require it.

## Submit and observe

```python
import os
from runstore.client import Client
from runstore.specs import JobSpec

client = Client("http://127.0.0.1:18110", os.environ["ASH_RUNSTORE_TOKEN"])
job_id = client.submit(JobSpec(
    kind="rollout", profile="codex",
    context={"benchmark": "my-benchmark", "task_id": "task-1"},
    spec={
        "prompt": "The task description...",
        "slot": "codex", "model": "openai.gpt-5.6-luna",
        "sandbox_image": "my-prepared-image-or-snapshot",
        "timeout_s": 3600,
        "extra": {"config_overrides": {"model_provider": '"amazon-bedrock"'}},
    },
), idempotency_key="experiment-1/task-1/parent")
print(client.get(job_id))
print(client.wait(job_id))
```

Repeating the same key/request returns the same job; a different request with
the same key returns 409. Profile defaults and their fingerprint are frozen at
HTTP submission. Changes to a profile are not silently applied to an old job.
`wait()` timeout only stops waiting; it does not cancel execution or consume the
durable result. A quarantined job is returned for inspection, not hidden forever.

REST endpoints (all require `Authorization: Bearer ...`):

| Endpoint | Result |
|---|---|
| `GET /health` | Authenticated API and fresh PostgreSQL-transaction readiness |
| `POST /v1/jobs` + `Idempotency-Key` | Accepted job, immediately |
| `GET /v1/jobs?state=running` | Job states/phases |
| `GET /v1/jobs/{id}` | Frozen request, lease, state, result/error |
| `GET /v1/jobs/{id}/attempts` | Attempts, heartbeat progress, artifact directory, query scope |
| `GET /v1/jobs/{id}/events?after=0` | Paginated full journal/trajectory events |
| `GET /v1/jobs/{id}/tools?after=0` | Ordered tool calls and responses |
| `GET /v1/jobs/{id}/recovery-points` | Message boundaries and snapshot/native pairs |
| `GET /v1/jobs/{id}/result` | Repeatably readable completion/result |
| `POST /v1/jobs/{id}/branch` | New queued branch from `point_id` |
| `POST /v1/jobs/{id}/cancel` | Cancel queued jobs immediately or persist cancellation intent for running jobs |
| `POST /v1/prefix/query` | Historical prefix matches and ancestor fallback |

Events/tools/recovery queries accept `attempt_id`; otherwise they select the
latest attempt. Event/tool cursors are the last returned `seq`/`depth`, respectively.
Job-list queries are bounded to the most recent 1,000 jobs in v1.

## Resume, branch and prefix lookup

```python
points = client.recovery_points(job_id)
point = next(point for point in points if point["available"])
branch_id = client.branch(
    job_id, point["id"], idempotency_key="experiment-1/task-1/branch-1",
    prompt="Continue from this state; investigate the parser next.",
    timeout_s=1800,
)
```

A branch is a fresh queued RunSpec, not mutation of its parent. In v1 it keeps
the source slot, environment/tool configuration and profile; only prompt, model
and budgets may change. Claude registers a separate native prefix session;
Codex forks a copied native prefix path. Neither adapter silently substitutes
the full parent conversation if an exact cut is unavailable.

Codex's identity-preserving facade uses app-server dynamic tools in namespace
`ash`, forwarding **only the supplied MCP panel** to the same Ash executor. This
gets the real native `callId` before execution without guessing from identical
arguments. Native MCP dispatch is disabled in that mode; native shell/edit/search
remain disabled. Physical capture stays inside the executor's serialization gate.

For prefix queries, get the exact `scope` from an attempt's `execution` record,
then pass ordered `{"name": ..., "arguments": ...}` calls. The reply distinguishes
`matched_depth` from `recovery_depth`. A prefix inside a two-tool message may
return historical responses, but its fallback is the **previous** closed message,
never the following snapshot. `replay_suffix` starts at that older recovery depth.
Occurrences from different attempts are returned separately, never spliced.

Native checksums and current AgentENV snapshot metadata are rechecked. Set
`snapshot_repository` for same-host manifest/commit checks as well. These are
availability checks, not a scrub of every disk layer; the actual VM restore is
still authoritative. A later invalidated message boundary revokes its published
point. A missing snapshot/prefix falls back to an earlier valid matching ancestor.

## Failure handling

`queued -> running -> succeeded|failed`; uncertain ownership/cleanup goes through
`quarantined`. An expired lease **never directly requeues execution**. Workers
fence the old token, identify the owned process by boot ID/PID/start time, stop
its process group, and reconcile its owned sandboxes before publication/retry.
A child cannot allocate anything until its process identity is durably recorded
and its stdin payload is delivered. Unknown allocation windows remain quarantined.

Worker freezes the effective spec, selected recovery point and profile config in
`rs_attempts.payload` with `payload_hash`, under the live attempt lease. This
payload is immutable through the Store API; heartbeat metadata lives separately
in `execution`. After spawning a child, Worker commits its process identity, then
sends the frozen JSON over stdin and closes the pipe. Only the artifact directory
and checksum are passed on argv, never the prompt or credential values. Payloads
are limited to 16 MiB; handoff has a bounded write deadline and the child rejects
missing, truncated, oversized, mismatched or late input before execution.

The child never queries PostgreSQL. Worker ingests using the exact frozen payload
and reconciliation reloads it from PostgreSQL, not current profile defaults or
`request.json`. New attempts do not create `request.json` or `permit.json`; old
files are left untouched and are never consulted by the new handoff path. Large
native logs, resource receipts and outcomes still use the shared artifact directory.

### Large journal events

Session-state events can contain a complete trajectory tree under
`rollout.session_state.state.metadata.tree_records`. For long agent runs, a single
canonical JSON event can be hundreds of MiB. Writing that object directly as
PostgreSQL `jsonb` can exceed PostgreSQL's roughly 256 MiB JSONB object-element
limit or the Run Store's normal five-second statement timeout.

Run Store therefore encodes canonical JSON events larger than 1 MiB with zlib and
Base64 before storing them in `rs_events.event`. The database wrapper is identified
by `encoding: "ash.runstore.zlib-json-v1"`; event reads transparently restore the
original object. Idempotency/conflict checks also compare restored events, so this
is a storage representation change rather than an API or journal-schema change.
Older uncompressed rows remain readable, and an application event is decoded only
when it has exactly the complete wrapper shape. Artifact files such as
`trajectory.jsonl` remain uncompressed and retain their original full content.
Compression reduces typical repeated SessionTree payloads but is not a hard
upper bound: an incompressible event can still approach PostgreSQL's JSONB limit.
Such an event fails explicitly; moving large opaque state to content-addressed
artifact storage would require a later schema change.

Encoding is completed before the transaction takes the job row lock, so CPU time
spent serializing a large tree does not block cancellation or heartbeat updates.
Two representative single-event measurements were 204.7 MB to 45.7 MB and
143.7 MB to 29.7 MB (approximately 78% and 79% smaller). These figures compare the
same PostgreSQL event before and after encoding; they are not complete trajectory
artifact sizes. Because compressed tree events may still be tens of MiB,
`append_events()` uses a 60-second local statement timeout. Cancellation uses a
20-second local timeout so it may wait behind a bounded in-flight write while
remaining below the client's 30-second deadline. Other Store operations retain
the five-second default, so these exceptional paths do not weaken general
control-plane failure detection.

On upgrade, stop new admission and drain old workers first, then add the nullable
payload columns:

```bash
python3.12 -m runstore init --config /path/to/worker-config.json
```

Restart both the Run Store API and workers so both processes load the same Python
revision; replacing source files does not hot-reload an existing process. Then
resume admission. Existing jobs/results are preserved. Legacy attempts without a
DB payload remain queryable but are not automatically reconciled by the new worker;
they stay quarantined if interrupted and need a separately scoped migration.
New queued attempts acquire their DB payload normally. A corrupt DB payload also
fails closed, with no fallback to a local handoff file or recomputed settings.

The manager only tracks occupied slots, claims jobs and collects completions.
Blocking preparation, indexing, cleanup and result publication run in per-attempt
supervision threads, never in its dispatch loop. A recovery exception is isolated
to that job; unresolved expired recoveries have a 30-second dispatch cooldown.
Queue/database errors back off exponentially, up to 30 seconds. Expiry scans skip
locked rows rather than waiting behind another attempt's transaction. Store
connections set a 5-second connect timeout, 5-second SQL statement timeout and
TCP keepalive/user-timeout settings; these are not a universal wall-clock I/O bound.

Each active attempt also has a lightweight watchdog that does not read its journal
or query PostgreSQL. It stops the owned child process group on run timeout, lease
loss or manager shutdown, even if that attempt's supervision thread is blocked.
Other attempts continue their own heartbeats. This deliberately does not keep
renewing a stalled attempt's lease forever. A still-blocked thread continues to
occupy a concurrency slot until it returns: Python threads cannot safely be killed,
and releasing that slot early would permit unbounded accumulation of stuck work.
If every slot is blocked, dispatch stops until recovery or manager restart.

SIGINT/SIGTERM stops new dispatch and requests shutdown of all owned attempts.
The manager waits up to 20 seconds for supervisors; unresolved work is left for
lease-expiry reconciliation after restart, not blindly replayed. A manager crash
still removes supervision of all its attempts; shared DB/host failures can affect
all tasks. This is bounded fault handling, not a guarantee against every outage.

Known expired attempts are reconciled by polling workers. Other quarantines
require investigation; explicit reconciliation is available with:

```bash
python3.12 -m runstore reconcile --config /path/to/worker-config.json --job-id JOB_ID
```

Infrastructure retries are capped at two additional attempts, with 30/60-second
backoff. Actor retry requires a valid complete-message continuation and remaining
wall-clock budget; uncertain shell calls are never blindly replayed. Cost-limited
retries also require measurable remaining cost; unknown/unpriced cost disables
that retry. Rollout timeout covers child startup and actor execution. Grader
`timeout_s` is the official test timeout, with an additional bounded 1,200-second
worker allowance for preparation/collection. An unresolved grade is a successful
grading job with `resolved: false`, never an infrastructure retry. A grader-only
failure requeues only that grading job, not the actor.

## Official grading

Queue `JobSpec(kind="grade", spec=asdict(GradeSpec(...)), profile="grade")`.
`GradeSpec` pins a dataset JSON/JSONL file by SHA256, instance ID, submission
snapshot, official grader revision, resources, timeout and separate verifier
network policy. `context` can record the actor job ID for application-level linkage.

- Verified invokes the installed **official `swebench` package**, in a separate
  interpreter entrypoint that avoids Ash's namesake package. Set `grader_revision`
  to the installed package version (validated here: `3.0.15`). Patch collection
  restores the submission snapshot and excludes a pristine image's untracked
  baseline. Official Docker verification applies explicit CPU/memory/network
  limits and saves the unmodified official report.
- Pro uses the existing pinned official assembly/test script/parser adapter in
  `swebench_pro.grade`, with separate collection and pristine verification VMs.
  Supply `harness_repo` at the pinned revision in `swebench_pro.tasks`.
- DeepSWE uses `deepswe.grade` unchanged for committed-patch collection and
  offline verification. Set `benchmark="deepswe"`, `harness_repo` to the
  `datacurve-ai/deep-swe` checkout and `grader_revision` to its Git commit.
  Each frozen dataset row contains `instance_id` and `task_files_sha256`,
  produced by `runstore.deepswe.task_manifest` from the task directory.
  The manifest covers task metadata, instructions and verifier files, never
  oracle solutions. Resources must match `task.toml`; use its execution and
  verifier timeouts with additional time for snapshot collection/finalization.
  Both collector and verifier sessions use the worker's resource ledger.
- Nonempty unresolved patches and golden patches were checked through the queue
  for both benchmarks. This validates those controls, not every benchmark image.

## Validation

```bash
ASH_RUNSTORE_TEST_DSN='host=127.0.0.1 dbname=isolated_test_db user=test_user' \
  PYTHONPATH=.:sdk python3.12 -m pytest runstore/tests -q
```

Each database test gets a unique temporary schema. Never point the test command
at a production database. Reproducible real-VM/offline-inference checks are in
`../artifacts/run-store-v1-20260912/`: Codex and Claude multi-tool message branches,
official grading positive/negative controls, HTTP restart with two independent
workers, and controller-SIGKILL/lease-expiry/paired-resume. They create new test
snapshots and do not run a full benchmark or import old trajectories.
