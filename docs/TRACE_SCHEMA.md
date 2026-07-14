# Tool trace schema v1

Ash writes one JSON object per line to `<trace>.events.jsonl`. The trace records
both the model-facing agent call and the routed runtime call so those schemas
can evolve independently.

## Common envelope

Every event contains:

| Field | Meaning |
|---|---|
| `v` | Schema version. Currently `1`. |
| `type` | Event type. v1 defines `tool.started` and `tool.finished`. |
| `ts` | UTC wall-clock timestamp in RFC 3339 format. |
| `seq` | Monotonic event order within this JSONL stream. |
| `run_id` | One complete run or rollout, shared by participating agents. |
| `agent_id` | Agent that initiated the call. |
| `sandbox_id` | Sandbox in which the runtime call executes. |

`seq` is a stream order, not a distributed logical clock. Cross-agent causal
relationships must use explicit IDs rather than timestamps or sequence values
from different streams.

## Tool events

`tool.started` adds:

- `turn_id`: groups calls emitted by one model response.
- `call_id`: identifies one invocation and joins its start and finish events.
- `agent`: the model-facing `name` and `args`.
- `runtime`: the routed `name` and `args` actually sent to the runtime.
- `parent_call_id`, when a call is automatically derived from another call.

`tool.finished` repeats `turn_id` and `call_id`, then adds:

- `status`: `ok` or `error` in v1.
- `result`: raw runtime `output`, `error`, `output_bytes`, and
  `output_truncated`.
- `duration_ms`: elapsed monotonic time from immediately before execution.
- `error_kind`: present on failures; v1 emits `routing` or `runtime`.
- `observation`: present only when formatting, warnings, or processors changed
  the text returned to the model.
- `process_id`: present when a background shell call creates a process.

`parent_call_id` is not used to pair start and finish events, group parallel
calls, relate agents, or reference a process. Those relationships use
`call_id`, `turn_id`, `parent_agent_id`, and `process_id`, respectively.

Framework-specific identifiers belong outside the core envelope. Producers may
add namespaced `extensions`, for example `extensions.swebench.instance_id` or
`extensions.verl.rollout_id`.

## Harness identities

AshAgent-based harnesses assign identities at the instance boundary:

| Harness | `run_id` | `agent_id` | `sandbox_id` |
|---|---|---|---|
| LiteLLM | One per instance | `agent` | Spawned container ID |
| Manager-worker | Shared by manager and workers | `manager`, `worker-<task>` | One shared container ID |
| Best-of-N | Shared by all candidates | `candidate-<index>` | One container ID per candidate |

Claude Code does not run through `AshAgent`; its tool events will be covered by
the MCP proxy audit layer rather than this writer.
