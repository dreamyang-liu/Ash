# `harness/` — multi-agent slot layer

Runs **any** of several agents against the same task, in the same sandbox, and
records one trajectory format for all of them. Built for rollback: every run's
journal can be paired with an AgentENV snapshot so a run can be resumed — or
**forked** — from any step.

Where this sits relative to `docs/ARCHITECTURE.md`: it is a second L3 (topology)
that is benchmark-agnostic. `swebench/harnesses/` answers "how do I score
SWE-bench"; `harness/` answers "how do I drive an arbitrary agent and keep a
replayable record". It reaches the sandbox through the same L2 seam every other
path uses — the MCP proxy (`swebench/mcp_server.py`), so the interceptor chain,
tool panel and traces behave normally.

```
Eval / CLI ──▶ slot driver ──tool calls──▶ MCP proxy (L2 chain) ──▶ sandbox (L1)
                   │                                                    │
                   │ normalized events                       AgentENV snapshots
                   ▼                                                    │
              Journal (append-only) ◀───── rollback pair (seq ↔ snapshot) ┘
                   │
                   └──▶ ATIF v1.8 export
```

The agent engine stays **outside** the VM. Packaging Claude Code (node + deps)
into a scratch/distroless rootfs would defeat the point of `envd` and the Go
runtime being static binaries; instead the agent runs host-side and every side
effect it can cause travels through MCP into the sandbox.

## Slots

| slot | driver | resume | fork | notes |
|---|---|---|---|---|
| `claude-code` | claude-agent-sdk (typed messages + PreToolUse hook) | yes | yes (`fork_session`) | builtin tools denied; verdict seam enforced in-process |
| `codex` | `codex exec --json` | yes (`exec resume`) | no | host sandbox `read-only` by default |
| `opencode` | `opencode run --format json` | yes | **yes, natively** (`--session --fork`) | `--pure` to ignore local plugins |

Verified against claude-agent-sdk 0.2.145 / claude 2.1.239, codex-cli 0.145.0,
opencode 1.18.5. Those versions are asserted by `contracts/ci_check.py`.

## Usage

```bash
# smallest run: agent works directly in --cwd, no sandbox wiring
python -m harness run --slot opencode --cwd /tmp/work "is there a bug in calc.py?"

# with the ash execution plane as a stdio MCP subprocess
python -m harness run --slot claude-code --cwd /tmp/work \
    --mcp-stdio "--image python:3.11" "fix the failing test"

# against a long-lived Execution Server (multi-slot, agent_id passthrough)
python -m harness run --slot codex --mcp-url http://localhost:8400/mcp "..."

python -m harness show      runs/<id>.jsonl        # event histogram
python -m harness atif      runs/<id>.jsonl -o t.json
python -m harness fork-plan runs/<id>.jsonl --step 12
```

## Layout

```
core/events.py     unified event model (v2), Usage with separated dimensions
core/journal.py    append-only JSONL writer + in-process event bus (subscribe)
core/slot.py       AgentSlot contract: run/kill/version + TaskSpec/McpWiring/SlotResult
normalize/*.py     native events -> journal events. Pure mapping tables, no I/O.
slots/cli_base.py  shared driver for JSONL-on-stdout CLIs (codex, opencode)
slots/*.py         per-agent drivers: command construction, MCP wiring, capabilities
execution/wiring.py  how a slot is told to reach the MCP proxy (stdio | http)
checkpointing.py   bridge: swebench Checkpointer -> RollbackLedger, quiesce rule
rollback.py        checkpoint pairing (env snapshot + session ref), fork plans
gateway/           inference gateway: model swap, wire tap, enforced budget
atif.py            journal -> ATIF v1.8 (interchange only; journal stays canonical)
demo_fork.py       acceptance demo: run -> branch at step N -> K continuations
tests/             fixtures captured from real CLI runs + fake agent/upstream E2E
```

## Invariants worth keeping

**The journal is canonical state.** Everything a slot learns is appended as it
happens, so a killed run still yields a usable trajectory (same rule as
`agent/checkpoints.py` persisting every step). Nothing rewrites history;
corrections are new events.

**Normalizers never drop.** Unmapped native payloads become `raw.<slot>` events
carrying the original. A silent drop is the one unacceptable failure mode: the
trajectory is wrong and nothing surfaces an error. `harness show` flags any
`raw.*` for exactly this reason, and a test asserts the recorded fixtures map
cleanly end to end.

**Usage dimensions stay separate.** `input / output / cached_input /
cache_creation / reasoning` tokens are tracked individually because providers
disagree about the breakdown and collapsing them loses cache economics
(opencode's `tokens.total` double counts cache writes, so we ignore it).

**Builtin tools are denied, not discouraged.** Claude Code's builtin
Bash/Read/Edit run on the *host*: their effects land outside the sandbox and
outside every snapshot, corrupting both the environment and the record. Denial
is enforced via `disallowed_tools` plus a PreToolUse verdict.

**PreToolUse, not `can_use_tool`.** Under `permission_mode="bypassPermissions"`
(what unattended runs need) the SDK auto-approves before `can_use_tool` is
consulted, so a callback there never fires — silently. It is also the step
boundary an external agent has, i.e. where checkpoints are taken; mounting it on
the shadowed seam would produce a rollback feature that records zero
checkpoints. `can_use_tool` is registered only for stricter modes.

**Timeouts use a watchdog, not `wait(timeout=)`.** Reading a child's stdout
blocks, so a CLI that hangs without emitting anything would never be reclaimed
and would stall a whole batch.

**Adapters depend on unpromised surfaces.** Flag names, event fields and SDK
kwargs are not covered by upstream compatibility guarantees, and
`ClaudeAgentOptions` kwargs we pass are dropped silently if renamed. `contracts/`
declares each assumption and `contracts/ci_check.py` asserts them — run it daily
and before bumping a pinned version.

## Rollback

A rollback point is a **pair**: an environment snapshot and the agent's
conversation reference. Neither half alone can branch a run — restoring files
under an agent whose memory disagrees with them produces nonsense, and branching
a conversation while siblings share one filesystem lets them corrupt each other.

`SnapshotBridge` wires both halves without any slot knowing it exists — it
subscribes to the journal:

```python
from harness.checkpointing import SnapshotBridge

bridge = SnapshotBridge.install(journal, session)      # CLI slots: turn boundaries
slot.run(task, journal, mcp)

bridge = SnapshotBridge.install(journal, session)      # SDK slot: PreToolUse
slot = ClaudeCodeSlot(on_tool_boundary=bridge.on_tool_boundary)
```

It only fires at a **quiesce point** — a step boundary with no in-flight tool
call. Snapshotting mid-call pairs an unresolved call with an ambiguous
filesystem, so the pair would not describe a resumable state. A session id that
arrives after the first checkpoint is backfilled by appending a correction (the
journal is never rewritten).

`fork_plan(journal, step)` resolves the pair and returns the `seq` boundary ATIF
needs for `is_copied_context`.

### Demo — verified end to end on AgentENV

```bash
export AENV_SERVER_URL=http://127.0.0.1:8000 AENV_API_KEY=...
python -m harness.demo_fork --slot claude-code --in-process --backend microvm \
    --sandbox-image docker.io/library/python:3.11-slim --branch-at 1 \
    --cwd /tmp/demo --prompt "..." \
    --direction "write branch_a.txt" --direction "is branch_a.txt there?"
```

Real result (Firecracker snapshots, claude-code, two branches off step 1):

```
branch step 1  pair complete: True
  env half          01a04476-a3fa-7bf1-8e2b-26e4a6d32407
  conversation half 8d3b6503-4f85-429d-8f11-a016a75cb183
branch 1 -> "base.txt branch_a.txt"   # inherited the parent's work, added its own
branch 2 -> "NO"                      # sees the parent's base.txt, NOT its sibling's file
distinct child sessions: yes
```

Branch 2 is the proof: it inherits everything the parent did and nothing its
sibling did. Each ATIF export shows `step 1 copied=True` (the shared prefix) and
`step 2 copied=False` (its own work).

Two requirements for that result, both of which fail loudly if unmet:

- **A snapshot-capable backend.** Docker cannot snapshot; only `MicroVMPool`
  (`--backend microvm`) can. On Docker the demo prints `env half ABSENT` rather
  than implying a complete pair — and an early run in that mode demonstrated
  exactly why the pair matters: branch 1 edited a file and branch 2, sharing the
  filesystem, reported "the file no longer contains the bug".
- **`--in-process`** for the SDK slot. The stdio `swebench.mcp_server` subprocess
  creates and owns its *own* sandbox, so the session being snapshotted would not
  be the one the agent worked in — the pair would reference an environment nobody
  touched. In-process keeps sandbox ownership in the harness.

## Inference gateway

The model seam (`harness/gateway/`). Wiring is one environment variable, which is
why it works for any agent:

```bash
python -m harness gateway --routes routes.json --mint agent-1 --budget-usd 2.50
# -> ANTHROPIC_BASE_URL=http://127.0.0.1:8787 + a per-slot token

python -m harness run --slot claude-code --gateway --budget-usd 2.50 "..."
```

Three things no slot can provide:

- **Model swap** — the reason it exists. A routing table entry redirects a model
  name to any endpoint (another provider, local vLLM, an RL checkpoint):
  `{"ash-rl-ckpt-42": {"base_url": "http://10.0.0.5:8000", "upstream_model": "checkpoint-42"}}`.
- **Wire-level tap** — exact token counts and the real request, independent of
  what the agent reports. Journaled as `gateway.request`.
- **Enforced budget** — 429 once a slot is over its ceiling. Slot-side accounting
  can only *ask* an agent to stop.

Implementation constraints worth knowing: headers pass through untouched
(`anthropic-beta`, `x-claude-code-session-id`/`-agent-id`/`-parent-agent-id` —
dropping them changes behaviour or loses subagent attribution); response bytes
are relayed verbatim (unknown fields such as thinking-block signatures must
survive byte-exact); streaming is a passthrough with a side parser for usage, so
the agent sees no added latency and frames are never modified; `/v1/models` must
answer because Claude Code treats discovery failure as fatal. The agent only ever
holds its own slot token — provider credentials stay in the routing table.

Verdict-style rewriting is *not* here: tool-call policy belongs in the MCP proxy
(L2) where a call is semantically addressable. This layer speaks HTTP and tokens.

## Threading rules (learned by breaking them)

`AshSession` drives a private event loop via `run_until_complete`. Two rules
follow, and violating either fails *quietly* — tool calls or snapshots return
`None` with an un-awaited coroutine warning:

- **Call the executor from a worker thread** (`asyncio.to_thread`) when inside an
  SDK tool handler; that handler already runs on a loop, and loops cannot nest.
- **Never checkpoint from a thread that has a running loop.** The bridge detects
  this and declines (counted in `_skipped_on_loop`) instead of failing. For the
  SDK slot the tool boundary — a worker thread — is the correct trigger anyway;
  the turn-boundary trigger serves the CLI slots, whose journal writes come from
  a plain reader thread.

## Not here yet

- Orchestrator (registry / scheduling / concurrency / guaranteed teardown) —
  currently one run per CLI invocation. Budget enforcement exists in the gateway.
- Live behaviour probes in `contracts/ci_check.py` (need credentials).
- Fork for the CLI slots against microvm: `opencode` has native session fork and
  `codex` replays its prompt, but neither has been run through the full pair
  end to end (the SDK slot has).
