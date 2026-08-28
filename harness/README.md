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

# the orchestrator owns the sandbox: it creates it, serves it, snapshots it,
# destroys it. --transport says only how the agent talks to it.
python -m harness run --slot claude-code --cwd /tmp/work \
    --sandbox-image python:3.11-slim --tools default \
    --backend microvm --runtime-bin runtime/ash-runtime \
    --transport http --snapshot-every-step "fix the failing test"

# same, with the server as a subprocess instead of in-process
python -m harness run --slot claude-code --transport stdio --sandbox-image ... "..."

# against a long-lived Execution Server somebody else runs (multi-slot,
# agent_id passthrough). No session here, so no snapshots -- see below.
python -m harness run --slot codex --mcp-url http://localhost:8400/mcp "..."

python -m harness show      runs/<id>.jsonl        # event histogram
python -m harness atif      runs/<id>.jsonl -o t.json
python -m harness fork-plan runs/<id>.jsonl --step 12
```

## Diagrams

- `docs/architecture-overview.excalidraw` — 一张图:分层架构 + 一次 run/fork 的时序图
- `docs/architecture-current.excalidraw` — 架构单独一张
- `docs/checkpoint-flow.excalidraw` — checkpoint 机制的流程图
- 都由 `docs/gen_*.py` 生成(几何校验:重叠/溢出直接拒绝写文件),改架构后重跑即可

## Layout

```
batch.py           worker pool + registry: concurrency, retry class, resumable
resources.py       write-ahead ledger of allocations (sandboxes, snapshots)
reap.py            reclaim what dead runs left behind
core/events.py     unified event model (v2), Usage with separated dimensions
core/journal.py    append-only JSONL writer + in-process event bus (subscribe)
core/slot.py       AgentSlot contract: run/kill/version + TaskSpec/McpWiring/SlotResult
normalize/*.py     native events -> journal events. Pure mapping tables, no I/O.
slots/cli_base.py  shared driver for JSONL-on-stdout CLIs (codex, opencode)
slots/*.py         per-agent drivers: command construction, MCP wiring, capabilities
execution/wiring.py  how a slot is told to reach the MCP proxy (stdio | http)
execution/session.py sandbox lifecycle + snapshots + the (tool, args) executor seam
execution/panel.py   the tool panel a model sees, compiled from a manifest
orchestrator/run.py  the shape of one run: owns the sandbox, the transport, teardown
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

## Batch runs and resource reclamation

```bash
python -m harness batch tasks.jsonl --slot opencode --workers 3 --out runs/b1
python -m harness reap --ledger runs/b1/resources.jsonl --dry-run
```

`batch.py` is deliberately thin — a worker pool plus a ledger, not a durable
scheduler. If a batch dies you re-run it; correctness of *cleanup* comes from the
ledger, and resumability from skipping tasks whose journal already reports a
terminal status. A persistent state machine is only worth building once RL
rollout defines what it must recover.

What it does have, because a few hundred tasks cannot work without it: bounded
concurrency, per-task failure isolation, per-task timeout, and **retry
classification** — a rate limit is retried, a context overflow or a safety
refusal is not (retrying a real outcome turns agent behaviour into apparent
environment noise).

Verified: 6 tasks / 3 workers, all correct, re-run skipped all 6.

**Per-task state isolation matters for concurrency.** The first real batch failed
2 of 6 with `database is locked`: opencode keeps sessions in one SQLite database,
and parallel lanes contend on it. The batch now gives each task its own
`XDG_DATA_HOME` (per *task*, not per attempt — a retry or later fork must find
the session the first attempt wrote). Same class of problem as `--pure` and
`setting_sources: []`: an agent's ambient state is not neutral.

### Resource ledger — the "no GC" problem

`resources.py` is a write-ahead ledger: record the intent to allocate *before*
allocating, mark it released after freeing. A crash therefore leaves a stale
claim (harmless — `reap` re-checks the backend) rather than an unreachable
resource. `claim.attach(journal)` records snapshots automatically from
`checkpoint.captured` events.

`reap` reads two sources: the ledger (knows the owner, so it can attribute an
orphan to a dead pid) and the backend (knows what exists, but has no owner
field — hence `--include-unknown --older-than 24h` as an explicit opt-in, since
"delete every snapshot older than N hours" is exactly the command that eats a
colleague's experiment).

**Finding: snapshot GC needs a backend change.** AgentENV exposes DELETE for
`/sandboxes/{id}` and `/templates/{id}` only; `DELETE /snapshots/{id}` answers
405, and `aenv snapshot` has no delete subcommand. A handful of fork-demo runs
left 17 orphaned snapshots that no client can reclaim. The reaper detects this
and reports it as *unsupported* rather than emitting N failures — but the fix
belongs in AgentENV.

Until then, `scripts/aenv-snapshot-gc.py` reclaims them offline (mark-and-sweep
over the on-disk store; keeps aliased templates and every layer they reference;
requires the server stopped). Worth knowing what it measured: 17 orphaned
snapshots freed **3.2 MB**, because overlaybd dedup means an incremental
disk-only checkpoint holds almost nothing of its own and the `chainSizeMB` the
API reports is the *logical* chain including shared base layers. Unbounded
snapshot growth is a count problem, not a capacity one.

## Transports, and why they are not equivalent

Both put the orchestrator in charge of the sandbox — it creates it, holds the
handle, and destroys it after the last snapshot. They differ in **where the tool
calls happen**, and that decides whether checkpointing works:

| | `--transport http` | `--transport stdio` |
|---|---|---|
| server | in this process, ephemeral port | the slot's own subprocess |
| sandbox handed over | the live handle (`pool.adopt`) | by id (`--attach`) |
| backends | any, Docker included | needs `attach` → microvm only |
| who constructs the machinery | the orchestrator | the server's own `main` |
| where records land | the journal, directly | `--checkpoint-log` JSONL, tailed into the journal live |

**One mechanism, mounted in whichever process serves the calls.** The machinery
is the `MutationTracker` interceptor (on the serving pipeline, so it sees every
call and knows a `text_editor view` changed nothing) plus `Checkpointer` fired
after every exec call: capture when something could have mutated, map a clean
step to the previous snapshot without paying for a capture, re-board when the
layer chain was compacted, squash at 128 layers. The transports differ only in
who constructs it and where the records land.

It was briefly two hand-rolled shortcuts instead, and each was wrong in its own
way — worth recording because both failures were *silent*: the http path built a
tracker nothing fed, so with `always` off every step after the first was recorded
as "clean" reuse of snapshot 1 while the agent was writing files (wrong pairs
that look complete); the stdio path re-implemented capture inline, so a `view`
paid for a snapshot, a `grep_files` step got no map entry at all, and none of the
chain upkeep ran. The tracker must sit on the pipeline that serves the calls.

Over stdio that serving process is the slot's subprocess — its loop is strictly
sequential so the boundary always fires quiesced, and `--attach` gives it its own
handle (microvm, which is also the only backend that can snapshot at all). Each
record is appended as one JSON line when it happens, not on exit: a killed run
keeps every line already written, which is the 300-snapshots-no-map lesson. The
parent **tails** the file during the run (`CheckpointTail`), so each pair lands
in the journal within a poll interval of the capture -- measured live: pairs at
t=17s and t=25s of a run that finished at t=30s. Pairing with the conversation
ref uses the bridge's backfill, so a pair tailed before the slot disclosed its
session id is corrected retroactively. `load_checkpoints` and `fork_plan` read
both transports identically.

This is also the answer to "should the journal be a server": for reading it
already is one (the file -- anyone can tail it, and it survives every kind of
death); for writing, each process appends to its own file and the owner tails
it. One writer, one reader, the file is the pipe. A socket-based journal server
would add a second write path at the cost of the property that makes the journal
worth trusting -- a killed run still leaves its record -- so that waits until
subagents genuinely need many-to-one.

Verified live with `always` off: write → capture, view → clean reuse, write →
new capture; identical records on both transports. `checkpoint.unavailable`
still fires on wirings with no boundary to stand at (hand-rolled
`mcp_stdio_args` without `--checkpoint-log`).

## Threading rules (learned by breaking them)

`SandboxSession` drives a private event loop via `run_until_complete`. Two rules
follow, and violating either fails *quietly* — tool calls or snapshots return
`None` with an un-awaited coroutine warning:

- **Call the executor from a worker thread** (`asyncio.to_thread`) when inside an
  SDK tool handler; that handler already runs on a loop, and loops cannot nest.
- **Never checkpoint from a thread that has a running loop.** The bridge detects
  this and declines instead of failing. That count is `bridge.skipped_on_loop`,
  and it is now *read* — it was private and looked at by nobody, which is how an
  orchestrator-owned run could report checkpointing as on and record zero
  snapshots. For the SDK slot the tool boundary — a worker thread — is the correct
  trigger anyway; the turn-boundary trigger serves the CLI slots, whose journal
  writes come from a plain reader thread.

## Not here yet

- A durable orchestrator: cross-process recovery, a persistent state machine, IAC
  routing for agents that spawn sub-agents. `batch.py` covers concurrency and
  isolation; the gateway covers budget; `reap` covers cleanup. What remains is
  only needed once RL rollout defines the recovery semantics.
- Live behaviour probes in `contracts/ci_check.py` (need credentials).
- Fork verified live for all three protocol drivers on microvm: claude-code
  (resume + fork_session), opencode (`/session/{id}/fork`), codex
  (`thread_fork`) -- each through the orchestrator, full pair, sibling isolation
  on both halves. The `-cli` fallback drivers remain unverified end to end.
- Verified on a real SWE-bench Verified instance (django__django-11848,
  microvm, http transport) with ALL THREE protocol drivers: claude-code (Opus 5
  on Bedrock), codex (gpt-5.6-sol, thread_fork), opencode (session fork). A
  pair per step with mutation gating live on real workloads -- opencode's run
  recorded 17 pairs of which 11 were clean reuses (its `view` steps cost no
  capture but kept the map complete); codex's turn-boundary pair reused too. On
  every agent: a FRESH microVM resumed from the pre-edit snapshot shows a clean
  tree and one from the post-edit snapshot shows exactly the fix; branching at
  the edit step produced one branch that validated the parent's fix against the
  full test module and one that reverted it and shipped a different
  implementation -- distinct sandboxes, distinct forked sessions, two different
  final diffs, no leaks.
- The ATIF inherited-prefix boundary is the checkpoint's own journal seq, which
  can under-include same-step events that journal after the boundary fires
  (slot-dependent ordering). The environment half is exact; the copied-context
  marking of the conversation prefix is approximate near the branch step.
