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
core/journal.py    append-only JSONL writer: monotonic seq, flush-per-line, thread-safe
core/slot.py       AgentSlot contract: run/kill/version + TaskSpec/McpWiring/SlotResult
normalize/*.py     native events -> journal events. Pure mapping tables, no I/O.
slots/cli_base.py  shared driver for JSONL-on-stdout CLIs (codex, opencode)
slots/*.py         per-agent drivers: command construction, MCP wiring, capabilities
execution/wiring.py  how a slot is told to reach the MCP proxy (stdio | http)
rollback.py        checkpoint pairing (env snapshot + session ref), fork plans
atif.py            journal -> ATIF v1.8 (interchange only; journal stays canonical)
tests/             fixtures captured from real CLI runs + a fake-agent E2E
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
under an agent whose memory disagrees with them produces nonsense.

```python
from harness.rollback import RollbackLedger
ledger = RollbackLedger(journal)
ledger.record(step, snapshot_id, session_ckpt=native_session_id)
```

Pair only at a quiesce point: a step boundary with no in-flight tool call
(`turn.completed` → next `turn.started`, or the PreToolUse callback for the SDK
slot). `fork_plan(journal, step)` resolves the pair and returns the `seq`
boundary that ATIF export needs for `is_copied_context`.

## Not here yet

- Orchestrator (registry / scheduling / budget enforcement / guaranteed
  teardown) — currently one run per CLI invocation.
- Inference gateway (model swap for eval matrices and RL serving). Keep
  `base_url` configuration-driven so it can be inserted without code changes.
- Live behaviour probes in `contracts/ci_check.py` (need credentials).
- Wiring `swebench/agent/checkpoints.py` into `RollbackLedger` for the three
  slots — the ledger is ready, the `Checkpointer` bridge is not written.
