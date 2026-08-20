# Ash Architecture

> Status: accepted design (2026-07). All four layers are implemented; the L2
> interceptor pipeline is mounted in the MCP proxy (opt-in via `--coordinate` /
> `--plugins`; see [Status](#status-implemented-vs-target)).

Ash is structured as four layers. Each layer has exactly one responsibility and
one home; every layer talks to the one below it through a single, frozen
contract.

```
┌─────────────────────────────────────────────────────────────────┐
│ L4  EVAL — score topologies on benchmarks                        │
│     SWE-bench today; Commit0 / PaperBench / custom tasks later   │
│     contract: Harness.run_instance(instance, dir) -> prediction  │
├─────────────────────────────────────────────────────────────────┤
│ L3  HARNESS — topology (who does what)                           │
│     single-agent | manager-worker | debate | best-of-n | ...     │
│     roles are prompts, orchestration is harness code             │
│     contract: AshAgent(config, executor) — agents only ever see  │
│               an executor: (tool_name, args) -> ToolResult       │
├─────────────────────────────────────────────────────────────────┤
│ L2  MCP PROXY — governance of the tool-call path                 │
│     an interceptor pipeline; coordination (Waggle), guardrails,  │
│     audit, per-agent tool ACLs are all interceptors              │
│     identity: MCP session = agent_id; routing key: sandbox_id    │
│     contract: same 7-tool schema in, same schema out             │
├─────────────────────────────────────────────────────────────────┤
│ L1  RUNTIME — execution                                          │
│     ash-runtime: single Go binary, 7 tools, HTTP / MCP / stdio   │
│     it executes tool calls; it never grows anything else         │
└─────────────────────────────────────────────────────────────────┘
```

The seam that composes everything is one function signature:

```
executor: (tool_name: str, args: dict) -> ToolResult
```

A sandbox session *produces* an executor, the proxy *wraps* executors, an agent
*consumes* an executor. The three never need to know about each other, which is
why topologies, coordination policies, and transports can change independently.

---

## L1 — Runtime (execution)

`ash-runtime` runs inside every sandbox and exposes exactly six tools
(`shell`, `process`, `text_editor`, `grep_files`, `web_fetch`,
`web_search`) over JSON-RPC via HTTP, MCP, or stdio.

Design rule: **the runtime only executes.** No coordination, no policy, no
middleware. Cross-cutting behavior belongs to L2. This keeps the binary small,
the protocol frozen, and every transport/client trivially compatible.

## L2 — MCP proxy (governance)

The proxy is the single front door for agents. Its core is not "coordination";
it is a **tool-call pipeline** with interception points. Coordination is just
one interceptor.

### Interceptor pipeline

```
                     MCP proxy
        ┌──────────────────────────────────────────┐
tool    │  ┌─────────┐  ┌────────┐  ┌───────┐      │
call ──►│─►│guardrail│─►│ waggle │─►│ audit │─► L1 │
        │  └─────────┘  └────────┘  └───────┘      │
result◄─│◄──── after hooks run in reverse ─────────│
        └──────────────────────────────────────────┘
        onion model: before in order, after in reverse
```

```python
class ToolInterceptor:
    tools: set[str] | Literal["*"] = "*"

    def before(self, ctx: CallContext) -> Verdict:
        """Continue | Reject(msg) | Rewrite(new_args) | ShortCircuit(result)"""
        return Continue()

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        return result
```

Pipeline rules:

- **Short-circuits still unwind the onion.** If `before` rejects, the `after`
  hooks of interceptors already entered still run — audit always sees rejected
  calls.
- **Fail-safety is per interceptor.** Safety interceptors fail closed (reject);
  observability interceptors fail open (pass + log).
- **The framework is stateless.** State (locks, versions) lives inside
  interceptors; the pipeline itself never becomes a concurrency bottleneck.

Assembly is plain Python — order is semantics, a list is the configuration:

```python
# my_plugins.py        ash-mcp-proxy --plugins my_plugins.py
PIPELINE = [
    GuardrailInterceptor(read_before_edit=False),   # nudges; Waggle enforces below
    ToolACLInterceptor({"investigator": {"text_editor", "grep_files"}}),  # planned
    WaggleInterceptor(policy=TeamPolicy()),                   # coordination
    TruncateInterceptor(),                                    # bound the result
    AuditInterceptor("audit.jsonl"),                          # observe last (planned)
]
```

`GuardrailInterceptor` (`swebench/agent/guardrails.py`), `WaggleInterceptor` and
`TruncateInterceptor` (`swebench/agent/interceptors.py`) exist today; the ACL and
audit seats are still to be written. Note `read_before_edit=False` above: that
rule is Waggle's when coordination is mounted, and two seats stating one rule
tells the model the same thing twice.

### Waggle — the coordination interceptor

Waggle provides optimistic concurrency control (OCC) for N agents sharing one
workspace, at tool-call granularity:

| Event | Mechanism |
|---|---|
| read (`text_editor` / `view`) | record per-agent snapshot of the file version seen |
| write (`str_replace` / `insert` / `write`) | arbitrate: stale snapshot → reject with a unified diff of what changed; the loser is granted a TTL **reservation** so it can re-read and re-apply without being overtaken |
| writer hits a foreign reservation | wait on the file's condition; on release, waiters re-arbitrate FIFO (commit-release → all conflict; expiry-release → a still-consistent waiter wins) |
| shell | effects detected post-hoc: registered files are fingerprinted after each call; drift from the last coordinated state is recorded as an `external` version (with an under-lock re-check so a raced coordinated commit is never misread as drift) |

Conflict *resolution* is delegated to the calling LLM: a rejection is an
ordinary failed tool result carrying the diff and instructions to re-read.
The merge engine is the model itself.

**Mechanism vs policy.** The kernel (version bookkeeping, per-file locking,
full-content history, invariants) is fixed. Decisions are policy, exposed as
Python hooks — code, not a YAML DSL:

```python
class WagglePolicy:
    def on_write(self, ctx) -> Allow | Reject | Wait | Defer: ...   # Defer = default OCC
    def on_conflict(self, ctx) -> Response: ...                     # default: reject + diff
    def on_drift(self, ctx) -> Action: ...                          # default: record
    def on_commit(self, ctx) -> None: ...                           # observe only
```

Policy guardrails: hooks return decisions but have no API to mutate the ledger
(account integrity is unreachable from policy code); a hook exception falls
back to default OCC and logs; hooks run inside the file's critical section, so
policy authors never reason about concurrency.

**Two extension surfaces — do not conflate them.** `WaggleInterceptor` *is a*
`ToolInterceptor`; **`WagglePolicy` is not** — it plugs into the Waggle
interceptor, one level deeper (like passport.js: the auth middleware is one
pipeline element; strategies plug into it and are not middleware themselves).
The levels differ in context, atomicity, and power:

| You want to customize | Use |
|---|---|
| what happens on conflict / who may write what / drift handling | `WagglePolicy` — sees kernel-computed coordination context (`snapshot_version`, diff, history), runs inside the file's critical section, cannot corrupt the ledger |
| a wholly different coordination mechanism | your own `ToolInterceptor` replacing `WaggleInterceptor` — you own all the bookkeeping |
| cross-cutting concerns beyond coordination (audit, ACL, budgets) | a sibling `ToolInterceptor` in the pipeline |

Flattening these into one level would force every policy author to rebuild the
OCC kernel (a raw interceptor only sees `(tool_name, args)` — it cannot know
what "stale" means), and would hand ledger-corrupting power to code that only
wants to choose a conflict response.

### Identity and routing

- `agent_id` = MCP session (external clients) or explicit id (harness threads).
- `sandbox_id` routes to the container; Waggle keys are
  `(agent_id, sandbox_id, path)`, so N:M works without changes.

Two orthogonal primitives cover every multi-agent shape:

|              | 1 sandbox              | N sandboxes                    |
|--------------|------------------------|--------------------------------|
| **1 agent**  | neither needed         | routing only (`sandbox_id`)    |
| **N agents** | arbitration (Waggle)   | routing + arbitration          |

## L3 — Harness (topology)

A harness decides *who does what*: how many agents, what roles, what order,
what feedback loops. It never implements coordination — with L2 in place,
overlap between agents' edits is arbitrated below it.

Registered harnesses: `litellm` (single agent), `claude-code`,
`manager-worker` (explore → decompose → parallel workers on one shared
sandbox), `best-of-n` (N fully isolated parallel candidates, one patch
selected by tests or heuristic). Planned: orchestrator-worker with
Magentic-style ledger/replan, debate.

## L4 — Eval

Runs a harness across a benchmark and scores it. Configs compose via
`extends:`; outputs are predictions, trajectories, and (when Waggle is active)
a per-instance coordination audit (`<iid>-waggle.json`) with every file's full
version chain.

The evaluation matrix is deliberately two-axis — **topology × coordination** —
so "does multi-agent help" and "does arbitration help" can be measured
independently.

---

## Decision records

**ADR-1: Coordination does not live in the runtime.**
The runtime's value is being a small, frozen, single-purpose execution binary.
Coordination is stateful policy; embedding it would couple the fleet's most
replicated component to its fastest-changing logic. The proxy is the chokepoint
that all agents already traverse, so the ledger lives there.
*Trade-off accepted:* the proxy cannot watch the container filesystem
(no fsnotify), so shell-drift detection stays scan-based (~10 ms per shell
call, measured) rather than event-based.

**ADR-2: The proxy's core abstraction is interception, not coordination.**
Guardrails, ACLs, audit, redaction, budgets and Waggle all need the same thing:
a seat on the tool-call path. One generic `ToolInterceptor` chain serves all of
them; coordination is one plugin among peers. This is the same model as Claude
Code's PreToolUse/PostToolUse hooks and Envoy filters.

**ADR-3: Policy is Python code, not a YAML DSL.**
Real policies are logic (ownership globs, auto-merge attempts, priorities).
A YAML schema for that inevitably grows into a crippled programming language.
Configuration keeps only true scalars (`ttl`, `enabled`); everything
conditional is a hook. Single-language plugins also erase the loading problem
(no WASM/webhook machinery until a concrete need exists).

**ADR-4: Conflict resolution is delegated to the LLM.**
Waggle never merges. It converts silent overwrites into explicit, readable
signals (diff + "re-read and re-apply"), and the agent performs the merge by
retrying. Validated end-to-end: two live agents forced onto the same file
produced one rejection, one self-served re-read/re-apply, zero lost updates.

**ADR-5: Same-file commit I/O happens under the file lock (deliberate).**
A commit's write and its fingerprint update must be atomic, or drift detection
could not distinguish coordinated writes from external ones. Same-file writes
are serialized on purpose; different files never contend. Sandbox calls are
localhost HTTP (milliseconds), which bounds the cost.

## Status: implemented vs target

| Piece | Today | Target |
|---|---|---|
| Runtime (L1) | ✅ as designed | unchanged, forever |
| Waggle kernel | ✅ `swebench/agent/waggle.py` (`WorkspaceCoordinator`), 10 unit tests + live-conflict experiment | same kernel, mounted in the proxy |
| Waggle mounting | ✅ `WaggleInterceptor` inside the MCP proxy (opt-in `--coordinate` / `--plugins`); `CoordinatedExecutor` kept as test fixture / proxy-less lite mode; `piped_executor` / `executor_for(pipeline=)` mounts the same chain on harness-thread agents | unchanged |
| Interceptor pipeline | ✅ `swebench/agent/pipeline.py`, mounted in `swebench/mcp_server.py` and consumed by the agent loop | ACL/audit interceptors still to come |
| Guardrails + truncation | ✅ migrated out of the loop into `GuardrailInterceptor` (`guardrails.py`) + `TruncateInterceptor` (`interceptors.py`); the loop mounts `default_pipeline()`, the proxy takes `--guardrails` | unchanged |
| Policy hooks | ✅ `WagglePolicy` (`on_write`/`on_conflict`/`on_drift`/`on_commit`), run inside the file's critical section | + two reference policies (ownership ACL, auto-merge-then-reject) |
| Harnesses (L3) | `litellm`, `claude-code`, `manager-worker`, `best-of-n` | + orchestrator-worker (ledger/replan), debate |
| Eval (L4) | SWE-bench (`extends:` configs, batch runner) | + more benchmarks; topology × coordination A/B matrix |

## Roadmap (ordered)

1. **A/B evidence** — manager-worker with Waggle on/off over a multi-file
   SWE-bench slice; publish numbers alongside the mechanism.
2. **Pipeline + proxy mounting** — ✅ interceptor chain built and mounted in
   `swebench/mcp_server.py`, Waggle behind it; guardrails and output truncation
   migrated out of the agent loop into interceptors, so agents arriving through
   the proxy get them too.
3. **`best-of-n` harness** — parallel candidates + objective test-based
   selection (the highest-confidence score lever).
4. **Policy hook surface** — ✅ `WagglePolicy` shipped; the two reference
   policies (ownership ACL, auto-merge-then-reject) still pending.
