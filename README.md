<p align="center">
  <img src="assets/logo.svg" width="200" alt="Ash Logo"/>
</p>

<h1 align="center">Ash — Agent Sandbox Hive</h1>

<p align="center">
  Scalable sandbox infrastructure for LLM agents and RL training.<br/>
  Isolated execution environments with a minimal tool protocol over HTTP, MCP, or stdio —<br/>
  and a snapshot of every step, so <b>any step of any run can be resumed or branched.</b>
</p>

<p align="center">
  <a href="#branching-a-failed-rollout">Branching</a> •
  <a href="#swe-bench-verified">Results</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#runtime">Runtime</a> •
  <a href="#python-client">Client</a> •
  <a href="#k8s-infrastructure">K8s Deploy</a>
</p>

---

<p align="center">
  <img src="assets/architecture.svg" width="800" alt="Ash Architecture"/>
</p>

## Branching a failed rollout

Every step an agent takes leaves a **rollback pair** — a filesystem snapshot plus
the agent's own conversation reference — so any step is resumable. Hand a later run
that snapshot as its image and the parent's session as its conversation and you get
a *branch*: same history, divergent future. Siblings each get their own microVM off
the same snapshot, so they cannot contaminate each other.

`fork_eval` turns that into a loop: run an attempt, grade it, and on failure let an
analyst model read the trajectory **plus the grading verdict** and fan out K
divergent continuations from the step it chooses.

```bash
python -m swebench.fork_eval \
    --instance sympy__sympy-13091 \
    --slot codex --model openai.gpt-5.6-luna \
    --rounds 2 --branches 3
```

Measured on `sympy__sympy-13091` (reference patch 522 lines): the first attempt
changed only `Basic.__eq__` and failed. The analyst named the mechanism — numeric
classes in `sympy/core/numbers.py` override comparison, so unknown-type comparisons
never reach `Basic.__eq__` — branched at the step before that edit, and **2 of 3
branches came back resolved**, all 89 `PASS_TO_PASS` regressions passing.

Three agent CLIs are supported and each has been driven through this end to end:
`claude-code` (resume + `fork_session`), `codex` (`thread_fork`), `opencode`
(`/session/{id}/fork`). The infrastructure is agent-agnostic: one orchestrator, one
checkpoint mechanism, one MCP server.

## SWE-bench Verified

<table>
<tr>
<th>Model</th>
<th>Tool Mode</th>
<th align="right">Resolved</th>
<th align="right">Rate</th>
</tr>
<tr>
<td><b>Claude Opus 4.6</b></td>
<td>5-tool</td>
<td align="right">397 / 500</td>
<td align="right"><b>79.4%</b></td>
</tr>
<tr>
<td><b>Claude Sonnet 4.6</b></td>
<td>5-tool</td>
<td align="right">377 / 500</td>
<td align="right"><b>75.4%</b></td>
</tr>
<tr>
<td><b>Claude Sonnet 4.6</b></td>
<td>bash-only</td>
<td align="right">362 / 500</td>
<td align="right">72.4%</td>
</tr>
<tr>
<td><b>Claude Sonnet 4.6</b></td>
<td>Claude Code</td>
<td align="right">371 / 500</td>
<td align="right">74.2%</td>
</tr>
<tr>
<td><b>MiniMax M2.5</b></td>
<td>bash-only</td>
<td align="right">378 / 500</td>
<td align="right"><b>75.6%</b></td>
</tr>
<tr>
<td>MiniMax M2.5</td>
<td>5-tool</td>
<td align="right">353 / 500</td>
<td align="right">70.6%</td>
</tr>
<tr>
<td><b>Kimi K2.5</b></td>
<td>bash-only</td>
<td align="right">350 / 500</td>
<td align="right"><b>70.0%</b></td>
</tr>
<tr>
<td>Kimi K2.5</td>
<td>5-tool</td>
<td align="right">343 / 500</td>
<td align="right">68.6%</td>
</tr>
</table>

All runs use the same Ash sandbox environment. **These numbers are history**: the
`5-tool` / `bash-only` rows were produced by Ash's own litellm agent loop, and the
`Claude Code` row by a `claude-code` harness — both since removed in favour of driving
real agent CLIs through the orchestrator (see [Branching](#branching-a-failed-rollout)).
The measurements stand; reproducing them would mean restoring that loop from git
history.

## Quick Start

```bash
# Build the runtime
cd runtime && go build -o ash-runtime .
```

### Run a single sandbox

```python
from ash_sandbox import Sandbox

async with Sandbox.connect("http://localhost:3000") as sb:
    result = await sb.call("shell", command="echo hello")
    print(result.output)
```

### Run many sandboxes with Docker

```python
from ash_sandbox import DockerPool

async with DockerPool(runtime_bin="./ash-runtime") as pool:
    sb = await pool.spawn(image="python:3.11")
    await sb.call("shell", command="pytest")
    await sb.call("text_editor", command="str_replace", path="app.py",
                  old_str="foo", new_str="bar")
```

### Run SWE-bench with snapshots and branching

```bash
pip install ./sdk datasets pyyaml

# microvm is the only backend that can snapshot -- Docker cannot
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_API_KEY=...

python -m swebench.fork_eval \
    --instance django__django-11848 \
    --slot codex --model openai.gpt-5.6-luna \
    --rounds 2 --branches 3 -o runs/fork-eval
```

One instance per invocation. Or drive a single agent yourself and branch by hand:

```bash
python -m harness run --slot claude-code \
    --sandbox-image python:3.11-slim \
    --backend microvm --runtime-bin runtime/ash-runtime \
    --transport http --tools default \
    --journal runs/one.jsonl "fix the failing test in /testbed"

python -m harness show      runs/one.jsonl        # what happened
python -m harness fork-plan runs/one.jsonl --step 7   # the pair at step 7
python -m harness reap                            # reclaim leaked sandboxes
```

### Deploy at scale on Kubernetes

```bash
cd k8s-config && bash deploy.sh
```

```python
from ash_sandbox import SandboxPool

async with SandboxPool(
    control_plane_url="http://control-plane:80",
    gateway_url="http://gateway:80",
) as pool:
    sandboxes = [await pool.spawn(image="my-task:latest") for _ in range(100)]
```

## Architecture

Four layers, each knowing only the one below it:

```
swebench/          what counts as an ANSWER (a patch; graded by the instance's tests)
        │
harness/           the agent runtime — never imports swebench/ (enforced by test)
  orchestrator/      one run: create the sandbox, serve it, snapshot each step, tear down
  slots/             per-agent drivers (claude-code, codex, opencode) behind one interface
  execution/         the tool path: session, compiled tool panel, MCP server, interceptors
  gateway/           model traffic: swap the model, price it, enforce a budget
        │
ash-runtime (Go)   8 tools over JSON-RPC, inside every sandbox
        │
AgentENV           Firecracker microVMs: snapshot, restore, fork
```

The load-bearing rule is that **granularity decides placement**: per tool call → an
interceptor, per step → a checkpoint, per run → the orchestrator. An agent is a black
box behind `run/kill/version`, which is why the same checkpoint machinery works for
three unrelated agent SDKs.

## Runtime

The `ash-runtime` binary runs inside each sandbox container. Single Go binary (~9MB), no dependencies beyond `ripgrep`.

### Run Modes

| Mode | Command | Use Case |
|------|---------|----------|
| HTTP | `ash-runtime --port 3000` | Container runtime (default) |
| stdio | `ash-runtime --mode stdio` | CLI, MCP stdio transport |
| MCP | `POST /mcp` endpoint | FastMCP, Claude Desktop, Cursor |

### Tools

| Tool | Description |
|------|-------------|
| `shell` | Execute commands. `background: true` returns a pid. |
| `process` | Read output or kill background processes. |
| `text_editor` | View, write/create, str_replace, insert files. |
| `grep_files` | Ripgrep search with pattern, glob, limit. |
| `web_fetch` | Fetch URLs. Formats: html, text, markdown. |
| `web_search` | Multi-engine search (Google, DuckDuckGo, Brave). |
| `artifact` | Fetch and verify a binary; backs manifest-defined custom tools. |
| `wait_for_events` | Observe async facts (process exits). Opt-in. |

The set is declared once, in Go, and everything downstream is derived: the schema is
dumped to `runtime/schema/tools.json`, and what a model actually sees is **compiled**
from that plus a panel manifest (`harness/tool_panels/`). A manifest offering a
parameter the runtime does not accept fails at startup instead of misleading a model.
The default panel is `shell` + `text_editor`; `full` offers all seven model-facing
tools (`artifact` is machinery, never offered).

## Python Client

```bash
pip install ash-sandbox
```

```python
from ash_sandbox import Sandbox

async with Sandbox.connect("http://localhost:3000") as sb:
    await sb.call("shell", command="pytest")
    await sb.call("text_editor", command="str_replace", path="app.py",
                  old_str="foo", new_str="bar")

    # Get schemas for LLM function calling
    tools = await sb.tool_schemas(format="openai")    # or "anthropic"

    # Execute model tool_calls directly
    result = await sb.execute_tool_call(model_tool_call)
```

### Backends

| Backend | Constructor | Routing |
|---------|-------------|---------|
| HTTP | `Sandbox.connect(url)` | Direct to one runtime |
| MCP | `Sandbox.mcp(url)` | MCP protocol handshake |
| CLI | `Sandbox.local(bin)` | Subprocess stdio |
| Gateway | `SandboxPool(cp, gw)` | X-Sandbox-ID header |

## K8s Infrastructure

| Component | Language | Role |
|-----------|----------|------|
| Control Plane | Go | REST API for sandbox lifecycle (spawn/destroy) |
| Gateway | Go | Reverse proxy, routes by sandbox ID via Redis |
| Redis | - | Session → pod routing table |
| ash-runtime | Go | Runs inside each sandbox pod |

### Control Plane API

```bash
# Create a sandbox
curl -X POST http://control-plane/create -d '{
  "image": "my-image:latest",
  "ports": [{"container_port": 3000}],
  "resources": {"requests": {"cpu": "500m", "memory": "512Mi"}}
}'
# → {"uuid": "sandbox-abc123-...", "status": "ready", ...}

# Destroy specific sandboxes
curl -X DELETE http://control-plane/destroy -d '{"ids": ["sandbox-abc123-..."]}'

# Destroy all
curl -X DELETE http://control-plane/destroy -d '{"all": true}'
```

## Project Structure

```
.
├── runtime/              # Go sandbox runtime (ash-runtime binary)
│   ├── main.go           # HTTP server + stdio + MCP endpoint
│   ├── tools/            # the 8 tool implementations
│   ├── schema/tools.json # --dump-schema output; panels compile against it
│   └── events/           # notification system
├── sdk/ash_sandbox/      # Python async client (Sandbox, DockerPool, MicroVMPool)
├── harness/              # agent runtime — see harness/README.md
│   ├── orchestrator/     # the shape of one run
│   ├── execution/        # session, tool panel, MCP server, interceptors
│   ├── slots/            # claude-code, codex, opencode drivers
│   ├── gateway/          # model swap, accounting, enforced budget
│   ├── core/journal.py   # append-only JSONL: the canonical state
│   ├── rollback.py       # checkpoint pairing and fork plans
│   └── tool_panels/      # default, full, bash_only, no_web
├── swebench/             # the eval layer — four files
│   ├── fork_eval.py      # run -> grade -> branch on failure
│   ├── dataset.py        # instances and test commands
│   └── patch.py          # what belongs in a diff
├── k8s-scaffold/         # K8s infrastructure (Go)
│   ├── control-plane/    # sandbox lifecycle API
│   └── gateway/          # session-routed reverse proxy
├── k8s-config/           # K8s manifests (deploy, rbac, infra)
└── docs/                 # generated diagrams (gen_*.py, geometry-validated)
```

## License

MIT
