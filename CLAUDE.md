# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What Ash is

**Ash — Agent Sandbox Hive** is scalable sandbox infrastructure for LLM agents and RL
training. It provides isolated execution environments exposed through a small, fixed tool
protocol that an agent can drive over **HTTP**, **MCP**, or **stdio**. The same protocol
works for one local container or thousands of pods behind a gateway on Kubernetes.

There are four independently-versioned pieces:

| Piece            | Language | Path             | Role                                                        |
|------------------|----------|------------------|-------------------------------------------------------------|
| `ash-runtime`    | Go       | `runtime/`       | The binary that runs *inside* every sandbox; serves 8 tools |
| `ash-sandbox`    | Python   | `sdk/`           | Async client SDK to drive runtimes and pools                |
| SWE-bench harness| Python   | `swebench/`      | Evaluation harness (agent loop + benchmark runner)          |
| K8s control plane| Go       | `k8s-scaffold/`  | Control plane + gateway for fleet-scale sandboxes           |

> Note: the Python SDK package `ash_sandbox` lives under **`sdk/ash_sandbox/`**, not at the
> repo root (the README's tree is slightly abbreviated).

## Repository layout

```
.
├── runtime/                 # Go sandbox runtime (the ash-runtime binary)
│   ├── main.go              # JSON-RPC server: HTTP (/), MCP (/mcp), and stdio modes
│   ├── tools/               # 8 tool implementations (shell, process, text_editor,
│   │                        #   grep_files, web_fetch, web_search, artifact, wait_for_events)
│   ├── schema/tools.json    # `--dump-schema` output, checked in; the panel compiles
│   │                        #   against it, and a test fails when it goes stale
│   └── events/              # Notification system (e.g. process_exited)
├── sdk/                     # Python async client — package `ash_sandbox`
│   └── ash_sandbox/
│       ├── sandbox.py       # Sandbox: call(), tool_schemas(), execute_tool_call()
│       ├── backends.py      # HTTP / MCP / CLI / Gateway transports
│       ├── pool.py          # DockerPool (local) + MicroVMPool (Firecracker) + SandboxPool (k8s)
│       ├── result.py        # ToolResult
│       └── cli.py           # `ash-sandbox` console script
├── swebench/                # SWE-bench evaluation harness — see "The four layers"
│   ├── __main__.py          # CLI entry (`python -m swebench`), YAML config w/ `extends`
│   ├── agent/               # L2: the interceptor chain + this repo's agent loop
│   │   ├── pipeline.py      #   the onion: verdicts, CallContext, ToolPipeline
│   │   ├── interceptors/    #   one package each: guardrail, truncate, present
│   │   └── tools.py         #   panel compilation + routing (ToolPanel)
│   ├── harnesses/           # L3 topologies: litellm, claude-code (base.py = the API)
│   ├── configs/             # Per-model YAML (`extends:`) + tool_panels/
│   ├── sandbox.py           # AshSession: sandbox lifecycle + the executor seam
│   ├── mcp_server.py        # MCP proxy: the same chain, for external agents
│   ├── submission.py        # L4: asking the agent for its own patch
│   ├── batch.py             # Parallel execution + dashboard
│   └── dataset.py           # Loads SWE-bench instances
├── k8s-scaffold/            # K8s infrastructure (Go)
│   ├── control-plane/       # REST API for sandbox lifecycle (spawn/destroy), Redis store
│   └── gateway/             # Reverse proxy, routes by sandbox ID via Redis
├── k8s-config/              # K8s manifests + deploy.sh (rbac, infra, infra-local)
├── Makefile                 # Top-level: builds + deploys k8s stack on minikube
└── results/                 # SWE-bench run outputs (large; not source — do not edit by hand)
```

## The four layers

Each layer knows only the one below it, and the boundaries are load-bearing: several of
the defects in this repo's history were one layer answering a question that belonged to
another.

```
┌─ L4  EVAL ───────────────────────────────── swebench/{__main__,batch,dataset,
│  What counts as an answer. Only this layer            prediction,submission}.py
│  knows what SWE-bench is.
│    __main__.py    the only entry point; YAML + `extends` + flag overrides
│    submission.py  asks the agent for its own patch, reserving steps to do it in
│    prediction.py  the preds.json format the grader reads
└───────────────────────┬──────────────────────────────────────────────────────
      harness.run_instance(instance) -> prediction
┌─ L3  HARNESS ────────────────────────────── swebench/harnesses/
│  Topology: how many agents, how many worktrees, who reports the answer.
│    litellm       one agent, one sandbox — any litellm model
│    claude-code   the Claude Code CLI over MCP; reaches L2 as an external
│                  agent (its own subprocess) and uses none of the shared layer
└───────────────────────┬──────────────────────────────────────────────────────
      ⇩ the one seam:  executor(tool_name, args) -> ToolResult
┌─ L2  TOOL PATH ──────────────────────────── swebench/agent/
│  Two halves. Keep them apart when reading:
│
│    the interceptor chain     pipeline.py + interceptors/
│      An onion around one executor: `before` in order, `after` in reverse.
│      Four verdicts — Continue / Rewrite / Reject / ShortCircuit — and per
│      interceptor fail-open or fail-closed. Never raises; every failure comes
│      back as a ToolResult, because the caller above is an agent loop and an
│      escaped exception kills a whole run.
│      Ships guardrail (read-before-edit, edit streaks), truncate (bound one
│      result), present (compose the model's text from the runtime's report).
│      Mount your own: `default_pipeline(extra=[...])`, or
│      `execution.interceptors: my_file.py`.
│
│    the agent loop           __init__.py + llm, conversation, prompts, hooks,
│                             trace, tools, custom_tools, checkpoints,
│                             context_window
│      This repo's own agent. A consumer of the chain, not part of it —
│      sandbox.py and mcp_server.py mount the chain without it.
└───────────────────────┬──────────────────────────────────────────────────────
      JSON-RPC over HTTP / MCP / stdio
┌─ L1  RUNTIME (Go, inside the sandbox) ───── runtime/
│  Executes and reports. 8 tools, one fixed protocol.
└──────────────────────────────────────────────────────────────────────────────

Two mount points, one chain:      sandbox.py      this repo's agents
                                  mcp_server.py   external agents, over MCP
```

Things that are configuration rather than code, each with one module that owns the
decision:

| | config | owner |
|---|---|---|
| where sandboxes come from | `execution.backend`: docker / microvm / k8s | `backends.py` |
| what tools a model sees | `agent.tools` → a panel manifest | `agent/tools.py` |
| what governs each call | `execution.interceptors` → a Python file | `agent/pipeline.py` |
| which topology runs | `--harness` | `harnesses/__init__.py` |

No call site names a `Pool`, a tool schema, or an interceptor class directly.

### Scope, and why it keeps mattering

State that belongs to one run must not live at module level. Three separate bugs came
from that: the active tool panel, the agent→runtime routing table, and the custom-tool
registry were each a process-wide singleton, so two configurations in one process saw
each other's. A benchmark run never noticed (one configuration, all workers alike); the
rollout server, which builds a configuration per request, is where it would have shown.
An `AshSession` owns its tool registry, an `AshAgent` owns its panel.

## The tool protocol (the core contract)

`ash-runtime` serves **8 tools** over JSON-RPC. Changing this set is a breaking change.

| Tool              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `shell`           | Run a command. `background: true` returns a pid.               |
| `process`         | Read output of / kill a background process.                    |
| `text_editor`     | `view` / `write` / `str_replace` / `insert`.                   |
| `grep_files`      | Ripgrep search (pattern, glob, limit). Requires `ripgrep`.     |
| `web_fetch`       | Fetch a URL as html / text / markdown.                         |
| `web_search`      | Multi-engine search (Google, DuckDuckGo, Brave).               |
| `artifact`        | Fetch + verify a binary. Backs manifest-defined custom tools.  |
| `wait_for_events` | Observe async facts (process exits, tool calls). Opt-in.       |

The set is declared once, in Go, and everything downstream is derived from it:

```
runtime/tools/*.go  Schema()          ← authoritative; it executes the calls
        │  ash-runtime --dump-schema
        ▼
runtime/schema/tools.json             ← checked in, so a panel compiles with no sandbox
        │  + swebench/configs/tool_panels/*.yaml   (what to offer, and how)
        ▼
the panel a model sees                ← compiled; see docs/TOOL_PANEL.md
```

So adding a tool means editing the Go, regenerating `tools.json`, and naming it in a
panel manifest if a model should see it. A test fails if `tools.json` is stale, and
another fails if a panel offers a parameter the runtime does not accept — that check
exists because the hand-written panel it replaced had drifted on four of seven tools,
one of them offering `web_fetch(max_length=…)`, which the runtime silently ignored.

Not every runtime tool is offered to a model: `artifact` is machinery the SDK uses to
expand a custom tool, so a panel that listed it would hand the model a download
primitive.

Runtime run modes: `--mode http` (default, port 3000), `--mode stdio`, the `POST /mcp`
endpoint (MCP protocol version `2025-03-26`), and `--dump-schema` (print and exit).

## Common commands

### Runtime (Go)

```bash
cd runtime
go build -o ash-runtime .        # build (~9MB single binary, needs ripgrep at runtime)
go vet ./...                     # static analysis (run before committing)

./ash-runtime --port 3000        # HTTP server
./ash-runtime --mode stdio       # stdio JSON-RPC (CLI / MCP stdio transport)
```

Go version: **1.22**. There is no Go unit-test suite — correctness is enforced by the CI
integration tests (see `.github/workflows/ci.yml`), which drive the binary over stdio, HTTP,
and MCP. Mirror those manual JSON-RPC checks when changing tool behavior.

### Python SDK (`sdk/`)

```bash
pip install ./sdk                # installs the ash-sandbox package (editable: pip install -e ./sdk)
```

```python
from ash_sandbox import Sandbox, DockerPool, SandboxPool

async with Sandbox.connect("http://localhost:3000") as sb:
    r = await sb.call("shell", command="pytest")
    print(r.output)
    tools = await sb.tool_schemas(format="openai")   # or "anthropic"
    r = await sb.execute_tool_call(model_tool_call)  # run a model's tool_call directly
```

Backends: `Sandbox.connect(url)` (HTTP), `Sandbox.mcp(url)` (MCP), `Sandbox.local(bin)`
(subprocess stdio), `SandboxPool(control_plane_url, gateway_url)` (routes by `X-Sandbox-ID`).
Requires Python ≥ 3.10; only runtime dependency is `httpx`.

### SWE-bench evaluation (`swebench/`)

```bash
pip install litellm pyyaml datasets

# Single instance
python -m swebench -c swebench/configs/bedrock-sonnet46.yaml -i django__django-11848

# Full run (workers set in config or via -w)
python -m swebench -c swebench/configs/bedrock-sonnet46-bash.yaml

# Claude Code harness instead of the litellm loop
python -m swebench -c swebench/configs/claude-opus.yaml --harness claude-code

# Run the same harness on Firecracker microVMs instead of containers
python -m swebench -c swebench/configs/bedrock-sonnet46.yaml --backend microvm
```

- **Sandbox backend** is config, not code (`swebench/backends.py`): `docker` (default),
  `microvm` (AgentENV/Firecracker), `k8s`. Settings go under `execution.<backend>`;
  `microvm` also reads `AENV_SERVER_URL` / `AENV_API_KEY`. No call site names a pool —
  `AshSession` and the MCP proxy both build theirs from config.
- Configs are YAML and compose via `extends: <name>` (resolved against `configs/`); CLI flags
  override file values which override defaults. See `swebench/__main__.py` for the flag/section
  mapping.
- **The tool panel** is compiled, not written (`docs/TOOL_PANEL.md`): `agent.tools`
  names a manifest — a shipped one (`swebench/configs/tool_panels/bash_only.yaml`) or a path to your own.
  A manifest declares views of runtime tools (rename, offer a subset of parameters,
  reword them) and, under `custom_tools:`, external binaries. `custom_tools_dir` still
  loads a directory of one-tool manifests.
- **Interceptors** are config too: `execution.interceptors: my_file.py`, where the file
  exports `PIPELINE = [MyInterceptor()]`. They mount outside the defaults, so they see a
  call before truncation spends anything on it and still see calls the inner ones reject.
- Three harnesses today (`swebench/harnesses/__init__.py`): `litellm` (custom agent
  loop, any model), `claude-code` (Claude Code CLI via MCP, which reaches L2 as an
  external agent and shares none of this layer), and `marathon` (SWE-Marathon's
  ultra-long-horizon tasks). The marathon path differs in where work comes from and
  how it is graded, not in the loop: a task is a **directory** (`swebench/marathon.py`
  reads `task.toml` / `instruction.md` / `environment/` / `tests/`), its image is
  **built locally** because several tasks bake encrypted verification assets in and no
  published image exists, and the grade is the task's **own `tests/test.sh`** run
  verbatim -- its anti-cheat (PATH sanitization, encrypted expected outputs, library
  fingerprinting) is part of the specification. Both the binary reward and
  `partial_score` are recorded, because on tasks this long nearly every attempt scores
  zero and 7-of-43 must be distinguishable from none. Grading needs the fixtures
  inside the sandbox, which is why pools gained `upload_file` (binary, multi-megabyte:
  the tool surface cannot carry it). Add new ones by subclassing `BaseHarness` and
  registering in `HARNESSES`. `manager-worker` and `best-of-n` were removed while the
  single-agent path settles, along with Waggle, the write-arbitration interceptor they
  used; the mount they relied on (several agents, one shared chain) still works.
- **Per-step checkpoints** (`swebench/agent/checkpoints.py`, `swebench/replay.py`):
  `checkpoints.enabled: true` snapshots the environment each step so a rollout can
  be restarted from any of them. Only steps that could have mutated are captured
  (a `MutationTracker` interceptor decides; `shell` counts as mutating because a
  command's effect is not in its text) and the rest map to the previous snapshot,
  so the step→snapshot map is complete either way — it lands in the trajectory's
  `info.checkpoints` and, for the RL path, in each rollout reply. `mode: disk_only`
  (default) skips the memory image; such snapshots **cold-boot**, so the microVM
  template must declare a startup command that launches the runtime, or
  `mode: full` (resume) must be used instead. A template made with `aenv snapshot
  create` records no startup command -- build one through AgentENV's template API
  (`POST /v3/templates`, then `POST /v2/templates/{id}/builds/{buildID}` with
  `startCmd: /usr/local/bin/ash-runtime --port 3000` and a `readyCmd` that probes
  the port) so cold-booted sandboxes get their runtime back. `swap_sandbox` probes
  a replacement before adopting it, so a template without one costs a deeper chain
  rather than a dead episode. When a capture shows the server
  compacted the layer chain (the layer count drops), the session continues on a
  sandbox started from that snapshot — the live layer stack is never compacted, so
  without that every later capture would re-compact the chain. `MicroVMPool` gained
  `snapshot`/`squash`/`get_snapshot`; other pools declare `supports_snapshot()` false.
  **Every checkpoint writes the trajectory** (pass `trajectory_path=` to
  `install`): snapshots outlive an interrupted run, and without the step→snapshot
  map beside them the survivors are unusable. Saving used to happen only after a
  clean finish, which is how a 5-hour marathon run was killed leaving 300
  snapshots and no record of which step each was; persisting belongs to
  checkpointing rather than to each harness, so a new harness cannot forget it.
  `--resume-from <snapshot>` (marathon) continues a task from a checkpoint
  instead of rebuilding its image -- the environment carries the work, the
  transcript does not, and the prompt says so.
  Each trajectory (and each rollout reply, success or failure) also carries an
  `environment` block, because a SWE-bench image name is usually a mutable tag and
  cannot identify what a run ran against: `base_image` is the origin, pinned when
  the task starts and unchanged by re-boarding (digest-pinned when the episode
  cold-started from an image, e.g. `…@sha256:1e0a86…`); `base_ref` is what the
  *current* sandbox started from, which becomes a snapshot id after a re-board;
  plus the repository `base_commit` and the sandbox id.
  `replay.environment_mismatch` compares `base_image` and `base_commit` -- not
  `base_ref`, which differs on every replay by design.
- **Two spawn entries on `MicroVMPool`**: `spawn(image=…)` starts from a snapshot
  or template (what a replay or re-board needs); `spawn_from_image(…)` cold-starts
  from an OCI image reference and is the only path that accepts one.
  `microvm.from_image: true` says this harness's image names are references to
  cold-start. Note the requirement that follows: a cold-started image must already
  bring up the runtime, and AgentENV runs no startup command for a plain image --
  so a benchmark whose per-instance images are raw (SWE-bench's are) needs a
  template built per instance rather than a raw cold start.
- **On-demand per-image templates** (`swebench/templates.py`): set
  `microvm.runtime_bin` to a local ash-runtime binary and `AshSession.create`
  turns each image into a template automatically -- cold-start the image, upload
  the binary through the backend's file service (envd, port 49983 via the proxy;
  the runtime cannot be used to install itself), snapshot (disk-only, unnamed so
  a retry never collides with a half-failed attempt's leftover), then build a
  template from that snapshot with a `RUN chmod` step, `startCmd`, and a
  port-probing `readyCmd`. Template names are content-addressed over (image,
  runtime-binary hash, port), so a batch builds each distinct image once and a
  rebuilt runtime gets fresh templates. Two lookup endpoints matter:
  `/snapshots/{name}` answers only for sandbox-sourced snapshots, so built
  (template-sourced) templates are checked via `/templates/aliases/{name}` --
  asking the wrong one makes every built template look missing and the rebuild
  collides with it. Names that are grammatically impossible as snapshot aliases
  (image references carry `/':'@`) skip the catalog lookup; a name the backend
  already knows (a replay's checkpoint snapshot id) passes through untouched.
  ripgrep is baked in alongside the runtime (downloaded once per host into
  `~/.cache/ash-swebench/`, part of the template's content-addressed identity):
  without it, every sandbox's *first* `grep_files` apt-gets rg -- measured at
  ~15s and +89 MiB of disk writes (the apt package indexes), all landing in the
  episode's first checkpoint. With it: 0.5s, +2 MiB. The runtime's own
  provisioning (for sandboxes without a baked template) was also reordered:
  static tarball first (arch-aware, verified by *running* rg -- LookPath
  accepted a wrong-arch binary), package managers as fallbacks that clean
  their indexes after installing. Measured: 0.5s/+7 MiB with a fetcher in the
  image, 4s/+33 MiB via slimmed apt on a bare image (was 15s/+89 MiB).
- **Context window** is managed at two seams, and they are not redundant:
  `TruncateInterceptor` (L2, tool path) bounds what each single result costs and
  also protects external agents through the MCP proxy; `context_window.py`'s
  `before_query` guard folds *accumulated* old tool outputs once the transcript
  passes budget. Flow vs stock. Elision, not summarization: assistant turns (what
  the agent did) stay verbatim, old tool outputs (what it saw, mostly re-obtainable)
  become one-line stubs; it cuts in bulk to a low target rather than one message
  per step, because rewriting old messages invalidates the prompt cache. Two
  strategies, selected by `execution.context_strategy`: `elide` (default; free,
  cannot invent) or `summarize` (one model call per firing writes the span's
  findings into the stub). Measured on a real 133-step transcript, both hit the
  same 23%-of-original target; elision cost 0.2s and nothing, the summary cost
  17.7s and $0.09 and kept facts the agent would otherwise re-derive (the exact
  build command, that `xxd`/`python3` are absent, the expected-hash table). A
  summary that fails or comes back empty falls back to elision, and its cost is
  charged to the run's own budget rather than hidden. Defaults differ by horizon
  on purpose: the `marathon` harness summarizes (measured: 7 of 8 sampled facts
  vanish under elision, tool calls included, because on that horizon the facts
  live only in tool output), while benchmark runs elide (at 30 steps nothing
  folds anyway).
  Both the budget and the measurement come from the provider, never from a guess:
  the budget is a fraction of the model's own `max_input_tokens` (litellm model
  metadata — a real 133-step run measured ~139K input tokens, fatal against a 200K
  window and 14% of a 1M one), and the count is `litellm.token_counter`, exact to a
  token and inclusive of tool calls. Character estimates were 2-3x off in both
  directions; a cheap char pre-gate only decides whether the tokenizer pass is
  needed. The protected recent tail is a floor: a target below it is unreachable,
  and the guard traces that instead of appearing to succeed.
- **Trajectories record `tool_calls`** on assistant messages (JSON-flattened by
  `conversation.plain_tool_calls`). They used to be dropped, which cost the record
  the agent's actual actions — a replay could see that it said it would edit a file
  but not what edit — and made any accounting of what the model saw understate it,
  since one `text_editor` write carries a whole file in its arguments.
- Output lands in `results/<run>/`. Treat `results/` as generated data.

### Kubernetes stack

```bash
# Local end-to-end on minikube (build images + apply manifests)
make all-local

# Or manifests only
cd k8s-config && bash deploy.sh
```

Control plane API (Go, `k8s-scaffold/control-plane/`):

```bash
curl -X POST   http://control-plane/create  -d '{"image":"my:latest","ports":[{"container_port":3000}]}'
curl -X DELETE http://control-plane/destroy -d '{"ids":["sandbox-abc123-..."]}'
curl -X DELETE http://control-plane/destroy -d '{"all":true}'
```

The gateway routes requests to the right pod by sandbox ID using a Redis routing table.

## Conventions & guardrails

- **The tool set is declared once, in Go.** Change `runtime/tools/`, then regenerate
  `runtime/schema/tools.json` (`./ash-runtime --dump-schema > runtime/schema/tools.json`)
  and name the tool in a panel manifest if a model should see it. This used to be a rule
  asking three hand-written copies to be edited together; the panel is compiled now and
  CI fails on a stale schema or a panel that offers a parameter the runtime rejects.
- **Go**: `gofmt`/`goimports` clean, wrap errors with `%w`, run `go vet ./...`. Small
  interfaces, accept interfaces / return structs.
- **Python**: PEP 8, type annotations on signatures, prefer immutable dataclasses; the SDK is
  fully `async`/`await`.
- **Don't hand-edit `results/`** — it is benchmark output (hundreds of MB).
- **Validate the runtime over all three transports** (stdio, HTTP, MCP) after touching tools,
  matching the CI steps in `.github/workflows/ci.yml`.
- An existing Claude Code skill for driving Ash sandboxes lives at
  `.claude/skills/ash/` (see `SKILL.md` and `tools-reference.md`).

## License

Runtime/repo: MIT. Python SDK (`sdk/`): Apache-2.0.
