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
| `ash-runtime`    | Go       | `runtime/`       | The binary that runs *inside* every sandbox; serves 7 tools |
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
│   ├── tools/               # 7 tool implementations (shell, process, readfile, edit, grep, web)
│   └── events/              # Notification system (e.g. process_exited)
├── sdk/                     # Python async client — package `ash_sandbox`
│   └── ash_sandbox/
│       ├── sandbox.py       # Sandbox: call(), tool_schemas(), execute_tool_call()
│       ├── backends.py      # HTTP / MCP / CLI / Gateway transports
│       ├── pool.py          # DockerPool (local) + MicroVMPool (Firecracker) + SandboxPool (k8s)
│       ├── result.py        # ToolResult
│       └── cli.py           # `ash-sandbox` console script
├── swebench/                # SWE-bench evaluation harness
│   ├── __main__.py          # CLI entry (`python -m swebench`), YAML config loader w/ `extends`
│   ├── agent/               # Agent loop + L2 pipeline: conversation, llm, tools, prompts,
│   │                        #   pipeline/interceptors/guardrails/waggle, hooks
│   ├── harnesses/           # Pluggable backends: litellm, claude-code (base.py defines API)
│   ├── configs/             # Per-model YAML configs (inherit via `extends:`)
│   ├── batch.py / runner.py # Parallel execution + dashboard
│   └── dataset.py           # Loads SWE-bench instances
├── k8s-scaffold/            # K8s infrastructure (Go)
│   ├── control-plane/       # REST API for sandbox lifecycle (spawn/destroy), Redis store
│   └── gateway/             # Reverse proxy, routes by sandbox ID via Redis
├── k8s-config/              # K8s manifests + deploy.sh (rbac, infra, infra-local)
├── Makefile                 # Top-level: builds + deploys k8s stack on minikube
└── results/                 # SWE-bench run outputs (large; not source — do not edit by hand)
```

## The tool protocol (the core contract)

`ash-runtime` exposes exactly **6 tools** over JSON-RPC. Changing this set is a breaking
change — keep the Go implementation, the SDK, and the SWE-bench agent tool list in sync.

| Tool          | Purpose                                                            |
|---------------|-------------------------------------------------------------------|
| `shell`       | Run a command. `background: true` returns a pid.                  |
| `process`     | Read output of / kill a background process.                       |
| `text_editor` | `view` / `write` / `str_replace` / `insert`.                     |
| `grep_files`  | Ripgrep search (pattern, glob, limit). Requires `ripgrep`.        |
| `web_fetch`   | Fetch a URL as html / text / markdown.                           |
| `web_search`  | Multi-engine search (Google, DuckDuckGo, Brave).                 |

Runtime run modes: `--mode http` (default, port 3000), `--mode stdio`, and the `POST /mcp`
endpoint (MCP protocol version `2025-03-26`).

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
- Two harnesses today (`swebench/harnesses/__init__.py`): `litellm` (custom agent loop, any
  model) and `claude-code` (Claude Code CLI via MCP). Add new ones by subclassing
  `BaseHarness` and registering in `HARNESSES`.
- Output lands in `results/<harness>/...`. Treat `results/` as generated data.

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

- **Keep the three tool views in sync.** Any change to the 7-tool set or a tool's
  arguments must land in `runtime/tools/`, the SDK (`sdk/ash_sandbox/`), and the SWE-bench
  agent tool list together.
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
