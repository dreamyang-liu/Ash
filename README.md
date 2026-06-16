<p align="center">
  <img src="logo.svg" width="120" alt="Ash Logo"/>
</p>

<h1 align="center">Ash — Agent Sandbox Hive</h1>

<p align="center">
  Scalable sandbox infrastructure for LLM agents and RL training.<br/>
  Isolated execution environments with a minimal tool protocol over HTTP, MCP, or stdio.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#tools">Tools</a> •
  <a href="#python-client">Client</a> •
  <a href="#k8s-infrastructure">K8s Deploy</a>
</p>

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Agent / Training Loop                                                │
│    Sandbox.connect(url)     → direct HTTP                            │
│    Sandbox.mcp(url)         → MCP Streamable HTTP                    │
│    Sandbox.local(bin)       → subprocess stdio                       │
│    DockerPool(bin)          → local multi-container                   │
│    SandboxPool(cp, gw)      → K8s gateway-routed                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌──────────────────┐  ┌──────────────┐  ┌────────────────────┐
│ ash-runtime      │  │ Gateway (Go) │  │ Control Plane (Go) │
│ (per sandbox)    │  │ Redis routing│  │ K8s lifecycle       │
│                  │  │ X-Session-ID │  │ spawn/destroy pods  │
│ POST /   JSON-RPC│  └──────┬───────┘  └────────────────────┘
│ POST /mcp  MCP   │         │
│ --mode stdio     │         │
│                  │◄────────┘
│ 7 tools:         │
│  shell, process  │
│  read_file       │
│  text_editor     │
│  grep_files      │
│  web_fetch       │
│  web_search      │
└──────────────────┘
```

## Quick Start

### Local (single sandbox)

```bash
cd runtime
go build -o ash-runtime .
./ash-runtime --port 3000
```

```python
from client import Sandbox

async with Sandbox.connect("http://localhost:3000") as sb:
    result = await sb.call("shell", command="echo hello")
    print(result.output)
```

### Local Docker (multiple sandboxes)

```python
from client import DockerPool

async with DockerPool(runtime_bin="./ash-runtime") as pool:
    sb1 = await pool.spawn(image="python:3.11")
    sb2 = await pool.spawn(image="node:20")
    await sb1.call("shell", command="pytest")
    await sb2.call("shell", command="npm test")
```

### Kubernetes (large-scale RL)

```bash
# Deploy infrastructure
cd k8s-config && bash deploy.sh

# Or manually
kubectl apply -f infra.yaml
kubectl apply -f rbac.yaml
```

```python
from client import SandboxPool

async with SandboxPool(
    control_plane_url="http://control-plane:80",
    gateway_url="http://gateway:80",
) as pool:
    sandboxes = [await pool.spawn(image="my-task:latest") for _ in range(100)]
    # Each sandbox is a K8s pod, routed via gateway
```

## Runtime

The `ash-runtime` binary runs inside each sandbox container. It's a single Go binary (~9MB) with no dependencies beyond `ripgrep` (auto-installed on first grep call).

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
| `read_file` | Read files with line numbers, offset, limit. |
| `text_editor` | View, create, str_replace, insert files. |
| `grep_files` | Ripgrep search with pattern, glob, limit. |
| `web_fetch` | Fetch URLs. Formats: html, text, markdown. |
| `web_search` | Multi-engine search (Google → DuckDuckGo → Brave). |

### Notifications

Every `tools/call` response includes a `notifications` array. Background process exits and file changes are automatically captured and delivered with the next response:

```json
{
  "content": [{"type": "text", "text": "..."}],
  "isError": false,
  "notifications": [
    {"kind": "process_exited", "data": {"pid": "abc123", "exit_code": 0}}
  ]
}
```

## Python Client

```bash
pip install httpx
```

```python
from client import Sandbox

async with Sandbox.connect("http://localhost:3000") as sb:
    # Direct tool calls
    await sb.call("shell", command="pytest")
    await sb.call("text_editor", command="str_replace", path="app.py", old_str="foo", new_str="bar")

    # Get schemas for LLM function calling
    tools = await sb.tool_schemas(format="openai")    # or "anthropic"

    # Execute model tool_calls directly
    result = await sb.execute_tool_call(model_tool_call)
    # result.output, result.is_error, result.notifications
```

### Backends

| Backend | Constructor | Routing |
|---------|-------------|---------|
| HTTP | `Sandbox.connect(url)` | Direct to one runtime |
| MCP | `Sandbox.mcp(url)` | MCP protocol handshake |
| CLI | `Sandbox.local(bin)` | Subprocess stdio |
| Gateway | `SandboxPool(cp, gw)` | X-Session-ID header |

## K8s Infrastructure

### Components

| Component | Language | Role |
|-----------|----------|------|
| Control Plane | Go | REST API for sandbox lifecycle (spawn/destroy) |
| Gateway | Go | Reverse proxy, routes by session ID via Redis |
| Redis | - | Session → pod routing table |
| ash-runtime | Go | Runs inside each sandbox pod |

### Control Plane API

```bash
# Spawn a sandbox
curl -X POST http://control-plane/spawn -d '{
  "image": "my-image:latest",
  "ports": [{"container_port": 3000}],
  "resources": {"requests": {"cpu": "500m", "memory": "512Mi"}}
}'
# → {"uuid": "sandbox-abc123-...", "status": "ready", ...}

# Destroy
curl -X POST http://control-plane/destroy -d '{"uuid": "sandbox-abc123-..."}'
```

### Gateway

Routes all tool requests to the correct sandbox pod:

```bash
curl -X POST http://gateway/ \
  -H "X-Session-ID: sandbox-abc123-..." \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"shell","arguments":{"command":"ls"}}}'
```

## Development

```bash
# Build runtime
cd runtime && go build -o ash-runtime .

# Run tests
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"shell","arguments":{"command":"echo ok"}}}' | ./ash-runtime --mode stdio

# Build K8s components
cd k8s-scaffold && make build
```

## Project Structure

```
.
├── runtime/              # Go sandbox runtime (ash-runtime binary)
│   ├── main.go           # HTTP server + stdio + MCP endpoint
│   ├── tools/            # 7 tool implementations
│   ├── events/           # Notification system
│   └── client/           # Python async client
├── k8s-scaffold/         # K8s infrastructure (Go)
│   ├── control-plane/    # Sandbox lifecycle API
│   └── gateway/          # Session-routed reverse proxy
├── k8s-config/           # K8s manifests (deploy, rbac, infra)
└── src/                  # Legacy Rust implementation (ash-mcp)
```

## License

Apache-2.0
