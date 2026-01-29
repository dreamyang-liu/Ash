<p align="center">
  <img src="logo.svg" alt="Ash Logo" width="280">
</p>

<h1 align="center">Ash</h1>
<p align="center"><em>Agent Sandbox Hive</em></p>

<p align="center">
  <strong>Scalable sandbox cluster for LLM Agents and Agent RL Rollouts</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Go-00ADD8?style=flat&logo=go&logoColor=white" alt="Go">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white" alt="Kubernetes">
  <img src="https://img.shields.io/badge/MCP-FF6B00?style=flat" alt="MCP">
  <img src="https://img.shields.io/badge/Kata_Containers-FFA500?style=flat" alt="Kata Containers">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
</p>

---

Ash provides **on-demand, isolated Kubernetes-based sandbox environments** for AI agents. Agents connect via [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to execute code, browse the web, and run tools safely. Spawn thousands of sandboxes dynamically with automatic routing, resource limits, and lifecycle management.

> **🔥 Kata Containers Branch**: This branch adds support for [Kata Containers](https://katacontainers.io/) to run sandboxes in lightweight microVMs (Firecracker/Cloud Hypervisor) for VM-level isolation. See [docs/kata-containers.md](docs/kata-containers.md) for details.

<br>

<table>
<tr>
<td width="33%" valign="top">

**Isolated Execution**

Each agent gets its own container with resource limits, network isolation, and automatic cleanup.

</td>
<td width="33%" valign="top">

**MCP Native**

First-class support for Model Context Protocol. Plug in terminal, web fetch, search, and custom tools.

</td>
<td width="33%" valign="top">

**Scale to Thousands**

Dynamic provisioning on Kubernetes. Auto-routing via Redis. Works on EKS, GKE, or local.

</td>
</tr>
</table>

<br>

## Quick Start

> If you don't have an existing Kubernetes cluster, see [Deployment](#deployment) first.

### Basic Usage

```python
import asyncio
from client import SandboxClient, SandboxConfig

async def main():
    config = SandboxConfig(
        control_plane_url="http://your-control-plane:80",
        gateway_url="http://your-gateway:80",
    )

    # Context manager handles cleanup automatically
    with SandboxClient(config) as client:
        # Spawn a sandbox
        sandbox = client.spawn()
        print(f"Sandbox ready: {sandbox.uuid}")

        # Connect via MCP and use tools
        mcp = client.connect()
        async with mcp:
            tools = await mcp.list_tools()
            print(f"Tools: {[t.name for t in tools]}")

            # Execute a shell command
            result = await mcp.call_tool(
                "terminal-controller_execute_command",
                {"command": "echo 'Hello from sandbox!'"}
            )
            print(result.content[0].text)

asyncio.run(main())
```

See [client/demo.py](./client/demo.py) for a complete benchmark example.

<br>

### Using MCP Tools

```python
mcp = client.connect()
async with mcp:
    # Terminal - execute commands
    await mcp.call_tool("terminal-controller_execute_command", {
        "command": "python --version"
    })

    # Fetch - get web content
    await mcp.call_tool("fetch_fetch", {
        "url": "https://example.com"
    })

    # Search - web search
    await mcp.call_tool("ddgs_search_mcp_search", {
        "query": "python asyncio tutorial",
        "max_results": 5
    })
```

<br>

### Available MCP Tools

The default image `timemagic/rl-mcp:general-1.7` includes:

| Tool | Description |
|:-----|:------------|
| `terminal-controller` | Execute shell commands in the sandbox |
| `fetch` | Fetch web content from URLs |
| `ddgs_search` | Web search via DuckDuckGo |

**Bring your own tools** — provide a custom image with your MCP server. See [sandbox-recipe/](./sandbox-recipe/) for examples.

<br>

---

## Architecture

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                      Kubernetes                          │
┌─────────┐                      │                                                          │
│         │   POST /spawn        │  ┌───────────────┐          ┌───────────────────┐        │
│  Agent  │─────────────────────▶│  │ Control Plane │─────────▶│    Sandbox Pod    │        │
│         │                      │  └───────┬───────┘  Create  │  ┌─────────────┐  │        │
└────┬────┘                      │          │                  │  │  FastMCP    │  │        │
     │                           │          │ Store route      │  │  Server     │  │        │
     │                           │          ▼                  │  └─────────────┘  │        │
     │                           │  ┌───────────────┐          └────────▲──────────┘        │
     │                           │  │     Redis     │                   │                   │
     │                           │  │               │                   │                   │
     │                           │  └───────▲───────┘                   │                   │
     │   MCP + X-Session-ID      │          │ Lookup                    │ Proxy             │
     │                           │  ┌───────┴───────┐                   │                   │
     └──────────────────────────▶│  │    Gateway    │───────────────────┘                   │
                                 │  └───────────────┘                                       │
                                 └──────────────────────────────────────────────────────────┘
```

<br>

### Components

| Component | Language | Description |
|:----------|:---------|:------------|
| **Control Plane** | Go | REST API for spawning/destroying sandbox pods |
| **Gateway** | Go | Routes MCP requests using `X-Session-ID` header |
| **Sandbox** | Python | Isolated container running FastMCP server |
| **Redis** | — | Session → sandbox routing table |

<br>

---

## Deployment

### Prerequisites

- Kubernetes cluster (EKS, GKE, or Minikube)
- `kubectl` configured
- Docker (for custom images)

<br>

### AWS EKS

<details>
<summary><strong>1. Create EKS Nodegroups</strong></summary>

```bash
# Infrastructure nodegroup (control-plane, gateway, redis)
eksctl create nodegroup \
  --cluster your-cluster \
  --name infra \
  --node-type m5.large \
  --nodes 3 \
  --node-labels "eks.amazonaws.com/nodegroup=infra"

# Sandbox nodegroup (where sandbox pods run)
eksctl create nodegroup \
  --cluster your-cluster \
  --name sandbox \
  --node-type m5.xlarge \
  --nodes-min 0 \
  --nodes-max 100 \
  --node-labels "eks.amazonaws.com/nodegroup=sandbox"
```

</details>

<details>
<summary><strong>2. Build Images (Optional)</strong></summary>

Skip if using pre-built images.

```bash
cd k8s-scaffold
make build

docker push timemagic/ash:control-plane-0.1
docker push timemagic/ash:gateway-0.1
```

</details>

<details>
<summary><strong>3. Deploy to Kubernetes</strong></summary>

```bash
cd k8s-config

# Create namespace and RBAC
kubectl apply -f rbac.yaml

# Deploy infrastructure
kubectl apply -f infra.yaml

# Wait for ready
kubectl -n ash rollout status deploy/redis
kubectl -n ash rollout status deploy/control-plane
kubectl -n ash rollout status deploy/gateway
```

</details>

<details>
<summary><strong>4. Get Service URLs</strong></summary>

```bash
kubectl -n ash get svc control-plane gateway

# Example output:
# NAME            TYPE           EXTERNAL-IP                              PORT(S)
# control-plane   LoadBalancer   abc123.us-west-2.elb.amazonaws.com       80:31234/TCP
# gateway         LoadBalancer   xyz789.us-west-2.elb.amazonaws.com       80:31235/TCP
```

</details>

<details>
<summary><strong>5. Configure Client</strong></summary>

```python
from client import SandboxClient, SandboxConfig

config = SandboxConfig(
    control_plane_url="http://abc123.us-west-2.elb.amazonaws.com",
    gateway_url="http://xyz789.us-west-2.elb.amazonaws.com",
    node_selector={"eks.amazonaws.com/nodegroup": "sandbox"},
)

client = SandboxClient(config)
sandbox = client.spawn()
```

</details>

<br>

### Local (Minikube)

```bash
# Start minikube and deploy everything
make all-local

# Get service URLs
minikube service control-plane -n ash --url
minikube service gateway -n ash --url
```

<br>

---

## Configuration

### SandboxConfig

```python
from client import SandboxClient, SandboxConfig, ResourceReq, ResourceSpec

config = SandboxConfig(
    # Connection URLs
    control_plane_url="http://control-plane:80",
    gateway_url="http://gateway:80",

    # Container settings
    image="custom-sandbox:latest",
    ports=[3000],
    env={"DEBUG": "true", "API_KEY": "..."},
    resources=ResourceReq(
        requests=ResourceSpec(cpu="100m", memory="256Mi"),
        limits=ResourceSpec(cpu="500m", memory="512Mi"),
    ),
    node_selector={"gpu": "true"},

    # Timeouts
    timeout=300,
    mcp_timeout=180,
)

client = SandboxClient(config)
```

<br>

### Config Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `control_plane_url` | — | Control plane endpoint (required) |
| `gateway_url` | — | Gateway endpoint for MCP (required) |
| `image` | `timemagic/rl-mcp:general-1.7` | Sandbox container image |
| `ports` | `[3000]` | Ports to expose |
| `env` | `{}` | Environment variables |
| `resources` | `None` | CPU/memory requests & limits |
| `node_selector` | `{}` | Kubernetes node selector |
| `timeout` | `300` | Spawn/destroy timeout (seconds) |
| `mcp_timeout` | `180` | MCP call timeout (seconds) |

<br>

### Node Scheduling

```python
# GPU nodes
config = SandboxConfig(..., node_selector={"gpu": "true"})

# Specific EKS nodegroup
config = SandboxConfig(..., node_selector={"eks.amazonaws.com/nodegroup": "sandbox"})

# Specific instance type
config = SandboxConfig(..., node_selector={"node.kubernetes.io/instance-type": "m5.large"})
```

<br>

### Multiple Sandboxes

```python
# Reuse config for multiple sandboxes
config = SandboxConfig(
    control_plane_url="http://control-plane:80",
    gateway_url="http://gateway:80",
)

# Each client manages one sandbox
clients = [SandboxClient(config) for _ in range(10)]
sandboxes = [c.spawn() for c in clients]
```

<br>

---

## Repository Structure

```
ash/
├── client/           # Python client library
├── example/          # Usage examples
├── k8s-scaffold/     # Control plane & gateway (Go)
├── k8s-config/       # Kubernetes manifests
└── sandbox-recipe/   # Sandbox container images
```

<br>

---

## License

Apache 2.0 — See [LICENSE](./LICENSE) for details.

<br>

---

<p align="center">
  <sub>Built for scalable AI agent infrastructure</sub>
</p>
