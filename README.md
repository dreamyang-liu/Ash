# StationHive

StationHive is a scalable workstation cluster for LLM Agents & RL Rollouts. It provides a Kubernetes-based infrastructure for creating, managing, and accessing isolated sandbox environments through a unified gateway.

## Overview

StationHive is designed to create isolated, ephemeral workstation environments that can be used for:
- LLM agent operations with tool access
- Reinforcement learning training and rollouts
- Algorithm experimentation and testing
- Scalable AI workloads

The architecture consists of the following key components:

- **Control Plane**: Manages the lifecycle of sandbox environments (creation, monitoring, deletion)
- **Gateway**: Routes requests to the appropriate sandbox based on session identifiers
- **MCP (Model Context Protocol) Servers**: Provides tools and resources for the sandboxed environments

## Repository Structure

```
├── Makefile                # Root makefile to build all components
├── k8s-config/             # Kubernetes configuration files
│   ├── deploy.sh           # Deployment script
│   ├── infra.yaml          # Infrastructure configuration
│   ├── Makefile            # Config-specific makefile
│   ├── rbac.yaml           # Role-based access control configuration
│   └── stateless-mcps.yaml # Stateless MCP services configuration
├── k8s-scaffold/           # Core infrastructure components
│   ├── Makefile            # Scaffold-specific makefile
│   ├── control-plane/      # Control plane service
│   │   ├── Dockerfile      # Container definition
│   │   ├── go.mod          # Go module definition
│   │   ├── go.sum          # Go dependencies
│   │   └── main.go         # Control plane implementation
│   └── gateway/            # API gateway service
│       ├── Dockerfile      # Container definition
│       ├── go.mod          # Go module definition
│       ├── go.sum          # Go dependencies
│       └── main.go         # Gateway implementation
├── sandbox-recipe/         # Sandbox environment definitions
│   ├── Makefile            # Recipe-specific makefile
│   └── general/            # General-purpose sandbox
│       ├── Dockerfile      # Container definition
│       ├── main.py         # Main entry point
│       └── ddgs_mcp/       # DuckDuckGo search MCP implementation
│           ├── .gitignore
│           ├── .python-version
│           ├── main.py     # DDGS MCP implementation
│           ├── pyproject.toml
│           ├── README.md
│           └── uv.lock
└── example/                # Example scripts
    ├── spawn_sandbox.py    # Script to create sandbox environments
    ├── deprovision_sandbox.py # Script to clean up sandbox environments
    └── mcp_example.py      # Example MCP usage
```

## Components

### Control Plane

The control plane is responsible for:

- Creating and managing sandbox environments
- Allocating resources based on requests
- Monitoring the health of sandbox environments
- Cleaning up environments when they expire or are no longer needed

The control plane exposes a REST API for creating, managing, and deprovisioning sandbox environments.

### Gateway

The gateway serves as the entry point for all requests to sandbox environments:

- Routes requests to the appropriate sandbox based on session headers
- Handles authentication and authorization
- Provides load balancing and failover capabilities
- Manages session affinity

### MCP (Model Context Protocol) Servers

MCP servers provide tools and resources to the sandbox environments, enabling LLM agents to access external capabilities:

- **sandbox-fusion-mcp**: A specialized MCP server for fusion operations
- **ddgs_mcp**: Provides search capabilities using DuckDuckGo search API
- **terminal-controller**: Enables terminal access and command execution
- **fetch**: Provides web content fetching capabilities

## Setup and Deployment

### Prerequisites

- Kubernetes cluster (Minikube for local development)
- Docker
- Go 1.x
- Python 3.x

### Building Components

To build all components:

```bash
make
```

This will build:
1. The control plane and gateway containers
2. The sandbox environment containers

### Local Deployment with Minikube

For a local development setup using Minikube:

```bash
make all-local
```

This single command will:
1. Start Minikube with LoadBalancer support (via MetalLB)
2. Build all necessary Docker images directly in the Minikube environment
3. Apply modified configuration files adapted for local development
4. Deploy all components to your local Minikube cluster

After deployment, you can access the services using:

```bash
# Get URLs for accessing services
minikube service control-plane -n apps --url
minikube service gateway -n apps --url
```

Alternatively, use port-forwarding for local access:

```bash
# Forward the control-plane service to localhost:8080
minikube kubectl -- -n apps port-forward svc/control-plane 8080:80

# In another terminal, forward the gateway service to localhost:8081
minikube kubectl -- -n apps port-forward svc/gateway 8081:80
```

### Standard Deployment

For deployment to a standard Kubernetes cluster:

```bash
cd k8s-config
./deploy.sh
```

This will:
1. Create the necessary Kubernetes namespace
2. Apply RBAC configurations
3. Deploy Redis for state management
4. Deploy the control plane and gateway services
5. Deploy the stateless MCP servers

## Using the Sandbox

### Creating a Sandbox Environment

To create a new sandbox environment, send a POST request to the control plane:

```bash
curl -X POST http://localhost:8080/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "image": "sandbox-general",
    "ports": [{"container_port": 3000}],
    "expose": "LoadBalancer",
    "env": {"PARAM1": "value1"},
    "replicas": 1
  }'
```

The response will include a UUID that can be used to access the sandbox environment.

#### Using the Python Example Script

The repository includes a Python script for creating sandboxes:

```bash
# Run the spawn script
python example/spawn_sandbox.py
```

This script will:
1. Automatically retrieve the control-plane service URL from Minikube
2. Send a request to create a new sandbox with the general-purpose image
3. Print the response, including the UUID needed to access the sandbox

### Accessing a Sandbox Environment

To access the sandbox environment, send requests to the gateway with the session UUID:

```bash
curl -H "X-MCP-Session-ID: <uuid>" http://gateway-endpoint/path
```

### Using MCP Tools in a Sandbox

The `example/mcp_example.py` script demonstrates how to use MCP tools in a sandbox:

```bash
# Update the UUID in the script first
nano example/mcp_example.py  # Edit the UUID from your spawn response

# Run the MCP example
python example/mcp_example.py
```

This example will:
1. Connect to the sandbox through the gateway using the specified UUID
2. List all available tools provided by the MCP servers
3. Execute a terminal command (`ls`) in the sandbox
4. Display the results

### Cleaning Up

To deprovision all sandbox environments:

```bash
curl -X DELETE http://localhost:8080/deprovision-all
```

To deprovision a specific sandbox:

```bash
curl -X DELETE http://localhost:8080/deprovision/<uuid>
```

You can also use the included Python script for cleanup:

```bash
# Clean up all sandboxes
python example/deprovision_sandbox.py
```

## MCP Servers

### DDGS Search MCP

The DDGS MCP provides search capabilities with the following tools:

- `search`: General web search with configurable parameters
- `search_images`: Image search with filtering options
- `search_videos`: Video search with duration and resolution filters
- `search_news`: News search for current events
- `search_books`: Book search using Anna's Archive digital library

## Configuration

Configuration is primarily done through environment variables. Key configuration options include:

### Control Plane

- `TARGET_NAMESPACE`: Kubernetes namespace for sandbox deployments (default: "apps")
- `REDIS_HOST`: Redis host for state management
- `REDIS_PORT`: Redis port
- `REDIS_DB`: Redis database index
- `SANDBOX_MAX_TTL_SEC`: Maximum TTL for sandbox environments in seconds

### Gateway

- `LISTEN_ADDR`: Gateway listen address (default: ":8080")
- `SESSION_HEADER`: Header for session identification (default: "X-MCP-Session-ID")
- `REDIS_ADDR`: Redis address for route table lookups
- `REDIS_DB`: Redis database index
- `REDIS_KEY_PREFIX`: Prefix for Redis keys (default: "sandbox:")

## License

[Include license information here]
