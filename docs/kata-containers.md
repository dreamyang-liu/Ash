# Kata Containers Integration for Ash

This document describes how to use [Kata Containers](https://katacontainers.io/) with Ash for enhanced sandbox isolation using microVMs.

## Overview

Kata Containers provides hardware-virtualized isolation for containers using lightweight VMs (microVMs). Instead of sharing the host kernel, each sandbox runs in its own VM with a dedicated kernel, providing an additional layer of security.

### Benefits
- **Stronger Isolation**: VM-level isolation vs container namespace isolation
- **Defense in Depth**: Even if a container escape vulnerability exists, the attacker is still inside a VM
- **Kernel Isolation**: Each sandbox has its own kernel, protecting against kernel exploits
- **Ideal for Untrusted Workloads**: Perfect for running untrusted agent code

### Trade-offs
- **Higher Overhead**: ~160-256MB extra memory per sandbox
- **Slower Startup**: ~1-2 seconds extra startup time
- **Resource Requirements**: Requires nested virtualization or bare-metal with KVM

## Supported Hypervisors

| RuntimeClass | Hypervisor | Boot Time | Memory | Best For |
|--------------|------------|-----------|--------|----------|
| `kata-fc` | Firecracker | ~125ms | ~160MB | Speed, serverless |
| `kata-clh` | Cloud Hypervisor | ~200ms | ~180MB | Features, hotplug |
| `kata-qemu` | QEMU | ~500ms | ~256MB | Compatibility |

**Recommendation**: Use `kata-fc` for Ash sandboxes unless you need specific QEMU features.

## Installation

### 1. Install Kata Containers on Your Cluster

```bash
# Apply Kata RBAC
kubectl apply -f https://raw.githubusercontent.com/kata-containers/kata-containers/main/tools/packaging/kata-deploy/kata-rbac/base/kata-rbac.yaml

# Deploy Kata (installs runtime on all nodes)
kubectl apply -f https://raw.githubusercontent.com/kata-containers/kata-containers/main/tools/packaging/kata-deploy/kata-deploy/base/kata-deploy.yaml

# Wait for installation to complete
kubectl -n kube-system wait --for=condition=Ready pod -l name=kata-deploy --timeout=300s

# Verify installation
kubectl get runtimeclass
```

### 2. Apply Ash RuntimeClasses

```bash
kubectl apply -f k8s-config/kata-runtimeclass.yaml
```

### 3. Verify Node Labels

Kata-deploy automatically labels nodes:
```bash
kubectl get nodes -l katacontainers.io/kata-runtime=true
```

## Usage

### Python Client

```python
from client import SandboxClient, SandboxConfig

# Create config with Kata runtime
config = SandboxConfig(
    control_plane_url="http://your-control-plane:80",
    gateway_url="http://your-gateway:80",
    runtime_class="kata-fc",  # Use Firecracker microVM
)

with SandboxClient(config) as client:
    sandbox = client.spawn()
    print(f"Sandbox running in microVM: {sandbox.uuid}")
    
    mcp = client.connect()
    async with mcp:
        # Tools run inside an isolated microVM
        result = await mcp.call_tool(
            "terminal-controller_execute_command",
            {"command": "uname -a"}  # Shows Kata's guest kernel
        )
```

### Direct API

```bash
curl -X POST http://control-plane/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "image": "timemagic/rl-mcp:general-1.7",
    "runtime_class": "kata-fc",
    "ports": [{"container_port": 3000}]
  }'
```

### Choosing a Runtime

```python
# Firecracker - fastest, lowest overhead
config = SandboxConfig(runtime_class="kata-fc", ...)

# Cloud Hypervisor - more features
config = SandboxConfig(runtime_class="kata-clh", ...)

# QEMU - maximum compatibility
config = SandboxConfig(runtime_class="kata-qemu", ...)

# Default (no Kata, standard container)
config = SandboxConfig(runtime_class="", ...)
```

## Resource Considerations

When using Kata, account for VM overhead in your resource requests:

```python
config = SandboxConfig(
    runtime_class="kata-fc",
    resources=ResourceReq(
        requests=ResourceSpec(
            cpu="250m",      # +250m for Kata overhead
            memory="512Mi",  # +160Mi for Kata overhead
        ),
        limits=ResourceSpec(
            cpu="1000m",
            memory="1Gi",
        ),
    ),
)
```

## Troubleshooting

### Pods Pending with Kata RuntimeClass

1. Check if nodes are labeled:
   ```bash
   kubectl get nodes -l katacontainers.io/kata-runtime=true
   ```

2. Check Kata installation:
   ```bash
   kubectl -n kube-system logs -l name=kata-deploy
   ```

3. Verify RuntimeClass exists:
   ```bash
   kubectl get runtimeclass kata-fc
   ```

### Performance Issues

- **Slow startup**: Normal for Kata; consider pre-warming sandboxes
- **High memory**: Reduce sandbox count or use larger node instance types
- **CPU overhead**: The ~250m overhead is mostly idle; real workloads are efficient

### Nested Virtualization (Cloud VMs)

If running on cloud VMs, enable nested virtualization:

- **AWS**: Use `.metal` instances (m5.metal, m6i.metal, etc.)
- **GCP**: Enable nested virtualization on the instance
- **Azure**: Use Dv3/Ev3 series with nested virtualization

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Node                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  MicroVM 1  │  │  MicroVM 2  │  │  MicroVM 3  │     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │
│  │ │Container│ │  │ │Container│ │  │ │Container│ │     │
│  │ │(Sandbox)│ │  │ │(Sandbox)│ │  │ │(Sandbox)│ │     │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │
│  │ Guest Kernel│  │ Guest Kernel│  │ Guest Kernel│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                │             │
│  ┌──────┴────────────────┴────────────────┴──────┐     │
│  │              Kata Runtime (containerd)         │     │
│  └────────────────────────────────────────────────┘     │
│                           │                             │
│  ┌────────────────────────┴────────────────────────┐   │
│  │           VMM (Firecracker/Cloud-Hypervisor)     │   │
│  └──────────────────────────────────────────────────┘   │
│                           │                             │
│  ┌────────────────────────┴────────────────────────┐   │
│  │                   KVM (Host Kernel)              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## References

- [Kata Containers Documentation](https://katacontainers.io/docs/)
- [Firecracker](https://firecracker-microvm.github.io/)
- [Cloud Hypervisor](https://www.cloudhypervisor.org/)
- [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/)
