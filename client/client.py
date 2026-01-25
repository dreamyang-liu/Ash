"""
Sandbox Client - Spin up, destroy, and connect to sandboxes for agent_loop.py

Control Plane API Reference (from Go server):
  POST /spawn              - Create new sandbox
  DELETE /deprovision/:uuid - Destroy sandbox by UUID
  DELETE /deprovision-all  - Destroy all sandboxes
  GET /healthz             - Health check
  GET /readyz              - Readiness check
"""
import requests
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from fastmcp import Client


# =============================================================================
# Configuration - All configurable parameters
# =============================================================================

@dataclass
class ResourceSpec:
    """Resource specification for CPU/Memory."""
    cpu: str = ""
    memory: str = ""


@dataclass
class ResourceReq:
    """Resource requests and limits."""
    requests: ResourceSpec = field(default_factory=ResourceSpec)
    limits: ResourceSpec = field(default_factory=ResourceSpec)


@dataclass
class SandboxConfig:
    """
    All configurable parameters for sandbox management.

    Control Plane URLs:
        spawner_url: URL of the control plane that manages K8s deployments
        gateway_url: URL of the MCP gateway for sandbox connections

    Container Settings:
        image: Docker image to use for sandbox containers
        ports: List of container ports to expose (default: [3000])

    Resource Limits:
        resources: CPU/Memory requests and limits

    Node Selection:
        node_selector: K8s node selector labels

    Environment:
        env: Environment variables to pass to container

    Timeouts:
        spawn_timeout: Timeout for spawn requests (seconds)
        destroy_timeout: Timeout for destroy requests (seconds)
        mcp_timeout: Timeout for MCP client operations (seconds)

    Note:
        - Service type is always ClusterIP (internal only)
        - Replicas is always 1 (single instance per sandbox)
    """
    # Control plane URLs
    spawner_url: str = "http://a8b4ee4606659412ca97fd13254655e1-1332289600.us-west-2.elb.amazonaws.com"
    gateway_url: str = "http://a8ba3420c90c64b2da804891c7d96d2f-1217634634.us-west-2.elb.amazonaws.com"

    # Container settings
    image: str = "timemagic/rl-mcp:general-1.7"
    ports: List[int] = field(default_factory=lambda: [3000])

    # Resource limits (optional)
    resources: ResourceReq = field(default_factory=ResourceReq)

    # Node selection (optional)
    node_selector: Dict[str, str] = field(default_factory=dict)

    # Environment variables (optional)
    env: Dict[str, str] = field(default_factory=dict)

    # Timeouts
    spawn_timeout: int = 300  # 5 minutes (control plane waits up to 120s for deploy ready)
    destroy_timeout: int = 30
    mcp_timeout: int = 60


# =============================================================================
# Sandbox Response Model
# =============================================================================

@dataclass
class Sandbox:
    """
    Represents a spawned sandbox instance.

    Attributes from control plane response:
        uuid: Unique identifier (format: {name}-{uuid})
        name: Deployment name
        namespace: K8s namespace
        status: "Ready" or "Starting"
        service_type: Always "ClusterIP" (internal only)
        cluster_ip: Internal cluster IP
        host: Internal DNS name ({name}.{namespace}.svc.cluster.local)
        ports: Service ports
        message: Additional status message
    """
    uuid: str
    name: str
    namespace: str
    status: str
    service_type: str
    cluster_ip: str = ""
    host: str = ""
    external_ip: str = ""
    external_hostname: str = ""
    ports: List[int] = field(default_factory=list)
    node_ports: List[int] = field(default_factory=list)
    message: str = ""
    config: SandboxConfig = field(default_factory=SandboxConfig)
    raw_response: dict = field(default_factory=dict)

    @property
    def mcp_url(self) -> str:
        return f"{self.config.gateway_url}/mcp"
    
    @property
    def is_ready(self) -> bool:
        return self.status.lower() == "ready"

    @classmethod
    def from_response(cls, data: dict, config: SandboxConfig) -> "Sandbox":
        return cls(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            namespace=data.get("namespace", ""),
            status=data.get("status", ""),
            service_type=data.get("service_type", ""),
            cluster_ip=data.get("cluster_ip", ""),
            host=data.get("host", ""),
            external_ip=data.get("external_ip", ""),
            external_hostname=data.get("external_hostname", ""),
            ports=data.get("ports", []),
            node_ports=data.get("node_ports", []),
            message=data.get("message", ""),
            config=config,
            raw_response=data,
        )


# =============================================================================
# Sandbox Client
# =============================================================================

class SandboxClient:
    """
    Client for managing sandbox lifecycle and MCP connections.
    
    Usage:
        client = SandboxClient()
        
        # Spawn a new sandbox
        sandbox = client.spawn()
        
        # Connect via MCP
        mcp = client.connect(sandbox.uuid)
        async with mcp:
            tools = await mcp.list_tools()
            result = await mcp.call_tool("tool_name", {"arg": "value"})
        
        # Cleanup
        client.destroy(sandbox.uuid)
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._mcp_client: Optional[Client] = None
        self._current_sandbox: Optional[Sandbox] = None

    def health_check(self) -> bool:
        """Check if control plane is healthy."""
        try:
            response = requests.get(
                f"{self.config.spawner_url}/healthz",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def ready_check(self) -> bool:
        """Check if control plane is ready (including Redis)."""
        try:
            response = requests.get(
                f"{self.config.spawner_url}/readyz",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def spawn(self, name: Optional[str] = None) -> Sandbox:
        """
        Spin up a new sandbox container.
        
        Args:
            name: Optional custom name (auto-generated if not provided)
        
        Returns:
            Sandbox object with connection details
        """
        url = f"{self.config.spawner_url}/spawn"
        
        # Build request matching Go SpawnReq struct
        data = {
            "image": self.config.image,
            "ports": [{"container_port": p} for p in self.config.ports],
        }
        
        if name:
            data["name"] = name
        
        if self.config.env:
            data["env"] = self.config.env
        
        if self.config.node_selector:
            data["node_selector"] = self.config.node_selector
        
        # Add resources if specified
        resources = {}
        if self.config.resources.requests.cpu or self.config.resources.requests.memory:
            resources["requests"] = {}
            if self.config.resources.requests.cpu:
                resources["requests"]["cpu"] = self.config.resources.requests.cpu
            if self.config.resources.requests.memory:
                resources["requests"]["memory"] = self.config.resources.requests.memory
        
        if self.config.resources.limits.cpu or self.config.resources.limits.memory:
            resources["limits"] = {}
            if self.config.resources.limits.cpu:
                resources["limits"]["cpu"] = self.config.resources.limits.cpu
            if self.config.resources.limits.memory:
                resources["limits"]["memory"] = self.config.resources.limits.memory
        
        if resources:
            data["resources"] = resources
        
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=self.config.spawn_timeout
        )
        response.raise_for_status()
        
        result = response.json()
        sandbox = Sandbox.from_response(result, self.config)
        self._current_sandbox = sandbox
        
        print(f"Sandbox spawned: {sandbox.uuid} (status: {sandbox.status})")
        if sandbox.message:
            print(f"  Message: {sandbox.message}")
        
        return sandbox

    def destroy(self, sandbox_uuid: Optional[str] = None) -> dict:
        """
        Destroy a sandbox container.
        
        Args:
            sandbox_uuid: UUID to destroy (uses current sandbox if not provided)
        
        Returns:
            Response dict with message and uuid
        """
        uuid = sandbox_uuid or (self._current_sandbox.uuid if self._current_sandbox else None)
        if not uuid:
            raise ValueError("No sandbox UUID provided and no current sandbox")
        
        # API uses DELETE /deprovision/:uuid
        url = f"{self.config.spawner_url}/deprovision/{uuid}"
        
        response = requests.delete(url, timeout=self.config.destroy_timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if self._current_sandbox and self._current_sandbox.uuid == uuid:
            self._current_sandbox = None
        
        print(f"Sandbox destroyed: {uuid}")
        return result

    def destroy_all(self) -> dict:
        """
        Destroy all sandboxes created by control plane.
        
        Returns:
            Response dict with deleted list, failed list, and count
        """
        url = f"{self.config.spawner_url}/deprovision-all"
        
        response = requests.delete(url, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        self._current_sandbox = None
        
        print(f"Destroyed {result.get('count', 0)} sandboxes")
        if result.get('failed'):
            print(f"  Failed: {result['failed']}")
        
        return result

    def connect(self, sandbox_uuid: Optional[str] = None) -> Client:
        """
        Get an MCP client connected to a sandbox.
        
        Args:
            sandbox_uuid: UUID to connect to (uses current sandbox if not provided)
        
        Returns:
            FastMCP Client configured for the sandbox
        """
        uuid = sandbox_uuid or (self._current_sandbox.uuid if self._current_sandbox else None)
        if not uuid:
            raise ValueError("No sandbox UUID provided and no current sandbox")
        
        mcp_config = {
            "mcpServers": {
                "sandbox": {
                    "transport": "http",
                    "url": f"{self.config.gateway_url}/mcp",
                    "headers": {"X-Session-ID": uuid},
                }
            },
        }
        self._mcp_client = Client(mcp_config, timeout=self.config.mcp_timeout)
        return self._mcp_client

    @property
    def current_sandbox(self) -> Optional[Sandbox]:
        return self._current_sandbox


# =============================================================================
# Demo / CLI
# =============================================================================

async def demo():
    """Demo usage of SandboxClient."""
    client = SandboxClient()
    
    # Health check
    print(f"Control plane healthy: {client.health_check()}")
    print(f"Control plane ready: {client.ready_check()}")
    
    # Spawn a new sandbox
    sandbox = client.spawn()
    print(f"Created sandbox: {sandbox.uuid}")
    print(f"  Host: {sandbox.host}")
    print(f"  Status: {sandbox.status}")
    print(f"  Ports: {sandbox.ports}")
    
    # Connect to it
    mcp = client.connect()
    
    async with mcp:
        await mcp.ping()
        print("Connected to sandbox MCP")
        
        tools = await mcp.list_tools()
        print(f"Available tools ({len(tools)}): {[t.name for t in tools[:5]]}...")
        
        result = await mcp.call_tool(
            "terminal-controller_execute_command",
            {"command": "echo 'Hello from sandbox!'"}
        )
        print(f"Command result: {result.content[0].text if result.content else result}")
    
    # Cleanup
    client.destroy()


if __name__ == "__main__":
    asyncio.run(demo())
