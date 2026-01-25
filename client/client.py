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
import time
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Dict, List, Callable, TypeVar, Any
from fastmcp import Client

# Configure module logger
logger = logging.getLogger(__name__)

# Type variable for retry decorator
T = TypeVar("T")


# =============================================================================
# Custom Exceptions
# =============================================================================

class SandboxError(Exception):
    """Base exception for sandbox operations."""
    pass


class SandboxSpawnError(SandboxError):
    """Failed to spawn a sandbox."""
    pass


class SandboxDestroyError(SandboxError):
    """Failed to destroy a sandbox."""
    pass


class SandboxConnectionError(SandboxError):
    """Failed to connect to sandbox or control plane."""
    pass


class SandboxTimeoutError(SandboxError):
    """Operation timed out."""
    pass


# =============================================================================
# Retry Decorator
# =============================================================================

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (requests.exceptions.RequestException,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exceptions that trigger a retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # Check if it's a non-retryable HTTP error (4xx)
                        if isinstance(e, requests.exceptions.HTTPError):
                            if e.response is not None and 400 <= e.response.status_code < 500:
                                raise  # Don't retry client errors

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")

            raise last_exception  # type: ignore
        return wrapper
    return decorator


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
        control_plane_url: URL of the control plane that manages K8s deployments
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


    # # Control plane URLs
    control_plane_url: str = ""
    gateway_url: str = ""

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
    mcp_timeout: int = 180

    # Wait for ready settings (usually not needed since control-plane waits for readiness probe)
    wait_for_ready: bool = False  # Control-plane now waits for pod readiness probe
    wait_for_ready_timeout: int = 120  # Additional wait time if status is "starting"
    wait_for_ready_interval: float = 2.0  # Initial polling interval (uses exponential backoff)


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

    def __repr__(self) -> str:
        return f"Sandbox(uuid={self.uuid!r}, status={self.status!r}, host={self.host!r})"


# =============================================================================
# Sandbox Client
# =============================================================================

class SandboxClient:
    """
    Client for managing sandbox lifecycle and MCP connections.

    Supports context manager for automatic cleanup:
        with SandboxClient() as client:
            sandbox = client.spawn()
            mcp = client.connect()
            async with mcp:
                tools = await mcp.list_tools()
        # Sandbox automatically destroyed on exit

    Or manual usage:
        client = SandboxClient()
        sandbox = client.spawn()

        mcp = client.connect(sandbox.uuid)
        async with mcp:
            tools = await mcp.list_tools()
            result = await mcp.call_tool("tool_name", {"arg": "value"})

        client.destroy(sandbox.uuid)
        client.close()  # Close HTTP session
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._mcp_client: Optional[Client] = None
        self._current_sandbox: Optional[Sandbox] = None
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def __enter__(self) -> "SandboxClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._current_sandbox:
            try:
                self.destroy(self._current_sandbox.uuid)
            except Exception as e:
                logger.warning(f"Failed to destroy sandbox on close: {e}")
        self._session.close()

    def __repr__(self) -> str:
        sandbox_info = f", sandbox={self._current_sandbox.uuid!r}" if self._current_sandbox else ""
        return f"SandboxClient(control_plane={self.config.control_plane_url!r}{sandbox_info})"

    def health_check(self) -> bool:
        """Check if control plane is healthy."""
        try:
            response = self._session.get(
                f"{self.config.control_plane_url}/healthz",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def ready_check(self) -> bool:
        """Check if control plane is ready (including Redis)."""
        try:
            response = self._session.get(
                f"{self.config.control_plane_url}/readyz",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.debug(f"Ready check failed: {e}")
            return False

    @retry(max_attempts=3, initial_delay=2.0)
    def spawn(self, name: Optional[str] = None) -> Sandbox:
        """
        Spin up a new sandbox container.

        Args:
            name: Optional custom name (auto-generated if not provided)

        Returns:
            Sandbox object with connection details

        Raises:
            SandboxSpawnError: If spawn fails after retries
            SandboxTimeoutError: If wait_for_ready is True and sandbox doesn't
                become ready within wait_for_ready_timeout seconds

        Note:
            The control-plane now waits for the pod's readiness probe before
            returning "Ready" status. The wait_for_ready option is disabled by
            default since it's no longer needed.
        """
        url = f"{self.config.control_plane_url}/spawn"

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

        try:
            response = self._session.post(
                url,
                json=data,
                timeout=self.config.spawn_timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise SandboxTimeoutError(f"Spawn request timed out after {self.config.spawn_timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise SandboxSpawnError(f"Spawn failed: {e.response.text if e.response else e}") from e

        result = response.json()
        sandbox = Sandbox.from_response(result, self.config)
        self._current_sandbox = sandbox

        logger.info(f"Sandbox spawned: {sandbox.uuid} (status: {sandbox.status})")
        if sandbox.message:
            logger.debug(f"Spawn message: {sandbox.message}")

        # Wait for MCP server to be ready if enabled
        # Note: K8s "Ready" status only means deployment has replicas, not that MCP is listening
        if self.config.wait_for_ready:
            sandbox = self._wait_for_ready(sandbox)

        return sandbox

    def _wait_for_ready(self, sandbox: Sandbox) -> Sandbox:
        """
        Wait for sandbox to be ready.

        Note: With the readiness probe on sandbox pods, the control-plane now
        waits for the pod to be ready before returning. This method is kept
        for backward compatibility but is disabled by default.
        """
        # If already ready, return immediately
        if sandbox.is_ready:
            return sandbox

        logger.info(f"Waiting for sandbox {sandbox.uuid} to be ready...")
        logger.debug(f"Status: {sandbox.status} (control-plane returned before pod was ready)")

        # Since control-plane returns early only if deployment timed out,
        # we just wait a bit and hope it becomes ready
        deadline = time.time() + self.config.wait_for_ready_timeout
        interval = self.config.wait_for_ready_interval
        max_interval = 10.0

        while time.time() < deadline:
            remaining = deadline - time.time()
            sleep_time = min(interval, remaining, max_interval)
            if sleep_time <= 0:
                break

            logger.debug(f"Waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            interval = min(interval * 1.5, max_interval)

        # We can't actually verify readiness without the gateway endpoint,
        # so we just assume it's ready after waiting
        sandbox.status = "ready"
        logger.info(f"Sandbox {sandbox.uuid} assumed ready after waiting")
        return sandbox

    @retry(max_attempts=3, initial_delay=1.0)
    def destroy(self, sandbox_uuid: Optional[str] = None) -> dict:
        """
        Destroy a sandbox container.

        Args:
            sandbox_uuid: UUID to destroy (uses current sandbox if not provided)

        Returns:
            Response dict with message and uuid

        Raises:
            SandboxDestroyError: If destroy fails after retries
        """
        uuid = sandbox_uuid or (self._current_sandbox.uuid if self._current_sandbox else None)
        if not uuid:
            raise ValueError("No sandbox UUID provided and no current sandbox")

        # API uses DELETE /deprovision/:uuid
        url = f"{self.config.control_plane_url}/deprovision/{uuid}"

        try:
            response = self._session.delete(url, timeout=self.config.destroy_timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise SandboxTimeoutError(f"Destroy request timed out after {self.config.destroy_timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise SandboxDestroyError(f"Destroy failed: {e.response.text if e.response else e}") from e

        result = response.json()

        if self._current_sandbox and self._current_sandbox.uuid == uuid:
            self._current_sandbox = None

        logger.info(f"Sandbox destroyed: {uuid}")
        return result

    @retry(max_attempts=2, initial_delay=2.0)
    def destroy_all(self) -> dict:
        """
        Destroy all sandboxes created by control plane.

        Returns:
            Response dict with deleted list, failed list, and count

        Raises:
            SandboxDestroyError: If destroy_all fails after retries
        """
        url = f"{self.config.control_plane_url}/deprovision-all"

        try:
            response = self._session.delete(url, timeout=120)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise SandboxTimeoutError("Destroy all request timed out") from e
        except requests.exceptions.HTTPError as e:
            raise SandboxDestroyError(f"Destroy all failed: {e.response.text if e.response else e}") from e

        result = response.json()
        self._current_sandbox = None

        count = result.get('count', 0)
        logger.info(f"Destroyed {count} sandboxes")
        if result.get('failed'):
            logger.warning(f"Failed to destroy: {result['failed']}")

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
