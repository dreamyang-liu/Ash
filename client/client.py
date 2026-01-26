"""
Sandbox Client - Spin up, destroy, and connect to sandboxes.

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
# Resource Configuration
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


# =============================================================================
# Sandbox Configuration
# =============================================================================

@dataclass
class SandboxConfig:
    """
    Configuration for sandbox client and containers.

    One config = one sandbox. Create multiple clients with the same config
    to manage multiple sandboxes. Different configs can connect to different
    clusters.

    Attributes:
        control_plane_url: URL of the control plane API
        gateway_url: URL of the MCP gateway
        image: Docker image for sandbox containers
        ports: Container ports to expose (default: [3000] for FastMCP)
        env: Environment variables to pass to the container
        resources: CPU/Memory requests and limits
        node_selector: Kubernetes node selector labels
        timeout: Timeout for spawn/destroy operations (seconds)
        mcp_timeout: Timeout for MCP client operations (seconds)

    Example:
        # Basic config
        config = SandboxConfig(
            control_plane_url="http://control-plane:80",
            gateway_url="http://gateway:80",
        )

        # Full config with resources
        config = SandboxConfig(
            control_plane_url="http://control-plane:80",
            gateway_url="http://gateway:80",
            image="custom-sandbox:latest",
            env={"DEBUG": "true", "API_KEY": "..."},
            resources=ResourceReq(
                requests=ResourceSpec(cpu="100m", memory="256Mi"),
                limits=ResourceSpec(cpu="500m", memory="512Mi"),
            ),
            node_selector={"gpu": "true"},
        )

        # Multi-cluster: use different configs
        config_us = SandboxConfig(control_plane_url="http://us-cluster:80", ...)
        config_eu = SandboxConfig(control_plane_url="http://eu-cluster:80", ...)
    """
    # Connection URLs
    control_plane_url: str = ""
    gateway_url: str = ""

    # Container settings
    image: str = "timemagic/rl-mcp:general-1.7"
    ports: List[int] = field(default_factory=lambda: [3000])
    env: Dict[str, str] = field(default_factory=dict)
    resources: ResourceReq = field(default_factory=ResourceReq)
    node_selector: Dict[str, str] = field(default_factory=dict)

    # Timeouts
    timeout: int = 300
    mcp_timeout: int = 180


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
        host: Internal DNS name ({name}.{namespace}.svc.cluster.local)
        ports: Service ports
    """
    uuid: str
    name: str
    namespace: str
    status: str
    host: str = ""
    ports: List[int] = field(default_factory=list)
    message: str = ""
    _gateway_url: str = field(default="", repr=False)

    @property
    def mcp_url(self) -> str:
        """MCP endpoint URL for this sandbox."""
        return f"{self._gateway_url}/mcp" if self._gateway_url else ""

    @property
    def is_ready(self) -> bool:
        return self.status.lower() == "ready"

    @classmethod
    def from_response(cls, data: dict, gateway_url: str = "") -> "Sandbox":
        return cls(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            namespace=data.get("namespace", ""),
            status=data.get("status", ""),
            host=data.get("host", ""),
            ports=data.get("ports", []),
            message=data.get("message", ""),
            _gateway_url=gateway_url,
        )

    def __repr__(self) -> str:
        return f"Sandbox(uuid={self.uuid!r}, status={self.status!r}, host={self.host!r})"


# =============================================================================
# Sandbox Client
# =============================================================================

class SandboxClient:
    """
    Client for managing a single sandbox lifecycle and MCP connection.

    One client manages one sandbox. For multiple sandboxes, create multiple
    clients (with the same or different configs).

    Args:
        config: SandboxConfig with connection URLs and container settings

    Example:
        # Single sandbox with context manager (auto-cleanup)
        config = SandboxConfig(
            control_plane_url="http://control-plane:80",
            gateway_url="http://gateway:80",
        )

        with SandboxClient(config) as client:
            sandbox = client.spawn()
            mcp = client.connect()
            async with mcp:
                tools = await mcp.list_tools()
        # Sandbox automatically destroyed on exit

        # Multiple sandboxes (reuse config)
        clients = [SandboxClient(config) for _ in range(10)]
        sandboxes = [c.spawn() for c in clients]

        # Multi-cluster
        config_a = SandboxConfig(control_plane_url="http://cluster-a:80", ...)
        config_b = SandboxConfig(control_plane_url="http://cluster-b:80", ...)
        client_a = SandboxClient(config_a)
        client_b = SandboxClient(config_b)
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._mcp_client = None
        self._sandbox: Optional[Sandbox] = None
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def __enter__(self) -> "SandboxClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._sandbox:
            try:
                self.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy sandbox on close: {e}")
        self._session.close()

    def __repr__(self) -> str:
        sandbox_info = f", sandbox={self._sandbox.uuid!r}" if self._sandbox else ""
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
        Spawn a new sandbox container using the client's config.

        Args:
            name: Optional custom name (auto-generated if not provided)

        Returns:
            Sandbox object with connection details

        Raises:
            SandboxSpawnError: If spawn fails after retries
            SandboxTimeoutError: If spawn times out
        """
        url = f"{self.config.control_plane_url}/spawn"

        # Build request matching Go SpawnReq struct
        data: Dict[str, Any] = {
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
        resources_dict: Dict[str, Any] = {}
        if self.config.resources.requests.cpu or self.config.resources.requests.memory:
            resources_dict["requests"] = {}
            if self.config.resources.requests.cpu:
                resources_dict["requests"]["cpu"] = self.config.resources.requests.cpu
            if self.config.resources.requests.memory:
                resources_dict["requests"]["memory"] = self.config.resources.requests.memory

        if self.config.resources.limits.cpu or self.config.resources.limits.memory:
            resources_dict["limits"] = {}
            if self.config.resources.limits.cpu:
                resources_dict["limits"]["cpu"] = self.config.resources.limits.cpu
            if self.config.resources.limits.memory:
                resources_dict["limits"]["memory"] = self.config.resources.limits.memory

        if resources_dict:
            data["resources"] = resources_dict

        try:
            response = self._session.post(url, json=data, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise SandboxTimeoutError(f"Spawn request timed out after {self.config.timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise SandboxSpawnError(f"Spawn failed: {e.response.text if e.response else e}") from e

        result = response.json()
        self._sandbox = Sandbox.from_response(result, self.config.gateway_url)

        logger.info(f"Sandbox spawned: {self._sandbox.uuid} (status: {self._sandbox.status})")
        if self._sandbox.message:
            logger.debug(f"Spawn message: {self._sandbox.message}")

        return self._sandbox

    @retry(max_attempts=3, initial_delay=1.0)
    def destroy(self) -> dict:
        """
        Destroy the client's sandbox.

        Returns:
            Response dict with message and uuid

        Raises:
            SandboxDestroyError: If destroy fails after retries
        """
        if not self._sandbox:
            raise ValueError("No sandbox to destroy - call spawn() first")

        url = f"{self.config.control_plane_url}/deprovision/{self._sandbox.uuid}"

        try:
            response = self._session.delete(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise SandboxTimeoutError("Destroy request timed out") from e
        except requests.exceptions.HTTPError as e:
            raise SandboxDestroyError(f"Destroy failed: {e.response.text if e.response else e}") from e

        result = response.json()
        uuid = self._sandbox.uuid
        self._sandbox = None

        logger.info(f"Sandbox destroyed: {uuid}")
        return result

    @retry(max_attempts=2, initial_delay=2.0)
    def destroy_all(self) -> dict:
        """
        Destroy all sandboxes in the namespace (use with caution).

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
        self._sandbox = None

        count = result.get('count', 0)
        logger.info(f"Destroyed {count} sandboxes")
        if result.get('failed'):
            logger.warning(f"Failed to destroy: {result['failed']}")

        return result

    def connect(self):
        """
        Get an MCP client connected to this client's sandbox.

        Returns:
            FastMCP Client configured for the sandbox

        Raises:
            ValueError: If no sandbox has been spawned

        Example:
            mcp = client.connect()
            async with mcp:
                tools = await mcp.list_tools()
                result = await mcp.call_tool("tool_name", {"arg": "value"})
        """
        from fastmcp import Client

        if not self._sandbox:
            raise ValueError("No sandbox to connect to - call spawn() first")

        mcp_config = {
            "mcpServers": {
                "sandbox": {
                    "transport": "http",
                    "url": f"{self.config.gateway_url}/mcp",
                    "headers": {"X-Session-ID": self._sandbox.uuid},
                }
            },
        }
        self._mcp_client = Client(mcp_config, timeout=self.config.mcp_timeout)
        return self._mcp_client

    @property
    def sandbox(self) -> Optional[Sandbox]:
        """The client's sandbox, if spawned."""
        return self._sandbox
