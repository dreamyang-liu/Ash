from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import shutil
import subprocess
from typing import Any

import httpx

from .backends import GatewayBackend, HTTPBackend
from .result import ToolResult
from .sandbox import Sandbox, _find_free_port


@dataclass(frozen=True)
class Snapshot:
    """A persistent snapshot, plus the facts needed to manage its chain.

    A snapshot is a stack of layers: the base image, then one layer per
    capture. `rootfs_layers` matters to callers because it is *inherited* --
    a sandbox started from this snapshot can never compact those layers, only
    add to them. Two decisions read it:

    - the count dropping between consecutive snapshots of one sandbox means
      the server compacted the chain, so that snapshot is a compact base and
      the sandbox should be replaced by one started from it (`re-board`);
    - a high count at branch time means children would inherit a deep prefix,
      so the snapshot is worth squashing first.
    """

    id: str
    names: tuple[str, ...] = ()
    rootfs_layers: int | None = None
    memory_layers: int | None = None
    chain_size_mb: int | None = None
    #: True when this snapshot carries no memory image, so sandboxes created
    #: from it cold-boot instead of resuming.
    disk_only: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, body: dict[str, Any], *, disk_only: bool = False) -> "Snapshot":
        return cls(
            id=body["snapshotID"],
            names=tuple(body.get("names") or ()),
            rootfs_layers=body.get("rootfsLayerCount"),
            memory_layers=body.get("memoryLayerCount"),
            chain_size_mb=body.get("chainSizeMB"),
            disk_only=disk_only,
            raw=body,
        )


class Pool(ABC):
    """A source of sandboxes.

    Implementations differ in where a sandbox comes from -- a local Docker
    daemon, a control plane, a microVM host later on -- but a harness that
    only needs "give me a sandbox, and tear it down afterwards" should not
    have to know which. Bookkeeping lives here so an implementation supplies
    just the create and remove steps for its own transport.
    """

    #: Live sandboxes by id, maintained by implementations.
    _sandboxes: dict[str, Sandbox]

    @abstractmethod
    async def spawn(
        self,
        image: str | None = None,
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
        agent_id: str = "",
    ) -> Sandbox:
        """Obtain a new sandbox.

        agent_id binds the returned handle's identity, so every call made
        through it is attributed to that agent. Worth passing whenever more
        than one agent will share the sandbox: the runtime keeps a per-identity
        cursor over the event log and anonymous callers share one, so two
        unnamed agents silently split events between them.

        Ids need only be distinct within a sandbox -- each runtime owns its own
        event log -- so one agent per sandbox may reuse the same name.
        """

    @abstractmethod
    async def destroy(self, *sandboxes: Sandbox) -> None:
        """Tear down specific sandboxes."""

    @abstractmethod
    async def destroy_all(self) -> None:
        """Tear down everything this pool created."""

    def list(self) -> list[Sandbox]:
        """Sandboxes this pool currently holds."""
        return list(self._sandboxes.values())

    # --- Optional capabilities ---
    #
    # Some sources of sandboxes can do more than create and destroy: a microVM
    # host can suspend one in milliseconds, or split a running one into
    # independent copies. Containers cannot. Rather than force every caller to
    # know which implementation it holds, capabilities are declared and the
    # base refuses them, so a harness can write "fork if you can, otherwise
    # rebuild" without a type check.

    def supports_pause(self) -> bool:
        """Whether this pool can suspend and resume a sandbox."""
        return False

    def supports_fork(self) -> bool:
        """Whether this pool can split a running sandbox into copies."""
        return False

    def supports_snapshot(self) -> bool:
        """Whether this pool can publish persistent snapshots."""
        return False

    async def pause(self, sandbox: Sandbox) -> None:
        """Suspend a sandbox, releasing its compute until resumed."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot pause sandboxes; "
            "check supports_pause() first"
        )

    async def resume(self, sandbox: Sandbox) -> None:
        """Bring a paused sandbox back."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot resume sandboxes; "
            "check supports_pause() first"
        )

    async def fork(self, sandbox: Sandbox, count: int = 1) -> list[Sandbox]:
        """Split a running sandbox into `count` independent copies.

        Each copy continues from the source's current state, which is what
        makes speculative branches cheap: the shared prefix is paid for once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot fork sandboxes; "
            "check supports_fork() first"
        )

    async def snapshot(self, sandbox: Sandbox, *, name: str | None = None,
                       disk_only: bool = False) -> "Snapshot":
        """Publish the sandbox's current state as a persistent snapshot.

        Unlike `pause`, the sandbox keeps running: this is the checkpoint
        primitive a rollout uses per step, and `spawn(image=<snapshot id>)`
        later starts a fresh sandbox from it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot snapshot sandboxes; "
            "check supports_snapshot() first"
        )

    async def squash(self, snapshot: "Snapshot | str", *,
                     name: str | None = None) -> "Snapshot":
        """Flatten a snapshot's layer chain, returning an equivalent snapshot.

        A sandbox started from a snapshot inherits its layers as a prefix it
        can never compact, so branching repeatedly from deep snapshots pays a
        growing cost. Squashing hands out a one-layer equivalent instead; the
        original stays valid.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot squash snapshots; "
            "check supports_snapshot() first"
        )

    async def close(self) -> None:
        await self.destroy_all()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class DockerPool(Pool):
    """Manages multiple sandboxes locally via Docker."""

    BOOTSTRAP_URL = "https://raw.githubusercontent.com/dreamyang-liu/Ash/main/runtime/bootstrap.sh"

    def __init__(self, runtime_bin: str | None = None, port: int = 3000):
        self.runtime_bin = runtime_bin or shutil.which("ash-runtime")
        self.port = port
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str = "ubuntu:24.04",
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
        agent_id: str = "",
    ) -> Sandbox:
        """Spawn a new sandbox.

        Args:
            image: Container image.
            entrypoint: Setup command run before ash-runtime starts.
            env: Environment variables for the container.
            resources: {"cpu": "2", "memory": "4g"} — mapped to Docker limits.
            agent_id: Identity bound to the returned handle (see Pool.spawn).
        """
        host_port = _find_free_port()

        docker_args = []
        if env:
            for k, v in env.items():
                docker_args.extend(["-e", f"{k}={v}"])
        if resources:
            if "cpu" in resources:
                docker_args.extend(["--cpus", str(resources["cpu"])])
            if "memory" in resources:
                docker_args.extend(["-m", str(resources["memory"])])

        label_args = ["--label", "ash.managed=1"]

        if self.runtime_bin:
            if entrypoint:
                container_cmd = f"({entrypoint}) && ash-runtime --port {self.port}"
            else:
                container_cmd = f"ash-runtime --port {self.port}"
            cmd = [
                "docker", "run", "-d",
                "-p", f"{host_port}:{self.port}",
                "-v", f"{self.runtime_bin}:/usr/local/bin/ash-runtime:ro",
                *label_args,
                *docker_args,
                image,
                "sh", "-c", container_cmd,
            ]
        else:
            bootstrap_cmd = f"curl -fsSL {self.BOOTSTRAP_URL} | ASH_PORT={self.port}"
            if entrypoint:
                escaped = entrypoint.replace('"', '\\"')
                bootstrap_cmd += f' ASH_SETUP="{escaped}"'
            bootstrap_cmd += " sh"
            cmd = [
                "docker", "run", "-d",
                "-p", f"{host_port}:{self.port}",
                *label_args,
                *docker_args,
                image,
                "sh", "-c", bootstrap_cmd,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        sb = Sandbox(backend=HTTPBackend(f"http://localhost:{host_port}"),
                     agent_id=agent_id)
        sb._container_id = container_id
        self._sandboxes[container_id] = sb

        await sb._wait_ready(timeout=60)
        return sb

    async def destroy(self, *sandboxes: Sandbox) -> None:
        for sandbox in sandboxes:
            cid = sandbox._container_id
            if not cid or cid not in self._sandboxes:
                continue
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            del self._sandboxes[cid]
            sandbox._container_id = None

    async def destroy_all(self):
        for cid in list(self._sandboxes.keys()):
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        self._sandboxes.clear()


class MicroVMPool(Pool):
    """Sandboxes as Firecracker microVMs, provisioned by an AgentENV server.

    Ash does not manage Firecracker itself; that is an entire storage stack
    (layered block devices, memory snapshots) and AgentENV already provides it
    behind an HTTP API. This pool is a client of that API: it asks for VMs from
    a template and reaches the runtime inside them through AgentENV's per-node
    proxy, which routes on a sandbox id and a target port.

    The template image must already contain ash-runtime, started on
    `runtime_port`. That is why the runtime is a static binary with embedded CA
    roots: it drops into any OCI image without further dependencies.

    What this buys over containers is the state operations -- pause, resume and
    fork all run in the tens of milliseconds -- which is why they are declared
    as supported capabilities here and refused by the base class elsewhere.

    Reaching the runtime only through a proxy has one consequence worth knowing:
    AgentENV gives an upstream 30 s to produce response headers
    (`response_header_timeout_ms`), and a tool call produces none until its
    command finishes. Runtimes from before mid-2026 therefore lose any call
    longer than that -- a 504 with the command still running inside the VM, so
    the work happens and the result is discarded. Current runtimes send headers
    up front and heartbeat while the command runs, which keeps the connection
    alive for as long as the call takes; there is nothing to configure. Against
    an older runtime, run long commands with `shell(background=True)` and poll
    `process read`, so every individual request stays short.
    """

    #: AgentENV's proxy routing headers (it also accepts E2B-compatible aliases).
    SANDBOX_ID_HEADER = "x-agentenv-sandbox-id"
    TARGET_PORT_HEADER = "x-agentenv-target-port"

    def __init__(self, server_url: str, default_template: str = "ubuntu",
                 runtime_port: int = 3000, api_key: str = "",
                 request_timeout: float = 120, sandbox_ttl: int = 300,
                 auto_resume: bool = True):
        """
        Args:
            server_url: AgentENV server, e.g. "http://127.0.0.1:8000".
            default_template: Template or snapshot name whose image already
                runs ash-runtime on `runtime_port` (e.g. built once with
                `aenv snapshot create <sandbox> --name ash-base`).
            runtime_port: Port ash-runtime listens on inside the VM.
            api_key: AgentENV API key. Sent as `X-API-KEY`; that is the header
                the server validates (a Bearer Authorization header is not).
            sandbox_ttl: Default sandbox time-to-live in seconds. AgentENV's
                own default is 15s, after which the VM auto-pauses -- far too
                short for an agent turn, so spawn() and resume() always send
                this instead.
            auto_resume: Ask AgentENV to transparently resume a sandbox that
                auto-paused at TTL expiry when the next proxied call arrives.
                With snapshot resume in the tens of milliseconds, an idle
                sandbox then costs nothing and the agent never notices.
        """
        self.server_url = server_url.rstrip("/")
        self.default_template = default_template
        self.runtime_port = runtime_port
        self.sandbox_ttl = sandbox_ttl
        self.auto_resume = auto_resume
        headers = {"X-API-KEY": api_key} if api_key else {}
        self._client = httpx.AsyncClient(timeout=request_timeout, headers=headers)
        self._sandboxes: dict[str, Sandbox] = {}

    # --- Capabilities ---

    def supports_pause(self) -> bool:
        return True

    def supports_fork(self) -> bool:
        return True

    def supports_snapshot(self) -> bool:
        return True

    # --- Lifecycle ---

    async def spawn(
        self,
        image: str | None = None,
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
        agent_id: str = "",
    ) -> Sandbox:
        """Start a microVM from a template and attach to its runtime.

        `entrypoint` is not supported: the runtime must already be baked into
        the template (VM resources likewise come from the template). Fail loud
        rather than silently ignoring a setup command the caller relies on.
        """
        if entrypoint:
            raise ValueError(
                "MicroVMPool cannot run an entrypoint; bake setup into the "
                "template instead (aenv snapshot create <sandbox> --name ...)"
            )
        if resources:
            raise ValueError(
                "MicroVMPool cannot set per-sandbox resources; they are fixed "
                "by the template"
            )
        payload: dict = {
            "templateID": image or self.default_template,
            "timeout": self.sandbox_ttl,
            "autoPause": True,
            "autoResume": {"enabled": self.auto_resume},
        }
        if env:
            payload["envVars"] = env

        resp = await self._client.post(f"{self.server_url}/sandboxes", json=payload)
        resp.raise_for_status()
        return self._attach(_sandbox_id(resp.json()), agent_id)

    async def destroy(self, *sandboxes: Sandbox) -> None:
        for sb in sandboxes:
            sid = sb._container_id
            if not sid or sid not in self._sandboxes:
                continue
            await self._client.delete(f"{self.server_url}/sandboxes/{sid}")
            del self._sandboxes[sid]
            sb._container_id = None

    async def destroy_all(self) -> None:
        await self.destroy(*self.list())
        self._sandboxes.clear()

    async def close(self) -> None:
        await self.destroy_all()
        await self._client.aclose()

    # --- State operations ---

    async def pause(self, sandbox: Sandbox) -> None:
        """Suspend a VM. Its memory and disk state are snapshotted."""
        sid = _require_id(sandbox)
        resp = await self._client.post(
            f"{self.server_url}/sandboxes/{sid}/pause")
        resp.raise_for_status()

    async def resume(self, sandbox: Sandbox) -> None:
        """Restore a paused VM from its snapshot.

        The endpoint requires a JSON body (415 without one); `timeout`
        restarts the sandbox's TTL clock from now.
        """
        sid = _require_id(sandbox)
        resp = await self._client.post(
            f"{self.server_url}/sandboxes/{sid}/resume",
            json={"timeout": self.sandbox_ttl})
        resp.raise_for_status()

    async def fork(self, sandbox: Sandbox, count: int = 1,
                   agent_ids: list[str] | None = None) -> list[Sandbox]:
        """Split a running VM into `count` copies of its current state.

        Useful for speculative work: every branch starts from the same
        prepared environment without repeating the setup that produced it.
        The source keeps running.

        Each child inherits the source's identity unless `agent_ids` names them
        individually. Inheriting is safe because a fork is a separate VM with
        its own event log, so two children called "worker" do not compete for a
        cursor -- pass agent_ids when the branches need telling apart in traces.
        """
        sid = _require_id(sandbox)
        resp = await self._client.post(
            f"{self.server_url}/sandboxes/{sid}/fork",
            json={"count": count, "timeout": self.sandbox_ttl})
        resp.raise_for_status()
        # The response is a list of per-fork results, each carrying either
        # `sandbox` or `error`. A 201 only means the snapshot succeeded, so
        # surface partial failures rather than silently returning fewer
        # children than asked for.
        results = resp.json()
        errors = [r["error"] for r in results if r.get("error")]
        if errors:
            raise RuntimeError(f"{len(errors)}/{len(results)} forks failed: {errors}")
        return [
            self._attach(
                _sandbox_id(result["sandbox"]),
                agent_ids[i] if agent_ids and i < len(agent_ids) else sandbox.agent_id,
            )
            for i, result in enumerate(results)
        ]

    async def snapshot(self, sandbox: Sandbox, *, name: str | None = None,
                       disk_only: bool = False) -> Snapshot:
        """Publish the sandbox's current state; the sandbox keeps running.

        `disk_only` skips the VM state and memory image, which is what makes a
        per-step checkpoint cheap: the cost is roughly the bytes written since
        the previous capture, and the guest filesystem is synced first so
        recent writes are on the virtual disk. Sandboxes created from a
        disk-only snapshot cold-boot (processes are gone, the template's
        startup command is re-run) instead of resuming.
        """
        sid = _require_id(sandbox)
        payload: dict = {}
        if name:
            payload["name"] = name
        if disk_only:
            payload["diskOnly"] = True
        resp = await self._client.post(
            f"{self.server_url}/sandboxes/{sid}/snapshots", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    async def squash(self, snapshot: Snapshot | str, *,
                     name: str | None = None) -> Snapshot:
        """Flatten a snapshot's chain into an equivalent one-layer snapshot.

        Returns a snapshot whose children start from a single inherited layer.
        The source snapshot is untouched and stays usable. Already-flat chains
        come back unchanged (the same id), so this is safe to call blindly.
        """
        snapshot_id = snapshot.id if isinstance(snapshot, Snapshot) else snapshot
        disk_only = snapshot.disk_only if isinstance(snapshot, Snapshot) else False
        payload: dict = {}
        if name:
            payload["name"] = name
        resp = await self._client.post(
            f"{self.server_url}/snapshots/{snapshot_id}/squash", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    async def get_snapshot(self, snapshot_id: str) -> Snapshot:
        """Look up a snapshot's current chain facts."""
        resp = await self._client.get(f"{self.server_url}/snapshots/{snapshot_id}")
        resp.raise_for_status()
        return Snapshot.from_api(resp.json())

    # --- Internals ---

    def _attach(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """Wrap an AgentENV sandbox id as a Sandbox reachable via the proxy."""
        sb = Sandbox(backend=GatewayBackend(
            self.server_url, sandbox_id,
            sandbox_id_header=self.SANDBOX_ID_HEADER,
            target_port=self.runtime_port,
            target_port_header=self.TARGET_PORT_HEADER,
        ), agent_id=agent_id)
        sb._container_id = sandbox_id
        self._sandboxes[sandbox_id] = sb
        return sb


def _sandbox_id(body: dict) -> str:
    """Read a sandbox id from an AgentENV response, whichever name it used."""
    for key in ("sandboxID", "sandbox_id", "id"):
        if body.get(key):
            return str(body[key])
    raise RuntimeError(f"no sandbox id in response: {body}")


def _require_id(sandbox: Sandbox) -> str:
    sid = sandbox._container_id
    if not sid:
        raise RuntimeError("sandbox has no id (already destroyed?)")
    return sid


class SandboxPool(Pool):
    """Manages multiple sandboxes via K8s control-plane + gateway."""

    def __init__(self, control_plane_url: str, gateway_url: str, default_image: str = "ubuntu:24.04"):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.default_image = default_image
        self._client = httpx.AsyncClient(timeout=120)
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str | None = None,
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
        agent_id: str = "",
    ) -> Sandbox:
        """Spawn a new sandbox.

        Args:
            image: Container image.
            entrypoint: Setup command run before ash-runtime starts.
            env: Environment variables for the container.
            resources: {"cpu": "1", "memory": "2Gi"} — K8s resource requests.
            agent_id: Identity bound to the returned handle (see Pool.spawn).
        """
        body: dict[str, Any] = {
            "image": image or self.default_image,
            "ports": [{"container_port": 3000}],
        }
        if entrypoint:
            body["entrypoint"] = entrypoint
        if env:
            body["env"] = env
        if resources:
            body["resources"] = {
                "requests": {k: v for k, v in resources.items()},
            }

        resp = await self._client.post(f"{self.control_plane_url}/create", json=body)
        resp.raise_for_status()
        data = resp.json()

        sandbox_id = data["uuid"]
        sb = Sandbox(backend=GatewayBackend(self.gateway_url, sandbox_id),
                     agent_id=agent_id)
        sb._container_id = sandbox_id
        self._sandboxes[sandbox_id] = sb

        await sb._wait_ready(timeout=60)
        return sb

    async def destroy(self, *sandboxes: Sandbox):
        """Destroy one or more sandboxes."""
        ids = [sb._container_id for sb in sandboxes if sb._container_id and sb._container_id in self._sandboxes]
        if not ids:
            return
        await self._client.request("DELETE", f"{self.control_plane_url}/destroy", json={"ids": ids})
        for sid in ids:
            if sid in self._sandboxes:
                del self._sandboxes[sid]
        for sb in sandboxes:
            sb._container_id = None

    async def destroy_all(self):
        """Destroy all sandboxes."""
        await self._client.request("DELETE", f"{self.control_plane_url}/destroy", json={"all": True})
        self._sandboxes.clear()

    async def close(self) -> None:
        await self.destroy_all()
        await self._client.aclose()
