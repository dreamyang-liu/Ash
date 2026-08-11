from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

import httpx

from . import schemas
from .backends import Backend, CLIBackend, HTTPBackend, MCPBackend
from .events import EventBatch, parse_batch, parse_events
from .result import ToolResult
from .toolset import ToolRegistry


@dataclass
class Sandbox:
    """Manages lifecycle + delegates execution to a Backend."""

    backend: Backend
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    # Who this handle acts as. Identity belongs to the connection rather than
    # to each call: the runtime keeps a per-agent cursor over the event log, so
    # forgetting it on one call silently shares the anonymous cursor with every
    # other anonymous caller and events start looking lost. Two handles onto
    # the same sandbox with different ids are two independent consumers, which
    # is what a parent and its subagent want.
    agent_id: str = ""
    # Custom tool name -> verified binary path inside *this* sandbox, learned
    # from the artifact step. Memoised per sandbox, never on the registry: a
    # registry is deliberately reusable across sandboxes, so caching a path
    # there would make one sandbox execute a path that exists only in another.
    _artifact_paths: dict[str, str] = field(default_factory=dict, repr=False)
    _container_id: str | None = field(default=None, repr=False)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _tools_cache: list[dict] | None = field(default=None, repr=False)

    # --- Constructors ---

    # Each constructor takes the same two optional extras as the dataclass:
    # `agent_id` (who this handle acts as) and `tools` (the manifest-defined
    # tools reachable through it). Without the latter, a caller holding a
    # registry would have to build the backend by hand just to attach it.

    @classmethod
    def connect(cls, url: str, agent_id: str = "",
                tools: ToolRegistry | None = None) -> Sandbox:
        """Connect to a running ash-runtime via HTTP."""
        return cls(backend=HTTPBackend(url), agent_id=agent_id,
                   tools=tools or ToolRegistry())

    @classmethod
    def mcp(cls, url: str, agent_id: str = "",
            tools: ToolRegistry | None = None) -> Sandbox:
        """Connect via MCP protocol."""
        return cls(backend=MCPBackend(url), agent_id=agent_id,
                   tools=tools or ToolRegistry())

    @classmethod
    def local(cls, bin_path: str | None = None, agent_id: str = "",
              tools: ToolRegistry | None = None) -> Sandbox:
        """Use CLI backend — no server needed, calls binary directly."""
        return cls(backend=CLIBackend(bin_path), agent_id=agent_id,
                   tools=tools or ToolRegistry())

    def as_agent(self, agent_id: str) -> Sandbox:
        """A handle onto the same sandbox acting as a different agent.

        Useful when a subagent should be handed a handle it cannot use under
        the wrong identity; passing agent_id per call is equally valid and is
        the simpler choice when one component dispatches for several agents.

        Everything describing the sandbox is shared -- the backend, the tool
        panel, and the resolved artifact paths, since a binary cached in this
        sandbox is cached for every identity using it. Only the identity, and
        therefore the cursor over the event log, differs.
        """
        clone = Sandbox(backend=self.backend, tools=self.tools, agent_id=agent_id)
        clone._artifact_paths = self._artifact_paths
        clone._tools_cache = self._tools_cache
        clone._container_id = self._container_id
        return clone

    def _whoami(self, agent_id: str = "") -> str:
        """An explicit per-call identity wins over the handle's own."""
        return agent_id or self.agent_id

    @property
    def sandbox_id(self) -> str | None:
        """Stable Docker or gateway identity, when this sandbox is managed."""
        return self._container_id

    @classmethod
    async def spawn(
        cls,
        image: str = "ubuntu:24.04",
        port: int = 3000,
        runtime_bin: str | None = None,
        docker_args: list[str] | None = None,
        agent_id: str = "",
        tools: ToolRegistry | None = None,
    ) -> Sandbox:
        """Spawn a Docker container with ash-runtime injected."""
        if runtime_bin is None:
            runtime_bin = shutil.which("ash-runtime")
            if runtime_bin is None:
                raise RuntimeError("ash-runtime binary not found in PATH")

        host_port = _find_free_port()
        cmd = [
            "docker", "run", "-d",
            "-p", f"{host_port}:{port}",
            "-v", f"{runtime_bin}:/usr/local/bin/ash-runtime:ro",
            *(docker_args or []),
            image,
            "ash-runtime", "--port", str(port),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        url = f"http://localhost:{host_port}"
        sb = cls(backend=HTTPBackend(url), agent_id=agent_id,
                 tools=tools or ToolRegistry())
        sb._container_id = container_id

        await sb._wait_ready()
        return sb

    # --- Lifecycle ---

    async def destroy(self):
        if self._container_id:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self._container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._container_id = None

    async def close(self):
        await self.backend.close()
        await self.destroy()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    # --- Execution ---

    async def call(self, tool_name: str, agent_id: str = "", **kwargs) -> ToolResult:
        """Call a tool by name.

        agent_id identifies the caller to the runtime, which uses it to decide
        whose event subscriptions this response carries and who is recorded as
        having caused the action. Harnesses assign these identities; the SDK
        only forwards them.
        """
        return await self.backend.call(tool_name, kwargs, self._whoami(agent_id))

    async def call_agent_tool(self, name: str, args: dict,
                              registry: ToolRegistry | None = None,
                              agent_id: str = "") -> ToolResult:
        """Call an agent-facing tool through a ToolRegistry.

        Builtin tools route by name; manifest-defined custom tools expand
        into their execution plan (artifact download -> shell, or direct
        shell for image-local binaries). This is the single place custom
        tool dispatch lives — any harness gets it via this method.

        agent_id is forwarded to every runtime call the dispatch makes, so a
        custom tool's download and execution are attributed to the same caller.
        """
        registry = registry or self.tools
        agent_id = self._whoami(agent_id)
        if registry.is_custom_tool(name):
            return await self._call_custom_tool(name, args, registry, agent_id)
        runtime_tool, runtime_args = registry.route(name, args)
        return await self.backend.call(runtime_tool, runtime_args, agent_id)

    async def _call_custom_tool(self, name: str, args: dict,
                                registry: ToolRegistry,
                                agent_id: str = "") -> ToolResult:
        """Run a manifest-defined tool: resolve its binary, then execute it.

        The artifact step is idempotent and cached inside the sandbox, so
        after the first resolution only the execution round-trip is needed.
        """
        plan = registry.plan_custom_tool(name, args)  # validates args first

        binary_path = plan.spec.path or self._artifact_paths.get(name, "")
        memoised = bool(binary_path) and not plan.spec.path
        if not binary_path:
            resolved = await self._resolve_artifact(plan, agent_id)
            if isinstance(resolved, ToolResult):
                return resolved  # download or verification failed
            binary_path = resolved
            self._artifact_paths[name] = binary_path

        tool, call_args = plan.shell_call(binary_path)
        result = await self.backend.call(tool, call_args, agent_id)

        # A memoised path can go stale (a cleaned /tmp, a recycled sandbox).
        # Re-resolve once rather than surfacing a confusing "not found".
        if memoised and result.is_error and _looks_missing(result.output):
            self._artifact_paths.pop(name, None)
            resolved = await self._resolve_artifact(plan, agent_id)
            if isinstance(resolved, ToolResult):
                return resolved
            self._artifact_paths[name] = resolved
            tool, call_args = plan.shell_call(resolved)
            return await self.backend.call(tool, call_args, agent_id)
        return result

    async def _resolve_artifact(self, plan, agent_id: str = "") -> str | ToolResult:
        """Return the local binary path, or the failing ToolResult."""
        tool, call_args = plan.artifact_call
        result = await self.backend.call(tool, call_args, agent_id)
        if result.is_error:
            return result
        return result.output.strip()

    async def prepare_tools(self, registry: ToolRegistry | None = None,
                            agent_id: str = "") -> dict[str, str]:
        """Resolve every custom tool's binary up front.

        Worth calling when a session starts: downloads happen once instead of
        during the first agent call, and a binary that cannot be fetched or
        fails verification surfaces immediately rather than mid-rollout.

        Returns the tool name -> local path mapping. Raises RuntimeError on the
        first tool that cannot be prepared.
        """
        registry = registry or self.tools
        agent_id = self._whoami(agent_id)
        for name, spec in registry.custom_specs.items():
            if spec.path:
                self._artifact_paths[name] = spec.path
                continue
            plan = registry.plan_custom_tool_for_prepare(name)
            resolved = await self._resolve_artifact(plan, agent_id)
            if isinstance(resolved, ToolResult):
                raise RuntimeError(
                    f"cannot prepare custom tool {name!r}: {resolved.output}"
                )
            self._artifact_paths[name] = resolved
        return dict(self._artifact_paths)

    async def execute_tool_call(self, tool_call: dict, agent_id: str = "") -> ToolResult:
        """Execute an LLM tool_call (OpenAI or Anthropic format)."""
        if "function" in tool_call:
            name = tool_call["function"]["name"]
            args_raw = tool_call["function"].get("arguments", "{}")
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        else:
            name = tool_call["name"]
            args = tool_call.get("input", {})
        return await self.backend.call(name, args, self._whoami(agent_id))

    async def execute_tool_calls(self, tool_calls: list[dict],
                                 agent_id: str = "") -> list[ToolResult]:
        """Execute multiple tool_calls sequentially.

        Sequential is only the default: a harness that knows its calls are
        independent can await them concurrently itself.
        """
        return [await self.execute_tool_call(tc, agent_id) for tc in tool_calls]

    # --- Events ---

    async def wait_for_events(self, kinds: list[str] | None = None,
                              sources: list[str] | None = None,
                              timeout: int = 30,
                              agent_id: str = "") -> EventBatch:
        """Block until a matching event occurs, or the timeout elapses.

        Filters are ANDed and each matches everything when omitted: `kinds`
        selects event types ("process_exited", "tool:web_fetch"), `sources`
        narrows to specific handles, so waiting on one background process is
        wait_for_events(["process_exited"], [pid]).

        Every event is delivered once per identity. A second call will not
        repeat what this handle already received -- and handles sharing an
        agent_id share that cursor, which is why an identity is worth setting
        when more than one consumer must each see everything.

        Prefer this over polling in a loop: it returns as soon as the event
        happens rather than on a timer.
        """
        args: dict = {"timeout": timeout}
        if kinds:
            args["kinds"] = kinds
        if sources:
            args["sources"] = sources
        result = await self.backend.call("wait_for_events", args,
                                         self._whoami(agent_id))
        if result.is_error:
            raise RuntimeError(f"wait_for_events failed: {result.output}")
        return parse_batch(result.output)

    async def poll_events(self, kinds: list[str] | None = None,
                          sources: list[str] | None = None,
                          agent_id: str = "") -> EventBatch:
        """Take whatever matching events are already available, without waiting.

        The non-blocking form of wait_for_events, for a harness that wants to
        check for environment changes at a convenient point in its own loop.
        """
        return await self.wait_for_events(kinds, sources, timeout=0,
                                          agent_id=agent_id)

    # --- Schemas ---

    async def tool_schemas(self, format: str = "openai",
                           registry: ToolRegistry | None = None) -> list[dict]:
        """Get the complete tool panel for LLM function calling.

        Includes both the runtime's builtin tools and the registry's
        manifest-defined ones. The runtime knows nothing about custom tools --
        they exist only as manifests on this side -- so assembling the full
        panel is the SDK's job, not something a harness should have to stitch
        together itself.

        format: "openai" | "anthropic" | "raw"
        """
        registry = registry or self.tools
        builtin = [
            schemas.render_runtime_tool(t, format)
            for t in await self._get_tools()
        ]
        return builtin + registry.custom_agent_schemas(format)

    # --- Internal ---

    async def _get_tools(self) -> list[dict]:
        if self._tools_cache is None:
            self._tools_cache = await self.backend.list_tools()
        return self._tools_cache

    async def _wait_ready(self, timeout: float = 60):
        deadline = asyncio.get_event_loop().time() + timeout
        client = httpx.AsyncClient(timeout=5)
        try:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(self.backend.url)
                    if resp.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, OSError):
                    pass
                await asyncio.sleep(1)
        finally:
            await client.aclose()
        raise TimeoutError(f"ash-runtime not ready after {timeout}s")


def _looks_missing(output: str) -> bool:
    """Whether a shell failure reads like the binary was not there.

    Used to decide whether a memoised artifact path has gone stale and is
    worth re-resolving once. Deliberately narrow: a genuine tool failure must
    not trigger a pointless retry.
    """
    lowered = output.lower()
    return (
        "no such file" in lowered
        or "not found" in lowered
        or "cannot execute" in lowered
    )


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]
