"""MCP server that exposes ash sandbox tools to external agents.

Supports multiple sandboxes with ownership + group-based visibility:
- Each sandbox has an owner (session) and optional groups
- A session sees: its own private sandboxes + any shared sandbox whose groups
  overlap with the session's groups
- Runs as HTTP/SSE for multi-session, or stdio for single-session (backwards-compat)
- Optionally routes exec tool calls through the L2 interceptor pipeline
  (docs/ARCHITECTURE.md): OFF by default; --guardrails mounts read-before-edit
  and edit-streak nudges, --plugins <file.py> supplies your own interceptors.
- Knows nothing about what a run's *answer* is, and needs no hook for it: the
  caller that provisioned the sandbox owns it and can extract whatever it wants,
  whenever it wants -- from a snapshot afterwards (harness/extract.py) or from the
  live sandbox before teardown. A benchmark that must do something at sandbox
  lifecycle points subclasses SandboxPool and passes `pool_cls` to main().

Usage:
    # HTTP mode (multi-session):
    python -m harness.execution.server --http --port 8400

    # HTTP mode with governance mounted on the tool path:
    python -m harness.execution.server --http --port 8400 --guardrails reject

    # Stdio mode (single-session, backwards-compat):
    python -m harness.execution.server --attach <sandbox-id>

    # Serve a compiled tool panel instead of the built-in four:
    python -m harness.execution.server --attach <sandbox-id> --tools default
"""

import asyncio
import copy
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash_sandbox import Pool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult

from harness.core.result import ToolResult
from harness.execution.backends import BACKENDS, BackendError, build_pool
from harness.execution.interceptors import GuardrailInterceptor, TruncateInterceptor
from harness.execution.pipeline import CallContext, ToolPipeline, load_pipeline


# ---------------------------------------------------------------------------
# Core state
# ---------------------------------------------------------------------------

@dataclass
class SandboxEntry:
    id: str
    sandbox: Sandbox
    image: str
    groups: list[str]        # visibility = group intersection
    base_commit: str = ""
    #: True when somebody else owns this sandbox's lifetime -- an orchestrator
    #: that created it and will snapshot and tear it down itself. The pool then
    #: serves calls into it but never destroys it: doing so would kill the
    #: environment its owner still needs for grading and extraction, and a second
    #: destroy from the real owner would then fail against a sandbox that is
    #: already gone.
    external: bool = False
    #: Scratch space for a SandboxPool subclass. The execution plane never
    #: reads it.
    meta: dict = field(default_factory=dict)

    def visible_to(self, session_groups: list[str]) -> bool:
        return bool(set(self.groups) & set(session_groups))


@dataclass
class Session:
    id: str
    groups: list[str] = field(default_factory=lambda: ["default"])
    # Fixed single-sandbox binding. Set once at startup in single-sandbox stdio
    # mode, or per session in HTTP mode from the `x-session-sandbox` header (an
    # orchestrator that provisioned the sandbox states which one this slot owns).
    # A bound session is served the single-sandbox tool schema, so the model
    # never sees a `sandbox_id` parameter and cannot name another sandbox.
    # Left None otherwise: an explicit sandbox_id is then required on every exec
    # call — there is no switchable "active" state.
    bound_id: str | None = None

    @property
    def owner_group(self) -> str:
        """The implicit private group for this session's owner."""
        return f"owner:{self.id}"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

LIFECYCLE_TOOLS = [
    {
        "name": "sandbox_create",
        "description": (
            "Create a new sandbox container from a Docker image.\n"
            "Your owner group is always attached (private by default).\n"
            "Pass additional groups to share with other sessions in those groups."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Docker image to spawn"},
                "groups": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional groups to share this sandbox with (your owner group is always included)",
                },
            },
            "required": ["image"],
        },
    },
    {
        "name": "sandbox_list",
        "description": "List sandboxes visible to you (your private + shared via groups).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string", "description": "Filter by group (omit for all visible)"},
            },
        },
    },
    {
        "name": "sandbox_destroy",
        "description": "Destroy a sandbox (must have your owner group).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sandbox_id": {"type": "string", "description": "Sandbox ID to destroy"},
            },
            "required": ["sandbox_id"],
        },
    },
]

EXEC_TOOLS = [
    {
        "name": "shell",
        "description": (
            "Execute a shell command in a sandbox container.\n"
            "Working directory defaults to /testbed.\n"
            "Use 'tail' to limit output. Use 'background: true' for long-running commands."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "background": {"type": "boolean", "default": False},
                "timeout": {"type": "integer", "default": 300},
                "tail": {"type": "integer", "description": "Only return last N lines"},
                "working_dir": {"type": "string", "description": "Working directory (default: /testbed)"},
                "sandbox_id": {"type": "string", "description": "Target sandbox (default: active)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "text_editor",
        "description": "View or edit files in a sandbox.\nCommands: view, str_replace, insert, write",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["view", "str_replace", "insert", "write"]},
                "path": {"type": "string"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "insert_text": {"type": "string"},
                "file_text": {"type": "string"},
                "sandbox_id": {"type": "string", "description": "Target sandbox (default: active)"},
            },
            "required": ["command", "path"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search files using ripgrep.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
                "limit": {"type": "integer"},
                "sandbox_id": {"type": "string", "description": "Target sandbox (default: active)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "process",
        "description": "Manage a background process (read output or kill).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "action": {"type": "string", "enum": ["read", "kill"]},
                "tail": {"type": "integer"},
                "sandbox_id": {"type": "string", "description": "Target sandbox (default: active)"},
            },
            "required": ["pid", "action"],
        },
    },
]

def _single_sandbox_tools() -> list[dict]:
    """Exec tools only, with the multi-sandbox `sandbox_id` arg removed.

    Used in single-sandbox stdio mode: the sandbox is pre-provisioned and bound
    at startup, so the agent should see only shell/text_editor/grep_files/
    process — no lifecycle tools, no sandbox routing.
    """
    tools = copy.deepcopy(EXEC_TOOLS)
    for t in tools:
        t["inputSchema"]["properties"].pop("sandbox_id", None)
    return tools


def _multi_sandbox_tools() -> list[dict]:
    """Exec tools with `sandbox_id` REQUIRED — stateless multi-sandbox mode.

    Every exec call names its target sandbox explicitly; there is no switchable
    "active" sandbox, so concurrent calls can't race and no lock is needed.
    """
    tools = copy.deepcopy(EXEC_TOOLS)
    for t in tools:
        props = t["inputSchema"]["properties"]
        if "sandbox_id" in props:
            props["sandbox_id"]["description"] = "Target sandbox ID (required)"
        req = t["inputSchema"].setdefault("required", [])
        if "sandbox_id" not in req:
            req.append("sandbox_id")
    return tools


EXEC_TOOLS_SINGLE = _single_sandbox_tools()
EXEC_TOOLS_MULTI = _multi_sandbox_tools()

# Multi-sandbox surface: lifecycle (create/list/destroy) + id-required exec tools.
ALL_TOOLS = LIFECYCLE_TOOLS + EXEC_TOOLS_MULTI


class ExecSurface:
    """The exec tools one server offers, and how a call on them routes.

    Two sources, one shape:

    - **the literals above** (default). Four tools, hand-written, kept because they
      are what every caller of this module has been served and what the in-process
      harnesses import to stay in step with it.
    - **a compiled panel** (``--tools <manifest>``). The manifest says which runtime
      tools to offer, under which names, with which parameters -- compiled against
      ``runtime/schema/tools.json``, so a manifest naming something the runtime does
      not serve fails at startup instead of at the model's first call.

    The panel format is ``"raw"``, which is already MCP's ``{name, description,
    inputSchema}``; nothing converts. What this class adds is the part a panel has
    no concept of: ``sandbox_id``. A panel describes one sandbox's tools, while this
    server may front many, so the multi-sandbox variant injects the argument and the
    single-sandbox variant leaves it out -- exactly as the literals are derived.

    Routing goes through the panel too, so a renamed view (``run_tests`` over
    ``shell``) reaches the runtime under its real name and interceptors keyed on
    ``shell`` do not go blind.
    """

    def __init__(self, panel: Any = None):
        self.panel = panel
        base = list(panel.schema) if panel is not None else EXEC_TOOLS
        if panel is None:
            self.single, self.multi = EXEC_TOOLS_SINGLE, EXEC_TOOLS_MULTI
        else:
            self.single = copy.deepcopy(base)
            self.multi = _with_sandbox_id(copy.deepcopy(base))

    @property
    def names(self) -> set:
        return {t["name"] for t in self.single}

    def route(self, name: str, args: dict) -> "tuple[str, dict]":
        """An agent-facing call as a runtime call. Identity without a panel."""
        if self.panel is None:
            return name, args
        return self.panel.route(name, args)

    def all_tools(self) -> list:
        return LIFECYCLE_TOOLS + self.multi


def _with_sandbox_id(tools: list) -> list:
    """Add the required ``sandbox_id`` argument to compiled tools.

    The literals declare it and ``_multi_sandbox_tools`` only promotes it; a panel
    never mentions it, because a panel is written about a sandbox's tools and not
    about a server that fronts several.
    """
    for tool in tools:
        schema = tool.setdefault("inputSchema", {"type": "object"})
        props = schema.setdefault("properties", {})
        props["sandbox_id"] = {"type": "string",
                               "description": "Target sandbox ID (required)"}
        required = schema.setdefault("required", [])
        if "sandbox_id" not in required:
            required.append("sandbox_id")
    return tools


#: What a server serves when no manifest is named: the literals, unchanged.
DEFAULT_SURFACE = ExecSurface()


# ---------------------------------------------------------------------------
# Sandbox Pool (shared across all sessions)
# ---------------------------------------------------------------------------

class SandboxPool:
    """Manages all sandboxes. Session-agnostic — visibility is enforced at the server layer."""

    def __init__(self, runtime_bin: str | None = None,
                 backend: dict | None = None):
        self.runtime_bin = runtime_bin
        #: Where sandboxes come from (``harness/execution/backends.py``); empty means
        #: local Docker. The proxy names no concrete pool, so every sandbox a
        #: client creates through it lands on the configured backend.
        self.backend = backend or {}
        self._pool: Pool | None = None
        self._sandboxes: dict[str, SandboxEntry] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"sb-{self._counter}"

    async def create(self, image: str, groups: list[str]) -> SandboxEntry:
        if not self._pool:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
        sandbox = await self._pool.spawn(image=image)
        r = await sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
        base_commit = r.output.strip() if not r.is_error else ""
        sb_id = self._next_id()
        entry = SandboxEntry(id=sb_id, sandbox=sandbox, image=image,
                             groups=groups, base_commit=base_commit)
        self._sandboxes[sb_id] = entry
        self._log(f"created {sb_id} groups={groups}")
        return entry

    async def destroy(self, sb_id: str) -> None:
        entry = self._sandboxes.pop(sb_id, None)
        if not entry:
            return
        if entry.external:
            # Stop serving it; leave it running. Its owner is still holding it.
            self._log(f"released {sb_id} (owned elsewhere)")
            return
        if self._pool:
            await self._pool.destroy(entry.sandbox)
        self._log(f"destroyed {sb_id}")

    async def destroy_all(self):
        for sb_id in list(self._sandboxes):
            await self.destroy(sb_id)

    def adopt(self, sandbox: Sandbox, groups: list[str], *,
              sandbox_id: str | None = None, image: str = "",
              base_commit: str = "") -> SandboxEntry:
        """Serve calls into a sandbox this process already holds a handle to.

        The in-process case, and the reason it is separate from :meth:`attach`:
        attach re-derives a handle *from an id*, which needs a backend that can
        (only microvm today) and yields a second handle to the same sandbox. This
        takes the handle the caller already has, so any backend works -- Docker
        included -- and there is exactly one of them.

        Used by the orchestrator when it runs the MCP server itself: the session
        and the server are then in the same process, so lending the sandbox needs
        no round trip and no re-derivation. The entry is marked ``external``, so
        this pool will not destroy it; the session that created it does that, after
        it has taken its last snapshot and extracted whatever it needed.

        Synchronous on purpose: there is nothing to await. ``attach`` probes the
        sandbox because it knows nothing about it; here the caller already does.
        """
        sb_id = sandbox_id or getattr(sandbox, "sandbox_id", None) or self._next_id()
        entry = SandboxEntry(id=sb_id, sandbox=sandbox, image=image, groups=groups,
                             base_commit=base_commit, external=True)
        self._sandboxes[sb_id] = entry
        self._log(f"adopted {sb_id} groups={groups} (owner keeps it)")
        return entry

    async def attach(self, sandbox_id: str, groups: list[str]) -> SandboxEntry:
        """Adopt a sandbox somebody else created, by id.

        This is how a caller that owns the sandbox lends it to the proxy: the
        orchestrator creates it (so it holds the handle, and can snapshot or
        extract afterwards) and the proxy only serves tool calls into it.
        Creating it here instead -- what ``--image`` did -- inverted that, and the
        owner then had no handle at all.
        """
        if not self._pool:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
        attach = getattr(self._pool, "attach", None) or getattr(self._pool, "_attach", None)
        if attach is None:
            raise BackendError(
                "backend %r cannot attach to an existing sandbox; it has no "
                "attach(). Use an http wiring against a server that can, or let "
                "this process create its own." % (self.backend.get("backend") or "docker"))
        sandbox = attach(sandbox_id)
        if hasattr(sandbox, "__await__"):
            sandbox = await sandbox
        result = await sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
        base_commit = result.output.strip() if not result.is_error else ""
        entry = SandboxEntry(id=sandbox_id, sandbox=sandbox, image="",
                             groups=groups, base_commit=base_commit)
        self._sandboxes[sandbox_id] = entry
        self._log(f"attached {sandbox_id} groups={groups}")
        return entry

    async def after_mutating_call(self, entry: SandboxEntry, tool_name: str,
                                  args: dict) -> None:
        """A call that may have changed ``entry`` completed. No-op by default.

        A subclass that keeps an artefact derived from the sandbox (a benchmark's
        answer, say) refreshes it here. Nothing in the execution plane depends on
        it, and a subclass that raises would surface as a tool error, so an
        override should swallow its own failures.
        """

    def get(self, sb_id: str) -> SandboxEntry | None:
        return self._sandboxes.get(sb_id)

    def visible_to(self, session: Session, group_filter: str | None = None) -> list[SandboxEntry]:
        results = []
        for entry in self._sandboxes.values():
            if not entry.visible_to(session.groups):
                continue
            if group_filter and group_filter not in entry.groups:
                continue
            results.append(entry)
        return results

    def _log(self, text: str):
        sys.stderr.write(f"[ash-pool] {text}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Session handler (per-connection tool dispatch)
# ---------------------------------------------------------------------------

class SessionHandler:
    """Handles tool calls scoped to a single session's visibility."""

    # Exec tools that can mutate the filesystem.
    _MUTATING = {"shell", "text_editor", "process"}

    def __init__(self, session: Session, pool: SandboxPool, notify_mutations: bool = False,
                 pipeline: "ToolPipeline | None" = None,
                 surface: "ExecSurface | None" = None):
        self.session = session
        self.pool = pool
        # surface: which exec tools exist and how they route. Defaults to the
        # literals, so a caller that never heard of manifests is unaffected.
        self.surface = surface or DEFAULT_SURFACE
        # notify_mutations: call the pool's after_mutating_call (a SandboxPool
        # subclass may override it) so a benchmark that keeps a running artefact
        # can refresh it. Plain SandboxPool does nothing.
        self.notify_mutations = notify_mutations
        # pipeline: L2 interceptor chain (shared across sessions — coordination
        # state must span agents). None = dispatch exactly as before (default).
        self.pipeline = pipeline

    def _resolve(self, sandbox_id: str | None) -> SandboxEntry | None:
        """Resolve the target sandbox by explicit id, falling back to the fixed
        single-sandbox binding. No switchable "active" state."""
        target = sandbox_id or self.session.bound_id
        if not target:
            return None
        entry = self.pool.get(target)
        if entry and entry.visible_to(self.session.groups):
            return entry
        return None

    async def call_tool(self, name: str, args: dict) -> dict:
        # -- Lifecycle tools --
        if name == "sandbox_create":
            # Always include the caller's owner group; add any extra shared groups.
            extra_groups = args.get("groups", [])
            groups = [self.session.owner_group] + extra_groups
            entry = await self.pool.create(args["image"], groups)
            return _ok(json.dumps({"id": entry.id, "groups": entry.groups}))

        if name == "sandbox_list":
            entries = self.pool.visible_to(self.session, args.get("group"))
            items = [{"id": e.id, "image": e.image, "groups": e.groups,
                      "mine": self.session.owner_group in e.groups} for e in entries]
            return _ok(json.dumps(items, indent=2))

        if name == "sandbox_destroy":
            sb_id = args["sandbox_id"]
            entry = self.pool.get(sb_id)
            if not entry:
                return _err(f"sandbox {sb_id} not found")
            if self.session.owner_group not in entry.groups:
                return _err(f"cannot destroy {sb_id}: not the owner")
            await self.pool.destroy(sb_id)
            return _ok(f"Destroyed {sb_id}.")

        # -- Exec tools -- sandbox_id is required in multi-sandbox mode; in
        # single-sandbox mode it is omitted and resolves to the bound sandbox.
        args = dict(args)  # copy so we never mutate the caller's argument dict
        sandbox_id = args.pop("sandbox_id", None)
        entry = self._resolve(sandbox_id)
        if not entry:
            return _err("sandbox_id is required and must reference a sandbox visible "
                        "to you (see sandbox_create / sandbox_list).")

        # Route BEFORE anything else looks at the call, so a renamed view reaches
        # the runtime under its real name -- and so the pipeline, the mutation
        # notification and the sandbox all agree on which tool ran. Routing after
        # any of them is how a panel's rename made interceptors keyed on `shell`
        # go blind.
        try:
            name, args = self.surface.route(name, args)
        except KeyError:
            return _err("unknown tool: %s" % name)
        except ValueError as exc:
            # The view does not offer that argument. Say so rather than dropping
            # it: a silently ignored parameter has the model believe a setting
            # took effect.
            return _err(str(exc))

        try:
            if self.pipeline is not None:
                content = await self._exec_via_pipeline(entry, name, args)
            else:
                result: SdkToolResult = await entry.sandbox.call(name, **args)
                content = {"type": "text", "text": result.output,
                           "isError": result.is_error}
        except Exception as e:
            return _err(str(e))

        if self.notify_mutations and name in self._MUTATING:
            # `entry` is the sandbox the call just ran in -- reuse it. This used to
            # re-resolve from `args.get("sandbox_id")`, which is always None here
            # because sandbox_id was popped above: in single-sandbox mode the
            # fallback to `bound_id` covered for it, but in multi-sandbox mode
            # nothing is bound, so the resolve failed and a SandboxPool subclass
            # hooking mutations was never called at all.
            await self.pool.after_mutating_call(entry, name, args)

        return content

    async def _exec_via_pipeline(self, entry: SandboxEntry, name: str,
                                 args: dict) -> dict:
        """Run one exec tool through the interceptor pipeline (L2 governance).

        The pipeline contract is synchronous, and an interceptor is allowed to
        block (waiting on a lock, say), so it runs on a worker thread; the raw
        executor bridges
        each inner/probe call back onto this event loop. agent_id is the MCP
        session identity; sandbox_id is the resolved sandbox.
        """
        loop = asyncio.get_running_loop()

        def raw_executor(tool: str, tool_args: dict) -> ToolResult:
            future = asyncio.run_coroutine_threadsafe(
                entry.sandbox.call(tool, **tool_args), loop)
            sdk = future.result()
            # from_sdk, so a command's outcome reaches interceptors on this path too
            # (a presenter rendering it, audit reading its byte counts).
            result = ToolResult.from_sdk(sdk)
            if sdk.is_error and not result.error:
                result.error = "tool error"
            return result

        ctx = CallContext(agent_id=self.session.id, sandbox_id=entry.id,
                          tool_name=name, args=dict(args),
                          metadata={"executor": raw_executor})
        result = await asyncio.to_thread(self.pipeline.execute, ctx, raw_executor)
        text = result.output if (result.success or result.output) \
            else f"Error: {result.error or 'unknown error'}"
        return {"type": "text", "text": text, "isError": not result.success}


def _ok(text: str) -> dict:
    return {"type": "text", "text": text}


def _err(text: str) -> dict:
    return {"type": "text", "text": f"Error: {text}", "isError": True}


# ---------------------------------------------------------------------------
# Transport: HTTP/SSE (multi-session)
# ---------------------------------------------------------------------------

class HttpMcpServer:
    """HTTP/SSE transport — one SandboxPool, multiple concurrent sessions."""

    def __init__(self, pool: SandboxPool, host: str = "0.0.0.0", port: int = 8400,
                 pipeline: "ToolPipeline | None" = None,
                 surface: "ExecSurface | None" = None,
                 notify_mutations: bool = False):
        self.pool = pool
        self.host = host
        self.port = port
        self.pipeline = pipeline
        self.surface = surface or DEFAULT_SURFACE
        # notify_mutations: call the pool's after_mutating_call after a call that
        # could have changed the filesystem. Off by default; the orchestrator turns
        # it on, because a tool call IS the step boundary for an external agent and
        # that is where a checkpoint belongs.
        self.notify_mutations = notify_mutations
        self._sessions: dict[str, Session] = {}

    #: Headers that identify the caller, most explicit first. ``mcp-session-id``
    #: is the protocol's own mechanism: ``initialize`` returns it both as a
    #: response header (which is what makes a conforming client echo it back)
    #: and in the result body.
    SESSION_HEADERS = ("x-session-owner", "mcp-session-id")

    #: Binds a session to one pre-provisioned sandbox. The caller that created
    #: the sandbox states its id here; that session is then served the
    #: single-sandbox schema and every exec call resolves to it implicitly.
    SANDBOX_HEADER = "x-session-sandbox"

    def _get_or_create_session(self, headers: dict) -> Session:
        owner = next((headers[h] for h in self.SESSION_HEADERS if headers.get(h)), None)
        if owner is None:
            # Anonymous request: a fresh identity, so sandboxes stay private.
            # Interceptors keyed by agent_id (the guardrails) can hold no
            # state across such requests — each one looks like a new agent — so
            # a client that wants governance must identify itself.
            owner = str(uuid.uuid4())
        if owner not in self._sessions:
            groups_header = headers.get("x-session-groups", "")
            explicit_groups = [g.strip() for g in groups_header.split(",") if g.strip()]
            # owner_group is always included — ensures private sandbox visibility
            session = Session(id=owner, groups=[f"owner:{owner}"] + explicit_groups)
            self._sessions[owner] = session

        session = self._sessions[owner]
        bound = headers.get(self.SANDBOX_HEADER)
        if bound:
            # Re-read every request: the header is the caller's standing
            # statement of which sandbox this slot owns, and a slot that is
            # re-boarded onto a new sandbox (checkpoint restore) says so by
            # changing it. Binding is a *default*, not a grant -- the visibility
            # check in _resolve still applies, so naming someone else's sandbox
            # here gains nothing.
            session.bound_id = bound
        return session

    @property
    def base_url(self) -> str:
        """Where a client should point. 0.0.0.0 is a bind address, not a
        destination -- a wiring built from it fails on some stacks."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", "::") else self.host
        return f"http://{host}:{self.port}/mcp"

    def _build_app(self):
        from aiohttp import web

        async def handle_mcp(request: web.Request) -> web.Response:
            headers = {k.lower(): v for k, v in request.headers.items()}
            session = self._get_or_create_session(headers)
            handler = SessionHandler(session, self.pool, pipeline=self.pipeline,
                                     surface=self.surface,
                                     notify_mutations=self.notify_mutations)

            body = await request.json()
            method = body.get("method", "")
            id_ = body.get("id")

            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ash-sandbox", "version": "2.0.0"},
                    "sessionId": session.id,
                }
                # The response *header* is what makes a conforming MCP client
                # echo the id back on later requests; the body field alone is
                # invisible to it, so without this every request looked like a
                # new anonymous session and sandboxes created by an earlier one
                # were no longer visible.
                return web.json_response(
                    {"jsonrpc": "2.0", "id": id_, "result": result},
                    headers={"Mcp-Session-Id": session.id},
                )

            elif method == "tools/list":
                # A bound session gets the single-sandbox surface: no sandbox_id
                # parameter to fill in, so the model cannot target another
                # sandbox and cannot omit the argument either.
                tools = (self.surface.single if session.bound_id
                         else self.surface.all_tools())
                return web.json_response({"jsonrpc": "2.0", "id": id_, "result": {"tools": tools}})

            elif method == "tools/call":
                params = body.get("params", {})
                # Stateless: every exec call carries its own sandbox_id, so
                # concurrent same-session requests share no mutable routing state.
                content = await handler.call_tool(params.get("name", ""), params.get("arguments", {}))
                return web.json_response({
                    "jsonrpc": "2.0", "id": id_,
                    "result": {"content": [content], "isError": content.get("isError", False)},
                })

            elif method == "ping":
                return web.json_response({"jsonrpc": "2.0", "id": id_, "result": {}})

            else:
                return web.json_response({
                    "jsonrpc": "2.0", "id": id_,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                })

        app = web.Application()
        app.router.add_post("/mcp", handle_mcp)
        return app

    async def run(self, sock=None, stop: "asyncio.Event | None" = None):
        """Serve until ``stop`` is set, or forever.

        ``sock`` is a pre-bound listening socket, which is how :meth:`start`
        knows the port before the loop exists: with ``port=0`` the kernel picks
        one, and asking aiohttp for it afterwards means racing the caller that
        wants to build a URL.
        """
        from aiohttp import web

        app = self._build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = (web.SockSite(runner, sock) if sock is not None
                else web.TCPSite(runner, self.host, self.port))
        await site.start()
        self._log(f"listening on {self.host}:{self.port}")
        try:
            await (stop or asyncio.Event()).wait()
        finally:
            # An adopted sandbox is *released* here, not destroyed: its owner is
            # still holding it (see SandboxPool.adopt).
            await self.pool.destroy_all()
            await runner.cleanup()

    # --- in-process transport ----------------------------------------------
    def start(self) -> "HttpMcpServer":
        """Run in a background thread; return once it is accepting connections.

        The orchestrator needs this because it owns the session: an out-of-process
        server would have to create its own sandbox, and the owner would then be
        unable to snapshot the environment its agent actually worked in -- the
        ownership inversion the ``--image`` flag used to cause. In-process, the
        sandbox is handed over with :meth:`SandboxPool.adopt` and there is one
        handle to it.

        The socket is bound *here*, before the thread starts, so ``port=0``
        resolves to a real port that :attr:`base_url` can report immediately.
        """
        import socket
        import threading

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(128)
        self.port = sock.getsockname()[1]

        ready = threading.Event()
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._stop: "asyncio.Event | None" = None

        def serve() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._stop = asyncio.Event()
            ready.set()
            try:
                loop.run_until_complete(self.run(sock=sock, stop=self._stop))
            finally:
                loop.close()

        self._thread = threading.Thread(target=serve, daemon=True,
                                        name="ash-mcp-http")
        self._thread.start()
        ready.wait(timeout=10)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Stop serving and release the pool. Safe to call twice."""
        loop, stop = getattr(self, "_loop", None), getattr(self, "_stop", None)
        if loop is not None and stop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop.set)
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None

    def _log(self, text: str):
        sys.stderr.write(f"[ash-mcp-http] {text}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Transport: stdio (single-session, backwards-compat)
# ---------------------------------------------------------------------------

class StdioMcpServer:
    """Stdio transport — single session, backwards-compatible."""

    def __init__(self, pool: SandboxPool, single_sandbox: bool = False,
                 pipeline: "ToolPipeline | None" = None,
                 surface: "ExecSurface | None" = None):
        self.pool = pool
        self.session = Session(id="stdio", groups=["owner:stdio", "default"])
        self.surface = surface or DEFAULT_SURFACE
        self.handler = SessionHandler(self.session, pool, notify_mutations=single_sandbox,
                                      pipeline=pipeline, surface=self.surface)
        # single_sandbox: expose only exec tools bound to the active sandbox
        # (lifecycle tools hidden — the harness pre-provisions the sandbox).
        self.single_sandbox = single_sandbox

    async def run(self):
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = await self._handle(msg)
                if response:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
        finally:
            await self.pool.destroy_all()

    async def _handle(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        id_ = msg.get("id")

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": id_, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ash-sandbox", "version": "2.0.0"},
            }}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            tools = (self.surface.single if self.single_sandbox
                     else self.surface.all_tools())
            return {"jsonrpc": "2.0", "id": id_, "result": {"tools": tools}}
        elif method == "tools/call":
            params = msg.get("params", {})
            content = await self.handler.call_tool(params.get("name", ""), params.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": id_,
                    "result": {"content": [content], "isError": content.get("isError", False)}}
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": id_, "result": {}}
        else:
            return {"jsonrpc": "2.0", "id": id_,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def _build_pipeline(args) -> "ToolPipeline | None":
    """Assemble the L2 interceptor pipeline (docs/ARCHITECTURE.md).

    Default OFF: with no --plugins or --guardrails the proxy dispatches tool calls
    exactly as before. --plugins replaces the default assembly entirely — the
    PIPELINE list in the file is the configuration, which is also how a
    coordination interceptor comes back (Waggle was removed; see git history).

    Order is semantics: guardrails annotate on the way out, so their ``after``
    runs last and a warning is appended to whatever result the model ends up
    seeing. Truncation sits innermost, so it bounds the runtime's result and the
    interceptors above it annotate already-bounded text.
    """
    if args.plugins:
        return load_pipeline(args.plugins)
    interceptors: list = []
    if args.guardrails:
        interceptors.append(GuardrailInterceptor(enforcement=args.guardrails))
    if interceptors and args.max_output_bytes > 0:
        # Only with another interceptor: mounting the chain for truncation alone would
        # change the default dispatch path, and the runtime already bounds its
        # own output (ASH_MAX_OUTPUT_BYTES).
        interceptors.append(TruncateInterceptor(max_len=args.max_output_bytes))
    return ToolPipeline(interceptors) if interceptors else None


def main(pool_cls=None):
    """Run the proxy. ``pool_cls`` lets a caller supply a SandboxPool subclass
    that hooks sandbox lifecycle (see swebench/mcp_server.py)."""
    import argparse
    parser = argparse.ArgumentParser(description="Ash sandbox MCP server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server (multi-session)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument(
        "--attach", default=None, metavar="SANDBOX_ID",
        help="Serve tool calls into a sandbox the caller already created (stdio "
             "mode). The caller keeps the handle, so it can snapshot the session "
             "and extract from it afterwards -- which is why the proxy no longer "
             "creates sandboxes itself.")
    parser.add_argument("--runtime-bin", default=None)
    parser.add_argument("--backend", default=None, choices=sorted(BACKENDS),
                        help="Where sandboxes come from (default: docker). "
                             "'microvm' needs AENV_SERVER_URL (+ AENV_API_KEY); "
                             "'k8s' needs ASH_CONTROL_PLANE_URL and "
                             "ASH_GATEWAY_URL.")
    parser.add_argument("--guardrails", nargs="?", const="warn", default=None,
                        choices=["warn", "reject"],
                        help="Mount read-before-edit / edit-streak guardrails: "
                             "'warn' annotates the result, 'reject' refuses the "
                             "call. Default: off.")
    parser.add_argument("--max-output-bytes", type=int, default=12000,
                        help="Bound each tool result to this many characters "
                             "when the pipeline is mounted (0 disables)")
    parser.add_argument("--plugins", default=None,
                        help="Python file exporting PIPELINE: list[ToolInterceptor]; "
                             "replaces the default pipeline assembly")
    parser.add_argument("--tools", default=None, metavar="PANEL",
                        help="Serve a compiled tool panel instead of the built-in "
                             "four: a shipped name (default, full, bash_only, "
                             "no_web) or a path to your own manifest. Compiled "
                             "against runtime/schema/tools.json, so a manifest the "
                             "runtime cannot serve fails here rather than at the "
                             "model's first call.")
    args = parser.parse_args()

    # Same reason as the backend check below: a panel that cannot compile must stop
    # startup. Serving a stale or partial tool list instead is precisely the drift
    # compiling the panel exists to prevent.
    surface = DEFAULT_SURFACE
    if args.tools:
        from harness.execution.panel import load_panel

        try:
            panel = load_panel(args.tools, format="raw")
        except Exception as exc:
            sys.stderr.write(f"[ash-mcp] cannot load tool panel {args.tools!r}: {exc}\n")
            raise SystemExit(2)
        surface = ExecSurface(panel)
        sys.stderr.write("[ash-mcp] tool panel %s: %s\n"
                         % (args.tools, ", ".join(sorted(surface.names))))
        sys.stderr.flush()

    # Fail at startup, not on the first sandbox_create: a proxy that accepted a
    # backend it cannot build would report the misconfiguration as a tool error
    # to whatever client happened to ask first.
    backend = {"backend": args.backend} if args.backend else {}
    try:
        build_pool(backend, runtime_bin=args.runtime_bin)
    except BackendError as exc:
        sys.stderr.write(f"[ash-mcp] {exc}\n")
        raise SystemExit(2)

    pool = (pool_cls or SandboxPool)(runtime_bin=args.runtime_bin, backend=backend)
    pipeline = _build_pipeline(args)
    if pipeline is not None:
        names = ", ".join(i.name for i in pipeline.interceptors) or "(empty)"
        sys.stderr.write(f"[ash-mcp] interceptor pipeline: {names}\n")
        if args.http:
            # Per-agent state is keyed by session identity, so an unidentified
            # client gets a new one per request and the chain can hold nothing.
            sys.stderr.write(
                "[ash-mcp] note: interceptor state is per session — clients must send "
                f"one of {', '.join(HttpMcpServer.SESSION_HEADERS)}\n")
        sys.stderr.flush()

    if args.http:
        server = HttpMcpServer(pool, host=args.host, port=args.port,
                               pipeline=pipeline, surface=surface)
        asyncio.run(server.run())
    else:
        async def run_stdio():
            # A bound session is served the single-sandbox schema, so the model
            # never sees a sandbox_id argument to fill in.
            single = bool(args.attach)
            stdio = StdioMcpServer(pool, single_sandbox=single,
                                   pipeline=pipeline, surface=surface)
            if args.attach:
                try:
                    entry = await pool.attach(args.attach,
                                              groups=["owner:stdio", "default"])
                except Exception as e:
                    sys.stderr.write(
                        f"[ash-mcp] cannot attach to sandbox '{args.attach}': {e}\n")
                    sys.stderr.flush()
                    raise SystemExit(1)
                stdio.session.bound_id = entry.id
            await stdio.run()
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
