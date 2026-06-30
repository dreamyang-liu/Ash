"""MCP server that exposes ash sandbox tools to Claude Code.

Supports multiple sandboxes with ownership + group-based visibility:
- Each sandbox has an owner (session) and optional groups
- A session sees: its own private sandboxes + any shared sandbox whose groups
  overlap with the session's groups
- Runs as HTTP/SSE for multi-session, or stdio for single-session (backwards-compat)

Usage:
    # HTTP mode (multi-session):
    python -m swebench.mcp_server --http --port 8400

    # Stdio mode (single-session, backwards-compat):
    python -m swebench.mcp_server --image <docker-image> --patch-dir /tmp/patches/
"""

import asyncio
import copy
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash_sandbox import DockerPool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult


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

    def visible_to(self, session_groups: list[str]) -> bool:
        return bool(set(self.groups) & set(session_groups))


@dataclass
class Session:
    id: str
    groups: list[str] = field(default_factory=lambda: ["default"])
    active_id: str | None = None

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
        "name": "sandbox_switch",
        "description": "Switch your active sandbox (default target for tool calls).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sandbox_id": {"type": "string", "description": "Sandbox ID to switch to"},
            },
            "required": ["sandbox_id"],
        },
    },
    {
        "name": "sandbox_destroy",
        "description": "Destroy a sandbox (must have your owner group). Extracts patch before destroying.",
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
        "name": "read_file",
        "description": "Read a file with line numbers from a sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
                "sandbox_id": {"type": "string", "description": "Target sandbox (default: active)"},
            },
            "required": ["path"],
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

ALL_TOOLS = LIFECYCLE_TOOLS + EXEC_TOOLS


def _single_sandbox_tools() -> list[dict]:
    """Exec tools only, with the multi-sandbox `sandbox_id` arg removed.

    Used in single-sandbox stdio mode: the sandbox is pre-provisioned and set
    active by the harness, so the agent should see only shell/text_editor/
    grep_files/read_file/process — no lifecycle tools, no sandbox routing.
    """
    tools = copy.deepcopy(EXEC_TOOLS)
    for t in tools:
        t["inputSchema"]["properties"].pop("sandbox_id", None)
    return tools


EXEC_TOOLS_SINGLE = _single_sandbox_tools()


# ---------------------------------------------------------------------------
# Sandbox Pool (shared across all sessions)
# ---------------------------------------------------------------------------

class SandboxPool:
    """Manages all sandboxes. Session-agnostic — visibility is enforced at the server layer."""

    def __init__(self, runtime_bin: str | None = None, patch_dir: str | None = None):
        self.runtime_bin = runtime_bin
        self.patch_dir = Path(patch_dir) if patch_dir else None
        self._pool: DockerPool | None = None
        self._sandboxes: dict[str, SandboxEntry] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"sb-{self._counter}"

    async def create(self, image: str, groups: list[str]) -> SandboxEntry:
        if not self._pool:
            self._pool = DockerPool(runtime_bin=self.runtime_bin)
        sandbox = await self._pool.spawn(image=image)
        r = await sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
        base_commit = r.output.strip() if not r.is_error else ""
        sb_id = self._next_id()
        entry = SandboxEntry(id=sb_id, sandbox=sandbox, image=image,
                             groups=groups, base_commit=base_commit)
        self._sandboxes[sb_id] = entry
        self._log(f"created {sb_id} groups={groups}")
        return entry

    async def destroy(self, sb_id: str) -> str:
        """Destroy sandbox, return its patch."""
        entry = self._sandboxes.pop(sb_id, None)
        if not entry:
            return ""
        patch = await self._extract_patch(entry)
        if self._pool:
            await self._pool.destroy(entry.sandbox)
        self._log(f"destroyed {sb_id}")
        return patch

    async def destroy_all(self):
        for sb_id in list(self._sandboxes):
            await self.destroy(sb_id)

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

    async def _extract_patch(self, entry: SandboxEntry) -> str:
        try:
            await entry.sandbox.call("shell", command="cd /testbed && git add -A")
            base = entry.base_commit or "HEAD"
            r = await entry.sandbox.call("shell", command=f"cd /testbed && git diff {base}")
            patch = r.output.rstrip("\r\n")
            if patch:
                patch += "\n"
            if self.patch_dir:
                self.patch_dir.mkdir(parents=True, exist_ok=True)
                (self.patch_dir / f"{entry.id}.diff").write_text(patch)
            return patch
        except Exception as e:
            self._log(f"patch extraction failed for {entry.id}: {e}")
            return ""

    def _log(self, text: str):
        sys.stderr.write(f"[ash-pool] {text}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Session handler (per-connection tool dispatch)
# ---------------------------------------------------------------------------

class SessionHandler:
    """Handles tool calls scoped to a single session's visibility."""

    # Exec tools that can mutate the filesystem — refresh the patch after these.
    _MUTATING = {"shell", "text_editor", "process"}

    def __init__(self, session: Session, pool: SandboxPool, auto_extract: bool = False):
        self.session = session
        self.pool = pool
        # auto_extract: re-extract the diff after every mutating tool call so the
        # patch file is always current (single-sandbox mode). Removes any reliance
        # on shutdown-time extraction, which races the harness read under load.
        self.auto_extract = auto_extract

    def _resolve(self, sandbox_id: str | None) -> SandboxEntry | None:
        """Resolve target sandbox, checking visibility via group intersection."""
        if sandbox_id:
            entry = self.pool.get(sandbox_id)
            if entry and entry.visible_to(self.session.groups):
                return entry
            return None
        if self.session.active_id:
            entry = self.pool.get(self.session.active_id)
            if entry and entry.visible_to(self.session.groups):
                return entry
        return None

    async def call_tool(self, name: str, args: dict) -> dict:
        # -- Lifecycle tools --
        if name == "sandbox_create":
            # Always include the caller's owner group; add any extra shared groups
            extra_groups = args.get("groups", [])
            groups = [self.session.owner_group] + extra_groups
            entry = await self.pool.create(args["image"], groups)
            self.session.active_id = entry.id
            return _ok(json.dumps({"id": entry.id, "groups": entry.groups}))

        if name == "sandbox_list":
            entries = self.pool.visible_to(self.session, args.get("group"))
            items = [{"id": e.id, "image": e.image, "groups": e.groups,
                      "mine": self.session.owner_group in e.groups,
                      "active": e.id == self.session.active_id} for e in entries]
            return _ok(json.dumps(items, indent=2))

        if name == "sandbox_switch":
            sb_id = args["sandbox_id"]
            entry = self.pool.get(sb_id)
            if not entry or not entry.visible_to(self.session.groups):
                return _err(f"sandbox {sb_id} not found or not accessible")
            self.session.active_id = sb_id
            return _ok(f"Switched to {sb_id}")

        if name == "sandbox_destroy":
            sb_id = args["sandbox_id"]
            entry = self.pool.get(sb_id)
            if not entry:
                return _err(f"sandbox {sb_id} not found")
            if self.session.owner_group not in entry.groups:
                return _err(f"cannot destroy {sb_id}: not the owner")
            patch = await self.pool.destroy(sb_id)
            if self.session.active_id == sb_id:
                visible = self.pool.visible_to(self.session)
                self.session.active_id = visible[0].id if visible else None
            return _ok(f"Destroyed {sb_id}. Patch: {len(patch)} chars.")

        # -- Exec tools --
        sandbox_id = args.pop("sandbox_id", None)
        entry = self._resolve(sandbox_id)
        if not entry:
            return _err("No active sandbox visible to you. Use sandbox_create or sandbox_switch.")

        try:
            result: SdkToolResult = await entry.sandbox.call(name, **args)
        except Exception as e:
            return _err(str(e))

        # Keep the patch file current so the harness never reads a stale/missing
        # diff (no dependence on shutdown timing). Best-effort: never fail the
        # tool call because extraction hiccupped.
        if self.auto_extract and self.pool.patch_dir and name in self._MUTATING:
            try:
                await self.pool._extract_patch(entry)
            except Exception:
                pass

        return {"type": "text", "text": result.output, "isError": result.is_error}


def _ok(text: str) -> dict:
    return {"type": "text", "text": text}


def _err(text: str) -> dict:
    return {"type": "text", "text": f"Error: {text}", "isError": True}


# ---------------------------------------------------------------------------
# Transport: HTTP/SSE (multi-session)
# ---------------------------------------------------------------------------

class HttpMcpServer:
    """HTTP/SSE transport — one SandboxPool, multiple concurrent sessions."""

    def __init__(self, pool: SandboxPool, host: str = "0.0.0.0", port: int = 8400):
        self.pool = pool
        self.host = host
        self.port = port
        self._sessions: dict[str, Session] = {}

    def _get_or_create_session(self, headers: dict) -> Session:
        owner = headers.get("x-session-owner", str(uuid.uuid4()))
        if owner not in self._sessions:
            groups_header = headers.get("x-session-groups", "")
            explicit_groups = [g.strip() for g in groups_header.split(",") if g.strip()]
            # owner_group is always included — ensures private sandbox visibility
            session = Session(id=owner, groups=[f"owner:{owner}"] + explicit_groups)
            self._sessions[owner] = session
        return self._sessions[owner]

    async def run(self):
        from aiohttp import web

        async def handle_mcp(request: web.Request) -> web.Response:
            headers = {k.lower(): v for k, v in request.headers.items()}
            session = self._get_or_create_session(headers)
            handler = SessionHandler(session, self.pool)

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
                return web.json_response({"jsonrpc": "2.0", "id": id_, "result": result})

            elif method == "tools/list":
                return web.json_response({"jsonrpc": "2.0", "id": id_, "result": {"tools": ALL_TOOLS}})

            elif method == "tools/call":
                params = body.get("params", {})
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

        self._log(f"listening on {self.host}:{self.port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        try:
            await asyncio.Event().wait()  # run forever
        finally:
            await self.pool.destroy_all()
            await runner.cleanup()

    def _log(self, text: str):
        sys.stderr.write(f"[ash-mcp-http] {text}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Transport: stdio (single-session, backwards-compat)
# ---------------------------------------------------------------------------

class StdioMcpServer:
    """Stdio transport — single session, backwards-compatible."""

    def __init__(self, pool: SandboxPool, single_sandbox: bool = False):
        self.pool = pool
        self.session = Session(id="stdio", groups=["owner:stdio", "default"])
        self.handler = SessionHandler(self.session, pool, auto_extract=single_sandbox)
        # single_sandbox: expose only exec tools bound to the active sandbox
        # (lifecycle tools hidden — the harness pre-provisions the sandbox).
        self.single_sandbox = single_sandbox

    async def run(self):
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

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
            tools = EXEC_TOOLS_SINGLE if self.single_sandbox else ALL_TOOLS
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ash sandbox MCP server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server (multi-session)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--image", default=None, help="Auto-create a sandbox on start (stdio mode)")
    parser.add_argument("--runtime-bin", default=None)
    parser.add_argument("--patch-dir", default=None, help="Directory for per-sandbox patch files")
    parser.add_argument("--patch-file", default=None, help="(deprecated) alias for --patch-dir parent")
    args = parser.parse_args()

    patch_dir = args.patch_dir
    if not patch_dir and args.patch_file:
        patch_dir = str(Path(args.patch_file).parent)

    pool = SandboxPool(runtime_bin=args.runtime_bin, patch_dir=patch_dir)

    if args.http:
        server = HttpMcpServer(pool, host=args.host, port=args.port)
        asyncio.run(server.run())
    else:
        async def run_stdio():
            # When an image is given, pre-provision the sandbox and set it active
            # so the agent gets a ready-to-use environment and only sees exec tools.
            single = bool(args.image)
            stdio = StdioMcpServer(pool, single_sandbox=single)
            if args.image:
                try:
                    entry = await pool.create(args.image, groups=["owner:stdio", "default"])
                except Exception as e:
                    sys.stderr.write(
                        f"[ash-mcp] failed to create sandbox from image '{args.image}': {e}\n")
                    sys.stderr.flush()
                    raise SystemExit(1)
                stdio.session.active_id = entry.id
                # Write an initial (empty) patch file immediately so the harness
                # always finds a current diff, even if the agent never edits.
                if pool.patch_dir:
                    try:
                        await pool._extract_patch(entry)
                    except Exception:
                        pass
            await stdio.run()
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
