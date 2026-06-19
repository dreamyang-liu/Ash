"""MCP server that exposes ash sandbox tools to Claude Code.

Usage:
    python -m swebench.mcp_server --image <docker-image> --patch-file /tmp/patch.diff

This starts a stdio MCP server with 5 tools (shell, text_editor, grep_files,
read_file, process) routed to an ash sandbox container. Claude Code connects
to this server and uses the tools directly.

On shutdown, extracts the git diff from /testbed and writes it to --patch-file.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ash_sandbox import DockerPool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult


TOOLS = [
    {
        "name": "shell",
        "description": (
            "Execute a shell command in the sandbox container.\n"
            "Working directory defaults to /testbed.\n"
            "Use 'tail' to limit output lines. Use 'background: true' for long-running commands."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "background": {"type": "boolean", "default": False, "description": "Run in background, returns pid"},
                "timeout": {"type": "integer", "default": 300, "description": "Timeout in seconds"},
                "tail": {"type": "integer", "description": "Only return last N lines of output"},
                "working_dir": {"type": "string", "description": "Working directory (default: /testbed)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "text_editor",
        "description": (
            "View or edit files in the sandbox.\n"
            "Commands: view, str_replace, insert, write"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["view", "str_replace", "insert", "write"]},
                "path": {"type": "string", "description": "File path"},
                "view_range": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] line range for view"},
                "old_str": {"type": "string", "description": "Text to find (str_replace)"},
                "new_str": {"type": "string", "description": "Replacement text (str_replace)"},
                "insert_line": {"type": "integer", "description": "Line number to insert after"},
                "insert_text": {"type": "string", "description": "Text to insert"},
                "file_text": {"type": "string", "description": "Full file content (write)"},
            },
            "required": ["command", "path"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search files using ripgrep with a regex pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search"},
                "path": {"type": "string", "description": "Directory or file to search (default: /testbed)"},
                "include": {"type": "string", "description": "File glob to include (e.g. '*.py')"},
                "limit": {"type": "integer", "description": "Max number of results"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file with line numbers from the sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "offset": {"type": "integer", "description": "Start line (1-based)"},
                "limit": {"type": "integer", "description": "Number of lines to read"},
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
                "pid": {"type": "string", "description": "Process ID from shell(background=true)"},
                "action": {"type": "string", "enum": ["read", "kill"]},
                "tail": {"type": "integer", "description": "Only return last N lines"},
            },
            "required": ["pid", "action"],
        },
    },
]


class McpServer:
    """Minimal MCP stdio server wrapping an ash sandbox."""

    def __init__(self, image: str, runtime_bin: str | None = None, patch_file: str | None = None):
        self.image = image
        self.runtime_bin = runtime_bin
        self.patch_file = patch_file
        self._pool: DockerPool | None = None
        self._sandbox: Sandbox | None = None
        self._base_commit: str = ""

    async def start(self):
        """Spawn the sandbox container."""
        self._pool = DockerPool(runtime_bin=self.runtime_bin)
        self._sandbox = await self._pool.spawn(image=self.image)
        cid = self._sandbox._container_id or "unknown"
        self._log(f"sandbox ready: {cid[:12]}")
        # Record initial HEAD for patch extraction
        r = await self._sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
        if not r.is_error:
            self._base_commit = r.output.strip()

    async def stop(self):
        """Extract patch then destroy the sandbox."""
        if self._sandbox and self.patch_file:
            await self._extract_patch()
        if self._pool and self._sandbox:
            await self._pool.destroy(self._sandbox)
            self._log("sandbox destroyed")

    async def _extract_patch(self):
        """Extract git diff and write to patch_file."""
        try:
            await self._sandbox.call("shell", command="cd /testbed && git add -A")
            base = self._base_commit or "HEAD"
            r = await self._sandbox.call("shell", command=f"cd /testbed && git diff {base}")
            patch = r.output.rstrip("\r\n")
            if patch:
                patch += "\n"
            Path(self.patch_file).write_text(patch)
            self._log(f"patch saved: {len(patch)} chars")
        except Exception as e:
            self._log(f"patch extraction failed: {e}")
            Path(self.patch_file).write_text("")

    async def call_tool(self, name: str, args: dict) -> dict:
        """Execute a tool in the sandbox, return MCP content."""
        if not self._sandbox:
            return {"type": "text", "text": "Error: no sandbox"}

        try:
            result: SdkToolResult = await self._sandbox.call(name, **args)
            return {
                "type": "text",
                "text": result.output,
                "isError": result.is_error,
            }
        except Exception as e:
            return {"type": "text", "text": f"Error: {e}", "isError": True}

    async def run_stdio(self):
        """Run the MCP stdio protocol loop."""
        await self.start()
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

                response = await self._handle_message(msg)
                if response:
                    self._write(response)
        finally:
            await self.stop()

    async def _handle_message(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        id_ = msg.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ash-sandbox", "version": "1.0.0"},
                },
            }

        elif method == "notifications/initialized":
            return None

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name", "")
            args = params.get("arguments", {})
            content = await self.call_tool(name, args)
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [content],
                    "isError": content.get("isError", False),
                },
            }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": id_, "result": {}}

        else:
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

    def _write(self, msg: dict):
        data = json.dumps(msg)
        sys.stdout.write(data + "\n")
        sys.stdout.flush()

    def _log(self, text: str):
        sys.stderr.write(f"[ash-mcp] {text}\n")
        sys.stderr.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ash sandbox MCP server")
    parser.add_argument("--image", required=True, help="Docker image to spawn")
    parser.add_argument("--runtime-bin", default=None, help="Path to ash-runtime binary")
    parser.add_argument("--patch-file", default=None, help="Write final git diff to this file on exit")
    args = parser.parse_args()

    server = McpServer(image=args.image, runtime_bin=args.runtime_bin, patch_file=args.patch_file)
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
