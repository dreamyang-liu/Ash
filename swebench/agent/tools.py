"""Tool schemas for SWE-bench agent.

Exposes ash-runtime tools directly via OpenAI function-calling format.
The agent calls tools by name (shell, text_editor, grep_files, etc.)
and the session routes them to the sandbox via SDK.
"""

BASH_ONLY_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

# Derived from the SDK's data-driven route table (single source of truth,
# ash_sandbox.toolset.BUILTIN_ROUTES). "bash" is excluded here: the
# executor handles the bash_only alias itself. Consumers unchanged:
# route_agent_tool below, agent/__init__.py, tests.
from ash_sandbox.toolset import BUILTIN_ROUTES as _BUILTIN_ROUTES

AGENT_TOOL_ROUTES = {k: v for k, v in _BUILTIN_ROUTES.items() if k != "bash"}


def route_agent_tool(name: str, args: dict) -> tuple[str, dict]:
    """Translate an agent-facing tool call to a runtime tool call."""
    runtime_tool = AGENT_TOOL_ROUTES.get(name)
    if runtime_tool is None:
        raise KeyError(f"unknown agent tool: {name}")
    return runtime_tool, dict(args)


def is_custom_tool(name: str) -> bool:
    """Whether name is a registered manifest-defined custom tool.

    Custom tools don't go through route_agent_tool; the session executor
    uses custom_tools.plan_custom_tool to expand them into artifact+shell.
    """
    from .custom_tools import CUSTOM_TOOL_SPECS

    return name in CUSTOM_TOOL_SPECS


def truncate_output(content: str, max_len: int = 12000) -> str:
    """Elide the middle of overly long tool output."""
    if len(content) <= max_len:
        return content
    head, tail = max_len * 2 // 3, max_len // 3  # ~8000 / ~4000 chars
    elided = len(content) - head - tail
    return (
        content[:head] +
        f"\n\n... [{elided} characters truncated — output too long. Use `tail` on shell "
        f"commands, `limit` on grep, or pipe through `grep` for targeted output] ...\n\n" +
        content[-tail:]
    )


def tool_summary(name: str, args: dict) -> str:
    """Build a human-readable one-line summary for a tool call (for display)."""
    if name == "shell":
        cmd = args.get("command", "")
        return cmd + (" &" if args.get("background") else "")
    elif name == "grep_files":
        parts = [f"/{args.get('pattern', '')}/"]
        if args.get("path"):
            parts.append(args["path"])
        if args.get("include"):
            parts.append(f"({args['include']})")
        return " ".join(parts)
    elif name == "text_editor":
        cmd = args.get("command", "")
        path = args.get("path", "")
        if cmd == "str_replace":
            preview = args.get("old_str", "")[:40].replace("\n", "\\n")
            return f'{path} [{cmd}] "{preview}"'
        elif cmd == "view":
            vr = args.get("view_range")
            return f"{path} [{vr[0]}:{vr[1]}]" if vr else f"{path} [view]"
        return f"{path} [{cmd}]"
    elif name == "process":
        return f"{args.get('pid', '?')} {args.get('action', '?')}"
    elif name == "web_fetch":
        return args.get("url", "")
    elif name == "web_search":
        return args.get("query", "")
    return args.get("command", "") or args.get("path", "") or str(args)[:80]


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Execute a shell command synchronously or in the background.\n"
                "Use for: running tests (pytest), installing packages (pip), git operations, "
                "and any other shell commands.\n"
                "Working directory defaults to /testbed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "background": {"type": "boolean", "default": False, "description": "Run in background, returns pid"},
                    "timeout": {"type": "integer", "default": 300, "description": "Timeout in seconds"},
                    "tail": {"type": "integer", "description": "Only return last N lines of output"},
                    "max_output_bytes": {
                        "type": "integer",
                        "default": 1048576,
                        "description": "Maximum captured bytes per output stream. Larger output keeps the first 40% and last 60%.",
                    },
                    "working_dir": {"type": "string", "description": "Working directory (default: /testbed)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_editor",
            "description": (
                "View or edit files.\n"
                "Commands:\n"
                "  view: Read file contents (optionally a line range)\n"
                "  str_replace: Replace exact text in a file\n"
                "  insert: Insert text after a specific line\n"
                "  write: Write full content to a file (creates or overwrites)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["view", "str_replace", "insert", "write"], "description": "Command to execute"},
                    "path": {"type": "string", "description": "File path"},
                    "view_range": {"type": "array", "items": {"type": "integer"}, "description": "For view only: [start, end] inclusive line range."},
                    "old_str": {"type": "string", "description": "Required for str_replace. Text to find; must match exactly once."},
                    "new_str": {"type": "string", "description": "Required for str_replace. Replacement text."},
                    "insert_line": {"type": "integer", "description": "Required for insert. Line number to insert after; use 0 to insert at the start."},
                    "insert_text": {"type": "string", "description": "Required for insert. Text to insert."},
                    "file_text": {"type": "string", "description": "Required for write. Full file content; creates or overwrites the file."},
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": (
                "Search files using ripgrep with a regex pattern.\n"
                "Use for finding code patterns, function definitions, imports, etc."
            ),
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "process",
            "description": (
                "Manage a background process started with shell(background=true).\n"
                "Actions:\n"
                "  read: Get the current bounded output snapshot\n"
                "  kill: Terminate the process"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "string", "description": "Process ID returned by shell(background=true)"},
                    "action": {"type": "string", "enum": ["read", "kill"], "description": "read or kill"},
                    "tail": {"type": "integer", "description": "Only return last N lines (read only)"},
                },
                "required": ["pid", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its content in the specified format",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "format": {"type": "string", "enum": ["html", "text", "markdown"], "default": "markdown"},
                    "headers": {"type": "object", "description": "Additional HTTP headers"},
                    "timeout": {
                        "type": "integer",
                        "default": 15,
                        "description": "Request timeout in seconds. Values are clamped to the runtime maximum.",
                    },
                    "max_length": {
                        "type": "integer",
                        "default": 10000,
                        "description": "Maximum returned characters. Values are clamped to the runtime maximum.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return results from multiple engines",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "backend": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "brave",
                            "duckduckgo",
                            "google",
                            "grokipedia",
                            "mojeek",
                            "startpage",
                            "wikipedia",
                            "yahoo",
                            "yandex",
                        ],
                        "default": "auto",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum results. Values are clamped to the runtime maximum.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_events",
            "description": (
                "Block until the sandbox reports an event (e.g. a background process "
                "exits or a file changes), or until the timeout elapses. Use this "
                "instead of polling in a loop when waiting on background work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            'Event kinds to wait for, e.g. ["process_exited", '
                            '"file_change"]. Omit to wait for any event.'
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 30,
                        "description": "Seconds to wait. Values are clamped to the runtime maximum.",
                    },
                },
                "required": [],
            },
        },
    },
]
