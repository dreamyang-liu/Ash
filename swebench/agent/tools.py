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
# ash_sandbox.toolset.BUILTIN_ROUTES). "bash" is excluded: this dict mirrors the
# default tool schema the model sees, and `bash` belongs to BASH_ONLY_SCHEMA
# instead. route_agent_tool handles it separately.
from ash_sandbox.toolset import BUILTIN_ROUTES as _BUILTIN_ROUTES

AGENT_TOOL_ROUTES = {k: v for k, v in _BUILTIN_ROUTES.items() if k != "bash"}

#: text_editor commands that modify a file. The single source of truth for
#: "this call is an edit", shared by everything that has to reason about it:
#: Waggle's write arbitration (waggle.py) and the guardrails (guardrails.py).
EDIT_COMMANDS = frozenset({"str_replace", "insert", "write"})

#: Edits that rewrite *existing* content, so "you did not read this file first"
#: is unambiguous. `write` is excluded on purpose: it also creates files, and
#: telling creation from overwrite needs a filesystem probe. Waggle pays for
#: that probe and refuses only blind overwrites (`_write_unregistered`); a rule
#: that cannot afford the probe must not claim to cover `write`, or creating a
#: new file becomes an unsatisfiable warning — or, when enforced, impossible.
CONTENT_EDIT_COMMANDS = frozenset({"str_replace", "insert"})


def route_agent_tool(name: str, args: dict) -> tuple[str, dict]:
    """Translate an agent-facing tool call to a runtime tool call.

    Covers the bash_only alias as well as the default surface. The agent loop
    routes before handing a call to its executor so that interceptors see the
    runtime tool: a seat keyed on `shell` must not go blind because a run is in
    bash_only mode.
    """
    runtime_tool = _BUILTIN_ROUTES.get(name)
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
                        "description": "Total bytes of output to return. Larger output is truncated per truncate_mode.",
                    },
                    "truncate_mode": {
                        "type": "string",
                        "default": "H2T3",
                        "description": (
                            'How to divide the byte budget when output is too long: "H<n>T<n>" '
                            "with weights for the head and tail sections. H2T3 keeps the first "
                            "40% and last 60%; T1 keeps only the tail (useful for build/test "
                            "errors); H1 keeps only the beginning."
                        ),
                    },
                    "working_dir": {"type": "string", "description": "Working directory (default: /testbed)"},
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Data to feed the command on standard input, then close it. "
                            "Lets you run 'python -', 'patch -p1', or 'sh -s' without "
                            "first writing a temporary file."
                        ),
                    },
                    "env": {
                        "type": "object",
                        "description": (
                            'Extra environment variables, e.g. {"PYTHONPATH": "/testbed"}. '
                            "Added to the existing environment rather than replacing it. "
                            "Prefer this over a 'KEY=value cmd' prefix: values needing "
                            "quotes or spaces are handled correctly."
                        ),
                    },
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
                "Observe asynchronous sandbox facts. Delivery is opt-in: "
                "action=subscribe registers interest in event kinds (optionally "
                "narrowed to specific sources) so they arrive with later tool "
                "responses; action=wait (the default) blocks until a matching event "
                "occurs or the timeout elapses -- use it instead of polling in a "
                "loop when waiting on background work. Events expire automatically "
                "after their time-to-live."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["wait", "subscribe", "unsubscribe", "subscriptions"],
                        "default": "wait",
                        "description": (
                            "wait: block for a matching event. subscribe: receive these "
                            "kinds with later tool responses (nothing is delivered "
                            "without a subscription). unsubscribe: stop receiving them. "
                            "subscriptions: list active ones."
                        ),
                    },
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            'Event kinds, e.g. ["process_exited", '
                            '"tool:text_editor", "tool:web_fetch"]. Any tool '
                            'call is observable as "tool:<name>". Required for '
                            "subscribe; omit when waiting to accept any kind."
                        ),
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Wait only for events from these handles: a pid returned by a "
                            "background shell call, or a file path. Omit to accept any "
                            "source. Use this to wait on one specific background process."
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
