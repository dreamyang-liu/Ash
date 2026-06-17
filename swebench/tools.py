"""Tool schemas for SWE-bench agent.

Exposes ash-runtime tools directly via OpenAI function-calling format.
The agent calls tools by name (shell, text_editor, grep_files, etc.)
and the session routes them to the sandbox via SDK.
"""

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
            "name": "read_file",
            "description": (
                "Read a file and return contents with line numbers.\n"
                "Use for reading specific sections of files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "offset": {"type": "integer", "description": "Start line (1-based, default: 1)"},
                    "limit": {"type": "integer", "description": "Number of lines to read (default: all)"},
                },
                "required": ["path"],
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
                "  read: Get new output lines since last read (incremental)\n"
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
]
