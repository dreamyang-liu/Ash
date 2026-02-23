"""Tool schemas for SWE-bench agent.

Single bash tool — agent writes ash CLI commands directly.
Session routing is handled by ASH_SESSION env var.
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the sandbox (/testbed).\n"
                "Use `ash` CLI for all operations:\n"
                "  ash grep \"<pattern>\" [path] [-i GLOB] [-l N]\n"
                "  ash edit view <file> [--start N --end N]\n"
                "  ash edit replace <file> --old \"...\" --new \"...\"\n"
                "  ash edit create <file> \"content\"\n"
                "  ash find \"<glob>\" [path] | ash outline <file> | ash ls <path>\n"
                "  ash run \"<shell_command>\" [--tail N]\n"
                "  ash undo [file] | ash buffer | ash terminal\n"
                "Run `ash --help` for all commands. Composable: &&, |, ;, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
