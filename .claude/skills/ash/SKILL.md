---
name: ash
description: Use Ash sandboxed environments to run code, execute commands, and manage files locally or in isolated Docker/K8s containers. All tool execution routes through ash-mcp via a Gateway for unified behavior. Invoke when the user wants to run untrusted code, test in clean environments, work with SWE-bench tasks, or needs isolated sandboxes.
allowed-tools: Bash(ash *), Bash(./ash *), Bash(./target/release/ash *), Bash(./target/debug/ash *)
argument-hint: "[command or description of what to do]"
---

# Ash - Sandboxed Execution Skill

You have access to `ash`, a CLI for managing sandboxed environments. Use it to create isolated containers, run commands, edit files, and manage processes in Docker or K8s sandboxes.

## Quick Reference

```bash
# Session lifecycle
ash session create                          # create sandbox (default image)
ash session create --image python:3.11      # custom image
ash session list                            # list all sessions
ash session destroy <session_id>            # cleanup

# Run commands in sandbox
ash --session <id> run "<command>"           # sync execution
ash --session <id> run "pip install numpy && python script.py"

# File operations in sandbox
ash --session <id> view /path/file.py                              # read file
ash --session <id> view /path/file.py -n 50 -l 20                 # lines 50-69
ash --session <id> grep "pattern" /path/ --include "*.py"          # search
ash --session <id> edit replace /path/file.py --old "x" --new "y"  # edit
ash --session <id> edit insert /path/file.py --line 10 --text "new line"

# Git in sandbox
ash --session <id> git-status
ash --session <id> git-diff
ash --session <id> git-log -n 5

# Async processes
ash --session <id> terminal start "long_running_command"
ash --session <id> terminal output <handle>
ash --session <id> terminal list
ash --session <id> terminal kill <handle>

# Local (no sandbox)
ash run "ls -la"
ash view ./src/main.rs
ash grep "TODO" ./src/
```

## Workflow

1. **Create a session** before doing sandbox work:
   ```bash
   ash session create --image python:3.11
   ```
   Save the returned `session_id`.

2. **Use `--session <id>`** for all subsequent commands to run them inside the sandbox.

3. **Destroy the session** when done to free resources:
   ```bash
   ash session destroy <session_id>
   ```

## Architecture

Ash uses a **Gateway + ash-mcp** architecture:

- **Gateway** (`ash gateway start/stop/status`): Routes tool calls to ash-mcp endpoints. Auto-starts on first CLI call via Unix socket (`~/.ash/gateway.sock`).
- **ash-mcp**: Executes all tools uniformly — same binary runs locally (as a subprocess), in Docker containers, and in K8s pods.
- Session management tools run inside the gateway; all other tools are forwarded to the appropriate ash-mcp instance.

## Key Rules

- **Always pass `--session <id>`** for sandbox operations. Without it, commands run locally.
- **View before editing** - read the file first with `ash view` or `ash edit view` to know what you're changing.
- **`str_replace` must be unique** - include enough surrounding context so the old text matches exactly once.
- For **long-running commands**, use `ash terminal start` instead of `ash run` to avoid timeout.
- Default `ash run` timeout is 300s. Override with `--timeout <secs>`.

## Output Format

- Default output is human-readable text.
- Add `--output json` for structured JSON output (useful for parsing).

For the complete tool reference with all 54 tools and their parameters, see [tools-reference.md](tools-reference.md).

## User Request

$ARGUMENTS
