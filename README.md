# Ash CLI - Agent Shell for Code Tasks

A minimal CLI and MCP server for AI agents to interact with sandboxed environments.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Agent     │────▶│  Control Plane   │────▶│   K8s Cluster   │
│   (LLM)     │     │   POST /spawn    │     │   (sandboxes)   │
└─────────────┘     │ DELETE /depr...  │     └─────────────────┘
       │            └──────────────────┘              │
       │                                              │
       │            ┌──────────────────┐              │
       └───────────▶│   MCP Gateway    │◀─────────────┘
                    │  X-Session-ID    │
                    │   routes calls   │
                    └──────────────────┘
```

## Quick Start

```bash
# Set endpoints (or use defaults)
export ASH_CONTROL_PLANE_URL="http://control-plane:80"
export ASH_GATEWAY_URL="http://gateway:80"

# Create a sandbox session
ash session create
# Output: {"session_id":"abc123","status":"Ready","host":"..."}

# Run commands in the sandbox
ash --session abc123 run "ls -la"
ash --session abc123 run "cat README.md"

# Destroy when done
ash session destroy abc123
```

---

## Tools Reference

### Session Management

#### `session_create`
Create a new sandbox. Returns a `session_id` for subsequent operations.

```bash
# Default image
ash session create

# Custom image
ash session create --image "python:3.11"

# With resources
ash session create --image "ubuntu:22.04" --memory "4Gi" --cpus "2"
```

**MCP:**
```json
{
  "name": "session_create",
  "arguments": {
    "image": "python:3.11",
    "env": {"DEBUG": "1"},
    "ports": [3000, 8080],
    "resources": {
      "requests": {"cpu": "100m", "memory": "256Mi"},
      "limits": {"cpu": "1", "memory": "1Gi"}
    },
    "node_selector": {"gpu": "true"}
  }
}
```

**Returns:**
```json
{"session_id": "abc123", "status": "Ready", "host": "sandbox-abc123.default.svc"}
```

#### `session_list`
List all active sessions.

```bash
ash session list
```

#### `session_destroy`
Destroy a sandbox by session_id.

```bash
ash session destroy abc123
```

---

### Shell Execution

#### `shell`
Execute a shell command. If `session_id` is provided, runs in that sandbox via MCP Gateway.

```bash
# Local execution
ash run "ls -la"

# In sandbox
ash --session abc123 run "ls -la"
ash --session abc123 run "python train.py"
ash --session abc123 run "pip install numpy && python -c 'import numpy; print(numpy.__version__)'"
```

**MCP:**
```json
{
  "name": "shell",
  "arguments": {
    "command": "python train.py",
    "session_id": "abc123",
    "timeout_secs": 300
  }
}
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `command` | string | yes | - | Shell command to execute |
| `session_id` | string | no | - | Execute in this sandbox |
| `timeout_secs` | int | no | 300 | Timeout in seconds |

---

### File Operations

#### `read_file`
Read file contents with line numbers.

```bash
ash view /path/to/file.py
ash view /path/to/file.py -n 50 -l 20  # Start at line 50, show 20 lines
```

**MCP:**
```json
{
  "name": "read_file",
  "arguments": {
    "file_path": "/testbed/src/main.py",
    "offset": 1,
    "limit": 100,
    "session_id": "abc123"
  }
}
```

**Output format:**
```
     1 | def main():
     2 |     print("Hello")
     3 |     return 0
```

#### `grep_files`
Search for patterns using ripgrep.

```bash
ash grep "def.*test" src/
ash grep "TODO" . --include "*.py"
```

**MCP:**
```json
{
  "name": "grep_files",
  "arguments": {
    "pattern": "def.*test",
    "path": "src/",
    "include": "*.py",
    "limit": 100,
    "session_id": "abc123"
  }
}
```

#### `text_editor`
Edit files with four commands: `view`, `str_replace`, `insert`, `create`.

**view** - View file with line range:
```bash
ash edit view /path/file.py --start 10 --end 30
```
```json
{"name": "text_editor", "arguments": {"command": "view", "path": "/testbed/src/lib.py", "view_range": [10, 30]}}
```

**str_replace** - Replace text (must be unique):
```bash
ash edit replace /path/file.py --old "old_text" --new "new_text"
```
```json
{"name": "text_editor", "arguments": {"command": "str_replace", "path": "/testbed/src/lib.py", "old_str": "def foo():", "new_str": "def foo(x):"}}
```

**insert** - Insert text after a line:
```bash
ash edit insert /path/file.py --line 10 --text "# New comment"
```
```json
{"name": "text_editor", "arguments": {"command": "insert", "path": "/testbed/src/lib.py", "insert_line": 10, "insert_text": "    # Added line"}}
```

**create** - Create a new file:
```bash
ash edit create /path/new_file.py "#!/usr/bin/env python3\nprint('hello')"
```
```json
{"name": "text_editor", "arguments": {"command": "create", "path": "/testbed/src/new.py", "file_text": "# New file\n"}}
```

---

### Git Operations

#### `git_status`
```bash
ash git-status
ash git-status --short
```
```json
{"name": "git_status", "arguments": {"short": true, "session_id": "abc123"}}
```

#### `git_diff`
```bash
ash git-diff
ash git-diff --staged
ash git-diff src/main.py src/lib.py
```
```json
{"name": "git_diff", "arguments": {"staged": true, "paths": ["src/main.py"], "session_id": "abc123"}}
```

#### `git_log`
```bash
ash git-log -n 5
ash git-log --oneline
```
```json
{"name": "git_log", "arguments": {"count": 10, "oneline": true, "session_id": "abc123"}}
```

---

### Clipboard (Agent Memory)

Clipboard provides named storage for text snippets - useful for tracking context across multiple operations.

#### `clip`
Save content to clipboard.

```bash
# Direct text
ash clip "important note" -n memo

# From file range
ash clip -f src/lib.rs:10-20 -n code_snippet
```

**MCP:**
```json
{"name": "clip", "arguments": {"content": "error: type mismatch", "name": "error_msg"}}
{"name": "clip", "arguments": {"file": "src/lib.py:50-60", "name": "function_def"}}
```

#### `paste`
Retrieve from clipboard.

```bash
ash paste              # Latest
ash paste error_msg    # By name
```

**MCP:**
```json
{"name": "paste", "arguments": {"name": "error_msg"}}
```

#### `clips`
List all clipboard entries.

```bash
ash clips
```

#### `clips_clear`
Clear clipboard entries.

```bash
ash clips-clear           # Clear all
ash clips-clear old_note  # Clear specific
```

---

## Common Workflows

### 1. Debug a Failing Test

```bash
# Create sandbox
ash session create --image python:3.11
# → {"session_id": "sess1", ...}

# Run the failing test
ash --session sess1 run "pytest tests/test_main.py -v"

# Find the error
ash --session sess1 run "grep -n 'def test_' tests/test_main.py"

# View the test
ash --session sess1 view tests/test_main.py -n 45 -l 20

# Fix it
ash --session sess1 edit replace tests/test_main.py \
  --old "assert result == 1" \
  --new "assert result == 2"

# Re-run
ash --session sess1 run "pytest tests/test_main.py -v"

# Cleanup
ash session destroy sess1
```

### 2. Explore and Modify Codebase

```bash
# Search for relevant code
ash --session sess1 grep "class.*Handler" src/

# View the file
ash --session sess1 view src/handlers.py -n 100 -l 50

# Save important snippet to clipboard
ash --session sess1 clip -f src/handlers.py:100-120 -n handler_class

# Make changes
ash --session sess1 edit replace src/handlers.py \
  --old "def process(self):" \
  --new "def process(self, data):"

# Check diff
ash --session sess1 git-diff src/handlers.py
```

### 3. SWE-bench Task Pattern

```bash
# 1. Create sandbox for the task
ash session create --image swebench/sweb.eval.x86_64.sympy__sympy

# 2. Explore the issue
ash --session $SID run "cat /testbed/issue.txt"
ash --session $SID grep "relevant_function" /testbed/

# 3. View and understand the code
ash --session $SID view /testbed/sympy/core/expr.py -n 500 -l 50

# 4. Make the fix
ash --session $SID edit replace /testbed/sympy/core/expr.py \
  --old "..." --new "..."

# 5. Test the fix
ash --session $SID run "python -c 'from sympy import ...; ...'"

# 6. Create patch
ash --session $SID run "git diff -- sympy/core/expr.py > /tmp/patch.txt"
ash --session $SID run "cat /tmp/patch.txt"

# 7. Submit
ash --session $SID run "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt"
```

---

## MCP Server Mode

Run ash as an MCP server for integration with MCP clients:

```bash
# Start MCP server (stdio transport)
ash-mcp

# Or with HTTP transport (future)
ash-mcp --transport http --port 8080
```

**MCP Protocol:**
- `initialize` - Handshake
- `tools/list` - List available tools
- `tools/call` - Execute a tool

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASH_CONTROL_PLANE_URL` | `http://localhost:8080` | Control plane endpoint |
| `ASH_GATEWAY_URL` | `http://localhost:8081` | MCP gateway endpoint |

### CLI Global Options

| Flag | Description |
|------|-------------|
| `--session <ID>` | Run commands in this session's sandbox |
| `--output text\|json` | Output format |
| `-h, --help` | Show help |
| `-V, --version` | Show version |

---

## Error Handling

**Common errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Session not found` | Invalid session_id | Check `ash session list` |
| `Connection refused` | Control plane down | Check endpoint URLs |
| `Timeout` | Long-running command | Increase `timeout_secs` |
| `Multiple matches` | str_replace not unique | Use more specific `old_str` |
| `No match found` | str_replace text not in file | Verify with `view` first |

---

## Tips for LLM Agents

1. **Always use `session_id`** for sandbox operations - don't rely on local filesystem.

2. **View before editing** - Use `read_file` or `text_editor view` to see current state.

3. **str_replace must be unique** - Include enough context to match exactly once.

4. **Use clipboard for context** - Store error messages, important code snippets for reference.

5. **Check exit codes** - Non-zero exit means failure; read stderr for details.

6. **Incremental changes** - Make small edits, test frequently, don't batch large changes.

7. **Clean up sessions** - Always destroy sessions when done to free resources.
