# ASH TOOL REFERENCE

Ash provides 54 tools via MCP protocol and an equivalent CLI. Every MCP tool has a 1:1 CLI counterpart.

Tools that accept `session_id` can execute inside a sandbox container. Without `session_id`, they run locally.

## INDEX

| # | Lines | Category | Tools | CLI |
|---|-------|----------|-------|-----|
| 1 | 25-128 | [File I/O](#1-file-io) | `read_file` `grep_files` `text_editor` | `ash view` `ash grep` `ash edit` |
| 2 | 129-236 | [Filesystem](#2-filesystem) | `fs_list_dir` `fs_stat` `fs_write` `fs_mkdir` `fs_remove` `fs_move` `fs_copy` | `ash fs <ls\|stat\|write\|mkdir\|rm\|mv\|cp>` |
| 3 | 237-258 | [Shell](#3-shell) | `shell` | `ash run` |
| 4 | 259-338 | [Async Terminal](#4-async-terminal) | `terminal_run_async` `terminal_get_output` `terminal_kill` `terminal_list` `terminal_remove` | `ash terminal <start\|output\|kill\|list\|remove>` |
| 5 | 339-413 | [Git](#5-git) | `git_status` `git_diff` `git_log` `git_add` `git_commit` | `ash git-status` `ash git-diff` `ash git-log` `ash git-add` `ash git-commit` |
| 6 | 414-478 | [Clipboard](#6-clipboard) | `clip` `paste` `clips` `clips_clear` | `ash clip` `ash paste` `ash clips` `ash clips-clear` |
| 7 | 479-614 | [Buffer](#7-buffer) | `buffer_read` `buffer_write` `buffer_delete` `buffer_replace` `buffer_list` `buffer_clear` `buffer_to_clip` `clip_to_buffer` | `ash buffer <read\|write\|delete\|replace\|list\|clear\|to-clip\|from-clip>` |
| 8 | 615-713 | [Session / Sandbox](#8-session--sandbox) | `session_create` `session_info` `session_list` `session_destroy` `backend_switch` `backend_status` | `ash session <create\|info\|list\|destroy\|switch>` `ash config` |
| 9 | 714-829 | [Utilities](#9-utilities) | `find_files` `tree` `diff_files` `patch_apply` `http_fetch` `file_info` `undo` | `ash find` `ash tree` `ash diff` `ash patch` `ash fetch` `ash file-info` `ash undo` |
| 10 | 830-881 | [Events](#10-events) | `events_subscribe` `events_poll` `events_push` | `ash events <subscribe\|poll\|push>` |
| 11 | 882-956 | [Custom Tools](#11-custom-tools) | `tool_create` `tool_list` `tool_view` `tool_run` `tool_remove` | `ash custom-tool <create\|list\|view\|run\|remove>` |

---

## 1. FILE I/O

### read_file
Read file contents with line numbers.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_path` | string | yes | | File path to read |
| `offset` | int | no | 1 | Start line (1-indexed) |
| `limit` | int | no | 100 | Max lines to show |
| `session_id` | string | no | | Execute in sandbox |

```json
{"name": "read_file", "arguments": {"file_path": "/testbed/src/main.py", "offset": 50, "limit": 30}}
```
```bash
ash view /testbed/src/main.py -n 50 -l 30
```

Output format:
```
    50 | def process(data):
    51 |     return data.strip()
```

### grep_files
Search for regex pattern in files using ripgrep.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pattern` | string | yes | | Regex pattern |
| `path` | string | no | `.` | Search directory |
| `include` | string | no | | File glob filter (e.g. `*.py`) |
| `limit` | int | no | 100 | Max results |
| `session_id` | string | no | | Execute in sandbox |

```json
{"name": "grep_files", "arguments": {"pattern": "def test_", "path": "src/", "include": "*.py"}}
```
```bash
ash grep "def test_" src/ --include "*.py"
```

### text_editor
Edit files with four commands.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | yes | `view`, `str_replace`, `insert`, or `create` |
| `path` | string | yes | File path |
| `session_id` | string | no | Execute in sandbox |

**view** — show lines:
| Param | Type | Description |
|-------|------|-------------|
| `view_range` | [int, int] | [start, end] line range |

```json
{"name": "text_editor", "arguments": {"command": "view", "path": "src/lib.py", "view_range": [10, 30]}}
```
```bash
ash edit view src/lib.py --start 10 --end 30
```

**str_replace** — find and replace (match must be unique):
| Param | Type | Description |
|-------|------|-------------|
| `old_str` | string | Text to find (must appear exactly once) |
| `new_str` | string | Replacement text |

```json
{"name": "text_editor", "arguments": {"command": "str_replace", "path": "src/lib.py", "old_str": "def foo():", "new_str": "def foo(x):"}}
```
```bash
ash edit replace src/lib.py --old "def foo():" --new "def foo(x):"
```

**insert** — insert text after a line:
| Param | Type | Description |
|-------|------|-------------|
| `insert_line` | int | Line number to insert after |
| `insert_text` | string | Text to insert |

```json
{"name": "text_editor", "arguments": {"command": "insert", "path": "src/lib.py", "insert_line": 10, "insert_text": "    # comment"}}
```
```bash
ash edit insert src/lib.py --line 10 --text "    # comment"
```

**create** — create new file:
| Param | Type | Description |
|-------|------|-------------|
| `file_text` | string | File content |

```json
{"name": "text_editor", "arguments": {"command": "create", "path": "src/new.py", "file_text": "#!/usr/bin/env python3\n"}}
```
```bash
ash edit create src/new.py "#!/usr/bin/env python3"
```

---

## 2. FILESYSTEM

### fs_list_dir
List directory contents with sizes.

| Param | Type | Required |
|-------|------|----------|
| `path` | string | yes |

```json
{"name": "fs_list_dir", "arguments": {"path": "/testbed/src"}}
```
```bash
ash fs ls /testbed/src
```

### fs_stat
Get file/directory metadata (size, permissions, timestamps).

| Param | Type | Required |
|-------|------|----------|
| `path` | string | yes |

```json
{"name": "fs_stat", "arguments": {"path": "/testbed/src/main.py"}}
```
```bash
ash fs stat /testbed/src/main.py
```

### fs_write
Write content to file. Creates parent directories automatically.

| Param | Type | Required |
|-------|------|----------|
| `path` | string | yes |
| `content` | string | yes |

```json
{"name": "fs_write", "arguments": {"path": "/testbed/output.txt", "content": "result: ok\n"}}
```
```bash
ash fs write /testbed/output.txt "result: ok"
```

### fs_mkdir
Create directory.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `path` | string | yes | |
| `recursive` | bool | no | true |

```json
{"name": "fs_mkdir", "arguments": {"path": "/testbed/src/utils"}}
```
```bash
ash fs mkdir /testbed/src/utils
```

### fs_remove
Remove file or directory.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `path` | string | yes | |
| `recursive` | bool | no | false |

```json
{"name": "fs_remove", "arguments": {"path": "/testbed/tmp", "recursive": true}}
```
```bash
ash fs rm /testbed/tmp -r
```

### fs_move
Move or rename file/directory.

| Param | Type | Required |
|-------|------|----------|
| `from` | string | yes |
| `to` | string | yes |

```json
{"name": "fs_move", "arguments": {"from": "old.py", "to": "new.py"}}
```
```bash
ash fs mv old.py new.py
```

### fs_copy
Copy file or directory.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `from` | string | yes | |
| `to` | string | yes | |
| `recursive` | bool | no | false |

```json
{"name": "fs_copy", "arguments": {"from": "src/", "to": "src_backup/", "recursive": true}}
```
```bash
ash fs cp src/ src_backup/ -r
```

---

## 3. SHELL

### shell
Execute a shell command synchronously. Returns stdout, stderr, and exit code.

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | string | yes | | Shell command |
| `timeout_secs` | int | no | 300 | Timeout in seconds |
| `working_dir` | string | no | | Working directory |
| `tail_lines` | int | no | | Only return last N lines |
| `session_id` | string | no | | Execute in sandbox |

```json
{"name": "shell", "arguments": {"command": "python -m pytest tests/ -v", "timeout_secs": 120, "session_id": "s1"}}
```
```bash
ash --session s1 run "python -m pytest tests/ -v" --timeout 120
```

---

## 4. ASYNC TERMINAL

For long-running processes. Start a process, get a handle, poll for output later.

### terminal_run_async
Start a background process. Returns a handle ID.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | yes | Shell command |
| `working_dir` | string | no | Working directory |
| `env` | object | no | Environment variables |

```json
{"name": "terminal_run_async", "arguments": {"command": "python train.py --epochs 100"}}
```
```bash
ash terminal start "python train.py --epochs 100"
```

Returns:
```json
{"handle": "a1b2c3", "pid": 12345, "status": "running"}
```

### terminal_get_output
Get output from an async process.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `handle` | string | yes | Process handle |
| `tail` | int | no | Only return last N lines |

```json
{"name": "terminal_get_output", "arguments": {"handle": "a1b2c3", "tail": 50}}
```
```bash
ash terminal output a1b2c3 --tail 50
```

### terminal_kill
Kill a running async process.

| Param | Type | Required |
|-------|------|----------|
| `handle` | string | yes |

```json
{"name": "terminal_kill", "arguments": {"handle": "a1b2c3"}}
```
```bash
ash terminal kill a1b2c3
```

### terminal_list
List all tracked async processes.

```json
{"name": "terminal_list", "arguments": {}}
```
```bash
ash terminal list
```

### terminal_remove
Remove a completed process from tracking.

| Param | Type | Required |
|-------|------|----------|
| `handle` | string | yes |

```json
{"name": "terminal_remove", "arguments": {"handle": "a1b2c3"}}
```
```bash
ash terminal remove a1b2c3
```

---

## 5. GIT

All git tools accept optional `session_id` to run inside a sandbox.

### git_status

| Param | Type | Default |
|-------|------|---------|
| `short` | bool | false |

```json
{"name": "git_status", "arguments": {"short": true}}
```
```bash
ash git-status --short
```

### git_diff

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `staged` | bool | false | Show staged changes (--staged) |
| `paths` | [string] | [] | Specific paths to diff |

```json
{"name": "git_diff", "arguments": {"staged": true, "paths": ["src/main.py"]}}
```
```bash
ash git-diff --staged src/main.py
```

### git_log

| Param | Type | Default |
|-------|------|---------|
| `count` | int | 10 |
| `oneline` | bool | false |

```json
{"name": "git_log", "arguments": {"count": 5, "oneline": true}}
```
```bash
ash git-log -n 5 --oneline
```

### git_add

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `paths` | [string] | [] | Files to stage |
| `all` | bool | false | Stage all (-A) |

```json
{"name": "git_add", "arguments": {"paths": ["src/fix.py"]}}
```
```bash
ash git-add src/fix.py
```

### git_commit

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `message` | string | yes | |
| `all` | bool | no | false |

```json
{"name": "git_commit", "arguments": {"message": "fix: handle edge case in parser"}}
```
```bash
ash git-commit -m "fix: handle edge case in parser"
```

---

## 6. CLIPBOARD

Named key-value storage for text snippets. Persists across tool calls.

### clip
Save content to a named clipboard entry.

| Param | Type | Description |
|-------|------|-------------|
| `content` | string | Direct text content |
| `file` | string | File path, optionally with `:start-end` (e.g. `src/lib.rs:10-20`) |
| `name` | string | Entry name (auto-generated if omitted) |
| `source` | string | Source reference override |

Provide either `content` or `file`, not both.

```json
{"name": "clip", "arguments": {"content": "TypeError: expected int", "name": "error"}}
{"name": "clip", "arguments": {"file": "src/lib.py:50-70", "name": "handler_fn"}}
```
```bash
ash clip "TypeError: expected int" -n error
ash clip -f src/lib.py:50-70 -n handler_fn
```

### paste
Retrieve a clipboard entry.

| Param | Type | Description |
|-------|------|-------------|
| `name` | string | Entry name (latest if omitted) |

```json
{"name": "paste", "arguments": {"name": "error"}}
```
```bash
ash paste error
```

### clips
List all clipboard entries.

```json
{"name": "clips", "arguments": {}}
```
```bash
ash clips
```

### clips_clear
Clear clipboard entries.

| Param | Type | Description |
|-------|------|-------------|
| `name` | string | Specific entry to remove (all if omitted) |

```json
{"name": "clips_clear", "arguments": {"name": "error"}}
```
```bash
ash clips-clear error
```

---

## 7. BUFFER

Named, line-addressable text buffers. A persistent scratchpad for composing content, accumulating results, or building files incrementally. Lines are 1-indexed.

### buffer_write
Write content to a buffer. Creates the buffer if it doesn't exist.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `main` | Buffer name |
| `content` | string | (required) | Text content |
| `at_line` | int | | Insert before this line |
| `append` | bool | false | Append to end |

Without `at_line` or `append`, replaces entire buffer.

```json
{"name": "buffer_write", "arguments": {"content": "def main():\n    pass\n"}}
{"name": "buffer_write", "arguments": {"content": "# appended", "append": true}}
{"name": "buffer_write", "arguments": {"name": "scratch", "content": "inserted", "at_line": 5}}
```
```bash
ash buffer write "def main():\n    pass\n"
ash buffer write --append "# appended"
ash buffer write -n scratch --at-line 5 "inserted"
```

### buffer_read
Read lines from a buffer.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `main` | Buffer name |
| `start_line` | int | 1 | Start line (1-indexed) |
| `end_line` | int | end | End line (inclusive) |

```json
{"name": "buffer_read", "arguments": {"name": "main", "start_line": 10, "end_line": 30}}
```
```bash
ash buffer read --start 10 --end 30
```

### buffer_delete
Delete a range of lines.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `main` | Buffer name |
| `start_line` | int | (required) | First line to delete |
| `end_line` | int | (required) | Last line to delete |

```json
{"name": "buffer_delete", "arguments": {"start_line": 5, "end_line": 10}}
```
```bash
ash buffer delete --start 5 --end 10
```

### buffer_replace
Replace a range of lines with new content.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `main` | Buffer name |
| `start_line` | int | (required) | First line to replace |
| `end_line` | int | (required) | Last line to replace |
| `content` | string | (required) | Replacement content |

```json
{"name": "buffer_replace", "arguments": {"start_line": 5, "end_line": 10, "content": "new line 1\nnew line 2"}}
```
```bash
ash buffer replace --start 5 --end 10 "new line 1\nnew line 2"
```

### buffer_list
List all buffers with line counts.

```json
{"name": "buffer_list", "arguments": {}}
```
```bash
ash buffer list
```

### buffer_clear
Clear buffer content or delete a buffer.

| Param | Type | Description |
|-------|------|-------------|
| `name` | string | Buffer to clear (all if omitted) |

```json
{"name": "buffer_clear", "arguments": {"name": "scratch"}}
```
```bash
ash buffer clear -n scratch
```

### buffer_to_clip
Copy a buffer range to a clipboard entry.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `buffer` | string | `main` | Source buffer |
| `start_line` | int | 1 | Start line |
| `end_line` | int | end | End line |
| `clip_name` | string | (required) | Clipboard entry name |

```json
{"name": "buffer_to_clip", "arguments": {"start_line": 1, "end_line": 20, "clip_name": "draft"}}
```
```bash
ash buffer to-clip --start 1 --end 20 draft
```

### clip_to_buffer
Paste a clipboard entry into a buffer.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_name` | string | (required) | Clipboard entry to paste |
| `buffer` | string | `main` | Target buffer |
| `at_line` | int | | Insert at this line |
| `append` | bool | false | Append to end |

```json
{"name": "clip_to_buffer", "arguments": {"clip_name": "draft", "append": true}}
```
```bash
ash buffer from-clip draft --append
```

---

## 8. SESSION / SANDBOX

Manage isolated container environments. Supports Docker (local) and K8s (remote) backends.

### session_create
Create a new sandbox session.

| Param | Type | Description |
|-------|------|-------------|
| `backend` | string | `docker` or `k8s` (auto-detected if omitted) |
| `name` | string | Custom session name |
| `image` | string | Container image |
| `env` | object | Environment variables |
| `ports` | [int] | Ports to expose |
| `working_dir` | string | Working directory |
| `resources` | object | `{cpu, memory, cpu_limit, memory_limit}` |
| `labels` | object | Labels / node selector |

```json
{"name": "session_create", "arguments": {"image": "python:3.11", "env": {"DEBUG": "1"}, "ports": [8080]}}
```
```bash
ash session create --image python:3.11 -e DEBUG=1 -p 8080
```

Returns:
```json
{"session_id": "abc123", "backend": "docker", "status": "running"}
```

### session_info
Get session details.

| Param | Type | Required |
|-------|------|----------|
| `session_id` | string | yes |

```json
{"name": "session_info", "arguments": {"session_id": "abc123"}}
```
```bash
ash session info abc123
```

### session_list
List all active sessions across all backends.

```json
{"name": "session_list", "arguments": {}}
```
```bash
ash session list
```

### session_destroy
Destroy a session and its container.

| Param | Type | Required |
|-------|------|----------|
| `session_id` | string | yes |

```json
{"name": "session_destroy", "arguments": {"session_id": "abc123"}}
```
```bash
ash session destroy abc123
```

### backend_switch
Switch a session to a different backend.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | |
| `backend` | string | yes | `docker` or `k8s` |

```json
{"name": "backend_switch", "arguments": {"session_id": "abc123", "backend": "k8s"}}
```
```bash
ash session switch abc123 k8s
```

### backend_status
Check backend health.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `backend` | string | yes | `docker` or `k8s` |

```json
{"name": "backend_status", "arguments": {"backend": "docker"}}
```
```bash
ash config
```

---

## 9. UTILITIES

### find_files
Find files by name pattern (glob).

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `pattern` | string | yes | |
| `path` | string | no | `.` |
| `max_depth` | int | no | |
| `limit` | int | no | 100 |

```json
{"name": "find_files", "arguments": {"pattern": "*.py", "path": "src/", "max_depth": 3}}
```
```bash
ash find "*.py" src/ --max-depth 3
```

### tree
Show directory tree.

| Param | Type | Default |
|-------|------|---------|
| `path` | string | `.` |
| `max_depth` | int | 3 |
| `show_hidden` | bool | false |

```json
{"name": "tree", "arguments": {"path": "/testbed", "max_depth": 2}}
```
```bash
ash tree /testbed --max-depth 2
```

### diff_files
Compare two files (unified diff).

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `file1` | string | yes | |
| `file2` | string | yes | |
| `context` | int | no | 3 |

```json
{"name": "diff_files", "arguments": {"file1": "a.py", "file2": "b.py"}}
```
```bash
ash diff a.py b.py
```

### patch_apply
Apply a unified diff patch.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `patch` | string | yes | |
| `path` | string | no | |
| `dry_run` | bool | no | false |

```json
{"name": "patch_apply", "arguments": {"patch": "--- a/f.py\n+++ b/f.py\n@@ ...", "dry_run": true}}
```
```bash
ash patch "--- a/f.py..." --dry-run
```

### http_fetch
HTTP GET request.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `url` | string | yes | |
| `headers` | object | no | |
| `timeout_secs` | int | no | 30 |

```json
{"name": "http_fetch", "arguments": {"url": "https://api.example.com/data"}}
```
```bash
ash fetch "https://api.example.com/data"
```

### file_info
Get file type and encoding info.

| Param | Type | Required |
|-------|------|----------|
| `path` | string | yes |

```json
{"name": "file_info", "arguments": {"path": "data.bin"}}
```
```bash
ash file-info data.bin
```

### undo
Undo last file edit made by `text_editor`.

| Param | Type | Description |
|-------|------|-------------|
| `path` | string | Specific file (defaults to last edited) |
| `list` | bool | List undo history instead of undoing |

```json
{"name": "undo", "arguments": {"list": true}}
{"name": "undo", "arguments": {"path": "src/main.py"}}
```
```bash
ash undo --list
ash undo src/main.py
```

---

## 10. EVENTS

Pub/sub event system for tracking process completions, file changes, and custom events.

### events_subscribe
Subscribe to event types.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `events` | [string] | yes | Event types: `process_complete`, `file_change`, `error`, `custom` |
| `unsubscribe` | bool | no | Unsubscribe instead |

```json
{"name": "events_subscribe", "arguments": {"events": ["process_complete", "file_change"]}}
```
```bash
ash events subscribe process_complete file_change
```

### events_poll
Poll pending events.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 10 | Max events to retrieve |
| `peek` | bool | false | Peek without removing from queue |

```json
{"name": "events_poll", "arguments": {"limit": 5}}
```
```bash
ash events poll --limit 5
```

### events_push
Push a custom event.

| Param | Type | Default |
|-------|------|---------|
| `kind` | string | (required) |
| `source` | string | `llm` |
| `data` | object | |

```json
{"name": "events_push", "arguments": {"kind": "task_complete", "data": {"result": "pass"}}}
```
```bash
ash events push task_complete --data '{"result":"pass"}'
```

---

## 11. CUSTOM TOOLS

Create, manage, and run user-defined shell/python scripts as tools.

### tool_create

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Tool name (becomes `<name>.sh` or `<name>.py`) |
| `script` | string | (required) | Script content (first `#` comment = description) |
| `lang` | string | `sh` | `sh` or `python` |

```json
{"name": "tool_create", "arguments": {"name": "count_lines", "script": "# Count lines in a file\nwc -l \"$1\"", "lang": "sh"}}
```
```bash
ash custom-tool create count_lines -s '# Count lines\nwc -l "$1"' -l sh
```

### tool_list
List all custom tools.

```json
{"name": "tool_list", "arguments": {}}
```
```bash
ash custom-tool list
```

### tool_view
View a custom tool's script.

| Param | Type | Required |
|-------|------|----------|
| `name` | string | yes |

```json
{"name": "tool_view", "arguments": {"name": "count_lines"}}
```
```bash
ash custom-tool view count_lines
```

### tool_run
Run a custom tool.

| Param | Type | Description |
|-------|------|-------------|
| `name` | string | Tool name |
| `args` | [string] | Positional arguments |
| `env` | object | Environment variables |

```json
{"name": "tool_run", "arguments": {"name": "count_lines", "args": ["src/main.py"]}}
```
```bash
ash custom-tool run count_lines src/main.py
```

### tool_remove
Remove a custom tool.

| Param | Type | Required |
|-------|------|----------|
| `name` | string | yes |

```json
{"name": "tool_remove", "arguments": {"name": "count_lines"}}
```
```bash
ash custom-tool remove count_lines
```

---

## QUICK REFERENCE

```
FILE I/O          read_file  grep_files  text_editor
FILESYSTEM        fs_list_dir  fs_stat  fs_write  fs_mkdir  fs_remove  fs_move  fs_copy
SHELL             shell
ASYNC TERMINAL    terminal_run_async  terminal_get_output  terminal_kill  terminal_list  terminal_remove
GIT               git_status  git_diff  git_log  git_add  git_commit
CLIPBOARD         clip  paste  clips  clips_clear
BUFFER            buffer_read  buffer_write  buffer_delete  buffer_replace  buffer_list  buffer_clear  buffer_to_clip  clip_to_buffer
SESSION           session_create  session_info  session_list  session_destroy  backend_switch  backend_status
UTILITIES         find_files  tree  diff_files  patch_apply  http_fetch  file_info  undo
EVENTS            events_subscribe  events_poll  events_push
CUSTOM TOOLS      tool_create  tool_list  tool_view  tool_run  tool_remove
```

Total: 54 tools.

## BEST PRACTICES

1. **View before editing.** Always `read_file` or `text_editor view` before making changes.
2. **str_replace must be unique.** Include enough surrounding context so the match is unambiguous. If it fails, view the file and retry with more context.
3. **Use `shell` for one-off commands, `terminal_run_async` for long-running processes.** Poll async output with `terminal_get_output`.
4. **Use clipboard for context.** Store error messages, code snippets, and intermediate results with `clip`. Retrieve later with `paste`.
5. **Use buffers for composition.** Build up files incrementally with `buffer_write`/`buffer_replace`, then write the result with `fs_write`.
6. **Always destroy sessions when done.** Free container resources with `session_destroy`.
7. **Incremental changes.** Make small edits, test after each change. Don't batch large modifications.
8. **Check git status.** After edits, use `git_diff` to verify changes match intent.
