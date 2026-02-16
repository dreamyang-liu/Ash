# Ash Tools Reference

Complete reference for all 54 tools available via `ash` CLI and MCP server.

---

## Session Management

### `session_create`
Create a new sandbox container.

```bash
ash session create
ash session create --image python:3.11
ash session create --image ubuntu:22.04 --port 3000 --port 8080
ash session create --env KEY=VALUE --cpu-limit 2 --memory-limit 4Gi
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--image` | string | default | Container image |
| `--name` | string | auto | Session name |
| `--port` | int[] | none | Ports to expose |
| `--env` | KEY=VALUE[] | none | Environment variables |
| `--cpu-request` | string | none | CPU request (e.g., "100m") |
| `--cpu-limit` | string | none | CPU limit (e.g., "2") |
| `--memory-request` | string | none | Memory request (e.g., "256Mi") |
| `--memory-limit` | string | none | Memory limit (e.g., "4Gi") |
| `--node-selector` | KEY=VALUE[] | none | K8s node selector |

Returns: `session_id`, `status`, `host`, `backend`

### `session_list`
```bash
ash session list
```

### `session_info`
```bash
ash session info <session_id>
```

### `session_destroy`
```bash
ash session destroy <session_id>
```

### `backend_status`
```bash
ash config
```

### `backend_switch`
```bash
ash session switch <session_id> docker
ash session switch <session_id> k8s
```

---

## Shell Execution

### `shell` (sync)
```bash
ash run "<command>"
ash --session <id> run "<command>"
ash --session <id> run "<command>" --timeout 600
ash --session <id> run "<command>" --tail 50    # last 50 lines only
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | string | required | Command to execute |
| `--timeout` | int | 300 | Timeout in seconds |
| `--tail` | int | none | Only show last N lines |

### `terminal_run_async`
Start a background process, returns a handle ID.

```bash
ash terminal start "<command>"
ash --session <id> terminal start "<command>"
ash terminal start "<command>" --workdir /path --env KEY=VALUE
```

### `terminal_get_output`
```bash
ash terminal output <handle>
ash terminal output <handle> --tail 100
```

### `terminal_kill`
```bash
ash terminal kill <handle>
```

### `terminal_list`
```bash
ash terminal list
```

### `terminal_remove`
```bash
ash terminal remove <handle>
```

---

## File Reading

### `read_file`
Read file with line numbers.

```bash
ash view <file_path>
ash view <file_path> -n 50 -l 20      # start at line 50, show 20 lines
ash --session <id> view /testbed/src/main.py
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | required | Path to file |
| `-n, --offset` | int | 1 | Start line |
| `-l, --limit` | int | 100 | Number of lines |

Output format:
```
     1 | def main():
     2 |     print("Hello")
```

### `grep_files`
Search for patterns (ripgrep).

```bash
ash grep "<pattern>" <path>
ash grep "def.*test" src/ --include "*.py" --limit 50
ash --session <id> grep "TODO" /testbed/
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | string | required | Regex pattern |
| `path` | string | "." | Search path |
| `--include` | string | none | File glob filter |
| `--limit` | int | 100 | Max results |

### `find_files`
Find files by name pattern (glob).

```bash
ash find "*.py" src/
ash find "test_*" . --max-depth 3 --limit 50
```

### `tree`
Show directory tree.

```bash
ash tree <path> --max-depth 3
ash tree . --show-hidden
```

### `file_info`
Get file type, encoding, size.

```bash
ash file-info <path>
```

---

## File Editing

### `text_editor`
Four subcommands: `view`, `replace`, `insert`, `create`.

**view** - View with line range:
```bash
ash edit view <path> --start 10 --end 30
```

**replace** (str_replace) - Must match exactly once:
```bash
ash edit replace <path> --old "old text" --new "new text"
```

**insert** - Insert after line:
```bash
ash edit insert <path> --line 10 --text "new line"
```

**create** - Create new file:
```bash
ash edit create <path> "file content here"
```

### `undo`
Undo last file edit.

```bash
ash undo                  # undo last edit
ash undo <path>           # undo specific file
ash undo --list           # show undo history
```

### `diff_files`
Compare two files (unified diff).

```bash
ash diff <file1> <file2>
ash diff <file1> <file2> --context 5
```

### `patch_apply`
Apply unified diff patch.

```bash
ash patch "<patch_content>" --path <base_path>
ash patch "<patch_content>" --dry-run
```

---

## Filesystem Operations

```bash
ash fs ls <path>                           # list directory
ash fs stat <path>                         # file/dir metadata
ash fs write <path> "<content>"            # write file
ash fs mkdir <path>                        # create directory (recursive)
ash fs rm <path>                           # remove file
ash fs rm <path> --recursive               # remove directory
ash fs mv <from> <to>                      # move/rename
ash fs cp <from> <to>                      # copy file
ash fs cp <from> <to> --recursive          # copy directory
```

---

## Git Operations

```bash
ash git-status                     # full status
ash git-status --short             # short format
ash git-diff                       # unstaged changes
ash git-diff --staged              # staged changes
ash git-diff <path1> <path2>       # specific files
ash git-log -n 10                  # last 10 commits
ash git-log --oneline              # one-line format
ash git-add <path1> <path2>        # stage files
ash git-add --all                  # stage all
ash git-commit -m "message"        # commit
ash git-commit -m "message" --all  # stage all + commit
```

All git commands work with `--session <id>` for sandbox repos.

---

## Clipboard (Agent Memory)

Named storage for text snippets across operations.

```bash
ash clip "content" --name memo         # save text
ash clip --file src/lib.rs:10-20 --name snippet  # save file range
ash paste                              # get latest
ash paste <name>                       # get by name
ash clips                             # list all
ash clips-clear                       # clear all
ash clips-clear <name>                # clear specific
```

---

## Buffer (Agent Workspace)

Named editable text buffers - a persistent scratchpad.

```bash
ash buffer write "content" --name main         # write/create
ash buffer write --append "more" --name main   # append
ash buffer write --at-line 5 "inserted"        # insert at line
ash buffer read --name main                    # read all
ash buffer read --name main --start 10 --end 50  # read range
ash buffer replace --name main --start 5 --end 10 "new content"
ash buffer delete --name main --start 5 --end 10
ash buffer list                                # list all buffers
ash buffer clear --name main                   # delete buffer
ash buffer clear                               # delete all

# Buffer <-> Clipboard transfer
ash buffer to-clip <clip_name> --start 10 --end 20
ash buffer from-clip <clip_name> --append
```

---

## Events

Subscribe to and poll events (process completions, file changes, etc.).

```bash
ash events subscribe process_complete file_change
ash events subscribe process_complete --unsubscribe
ash events poll --limit 10
ash events poll --peek           # peek without consuming
ash events push custom --source agent --data '{"key":"value"}'
```

---

## HTTP

```bash
ash fetch <url>
ash fetch <url> --timeout 60
```

---

## Custom Tools

Register reusable scripts as tools.

```bash
ash custom-tool create my-lint --script "flake8 ." --lang sh
ash custom-tool list
ash custom-tool view my-lint
ash custom-tool run my-lint
ash custom-tool remove my-lint
```

---

## Gateway

Routing layer that forwards tool calls to `ash-mcp` processes. The gateway manages session routing, local ash-mcp subprocess lifecycle, and ensures all tool execution happens uniformly through ash-mcp (local, Docker, or K8s).

The gateway auto-starts on the first CLI call — no manual setup needed.

```bash
ash gateway start                # detach to background
ash gateway start --foreground   # run in foreground
ash gateway status               # check if running
ash gateway stop                 # stop gateway + local ash-mcp
```

Architecture:
- **CLI → Gateway**: Unix socket (`~/.ash/gateway.sock`), JSON-RPC
- **Gateway → ash-mcp**: HTTP, JSON-RPC (local subprocess on auto-assigned port, or Docker/K8s endpoints)
- **Session tools** (`session_create`, `session_destroy`, etc.) execute inside the gateway (they manage infrastructure)
- **All other tools** are forwarded to the appropriate ash-mcp endpoint
- If the gateway is unavailable, the CLI falls back to direct local execution

---

## System Info

```bash
ash info          # show backends, sessions, processes, tools count
ash tools         # list all available tools
```
