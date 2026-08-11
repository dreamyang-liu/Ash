# ash-sandbox

Python SDK for [Ash](https://github.com/dreamyang-liu/Ash) — Agent Sandbox Hive.

It gives a harness two things: a way to obtain sandboxes, and a way to talk to
the tools inside one. Policy — what the agent loop does, which tools an agent is
offered, who gets to be an agent — stays in the harness.

## Install

```bash
pip install ash-sandbox
```

## Connecting

```python
from ash_sandbox import Sandbox

async with Sandbox.connect("http://localhost:3000") as sb:
    result = await sb.call("shell", command="echo hello")
    print(result.output)          # hello
```

`Sandbox.connect(url)` speaks HTTP; `Sandbox.mcp(url)` speaks MCP;
`Sandbox.local(bin)` runs the runtime binary directly with no server.

All three behave alike, including events and background processes: each keeps
one runtime process for the life of the handle. A sandbox's state -- its event
log, its running jobs, its artifact cache -- lives inside that process, so
`close()` ends it. Call it, or use the handle as a context manager.

## Identity

Pass `agent_id` whenever more than one agent shares a sandbox. The runtime
keeps a per-identity cursor over its event log, so two anonymous clients share
one cursor and each sees only part of what happened.

```python
sb = Sandbox.connect(url, agent_id="reviewer")     # bound to the handle
await sb.call("shell", command="ls", agent_id="x") # or per call, which wins
```

Identity travels beside the arguments, not inside them, and the runtime
overwrites whatever a caller supplied. A model therefore cannot name itself —
if it could, it could read events addressed to another agent.

Bind it at construction where you can. An agent handed a pre-named channel
cannot act as anyone else, and no call site can forget it:

```python
async with DockerPool() as pool:
    worker = await pool.spawn(image="python:3.12", agent_id="worker-1")
```

## The eight builtin tools

`shell`, `text_editor`, `grep_files`, `process`, `web_search`, `web_fetch`,
`artifact`, `wait_for_events`.

```python
# stdin and environment, without shell quoting
await sb.call("shell", command="cat; echo $GREETING",
              stdin="piped\n", env={"GREETING": "hi"})

# bound the output so a runaway command cannot OOM the sandbox.
# truncate_mode "H<n>T<n>" weights head vs tail; the floor is 1 KiB per stream.
await sb.call("shell", command="seq 1 100000",
              max_output_bytes=64_000, truncate_mode="H1T3")
```

Truncation is always announced in the output, never silent.

## Events

Anything asynchronous — a background process exiting, another agent's tool call
— arrives as an event. Delivery is **opt-in**: subscribe to a kind, or wait for
it explicitly. Nothing is retained for an identity that asked for nothing.

```python
# Wait for one specific background process, not "any exit"
result = await sb.call("shell", command="make -j8", background=True)
pid = json.loads(result.output)["pid"]

batch = await sb.wait_for_events(kinds=["process_exited"], sources=[pid], timeout=60)
for event in batch:
    print(event.kind, event.data["exit_code"])

if batch.missed:
    print(f"{batch.missed} event(s) expired before we read them")
```

`poll_events(...)` is the same thing without blocking. Events are delivered
once per identity, so a second call will not repeat what this handle already
received. Long-polling is deliberate: the sandbox never opens connections
outward, so every transport behaves identically.

Observing another agent needs both a subscription and a distinct identity:

```python
observer = Sandbox.connect(url, agent_id="observer")
await observer.call("wait_for_events", action="subscribe", kinds=["tool:text_editor"])
batch = await observer.wait_for_events(kinds=["tool:text_editor"], timeout=30)
for event in batch:
    print(f"{event.origin} edited {event.source}")
```

See `examples/multi_agent_shared_sandbox.py` for three named agents cooperating
in one sandbox, coordinating through the event log rather than through the host
process.

## Pools

```python
from ash_sandbox import DockerPool, MicroVMPool

async with DockerPool() as pool:                 # local containers
    sb = await pool.spawn(image="python:3.12", entrypoint="pip install pytest")
```

`Pool` is the interface: `spawn` / `destroy` / `destroy_all` / `list` / `close`.
Some sources can do more, and say so rather than making callers check types:

```python
if pool.supports_fork():
    branches = await pool.fork(sb, count=4)   # each continues from sb's state
```

`MicroVMPool` (Firecracker via [AgentENV](https://github.com/kvcache-ai/AgentEnv))
supports pause/resume/fork; `DockerPool` declares `False` and refuses them.
`SandboxPool` provisions through a Kubernetes control plane.

## Tools for an LLM

```python
tools = await sb.tool_schemas(format="openai")     # or "anthropic", "raw"
result = await sb.execute_tool_call(tool_call)     # OpenAI or Anthropic shape
```

## Custom tools

A tool can be declared as data rather than code: a binary (URL + optional
sha256, or a path already in the image) plus how its arguments become argv.

```yaml
name: ruff
description: Lint Python files
binary:
  url: https://example.com/ruff
  sha256: <64 hex chars>
parameters:
  path:
    type: string
    required: true
    map: {positional: 0}
  fix:
    type: boolean
    map: {flag: "--fix"}
```

```python
from ash_sandbox import ToolRegistry, parse_manifest

registry = ToolRegistry()
registry.register(parse_manifest(yaml.safe_load(text)))

sb = Sandbox.connect(url, tools=registry)
await sb.prepare_tools()                       # download + verify up front
await sb.call_agent_tool("ruff", {"path": "src/", "fix": True})

tools = await sb.tool_schemas(format="openai")  # builtin + custom, one panel
```

Arguments are compiled into argv slots, never interpolated into a shell string,
so an argument cannot become a command. Binaries are content-addressed and
cached per sandbox: the download happens once, the sha256 is verified, and a
stale cache is re-resolved rather than reported as "not found".
