# ash-sandbox

Python SDK for [Ash](https://github.com/dreamyang-liu/Ash) — Agent Sandbox Hive.

## Install

```bash
pip install ash-sandbox
```

## Usage

```python
from ash_sandbox import Sandbox, DockerPool, SandboxPool

# Connect to a running ash-runtime
async with Sandbox.connect("http://localhost:3000") as sb:
    result = await sb.call("shell", command="echo hello")
    print(result.output)

# Spawn Docker containers
async with DockerPool() as pool:
    sb = await pool.spawn(image="python:3.11", entrypoint="pip install pytest")
    result = await sb.call("shell", command="pytest")

# Get tool schemas for LLM function calling
tools = await sb.tool_schemas(format="openai")

# Execute model tool_calls directly
result = await sb.execute_tool_call(tool_call)
```
