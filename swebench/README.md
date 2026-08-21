# SWE-bench evaluation harness

Runs an LLM agent against SWE-bench instances in isolated sandboxes — containers,
Firecracker microVMs, or Kubernetes pods, chosen by config rather than by code.

## Quick start

```bash
pip install litellm pyyaml datasets
pip install ./sdk                      # the ash-sandbox client
export ANTHROPIC_API_KEY=sk-...        # or AWS creds for Bedrock, etc.

# One instance
python -m swebench -c swebench/configs/bedrock-sonnet46.yaml -i django__django-11848

# A full run (workers come from the config, or -w)
python -m swebench -c swebench/configs/bedrock-sonnet46.yaml

# The same config on Firecracker microVMs instead of containers
python -m swebench -c swebench/configs/bedrock-sonnet46.yaml --backend microvm

# A different harness (topology)
python -m swebench -c swebench/configs/claude-opus.yaml --harness claude-code
```

`python -m swebench` is the only entry point. Configs are YAML and compose via
`extends:`; CLI flags override file values, which override defaults. See
`__main__.py` for the flag-to-section mapping, and `configs/` for examples.

## Prerequisites

- Python ≥ 3.10, plus the `ash-sandbox` SDK (`pip install ./sdk`)
- A sandbox backend, selected with `--backend` or `execution.backend`:
  - `docker` (default) — Docker running, SWE-bench images pullable
  - `microvm` — an AgentENV server; settings under `execution.microvm`, or
    `AENV_SERVER_URL` / `AENV_API_KEY`
  - `k8s` — the control plane and gateway from `k8s-scaffold/`
- Nothing needs installing inside the image: the `ash-runtime` binary is either
  mounted (`--runtime-bin`) or fetched by `bootstrap.sh` at startup.

## Layout

```
swebench/
├── __main__.py       # CLI entry: config loading, flag merge, dispatch
├── batch.py          # Parallel execution + live dashboard
├── dataset.py        # Instance loading, image resolution, task prompts
├── models.py         # Generic types: ToolResult, CommandOutcome, AgentConfig,
│                     #   CostTracker, Trajectory
├── backends.py       # Sandbox backend from config (docker | microvm | k8s)
├── sandbox.py        # AshSession: sandbox lifecycle + the executor seam
├── prediction.py     # The SWE-bench prediction format (eval layer)
├── submission.py     # Asking the agent to hand in its patch (eval layer)
├── patch.py          # Diffing a worktree, for shared-tree topologies
├── agent/            # The agent loop and the L2 interceptor pipeline:
│                     #   conversation, llm, tools, prompts, hooks,
│                     #   pipeline/interceptors/guardrails
├── harnesses/        # Pluggable topologies; base.py defines the API
├── configs/          # Per-model YAML, composed with `extends:`
├── mcp_server.py     # MCP proxy: the same pipeline for external agents
└── rollout_server.py # RL rollout endpoint (agent + in-sandbox grading)
```

## How a run works

1. **Sandbox** — `AshSession.create(image)` starts one from the configured
   backend and connects to the `ash-runtime` serving tools inside it.
2. **Agent loop** — the model gets a system prompt and the issue, and drives the
   runtime's tools (`shell`, `text_editor`, `process`, `grep_files`, `web_*`; or a
   single `bash` tool with `tools: bash_only`). Every call crosses one seam —
   `executor(tool_name, args) -> ToolResult` — which is where output truncation,
   guardrails, and multi-agent coordination mount as interceptors.
3. **Submission** — the agent is asked for its own diff, with steps reserved so it
   still has turns to answer in (`submission.py`); it knows which files it changed,
   which a harness reading git state can only guess. Shared-worktree topologies
   diff the tree instead, since there no single agent knows the change set.
4. **Cleanup** — `session.destroy()` runs in a `finally`, so the sandbox goes away
   even when the run raises.

Output lands in `results/<run>/`: `preds.json`, plus per-instance trajectories and
traces. Treat it as generated data.

## Harnesses

Registered in `harnesses/__init__.py`:

| Harness       | Topology                                     |
|---------------|----------------------------------------------|
| `litellm`     | one agent, one sandbox — any litellm model    |
| `claude-code` | the Claude Code CLI, driven over MCP         |

Add one by subclassing `BaseHarness` and registering it in `HARNESSES`.

`manager-worker` and `best-of-n` were removed while the single-agent path is being
settled, and Waggle (the write-arbitration seat they used) with them. Mounting one
shared chain across several agents still works and is still tested — a coordination
seat comes back as a plugin, or by reverting.

## Evaluating results

```bash
pip install sb-cli
sb-cli submit swe-bench_verified test \
  --predictions_path results/<run>/preds.json --run_id my-run
```
