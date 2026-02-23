# SWE-bench Benchmark for ash-cli

Run an LLM agent with a bash tool on SWE-bench instances in isolated Docker sandboxes.

## Quick Start

```bash
# Install dependencies
pip install litellm datasets

# Set API key (Claude, GPT-4, Gemini, etc. via litellm)
export ANTHROPIC_API_KEY=sk-...

# Run on a single instance
python -m swebench.runner -i sympy__sympy-15599

# Run batch on SWE-bench Lite
python -m swebench.runner --subset lite --split test -o results/

# Run on SWE-bench Verified with 4 workers
python -m swebench.runner --subset verified --split test -w 4 -o results/
```

## Prerequisites

- `ash` binary on PATH (or use `--ash-binary ./target/release/ash`)
- Docker running (the runner creates Docker containers per instance)
- The swebench Docker images pulled or pullable

## Evaluate Results

```bash
# Using sb-cli (free cloud evaluation)
pip install sb-cli
sb-cli submit swe-bench_verified test --predictions_path results/preds.json --run_id my-run
```

## Architecture

```
swebench/
├── AGENT.md       # System prompt / manual given to the LLM
├── types.py       # Core types: AgentConfig, Trajectory, ToolResult, CostTracker
├── ash_cli.py     # AshSession: session lifecycle + bash execution via ash CLI
├── tools.py       # Single "bash" tool schema (OpenAI function calling format)
├── agent.py       # AshAgent: litellm agent loop
└── runner.py      # CLI runner for single/batch execution
```

## How It Works

1. **Session Setup** — `ash session create --image <swebench-image>` creates a Docker sandbox with ash-mcp running inside
2. **Agent Loop** — The LLM gets AGENT.md as system prompt + the issue. It has one tool: `bash`. Commands run via `ash --session <id> run "<command>"`
3. **Patch Extraction** — After the agent finishes, `ash --session <id> git-diff` captures the changes
4. **Cleanup** — `ash session destroy <id>` removes the container

## CLI Options

```
python -m swebench.runner [OPTIONS]

Dataset:
  --subset SUBSET       lite, verified, or full (default: lite)
  --split SPLIT         Dataset split (default: test)
  --instance ID         Single instance by ID or index
  --slice SPEC          Slice (e.g., "0:10")
  --filter REGEX        Filter instance IDs

Model:
  --model MODEL         litellm model name (default: anthropic/claude-sonnet-4-5-20250929)
  --step-limit N        Max agent steps (default: 250)
  --cost-limit N        Max cost in USD (default: 3.0)
  --temperature T       Sampling temperature (default: 0.0)

Ash:
  --ash-binary PATH     Path to ash binary (default: ash)

Execution:
  --output DIR          Output directory (default: swebench_results/)
  --workers N           Parallel workers (default: 1)
```

## Tool Flow

```
LLM → tool_call(bash, {command: "ash grep 'def solve' src/"})
  → AshAgent._execute_tool_calls()
    → AshSession.run_command("ash grep 'def solve' src/")
      → subprocess: ash --session <id> run "ash grep 'def solve' src/"
        → ash CLI → Gateway → ash-mcp inside Docker container
      ← stdout/stderr as ToolResult
    ← observation message
  ← append to messages, next LLM call
```
