# SWE-bench Integration for ash-cli

Run ash agent on SWE-bench for evaluation.

## Quick Start

```bash
# Install dependencies
pip install litellm datasets

# Run on a single instance (debug mode)
python -m swebench.runner --instance sympy__sympy-15599 --model anthropic/claude-sonnet-4-5-20250929

# Run batch on SWE-bench Lite
python -m swebench.runner --subset lite --split dev -o results/

# Run on SWE-bench Verified with 4 workers
python -m swebench.runner --subset verified --split test --workers 4 -o results/
```

## Evaluate Results

```bash
# Using sb-cli (recommended, free cloud evaluation)
pip install sb-cli
sb-cli submit swe-bench_verified test --predictions_path results/preds.json --run_id my-run

# Or local evaluation
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path results/preds.json \
    --max_workers 4 \
    --run_id my-run
```

## Architecture

```
swebench/
├── __init__.py      # Core types: AgentConfig, Trajectory, tool definitions
├── agent.py         # AshAgent - main agent loop using litellm + ash tools
├── docker_env.py    # Docker environment for sandboxed execution
└── runner.py        # CLI runner for single/batch execution
```

## How It Works

1. **Agent Loop** (`agent.py`)
   - Uses litellm to support any model (Claude, GPT-4, Gemini, etc.)
   - Provides ash tools via OpenAI-compatible function calling
   - Tracks cost and step limits
   - Saves trajectories in mini-swe-agent compatible format

2. **Docker Environment** (`docker_env.py`)
   - Starts SWE-bench Docker containers for each instance
   - Executes ash tools by translating to shell commands inside container
   - Handles file reading, editing, grep via standard Unix tools

3. **Runner** (`runner.py`)
   - Loads instances from HuggingFace datasets
   - Supports filtering, slicing, resuming
   - Parallel execution with configurable workers
   - Outputs `preds.json` for sb-cli evaluation

## CLI Options

```
Usage: python -m swebench.runner [OPTIONS]

Data Selection:
  --subset SUBSET       SWE-bench subset: lite, verified, full (default: lite)
  --split SPLIT         Dataset split: dev, test (default: dev)
  --instance ID         Run single instance by ID or index
  --slice SPEC          Slice instances (e.g., "0:10")
  --filter REGEX        Filter instance IDs

Model Config:
  --model MODEL         Model name (default: anthropic/claude-sonnet-4-5-20250929)
  --step-limit N        Max agent steps (default: 250)
  --cost-limit N        Max cost in USD (default: 3.0)
  --temperature T       Sampling temperature (default: 0.0)

Execution:
  --output DIR          Output directory (default: swebench_results/)
  --workers N           Parallel workers (default: 1)
  --no-docker           Run locally without Docker
  --ash-binary PATH     Path to ash binary
```

## Tool Mapping

| ash tool | SWE-bench action |
|----------|-----------------|
| `read_file` | `sed -n 'start,end'p file | cat -n` |
| `grep_files` | `rg` or `grep -rn` |
| `text_editor.view` | `sed -n 'start,end'p file` |
| `text_editor.str_replace` | Python string replace |
| `text_editor.insert` | `sed -i 'line a\text'` |
| `text_editor.create` | `cat > file << EOF` |
| `shell` | Direct `docker exec` |
| `git_*` | Native git commands |

## Comparison with mini-swe-agent

| Feature | mini-swe-agent | ash-agent |
|---------|---------------|-----------|
| Core | ~100 lines Python | ~300 lines Python + Rust CLI |
| Tools | bash only | Structured tools (read, grep, edit) |
| Execution | subprocess.run | ash CLI or MCP |
| Format | Linear messages | Tool calls + observations |

ash-agent uses structured tools instead of raw bash, which:
- ✅ Clearer tool boundaries for the model
- ✅ Better error handling
- ✅ Consistent interface across local/docker
- ⚠️ May need prompt tuning for different models
