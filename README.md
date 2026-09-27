# Running the Qwen Experiments

Use the **`exp/qwen-experiments`** branch. Select one of 12 experiments with `--dataset` and `--method`: three datasets and four methods. Run all commands in Bash from the Ash repository root unless stated otherwise.

## 1. Install the runtime and dependencies

Prerequisites: Python 3.12+, Go 1.25+, a working AgentENV snapshot service, and network access to the task image registries. Deploy AgentENV separately or use an existing service. Clone the complete Ash repository because the experiment runner imports other Ash modules.

```bash
git clone --single-branch --branch exp/qwen-experiments https://github.com/dreamyang-liu/Ash.git
cd Ash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r experiments/qwen/requirements.txt
pip install ./sdk
export PYTHONPATH=.:sdk
export LITELLM_LOCAL_MODEL_COST_MAP=True
(cd runtime && go build -o ash-runtime .)
```

Configure your model endpoint and credentials. Replace the placeholders below with your own settings. Keep credentials outside Git.

```bash
export QWEN_BASE_URL=http://your-private-model-endpoint
read -rs -p 'Qwen API key: ' QWEN_API_KEY; echo
export QWEN_API_KEY
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_KEY_FILE=/path/to/aenv-key
```

## 2. Download the official datasets and select tasks

Download the datasets and their environment images separately. This repository includes experiment code and the HARD-51 selection list. Each task directory must retain its `task.toml`, instruction, environment configuration, and official verifier scripts. The runner requires these task files in addition to any tabular dataset metadata.

| Dataset | `--dataset` | Official sources and preparation requirements |
|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | [Official Scale benchmark page](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2); [official V2 dataset and setup instructions](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2); [official HARD-51 task IDs](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/hard51_ids.txt). Clone the official repository and use its `v2/tasks` directory with the included `experiments/qwen/swebenchpro-v2-hard/selection.json`. |
| TerminalBench 2.1 | `terminalbench21` | [Official dataset repository](https://github.com/harbor-framework/terminal-bench-2-1); [official Harbor Hub dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest). Use the repository's `tasks` directory and an explicit task selection file. |
| DeepSWE | `deepswe` | [Official benchmark website](https://deepswe.datacurve.ai/); [official dataset repository](https://github.com/datacurve-ai/deep-swe). Use the repository's `tasks` directory and an explicit task selection file. |

The Scale benchmark page links to the official GitHub dataset. That repository publishes V2 tasks under `v2/tasks` and defines HARD-51 in `v2/hard51_ids.txt`. The same dataset is available on [Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro), with the `hard` configuration selecting HARD-51. For this runner, obtain the task directories and verifiers from the official GitHub repository. The included 51 task IDs were checked against the official list on September 27, 2026.

For a fresh machine, download the task repositories into a directory you own:

```bash
export DATASETS="$HOME/datasets"
mkdir -p "$DATASETS"
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$DATASETS/SWE-bench_Pro-os"
git clone https://github.com/harbor-framework/terminal-bench-2-1.git "$DATASETS/terminal-bench-2-1"
git clone https://github.com/datacurve-ai/deep-swe.git "$DATASETS/deep-swe"
mkdir -p "$DATASETS/cohorts"
```

Record the dataset commits used for your evaluation and keep task files unchanged across methods. Follow the upstream release instructions for environment images and verification requirements. Revalidate compatibility when changing dataset versions.

A selection file is a JSON array of task directory names. For example, `["largest-eigenval"]` selects one TerminalBench task. Create `terminalbench21.json` and `deepswe.json` in the `cohorts` directory with the tasks you intend to evaluate. Use the same selection for all four methods. The HARD-51 list is already included in Ash.

Set the following variables and Bash arrays once per shell session. Adjust paths for your own machine, and keep outputs outside the Ash checkout:

```bash
export PRO_TASKS="$DATASETS/SWE-bench_Pro-os/v2/tasks"
export TB_TASKS="$DATASETS/terminal-bench-2-1/tasks"
export DEEP_TASKS="$DATASETS/deep-swe/tasks"
export TB_SELECTION="$DATASETS/cohorts/terminalbench21.json"
export DEEP_SELECTION="$DATASETS/cohorts/deepswe.json"
export RUNS="$HOME/ash-runs/qwen"

COMMON=(--runtime-bin "$PWD/runtime/ash-runtime" --api-key-file "$AENV_KEY_FILE" --workers 16)
PRO=(--dataset swebenchpro-v2-hard --tasks-dir "$PRO_TASKS" --selection "$PWD/experiments/qwen/swebenchpro-v2-hard/selection.json")
TB=(--dataset terminalbench21 --tasks-dir "$TB_TASKS" --selection "$TB_SELECTION")
DEEP=(--dataset deepswe --tasks-dir "$DEEP_TASKS" --selection "$DEEP_SELECTION")
```

In a new terminal, reactivate the virtual environment and set these environment variables and Bash arrays again.

## 3. Run one of the 12 experiments

For each dataset, finish **baseline** first, then run SPROUT, BPO, and Shepherd. All three branch methods use `--baseline-root` to share the same initial rollout.

Defaults: mini-swe-agent, `qwen3.8-27b`, `high` reasoning effort, assistant-turn branching, up to 300 model turns per rollout, and a maximum output-token setting of 64000. Baseline always runs four independent rollouts, regardless of success. Branch methods skip tasks whose first rollout succeeds; otherwise, each method allows at most eight attempts including that shared initial rollout.

### SWE-bench-Pro v2 HARD-51: experiments 1–4

```bash
# 1. baseline
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/swebenchpro-v2-hard-baseline"
# 2. SPROUT
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-sprout"
# 3. entropy-based BPO
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-bpo"
# 4. Shepherd
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-shepherd"
```

### TerminalBench 2.1: experiments 5–8

```bash
# 5. baseline
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/terminalbench21-baseline"
# 6. SPROUT
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-sprout"
# 7. entropy-based BPO
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-bpo"
# 8. Shepherd
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-shepherd"
```

### DeepSWE: experiments 9–12

```bash
# 9. baseline
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/deepswe-baseline"
# 10. SPROUT
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-sprout"
# 11. entropy-based BPO
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-bpo"
# 12. Shepherd
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-shepherd"
```

Each command defaults to 16 task workers. Concurrent commands add to the total worker count. Reduce `--workers` in `COMMON` to limit resource usage.

## 4. Optional: run all four methods for one dataset

Choose either the separate runs above or this combined mode. With `all`, each task runs its four baseline rollouts and then the three branch methods in sequence.

```bash
python -m experiments.qwen.run "${PRO[@]}" "${COMMON[@]}" --method all --output "$RUNS/swebenchpro-v2-hard-all"
python -m experiments.qwen.run "${TB[@]}" "${COMMON[@]}" --method all --output "$RUNS/terminalbench21-all"
python -m experiments.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method all --output "$RUNS/deepswe-all"
```

## 5. Preflight checks and small smoke tests

Append `--plan` to a command in Section 3 or 4 to check task files and print the execution plan without model calls or VM creation. Network access, images, and official verifiers require an actual smoke test.

These commands each select one task and exercise all four methods with a small budget:

```bash
python -m experiments.qwen.run --dataset terminalbench21 --method all --tasks-dir "$TB_TASKS" --task largest-eigenval "${COMMON[@]}" --workers 1 --max-turns 2 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/terminalbench21-smoke"
python -m experiments.qwen.run --dataset deepswe --method all --tasks-dir "$DEEP_TASKS" --task superjson-error-stack-serialization "${COMMON[@]}" --workers 1 --max-turns 3 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/deepswe-smoke"
```

The smoke configuration still runs four baseline rollouts; each branch method has a total attempt budget of two. Smoke results validate integration and are not full benchmark scores.

## 6. Export the five-method, four-metric comparison

After the separate runs from Section 3 finish, select a dataset and combine their results:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m experiments.qwen.compare \
  --baseline "$RUNS/$DS-baseline" \
  --sprout "$RUNS/$DS-sprout" \
  --bpo "$RUNS/$DS-bpo" \
  --shepherd "$RUNS/$DS-shepherd" \
  --output "$RUNS/$DS-comparison"
```

This writes `comparison.csv` and `comparison.json`. The five rows are pass@1, pass@4 (baseline), BPO, Shepherd, and SPROUT. The four metrics are Resolve Rate, Recovery, Steps, and Tokens (M).

For the combined `all` mode from Section 4, generate a JSON report containing the five-row `comparison` array:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m experiments.qwen.report "$RUNS/$DS-all" --method all --save "$RUNS/$DS-all/comparison.json"
```

## 7. Resume runs and retain checkpoints

- Repeating the original command with the same configuration and code commit reuses completed stages. Incomplete stages are protected from overwriting: retain their outputs and usage records, then retry in a fresh output directory.
- This entry point requires its own manifests and stage receipts. Outputs from other controllers cannot be passed directly as `--baseline-root`.
- A failed first rollout retains `retained-checkpoints.json`, journals, and exact conversation-prefix references. Recovery also requires the snapshot objects in the AgentENV backend. Keep both the output directory and backend snapshot storage.
- The experiment package does not clean up snapshots. Disk-only snapshots do not preserve RAM, live processes, or tmpfs. Tasks that depend on those require full snapshot support and a separately labelled experiment.
- BPO and Shepherd use inference-time policy adaptations. See the [experiment protocol](experiments/qwen/PROTOCOL.md) for budget, accounting, and recovery details.
