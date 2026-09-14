# Terminal-Bench 4.0 evaluation

This entry point pins `terminal-bench/terminal-bench@4.0.0` and Harbor 0.22.0.
Harbor resolves the dataset, creates each task's environment, runs the agent,
collects task artifacts, runs the task's verifier, saves results and tears down
the environment. Task CPU/GPU/memory/network policies and agent/verifier timeouts
are not overridden. Version 4.0 has 66 tasks, including GPU and multi-container
tasks; its configured agent timeout is eight hours.

Sources: [official release](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0),
[dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4).

## Agent and tools

The default is now **AgentENV**. Harbor owns task loading, agent/verifier phases,
artifact collection and grading. `terminalbench.agentenv:AgentENVEnvironment`
implements its environment operations using Ash's microVM session: create,
execute, file/directory transfer and teardown. OCI images are digest-pinned;
CPU, memory and disk resources reach the cold start and template identity.
Commands preserve the image/task environment, working directory and execution
user. The agent's VM stays alive for collection; a separate verifier gets a
fresh AgentENV VM and the task-defined artifacts.

`terminalbench.agentenv_agent:AgentENVClaudeCode` runs the existing Ash Claude
Code slot on the host, connected to that VM through MCP. The only sandbox tool
is `shell`; built-in host tools and `text_editor` are denied. The existing
mutation tracker and exact tool-call identity checkpoint bridge are mounted on
the live tool path, so journals pair each executed step with its snapshot.
The default `--checkpoint-mode full` preserves guest processes as well as disk.
`disk_only` is cheaper but cannot restore background processes. Journals,
native session references and a final actor snapshot are retained for replay.
The TB4 CLI does not yet schedule analyst/reviewer branches; `--attempts` still
means independent trials, and `--resume` resumes a Harbor job, not a tool step.

Explicit `--env docker`, `--env modal`, etc. retain the earlier Harbor-native
container-side Claude Code driver (`terminalbench.agent:AshClaudeCode`). That
route retains task MCP services but has no AgentENV checkpoints. There is no
automatic fallback between the two routes.

### AgentENV admission limits

Before an AgentENV run, every selected task's agent and verifier definitions
are checked and written to `agentenv-admission.json`. If any are unsupported,
the whole job stops before starting an actor; tasks are never silently omitted.
Current unsupported requirements include GPU/TPU, Windows, Compose sidecars,
task-provided MCP services, network allowlists or in-place network-policy
changes, arbitrary host bind mounts and multi-step tasks. CPU and memory limits
are supported; resource reservation/guarantee policies are not claimed.

The current AgentENV OCI layer implementation has a **64 GiB minimum virtual
disk** and cannot shrink the base image. Smaller explicit task disk budgets are
rejected rather than silently enlarged. Disk sizes must also be multiples of
1024 MiB. Thus this backend cannot currently execute the full unmodified TB4
cohort. Admission is a compatibility check, not proof that an image will build
or a task's verifier is correct; real oracle/nop gates remain necessary.

Prebuilt images must be reachable from AgentENV. Tasks or separate verifiers
with only a Dockerfile require `--image-registry HOST:PORT`: the wrapper builds
the image with Docker and pushes it to that registry. Configure registry access
for both local regctl and the AgentENV service. No registry credentials are
written to result manifests.

## Install and plan

From the Ash repository, use Python 3.12 or newer:

```bash
python3.12 -m venv .venv-tb4
.venv-tb4/bin/pip install -r terminalbench/requirements.txt
cd runtime && go build -o /tmp/ash-runtime . && cd ..
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench \
  --model anthropic/claude-sonnet-4-6 --runtime-bin /tmp/ash-runtime --workers 32
```

The default phase is `plan`: validate and print the exact Harbor config without
downloading tasks, creating environments or calling a model. Set `AENV_SERVER_URL`
and `AENV_API_KEY` (or `--server-url` and `--api-key-file`), plus credentials for
the selected model provider. The default lease is ten hours, covering the
eight-hour actor budget plus setup/collection. Local Docker execution requires
Docker Compose v2. For a remote provider, install its Harbor extras, e.g.
`pip install 'harbor[modal]==0.22.0'`, and configure the provider credentials.
The selected provider must support the tasks' resources and services; no tasks
are silently dropped to fit the local machine.

## Run

Audit the selected official tasks without starting VMs or model inference:

```bash
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench --phase audit \
  --model anthropic/claude-sonnet-4-6 --runtime-bin /tmp/ash-runtime \
  --image-registry localhost:5000 --output runs/tb4-admission
```

Use `--phase run` with the same options and `--resume` after successful admission,
or choose a new output directory. A failing full-cohort audit does not authorize
changing task resources: explicitly choose a supported subset for development,
or use another supported backend. The summary labels the actual selected cohort.

Use `--task NAME` repeatedly to select a subset, `--env docker` for local
execution, or `--tasks-dir PATH` for an explicitly local Harbor dataset.
Subset/local runs are identified in the summary and are not full TB4 scores.
`--resume` requires the original output directory and identical job options;
Harbor reconciles existing trial results. Automatic retries are disabled to
avoid silently spending another model attempt. Resume does not imply retrying
every already-recorded failed trial; use a separate named run for deliberate
retries. Existing results are not overwritten by an implicit fresh run.

## Results

```text
runs/tb4-parent/
  ash-eval.json                 dataset, settings, dependency version, trial count
  config.json                   Harbor config
  result.json                   official aggregate
  ash-summary.json              measured rewards, errors, unfinished coverage
  agentenv-admission.json        per-task capability checks (AgentENV route)
  <trial>/result.json            official per-trial reward and exception details
  <trial>/*.agentenv.json        VM ID, image digest, resources and network policy
  <trial>/agent/trajectory.jsonl Ash journal with exact checkpoint pairs
  <trial>/agent/trajectory.json  ATIF trajectory
  <trial>/agent/snapshot.json    final actor snapshot and session reference
  <trial>/agent/execution.json   actor outcome
  <trial>/verifier/             verifier output and reward files
```

On the explicit Harbor-native route, agent logs instead include `ash-agent.json`,
`claude-code.txt`, `sessions/` and the official ATIF trajectory. AgentENV's native
Claude session is stored by the host SDK under its usual project/session path.

Summarizing needs no running environment:

```bash
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench \
  --phase summarize --output runs/tb4-parent
```

An official reward of zero is a completed negative result. Missing/malformed
rewards and exceptions are separate errors. Missing/unfinished trial reports
are incomplete. `final_mean_reward` stays null until all planned trials have
valid rewards; `mean_reward_lower_bound` uses the full planned denominator.
Completed-only means are diagnostic, not replacements for the benchmark score.
The driver returns 2 for incomplete/error coverage and 0 for fully evaluated
coverage, even if all valid rewards are zero.

## Validate without model calls

```bash
PYTHONPATH=.:sdk .venv-tb4/bin/python -m pytest terminalbench/tests -q
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench --phase run \
  --agent nop --tasks-dir terminalbench/tests/agentenv_fixtures --runtime-bin /tmp/ash-runtime \
  --workers 1 --output runs/tb4-smoke-nop
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench --phase run \
  --agent oracle --tasks-dir terminalbench/tests/agentenv_fixtures --runtime-bin /tmp/ash-runtime \
  --workers 1 --output runs/tb4-smoke-oracle
PYTHONPATH=.:sdk .venv-tb4/bin/python -m terminalbench.tests.live_gate \
  --runtime-bin /tmp/ash-runtime --output runs/tb4-checkpoint-gate
```

The scripted gate makes two real MCP calls without model inference, runs the
official Harbor verification lifecycle, restores the first full snapshot and
checks that both its earlier file contents and a background process survived.
For separate verifier coverage, pass `--tasks-dir
terminalbench/tests/agentenv_separate_fixtures --image-registry localhost:5000`
to the same gate. Its verifier asserts that the actor's `/app/answer.txt` is
absent and grades only the collected artifact in the fresh VM. Run that fixture
with `--agent nop` as well; expected rewards are scripted/oracle=1 and nop=0.
These are synthetic transport gates, not full TB4 or live model validation.
Install pytest separately for unit tests. Snapshots remain available; gates do
not garbage-collect snapshot history.
