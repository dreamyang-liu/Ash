# Paired BPO / Shepherd sampling

This entry point runs a fixed DeepSWE task list through Ash's Claude Code
harness, checkpoint/restore path, and existing DeepSWE verifier. It implements
the **sampling policies**, with frozen API model weights; it does not claim to
reproduce the papers' policy-gradient training or reported training results.

This branch extends the published Shepherd implementation. Its original live
validation is recorded in [VALIDATION.md](VALIDATION.md).

| Method | Selection | Additional rollouts |
| --- | --- | --- |
| `baseline` | Independent fresh starts | Up to 7 |
| `bpo` | Highest first-token entropy scores on the shared failed backbone, with minimum spacing | Up to 7, distributed over selected states |
| `shepherd` | A meta-agent selects one checkpoint on the shared failed backbone | Up to 7 independent siblings at that state |

One shared initial rollout counts toward **every method's total budget of 8**.
If it passes, all methods skip the task. Otherwise each method stops immediately
at its first verified success or budget exhaustion. This differs from historical
server experiments that finished an entire 4/3 batch before stopping. The new
runner records its stopping rule; do not silently pool those protocols.

## Install and run

Use a Linux machine with a reachable AgentENV
[`v0.1.2-ash.1`](https://github.com/dreamyang-liu/AgentENV/releases/tag/v0.1.2-ash.1)
server. This implementation uses Ash's existing microVM snapshot backend; Docker
is not needed and is not a second, untested restore implementation.

From the Ash checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install ./sdk -r configs/branchbench/requirements.txt
(cd runtime && go build -o ash-runtime .)
git clone https://github.com/datacurve-ai/deep-swe ../deep-swe
export PYTHONPATH=.:sdk
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_API_KEY='your-agentenv-key'
export OPENAI_BASE_URL='https://your-provider/v1'
export OPENAI_API_KEY='your-model-key'
umask 077
```

Keys are read only from environment variables. Never put them in configs, Git,
or the SDK owner label. Raw provider audits contain model prompts and responses;
keep run directories private.

First validate the selected task images and verifier using `deepswe.gate` as
described in the repository's `AGENTS.md`. When reusing an existing cohort, retain
its task checkout, passing gate results, and digest-pinned image lock file.

Start a **separate local bridge** in terminal 1:

```bash
python -m deepswe.branching.bridge \
  --model deepseek-v4.1-flash --effort high --port 18187 \
  --audit-root runs/branchbench-deepseek
```

Then in terminal 2 with the same environment:

```bash
python -m deepswe.branching.runner --config configs/branchbench/deepseek.json
```

For Qwen, the corresponding two commands are:

```bash
python -m deepswe.branching.bridge \
  --model qwen3.8-27b --effort high --port 18186 \
  --audit-root runs/branchbench-qwen \
  --extra-json '{"top_k":20,"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}'

python -m deepswe.branching.runner --config configs/branchbench/qwen.json
```

The bridge retains `reasoning_content`, tool call IDs and tool results, and
records the exact provider requests used to align BPO scores with checkpoints.
It queues at most two upstream calls by default. Its `/health` reports the model,
reasoning effort and audit directory; the runner checks these before spending
rollouts. Do not point new validation at an existing experiment's bridge.

`api_timeout_ms` defaults to 1,860,000 (31 minutes) for the Claude Code client,
slightly above the bridge's 30-minute upstream deadline. This is distinct from
the task's overall actor timeout. The runner also configures the separate
`CLAUDE_STREAM_IDLE_TIMEOUT_MS` and byte-level watchdog; raising only
`API_TIMEOUT_MS` is insufficient. The CLI's event watchdog can cancel buffered
reasoning requests even while SSE heartbeat pings arrive. Interrupted
requests are retained with `CancelledError`; unknown usage is never called zero
cost. See the [Claude Code environment reference](https://code.claude.com/docs/en/env-vars).

For another model, copy one config, change `model`, `output` and bridge port,
then start the bridge with the same values. Remove Qwen-only extras unless the
provider supports them. `tasks: ["task-id"]` selects a small fixed subset; omitting
it selects all tasks in the pinned dataset checkout. `max_rollouts: 2` performs
one shared root plus one continuation per method for a small integration test.
The full run uses the task's own timeout; a `timeout` override is recorded and
should be used only for explicitly labeled pilot runs.

## Reuse the server's existing initial rollouts

The original experiments and their journals are read-only inputs. Add these
fields to a config, using your own paths:

```json
{
  "initial_root": "/data/existing-cohort/initial",
  "provider_audit_root": "/data/existing-cohort",
  "initial_owner_template": "experiment:{task}/initial/parent",
  "task_locks": "/data/existing-cohort/task-locks.json"
}
```

`initial_root/<task>/parent.jsonl` must have a completed run and a sibling
`parent.result.json` with its grade. The original Claude Code transcripts and
AgentENV snapshots must still exist on this host. The runner validates the
parent model and freezes hashes of the journal and verdict. The optional image
lock has `{ "task-id": { "pinned_image": "registry/image@sha256:..." } }`.
Historical provider audits may be `.json` or `.json.gz`; the adapter supports
the running experiment's `actor-usage.jsonl` / `provider-responses` layout.

Ash's SPROUT implementation remains in `swebench.fork_eval`. It can consume the
same initial journals with `--benchmark deepswe --tasks-dir ... --parent-from
.../initial --branches 4,3 --branch-count-mode fixed --rounds 2`. Its multi-round
analyst/reviewer and round-level stop protocol are unchanged by this addition.

## Policy fidelity and API limitations

[BPO §4.2](https://arxiv.org/html/2607.14171#S4.SS2) selects top-M decision
boundaries using the first-token distribution, with 64 completion tokens of
minimum spacing. Here checkpoint N is state **after tool turn N**, before the
next model decision. Only exact, completed turn checkpoints are eligible; no
nearest-snapshot substitution or full-conversation fallback is allowed.

OpenAI-compatible APIs normally expose only top-k **content-token** probabilities.
We compute `-sum(p*log(p))` over them and the single aggregated tail bin
`1-sum(p)`. This is explicitly a **lower bound**, not full-vocabulary Shannon
entropy. With hidden reasoning models the first reported content token occurs
after latent reasoning, so its entropy is an observable proxy for the paper's
decision-boundary distribution. Original probabilities are used when available;
otherwise the exact saved request is rescored, changing only the probability
request, streaming flag and output cap. Requests, returned token counts and
scoring usage are saved. A provider that rejects/omits logprobs blocks BPO;
there is no random, self-reported-confidence, or alternate-model fallback.

The validated DeepSeek v4.1 endpoint rejects logprobs, so its example enables
baseline and Shepherd. The Qwen example also enables BPO. If fewer than seven
separated states are eligible, remaining siblings are distributed round-robin
across the selected states, with the actual topology recorded in `plan.json`.
Branches always fork the initial backbone, not later successful/failed children.

[Shepherd Algorithm 2](https://arxiv.org/html/2605.10913#A6.SS6) selects one fork
state with a meta-agent, restores the prefix, and samples K independent suffixes.
We use the same configured model at `high` effort for this selector. It receives
the task, observable trajectory, binary reward, and exact eligible checkpoints.
It does **not** receive hidden tests, oracle patches, or verifier details. This
is the paper's **selection-only** algorithm, not its appendix's additional
replacement-tool-call probe or trajectory-compression hint variant. Its reason
is retained for audit and never injected into the worker. Both methods receive
the same short, fixed neutral continuation prompt required by the CLI resume
interface; this adaptation is recorded as `hint_delivery: fixed-neutral`.

The native transcript is copied only through the chosen tool-result cut and
registered as an independent Claude session. Every sibling gets its own sandbox
and working directory, with local Claude settings disabled. Disk snapshots do
not preserve background processes or RAM; this inherits Ash's foreground-tool
execution contract. Snapshot/session inconsistencies stop that method.

## Records and recovery

`benchmark-manifest.json` freezes settings, task list, dataset file hashes, image
references and the Ash commit. `parents/` freezes shared roots. Each method/task
contains its plan, scoring or selector audit, branch journals, exact transcript
prefix receipts and verifier outputs. `task-summary/` and `summary.json` distinguish
success, exhausted budgets and blocked infrastructure. A blocked task is never
counted as an ordinary failure. The program exits nonzero if any method blocks.

Rerun the same command to reuse plans, probability scores and completed outcomes.
Finished but ungraded journals can be graded without another actor call. An
interrupted actor journal requires explicit recovery; the runner never spends
an unrecorded replacement rollout. A file lock prevents two controllers from
running in the same output directory. Use a new directory when changing settings
or code revision. The runner never deletes snapshots or existing experiments.

Raw API audits separately preserve actor usage, BPO scoring usage and Shepherd
selector usage, including reasoning tokens where the endpoint reports them.
Count these overheads when comparing compute; 8 leaves alone is not equal token
cost. No assumed model prices are used to invent dollar totals. Aggregate with
`python -m deepswe.branching.report runs/branchbench-qwen`; this writes
`metrics.json`. Success rates remain null until all requested tasks for that
method are complete, so infrastructure blocks cannot inflate reported accuracy.

```bash
PYTHONPATH=.:sdk python -m pytest deepswe/tests/test_bpo_sampling.py deepswe/tests/test_shepherd_sampling.py -q
PYTHONPATH=.:sdk python -m pytest harness/tests swebench/tests sdk/tests deepswe/tests -m 'not slow' -q
```
