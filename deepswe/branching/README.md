# Shepherd sampling benchmark

Run a fixed DeepSWE task set through Ash's mini-swe-agent harness, exact checkpoint
restore, and existing verifier. This implements the **inference-time sampling
policy** from [Shepherd Algorithm 2](https://arxiv.org/html/2605.10913#A6.SS6),
with frozen API model weights, not the paper's reinforcement-learning training.

One shared initial rollout counts toward each method's **total budget of 8**.
If it passes, skip further sampling. Otherwise:

- `baseline`: up to seven independent fresh rollouts.
- `shepherd`: the meta-agent selects one exact checkpoint on the failed initial
  trajectory; sample up to seven independent siblings from that same state.

Both methods stop immediately at the first verified success. This differs from
historical experiments that finished an entire 4/3 batch before stopping; do not
silently pool their measurements. See [validation results](VALIDATION.md).

## Setup

Use Linux and Python 3.12. Shepherd branches use a project binding archive, not
an AgentENV or whole-sandbox snapshot. A benchmark adapter may use any fresh
sandbox provider for execution, but every sibling starts from the original task
image and restores only the declared project workdir.

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

Keep keys in environment variables, never in configs or Git. Raw provider audits
contain prompts and responses; keep run directories private. Validate the chosen
task images/verifier with `deepswe.gate` as described in the repository's
`AGENTS.md`. Reused cohorts should retain their passing gates, dataset checkout
and digest-pinned image locks.

## Run

In terminal 1, start a dedicated bridge:

```bash
python -m deepswe.branching.bridge \
  --model deepseek-v4.1-flash --effort high --port 18187 \
  --audit-root runs/branchbench-deepseek
```

In terminal 2 with the same environment:

```bash
python -m deepswe.branching.runner --config configs/branchbench/deepseek.json
python -m deepswe.branching.report runs/branchbench-deepseek
```

The example enables both methods. Set `"methods": ["shepherd"]` to run Shepherd
only. Set `"tasks": ["task-id"]` and `"max_rollouts": 2` for a pilot comprising
one shared root plus one continuation. Omitting `tasks` selects the full pinned
dataset checkout. The default actor deadline is the task's own timeout; any
`timeout` override is recorded and should be labeled as a pilot setting.

For another OpenAI-compatible API, copy the JSON and change `model`, `output`
and `bridge_url`; start a bridge with the same model, effort, output directory
and port. This release's real-model validation used DeepSeek. Provider-specific
worker options can be supplied with bridge `--extra-json`; selector options use
config `meta_extra`. Do not override protected model/messages/effort fields.

The bridge preserves reasoning content, tool call IDs and tool results. It uses
at most two concurrent upstream calls by default. `/health` exposes the model,
effort and audit directory, which the runner checks before sampling. Do not
point validation at an existing experiment's bridge.

The bridge's upstream timeout defaults to 30 minutes. mini-swe-agent uses
OpenAI-compatible non-streaming Chat Completions and remains bounded by the task's
overall actor deadline. Interrupted requests retain cancellation audits and unknown
usage is not represented as zero cost.

## Reuse existing initial rollouts

Add the following to the configuration, using your own paths:

```json
{
  "initial_root": "/data/existing-cohort/initial",
  "task_locks": "/data/existing-cohort/task-locks.json"
}
```

`initial_root/<task>/parent.jsonl` needs a completed run and a sibling
`parent.result.json` with its grade. Original mini native transcripts and
project-binding snapshots must still exist on this host. The runner checks the parent model and
freezes journal/verdict hashes. The optional image lock format is
`{"task-id": {"pinned_image": "registry/image@sha256:..."}}`.

Ash's SPROUT remains unchanged in `swebench.fork_eval`. It can use the same
initial journals with `--benchmark deepswe --tasks-dir ... --parent-from
.../initial --branches 4,3 --branch-count-mode fixed --rounds 2`; its multi-round
analyst/reviewer and round-level stopping protocol are separate from this runner.

## Selection and restoration contract

The same configured model at `high` effort receives the task, observable
trajectory, binary reward, and eligible checkpoints. It does not receive hidden
tests, oracle patches or verifier details. It selects one checkpoint and a short
reason. This is **selection-only Algorithm 2**, not the appendix's additional
replacement-tool-call probe or trajectory-compression hint variant.

Checkpoint N means state **after completed tool step N**, before the next model
decision. Out-of-range or incomplete decisions fail closed. The selector's reason
is saved for audit but never given to the worker. Siblings resume the exact mini
history directly, without a user hint; this is recorded as
`hint_delivery: point-only`.

Each branch restores the exact project binding and copies the native conversation
only through that checkpoint's tool-result cut. Each sibling has its own agent
session, working directory and fresh sandbox, with user/project settings disabled.
There is no nearest-snapshot substitution or full-future-history fallback. Files
outside the project binding, RAM, background processes and external side effects
are deliberately not inherited. Required services must be rebuilt by the branch.

`project_scope.ProjectBindingSnapshot` is the durable fork contract. Archives are
content-addressed and verified before every restore; replacement is refused for
system roots. Whole-sandbox snapshots are not accepted as branch images.
The boundary follows the project-workdir MetaGit scopes in Shepherd Experiments
commit `c12ebd1b774cf12f70ef2b4486e61e7052f3e3ab`; the tar backend is Ash's
portable binding transport for Terminal-Bench images that do not ship MetaGit.

## Records, recovery and accounting

`benchmark-manifest.json` freezes configuration, task list, dataset/image references,
Git revision, source fingerprint and runtime hash. `parents/` freezes initial
journals. Each method/task stores its selection, selector request, branch journals,
native-prefix references, results and verifier outputs. Infrastructure failures are marked
`blocked`, not ordinary model failures; the runner exits nonzero for blocked tasks.

Rerunning the same configuration/code reuses plans and completed outcomes. Finished
but ungraded journals can be graded without another actor call. Interrupted actors
require explicit recovery; the runner never silently spends a replacement rollout.
A file lock prevents two controllers sharing an output directory. Use a new output
directory when changing settings or code revision. Existing snapshots/experiments
are never deleted.

The report separately counts actor and selector usage, including cached input and
reasoning tokens when provided. Imported root costs remain in the source cohort.
Eight leaves alone do not imply equal compute. No token prices are assumed, and
success rates remain null while any requested task is pending/blocked.

```bash
PYTHONPATH=.:sdk python -m pytest deepswe/tests/test_shepherd_sampling.py -q
PYTHONPATH=.:sdk python -m pytest harness/tests swebench/tests sdk/tests deepswe/tests -m 'not slow' -q
```
