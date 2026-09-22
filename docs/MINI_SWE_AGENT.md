# mini-swe-agent

`--slot mini-swe-agent` drives the upstream `DefaultAgent` from mini-swe-agent
2.4.6. Its model sees the upstream `bash` tool. Commands execute through Ash's
HTTP MCP shell; Ash owns the VM, command timeout, snapshots, grading and rollout
budgets. Each command preserves its real exit code and stdout/stderr.

## Install and run

Install into the interpreter used by the corresponding Run Store profile:

```bash
python -m pip install -e ./sdk -r harness/requirements-mini-swe-agent.txt
PYTHONPATH=.:sdk python contracts/ci_check.py --slot mini-swe-agent
```

The model endpoint must implement nonstreaming Chat Completions with function
tools. Ash preserves the served model name, `temperature`, `top_p`, `top_k`,
output limit and text stop sequences. The gateway supplies provider credentials;
they are not stored in the native history.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export OPENAI_API_KEY=...

python -m harness run --slot mini-swe-agent --model served-policy \
  --backend microvm --runtime-bin runtime/ash-runtime \
  --transport http --tools shell_only --sandbox-image task-template \
  --journal runs/mini/trajectory.jsonl \
  --extra '{"mini":{"environment":{"cwd":"/repo","timeout":60}}}' \
  'Fix the issue in this repository.'
```

Alternatively use the existing `--gateway --routes ...` configuration. The
orchestrator points mini at that gateway automatically. The `mini.environment`
working directory is inside the sandbox; harness `cwd` remains a host workspace
for bookkeeping.

The default prompt is upstream `config/mini.yaml`. `extra.mini.agent` can
override its system/instance templates. `extra.mini.environment` accepts `cwd`,
`timeout` and string environment variables; for images requiring conda setup,
configure `BASH_ENV` explicitly. `extra.mini.model` accepts upstream observation
and format-error templates, plus supported `model_kwargs`. Ash owns step/time/
cost budgets and output files, so conflicting upstream limits are rejected.

## Run Store and RL

New `ash-rollout-v3` requests default to mini when slot/profile are omitted.
New mini branch requests default to assistant-turn delivery and require an
authored assistant message; see [RL assistant-turn branches](RL_ASSISTANT_TURN.md)
for the API, retry behavior and scheduling boundary.

Add a profile using the existing microVM/backend and credential references,
the interpreter containing mini, and these rollout defaults:

```json
{
  "slot": "mini-swe-agent",
  "tools": "shell_only",
  "transport": "http",
  "extra": {
    "mini": {
      "environment": {"cwd": "/repo", "timeout": 60}
    }
  }
}
```

Set the Driver's `profile` to that profile and `run_defaults.slot` to
`mini-swe-agent`. Configure `tasks[task_id].grade` as for other agents.
Use **`ash-rollout-v3`** / Miles `AshMessageRolloutFn`; the older v2 token-session
adapter rejects mini explicitly.

The v3 caller supplies `model_endpoint`, `model`, `sampling_params`, `max_turns`,
wall-time budget and task/image references. The worker returns actual messages,
the `bash` function schema, reasoning text, tool results, grading reward and
branch origin. The trainer can then tokenize and construct loss masks using
the same v3 consumer as the other agents. Do not rename `bash` to `shell` in
training messages: the model sampled `bash`, while `shell` is the environment
transport.

This checkout's Driver uses its existing independent-sample scheduling policy.
Queue branches use `POST /v1/jobs/{job_id}/branch`. Deployments with automatic
review/branch scheduling must also enable mini in their sequence counter,
hint-delivery and timeout-prefix policies; replacing those deployment files
with older checkout versions would discard existing behavior. Sequence-limited
profiles additionally need the deployment's tokenizer dependencies and the same
tokenizer/chat template as the learner.

## Checkpoints, branches and interruption

Native JSONL is written and flushed as events happen. Every successful command
can have a disk snapshot; an eligible conversation cut requires the **entire
model response's tool results**. A response containing two commands can branch
after the second, never between its unresolved tool calls.

Run Store indexes these boundaries, hashes the exact native prefix, checks the
snapshot and gives each child its own native history. The child receives only
that prefix plus its branch instruction. Training export removes the marked
instruction from user/system messages and preserves assistant/tool content.
`swebench.fork_eval` uses the same mini prefix index.

Submission is the upstream `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` command. Its
actual observation is persisted before exit, keeping training tool calls paired.
Transport errors and invalid/incomplete exports are errors, not reward zero.

A model request is cancelled and its HTTP client closed when the rollout stops.
An executing shell command is allowed to settle under its runtime timeout before
final snapshotting. If a time limit interrupts a response between commands, the
worker can retain the previous closed native prefix **and its matching snapshot**
for grading, recording the discarded later snapshot in `timeout_fallback`.
It never combines an earlier message prefix with a later filesystem state.

## Verification

```bash
python -m pytest harness/tests/test_mini_swe.py harness/tests/test_chat_gateway.py \
  runstore/tests/test_mini_native.py -q
```

These tests use the real upstream loop, real HTTP boundaries and real shell
commands on temporary filesystems, with controlled model replies. They cover
multiple actions, exit status, submission, turn limits, exact branching,
hint removal, native-prefix integrity and partial-response timeout recovery.
Real VM/queue/grader and learner validation should additionally be run against
the deployment that will consume the feature.
