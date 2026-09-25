# Branch guidance modes

`swebench.fork_eval` supports three guidance modes:

The default agent is `mini-swe-agent`, and its default guidance is `assistant-turn`.
With no `--slot` or `--branch-guidance` flags, both defaults apply. Explicitly
selecting another agent (such as `--slot codex` or `--slot claude-code`) keeps
`user-hint` as that agent's default. An explicit guidance flag always wins,
subject to the compatibility checks below.

| Flag | Branch continuation |
|---|---|
| `--branch-guidance user-hint` | Deliver the reviewer's hint as before; default for non-mini slots. |
| `--branch-guidance assistant-turn` | Restore a mini prefix, execute a reviewer-authored assistant turn, then continue mini; default for mini. |
| `--branch-guidance none` | Select a mini checkpoint and query the actor directly with its retained history alone. |

The `assistant-turn` and `none` modes accept **only mini-swe-agent parents and mini continuations**.
They require an exact completed-turn checkpoint; `--fork-full-conversation` is
rejected. Existing Claude Code/Codex trajectories are not converted.

These are branching-controller defaults. Low-level `RunSpec` and existing
Run Store profile defaults are unchanged, and `harness run` still asks for an
explicit slot. Configure mini's Chat Completions endpoint as described in
[mini setup](MINI_SWE_AGENT.md); changing the default slot does not configure
model credentials or analyst/reviewer routes.

In `none` mode, the reviewer chooses restart points only. No user message,
empty message, assistant response, command or continuation instruction is added.
The first child model request contains exactly the retained native prefix,
using mini's normal request serialization. Tools execute only after that model
produces its own response.

The analyst and branch-count policy remain in place. Reviewer output uses
the same `branch-plan` fence but each branch permits only `name`, `base`,
`branch_step` and `why`:

```branch-plan
{
  "synthesis": "Restart after the initial inspection.",
  "branches": [{
    "name": "after-inspection",
    "base": "parent",
    "branch_step": 12,
    "why": "Useful observations are available before the unsuccessful edit."
  }]
}
```

`synthesis` and `why` remain controller metadata and never enter the actor
conversation. Plans with `hint`, `assistant_turn`, commands or other extra
branch fields are rejected. Existing messages in the retained prefix are
preserved; this mode does not delete earlier history.

Example using an existing mini parent:

```bash
python -m swebench.fork_eval \
  --slot mini-swe-agent \
  --model YOUR_ACTOR_MODEL \
  --analyst-model YOUR_REVIEWER_MODEL \
  --instance YOUR_INSTANCE_ID \
  --parent-from runs/mini-parent \
  --branch-guidance assistant-turn \
  --rounds 2 --branches 4,3 \
  -o runs/mini-assistant-branches
```

Replace `--branch-guidance assistant-turn` with `--branch-guidance none` to
run the point-only variant. The following assistant-turn schema and injected
execution sequence apply only to `assistant-turn`.

Keep the same benchmark, dataset, environment and model endpoint configuration
used by the mini parent. This flag works in the shared `fork_eval` controller
for SWE-bench, DeepSWE and SWE-bench Pro; it does not modify archived batch
controllers or launch experiments itself. Mini's dependency remains pinned in
`harness/requirements-mini-swe-agent.txt`.

## Reviewer output

The analyst stage, branch counts and base/step selection are retained. The
reviewer receives analyses, the task, and each attempt's native messages with
`prefix_message_counts`: the exact number of messages retained for every
eligible step. This includes inherited messages for branches of branches.
The reviewer must distinguish the selected prefix from its discarded suffix.

Instead of `hint`, each new-mode branch contains `assistant_turn`:

```branch-plan
{
  "synthesis": "Check whether the branch handles the empty input.",
  "branches": [{
    "name": "empty-input",
    "base": "parent",
    "branch_step": 12,
    "why": "The relevant function has just been inspected.",
    "assistant_turn": {
      "role": "assistant",
      "content": "The empty-input path may bypass this check. I'll inspect the caller before changing it.",
      "tool_calls": [{
        "id": "review-empty-input-1",
        "type": "function",
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"rg -n 'def parse|parse\\\\(' src\"}"
        }
      }]
    }
  }]
}
```

`content` is ordinary assistant text, not a provider's private reasoning field.
The message must contain at least one `bash` call. Arguments are a JSON string
with exactly one non-empty string `command`; IDs must be unique within the
message and absent from the selected history. Content and commands are not
rewritten. Additional fields, native thinking, observations and a simultaneous
`hint` are rejected before branch execution.

## Execution and records

After restoring the selected snapshot and native prefix:

1. Mini keeps the original system/task messages and does not append a user hint.
2. The reviewer turn is durably recorded as an assistant message. Mini's existing
   tool parser validates it and the existing MCP environment executes the calls
   sequentially, using the normal command timeout and checkpoint path.
3. Actual outputs and exit codes become normal tool observations. An ordinary
   nonzero exit is an observation, not a fabricated successful result.
4. Only a fully observed turn publishes a branchable native boundary. An
   interruption between calls leaves no completed boundary for that partial turn.
5. Mini queries the actor model with the retained prefix, reviewer turn and real
   observations, then continues its usual loop.

Injection itself does not call the actor model or invent actor usage. Its tools
consume the normal execution time/tool limits; subsequent actor requests remain
subject to the existing model budgets.

The plan, `fork.origin` and `branch.assistant_turn` journal event record the
reviewer origin. The native assistant message's `extra.source` also records
`reviewer`, but this bookkeeping is not sent to the actor. Native turn records
remain indexable. Resuming after the injected turn inherits its completed work;
it does not execute the injection again. Training-message export retains the
ordinary assistant/tool messages and their call IDs.

The reviewer sees native context without a new lossy truncation step; long
histories therefore increase its input size. Prompt instructions encourage
natural continuation and prohibit invented prior observations or hidden grader
details. Structural checks establish execution and history consistency, not
that the generated reasoning is natural or that the repair will succeed.

## Verification

The new tests drive the pinned mini loop through local HTTP model and MCP
servers. Commands run against temporary filesystems with checkpoint copies;
they make no external model calls and do not create microVMs.

```bash
PYTHONPATH=.:sdk python -m pytest \
  harness/tests/test_assistant_turn.py \
  runstore/tests/test_assistant_turn_branch.py \
  runstore/tests/test_no_hint_branch.py \
  swebench/tests/test_assistant_branch.py -q
```

Use a Python environment with the mini dependency installed to run the integration
cases; they are explicitly skipped when mini is unavailable.
