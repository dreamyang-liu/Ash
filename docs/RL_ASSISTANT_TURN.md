# Mini assistant-turn branches in RL

For `ash-rollout-v3` message rollouts, omitted `run_defaults.slot` now selects
`mini-swe-agent`; an omitted profile selects `mini-swe-agent` for that slot.
Explicit agent/profile configuration is preserved. The older v2 token-session
protocol retains its compatibility behavior and does not support mini.

Install mini into the Python interpreter named by the worker profile. The
examples in `rl_driver/config.example.json` and `runstore/config.example.json`
include the mini profile. Configure the image resources, grading tasks and
model endpoint/credential references for the deployment.

## Branch submission

New mini branches submitted through the Run Store branch endpoint default to
`assistant-turn`. The caller supplies a complete reviewer-authored message:

```json
{
  "point_id": "SELECTED_RECOVERY_POINT",
  "overrides": {
    "assistant_turn": {
      "role": "assistant",
      "content": "I will verify the current working directory before the next edit.",
      "tool_calls": [
        {
          "id": "unique-reviewer-call",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\":\"pwd\"}"
          }
        }
      ]
    }
  }
}
```

Send this to `POST /v1/jobs/{parent_job_id}/branch`. A generic RL execution group
can provide the same object under a sample's `branch.overrides`; the driver
forwards it without running the agent or editing the tool call.

The API checks that the point belongs to the parent, its native prefix and
snapshot remain valid, the slot matches, and the assistant turn contains only
valid bash calls with IDs absent from the retained history. Invalid guidance
returns a validation error before any child is queued.

The child restores the original prefix and snapshot, executes the injected
assistant turn, records actual observations, then asks mini to continue.
It does not append the branch prompt as a user message. Only normal actor
requests consume the actor's model-call count; injected commands still execute
through the ordinary tool/checkpoint path.

Explicit alternatives are retained:

- `branch_guidance: "user-hint"` with `prompt` for the legacy hint behavior.
- `branch_guidance: "none"` without a prompt or assistant turn for a point-only continuation.
- Non-mini slots default to user hints and reject assistant-turn/point-only modes.

Assistant-turn branches require a message; the API never invents an empty seed
or silently falls back to a hint. Existing stored jobs without a guidance marker
retain their historical interpretation.

## Retry and export

A branch of a branch receives a new seed only when one is supplied. Its parent's
seed remains part of the retained history and is not executed again.
An infrastructure retry from a point in the same job resumes mini without
reinjecting that job's original seed.

Message export retains ordinary reviewer assistant content and real tool
observations. `training_origin` and the RL trajectory's origin metadata include
the guidance mode, reviewer source and injected call IDs, where applicable.
This is provenance for consumers; it does not silently change learner loss masks.

## Scheduling boundary

These changes select the agent and how a requested branch is delivered. They
do not introduce automatic reviewer sampling or a 4→3 search schedule into the
v3 independent-sample planner. The caller currently chooses when to branch and
supplies the reviewer message. Automatic search must separately define how
roots, children, returned samples and reviewer calls fit the RL group's budget.

## Verification

```bash
PYTHONPATH=.:sdk python -m pytest \
  runstore/tests/test_rl_assistant_guidance.py \
  rl_driver/tests/test_branch_guidance.py \
  rl_driver/tests/test_message_rollout.py -q
```

The tests cover API rejection, driver forwarding, real mini execution against a
temporary filesystem, exact prefix restoration, seed non-reexecution on retry
and exported assistant/tool content and provenance.
