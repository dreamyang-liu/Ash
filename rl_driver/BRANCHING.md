# Review-guided RL branching

Branching is opt-in for `ash-rollout-v3`. Each prompt progresses independently:

1. Run one root trajectory and grade its final snapshot.
2. If unresolved, review to select up to **4** repair branches in the first
   round and, if needed, up to **3** in the second.
3. If the root is already resolved, run one review round selecting up to **2**
   plausible mistaken continuations, seeking negative examples, then finish.
4. The reviewer decides the actual count and each parent, point and direction.
   Different directions may share a point. Caps are not mandatory counts.
5. Run and grade every child selected for the current round. Only after the
   whole round finishes, check the root and all completed children for both
   positive and negative rewards. If both exist, do not start another round.

Only the real grader's `resolved` boolean controls direction and stopping.
Reviewer predictions, actor self-assessment and infrastructure errors do not
create rewards. The second review sees the root and all previous children,
indexed tool steps, grades and previous directions. It may select any available
point from any of those attempts. A direction intended to fail is not assigned
a negative reward unless its real grader reports unresolved.

## Configuration

Add this under the driver's `miles` configuration:

```json
{
  "branching": {
    "enabled": false,
    "branch_limits": [4, 3],
    "successful_root_limit": 2,
    "reviewer_model": "<your existing Bedrock reviewer model>",
    "reviewer_region": "us-west-2",
    "reviewer_timeout_s": 300,
    "reviewer_workers": 2,
    "return_mode": "pair"
  }
}
```

Review uses the existing tool-free Bedrock transport, shared with
`swebench.fork_eval` through `model_review.py`. The driver process requires
`AWS_BEARER_TOKEN_BEDROCK`. Missing model configuration or credentials is
rejected before submitting the root. Credentials are never put in the ledger.

Enable the deployment default with `miles.branching.enabled=true` or the driver
CLI `--branching`; `--no-branching` clears that default. A v3 request can opt in
with `"branching": true`. Disabled requests retain independent sampling and the
old wire shape. V2 does not support this policy.

The matching Miles caller uses:

```bash
--rollout-function-path miles.rollout.ash.message_rollout.AshMessageRolloutFn \
--ash-rollout-branching \
--n-samples-per-prompt 2
```

For an owned local reviewer, set `reviewer_endpoint` to its serving base URL
and `reviewer_model` to its base model name. This uses Chat Completions and does
not require Bedrock credentials. `reviewer_api_key_env` is optional;
`reviewer_max_tokens` defaults to16384. In-flight review calls are drained
before a timed-out group becomes ready, so a colocated trainer does not offload
inference while its review call is still active.

To bound local reviewer reasoning independently of its final JSON, configure:

```json
{
  "reviewer_max_tokens": 32768,
  "reviewer_thinking_budget": 16384,
  "reviewer_timeout_s": 2400
}
```

The total output budget includes reasoning and final content. A thinking budget
requires SGLang `enable_strict_thinking`; the client verifies this before making
the model request and sends `custom_params.thinking_budget`. It also requests a
JSON schema for the final plan. A backend without enforcement is rejected.
The serving context limit is separate and must fit the review input plus output.

## Sequence limits and discounted rewards

V3 requests can set `max_sequence_tokens` and `truncated_reward_scale`.
Configure `miles.sequence_tokenizers` as an exact model-name→local tokenizer
directory mapping (include the LoRA-qualified name when serving an adapter).
The worker bounds native output by the remaining sequence budget. At final
export it counts the cleaned history with that tokenizer. If tool output or
history still exceeds the limit, it chooses a complete native checkpoint prefix
that fits and grades that prefix's paired snapshot. The later full-run snapshot
is retained separately, never used to score the shortened training sequence.
No eligible prefix is an explicit unusable episode, not arbitrary token slicing.

Correct truncated episodes receive `resolved × truncated_reward_scale`;
natural successes remain1, failures0. Search decisions continue to use the real
grader boolean. Branch review receives retained-prefix token counts and excludes
discarded or over-budget points. Defaults preserve the previous uncapped,
undiscounted wire contract.

## Samples and stopping

`pair` requires two allocated slots and `minimum_returned_samples=2`. At the end
of the search, it returns the root and the first child with the opposite reward;
if no child has the opposite reward, it returns the root and last child.
Unselected attempts remain
in Run Store and the driver ledger. `search_branches` and model/tool usage count
all executed children, including unselected ones.

No trajectory is duplicated to fill a slot. Missing recovery points or failure
before a valid pair exists produce an explicit shortfall, which the fixed-size
Miles caller rejects. A fresh root never substitutes for an exact branch.

Reviewer output is a JSON object with a `branches` list and optional `synthesis`.
Each entry contains `job_id`, `point_id`, `reason` and `hint`. The list may contain
fewer entries than the cap, including zero with an explanation. An over-cap plan
is rejected as a whole, without trimming or filling. A zero-branch decision ends
the search explicitly. Root-success review is capped at two children regardless
of whether they ultimately fail or succeed.

For callers supporting variable group sizes, `return_mode="all"` returns every
executed trajectory. With the default schedule, allocate at least **8** slots
(root + 4 + 3) and set
`minimum_returned_samples=1`; early stopping leaves unused slots empty. The
standard Miles flag uses fixed pairs, not this variable-size mode.

Root, reviews and children share the execution deadline. No new review/child
starts after it. Snapshot export and grading retain the finalization allowance.
Per-trajectory `max_turns` and native sampling settings are inherited.

## Hints and training

Only the reviewer's actor-facing hint becomes the continuation. The worker
wraps it in `<ash_training_hint>...</ash_training_hint>` after exact restoration.
Export removes marked user/system/developer injections across inherited rounds.
Training metadata retains lineage and outcomes, excluding hints and review text.
Assistant/tool text stays as actually recorded, including any literal quotation;
export does not perform global string replacement on those outputs.

Miles re-tokenizes cleaned histories and recomputes logprobs. Hint-conditioned
rollout logprobs are not reused. Existing per-sample loss/advantage semantics
remain: this feature adds neither tree-specific shared-prefix weighting nor
importance correction.

## Durability

Reviews execute outside the polling loop. Evidence, accepted plans, rounds and
submission intentions are durable. Restart may repeat an unacknowledged review;
accepted plans are reused. Actor/grader submissions retain stable idempotency
keys after a lost HTTP acknowledgement.

Every point in a round is checked before any child is appended, and Run Store
checks again before execution. Invalid points are rejected without substitution.
Child identities include the round and within-round index, so multi-child
submissions remain idempotent after restart or lost acknowledgements.
Cancellation discards pending
review results and prevents new children. A running review HTTP call may finish
without its result being consumed. Worker/Run Store retain ownership of running
actor cleanup. Full review evidence stays in the diagnostic ledger, not the v3
training export.

## Validation

`rl_driver/tests/test_branching.py` covers both directions, 4/3 and successful-root
2 caps, reviewer-selected counts and points, whole-round stopping, real grading,
selection, hint removal, point/configuration failures, cancellation, deadlines,
concurrent roots and multi-child restart/idempotency. Miles tests cover pair
import even when seven search branches were executed. These are controlled
HTTP/execution tests, not a claim of live model branching or an optimizer update.
