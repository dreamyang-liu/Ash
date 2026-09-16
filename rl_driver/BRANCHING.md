# Review-guided RL branching

Branching is opt-in for `ash-rollout-v3`. Each prompt progresses independently:

1. Run one root trajectory and grade its final snapshot.
2. If unresolved, review the attempts to find a repair direction likely to resolve.
3. If resolved, review for a plausible mistaken reasoning or implementation
   direction likely to produce an unresolved continuation.
4. Select one available recovery point, restore its exact conversation prefix
   and filesystem, and execute one child with the reviewer's direction.
5. Grade the child. Stop on the first opposite signal, or review again.
   The default is at most two rounds after the root.

Only the real grader's `resolved` boolean controls direction and stopping.
Reviewer predictions, actor self-assessment and infrastructure errors do not
create rewards. The second review sees the root and previous child, indexed
tool steps, grades and previous directions. It may select a point from either.

## Configuration

Add this under the driver's `miles` configuration:

```json
{
  "branching": {
    "enabled": false,
    "max_rounds": 2,
    "reviewer_model": "<your existing Bedrock reviewer model>",
    "reviewer_region": "us-west-2",
    "reviewer_timeout_s": 300,
    "reviewer_workers": 2,
    "stop_on_negative": true,
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

## Samples and stopping

`pair` requires two allocated slots and `minimum_returned_samples=2`. It returns
the root and the first child with the opposite reward; if both rounds retain the
root's reward, it returns the root and last child. Intermediate attempts remain
in Run Store and the driver ledger. `search_branches` and model/tool usage count
all executed children, including unselected ones.

No trajectory is duplicated to fill a slot. Missing recovery points or failure
before a valid pair exists produce an explicit shortfall, which the fixed-size
Miles caller rejects. A fresh root never substitutes for an exact branch.

`stop_on_negative=false` runs both rounds for an initially successful prompt,
even if its first child is unresolved. A failed root always stops when resolved.

For callers supporting variable group sizes, `return_mode="all"` returns every
executed trajectory. Allocate at least `max_rounds+1` slots and set
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

Points are checked before and after review and by Run Store before execution.
Invalid points are rejected without substitution. Cancellation discards pending
review results and prevents new children. A running review HTTP call may finish
without its result being consumed. Worker/Run Store retain ownership of running
actor cleanup. Full review evidence stays in the diagnostic ledger, not the v3
training export.

## Validation

`rl_driver/tests/test_branching.py` covers both directions, real grading, round
limits, selection, hint removal, point/configuration failures, cancellation,
deadlines, concurrent roots and restart/idempotency. Miles tests cover the flag
wire format and pair import. These are controlled HTTP/execution tests, not a
claim of live model branching or an optimizer update.
