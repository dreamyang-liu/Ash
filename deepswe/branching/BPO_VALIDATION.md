# BPO sampling validation

Date: 2026-09-20. This branch extends the published Shepherd implementation
`0a0c6de7d7b49b45040951c0374473aeec347ce2`, retaining its baseline, harness,
checkpoint-restore and verifier integration.

## Automated checks

- Linux: **1,024 passed, 17 skipped, 6 deselected** across `harness/tests`,
  `swebench/tests`, `sdk/tests` and `deepswe/tests`, using `-m 'not slow'`.
- Windows DeepSWE tests: **91 passed**.
- Clean Python 3.12 environment, installed from the documented requirements:
  **305 passed** across SWE-bench, BPO/Shepherd sampling, bridge lifecycle and exact-prefix tests.
  CI explicitly runs both sampling suites and retains the tool-panel guards.
- SDK/runtime contracts: **58 passed, 0 failed, 7 skipped** (unavailable unrelated
  harness CLIs).

Before the bridge lifecycle fix below, the Linux release checkout, local checkout
and live validation implementation shared this non-test Python source fingerprint:
`fd1df5bd973a8263b5f88a0a363b1c888e089231f111a8ea48c658c2a83728cf`.

## Real Qwen probability and restoration pilot

Model: `qwen3.8-27b`; Claude Code harness; reasoning `high`; AgentENV
`v0.1.2-ash.1`. Task: `etree-xml-diff-patch`, using the existing failed initial
rollout and its digest-pinned task image. Budget: one shared initial rollout plus
one new branch (`max_rollouts: 2`). This is an integration pilot, not an
eight-rollout/full-cohort accuracy measurement.

All **45** eligible candidate scoring requests were checked against their exact
original provider requests. Only probability-output options, streaming flag and
output cap changed for historical-prefix rescoring. Returned first-token
probabilities were matched to their audit records and every entropy value was
recomputed. Applying the published selection code independently reproduced the
saved branch plan: checkpoint **9**, entropy lower bound
**1.7264729468646447 nats**.

The observable statistic is the **first reported content token's top-k-plus-tail
entropy lower bound**, not full-vocabulary entropy or hidden reasoning-token
entropy. This is explicitly an API-constrained sampling adaptation, not a claim
to reproduce the paper's training algorithm or training results. Missing/rejected
logprobs blocks sampling; no random or alternate-model fallback is used. The
available DeepSeek endpoint rejects logprobs, so the real probability validation
uses Qwen as requested.

The worker **completed 101 tool calls** and reached the independent task verifier:
**F2P failed, P2P passed, resolved=false**, with no grading infrastructure error.
This validates the selection/restore/continuation/grading pipeline, not task success.

The selected native prefix contained **36 entries**, excluding **265 later source
entries**. Its exact contents and first worker request were verified against the
original history. Prefix SHA-256:
`2d5dcdd3fc7a34ff01961fe96c3ab02b1fd72de594335a28de78528c13b28e52`.
Final journal SHA-256:
`3c3dd04614b1ec4e812265280cee72f1745b15e44d966111c09f953713b93b96`.

Reported actor usage: 8,446,327 input tokens (7,482,880 cached), 135,971 output
tokens (83,602 reasoning). There were 94 finalized request records: 89 successful
responses and five without usage. **Seven additional interrupted audits have no
final accounting** and are reported separately; these totals are not a complete
bill. The 45 cached scoring requests report 4,276,680 input and 45 output tokens;
they were reused, not charged again for the final pilot. No dollar cost is inferred.

## Step-92 bridge lifecycle recovery

The Qwen worker completed 92 tool calls, then its native Claude Code context
compaction encountered upstream 502/timeouts. The original bridge failed to cancel
non-streaming work after client disconnects, allowing obsolete retries to remain
active or queued. This was an implementation defect, not a BPO entropy-selection
failure. It is fixed with disconnect cancellation, bounded queue admission, a
total upstream deadline, and at most three audited transient-error attempts.

Nine lifecycle tests cover real HTTP disconnects (streaming and non-streaming),
queue cancellation/deadlines, semaphore release, upstream cancellation and bounded
retries. No real model calls are made by these tests.

Recovery replaced only the isolated bridge. The existing controller, Claude actor
and sandbox remained alive at step 92; no new rollout or cold restore was launched.
The snapshot exists, but the ordinary exact-cut recovery check did not return a
loadable cut at this unfinished turn, so a cold restart was deliberately avoided.
The original journal was retained and continued; the manifest was not rewritten.
Separate recovery receipts
record bridge hashes, preserved process/session identities and unfinalized old
requests whose upstream usage is unknown. This is a mixed-version operational
recovery, not an uninterrupted run of one release revision.

After the fix, the same actor completed compaction, continued from step 92 to
step 101, and finished grading without launching a replacement rollout.

Final release source fingerprint (including explicit unfinalized-usage reporting):
`b241b2d935fc414d1954e51a8ca634159795d29021e680cdf99e81ee7ef81ea2`.

## Retained development diagnostics

Early pilots exposed a restored system-role message unsupported by the first
bridge and independent CLI total-request/event-stream deadlines. These were
fixed and regression-tested. Provider 502/403 responses and request cancellations
remain in private audits rather than being counted as ordinary model failures.
Another pilot was explicitly paused by the user while Shepherd was published;
it is not a completed BPO measurement.

The final pilot reuses the original 45 scoring audits without paying for new
scores. Actor usage and probability-scoring usage are reported separately;
imported parent costs remain in the original cohort. Interrupted requests with
unknown usage are not assigned zero cost. No model prices are assumed.

Original experiment services, checkpoints and outputs were read-only inputs.
New validation used separate bridges, directories and sandboxes. Credentials
remain in environment variables; raw prompts, responses and private server paths
are not included in this public validation note.
