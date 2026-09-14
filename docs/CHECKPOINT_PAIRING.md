# Exact checkpoint pairing

Implemented 2026-09-06 for the owned Claude Code rollout path (HTTP and stdio).

A branch at step N must combine the environment after tool N with the native
conversation ending at that same tool's result. The conversation's start order
is not interchangeable with the count/order of successful executor callbacks.

## What changed

- Claude CLI MCP total and idle-progress waits both default explicitly to600
  seconds via `MCP_TOOL_TIMEOUT=600000`,
  `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=600000`, and each configured MCP server's
  `timeout: 600000`. The server-level timeout is required because Claude CLI
  prioritizes it over the global idle setting for configured servers. Task-specific env wins over inherited process env,
  which wins over this default. The effective value is recorded in
  `run.started.config.mcp_tool_timeout_ms`; it is independent of the command's
  own timeout and the whole rollout cap. Existing frozen batches are not edited
  or restarted by changing this default in the main checkout.
- The approval hook and native event stream register the same tool ID once in
  the journal. `tool.started.step` is assigned there, including calls which never
  reach the executor. The hook carries `__ash_checkpoint_call: {step, call_id}`
  through updatedInput; the server removes it before runtime dispatch. The
  model-facing tool schemas and AgentENV snapshot API did not change.
- One same-sandbox gate covers execution and capture. Cancelling an HTTP handler
  does not release the gate while its worker or snapshot is still active.
  Session event-loop access is also serialized across worker threads.
- New checkpoint events carry `pairing: "call-id-v1"`, `call_id`, and
  `prefix_complete`. Failed, cancelled, missing, and non-prefix boundaries do
  not become branch candidates. Known state-preserving steps can reuse a
  snapshot without losing their own identity.
- HTTP dispatch checks the identity against the approved tool and arguments;
  duplicate and already-finished/late requests cannot mutate the environment.
  A transport failure with unknown execution state blocks subsequent mutations
  and captures. It also requests out-of-band termination of the model driver;
  refusal text alone must not leave a model repeatedly asking for tools. The
  Claude SDK stream is read and closed by one owning task, with an abort wakeup
  even while no model message arrives. Ordinary command exit failures are not
  transport uncertainty. As of 2026-09-07, a runtime timeout with
  `running=false` and a known exit code is also a completed command outcome:
  capture its actual post-timeout filesystem, return the output with the
  `[timed out]` marker through the default presenter, and let the agent continue.
  Files written before timeout remain; there is no implicit rollback or retry.
  A still-running command, a timeout without an exit code, or a transport
  failure remains terminal. This requires runtime timeout containment (the
  v0.1.4 process-group cancellation fix); it does not recover unknown execution.
  `execution_detail` retains the first trigger's kind/exception or runtime flags.
- Native cuts reject unresolved parallel tool calls and messages containing
  results from beyond the requested step. Per-tool disk snapshots therefore do
  not imply every intermediate parallel tool has an independently loadable cut.
- Fork planning requires an exact snapshot/conversation pair. Grading instead
  evaluates the last successfully persisted snapshot, even if later operations
  or captures failed. The benchmark verifier is unchanged. `Grade.grading_snapshot`
  records policy `last_successful_snapshot`, the selected capture step/sequence
  and snapshot ID, plus later tool calls and checkpoint issues. Failed captures
  and session-reference backfills cannot replace a newer successful capture.
  A conversation reference is unnecessary for grading a saved environment.
  No successful capture still means there is nothing to grade. This grading
  policy was adopted on 2026-09-08; old frozen batches and grades are unchanged.
- Shutdown drains execution/capture before releasing the pool. If draining times
  out, the owner reports an error and retains the sandbox for TTL/ledger cleanup,
  instead of destroying it under the worker. Closed journals reject late captures.

## Scope and compatibility

For SDK 0.2.145 with its bundled CLI 2.1.247, a tool-result UUID explicitly listed in
`compactMetadata.preservedMessages.uuids` can survive one later compaction.
Admission requires no intervening assistant/user content between that result
and the compaction: otherwise the summary could carry observations from beyond
the selected state. Post-cut attachments are also rejected except for the
observed token-budget reminder. Missing/malformed preservation metadata and multiple later
compactions remain rejected. Ordinary turn-completion and exact-call checks
still apply. This narrow exception restores the compressed prefix plus the
preserved result, not every original raw message or the parent's later answer.

The real Koota step84 probe accepted the UUID through the installed SDK using a
local canned provider and independently verified the restored snapshot against
the original graded patch. Evidence:
`runs/koota-native-cut-probe-20260907-gBiTkL/verified.json`. This is not a general
claim that arbitrary pre-compaction cuts are loadable.

### Original prefixes before compaction

`conversation_restore()` now chooses between that native route and an independent
original-prefix session. `available_branch_points()` and `prepare_branches()` use
this shared definition, so Analyst and Reviewer can select an earlier complete
turn even when `conversation_cut()` rejects loading it from the current session.
Only the compaction restriction is bypassed for prefix reconstruction; missing
snapshots/call identities, incomplete parallel turns and mismatched results stay
ineligible. Missing or malformed original-prefix records are not reconstructed.

For each selected prefix branch, `harness/slots/claude_history.py` copies records
only through the selected result UUID into a new session ID, registers it in the
native project directory for that branch's actual host cwd, and resumes/forks
that session at the same UUID. It excludes later messages and later compaction
summaries. Original message payloads, entry UUIDs and parent links are retained;
only session/cwd metadata changes. Local project settings are disabled for the
isolated actor workspace. Original parent/session files are never rewritten.

Each branch saves `conversation-prefixes/<branch>/prefix.jsonl` and `manifest.json`
next to its journal. The manifest pins the source transcript and any referenced
persisted-output files by hash; those originals must remain available. Changed
sources or unavailable output references stop preparation instead of silently
changing history. Plans and `fork.origin` record `conversation_restore`, the
source and resume session IDs, and the prefix manifest path. Native registration
uses the installed SDK's path sanitizer; verify this seam on SDK upgrades.

Production-path verification:
`runs/prefix-branch-integration-20260907-ukuck4/verified.json` exercises `run_one`
with a canned Analyst/Reviewer/verifier and local Messages API, real SDK/CLI,
real HTTP MCP and a real step40 sandbox. Its first model request contains exactly
the40 expected tool-result IDs; the next real tool verifies the step40 edit is
present and step41's edit absent. The original parent stays unchanged and the
probe VM is destroyed. This tests plumbing, not benchmark improvement.

The orchestrator enables identity propagation automatically for owned Claude
Code HTTP/stdio runs. Other slots retain legacy pairing; they receive the
executor ordering protection but are not certified to provide native-ID exact
cuts. In stdio, a missing predecessor cannot be certified from the subprocess's
map alone, so such prefixes are conservatively excluded.

Legacy journals remain readable; these changes do not repair or certify their
old step labels. No historical trajectories or snapshots were rewritten.
Prompt changes, benchmark/model reruns and AgentENV deployment are separate.

## Verification

From the Ash root:

```bash
PYTHONPATH=.:sdk python3.12 -m pytest harness/tests swebench/tests sdk/tests deepswe/tests -q
```

Result: 707 passed, 7 skipped. New regressions live in
`harness/tests/test_checkpoint_identity.py`, `test_checkpoint_ordering.py`, and
`swebench/tests/test_conversation_cut.py`. They cover cancelled execution/capture,
missing and late requests, reorderings, capture failure, read-only reuse, shared
loop access, shutdown, stdio map tailing, and native parallel cuts.

The installed Claude SDK/CLI was also exercised against a local canned Messages
API and the real HTTP MCP server with an in-memory sandbox. Both tool IDs crossed
the real hook boundary correctly, snapshots matched A then A+B, and both real
native conversation cuts were found. Reproduce from the LBP root:

```bash
python3.12 artifacts/checkpoint-fix-20260906-UdrSQa/offline_sdk_probe.py
```

Evidence: `artifacts/checkpoint-fix-20260906-UdrSQa/sdk-probe-5ik_lorp/probe.jsonl`.
This probe uses no paid model, AgentENV or VM. Actual disk snapshot/restore and
the paused DeepSWE pilot have not been rerun.

The generic `contracts/ci_check.py` did not pass: its existing SDK field checker
calls `Report.ok` with an extra argument and raises TypeError. Its Claude version
entry also predates the already-installed CLI. Neither checker nor dependency
versions was changed in this repair.

## Follow-up: terminate quarantined model loops

The initial refusal-only implementation left real Arcane/Awilix actors retrying
rejected tools. The 2026-09-06 follow-up connects checkpoint execution_uncertain
events to a per-run RunControl and the model driver's stream cancellation.
Regression tests in `harness/tests/test_run_abort.py` cover silent streams,
deadlines, same-task close, and actual orchestrator-to-driver termination for
transport exceptions, timeouts without a known exit code and still-running
results. The 2026-09-07 continuation tests verify that settled timeouts reach
the SDK driver with output and the timeout marker, retain partial writes in an
exact snapshot, permit the next command, and remain eligible for grading even
when the timed-out command was the final call. Both dispatch paths also verify
that capture finishes before the next command and uncertain outcomes stay blocked.

The complete Python suite on 2026-09-07 passed 758 tests with 7 skipped using
the command above. These continuation regressions use simulated runtime
outcomes and a simulated SDK response stream with the real harness wiring;
they do not establish live model recovery success or certify old failed runs.

`artifacts/quarantine-fix-20260906-mxWYSz/offline_abort_probe.py` exercised the
installed SDK/CLI with a localhost canned provider and the real MCP server.
An injected transport uncertainty ended the whole rollout in about 1.9 seconds,
with one backend request and an explicit error; normal two-tool execution also
passed. No paid model or VM was used for that probe. Frozen previous attempts
remain untouched; affected tasks are rerun in a separately versioned batch.
# Completed model-turn branching (2026-09-06)

Storage remains per tool call, with its original step/call_id. New Claude Code
runs also record `branch.boundary.policy=completed-model-turn-v1` and
`model.turn.tool`, `model.turn.output_completed`, `model.turn.completed` events.
SDK AssistantMessages sharing message_id are one model response, not separate
turns. The next distinct response or successful query ResultMessage certifies
that response's output is closed; all of its tool results must also be recorded
before a complete-turn boundary is published. This intentionally delays live
publication until the next response; a temporarily empty pending set does not
close a streaming response. Missing identities and reopened responses fail closed.

`branch_checkpoints()` remains the exact-pair view used by fork planning.
Grading selects the last successful capture independently of conversation pairing.
`turn_branch_checkpoints()` intersects it with completed model turns;
`swebench.fork_eval.available_branch_points()` additionally verifies the native
cut. Analysts, reviewers and the batch canary consume that candidate definition.
The native cut reader groups split assistant entries by message.id and still
rejects pending calls and future results. Cuts invalidated only by later
compaction may use the checked original-prefix reconstruction path above.
No historical snapshot/step is removed or renumbered. To regenerate a turn,
branch before its whole response/tool group, not partway through the group.

Real SDK/CLI + AgentENV verification (local canned model, no paid inference):
`../artifacts/turn-boundary-20260906-jcFpxy/probe.py`; the recorded run proves
per-tool disk states, exclusion of a mid-turn snapshot, and native resume with
both tool results but without the parent's later disk/context changes.
# VM lifetime is independent of MCP/command timeouts (2026-09-06)

`swebench.fork_eval.backend_for()` now budgets the microVM lease from the actor
wall-clock timeout plus600 seconds of provisioning/teardown margin. A10800s
DeepSWE attempt therefore uses sandbox_ttl11400, not the generic600s pool
default. The same configured pool value reaches template/cold creation,
snapshot restoration, reboarding, explicit resume and VM forks. Longer explicit
leases passed through `with_sandbox_budget()` are preserved; input configuration
is not mutated. New fork-eval summaries record sandbox_ttl separately from the
actor timeout. Command timeouts are unchanged; MCP_TOOL_TIMEOUT,
CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT and the per-server timeout are600000 for new runs.

The generic pool defaults remain unchanged for non-rollout users. Existing
frozen campaigns do not pick up main-source changes automatically. The held
bIaGkT campaign uses a separately logged temporary lease keeper for confirmed
Running actor VMs and their recorded snapshot descendants; it neither resumes
Paused VMs nor changes code/results. It stops when those workers drain.

Avoid treating all sandbox endpoints alike: POST /refreshes has a3600s schema
limit, whereas POST /timeout supports the full operation lease. Neither should
be confused with the600-second MCP total/idle response waits. Auto-resume does not make a
short lease harmless: a call arriving while the VM is Pausing can receive410.
