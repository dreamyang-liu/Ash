# SWE-bench Pro on Ash

Runs the public Pro dataset through the existing `fork_eval` actor, MCP tools,
per-tool checkpoints, native conversation cuts, branching and regrading.
Pro's task-specific `run_script.sh` and `parser.py` come from the official
[ScaleAI repository](https://github.com/scaleapi/SWE-bench_Pro-os).

### Changing resources during recovery

Prepared snapshots retain their original VM CPU and memory. Changing only the
batch manifest does not resize them. Build replacement templates from the cached
OCI images and retain original continuation snapshots when preserving exact
trajectories. Workers forward `cpu` and `memory_mb` to the shared harness.

Recovery manifests can set `require_resource_receipts: true` to reject receipts
without matching resources and a guest-memory probe. `prepare_only: true` stops
after image validation without admitting actors. `deferred_image_indices` delays
selected images until the second preparation pass, after other images finish.
Old templates must not be retired until their replacements pass validation;
shared disk layers and trajectory snapshots must remain intact.

Maintenance recovery distinguishes finished actors awaiting verification from
unfinished actors: regrade-only workers reuse the saved outcome and final journal
without another model call. Resumed actors record their restored initial snapshot
so it remains gradable even if they finish without further tool calls. Batch
controllers prioritize pending actor canaries before other recovery work; recovery
actions and explicitly approved restarts are exposed in the summary.

## Setup

From the Ash checkout, install the normal SDK/agent dependencies and obtain the
official assets. No Modal account or Docker daemon is needed for this adapter;
the actor and verifier both use AgentENV microVMs.

```bash
pip install ./sdk datasets pandas tqdm
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git ../SWE-bench_Pro-os
git -C ../SWE-bench_Pro-os checkout ca10a60a5fcae51e6948ffe1485d4153d421e6c5
cd runtime && CGO_ENABLED=0 go build -o ash-runtime . && cd ..
export PYTHONPATH=.:sdk
export AENV_SERVER_URL=http://127.0.0.1:18000
export AENV_API_KEY=<current-server-key>
```

Configure your chosen agent slot/model as for DeepSWE. Use a static runtime
build: some Pro images have an older glibc than the
host. `CGO_ENABLED=0` avoids linking the runtime to the host's libc.

The default public data revision is `7ab5114912baf22bb098818e604c02fe7ad2c11f` of
`ScaleAI/SWE-bench_Pro`, test split (731 tasks). The loader requires the pinned
official checkout and rejects modifications to its grader/assets. It records
the dataset source, sample hash, official revision and asset hashes in artifacts.
`--pro-data data.csv` or `--pro-data data.jsonl` selects a local export instead;
`--pro-dataset-revision <full-sha>` explicitly selects another data revision.

## One trajectory

An interrupted full batch can be reconciled into a new recovery directory:

```bash
python -m swebench_pro.resume launch \
  --source runs/swebench-pro731-single-20260909 \
  --out runs/pro-recovery \
  --interrupted-at 2026-09-09T09:14:48+00:00
```

This local-AgentENV recovery path preserves valid grades and the old source/run
files. It prepares and probes every image before allowing actors to start,
including images for preserved tasks. `recovery-status.json` records this phase;
`controller.json` takes over when rollouts begin. Failed image preparation or a
disk reserve below160GiB keeps actor admission closed. Image preparation uses
two workers with one additional retry pass; no actor is automatically retried.

Interrupted trajectories continue from the latest checkpoint having both a
parseable on-disk manifest and an exact native conversation prefix. The selected
step and latest originally recorded step are both recorded; corrupt recent
checkpoints are not claimed as successfully restored. Original native histories
are not edited: a verified prefix is registered in a new actor workspace.
Previously consumed wall/tool budgets are deducted, with wall time charged up
to the explicitly supplied interruption time. The inference/verifier code is
copied from the old frozen batch; only recovery orchestration is overlaid.
Readiness is checked again in the worker. Once every image and continuation
probe passes, the controller resumes unfinished tasks with two new canaries
before opening16-worker admission. Preserved tasks are not run again.

For a frozen, background single-pass batch, `python -m swebench_pro.batch launch`
takes `--out`, `--pro-repo`, `--runtime-bin`, `--workers` (default16) and `--model`.
It requires current AgentENV and Bedrock credentials, freezes all731 tasks and
the execution source, and launches one fresh Claude Code parent per task.
There is no automatic retry or branching. Two tasks first establish live
tool/checkpoint evidence before opening full admission. Image preparation and
grading each have four slots; failures remain explicit in `controller.json`.
Patch collection keeps its configuration, alternate index and output in the
resolved Git metadata directory, not `/tmp`, which image boot services may clean.
Patch bytes and CRLF are preserved through JSON transport with UTF-8
`surrogateescape`; the official assembler still applies its own binary-hunk
filter. The guard normalizes the pinned legacy Docker ENV exports in the
entryscript prelude while retaining strict setup failure handling and unchanged
official test/parser commands.

For selected Pro runs, `--pro-runtime-port 34122` sets the same control port for
actor/verifier templates, readiness and requests. Defaults remain unchanged for
other runs. Template names include the port; base-image receipts must also match
the run's `runtime_port`, so old port3000 receipts cannot be silently reused.
When grading an old port3000 snapshot, add `--pro-collector-runtime-port 3000`:
only patch collection uses the legacy connection, while official tests run in
the new-port verifier. Record this compatibility choice in grading metadata;
do not edit old snapshots or change the application's own test ports.
`python -m swebench_pro.batch summarize --out <batch>` reads current results.
A `STOP_REQUEST.json` file stops admission and interrupts active workers.

The batch maps the pinned official SWE-agent tool config to450s per command,
1800s aggregate command execution, and three consecutive command timeouts.
Actor wall time is3600s, matching the official Modal deployment lifetime;
verifier timeout is3600s. These are separate budgets, not one generic timeout.
The official wrapper's three-call debugging limit is not used. Claude Code's
native model/sampling/tool protocol differs from SWE-agent; the manifest lists
the mapping and adaptations instead of claiming identical scaffold settings.

```bash
python -m swebench_pro \
  --pro-repo ../SWE-bench_Pro-os \
  --instance instance_ansible__ansible-f327e65d11bb905ed9f15996024f857a95592629-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5 \
  --slot claude-code --model us.anthropic.claude-sonnet-4-6 \
  --rounds 0 --timeout 10800 -o runs/pro-single
```

This is equivalent to `python -m swebench.fork_eval --benchmark swebench-pro ...`.
Supply comma-separated instance IDs or `--instance all` for sequential evaluation
of the loaded cohort. Concurrent workers must use separate output directories.
No model evaluation starts merely by importing the adapter or loading tasks.

To enable branching, replace `--rounds 0` with, for example,
`--rounds 2 --branches 4,3 --analyst-model <model>`. Existing branch count modes,
`--parent-from`, exact native conversation cuts and explicit
`--fork-full-conversation` retain their shared meanings. Only fresh parents use
the preparation hook; branches restore the selected snapshot without resetting
it to the base commit. Pro grading errors stop further branching for that task.

To grade recorded attempts again without actor calls:

```bash
python -m swebench_pro --pro-repo ../SWE-bench_Pro-os \
  --regrade -o runs/pro-single
```

The shared regrade loop evaluates attempts in order and stops at the first
resolved attempt per task. It does not re-run historical branch decisions.

## Environment and patch contract

The prompt includes `problem_statement`, `requirements` and `interface`.
Gold patches and verifier metadata are not included in the actor prompt.
The repository is `/app`. Fresh parents start from a prepared snapshot at the
dataset base commit; ignored/preinstalled dependencies are retained.

Both committed and uncommitted changes, including new files, are exported.
Collection uses a temporary Git index, leaving the actor's saved index intact.
Paths already untracked in the pristine environment are excluded. As in the
pinned upstream assembler, binary diff sections are stripped from the patch
that is tested; the original `model.patch` and effective `patch.diff` are both
retained. This is not DeepSWE's committed-work-only contract.

Default resources are 4 CPUs / 16 GiB, with a 3600-second verifier command
timeout. Adjust `--pro-cpus`, `--pro-memory-mb`, `--pro-verifier-timeout` and
the actor's `--timeout` independently. Network is enabled, matching the official
Pro default; some scripts install dependencies or start services. Explicit
`--pro-block-network` supplies a deny default for both actor and verifier.
Independent `--agent-network allow|deny` and `--verifier-network allow|deny`
override that default for their phase only. For example, `--agent-network deny
--verifier-network allow` keeps the actor offline but permits verifier dependency
installation. The verifier policy also applies to patch collectors. Host
inference/MCP and image downloads are unaffected; Codex-native tools remain
controlled separately. The runtime inherits the image's environment, including PATH.

`swebench_pro.batch launch` accepts the same two flags and freezes their values
in the new manifest as `agent_network` and `verifier_network`. Workers forward
the frozen settings, with per-task values taking precedence when supplied.
Existing `controller`/`summarize` commands reject network flag overrides rather
than changing old runs silently. New summary files record both phase policies;
omitted settings retain benchmark/backend defaults. No existing manifest or
historical result is migrated by adding this interface.

## Grading and evidence

The adapter restores the actor snapshot to collect its diff, and starts a
separate pristine VM to verify it. It uses the pinned official
`assemble_workspace_files()` to generate the patch, runner, parser and original
entryscript. The test runner and parser are uploaded unchanged. A guarded
entryscript checks setup/patch failures but still invokes the parser after a
nonzero test exit. Empty patches run the unchanged baseline tests.

Acceptance matches the official predicate: every F2P and P2P test name must
appear with `PASSED` status in the parser output. Missing test names count as
failed. Missing/malformed parser output, transport timeouts and setup failures
are grading errors; a patch that cannot apply is an unresolved attempt.
This preserves the official tests and acceptance rule while distinguishing
infrastructure failures from measured failures. The microVM transport/resource
configuration differs from the upstream Docker/Modal launch, so dataset-wide
equivalence requires the oracle/no-op gate, not just unit tests.

Each task keeps `parent.jsonl` and branch journals plus preparation metadata.
Each verdict gets an independent `<attempt>.verifier/verify-*/` directory with
the original/effective patches, exact scripts, parsed output when available,
`metadata.json`, `grade.json`, and `verifier-logs.tar.gz`. Collection failures
retain whatever evidence was available. Temporary VMs are torn down in finally
blocks. Snapshots remain available for replay under the existing retention policy.

`summary.json` records per-attempt `grading_error` and archive errors, and adds
`expected_tasks`, `pending_task_ids`, `grading_complete`, `grading_error_ids`, `resolved_lower_bound` and
`final_resolved_rate`. The final rate is null when grading is incomplete;
pending tasks remain in the selected-cohort denominator. The score describes
the explicitly selected cohort, not necessarily all 731 tasks.
The shared grading selector uses the last successful snapshot and records any
later calls/checkpoint issues in `grading_snapshot`; it may precede the last
actor operation if a later capture failed.

## Validate before a batch

### Isolated infrastructure retries

New `swebench_pro.batch` manifests select `failure_policy: isolated`, with at
most two additional infrastructure attempts and 30/60-second backoffs. Existing
frozen manifests retain their previous policy; do not modify old run inputs to
silently enable this behavior. This queue still runs single parent rollouts,
not an analyst/branch planner.

Workers write only `shard-NNN/attempts/attempt-NNN/`; the controller owns the
canonical `shard-NNN/worker.json` and durable `job.json`. Each request has a
unique attempt ID and hash. A stale result cannot replace a newer result.
Canonical `journal_path` selects the accepted attempt; `--parent-from` respects
that publication rather than taking the first historical attempt it finds.

Valid unresolved grades, including valid empty parser test lists, are terminal.
Grader infrastructure faults reuse the completed actor journal/snapshot without
model calls. Actor interruption requires an intact snapshot and matching native
conversation prefix; recovery carries spent actor/tool budgets. With no safe
pair, the job is held rather than restarted from scratch. Uncertain shell calls
are never automatically replayed against the same live sandbox.

Redispatch requires the old worker process group to have exited, no uncertain or
in-flight VM creation, and an inventory check confirming its owned VMs are gone.
Unknown creation outcomes or surviving child processes quarantine that job;
they are not killed or retried blindly. A quarantined job reserves a worker slot
until explicitly reconciled, including ambiguous controller dispatch crashes.
Normal errors never write a global STOP.
A backend outage, unprovable paginated inventory, or less than 80 GiB free disk
pauses new admission. Explicit STOP/SIGINT/SIGTERM drains healthy workers.
All-canary readiness failure also pauses admission. Restarting the controller
adopts identifiable old workers; ambiguous dispatch crash windows quarantine.

`retry_history` retains each attempt and `usage` sums reported actor usage,
excluding grading-only copies. `reported_usage_complete: false` means at least
one actor attempt has no reported usage, so the sum is not a complete billing
measurement. Retry attempts do not increase the task denominator. This is a
code-level safety mechanism, not live validation of old-snapshot port migration
or a full branching experiment.

### Tests and oracle gate

```bash
PYTHONPATH=.:sdk python3.12 -m pytest swebench_pro/tests swebench/tests deepswe/tests -q
python -m swebench_pro.gate \
  --pro-repo ../SWE-bench_Pro-os \
  --instance instance_ansible__ansible-f327e65d11bb905ed9f15996024f857a95592629-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5 \
  -o runs/pro-gate
```

The gate makes an oracle snapshot with the dataset's reference patch and a
no-op snapshot, then calls the same collection/verifier path as a real attempt.
It requires oracle success and no-op failure, with no grading/archive errors.
It makes no model calls. Gate outputs are unique per invocation, avoiding stale
verdict reuse; use `--mode oracle` or `--mode nop` to inspect either path.

Validation on 2026-09-09 loaded all 731 rows and their assets, and assembled and
syntax-checked one official verifier per repository (11 repositories). A live
Ansible microVM gate passed: gold patch 4/4 F2P and 171/171 P2P; no-op 0/4 F2P
and 171/171 P2P, with logs retained and no VM leaks. This is a single-task live
check, not a 731-task parity claim or a model performance measurement.
