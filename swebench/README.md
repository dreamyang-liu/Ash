# SWE-bench evaluation

This layer knows two things the rest of the repository deliberately does not:
that the answer to a SWE-bench instance is a **patch**, and that a patch is right
when the instance's `FAIL_TO_PASS` tests pass without breaking `PASS_TO_PASS`.
Everything about *running* an agent lives in [`harness/`](../harness/README.md).

For the ScaleAI Pro public benchmark, use the separate
[`swebench_pro` adapter](../swebench_pro/README.md). It shares the trajectory and
branching loop, but uses Pro's official per-instance test scripts and parser.

## Quick start

```bash
pip install ./sdk pyyaml datasets
cd runtime && go build -o ash-runtime . && cd ..
export AENV_SERVER_URL=http://127.0.0.1:8000   # microvm: the only snapshot backend
export AENV_API_KEY=...
export AWS_BEARER_TOKEN_BEDROCK=...            # for codex on Bedrock

# no config files: fork_eval takes its arguments on the command line. The 24
# per-model YAMLs went with the batch runner that was their only reader.

python -m swebench.fork_eval \
    --instance sympy__sympy-13091 \
    --slot codex --model openai.gpt-5.6-luna \
    --rounds 2 --branches 3 --fork-full-conversation \
    -o runs/fork-eval
```

One instance at a time. The loop is attempt → grade → branch on failure; see
[`../CLAUDE.md`](../CLAUDE.md) for what each step does and why.

Branch counts have two modes:

```bash
--branch-count-mode adaptive --branches 4,3   # default: at most 4, then at most 3
--branch-count-mode fixed --branches 4,3      # require exactly 4, then exactly 3
```

Both let the reviewer choose each base/step/hint, including repeated positions.
Adaptive mode may return fewer branches; fixed mode rejects a plan with too few
or too many branches before launch. It does not fabricate hints to fill the list
or silently reduce the count. A later round still runs only if no prior attempt
resolved the task. Mode and count semantics are recorded in plans and summaries.
Normal mode requires an exact recorded snapshot/session pair and an available
Claude Code native conversation cut. The Codex example explicitly opts into
full-conversation context; there is no automatic full-conversation fallback.

## Sandbox network policy

Actor and verifier egress can be controlled independently on `swebench.fork_eval`:

```bash
--agent-network deny --verifier-network allow
```

Each flag accepts `allow` or `deny`. The actor setting applies to fresh attempts,
branches and restored actor sandboxes. The verifier setting applies to grading,
including restored patch collectors and verifier setup/test commands. With
`--regrade`, only the verifier runs; historical actor execution is unchanged.
`--parent-from` likewise does not retroactively alter the imported parent's policy.

Omitted flags preserve benchmark defaults: DeepSWE denies egress, while standard
SWE-bench and Pro leave it to the backend default. Legacy `--pro-block-network`
sets Pro's default to deny for both phases; an explicit per-phase flag overrides
that phase only. An override can differ from the benchmark's canonical protocol.

These settings reach AgentENV sandbox creation, not just the actor prompt. They
do not block host inference, MCP control traffic or host image downloads, and
do not enable Codex-native web search. Allowing Ash shell egress still permits
commands such as `curl`, `git fetch` and `pip install`. Provisioning shared base
templates is infrastructure setup, not a model-controlled actor execution.

`summary.json` records `network_requested` and `network_policy` per phase;
`backend-default` means no explicit flag was sent, not measured internet access.
Legacy `no_network` is null for differing phase policies; use `network_policy`
instead. Each `grading_snapshot` records its verifier network policy. Regrade
reports label actor policy `not-rerun` and leave the original summary intact.
Per-attempt network metadata labels an imported parent `recorded` rather than
claiming that its historical execution used the new actor setting.

## Files

SWE microVM templates activate the image's existing `testbed` conda environment
before starting the runtime. The generic `microvm.runtime_init` option is part
of template identity; changing initialization never silently reuses an older
template. Verifier commands also explicitly activate and check that environment,
including when grading an older snapshot. No packages are installed by this check.

The normal grading path retains separate complete stdout/stderr files, command
records and a `verifier-logs.tar.gz` archive under each attempt's `.verifier`
directory. Export happens before sandbox destruction, including on failures.
The sandbox lease covers both test phases and cleanup independently of the Actor
budget. Missing test environments, invalid runner invocations and timeouts are
grading errors, not ordinary failed assertions. Genuine assertion failures remain
eligible for branching.

Batch supervisors must distinguish the benchmark CLI's exit `1` (a completed
unresolved task) from a failed worker. `batch_status.classify_worker_exit` checks
the terminal worker record and cleanup evidence as well as the exit code.
Actor timeouts and ungradable individual attempts can be isolated without
stopping unrelated work; missing exact snapshots are still never graded from
an older prefix. Lost verifier archives and unproven cleanup remain safety stops.

Batch admission can use `swebench.baseline.inspect_baseline` to distinguish the
dataset source commit from a prepared image commit. Exact baselines and
direct-child preparations with unchanged content or only executable-bit additions
are accepted. Astropy's known setuptools pin is verified by byte-for-byte content
comparison rather than a per-task SHA whitelist. Sphinx's known dependency
constraints and pytest-report preparation are likewise checked against full
file transformations, with incidental py311 URL changes explicitly recorded.
Requests' existing build copies are accepted only after matching their tracked
source bytes. Structured probe stdout is decoded separately from retained stderr
warnings. Other content changes stay rejected, with evidence attached for review;
passing baseline admission does not replace positive/negative grading controls.

```
swebench/
├── fork_eval.py     the loop: attempt, grade from a snapshot, branch on failure
├── branching.py     per-branch naming and old/new plan metadata readers
├── dataset.py       instances, test commands, the runner bare test ids need
└── patch.py         what belongs in a diff: staged + untracked-minus-baseline
```

The re-export shims (models, backends, templates, mcp_server) and
style.py went once nothing imported them: a shim with no importer is a path for
code to drift back across a layer boundary. Import from `harness.core.result`,
`harness.execution.backends`, `harness.execution.templates` and
`harness.execution.server` directly.

## Grading, and why it is easy to get wrong

`grade_snapshot` restores the attempt's **last snapshot into a fresh microVM** and
runs the tests there. That is deliberate: it proves the snapshot carries the work,
and it lets grading happen after the agent's sandbox is gone.

Before running anything it applies the dataset's `test_patch` — the tests the
image ships predate the fix, so a graded test may assert the *old* behaviour, or
not exist at all. A `test_patch` that will not apply is reported as a grading
error rather than falling back to the stale copies: it means the agent's edits
collided with the graded tests themselves.

**Validate any change to this code against an input that must fail.** Grade a
snapshot from *before* the agent's edit; if it "passes", the grader is broken.
That check is what caught four separate defects here, each of which had been
reporting a confident wrong number:

- sympy reports **bare test-function names** (all 75 of its Verified instances).
  Handed to pytest as paths they collect nothing, so the run fails whatever the
  agent did. `build_batch_test_command` now refuses ids it cannot express.
- `bin/test -k EXPR FILE` ignores FILE, matches nothing, and **exits 0** — and
  these images ship no pytest at all.
- `sympy.test(file, kw=[...])` also returns truthy for a zero-match run, and its
  progress output cannot be captured (it holds its own stream reference).
  `SYMPY_RUNNER` imports the module and **calls** the named functions instead,
  exiting `2` ("grader broken") when nothing ran.
- the graded tests come from `test_patch`, not from the image.

The full-500 run added three more, worth ~30 points of fiction (43.0% reported,
72.6% after re-grading the same snapshots — no agent was re-run):

- the dataset splits parametrised ids on the commas **inside** the parameters
  (65 instances), and harvested **django's docstring display strings** as test
  ids (165 of 231 django instances). A prose id handed to `runtests.py` as a
  label kills collection before any test runs. django is therefore graded the
  official way: run the covering modules at `--verbosity 2` and match ids
  against the **printed lines** (`parse_django_verbose`) — the only
  representation in which a docstring id exists. An id the output never names
  is a failure.
- `--verbosity 2` makes django print `Creating tables…`, and one ellipsis under
  the images' ascii locale killed the run with UnicodeEncodeError. Found because
  the MUST-PASS validation case failed — which is exactly what it is for.
- an agent's edits to graded test files are **reverted before grading** (the
  public-leaderboard convention: the model's patch excludes tests), recorded in
  `Grade.reverted_test_edits`, instead of being graded as a fatal collision.
  Five of five spot-checked "collisions" were real edits to graded tests — and
  the first one re-graded under the convention was simply resolved.

## DeepSWE verifier logs

Each verification exports `/logs/verifier/` before destroying its temporary VM,
including failed or timed-out runs. The attempt's `parent.verifier/verify-*/`
directory (or the corresponding branch name) contains `verifier-logs.tar.gz`,
captured `stdout.txt`, and `metadata.json` with exit code, timeout/running flags
when available, the grading error and any export failure. Archives include nested
reports and bypass runtime text-output truncation. Regrading creates a new
directory, preserving previous evidence.

`Grade`, batch summaries and regrade summaries expose `verifier_artifacts` and
`verifier_artifact_error`. Direct `deepswe.grade.verify_patch` or `grade_snapshot`
callers can supply `artifacts_dir`; their default is `runs/verifier-logs/`.
Export failure is reported and does not change the verifier's score or skip VM
cleanup. An unreachable VM or a killed host process can still prevent export.
This does not change tests, time limits or existing grading results.

## Not here any more

This repository's own litellm agent loop, the four `harnesses/` topologies,
SWE-Marathon, the batch runner (`python -m swebench`), the RL rollout server and
step-replay were deleted. Each held a second copy of something the orchestrator
does properly now — sandbox lifecycle, per-step checkpoints, agent drivers — and
none was in use once `fork_eval` existed. Batch and rollout return on top of the
orchestrator when they are needed, rather than being carried along broken.
