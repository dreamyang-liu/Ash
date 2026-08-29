# SWE-bench evaluation

This layer knows two things the rest of the repository deliberately does not:
that the answer to a SWE-bench instance is a **patch**, and that a patch is right
when the instance's `FAIL_TO_PASS` tests pass without breaking `PASS_TO_PASS`.
Everything about *running* an agent lives in [`harness/`](../harness/README.md).

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
    --rounds 2 --branches 3 \
    -o runs/fork-eval
```

One instance at a time. The loop is attempt → grade → branch on failure; see
[`../CLAUDE.md`](../CLAUDE.md) for what each step does and why.

## Files

```
swebench/
├── fork_eval.py     the loop: attempt, grade from a snapshot, branch on failure
├── dataset.py       instances, test commands, the runner bare test ids need
├── patch.py         what belongs in a diff: staged + untracked-minus-baseline
├── models.py        shim -> harness.core.result
├── backends.py      shim -> harness.execution.backends
├── templates.py     shim -> harness.execution.templates
├── mcp_server.py    shim -> harness.execution.server
└── style.py         terminal colours
```

The four shims exist because the modules moved into the execution plane during the
layering inversion, and documented import paths should keep working.

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

## Not here any more

This repository's own litellm agent loop, the four `harnesses/` topologies,
SWE-Marathon, the batch runner (`python -m swebench`), the RL rollout server and
step-replay were deleted. Each held a second copy of something the orchestrator
does properly now — sandbox lifecycle, per-step checkpoints, agent drivers — and
none was in use once `fork_eval` existed. Batch and rollout return on top of the
orchestrator when they are needed, rather than being carried along broken.
