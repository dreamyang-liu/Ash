# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

> **Ash — Agent Sandbox Hive** is sandbox infrastructure for LLM agents and RL
> training: isolated environments behind a small fixed tool protocol, with a
> per-step snapshot of every one of them so any step of any run can be resumed
> or **branched**.
>
> This file is task-first: how to run SWE-bench, how to branch a failed rollout,
> and the facts you need to not get burned doing it. For the internals of each
> layer see [`harness/README.md`](harness/README.md); for the three-repo stack
> (Ash + AgentENV + patched firecracker) see [`AGENTS.md`](AGENTS.md).

## The one-paragraph model

An **orchestrator** owns a run: it creates the microVM, serves it to the agent
over MCP, snapshots the filesystem after every step that could have changed it,
and tears it down. Each snapshot is paired with the agent's conversation
reference, so a `(step → snapshot + session)` pair makes that step *resumable*.
Give a later run that snapshot as its image and the parent's session as its
conversation, and you have a **branch**: same history, divergent future. The
agent is a black box behind a slot interface (`claude-code`, `codex`,
`opencode`), so none of this is specific to one vendor's SDK.

---

## Prerequisites

```bash
pip install ./sdk                      # the ash-sandbox client
pip install litellm pyyaml datasets    # eval deps
cd runtime && go build -o ash-runtime . && cd ..   # the in-sandbox binary
```

**A snapshot backend is mandatory for anything in this file.** Docker cannot
snapshot; `--backend microvm` (AgentENV/Firecracker) is the one that can. It
reads:

```bash
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_API_KEY=...
```

Model credentials, by slot:

| Slot | How it reaches a model |
|---|---|
| `claude-code` | `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_USE_BEDROCK=1` + `AWS_BEARER_TOKEN_BEDROCK` |
| `codex` | `model_provider = "amazon-bedrock"` in `~/.codex/config.toml` + `AWS_BEARER_TOKEN_BEDROCK` (GPT-5.6 is hosted on Bedrock; no login needed). **Top-level keys must precede the first `[section]`** or TOML puts them inside it and codex silently falls back to api.openai.com. |
| `opencode` | its own config, or AWS env vars for its Bedrock provider |

---

## Run SWE-bench, grade it, branch what fails

One command does the whole loop — attempt, grade, and on failure let an analyst
model read the trajectory and fan out branches:

```bash
python -m swebench.fork_eval \
    --instance sympy__sympy-13091 \
    --slot codex --model openai.gpt-5.6-luna \
    --analyst-model openai.gpt-5.6-luna \
    --rounds 2 --branches 3 \
    -o runs/fork-eval
```

What happens, in order (`swebench/fork_eval.py`):

1. **Attempt.** One orchestrator run on a fresh microVM. Every step that could
   have mutated the filesystem leaves a `(snapshot, session)` pair in the
   journal; read-only steps map to the previous snapshot for free, so *every*
   step is a valid branch point without paying for a snapshot per step.
2. **Grade.** The **last snapshot is restored into a NEW microVM** and the tests
   run there. Grading in a restored sandbox rather than the live one proves the
   snapshot carries the work, and lets grading happen after the agent is gone.
   The dataset's `test_patch` is applied first — the tests the image ships
   predate the fix and the graded test may not exist in it at all.
3. **Branch on failure.** The analyst gets the journal rendered as one line per
   step **plus the grading verdict** (which test failed, the patch, the output).
   That verdict is the point: on a benchmark the agent usually believes it
   succeeded, so "what went wrong" is only answerable from outside. It returns a
   branch step and K diverse directions; each becomes another run whose sandbox
   image *is* that step's snapshot and whose conversation forks the parent's.
4. **Repeat** from the best-scoring attempt, up to `--rounds`.

Scores: `3` resolved, `2` target tests pass but something regressed, `1` a patch
exists, `0` nothing. `summary.json` and every journal land in `-o`.

Measured on `sympy__sympy-13091` (reference patch 522 lines): the parent changed
only `Basic.__eq__` and failed; the analyst named the mechanism (numeric classes
in `sympy/core/numbers.py` override comparison, so unknown-type comparisons never
reach `Basic.__eq__`), branched at step 5, and **2 of 3 branches came back
resolved** with all 89 `PASS_TO_PASS` regressions passing.

### Useful knobs

- `--analyst-tokens 100000` — how much trajectory the analyst sees. The budget is
  spent per-line first: tool *results* get 6000 characters, kept **head and
  tail**, because a test run's verdict is at the end.
- `--branches`, `--rounds` — width and depth. Branches within a round are
  independent; each gets its own sandbox off the same snapshot, so siblings
  cannot contaminate each other.
- `--slot claude-code|codex|opencode` — all three verified end to end on this
  loop.

---

## Run one agent by hand

When you want a single run rather than an eval loop:

```bash
python -m harness run --slot codex \
    --sandbox-image python:3.11-slim \
    --backend microvm --runtime-bin runtime/ash-runtime \
    --transport http --tools default \
    --cwd /tmp --journal runs/one.jsonl \
    "fix the failing test in /testbed"
```

The orchestrator owns the sandbox either way; `--transport` only decides how the
agent talks to it (`http` = an MCP server inside this process, `stdio` = a
subprocess). Both checkpoint at the tool boundary. Add `--gateway --routes
routes.json --budget-usd 5` to route the model traffic through the inference
gateway (model swap, real accounting, enforced budget).

Then inspect and branch by hand:

```bash
python -m harness show      runs/one.jsonl        # event histogram
python -m harness fork-plan runs/one.jsonl --step 7   # the pair at step 7
python -m harness atif      runs/one.jsonl -o t.json  # ATIF v1.8 export
python -m harness reap                            # reclaim leaked sandboxes
```

`python -m harness.demo_fork --slot opencode --image python:3.11-slim
--prompt … --branch-at 2 --direction "try X" --direction "try Y"` is the
minimal fork demo without any benchmark attached.

---

## What is deliberately absent

`python -m swebench` (the batch runner), the four `harnesses/`, this
repository's own litellm agent loop, SWE-Marathon, the RL rollout server and
step-replay have all been **deleted**. Each kept a second copy of something the
orchestrator now does properly — sandbox lifecycle, per-step checkpoints, agent
drivers — and none was in use once `fork_eval` existed.

`fork_eval` runs **one instance at a time** and takes its arguments on the
command line — the 24 per-model YAML configs went too, since the batch runner was
their only reader. Batch (and the rollout server) come back on top of the
orchestrator when they are needed, rather than being carried along broken.
`results/` from those older runs is still on disk: generated data, untracked.

---

## Facts that will burn you

Each of these cost real debugging time, most of it in a single session.

**Grading lies quietly.** A grader must be validated against inputs that *must*
fail AND that *must* pass — the must-pass case is what caught django dying of a
UnicodeEncodeError under `--verbosity 2`. Seven separate defects hid behind
confident numbers here: sympy's bare test names, `bin/test -k` exiting 0 on no
match, `sympy.test(...)` truthy on zero-match, graded tests coming from
`test_patch` not the image, the dataset splitting parametrised ids on internal
commas, django docstrings harvested as test ids (165/231 django instances —
grade by parsing `--verbosity 2` output, never by labels), and agent test-file
edits graded as fatal instead of reverted per the public convention. The first
full 500 reported 43.0%; the same snapshots re-graded honestly are 72.6%.

**Every agent silently bypasses a gateway** unless its own provider-direct mode
is disabled, and each does it differently: `claude-code` needs
`CLAUDE_CODE_USE_BEDROCK/VERTEX=0`, `opencode` ignores `ANTHROPIC_BASE_URL`
entirely and needs its config file written plus `AWS_*` cleared, `codex` needs a
custom `model_providers` entry. The slots handle this; the lesson is that
"traffic will go through X" must be *verified*, never assumed.

**A budget without prices cannot bind.** Providers report tokens, not dollars.
A gateway route needs `pricing` or `budget_usd` never fires — the gateway
journals `budget_unenforceable` once rather than pretending. And refuse over
budget with a **non-retryable 400**: a 429 tells every SDK to back off and retry
something that can never succeed.

**Disk-only snapshots cold-boot.** Processes do not survive them, which is why
the default tool panel withholds `background` (and therefore `process`): a
replay of a step taken while a background process ran diverges. A microVM
template must declare a startup command that launches `ash-runtime`, or a
restored sandbox has no runtime and every tool call 502s. `microvm.runtime_bin`
makes `SandboxSession.create` build such a template per image on demand.

**A run that cannot say what it ran against is not reproducible.** Image names
are mutable tags, so every trajectory carries `base_image` (digest-pinned),
`base_ref`, `base_commit` and the sandbox id.

**A journal under `/tmp` schedules its own destruction.** It is the run's only
record — snapshot ids, every step, the grading evidence. A 32-instance batch's
journals lived in `/tmp` when the host rebooted mid-regrade; hours of agent time
now exist only as prose. `harness run` and `fork_eval` now **refuse** volatile
output paths (`--volatile-ok` to override); put runs under `runs/`.

**The `cwd` you give an agent is not the sandbox.** It is where the CLI process
runs on the host; keep it neutral (`/tmp`). Pointing it at this repository once
handed an agent this repo's own `.claude/` skills mid-task.

---

## Repository layout

```
runtime/          Go binary inside every sandbox: 8 tools, one JSON-RPC protocol
  tools/          the 8 tool implementations
  schema/tools.json   `--dump-schema` output, checked in; panels compile against it
sdk/ash_sandbox/  async Python client: Sandbox, DockerPool, MicroVMPool, pool.py
harness/          agent runtime — see harness/README.md
  orchestrator/   run.py: the shape of one run (sandbox, transport, teardown)
  execution/      the execution plane: session, panel, server, interceptors
  slots/          per-agent drivers; normalize/ maps native events to journal ones
  gateway/        inference gateway: model swap, wire tap, enforced budget
  core/journal.py append-only JSONL, the canonical state
  rollback.py     checkpoint pairing and fork plans
  tool_panels/    default (shell+text_editor), full, bash_only, no_web
swebench/         the eval layer: what counts as an answer. Four files.
  fork_eval.py    run -> grade -> branch on failure (the loop above)
  dataset.py      instances, test commands, the bare-name runner
  patch.py        what belongs in a diff
k8s-scaffold/     Go control plane + gateway for fleet-scale sandboxes
docs/             generated diagrams (gen_*.py) — geometry-validated
results/          benchmark output. Generated data.
```

### The layering rule

`harness/` is the agent runtime and **never imports `swebench/`** (an AST test
enforces it). "What counts as the answer" — a patch, a grade — lives only in
`swebench/`. Granularity decides placement: per tool call → an interceptor, per
step → a checkpoint, per run → the orchestrator.

State belonging to one run must not live at module level. Three bugs came from
that (tool panel, routing table, custom-tool registry were each process-wide);
a session owns its registry, an agent owns its panel.

---

## The tool protocol

`ash-runtime` serves **8 tools** over JSON-RPC. Changing this set is a breaking
change.

| Tool | Purpose |
|---|---|
| `shell` | Run a command. `background: true` returns a pid. |
| `process` | Read output of / kill a background process. |
| `text_editor` | `view` / `write` / `str_replace` / `insert`. |
| `grep_files` | Ripgrep search (pattern, glob, limit). |
| `web_fetch` | Fetch a URL as html / text / markdown. |
| `web_search` | Multi-engine search. |
| `artifact` | Fetch + verify a binary; backs manifest-defined custom tools. |
| `wait_for_events` | Observe async facts (process exits). Opt-in. |

Declared once, in Go; everything downstream is derived:

```
runtime/tools/*.go  Schema()      ← authoritative
        │  ash-runtime --dump-schema
        ▼
runtime/schema/tools.json         ← checked in, so a panel compiles with no sandbox
        │  + harness/tool_panels/*.yaml   (what to offer, and how)
        ▼
the panel a model sees            ← compiled; docs/TOOL_PANEL.md
```

Adding a tool: edit the Go, regenerate `tools.json`, name it in a panel manifest
if a model should see it. A test fails on a stale schema, another on a panel
offering a parameter the runtime rejects — the hand-written panel that replaced
had drifted on four of seven tools.

Panels: `default` (shell + text_editor — enough to run commands and read/write
files), `full` (all seven model-facing tools; what the SWE-bench configs name),
`bash_only`, `no_web`. Not every runtime tool is offered: `artifact` is machinery
the SDK uses, so a panel listing it would hand the model a download primitive.

---

## Conventions

- **Go**: `gofmt`/`goimports` clean, wrap errors with `%w`, `go vet ./...` before
  committing. Go 1.22. No Go unit tests — correctness comes from the CI
  integration tests, which drive the binary over stdio, HTTP and MCP. Mirror
  those checks after touching tools.
- **Python**: PEP 8, annotations on signatures, prefer immutable dataclasses; the
  SDK is fully async. Tests: `python -m pytest harness/tests swebench/tests -q`
  (819 passing), plus `python contracts/ci_check.py` (117 checks that upstream
  CLI flags and SDK APIs we depend on still exist).
- **Don't hand-edit `results/`.**
- Diagrams are generated: edit `docs/gen_*.py` and re-run it. The generator
  refuses to write a file whose boxes overlap or whose text escapes its box.
- The `ash` skill for driving sandboxes lives at `.claude/skills/ash/`.

## License

Runtime/repo: MIT. Python SDK (`sdk/`): Apache-2.0.
