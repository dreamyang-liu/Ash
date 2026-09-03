# The checkpointed-rollout stack

Three repositories that together let an agent rollout be **snapshotted at every
step and resumed or branched from any of them** — the capability an RL loop
needs to retry a trajectory from where it went wrong instead of from the start.

```
Ash  ─────────────── the agent, and the policy about when to checkpoint
  │                  (Python: SDK, agent loop, SWE-bench + SWE-Marathon harnesses)
  │  HTTP
AgentENV ─────────── the sandboxes and the snapshot storage
  │                  (Rust: Firecracker microVMs, OverlayBD layer chains, E2B-ish API)
  │  downloads a binary at startup
firecracker (fork) ── one added API the incremental memory snapshots need
                     (patched build, published as a release asset)
```

Read this file to understand how the three fit together and how to run
anything. Each repository's own `CLAUDE.md` covers its internals.

| Repository | Branch | Role |
|---|---|---|
| [dreamyang-liu/Ash](https://github.com/dreamyang-liu/Ash) | `feat/per-step-checkpoints` | Agent loop, tool protocol, harnesses, checkpoint policy, replay |
| [dreamyang-liu/AgentENV](https://github.com/dreamyang-liu/AgentENV) | `feat/incremental-memory-layers` | microVM lifecycle, snapshots, layer-chain compaction, squash |
| [dreamyang-liu/firecracker](https://github.com/dreamyang-liu/firecracker) | `feat/dirty-memory-ranges-reset` | `PUT /vm/dirty-memory-ranges/reset`, so memory snapshots can be incremental |

Local checkouts on this machine: `~/projects/LBP/{Ash,AgentENV,firecracker-fork}`.

> This file is also `CLAUDE.md` (a symlink): one document serves every
> agent and every human. Task-first sections come first; the stack's
> internals follow.

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

## Running things

Everything below assumes the env file this machine keeps at
`~/aenv-bench/env.sh` (AENV_SERVER_URL + AENV_API_KEY, chmod 600 — the key is
whatever the server was STARTED with, so a restart may mint a new one) plus:

```bash
source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK=...          # claude-code & the analyst
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2
export PYTHONPATH=.:sdk                      # from the Ash checkout
```

### Recovering the server after a reboot

A reboot (or a kernel update) is the common way this machine breaks:

```bash
# 1. ublk module gone? Rebuild against the RUNNING kernel:
cd ~/projects/LBP/artifacts/ublk-build
make -C /usr/src/kernels/$(uname -r) M=$PWD modules
sudo cp ublk_drv.ko /lib/modules/$(uname -r)/extra/ && sudo depmod -a
sudo modprobe ublk_drv

# 2. start the bench instance (setsid so a later shell mishap cannot kill it)
sudo setsid env \
  AENV_CONFIG_PATH=/opt/aenv-bench/config.toml \
  AENV_HOME_PATH=/opt/aenv-bench/home \
  AENV_RUNTIME_PATH=/opt/aenv-bench/run \
  AENV_DEPS_PATH=/var/lib/aenv/deps \
  API_ADDR=127.0.0.1:18000 \
  AENV_API_KEY="$AENV_API_KEY" RUST_LOG=info \
  /opt/aenv-bench/bin/server >> ~/aenv-bench/server.log 2>&1 < /dev/null &
# give it ~20s, then:
curl -s -H "X-API-Key: $AENV_API_KEY" http://127.0.0.1:18000/sandboxes
```

`/opt/aenv-bench/home` (image cache, snapshot store) survives reboots; the
templates and snapshots come back by themselves. Docker Hub credentials do
NOT flow from `docker login` to the server: its regctl needs its own
`sudo regctl registry login docker.io` (see "Things that will bite you").

### One SWE-bench instance, with branching on failure

```bash
python -m swebench.fork_eval \
    --instance sympy__sympy-13091 \
    --slot claude-code --model us.anthropic.claude-sonnet-4-6 \
    --analyst-model us.anthropic.claude-sonnet-4-6 \
    --rounds 2 --branches 4,3 --timeout 1200 \
    -o runs/my-run
```

`--rounds 0` = single attempt, no branching. `--branches 4,3` = width per
round. `--instance` takes a comma list; each instance gets its own directory
under `-o`, and `summary.json` is rewritten after every instance. Volatile
output paths (/tmp and friends) are refused — journals are the run's only
record.

### Full-dataset batches: pre-pull, then shard

Pulling images and running agents consume different scarce resources
(Docker Hub quota vs agent time); mixing them cost one batch 49 instances.
`scripts/run_v500_when_ready.sh` is the whole pattern: resumable per-image
pre-pull (`scripts/prepull_images.py`, 8 workers), retry rounds across
rate-limit windows, a refuses-to-launch check if too many images are missing,
then 8 sharded `fork_eval` workers. To run a fresh batch:

```bash
# shard ids round-robin (mixes heavy repos across workers)
python3 - <<'EOF'
from swebench.dataset import load_instances
import pathlib
ids = sorted(i['instance_id'] for i in load_instances('verified'))
out = pathlib.Path('runs/mybatch'); out.mkdir(parents=True, exist_ok=True)
for k in range(8):
    (out / f'shard-{k}.txt').write_text(','.join(ids[k::8]))
EOF

for k in 0 1 2 3 4 5 6 7; do
  setsid nohup python3.12 -u -m swebench.fork_eval \
    --instance "$(cat runs/mybatch/shard-$k.txt)" --slot claude-code \
    --model us.anthropic.claude-sonnet-4-6 \
    --analyst-model us.anthropic.claude-sonnet-4-6 \
    --rounds 2 --branches 4,3 --timeout 1200 \
    -o runs/mybatch/shard-$k > runs/mybatch/shard-$k.log 2>&1 < /dev/null &
  sleep 3
done
```

8 workers is the proven throughput (24 courts Bedrock throttling). Monitor:

```bash
python3 -c "
import json, glob
d = ok = 0
for f in glob.glob('runs/mybatch/shard-*/summary.json'):
    for i in json.load(open(f))['instances']:
        d += 1; ok += i['resolved']
print(f'{ok}/{d} resolved')"
```

### Re-grading without re-running

A grader fix applies to finished runs — every attempt left a snapshot:

```bash
python -m swebench.fork_eval --regrade -o runs/mybatch/shard-0 \
    --slot claude-code --model us.anthropic.claude-sonnet-4-6
```

Grades in attempt order, stops at the first resolved one (a correct grader
would have stopped the loop there too). ~150 verdicts flipped across three
regrade waves in this repo's history for ~20 minutes of compute total.

### Cleaning up

Nothing deletes snapshots automatically — runs end, history stays, and
`harness reap` reclaims only compute (leaked sandboxes), never snapshots.
Deletion takes a deliberate act, twice over:

```bash
python -m harness reap                     # leaked sandboxes (safe, habitual)
python -m harness sweep runs/old/*/parent.jsonl        # dry-run: what would die
python -m harness sweep runs/old/*/parent.jsonl --yes  # delete + journal marker
sudo python3 scripts/gc_layers.py --store /opt/aenv-bench/home/snapshot-store
sudo python3 scripts/gc_layers.py --store ... --yes    # reclaim orphaned layers
```

`sweep` deletes the snapshots a NAMED journal records (the path is the
consent); `gc_layers` mark-and-sweeps `managed-layers/` against every catalog
record, keeps anything referenced or younger than 2h, and is dry-run by
default like everything else here.

## What a run produces

```
results/<run>/
  trajectories/<id>.json     messages (with tool_calls and tool_call_id),
                             info.checkpoints.step_snapshots, info.environment
  traces/<id>.events.jsonl   one JSON per tool call and checkpoint
  preds.json                 the grader's format
```

Snapshots live on the AgentENV side (`$AENV_HOME/snapshot-store/`), content
addressed, so the layers a base image contributes are stored once no matter how
many snapshots reference them.

Measured on a real 133-step marathon attempt: checkpoint overhead **0.1% of
wall clock** (p50 0.03 s per capture, ~40 KB per step), and 99.8% of the time
went to model inference. Infrastructure is not the bottleneck on this workload
and cannot become one.

## What is deliberately absent

`python -m swebench` (the batch runner), the four `harnesses/`, this
repository's own litellm agent loop, SWE-Marathon, the RL rollout server and
step-replay have all been **deleted**. Each kept a second copy of something the
orchestrator now does properly — sandbox lifecycle, per-step checkpoints, agent
drivers — and none was in use once `fork_eval` existed.

`fork_eval` takes its arguments on the command line — the 24 per-model YAML
configs went too, since the batch runner was their only reader. `--instance`
accepts a comma list (each instance gets its own output directory; summary.json
is rewritten after every one), `--branches 4,3` varies width per round,
`--rounds 0` disables branching, and `--regrade` re-runs the current grader over
a finished run's snapshots without spending agent time. For full-dataset scale,
shard the ids across parallel invocations (scripts/run_v500_when_ready.sh shows
the pattern: pre-pull images first, then launch). The rollout server comes back
on top of the orchestrator when it is needed.
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

## Things that will bite you

- **Disk-only snapshots cold-boot.** The runtime only comes back if the template
  declares a startup command; a template made with `aenv snapshot create`
  records none. `swap_sandbox` probes a replacement before adopting it, so a
  template without one costs a deeper chain rather than a dead episode.
- **Background processes do not survive a disk-only checkpoint.** Files
  (including `/tmp`, which is on the rootfs), git state and installed packages
  do; `/dev/shm`, sysctls, extra mounts and live processes do not. Steps that
  had background processes are flagged `live_background` and
  `replay_caveats()` reports them.
- **A capture that fails is recorded as `reason: "failed"`, not silently.** It
  used to be indistinguishable from a step that had nothing to capture, which
  is how a run with 68 records and one snapshot looked like a very clean
  workload.
- **Two runs of one task must not share snapshot names.** Aliases are unique
  per repository, so names include the run id; without that, the second run's
  every capture fails on a collision.
- **Deleting a snapshot reclaims no bytes by itself.** `DELETE /snapshots/{id}`
  (added on the `feat/snapshot-delete` branch) removes the record — the layers
  sit orphaned in `managed-layers/` because neither our AgentENV nor upstream
  has a layer GC. `scripts/gc_layers.py` is that GC: mark every digest any
  catalog record mentions, sweep unreferenced layers older than 2h. Measured:
  ~25k orphaned records ≈ only 15 GiB, because per-step deltas are small — the
  store's weight is template commits (~3.9 GB per unique rootfs, deduplicated
  by content) that live history genuinely references.
- **Ops footguns that each cost a session real time.** `pkill -f <pattern>`
  matches the shell issuing it when the pattern appears in the command line —
  bracket a character (`serve[r]`) or the kill takes you with it. A compound
  `cd X && ... &` backgrounds the `cd` too; later commands in the same shell
  run from the old directory. Workers sometimes hang AFTER writing their final
  summary (exit-path bug, unfixed) — check summaries, not process counts, and
  kill stragglers freely once summaries are complete. Journals under /tmp die
  on reboot; the entry points refuse volatile paths for exactly that reason.
- **A streaming response can go silent, and the request timeout will not save
  you.** `timeout=` bounds getting a response started; once the stream is open,
  iterating it blocks indefinitely if the provider stops sending — seen as a
  socket in `CLOSE-WAIT` with the process in `futex_wait`. Measured twice:
  2h48m of silence on one run, and 20 minutes on another *with* a 900s request
  timeout configured, zero retries either time. A separate watchdog now
  abandons a stream that produces nothing for 180s, and a stalled stream is
  retryable.
- **A completion can come back with nothing in it, and that is not the model
  giving up.** Traced on one proxy: extended thinking consumed the whole output
  budget, so the reply carried an empty `thinking` block and no `text` block —
  litellm turns that into `content: None`. Sending that empty assistant message
  back made the proxy substitute
  `[System: Empty message content sanitised to satisfy protocol]` for it, which
  reads like a text-only answer and tripped the loop's two-strikes finish rule:
  a run reported `completed` at step 57 over an unbuilt decoder. Empty
  completions are now retried in the LLM client (3 attempts, streaming judged
  after assembly), and if they persist the loop re-prompts rather than counting
  them toward finishing.
- **Killing a run leaks its sandbox.** `session.destroy()` runs in a `finally`
  that a `kill -9` skips; the sandbox then lives until its TTL. Delete it with
  `DELETE /sandboxes/<id>`.

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

## How the pieces depend on each other

**Ash → AgentENV** over HTTP. Ash never manages Firecracker itself; it asks for
sandboxes and snapshots through the API (`harness/execution/backends.py` builds
the pool from config, so no call site names a backend). Ash is what decides *when* to
checkpoint, when to re-board, when to squash — AgentENV only provides the
mechanisms.

**AgentENV → firecracker** by download. `config/deps_manifest.toml` pins:

```toml
[firecracker.kvm]
version = "1.15.1-patch-v2"
url = "https://github.com/dreamyang-liu/firecracker/releases/download/aenv-deps/firecracker-{version}-{arch}.tgz"
```

The server fetches that on first start. The patch adds one endpoint —
`PUT /vm/dirty-memory-ranges/reset`, valid only while a VM is paused — which
lets a capture reset the dirty-page baseline so the *next* memory snapshot
stores an interval instead of everything ever touched. Without it the stack
still works: memory captures fall back to cumulative, costing storage, not
correctness.

## Why it is built this way

Three decisions explain most of the design, and each was measured rather than
assumed:

**Disk-only snapshots for per-step checkpoints.** A full snapshot also stores
the VM's memory, which is charged for every page *touched* since the last
capture — read a file and its page cache counts. Measured on one SWE-Marathon
attempt: 37 MB per episode disk-only versus 2.2–4.7 GB full, and 0.03 s versus
0.2–1.3 s per capture. A replay restores the disk and re-feeds the transcript,
so memory is dead weight. Full snapshots keep one advantage — they resume with
processes alive (0.27 s vs 1.2 s), so they suit milestones, not every step.

**Only capture steps that could have changed something.** The runtime's shell
is stateless (each call is a fresh `sh -c`), so a step that only read leaves the
environment byte-identical and the previous snapshot already *is* its state. A
`MutationTracker` interceptor decides; `shell` always counts as mutating,
because a command's effect is not in its text. Measured savings vary by task
shape: 34% on SWE-bench, 32% on a Rust-editing marathon task, 0% on one that
used `cat > file` for everything.

**Re-board when the chain compacts.** A capture appends a layer; the server
compacts the chain when it crosses a configured budget. A running sandbox's own
layer stack is never compacted, so after a compaction every later capture would
re-merge the whole chain. Ash detects a compaction as *a layer count that
failed to grow* and continues on a sandbox started from that snapshot. Watching
for that rather than counting steps means the policy follows whatever trigger
the server is configured with.

## Setting it up

### 1. AgentENV server

Needs a Linux host with `/dev/kvm`, the `ublk_drv` module, and a one-time root
setup. Build and run:

```bash
cd ~/projects/LBP/AgentENV
make                      # build the workspace
sudo ./target/release/server --setup-host \
    --runtime-user "$USER" --runtime-group "$USER"   # once per machine
make start-server         # or run target/release/server directly
```

The bench instance used for all the measurements in this file runs on an
isolated config so it cannot disturb a production one:

```bash
sudo env \
  AENV_CONFIG_PATH=/opt/aenv-bench/config.toml \
  AENV_HOME_PATH=/opt/aenv-bench/home \
  AENV_RUNTIME_PATH=/opt/aenv-bench/run \
  AENV_DEPS_PATH=/var/lib/aenv/deps \
  API_ADDR=127.0.0.1:18000 \
  AENV_API_KEY=<key> RUST_LOG=info \
  /opt/aenv-bench/bin/server >> ~/aenv-bench/server.log 2>&1 &
```

Settings that matter to checkpointing, with the values this machine uses:

```toml
[snapshot]
repository_backend = "posix_fs"
max_stacked_layers = 200      # compact when the chain passes this many layers
max_chain_size_mib = 1024     # ...or this many MiB, whichever trips first (0 = off)

[memory_snapshot]             # full snapshots only
track_dirty_pages = true
incremental_layers = true     # needs the patched firecracker
compression_enabled = true    # zstd; free on text, useless on already-packed bytes
```

`max_chain_size_mib` bounds how long one compaction pauses a capture:
roughly chain bytes ÷ the snapshot store's write bandwidth (~125 MB/s on gp3
here, so ~8 s per GiB). Set it *above* what a single step writes — a budget
smaller than one step's writes makes every capture compact, and the guard in
Ash warns when it sees that.

### 2. ash-runtime inside the sandbox

Every sandbox runs `ash-runtime` (Go, serves the 8-tool protocol). For microVM
work you only need it on the host; the harness uploads it while building a
template:

```bash
cd ~/projects/LBP/Ash/runtime && go build -o /tmp/ash-runtime .
```

### 3. Python side

```bash
cd ~/projects/LBP/Ash
pip install ./sdk litellm pyyaml datasets
```

## Tests

```bash
cd ~/projects/LBP/Ash    && python -m pytest swebench/tests sdk/tests -q   # 516
cd ~/projects/LBP/AgentENV && make test-unit                                # 761
cd ~/projects/LBP/AgentENV && make fmt clippy
```

Several tests assert *wiring* rather than logic — that every harness passes its
backend config through, reports through the prediction builder, and mounts the
context guard. They exist because the recurring failure in this stack has not
been wrong code but working code nothing calls: a context guard written and
never mounted, a config key that never reached the harness, a CLI that rejected
a registered harness. When adding a capability, add the test that asserts
something consumes it.

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
