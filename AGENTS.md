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

## How the pieces depend on each other

**Ash → AgentENV** over HTTP. Ash never manages Firecracker itself; it asks for
sandboxes and snapshots through the API (`swebench/backends.py` builds the pool
from config, so no call site names a backend). Ash is what decides *when* to
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
