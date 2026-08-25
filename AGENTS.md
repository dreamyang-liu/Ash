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

### A SWE-bench instance on microVMs, checkpointing every step

```yaml
# swebench/configs/mine.yaml
extends: bedrock-sonnet46
execution:
  backend: microvm
  microvm:
    server_url: http://127.0.0.1:18000
    api_key: <AgentENV key>
    runtime_bin: /tmp/ash-runtime     # set this and templates build themselves
  checkpoints:
    enabled: true
    mode: disk_only        # or full, for milestones that must resume live
    trigger: mutation      # or every_step
```

```bash
python -m swebench -c swebench/configs/mine.yaml -i django__django-11848
```

With `runtime_bin` set, each benchmark image is turned into a microVM template
on demand — cold-start the image, upload the runtime and ripgrep through the
backend's file service, snapshot, then build a template declaring a startup
command. Templates are content-addressed over (image, runtime hash, port), so a
batch builds each image once (~30–60 s) and every later run reuses it (~0.3 s).

### A SWE-Marathon task

Marathon tasks are directories, not dataset rows, and each brings its own
Dockerfile and verifier:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/abundant-ai/swe-marathon /tmp/swe-marathon
cd /tmp/swe-marathon && git sparse-checkout set tasks

# one task
python -m swebench --harness marathon -c swebench/configs/marathon-full.yaml \
    --task-dir /tmp/swe-marathon/tasks/wasm-simd

# every task in the checkout
python -m swebench --harness marathon -c swebench/configs/marathon-full.yaml \
    --task-dir /tmp/swe-marathon
```

The harness builds the task's image locally (several bake encrypted
verification assets in, so nothing is published to pull), pushes it to a
registry the backend can reach, and grades by running the task's own
`tests/test.sh` verbatim — its anti-cheat is part of the specification. Both the
binary reward and `partial_score` are recorded, because nearly every attempt on
these tasks scores zero and 7-of-43 has to be distinguishable from none.

A local registry is the simplest reachable target:

```bash
docker run -d --name local-registry -p 5000:5000 registry:2
# the server runs as root, so let its regctl use plain HTTP for it:
sudo mkdir -p /root/.regctl
echo '{"hosts":{"localhost:5000":{"tls":"disabled"}}}' | sudo tee /root/.regctl/config.json
```

### Resuming an interrupted run

Every checkpoint writes the trajectory next to itself, so an interrupted run
leaves both the snapshots and the map from step to snapshot:

```bash
python -m swebench --harness marathon -c swebench/configs/marathon-full.yaml \
    --task-dir /tmp/swe-marathon/tasks/wasm-simd \
    --resume-from <snapshot-id>
```

The environment carries the work; the transcript does not, and the prompt says
so. To find the snapshot for a particular step, read the trajectory:

```python
from swebench.replay import (load_step_snapshots, messages_through_step,
                             replay_caveats, environment_mismatch)

step_snapshots = load_step_snapshots("results/.../trajectories/task.json")
snapshot = step_snapshots[156]              # environment as of step 156
prefix = messages_through_step(path, 156)   # transcript to re-feed
print(replay_caveats(path, 156))            # e.g. background processes were live
```

### Using a model behind a proxy

litellm reads **`ANTHROPIC_API_KEY`** and **`ANTHROPIC_API_BASE`** — *not* Claude
Code's `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL`, which is the mistake worth
knowing about. Either export those, or put them in the config:

```yaml
model:
  name: anthropic/deepseek-v4-flash    # `anthropic/` = speak the Messages API
  api_base: http://<proxy>
  api_key: sk-...                       # keep such a config out of git
execution:
  # litellm has no metadata for a proxy-served name, so state the window or the
  # guard assumes a conservative 200K and folds far too early.
  context_window_tokens: 1000000
  context_budget_fraction: 0.60
```

A large window is what the model *accepts*, not what to fill: on the proxy
measured here, latency grew ~13 s per 100K input tokens (29 s at 200K, 137 s at
1M), so every token kept is paid for again on every later step. Note also that
a proxy which does not report cost makes `cost_limit` inert — only `step_limit`
and the wall-clock timeout bind.

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
- **There is no GC yet.** Deleting a snapshot does not reclaim its layers, and
  squash adds a copy until the originals become unreferenced. A 500-step
  disk-only chain is ~1.4 GB; the store on this machine grew to 85 GB across a
  few days of experiments. Reclaiming means stopping the server and removing
  `$AENV_HOME/snapshot-store` (task images then reconvert, minutes each).
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
