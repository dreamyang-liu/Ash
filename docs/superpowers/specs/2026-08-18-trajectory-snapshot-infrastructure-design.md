# Trajectory Snapshot Infrastructure — Design (draft)

**Date:** 2026-08-18
**Status:** Design discussion complete; section-by-section review NOT yet done.
This document captures the agreed conclusions of a long design conversation so
work can continue on another machine. Next step: review each section, then
write the implementation plan.

## 1. Problem

Give Ash **resumable, branchable trajectories**: while running a benchmark
eval (SWE-bench first), snapshot the environment at every state-mutating step
so that — mid-rollout or after the rollout finished, even after the container
is destroyed — we can **restore the state as of any step k and continue from
there, or fork K parallel branches**.

Motivation: exploring alternative ways to *expand a step in a trajectory*
(BPO's entropy-based branch points are one instance; we want infrastructure
that makes such experiments cheap). This is the shared substrate of three
papers:

- **Shepherd** (arXiv 2605.10913) — execution as a first-class reversible
  value; Git-like trace of typed effects; scope fork via overlayfs layer
  freeze (~72 ms, O(1)). Repo `shepherd-agents/shepherd` is early alpha;
  meta-agent APIs not shipped; used as *design reference, not dependency*.
- **TVCache** (arXiv 2602.10986) — tool-call prefix tree as source of truth;
  snapshots are just a cache; restore = nearest-ancestor snapshot + replay of
  the un-snapshotted suffix; `will_mutate_state()` annotations skip read-only
  calls.
- **BPO** (arXiv 2607.14171) — sandbox snapshot/restore as an RL training
  primitive; sibling-baseline advantages at branch points; got all its gains
  on *filesystem-only*, seconds-slow snapshots (Docker/overlayfs 1.9 s on
  SWE-bench) — so fs-only fidelity is sufficient for coding benchmarks.

Survey conclusion: nothing off-the-shelf couples **agent messages + environment
state per step**. Eval frameworks (LangGraph, OpenHands, Inspect) checkpoint
messages but not the world; sandbox providers (E2B, Morph, Daytona, AgentENV)
snapshot the world but know nothing about the agent loop. Ash already owns the
right seams: the runtime mediates every tool call, `events.jsonl`
(ToolTraceWriter) already records ordered `tool.started`/`tool.finished`
events with both agent-level and runtime-level `{name, args}`, and `Pool`
already has a capability pattern (`supports_fork()`).

## 2. Core model: trace is the source of truth, snapshots are a cache

- Every mutating tool call gets a trace event; a `snapshot.created` event
  (with `seq`, `turn_id`, `call_id`, message index, snapshot `ref`,
  `fidelity`, covered paths) binds the conversation position to an
  environment snapshot.
- **Restore(step k)** = find nearest ancestor snapshot ≤ k, instantiate it,
  replay the runtime-level commands of the suffix (k′..k]. Replay skips
  read-only tools (`readfile`, `grep`, …) — TVCache's correctness argument.
- Consequences: snapshots may be sparse or GC'd without losing reachability;
  fidelity gaps (out-of-tree writes, dead background processes) are repaired
  by replay; the events.jsonl format needs no change (runtime `{name,args}`
  is already recorded).

## 3. Snapshotter interface (mechanism-agnostic)

Extend `Pool` capabilities in `sdk/ash_sandbox/pool.py`:

```
supports_snapshot() -> bool
snapshot(sandbox, label) -> Snapshot(ref, fidelity, covered_paths)
rewind(sandbox, ref)                  # in-place, optional capability
spawn_from_snapshot(ref) -> Sandbox   # fork / offline resume
```

`fidelity` ∈ {"filesystem", "full"}. Trace records it so the replayer knows
whether background processes must be re-spawned.

### Backends (priority order — machine with KVM has ARRIVED)

1. **AgentENV** (`kvcache-ai/AgentEnv`) — **primary backend now that a bare
   metal machine is available.** Source audit (2026-08-18, commit 6988cb0)
   confirmed:
   - `POST /sandboxes/{id}/snapshots` creates persistent named snapshots;
     source VM pauses <100 ms and resumes in place; snapshots survive VM
     destruction; unlimited re-instantiation via `POST /sandboxes
     {"templateID": "<snapshot-id>"}`.
   - Incremental: memory = dirty-page delta layers (overlaybd/LSMT over
     ublk), disk = sealed upper layers; content-addressed dedup across a
     lineage; auto-compaction past 32 layers (solves the thousands-of-steps
     problem structurally).
   - Fork: source keeps running, N children boot in parallel <50 ms sharing
     one read-only memory device.
   - OCI images → templates via `aenv pull` (shared base layers convert
     once) or `POST /sandboxes-cold` directly from an image — covers the
     500 SWE-bench per-instance images.
   - Requirements: kernel ≥ 6.8, `/dev/kvm` (bare metal OK, no PVM kernel
     needed), trusted network (**no auth**), no GPU. Project is ~4 weeks
     old; pin the version (`/resume` already deprecated in favor of
     `/connect`).
   - Ash's `MicroVMPool` already implements spawn/destroy/pause/resume/fork
     against it; only the **snapshots endpoints need adding** (~3 HTTP
     calls).
2. **podman/docker commit** — fallback for machines without KVM. Measured on
   AWS (6.8 kernel, NVMe, podman 3.4.4): commit 219 ms @5 MB dirty,
   2.8 s @300 MB; run-from-image 121 ms. ~50 lines; zero privileges; Docker
   image layers are content-addressed and parent-shared, so a commit chain
   is already a forkable tree. Caveats: fs-only; volumes excluded; images
   local to the daemon.
3. **Runtime-side overlay freeze** — *reserved, de-prioritized* now that
   metal is available (its raison d'être was the no-KVM fast path). Design
   retained in §4 in case it's needed; measured 8–96 ms O(1) freeze,
   rootless kernel overlayfs works in userns on kernel 6.8 (verified).

**Rejected:** CRIU via podman (restore SIGSEGV'd in testing; root-only;
full-RSS dumps, no incremental); building our own microVM stack (AgentENV
already is one, MIT-licensed; our differentiation is the *trajectory layer*,
not virtualization).

## 4. Reserved design: runtime overlay freeze (backend 3)

Recorded for completeness; implement only if commit latency hurts and no KVM
box is available (e.g. k8s pods, where there is no docker daemon to commit).

- **Lazy init**: before the first tool call, harness calls
  `runtime.overlay_init(paths)`; ash-runtime overlays each configured
  mutable subtree (`/testbed`, conda prefix, `/tmp`, `/home`, …) in place
  (`mv dir dir.base; mount overlay`). Images need zero changes. The only
  spawn-time requirement: mount a host **layer volume**
  (`-v /nvme/ash-layers/<run>:/ash/layers`) so frozen layers are shared
  across containers and outlive them.
- **Freeze** (per mutating step, ~10 ms O(1)):
  `umount → mv upper frozen_k → mkdir upper → remount with frozen_k
  prepended to lowerdir`. Cheapness comes from rename being an inode
  metadata op; contrast docker commit which diffs+tars+registers the same
  upper (that's the 300× gap).
- **Rewind** (same container): kill agent-spawned processes (runtime owns
  the process table; kill BEFORE umount to avoid EBUSY), restack lowers to
  frozen_k + fresh upper. Discarded layers can be kept for redo.
- **Fork**: new container, same image + same layer volume; mount
  `lowerdir=frozen_k:…:base` + private upper. Cold fork ~1–2 s; warm
  parked-container pool is a v2 item.
- **EBUSY** on umount (process cwd/open file inside subtree): degrade to
  "skip freeze this step, record in trace" — replay covers the gap.
- **Whole-rootfs is not achievable at file level** (can't umount a root
  that's in use; container `/` belongs to the engine). Cover the mutable
  directory set instead; out-of-tree writes are repaired by replay.
- **Thousands of steps**: kernel overlayfs caps at ~500 lower layers and
  lookup cost is O(depth). Solution: **milestone consolidation** — every M
  steps (e.g. 64), materialize the merged view in the background with
  `rsync --link-dest` (unchanged files hard-linked); restore chain is then
  ≤ M+1 layers forever. Frozen layers are immutable so this races nothing.
  Default off for evals; needed for RL-scale horizons. (AgentENV has this
  built in as 32-layer compaction.)

## 5. Harness integration

- **Agent loop hook**: in `swebench/agent/__init__.py::_run_tool`, after a
  mutating call (`shell`, `edit`; not `readfile`/`grep`), call an injected
  `on_mutation` callback; it snapshots and emits `snapshot.created` into the
  existing events.jsonl. Agent stays sandbox-agnostic.
- **Resume/branch entrypoint**:
  `python -m swebench resume <run> --instance <id> --at <seq> [--fork N]` —
  load trajectory, truncate messages at the recorded index, instantiate
  snapshot (nearest ancestor + replay suffix), continue the loop. `--fork N`
  with different temperatures is the BPO-style branch primitive; branch-point
  selection policies (entropy, etc.) are pluggable filters over the trace —
  pure harness logic, orthogonal to the snapshot layer.
- **GC**: per-run snapshot budget; keep milestones, evict fine-grained
  layers/images by age & reuse; trace replay guarantees reachability of
  everything from the base image, so GC is a speed/disk tradeoff, never a
  correctness one.

## 6. Open items for the next session

1. Section-by-section design review (was about to start when the
   conversation migrated).
2. Decide v1 scope now that metal is available: AgentENV backend + commit
   fallback + trace/resume/replay; overlay backend probably deferred.
3. AgentENV deployment on the new machine (network isolation — it has no
   auth; version pinning).
4. Template pipeline: `aenv pull` the SWE-bench images vs `sandboxes-cold`
   + first-step snapshot.
5. Then: writing-plans skill → implementation plan → TDD.

## Appendix: measured numbers (this AWS box, 2026-08-18)

| Mechanism | Result |
|---|---|
| podman commit, 5 MB dirty | 219 ms |
| podman commit, 300 MB dirty | 2.8 s |
| podman run from committed image | 121 ms |
| CRIU checkpoint 355 MB RSS (root, runc) | 724 ms; **restore SIGSEGV — rejected** |
| rootless userns kernel overlayfs mount | works (kernel 6.8) |
| overlay freeze, 50 MB dirty | 96 ms |
| overlay freeze, 500 MB dirty (synced) | 8 ms (O(1) confirmed) |
| /dev/kvm on this box | absent (non-metal) — AgentENV needs the new metal machine |
