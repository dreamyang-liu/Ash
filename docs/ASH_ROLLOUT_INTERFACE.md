# Ash Rollout Interface

This branch adds a strategy-neutral Ash rollout interface for Miles. It is
not a replacement for Ash's branch algorithm, and it does not require a
particular checkpoint index or branch-selection policy.

## Boundary

```text
Miles POST /rollout-groups
  -> validated group request and sample slots
  -> Ash job lifecycle (queued/running/terminal)
  -> injected strategy
  -> trajectories bound to the original slots
  -> Miles GET result
```

The service owns request validation, idempotency, cancellation and cleanup. The
strategy owns agent prompting, tool execution, checkpoint selection, fork and
trajectory export. A later policy can therefore change the tree-growth
algorithm without changing the Miles-facing API.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /rollout-groups` | Validate a group and enqueue an asynchronous job. Repeating the same job ID with the same request is idempotent; reusing it for another request is rejected. |
| `GET /rollout-groups/{job_id}` | Read progress or the terminal result. A terminal result contains complete trajectories, not just final text. |
| `DELETE /rollout-groups/{job_id}` | Set a cancellation event. Strategies must check it before creating new branches and release their resources in `close()`. |

The wire version is `ash-rollout-v1`, matching the Miles adapter. Each
returned leaf must retain its `sample_slot_id`; Ash branch/checkpoint IDs are
lineage metadata. A trajectory includes exact token IDs, generated spans,
weight version, ordered messages, reward and branch lineage.

## Extension points

- `RolloutStrategy.run(request, context)` is the algorithm seam.
- `EnvironmentProvider` describes the environment lifecycle required by a
  strategy and can wrap `MicroVMPool`, Docker or a remote AgentENV pool.
- `ModelClient` receives the endpoint supplied by Miles. `EndpointModelClient`
  is a minimal raw SGLang implementation; a Session Server client can replace
  it when Miles needs SessionTree recording.

## What this branch proves

The protocol and HTTP lifecycle are covered by unit tests; malformed requests,
duplicate IDs, cancellation and result identity are checked before import into
Miles. Full Ash SDK regression is `486 passed, 4 skipped` in the current test
environment. The sequential interface baseline has an HTTP end-to-end test; a
production strategy and real Miles Session Server/RL integration are still
pending.
