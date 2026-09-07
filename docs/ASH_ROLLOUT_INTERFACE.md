# Ash Rollout Interface

This branch adds a strategy-neutral Ash rollout interface for Miles. It starts
from the Ash `main` branch and keeps the rollout protocol independent from any
branching algorithm. The first executable strategy is the branch-free
`SequentialRolloutStrategy`; no checkpoint index or branch-selection policy is
part of the interface layer. `CheckpointAgentLoopRolloutStrategy` is a
functional reference that exercises the AgentENV checkpoint contract; its
fixed first-checkpoint rule is not part of that contract.

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
- `EnvironmentCheckpoint` is an opaque, owned handle. Strategies create,
  restore and release it through `EnvironmentProvider`; they do not call
  AgentENV snapshot URLs directly.
- `CheckpointCapabilities` distinguishes full runtime state from a filesystem
  image and declares repeated-restore and explicit-release semantics. The
  AgentENV microVM backend satisfies the current branch contract; Docker does
  not advertise it as equivalent.
- `ModelClient` receives the endpoint supplied by Miles. `EndpointModelClient`
  is a minimal raw SGLang implementation; a Session Server client can replace
  it when Miles needs SessionTree recording.

## Starting the service

The generic service can be started without changing the strategy code. The
following command selects the real AgentENV-backed environment provider and the
Miles v2 Session Server:

```bash
PYTHONPATH=sdk:. python -m swebench.rollout_groups.server \
  --strategy agent-loop \
  --image "$ASH_ROLLOUT_IMAGE" \
  --backend-json '{"backend":"microvm","microvm":{"server_url":"http://agentenv:8000","template":"ash-runtime-template","runtime_port":3000,"api_key_file":"/run/secrets/agentenv-api-key"}}' \
  --miles-session-endpoint "http://miles-session:30000" \
  --model openai/local
```

`ASH_ROLLOUT_IMAGE` is an AgentENV template or snapshot containing
`ash-runtime`; it is not a Docker entrypoint. `server_url` and the API-key
file are deployment settings supplied by the environment owner. The process
exposes `POST/GET/DELETE /rollout-groups` on port `11001` by default. A later
branch policy can replace `MilesSessionAgentRolloutStrategy` without changing
this transport or environment wiring.

The checkpoint reference strategy uses the same service with
`--strategy checkpoint-agent-loop-v1`. It records the first tool-complete
boundary, restores the remaining allocated slots from that checkpoint and
releases the persistent AgentENV snapshot after the group finishes.

## What this branch proves

The protocol and HTTP lifecycle are covered by unit tests; malformed requests,
duplicate IDs, cancellation and result identity are checked before import into
Miles. The executable sequential path additionally verifies that a model
client is called once per slot, an environment is created and destroyed for
each slot, and the returned token sequence is exported as a trajectory. Full
Ash regression is `414 passed, 4 skipped` after removing the unrelated
checkpoint-cache experiment suite. The tests cover checkpoint capabilities,
AgentENV create/restore/release HTTP paths, ownership, release failure and the
parent/child SessionTree reference strategy. An earlier revision using the
same AgentENV create/restore endpoints also ran the real
`AshAgent -> AgentENV -> Miles Session Server -> GRPO/Megatron` path. The new
ownership/release wrapper still needs a live deployment rerun; performance,
dynamic checkpoint selection and cross-job checkpoint retention remain
outside this functional reference.
