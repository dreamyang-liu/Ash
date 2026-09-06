# Ash Rollout Interface

This branch adds a strategy-neutral Ash rollout interface for Miles. It starts
from the Ash `main` branch and keeps the rollout protocol independent from any
branching algorithm. The first executable strategy is the branch-free
`SequentialRolloutStrategy`; no checkpoint index or branch-selection policy is
part of the interface layer.

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

## What this branch proves

The protocol and HTTP lifecycle are covered by unit tests; malformed requests,
duplicate IDs, cancellation and result identity are checked before import into
Miles. The executable sequential path additionally verifies that a model
client is called once per slot, an environment is created and destroyed for
each slot, and the returned token sequence is exported as a trajectory. Full
Ash SDK regression is `490 passed, 4 skipped` in the current test environment.
The service builder and real `AshAgent -> AshSession -> AgentENV` wiring are
implemented. A live AgentENV endpoint and template are still required for a
deployment-level run; the Miles-side GRPO/Megatron smoke must then be launched
with the returned trajectories.
