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

`ash-runtime` is the sandbox-side execution service, not the rollout service.
It is a small Go binary that exposes Ash's eight builtin tools (`shell`,
`process`, `text_editor`, `grep_files`, `web_fetch`, `web_search`, `artifact`,
and `wait_for_events`) over the tool protocol. The rollout service and Python
SDK call it; AgentENV creates and snapshots the microVM in which it runs; Miles
never calls it directly.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /rollout-environments` | List deployment-approved static environment references. The response omits backend `spawn_ref` values and credentials. |
| `POST /rollout-groups` | Validate a group and enqueue an asynchronous job. Repeating the same job ID with the same request is idempotent and returns the job's current status, including a terminal status if it already finished; reusing it for another request is rejected. |
| `GET /rollout-groups/{job_id}` | Read progress or the terminal result. A terminal result contains complete trajectories, not just final text. |
| `DELETE /rollout-groups/{job_id}` | Cancel unfinished work and release the in-memory job record. Miles calls it after consuming a terminal result as well as on failure; the response is a lightweight job ID/status acknowledgement, not a second copy of the trajectories. |

Terminal results that are not deleted, for example after a Miles process
failure, become eligible for removal after `--result-ttl-seconds` (300 seconds
by default). The in-memory service performs this pruning lazily on a later
submit/get/delete operation; it has no background expiry worker or persistent
job store.

The wire version is `ash-rollout-v2`, matching the Miles adapter. Each
returned leaf must retain its `sample_slot_id`; Ash branch/checkpoint IDs are
lineage metadata. A trajectory includes exact token IDs, generated spans,
weight version, ordered messages, reward and branch lineage.

### Request and response bodies

Miles submits one complete prompt group. `environment_ref` is either a logical,
deployment-allowlisted template/snapshot identity or a digest-pinned OCI image
identity. It never contains a backend endpoint or credential. `sample_slots`
are the training slots Miles has reserved, while `budgets` bound the work Ash
may perform.

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "rollout_id": 17,
  "prompt_group_id": "group-42",
  "task_id": "swebench__repo-123",
  "environment_ref": {
    "kind": "image",
    "id": "docker.io/library/ubuntu",
    "revision": "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254",
    "resource_profile": "standard"
  },
  "sample_slots": [
    {"sample_slot_id": "...:slot:0", "sample_index": 0},
    {"sample_slot_id": "...:slot:1", "sample_index": 1}
  ],
  "max_samples": 2,
  "minimum_returned_samples": 2,
  "prompt": "<text or ordered messages>",
  "prompt_token_ids": [1, 2, 3],
  "model_endpoint": "http://model-router:port",
  "session_server_endpoint": "http://session-server:port",
  "model": "local-model",
  "expected_weight_version": "1",
  "return_rollout_logprobs": false,
  "sampling_params": {"temperature": 0.6, "max_new_tokens": 2048},
  "budgets": {
    "max_model_calls": 6,
    "max_tool_calls": 2,
    "max_wall_time_seconds": 1800
  }
}
```

In the current agent-loop strategies, `task_id` becomes the AshAgent
`instance_id` used for task/trace identity. It does not by itself install a
repository, select a working directory, configure tools, or bind a reward.
Those task-specific effects must already be represented by the selected
environment and prompt, or be added by a future task-initialization contract.
Only `environment_ref` selects the sandbox creation artifact.

The protocol accepts an extensible `sampling_params` object. The shared
AshAgent adapter maps `model`, `max_tokens`/`max_new_tokens`, `temperature`,
`top_p`, `top_k`, `stop`, `stop_token_ids`, `skip_special_tokens`,
`no_stop_trim`, `spaces_between_special_tokens`, and `chat_template_kwargs` into the
OpenAI-compatible model request. An explicit `extra_body` object may carry
additional provider fields; future sampling controls still require an adapter
extension before they can be relied on.

The first `POST /rollout-groups` normally returns this asynchronous
acknowledgement:

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "status": "queued"
}
```

If the client retries the same POST after losing the response, `status` is the
job's current `queued`, `running`, or terminal state. The response still does
not contain trajectories; Miles obtains those through GET.

Miles polls `GET /rollout-groups/{rollout_job_id}`. A terminal response has
the following shape (the full `messages`, token IDs and spans are not shortened
in the actual response):

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "prompt_group_id": "group-42",
  "status": "completed",
  "max_samples": 2,
  "actual_samples": 2,
  "stop_reason": null,
  "search_branches": 1,
  "consumed_budget": {"model_calls": 3, "tool_calls": 1, "session_tree_leaves": 2},
  "trajectories": [
    {
      "sample_slot_id": "...:slot:0",
      "branch_id": "...:root:0",
      "parent_branch_id": null,
      "branch_point_token_count": null,
      "messages": [
        {"role": "user", "content": "Inspect the workspace."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
        {"role": "assistant", "content": "Parent completed."}
      ],
      "token_ids": [1, 2, 3, 4, 5, 6],
      "prompt_length": 3,
      "generated_spans": [
        {
          "response_id": "response-tool",
          "start": 3,
          "end": 4,
          "input_token_ids": [1, 2, 3],
          "output_token_ids": [4],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "tool_calls"
        },
        {
          "response_id": "response-parent",
          "start": 5,
          "end": 6,
          "input_token_ids": [1, 2, 3, 4, 5],
          "output_token_ids": [6],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "stop"
        }
      ],
      "response_text": "Parent completed.",
      "reward": null,
      "status": "completed",
      "metadata": {"environment_checkpoint_id": "opaque-to-Miles"}
    },
    {
      "sample_slot_id": "...:slot:1",
      "branch_id": "...:child:1",
      "parent_branch_id": "...:root:0",
      "branch_point_token_count": 5,
      "messages": [
        {"role": "user", "content": "Inspect the workspace."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
        {"role": "assistant", "content": "Child completed."}
      ],
      "token_ids": [1, 2, 3, 4, 5, 7],
      "prompt_length": 3,
      "generated_spans": [
        {
          "response_id": "response-tool",
          "start": 3,
          "end": 4,
          "input_token_ids": [1, 2, 3],
          "output_token_ids": [4],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "tool_calls"
        },
        {
          "response_id": "response-child",
          "start": 5,
          "end": 6,
          "input_token_ids": [1, 2, 3, 4, 5],
          "output_token_ids": [7],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "stop"
        }
      ],
      "response_text": "Child completed.",
      "reward": null,
      "status": "completed",
      "metadata": {"environment_checkpoint_id": "opaque-to-Miles"}
    }
  ]
}
```

Here `actual_samples` is exactly the length of `trajectories`. The child uses
`parent_branch_id` and `branch_point_token_count` to identify its model-prefix
attachment. Ash may
keep environment checkpoint and sandbox handles in `metadata`; Miles treats
those values as opaque lineage information. After consuming a terminal result,
Miles calls `DELETE /rollout-groups/{rollout_job_id}`; the response is only a
status acknowledgement:

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "status": "completed"
}
```

## Extension points

- `RolloutStrategy.run(request, context)` is the algorithm seam.
- `EnvironmentProvider` describes the environment lifecycle required by a
  strategy and can wrap `MicroVMPool`, Docker or a remote AgentENV pool.
- `EnvironmentCatalog` maps the request's immutable logical environment
  identity to the concrete reference accepted by Ash's native
  `Pool.spawn(image=...)`. Unknown revisions or resource profiles are rejected
  before the job is enqueued.
- `AgentEnvOCIResolver` is the optional dynamic path for a digest-pinned
  `kind: image` that is not already in the static catalog. It validates the
  registry and resource profile before enqueue, then prepares or reuses a
  runtime-ready AgentENV snapshot inside the asynchronous job. Its `aenv` CLI
  subprocesses receive the same AgentENV URL and API key as the microVM
  provider through an isolated temporary credentials file; they do not depend
  on a developer's prior `aenv auth` state.
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
- `SessionAgentStrategySupport` owns the common Miles SessionTree agent
  execution path, tool-budget enforcement and model configuration.
  `trajectory_from_session` converts SessionTree records into the wire
  trajectory. A new branch policy can reuse both and implement only its tree
  growth, checkpoint selection and scheduling decisions.

## Starting the service

The generic service can be started without changing the strategy code. Each
rollout request selects its environment at sandbox-creation granularity through
`environment_ref`; `task_id` supplies task identity, not a second environment
selector. The service does not own a default image. A
deployment may enable a static catalog, the AgentENV OCI resolver, or both. The
catalog is an exact allowlist and provider-specific translation:

```json
{
  "environments": [
    {
      "kind": "template",
      "id": "swebench-runtime",
      "revision": "sha256:immutable-revision",
      "resource_profile": "standard",
      "spawn_ref": "agentenv-template-17"
    }
  ]
}
```

`spawn_ref` is never accepted from Miles. For AgentENV it names a prepared
template or snapshot; for Docker or Kubernetes it can name an allowlisted OCI
image. A static entry may also map a public `kind: image` identity to an
already-prepared AgentENV snapshot. The request must match all four public
fields exactly. This makes a resource profile part of the trusted environment
identity; the selected template, image, or snapshot is responsible for
implementing that profile.

`GET /rollout-environments` returns the public four-field identity of every
static catalog entry in stable order:

```json
{
  "protocol_version": "ash-rollout-v2",
  "environments": [
    {
      "kind": "template",
      "id": "swebench-runtime",
      "revision": "sha256:immutable-revision",
      "resource_profile": "standard"
    }
  ]
}
```

It does not return `spawn_ref`, AgentENV credentials, or every template in the
underlying AgentENV inventory. The endpoint is discovery-only; registration
remains a deployment control-plane operation. An unknown static identity is
rejected with HTTP 400 during POST validation. Dynamic OCI identities are
validated from resolver policy at submission time and do not become static
catalog entries merely because their prepared snapshots are cached.

### Public OCI images on AgentENV

An AgentENV `templateID` is local to one AgentENV deployment; it is not an OCI
URL. A normal public image also does not contain the in-sandbox `ash-runtime`
service required by the Ash tool client. The optional AgentENV OCI resolver
bridges that gap inside the existing asynchronous rollout job:

1. validate a digest-pinned OCI repository against a registry allowlist;
2. derive a cache key from the source digest, resource profile, runtime binary
   digest and preparation schema;
3. reuse the corresponding runtime-ready snapshot when it already exists;
4. otherwise import the OCI image as a temporary AgentENV template, start a
   builder sandbox, upload and start `ash-runtime`, then snapshot it;
5. destroy the temporary sandbox and template, and spawn the rollout sandbox
   from the cached snapshot.

Miles still makes only one `POST /rollout-groups` call. Environment preparation
happens inside that asynchronous job; later groups using the same cache key
take the warm path. Enable this behavior with a deployment-owned policy file:

```bash
PYTHONPATH=sdk:. python -m swebench.rollout_groups.server \
  --strategy agent-loop \
  --agentenv-oci-resolver-config configs/agentenv_oci_resolver.example.json \
  --backend-json '{"backend":"microvm","microvm":{"server_url":"http://agentenv:8000","runtime_port":3000,"api_key_file":"/run/secrets/agentenv-api-key"}}' \
  --miles-session-endpoint "http://miles-session:30000" \
  --model openai/local
```

The resolver config contains only preparation policy and the local runtime
artifact. AgentENV endpoint and credential settings stay in `--backend-json`
and are shared with the resolver internally. The matching request uses
`kind: image`, keeps the OCI repository in `id`, and puts the immutable
manifest digest in `revision`:

```json
{
  "kind": "image",
  "id": "docker.io/library/ubuntu",
  "revision": "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254",
  "resource_profile": "standard"
}
```

Mutable tags, URL schemes, embedded credentials, unknown registries and unknown
resource profiles are rejected before the job is enqueued. The first
implementation expects a Linux image in which the builder can create the
runtime destination and invoke a POSIX shell; minimal/distroless images need a
provider-specific preparation strategy. Static catalog entries remain the
preferred path for centrally prepared templates and snapshots.

Docker can bind-mount `ash-runtime` at container creation time, whereas an
AgentENV microVM must receive it during preparation and preserve the running
process in the cached full-runtime snapshot.

The following command selects the real AgentENV-backed environment provider,
the catalog and the Miles v2 Session Server:

```bash
PYTHONPATH=sdk:. python -m swebench.rollout_groups.server \
  --strategy agent-loop \
  --environment-catalog /etc/ash/environments.json \
  --backend-json '{"backend":"microvm","microvm":{"server_url":"http://agentenv:8000","runtime_port":3000,"api_key_file":"/run/secrets/agentenv-api-key"}}' \
  --miles-session-endpoint "http://miles-session:30000" \
  --model openai/local
```

`server_url`, credentials and runtime port remain deployment settings and never
cross the rollout request. The process exposes `POST/GET/DELETE
/rollout-groups` on port `11001` by default. A later branch policy can replace
`MilesSessionAgentRolloutStrategy` without changing this transport or
environment wiring.

The checkpoint reference strategy uses the same service with
`--strategy checkpoint-agent-loop-v1`. It records the first tool-complete
boundary, restores the remaining allocated slots from that checkpoint and
releases the persistent AgentENV snapshot after the group finishes.

The older `swebench/rollout_server.py` exposes a separate `/run` endpoint for
one episode. It is retained for compatibility with that caller and is not an
alias for `ash-rollout-v2`: Miles tree rollout uses `rollout_groups/server.py`
and the three `/rollout-groups` endpoints documented here.

## What this branch proves

The protocol and HTTP lifecycle are covered by unit tests; malformed requests,
duplicate IDs, idempotent retries before and after completion, cancellation,
terminal-result release and result identity are checked before import into
Miles. The executable sequential path additionally verifies that a model
client is called once per slot and the returned token sequence is exported as
a trajectory. A separate injected-provider test verifies per-slot environment
creation and destruction; the command-line `sequential` service itself does
not attach an environment provider. The tests also cover checkpoint
capabilities, AgentENV create/restore/release HTTP paths, ownership, release
failure and the parent/child SessionTree reference strategy.

Two real-backend environment paths completed the same AgentENV/Firecracker +
Miles GRPO smoke test on 2026-09-11:

- The prebuilt-template path performed one checkpoint restore, one tool
  execution, five model calls, produced two SessionTree leaves, completed an
  optimizer step (`grad_norm=32.4090`), and updated SGLang `weight_version`
  1 -> 2.
- The dynamic OCI path first pulled a digest-pinned public Ubuntu 24.04 image,
  injected `ash-runtime`, and created or reused a runtime-ready AgentENV
  snapshot. It then completed the same branch and training chain with
  `grad_norm=32.4122` and SGLang `weight_version` 1 -> 2. A separate resolver
  probe measured 18.46 seconds for cold preparation and 0.001 seconds for an
  identical cached resolution; those preparation timings are not end-to-end
  GRPO latency measurements.

These are functional integration checks, not claims about rollout performance,
training quality, dynamic checkpoint selection, or cross-job checkpoint
retention.
