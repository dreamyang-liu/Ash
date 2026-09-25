# BPO sampling benchmark

This directory contains only the inference-time BPO sampling baseline. It runs
the latest Ash mini-swe-agent harness (`mini-swe-agent==2.4.6`), records exact
native conversation prefixes and checkpoints, ranks failed-backbone decision
boundaries by first-reported-token entropy, and launches point-only continuations.

The initial rollout counts toward the total budget. If it succeeds, BPO is
skipped. If it fails, BPO launches at most `max_rollouts - 1` continuations and
stops at the first verified success.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install ./sdk -r configs/branchbench/requirements.txt
(cd runtime && go build -o ash-runtime .)
git clone https://github.com/datacurve-ai/deep-swe ../deep-swe
export PYTHONPATH=.:sdk
export AENV_SERVER_URL=http://127.0.0.1:8000
export AENV_API_KEY='your-agentenv-key'
export OPENAI_BASE_URL='https://your-provider/v1'
export OPENAI_API_KEY='your-model-key'
umask 077
```

Credentials are read only from environment variables. Provider audits contain
prompts and responses, so keep run directories private.

## Run

Start the audited OpenAI-compatible bridge:

```bash
python -m deepswe.branching.bridge \
  --model qwen3.8-27b --effort high --port 18186 \
  --audit-root runs/branchbench-qwen \
  --extra-json '{"top_k":20,"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}'
```

Then run BPO and produce its report:

```bash
python -m deepswe.branching.runner --config configs/branchbench/qwen.json
python -m deepswe.branching.report runs/branchbench-qwen
```

The bridge exposes `/v1/chat/completions` for mini-swe-agent and preserves each
native request/response for checkpoint-to-token alignment. Its `/health` model,
effort and audit directory must match the runner configuration.

## BPO contract

Checkpoint N is the state after completed tool step N and before the next model
decision. Only exact checkpoint/native-prefix pairs are eligible. BPO ranks
them by Shannon entropy over the provider's top-k probabilities plus one
aggregated unreported-tail bin. This is a lower bound on full-vocabulary
entropy. If the original response did not include logprobs, the exact saved
request is rescored with one output token. A provider that rejects or omits
logprobs blocks the task; there is no random or alternate-model fallback.

Selected points obey `bpo_min_spacing` in completion-token position. Up to
`bpo_max_points` distinct points are selected, and the continuation budget is
distributed round-robin when fewer points are available. Every continuation
restores the selected sandbox checkpoint and exact mini native history prefix,
then resumes without a user hint (`hint_delivery: point-only`).

## Reusing initial rollouts

Set `initial_root`, `provider_audit_root`, `initial_owner_template`, and optional
`task_locks` in the config. Imported parents must be completed mini-swe-agent
runs with unchanged model, journal and grade hashes. Their native histories,
snapshots and provider audits must remain available.

## Records and tests

The runner saves the manifest, parent identity, entropy requests, plans,
journals, verifier outputs and task summaries. Infrastructure failures are
marked `blocked` and are never counted as ordinary model failures. Interrupted
actor journals require explicit recovery.

```bash
PYTHONPATH=.:sdk python -m pytest deepswe/tests/test_bpo_sampling.py -q
PYTHONPATH=.:sdk python -m pytest harness/tests/test_mini_swe.py \
  harness/tests/test_assistant_turn.py swebench/tests/test_assistant_branch.py -q
```
