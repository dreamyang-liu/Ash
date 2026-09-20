# Shepherd validation

Date: 2026-09-20. Based on Ash `dev`
`c5a1d30795af974ad0c6fe1923e4e08878d08d45`.

## Real DeepSeek / AgentENV pilot

- Model: `deepseek-v4.1-flash`; Claude Code harness; reasoning `high`.
- AgentENV: `v0.1.2-ash.1`; original digest-pinned task image and verifier.
- Task: `termenv-preserve-ansi-resets`; imported failed initial rollout.
- Budget: one parent plus **one new branch** (`max_rollouts: 2`).
- Selector chose checkpoint **18**, after that completed tool turn.
- Exact native prefix: **72 entries**, excluding **133 later entries**. Verified
  against the original history, allowing only independent session ID and working
  directory substitutions. Prefix SHA-256:
  `6db9b51eb9f84938b5e4ea4a3b13a30b1d8d5f167582f21cad43e93cf7cd6867`.
- New actor completed **24 tool turns**, produced a patch and reached the real
  verifier. **F2P failed, P2P passed, resolved=false**. The pipeline completed;
  this is not a claim that the task was solved.

Actor usage: 23 API calls; 1,522,198 input tokens (1,363,328 cached), 29,608 output
tokens (21,277 reasoning). Selector overhead: 34,980 input tokens, 50,136 output
tokens (49,993 reasoning). Imported parent cost is separate.

An earlier 16K selector budget was exhausted by reasoning. No partial decision
was executed; the retained completed pilot used 65,536 tokens. Raw requests,
patches and verifier artifacts remain private on the validation host. No API
credentials or private server paths are published.

This is an end-to-end integration pilot, **not** a full-cohort/eight-rollout
accuracy estimate or a reproduction of the paper's training results. The live
pilot ran before the Shepherd-only packaging split and timeout/audit hardening;
the selection, exact-restore and grading behavior is unchanged. The isolated
release is regression-tested again after the split.

## Runtime and automated checks

Python 3.12, Claude Agent SDK 0.2.145, MCP 2.2.0, HTTPX 0.28.1, FastAPI 0.141.1,
Uvicorn 0.53.0 and FileLock 3.32.7.

- Windows DeepSWE tests: **61 passed**.
- Linux release regression: **994 passed, 17 skipped, 6 deselected**, covering
  `harness/tests`, `swebench/tests`, `sdk/tests` and `deepswe/tests` with
  `-m 'not slow'`.
- SDK/runtime contracts: **58 passed, 0 failed, 7 skipped**
  (unavailable unrelated harness CLIs).

The release source fingerprint matches between Windows and Linux:
`990b99c8f34b70a8a85e7f994af8a659b7f25810153daaa76ecbf3960cc00e3b`.
This hashes non-test Python sources in the harness, benchmarks and SDK. The
completed pilot's prefix, journal and verifier result were rechecked using the
Shepherd-only release code; no extra model rollout was charged for packaging.

Original production checkouts, experiment services and outputs were not modified.
