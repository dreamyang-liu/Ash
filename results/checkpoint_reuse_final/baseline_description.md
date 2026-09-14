# Checkpoint Reuse Baseline Description

## No Cache
Each rollout starts from the initial environment. No previous trajectory or checkpoint is reused.

## Exact Full History
Cache key is the complete normalized trajectory history. Reuse requires exact history equality. This provides strict correctness but has very low recall because small trajectory differences prevent reuse.

## TVCACHE TCG
Implemented as the external cache baseline. It is based on structured tool-call/cache matching rather than whole-history string equality. The exact cache-key and mutation handling should be verified against the original implementation before final paper submission.

## Full Workspace Hash
Uses whole-workspace state hashing as a strong oracle-style baseline. If the entire workspace hash matches, the environment state is considered reusable. It achieves higher recall but requires scanning the workspace state.

## Ours: Relaxed Projection
Instead of scanning the entire workspace, classify tool interactions into:
- safe read: can be ignored for state equivalence
- mutation: must be preserved
- barrier: conservatively preserved when safety cannot be proven

Goal: achieve state-equivalent reuse with lower matching overhead.
