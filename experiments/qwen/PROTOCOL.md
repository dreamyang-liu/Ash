# Experiment protocol

All runs use mini-swe-agent 2.4.6, Qwen `qwen3.8-27b`, high reasoning, and exact
closed assistant-turn boundaries. Task input contents are SHA-256 pinned.
An explicit cohort is required. Default task concurrency is 16; smoke tests use 1.

Every task runs four independent baseline trajectories even if the first passes.
Each branch method shares baseline attempt 1 (not the best of four). Initial
success skips branching. An initial failure permits at most seven new evaluated
continuations: at most eight including that shared initial trajectory.

SPROUT uses Ash dev's analyst/reviewer control flow and assistant-turn replacement,
with adaptive round caps 4 then 3. BPO ranks exact backbone states by first visible
continuation token entropy (top-5 plus residual mass, a lower bound), with 64
completion-token spacing, then samples siblings without hints. Shepherd uses a
Qwen meta-agent to choose one exact state and samples siblings without hints.
BPO and Shepherd are **inference-time policy adaptations**, not the original
training algorithms or a claim of reproducing their published training results.
No missing logprobs are replaced with random scores. Invalid selections fail.

Failed initial trajectories retain disk-only checkpoint IDs, content-addressed
layers, exact native transcript offsets/hashes and canonical journals permanently
for future strategies. Transcripts and base layers are shared, not copied per
checkpoint. Do not remove their output directory or sweep those snapshot IDs.
Disk-only restore does not preserve live processes, tmpfs, RAM or external services;
tasks requiring them need full snapshot support and a separately labelled run.
Noninitial leaf runs disable per-step capture, retaining only the final grading
snapshot for verification. This experiment package does not delete snapshots,
images, layers or run artifacts. Successful first rollouts do not get a retained
checkpoint manifest; their already captured snapshots remain on the backend until
the operator applies a separate retention policy. SPROUT intermediate checkpoints
remain available after the experiment. No deletion script is included.

Results record initial and additional phases separately. Success@1/@5/@8 uses all
selected tasks as denominator; baseline @5/@8 stops at the observed four attempts.
Reports show coverage, initial-failure recovery, successful/graded trajectory
counts, positive rate, time when the first successful verifier completed, newly
executed tool steps, input/cached-input/output tokens and source-reported cost.
Shared initial work is counted once in each method comparison, never four times
in physical expenditure. Copied journals are deduplicated by content hash.
Infrastructure failures retain all output and usage; they are incomplete, not
ordinary negative rewards. No automatic stage retry silently spends extra rollouts.
HTTP retries remain visible, and unreported usage/billing stays unknown.

Stage wall time covers setup, actor, critic, snapshot, restore and verification.
Critic/BPO probe usage contributes to additional
phase costs. Missing provider prices are null, not zero; known charges remain
available as lower bounds. Host CPU/storage dollar pricing is not assumed.
Budget failures retain their consumed steps, tokens and costs. A completed stage
can be reused with the same immutable manifest. An incomplete stage requires a
fresh output root so failed execution history is never erased.
