# Ash source for the B300 LoRA training trial

This branch accompanies `dreamyang-liu/miles`, branch
`codex/b300-lora32-32tasks-20260917`. It includes the review-guided branching,
graded messages-v3 export, LoRA routing, and sequence-budget work in its
commit history, together with the runtime fixes used by the B300 deployment.

The task driver uses `miles.branching.return_mode=all` and
`miles.max_samples=8`. The reviewer chooses up to four children in its first
round and three in its second; a successful root allows up to two children.
Every selected sibling is graded before a round is considered complete.
The learner retains each valid graded trajectory and weights tasks equally.
Incomplete or unverifiable execution does not become a negative reward.

The additional deployment fixes cover:

- Serializing tokenizer initialization and token counting within each
  process, preventing concurrent cold loads from entering Transformers'
  lazy initialization together.
- Preserving NUL-containing tool output and JSON values losslessly in
  PostgreSQL JSONB, including literal values shaped like the storage envelope.
  Decoded values retain their original content and identity.
- Incremental journal ingestion and an independent lease heartbeat so a
  large journal does not prevent lease renewal.
- Graceful worker draining through `SIGUSR1`, plus an optional
  `max_running_jobs` admission limit shared through the database.
  Existing attempts retain their leases during a handover.
- Explicit Messages authentication selection and optional removal of only
  the incompatible Anthropic effort field for Qwen routes.
- Reporting controlled execution limits distinctly from uncertain tool
  execution, so only verified final snapshots can be graded/exported.
- Matching the deployed synchronous shell schema and file-tool panel.

No production configuration, credentials, task artifacts, or database
contents are part of this change. Database tests use
`ASH_RUNSTORE_TEST_DSN` and create isolated temporary schemas.

The source records the deployment used for the trial. Operational recovery
still requires checking snapshot availability, native conversation prefixes,
remaining execution budgets, and the policy version before resuming work.
