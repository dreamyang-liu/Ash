# SWE-bench Verified: measured results

Everything below is measured, not estimated, except where marked *(est)*. Agent
costs are the Claude Code CLI's own `cost_usd` accounting summed from journals;
times are journal timestamps; resolve verdicts are the fixed grader run against
restored snapshots. Data: `runs/v500`, `runs/branch134`, `runs/pass4` (persistent
disk). As of 2026-09-03; all four baseline passes complete.

**Setup**: agent = Claude Code CLI + `us.anthropic.claude-sonnet-4-6` (Bedrock),
two MCP tools (shell + text_editor), microVM sandboxes with a snapshot per step,
1200 s timeout per run, 8 parallel workers. Analyst (branching only) = the same
sonnet 4.6.

---

## Resolve rate

| Configuration | Resolved | Rate | Note |
|---|---:|---:|---|
| single attempt, old prompt, **broken grader** | 215/500 | 43.0% | the first full run's reported number |
| single attempt, old prompt, fixed grader | 364/500 | 72.8% | same snapshots re-graded; **no agent re-run** |
| single attempt, new prompt (pass 2) | 371/500 | 74.2% | prompt alone is worth +1.6 pt |
| pass@2 (union of two singles) | 386/500 | 77.2% | |
| pass@3 | 392/500 | 78.4% | |
| pass@4 (four independent singles) | 395/500 | 79.0% | measured, not extrapolated: the independence formula promised ~99% |
| **single + verifier-guided branching** | **465/500** | **93.0%** | 134 failures re-run with branching (2 rounds, width 4→3) |

**The head-to-head this table exists for**: at near-equal budget, four blind
samples reach 79.0% (pass@4, $739 total) while verifier-guided branching
reaches 93.0% (pass@3.9, $663). Same model, same sandboxes, same grader. Blind
retries fail in correlated ways — 105 instances failed all four samples, and
the marginal sample decays fast (+4.4, +1.2, +0.6 pt) — while branching
rescued 101 of its 134 targets because each branch starts from a chosen step
of a failed trajectory WITH the verdict (which test failed, what broke). The
verifier signal is precisely what breaks the failure correlation.

### The stronger baseline, run for real: adaptive blind retry

The obvious objection to pass@4: branching only spends extra on the failures,
while full passes waste money re-running solved instances. So the fair opponent
was RUN as its own experiment (`runs/adaptive`, 2026-09-03): one fresh full
pass (365/500 = 73.0%, consistent with the other four singles), then FIVE blind
single-attempt retries over exactly its 135 failures — ~5.0 runs per failed
instance, versus branching's 5.5. Budget matched, verifier used identically for
instance selection; the only variable left is whether the retry carries the
verdict.

The pass@4 experiment's rounds also cover these 135 instances, so the blind
curve extends to NINE independent samples for free:

| blind samples on the same 135 failures | rescued (union) | marginal |
|---|---:|---:|
| 1–5 (the adaptive retries) | 28/135 = 21% | +14, +7, +3, +3, +1 |
| 6–8 (the pass@4 rounds, same prompt) | 33/135 = 24% | +4, **+0**, +1 |
| 9 (the v500 pass, OLD prompt) | 36/135 = 27% | +3 |
| branching, 5.5 runs/instance | **101/134 = 75%** | |

Budget-equalised exactly, by grafting the pass@4 rounds' samples on these 135
(costed at only those instances' runs): **blind is bracketed from both sides of
branching's $663 — $644 (1+6 draws) → 79.4%, $714 (1+7 draws) → 79.4%, the
seventh draw adding zero across a full round — while branching's $663 → 93.0%.
Same money, the verdict is worth 13.6 points (68 instances).** Nine blind samples cap at 27% on the
failure set — sample 7 added ZERO new solves across a full round — while the
verdict-guided branches reached 75% in 5.5. Two footnotes the curve buys:
same-model blind resampling is exhausted by ~5 draws, and the one sample with a
DIFFERENT prompt (old-prompt v500, +3) out-earned same-prompt draws 6–8 —
distribution diversity beats redrawing, and the verdict is the strongest
distribution shift available.

Reference points: public leaderboard numbers for sonnet 4.6 with tuned harnesses
are ~77–80%. **The 93.0% is not leaderboard-comparable**: the analyst reads the
official tests' verdict between rounds, so it is verifier-guided. The honest
framing is RL-rollout economics — what it costs to turn a failed trajectory into
a successful one with full per-step snapshots.

### Grader corrections (43.0% → 72.8%, zero agent time)

Eight defects, each found by validating against must-fail *and* must-pass
inputs or by spot-checking; ~30 points of fiction total. The dataset itself
contributed four (bare sympy test names, parametrised ids split on internal
commas, django docstrings harvested as test ids — 165/231 django instances,
pytest's `[100%]` progress marker as a test id); the grader contributed the
rest (label-passing instead of output-parsing for django, an ascii-locale
UnicodeEncodeError under `--verbosity 2`, treating agent test-edits as fatal
instead of reverting them per the public convention). Two instances remain
ungradable (every FAIL_TO_PASS id damaged).

---

## Branching (`runs/branch134`: the 134 instances that failed the single pass)

| | |
|---|---:|
| rescued | **101/134 = 75.4%** |
| — by the parent re-run alone (new prompt) | 17 |
| — by round 1 (width 4) | 71 |
| — by round 2 (width 3, only when round 1 failed) | 13 |
| unrescued | 33 |
| per-branch success rate (595 graded branches) | 45% |

Mechanism: map-reduce analysis. Every failed attempt is analysed separately
(failure mechanism, lesson, salvage, candidate fork steps); one reviewer reads
all analyses — parent included — and picks a base attempt + fork step + K
divergent hints. Branches start from the base's exact filesystem (snapshot as
image). **Correction (2026-09-04):** in this run the conversation was NOT cut
at the fork — `fork_session` resumed the parent's whole transcript (all steps
and its closing summary), so a branch "at step N" knew what the parent did
after N while its disk was at N. Round-1 forks here carried a median 32% of the
parent trajectory beyond the fork (70/117 ≥25%). The true-fork mechanism
(`resume_session_at`) exists since 2026-09-04 and was measured on DeepSWE (see
below); this SWE-bench run has not been repeated with it.

### Where forks succeed (595 branches)

| fork position in base trajectory | success |
|---|---:|
| 0–33% | 55% |
| 33–66% | 58% |
| 66–100% | 41% |

The analyst prompt said "later is better" (median chosen position: ~73%); the
data says the last third is the worst region. Not yet corrected in the prompt.

### By failure shape

| prior failure shape | n | rescue rate |
|---|---:|---:|
| target passed, regressions broke ("almost there") | 15 | **33%** |
| target failed, patch existed | 111 | 81% |
| near-empty / timeout | 6 | 83% |

"Almost there" is a local optimum, not almost there: every fork tried to
preserve the passing-but-wrong fix (zero early forks in this class), and paid
for it. Early-fork + redesign directions for this class are untested.

---

## Cost

| Phase | Agent runs | Cost | per instance |
|---|---:|---:|---:|
| single pass, old prompt (v500) | 500 | $171 | $0.34 |
| branching rescue (branch134) | 740 | $462 | $3.45/failed instance |
| analyst calls (~340 case analyses + reviews) | — | ~$30 *(est)* | — |
| **full pipeline (single + rescue)** | 1240 | **$663** | **$1.33** |
| single pass, new prompt (pass 2) | 500 | $194 | $0.39 |
| pass 3 + pass 4 (baseline experiment) | 1000 | $374 | $0.37 |

**The pipeline costs pass@3.9** (663 / (500 × $0.34)) and reached 93.0%. The
empirical pass@2 already shows why blind retries cannot match it at that
budget: two samples overlap on 349 solves and add only 37 — same-model retries
fail in correlated ways, and the verifier signal is exactly what breaks the
correlation. ~99% of input tokens were prompt-cache reads.

---

## Time

| Phase | runs | wall (8 workers) | agent compute | per run |
|---|---:|---:|---:|---:|
| v500 single | 500 | 3.1 h | 18.6 h | 2.2 min |
| branch134 | 740 | 7.5 h | 46.1 h | 3.7 min |
| pass 2 single | 500 | 3.3 h | 19.3 h | 2.3 min |

Branch runs are *not* slower than parents at equal difficulty — on the same 134
hard instances, parents average 4.2 min and branches 3.6 (a branch inherits the
parent's exploration). The apparent 68% branch premium was selection bias:
branches only exist on hard instances, and hard instances take ~2× regardless
(median 2.1 min vs 1.1 for the full set).

Grader re-runs are nearly free: all three regrade waves (149 verdicts flipped)
took ~20 minutes total, because grading restores a snapshot instead of
re-running an agent.

---

## Pending

- Analyst-prompt calibration from the fork-position and failure-shape tables.
- 33 unrescued instances' failure reports unexamined.


---

# DeepSWE (2026-09-04)

Generated by `scripts/deepswe_report.py runs/deepswe-final.json runs/deepswe-details.jsonl --extra-cost runs/deepswe`; inputs from `scripts/deepswe_aggregate.py runs/deepswe runs/deepswe-rerun1` and `scripts/deepswe_regrade_details.py`. Dataset `~/projects/LBP/deep-swe` @ `0b9fabbb`.

## DeepSWE (datacurve-ai/deep-swe, 113 tasks) — single attempt, no branching

**pass@1 = 31/113 = 27.4%**  (claude-code slot, `us.anthropic.claude-sonnet-4-6`, sandbox offline, 2 CPU / 8 GB, agent cap 10800 s, their verifier verbatim in a pristine offline microVM)

Not leaderboard-comparable: the leaderboard runs mini-swe-agent; this is Claude Code over Ash's two MCP tools.

### Cost and time (final-verdict runs only)

- agent cost: **$341 total**, mean $3.02 / task, median $2.64, max $7.91
- plus $223 spent on 14 runs whose verdict was discarded (infra-affected base runs, rerun cleanly)
- wall per task: mean 930 s, median 847 s, max 2353 s (cap 10800 s; 0 hit the cap)
- tool calls per task: mean 72, median 62, max 253

### By language

| language | tasks | resolved | pass@1 | mean cost |
|---|---|---|---|---|
| go | 34 | 6 | 17.6% | $2.87 |
| javascript | 5 | 1 | 20.0% | $2.24 |
| python | 34 | 9 | 26.5% | $2.86 |
| rust | 5 | 1 | 20.0% | $4.63 |
| typescript | 35 | 14 | 40.0% | $3.20 |

### Failure shape (from their reward.json, re-graded from snapshots)

| shape | tasks |
|---|---|
| resolved | 31 |
| near miss: >=90% of target tests pass | 39 |
| target tests all pass, regression(s) broke | 3 |
| partial: 50-90% of target tests pass | 31 |
| weak: <50% of target tests pass | 3 |
| no target test passes | 5 |
| agent error (run did not complete) | 1 |

Among failures with a verifier score, the target-test pass fraction has median 0.90 (quartiles 0.79 / 0.96); 71 of 82 failures pass every regression test.

### Integrity checks

- independent re-grade from snapshots: 0 verdict flips out of 113 re-graded
- runs with proxy-404 tool results in the FINAL verdict set: 0 (must be 0)
- agent runs that did not complete: kea-atomic-signal-selectors (error)

### Per task

| task | lang | verdict | f2p | p2p | cost | wall s | shape |
|---|---|---|---|---|---|---|---|
| anko-default-function-arguments | go | PASS | 2/2 | 119/119 | $3.85 | 1426 | resolved |
| awilix-async-container-initialization | typescript | PASS | 24/24 | 162/162 | $1.82 | 841 | resolved |
| bandit-interprocedural-taint-checks | python | PASS | 66/66 | 293/293 | $1.71 | 573 | resolved |
| cattrs-partial-structuring-recovery | python | PASS | 69/69 | 7/7 | $1.75 | 630 | resolved |
| claude-code-by-agents-recursive-delegation | typescript | PASS | 7/7 | 31/31 | $1.22 | 417 | resolved |
| drizzle-orm-window-function-builders | typescript | PASS | 130/130 | 566/566 | $3.14 | 596 | resolved |
| fd-deterministic-multi-key-sorting | rust | PASS | 43/43 | 109/109 | $2.06 | 461 | resolved |
| geo-shapeindex-serialization | go | PASS | 24/24 | 599/599 | $1.20 | 380 | resolved |
| happy-dom-abort-pending-body-reads | typescript | PASS | 14/14 | 165/165 | $4.04 | 1281 | resolved |
| happy-dom-deterministic-intersectionobserver | typescript | PASS | 14/14 | 9/9 | $1.95 | 670 | resolved |
| helm-unified-manifest-stream | go | PASS | 5/5 | 2/2 | $3.06 | 934 | resolved |
| httpx-deterministic-cookie-store | typescript | PASS | 115/115 | 1281/1281 | $1.41 | 430 | resolved |
| httpx-multipart-response-parsing | python | PASS | 122/122 | 1272/1272 | $1.47 | 488 | resolved |
| igel-persist-feature-schema | python | PASS | 24/24 | 2/2 | $1.79 | 500 | resolved |
| koota-entity-snapshot-rollback | python | PASS | 84/84 | 47/47 | $2.80 | 843 | resolved |
| koota-query-predicates | typescript | PASS | 43/43 | 172/172 | $3.27 | 1015 | resolved |
| narwhals-rolling-window-suite | python | PASS | 103/103 | 10093/10093 | $6.13 | 1271 | resolved |
| numba-stencil-boundary-modes | python | PASS | 29/29 | 827/827 | $5.90 | 2353 | resolved |
| obsidian-linter-link-format-conversion | typescript | PASS | 60/60 | 1131/1131 | $1.97 | 732 | resolved |
| ofetch-per-origin-circuit-breaker | typescript | PASS | 47/47 | 13/13 | $2.01 | 875 | resolved |
| opa-rego-rule-profiling | go | PASS | 25/25 | 6/6 | $1.68 | 521 | resolved |
| opa-template-string-reconstruction | go | PASS | 5/5 | 4/4 | $4.51 | 1467 | resolved |
| query-persist-restored-query-state | typescript | PASS | 8/8 | 42/42 | $1.95 | 581 | resolved |
| returns-validated-error-accumulation | python | PASS | 159/159 | 61/61 | $1.21 | 357 | resolved |
| skrub-duration-encoding | python | PASS | 130/130 | 2784/2784 | $2.88 | 852 | resolved |
| sql-formatter-bigquery-pipe-formatting | typescript | PASS | 26/26 | 5709/5709 | $3.10 | 946 | resolved |
| testem-per-launcher-reports | javascript | PASS | 65/65 | 469/469 | $0.92 | 308 | resolved |
| true-myth-iterable-collection-combinators | typescript | PASS | 96/96 | 561/561 | $1.42 | 264 | resolved |
| ts-pattern-match-each | typescript | PASS | 85/85 | 6/6 | $1.22 | 494 | resolved |
| vitest-duration-sharding | typescript | PASS | 56/56 | 24/24 | $2.05 | 430 | resolved |
| ytt-jsonpath-query-api | go | PASS | 103/103 | 1/1 | $3.41 | 1733 | resolved |
| abs-module-cache-flags | go | fail | 13/20 | 3/3 | $1.75 | 441 | partial: 50-90% of target tests pass |
| abs-stepped-slices | go | fail | 5/6 | 6/6 | $2.12 | 557 | partial: 50-90% of target tests pass |
| actionlint-action-pinning-lint | go | fail | 54/55 | 145/145 | $2.15 | 548 | near miss: >=90% of target tests pass |
| adaptix-name-mapping-aliases | python | fail | 42/44 | 2738/2738 | $4.89 | 1135 | near miss: >=90% of target tests pass |
| aiomonitor-task-snapshots-diff | python | fail | 41/53 | 8/8 | $1.52 | 356 | partial: 50-90% of target tests pass |
| anko-typed-variable-bindings | go | fail | 5/9 | 94/94 | $1.78 | 397 | partial: 50-90% of target tests pass |
| arcane-drift-detection-baselines | go | fail | 67/82 | 2/2 | $1.91 | 574 | partial: 50-90% of target tests pass |
| arktype-json-schema-refs-dependencies | typescript | fail | 23/25 | 1679/1679 | $3.77 | 1205 | near miss: >=90% of target tests pass |
| bandit-incremental-cache-control | python | fail | 83/88 | 275/275 | $1.71 | 515 | near miss: >=90% of target tests pass |
| bandit-structured-nosec-directives | python | fail | 68/69 | 282/282 | $2.20 | 712 | near miss: >=90% of target tests pass |
| boa-hierarchical-evaluation-cancellation | rust | fail | 15/17 | 7/7 | $6.27 | 1435 | partial: 50-90% of target tests pass |
| clack-async-autocomplete-options | typescript | fail | 71/82 | 643/643 | $3.76 | 1812 | partial: 50-90% of target tests pass |
| cliffy-config-file-parsing | typescript | fail | 35/37 | 451/451 | $2.81 | 757 | near miss: >=90% of target tests pass |
| csstree-shorthand-expansion-compression | javascript | fail | 71/79 | 16715/16715 | $3.08 | 957 | partial: 50-90% of target tests pass |
| dasel-html-document-format | go | fail | 132/146 | 1012/1012 | $1.87 | 568 | near miss: >=90% of target tests pass |
| dateutil-rfc5545-timezone-interop | python | fail | 53/67 | 2035/2035 | $1.30 | 357 | partial: 50-90% of target tests pass |
| dynamodb-toolbox-conditional-attribute-requirements | typescript | fail | 29/31 | 1267/1267 | $5.06 | 1103 | near miss: >=90% of target tests pass |
| dynamodb-toolbox-lazy-recursive-schemas | typescript | fail | 36/37 | 1267/1267 | $4.39 | 1129 | near miss: >=90% of target tests pass |
| effect-sse-httpapi-streaming | typescript | fail | 38/47 | 70/70 | $4.35 | 817 | partial: 50-90% of target tests pass |
| eicrud-keyset-pagination-cursor | typescript | fail | 0/14 | 163/168 | $3.18 | 1043 | no target test passes |
| etree-xml-diff-patch | go | fail | 51/52 | 15/15 | $2.20 | 1043 | near miss: >=90% of target tests pass |
| expr-try-catch-errors | go | fail | 73/79 | 66265/66265 | $7.05 | 1938 | near miss: >=90% of target tests pass |
| fastapi-deprecation-response-headers | python | fail | 127/137 | 3134/3134 | $5.12 | 830 | near miss: >=90% of target tests pass |
| fastapi-implicit-head-options | python | fail | 35/43 | 3134/3134 | $6.11 | 1432 | partial: 50-90% of target tests pass |
| go-critic-doc-link-checker | go | fail | 2/3 | 15/16 | $2.64 | 936 | partial: 50-90% of target tests pass |
| go-genai-streamed-function-args | go | fail | 3/6 | 62/62 | $2.17 | 748 | partial: 50-90% of target tests pass |
| go-git-worktree-merge-conflicts | go | fail | 14/17 | 2/2 | $5.22 | 1683 | partial: 50-90% of target tests pass |
| goreleaser-retry-publish-auditing | go | fail | 13/29 | 29/29 | $1.45 | 581 | weak: <50% of target tests pass |
| gql-incremental-graphql-delivery | python | fail | 15/17 | 810/811 | $2.23 | 611 | partial: 50-90% of target tests pass |
| helm-array-merge-strategies | go | fail | 35/47 | 12/12 | $2.41 | 889 | partial: 50-90% of target tests pass |
| httpx-streaming-json-iteration | python | fail | 105/108 | 1404/1404 | $1.54 | 448 | near miss: >=90% of target tests pass |
| ink-grid-box-layout | typescript | fail | 22/25 | 49/49 | $3.29 | 1014 | partial: 50-90% of target tests pass |
| ipython-session-bundle-replay | python | fail | 13/17 | 29/29 | $1.65 | 537 | partial: 50-90% of target tests pass |
| katex-multicolumn-array-spans | javascript | fail | 86/94 | 599/599 | $2.67 | 1138 | near miss: >=90% of target tests pass |
| kcp-go-multiplexed-kcp-streams | go | fail | 29/30 | 12/12 | $1.55 | 885 | near miss: >=90% of target tests pass |
| kea-atomic-signal-selectors | typescript | fail | 0/12 | 139/139 | $2.91 | 1986 | agent error (run did not complete) |
| kgateway-consistent-hash-policy | go | fail | 0/2 | 214/214 | $3.76 | 783 | no target test passes |
| kombu-single-active-consumer-priority | python | fail | 82/85 | 1421/1421 | $1.60 | 485 | near miss: >=90% of target tests pass |
| kombu-virtual-queue-dead-lettering | python | fail | 65/76 | 1412/1412 | $2.90 | 725 | partial: 50-90% of target tests pass |
| koota-composite-trait-aspects | typescript | fail | 47/51 | 172/172 | $2.63 | 761 | near miss: >=90% of target tests pass |
| koota-deferred-mutation-buffer | typescript | fail | 47/71 | 128/128 | $4.07 | 1225 | partial: 50-90% of target tests pass |
| koota-pair-relation-tracking | typescript | fail | 31/38 | 172/172 | $5.33 | 1566 | partial: 50-90% of target tests pass |
| kysely-window-grouping-helpers | typescript | fail | 0/254 | 22/22 | $3.84 | 693 | no target test passes |
| langchain-request-coalescing | python | fail | 48/50 | 232/232 | $2.29 | 847 | near miss: >=90% of target tests pass |
| mashumaro-flattened-dataclass-fields | python | fail | 64/66 | 30014/30014 | $4.24 | 1241 | near miss: >=90% of target tests pass |
| meriyah-explicit-resource-declarations | typescript | fail | 43/49 | 51468/51469 | $4.05 | 1551 | partial: 50-90% of target tests pass |
| mnamer-daemon-watch-lifecycle | python | fail | 51/51 | 316/319 | $1.74 | 777 | target tests all pass, regression(s) broke |
| mobly-grouped-test-barriers | python | fail | 70/79 | 808/808 | $3.00 | 1057 | partial: 50-90% of target tests pass |
| obsidian-linter-auto-table-of-contents | typescript | fail | 0/41 | 1131/1131 | $1.85 | 611 | no target test passes |
| obsidian-linter-scoped-ignore-markers | typescript | fail | 27/33 | 1132/1133 | $3.31 | 1395 | partial: 50-90% of target tests pass |
| onedump-dump-encryption-pipeline | go | fail | 76/82 | 6/6 | $1.58 | 558 | near miss: >=90% of target tests pass |
| optique-conditional-option-dependencies | typescript | fail | 35/36 | 2034/2034 | $6.36 | 1639 | near miss: >=90% of target tests pass |
| oxvg-structural-selector-preservation | rust | fail | 3/6 | 62/62 | $3.58 | 1417 | partial: 50-90% of target tests pass |
| participle-grammar-conflict-analysis | go | fail | 90/91 | 153/153 | $2.28 | 815 | near miss: >=90% of target tests pass |
| pebble-durability-wait-apis | go | fail | 25/59 | 44/44 | $4.31 | 1206 | weak: <50% of target tests pass |
| pest-character-class-coalescing | rust | fail | 100/104 | 250/250 | $3.33 | 1019 | near miss: >=90% of target tests pass |
| prometheus-transactional-reload-status | typescript | fail | 15/15 | 81/82 | $4.58 | 974 | target tests all pass, regression(s) broke |
| prometheus-typed-label-sorting | go | fail | 16/17 | 28/28 | $4.27 | 1374 | near miss: >=90% of target tests pass |
| psd-tools-blend-range-api | python | fail | 42/45 | 979/979 | $1.07 | 390 | near miss: >=90% of target tests pass |
| pwntools-tube-multiplexing | python | fail | 71/73 | 1/1 | $3.39 | 1220 | near miss: >=90% of target tests pass |
| python-statemachine-state-data-scoping | python | fail | 69/72 | 1286/1286 | $4.37 | 1306 | near miss: >=90% of target tests pass |
| quill-shared-toolbar-focus | typescript | fail | 8/13 | 22/22 | $3.98 | 1405 | partial: 50-90% of target tests pass |
| scc-bounded-memory-spilling | go | fail | 28/31 | 286/286 | $1.73 | 548 | near miss: >=90% of target tests pass |
| scriggo-method-declarations | go | fail | 41/48 | 1049/1049 | $6.99 | 1605 | partial: 50-90% of target tests pass |
| sqlfmt-create-table-ddl-formatting | python | fail | 30/32 | 1273/1273 | $4.76 | 1541 | near miss: >=90% of target tests pass |
| sqlite-utils-safe-import-checkpoints | python | fail | 59/60 | 1038/1038 | $4.43 | 1133 | near miss: >=90% of target tests pass |
| superjson-error-stack-serialization | typescript | fail | 76/80 | 114/116 | $2.18 | 974 | near miss: >=90% of target tests pass |
| task-task-graph-export | go | fail | 18/20 | 17/17 | $2.29 | 597 | near miss: >=90% of target tests pass |
| tengo-callable-instance-isolation | go | fail | 22/23 | 122/122 | $2.58 | 812 | near miss: >=90% of target tests pass |
| tengo-destructuring-bindings | go | fail | 86/91 | 132/132 | $4.11 | 1026 | near miss: >=90% of target tests pass |
| termenv-preserve-ansi-resets | go | fail | 20/35 | 87/87 | $2.32 | 1148 | partial: 50-90% of target tests pass |
| testem-bail-on-test-failure | javascript | fail | 88/90 | 489/489 | $2.62 | 627 | near miss: >=90% of target tests pass |
| textual-kitty-key-phases | python | fail | 19/23 | 57/57 | $2.32 | 1023 | partial: 50-90% of target tests pass |
| textual-richlog-follow-state | python | fail | 19/20 | 6/6 | $2.77 | 954 | near miss: >=90% of target tests pass |
| tomlkit-toml-table-converters | python | fail | 59/60 | 964/964 | $2.91 | 1079 | near miss: >=90% of target tests pass |
| updo-policy-alerting | go | fail | 15/17 | 123/123 | $1.60 | 473 | partial: 50-90% of target tests pass |
| valibot-recursive-schema-composition | typescript | fail | 2/10 | 209/209 | $5.58 | 2027 | weak: <50% of target tests pass |
| vulture-persistent-analysis-cache | python | fail | 23/24 | 291/295 | $1.71 | 610 | near miss: >=90% of target tests pass |
| wasmi-trap-coredumps | rust | fail | 22/22 | 57/58 | $7.91 | 1709 | target tests all pass, regression(s) broke |
| wazero-multi-module-snapshots | go | fail | 76/78 | 2/2 | $1.32 | 475 | near miss: >=90% of target tests pass |
| yaegi-go-embed-directives | go | fail | 0/38 | 0/58 | $4.53 | 1184 | no target test passes |
| yjs-map-conflict-detection | javascript | fail | 8/9 | 231/231 | $1.89 | 689 | partial: 50-90% of target tests pass |


---

# DeepSWE branching (2026-09-04)

Generated by `scripts/deepswe_branch_report.py runs/deepswe-branch --single runs/deepswe-final.json --details runs/deepswe-details.jsonl`. Analyst/reviewer calls (Bedrock Converse, sonnet 4.6) are not in the journals and are NOT in the cost figure. Resolved trajectories (journals + plans + ATIF): `runs/deepswe-resolved-trajectories.tar.gz` (`scripts/deepswe_export_resolved.py`).

## DeepSWE branching (`runs/deepswe-branch`): the 82 tasks the single pass failed

Recipe: recorded single-pass parent as base (no re-run), verifier-guided branching, 2 rounds, width 4 then 3; analyst = agent model. 82/82 tasks finished.

| | |
|---|---:|
| rescued | **58/82 = 70.7%** |
| — by round 1 | 41 |
| — by round 2 | 17 |
| unrescued | 24 |
| per-branch success rate (451 graded branches) | 36% |
| — round 1 branches | 120/328 = 37% |
| — round 2 branches | 42/123 = 34% |

Branch cost: **$649 total**, mean $1.44 / branch (median $1.20); mean wall 320 s, mean 25 tool calls.

**Combined: single pass 31 + rescued 58 = 89/113 = 78.8%** (single pass alone 31/113 = 27.4%).

### Where forks were placed (reviewer's chosen step / base trajectory length)

| fork position | rounds | round rescued |
|---|---:|---:|
| 0–33% | 2 | 50% |
| 33–66% | 16 | 56% |
| 66–100% | 105 | 46% |

median chosen position: 97%

### Rescue rate by single-pass failure shape

| prior failure shape | n | rescued |
|---|---:|---:|
| near miss (>=90% f2p) | 39 | 74% |
| partial (50-90%) | 31 | 77% |
| weak (<50%) | 9 | 22% |
| target all pass, regression broke | 3 | 100% |

### Unrescued

- clack-async-autocomplete-options
- dynamodb-toolbox-lazy-recursive-schemas
- effect-sse-httpapi-streaming
- eicrud-keyset-pagination-cursor
- httpx-streaming-json-iteration
- kea-atomic-signal-selectors
- kgateway-consistent-hash-policy
- koota-composite-trait-aspects
- koota-deferred-mutation-buffer
- koota-pair-relation-tracking
- kysely-window-grouping-helpers
- obsidian-linter-auto-table-of-contents
- obsidian-linter-scoped-ignore-markers
- optique-conditional-option-dependencies
- pebble-durability-wait-apis
- pest-character-class-coalescing
- pwntools-tube-multiplexing
- quill-shared-toolbar-focus
- scc-bounded-memory-spilling
- tengo-callable-instance-isolation
- termenv-preserve-ansi-resets
- textual-richlog-follow-state
- wazero-multi-module-snapshots
- yaegi-go-embed-directives



---

# DeepSWE branching, true fork (2026-09-04)

Same 82 tasks and recorded parents as the run above; three changes, nothing else: (1) the branch conversation is cut at the fork step (`resume_session_at` at the tool_result of step N; `conversation_cut`), (2) the branch is told about the fork as an environment note (`deepswe.bench.branch_note`, a `<system-reminder>` with verdict facts, analyst failure_reason/lesson, reviewer hint), (3) Claude Code out-of-band builtins (Task, Cron*, ScheduleWakeup, Workflow, SendMessage, Worktree, Skill, ReportFindings) denied. Old mechanism preserved at git tag `branching-fullconv-2026-09-04`; `--fork-full-conversation` reproduces it.

| | full-conversation fork | true fork |
|---|---:|---:|
| rescued | 58/82 = 70.7% | **70/82 = 85.4%** |
| — round 1 / round 2 | 41 / 17 | 52 / 18 |
| per-branch success | 36% (451) | **48% (418)** |
| branch cost | $649, $1.44 mean, 25 calls | $554, $1.33 mean, 22 calls |
| combined 113 | 89/113 = 78.8% | **101/113 = 89.4%** |

Verification: 414 of 418 branch transcripts contain none of their base's post-fork tool ids; the other 4 ran with the full conversation by design (parent auto-compacted before the fork step, recorded in `origin.cut_note`); 0 branches received foreign queued messages; 0 proxy 404s; 22 branches (main batch, pre-fix code) saw the transient "different event loop" error after a re-board. Generated by `scripts/deepswe_branch_report.py runs/deepswe-branch-truefork runs/deepswe-branch-truefork-fix --single runs/deepswe-final.json --details runs/deepswe-details.jsonl`. Trajectories: `runs/deepswe-resolved-trajectories-truefork.tar.gz`.

## DeepSWE branching (`runs/deepswe-branch-truefork, runs/deepswe-branch-truefork-fix`): the 82 tasks the single pass failed

Recipe: recorded single-pass parent as base (no re-run), verifier-guided branching, 2 rounds, width 4 then 3; analyst = agent model. 82/82 tasks finished.

| | |
|---|---:|
| rescued | **70/82 = 85.4%** |
| — by round 1 | 52 |
| — by round 2 | 18 |
| unrescued | 12 |
| branches run with the FULL conversation (parent compacted before the fork step) | 4 |
| per-branch success rate (418 graded branches) | 48% |
| — round 1 branches | 159/328 = 48% |
| — round 2 branches | 42/90 = 47% |

Branch cost: **$554 total**, mean $1.33 / branch (median $1.01); mean wall 323 s, mean 22 tool calls.

**Combined: single pass 31 + rescued 70 = 101/113 = 89.4%** (single pass alone 31/113 = 27.4%).

### Where forks were placed (reviewer's chosen step / base trajectory length)

| fork position | rounds | round rescued |
|---|---:|---:|
| 0–33% | 6 | 83% |
| 33–66% | 19 | 63% |
| 66–100% | 87 | 61% |

median chosen position: 96%

### Rescue rate by single-pass failure shape

| prior failure shape | n | rescued |
|---|---:|---:|
| near miss (>=90% f2p) | 39 | 87% |
| partial (50-90%) | 31 | 90% |
| weak (<50%) | 9 | 56% |
| target all pass, regression broke | 3 | 100% |

### Unrescued

- dynamodb-toolbox-lazy-recursive-schemas
- eicrud-keyset-pagination-cursor
- helm-array-merge-strategies
- httpx-streaming-json-iteration
- kea-atomic-signal-selectors
- kgateway-consistent-hash-policy
- koota-deferred-mutation-buffer
- kysely-window-grouping-helpers
- pest-character-class-coalescing
- pwntools-tube-multiplexing
- quill-shared-toolbar-focus
- tengo-callable-instance-isolation

