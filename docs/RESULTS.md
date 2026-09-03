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
divergent hints. Branches inherit the base's conversation up to the fork
(`fork_session`) and its exact filesystem (snapshot as image).

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
