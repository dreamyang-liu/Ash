# Official SWE-bench re-verification

The recorded Verified 500 exports are evaluated with the installed official
SWE-bench Docker harness. `scripts/official_swebench.py` owns resumption and
coverage accounting; `scripts/official_swebench_runner.py` isolates the installed
package from Ash's identically named `swebench` package. Official patch application,
eval scripts and grading rules are unchanged. This does not replace the local
`fork_eval` grader or recompute historical branch decisions.

The existing run driver consumes both helpers:

```bash
python3.12 runs/swe-verified500-official-reverify-20260908/official_reverify.py --phase summarize
```

Summarizing reads existing artifacts only; it does not start Docker or agents.
The expected cohort comes from the source manifests, including tasks absent from
the export. `official-attempt-results.jsonl` includes every exported attempt:

- `completed`: a valid official report supplies `official_resolved`.
- `empty_patch`: unresolved, but no official test report is expected.
- `missing_report`: unmeasured, with an error-log path when available. This may
  mean a harness failure or unfinished evaluation; it is not automatically an
  infrastructure diagnosis.
- `export_error`: unmeasured; exporting the artifact must be repaired separately.

`parent_submitted` counts nonempty predictions, `parent_completed` counts reports,
and `parent_unresolved` counts negative reports only. Empty patches have their own
count. `parent_final_rate` stays null until every expected parent has a known
outcome. `parent_resolved_total_lower_bound` uses the whole cohort;
`parent_resolved_completed_rate` is a diagnostic on a potentially biased subset,
not a replacement benchmark score. Branch and combined successes remain partial
while recorded attempts are incomplete. No historical official reports are edited.

To resume just the missing parent reports, when evaluation is to be resumed:

```bash
python3.12 runs/swe-verified500-official-reverify-20260908/official_reverify.py \
  --phase retry --batches parent --prepare-workers 4 --retry-workers 8 --retry-rounds 2
```

`--phase evaluate` uses 32 test workers initially; `--phase retry` starts at 8.
Both skip existing positive and negative reports. Omit `--batches` to include
parent and every branch slot (including slots not yet started). Before each
pass, missing images are pulled or built with four workers, existing images are
reused, and all preparation must succeed before tests start. Instance images are
retained by default. Container creation is separately capped at four concurrent
Docker calls (`--startup-workers`), while tests can run with 32 workers.
Two additional passes retry only predictions still lacking
reports, at eight workers. These limits are configurable and have not yet been
validated by a new live evaluation of this batch.

Each pass saves its exact pending predictions, preparation records and separate
logs; `official/retry-runs.jsonl` is append-only. A zero harness process exit does
not establish completion: remaining missing reports cause a nonzero driver exit
after the retry budget, and a summary is still written. Corrupt reports stop
resumption for inspection. Cleanup only targets containers belonging to the
current batch, has a 60-second command timeout, and stops on stuck removals.
It never restarts Docker or cleans unrelated runs.

Validation (no live evaluation):

```bash
PYTHONPATH=.:sdk python3.12 -m pytest swebench/tests/test_official_reverify.py -q
```
