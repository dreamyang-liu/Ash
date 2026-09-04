# Checkpoint State Reuse: 10-Task Results

This is the compact, report-ready summary for the checkpoint search and
storage experiments. Generated artifacts remain under `results/` and are not
versioned.

## Protocol

- 10 SWE-Marathon tasks.
- 28 independent Qwen/Luna rollouts.
- 134 complete recorded tool boundaries.
- 52 ordered source-to-query rollout pairs within the same task.
- 248 cross-rollout queries.
- Every reported hit was checked against an independently replayed Docker
  workspace digest.

## Main comparison

| Method | Role/source | Cross-rollout hit rate | Restore precision | Online decision cost | Retained candidates |
|---|---|---:|---:|---:|---:|
| No Cache | control | 0/248 (0.0%) | N/A | 0 | 134/134 |
| Exact Full History | strict-match control | 2/248 (0.8%) | 100% | 0.0003 ms | 133/134 |
| Official TVCACHE TCG | [paper](https://arxiv.org/abs/2602.10986), [official code](https://github.com/TVCache/TVCache) | 8/248 (3.2%) | 100% | 0.0017 ms | 133/134 |
| Full-Workspace Hash | scan-based upper-bound baseline | 91/248 (36.7%) | 100% | 79.1 ms scan | 38/134 |
| **Relaxed Projection (ours)** | scan-free | **35/248 (14.1%)** | **100%** | **0.201 ms** | **82/134** |

Relaxed Projection improves over official TVCACHE by 10.9 percentage points;
the task-cluster bootstrap 95% confidence interval for that difference is
[1.6, 20.5] percentage points. On 184 cross-model queries, Exact and TVCACHE
both have zero hits while Relaxed Projection has 18/184 (9.8%). The scoped
claim is **best among the evaluated scan-free methods**, not an unconditional
highest hit rate: full-workspace scanning finds more matches at much higher
online cost.

## Real Docker measurements

| Measurement | Baseline | Relaxed/checkpoint result |
|---|---:|---:|
| Five-branch Exact miss plus replay vs restore | 13.09 s | 1.39 s (9.41x faster) |
| Zero-diff checkpoint commit | N/A | 414 ms median despite 0 B writable delta |
| 18 retained Docker snapshots | N/A | 24.6 kB median; 50.14 MB total |

`134 -> 82` is a dense-checkpoint-policy simulation over real tool boundaries:
it counts how many distinct checkpoint entries would need to be retained. It
does **not** mean that 134 Docker images were physically created in this run.
Physical byte measurements use the separate 18-snapshot experiment.

## Reproduction

```bash
PYTHONPATH=sdk:. python scripts/build_paper_cache_corpus.py
PYTHONPATH=sdk:. python scripts/evaluate_paper_cache_baselines.py \
  --tvcache-root /path/to/TVCache --workers 4 --force-replay
PYTHONPATH=sdk:. python scripts/summarize_paper_cache_baselines.py
```

The corpus builder expects the locally generated rollout artifacts documented
in `docs/PREFIX_CHECKPOINT_SEARCH_README.md`. Results are intentionally ignored
by Git.
