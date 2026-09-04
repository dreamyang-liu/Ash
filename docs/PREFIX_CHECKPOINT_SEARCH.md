# Prefix / Checkpoint Storage & Search

## Scope

This layer stores and searches reusable **environment checkpoints** for long-horizon branching. It does not choose branch points and does not reuse another trajectory's model history/KV cache.

```text
model_prefix_reusable = false
kv_reuse = false
```

Core responsibilities:

1. content-addressed Prefix DAG / hash-chain storage;
2. exact longest-prefix checkpoint lookup;
3. conservative state-equivalent lookup;
4. pre-snapshot zero-diff deduplication;
5. persistent checkpoint metadata and restore integration;
6. cache-hit, snapshot, storage, and lookup accounting.

## Safety model

Relaxed matching projects out only operations that are **proven** not to change the environment.

Supported conservative reads include `grep_files`, `text_editor view/read`, and restricted shell forms of `pwd`, `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `rg`, `stat`, `file`, `sort`, `uniq`, `tr`, `echo`, print-only `sed`, read-only `find`, `which`, and controlled `cd` / pipelines / `&&` composition.

Examples that remain barriers include:

- write redirection;
- `sed -i`;
- `find -delete/-exec`;
- `tail -f`;
- `rg --pre`;
- shell substitution/background execution;
- arbitrary Python/scripts/build/test/package commands.

Mutation/unknown events retain an observed-result digest in the projected state key.

## Baselines

The current benchmark suite covers:

- No reuse.
- Linear exact-prefix scan.
- Hash / exact-prefix DAG.
- Rolling-hash exact prefix.
- TVCACHE LPM / full tool sequence.
- TVCACHE Stateful with coarse/static tool annotations.
- Adjacent zero-diff coalescing.
- Oracle full-state fingerprint cache.
- TVCACHE Stateful with oracle per-call mutation labels.
- Stateless `(tool_name, args)` cache as an unsafe diagnostic.
- Naive read-command classification as an unsafe ablation.
- Ours: proof-based global state-equivalent checkpoint index.

Classic data-structure microbenchmarks additionally compare a flat full-prefix list, SHA-256 HashMap, Trie/Tool-Call Graph, and the persistent SQLite hash-chain DAG.

## Full SWE-Marathon result

9 real Qwen3.8-27B trajectories, 895 tool boundaries:

| Method | Reusable hits | Physical snapshots | Reduction |
|---|---:|---:|---:|
| No reuse | 0 | 895 | 0.00% |
| Exact / rolling / TVCACHE LPM | 0 | 895 | 0.00% |
| TVCACHE Stateful — static | 19 | 876 | 2.12% |
| Adjacent zero-diff | **186** | **709** | **20.78%** |
| Full-state oracle | **186** | **709** | **20.78%** |
| TVCACHE Stateful — per-call oracle | **186** | **709** | **20.78%** |
| **Ours** | **186** | **709** | **20.78%** |

### Interpretation

The 20.78% within-rollout result is a **zero-diff/coalescing result**, not a unique benefit of global search: on these trajectories, all reusable states found by the global index were adjacent to the previous equivalent state.

The global index is justified by **cross-rollout reuse**. In dense same-task Qwen runs, Exact gives 0 hits while Relaxed gives 2/4 = 50% in both A→B and B→A directions. In the Qwen↔Luna controlled policy-shift test over the same 9 tasks, Exact/LPM/static-TVCACHE give 0 cross-model hits while the state-equivalent index retains 2 hits in each direction.

TVCACHE already supports stateful prefix matching when developers provide state-preserving annotations. Our defensible distinction is automatic conservative **per-invocation shell proof** for a generic shell interface plus persistent environment-checkpoint indexing; the automatic method matches the manually annotated per-call TVCACHE oracle on the current traces.

## Cross-model controlled results

| Model | Tasks | Tool boundaries | TVCACHE static | TVCACHE oracle | Ours | Snapshots | Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 9 | 56 | 3 | 22 | **22** | 56→**34** | **39.29%** |
| Luna Max | 9 | 36 | 2 | 3 | **3** | 36→**33** | **8.33%** |

Actual SQLite lookup on these controlled traces:

- Qwen: Exact 0.351 ms, Relaxed 0.618 ms.
- Luna: Exact 0.461 ms, Relaxed 0.721 ms.

The controlled lightstage runs use official SWE-Marathon task instructions and agent-visible workspaces but not the full verifier/toolchain environment, so they are mechanism validation only, not solve-rate claims.

## Storage accounting

Fresh Docker probe, five repetitions per condition:

| State before commit | Median commit | Mean commit | Writable layer |
|---|---:|---:|---:|
| Clean | 514 ms | 728 ms | 0 B |
| Proven read-only | **414 ms** | **442 ms** | **0 B** |
| One-byte mutation | 342 ms | 382 ms | 12.3 kB |

Across 18 retained controlled snapshots, the writable-layer delta is 24.6 kB median and 2.79 MB mean.

Do **not** translate a snapshot-count reduction directly into the same percentage of disk-byte savings. Zero-diff snapshots can add 0 B but still cost hundreds of milliseconds to commit. Report snapshot count, wall time, writable bytes, and lookup latency separately.

## Reproduce

```bash
cd ~/Desktop/rl-infra/Ash

# Classic + state-aware baselines on 9 full SWE-Marathon trajectories.
PYTHONPATH=sdk:. .venv/bin/python scripts/compare_prefix_storage_baselines.py \
  --audit-json results/relaxed-search-dense-final/real_trajectory_search.json \
  --output-dir results/prefix-storage-baselines/qwen38-full-swemarathon

# TVCACHE-style, unsafe ablations, safety suite, and index microbenchmark.
PYTHONPATH=sdk:. .venv/bin/python scripts/benchmark_search_storage_baselines.py \
  --audit-json results/relaxed-search-dense-final/real_trajectory_search.json \
  --output-dir results/prefix-baseline-comparison \
  --index-depth 3000

# Cross-model / policy-shift global-cache test.
PYTHONPATH=sdk:. .venv/bin/python scripts/benchmark_cross_model_cache_baselines.py

# Actual Docker commit + writable-layer measurement.
.venv/bin/python benchmarks/bench_snapshot_storage_delta.py --repeats 5
```

## Related systems

The most direct recent baseline is **TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents** (arXiv:2602.10986), which builds a Tool Call Graph, performs longest-prefix matching, and supports state-preserving annotations.

Model-side KV caches such as vLLM Automatic Prefix Caching and LMCache/CacheBlend are related but orthogonal: they cache transformer KV/prefill state, while this layer caches executable environment checkpoints.
