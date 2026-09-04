# Prefix / Checkpoint Search README

## What to use

The storage/search layer is responsible for:

- shared prefix storage;
- exact checkpoint lookup;
- proof-based environment-state lookup;
- pre-snapshot zero-diff deduplication;
- checkpoint restore metadata;
- cache-hit / snapshot / storage / latency accounting.

It is **not** the branch-selection policy.

## Main files

| File | Purpose |
|---|---|
| `sdk/ash_sandbox/prefix_index.py` | Persistent exact Prefix DAG / hash-chain index. |
| `sdk/ash_sandbox/relaxed_prefix.py` | Conservative command-level environment-effect proof. |
| `sdk/ash_sandbox/relaxed_change_index.py` | Relaxed state-equivalent index. |
| `sdk/ash_sandbox/trajectory_cache.py` | Checkpoint cache lookup / registration integration. |
| `scripts/compare_prefix_storage_baselines.py` | Classic exact, adjacent zero-diff, full-state oracle, ours. |
| `scripts/benchmark_search_storage_baselines.py` | TVCACHE-style baselines, unsafe ablations, safety suite, index microbenchmark. |
| `scripts/benchmark_cross_model_cache_baselines.py` | Cross-rollout / model-shift global reuse. |
| `scripts/benchmark_relaxed_lookup_table.py` | Actual SQLite Exact-vs-Relaxed lookup latency. |
| `benchmarks/bench_snapshot_storage_delta.py` | Real Docker commit time + writable-layer bytes. |

## Reproduce the final table

```bash
cd ~/Desktop/rl-infra/Ash

PYTHONPATH=sdk:. .venv/bin/python scripts/compare_prefix_storage_baselines.py \
  --audit-json results/relaxed-search-dense-final/real_trajectory_search.json \
  --output-dir results/prefix-storage-baselines/qwen38-full-swemarathon

PYTHONPATH=sdk:. .venv/bin/python scripts/benchmark_search_storage_baselines.py \
  --audit-json results/relaxed-search-dense-final/real_trajectory_search.json \
  --output-dir results/prefix-baseline-comparison \
  --index-depth 3000

PYTHONPATH=sdk:. .venv/bin/python scripts/benchmark_cross_model_cache_baselines.py

.venv/bin/python benchmarks/bench_snapshot_storage_delta.py --repeats 5
```

## Main numbers

### Within one long rollout

9 real Qwen3.8-27B SWE-Marathon trajectories, 895 tool boundaries:

| Method | Hits / avoided | Snapshots | Reduction |
|---|---:|---:|---:|
| No reuse | 0 | 895 | 0.00% |
| Exact / rolling / TVCACHE LPM | 0 | 895 | 0.00% |
| TVCACHE Stateful (coarse/static) | 19 | 876 | 2.12% |
| Adjacent zero-diff | **186** | **709** | **20.78%** |
| TVCACHE per-call oracle | **186** | **709** | **20.78%** |
| **Ours** | **186** | **709** | **20.78%** |

The 20.78% result is **not uniquely caused by global search**. On these single trajectories the useful states are adjacent zero-diff repeats, so simple adjacent coalescing ties the global index.

### Why keep the global index

The global index matters across trajectories / branches / pass@N:

- dense same-model `find-network-alignments` A→B: Exact 0, Relaxed **2/4 = 50%**;
- B→A: Exact 0, Relaxed **2/4 = 50%**;
- Qwen cache → Luna queries across 9 tasks: Exact/TVCACHE-LPM/static-TVCACHE 0, Ours **2/36 = 5.56%**;
- Luna cache → Qwen queries: Exact/TVCACHE-LPM/static-TVCACHE 0, Ours **2/56 = 3.57%**.

This is the intended RL / repeated-rollout workload.

## TVCACHE comparison

TVCACHE already supports stateful prefix matching and may skip calls marked state-preserving. Therefore the difference is **not simply read-only filtering**.

A SWE agent commonly exposes a generic `shell` tool containing both read-only and mutating invocations. Coarse tool-level annotation cannot safely declare `shell` stateless. The Ash classifier instead proves individual invocations conservatively. On the current trajectories it matches a TVCACHE oracle that is given perfect per-call mutation labels, without requiring those labels manually.

## Safety

Examples automatically accepted as state-preserving include conservative forms of:

```text
ls
cat
grep / rg
cat | head
cd /app && rg ...
sed -n ...
find ... -name ...
```

Examples rejected as barriers include:

```text
echo x > file
sed -i ...
find ... -delete
tail -f ...
rg --pre ...
python scripts
build / test commands
```

A naive command-name-only classifier appears to obtain 73.41% reduction, but it incorrectly treats several mutating forms above as read-only and is therefore only an unsafe ablation.

## Actual storage result

Five-repeat Docker probe:

| State before commit | Median commit | Mean commit | Writable layer |
|---|---:|---:|---:|
| Clean | 514 ms | 728 ms | 0 B |
| Proven read-only | 414 ms | 442 ms | **0 B** |
| One-byte mutation | 342 ms | 382 ms | 12.3 kB |

Across 18 retained controlled snapshots, writable-layer delta is **24.6 kB median / 2.79 MB mean**.

Do not convert snapshot-count reduction into the same percentage of disk savings. Zero-diff snapshots may add 0 filesystem bytes while still costing hundreds of milliseconds to create/manage.

## Reporting boundary

Environment snapshot reuse never implies model-state reuse:

```text
model_prefix_reusable = false
kv_reuse = false
```

For the controlled lightstage runs, official task instructions and agent-visible workspaces are used, but not the full verifier/toolchain environment. Those runs are systems-mechanism validation, not solve-rate results.
