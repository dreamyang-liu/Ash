# Run SWE-bench-Pro v2 HARD-51

Prepare the task sources from [SWE-bench-Pro v2 HARD-51](https://github.com/scaleapi/SWE-bench_Pro-os), or use your existing copy.
Keep the task files and their verifiers unchanged. The runner hashes all selected
task files before executing. Use the identical cohort for all four runs.

```bash
python -m experiments.qwen.run --dataset swebenchpro-v2-hard --method all \
  --tasks-dir /data/SWE-bench_Pro-os/v2/tasks --selection experiments/qwen/swebenchpro-v2-hard/selection.json \
  --runtime-bin runtime/ash-runtime --api-key-file /path/to/aenv-key \
  --workers 16 --output /data/runs/swebenchpro-v2-hard-all
python -m experiments.qwen.report /data/runs/swebenchpro-v2-hard-all --method all \
  --save /data/runs/swebenchpro-v2-hard-all/comparison.json
```

Set `QWEN_BASE_URL`, `QWEN_API_KEY` and `AENV_SERVER_URL` outside Git first.
Use `--workers 1` for a smoke test. Add `--plan` for a no-execution preflight.
Defaults are Qwen3.8-27b, high, four baseline samples and an eight-sample branch cap.
For separate runs, use the method folders below and `--baseline-root` for the
three branch methods. Failed initial checkpoints are retained; no deletion runs.
The report's `comparison` array gives pass@1, pass@4, BPO, Shepherd and SPROUT
with Resolve Rate, Recovery, Steps, Tokens (M) and completion coverage.
