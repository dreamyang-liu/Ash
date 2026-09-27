# Run DeepSWE

Prepare the task sources from [DeepSWE](https://github.com/datacurve-ai/deep-swe), or use your existing copy.
Keep the task files and their verifiers unchanged. The runner hashes all selected
task files before executing. Use the identical cohort for all four runs.

```bash
python -m experiments.qwen.run --dataset deepswe --method all \
  --tasks-dir /data/deep-swe/tasks --task superjson-error-stack-serialization \
  --runtime-bin runtime/ash-runtime --api-key-file /path/to/aenv-key \
  --workers 16 --output /data/runs/deepswe-all
python -m experiments.qwen.report /data/runs/deepswe-all --method all \
  --save /data/runs/deepswe-all/comparison.json
```

Set `QWEN_BASE_URL`, `QWEN_API_KEY` and `AENV_SERVER_URL` outside Git first.
Use `--workers 1` for a smoke test. Add `--plan` for a no-execution preflight.
Defaults are Qwen3.8-27b, high, four baseline samples and an eight-sample branch cap.
For separate runs, use the method folders below and `--baseline-root` for the
three branch methods. Failed initial checkpoints are retained; no deletion runs.
The report's `comparison` array gives pass@1, pass@4, BPO, Shepherd and SPROUT
with Resolve Rate, Recovery, Steps, Tokens (M) and completion coverage.
