# Run TerminalBench 2.1

Prepare the task sources from [TerminalBench 2.1](https://github.com/harbor-framework/terminal-bench-2-1), or use your existing copy.
Keep the task files and their verifiers unchanged. The runner hashes all selected
task files before executing. Use the identical cohort for all four runs.

```bash
python -m experiments.qwen.run --dataset terminalbench21 --method all \
  --tasks-dir /data/terminal-bench-2-1/tasks --task largest-eigenval \
  --runtime-bin runtime/ash-runtime --api-key-file /path/to/aenv-key \
  --workers 16 --output /data/runs/terminalbench21-all
python -m experiments.qwen.report /data/runs/terminalbench21-all --method all \
  --save /data/runs/terminalbench21-all/comparison.json
```

Set `QWEN_BASE_URL`, `QWEN_API_KEY` and `AENV_SERVER_URL` outside Git first.
Use `--workers 1` for a smoke test. Add `--plan` for a no-execution preflight.
Defaults are Qwen3.8-27b, high, four baseline samples and an eight-sample branch cap.
For separate runs, use the method folders below and `--baseline-root` for the
three branch methods. Failed initial checkpoints are retained; no deletion runs.
The report's `comparison` array gives pass@1, pass@4, BPO, Shepherd and SPROUT
with Resolve Rate, Recovery, Steps, Tokens (M) and completion coverage.
