# Run Qwen experiments

Use Python 3.12+ and an existing AgentENV server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/qwen/requirements.txt
pip install ./sdk
export PYTHONPATH=.:sdk
(cd runtime && go build -o ash-runtime .)
export QWEN_BASE_URL=http://your-private-model-endpoint
read -rs -p 'Qwen key: ' QWEN_API_KEY; export QWEN_API_KEY
export AENV_SERVER_URL=http://127.0.0.1:8000
python -m experiments.qwen.run --dataset deepswe --method all \
  --tasks-dir /data/deep-swe/tasks --task superjson-error-stack-serialization \
  --runtime-bin runtime/ash-runtime --api-key-file /path/to/aenv-key \
  --workers 1 --output /data/runs/qwen-smoke
python -m experiments.qwen.report /data/runs/qwen-smoke --method all
```

For a full cohort, provide `--selection cohort.json` instead of `--task` and
use `--workers 16`. Add `--plan` to validate inputs without inference.
See [protocol and output fields](experiments/qwen/PROTOCOL.md).
