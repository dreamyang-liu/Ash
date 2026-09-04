#!/usr/bin/env bash
# DeepSWE, all 113 tasks, single attempt each, sharded across N workers.
#
#   scripts/run_deepswe.sh                 # 16 workers -> runs/deepswe
#   WORKERS=8 OUT=runs/deepswe-b scripts/run_deepswe.sh
#
# Agreement (STATUS.md, 2026-09-03): --rounds 0 (no branching yet), claude-code
# + sonnet 4.6, --timeout 10800 (the dataset's own agent limit), sandbox
# offline and 2 CPU / 8 GB (from task.toml; the deepswe benchmark adapter
# applies both). Shards are round-robin over the sorted task ids so every
# worker gets a mix of languages. Run detached:
#
#   setsid nohup scripts/run_deepswe.sh > runs/deepswe/chain.log 2>&1 < /dev/null &
#
# Monitor:  python3.11 scripts/deepswe_progress.py runs/deepswe
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-16}"
OUT="${OUT:-runs/deepswe}"
TASKS_DIR="${TASKS_DIR:-$HOME/projects/LBP/deep-swe/tasks}"
MODEL="${MODEL:-us.anthropic.claude-sonnet-4-6}"
TIMEOUT="${TIMEOUT:-10800}"
PY="${PY:-python3.12}"

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('$HOME/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

mkdir -p "$OUT"
git -C "$(dirname "$TASKS_DIR")" rev-parse HEAD > "$OUT/dataset-commit.txt"

# shard ids round-robin
$PY - "$TASKS_DIR" "$OUT" "$WORKERS" <<'EOF'
import pathlib, sys
from deepswe.tasks import load_tasks
tasks_dir, out, workers = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
ids = [t.task_id for t in load_tasks(tasks_dir)]
for k in range(workers):
    (out / f"shard-{k}.txt").write_text(",".join(ids[k::workers]))
print(f"{len(ids)} tasks over {workers} shards")
EOF

for k in $(seq 0 $((WORKERS - 1))); do
  setsid nohup "$PY" -u -m swebench.fork_eval \
    --benchmark deepswe --tasks-dir "$TASKS_DIR" \
    --instance "$(cat "$OUT/shard-$k.txt")" \
    --slot claude-code --model "$MODEL" --analyst-model "$MODEL" \
    --rounds 0 --timeout "$TIMEOUT" \
    -o "$OUT/shard-$k" > "$OUT/shard-$k.log" 2>&1 < /dev/null &
  echo "shard $k pid $!"
  sleep 3
done
echo "launched $WORKERS workers -> $OUT"
