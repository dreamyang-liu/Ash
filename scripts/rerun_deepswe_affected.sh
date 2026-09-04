#!/usr/bin/env bash
# Rerun the DeepSWE tasks whose runs were infra-affected (proxy 404 after a
# re-board, Failure log #7) with the fixed code. Same settings as
# run_deepswe.sh; output goes to a separate batch dir so the base batch stays
# intact and scripts/deepswe_aggregate.py can layer the reruns on top.
#
#   scripts/rerun_deepswe_affected.sh                # BASE=runs/deepswe OUT=runs/deepswe-rerun1
#   BASE=runs/deepswe-rerun1 OUT=runs/deepswe-rerun2 scripts/rerun_deepswe_affected.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-runs/deepswe}"
OUT="${OUT:-runs/deepswe-rerun1}"
WORKERS="${WORKERS:-8}"
TASKS_DIR="${TASKS_DIR:-$HOME/projects/LBP/deep-swe/tasks}"
MODEL="${MODEL:-us.anthropic.claude-sonnet-4-6}"
TIMEOUT="${TIMEOUT:-10800}"
PY="${PY:-python3.12}"

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('$HOME/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

mkdir -p "$OUT"
python3.11 - "$BASE" "$OUT" "$WORKERS" <<'EOF'
import glob, json, pathlib, sys
base, out, workers = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
affected = sorted({p.split("/")[-2] for p in glob.glob(str(base / "**" / "parent.jsonl"), recursive=True)
                   if any('"tool.finished"' in l and "Client error '404" in l for l in open(p))})
(out / "affected.txt").write_text("\n".join(affected) + "\n")
workers = max(1, min(workers, len(affected)))
for k in range(workers):
    (out / f"shard-{k}.txt").write_text(",".join(affected[k::workers]))
print(f"{len(affected)} affected task(s) over {workers} shard(s): {affected}")
(out / "workers").write_text(str(workers))
EOF

WORKERS=$(cat "$OUT/workers")
for k in $(seq 0 $((WORKERS - 1))); do
  [ -s "$OUT/shard-$k.txt" ] || continue
  setsid nohup "$PY" -u -m swebench.fork_eval \
    --benchmark deepswe --tasks-dir "$TASKS_DIR" \
    --instance "$(cat "$OUT/shard-$k.txt")" \
    --slot claude-code --model "$MODEL" --analyst-model "$MODEL" \
    --rounds 0 --timeout "$TIMEOUT" \
    -o "$OUT/shard-$k" > "$OUT/shard-$k.log" 2>&1 < /dev/null &
  echo "shard $k pid $!"
  sleep 3
done
echo "launched $WORKERS rerun workers -> $OUT"
