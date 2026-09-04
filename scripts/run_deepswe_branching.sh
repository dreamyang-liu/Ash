#!/usr/bin/env bash
# DeepSWE branching round: the tasks the single pass failed, re-run through the
# full loop -- fresh parent attempt, then verifier-guided branching, width 4
# then 3 (the SWE-bench branch134 recipe, per the user: "按照同样的来吧，先4再3").
#
#   scripts/run_deepswe_branching.sh                       # 20 workers -> runs/deepswe-branch
#   FINAL=runs/deepswe-final.json WORKERS=16 OUT=... scripts/run_deepswe_branching.sh
#
# Failed = `resolved: false` in FINAL (scripts/deepswe_aggregate.py output).
# Same model for agent and analyst (decision 2026-09-01). Detached:
#   setsid nohup scripts/run_deepswe_branching.sh > runs/deepswe-branch/chain.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."

FINAL="${FINAL:-runs/deepswe-final.json}"
OUT="${OUT:-runs/deepswe-branch}"
# 32 per the user (2026-09-04). One sandbox per worker at a time.
WORKERS="${WORKERS:-32}"
# The recorded single-pass parent is the base for branching; no parent re-run
# (user: "你为啥还要跑parent" -- the prompt is unchanged, so a fresh parent is a
# blind retry, not a branch). FINAL names each task's final journal.
PARENT_FROM="${PARENT_FROM:-$FINAL}"
TASKS_DIR="${TASKS_DIR:-$HOME/projects/LBP/deep-swe/tasks}"
MODEL="${MODEL:-us.anthropic.claude-sonnet-4-6}"
TIMEOUT="${TIMEOUT:-10800}"
ROUNDS="${ROUNDS:-2}"
BRANCHES="${BRANCHES:-4,3}"
PY="${PY:-python3.12}"

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('$HOME/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

mkdir -p "$OUT"
git -C "$(dirname "$TASKS_DIR")" rev-parse HEAD > "$OUT/dataset-commit.txt"
cp "$FINAL" "$OUT/single-pass-final.json"

python3.11 - "$FINAL" "$OUT" "$WORKERS" <<'EOF'
import json, pathlib, sys
final, out, workers = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2]), int(sys.argv[3])
failed = sorted(t["task"] for t in final["tasks"] if not t["resolved"])
(out / "failed.txt").write_text("\n".join(failed) + "\n")
workers = max(1, min(workers, len(failed)))
for k in range(workers):
    (out / f"shard-{k}.txt").write_text(",".join(failed[k::workers]))
(out / "workers").write_text(str(workers))
print(f"{len(failed)} failed task(s) over {workers} shards")
EOF

WORKERS=$(cat "$OUT/workers")
for k in $(seq 0 $((WORKERS - 1))); do
  [ -s "$OUT/shard-$k.txt" ] || continue
  setsid nohup "$PY" -u -m swebench.fork_eval \
    --benchmark deepswe --tasks-dir "$TASKS_DIR" \
    --instance "$(cat "$OUT/shard-$k.txt")" \
    --slot claude-code --model "$MODEL" --analyst-model "$MODEL" \
    --rounds "$ROUNDS" --branches "$BRANCHES" --timeout "$TIMEOUT" \
    --parent-from "$PARENT_FROM" \
    -o "$OUT/shard-$k" > "$OUT/shard-$k.log" 2>&1 < /dev/null &
  echo "shard $k pid $!"
  sleep 3
done
echo "launched $WORKERS branching workers -> $OUT (rounds=$ROUNDS branches=$BRANCHES)"
