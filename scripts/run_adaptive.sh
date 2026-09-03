#!/usr/bin/env bash
# Adaptive blind-retry experiment: one full single-attempt pass, then five
# blind single-attempt retries over exactly the instances that failed it.
# 32-way parallel throughout.
#
# This is the empirical form of the strongest objection to the branching
# result ("concentrate the retry budget on the failures instead of full
# passes") -- measured instead of extrapolated. No branching anywhere:
# every run is --rounds 0.
#
# Lessons wired in from the pass@4 chain: phase completion is judged by
# SUMMARY COUNTS, not process exit (workers sometimes hang after writing
# their final summary), and stragglers are killed once the summaries say the
# phase is done.
#
# Run detached:  setsid nohup scripts/run_adaptive.sh > runs/adaptive/chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('/home/ec2-user/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

WORKERS=32
BASE=runs/adaptive
mkdir -p "$BASE"

launch_phase() {  # $1 = phase dir, $2 = ids file
    local out="$1" ids="$2"
    mkdir -p "$out"
    python3.11 - "$out" "$ids" "$WORKERS" <<'PY'
import pathlib, sys
out, ids_file, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = pathlib.Path(ids_file).read_text().strip().split(',')
for k in range(n):
    shard = ids[k::n]
    if shard:
        (pathlib.Path(out) / ('shard-%d.txt' % k)).write_text(','.join(shard))
print('%d ids across %d shards' % (len(ids), min(n, len(ids))))
PY
    for shard in "$out"/shard-*.txt; do
        k=$(basename "$shard" .txt | cut -d- -f2)
        setsid nohup python3.12 -u -m swebench.fork_eval \
            --instance "$(cat "$shard")" --slot claude-code \
            --model us.anthropic.claude-sonnet-4-6 \
            --rounds 0 --timeout 1200 \
            -o "$out/shard-$k" > "$out/shard-$k.log" 2>&1 < /dev/null &
        sleep 1
    done
}

wait_phase() {  # $1 = phase dir; returns when summaries cover every shard list
    local out="$1"
    while true; do
        sleep 120
        local verdict
        verdict=$(python3.11 - "$out" <<'PY'
import json, glob, pathlib, sys
out = sys.argv[1]
expected = done = 0
for lst in glob.glob(out + '/shard-*.txt'):
    want = len(pathlib.Path(lst).read_text().strip().split(','))
    expected += want
    summary = lst[:-4] + '/summary.json'
    try:
        done += len(json.load(open(summary))['instances'])
    except Exception:
        pass
print('DONE' if done >= expected and expected > 0 else 'WAIT %d/%d' % (done, expected))
PY
)
        echo "[chain] $out: $verdict"
        [ "$verdict" = "DONE" ] && break
    done
    # summaries complete -> anything still running for this phase is a hung tail
    python3.11 - "$out" <<'PY'
import os, signal, sys
needle = sys.argv[1] + '/shard-'
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open('/proc/%s/cmdline' % pid, 'rb').read().decode().replace('\0', ' ')
    except OSError:
        continue
    if 'swebench.fork_eval' in cmd and needle in cmd:
        os.kill(int(pid), signal.SIGTERM)
        print('[chain] reaped hung worker', pid)
PY
}

# ---- phase 1: one full pass -------------------------------------------------
if [ ! -f "$BASE/p1/.complete" ]; then
    echo "[chain] phase 1: full 500, $WORKERS workers, $(date)"
    python3.11 - <<'PY'
from swebench.dataset import load_instances
import pathlib
ids = sorted(i['instance_id'] for i in load_instances('verified'))
pathlib.Path('runs/adaptive/all.txt').write_text(','.join(ids))
PY
    launch_phase "$BASE/p1" "$BASE/all.txt"
    wait_phase "$BASE/p1"
    touch "$BASE/p1/.complete"
fi

# ---- the failed set ----------------------------------------------------------
python3.11 - <<'PY'
import json, glob, pathlib
failed = []
for f in sorted(glob.glob('runs/adaptive/p1/shard-*/summary.json')):
    for i in json.load(open(f))['instances']:
        if not i['resolved']:
            failed.append(i['instance'])
pathlib.Path('runs/adaptive/failed.txt').write_text(','.join(sorted(failed)))
print('[chain] phase 1 failures: %d' % len(failed))
PY

# ---- phase 2: five blind retries over the failures ---------------------------
for r in 1 2 3 4 5; do
    if [ -f "$BASE/retry$r/.complete" ]; then
        echo "[chain] retry $r already complete"
        continue
    fi
    echo "[chain] retry $r starting $(date)"
    launch_phase "$BASE/retry$r" "$BASE/failed.txt"
    wait_phase "$BASE/retry$r"
    touch "$BASE/retry$r/.complete"
done
echo "[chain] all done $(date)"
