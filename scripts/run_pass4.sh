#!/usr/bin/env bash
# Three more independent full-500 single-attempt passes, run back to back.
#
# Purpose: an EMPIRICAL pass@4 baseline. The branching pipeline hit 93.0% at
# 3.9x the single-pass spend; the honest comparison is what 4 independent
# blind samples buy at the same budget, measured rather than extrapolated
# from an independence assumption that same-model retries are known to break.
#
# Passes run sequentially (8 workers each) because 24 concurrent agents would
# court Bedrock throttling; 8 is the proven throughput. Each pass is sharded,
# journaled on persistent disk, and graded inline by the current grader.
#
# Run detached:  setsid nohup scripts/run_pass4.sh > runs/pass4/chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('/home/ec2-user/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

mkdir -p runs/pass4
if [ ! -f runs/pass4/shard-0.txt ]; then
    python3.11 - <<'PY'
from swebench.dataset import load_instances
import pathlib
ids = sorted(i['instance_id'] for i in load_instances('verified'))
out = pathlib.Path('runs/pass4')
for k in range(8):
    (out / ('shard-%d.txt' % k)).write_text(','.join(ids[k::8]))
print('shards ready')
PY
fi

for pass_no in 2 3 4; do
    out="runs/pass4/p${pass_no}"
    if [ -f "${out}/.complete" ]; then
        echo "[chain] pass ${pass_no} already complete, skipping"
        continue
    fi
    echo "[chain] === pass ${pass_no} starting $(date) ==="
    mkdir -p "${out}"
    pids=""
    for k in 0 1 2 3 4 5 6 7; do
        setsid nohup python3.12 -u -m swebench.fork_eval \
            --instance "$(cat runs/pass4/shard-$k.txt)" --slot claude-code \
            --model us.anthropic.claude-sonnet-4-6 \
            --rounds 0 --timeout 1200 \
            -o "${out}/shard-$k" > "${out}/shard-$k.log" 2>&1 < /dev/null &
        pids="$pids $!"
        sleep 3
    done
    wait $pids
    echo "[chain] pass ${pass_no} workers exited $(date)"
    python3.11 - "$out" <<'PY'
import json, glob, sys
out = sys.argv[1]
done = ok = 0
for f in glob.glob(out + '/shard-*/summary.json'):
    for i in json.load(open(f))['instances']:
        done += 1; ok += i['resolved']
print('[chain] pass result: %d/%d resolved of %d run' % (ok, done, done))
if done >= 495:   # tolerate a few stragglers; mark complete
    open(out + '/.complete', 'w').write('%d/%d\n' % (ok, done))
PY
done
echo "[chain] all passes done $(date)"
