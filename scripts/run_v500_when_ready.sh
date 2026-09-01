#!/usr/bin/env bash
# Chain: wait for the image pre-pull, retry its failures, then launch the eval.
#
# Exists because the two phases consume different scarce resources -- Docker Hub
# pull budget vs agent time -- and the first full launch mixed them: 49 agent
# slots died on 429s. This script keeps them sequential and unattended: the
# pre-pull's state file is per-image, so each retry round only touches what
# failed, and the eval only starts once every image the run will ask for is
# already local.
#
# Run it detached:  setsid nohup scripts/run_v500_when_ready.sh > runs/v500/autorun.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source ~/aenv-bench/env.sh
export AWS_BEARER_TOKEN_BEDROCK="$(python3.11 -c "import json; print(json.load(open('/home/ec2-user/.claude/settings.json'))['env']['AWS_BEARER_TOKEN_BEDROCK'])")"
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 PYTHONPATH=.:sdk

echo "[chain] waiting for the running pre-pull to finish..."
while pgrep -f "prepull_image[s].py" >/dev/null; do sleep 60; done

# Retry rounds: each one re-runs the (idempotent, per-image-state) pre-pull.
# 25 minutes between rounds -- rate-limit windows recover on the hour scale,
# and a tighter loop would just burn the remaining budget on the same 429s.
for round in 1 2 3 4 5 6 7 8; do
    # 8 workers: one measured conversion is ~37s, so 2 workers (~197/h) sat
    # exactly at the 200/h quota -- any slower image and the window leaks.
    # At 8, conversion can never be the bottleneck; each hourly window is
    # drained as soon as it opens, and 429s cost nothing.
    if python3.11 -u scripts/prepull_images.py --workers 8; then
        echo "[chain] pre-pull complete, every image converted"
        break
    fi
    echo "[chain] round ${round} left failures; sleeping 20min for the rate-limit window"
    sleep 1200
done

remaining=$(python3.11 - <<'PY'
import json, pathlib
from swebench.dataset import load_instances, resolve_image
need = {resolve_image(i) for i in load_instances("verified")}
ok = set()
state = pathlib.Path("runs/v500/prepull.jsonl")
if state.exists():
    for line in state.read_text().splitlines():
        r = json.loads(line)
        if r.get("ok"):
            ok.add(r["image"])
print(len(need - ok))
PY
)
echo "[chain] images still missing: ${remaining}"
if [ "${remaining}" -gt 25 ]; then
    # A quarter of the dataset missing means something structural (auth expired,
    # registry down); launching would recreate the failure this script exists to
    # prevent. Stop and say so instead.
    echo "[chain] too many missing -- NOT launching. Fix the pull, re-run this script."
    exit 1
fi

echo "[chain] launching 8 workers"
for k in 0 1 2 3 4 5 6 7; do
    setsid nohup python3.12 -u -m swebench.fork_eval \
        --instance "$(cat runs/v500/shard-$k.txt)" --slot claude-code \
        --model us.anthropic.claude-sonnet-4-6 \
        --rounds 0 --timeout 1200 \
        -o runs/v500/shard-$k > runs/v500/shard-$k.log 2>&1 < /dev/null &
    sleep 3
done
echo "[chain] workers up: $(ps aux | grep -c '[f]ork_eval --instance')"
