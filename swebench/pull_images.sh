#!/usr/bin/env bash
# Pull all SWE-bench Docker images in parallel.
#
# Usage:
#   ./swebench/pull_images.sh              # SWE-bench Lite, 8 parallel
#   ./swebench/pull_images.sh verified 16  # SWE-bench Verified, 16 parallel
#   ./swebench/pull_images.sh lite 4       # SWE-bench Lite, 4 parallel

set -euo pipefail

SUBSET="${1:-lite}"
PARALLEL="${2:-8}"

DATASET_MAP_lite="princeton-nlp/SWE-bench_Lite"
DATASET_MAP_verified="princeton-nlp/SWE-bench_Verified"
DATASET_MAP_full="princeton-nlp/SWE-bench"

VAR="DATASET_MAP_${SUBSET}"
DATASET="${!VAR:-}"
if [[ -z "$DATASET" ]]; then
    echo "Unknown subset: $SUBSET (use: lite, verified, full)"
    exit 1
fi

echo "=== Pulling SWE-bench images ==="
echo "  subset:   $SUBSET ($DATASET)"
echo "  parallel: $PARALLEL"
echo ""

# Generate image list (__ -> _1776_ for Docker compatibility, lowercase)
IMAGES=$(python -c "
from datasets import load_dataset
ds = load_dataset('$DATASET', split='test')
for inst in ds:
    iid = inst['instance_id'].replace('__', '_1776_').lower()
    name = inst.get('image_name') or f'swebench/sweb.eval.x86_64.{iid}:latest'
    print(name)
" 2>/dev/null)

TOTAL=$(echo "$IMAGES" | wc -l)
echo "  total:    $TOTAL images"
echo ""

# Pull in parallel, track progress
DONE=0
FAIL=0
LOGFILE="/tmp/swebench_pull_$(date +%s).log"

pull_one() {
    local img="$1"
    if docker image inspect "$img" > /dev/null 2>&1; then
        echo "  - $img (exists)"
        return 0
    fi
    if docker pull "$img" >> "$LOGFILE" 2>&1; then
        echo "  ✓ $img"
    else
        echo "  ✗ $img (FAILED)" >&2
        return 1
    fi
}
export -f pull_one
export LOGFILE

echo "$IMAGES" | xargs -P "$PARALLEL" -I {} bash -c 'pull_one "$@"' _ {}

echo ""
echo "=== Done. Log: $LOGFILE ==="
