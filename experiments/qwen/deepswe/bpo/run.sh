#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:$repo/sdk${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
exec python -m experiments.qwen.run --dataset deepswe --method bpo "$@"
