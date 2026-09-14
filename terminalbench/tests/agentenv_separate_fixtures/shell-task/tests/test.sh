#!/bin/bash
set -eu
mkdir -p /logs/verifier
python3 - <<'PY'
from pathlib import Path

answer = Path("/logs/artifacts/answer.txt")
assert not Path("/app/answer.txt").exists(), "verifier must start in a fresh VM"
reward = int(answer.exists() and answer.read_text() == "ready\n")
Path("/logs/verifier/reward.txt").write_text(str(reward))
PY
