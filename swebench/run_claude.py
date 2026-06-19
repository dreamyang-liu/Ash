"""Backwards-compat entry point — use `python -m swebench --harness claude-code` instead."""

import sys

print("NOTE: run_claude.py is deprecated. Use:", file=sys.stderr)
print("  python -m swebench -c swebench/configs/claude-opus.yaml", file=sys.stderr)
print("  python -m swebench --harness claude-code --model opus -i <instance>", file=sys.stderr)
print("", file=sys.stderr)

# Forward to unified CLI
from .__main__ import main
sys.argv = [sys.argv[0]] + ["--harness", "claude-code"] + sys.argv[1:]
main()
