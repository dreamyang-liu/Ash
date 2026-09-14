"""Run Pro with the shared checkpointed rollout CLI."""

import sys

from swebench.fork_eval import main

if __name__ == "__main__":
    raise SystemExit(main(["--benchmark", "swebench-pro", *sys.argv[1:]]))
