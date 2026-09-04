"""DeepSWE (datacurve-ai/deep-swe) on the checkpointed path.

Sits beside ``swebench/`` and reuses its loop: ``swebench.fork_eval`` runs the
attempt and decides when to branch; this package supplies what differs per
benchmark -- where tasks come from (``tasks.py``), what the agent is told
(``bench.py``), and how a snapshot is turned into a verdict (``grade.py``).

The grader is *theirs*, verbatim: every task ships ``tests/{test.sh, grader.py,
test.patch, config.json}`` and a two-line ``tests/Dockerfile`` that copies them
onto the task image. We replay that Dockerfile inside a fresh, offline microVM
and run ``test.sh`` -- the only thing we add is transport.

    python -m swebench.fork_eval --benchmark deepswe \\
        --tasks-dir ~/projects/LBP/deep-swe/tasks \\
        --instance ytt-jsonpath-query-api --rounds 0 --timeout 10800 \\
        --slot claude-code --model us.anthropic.claude-sonnet-4-6 -o runs/deepswe
"""
