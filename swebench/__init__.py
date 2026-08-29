"""SWE-bench evaluation: what counts as an answer, and how to branch a wrong one.

This layer knows two things the rest of the repository deliberately does not:
that the answer is a **patch**, and that a patch is right when the instance's
FAIL_TO_PASS tests pass without breaking PASS_TO_PASS. Everything about *running*
an agent lives in ``harness/``.

Usage:
    python -m swebench.fork_eval --instance sympy__sympy-13091 \\
        --slot codex --model openai.gpt-5.6-luna --rounds 2 --branches 3

What used to be here and is gone: this repository's own litellm agent loop, the
four ``harnesses/`` topologies, SWE-Marathon, the batch runner, the RL rollout
server, step-replay, and the re-export shims that pointed at modules which had
moved into ``harness/`` (backends, templates, mcp_server, models). All of them
predate the orchestrator owning a run, and the shims had no importers left once
the code above them went. Running one instance through ``fork_eval`` covers what
we actually do today; batch and rollout come back on top of the orchestrator when
they are needed, rather than being carried along broken.

Three modules remain, and each is here because it answers a question only this
layer may answer: ``dataset`` (which instances, and what command decides a test
passed), ``patch`` (what belongs in a diff), ``fork_eval`` (the loop).
"""

from .dataset import format_task_prompt, load_instances, resolve_image

__all__ = ["format_task_prompt", "load_instances", "resolve_image"]
