"""What a benchmark run reports back — the eval layer's own vocabulary.

A prediction is SWE-bench's format, not something the machinery below should
know about. It was being assembled by hand in nine places (four harnesses, the
single-instance runner, twice each for the success and failure paths), which is
how `manager-worker` came to drop the `tools` setting and how `custom_tools_dir`
ended up reaching `AgentConfig` from nowhere: a decision repeated nine times is a
decision made differently somewhere.

Two shapes, one place:

    prediction(instance_id, model, patch, exit_status)   a finished attempt
    failure(instance_id, model, exit_status)             no patch to report

`model_name_or_path` is the field SWE-bench's grader reads, so the name stays
even though it holds a model name; and an empty `model_patch` is how a failure is
reported, since the grader treats a missing patch and an empty one alike.
"""

from __future__ import annotations

__all__ = ["prediction", "failure", "PREDICTION_KEYS"]

#: The keys SWE-bench's grader expects. Named so a test can assert the shape
#: rather than restating the literal in another place.
PREDICTION_KEYS = ("instance_id", "model_patch", "model_name_or_path",
                   "exit_status")

UNKNOWN_MODEL = "unknown"


def prediction(instance_id: str, model: str | None, patch: str,
               exit_status: str) -> dict:
    """One attempt at one instance, in the format the grader reads.

    ``exit_status`` is Ash's own vocabulary (``completed``, ``cost_limit``,
    ``step_limit``, ``error: …``) and is carried through untouched: the grader
    ignores it, but a run is much easier to read afterwards when a missing patch
    says *why* it is missing.
    """
    return {
        "instance_id": instance_id,
        "model_patch": patch or "",
        "model_name_or_path": model or UNKNOWN_MODEL,
        "exit_status": exit_status,
    }


def failure(instance_id: str, model: str | None, exit_status: str) -> dict:
    """An attempt that produced nothing to submit.

    Distinct from ``prediction(..., patch="")`` only in intent, which is the
    point: the call sites that meant "this failed" now say so.
    """
    return prediction(instance_id, model, "", exit_status)
