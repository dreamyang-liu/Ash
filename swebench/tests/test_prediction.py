"""Prediction construction (swebench/prediction.py).

The format belongs to the benchmark, but it was assembled by hand in nine places
-- four harnesses, the single-instance runner and the batch runner, twice each for
the success and failure paths. A decision repeated nine times is a decision made
differently somewhere, which is exactly how `manager-worker` came to drop the
`tools` setting from its config builder.

These tests pin the format itself, the fallbacks each old site relied on, and
that no site builds one by hand again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swebench.prediction import PREDICTION_KEYS, failure, prediction


def test_a_prediction_has_exactly_the_keys_the_grader_reads():
    p = prediction("django__django-10880", "bedrock/sonnet", "diff --git\n", "completed")
    assert tuple(p) == PREDICTION_KEYS
    assert p == {
        "instance_id": "django__django-10880",
        "model_patch": "diff --git\n",
        "model_name_or_path": "bedrock/sonnet",
        "exit_status": "completed",
    }


def test_a_failure_is_a_prediction_with_no_patch():
    """SWE-bench's grader treats a missing patch and an empty one alike, so a
    failure is reported in the same shape rather than a different one."""
    f = failure("iid", "m", "session_failed")
    assert f == prediction("iid", "m", "", "session_failed")
    assert f["model_patch"] == ""


@pytest.mark.parametrize("model,expected", [
    ("bedrock/sonnet", "bedrock/sonnet"),
    ("claude-code/opus", "claude-code/opus"),   # the claude-code harness prefixes
    (None, "unknown"),                          # c.get("model") with no model set
    ("", "unknown"),
])
def test_a_missing_model_reports_unknown(model, expected):
    """Each old site wrote `c.get("model", "unknown")` by hand; one of them could
    have written something else."""
    assert prediction("iid", model, "p", "completed")["model_name_or_path"] == expected


def test_a_none_patch_is_reported_as_empty():
    """Harnesses pass whatever get_patch returned; `None` must not reach the
    grader as the string "None"."""
    assert prediction("iid", "m", None, "completed")["model_patch"] == ""


def test_the_exit_status_is_carried_through_untouched():
    """The grader ignores it, but a run is far easier to read afterwards when a
    missing patch says why it is missing."""
    for status in ("completed", "cost_limit", "step_limit", "error: boom",
                   "session_failed", "no_base_commit"):
        assert prediction("iid", "m", "", status)["exit_status"] == status


def test_a_prediction_survives_json_round_trip():
    """It is written to preds.json and read by an external grader."""
    p = prediction("iid", "m", "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n", "completed")
    assert json.loads(json.dumps(p)) == p


# --------------------------------------------------------------------------- #
#  One builder, every caller
# --------------------------------------------------------------------------- #

def test_no_module_assembles_a_prediction_by_hand():
    """The point of the module. A tenth hand-written copy would compile, pass
    every test, and quietly disagree about a fallback."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path) or path.name in ("prediction.py",):
            continue
        if "/tests/" in str(path):
            continue
        src = path.read_text()
        # A construction, not a read: `"model_patch":` with a value beside it.
        if '"model_patch":' in src:
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], f"these build a prediction by hand: {offenders}"


def test_every_harness_reports_through_the_builder():
    """A harness that returned a bare dict would bypass the fallbacks above."""
    root = Path(__file__).resolve().parents[1] / "harnesses"
    for path in sorted(root.glob("*.py")):
        if path.name in ("__init__.py", "base.py"):
            continue                      # the registry and the ABC report nothing
        src = path.read_text()
        assert "from ..prediction import" in src, \
            f"{path.name} does not use the builder"
