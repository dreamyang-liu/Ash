"""How a tool result is shaped and rendered (swebench/models.py, agent/__init__.py).

`output` and `error` are alternatives, never two copies of one message. The wire
format carries a single text slot plus an `is_error` flag, so a failed call gives
one string; storing it in both fields made the agent loop render it twice, and
the model paid for both copies.

Covered:
- ToolResult.from_sdk puts a failure's text in `output` only
- both conversion sites (AshSession, manager-worker's thread executor) go
  through it, so they cannot drift apart again
- _observation renders each shape with no filler and no duplication
- a failing command's rendered size is halved versus the duplicating form
- truncation's budget is spent on real output rather than on a second copy
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swebench.agent import _observation
from swebench.agent.interceptors import default_pipeline
from swebench.agent.pipeline import CallContext
from swebench.models import ToolResult


def _sdk(output: str, is_error: bool = False) -> SimpleNamespace:
    return SimpleNamespace(output=output, is_error=is_error)


# --------------------------------------------------------------------------- #
#  Conversion
# --------------------------------------------------------------------------- #

def test_from_sdk_keeps_a_failures_text_in_output_only():
    result = ToolResult.from_sdk(_sdk("pytest failed\nAssertionError", is_error=True))
    assert not result.success
    assert result.output == "pytest failed\nAssertionError"
    assert result.error is None, "the text is output, not a second copy under error"


def test_from_sdk_maps_success():
    result = ToolResult.from_sdk(_sdk("all good"))
    assert result.success and result.output == "all good" and result.error is None


def test_both_conversion_sites_share_one_implementation():
    """Two hand-rolled copies of this conversion is how the duplication got in
    (and stayed in two files). Grep is the cheapest guard against a third."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("sandbox.py", "harnesses/manager_worker.py"):
        source = (root / name).read_text()
        assert "from_sdk" in source, f"{name} should convert via ToolResult.from_sdk"
        assert "error=r.output" not in source and \
               "error=sdk_result.output" not in source, \
               f"{name} still duplicates the output into error"


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("success,output,error,expected", [
    # A command that ran and failed: real output, no error field.
    (False, "AssertionError: got 4", None, "Error: AssertionError: got 4"),
    # A call the tool refused: an error, no output.
    (False, "", "unknown agent tool: xyz", "Error: unknown agent tool: xyz"),
    (False, "", "No active sandbox", "Error: No active sandbox"),
    # Both, and genuinely different: both are shown.
    (False, "partial output", "exit 2", "Error: exit 2\npartial output"),
    # Nothing at all still says something.
    (False, "", None, "Error: Unknown error"),
    # Success is just the output.
    (True, "done", None, "done"),
])
def test_observation_shapes(success, output, error, expected):
    assert _observation(success, output, error) == expected


def test_observation_never_prints_one_message_twice():
    """Defensive: a producer that still fills both fields identically must not
    cost the model two copies."""
    text = "AssertionError: expected 3, got 4"
    assert _observation(False, text, text) == f"Error: {text}"


def test_a_failing_command_renders_at_half_the_duplicating_size():
    payload = "F" * 200_000
    duplicating = f"Error: {payload}\n{payload}"          # the shape before the fix
    result = ToolResult.from_sdk(_sdk(payload, is_error=True))
    rendered = _observation(result.success, result.output, result.error)
    assert len(rendered) < len(duplicating) * 0.55


def test_truncation_budget_buys_real_output_not_a_second_copy():
    """With the text in one field, the whole bound goes to distinct content."""
    huge = "F" * 3_000_000
    ctx = CallContext(agent_id="A", sandbox_id="s", tool_name="shell",
                      args={"command": "pytest"}, metadata={})
    result = default_pipeline(max_output_len=12000).execute(
        ctx, lambda t, a: ToolResult.from_sdk(_sdk(huge, is_error=True)))
    rendered = _observation(result.success, result.output, result.error)
    assert len(rendered) < 30_000
    assert not result.error, "nothing should have been mirrored into error"
