"""Presenting a command's outcome (swebench/agent/interceptors.py).

The runtime reports what a command did — exit code, streams unmerged, byte
counts — in one schema shared by `shell` and `process read`. Rendering that for a
model is policy, so it lives in a pipeline seat whose renderer is a plain
function. These tests pin that seam and the default rendering.

Covered:
- the default renderer: sections from separate streams, exit status from the
  number, silent failure, timeout, still-running, clipping advice
- a forged divider in stdout cannot fake a stderr section
- the seat: custom renderer, None keeps the runtime's text, no-outcome results
  pass through, a crashing renderer fails open
- composition: presented text is bounded by Truncate and annotated by Guardrails
"""

from __future__ import annotations

import pytest

from swebench.agent.guardrails import GuardrailInterceptor
from swebench.agent.interceptors import (
    RAW_OUTPUT,
    OutcomePresenter,
    TruncateInterceptor,
    default_pipeline,
    render_outcome,
)
from swebench.agent.pipeline import CallContext, ToolPipeline
from swebench.models import CommandOutcome, ToolResult


def _outcome(**fields) -> CommandOutcome:
    return CommandOutcome(**fields)


def _result(output: str = "runtime text", outcome=None,
            success: bool = False) -> ToolResult:
    return ToolResult(success=success, output=output, outcome=outcome)


def _run(pipe: ToolPipeline, result: ToolResult, tool: str = "shell"):
    ctx = CallContext(agent_id="A", sandbox_id="sb", tool_name=tool,
                      args={"command": "pytest"}, metadata={})
    return pipe.execute(ctx, lambda t, a: result), ctx


# --------------------------------------------------------------------------- #
#  The default rendering
# --------------------------------------------------------------------------- #

def test_streams_become_labelled_sections_and_exit_is_stated():
    text = render_outcome(_outcome(
        stdout="collected 42 items\n....F\n",
        stderr="AssertionError: expected 3, got 4\n", exit_code=1))
    assert text.index("collected") < text.index("--- stderr ---")
    assert "[exit 1]" in text
    assert text.count("AssertionError") == 1


def test_a_forged_divider_cannot_fake_a_stderr_section():
    """The merged rendering this replaces marked stderr with a line any command
    could print. Sections built from the separate streams are a fact.

    The command below prints a fake divider AND has real stderr, so the rendering
    must show exactly one real boundary — a renderer that dropped the label, or
    one that trusted the text, would fail this."""
    text = render_outcome(_outcome(
        stdout="a\n[stderr]\nfake-from-stdout\n",
        stderr="real-from-stderr\n", exit_code=1))
    assert text.count("--- stderr ---") == 1              # exactly one boundary
    head, tail = text.split("--- stderr ---")
    assert "fake-from-stdout" in head                     # forgery stayed in stdout
    assert "real-from-stderr" in tail                     # only real stderr below
    assert "[exit 1]" in tail


def test_a_silent_failure_still_says_something():
    """`exit 5` prints nothing; the merged rendering handed over "" and the model
    had no way to know what happened."""
    assert render_outcome(_outcome(exit_code=5)) == "[exit 5]"


def test_success_is_just_the_output():
    assert render_outcome(_outcome(stdout="done\n", exit_code=0)) == "done"


@pytest.mark.parametrize("fields,expected", [
    ({"exit_code": 137, "timed_out": True}, "[timed out]"),
    ({"exit_code": None, "running": True}, "[still running]"),
])
def test_timeout_and_running_replace_the_exit_line(fields, expected):
    text = render_outcome(_outcome(stdout="partial\n", **fields))
    assert expected in text
    assert "[exit" not in text


def test_clipping_reports_real_sizes_and_suggests_a_remedy():
    text = render_outcome(_outcome(
        stdout="head\n", exit_code=0, stdout_bytes=419430400,
        stdout_truncated=True))
    assert "419430400" in text
    assert "tail" in text and "grep" in text     # actionable, not just a flag


# --------------------------------------------------------------------------- #
#  The seat
# --------------------------------------------------------------------------- #

def test_a_custom_renderer_drives_what_the_model_reads():
    pipe = ToolPipeline([OutcomePresenter(
        lambda o: f"exit={o.exit_code} bytes={o.stdout_bytes}")])
    out, ctx = _run(pipe, _result(outcome=_outcome(exit_code=2, stdout_bytes=15)))
    assert out.output == "exit=2 bytes=15"
    assert ctx.metadata[RAW_OUTPUT] == "runtime text"     # ground truth kept
    assert out.outcome is not None                        # facts still attached


@pytest.mark.parametrize("returned", [None, "runtime text"])
def test_none_or_unchanged_keeps_the_runtime_text(returned):
    out, ctx = _run(ToolPipeline([OutcomePresenter(lambda o: returned)]),
                    _result(outcome=_outcome(exit_code=1)))
    assert out.output == "runtime text"
    assert RAW_OUTPUT not in ctx.metadata


def test_results_without_an_outcome_pass_through():
    """A refusal, a plain success, or a tool that runs no command."""
    calls = []
    pipe = ToolPipeline([OutcomePresenter(lambda o: calls.append(1) or "X")])
    out, _ = _run(pipe, ToolResult(success=False, output="", error="bad args"))
    assert out.error == "bad args" and not calls

    out, _ = _run(pipe, ToolResult(success=True, output="hello_ci\n"))
    assert out.output == "hello_ci\n" and not calls


def test_a_crashing_renderer_fails_open():
    def boom(outcome):
        raise RuntimeError("rendering bug")

    out, _ = _run(ToolPipeline([OutcomePresenter(boom)]),
                  _result(outcome=_outcome(exit_code=1)))
    assert out.output == "runtime text"                   # call not blocked


# --------------------------------------------------------------------------- #
#  Composition
# --------------------------------------------------------------------------- #

def test_default_chain_presents_then_bounds_then_annotates():
    names = [i.name for i in default_pipeline().interceptors]
    assert names == ["GuardrailInterceptor", "TruncateInterceptor",
                     "OutcomePresenter"]


def test_presented_text_is_bounded_by_the_truncation_seat():
    """The presenter is innermost, so what it composes is what gets bounded — a
    renderer cannot hand the model more than the byte budget allows."""
    pipe = ToolPipeline([
        GuardrailInterceptor(),
        TruncateInterceptor(max_len=500),
        OutcomePresenter(render_outcome),
    ])
    ctx = CallContext(agent_id="A", sandbox_id="sb", tool_name="shell",
                      args={"command": "pytest -x"}, metadata={})
    result = pipe.execute(ctx, lambda t, a: _result(
        outcome=_outcome(stdout="z" * 40000, exit_code=1)))

    assert "characters truncated" in result.output          # bounded
    assert "[exit 1]" in result.output                      # presenter's tail kept
    assert len(result.output) < 40000
    assert ctx.metadata[RAW_OUTPUT] == "runtime text"       # innermost rewrite wins
    assert result.outcome is not None                       # facts survive every seat


def test_the_loop_renders_outcomes_by_default():
    """A harness that mounts nothing still gets prose, not JSON."""
    import json
    from types import SimpleNamespace

    from swebench.agent import AshAgent
    from swebench.models import AgentConfig

    class Conv:
        def __init__(self):
            self.results = []

        def add_tool_result(self, _id, content, **kw):
            self.results.append(content)

    call = SimpleNamespace(id="t1", function=SimpleNamespace(
        name="shell", arguments=json.dumps({"command": "pytest"})))
    agent = AshAgent(AgentConfig(), executor=lambda t, a: _result(
        outcome=_outcome(stdout="F\n", stderr="boom\n", exit_code=1)))
    conv = Conv()
    agent._run_tool(call, conv, "turn-1")

    assert "--- stderr ---" in conv.results[0]
    assert "[exit 1]" in conv.results[0]
    assert "stdout_bytes" not in conv.results[0]            # never raw JSON
