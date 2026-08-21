"""Unit tests for the interceptors migrated out of the agent loop.

Covers GuardrailInterceptor (swebench/agent/guardrails.py), TruncateInterceptor
and the loop's default chain (swebench/agent/interceptors.py), plus the agent
loop consuming a pipeline (swebench/agent/__init__.py).

No Docker, no model calls.

Covered:
- read-before-edit: warn mode annotates, reject mode refuses
- reads recorded only on SUCCESS, and only in `after` (a failed view must not
  unlock editing — the loop's inline version had this backwards)
- state keyed by (agent_id, sandbox_id): A's read never excuses B's blind edit
- edit-streak counting and its reset by a test-running shell command
- truncation bounds ToolResult.output, preserves the original in metadata,
  and never elides error text
- ordering: a guardrail warning survives truncation of a huge output
- agent loop: default chain mounted, custom chain respected, ToolPipeline([])
  disables interception, guardrail state does not leak across runs
- trace fidelity: events report the runtime's byte count, not the bounded one
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from swebench.agent import AshAgent
from swebench.agent.pipeline import (RAW_ERROR, RAW_OUTPUT, CallContext,
                                     ToolPipeline)
from swebench.agent.seats import (
    EDIT_STREAK_LIMIT,
    GuardrailInterceptor,
    GuardrailState,
    TruncateInterceptor,
    default_pipeline,
)
from swebench.agent.trace import ToolTraceWriter
from swebench.models import AgentConfig, ToolResult

PATH = "/testbed/target.py"


def _ok(output: str = "ok") -> ToolResult:
    return ToolResult(success=True, output=output)


def _ctx(tool: str, args: dict, agent: str = "A", sandbox: str = "sb-1") -> CallContext:
    return CallContext(agent_id=agent, sandbox_id=sandbox, tool_name=tool,
                       args=dict(args), metadata={})


def _view(path: str = PATH) -> tuple[str, dict]:
    return "text_editor", {"command": "view", "path": path}


def _edit(path: str = PATH) -> tuple[str, dict]:
    return "text_editor", {"command": "str_replace", "path": path,
                           "old_str": "a", "new_str": "b"}


def _run(pipe: ToolPipeline, tool: str, args: dict, agent: str = "A",
         sandbox: str = "sb-1", result: ToolResult | None = None) -> ToolResult:
    inner = lambda t, a: (result if result is not None else _ok())  # noqa: E731
    return pipe.execute(_ctx(tool, args, agent, sandbox), inner)


# --------------------------------------------------------------------------- #
#  GuardrailInterceptor: read-before-edit
# --------------------------------------------------------------------------- #

def test_warn_mode_appends_warning_but_lets_the_edit_through():
    pipe = ToolPipeline([GuardrailInterceptor()])
    result = _run(pipe, *_edit())
    assert result.success                                  # advisory, not a block
    assert "without reading it first" in result.output
    assert result.output.startswith("ok")                  # tool output preserved


def test_reject_mode_refuses_the_edit():
    pipe = ToolPipeline([GuardrailInterceptor(enforcement="reject")])
    result = _run(pipe, *_edit())
    assert not result.success
    assert result.error == "rejected by GuardrailInterceptor"
    assert "without reading it first" in result.output


def test_reading_first_silences_the_warning():
    pipe = ToolPipeline([GuardrailInterceptor()])
    assert _run(pipe, *_view()).success
    result = _run(pipe, *_edit())
    assert result.success
    assert "Warning" not in result.output


def test_failed_view_does_not_unlock_editing():
    """A view that errored is not a read. The loop's inline version recorded
    reads before execution, so viewing a nonexistent file counted."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    failed = ToolResult(success=False, output="", error="No such file")
    _run(pipe, *_view(), result=failed)
    assert "without reading it first" in _run(pipe, *_edit()).output


def test_read_state_is_keyed_by_agent():
    """Shared interceptor instance: A's read must not excuse B's blind edit."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    _run(pipe, *_view(), agent="A")
    assert "Warning" not in _run(pipe, *_edit(), agent="A").output
    assert "without reading it first" in _run(pipe, *_edit(), agent="B").output


def test_read_state_is_keyed_by_sandbox():
    """Same agent, two workspaces: reading a path in one is not reading it in
    the other — the file behind that path is a different file."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    _run(pipe, *_view(), sandbox="sb-1")
    assert "without reading it first" in _run(pipe, *_edit(), sandbox="sb-2").output


@pytest.mark.parametrize("command,args", [
    ("str_replace", {"old_str": "a", "new_str": "b"}),
    ("insert", {"insert_line": 1, "new_str": "x"}),
])
def test_content_edits_require_a_read(command, args):
    pipe = ToolPipeline([GuardrailInterceptor()])
    result = _run(pipe, "text_editor", {"command": command, "path": PATH, **args})
    assert "without reading it first" in result.output


def test_write_does_not_require_a_read_because_it_also_creates():
    """`view` on a nonexistent path fails, so a creating `write` could never
    satisfy read-before-edit: in warn mode that is noise on the documented
    "write a reproduce script" workflow, and in reject mode creating any new
    file becomes impossible. A seat that can afford an existence probe may refuse
    blind *overwrites*; a rule that cannot afford the probe must not claim
    `write`."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    result = _run(pipe, "text_editor", {"command": "write", "path": PATH,
                                        "file_text": "new file"})
    assert "without reading it first" not in result.output

    strict = ToolPipeline([GuardrailInterceptor(enforcement="reject")])
    created = _run(strict, "text_editor", {"command": "write", "path": PATH,
                                           "file_text": "new file"})
    assert created.success, "reject mode must not make file creation impossible"


def test_write_still_counts_toward_the_edit_streak():
    """Not requiring a read is not being unguarded: `write` is still an edit."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    outputs = [_run(pipe, "text_editor", {"command": "write", "path": PATH,
                                          "file_text": f"v{i}"}).output
               for i in range(EDIT_STREAK_LIMIT)]
    assert "without running tests" in outputs[-1]


def test_one_place_states_what_counts_as_an_edit():
    """Two sets, one narrower than the other, and the difference is `write`:
    everything that mutates a file (EDIT_COMMANDS, for streak counting) versus
    everything that mutates *existing* content (CONTENT_EDIT_COMMANDS, for
    read-before-edit). A coordination seat keys off the wider one; that seat was
    removed with Waggle, and the distinction is the part worth keeping."""
    from swebench.agent.tools import CONTENT_EDIT_COMMANDS, EDIT_COMMANDS
    assert "write" in EDIT_COMMANDS
    assert "write" not in CONTENT_EDIT_COMMANDS
    assert CONTENT_EDIT_COMMANDS < EDIT_COMMANDS


@pytest.mark.parametrize("junk", [["str_replace"], {"a": 1}, 5, None])
def test_malformed_command_argument_is_not_a_policy_rejection(junk):
    """A model can put anything in args. Membership against a frozenset raises
    TypeError on unhashable values, and reject mode is fail-closed — so junk
    would surface as "rejected by GuardrailInterceptor" instead of letting the
    runtime report what was actually wrong."""
    pipe = ToolPipeline([GuardrailInterceptor(enforcement="reject")])
    result = _run(pipe, "text_editor", {"command": junk, "path": PATH})
    assert result.success
    assert result.error != "rejected by GuardrailInterceptor"


def test_warning_survives_a_rejection_from_a_deeper_seat():
    """The warning is stashed in `before` and emitted in `after`. A deeper seat
    rejecting must not swallow it — the guardrail was entered, so per the onion
    rules its `after` still runs."""
    from swebench.agent.pipeline import Reject, ToolInterceptor

    class DeepRejecter(ToolInterceptor):
        def before(self, ctx):
            return Reject("deeper seat said no")

    ctx = _ctx("text_editor", {"command": "str_replace", "path": PATH,
                               "old_str": "a", "new_str": "b"})
    result = ToolPipeline([GuardrailInterceptor(), DeepRejecter()]) \
        .execute(ctx, lambda t, a: _ok())
    assert not result.success
    assert "deeper seat said no" in result.output       # rejection preserved
    assert "without reading it first" in result.output  # warning still delivered
    assert "guardrail_warnings" not in ctx.metadata     # stash drained


def test_rejecting_guardrail_leaves_no_orphaned_stash():
    """In reject mode the guardrail's own `after` never runs (it produced the
    result), so the message must travel in the Reject, not the stash."""
    ctx = _ctx("text_editor", {"command": "str_replace", "path": PATH,
                               "old_str": "a", "new_str": "b"})
    result = ToolPipeline([GuardrailInterceptor(enforcement="reject")]) \
        .execute(ctx, lambda t, a: _ok())
    assert "without reading it first" in result.output
    assert "guardrail_warnings" not in ctx.metadata


def test_reject_mode_is_fail_closed_and_warn_mode_fail_open():
    assert GuardrailInterceptor(enforcement="reject").fail_mode == "closed"
    assert GuardrailInterceptor().fail_mode == "open"


def test_invalid_enforcement_is_rejected_at_construction():
    with pytest.raises(ValueError, match="enforcement"):
        GuardrailInterceptor(enforcement="block")


# --------------------------------------------------------------------------- #
#  GuardrailInterceptor: edit streaks
# --------------------------------------------------------------------------- #

def test_edit_streak_warns_at_the_limit_and_a_test_run_resets_it():
    pipe = ToolPipeline([GuardrailInterceptor()])
    _run(pipe, *_view())
    for _ in range(EDIT_STREAK_LIMIT - 1):
        assert "without running tests" not in _run(pipe, *_edit()).output
    assert "without running tests" in _run(pipe, *_edit()).output

    _run(pipe, "shell", {"command": "pytest tests/"})       # streak reset
    assert "without running tests" not in _run(pipe, *_edit()).output


def test_non_test_shell_command_does_not_reset_the_streak():
    pipe = ToolPipeline([GuardrailInterceptor()])
    _run(pipe, *_view())
    for _ in range(EDIT_STREAK_LIMIT):
        _run(pipe, *_edit())
    _run(pipe, "shell", {"command": "ls -la"})
    assert "without running tests" in _run(pipe, *_edit()).output


def test_edit_streaks_are_per_agent():
    pipe = ToolPipeline([GuardrailInterceptor()])
    for agent in ("A", "B"):
        _run(pipe, *_view(), agent=agent)
    for _ in range(EDIT_STREAK_LIMIT):
        _run(pipe, *_edit(), agent="A")
    assert "without running tests" not in _run(pipe, *_edit(), agent="B").output


def test_state_dump_reports_reads_and_streaks():
    state = GuardrailState()
    pipe = ToolPipeline([GuardrailInterceptor(state=state)])
    _run(pipe, *_view())
    _run(pipe, *_edit())
    dumped = state.dump()["A:sb-1"]
    assert dumped["files_read"] == [PATH]
    assert dumped["edit_streak"] == {PATH: 1}


def test_state_dump_includes_agents_that_only_edited_blindly():
    """Keying the dump off reads alone hides the agent that never read anything
    — exactly the behavior this audit exists to surface."""
    state = GuardrailState()
    pipe = ToolPipeline([GuardrailInterceptor(state=state)])
    _run(pipe, *_view(), agent="careful")
    _run(pipe, *_edit(), agent="blind")
    assert set(state.dump()) == {"careful:sb-1", "blind:sb-1"}
    assert state.dump()["blind:sb-1"] == {"files_read": [], "edit_streak": {PATH: 1}}


def test_warning_is_separated_from_an_empty_result_body():
    """A failure with empty output: the loop renders "Error: …\\n" + this text,
    so a bare join would glue the warning onto the error line."""
    pipe = ToolPipeline([GuardrailInterceptor()])
    result = _run(pipe, *_edit(), result=ToolResult(success=False, output="",
                                                   error="No match found"))
    assert result.output.startswith("\n\n[Warning]")


# --------------------------------------------------------------------------- #
#  TruncateInterceptor
# --------------------------------------------------------------------------- #

def test_short_output_passes_through_untouched():
    pipe = ToolPipeline([TruncateInterceptor(max_len=100)])
    result = _run(pipe, "shell", {"command": "ls"}, result=_ok("small"))
    assert result.output == "small"


def test_long_output_is_elided_and_the_original_preserved_in_metadata():
    huge = "x" * 500
    interceptor = TruncateInterceptor(max_len=100)
    ctx = _ctx("shell", {"command": "cat big"})
    result = ToolPipeline([interceptor]).execute(ctx, lambda t, a: _ok(huge))
    assert len(result.output) < len(huge)
    assert "characters truncated" in result.output
    assert ctx.metadata[RAW_OUTPUT] == huge                # evidence kept


def test_truncation_preserves_success_and_short_error():
    failed = ToolResult(success=False, output="y" * 500, error="exit 1")
    pipe = ToolPipeline([TruncateInterceptor(max_len=100)])
    result = _run(pipe, "shell", {"command": "boom"}, result=failed)
    assert not result.success
    assert result.error == "exit 1"                        # short error kept whole


def test_a_failing_command_is_bounded_including_its_error():
    """The one that matters most. Every executor here reports a failure as
    ToolResult(success=False, output=X, error=X) (sandbox.py), and the loop shows
    the model f"Error: {error}\\n{output}". Bounding only `output` leaves the
    payload unbounded, because `error` carries the same bytes — a failing pytest
    without `tail` would blow the context window."""
    huge = "F" * 3_000_000                                 # a failing pytest
    pipe = default_pipeline(max_output_len=12000)
    ctx = _ctx("shell", {"command": "pytest"})
    result = pipe.execute(ctx, lambda t, a: ToolResult(success=False, output=huge,
                                                      error=huge))
    model_sees = f"Error: {result.error or 'Unknown error'}\n{result.output}"
    assert len(model_sees) < 30_000, f"model received {len(model_sees):,} chars"
    assert ctx.metadata[RAW_OUTPUT] == huge                # evidence kept
    assert ctx.metadata[RAW_ERROR] == huge


def test_a_huge_error_is_bounded_even_when_output_is_short():
    """The case a per-field `output` check cannot see: a command that failed
    with everything on stderr. `output` is under the bound, so a guard that only
    measures `output` returns early and the error reaches the model whole."""
    huge = "E" * 3_000_000
    pipe = ToolPipeline([TruncateInterceptor(max_len=12000)])
    ctx = _ctx("shell", {"command": "pytest"})
    result = pipe.execute(ctx, lambda t, a: ToolResult(success=False, output="",
                                                      error=huge))
    assert len(result.error) < 30_000, f"error was {len(result.error):,} chars"
    assert ctx.metadata[RAW_ERROR] == huge


def test_combined_budget_bounds_output_and_error_together():
    """Bounding each field to max_len independently would pass 2x the budget."""
    pipe = ToolPipeline([TruncateInterceptor(max_len=1000)])
    result = _run(pipe, "shell", {"command": "boom"},
                  result=ToolResult(success=False, output="o" * 50_000,
                                    error="e" * 50_000))
    assert len(result.output) + len(result.error) <= 1000 * 2


def test_truncation_leaves_a_failing_call_with_a_usable_error():
    """Splitting the budget must not leave the error empty — it is the headline
    of a failure, and an agent cannot act on `Error: `."""
    pipe = ToolPipeline([TruncateInterceptor(max_len=1000)])
    result = _run(pipe, "shell", {"command": "boom"},
                  result=ToolResult(success=False, output="o" * 50_000,
                                    error="AssertionError: expected 3, got 4\n" + "e" * 50_000))
    assert "AssertionError: expected 3, got 4" in result.error
    assert result.output                                   # output not starved either


# --------------------------------------------------------------------------- #
#  Default chain: order is semantics
# --------------------------------------------------------------------------- #

def test_guardrail_warning_survives_truncation_of_a_huge_output():
    """Guardrails sit outermost, so the warning is appended after truncation."""
    pipe = default_pipeline(max_output_len=200)
    result = _run(pipe, *_edit(), result=_ok("z" * 5000))
    assert "characters truncated" in result.output
    assert "without reading it first" in result.output      # not elided


def test_default_chain_presents_then_bounds_then_annotates():
    """Order is semantics, read innermost-out: the presenter composes text from
    a command's outcome, truncation bounds it, guardrails annotate last."""
    names = [i.name for i in default_pipeline().interceptors]
    assert names == ["GuardrailInterceptor", "TruncateInterceptor",
                     "OutcomePresenter"]


# --------------------------------------------------------------------------- #
#  read_before_edit as a standalone switch
# --------------------------------------------------------------------------- #

def _chain(read_before_edit: bool):
    from swebench.tests.fake_sandbox import FakeSandbox

    sandbox = FakeSandbox({PATH: "base"})
    pipe = ToolPipeline([
        GuardrailInterceptor(enforcement="warn", read_before_edit=read_before_edit),
    ])

    def call(tool: str, args: dict) -> ToolResult:
        executor = sandbox.executor()
        ctx = CallContext(agent_id="A", sandbox_id="default", tool_name=tool,
                          args=dict(args), metadata={"executor": executor})
        return pipe.execute(ctx, executor)

    return call


def test_read_before_edit_can_be_turned_off():
    """The switch exists so a chain with a coordination seat below does not state
    one rule twice -- that seat left with Waggle, but the switch is independent of
    it and a chain composed by hand still needs it."""
    blind = {"command": "str_replace", "path": PATH, "old_str": "base",
             "new_str": "x"}
    assert "without reading it first" in _chain(True)("text_editor", blind).output
    assert "without reading it first" not in _chain(False)("text_editor", blind).output


def test_edit_streak_nudges_survive_with_read_before_edit_off():
    """Turning off one rule must not disable the other."""
    call = _chain(read_before_edit=False)
    call("text_editor", {"command": "view", "path": PATH})
    outputs = [call("text_editor", {"command": "write", "path": PATH,
                                    "file_text": f"v{i}"}).output
               for i in range(EDIT_STREAK_LIMIT)]
    assert "without running tests" in outputs[-1]


# --------------------------------------------------------------------------- #
#  Agent loop consuming the pipeline
# --------------------------------------------------------------------------- #

def _tool_call(name: str, args: dict):
    return SimpleNamespace(id="tc1",
                           function=SimpleNamespace(name=name,
                                                    arguments=json.dumps(args)))


class _Conv:
    def __init__(self):
        self.results = []

    def add_tool_result(self, _id, content, **kw):
        self.results.append(content)


def _agent(executor, **kw) -> AshAgent:
    return AshAgent(AgentConfig(), executor=executor, **kw)


def test_loop_mounts_the_default_chain_and_truncates():
    agent = _agent(lambda n, a: _ok("q" * 40000))
    conv = _Conv()
    agent._run_tool(_tool_call("shell", {"command": "cat big"}), conv, "turn-1")
    assert "characters truncated" in conv.results[0]


def test_loop_default_chain_warns_on_blind_edit():
    agent = _agent(lambda n, a: _ok())
    conv = _Conv()
    agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                              "path": PATH, "old_str": "a",
                                              "new_str": "b"}), conv, "turn-1")
    assert "without reading it first" in conv.results[0]


def test_empty_pipeline_disables_interception():
    agent = _agent(lambda n, a: _ok("q" * 40000), pipeline=ToolPipeline([]))
    conv = _Conv()
    agent._run_tool(_tool_call("shell", {"command": "cat big"}), conv, "turn-1")
    assert "characters truncated" not in conv.results[0]
    assert len(conv.results[0]) == 40000


def test_caller_supplied_pipeline_is_used_and_sees_agent_identity():
    seen = []

    class Spy(TruncateInterceptor):
        def after(self, ctx, result):
            seen.append((ctx.agent_id, ctx.sandbox_id, ctx.tool_name))
            return result

    agent = _agent(lambda n, a: _ok(), pipeline=ToolPipeline([Spy()]),
                   agent_id="worker-3", sandbox_id="shared")
    agent._run_tool(_tool_call("shell", {"command": "ls"}), _Conv(), "turn-1")
    assert seen == [("worker-3", "shared", "shell")]


def test_interceptors_receive_the_routed_runtime_tool_not_the_agent_alias():
    """bash_only mode: the model calls `bash`, the runtime tool is `shell`.
    Interceptors must see what actually executes."""
    seen = []

    class Spy(TruncateInterceptor):
        def before(self, ctx):
            seen.append(ctx.tool_name)
            return super().before(ctx)

    agent = _agent(lambda n, a: _ok(), pipeline=ToolPipeline([Spy()]))
    agent._run_tool(_tool_call("bash", {"command": "ls"}), _Conv(), "turn-1")
    assert seen == ["shell"]


def test_guardrail_state_does_not_leak_between_runs(monkeypatch):
    """Each run() gets a fresh default chain: an edit streak from a previous
    instance must not warn on the first edit of the next one."""
    agent = _agent(lambda n, a: _ok())
    conv = _Conv()
    for _ in range(EDIT_STREAK_LIMIT):
        agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                                  "path": PATH, "old_str": "a",
                                                  "new_str": "b"}), conv, "turn-1")
    assert "without running tests" in conv.results[-1]

    # run() re-resolves the default chain; stub the model call out immediately.
    monkeypatch.setattr(agent, "_query", lambda *a, **kw: None)
    agent.run("task")
    conv2 = _Conv()
    agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                               "path": PATH, "old_str": "a",
                                               "new_str": "b"}), conv2, "turn-1")
    assert "without running tests" not in conv2.results[0]


def test_trace_is_not_polluted_by_a_guardrail_warning(tmp_path):
    """The warning is for the model. Counting its bytes as the tool's output
    inflates output_bytes and — because the polluted text then equals what the
    model saw — suppresses the `observation` that documents the nudge."""
    agent = _agent(lambda n, a: _ok("File edited successfully"))
    path = tmp_path / "trace.events.jsonl"
    agent._event_trace = ToolTraceWriter(path, run_id="r1", agent_id="A",
                                        sandbox_id="sb")
    conv = _Conv()
    agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                              "path": PATH, "old_str": "a",
                                              "new_str": "b"}), conv, "turn-1")
    agent._event_trace.close()

    _, finished = [json.loads(line) for line in path.read_text().splitlines()]
    assert finished["result"]["output"] == "File edited successfully"
    assert finished["result"]["output_bytes"] == len(b"File edited successfully")
    assert "without reading it first" in finished["observation"]
    assert "without reading it first" in conv.results[0]     # model saw it


def test_trace_reports_the_runtime_error_not_the_truncated_one(tmp_path):
    huge = "F" * 40000
    agent = _agent(lambda n, a: ToolResult(success=False, output=huge, error=huge))
    path = tmp_path / "trace.events.jsonl"
    agent._event_trace = ToolTraceWriter(path, run_id="r1", agent_id="A",
                                        sandbox_id="sb")
    conv = _Conv()
    agent._run_tool(_tool_call("shell", {"command": "pytest"}), conv, "turn-1")
    agent._event_trace.close()

    _, finished = [json.loads(line) for line in path.read_text().splitlines()]
    assert finished["result"]["error"] == huge               # ground truth
    assert finished["result"]["output_bytes"] == len(huge.encode())
    assert len(conv.results[0]) < 30_000                    # model got a bound


def test_trace_records_runtime_bytes_while_the_model_sees_bounded_output(tmp_path):
    """Truncation protects the model's context; the trace keeps ground truth."""
    huge = "w" * 40000
    agent = _agent(lambda n, a: _ok(huge))
    path = tmp_path / "trace.events.jsonl"
    agent._event_trace = ToolTraceWriter(path, run_id="r1", agent_id="A",
                                        sandbox_id="sb")
    conv = _Conv()
    agent._run_tool(_tool_call("shell", {"command": "cat big"}), conv, "turn-1")
    agent._event_trace.close()

    _, finished = [json.loads(line) for line in path.read_text().splitlines()]
    assert finished["result"]["output_bytes"] == len(huge.encode())
    assert finished["result"]["output"] == huge             # untruncated
    assert "characters truncated" in finished["observation"]  # what the model saw
    assert "characters truncated" in conv.results[0]


# --------------------------------------------------------------------------- #
#  MCP proxy assembly + session identity
# --------------------------------------------------------------------------- #

def _args(**kw):
    import types
    base = dict(plugins=None, guardrails=None, max_output_bytes=12000)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_proxy_pipeline_is_off_by_default():
    from swebench.mcp_server import _build_pipeline
    assert _build_pipeline(_args()) is None


def test_proxy_bounds_output_whenever_it_mounts_the_chain():
    """Truncation only reaches proxy clients if the assembly actually includes
    it — the flag combinations, not the docstring, are the contract."""
    from swebench.mcp_server import _build_pipeline
    for args in (_args(guardrails="warn"), _args(guardrails="reject")):
        names = [i.name for i in _build_pipeline(args).interceptors]
        assert names[-1] == "TruncateInterceptor", names   # innermost seat


def test_proxy_guardrails_enforce_read_before_edit():
    """It used to be deferred to a coordination seat below, which is gone; with
    guardrails as the only seat, the rule has to be on."""
    from swebench.mcp_server import _build_pipeline
    seats = {i.name: i for i in
             _build_pipeline(_args(guardrails="warn")).interceptors}
    assert seats["GuardrailInterceptor"].read_before_edit is True


def test_a_plugins_file_replaces_the_assembly(tmp_path):
    """How a coordination seat comes back after Waggle's removal: as a plugin."""
    from swebench.mcp_server import _build_pipeline
    path = tmp_path / "seats.py"
    path.write_text(
        "from swebench.agent.pipeline import ToolInterceptor\n"
        "class MySeat(ToolInterceptor):\n    pass\n"
        "PIPELINE = [MySeat()]\n")
    names = [i.name for i in
             _build_pipeline(_args(plugins=str(path))).interceptors]
    assert names == ["MySeat"]


def test_http_sessions_are_stable_for_an_identified_client():
    """Interceptor state is keyed by session identity, so a client that cannot
    be identified gets a new one per request and no seat can hold state. MCP's
    own `initialize` hands out a sessionId; honoring it back is what makes
    read-before-edit satisfiable over HTTP."""
    from swebench.mcp_server import HttpMcpServer, SandboxPool
    server = HttpMcpServer(SandboxPool())
    echoed = [server._get_or_create_session({"mcp-session-id": "s-1"}).id
              for _ in range(3)]
    assert len(set(echoed)) == 1
    owner = [server._get_or_create_session({"x-session-owner": "claude"}).id
             for _ in range(3)]
    assert len(set(owner)) == 1
    anon = [server._get_or_create_session({}).id for _ in range(3)]
    assert len(set(anon)) == 3, "an unidentified client stays isolated"


# --------------------------------------------------------------------------- #
#  Composing with PR #21's harness-side mount (executor_for(pipeline=))
# --------------------------------------------------------------------------- #

def test_an_already_governed_executor_is_not_governed_twice():
    """`executor_for(pipeline=…)` folds a chain into the executor. If the agent
    then mounted its default on top, every rule the two share would be stated
    to the model twice."""
    from swebench.agent.pipeline import piped_executor

    ex = piped_executor(ToolPipeline([GuardrailInterceptor()]),
                        lambda t, a: _ok(), "w1", "sb")
    agent = _agent(ex)
    conv = _Conv()
    agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                              "path": PATH, "old_str": "a",
                                              "new_str": "b"}), conv, "turn-1")
    assert conv.results[0].count("without reading it first") == 1


def test_an_explicit_pipeline_still_wins_over_a_mounted_one():
    """Deferring to the executor's chain must not override an explicit choice."""
    from swebench.agent.pipeline import piped_executor

    ex = piped_executor(ToolPipeline([]), lambda t, a: _ok(), "w1", "sb")
    agent = _agent(ex, pipeline=ToolPipeline([GuardrailInterceptor()]))
    conv = _Conv()
    agent._run_tool(_tool_call("text_editor", {"command": "str_replace",
                                              "path": PATH, "old_str": "a",
                                              "new_str": "b"}), conv, "turn-1")
    assert "without reading it first" in conv.results[0]


def test_a_plain_executor_still_gets_the_default_chain():
    """The deference must key off an actually-mounted chain, not fire always."""
    agent = _agent(lambda t, a: _ok("z" * 40000))
    conv = _Conv()
    agent._run_tool(_tool_call("shell", {"command": "cat big"}), conv, "turn-1")
    assert "characters truncated" in conv.results[0]


# --------------------------------------------------------------------------- #
#  A seat that rewrites a result must not destroy the rest of it
# --------------------------------------------------------------------------- #

def test_no_seat_drops_the_structured_outcome():
    """Every seat here rebuilds ToolResult to change one field, and `outcome` has
    to survive the rebuild: it is the runtime's structured report (exit code, the
    two streams, byte counts), and a seat further out may want to read it.

    The guardrail seat used to drop it. That was invisible while it was outermost
    -- nothing downstream looked -- and became reachable as soon as `extra=` let a
    caller mount a seat outside it. Swept across the chain rather than tested on
    one seat, so a new seat inherits the check."""
    from swebench.models import CommandOutcome

    outcome = CommandOutcome(exit_code=1, stdout="out", stderr="err",
                             stdout_bytes=3, stderr_bytes=3)
    edit = {"command": "str_replace", "path": PATH, "old_str": "a", "new_str": "b"}

    for seat in default_pipeline().interceptors:
        # Long enough to trigger truncation, and an edit so the guardrail warns:
        # each seat must be on its rewriting path, not its pass-through path.
        incoming = ToolResult(success=False, output="x" * 40_000,
                              error="y" * 40_000, outcome=outcome)
        ctx = CallContext(agent_id="A", sandbox_id="sb", tool_name="text_editor",
                          args=dict(edit), metadata={})
        pipe = ToolPipeline([seat])
        result = pipe.execute(ctx, lambda t, a: incoming)
        assert result.outcome is outcome, \
            f"{seat.name} dropped the outcome while rewriting the result"


def test_the_whole_default_chain_preserves_the_outcome():
    """End to end, not seat by seat: what the agent loop receives still carries
    the structured report."""
    from swebench.models import CommandOutcome

    outcome = CommandOutcome(exit_code=1, stdout="out", stderr="err")
    incoming = ToolResult(success=False, output="x" * 40_000, error=None,
                          outcome=outcome)
    ctx = CallContext(agent_id="A", sandbox_id="sb", tool_name="text_editor",
                      args={"command": "str_replace", "path": PATH,
                            "old_str": "a", "new_str": "b"}, metadata={})
    result = default_pipeline().execute(ctx, lambda t, a: incoming)
    assert result.outcome is outcome
