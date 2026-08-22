"""Unit tests for the tool interceptor pipeline (swebench.agent.pipeline).

No Docker, no model calls — the interceptors are exercised against plain
functions, and where a filesystem is needed, tests/fake_sandbox.py stands in.

Covered:
- onion ordering: befores in order, afters in REVERSE
- short-circuit / reject still unwind already-entered afters (audit sees all)
- Rewrite composition + context immutability
- per-interceptor fail-open vs fail-closed (before and after hooks)
- tools filtering, inner-exception conversion, plugin loading
- piped_executor: the harness-side mount (identity binding, raw-executor
  metadata, per-call sandbox-id resolution, and two agents sharing one
  stateful interceptor -- what a multi-agent topology needs from the mount)
"""

from __future__ import annotations

import pytest

from swebench.agent.pipeline import (
    CallContext,
    Continue,
    Reject,
    Rewrite,
    ShortCircuit,
    ToolInterceptor,
    ToolPipeline,
    load_pipeline,
    piped_executor,
)
from swebench.models import ToolResult

PATH = "/testbed/target.txt"


def _ok(output: str = "ok") -> ToolResult:
    return ToolResult(success=True, output=output)


def _inner_recorder(log: list):
    def inner(tool: str, args: dict) -> ToolResult:
        log.append(("inner", tool, dict(args)))
        return _ok("ran")
    return inner


def _ctx(tool: str = "shell", args: dict | None = None, agent: str = "A") -> CallContext:
    return CallContext(agent_id=agent, sandbox_id="sb-1", tool_name=tool,
                       args=dict(args or {}), metadata={})


class Tap(ToolInterceptor):
    """Records its hook invocations and passes everything through."""

    def __init__(self, label: str, log: list, tools="*"):
        self.label = label
        self.log = log
        self.tools = tools

    def before(self, ctx: CallContext):
        self.log.append((self.label, "before", ctx.tool_name, dict(ctx.args)))
        return Continue()

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        self.log.append((self.label, "after", result.success, result.output))
        return result


# --------------------------------------------------------------------------- #
#  Onion semantics
# --------------------------------------------------------------------------- #

def test_empty_pipeline_calls_inner_directly():
    log = []
    result = ToolPipeline([]).execute(_ctx("shell", {"command": "ls"}),
                                      _inner_recorder(log))
    assert result.success and result.output == "ran"
    assert log == [("inner", "shell", {"command": "ls"})]


def test_onion_before_in_order_after_in_reverse():
    log = []
    ToolPipeline([Tap("a", log), Tap("b", log)]).execute(_ctx(), _inner_recorder(log))
    assert [e[:2] for e in log] == [
        ("a", "before"), ("b", "before"), ("inner", "shell"),
        ("b", "after"), ("a", "after"),
    ]


def test_reject_skips_inner_and_becomes_failed_result():
    class Rejecter(ToolInterceptor):
        def before(self, ctx):
            return Reject("not allowed")

    log = []
    result = ToolPipeline([Rejecter()]).execute(_ctx(), _inner_recorder(log))
    assert not result.success
    assert result.output == "not allowed"
    assert result.error == "rejected by Rejecter"
    assert log == []                                       # inner never ran


def test_short_circuit_skips_inner_and_deeper_but_unwinds_entered_afters():
    log = []
    canned = ToolResult(success=True, output="cached")

    class Cache(ToolInterceptor):
        def before(self, ctx):
            return ShortCircuit(canned)

    result = ToolPipeline([Tap("audit", log), Cache(), Tap("deeper", log)]) \
        .execute(_ctx(), _inner_recorder(log))
    assert result.output == "cached"
    assert [e[:2] for e in log] == [("audit", "before"), ("audit", "after")]
    assert log[-1] == ("audit", "after", True, "cached")   # audit saw the result


def test_audit_style_interceptor_still_sees_rejected_calls():
    """Short-circuits still unwind the onion (ARCHITECTURE.md pipeline rule)."""
    log = []

    class Rejecter(ToolInterceptor):
        def before(self, ctx):
            log.append(("rejecter", "before"))
            return Reject("nope")

        def after(self, ctx, result):
            log.append(("rejecter", "after"))
            return result

    result = ToolPipeline([Tap("audit", log), Rejecter()]) \
        .execute(_ctx(), _inner_recorder(log))
    assert not result.success
    assert ("audit", "after", False, "nope") in log        # audit saw the rejection
    assert ("rejecter", "after") not in log                # terminator's after skipped


# --------------------------------------------------------------------------- #
#  Rewrite
# --------------------------------------------------------------------------- #

class AddKey(ToolInterceptor):
    def __init__(self, key: str, value):
        self.key, self.value = key, value

    def before(self, ctx):
        return Rewrite({**ctx.args, self.key: self.value})


def test_rewrite_composes_and_inner_sees_final_args():
    log = []
    original = _ctx("shell", {"command": "ls"})
    result = ToolPipeline([AddKey("a", 1), AddKey("b", 2)]) \
        .execute(original, _inner_recorder(log))
    assert result.success
    assert log == [("inner", "shell", {"command": "ls", "a": 1, "b": 2})]
    assert original.args == {"command": "ls"}              # contexts are immutable


def test_after_receives_the_context_its_before_saw():
    seen = {}

    class Rewriter(ToolInterceptor):
        def before(self, ctx):
            return Rewrite({**ctx.args, "extra": True})

        def after(self, ctx, result):
            seen["rewriter_after"] = dict(ctx.args)
            return result

    class Deeper(ToolInterceptor):
        def before(self, ctx):
            seen["deeper_before"] = dict(ctx.args)
            return Continue()

        def after(self, ctx, result):
            seen["deeper_after"] = dict(ctx.args)
            return result

    ToolPipeline([Rewriter(), Deeper()]).execute(
        _ctx("shell", {"command": "ls"}), lambda t, a: _ok())
    assert seen["rewriter_after"] == {"command": "ls"}     # its own pre-rewrite view
    assert seen["deeper_before"] == {"command": "ls", "extra": True}
    assert seen["deeper_after"] == {"command": "ls", "extra": True}


# --------------------------------------------------------------------------- #
#  Fail-open vs fail-closed
# --------------------------------------------------------------------------- #

class _BeforeBoom(ToolInterceptor):
    def __init__(self, fail_mode: str):
        self.fail_mode = fail_mode

    def before(self, ctx):
        raise RuntimeError("boom")


class _AfterBoom(ToolInterceptor):
    def __init__(self, fail_mode: str):
        self.fail_mode = fail_mode

    def after(self, ctx, result):
        raise RuntimeError("boom")


def test_fail_open_before_logs_and_continues(caplog):
    log = []
    with caplog.at_level("WARNING", logger="ash.pipeline"):
        result = ToolPipeline([_BeforeBoom("open")]).execute(_ctx(), _inner_recorder(log))
    assert result.success and log                          # call went through
    assert any("failed open" in r.message for r in caplog.records)


def test_fail_closed_before_rejects():
    log = []
    result = ToolPipeline([_BeforeBoom("closed")]).execute(_ctx(), _inner_recorder(log))
    assert not result.success
    assert result.error == "rejected by _BeforeBoom"
    assert "failed closed" in result.output
    assert log == []                                       # inner never ran


def test_fail_open_after_passes_result_through():
    result = ToolPipeline([_AfterBoom("open")]).execute(_ctx(), lambda t, a: _ok("fine"))
    assert result.success and result.output == "fine"


def test_fail_closed_after_rejects_and_outer_unwind_continues():
    log = []
    result = ToolPipeline([Tap("outer", log), _AfterBoom("closed")]) \
        .execute(_ctx(), lambda t, a: _ok("fine"))
    assert not result.success
    assert result.error == "rejected by _AfterBoom"
    assert any(e[:2] == ("outer", "after") and e[2] is False for e in log)


def test_invalid_verdict_is_a_failure_honoring_fail_mode():
    class Junk(ToolInterceptor):
        fail_mode = "closed"

        def before(self, ctx):
            return "not a verdict"

    result = ToolPipeline([Junk()]).execute(_ctx(), lambda t, a: _ok())
    assert not result.success
    assert result.error == "rejected by Junk"


# --------------------------------------------------------------------------- #
#  Tools filtering + inner failures
# --------------------------------------------------------------------------- #

def test_tools_filter_skips_non_matching_calls():
    log = []
    pipe = ToolPipeline([Tap("shell_only", log, tools={"shell"})])
    pipe.execute(_ctx("text_editor", {"command": "view", "path": PATH}), _inner_recorder(log))
    assert [e[0] for e in log] == ["inner"]                # neither hook ran
    pipe.execute(_ctx("shell", {"command": "ls"}), _inner_recorder(log))
    assert ("shell_only", "before", "shell", {"command": "ls"}) in log


def test_inner_exception_becomes_failed_result_and_still_unwinds():
    log = []

    def broken(tool, args):
        raise RuntimeError("connection lost")

    result = ToolPipeline([Tap("audit", log)]).execute(_ctx(), broken)
    assert not result.success
    assert result.error == "connection lost"
    assert ("audit", "after", False, "") in log            # audit saw the failure


# --------------------------------------------------------------------------- #
#  Plugin loading
# --------------------------------------------------------------------------- #

_PLUGIN_SOURCE = """\
from swebench.agent.pipeline import Continue, ToolInterceptor

class Stamp(ToolInterceptor):
    def before(self, ctx):
        ctx.metadata["stamped"] = True
        return Continue()

PIPELINE = [Stamp()]
"""


def test_load_pipeline_from_plugins_file(tmp_path):
    plugin = tmp_path / "my_plugins.py"
    plugin.write_text(_PLUGIN_SOURCE)
    pipe = load_pipeline(str(plugin))
    assert len(pipe.interceptors) == 1
    ctx = _ctx()
    assert pipe.execute(ctx, lambda t, a: _ok()).success
    assert ctx.metadata["stamped"] is True


def test_load_pipeline_requires_module_level_list(tmp_path):
    plugin = tmp_path / "empty_plugins.py"
    plugin.write_text("X = 1\n")
    with pytest.raises(ValueError, match="PIPELINE"):
        load_pipeline(str(plugin))


# --------------------------------------------------------------------------- #
#  piped_executor — the harness-side mount
# --------------------------------------------------------------------------- #

def test_piped_executor_is_a_plain_executor():
    # The whole point of the mount: an AshAgent consumes it without knowing
    # the pipeline exists, and the onion runs on every call.
    log = []
    execute = piped_executor(ToolPipeline([Tap("a", log)]),
                             _inner_recorder(log), agent_id="A")
    result = execute("shell", {"command": "ls"})
    assert result.success and result.output == "ran"
    assert [e[:2] for e in log] == [("a", "before"), ("inner", "shell"), ("a", "after")]


def test_piped_executor_binds_identity_and_supplies_raw_executor():
    seen = {}

    class Probe(ToolInterceptor):
        def before(self, ctx):
            seen["agent"] = ctx.agent_id
            seen["sandbox"] = ctx.sandbox_id
            seen["raw"] = ctx.metadata.get("executor")
            return Continue()

    def inner(tool, args):
        return _ok()

    execute = piped_executor(ToolPipeline([Probe()]), inner,
                             agent_id="worker-1", sandbox_id="sb-42")
    execute("shell", {})
    assert seen["agent"] == "worker-1"
    assert seen["sandbox"] == "sb-42"
    # The contract for probe traffic: the raw executor rides in metadata so a
    # interceptor's own calls never re-enter the pipeline.
    assert seen["raw"] is inner


def test_piped_executor_resolves_callable_sandbox_id_per_call():
    # A session hands out executors before its sandbox exists; the id must be
    # read at call time, not frozen at mount time.
    ids = []

    class Probe(ToolInterceptor):
        def before(self, ctx):
            ids.append(ctx.sandbox_id)
            return Continue()

    current = {"id": "unknown"}
    execute = piped_executor(ToolPipeline([Probe()]), lambda t, a: _ok(),
                             agent_id="A", sandbox_id=lambda: current["id"])
    execute("shell", {})
    current["id"] = "sb-7"          # sandbox spawned after the mount
    execute("shell", {})
    assert ids == ["unknown", "sb-7"]


def test_piped_executor_does_not_leak_arg_mutations_between_layers():
    # The mount copies args into the context, so a Rewrite downstream can
    # never mutate the dict the agent still holds.
    class Rewriter(ToolInterceptor):
        def before(self, ctx):
            return Rewrite({**ctx.args, "injected": True})

    captured = {}

    def inner(tool, args):
        captured.update(args)
        return _ok()

    execute = piped_executor(ToolPipeline([Rewriter()]), inner, agent_id="A")
    original = {"command": "ls"}
    execute("shell", original)
    assert captured == {"command": "ls", "injected": True}
    assert original == {"command": "ls"}


def test_two_mounted_agents_share_one_chain_and_its_state():
    """One interceptor instance, two executors: state is shared, so an interceptor can
    arbitrate between agents. This is what a multi-agent topology needs from the
    mount, and it was previously asserted through Waggle -- the property belongs to
    piped_executor, so it is checked here without a coordinator.
    """
    class FirstWriterWins(ToolInterceptor):
        """Minimal arbitration: whoever writes a path first owns it."""
        tools = {"text_editor"}

        def __init__(self):
            self.owner: dict[str, str] = {}

        def before(self, ctx):
            path = ctx.args.get("path", "")
            held = self.owner.setdefault(path, ctx.agent_id)
            if held != ctx.agent_id:
                return Reject(f"{path} is held by {held}")
            return Continue()

    interceptor = FirstWriterWins()                  # ONE shared instance
    pipeline = ToolPipeline([interceptor])
    a = piped_executor(pipeline, lambda t, args: _ok(), agent_id="A")
    b = piped_executor(pipeline, lambda t, args: _ok(), agent_id="B")

    assert a("text_editor", {"command": "write", "path": PATH}).success
    loser = b("text_editor", {"command": "write", "path": PATH})
    assert not loser.success and "held by A" in loser.output
    assert interceptor.owner[PATH] == "A", \
        "the two mounts did not share the interceptor's state"


# --------------------------------------------------------------------------- #
#  Why the verdicts are four and not two
# --------------------------------------------------------------------------- #
#  Rewrite and ShortCircuit have no production caller since Waggle was removed,
#  so these pin what would be lost by deleting them. Both failures are quiet:
#  the workaround runs fine and reports the wrong thing.

def test_rewrite_keeps_each_interceptors_after_consistent_with_its_own_before():
    """An audit interceptor records "the agent asked for X" on the way in and "the result
    was Y" on the way out. Rewrite derives a fresh context, so an inner interceptor adding
    `timeout 300` cannot make that pair describe two different requests."""
    class Auditor(ToolInterceptor):
        def __init__(self):
            self.at_before = self.at_after = None

        def before(self, ctx):
            self.at_before = ctx.args["command"]
            return Continue()

        def after(self, ctx, result):
            self.at_after = ctx.args["command"]
            return result

    class AddTimeout(ToolInterceptor):
        def before(self, ctx):
            return Rewrite({**ctx.args, "command": "timeout 300 " + ctx.args["command"]})

    auditor = Auditor()
    inner_saw = {}

    def inner(tool, args):
        inner_saw.update(args)
        return _ok()

    piped_executor(ToolPipeline([auditor, AddTimeout()]), inner,
                   agent_id="A")("shell", {"command": "pytest -x"})

    assert auditor.at_before == auditor.at_after == "pytest -x"
    assert inner_saw["command"] == "timeout 300 pytest -x", "the rewrite never applied"


def test_mutating_args_in_place_is_what_rewrite_avoids():
    """The workaround, shown failing: CallContext is frozen, but that binds the
    fields, not the dict inside -- so an in-place edit leaks outward and the outer
    interceptor's after no longer matches its own before."""
    class Auditor(ToolInterceptor):
        def __init__(self):
            self.at_before = self.at_after = None

        def before(self, ctx):
            self.at_before = ctx.args["command"]
            return Continue()

        def after(self, ctx, result):
            self.at_after = ctx.args["command"]
            return result

    class MutateInPlace(ToolInterceptor):
        def before(self, ctx):
            ctx.args["command"] = "timeout 300 " + ctx.args["command"]
            return Continue()

    auditor = Auditor()
    piped_executor(ToolPipeline([auditor, MutateInPlace()]),
                   lambda t, a: _ok(), agent_id="A")("shell", {"command": "pytest -x"})

    assert auditor.at_before == "pytest -x"
    assert auditor.at_after == "timeout 300 pytest -x"   # the leak Rewrite prevents


def test_short_circuit_can_answer_successfully_and_reject_cannot():
    """A cache interceptor holding a passing test result has to report success. Through
    Reject it would report failure on a green suite, and an agent reading that goes
    off to fix a test that was never broken."""
    answer = ToolResult(success=True, output="5 passed")

    class Cache(ToolInterceptor):
        def before(self, ctx):
            return ShortCircuit(answer)

    class CacheViaReject(ToolInterceptor):
        def before(self, ctx):
            return Reject("5 passed")

    def inner(tool, args):
        raise AssertionError("the inner executor should not be reached")

    served = ToolPipeline([Cache()]).execute(
        CallContext("A", "sb", "shell", {"command": "pytest"}), inner)
    assert served.success is True and served.output == "5 passed"

    faked = ToolPipeline([CacheViaReject()]).execute(
        CallContext("A", "sb", "shell", {"command": "pytest"}), inner)
    assert faked.success is False, "Reject must keep meaning 'this did not happen'"
