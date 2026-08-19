"""Unit tests for the tool interceptor pipeline (swebench.agent.pipeline)
and for Waggle mounted as an interceptor (swebench.agent.waggle).

No Docker, no model calls — the sandbox is the in-memory FakeSandbox from
test_waggle (imported, not copied).

Covered:
- onion ordering: befores in order, afters in REVERSE
- short-circuit / reject still unwind already-entered afters (audit sees all)
- Rewrite composition + context immutability
- per-interceptor fail-open vs fail-closed (before and after hooks)
- tools filtering, inner-exception conversion, plugin loading
- WaggleInterceptor reproduces CoordinatedExecutor's OCC behavior
- WagglePolicy hooks: on_write / on_conflict / on_drift / on_commit,
  hook-exception fallback to default OCC
- piped_executor: the harness-side mount (identity binding, raw-executor
  metadata, per-call sandbox-id resolution, two mounted agents sharing
  Waggle arbitration)
"""

from __future__ import annotations

import time

import pytest

from swebench.agent import waggle
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
from swebench.agent.waggle import (
    Allow,
    CoordinatedExecutor,
    Ignore,
    WaggleInterceptor,
    WagglePolicy,
    WorkspaceCoordinator,
)
from swebench.models import ToolResult
from swebench.tests.test_waggle import FakeSandbox

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
#  WaggleInterceptor: OCC behavior reproduced through the pipeline
# --------------------------------------------------------------------------- #

def _waggle_setup(files: dict[str, str] | None = None, ttl: float = 5.0,
                  policy: WagglePolicy | None = None):
    sandbox = FakeSandbox(files)
    interceptor = WaggleInterceptor(state=WorkspaceCoordinator(ttl=ttl), policy=policy)
    pipe = ToolPipeline([interceptor])

    def call(agent: str, tool: str, args: dict) -> ToolResult:
        executor = sandbox.executor()
        ctx = CallContext(agent_id=agent, sandbox_id="default", tool_name=tool,
                          args=dict(args), metadata={"executor": executor})
        return pipe.execute(ctx, executor)

    return sandbox, interceptor, call


def _read(call, agent: str, path: str = PATH) -> ToolResult:
    return call(agent, "text_editor", {"command": "view", "path": path})


def _write(call, agent: str, text: str, path: str = PATH) -> ToolResult:
    return call(agent, "text_editor",
                {"command": "write", "path": path, "file_text": text})


def test_waggle_interceptor_read_write_records_history():
    sandbox, icpt, call = _waggle_setup({PATH: "base"})
    assert _read(call, "A").success
    assert _write(call, "A", "v2 content").success
    rec = icpt.state.file("default", PATH)
    assert rec.version == 2
    assert [r.op for r in rec.history] == ["baseline", "write"]
    assert sandbox.read(PATH) == "v2 content"


def test_waggle_interceptor_rejects_blind_write():
    sandbox, _, call = _waggle_setup({PATH: "base"})
    result = _write(call, "A", "blind")
    assert not result.success
    assert "text_editor view first" in result.output
    assert sandbox.read(PATH) == "base"


def test_waggle_interceptor_stale_write_rejected_then_retry_wins():
    sandbox, icpt, call = _waggle_setup({PATH: "line1\nline2"})
    _read(call, "A"), _read(call, "B")
    assert _write(call, "A", "line1\nline2 changed by A").success
    rejection = _write(call, "B", "line1 changed by B\nline2")
    assert not rejection.success
    assert "[WAGGLE]" in rejection.output
    assert "your snapshot : v1" in rejection.output
    assert "+line2 changed by A" in rejection.output       # unified diff present
    reservation = icpt.state.file("default", PATH).reservation
    assert reservation and reservation.agent == "B"        # loser is protected
    _read(call, "B")
    assert _write(call, "B", "merged by B").success
    assert sandbox.read(PATH) == "merged by B"


def test_waggle_interceptor_detects_shell_drift():
    sandbox, icpt, call = _waggle_setup({PATH: "base"})
    _read(call, "A"), _read(call, "B")
    sandbox.mutate(PATH, "changed out of band")
    call("B", "shell", {"command": "run-something"})
    rec = icpt.state.file("default", PATH)
    assert rec.version == 2
    assert rec.history[-1].op == "external"
    rejection = _write(call, "A", "based on stale v1")
    assert not rejection.success
    assert "external (detected by B)" in rejection.output


def test_waggle_interceptor_matches_coordinated_executor_ledger():
    """Same op sequence through both mountings -> identical version ledger."""
    script = [("A", "read", ""), ("B", "read", ""), ("A", "write", "by A"),
              ("B", "write", "by B"), ("B", "read", ""), ("B", "write", "by B v2")]

    def run_via_executor():
        sandbox = FakeSandbox({PATH: "base"})
        state = WorkspaceCoordinator(ttl=5.0)
        agents = {a: CoordinatedExecutor(sandbox.executor(), state, agent_id=a)
                  for a in ("A", "B")}
        outcomes = []
        for agent, op, text in script:
            if op == "read":
                outcomes.append(agents[agent]("text_editor", {"command": "view", "path": PATH}).success)
            else:
                outcomes.append(agents[agent]("text_editor", {
                    "command": "write", "path": PATH, "file_text": text}).success)
        return outcomes, state.dump()

    def run_via_pipeline():
        _, icpt, call = _waggle_setup({PATH: "base"})
        outcomes = [(_read(call, agent) if op == "read"
                     else _write(call, agent, text)).success
                    for agent, op, text in script]
        return outcomes, icpt.dump()

    exec_outcomes, exec_dump = run_via_executor()
    pipe_outcomes, pipe_dump = run_via_pipeline()

    def strip(dump: dict) -> dict:  # timestamps differ run-to-run
        return {key: [{f: r[f] for f in ("version", "author", "op", "bytes")}
                      for r in records]
                for key, records in dump.items()}

    assert exec_outcomes == pipe_outcomes == [True, True, True, False, True, True]
    assert strip(exec_dump) == strip(pipe_dump)


def test_waggle_interceptor_fails_closed_without_executor_metadata():
    sandbox = FakeSandbox({PATH: "base"})
    pipe = ToolPipeline([WaggleInterceptor(state=WorkspaceCoordinator(ttl=5.0))])
    ctx = CallContext(agent_id="A", sandbox_id="default", tool_name="text_editor",
                      args={"command": "write", "path": PATH, "file_text": "x"},
                      metadata={})                         # executor missing
    result = pipe.execute(ctx, sandbox.executor())
    assert not result.success
    assert result.error == "rejected by WaggleInterceptor"
    assert sandbox.read(PATH) == "base"                    # nothing written


# --------------------------------------------------------------------------- #
#  WagglePolicy hooks (invoked inside the arbitration critical section)
# --------------------------------------------------------------------------- #

class _RejectWrites(WagglePolicy):
    def on_write(self, ctx):
        return waggle.Reject("A owns this file")


def test_policy_on_write_reject_blocks_the_write():
    sandbox, _, call = _waggle_setup({PATH: "base"}, policy=_RejectWrites())
    _read(call, "B")
    result = _write(call, "B", "denied")
    assert not result.success
    assert "rejected by policy" in result.output
    assert "A owns this file" in result.output
    assert sandbox.read(PATH) == "base"


class _AllowAll(WagglePolicy):
    def on_write(self, ctx):
        return Allow()


def test_policy_on_write_allow_bypasses_occ():
    sandbox, icpt, call = _waggle_setup({PATH: "base"}, policy=_AllowAll())
    result = _write(call, "B", "forced")                   # B never read the file
    assert result.success
    assert sandbox.read(PATH) == "forced"
    assert icpt.state.file("default", PATH).history[-1].author == "B"


class _LastWriterWins(WagglePolicy):
    def on_conflict(self, ctx):
        assert ctx.diff                                    # kernel-computed context
        assert ctx.snapshot_version == 1
        return Allow()


def test_policy_on_conflict_allow_overrides_stale_rejection():
    sandbox, _, call = _waggle_setup({PATH: "base"}, policy=_LastWriterWins())
    _read(call, "A"), _read(call, "B")
    assert _write(call, "A", "by A").success
    result = _write(call, "B", "by B")                     # stale, but policy allows
    assert result.success
    assert sandbox.read(PATH) == "by B"


class _CommitLog(WagglePolicy):
    def __init__(self):
        self.commits = []

    def on_commit(self, ctx):
        self.commits.append((ctx.agent_id, ctx.path, ctx.current_version))


def test_policy_on_commit_observes_commits():
    policy = _CommitLog()
    _, _, call = _waggle_setup({PATH: "base"}, policy=policy)
    _read(call, "A")
    assert _write(call, "A", "v2").success
    assert policy.commits == [("A", PATH, 2)]


class _IgnoreDrift(WagglePolicy):
    def on_drift(self, ctx):
        return Ignore()


def test_policy_on_drift_ignore_skips_recording():
    sandbox, icpt, call = _waggle_setup({PATH: "base"}, policy=_IgnoreDrift())
    _read(call, "A")
    sandbox.mutate(PATH, "out of band")
    call("A", "shell", {"command": "anything"})
    assert icpt.state.file("default", PATH).version == 1   # drift not recorded


class _Crashing(WagglePolicy):
    def on_write(self, ctx):
        raise RuntimeError("policy bug")

    def on_conflict(self, ctx):
        raise RuntimeError("policy bug")


def test_policy_exception_falls_back_to_default_occ(caplog):
    sandbox, _, call = _waggle_setup({PATH: "base"}, policy=_Crashing())
    with caplog.at_level("WARNING", logger="ash.waggle"):
        result = _write(call, "A", "blind")                # default OCC: unread -> reject
    assert not result.success
    assert "text_editor view first" in result.output
    assert any("on_write" in r.message for r in caplog.records)
    _read(call, "A")
    assert _write(call, "A", "v2").success                 # normal path still works
    assert sandbox.read(PATH) == "v2"


class _AlwaysWait(WagglePolicy):
    def on_write(self, ctx):
        return waggle.Wait()


def test_policy_wait_times_out_within_the_write_deadline():
    _, _, call = _waggle_setup({PATH: "base"}, ttl=0.4, policy=_AlwaysWait())
    _read(call, "A")
    start = time.monotonic()
    result = _write(call, "A", "never lands")
    assert not result.success
    assert "[WAGGLE]" in result.output and "on hold" in result.output
    assert time.monotonic() - start < 5.0


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
    # Waggle's contract: the raw executor rides in metadata so probe traffic
    # and arbitrated writes never re-enter the pipeline.
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


def test_two_mounted_agents_share_waggle_arbitration():
    # End-to-end through the mount: same kernel, two executors, OCC fires.
    # This is the manager-worker layout expressed via executor_for(pipeline=).
    sandbox = FakeSandbox({PATH: "base"})
    pipeline = ToolPipeline([WaggleInterceptor()])   # ONE shared instance
    a = piped_executor(pipeline, sandbox.executor(), agent_id="A")
    b = piped_executor(pipeline, sandbox.executor(), agent_id="B")

    for execute in (a, b):
        assert execute("text_editor", {"command": "view", "path": PATH}).success
    assert a("text_editor", {"command": "write", "path": PATH,
                             "file_text": "A wins"}).success
    stale = b("text_editor", {"command": "write", "path": PATH,
                              "file_text": "B stale"})
    assert not stale.success
    assert "[WAGGLE]" in stale.output          # rejection carries the diff
    assert sandbox.read(PATH) == "A wins"      # no lost update

    # The loser re-reads and re-applies — the LLM-as-merge-engine loop.
    assert b("text_editor", {"command": "view", "path": PATH}).success
    assert b("text_editor", {"command": "write", "path": PATH,
                             "file_text": "B rebased"}).success
    assert sandbox.read(PATH) == "B rebased"
