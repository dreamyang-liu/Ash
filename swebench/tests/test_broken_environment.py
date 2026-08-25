"""A dead sandbox is not the agent reporting success."""

from swebench.agent import (BROKEN_ENVIRONMENT_STRIKES, AshAgent,
                            _looks_unreachable)
from swebench.models import AgentConfig, ToolResult


def test_transport_failures_are_distinguished_from_failed_commands():
    """A command that ran and failed is normal work; a call that never reached
    the guest means the environment is gone."""
    def failed(error="", output=""):
        return ToolResult(success=False, output=output, error=error)

    assert _looks_unreachable(failed(
        error="Client error '404 Not Found' for url 'http://127.0.0.1:18000'"))
    assert _looks_unreachable(failed(error="Connection refused"))
    # Ordinary failures: a bad path, a failing build, a non-zero exit.
    assert not _looks_unreachable(failed(error="make: *** [all] Error 1"))
    assert not _looks_unreachable(failed(output="FileNotFoundError: /app/nope"))
    assert not _looks_unreachable(ToolResult(success=True, output="ok", error=""))


def test_a_dead_environment_ends_the_run_as_an_error_not_completed():
    """Measured: a 1473-turn run reported `completed` after its sandbox was
    deleted from under it. Every tool call answered 404, so the agent could
    only produce prose -- and prose with no tool call is exactly what "I am
    finished" looks like. The run then graded 0.0 with empty metrics, while a
    re-grade from its final checkpoint scored 8/43. Environment failure must
    not be recorded as a verdict on the work."""
    calls = {"n": 0}

    def dead_executor(name, args, timeout=None):
        calls["n"] += 1
        return ToolResult(
            success=False, output="",
            error="Client error '404 Not Found' for url 'http://127.0.0.1:18000'")

    class Msg:
        thinking_blocks = None

        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    def tool_call(i):
        return type("TC", (), {
            "id": f"c{i}", "type": "function",
            "function": type("F", (), {"name": "shell",
                                       "arguments": '{"command": "make"}'})()})()

    # The shape the real run ended in: tool call, then prose, alternating.
    script = []
    for i in range(BROKEN_ENVIRONMENT_STRIKES + 2):
        script.append(Msg("", [tool_call(i)]))
        script.append(Msg("The fix is not complete. Tools are unavailable."))

    seen = {"i": 0}

    class Resp:
        def __init__(self, message):
            self.choices = [type("C", (), {"message": message,
                                           "finish_reason": "stop"})()]
            self.usage = None

    def fake_completion(**kwargs):
        message = script[min(seen["i"], len(script) - 1)]
        seen["i"] += 1
        return Resp(message)

    import swebench.agent.llm as llm
    original = llm._get_litellm
    llm._get_litellm = lambda: (fake_completion, lambda **k: None)
    try:
        agent = AshAgent(AgentConfig(model="m", prompt_cache=False,
                                     step_limit=60),
                         executor=dead_executor, agent_id="a")
        agent.stream = False
        status = agent.run("build it", instance_id="x")
    finally:
        llm._get_litellm = original

    assert status == "environment_error", (
        f"got {status!r}: a run whose tools all 404 has not completed its work")
    assert calls["n"] >= BROKEN_ENVIRONMENT_STRIKES


def test_intermittent_transport_failures_do_not_end_a_run():
    """One 404 among working calls is a hiccup, not a dead environment: the
    counter resets on the next call that executes."""
    results = [
        ToolResult(success=False, output="", error="Connection reset by peer"),
        ToolResult(success=True, output="ok", error=""),
    ]

    def flaky(name, args, timeout=None):
        return results[min(len(results) - 1, flaky.n if hasattr(flaky, "n") else 0)]

    agent = AshAgent(AgentConfig(model="m"), executor=flaky, agent_id="a")
    agent.consecutive_tool_failures = 0
    from swebench.agent.conversation import Conversation
    from swebench.models import Trajectory
    conv = Conversation(Trajectory())

    class TC:
        id = "c1"
        function = type("F", (), {"name": "shell", "arguments": "{}"})()

    agent._pipeline = None
    agent._run_tool(TC(), conv, "turn-1")
    assert agent.consecutive_tool_failures == 1
    flaky.n = 1
    agent._run_tool(TC(), conv, "turn-2")
    assert agent.consecutive_tool_failures == 0, "a working call clears it"


def test_the_loop_honours_every_nudge_verdict():
    """The loop compared the nudge's return to "completed" and dropped
    anything else, so the environment-error verdict was silently ignored and
    the run went on to exhaust its step limit against a dead sandbox. Finish
    hooks exist to extend a *successful* stop; they do not get a say here."""
    import inspect
    from swebench.agent import AshAgent
    source = inspect.getsource(AshAgent.run)
    assert "elif verdict is not None:" in source
    assert "return verdict" in source
