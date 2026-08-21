"""Mounting your own interceptor: write a class, put it in a list.

The mechanism is deliberately not a registry or a YAML schema. ADR-3 says policy
is Python code -- a config dialect for "reject writes under src/ unless the agent
read the file first" becomes a crippled programming language -- so a plugins file
holds the instances and config holds only the path to it.

Covered:
- `extra=` puts your seats outside the defaults, and outside is what makes them
  able to reject a call before the inner seats spend anything on it
- the config key reaches the agent's chain
- with no config key, nothing changes
"""

from __future__ import annotations

import textwrap

import pytest

from swebench.agent import AshAgent
from swebench.agent.seats import default_pipeline
from swebench.agent.pipeline import (CallContext, Continue, Reject,
                                     ToolInterceptor, ToolPipeline,
                                     load_pipeline)
from swebench.models import AgentConfig, ToolResult

PLUGIN_SOURCE = textwrap.dedent('''
    from swebench.agent.pipeline import (CallContext, Continue, Reject,
                                         ToolInterceptor, Verdict)

    class NoDestructiveShell(ToolInterceptor):
        tools = {"shell"}
        fail_mode = "closed"

        def before(self, ctx):
            if "rm -rf" in ctx.args.get("command", ""):
                return Reject("refused: rm -rf is not allowed")
            return Continue()

    PIPELINE = [NoDestructiveShell()]
''')


class Counter(ToolInterceptor):
    """An observer, to check what a seat placed outside actually sees."""

    tools = "*"

    def __init__(self):
        self.seen = []

    def before(self, ctx: CallContext) -> Continue:
        self.seen.append(ctx.tool_name)
        return Continue()


class Blocker(ToolInterceptor):
    tools = {"shell"}

    def before(self, ctx: CallContext) -> Reject:
        return Reject("blocked")


@pytest.fixture
def plugin_file(tmp_path):
    path = tmp_path / "my_seats.py"
    path.write_text(PLUGIN_SOURCE)
    return str(path)


def _ok(tool, args):
    return ToolResult(success=True, output="ok")


# --------------------------------------------------------------------------- #
#  extra=
# --------------------------------------------------------------------------- #

def test_extra_seats_are_mounted_with_the_defaults():
    chain = default_pipeline(extra=[Blocker()])
    names = [i.name for i in chain.interceptors]
    assert names == ["Blocker", "GuardrailInterceptor",
                     "TruncateInterceptor", "OutcomePresenter"]


def test_no_extra_leaves_the_default_chain_alone():
    """The argument is additive: absent it, this is the historical behaviour."""
    assert [i.name for i in default_pipeline().interceptors] == [
        "GuardrailInterceptor", "TruncateInterceptor", "OutcomePresenter"]


def test_an_extra_seat_can_stop_a_call_reaching_the_runtime():
    reached = []

    def runtime(tool, args):
        reached.append(args.get("command"))
        return ToolResult(success=True, output="ok")

    chain = default_pipeline(extra=[Blocker()])
    result = chain.execute(
        CallContext("a", "sb", "shell", {"command": "rm -rf /"}), runtime)

    assert result.success is False
    assert reached == [], "the call reached the runtime despite a rejection"


def test_extra_seats_sit_outside_so_they_see_rejected_calls():
    """Why `extra` mounts outside rather than inside: a short circuit unwinds the
    onion through every seat already entered, so an outer observer sees calls the
    inner seats refuse. An inner one would not -- which is the whole reason to
    care where the seat goes."""
    counter = Counter()
    chain = default_pipeline(extra=[counter, Blocker()])
    chain.execute(CallContext("a", "sb", "shell", {"command": "rm -rf /"}), _ok)
    assert counter.seen == ["shell"]


# --------------------------------------------------------------------------- #
#  Through a plugins file, the way config does it
# --------------------------------------------------------------------------- #

def test_a_plugins_file_supplies_the_seats(plugin_file):
    chain = default_pipeline(extra=load_pipeline(plugin_file).interceptors)
    assert [i.name for i in chain.interceptors][0] == "NoDestructiveShell"


def test_the_seats_from_a_file_actually_run(plugin_file):
    chain = default_pipeline(extra=load_pipeline(plugin_file).interceptors)
    refused = chain.execute(
        CallContext("a", "sb", "shell", {"command": "rm -rf /testbed"}), _ok)
    allowed = chain.execute(
        CallContext("a", "sb", "shell", {"command": "pytest -x"}), _ok)
    assert refused.success is False and "rm -rf" in refused.output
    assert allowed.success is True


def test_a_bad_plugins_path_is_reported_not_ignored(tmp_path):
    """A silently ignored guardrail is worse than a crash: the run looks governed."""
    with pytest.raises(ValueError):
        load_pipeline(str(tmp_path / "does_not_exist.py"))


def test_a_file_without_PIPELINE_is_rejected(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("X = 1\n")
    with pytest.raises(ValueError):
        load_pipeline(str(path))


# --------------------------------------------------------------------------- #
#  Reaching the agent
# --------------------------------------------------------------------------- #

def test_a_chain_assigned_after_construction_is_the_one_used():
    """How the harness mounts it: build the agent, then set .pipeline. run()
    re-resolves per run, so assigning after __init__ has to take effect."""
    agent = AshAgent(AgentConfig(), executor=_ok)
    agent.pipeline = default_pipeline(extra=[Blocker()])
    assert [i.name for i in agent._resolve_pipeline().interceptors][0] == "Blocker"

    result = agent._governed({})("shell", {"command": "rm -rf /"})
    assert result.success is False


def test_the_harness_reads_the_config_key():
    """Guards the wiring: the key has to be read somewhere, or the flag is inert
    -- which is exactly how `custom_tools_dir` came to never reach AgentConfig."""
    import inspect
    from swebench.harnesses import litellm
    source = inspect.getsource(litellm)
    assert 'c.get("interceptors")' in source
    assert "load_pipeline" in source


def test_the_config_key_survives_the_flag_flattening():
    """`execution.interceptors` has to be in the mapping table, or a config that
    sets it is silently dropped."""
    import inspect
    from swebench import __main__ as main
    assert '("execution", "interceptors")' in inspect.getsource(main)


def test_a_broken_plugin_is_reported_as_a_plugin_failure(tmp_path):
    """A syntax error used to surface as a bare SyntaxError with no mention of
    plugins, so the operator had to guess which file the run choked on."""
    path = tmp_path / "broken.py"
    path.write_text("PIPELINE = [\n")          # unclosed bracket
    with pytest.raises(ValueError) as caught:
        load_pipeline(str(path))
    assert "broken.py" in str(caught.value)
