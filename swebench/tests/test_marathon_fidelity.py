"""Staying on the same axis as the benchmark's published numbers."""

import json

import pytest

from swebench.marathon import _KNOWN_OPEN_TASKS, discover_tasks, load_task
from swebench.tests.test_marathon import write_task


def with_runner(root, restricted, open_):
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash"]
    for name in open_:
        lines.append(f"harbor run -p tasks/{name} -a claude-code -e modal")
    for name in restricted:
        lines.append(f"harbor run -p tasks/{name} -a claude-code -e modal "
                     f"--allow-agent-host api.anthropic.com "
                     f'--agent-kwarg disallowed_tools="WebSearch WebFetch"')
    (root / "scripts" / "run-benchmark.sh").write_text("\n".join(lines) + "\n")


def test_web_tools_follow_the_reference_runner(tmp_path):
    """The restriction lives in the run command, not in task.toml: the
    benchmark's own runner passes disallowed_tools for 16 of the 20 tasks,
    because their sandboxes block the network while a provider-side web tool
    would reach out anyway. Offering those tools would answer an easier
    question on most of the suite."""
    write_task(tmp_path, "zstd-decoder")
    write_task(tmp_path, "slack-clone")
    with_runner(tmp_path, restricted=["zstd-decoder"], open_=["slack-clone"])

    tasks = {t.instance_id: t for t in discover_tasks(tmp_path)}
    assert tasks["zstd-decoder"].internet_restricted
    assert not tasks["slack-clone"].internet_restricted


def test_an_unknown_task_is_treated_as_restricted(tmp_path):
    """Without the runner in the checkout, the published open set decides, and
    anything else is restricted -- the error that matters is handing an agent
    network access the benchmark denied every published trial."""
    write_task(tmp_path, "some-new-task")
    write_task(tmp_path, "slack-clone")
    tasks = {t.instance_id: t for t in discover_tasks(tmp_path)}
    assert tasks["some-new-task"].internet_restricted
    assert not tasks["slack-clone"].internet_restricted
    assert "slack-clone" in _KNOWN_OPEN_TASKS


def test_the_panel_drops_both_halves(tmp_path):
    """Schema and routing together: a tool the schema omits but the views still
    route is reachable through a rename, which is how a panel came to offer a
    parameter the runtime rejected."""
    from swebench.agent.tools import DEFAULT_PANEL, build_panel

    panel = build_panel(DEFAULT_PANEL, None)
    stripped = panel.without(("web_fetch", "web_search"))
    names = [e["function"]["name"] for e in stripped.schema]
    assert "web_fetch" not in names and "web_search" not in names
    assert "web_fetch" not in stripped.views
    assert "shell" in names and "shell" in stripped.views
    assert len(names) == len(panel.schema) - 2


def test_the_task_wall_clock_is_enforced():
    """task.toml allots 4 to 10 hours depending on the task, and the reference
    harness stops there. Without it a run could quietly exceed what every
    published trial was allowed -- incomparable in the direction that flatters
    us."""
    from swebench.harnesses.marathon import _Deadline, _make_deadline

    assert _make_deadline(0) is None, "no clock, no hook"

    hook = _make_deadline(3600)
    class Agent:
        def _trace(self, _text): pass
    hook(Agent(), None)          # plenty of time left: silent

    import swebench.harnesses.marathon as harness
    elapsed = iter([0.0, 4000.0])
    original = harness.__dict__.get("time")
    hook = _make_deadline(3600)
    with pytest.raises(_Deadline):
        import time as real_time
        saved = real_time.monotonic
        try:
            real_time.monotonic = lambda: 1e9      # far past the deadline
            hook(Agent(), None)
        finally:
            real_time.monotonic = saved


def test_the_completion_gate_is_off_by_default():
    """Two of the five failure buckets the benchmark's audit assigns --
    premature_termination and poor_self_verification -- are what the gate
    suppresses, and the reference harness has nothing like it. On by default it
    would produce numbers that do not belong beside a leaderboard score."""
    import inspect
    from swebench.harnesses import marathon
    source = inspect.getsource(marathon)
    assert 'config.get("completion_gate", False)' in source
    assert "premature_termination" in source, "the reason is recorded here"
