"""Sandbox observers: the seam that keeps "what is the answer" out of the server.

Covers the fan-out contract (failure isolation, ordering) and SWE-bench's
``PatchObserver`` as its first implementation, using a scripted fake sandbox so
no container is needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

import pytest

from harness.execution.observers import ObserverSet, load_observer


# --- fakes -----------------------------------------------------------------
@dataclass
class FakeResult:
    output: str = ""
    is_error: bool = False


class ScriptedSandbox:
    """Answers shell calls from a script keyed by substring."""

    def __init__(self, script):
        self.script = script
        self.calls: List[str] = []

    async def call(self, tool, **kwargs):
        command = kwargs.get("command", "")
        self.calls.append(command)
        for needle, result in self.script:
            if needle in command:
                return result
        return FakeResult(output="")


@dataclass
class FakeEntry:
    id: str = "sb-1"
    base_commit: str = "abc123"
    sandbox: object = None
    meta: dict = field(default_factory=dict)


@dataclass
class Recorder:
    name: str = "recorder"
    events: List[str] = field(default_factory=list)

    async def on_created(self, entry):
        self.events.append("created:%s" % entry.id)

    async def after_mutating_call(self, entry, tool_name, args):
        self.events.append("mutated:%s:%s" % (entry.id, tool_name))

    async def on_destroy(self, entry):
        self.events.append("destroyed:%s" % entry.id)


class Exploder:
    name = "exploder"

    async def on_created(self, entry):
        raise RuntimeError("observer is broken")

    async def after_mutating_call(self, entry, tool_name, args):
        raise RuntimeError("observer is broken")

    async def on_destroy(self, entry):
        raise RuntimeError("observer is broken")


# --- ObserverSet -----------------------------------------------------------
def test_hooks_fan_out_in_order():
    a, b = Recorder(name="a"), Recorder(name="b")
    observers = ObserverSet([a, b])
    entry = FakeEntry()

    asyncio.run(observers.on_created(entry))
    asyncio.run(observers.after_mutating_call(entry, "shell", {}))
    asyncio.run(observers.on_destroy(entry))

    assert a.events == ["created:sb-1", "mutated:sb-1:shell", "destroyed:sb-1"]
    assert b.events == a.events
    assert observers.names() == ["a", "b"]


def test_a_broken_observer_cannot_stop_the_others_or_the_run(capsys):
    """Advisory by contract: the agent cannot act on an extractor failure, and an
    escaped exception would kill the run."""
    good = Recorder()
    observers = ObserverSet([Exploder(), good])
    entry = FakeEntry()

    asyncio.run(observers.on_created(entry))          # must not raise
    assert good.events == ["created:sb-1"]
    assert "exploder" in capsys.readouterr().err


def test_empty_set_is_falsy_and_a_no_op():
    observers = ObserverSet()
    assert not observers
    asyncio.run(observers.on_created(FakeEntry()))    # no hooks, no error


def test_observers_without_a_hook_are_skipped():
    class Partial:
        name = "partial"

        async def on_destroy(self, entry):
            self.seen = True

    partial = Partial()
    observers = ObserverSet([partial])
    asyncio.run(observers.on_created(FakeEntry()))    # method absent -> skipped
    asyncio.run(observers.on_destroy(FakeEntry()))
    assert partial.seen is True


def test_load_observer_resolves_module_and_factory():
    observer = load_observer("swebench.patch:patch_observer")
    assert observer.name == "swebench-patch"


def test_load_observer_rejects_a_malformed_spec():
    with pytest.raises(ValueError):
        load_observer("swebench.patch.patch_observer")


# --- PatchObserver ---------------------------------------------------------
def _observer(tmp_path):
    from swebench.patch import PatchObserver

    return PatchObserver(tmp_path)


def test_baseline_is_taken_before_the_agent_runs(tmp_path):
    """Untracked paths the image shipped must be recorded first, or a build/ tree
    is indistinguishable later from the agent's work."""
    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output="build/lib.so\ndocs/cache\n")),
        ("diff", FakeResult(output="")),
    ])
    entry = FakeEntry(sandbox=sandbox)
    asyncio.run(_observer(tmp_path).on_created(entry))

    assert entry.meta["baseline_untracked"] == {"build/lib.so", "docs/cache"}
    # And an empty diff exists immediately, so a reader never finds nothing.
    assert (tmp_path / "sb-1.diff").read_text() == ""


def test_mutating_calls_refresh_the_diff(tmp_path):
    """Extraction at shutdown races the reader and loses a killed run entirely."""
    diff = "diff --git a/f.py b/f.py\n@@ -1 +1 @@\n-a\n+b"
    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output="")),
        ("diff", FakeResult(output=diff)),
    ])
    entry = FakeEntry(sandbox=sandbox)
    observer = _observer(tmp_path)

    asyncio.run(observer.on_created(entry))
    asyncio.run(observer.after_mutating_call(entry, "text_editor", {}))

    written = (tmp_path / "sb-1.diff").read_text()
    assert written.startswith("diff --git a/f.py")
    assert observer.patches["sb-1"] == written


def test_newly_added_files_are_staged_before_diffing(tmp_path):
    """A file the agent created is part of the answer, but `git add -A` would also
    stage everything the image shipped -- hence the baseline."""
    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output="build/lib.so\nnew_feature.py\n")),
        ("diff", FakeResult(output="diff --git a/new_feature.py b/new_feature.py")),
    ])
    entry = FakeEntry(sandbox=sandbox, meta={"baseline_untracked": {"build/lib.so"}})
    asyncio.run(_observer(tmp_path).after_mutating_call(entry, "shell", {}))

    staged = [c for c in sandbox.calls if "add" in c]
    assert any("new_feature.py" in c for c in staged)
    assert not any("build/lib.so" in c for c in staged), "staged an image-shipped path"


def test_a_sandbox_without_a_repository_yields_an_empty_patch(tmp_path):
    """Regression: the shell's error payload used to be written into the .diff and
    read back as if it were a prediction."""
    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output='{"stderr":"cd: can\'t cd to /testbed"}', is_error=True)),
    ])
    entry = FakeEntry(sandbox=sandbox)
    asyncio.run(_observer(tmp_path).on_created(entry))

    assert (tmp_path / "sb-1.diff").read_text() == ""


def test_diff_failure_also_yields_an_empty_patch(tmp_path):
    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output="")),
        ("diff", FakeResult(output="fatal: bad revision", is_error=True)),
    ])
    entry = FakeEntry(sandbox=sandbox)
    asyncio.run(_observer(tmp_path).after_mutating_call(entry, "shell", {}))

    assert (tmp_path / "sb-1.diff").read_text() == ""


def test_patch_observer_works_without_a_directory(tmp_path):
    """In-memory only: the harness reads `.patches` instead of a file."""
    from swebench.patch import PatchObserver

    sandbox = ScriptedSandbox([
        ("--others", FakeResult(output="")),
        ("diff", FakeResult(output="diff --git a/x b/x")),
    ])
    entry = FakeEntry(sandbox=sandbox)
    observer = PatchObserver(None)
    asyncio.run(observer.after_mutating_call(entry, "shell", {}))

    assert observer.patches["sb-1"].startswith("diff --git")
    assert not list(tmp_path.iterdir())
