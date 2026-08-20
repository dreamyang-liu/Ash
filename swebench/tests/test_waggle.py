"""Unit tests for Waggle (swebench.agent.waggle).

Runs against an in-memory FakeSandbox — no Docker, no model calls.

Covered:
- read/write happy path (version bump + full-content history)
- require_read rejection and new-file creation
- stale write -> rejection with diff + reservation grant
- reservation wait: holder commits -> waiter wakes to a conflict
- reservation TTL expiry -> still-consistent waiter wins
- shell effect detection (out-of-band write bumps version)
- zero lost updates under two-agent write contention (the money test)
"""

from __future__ import annotations

import hashlib
import shlex
import threading
import time

from swebench.agent.waggle import CoordinatedExecutor, WorkspaceCoordinator
from swebench.models import ToolResult

PATH = "/testbed/target.txt"


class FakeSandbox:
    """In-memory stand-in for the ash runtime: a dict of files + minimal shell."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files = dict(files or {})
        self._lock = threading.Lock()

    def executor(self):
        return lambda tool, args: self._dispatch(tool, dict(args))

    def read(self, path: str) -> str | None:
        with self._lock:
            return self._files.get(path)

    def mutate(self, path: str, content: str) -> None:
        """Out-of-band write (simulates a shell side effect)."""
        with self._lock:
            self._files[path] = content

    # -- dispatch --------------------------------------------------------- #

    def _dispatch(self, tool: str, args: dict) -> ToolResult:
        if tool == "text_editor":
            return self._text_editor(args)
        if tool == "shell":
            return self._shell(args.get("command", ""))
        return ToolResult(success=True, output="")

    def _text_editor(self, args: dict) -> ToolResult:
        command, path = args["command"], args["path"]
        with self._lock:
            if command == "view":
                if path not in self._files:
                    return ToolResult(success=False, output="", error="not found")
                return ToolResult(success=True, output=self._files[path])
            if command == "write":
                self._files[path] = args["file_text"]
                return ToolResult(success=True, output="ok")
            if command == "str_replace":
                content = self._files.get(path, "")
                if args["old_str"] not in content:
                    return ToolResult(success=False, output="", error="no match")
                self._files[path] = content.replace(args["old_str"], args["new_str"], 1)
                return ToolResult(success=True, output="ok")
        return ToolResult(success=False, output="", error=f"unsupported: {command}")

    def _shell(self, command: str) -> ToolResult:
        tokens = shlex.split(command.replace("&&", " ").replace("||", " "))
        with self._lock:
            if tokens[:2] == ["cat", "--"]:
                content = self._files.get(tokens[2])
                if content is None:
                    return ToolResult(success=False, output="", error="not found")
                return ToolResult(success=True, output=content)
            if tokens[:2] == ["md5sum", "--"]:
                lines = [f"{_md5(self._files[p])}  {p}"
                         for p in tokens[2:] if p in self._files]
                return ToolResult(success=True, output="\n".join(lines))
            if tokens[:2] == ["test", "-f"]:
                exists = tokens[2] in self._files
                return ToolResult(success=True, output="EXISTS" if exists else "MISSING")
        return ToolResult(success=True, output="")


def _md5(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _setup(files: dict[str, str] | None = None, ttl: float = 5.0):
    sandbox = FakeSandbox(files)
    state = WorkspaceCoordinator(ttl=ttl)

    def make(agent: str) -> CoordinatedExecutor:
        return CoordinatedExecutor(sandbox.executor(), state, agent_id=agent)

    return sandbox, state, make


def _write(ex: CoordinatedExecutor, text: str, path: str = PATH) -> ToolResult:
    return ex("text_editor", {"command": "write", "path": path, "file_text": text})


def _read(ex: CoordinatedExecutor, path: str = PATH) -> ToolResult:
    return ex("text_editor", {"command": "view", "path": path})


# --------------------------------------------------------------------------- #
#  Basics
# --------------------------------------------------------------------------- #

def test_read_then_write_bumps_version_and_records_history():
    sandbox, state, make = _setup({PATH: "base"})
    a = make("A")

    assert _read(a).success
    assert _write(a, "v2 content").success

    rec = state.file("default", PATH)
    assert rec.version == 2                      # baseline v1 + write v2
    assert [r.op for r in rec.history] == ["baseline", "write"]
    assert rec.history[-1].content == "v2 content"     # full text, not a diff
    assert sandbox.read(PATH) == "v2 content"


def test_write_without_read_is_rejected():
    _, _, make = _setup({PATH: "base"})
    result = _write(make("A"), "blind overwrite")
    assert not result.success
    assert "text_editor view first" in result.output


def test_new_file_creation_is_allowed_without_read():
    sandbox, state, make = _setup()
    result = _write(make("A"), "hello", path="/testbed/new.txt")
    assert result.success
    assert sandbox.read("/testbed/new.txt") == "hello"
    assert state.file("default", "/testbed/new.txt").history[-1].op == "create"


def test_stale_write_rejected_with_diff_and_reservation():
    _, state, make = _setup({PATH: "line1\nline2"})
    a, b = make("A"), make("B")
    _read(a), _read(b)                           # both snapshot v1

    assert _write(a, "line1\nline2 changed by A").success
    rejection = _write(b, "line1 changed by B\nline2")

    assert not rejection.success
    assert "[WAGGLE]" in rejection.output
    assert "your snapshot : v1" in rejection.output
    assert "changed by: A" in rejection.output
    assert "+line2 changed by A" in rejection.output      # unified diff present
    reservation = state.file("default", PATH).reservation
    assert reservation and reservation.agent == "B"       # loser is protected

    _read(b)                                              # re-read latest
    assert _write(b, "merged by B").success               # protected retry wins


# --------------------------------------------------------------------------- #
#  Reservation waiting
# --------------------------------------------------------------------------- #

def test_waiter_wakes_to_conflict_after_holder_commits():
    _, state, make = _setup({PATH: "base"})
    a, b, c = make("A"), make("B"), make("C")
    _read(a), _read(b)
    assert _write(a, "by A").success             # v2
    assert not _write(b, "by B").success         # stale -> B holds reservation
    _read(c)                                     # C snapshots current v2

    outcome: dict = {}

    def blocked_write():
        outcome["result"] = _write(c, "by C")    # must wait on B's reservation

    thread = threading.Thread(target=blocked_write)
    thread.start()
    time.sleep(0.3)                              # let C park on the reservation
    assert "result" not in outcome               # still waiting

    _read(b)
    assert _write(b, "by B, take 2").success     # commit -> release -> wake C
    thread.join(timeout=3)

    result = outcome["result"]
    assert not result.success                    # woke to a truthful conflict
    assert "[WAGGLE]" in result.output
    reservation = state.file("default", PATH).reservation
    assert reservation and reservation.agent == "C"       # C protected next


def test_expired_reservation_lets_consistent_waiter_win():
    sandbox, _, make = _setup({PATH: "base"}, ttl=0.4)
    a, b, c = make("A"), make("B"), make("C")
    _read(a), _read(b)
    assert _write(a, "by A").success             # v2
    assert not _write(b, "by B").success         # B reserved... then goes silent
    _read(c)                                     # C snapshots current v2

    start = time.monotonic()
    result = _write(c, "by C")                   # waits ~ttl, then re-arbitrates
    elapsed = time.monotonic() - start

    assert result.success                        # version unchanged -> C wins
    assert elapsed < 3.0
    assert sandbox.read(PATH) == "by C"


# --------------------------------------------------------------------------- #
#  Shell effect detection
# --------------------------------------------------------------------------- #

def test_shell_effect_bumps_version_and_stales_other_agents():
    sandbox, state, make = _setup({PATH: "base"})
    a, b = make("A"), make("B")
    _read(a), _read(b)                           # both snapshot v1

    sandbox.mutate(PATH, "changed out of band")  # what the shell command "did"
    b("shell", {"command": "run-something"})     # B's shell triggers the scan

    rec = state.file("default", PATH)
    assert rec.version == 2
    assert rec.history[-1].op == "external"
    assert rec.history[-1].author == "external (detected by B)"
    assert rec.history[-1].content == "changed out of band"

    rejection = _write(a, "based on stale v1")   # A must now be stale
    assert not rejection.success
    assert "external (detected by B)" in rejection.output


def test_scan_double_check_prevents_phantom_version():
    """A bulk fingerprint taken just before a coordinated commit must not be
    misread as external drift: the under-lock re-check must catch the race."""
    sandbox, state, make = _setup({PATH: "base"})
    a = make("A")
    _read(a)
    assert _write(a, "v2").success               # rec.digest now tracks v2
    rec = state.file("default", PATH)
    assert rec.version == 2

    real_fetch = a._fetch_digests
    calls = {"count": 0}

    def stale_bulk_then_fresh(paths):
        calls["count"] += 1
        if calls["count"] == 1:                  # bulk photo: stale, pre-commit
            return {PATH: "0" * 32}
        return real_fetch(paths)                 # under-lock double check: fresh

    a._fetch_digests = stale_bulk_then_fresh
    a("shell", {"command": "true"})              # triggers the scan

    assert rec.version == 2                      # no phantom "external" version
    assert calls["count"] == 2                   # the double check actually ran


# --------------------------------------------------------------------------- #
#  The money test: zero lost updates under contention
# --------------------------------------------------------------------------- #

def test_no_lost_updates_under_two_agent_contention():
    rounds = 6
    sandbox, _, make = _setup({PATH: "A=0;B=0"})

    def parse(text: str) -> dict[str, int]:
        return {k: int(v) for k, v in (kv.split("=") for kv in text.split(";"))}

    def fmt(data: dict[str, int]) -> str:
        return ";".join(f"{k}={v}" for k, v in sorted(data.items()))

    def agent_loop(ex: CoordinatedExecutor, key: str):
        for _ in range(rounds):
            while True:                                   # LLM-style retry loop
                seen = _read(ex)
                data = parse(seen.output)
                data[key] += 1
                if _write(ex, fmt(data)).success:
                    break                                 # rejected -> re-read

    threads = [threading.Thread(target=agent_loop, args=(make(k), k))
               for k in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    final = parse(sandbox.read(PATH))
    assert final == {"A": rounds, "B": rounds}   # nothing overwritten silently


def test_dump_is_json_friendly_audit():
    _, state, make = _setup({PATH: "base"})
    a = make("A")
    _read(a)
    _write(a, "v2")

    dump = state.dump()
    (key, records), = dump.items()
    assert key == f"default:{PATH}"
    assert [r["version"] for r in records] == [1, 2]
    assert {"version", "author", "op", "timestamp", "bytes"} <= records[0].keys()


# --------------------------------------------------------------------------- #
#  Opaque writers: tools whose file effects Waggle cannot see directly
# --------------------------------------------------------------------------- #

def test_shell_is_an_opaque_writer_by_default():
    """Default behaviour is unchanged: shell is scanned, nothing else is."""
    from swebench.agent.waggle import WaggleInterceptor

    icpt = WaggleInterceptor()
    assert icpt.opaque_writers == {"shell"}
    assert icpt.tools == {"text_editor", "shell"}
    assert not icpt.applies_to("mytool")


def test_declaring_a_tool_an_opaque_writer_restores_drift_detection():
    """A tool whose dispatch happens below this seat arrives as one opaque call,
    so Waggle has no `shell` to key on. Undeclared, an agent can then write over
    a file version it never read: `str_replace` may coexist by luck, but a whole
    file `write` silently discards the other agent's work.

    Naming the tool is all it takes -- Waggle is a pluggable suite, so which
    calls it watches is its own configuration.
    """
    from swebench.agent.pipeline import CallContext, ToolPipeline
    from swebench.agent.waggle import WaggleInterceptor, WorkspaceCoordinator

    path = "/testbed/app.py"
    original = "def f():\n    return 1\n"
    after_tool = original + "\ndef g():\n    return 'other agent'\n"

    def run(declared: bool):
        sandbox = FakeSandbox({path: original})
        icpt = WaggleInterceptor(state=WorkspaceCoordinator(ttl=5.0),
                                 opaque_writers={"mycodegen"} if declared else None)
        pipe = ToolPipeline([icpt])

        def call(agent, tool, args):
            executor = sandbox.executor()
            return pipe.execute(
                CallContext(agent_id=agent, sandbox_id="default", tool_name=tool,
                            args=dict(args), metadata={"executor": executor}),
                executor)

        call("A", "text_editor", {"command": "view", "path": path})
        sandbox.mutate(path, after_tool)                 # the tool's effect
        call("B", "mycodegen", {"target": path})          # one opaque call
        result = call("A", "text_editor", {
            "command": "write", "path": path, "file_text": "def f():\n    return 2\n"})
        return result, sandbox.read(path)

    undeclared, content = run(declared=False)
    assert undeclared.success                            # nothing stopped it
    assert "other agent" not in content                  # ... and the work is gone

    declared, content = run(declared=True)
    assert not declared.success
    assert "[WAGGLE]" in declared.output                 # rejected with a diff
    assert "other agent" in content                      # the work survived


def test_the_lite_mounting_shares_the_opaque_writer_setting():
    """Both mountings must watch the same calls, or switching between them
    quietly changes what coordination covers."""
    from swebench.agent.waggle import CoordinatedExecutor, WorkspaceCoordinator

    sandbox = FakeSandbox({"/testbed/a.py": "v1"})
    ex = CoordinatedExecutor(sandbox.executor(), WorkspaceCoordinator(ttl=5.0),
                             agent_id="A", opaque_writers={"mytool"})
    assert ex._opaque_writers == {"shell", "mytool"}


def test_an_opaque_writer_reaches_the_runtime_under_its_own_name():
    """The executor below this seat expands a manifest tool by name, so renaming
    it to `shell` on the way through would break dispatch entirely."""
    from swebench.agent.waggle import CoordinatedExecutor, WorkspaceCoordinator

    seen = []
    sandbox = FakeSandbox({"/testbed/a.py": "v1"})
    inner = sandbox.executor()

    def spy(tool, args):
        seen.append(tool)
        return inner(tool, args)

    ex = CoordinatedExecutor(spy, WorkspaceCoordinator(ttl=5.0), agent_id="A",
                             opaque_writers={"mytool"})
    ex("mytool", {"target": "/testbed/a.py"})
    assert seen == ["mytool"]


def test_the_lite_mounting_scans_after_a_declared_opaque_writer():
    """Not just that the setting is stored -- that it drives the scan. Without
    this, an agent could write over a version it never read."""
    from swebench.agent.waggle import CoordinatedExecutor, WorkspaceCoordinator

    path = "/testbed/a.py"
    sandbox = FakeSandbox({path: "v1"})
    state = WorkspaceCoordinator(ttl=5.0)
    a = CoordinatedExecutor(sandbox.executor(), state, agent_id="A",
                            opaque_writers={"mytool"})
    b = CoordinatedExecutor(sandbox.executor(), state, agent_id="B",
                            opaque_writers={"mytool"})

    a("text_editor", {"command": "view", "path": path})     # A holds v1
    sandbox.mutate(path, "v1\nchanged by the tool\n")       # the tool's effect
    b("mytool", {"target": path})                           # opaque call -> scan

    result = a("text_editor", {"command": "write", "path": path,
                               "file_text": "rewritten by A\n"})
    assert not result.success                                # drift was noticed
    assert "changed by the tool" in sandbox.read(path)       # nothing was lost
