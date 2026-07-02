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
        if tool == "read_file":
            return self._read_file(args["path"])
        if tool == "text_editor":
            return self._text_editor(args)
        if tool == "shell":
            return self._shell(args.get("command", ""))
        return ToolResult(success=True, output="")

    def _read_file(self, path: str) -> ToolResult:
        with self._lock:
            if path not in self._files:
                return ToolResult(success=False, output="", error="not found")
            return ToolResult(success=True, output=self._files[path])

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
    return ex("read_file", {"path": path})


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
    assert "read_file first" in result.output


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
    assert rec.history[-1].op == "shell"
    assert rec.history[-1].author == "shell(B)"
    assert rec.history[-1].content == "changed out of band"

    rejection = _write(a, "based on stale v1")   # A must now be stale
    assert not rejection.success
    assert "shell(B)" in rejection.output


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
