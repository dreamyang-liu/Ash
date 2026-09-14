"""Control-plane cutoffs must wake a native SDK stream and retain its identity."""
from types import SimpleNamespace
import threading

import pytest

from harness.core.control import RunControl
from harness.core.journal import JournalWriter
from harness.core.slot import TaskSpec
from harness.slots.codex_sdk import CodexSdkSlot


def test_control_interrupts_a_silent_sdk_stream(tmp_path, monkeypatch):
    sdk = pytest.importorskip("openai_codex")
    import openai_codex.client

    interrupted = threading.Event()
    closed = []
    control = RunControl()

    class Client:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def initialize(self):
            pass
        def thread_start(self, params):
            return {"thread": {"id": "native-session"}}
        def close(self):
            closed.append(True)

    class Thread:
        def __init__(self, **kwargs):
            pass
        def turn(self, *args, **kwargs):
            return SimpleNamespace(interrupt=interrupted.set)

    def collect(handle, journal):
        control.request_stop("deadline", stop_reason="timeout")
        assert interrupted.wait(2), "SDK stream did not receive the interrupt"
        return SimpleNamespace(status="interrupted", error=None, final_response="", usage=None), {}

    monkeypatch.setattr(sdk, "Thread", Thread)
    monkeypatch.setattr(openai_codex.client, "CodexClient", Client)
    monkeypatch.setattr("harness.slots.codex_sdk.normalize.collect_and_journal", collect)
    slot = CodexSdkSlot()
    monkeypatch.setattr(slot, "version", lambda: "fixture")
    monkeypatch.setattr(slot, "_child_env", lambda task: {})
    monkeypatch.setattr(slot, "_config_overrides", lambda mcp, extra: [])
    with JournalWriter(tmp_path / "journal", run_id="run") as journal:
        result = slot.run(TaskSpec(prompt="task", cwd=str(tmp_path), control=control), journal)
    assert interrupted.is_set() and closed
    assert result.native_session_id == "native-session"
    assert control.stop_reason == "timeout"
