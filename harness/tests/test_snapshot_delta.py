from types import SimpleNamespace

import pytest

from ash_sandbox.pool import Snapshot
from harness.checkpointing import SnapshotBridge
from harness.core.journal import JournalWriter, read_journal
from harness.execution.checkpoints import Checkpointer, MutationTracker
from harness.rollback import load_checkpoints


@pytest.mark.parametrize("flag", [True, False, None])
def test_capture_tag_survives_ledger_and_late_session_reference(tmp_path, flag):
    path = tmp_path / "journal.jsonl"
    tracker = MutationTracker()
    session = SimpleNamespace(supports_snapshot=lambda: True,
                              snapshot=lambda **kwargs: Snapshot("s", delta_empty=flag))
    journal = JournalWriter(path)
    bridge = SnapshotBridge.install(journal, session, tracker=tracker)
    bridge.on_tool_boundary(1)
    journal.emit("session.ref", native_session_id="session")
    journal.close()
    assert all(c.delta_empty is flag for c in load_checkpoints(path))
    assert any(e.get("delta_empty") is flag for e in read_journal(path)
               if e.get("type") == "checkpoint.captured")


def test_skipped_and_failed_capture_do_not_inherit_previous_measurement():
    tracker = MutationTracker()
    session = SimpleNamespace(supports_snapshot=lambda: True,
                              snapshot=lambda **kwargs: Snapshot("s", delta_empty=True))
    checkpointer = Checkpointer(session, tracker=tracker)
    assert checkpointer.after_step(1).delta_empty is True
    clean = checkpointer.after_step(2)
    assert clean.reason == "clean" and clean.delta_empty is None
    tracker.before(SimpleNamespace(tool_name="shell", args={"command": "write"}))
    session.snapshot = lambda **kwargs: None
    failed = checkpointer.after_step(3)
    assert failed.reason == "failed" and failed.delta_empty is None
