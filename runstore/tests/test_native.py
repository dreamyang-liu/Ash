import hashlib
import json

from harness.core.journal import JournalWriter
from runstore.native import index_native, read_prefix


def test_codex_group_includes_all_tools_and_never_cuts_an_incomplete_response(tmp_path):
    journal = tmp_path / "trajectory.jsonl"
    with JournalWriter(journal) as writer:
        writer.emit("checkpoint.policy", pairing="call-id-v1")
        for depth in range(1, 4):
            writer.emit("tool.started", call_id=f"call-{depth}", name="ash__shell", args={})
            writer.emit("tool.finished", call_id=f"call-{depth}", output="response")
            writer.emit("checkpoint.captured", step=depth, call_id=f"call-{depth}",
                        pairing="call-id-v1", prefix_complete=True, snapshot_id=f"snap-{depth}",
                        session_ckpt="session", reason="captured")
    def entry(kind, **payload):
        return {"type": kind, "payload": payload}
    entries = [entry("session_meta", id="session"),
               entry("response_item", type="function_call", call_id="call-1"),
               entry("response_item", type="function_call", call_id="call-2"),
               entry("response_item", type="function_call_output", call_id="call-1"),
               entry("response_item", type="function_call_output", call_id="call-2"),
               entry("event_msg", type="token_count"),
               entry("response_item", type="function_call", call_id="call-3")]
    transcript = tmp_path / "native.jsonl"
    transcript.write_text("".join(json.dumps(item) + "\n" for item in entries))
    points = index_native(journal, transcript, "codex", "session")
    assert [(point.message_step, point.tool_depth, point.snapshot_id) for point in points] == [(1, 2, "snap-2")]
    reference = points[0].native
    prefix = read_prefix(reference)
    assert b"call-3" not in prefix
    with transcript.open("a") as stream:
        stream.write(json.dumps(entry("response_item", type="function_call_output", call_id="call-3")) + "\n")
        stream.write(json.dumps(entry("event_msg", type="token_count")) + "\n")
    assert read_prefix(reference) == prefix
    assert len(index_native(journal, transcript, "codex", "session")) == 2
