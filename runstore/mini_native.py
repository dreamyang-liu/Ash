"""mini response boundaries indexed against observed calls and exact snapshots."""

import hashlib
import json
from pathlib import Path

from harness.normalize.claude_turns import completed_turn_steps
from harness.rollback import branch_checkpoints
from harness.slots.mini_history import read_entries


def index_mini(journal: Path, transcript: Path, session_id: str, *,
               events: list[dict], inherited_native: dict | None = None) -> list:
    from runstore.native import NativePoint, read_prefix

    local_calls = [e["call_id"] for e in events if e.get("type") == "tool.started"]
    positions = {call: i for i, call in enumerate(local_calls, 1)}
    admitted = set(local_calls)
    if inherited_native:
        if inherited_native["slot"] != "mini-swe-agent":
            raise ValueError("Inherited native prefix uses a different slot")
        for entry in read_entries(read_prefix(inherited_native)):
            admitted.update(call["id"] for call in entry.get("message", {}).get("tool_calls") or [])
    checkpoints = branch_checkpoints(journal, events=events)
    turns = completed_turn_steps(events) or {}
    closed = {value["turn_id"]: value for value in turns.values()}
    points, pending, seen, group = [], set(), set(), []
    checksum, offset, message_step = hashlib.sha256(), 0, 0
    for line in Path(transcript).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row = json.loads(line)
        checksum.update(line)
        offset += len(line)
        kind = row.get("type")
        if kind == "mini.session":
            if row.get("format") != "ash-mini-v1" or row.get("mini_version") != "2.4.6":
                raise ValueError("Unsupported mini native history version")
        elif kind == "mini.message":
            message = row["message"]
            calls = message.get("tool_calls") or []
            if calls:
                if pending or group:
                    break
                group = [call["id"] for call in calls]
                if len(set(group)) != len(group) or not set(group) <= admitted or set(group) & seen:
                    break
                pending.update(group)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    break
                pending.remove(call_id)
                seen.add(call_id)
        elif kind == "mini.turn":
            if pending or row.get("call_ids") != group:
                break
            message_step += 1
            local = [call for call in group if call in positions]
            if local:
                depth = max(positions[call] for call in local)
                turn = closed.get(row.get("turn_id"), {})
                point = checkpoints.get(depth)
                if (turn.get("call_ids") != group or turn.get("step") != depth
                        or not set(local_calls[:depth]) <= seen or set(local_calls[depth:]) & seen):
                    break
                if (point and point.pairing == "call-id-v1" and point.prefix_complete is True
                        and point.session_ckpt == session_id):
                    points.append(NativePoint(message_step, depth, point.snapshot_id, {
                        "slot": "mini-swe-agent", "session_id": session_id,
                        "path": str(Path(transcript).resolve()), "byte_length": offset,
                        "sha256": checksum.hexdigest(), "cut": row["turn_id"],
                    }))
            group = []
        else:
            raise ValueError("Unknown mini native record")
    return points


def reference_at(journal: Path, step: int, session_id: str) -> dict | None:
    from runstore.files import journal_events

    events = journal_events(Path(journal))
    reference = next((e for e in reversed(events) if e.get("type") == "session.ref"
                      and e.get("native_session_id") == session_id and e.get("transcript_path")), None)
    if reference is None:
        return None
    inherited = next((e.get("inherited_native") for e in events
                      if e.get("type") == "mini.restored"), None)
    points = index_mini(journal, reference["transcript_path"], session_id,
                        events=events, inherited_native=inherited)
    return next((point.native for point in points if point.tool_depth == step), None)
