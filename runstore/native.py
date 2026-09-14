"""Immutable views into native history, always cut at a closed response group."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from harness.normalize.claude_turns import completed_turn_steps
from harness.rollback import branch_checkpoints


@dataclass(frozen=True)
class NativePoint:
    message_step: int
    tool_depth: int
    snapshot_id: str
    native: dict


def read_prefix(reference: dict) -> bytes:
    length = reference["byte_length"]
    if type(length) is not int or length <= 0:
        raise ValueError("Invalid native prefix length")
    with Path(reference["path"]).open("rb") as stream:
        data = stream.read(length)
    if len(data) != length or not data.endswith(b"\n"):
        raise ValueError("Native prefix missing or truncated")
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Native prefix checksum changed")
    for item in reference.get("referenced_outputs", []):
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Persisted native output changed")
    return data


def valid_prefix(reference: dict) -> bool:
    try:
        read_prefix(reference)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _inherited_call_ids(reference: dict | None, slot: str) -> set[str]:
    if reference is None:
        return set()
    if reference.get("slot") != slot:
        raise ValueError("Inherited native prefix uses a different slot")
    calls = set()
    for line in read_prefix(reference).splitlines():
        entry = json.loads(line)
        if slot == "codex":
            payload = entry.get("payload", {})
            if entry.get("type") == "response_item" and payload.get("type") in {"function_call", "custom_tool_call"}:
                calls.add(payload["call_id"])
        else:
            content = entry.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.add(block["id"])
    return calls


def index_native(journal: Path, transcript: Path, slot: str, session_id: str,
                 allowed_outputs=(), *, events: list[dict] | None = None,
                 inherited_native: dict | None = None) -> list[NativePoint]:
    if events is None:
        events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    else:
        events = list(events)
    calls = [event["call_id"] for event in events if event.get("type") == "tool.started"]
    positions = {call_id: depth for depth, call_id in enumerate(calls, 1)}
    checkpoints = branch_checkpoints(journal, events=events)
    admitted_calls = set(positions) | _inherited_call_ids(inherited_native, slot)
    checkpoints = {depth: point for depth, point in checkpoints.items()
                   if point.pairing == "call-id-v1" and point.prefix_complete is True}
    turns = completed_turn_steps(events) if slot == "claude-code" else None
    if slot == "claude-code" and turns is None:
        return []
    completed_groups = {tuple(value["call_ids"]): depth for depth, value in (turns or {}).items()}
    pending: set[str] = set()
    seen: set[str] = set()
    group: list[str] = []
    points = []
    offset = 0
    checksum = hashlib.sha256()
    message_step = 0
    native_entries = []
    for line in transcript.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        entry = json.loads(line)
        native_entries.append(entry)
        offset += len(line)
        checksum.update(line)
        close = False
        payload = entry.get("payload", {}) if slot == "codex" else entry.get("message", {})
        if slot == "codex":
            kind = entry.get("type")
            subtype = payload.get("type")
            if kind == "compacted" or (kind == "event_msg" and subtype == "thread_rolled_back"):
                break
            if kind == "response_item" and subtype in {"function_call", "custom_tool_call"}:
                call_id = payload["call_id"]
                if call_id not in admitted_calls or call_id in pending or call_id in seen:
                    break
                pending.add(call_id)
                group.append(call_id)
            elif kind == "response_item" and subtype in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload["call_id"]
                if call_id not in pending:
                    break
                pending.remove(call_id)
                seen.add(call_id)
            elif kind == "event_msg" and subtype == "token_count":
                close = True
        elif slot == "claude-code":
            if entry.get("subtype") == "compact_boundary":
                break
            content = payload.get("content", [])
            unobserved_call = False
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    if block["id"] not in admitted_calls:
                        unobserved_call = True
                        break
                    pending.add(block["id"])
                elif block.get("type") == "tool_result":
                    pending.discard(block["tool_use_id"])
                    seen.add(block["tool_use_id"])
            if unobserved_call:
                break
            if entry.get("type") == "user" and not pending:
                candidates = [(ids, depth) for ids, depth in completed_groups.items()
                              if set(ids) <= seen and depth > (points[-1].tool_depth if points else 0)]
                if candidates:
                    group = list(max(candidates, key=lambda candidate: candidate[1])[0])
                    close = True
        else:
            raise ValueError("Unsupported native history slot")
        if not close:
            continue
        message_step += 1
        local = [call_id for call_id in group if call_id in positions]
        if local and not pending:
            depth = max(positions[call_id] for call_id in local)
            if (set(calls[:depth]) <= seen and not set(calls[depth:]) & seen
                    and depth in checkpoints):
                reference = {"slot": slot, "session_id": session_id,
                             "path": str(transcript.resolve()), "byte_length": offset,
                             "sha256": checksum.hexdigest(), "cut": entry.get("uuid")}
                if slot == "claude-code":
                    from harness.slots.claude_history import _referenced_outputs

                    reference["referenced_outputs"] = _referenced_outputs(native_entries, transcript, allowed_outputs)
                points.append(NativePoint(message_step, depth, checkpoints[depth].snapshot_id, reference))
        group = []
    return points


def materialize(reference: dict, cwd: Path, destination: Path, native_home: Path) -> dict:
    data = read_prefix(reference)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prefix.jsonl"
    path.write_bytes(data)
    if reference["slot"] == "codex":
        return {"resume_session_id": reference["session_id"], "fork": True,
                "native_prefix": {"path": str(path.resolve()), "sha256": reference["sha256"]}}
    from harness.slots.claude_history import PrefixSource, prepare_prefix

    original = Path(reference["path"])
    full = original.read_bytes()
    if full[:reference["byte_length"]] != data:
        raise ValueError("Native source changed during restoration")
    source = PrefixSource(original, hashlib.sha256(full).hexdigest(), reference["session_id"], reference["cut"],
                          tuple((item["path"], item["sha256"])
                                for item in reference.get("referenced_outputs", [])))
    receipt = prepare_prefix(source, cwd, destination / "registration", native_home / "projects")
    return {"resume_session_id": receipt["resume_session_id"], "fork": False}
