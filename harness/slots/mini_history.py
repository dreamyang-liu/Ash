"""Durable mini messages and closed model turns; no upstream dependency."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path


def read_entries(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.splitlines(keepends=True)
            if line.endswith(b"\n") and line.strip()]


def training_messages(entries: list[dict]) -> list[dict]:
    messages = []
    for entry in entries:
        if entry.get("type") != "mini.message":
            continue
        native = entry["message"]
        if native.get("role") == "exit":
            continue
        message = {key: deepcopy(native[key]) for key in (
            "role", "content", "tool_calls", "tool_call_id", "reasoning_content")
            if native.get(key) is not None}
        message.setdefault("content", "")
        if not isinstance(message["content"], str):
            raise ValueError("mini training requires text content")
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {"id": call["id"], "type": call["type"],
                 "function": {"name": call["function"]["name"],
                              "arguments": call["function"]["arguments"]}}
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


def load_prefix(reference: dict) -> list[dict]:
    path = Path(reference["path"])
    with path.open("rb") as stream:
        data = stream.read(reference.get("byte_length", -1))
    if (not data.endswith(b"\n") or hashlib.sha256(data).hexdigest() != reference["sha256"]):
        raise ValueError("mini native prefix checksum changed")
    entries = read_entries(data)
    if not entries or entries[-1].get("type") != "mini.turn":
        raise ValueError("mini restoration requires a closed model turn")
    metadata = next((e for e in entries if e.get("type") == "mini.session"), {})
    if metadata.get("format") != "ash-mini-v1" or metadata.get("mini_version") != "2.4.6":
        raise ValueError("Unsupported mini native history version")
    return entries


class History:
    def __init__(self, path: Path | str, session_id: str, *, prefix=(), workspace=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        metadata = {"type": "mini.session", "format": "ash-mini-v1",
                    "mini_version": "2.4.6", "session_id": session_id}
        if workspace is not None:
            metadata["workspace"] = deepcopy(workspace)
        self.append(metadata)
        self.prefix_messages = []
        for entry in prefix:
            if entry["type"] != "mini.session":
                self.append(entry)
            if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
                self.prefix_messages.append(deepcopy(entry["message"]))

    def append(self, entry: dict) -> None:
        self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()
