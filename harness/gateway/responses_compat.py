"""Opt-in Responses namespace adaptation for flat-function-only upstreams."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


class ResponsesNamespaceAdapter:
    def __init__(self, payload: dict):
        self._names: dict[tuple[str, str], str] = {}
        self._original: dict[str, tuple[str, str]] = {}
        self._buffer = b""
        self.payload = deepcopy(payload)
        tools = self.payload.get("tools", [])
        if tools is None:
            tools = []
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("Responses tools must be a list of objects")
        if any("name" in tool and not isinstance(tool["name"], str) for tool in tools):
            raise ValueError("Tool names must be strings")
        occupied = {tool.get("name") for tool in tools if tool.get("type") != "namespace"}
        flattened = []
        for tool in tools:
            if tool.get("type") != "namespace":
                flattened.append(tool)
                continue
            namespace = tool.get("name")
            if not isinstance(namespace, str) or not namespace:
                raise ValueError("Namespace tool requires a nonempty name")
            nested_tools = tool.get("tools", [])
            if not isinstance(nested_tools, list) or any(not isinstance(tool, dict) for tool in nested_tools):
                raise ValueError("Namespace tools must be a list of objects")
            for nested in nested_tools:
                name = nested.get("name")
                if nested.get("type") != "function" or not isinstance(name, str) or not name:
                    raise ValueError("Only named function tools can be flattened from a namespace")
                qualified = f"{namespace}__{name}"
                if qualified in occupied:
                    raise ValueError(f"Flattened tool name collision: {qualified}")
                occupied.add(qualified)
                self._names[namespace, name] = qualified
                self._original[qualified] = namespace, name
                flattened.append({**nested, "name": qualified})
        if "tools" in self.payload:
            self.payload["tools"] = flattened
        for key in ("input", "tool_choice"):
            if key in self.payload:
                self.payload[key] = self._flatten_references(self.payload[key])

    def _flatten_references(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._flatten_references(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: self._flatten_references(item) for key, item in value.items()}
        namespace = result.get("namespace")
        if result.get("type") in ("function_call", "function") and namespace:
            if not isinstance(namespace, str) or not isinstance(result.get("name"), str):
                raise ValueError("Namespaced call requires string namespace and name")
            identity = namespace, result.get("name")
            if identity not in self._names:
                raise ValueError("Namespaced call has no matching tool definition")
            result["name"] = self._names[identity]
            result.pop("namespace")
        return result

    def restore(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.restore(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: self.restore(item) for key, item in value.items()}
        kind = result.get("type")
        if kind in ("function_call", "response.function_call_arguments.done") and isinstance(result.get("name"), str):
            identity = self._original.get(result.get("name"))
            if identity:
                result["namespace"], result["name"] = identity
        return result

    def _rewrite_event(self, frame: bytes) -> bytes:
        lines = frame.splitlines()
        data = [line[5:].lstrip() for line in lines if line.startswith(b"data:")]
        if not data:
            return frame
        try:
            payload = json.loads(b"\n".join(data))
        except (ValueError, UnicodeError):
            return frame
        restored = self.restore(payload)
        if restored == payload:
            return frame
        encoded = b"data: " + json.dumps(restored, ensure_ascii=False, separators=(",", ":")).encode()
        rebuilt = []
        inserted = False
        for line in lines:
            if line.startswith(b"data:"):
                if not inserted:
                    rebuilt.append(encoded)
                    inserted = True
            else:
                rebuilt.append(line)
        separator = b"\r\n" if b"\r\n" in frame else b"\n"
        return separator.join(rebuilt)

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer += chunk
        frames = []
        while match := re.search(br"\r?\n\r?\n", self._buffer):
            frames.append(self._rewrite_event(self._buffer[:match.start()]) + match.group())
            self._buffer = self._buffer[match.end():]
        return frames

    def finish(self) -> bytes:
        remaining = self._rewrite_event(self._buffer) if self._buffer else b""
        self._buffer = b""
        return remaining
