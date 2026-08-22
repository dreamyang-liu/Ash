"""An in-memory stand-in for the ash runtime, shared by the L2 tests.

Lets an interceptor chain be exercised with no sandbox, no container and no model:
a dict of files, plus enough of `text_editor` and `shell` to write and read back.
It lived in the Waggle tests, which was the wrong home -- it is a fake runtime,
not a fake coordinator, and the pipeline and interceptor tests were reaching
across into that module to get it.
"""

from __future__ import annotations

import hashlib
import shlex
import threading

from swebench.models import ToolResult


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
