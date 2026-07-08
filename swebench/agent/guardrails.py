"""Tool-level guardrails: nudge the agent away from common failure patterns.

Tracks which files have been read and how many times a file has been edited
without running tests, returning warning strings appended to tool results.
"""


class Guardrails:
    """Stateful guardrail checker, one per agent run."""

    def __init__(self):
        self._files_read: set[str] = set()          # files seen via text_editor view
        self._consecutive_edits: dict[str, int] = {} # file -> edits since last test

    def check(self, name: str, args: dict) -> str:
        """Inspect a tool call before execution. Returns a warning string or ''."""
        warnings = []
        path = args.get("path", "")

        # Track file reads
        if name == "text_editor" and args.get("command") == "view":
            self._files_read.add(path)

        # Read-before-edit: must have read a file before editing it
        if name == "text_editor" and args.get("command") in ("str_replace", "insert"):
            if path and path not in self._files_read:
                warnings.append(
                    f"[Warning] You are editing {path} without reading it first. "
                    f"Use text_editor(view) first to see the current content."
                )
            self._consecutive_edits[path] = self._consecutive_edits.get(path, 0) + 1
            if self._consecutive_edits[path] >= 3:
                warnings.append(
                    f"[Warning] This is edit #{self._consecutive_edits[path]} to {path} "
                    f"without running tests. Consider testing before making more changes."
                )

        # Reset edit counters when running tests
        if name == "shell":
            cmd = args.get("command", "")
            if "pytest" in cmd or "test_" in cmd or "assert" in cmd:
                self._consecutive_edits.clear()

        return "\n".join(warnings)
