"""Ash CLI wrapper for executing commands in sessions.

Manages ash sessions and routes commands via ASH_SESSION env var:
    ASH_SESSION=<id> ash <subcommand> [args...]
"""

import json
import os
import subprocess
from typing import Optional

from . import style as S
from .models import ToolResult


def _validate_ash_only(command: str) -> Optional[str]:
    """Validate that every statement in the command invokes `ash`.

    Splits on unquoted &&, ||, ; and checks that the first word of each
    statement (before any pipes) is 'ash'.

    Returns an error message if invalid, None if OK.
    """
    # Split into statements respecting quotes
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    chars = command

    while i < len(chars):
        c = chars[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            # && or ||
            if c in ('&', '|') and i + 1 < len(chars) and chars[i + 1] == c:
                statements.append("".join(current))
                current = []
                i += 2
                continue
            # ;
            if c == ';':
                statements.append("".join(current))
                current = []
                i += 1
                continue
        current.append(c)
        i += 1

    statements.append("".join(current))

    for idx, stmt in enumerate(statements):
        stmt = stmt.strip()
        if not stmt:
            continue
        # Take the first command in a pipe chain: "ash grep ... | wc -l" → "ash grep ..."
        first_cmd = stmt.split("|")[0].strip()
        first_word = first_cmd.split()[0] if first_cmd.split() else ""
        # Allow `sleep` only as the very first statement
        if first_word == "sleep":
            if idx == 0:
                continue
            return (
                f"`sleep` is only allowed at the start of a command. Got: `{stmt}`\n"
                f"Move `sleep` to the beginning, e.g.: sleep 2 && ash ..."
            )
        if first_word != "ash":
            return (
                f"Only `ash` commands are allowed. Got: `{stmt}`\n"
                f"Run `ash --help` to see available subcommands."
            )

    return None


class AshSession:
    """Manages an ash session for SWE-bench evaluation.

    Creates a Docker-backed sandbox session via `ash session create`.
    Commands run with ASH_SESSION env var set, so callers just write
    `ash grep ...` / `ash run "..."` without passing --session.
    """

    def __init__(self, ash_binary: str = "ash", timeout: float = 300.0):
        self.ash_binary = ash_binary
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._env: dict[str, str] = {}

    def create(self, image: str) -> bool:
        """Create a new session with the given Docker image."""
        result = self._ash(["session", "create", "--image", image], timeout=180.0)
        if not result.success:
            print(f"  {S.bright_red('!')} Failed to create session: {result.error}")
            return False

        try:
            data = json.loads(result.output.strip())
            self.session_id = data.get("session_id")
        except (json.JSONDecodeError, TypeError):
            output = result.output.strip()
            if output:
                self.session_id = output.split()[-1]

        if not self.session_id:
            print(f"  {S.bright_red('!')} Could not parse session ID from: {result.output}")
            return False

        # Build env once — all subsequent commands inherit ASH_SESSION + ASH_AGENT
        self._env = {**os.environ, "ASH_SESSION": self.session_id, "ASH_AGENT": "1"}
        print(S.kv("session ", S.cyan(self.session_id)))
        return True

    def destroy(self):
        """Destroy the session."""
        if self.session_id:
            self._ash(["session", "destroy", self.session_id], timeout=30.0)
            print(S.kv("cleanup ", S.dim(f"destroyed {self.session_id}")))
            self.session_id = None
            self._env = {}

    def execute(self, command: str, timeout: float = 300.0) -> ToolResult:
        """Execute a shell command with ASH_SESSION set.

        The command is run as-is via shell. Callers write ash CLI commands
        directly: `ash grep "pattern" src/`, `ash run "pytest" --tail 30`, etc.

        Commands that don't start with `ash` are rejected with a hint.
        """
        if not self.session_id:
            return ToolResult(success=False, output="", error="No active session")

        # Intercept: only ash commands allowed
        err = _validate_ash_only(command)
        if err:
            return ToolResult(success=False, output="", error=err)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env,
            )
            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                error = result.stderr.strip() or f"Exit code {result.returncode}"
                return ToolResult(success=False, output=result.stdout, error=error)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def get_patch(self) -> str:
        """Get the git diff (patch) from the session."""
        result = self.execute(f'{self.ash_binary} run "git diff"')
        if result.success and result.output.strip():
            return result.output.strip()

        # Also check untracked files
        self.execute(f'{self.ash_binary} run "git add -N ."')
        result = self.execute(f'{self.ash_binary} run "git diff"')
        return result.output.strip() if result.success else ""

    # --- Helpers ---

    def _ash(self, args: list[str], timeout: Optional[float] = None) -> ToolResult:
        """Execute an ash CLI command (without session env — for session management)."""
        cmd = [self.ash_binary] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                error = result.stderr.strip() or f"Exit code {result.returncode}"
                return ToolResult(success=False, output=result.stdout, error=error)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output="",
                error=f"Timed out after {timeout or self.timeout}s",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, output="",
                error=f"ash binary not found: {self.ash_binary}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
