"""What an agent has read and how long it has been editing.

Keyed by ``(agent_id, sandbox_id)``: two agents in one sandbox must not satisfy
each other's read-before-edit, and one agent across two sandboxes has not read a
path in the second just because it read it in the first.
"""

from __future__ import annotations

import threading

__all__ = ["GuardrailState"]


class GuardrailState:
    """Read/edit bookkeeping, keyed by ``(agent_id, sandbox_id)``.

    One instance per interceptor, shared across agents — hence the keying:
    files A read must never excuse B's blind edit. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files_read: dict[tuple[str, str], set[str]] = {}
        self._edit_streak: dict[tuple[str, str], dict[str, int]] = {}

    # -- reads --------------------------------------------------------------- #

    def record_read(self, agent_id: str, sandbox_id: str, path: str) -> None:
        with self._lock:
            self._files_read.setdefault((agent_id, sandbox_id), set()).add(path)

    def has_read(self, agent_id: str, sandbox_id: str, path: str) -> bool:
        with self._lock:
            return path in self._files_read.get((agent_id, sandbox_id), ())

    # -- edit streaks -------------------------------------------------------- #

    def record_edit(self, agent_id: str, sandbox_id: str, path: str) -> int:
        """Count this edit and return the streak length since the last test run."""
        with self._lock:
            streaks = self._edit_streak.setdefault((agent_id, sandbox_id), {})
            streaks[path] = streaks.get(path, 0) + 1
            return streaks[path]

    def reset_edits(self, agent_id: str, sandbox_id: str) -> None:
        with self._lock:
            self._edit_streak.pop((agent_id, sandbox_id), None)

    def dump(self) -> dict:
        """JSON-friendly snapshot, so an audit can read this interceptor's state.

        Keyed over reads *and* streaks: an agent that only ever edited blindly
        has no read entry, and it is exactly the behavior this audit exists to
        surface.
        """
        with self._lock:
            keys = set(self._files_read) | set(self._edit_streak)
            return {
                f"{agent}:{sbx}": {
                    "files_read": sorted(self._files_read.get((agent, sbx), ())),
                    "edit_streak": dict(self._edit_streak.get((agent, sbx), {})),
                }
                for agent, sbx in sorted(keys)
            }
