"""Sandbox observers: the pluggable seam for "what is this run's answer".

The execution plane knows how to run tools in a sandbox. It must not know what
counts as a submission -- for SWE-bench that is a git diff, for SWE-Marathon it
is the verifier's score, for a future benchmark it could be a generated file or
a sequence of API calls. Those all need the *same* hook points, so the server
takes observers instead of hard-coding any of them:

- :meth:`SandboxObserver.on_created` -- baseline the sandbox before an agent
  touches it (SWE-bench must know which untracked paths the image shipped, or a
  ``build/`` tree it created is indistinguishable later from the agent's work).
- :meth:`SandboxObserver.after_mutating_call` -- refresh the artefact after every
  call that could have changed the sandbox, so it is always current. Extraction
  only at shutdown races the reader under load, and a killed run leaves nothing.
- :meth:`SandboxObserver.on_destroy` -- last chance, before the sandbox is gone.

Observers are advisory: a failure is logged and swallowed, never turned into a
tool error, because the agent cannot act on it and an escaped exception would
kill the run. They also must not mutate the ToolResult -- shaping what the model
sees is the interceptor pipeline's job, on the other side of the seam.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class SandboxObserver(Protocol):
    """Reacts to sandbox lifecycle and mutating tool calls."""

    name: str

    async def on_created(self, entry: Any) -> None:
        """A sandbox was created and is ready; capture any baseline."""

    async def after_mutating_call(self, entry: Any, tool_name: str, args: dict) -> None:
        """A call that may have changed the sandbox has completed."""

    async def on_destroy(self, entry: Any) -> None:
        """The sandbox is about to be destroyed."""


class ObserverSet:
    """Fan-out with failure isolation.

    Holds zero or more observers; every call is best-effort per observer, so one
    misbehaving extractor cannot stop the others or break the run.
    """

    def __init__(self, observers: Optional[Iterable[SandboxObserver]] = None) -> None:
        self.observers: List[SandboxObserver] = list(observers or [])

    def __bool__(self) -> bool:
        return bool(self.observers)

    def add(self, observer: SandboxObserver) -> None:
        self.observers.append(observer)

    def names(self) -> List[str]:
        return [getattr(o, "name", type(o).__name__) for o in self.observers]

    async def on_created(self, entry: Any) -> None:
        await self._fan("on_created", entry)

    async def after_mutating_call(self, entry: Any, tool_name: str, args: dict) -> None:
        await self._fan("after_mutating_call", entry, tool_name, args)

    async def on_destroy(self, entry: Any) -> None:
        await self._fan("on_destroy", entry)

    async def _fan(self, hook: str, *args) -> None:
        for observer in self.observers:
            method = getattr(observer, hook, None)
            if method is None:
                continue
            try:
                await method(*args)
            except Exception as exc:  # noqa: BLE001 - advisory by contract
                sys.stderr.write(
                    "[ash-exec] observer %s.%s failed: %s\n"
                    % (getattr(observer, "name", type(observer).__name__), hook, exc)
                )
                sys.stderr.flush()


def load_observer(spec: str) -> SandboxObserver:
    """Load ``module:factory`` and call it with no arguments.

    Mirrors how ``--plugins`` loads interceptors: the benchmark side supplies its
    own extractor without the execution plane importing it.
    """
    import importlib

    if ":" not in spec:
        raise ValueError("observer spec must be 'module:factory', got %r" % spec)
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    return factory()
