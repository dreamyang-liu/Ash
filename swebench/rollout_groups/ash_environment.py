"""EnvironmentProvider backed by AshSession and the real Ash sandbox SDK."""

from __future__ import annotations

import uuid
from typing import Any

from ..sandbox import AshSession


class AshSessionEnvironmentProvider:
    """Create one independent AshSession for each rollout sample.

    ``backend`` is passed unchanged to :class:`AshSession`; selecting
    ``backend: microvm`` therefore uses AgentENV/Firecracker, while tests can
    inject Docker or a fake backend without changing the strategy.
    """

    def __init__(
        self,
        image: str,
        *,
        backend: dict[str, Any] | None = None,
        runtime_bin: str | None = None,
        timeout: float = 300.0,
        quiet: bool = True,
    ) -> None:
        if not image:
            raise ValueError("image must be non-empty")
        self.image = image
        self.backend = dict(backend or {})
        self.runtime_bin = runtime_bin
        self.timeout = timeout
        self.quiet = quiet

    def spawn(self, _request):
        session = AshSession(
            runtime_bin=self.runtime_bin,
            timeout=self.timeout,
            quiet=self.quiet,
            backend=self.backend,
        )
        if not session.create(self.image):
            session.destroy()
            raise RuntimeError(f"failed to create Ash sandbox from image {self.image!r}")
        return session

    def destroy(self, sandbox) -> None:
        sandbox.destroy()

    def snapshot(self, sandbox, *, name: str) -> str:
        return sandbox.snapshot(name=name)

    def restore(self, snapshot_id: str):
        session = AshSession(
            runtime_bin=self.runtime_bin,
            timeout=self.timeout,
            quiet=self.quiet,
            backend=self.backend,
        )
        if not session.restore(snapshot_id):
            session.destroy()
            raise RuntimeError(f"failed to restore Ash snapshot {snapshot_id!r}")
        return session

    def fork(self, sandbox, *, count: int):
        if count < 0:
            raise ValueError("count must be non-negative")
        # Pool-level fork is provider-specific; the generic strategy only
        # relies on restore, so snapshots remain the portable branch seam.
        snapshot_id = self.snapshot(sandbox, name=f"rollout-fork-{uuid.uuid4().hex}")
        return [self.restore(snapshot_id) for _ in range(count)]
