"""EnvironmentProvider backed by AshSession and the real Ash sandbox SDK."""

from __future__ import annotations

import threading
from typing import Any

from ..sandbox import AshSession
from .runner import EnvironmentCheckpoint


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
        self._checkpoint_lock = threading.RLock()
        self._checkpoint_owners: dict[
            str, tuple[EnvironmentCheckpoint, AshSession]
        ] = {}

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

    def create_checkpoint(
        self,
        sandbox,
        *,
        owner_job_id: str,
        name: str | None = None,
    ) -> EnvironmentCheckpoint:
        if not owner_job_id:
            raise ValueError("owner_job_id must be non-empty")
        capabilities = sandbox.checkpoint_capabilities
        if capabilities is None:
            raise NotImplementedError("selected environment backend does not support checkpoints")
        if capabilities.state_scope != "full-runtime":
            raise RuntimeError(
                "agent rollout branching requires a full-runtime checkpoint; "
                f"backend provides {capabilities.state_scope!r}"
            )
        checkpoint_id = sandbox.create_checkpoint(name=name)
        checkpoint = EnvironmentCheckpoint(
            checkpoint_id=checkpoint_id,
            owner_job_id=owner_job_id,
            source_sandbox_id=sandbox.sandbox_id,
            backend="agentenv-microvm",
            state_scope=capabilities.state_scope,
            multiple_restore=capabilities.multiple_restore,
            explicit_release=capabilities.explicit_release,
        )
        with self._checkpoint_lock:
            if checkpoint_id in self._checkpoint_owners:
                raise RuntimeError(f"checkpoint id collision: {checkpoint_id!r}")
            self._checkpoint_owners[checkpoint_id] = (checkpoint, sandbox)
        return checkpoint

    def restore_checkpoint(
        self,
        checkpoint: EnvironmentCheckpoint,
        *,
        agent_id: str = "",
    ):
        if checkpoint.backend != "agentenv-microvm" or checkpoint.state_scope != "full-runtime":
            raise ValueError("checkpoint is incompatible with the AgentENV microVM provider")
        self._owned_checkpoint(checkpoint)
        session = AshSession(
            runtime_bin=self.runtime_bin,
            timeout=self.timeout,
            quiet=self.quiet,
            backend=self.backend,
        )
        try:
            session.restore_checkpoint(checkpoint.checkpoint_id, agent_id=agent_id)
        except Exception:
            session.destroy()
            raise
        return session

    def release_checkpoint(self, checkpoint: EnvironmentCheckpoint) -> bool:
        with self._checkpoint_lock:
            owned = self._checkpoint_owners.get(checkpoint.checkpoint_id)
        if owned is None:
            return False
        registered, owner = owned
        if checkpoint != registered:
            raise ValueError("checkpoint metadata does not match the creating rollout job")
        with self._checkpoint_lock:
            current = self._checkpoint_owners.get(checkpoint.checkpoint_id)
            if current != owned:
                raise RuntimeError("checkpoint ownership changed during release")
            del self._checkpoint_owners[checkpoint.checkpoint_id]
        try:
            owner.release_checkpoint(checkpoint.checkpoint_id)
        except Exception:
            with self._checkpoint_lock:
                self._checkpoint_owners.setdefault(checkpoint.checkpoint_id, owned)
            raise
        return True

    def _owned_checkpoint(self, checkpoint: EnvironmentCheckpoint) -> AshSession:
        with self._checkpoint_lock:
            owned = self._checkpoint_owners.get(checkpoint.checkpoint_id)
        if owned is None:
            raise ValueError("checkpoint is unknown or has already been released")
        registered, owner = owned
        if checkpoint != registered:
            raise ValueError("checkpoint metadata does not match the creating rollout job")
        return owner
