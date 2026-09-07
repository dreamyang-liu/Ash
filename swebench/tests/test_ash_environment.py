from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import swebench.rollout_groups.ash_environment as environment_module
from swebench.rollout_groups.ash_environment import AshSessionEnvironmentProvider


class _Session:
    instances = []

    def __init__(self, **_kwargs):
        self.sandbox_id = "restored-child"
        self.restored = None
        self.destroyed = False
        self.__class__.instances.append(self)

    def restore_checkpoint(self, checkpoint_id, *, agent_id=""):
        self.restored = (checkpoint_id, agent_id)

    def destroy(self):
        self.destroyed = True


class _Parent:
    sandbox_id = "parent-sandbox"
    checkpoint_capabilities = SimpleNamespace(
        state_scope="full-runtime",
        multiple_restore=True,
        explicit_release=True,
    )

    def __init__(self):
        self.released = []
        self.fail_release = False

    def create_checkpoint(self, *, name=None):
        assert name == "job-step-1"
        return "checkpoint-1"

    def release_checkpoint(self, checkpoint_id):
        if self.fail_release:
            raise RuntimeError("AgentENV unavailable")
        self.released.append(checkpoint_id)


def _provider(monkeypatch):
    _Session.instances.clear()
    monkeypatch.setattr(environment_module, "AshSession", _Session)
    return AshSessionEnvironmentProvider(
        image="template-1",
        backend={"backend": "microvm"},
    )


def test_provider_creates_restores_and_idempotently_releases_checkpoint(monkeypatch):
    provider = _provider(monkeypatch)
    parent = _Parent()

    checkpoint = provider.create_checkpoint(
        parent, owner_job_id="job-1", name="job-step-1"
    )
    child = provider.restore_checkpoint(checkpoint, agent_id="child-1")

    assert checkpoint.owner_job_id == "job-1"
    assert checkpoint.state_scope == "full-runtime"
    assert child.restored == ("checkpoint-1", "child-1")
    assert provider.release_checkpoint(checkpoint) is True
    assert provider.release_checkpoint(checkpoint) is False
    assert parent.released == ["checkpoint-1"]


def test_provider_rejects_checkpoint_metadata_from_another_owner(monkeypatch):
    provider = _provider(monkeypatch)
    parent = _Parent()
    checkpoint = provider.create_checkpoint(
        parent, owner_job_id="job-1", name="job-step-1"
    )
    forged = replace(checkpoint, owner_job_id="job-2")

    with pytest.raises(ValueError, match="metadata does not match"):
        provider.restore_checkpoint(forged)
    with pytest.raises(ValueError, match="metadata does not match"):
        provider.release_checkpoint(forged)

    assert provider.release_checkpoint(checkpoint) is True


def test_provider_retains_ownership_when_release_fails(monkeypatch):
    provider = _provider(monkeypatch)
    parent = _Parent()
    checkpoint = provider.create_checkpoint(
        parent, owner_job_id="job-1", name="job-step-1"
    )
    parent.fail_release = True

    with pytest.raises(RuntimeError, match="AgentENV unavailable"):
        provider.release_checkpoint(checkpoint)

    parent.fail_release = False
    assert provider.release_checkpoint(checkpoint) is True


def test_provider_refuses_filesystem_only_checkpoint_backend(monkeypatch):
    provider = _provider(monkeypatch)
    parent = _Parent()
    parent.checkpoint_capabilities = SimpleNamespace(
        state_scope="filesystem-only",
        multiple_restore=True,
        explicit_release=True,
    )

    with pytest.raises(RuntimeError, match="requires a full-runtime checkpoint"):
        provider.create_checkpoint(
            parent, owner_job_id="job-1", name="job-step-1"
        )
