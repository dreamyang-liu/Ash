from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import swebench.rollout_groups.ash_environment as environment_module
from swebench.rollout_groups.ash_environment import AshSessionEnvironmentProvider
from swebench.rollout_groups.environment_catalog import EnvironmentCatalog
from swebench.rollout_groups.protocol import EnvironmentRef


class _Session:
    instances = []

    def __init__(self, **_kwargs):
        self.sandbox_id = "restored-child"
        self.created = None
        self.restored = None
        self.destroyed = False
        self.__class__.instances.append(self)

    def create(self, image):
        self.created = image
        return True

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
        catalog=EnvironmentCatalog.from_dict(
            {
                "environments": [
                    {
                        "kind": "template",
                        "id": "swebench-runtime",
                        "revision": "sha256:test",
                        "resource_profile": "standard",
                        "spawn_ref": "template-1",
                    }
                ]
            }
        ),
        backend={"backend": "microvm"},
    )


def _request(**overrides):
    values = {
        "kind": "template",
        "id": "swebench-runtime",
        "revision": "sha256:test",
        "resource_profile": "standard",
    }
    values.update(overrides)
    return SimpleNamespace(environment_ref=EnvironmentRef(**values))


def test_provider_resolves_request_environment_at_spawn(monkeypatch):
    provider = _provider(monkeypatch)

    session = provider.spawn(_request())

    assert session.created == "template-1"


def test_provider_rejects_unlisted_or_backend_incompatible_environment(monkeypatch):
    provider = _provider(monkeypatch)

    with pytest.raises(ValueError, match="not allowlisted"):
        provider.spawn(_request(revision="sha256:other"))
    with pytest.raises(ValueError, match="requires an OCI resolver"):
        provider.spawn(_request(kind="image"))


def test_provider_resolves_digest_pinned_image_through_oci_resolver(monkeypatch):
    class _Resolver:
        def __init__(self):
            self.validated = []
            self.resolved = []

        def validate(self, ref):
            self.validated.append(ref)

        def resolve(self, ref):
            self.resolved.append(ref)
            return SimpleNamespace(spawn_ref="runtime-snapshot-1")

    _Session.instances.clear()
    monkeypatch.setattr(environment_module, "AshSession", _Session)
    resolver = _Resolver()
    provider = AshSessionEnvironmentProvider(
        catalog=None,
        oci_resolver=resolver,
        backend={"backend": "microvm"},
    )
    request = _request(
        kind="image",
        id="docker.io/example/task-env",
        revision="sha256:" + "a" * 64,
    )

    provider.validate_request(request)
    session = provider.spawn(request)

    assert resolver.validated == [request.environment_ref, request.environment_ref]
    assert resolver.resolved == [request.environment_ref]
    assert session.created == "runtime-snapshot-1"


def test_provider_can_use_pre_resolved_image_from_static_catalog(monkeypatch):
    _Session.instances.clear()
    monkeypatch.setattr(environment_module, "AshSession", _Session)
    provider = AshSessionEnvironmentProvider(
        catalog=EnvironmentCatalog.from_dict(
            {
                "environments": [
                    {
                        "kind": "image",
                        "id": "docker.io/example/task-env",
                        "revision": "sha256:" + "a" * 64,
                        "resource_profile": "standard",
                        "spawn_ref": "prebuilt-runtime-snapshot",
                    }
                ]
            }
        ),
        backend={"backend": "microvm"},
    )
    request = _request(
        kind="image",
        id="docker.io/example/task-env",
        revision="sha256:" + "a" * 64,
    )

    session = provider.spawn(request)

    assert session.created == "prebuilt-runtime-snapshot"


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
