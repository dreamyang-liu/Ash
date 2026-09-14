"""The common rollout entry must propagate its budget before creating the VM."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from harness.execution.backends import BackendError
from harness.orchestrator.run import Orchestrator, RunSpec


@pytest.fixture
def recorded_session(monkeypatch):
    observed = {}

    def create(image, resources=None):
        observed["image"] = image
        observed["resources"] = resources
        return True

    session = SimpleNamespace(create=create, sandbox_id="fixture-vm", supports_snapshot=lambda: False)

    def make_session(**kwargs):
        observed["constructor"] = kwargs
        return session

    monkeypatch.setattr("harness.execution.session.SandboxSession", make_session)
    monkeypatch.setattr(Orchestrator, "_serve_in_process", lambda *args: "fixture-http-wiring")
    return observed


@pytest.mark.parametrize("transport", ["http", "stdio"])
@pytest.mark.parametrize("budget,explicit,expected", [
    (1800, None, 2400), (1800, 600, 2400), (1800, 7200, 7200),
    (3600, None, 4200), (10800, 600, 11400), (1.2, None, 602),
])
def test_owned_rollout_passes_longer_lifetime_without_changing_budget(
        recorded_session, transport, budget, explicit, expected):
    backend = {"backend": "microvm", "microvm": {"server_url": "http://fixture", "allow_internet": False}}
    if explicit is not None:
        backend["microvm"]["sandbox_ttl"] = explicit
    before = deepcopy(backend)
    spec = RunSpec(prompt="fixture", sandbox_image="ready-template", backend=backend,
                   timeout_s=budget, transport=transport, sandbox_resources={"cpu": 2, "memory_mb": 12288})
    Orchestrator()._own_sandbox(spec, None)
    effective = recorded_session["constructor"]["backend"]
    assert effective["microvm"]["sandbox_ttl"] == expected
    assert effective["microvm"]["allow_internet"] is False
    assert effective["microvm"]["server_url"] == "http://fixture"
    assert spec.timeout_s == budget and spec.backend == before
    assert recorded_session["image"] == "ready-template"
    assert recorded_session["resources"] == spec.sandbox_resources


def test_snapshot_start_uses_the_continuation_budget(recorded_session):
    spec = RunSpec(prompt="continue", sandbox_image="parent-snapshot", timeout_s=125.5,
                   backend={"backend": "microvm"}, transport="http",
                   origin={"snapshot_id": "parent-snapshot", "message_step": 55})
    Orchestrator()._wire_sandbox(spec, None)
    assert recorded_session["constructor"]["backend"]["microvm"]["sandbox_ttl"] == 726
    assert recorded_session["image"] == "parent-snapshot"
    assert spec.timeout_s == 125.5


@pytest.mark.parametrize("backend", [{}, {"backend": "docker"}])
def test_non_microvm_owned_runs_keep_backend_defaults(recorded_session, backend):
    spec = RunSpec(prompt="fixture", sandbox_image="image", backend=backend)
    Orchestrator()._own_sandbox(spec, None)
    assert recorded_session["constructor"]["backend"] == backend


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_vm_budget_is_rejected_before_allocation(recorded_session, budget):
    spec = RunSpec(prompt="fixture", sandbox_image="image", timeout_s=budget, backend={"backend": "microvm"})
    with pytest.raises(BackendError, match="operation budget"):
        Orchestrator()._own_sandbox(spec, None)
    assert recorded_session == {}


def test_externally_owned_server_is_not_silently_reconfigured(recorded_session):
    spec = RunSpec(prompt="fixture", mcp_url="http://external/mcp", sandbox_id="external-vm",
                   timeout_s=1800, backend={"backend": "microvm", "microvm": {"sandbox_ttl": 600}})
    Orchestrator()._wire_sandbox(spec, None)
    assert recorded_session == {}
    assert spec.backend["microvm"]["sandbox_ttl"] == 600


def test_longer_vm_lifetime_reaches_agentenv_request(recorded_session):
    import asyncio
    import json

    import httpx

    from harness.execution.backends import build_pool

    spec = RunSpec(prompt="fixture", sandbox_image="ready-template", timeout_s=1800,
                   backend={"backend": "microvm", "microvm": {"server_url": "http://fixture"}})
    Orchestrator()._own_sandbox(spec, None)
    requests = []

    def respond(request):
        if request.method == "POST" and request.url.path == "/sandboxes":
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"sandboxID": "fixture-vm", "templateID": "ready-template"})
        assert request.method == "DELETE"
        return httpx.Response(204)

    async def create():
        pool = build_pool(recorded_session["constructor"]["backend"])
        await pool._client.aclose()
        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            await pool.spawn("ready-template")
        finally:
            await pool.close()

    asyncio.run(create())
    assert len(requests) == 1
    assert requests[0]["timeout"] == 2400
    assert requests[0]["autoPause"] is True
    assert requests[0]["templateID"] == "ready-template"
