import pytest
from types import SimpleNamespace

from runstore.api import create_app
from runstore.tests.test_store import store


def test_health_is_authenticated_and_reports_store_readiness():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    fake_store = SimpleNamespace(
        health=lambda: {"status": "ok", "database": "ready"}
    )
    with TestClient(create_app(fake_store, "fixture-token")) as client:
        assert client.get("/health").status_code == 401
        response = client.get(
            "/health", headers={"Authorization": "Bearer fixture-token"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "ready"}


def test_api_auth_validation_status_and_durable_result(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(store, "fixture-token")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/v1/jobs").status_code == 401
        client.headers["Authorization"] = "Bearer fixture-token"
        assert client.get("/health").json() == {
            "status": "ok",
            "database": "ready",
        }
        assert client.post("/v1/jobs", json={"kind": "rollout", "spec": {"prompt": "fixture"}}).status_code == 422
        created = client.post("/v1/jobs", headers={"Idempotency-Key": "fixture"},
                              json={"kind": "rollout", "spec": {"prompt": "fixture"}})
        assert created.status_code == 202, created.text
        job_id = created.json()["id"]
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"
        claim = store.claim("fixture-worker")
        store.finish(job_id, claim["lease_token"], {"final_text": "durable"})
        for repeat in range(2):
            result = client.get(f"/v1/jobs/{job_id}/result").json()
            assert result["ready"] and result["result"]["final_text"] == "durable"
        assert len(client.get(f"/v1/jobs/{job_id}/attempts").json()) == 1


def test_http_submission_freezes_profile_defaults(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from runstore.specs import digest

    profile = {"python": "/usr/bin/python3.11", "run_defaults": {"slot": "codex", "tools": "shell_only"}}
    with TestClient(create_app(store, "fixture", profiles={"default": profile})) as client:
        response = client.post("/v1/jobs", headers={"Authorization": "Bearer fixture", "Idempotency-Key": "profile"},
                               json={"kind": "rollout", "spec": {"prompt": "fixture"}})
        assert response.status_code == 202, response.text
        request = response.json()["request"]
        assert request["spec"]["slot"] == "codex"
        assert request["spec"]["tools"] == "shell_only"
        assert request["profile_hash"] == digest(profile)
