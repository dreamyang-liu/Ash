from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from swebench.rollout_groups.protocol import (
    GeneratedSpan,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutDeletion,
    RolloutSubmission,
    Trajectory,
)
from swebench.rollout_groups.runner import GroupRolloutService
from swebench.rollout_groups.server import RolloutGroupsHTTPServer
from swebench.rollout_groups.environment_catalog import EnvironmentCatalog
from swebench.rollout_groups.ash_environment import AshSessionEnvironmentProvider
import swebench.rollout_groups.server as server_module


def request_payload(job_id="job-1"):
    return {
        "rollout_job_id": job_id,
        "rollout_id": 0,
        "prompt_group_id": "group-1",
        "task_id": "task-1",
        "environment_ref": {
            "kind": "template",
            "id": "swebench-runtime",
            "revision": "sha256:test",
            "resource_profile": "standard",
        },
        "sample_slots": [{"sample_slot_id": "slot-0", "sample_index": 0}],
        "max_samples": 1,
        "minimum_returned_samples": 1,
        "prompt": "hello",
        "prompt_token_ids": [1],
        "model_endpoint": "http://model",
        "expected_weight_version": "7",
        "return_rollout_logprobs": False,
        "sampling_params": {},
        "budgets": {"max_model_calls": 2, "max_tool_calls": 3,
                     "max_wall_time_seconds": 10},
    }


def complete_result(request, _context):
    span = GeneratedSpan(
        response_id="response-1", start=1, end=2,
        input_token_ids=(1,), output_token_ids=(2,), weight_version="7",
        finish_reason="stop",
    )
    trajectory = Trajectory(
        sample_slot_id=request.sample_slots[0].sample_slot_id,
        branch_id="root", messages=[{"role": "user", "content": "hello"}],
        token_ids=[1, 2], prompt_length=1, generated_spans=[span],
        response_text="world",
    )
    return RolloutGroupResult(
        rollout_job_id=request.rollout_job_id,
        prompt_group_id=request.prompt_group_id,
        status="completed", max_samples=1, trajectories=[trajectory],
    )


class _ResultStrategy:
    def __init__(self, fn):
        self.fn = fn

    def run(self, request, context):
        return self.fn(request, context)


def test_protocol_round_trip_and_span_validation():
    request = RolloutGroupRequest.from_dict(request_payload())
    assert request.to_dict()["prompt_token_ids"] == [1]
    with pytest.raises(ValueError, match="output_token_ids length"):
        GeneratedSpan.from_dict({
            "response_id": "r", "start": 1, "end": 3,
            "input_token_ids": [1], "output_token_ids": [2],
            "weight_version": "1", "finish_reason": "stop",
        })


def test_submission_rejects_unknown_status_or_protocol_version():
    with pytest.raises(ValueError, match="unknown submission status"):
        RolloutSubmission("job-1", "unknown")
    with pytest.raises(ValueError, match="unsupported protocol_version"):
        RolloutSubmission("job-1", "queued", protocol_version="future-version")


def test_deletion_rejects_unknown_protocol_version():
    with pytest.raises(ValueError, match="unsupported protocol_version"):
        RolloutDeletion("job-1", "completed", protocol_version="future-version")


def test_protocol_requires_digest_for_image_environment():
    payload = request_payload()
    payload["environment_ref"] = {
        "kind": "image",
        "id": "docker.io/example/task-env",
        "revision": "latest",
        "resource_profile": "standard",
    }

    with pytest.raises(ValueError, match="sha256 digest"):
        RolloutGroupRequest.from_dict(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("task_id",), "   "),
        (("model_endpoint",), "\t"),
        (("environment_ref", "id"), "\n"),
    ],
)
def test_protocol_rejects_blank_required_strings(path, value):
    payload = request_payload()
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="non-empty string"):
        RolloutGroupRequest.from_dict(payload)


def test_protocol_rejects_boolean_integer_fields():
    payload = request_payload()
    payload["sample_slots"][0]["sample_index"] = True

    with pytest.raises(ValueError, match="non-negative integer"):
        RolloutGroupRequest.from_dict(payload)


@pytest.mark.parametrize("field", ["max_model_calls", "max_tool_calls", "max_wall_time_seconds"])
def test_protocol_rejects_boolean_budget_fields(field):
    payload = request_payload()
    payload["budgets"][field] = True

    with pytest.raises(ValueError):
        RolloutGroupRequest.from_dict(payload)


def test_protocol_rejects_non_numeric_consumed_budget():
    with pytest.raises(ValueError, match="finite non-negative number"):
        RolloutGroupResult(
            rollout_job_id="job-invalid-budget",
            prompt_group_id="group-1",
            status="completed",
            max_samples=1,
            consumed_budget={"parent_status": "step_limit"},
        )


@pytest.mark.parametrize("reward", [True, float("inf"), float("nan")])
def test_trajectory_rejects_invalid_numeric_reward(reward):
    payload = complete_result(RolloutGroupRequest.from_dict(request_payload()), None)
    trajectory = payload.trajectories[0].to_dict()
    trajectory["reward"] = reward

    with pytest.raises(ValueError, match="finite number or object"):
        Trajectory.from_dict(trajectory)


def test_result_rejects_unknown_protocol_version():
    with pytest.raises(ValueError, match="unsupported protocol_version"):
        RolloutGroupResult(
            rollout_job_id="job-future-result",
            prompt_group_id="group-1",
            status="completed",
            max_samples=1,
            protocol_version="future-version",
        )


@pytest.mark.parametrize(
    "path, field",
    [
        ((), "future_request_field"),
        (("budgets",), "future_budget_field"),
        (("sample_slots", 0), "future_slot_field"),
    ],
)
def test_protocol_rejects_unknown_request_fields(path, field):
    payload = request_payload()
    target = payload
    for component in path:
        target = target[component]
    target[field] = True

    with pytest.raises(ValueError, match="unknown fields"):
        RolloutGroupRequest.from_dict(payload)


def test_service_idempotency_and_delete():
    service = GroupRolloutService(lambda request, context: _ResultStrategy(complete_result))
    request = RolloutGroupRequest.from_dict(request_payload())
    first = service.submit(request)
    second = service.submit(request)
    assert first.rollout_job_id == second.rollout_job_id
    deadline = time.time() + 2
    while time.time() < deadline and service.get(request.rollout_job_id).status in {"queued", "running"}:
        time.sleep(0.01)
    assert service.get(request.rollout_job_id).status == "completed"
    # A POST response can be lost while the job keeps running. Retrying the
    # same request is still idempotent after completion and reports the
    # current state instead of pretending to enqueue another job.
    assert service.submit(request).status == "completed"
    with pytest.raises(ValueError):
        service.submit(RolloutGroupRequest.from_dict(request_payload(job_id="job-1") | {
            "prompt_group_id": "different",
        }))
    deleted = service.delete(request.rollout_job_id)
    assert deleted.status == "completed"
    with pytest.raises(KeyError):
        service.get(request.rollout_job_id)


def test_optional_environment_validation_runs_before_enqueue():
    class _RejectingEnvironment:
        def __init__(self):
            self.calls = 0

        def validate_request(self, _request):
            self.calls += 1
            raise ValueError("environment_ref is not allowlisted")

    environment = _RejectingEnvironment()
    service = GroupRolloutService(
        lambda request, context: _ResultStrategy(complete_result),
        environment_provider=environment,
    )
    request = RolloutGroupRequest.from_dict(request_payload("invalid-environment"))

    with pytest.raises(ValueError, match="not allowlisted"):
        service.submit(request)

    assert environment.calls == 1
    with pytest.raises(KeyError):
        service.get(request.rollout_job_id)


def test_delete_cancels_active_job_and_releases_record():
    started = threading.Event()

    def wait_for_cancel(request, context):
        started.set()
        while not context.cancel_event.wait(0.01):
            pass
        context.check_cancelled()

    service = GroupRolloutService(lambda request, context: _ResultStrategy(wait_for_cancel))
    request = RolloutGroupRequest.from_dict(request_payload("active-job"))
    service.submit(request)
    assert started.wait(timeout=2)

    deleted = service.delete(request.rollout_job_id)

    assert deleted.status == "cancelled"
    with pytest.raises(KeyError):
        service.get(request.rollout_job_id)


def test_terminal_job_expires_if_consumer_does_not_delete_it():
    service = GroupRolloutService(
        lambda request, context: _ResultStrategy(complete_result),
        result_ttl_seconds=0.01,
    )
    request = RolloutGroupRequest.from_dict(request_payload("expiring-job"))
    service.submit(request)
    deadline = time.time() + 2
    while time.time() < deadline and service.get(request.rollout_job_id).status in {"queued", "running"}:
        time.sleep(0.001)
    assert service.get(request.rollout_job_id).status == "completed"

    time.sleep(0.02)

    with pytest.raises(KeyError):
        service.get(request.rollout_job_id)


def test_http_endpoints():
    service = GroupRolloutService(lambda request, context: _ResultStrategy(complete_result))
    server = RolloutGroupsHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        body = json.dumps(request_payload("http-job")).encode()
        req = urllib.request.Request(base + "/rollout-groups", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            assert response.status == 202
            assert json.load(response)["rollout_job_id"] == "http-job"
        with urllib.request.urlopen(base + "/rollout-environments") as response:
            assert response.status == 200
            assert json.load(response) == {
                "protocol_version": "ash-rollout-v2",
                "environments": [],
            }
        deadline = time.time() + 2
        while time.time() < deadline:
            with urllib.request.urlopen(base + "/rollout-groups/http-job") as response:
                result = json.load(response)
            if result["status"] == "completed":
                break
            time.sleep(0.01)
        assert result["actual_samples"] == 1
        assert result["trajectories"][0]["branch_id"] == "root"
        delete = urllib.request.Request(
            base + "/rollout-groups/http-job",
            method="DELETE",
        )
        with urllib.request.urlopen(delete) as response:
            assert response.status == 200
            deletion = json.load(response)
            assert deletion == {
                "protocol_version": "ash-rollout-v2",
                "rollout_job_id": "http-job",
                "status": "completed",
            }
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(base + "/rollout-groups/http-job")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_lists_static_environment_refs_without_spawn_handles():
    catalog = EnvironmentCatalog.from_dict(
        {
            "environments": [
                {
                    "kind": "template",
                    "id": "runtime-ready",
                    "revision": "v1",
                    "resource_profile": "standard",
                    "spawn_ref": "agentenv-internal-template-id",
                }
            ]
        }
    )
    environment = AshSessionEnvironmentProvider(
        catalog,
        backend={"backend": "microvm"},
    )
    service = GroupRolloutService(
        lambda request, context: _ResultStrategy(complete_result),
        environment_provider=environment,
    )
    server = RolloutGroupsHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/rollout-environments"
        ) as response:
            payload = json.load(response)
        assert payload == {
            "protocol_version": "ash-rollout-v2",
            "environments": [
                {
                    "kind": "template",
                    "id": "runtime-ready",
                    "revision": "v1",
                    "resource_profile": "standard",
                }
            ],
        }
        assert "spawn_ref" not in json.dumps(payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_strategy_cannot_leave_job_non_terminal():
    def bad_strategy(request, _context):
        return _ResultStrategy(lambda req, ctx: RolloutGroupResult(
            rollout_job_id=req.rollout_job_id,
            prompt_group_id=req.prompt_group_id,
            status="running",
            max_samples=req.max_samples,
        ))

    service = GroupRolloutService(bad_strategy)
    request = RolloutGroupRequest.from_dict(request_payload("non-terminal"))
    service.submit(request)
    deadline = time.time() + 2
    while time.time() < deadline:
        result = service.get(request.rollout_job_id)
        if result.status == "failed":
            break
        time.sleep(0.01)
    assert result.status == "failed"
    assert "terminal" in (result.stop_reason or "")


def test_server_shares_agentenv_connection_with_oci_resolver(tmp_path, monkeypatch):
    resolver_config = tmp_path / "resolver.json"
    resolver_config.write_text("{}", encoding="utf-8")
    api_key_file = tmp_path / "agentenv-key"
    api_key_file.write_text("not-a-real-key", encoding="utf-8")
    captured = {}

    class _ResolverConfig:
        @classmethod
        def from_file(cls, path):
            captured["resolver_config_path"] = path
            return object()

    class _Resolver:
        def __init__(self, config, **kwargs):
            captured["resolver_config"] = config
            captured["resolver_kwargs"] = kwargs

    sentinel_service = object()

    def build_service(**kwargs):
        captured["service_kwargs"] = kwargs
        return sentinel_service

    def serve(service, **kwargs):
        captured["served"] = (service, kwargs)

    backend = {
        "backend": "microvm",
        "microvm": {
            "server_url": "http://agentenv.example:8000",
            "runtime_port": 3000,
            "api_key_file": str(api_key_file),
        },
    }
    monkeypatch.setattr(server_module, "AgentEnvOCIResolverConfig", _ResolverConfig)
    monkeypatch.setattr(server_module, "AgentEnvOCIResolver", _Resolver)
    monkeypatch.setattr(server_module, "build_service", build_service)
    monkeypatch.setattr(server_module, "serve", serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ash-rollout-server",
            "--strategy",
            "checkpoint-agent-loop-v1",
            "--agentenv-oci-resolver-config",
            str(resolver_config),
            "--backend-json",
            json.dumps(backend),
            "--miles-session-endpoint",
            "http://miles-session.example:30000",
        ],
    )

    server_module.main()

    assert captured["resolver_config_path"] == str(resolver_config)
    assert captured["resolver_kwargs"] == {
        "aenv_server_url": "http://agentenv.example:8000",
        "aenv_api_key": None,
        "aenv_api_key_file": str(api_key_file),
    }
    assert captured["service_kwargs"]["oci_resolver"].__class__ is _Resolver
    assert captured["service_kwargs"]["backend"] == backend
    assert captured["served"] == (sentinel_service, {"host": "0.0.0.0", "port": 11001})


def test_cli_sequential_service_has_a_real_model_client():
    service = server_module.build_service(strategy="sequential")
    assert service.model_client is not None
