import json
import time
from types import SimpleNamespace

import httpx
import pytest

from harness.core.control import RunControl
from harness.core.journal import JournalWriter, read_journal
from harness.core.result import ToolResult
from harness.core.slot import SlotResult
from harness.execution.pipeline import CallContext, ToolPipeline
from harness.orchestrator.run import Orchestrator, RunSpec
from harness.rollout import RolloutControls
from harness.tests.test_gateway import upstream as upstream


def contract(url, **changes):
    return {"model_endpoint": url, "model": "checkpoint-model", "deadline_at": time.time() + 30,
            "max_model_calls": 2, "max_tool_calls": 1,
            "sampling_params": {"temperature": 0.4, "max_new_tokens": 7}, **changes}


@pytest.mark.parametrize("slot_name,shape,length_field", [("codex", "responses", "max_output_tokens"),
                                                         ("claude-code", "messages", "max_tokens")])
@pytest.mark.parametrize("message_mode", [False, True])
@pytest.mark.parametrize("bearer_auth", [False, True])
def test_orchestrator_mounts_real_endpoint_sampling_and_admission_caps(tmp_path, upstream, monkeypatch,
                                                                     slot_name, shape, length_field, message_mode,
                                                                     bearer_auth):
    server = SimpleNamespace(pipeline=ToolPipeline())
    destroyed, executed = [], []
    owned = SimpleNamespace(server=server, sandbox_id="test-sandbox", destroy=lambda: destroyed.append(True))
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: (owned, None))
    monkeypatch.setattr(Orchestrator, "_wire_checkpoints", lambda *args: None)

    class Slot:
        def run(self, task, journal, mcp):
            url = task.env["ANTHROPIC_BASE_URL"] + "/v1/" + shape
            headers = {"Authorization": "Bearer " + task.env["ANTHROPIC_AUTH_TOKEN"]}
            for _ in range(2):
                response = httpx.post(url, headers=headers, json={"model": "agent-requested", "temperature": 1,
                    "output_config": {"effort": "xhigh", "format": {"type": "json_schema"}}}, timeout=5)
                assert response.status_code == 200
            ctx = CallContext("agent", "sandbox", "shell", {"command": "true"})
            def execute(name, args):
                executed.append(name)
                return ToolResult(True, "ok")
            assert server.pipeline.execute(ctx, execute).success
            if message_mode:
                for _ in range(9):
                    assert server.pipeline.execute(ctx, execute).success
            else:
                assert not server.pipeline.execute(ctx, execute).success
            assert httpx.post(url, headers=headers, json={"model": "agent-requested"}, timeout=5).status_code == 400
            return SlotResult(status="completed")

    monkeypatch.setattr("harness.slots.load_slot", lambda name: Slot)
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"routes": {
        "default": {"base_url": "http://unused.invalid", "api_key": "must-not-leak",
                    "auth_scheme": "bearer" if bearer_auth else "protocol",
                    "omit_anthropic_effort": bearer_auth},
        "agent-requested": {"base_url": "http://unused.invalid", "api_key": "must-not-leak"},
    }}))
    path = tmp_path / "journal.jsonl"
    controls = contract(upstream.base_url + "/v1")
    if bearer_auth:
        monkeypatch.setenv("ASH_ROLLOUT_TEST_KEY", "explicit-rollout-key")
        controls["api_key_env"] = "ASH_ROLLOUT_TEST_KEY"
    if message_mode:
        controls["message_export"] = True
        controls["max_turns"] = controls.pop("max_model_calls")
        controls.pop("max_tool_calls")
        controls["sampling_params"].update(top_k=11, stop=["stop-here"])
    result = Orchestrator().run(RunSpec(prompt="task", slot=slot_name, transport="http", routes_file=str(routes),
                                       journal_path=path, extra={"rollout_contract": controls}))
    assert result.status == "error" and "budget" in result.error
    expected_tools = 10 if message_mode else 1
    assert len(upstream.requests) == 2 and executed == ["shell"] * expected_tools and destroyed
    for item in upstream.requests:
        assert item["body"]["temperature"] == 0.4
        assert item["body"][length_field] == 7
        assert item["body"]["model"] == "checkpoint-model"
        if message_mode:
            assert item["body"]["top_k"] == 11
            assert item["body"]["stop_sequences" if shape == "messages" else "stop"] == ["stop-here"]
        assert "must-not-leak" not in json.dumps(item["headers"])
        assert ("effort" not in item["body"]["output_config"]) == (bearer_auth and shape == "messages")
        assert item["body"]["output_config"]["format"] == {"type": "json_schema"}
        if bearer_auth:
            headers = {key.lower(): value for key, value in item["headers"].items()}
            assert headers["authorization"] == "Bearer explicit-rollout-key"
            assert "x-api-key" not in headers
    usage = [e for e in read_journal(path) if e["type"] == "rollout.usage"][-1]
    assert usage["model_calls"] == 2 and usage["tool_calls"] == expected_tools


@pytest.mark.parametrize("max_turns", [None, 0, -1, True, 1.5])
def test_message_controls_reject_invalid_turn_limit(tmp_path, max_turns):
    controls = contract("http://model")
    controls.pop("max_model_calls")
    controls.pop("max_tool_calls")
    controls.update(message_export=True, max_turns=max_turns)
    with JournalWriter(tmp_path / "journal", run_id="run") as journal:
        with pytest.raises(ValueError, match="max_turns"):
            RolloutControls(controls, journal, RunControl())


def test_message_controls_reject_legacy_count_caps(tmp_path):
    controls = contract("http://model", message_export=True, max_turns=2)
    with JournalWriter(tmp_path / "journal", run_id="run") as journal:
        with pytest.raises(ValueError, match="model/tool call budgets"):
            RolloutControls(controls, journal, RunControl())


def test_expired_group_never_allocates_a_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.slots.load_slot", lambda name: lambda: SimpleNamespace())
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: pytest.fail("Expired group allocated a VM"))
    result = Orchestrator().run(RunSpec(prompt="task", transport="http", journal_path=tmp_path / "journal",
                                       extra={"rollout_contract": contract("http://model", deadline_at=time.time()-1)}))
    assert "expired before execution" in result.error


def test_session_endpoint_is_used_and_records_are_saved_before_release(tmp_path, monkeypatch):
    calls = []
    state = {"records": [{"request": {"input_ids": [1]}}], "metadata": {"accumulated_token_ids": [1, 2]}}
    def call(method, url, **kwargs):
        calls.append((method, url))
        return httpx.Response(200, request=httpx.Request(method, url),
                              json={"session_id": "recorded"} if method == "POST" else state)
    for method in ("post", "get", "delete"):
        request_method = method.upper()
        monkeypatch.setattr(httpx, method, lambda url, _method=request_method, **kw: call(_method, url, **kw))
    path = tmp_path / "journal"
    with JournalWriter(path, run_id="run") as journal:
        controls = RolloutControls(contract("http://model", session_server_endpoint="http://sessions:30000"), journal, RunControl())
        controls.start()
        assert controls.base_url == "http://sessions:30000/sessions/recorded"
        controls.finish()
    assert [method for method, _ in calls] == ["POST", "GET", "DELETE"]
    assert [e for e in read_journal(path) if e["type"] == "rollout.session_state"][0]["state"] == state
