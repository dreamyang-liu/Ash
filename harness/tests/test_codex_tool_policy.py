"""Codex policy checks, including real binaries against offline fake services."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from harness.core.journal import JournalWriter
from harness.core.slot import McpWiring, TaskSpec
from harness.slots.codex_config import _config_pairs, _mcp_tool_policy
from harness.slots.codex_sdk import CodexSdkSlot


RESOURCE_TOOLS = {"list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"}


def test_native_tool_policy_overrides_caller_settings_only_with_mcp():
    hostile = {"web_search": '"live"', "features.shell_tool": "true",
               "features.multi_agent": "true", "model_provider": '"fixture"'}
    wiring = McpWiring(name="ash", url="http://fixture/mcp")

    def overrides(mcp):
        return CodexSdkSlot()._config_overrides(mcp, {"config_overrides": hostile})

    effective = dict(value.split("=", 1) for value in overrides(wiring))
    for pair in _config_pairs(_mcp_tool_policy()):
        key, value = pair.split("=", 1)
        assert effective[key] == value
    assert effective["model_provider"] == '"fixture"'
    assert effective["mcp_servers.ash.url"] == '"http://fixture/mcp"'
    assert dict(value.split("=", 1) for value in overrides(None)) == hostile


@pytest.mark.parametrize("mode", ["start", "resume", "fork", "fork_at"])
def test_thread_operations_carry_scoped_tool_policy(mode):
    slot = CodexSdkSlot()
    slot._tool_config = _mcp_tool_policy()
    calls = []

    def record(*args):
        calls.append(args[-1])
        return SimpleNamespace(thread=SimpleNamespace(id="thread-fixture"))

    client = SimpleNamespace(thread_start=record, thread_resume=record, thread_fork=record)
    if mode == "fork_at":
        slot._client = client
        slot.fork_at("parent", config={"web_search": "live"})
    else:
        extra = {} if mode == "start" else {"resume_session_id": "parent", "fork": mode == "fork"}
        slot._open_thread(client, TaskSpec(prompt="fixture", cwd="/tmp"), extra, None)
    assert calls[0]["config"] == slot._tool_config
    assert calls[0]["config"]["web_search"] == "disabled"


@pytest.mark.parametrize("method,params,reply", [
    ("item/commandExecution/requestApproval", {}, {"decision": "decline"}),
    ("item/fileChange/requestApproval", {}, {"decision": "decline"}),
    ("item/permissions/requestApproval", {"requestedPermissions": {"network": True}},
     {"permissions": {}, "scope": "turn"}),
    ("mcpServer/elicitation/request", {"serverName": "unrelated"}, {"action": "decline"}),
    ("mcpServer/elicitation/request", {"serverName": "ash"}, {"action": "accept", "content": {}}),
])
def test_scoped_approvals_cannot_enable_native_execution(method, params, reply):
    slot = CodexSdkSlot(policy=lambda *args: ("allow", None))
    slot._mcp_name = "ash"
    slot._journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    assert slot._on_approval(method, params) == reply


@pytest.fixture
def offline_services():
    captured = {"requests": [], "mcp_calls": [], "unrelated": [], "flat_only": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(405)
            self.end_headers()

        def do_DELETE(self):
            self.send_response(204)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path.endswith("mcp"):
                method = body.get("method")
                if self.path != "/mcp":
                    captured["unrelated"].append(body)
                if method == "initialize":
                    result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                              "serverInfo": {"name": "offline-fixture", "version": "1"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": name, "description": "Offline fixture tool",
                                         "inputSchema": {"type": "object", "properties": {}}}
                                        for name in ("shell", "text_editor")]}
                elif method == "tools/call":
                    captured["mcp_calls"].append(body["params"])
                    result = {"content": [{"type": "text", "text": "offline tool executed"}]}
                elif "id" not in body:
                    self.send_response(202)
                    self.end_headers()
                    return
                else:
                    result = {}
                self.respond("application/json", json.dumps(
                    {"jsonrpc": "2.0", "id": body["id"], "result": result}).encode())
                return
            captured["requests"].append(body)
            gate = captured.get("gates", {}).get(len(captured["requests"]))
            if gate is not None:
                gate.wait(120)
            if captured["flat_only"] and any(tool.get("type") != "function" for tool in body.get("tools", [])):
                self.send_response(400)
                self.end_headers()
                return
            if len(captured["requests"]) == 1:
                item = {"type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
                        "name": "shell", "namespace": captured.get("namespace", "mcp__ash"), "arguments": "{}"}
                if captured["flat_only"]:
                    item.update(name="mcp__ash__shell")
                    item.pop("namespace")
            else:
                item = {"type": "message", "id": "msg_fixture", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": "Offline fixture complete.", "annotations": []}]}
            outputs = captured.get("batches", {}).get(len(captured["requests"]), [item])
            response = {"id": f"resp_{len(captured['requests'])}", "object": "response", "created_at": 1,
                        "status": "completed", "output": outputs,
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
            events = [{"type": "response.output_item.done", "output_index": index, "item": output}
                      for index, output in enumerate(outputs)]
            events.append({"type": "response.completed", "response": response})
            payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
            self.respond("text/event-stream", payload.encode())

        def respond(self, content_type, payload):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_codex_sdk_exposes_only_mcp_and_resource_helpers(tmp_path, offline_services):
    pytest.importorskip("openai_codex")
    slot = CodexSdkSlot()
    endpoint, captured = offline_services
    home = tmp_path / "codex-home"
    home.mkdir()
    inherited = ('web_search = "live"\n[features]\nshell_tool = true\nshell_snapshot = true\nmulti_agent = true\n'
                 '[mcp_servers."unrelated.with.dot"]\nurl = ' + json.dumps(endpoint + "/other-mcp") + '\n')
    (home / "config.toml").write_text(inherited)
    overrides = {
        "model_provider": '"fixture"',
        "model_providers.fixture.name": '"Offline fixture"',
        "model_providers.fixture.base_url": json.dumps(endpoint + "/v1"),
        "model_providers.fixture.wire_api": '"responses"',
        "model_providers.fixture.requires_openai_auth": "false",
        "features.enable_request_compression": "false",
        "web_search": '"live"', "features.shell_tool": "true", "features.multi_agent": "true",
        "features.shell_snapshot": "true",
    }
    task = TaskSpec(prompt="Offline tool protocol fixture.", cwd=str(tmp_path),
                    model="openai.gpt-5.6-luna", timeout_s=30,
                    env={"CODEX_HOME": str(home), "OPENAI_API_KEY": "offline-fixture",
                         "ASH_SECRET_SENTINEL": "codex-cache-test-secret-not-a-real-key"},
                    extra={"config_overrides": overrides})
    with JournalWriter(tmp_path / "journal.jsonl", run_id="fixture") as journal:
        result = slot.run(task, journal, McpWiring(name="ash", url=endpoint + "/mcp"))
    (tmp_path / "capture.json").write_text(json.dumps(captured, indent=2))
    assert result.status == "completed", result.error
    assert len(captured["requests"]) == 2
    assert [{key: call[key] for key in ("name", "arguments")} for call in captured["mcp_calls"]] == [
        {"name": "shell", "arguments": {}}]
    assert captured["unrelated"] == []
    for request in captured["requests"]:
        tools = request["tools"]
        assert {tool["name"] for tool in tools if tool["type"] == "function"} == RESOURCE_TOOLS
        namespaces = [tool for tool in tools if tool["type"] == "namespace"]
        assert len(namespaces) == 1 and namespaces[0]["name"] == "mcp__ash"
        assert {tool["name"] for tool in namespaces[0]["tools"]} == {"shell", "text_editor"}
        assert all(tool["type"] in ("function", "namespace") for tool in tools)
    assert (home / "config.toml").read_text() == inherited
    assert not list((home / "shell_snapshots").glob("*.sh"))
    assert not any(b"codex-cache-test-secret-not-a-real-key" in path.read_bytes()
                   for path in home.rglob("*") if path.is_file())


def test_sdk_tools_round_trip_through_flat_provider_gateway(tmp_path, offline_services):
    pytest.importorskip("openai_codex")
    from harness.gateway.routing import ModelRoute, RoutingTable
    from harness.gateway.server import GatewayServer

    endpoint, captured = offline_services
    captured["flat_only"] = True
    home = tmp_path / "codex-home"
    home.mkdir()
    table = RoutingTable()
    table.add_route("default", ModelRoute(base_url=endpoint, flatten_tool_namespaces=True))
    with GatewayServer(table) as gateway:
        token = table.mint("fixture")
        overrides = {
            "model_provider": '"fixture"',
            "model_providers.fixture.name": '"Offline namespace adapter"',
            "model_providers.fixture.base_url": json.dumps(gateway.base_url + "/v1"),
            "model_providers.fixture.wire_api": '"responses"',
            "model_providers.fixture.requires_openai_auth": "false",
            "model_providers.fixture.env_key": '"ASH_FIXTURE_GATEWAY_TOKEN"',
            "features.enable_request_compression": "false",
        }
        task = TaskSpec(prompt="Offline tool round trip.", cwd=str(tmp_path), model="Qwen/Qwen3.8-27B",
                        timeout_s=30, env={"CODEX_HOME": str(home), "ASH_FIXTURE_GATEWAY_TOKEN": token.token},
                        extra={"config_overrides": overrides})
        with JournalWriter(tmp_path / "journal.jsonl", run_id="fixture") as journal:
            outcome = CodexSdkSlot().run(task, journal, McpWiring(name="ash", url=endpoint + "/mcp"))
    assert outcome.status == "completed", outcome.error
    assert len(captured["requests"]) == 2
    assert len(captured["mcp_calls"]) == 1 and captured["mcp_calls"][0]["name"] == "shell"
    assert all(tool["type"] == "function" for request in captured["requests"] for tool in request["tools"])
    assert not list((home / "shell_snapshots").glob("*.sh"))
