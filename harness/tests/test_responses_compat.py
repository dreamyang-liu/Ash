import json

import pytest

from harness.gateway.responses_compat import ResponsesNamespaceAdapter
from harness.gateway.routing import ModelRoute


def request():
    return {"model": "served-model", "tools": [
        {"type": "function", "name": "read_mcp_resource", "parameters": {}},
        {"type": "namespace", "name": "mcp__ash", "tools": [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {"type": "function", "name": "text_editor", "parameters": {"type": "object"}}]}],
        "input": [{"type": "function_call", "namespace": "mcp__ash", "name": "shell",
                   "call_id": "call-1", "arguments": '{"command":"echo ok"}'},
                  {"type": "function_call_output", "call_id": "call-1", "output": "ok"}],
        "tool_choice": {"type": "function", "namespace": "mcp__ash", "name": "shell"}}


def test_namespace_request_and_history_are_flattened_without_mutating_source():
    original = request()
    before = json.dumps(original)
    adapter = ResponsesNamespaceAdapter(original)
    assert json.dumps(original) == before
    assert [tool["name"] for tool in adapter.payload["tools"]] == [
        "read_mcp_resource", "mcp__ash__shell", "mcp__ash__text_editor"]
    assert all(tool["type"] == "function" for tool in adapter.payload["tools"])
    assert adapter.payload["input"][0]["name"] == "mcp__ash__shell"
    assert "namespace" not in adapter.payload["input"][0]
    assert adapter.payload["input"][1] == original["input"][1]
    assert adapter.payload["tool_choice"] == {"type": "function", "name": "mcp__ash__shell"}


def test_restored_tool_call_keeps_ids_arguments_and_unrelated_fields():
    adapter = ResponsesNamespaceAdapter(request())
    response = {"id": "response-1", "output": [
        {"type": "function_call", "name": "mcp__ash__shell", "call_id": "call-1",
         "arguments": '{"command":"echo ok"}', "id": "item-1"},
        {"type": "reasoning", "encrypted_content": "leave-byte-content-alone"}],
        "usage": {"input_tokens": 5}}
    restored = adapter.restore(response)
    assert restored["output"][0] == {**response["output"][0], "name": "shell", "namespace": "mcp__ash"}
    assert restored["output"][1] == response["output"][1]
    assert restored["usage"] == response["usage"]


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
def test_sse_handles_split_unicode_and_preserves_unrelated_frames(separator):
    adapter = ResponsesNamespaceAdapter(request())
    payload = {"type": "response.output_item.done", "item": {
        "type": "function_call", "name": "mcp__ash__shell", "call_id": "call-1", "arguments": "你好"}}
    unrelated = b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n"
    framed = separator.join([b"event: response.output_item.done",
                             b"data: " + json.dumps(payload, ensure_ascii=False).encode(), b"", b""])
    output = []
    for byte in framed + unrelated + b"data: [DONE]\n\n":
        output.extend(adapter.feed(bytes([byte])))
    output.append(adapter.finish())
    assert unrelated in b"".join(output)
    data = next(line[5:].strip() for line in output[0].splitlines() if line.startswith(b"data:"))
    parsed = json.loads(data)
    assert parsed["item"]["name"] == "shell" and parsed["item"]["namespace"] == "mcp__ash"
    assert parsed["item"]["arguments"] == "你好"
    assert b"data: [DONE]\n\n" in output


def test_tool_name_collisions_are_rejected_instead_of_misrouting_calls():
    payload = request()
    payload["tools"].append({"type": "function", "name": "mcp__ash__shell"})
    with pytest.raises(ValueError, match="collision"):
        ResponsesNamespaceAdapter(payload)


def test_unsupported_nested_tools_and_unmatched_history_are_rejected():
    payload = request()
    payload["tools"][1]["tools"][0]["type"] = "custom"
    with pytest.raises(ValueError, match="Only named function"):
        ResponsesNamespaceAdapter(payload)
    payload = request()
    payload["input"][0]["name"] = "missing"
    with pytest.raises(ValueError, match="no matching"):
        ResponsesNamespaceAdapter(payload)


def test_flat_requests_and_unknown_response_fields_are_preserved():
    payload = {"model": "model", "input": "hello", "tools": [{"type": "function", "name": "echo"}]}
    adapter = ResponsesNamespaceAdapter(payload)
    assert adapter.payload == payload
    assert adapter.restore({"type": "function_call", "name": "unknown"}) == {
        "type": "function_call", "name": "unknown"}
    assert adapter.feed(b"event: unknown\ndata: not-json\n\n") == [b"event: unknown\ndata: not-json\n\n"]
    assert adapter.feed(b"data: [DONE]") == []
    assert adapter.finish() == b"data: [DONE]"


@pytest.mark.parametrize("tools", ["invalid", {}, False, 0, [None], [{"type": "function", "name": []}],
                                   [{"type": "namespace", "name": "ash", "tools": None}]])
def test_malformed_tool_shapes_fail_with_a_validation_error(tools):
    with pytest.raises(ValueError):
        ResponsesNamespaceAdapter({"tools": tools})


def test_nullable_tools_are_supported_without_changing_input():
    adapter = ResponsesNamespaceAdapter({"tools": None, "input": "hello"})
    assert adapter.payload == {"tools": [], "input": "hello"}


def test_route_adaptation_is_explicit_opt_in():
    assert ModelRoute.from_dict({}).flatten_tool_namespaces is False
    assert ModelRoute.from_dict({"flatten_tool_namespaces": True}).flatten_tool_namespaces is True
    with pytest.raises(ValueError, match="boolean"):
        ModelRoute.from_dict({"flatten_tool_namespaces": "false"})


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_gateway_consumes_namespace_option_for_request_and_response(streaming, enabled):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    import httpx

    from harness.gateway.routing import RoutingTable
    from harness.gateway.server import GatewayServer

    captured = []
    result = {"id": "response-1", "model": "served-model", "output": [{
        "type": "function_call", "name": "mcp__ash__shell", "call_id": "call-1", "arguments": "{}"}],
        "usage": {"input_tokens": 3, "output_tokens": 2}}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if streaming:
                body = ("event: response.completed\ndata: " + json.dumps({
                    "type": "response.completed", "response": result}) + "\n\n").encode()
            else:
                body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    table = RoutingTable()
    table.add_route("default", ModelRoute(base_url=f"http://127.0.0.1:{upstream.server_port}",
                                          flatten_tool_namespaces=enabled))
    try:
        with GatewayServer(table, require_token=False) as gateway:
            response = httpx.post(gateway.base_url + "/v1/responses", json={**request(), "stream": streaming})
        assert response.status_code == 200
        expected = ResponsesNamespaceAdapter(request()).payload if enabled else request()
        assert captured == [{**expected, "stream": streaming}]
        if streaming:
            event = json.loads(next(line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")))
            body = event["response"]
        else:
            body = response.json()
        assert body["output"][0]["name"] == ("shell" if enabled else "mcp__ash__shell")
        assert body["output"][0].get("namespace") == ("mcp__ash" if enabled else None)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
