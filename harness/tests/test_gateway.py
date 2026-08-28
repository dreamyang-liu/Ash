"""Gateway tests against a fake upstream provider.

A stub Anthropic-shaped server stands in for the real API so the whole path --
auth, routing, model rewrite, header forwarding, streaming relay, usage tap,
budget refusal -- is covered without credentials or network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")

from harness.core.journal import JournalWriter, read_journal
from harness.gateway.routing import ModelRoute, RoutingTable
from harness.gateway.server import GATEWAY_EVENT, GatewayServer

RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "upstream-model",
    "content": [{"type": "text", "text": "pong"}],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 5,
    },
}

SSE = (
    b'data: {"type":"message_start","message":{"model":"upstream-model",'
    b'"usage":{"input_tokens":11,"cache_read_input_tokens":3}}}\n\n'
    b'data: {"type":"content_block_delta","delta":{"text":"po"}}\n\n'
    b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    b'data: {"type":"message_stop"}\n\n'
)


class _Upstream:
    """Records what it received so forwarding can be asserted."""

    def __init__(self):
        self.requests = []
        handler = _make_upstream_handler(self)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d" % self._httpd.server_address[1]

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


def _make_upstream_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append({"headers": dict(self.headers), "body": body})

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for frame in SSE.split(b"\n\n"):
                    if not frame.strip():
                        continue
                    chunk = frame + b"\n\n"
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return

            payload = json.dumps(RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("anthropic-request-id", "req_abc")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def upstream():
    server = _Upstream()
    yield server
    server.stop()


@pytest.fixture
def gateway(upstream, tmp_path):
    table = RoutingTable(
        {
            "default": ModelRoute(base_url=upstream.base_url, api_key="real-provider-key"),
            "ash-rl-ckpt-42": ModelRoute(
                base_url=upstream.base_url,
                api_key="rl-key",
                upstream_model="checkpoint-42",
                headers={"X-Rl-Run": "42"},
            ),
        }
    )
    journal = JournalWriter(tmp_path / "gw.jsonl", run_id="gwrun")
    server = GatewayServer(table, journal=journal, port=0)
    server.start()
    yield server, table, journal, tmp_path / "gw.jsonl"
    server.stop()
    journal.close()


def post(server, token, body):
    return httpx.post(
        server.base_url + "/v1/messages",
        json=body,
        headers={"Authorization": "Bearer " + token},
        timeout=30,
    )


# --- auth ------------------------------------------------------------------
def test_unknown_token_is_rejected(gateway):
    server, _table, _journal, _path = gateway
    response = post(server, "not-a-real-token", {"model": "m", "messages": []})
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_provider_key_never_reaches_the_agent(gateway, upstream):
    """The agent presents its slot token; the gateway swaps in the real key."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1", run_id="gwrun")
    post(server, token.token, {"model": "m", "messages": []})

    sent = upstream.requests[-1]["headers"]
    assert sent["x-api-key"] == "real-provider-key"
    assert token.token not in json.dumps(sent)
    assert "Authorization" not in sent


# --- routing / model swap --------------------------------------------------
def test_model_swap_rewrites_name_and_targets_route(gateway, upstream):
    """The whole reason the gateway exists: redirect a model without touching
    the agent."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "ash-rl-ckpt-42", "messages": []})
    assert response.status_code == 200

    request = upstream.requests[-1]
    assert request["body"]["model"] == "checkpoint-42"   # rewritten
    assert request["headers"]["x-api-key"] == "rl-key"   # route's own credential
    assert request["headers"]["X-Rl-Run"] == "42"        # route headers added


def test_unknown_model_falls_back_to_default_route(gateway, upstream):
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    post(server, token.token, {"model": "some-unmapped-model", "messages": []})
    # default route does not rename
    assert upstream.requests[-1]["body"]["model"] == "some-unmapped-model"


def test_models_endpoint_answers_for_discovery(gateway):
    """Claude Code probes /v1/models and treats failure as fatal."""
    server, _table, _journal, _path = gateway
    response = httpx.get(server.base_url + "/v1/models", timeout=10)
    assert response.status_code == 200
    assert "ash-rl-ckpt-42" in [m["id"] for m in response.json()["data"]]


# --- header forwarding -----------------------------------------------------
def test_correlation_and_beta_headers_are_forwarded(gateway, upstream):
    """Dropping these loses per-subagent attribution or changes behaviour."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    httpx.post(
        server.base_url + "/v1/messages",
        json={"model": "m", "messages": []},
        headers={
            "Authorization": "Bearer " + token.token,
            "anthropic-beta": "some-beta-2026-01-01",
            "x-claude-code-session-id": "sess-9",
            "x-claude-code-agent-id": "sub-3",
            "x-claude-code-parent-agent-id": "root-1",
        },
        timeout=30,
    )
    sent = {k.lower(): v for k, v in upstream.requests[-1]["headers"].items()}
    assert sent["anthropic-beta"] == "some-beta-2026-01-01"
    assert sent["x-claude-code-session-id"] == "sess-9"
    assert sent["x-claude-code-agent-id"] == "sub-3"
    assert sent["x-claude-code-parent-agent-id"] == "root-1"


def test_response_body_is_relayed_byte_exact(gateway):
    """Unknown fields (e.g. thinking signatures) must survive untouched."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "m", "messages": []})
    assert response.json() == RESPONSE
    assert response.headers["anthropic-request-id"] == "req_abc"


# --- usage tap + budget ----------------------------------------------------
def test_usage_is_tapped_into_the_journal(gateway):
    server, table, _journal, path = gateway
    token = table.mint("agent-1")
    post(server, token.token, {"model": "m", "messages": []})

    events = [r for r in read_journal(path) if r["type"] == GATEWAY_EVENT]
    assert len(events) == 1
    tap = events[0]
    assert tap["agent_id"] == "agent-1"
    assert tap["status"] == "ok"
    assert tap["usage"]["input_tokens"] == 11
    assert tap["usage"]["output_tokens"] == 7
    assert tap["usage"]["cached_input_tokens"] == 3
    assert tap["usage"]["cache_creation_tokens"] == 5
    assert token.input_tokens == 11 and token.output_tokens == 7


def test_streaming_relays_frames_and_still_taps_usage(gateway):
    server, table, _journal, path = gateway
    token = table.mint("agent-1")
    frames = []
    with httpx.stream(
        "POST",
        server.base_url + "/v1/messages",
        json={"model": "m", "messages": [], "stream": True},
        headers={"Authorization": "Bearer " + token.token},
        timeout=30,
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                frames.append(line)

    kinds = [json.loads(f[5:])["type"] for f in frames]
    assert kinds == ["message_start", "content_block_delta", "message_delta", "message_stop"]

    tap = [r for r in read_journal(path) if r["type"] == GATEWAY_EVENT][-1]
    assert tap["streaming"] is True
    assert tap["usage"]["input_tokens"] == 11
    assert tap["usage"]["output_tokens"] == 7   # from message_delta


def test_budget_is_enforced_not_merely_reported(gateway):
    """A slot-side accountant can only ask an agent to stop; this makes it."""
    server, table, _journal, path = gateway
    token = table.mint("agent-1", budget_usd=0.001)
    table.charge(token, cost_usd=0.002)          # simulate prior spend

    response = post(server, token.token, {"model": "m", "messages": []})
    # 400, not 429: an exhausted budget is terminal, and 429 invites the SDK to
    # back off and retry a request that can never succeed (measured live).
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "budget exhausted" in response.json()["error"]["message"]
    assert "budget" in response.json()["error"]["message"]
    assert token.blocked == 1

    tap = [r for r in read_journal(path) if r["type"] == GATEWAY_EVENT][-1]
    assert tap["status"] == "budget_exceeded"


def test_upstream_failure_becomes_502_not_a_crash(gateway):
    server, table, _journal, path = gateway
    table.add_route("dead-model", ModelRoute(base_url="http://127.0.0.1:1", api_key="x"))
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "dead-model", "messages": []})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    # the gateway is still serving
    assert httpx.get(server.base_url + "/health", timeout=10).status_code == 200


def test_env_for_gives_an_agent_everything_it_needs(gateway):
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    env = server.env_for(token)
    assert env["ANTHROPIC_BASE_URL"] == server.base_url
    assert env["ANTHROPIC_AUTH_TOKEN"] == token.token


def test_stats_endpoint_reports_per_slot_spend(gateway):
    server, table, _journal, _path = gateway
    token = table.mint("agent-1", budget_usd=5.0)
    post(server, token.token, {"model": "m", "messages": []})
    slots = httpx.get(server.base_url + "/_ash/stats", timeout=10).json()["slots"]
    entry = next(s for s in slots if s["agent_id"] == "agent-1")
    assert entry["requests"] == 1
    assert entry["input_tokens"] == 11
    assert entry["budget_usd"] == 5.0


# --- routing table ---------------------------------------------------------
def test_routing_table_from_file(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps(
            {
                "routes": {
                    "default": {"base_url": "https://api.anthropic.com",
                                "api_key_env": "SOME_KEY_VAR"},
                    "local": {"base_url": "http://127.0.0.1:8000",
                              "upstream_model": "ckpt"},
                }
            }
        )
    )
    table = RoutingTable.from_file(path)
    assert table.route_for("local").upstream_model == "ckpt"
    assert table.models() == ["local"]


def test_api_key_env_is_resolved_lazily(monkeypatch):
    route = ModelRoute(api_key_env="ASH_TEST_KEY")
    monkeypatch.delenv("ASH_TEST_KEY", raising=False)
    assert route.resolve_key() is None
    monkeypatch.setenv("ASH_TEST_KEY", "later-value")
    assert route.resolve_key() == "later-value"


def test_tokens_are_unguessable_and_distinct():
    table = RoutingTable()
    a = table.mint("agent-a")
    b = table.mint("agent-b")
    assert a.token != b.token
    assert len(a.token) > 24
    assert table.lookup(a.token).agent_id == "agent-a"
    assert table.lookup("Bearer " + b.token).agent_id == "agent-b"
    assert table.lookup(None) is None


# --- pricing: what makes budget_usd real -------------------------------------
def test_a_route_prices_usage_and_uncached_input_is_not_double_billed():
    """The provider reports token counts but no price; without a conversion the
    budget never binds. cache reads bill at their own rate and are a SUBSET of
    input_tokens, so they must come out of the input bucket."""
    from harness.core.events import Usage
    from harness.gateway.routing import ModelRoute

    route = ModelRoute(pricing={"input": 3.0, "output": 15.0,
                                "cache_read": 0.30, "cache_write": 3.75})
    usage = Usage(input_tokens=1000, output_tokens=100,
                  cached_input_tokens=400, cache_creation_tokens=200)
    # 600 uncached * 3 + 100 * 15 + 400 * 0.30 + 200 * 3.75 (per mtok)
    assert abs(route.price(usage) - 0.00417) < 1e-9
    assert ModelRoute().price(usage) == 0.0, "no pricing -> no invented cost"


def test_pricing_survives_the_routes_file():
    from harness.gateway.routing import ModelRoute

    route = ModelRoute.from_dict({"base_url": "http://up", "api_key": "k",
                                  "pricing": {"input": 3.0}})
    assert route.pricing == {"input": 3.0}


def test_an_unpriced_budget_says_so_once(tmp_path):
    """A budget over a route with no pricing can never bind -- spent stays 0
    forever. Found live: the 'enforced budget' was decorative on every real
    Anthropic upstream. The gateway must say so, once, rather than silently
    enforcing nothing."""
    import json
    import urllib.request

    from harness.core.journal import JournalWriter, read_journal
    from harness.gateway.routing import ModelRoute, RoutingTable
    from harness.gateway.server import GatewayServer

    upstream = _Upstream()           # anthropic-shaped, reports tokens, no cost
    table = RoutingTable()
    table.add_route("default", ModelRoute(base_url=upstream.base_url, api_key="k"))
    journal_path = tmp_path / "j.jsonl"
    with JournalWriter(journal_path, run_id="r", agent_id="a") as journal:
        gateway = GatewayServer(table, journal=journal, port=0).start()
        try:
            token = table.mint("agent-1", run_id="r", budget_usd=1.0)
            for _ in range(2):       # twice: the warning must not repeat
                body = json.dumps({"model": "m", "max_tokens": 8,
                                   "messages": [{"role": "user", "content": "hi"}]}).encode()
                req = urllib.request.Request(
                    gateway.base_url + "/v1/messages", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer %s" % token.token})
                urllib.request.urlopen(req, timeout=10).read()
        finally:
            gateway.stop()
            upstream.stop()
    warnings = [r for r in read_journal(journal_path)
                if r.get("status") == "budget_unenforceable"]
    assert len(warnings) == 1, "say it once; per-request is noise"
