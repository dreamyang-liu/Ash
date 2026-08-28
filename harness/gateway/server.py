"""The gateway HTTP server.

Anthropic Messages API shape only (``POST /v1/messages`` + ``GET /v1/models``),
which is all any Claude-speaking agent needs. Requests are forwarded verbatim
except for the two things we own: the credential (agent's slot token -> the
route's real key) and, optionally, the model name.

Implementation notes that are not obvious:

- **Headers pass through untouched** apart from hop-by-hop and auth. Agents send
  ``anthropic-beta`` and session-correlation headers
  (``x-claude-code-session-id``, ``x-claude-code-agent-id``,
  ``x-claude-code-parent-agent-id``); dropping them changes behaviour or loses
  per-subagent attribution. Rewriting response bytes is likewise avoided --
  unknown fields (thinking-block signatures) must survive byte-exact or the next
  request is rejected upstream.
- **Streaming is a passthrough with a parser tap.** SSE frames are relayed as
  they arrive (so the agent sees no added latency) while a side parser
  accumulates ``message_start`` / ``message_delta`` usage for the journal. We do
  not buffer whole responses, and we do not modify frames -- verdict-style
  rewriting on the model stream is not this layer's job.
- **Budget is checked before forwarding**, refused with a NON-RETRYABLE 400 and
  a JSON error the agent can read. Enforcement here is real, unlike asking an
  agent to stop; and it must be terminal -- a 429 reads as "retry later" to
  every SDK, and an agent then burns wall-clock retrying the unwinnable.
- ``GET /v1/models`` must answer: Claude Code probes it for model discovery and
  treats failure (including any redirect) as fatal.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from harness.core.events import Usage
from harness.gateway.routing import RoutingTable

#: Never forwarded upstream (hop-by-hop or replaced by us).
_STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "authorization",
    "x-api-key",
    "accept-encoding",  # we relay bytes; let httpx negotiate identity
}

_STRIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
}

GATEWAY_EVENT = "gateway.request"   # wire-level tap event type


class GatewayServer:
    """Owns the routing table, the journal tap and the HTTP server."""

    def __init__(
        self,
        table: RoutingTable,
        *,
        journal=None,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout_s: float = 900.0,
        require_token: bool = True,
    ) -> None:
        self.table = table
        self.journal = journal
        self.timeout_s = timeout_s
        self.require_token = require_token
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None
        self._warned_unpriced = False

    # --- lifecycle ---------------------------------------------------------
    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return "http://%s:%s" % (host, port)

    def start(self) -> "GatewayServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=10)

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def env_for(self, token) -> dict:
        """Environment an agent needs to talk to this gateway."""
        return {
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_AUTH_TOKEN": token.token,
            "ANTHROPIC_API_KEY": token.token,
        }

    # --- tap ---------------------------------------------------------------
    def record(self, **payload) -> None:
        if self.journal is not None:
            self.journal.emit(GATEWAY_EVENT, **payload)


def _make_handler(gateway: GatewayServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ash-gateway/1"

        # --- plumbing ---------------------------------------------------
        def log_message(self, *args) -> None:  # noqa: A003 - silence stderr spam
            pass

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
            self._json(status, {"type": "error", "error": {"type": kind, "message": message}})

        def _authenticate(self):
            presented = self.headers.get("Authorization") or self.headers.get("x-api-key")
            token = gateway.table.lookup(presented)
            if token is None and gateway.require_token:
                return None
            return token

        # --- routes -----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.startswith("/v1/models"):
                # Claude Code probes this for discovery; failure is fatal there.
                self._json(
                    200,
                    {
                        "data": [
                            {"type": "model", "id": name, "display_name": name}
                            for name in gateway.table.models()
                        ],
                        "has_more": False,
                    },
                )
                return
            if self.path == "/_ash/stats":
                self._json(200, {"slots": gateway.table.stats()})
                return
            if self.path in ("/health", "/_ash/health"):
                self._json(200, {"status": "ok"})
                return
            self._error(404, "unknown path %s" % self.path)

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/v1/messages"):
                self._error(404, "unknown path %s" % self.path)
                return

            token = self._authenticate()
            if token is None and gateway.require_token:
                self._error(401, "unknown or missing slot token", "authentication_error")
                return

            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                self._error(400, "body is not valid JSON")
                return

            if token is not None and token.over_budget():
                token.blocked += 1
                gateway.record(
                    agent_id=token.agent_id,
                    status="budget_exceeded",
                    spent_usd=round(token.spent_usd, 6),
                    budget_usd=token.budget_usd,
                )
                # 400/invalid_request, NOT 429/rate_limit: an exhausted budget
                # is terminal, and 429 tells every SDK "back off and retry" --
                # measured live, the agent then burned minutes retrying a
                # request that could never succeed. A non-retryable error fails
                # the run NOW, which is what a kill switch is for.
                self._error(
                    400,
                    "run budget exhausted: spent $%.4f of $%.4f -- this will "
                    "not succeed on retry"
                    % (token.spent_usd, token.budget_usd),
                    "invalid_request_error",
                )
                return

            requested_model = payload.get("model")
            route = gateway.table.route_for(requested_model)
            if route.upstream_model:
                payload["model"] = route.upstream_model
            body = json.dumps(payload).encode()
            streaming = bool(payload.get("stream"))

            if token is not None:
                token.requests += 1

            try:
                self._forward(route, body, streaming, token, requested_model)
            except Exception as exc:  # noqa: BLE001 - report, never crash the server
                gateway.record(
                    agent_id=getattr(token, "agent_id", None),
                    status="upstream_error",
                    model=requested_model,
                    error="%s: %s" % (type(exc).__name__, exc),
                )
                try:
                    self._error(502, "upstream failure: %s" % exc, "api_error")
                except Exception:  # pragma: no cover - client already gone
                    pass

        # --- forwarding -------------------------------------------------
        def _upstream_headers(self, route) -> dict:
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _STRIP_REQUEST_HEADERS
            }
            headers.update(route.headers)
            key = route.resolve_key()
            if key:
                headers["x-api-key"] = key
            headers.setdefault("anthropic-version", "2023-06-01")
            headers["Content-Type"] = "application/json"
            return headers

        def _forward(self, route, body: bytes, streaming: bool, token, requested_model) -> None:
            import httpx

            url = route.base_url.rstrip("/") + "/v1/messages"
            headers = self._upstream_headers(route)

            with httpx.Client(timeout=gateway.timeout_s) as client:
                if not streaming:
                    upstream = client.post(url, content=body, headers=headers)
                    self._relay_head(upstream.status_code, upstream.headers, len(upstream.content))
                    self.wfile.write(upstream.content)
                    usage, model = _usage_from_response(upstream)
                    self._tap(token, requested_model, model, route, upstream.status_code, usage, False)
                    return

                with client.stream("POST", url, content=body, headers=headers) as upstream:
                    self._relay_head(upstream.status_code, upstream.headers, None)
                    usage = Usage()
                    model = None
                    for chunk in upstream.iter_raw():
                        if not chunk:
                            continue
                        # Chunked framing: relay immediately, no buffering.
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                        found = _scan_sse(chunk, usage)
                        model = model or found
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    self._tap(token, requested_model, model, route, upstream.status_code, usage, True)

        def _relay_head(self, status: int, headers, content_length: Optional[int]) -> None:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() in _STRIP_RESPONSE_HEADERS:
                    continue
                self.send_header(key, value)
            if content_length is None:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(content_length))
            self.end_headers()

        def _tap(self, token, requested_model, upstream_model, route, status, usage, streaming):
            cost = usage.cost_usd or route.price(usage)
            if token is not None:
                gateway.table.charge(
                    token,
                    cost_usd=cost,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                if (token.budget_usd is not None and cost == 0.0
                        and (usage.input_tokens or usage.output_tokens)
                        and not gateway._warned_unpriced):
                    # A budget over a route with no pricing can never bind: the
                    # provider reports tokens, nobody converts them to dollars,
                    # spent stays 0 forever. Found live -- the "enforced budget"
                    # was decorative on every real Anthropic upstream. Once per
                    # gateway, because once is a diagnosis and per-request is
                    # noise.
                    gateway._warned_unpriced = True
                    gateway.record(
                        status="budget_unenforceable",
                        agent_id=getattr(token, "agent_id", None),
                        base_url=route.base_url,
                        reason="route has no pricing; budget_usd cannot bind "
                               "-- add pricing to the route",
                    )
            gateway.record(
                agent_id=getattr(token, "agent_id", None),
                status="ok" if status < 400 else "upstream_%d" % status,
                requested_model=requested_model,
                upstream_model=upstream_model or route.upstream_model or requested_model,
                base_url=route.base_url,
                streaming=streaming,
                usage=dict(usage.as_dict(), cost_usd=round(cost, 6)),
                spent_usd=round(token.spent_usd, 6) if token else None,
            )

    return Handler


# --- usage extraction ------------------------------------------------------
def _usage_from_response(response) -> Tuple[Usage, Optional[str]]:
    usage = Usage()
    model = None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - non-JSON error body
        return usage, model
    if isinstance(payload, dict):
        model = payload.get("model")
        _absorb_usage(payload.get("usage"), usage)
    return usage, model


def _absorb_usage(native, usage: Usage) -> None:
    if not isinstance(native, dict):
        return
    usage.input_tokens += int(native.get("input_tokens") or 0)
    usage.output_tokens += int(native.get("output_tokens") or 0)
    usage.cached_input_tokens += int(native.get("cache_read_input_tokens") or 0)
    usage.cache_creation_tokens += int(native.get("cache_creation_input_tokens") or 0)


def _scan_sse(chunk: bytes, usage: Usage) -> Optional[str]:
    """Accumulate usage from relayed SSE bytes. Best-effort by design.

    A chunk can split a frame; a partial JSON line is simply skipped rather than
    reassembled, because this tap must never affect what the agent receives.
    ``message_start`` carries input/cache counts, ``message_delta`` the output
    total, so the important numbers arrive in single small frames.
    """
    model = None
    for line in chunk.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        blob = line[5:].strip()
        if not blob or blob == b"[DONE]":
            continue
        try:
            event = json.loads(blob)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message") or {}
            model = message.get("model") or model
            _absorb_usage(message.get("usage"), usage)
        elif etype == "message_delta":
            native = event.get("usage") or {}
            # output_tokens in message_delta is cumulative for the message.
            output = int(native.get("output_tokens") or 0)
            if output:
                usage.output_tokens = max(usage.output_tokens, output)
    return model


def serve(
    table: Optional[RoutingTable] = None,
    *,
    journal=None,
    host: str = "127.0.0.1",
    port: int = 8787,
    require_token: bool = True,
) -> GatewayServer:
    """Start a gateway in a background thread."""
    return GatewayServer(
        table or RoutingTable(),
        journal=journal,
        host=host,
        port=port,
        require_token=require_token,
    ).start()
