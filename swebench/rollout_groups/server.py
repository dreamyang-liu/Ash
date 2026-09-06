"""Stdlib HTTP adapter for :class:`GroupRolloutService`."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .protocol import RolloutGroupRequest
from .runner import GroupRolloutService


def build_service(
    *,
    strategy: str = "agent-loop",
    image: str | None = None,
    backend: dict[str, Any] | None = None,
    miles_session_endpoint: str | None = None,
    model: str | None = None,
    allow_text_prompt: bool = True,
) -> GroupRolloutService:
    """Build a production rollout service from transport-level settings.

    The service owns orchestration, while the selected strategy owns branch
    behaviour.  ``agent-loop`` is the real integration path: it creates
    :class:`AshSession` instances through the configured backend (``microvm``
    reaches AgentENV) and drives Miles' v2 Session Server.  The environment
    provider is intentionally constructed once and reused by all worker jobs;
    each strategy invocation still creates and destroys its own sandbox.
    """
    normalized = strategy.strip().lower()
    if normalized == "agent-loop":
        if not image:
            raise ValueError("agent-loop service requires an image/template")
        if not miles_session_endpoint:
            raise ValueError("agent-loop service requires miles_session_endpoint")
        from .ash_environment import AshSessionEnvironmentProvider
        from .strategies.agent_loop import MilesSessionAgentRolloutStrategy
        from ..models import AgentConfig

        provider = AshSessionEnvironmentProvider(image=image, backend=backend)
        agent_config = AgentConfig(model=model or "openai/local")

        def factory(request, _context):
            # The endpoint in the request is authoritative when supplied by
            # Miles; the process-level value is the convenient service default.
            if request.session_server_endpoint is None:
                request = type(request)(
                    **{
                        **request.__dict__,
                        "session_server_endpoint": miles_session_endpoint,
                        "model": request.model or model,
                    }
                )
            return MilesSessionAgentRolloutStrategy(
                agent_config=agent_config,
                allow_text_prompt=allow_text_prompt,
            )

        return GroupRolloutService(factory, environment_provider=provider)

    if normalized == "sequential":
        from .strategies.sequential import SequentialRolloutStrategy

        return GroupRolloutService(
            lambda _request, _context: SequentialRolloutStrategy(allow_deterministic_fallback=False)
        )
    raise ValueError(f"unknown rollout strategy {strategy!r}; choose agent-loop or sequential")


class RolloutGroupsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, service: GroupRolloutService):
        super().__init__(address, RolloutGroupsRequestHandler)
        self.service = service


class RolloutGroupsRequestHandler(BaseHTTPRequestHandler):
    server: RolloutGroupsHTTPServer

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/rollout-groups":
            self._send({"error": "unknown path"}, 404)
            return
        try:
            request = RolloutGroupRequest.from_dict(self._read())
            submission = self.server.service.submit(request)
            self._send(submission.to_dict(), 202)
        except (ValueError, TypeError) as exc:
            self._send({"error": str(exc)}, 400)

    def do_GET(self) -> None:
        prefix = "/rollout-groups/"
        if not self.path.startswith(prefix) or not self.path[len(prefix):]:
            self._send({"error": "unknown path"}, 404)
            return
        try:
            self._send(self.server.service.get(self.path[len(prefix):]).to_dict(), 200)
        except KeyError:
            self._send({"error": "unknown rollout_job_id"}, 404)

    def do_DELETE(self) -> None:
        prefix = "/rollout-groups/"
        if not self.path.startswith(prefix) or not self.path[len(prefix):]:
            self._send({"error": "unknown path"}, 404)
            return
        try:
            self._send(self.server.service.cancel(self.path[len(prefix):]).to_dict(), 200)
        except KeyError:
            self._send({"error": "unknown rollout_job_id"}, 404)

    def _read(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _send(self, payload: dict[str, Any], status: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(service: GroupRolloutService, *, host: str = "0.0.0.0", port: int = 11001) -> None:
    """Run the service until interrupted."""
    server = RolloutGroupsHTTPServer((host, port), service)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ash rollout interface service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11001)
    parser.add_argument("--strategy", choices=["agent-loop", "sequential"], default="agent-loop")
    parser.add_argument("--image", default=os.environ.get("ASH_ROLLOUT_IMAGE"))
    parser.add_argument(
        "--backend-json",
        default=os.environ.get("ASH_ROLLOUT_BACKEND_JSON", "{}"),
        help="JSON object passed to AshSession, e.g. {'backend':'microvm', ...}",
    )
    parser.add_argument(
        "--miles-session-endpoint",
        default=os.environ.get("MILES_SESSION_SERVER_ENDPOINT"),
        help="Miles v2 Session Server base URL",
    )
    parser.add_argument("--model", default=os.environ.get("ASH_ROLLOUT_MODEL", "openai/local"))
    args = parser.parse_args()
    try:
        backend = json.loads(args.backend_json)
        if not isinstance(backend, dict):
            raise ValueError("--backend-json must decode to an object")
        service = build_service(
            strategy=args.strategy,
            image=args.image,
            backend=backend,
            miles_session_endpoint=args.miles_session_endpoint,
            model=args.model,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
