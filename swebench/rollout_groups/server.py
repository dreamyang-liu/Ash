"""Stdlib HTTP adapter for :class:`GroupRolloutService`."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .protocol import RolloutGroupRequest
from .runner import GroupRolloutService


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
    args = parser.parse_args()
    raise SystemExit("No rollout strategy configured; embed GroupRolloutService in an Ash application")


if __name__ == "__main__":
    main()
