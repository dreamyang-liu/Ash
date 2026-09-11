"""Stdlib HTTP adapter for :class:`GroupRolloutService`."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .environment_catalog import EnvironmentCatalog
from .environment_resolver import AgentEnvOCIResolver, AgentEnvOCIResolverConfig
from .protocol import RolloutGroupRequest
from .runner import EndpointModelClient, GroupRolloutService


def build_service(
    *,
    strategy: str = "agent-loop",
    environment_catalog: EnvironmentCatalog | None = None,
    oci_resolver: AgentEnvOCIResolver | None = None,
    backend: dict[str, Any] | None = None,
    miles_session_endpoint: str | None = None,
    model: str | None = None,
    allow_text_prompt: bool = True,
    result_ttl_seconds: float = 300.0,
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
        if environment_catalog is None and oci_resolver is None:
            raise ValueError("agent-loop service requires an environment catalog or OCI resolver")
        if not miles_session_endpoint:
            raise ValueError("agent-loop service requires miles_session_endpoint")
        from .ash_environment import AshSessionEnvironmentProvider
        from .strategies.agent_loop import MilesSessionAgentRolloutStrategy
        from ..models import AgentConfig

        provider = AshSessionEnvironmentProvider(
            catalog=environment_catalog, oci_resolver=oci_resolver, backend=backend
        )
        agent_config = AgentConfig(model=model or "openai/local")

        def factory(_request, _context):
            return MilesSessionAgentRolloutStrategy(
                agent_config=agent_config,
                session_server_endpoint=miles_session_endpoint,
                allow_text_prompt=allow_text_prompt,
            )

        return GroupRolloutService(
            factory,
            environment_provider=provider,
            result_ttl_seconds=result_ttl_seconds,
        )

    if normalized == "checkpoint-agent-loop-v1":
        if environment_catalog is None and oci_resolver is None:
            raise ValueError(
                "checkpoint-agent-loop-v1 service requires an environment catalog or OCI resolver"
            )
        if not miles_session_endpoint:
            raise ValueError("checkpoint-agent-loop-v1 service requires miles_session_endpoint")
        from .ash_environment import AshSessionEnvironmentProvider
        from .strategies.checkpoint_agent_loop import CheckpointAgentLoopRolloutStrategy
        from ..models import AgentConfig

        provider = AshSessionEnvironmentProvider(
            catalog=environment_catalog, oci_resolver=oci_resolver, backend=backend
        )
        agent_config = AgentConfig(model=model or "openai/local")

        def factory(_request, _context):
            return CheckpointAgentLoopRolloutStrategy(
                agent_config=agent_config,
                session_server_endpoint=miles_session_endpoint,
                allow_text_prompt=allow_text_prompt,
            )

        return GroupRolloutService(
            factory,
            environment_provider=provider,
            result_ttl_seconds=result_ttl_seconds,
        )

    if normalized == "sequential":
        from .strategies.sequential import SequentialRolloutStrategy

        return GroupRolloutService(
            lambda _request, _context: SequentialRolloutStrategy(allow_deterministic_fallback=False),
            model_client=EndpointModelClient(),
            result_ttl_seconds=result_ttl_seconds,
        )
    raise ValueError(
        f"unknown rollout strategy {strategy!r}; choose agent-loop, checkpoint-agent-loop-v1, or sequential"
    )


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
        if self.path.rstrip("/") == "/rollout-environments":
            self._send(self.server.service.list_environments(), 200)
            return
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
            self._send(self.server.service.delete(self.path[len(prefix):]).to_dict(), 200)
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
    parser.add_argument(
        "--strategy", choices=["agent-loop", "checkpoint-agent-loop-v1", "sequential"], default="agent-loop"
    )
    parser.add_argument(
        "--environment-catalog",
        default=os.environ.get("ASH_ROLLOUT_ENVIRONMENT_CATALOG"),
        help="JSON catalog mapping trusted environment refs to native Ash spawn refs",
    )
    parser.add_argument(
        "--agentenv-oci-resolver-config",
        default=os.environ.get("ASH_AGENTENV_OCI_RESOLVER_CONFIG"),
        help=(
            "optional JSON policy for resolving digest-pinned OCI images into "
            "runtime-ready AgentENV snapshots"
        ),
    )
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
    parser.add_argument(
        "--result-ttl-seconds",
        type=float,
        default=float(os.environ.get("ASH_ROLLOUT_RESULT_TTL_SECONDS", "300")),
        help="Fallback retention for terminal results that Miles does not delete",
    )
    args = parser.parse_args()
    try:
        backend = json.loads(args.backend_json)
        if not isinstance(backend, dict):
            raise ValueError("--backend-json must decode to an object")
        environment_catalog = (
            EnvironmentCatalog.from_file(args.environment_catalog)
            if args.environment_catalog
            else None
        )
        oci_resolver = None
        if args.agentenv_oci_resolver_config:
            microvm = backend.get("microvm") or {}
            if not isinstance(microvm, dict):
                raise ValueError("backend.microvm must be an object")
            aenv_server_url = microvm.get("server_url") or os.environ.get(
                "AENV_SERVER_URL"
            )
            aenv_api_key = microvm.get("api_key") or os.environ.get("AENV_API_KEY")
            aenv_api_key_file = microvm.get("api_key_file")
            if aenv_api_key:
                aenv_api_key_file = None
            oci_resolver = AgentEnvOCIResolver(
                AgentEnvOCIResolverConfig.from_file(
                    args.agentenv_oci_resolver_config
                ),
                aenv_server_url=aenv_server_url,
                aenv_api_key=aenv_api_key,
                aenv_api_key_file=aenv_api_key_file,
            )
        service = build_service(
            strategy=args.strategy,
            environment_catalog=environment_catalog,
            oci_resolver=oci_resolver,
            backend=backend,
            miles_session_endpoint=args.miles_session_endpoint,
            model=args.model,
            result_ttl_seconds=args.result_ttl_seconds,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
