"""HTTP rollout server bridging miles RL training to Ash SWE-bench episodes.

miles (radixark/miles, a slime fork: SGLang rollout + Megatron training) drives
SWE-bench RL through a task-agnostic agent function
(``examples/experimental/swe-agent-v2/swe_agent_function.py``) that POSTs one
JSON request per episode to ``${AGENT_SERVER_URL}/run`` and merges the JSON
reply into the sample's metadata (reward is read back by ``--custom-rm-path``).
This module is the Ash side of that contract.

Request (``POST /run``) — the sample's metadata spread into the body, plus:

- ``instance_id``      SWE-bench instance to run (required)
- ``subset``           dataset subset, default from ``--subset`` (``verified``)
- ``base_url``         OpenAI-compatible endpoint of the policy being trained
                       (the miles session server fronting SGLang), ends in /v1
- ``model``            litellm model name, e.g. ``openai/model``
- ``sampling_params``  ``{"temperature": ..., "max_tokens": ...}`` (extra keys ignored)
- ``max_seq_len``      accepted but unused — miles truncates trainer-side

Reply — always HTTP 200 with a structured body; any failure becomes a
zero-reward reply (kinder to training than a transport error, which miles
turns into ``None``):

- ``reward``         fraction of FAIL_TO_PASS tests passing after the episode
                     (``ASH_REWARD_MODE=binary``: 1.0 only if all pass)
- ``exit_status``    agent loop status (``completed`` | ``step_limit`` |
                     ``cost_limit`` | ``error``) or ``"error: ..."``
- ``eval_report``    ``{"f2p_total", "f2p_passed", "patch_chars"}``
- ``agent_metrics``  ``{"turns", "tool_calls", "agent_run_time", "eval_time",
                     "total_time", "time_per_turn"}`` — keys miles aggregates

Each request runs on its own thread (``ThreadingHTTPServer``) with its own
sandbox (``AshSession`` → docker) and its own agent loop, mirroring the
thread-per-rollout pattern of the best-of-n harness. Stdlib HTTP only.

Run::

    python -m swebench.rollout_server --port 11000 --subset verified
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from .agent import AshAgent, TOOLS_SCHEMA
from .dataset import (
    format_task_prompt,
    image_registry_for_subset,
    load_instances,
    resolve_image,
)
from .harnesses.best_of_n import build_test_command, parse_test_list
from .models import AgentConfig
from .sandbox import AshSession

logger = logging.getLogger("swebench.rollout_server")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 11000
DEFAULT_SUBSET = "verified"
DEFAULT_STEP_LIMIT = 100
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MODEL = "openai/model"
# The policy endpoint is a local SGLang server behind the miles session server:
# requests are free, and litellm cannot price unknown model names anyway
# (CostTracker already swallows that). Keep the cost cap out of the way.
UNLIMITED_COST = 1e9
# Local OpenAI-compatible servers ignore the key, but litellm wants one set.
DEFAULT_API_KEY = "unused"

REWARD_FRACTION = "fraction"
REWARD_BINARY = "binary"

REWARD_MODE_ENV = "ASH_REWARD_MODE"
RUNTIME_BIN_ENV = "ASH_RUNTIME_BIN"
STEP_LIMIT_ENV = "ASH_STEP_LIMIT"


# --------------------------------------------------------------------------- #
#  Episode logic (dependency-injected; unit-tested without docker or models)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeDeps:
    """Injectable collaborators for :func:`run_episode` (fakes in unit tests).

    ``get_instance(instance_id, subset)`` returns a SWE-bench instance dict
    (raising on unknown ids); ``make_session()`` returns an AshSession-like
    object (create/execute/get_patch/destroy); ``make_agent(config, executor)``
    returns an AshAgent-like object (run/cost).
    """

    get_instance: Callable[[str, str], dict]
    make_session: Callable[[], Any]
    make_agent: Callable[[AgentConfig, Callable[[str, dict], Any]], Any]
    subset: str = DEFAULT_SUBSET
    step_limit: int = DEFAULT_STEP_LIMIT
    cost_limit: float = UNLIMITED_COST
    max_tokens: int = DEFAULT_MAX_TOKENS
    reward_mode: str = REWARD_FRACTION


class CountingExecutor:
    """Wraps a session executor, counting agent tool invocations for metrics."""

    def __init__(self, execute: Callable[[str, dict], Any]):
        self._execute = execute
        self.calls = 0

    def __call__(self, tool_name: str, args: dict) -> Any:
        self.calls += 1
        return self._execute(tool_name, args)


def run_episode(request: dict[str, Any], deps: EpisodeDeps) -> dict[str, Any]:
    """Run one graded SWE-bench episode. Never raises — every failure becomes
    a structured zero-reward reply."""
    started_at = time.monotonic()
    instance_id = str(request.get("instance_id") or "")
    if not instance_id:
        return _failure_reply("error: request missing instance_id", started_at)

    session: Any = None
    try:
        subset = str(request.get("subset") or deps.subset)
        instance = deps.get_instance(instance_id, subset)
        image = resolve_image(instance, registry=image_registry_for_subset(subset))
        session = deps.make_session()
        if not session.create(image):
            return _failure_reply(f"error: sandbox creation failed for {image}", started_at)
        return _run_and_grade(request, instance, session, deps, started_at)
    except Exception as exc:  # noqa: BLE001 — contract: reply, don't raise
        logger.exception("episode %s failed", instance_id)
        return _failure_reply(f"error: {exc}", started_at)
    finally:
        if session is not None:
            _destroy_quietly(session)


def _run_and_grade(request: dict[str, Any], instance: dict, session: Any,
                   deps: EpisodeDeps, started_at: float) -> dict[str, Any]:
    """Agent loop, then patch extraction + FAIL_TO_PASS grading in-sandbox."""
    config = _agent_config(request, deps)
    executor = CountingExecutor(session.execute)
    agent = deps.make_agent(config, executor)

    agent_started = time.monotonic()
    exit_status = agent.run(format_task_prompt(instance),
                            instance_id=str(instance.get("instance_id", "")))
    agent_run_time = time.monotonic() - agent_started

    eval_started = time.monotonic()
    patch = session.get_patch()
    reward, passed, total = _grade_patch(session, instance, patch, deps.reward_mode)
    eval_time = time.monotonic() - eval_started

    turns = agent.cost.api_calls
    total_time = time.monotonic() - started_at
    return {
        "reward": reward,
        "exit_status": exit_status,
        "eval_report": {
            "f2p_total": total,
            "f2p_passed": passed,
            "patch_chars": len(patch),
        },
        "agent_metrics": {
            "turns": turns,
            "tool_calls": executor.calls,
            "agent_run_time": round(agent_run_time, 3),
            "eval_time": round(eval_time, 3),
            "total_time": round(total_time, 3),
            "time_per_turn": round(agent_run_time / turns, 3) if turns else 0.0,
        },
    }


def _grade_patch(session: Any, instance: dict, patch: str,
                 reward_mode: str) -> tuple[float, int, int]:
    """Reward = FAIL_TO_PASS pass fraction (binary mode: all-or-nothing).

    An empty patch or an instance without parseable FAIL_TO_PASS tests scores
    0.0 without running anything.
    """
    tests = parse_test_list(instance.get("FAIL_TO_PASS"))
    if not patch.strip() or not tests:
        return 0.0, 0, len(tests)

    repo = str(instance.get("repo", ""))
    passed = sum(
        1 for test_id in tests
        if session.execute("shell", {"command": build_test_command(repo, test_id),
                                     "working_dir": "/testbed"}).success
    )
    if reward_mode == REWARD_BINARY:
        return (1.0 if passed == len(tests) else 0.0), passed, len(tests)
    return passed / len(tests), passed, len(tests)


def _agent_config(request: dict[str, Any], deps: EpisodeDeps) -> AgentConfig:
    """Map the miles request onto the standard agent loop configuration."""
    sampling = request.get("sampling_params") or {}
    raw_temperature = sampling.get("temperature")
    max_tokens = sampling.get("max_tokens") or sampling.get("max_new_tokens")
    return AgentConfig(
        model=str(request.get("model") or DEFAULT_MODEL),
        api_base=request.get("base_url"),
        api_key=str(request.get("api_key") or DEFAULT_API_KEY),
        max_tokens=int(max_tokens or deps.max_tokens),
        step_limit=deps.step_limit,
        cost_limit=deps.cost_limit,
        temperature=float(raw_temperature) if raw_temperature is not None else None,
        prompt_cache=False,  # OpenAI-compatible SGLang endpoint: no Anthropic caching
    )


def _failure_reply(status: str, started_at: float) -> dict[str, Any]:
    """Structured zero-reward reply — miles treats it as a graded failure."""
    return {
        "reward": 0.0,
        "exit_status": status,
        "eval_report": {},
        "agent_metrics": {"total_time": round(time.monotonic() - started_at, 3)},
    }


def _destroy_quietly(session: Any) -> None:
    try:
        session.destroy()
    except Exception:  # noqa: BLE001 — cleanup must never mask the reply
        logger.warning("session destroy failed", exc_info=True)


def reward_mode_from_env() -> str:
    """Read ``ASH_REWARD_MODE`` (``fraction`` default, or ``binary``)."""
    mode = os.environ.get(REWARD_MODE_ENV, "").strip().lower() or REWARD_FRACTION
    if mode not in (REWARD_FRACTION, REWARD_BINARY):
        raise ValueError(
            f"{REWARD_MODE_ENV} must be {REWARD_FRACTION!r} or {REWARD_BINARY!r}, got {mode!r}"
        )
    return mode


# --------------------------------------------------------------------------- #
#  Dataset access
# --------------------------------------------------------------------------- #

class InstanceStore:
    """Thread-safe, lazily-loaded ``id -> instance`` maps, one per subset."""

    def __init__(self, loader: Callable[..., list[dict]] = load_instances):
        self._loader = loader
        self._lock = threading.Lock()
        self._tables: dict[str, dict[str, dict]] = {}

    def _table(self, subset: str) -> dict[str, dict]:
        with self._lock:
            if subset not in self._tables:
                logger.info("loading SWE-bench subset %r ...", subset)
                instances = self._loader(subset=subset)
                self._tables[subset] = {i["instance_id"]: i for i in instances}
                logger.info("loaded %d instances for subset %r",
                            len(self._tables[subset]), subset)
            return self._tables[subset]

    def get(self, instance_id: str, subset: str) -> dict:
        table = self._table(subset)
        if instance_id not in table:
            raise KeyError(f"unknown instance_id {instance_id!r} in subset {subset!r}")
        return table[instance_id]

    def count(self, subset: str) -> int:
        return len(self._table(subset))


# --------------------------------------------------------------------------- #
#  HTTP layer
# --------------------------------------------------------------------------- #

class RolloutHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server carrying the episode dependencies."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], deps: EpisodeDeps,
                 instance_count: Callable[[], int]):
        super().__init__(address, RolloutRequestHandler)
        self.deps = deps
        self.instance_count = instance_count


class RolloutRequestHandler(BaseHTTPRequestHandler):
    """``GET /health`` and ``POST /run`` (see module docstring for the contract)."""

    server: RolloutHTTPServer  # narrowed for type checkers

    def log_message(self, fmt: str, *args: Any) -> None:  # route BaseHTTP noise
        logger.debug("%s %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._send_json({"error": f"unknown path {self.path}"}, status=404)
            return
        try:
            count = self.server.instance_count()
        except Exception as exc:  # noqa: BLE001 — dataset load can fail
            logger.exception("health check failed")
            self._send_json({"status": "error", "error": str(exc)}, status=503)
            return
        self._send_json({"status": "ok", "instances": count})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/run":
            self._send_json({"error": f"unknown path {self.path}"}, status=404)
            return
        request = self._read_json_body()
        reply = run_episode(request, self.server.deps)  # never raises
        _log_episode(request, reply)
        self._send_json(reply)

    def _read_json_body(self) -> dict[str, Any]:
        """Parse the request body; malformed input degrades to ``{}`` so the
        episode path returns a structured zero-reward reply."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _log_episode(request: dict[str, Any], reply: dict[str, Any]) -> None:
    """One line per episode: instance, reward, turns, seconds."""
    metrics = reply.get("agent_metrics") or {}
    logger.info(
        "episode instance=%s reward=%.3f exit=%s turns=%s tool_calls=%s time=%.1fs",
        request.get("instance_id"), reply.get("reward", 0.0),
        reply.get("exit_status", ""), metrics.get("turns", 0),
        metrics.get("tool_calls", 0), metrics.get("total_time", 0.0),
    )


# --------------------------------------------------------------------------- #
#  Wiring + CLI
# --------------------------------------------------------------------------- #

def build_default_deps(subset: str, runtime_bin: Optional[str], step_limit: int,
                       reward_mode: str, store: InstanceStore) -> EpisodeDeps:
    """Production dependencies: real dataset, docker sandboxes, agent loop."""

    def make_session() -> AshSession:
        return AshSession(runtime_bin=runtime_bin, quiet=True)

    def make_agent(config: AgentConfig,
                   executor: Callable[[str, dict], Any]) -> AshAgent:
        agent = AshAgent(config, executor=executor)
        agent.stream = False
        agent.set_tools_schema(TOOLS_SCHEMA)
        return agent

    return EpisodeDeps(
        get_instance=store.get,
        make_session=make_session,
        make_agent=make_agent,
        subset=subset,
        step_limit=step_limit,
        reward_mode=reward_mode,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swebench.rollout_server",
        description="Ash rollout server for miles RL training (SWE-bench episodes).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--subset", default=DEFAULT_SUBSET,
                        help=f"default SWE-bench subset (default: {DEFAULT_SUBSET})")
    parser.add_argument("--runtime-bin", default=os.environ.get(RUNTIME_BIN_ENV),
                        help=f"ash-runtime binary for sandboxes (default: ${RUNTIME_BIN_ENV})")
    parser.add_argument("--step-limit", type=int,
                        default=int(os.environ.get(STEP_LIMIT_ENV, DEFAULT_STEP_LIMIT)),
                        help=f"agent steps per episode (default: ${STEP_LIMIT_ENV} "
                             f"or {DEFAULT_STEP_LIMIT})")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        reward_mode = reward_mode_from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    store = InstanceStore()
    deps = build_default_deps(subset=args.subset, runtime_bin=args.runtime_bin,
                              step_limit=args.step_limit, reward_mode=reward_mode,
                              store=store)
    server = RolloutHTTPServer((args.host, args.port), deps,
                               lambda: store.count(args.subset))
    logger.info("Ash rollout server on %s:%d (subset=%s step_limit=%d reward_mode=%s)",
                args.host, args.port, args.subset, args.step_limit, reward_mode)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted — shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
