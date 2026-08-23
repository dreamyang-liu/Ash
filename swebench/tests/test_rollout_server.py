"""Unit tests for the miles rollout server (swebench.rollout_server).

No docker, no model calls, no dataset downloads — everything runs against
fakes injected through ``EpisodeDeps``:

- request -> reply contract shape (the miles swe_agent_function contract)
- reward = FAIL_TO_PASS pass fraction; ASH_REWARD_MODE=binary semantics
- empty patch / missing tests -> reward 0 without running any test command
- failures (missing/unknown instance, sandbox, agent crash) -> structured
  zero-reward reply, session always destroyed
- AgentConfig wiring from the request body
- HTTP layer (/health, /run, 404) on a real ThreadingHTTPServer with fakes
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

import pytest

from swebench.models import AgentConfig, CostTracker, ToolResult
from swebench.rollout_server import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STEP_LIMIT,
    EpisodeDeps,
    RolloutHTTPServer,
    reward_mode_from_env,
    run_episode,
)

INSTANCE_ID = "astropy__astropy-12907"
F2P_TESTS = ["test_a", "test_b", "test_c", "test_d"]
INSTANCE = {
    "instance_id": INSTANCE_ID,
    "repo": "astropy/astropy",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "problem_statement": "Modeling's separability matrix is wrong.",
    "FAIL_TO_PASS": json.dumps(F2P_TESTS),
}
PATCH = (
    "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n"
    "--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n"
    "@@ -1 +1 @@\n-old\n+new\n"
)

REPLY_KEYS = {"reward", "exit_status", "eval_report", "agent_metrics",
              "environment"}
EVAL_KEYS = {"f2p_total", "f2p_passed", "patch_chars"}
METRIC_KEYS = {"turns", "tool_calls", "agent_run_time", "eval_time",
               "total_time", "time_per_turn"}


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #

class FakeSession:
    """In-memory AshSession stand-in; ``passing`` marks test ids that pass."""

    def __init__(self, patch: str = PATCH, create_ok: bool = True,
                 passing: tuple[str, ...] = ()):
        self.patch = patch
        self.create_ok = create_ok
        self.passing = passing
        self.commands: list[str] = []
        self.callers: list[str] = []  # identity each call was attributed to
        self.created_with: Optional[str] = None
        self.destroyed = False

    def create(self, image: str) -> bool:
        self.created_with = image
        return self.create_ok

    def environment(self) -> dict:
        """Mirrors AshSession: what the episode ran against. A mutable image
        name is not identity, so the resolved reference is reported too."""
        return {
            "requested_image": self.created_with or "",
            "base_ref": f"registry/{self.created_with}@sha256:feed" if self.created_with else "",
            "base_commit": "abc123",
            "sandbox_id": "fake-sandbox",
        }

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        return self._run(tool_name, args, "harness")

    def executor_for(self, agent_id: str):
        """Mirrors AshSession: an executor with one agent's identity bound."""
        def run(tool_name: str, args: dict) -> ToolResult:
            return self._run(tool_name, args, agent_id)
        return run

    def _run(self, tool_name: str, args: dict, agent_id: str) -> ToolResult:
        command = args.get("command", "")
        self.commands.append(command)
        self.callers.append(agent_id)
        passed = any(test_id in command for test_id in self.passing)
        return ToolResult(success=passed, output="")

    def get_patch(self) -> str:
        return self.patch

    def destroy(self) -> None:
        self.destroyed = True


class FakeAgent:
    """Scripted agent: makes ``tool_calls`` executor calls, then exits."""

    def __init__(self, executor: Callable[[str, dict], Any],
                 exit_status: str = "completed", turns: int = 5,
                 tool_calls: int = 0):
        self._executor = executor
        self._exit_status = exit_status
        self._tool_calls = tool_calls
        self.cost = CostTracker()
        self.cost.api_calls = turns
        self.task: Optional[str] = None
        self.instance_id: Optional[str] = None

    def run(self, task: str, instance_id: str = "") -> str:
        self.task = task
        self.instance_id = instance_id
        for n in range(self._tool_calls):
            self._executor("shell", {"command": f"agent-cmd-{n}"})
        return self._exit_status


def make_deps(session: FakeSession, instance: dict = INSTANCE,
              reward_mode: str = "fraction", agent_kwargs: Optional[dict] = None,
              seen_configs: Optional[list] = None,
              seen_agents: Optional[list] = None,
              seen_subsets: Optional[list] = None) -> EpisodeDeps:
    instances = {instance["instance_id"]: instance}

    def get_instance(instance_id: str, subset: str) -> dict:
        if seen_subsets is not None:
            seen_subsets.append(subset)
        return instances[instance_id]  # KeyError for unknown ids

    def make_agent(config: AgentConfig, executor: Callable,
                   session=None) -> FakeAgent:
        if seen_configs is not None:
            seen_configs.append(config)
        agent = FakeAgent(executor, **(agent_kwargs or {}))
        if seen_agents is not None:
            seen_agents.append(agent)
        return agent

    return EpisodeDeps(get_instance=get_instance, make_session=lambda: session,
                       make_agent=make_agent, reward_mode=reward_mode)


def make_request(**overrides: Any) -> dict:
    request = {
        "instance_id": INSTANCE_ID,
        "base_url": "http://127.0.0.1:30001/v1",
        "model": "openai/model",
        "sampling_params": {"temperature": 0.8, "max_tokens": 4096},
    }
    request.update(overrides)
    return request


# --------------------------------------------------------------------------- #
#  Contract shape
# --------------------------------------------------------------------------- #

def test_reply_matches_miles_contract_shape():
    session = FakeSession(passing=tuple(F2P_TESTS))
    reply = run_episode(make_request(), make_deps(session))

    assert set(reply) == REPLY_KEYS
    assert set(reply["eval_report"]) == EVAL_KEYS
    assert set(reply["agent_metrics"]) == METRIC_KEYS
    assert isinstance(reply["reward"], float)
    assert reply["exit_status"] == "completed"


def test_task_prompt_and_instance_id_reach_the_agent():
    session = FakeSession()
    agents: list[FakeAgent] = []
    run_episode(make_request(), make_deps(session, seen_agents=agents))

    (agent,) = agents
    assert INSTANCE["problem_statement"] in agent.task  # format_task_prompt
    assert INSTANCE["repo"] in agent.task
    assert agent.instance_id == INSTANCE_ID
    assert session.created_with  # image resolved from the instance
    assert session.destroyed


# --------------------------------------------------------------------------- #
#  Reward math
# --------------------------------------------------------------------------- #

def test_reward_is_fraction_of_passing_fail_to_pass_tests():
    session = FakeSession(passing=("test_a", "test_b", "test_c"))
    reply = run_episode(make_request(), make_deps(session))

    assert reply["reward"] == pytest.approx(0.75)
    assert reply["eval_report"] == {"f2p_total": 4, "f2p_passed": 3,
                                    "patch_chars": len(PATCH)}
    # one shell invocation per FAIL_TO_PASS test
    assert len(session.commands) == len(F2P_TESTS)


def test_reward_is_one_when_all_tests_pass():
    session = FakeSession(passing=tuple(F2P_TESTS))
    reply = run_episode(make_request(), make_deps(session))
    assert reply["reward"] == pytest.approx(1.0)


def test_binary_mode_is_all_or_nothing():
    partial = FakeSession(passing=("test_a", "test_b", "test_c"))
    reply = run_episode(make_request(), make_deps(partial, reward_mode="binary"))
    assert reply["reward"] == 0.0
    assert reply["eval_report"]["f2p_passed"] == 3

    full = FakeSession(passing=tuple(F2P_TESTS))
    reply = run_episode(make_request(), make_deps(full, reward_mode="binary"))
    assert reply["reward"] == 1.0


def test_empty_patch_scores_zero_without_running_tests():
    session = FakeSession(patch="")
    reply = run_episode(make_request(), make_deps(session))

    assert reply["reward"] == 0.0
    assert reply["exit_status"] == "completed"  # agent status is preserved
    assert reply["eval_report"] == {"f2p_total": 4, "f2p_passed": 0,
                                    "patch_chars": 0}
    assert session.commands == []  # no test commands were executed


def test_no_fail_to_pass_tests_scores_zero():
    instance = {**INSTANCE, "FAIL_TO_PASS": "[]"}
    session = FakeSession()
    reply = run_episode(make_request(), make_deps(session, instance=instance))
    assert reply["reward"] == 0.0
    assert reply["eval_report"]["f2p_total"] == 0
    assert session.commands == []


def test_reward_mode_from_env(monkeypatch):
    monkeypatch.delenv("ASH_REWARD_MODE", raising=False)
    assert reward_mode_from_env() == "fraction"
    monkeypatch.setenv("ASH_REWARD_MODE", "binary")
    assert reward_mode_from_env() == "binary"
    monkeypatch.setenv("ASH_REWARD_MODE", "bogus")
    with pytest.raises(ValueError):
        reward_mode_from_env()


# --------------------------------------------------------------------------- #
#  Failure paths -> structured zero-reward replies
# --------------------------------------------------------------------------- #

def _assert_failure_reply(reply: dict):
    assert set(reply) == REPLY_KEYS
    assert reply["reward"] == 0.0
    assert reply["exit_status"].startswith("error")
    assert reply["eval_report"] == {}
    assert "total_time" in reply["agent_metrics"]


def test_missing_instance_id_returns_zero_reward():
    reply = run_episode(make_request(instance_id=""), make_deps(FakeSession()))
    _assert_failure_reply(reply)
    assert "instance_id" in reply["exit_status"]


def test_unknown_instance_returns_zero_reward():
    reply = run_episode(make_request(instance_id="nope__nope-1"),
                        make_deps(FakeSession()))
    _assert_failure_reply(reply)


def test_sandbox_creation_failure_returns_zero_reward_and_destroys():
    session = FakeSession(create_ok=False)
    reply = run_episode(make_request(), make_deps(session))
    _assert_failure_reply(reply)
    assert "sandbox creation failed" in reply["exit_status"]
    assert session.destroyed


def test_agent_crash_returns_zero_reward_and_destroys_session():
    session = FakeSession()

    def exploding_make_agent(config: AgentConfig, executor: Callable,
                             session=None):
        raise RuntimeError("boom")

    deps = EpisodeDeps(get_instance=lambda i, s: INSTANCE,
                       make_session=lambda: session,
                       make_agent=exploding_make_agent)
    reply = run_episode(make_request(), deps)
    _assert_failure_reply(reply)
    assert "boom" in reply["exit_status"]
    assert session.destroyed


def test_session_destroyed_on_success():
    session = FakeSession(passing=tuple(F2P_TESTS))
    run_episode(make_request(), make_deps(session))
    assert session.destroyed


# --------------------------------------------------------------------------- #
#  Request -> AgentConfig wiring and metrics
# --------------------------------------------------------------------------- #

def test_agent_config_built_from_request():
    configs: list[AgentConfig] = []
    run_episode(make_request(), make_deps(FakeSession(), seen_configs=configs))

    (config,) = configs
    assert config.model == "openai/model"
    assert config.api_base == "http://127.0.0.1:30001/v1"
    assert config.temperature == pytest.approx(0.8)
    assert config.max_tokens == 4096
    assert config.step_limit == DEFAULT_STEP_LIMIT
    assert config.prompt_cache is False
    assert config.cost_limit >= 1e9  # local policy: cost cap out of the way


def test_agent_config_defaults_without_sampling_params():
    configs: list[AgentConfig] = []
    run_episode(make_request(sampling_params={}),
                make_deps(FakeSession(), seen_configs=configs))
    (config,) = configs
    assert config.max_tokens == DEFAULT_MAX_TOKENS
    assert config.temperature is None


def test_subset_defaults_and_request_override():
    subsets: list[str] = []
    run_episode(make_request(), make_deps(FakeSession(), seen_subsets=subsets))
    run_episode(make_request(subset="lite"),
                make_deps(FakeSession(), seen_subsets=subsets))
    assert subsets == ["verified", "lite"]


def test_metrics_report_turns_and_tool_calls():
    session = FakeSession(patch="")
    deps = make_deps(session, agent_kwargs={"turns": 9, "tool_calls": 3})
    reply = run_episode(make_request(), deps)

    metrics = reply["agent_metrics"]
    assert metrics["turns"] == 9
    assert metrics["tool_calls"] == 3
    assert metrics["total_time"] >= metrics["agent_run_time"] >= 0.0
    # the 3 scripted agent commands went through the counting executor
    assert session.commands == ["agent-cmd-0", "agent-cmd-1", "agent-cmd-2"]


# --------------------------------------------------------------------------- #
#  HTTP layer (real ThreadingHTTPServer on an ephemeral port, fake episode deps)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def http_server():
    deps = make_deps(FakeSession(passing=tuple(F2P_TESTS)))
    server = RolloutHTTPServer(("127.0.0.1", 0), deps, lambda: 42)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_health_endpoint(http_server):
    status, payload = _get(f"{http_server}/health")
    assert status == 200
    assert payload == {"status": "ok", "instances": 42}


def test_run_endpoint_returns_contract_reply(http_server):
    body = json.dumps(make_request()).encode("utf-8")
    status, payload = _post(f"{http_server}/run", body)
    assert status == 200
    assert set(payload) == REPLY_KEYS
    assert payload["reward"] == pytest.approx(1.0)


def test_run_endpoint_tolerates_malformed_json(http_server):
    status, payload = _post(f"{http_server}/run", b"{not json")
    assert status == 200
    assert payload["reward"] == 0.0
    assert payload["exit_status"].startswith("error")


def test_unknown_paths_return_404(http_server):
    status, _ = _get(f"{http_server}/nope")
    assert status == 404
    status, _ = _post(f"{http_server}/nope", b"{}")
    assert status == 404
