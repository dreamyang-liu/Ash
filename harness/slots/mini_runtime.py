"""Hooks around upstream DefaultAgent for Ash execution, durability and limits.

The upstream run/step/query loop and tool parser/observation templates are used
directly. Hooks persist messages before execution, close each real observation,
restore an exact prefix and keep the submission command's actual result.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import threading
import time
from uuid import uuid4

import httpx
import litellm
import yaml

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from harness.core.checkpoint_identity import CALL_IDENTITY_KEY
from harness.core.assistant_turn import validate_assistant_turn
from harness.core.mini_tools import mini_tool_schema
from harness.core.control import RunControl
from harness.core.events import Usage
from harness.core.http import post
from harness.core.journal import JournalWriter
from harness.core.slot import McpWiring, SlotResult, TaskSpec
from harness.gateway.server import _absorb_usage
from harness.normalize.claude_turns import TURN_BRANCH_POLICY
from harness.slots.mini_history import History, load_prefix


class ChatModel(LitellmModel):
    # HTTP transport errors end this attempt. Queue policy owns retries; a
    # gateway budget rejection must not start upstream's exponential retry loop.
    abort_exceptions = [Exception, KeyboardInterrupt]

    def __init__(self, task: TaskSpec, journal: JournalWriter, deadline: float,
                 control: RunControl, **config) -> None:
        super().__init__(model_name=task.model, **config)
        self.journal, self.deadline, self.control = journal, deadline, control
        env = {**os.environ, **task.env}
        base = env.get("OPENAI_BASE_URL", "")
        if not base:
            raise ValueError("Set OPENAI_BASE_URL or configure the Ash model gateway")
        self.url = base.rstrip("/") + ("/chat/completions" if base.rstrip("/").endswith("/v1")
                                       else "/v1/chat/completions")
        self.key = env.get("OPENAI_API_KEY", "")
        self.usage = Usage()
        self.tools = mini_tool_schema(BASH_TOOL)

    def _query(self, messages: list[dict], **kwargs) -> litellm.ModelResponse:
        self.control.raise_if_stopped()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
            self.control.raise_if_stopped()
        payload = {"model": self.config.model_name, "messages": messages, "tools": self.tools,
                   **self.config.model_kwargs, **kwargs, "stream": False}
        response = post(self.url, json=payload, timeout_s=remaining, control=self.control,
                        headers={"Authorization": "Bearer " + self.key})
        response.raise_for_status()
        raw = response.json()
        if (not isinstance(raw, dict) or len(raw.get("choices", [])) != 1
                or raw["choices"][0].get("finish_reason") not in {"stop", "tool_calls", "length"}):
            raise ValueError("mini model response is missing a valid terminal choice")
        message = raw["choices"][0].get("message", {})
        if message.get("role") != "assistant" or not isinstance(message.get("content") or "", str):
            raise ValueError("mini requires a text assistant completion")
        # Store the original provider response even if parsing fails.
        self.journal.emit("raw.mini-swe-agent", response=raw)
        usage = Usage()
        _absorb_usage(raw.get("usage"), usage)
        self.usage.add(usage)
        self.journal.emit("usage.updated", usage=usage.as_dict())
        return litellm.ModelResponse(**raw)

    def _calculate_cost(self, response: litellm.ModelResponse) -> dict:
        # The gateway's configured route owns authoritative pricing. Local
        # open-weight policies need no invented LiteLLM price registration.
        return {"cost": 0.0}

    def _parse_actions(self, response: litellm.ModelResponse) -> list[dict]:
        actions = super()._parse_actions(response)
        calls = response.choices[0].message.tool_calls
        if (len({call.id for call in calls}) != len(calls)
                or any(not isinstance(call.id, str) or not call.id
                       or set(json.loads(call.function.arguments)) != {"command"}
                       or not isinstance(action["command"], str)
                       for call, action in zip(calls, actions, strict=True))):
            raise ValueError("mini bash calls require unique ids and exactly one string command")
        return actions


class McpEnvironment:
    def __init__(self, wiring: McpWiring, journal: JournalWriter, deadline: float,
                 control: RunControl, config: dict) -> None:
        self.wiring, self.journal = wiring, journal
        self.deadline, self.control = deadline, control
        self.config = config
        self.headers = dict(wiring.headers)
        self.called = set()
        self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "ash-mini", "version": "1"}})
        tools = self.request("tools/list", {}).get("tools", [])
        shell = next((tool for tool in tools if tool["name"] == "shell"), None)
        if shell is None or not {"command", "working_dir", "timeout"} <= set(
                shell["inputSchema"].get("properties", {})):
            raise ValueError("mini requires Ash shell(command, working_dir, timeout)")

    def request(self, method: str, params: dict) -> dict:
        identifier = uuid4().hex
        response = httpx.post(
            self.wiring.url, json={"jsonrpc": "2.0", "id": identifier, "method": method, "params": params},
            headers={"Accept": "application/json, text/event-stream", **self.headers},
            timeout=max(1, self.deadline - time.monotonic()) + 90)
        response.raise_for_status()
        if response.headers.get("mcp-session-id"):
            self.headers["mcp-session-id"] = response.headers["mcp-session-id"]
        if "text/event-stream" in response.headers.get("content-type", ""):
            rows = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]
            body = next((row for row in rows if row.get("id") == identifier), {})
        else:
            body = response.json()
        if body.get("id") != identifier or "error" in body:
            raise ValueError("Invalid Ash MCP response")
        return body["result"]

    def execute(self, action: dict, cwd: str = "") -> dict:
        self.control.raise_if_stopped()
        call_id, command = action.get("tool_call_id"), action.get("command")
        if (not isinstance(call_id, str) or not call_id or call_id in self.called
                or not isinstance(command, str)):
            raise ValueError("Invalid or repeated mini bash action")
        self.called.add(call_id)
        env = self.config.get("env", {})
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in env):
            raise ValueError("Invalid sandbox environment variable name")
        prefix = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        args = {"command": f"env {prefix} bash -c {shlex.quote(command)}",
                "working_dir": cwd or self.config["cwd"],
                "timeout": max(1, min(self.config.get("timeout", 60),
                                      math.ceil(self.deadline - time.monotonic())))}
        started = self.journal.emit("tool.started", call_id=call_id, name="ash__shell", args=args)
        result = self.request("tools/call", {"name": "shell", "arguments": {
            **args, CALL_IDENTITY_KEY: {"call_id": call_id, "step": started["step"]}}})
        outcome = result.get("structuredContent", {}).get("command_outcome")
        content = result.get("content", [])
        if any(part.get("type") != "text" for part in content):
            raise ValueError("mini requires text shell output")
        text = "".join(part["text"] for part in content)
        self.journal.emit("tool.finished", call_id=call_id, name="ash__shell",
                          status="error" if result.get("isError") else "ok",
                          output=text, outcome=outcome)
        if not isinstance(outcome, dict) or type(outcome.get("exit_code")) is not int or outcome.get("running"):
            raise ValueError("Ash shell did not return a settled command outcome")
        # The generic MCP text may be JSON or include presentation annotations.
        # mini's completion sentinel and observation template consume streams.
        output = outcome.get("stdout", "") + outcome.get("stderr", "")
        return {"output": output, "returncode": outcome["exit_code"],
                "exception_info": "Command timed out" if outcome.get("timed_out") else "",
                "extra": {"timed_out": outcome.get("timed_out", False)}}

    def get_template_vars(self, **kwargs) -> dict:
        # No host environment or host uname enters the sandbox prompt.
        return {**self.config, "repository_dir": getattr(self, "repository_dir", None),
                "system": "Linux", "release": "", "version": "", "machine": "", **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config, "environment_type": "ash-http-mcp"}}}


class CheckpointAgent(DefaultAgent):
    def __init__(self, model: ChatModel, env: McpEnvironment, history: History,
                 journal: JournalWriter, control: RunControl, *, assistant_turn: dict | None = None,
                 resume_without_hint: bool = False,
                 **kwargs) -> None:
        super().__init__(model, env, **kwargs)
        self.history, self.journal, self.control = history, journal, control
        self.initialized = False
        self.turn_id = None
        self.turn_count = 0
        self.turn_source = "actor"
        if type(resume_without_hint) is not bool:
            raise ValueError("resume_without_hint must be a boolean")
        if assistant_turn is not None and resume_without_hint:
            raise ValueError("Point-only continuation cannot include an assistant_turn")
        if (assistant_turn is not None or resume_without_hint) and not history.prefix_messages:
            raise ValueError("This branch mode requires an exact mini history prefix")
        self.pending_turn = (validate_assistant_turn(assistant_turn, history=history.prefix_messages, tools=model.tools)
                             if assistant_turn is not None else None)
        self.omit_user_hint = assistant_turn is not None or resume_without_hint

    def add_messages(self, *messages: dict) -> list[dict]:
        if not self.initialized:
            self.initialized = True
            if self.history.prefix_messages:
                self.messages = deepcopy(self.history.prefix_messages)
                # The new mode continues the prefix directly, without a user hint.
                messages = (() if self.omit_user_hint else (
                    self.model.format_message(role="user", content=self.extra_template_vars["task"]),))
        for message in messages:
            response = message.get("extra", {}).get("response")
            if message.get("extra", {}).get("interrupt_type") == "FormatError" and isinstance(response, dict):
                # Upstream retains a malformed completion inside the feedback's
                # extra field. Preserve it as actual assistant context as well.
                original = deepcopy(response["choices"][0]["message"])
                super().add_messages(original)
                self.history.append({"type": "mini.message", "message": original})
                for call in original.get("tool_calls") or []:
                    rejection = {"role": "tool", "tool_call_id": call["id"],
                                 "content": "Action rejected before execution: " + message["content"]}
                    super().add_messages(rejection)
                    self.history.append({"type": "mini.message", "message": rejection})
            super().add_messages(message)
            self.history.append({"type": "mini.message", "message": message})
            if message.get("role") == "assistant":
                if message.get("content"):
                    self.journal.emit("agent.message", text=message["content"])
                if message.get("reasoning_content"):
                    self.journal.emit("agent.thinking", text=message["reasoning_content"])
        return list(messages)

    def query(self) -> dict:
        self.control.raise_if_stopped()
        self.turn_id = uuid4().hex
        self.turn_count += 1
        self.turn_source = "reviewer" if self.pending_turn is not None else "actor"
        self.journal.emit("turn.started", turn=self.turn_count, source=self.turn_source)
        if self.pending_turn is not None:
            message, self.pending_turn = self.pending_turn, None
            # Use mini's real parser; no model request, usage or fake observation.
            response = litellm.ModelResponse(choices=[{
                "index": 0, "finish_reason": "tool_calls", "message": message}])
            message["extra"] = {"actions": self.model._parse_actions(response), "source": "reviewer"}
            self.journal.emit("branch.assistant_turn", source="reviewer", turn_id=self.turn_id,
                              message={k: v for k, v in message.items() if k != "extra"})
            self.add_messages(message)
            return message
        return super().query()

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        # The model's entire response is already durable. Execute sequentially
        # and persist each real result, including COMPLETE_TASK_AND_SUBMIT.
        observed, submitted = [], None
        for action in actions:
            output = self.env.execute(action)
            one = {**message, "extra": {**message["extra"], "actions": [action]}}
            observations = self.model.format_observation_messages(one, [output], self.get_template_vars())
            observed.extend(self.add_messages(*observations))
            try:
                LocalEnvironment._check_finished(self.env, output)
            except Submitted as end:
                submitted = end
                break
        calls = [action["tool_call_id"] for action in actions]
        if len(observed) == len(actions):
            self.journal.emit("model.turn.output_completed", turn_id=self.turn_id,
                              call_ids=calls, closure="mini-response-and-observations", source=self.turn_source)
            self.journal.emit("model.turn.completed", turn_id=self.turn_id, call_ids=calls,
                              step=self.journal.tool_calls()[-1]["step"], source=self.turn_source)
            self.history.append({"type": "mini.turn", "turn_id": self.turn_id,
                                 "call_ids": calls, "source": self.turn_source})
            self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source)
        if submitted:
            if len(observed) != len(actions):
                raise ValueError("mini submitted before completing its model response's actions")
            raise submitted
        return observed


def run(task: TaskSpec, journal: JournalWriter, mcp: McpWiring,
        home: Path, session_id: str) -> SlotResult:
    config = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
    overrides = task.extra.get("mini", {})
    if not isinstance(overrides, dict) or set(overrides) - {"agent", "environment", "model"}:
        raise ValueError("mini config accepts agent/environment/model overrides")
    for section, value in overrides.items():
        config[section] = {**config.get(section, {}), **value}
    # Admission and pricing belong to Ash. No hidden upstream step/cost budget.
    agent_config = {**config["agent"], "step_limit": 0, "cost_limit": 0,
                    "wall_time_limit_seconds": 0, "output_path": home / f"{session_id}.traj.json"}
    for key in ("step_limit", "cost_limit", "wall_time_limit_seconds", "output_path"):
        if key in overrides.get("agent", {}):
            raise ValueError(f"Use Ash rollout controls instead of mini agent.{key}")
    env_config = {"cwd": None, "timeout": 60, **config.get("environment", {})}
    if set(env_config) - {"cwd", "timeout", "env"}:
        raise ValueError("mini environment only accepts cwd, timeout and env")
    if (not isinstance(env_config["cwd"], str) or not env_config["cwd"].startswith("/")
            or type(env_config["timeout"]) is not int or env_config["timeout"] <= 0
            or not isinstance(env_config.get("env", {}), dict)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in env_config.get("env", {}).items())):
        raise ValueError("mini requires a discovered or explicit absolute sandbox cwd, positive timeout and text env")
    workspace = task.extra.get("mini_workspace") or {"cwd": env_config["cwd"], "repository_dir": None}
    if workspace["cwd"] != env_config["cwd"]:
        raise ValueError("mini prompt workspace and execution working directory differ")
    agent_config["system_template"] += (
        "\n\n<workspace>\n"
        "Default working directory for every bash call: {{cwd}}\n"
        "{% if repository_dir %}Repository root: {{repository_dir}}\n{% endif %}"
        "Each call runs in a new shell. Changes made by cd or export do not persist across calls.\n"
        "</workspace>"
    )
    model_config = dict(config["model"])
    # This is an OpenAI-compatible wire adapter, not LiteLLM provider selection.
    model_config.pop("model_name", None)
    model_kwargs = model_config.pop("model_kwargs", {})
    model_kwargs.pop("drop_params", None)
    if set(model_kwargs) - {"temperature", "top_p", "top_k", "max_tokens", "stop", "parallel_tool_calls"}:
        raise ValueError("Unsupported mini model_kwargs; parameters must reach the model unchanged")
    model_config["model_kwargs"] = model_kwargs
    control = task.control or RunControl()
    deadline = time.monotonic() + task.timeout_s
    prefix = load_prefix(task.extra["native_prefix"]) if task.extra.get("native_prefix") else ()
    inherited = task.extra.get("native_prefix")
    if task.extra.get("resume_session_id") and not prefix:
        source = home / f"{task.extra['resume_session_id']}.jsonl"
        data, cut = b"", task.extra.get("resume_session_at")
        if not cut:
            raise ValueError("mini resume requires an exact native prefix or resume_session_at")
        for line in source.read_bytes().splitlines(keepends=True):
            data += line
            if json.loads(line).get("turn_id") == cut:
                break
        else:
            raise ValueError("mini native cut not found")
        inherited = {"path": str(source), "byte_length": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()}
        prefix = load_prefix(inherited)
    if prefix:
        parent = next(entry for entry in prefix if entry.get("type") == "mini.session")
        if parent.get("workspace") is not None and parent["workspace"] != workspace:
            raise ValueError("mini branch workspace differs from its inherited history")
    assistant_turn = task.extra.get("assistant_turn")
    resume_without_hint = task.extra.get("resume_without_hint", False)
    if type(resume_without_hint) is not bool:
        raise ValueError("resume_without_hint must be a boolean")
    if assistant_turn is not None and resume_without_hint:
        raise ValueError("Point-only continuation cannot include an assistant_turn")
    if (assistant_turn is not None or resume_without_hint) and not prefix:
        raise ValueError("This branch mode requires an exact mini history prefix")
    if assistant_turn is not None:
        validate_assistant_turn(assistant_turn, history=[
            entry["message"] for entry in prefix if entry.get("type") == "mini.message"])
    history = History(home / f"{session_id}.jsonl", session_id, prefix=prefix, workspace=workspace)
    model, agent = None, None
    status, error, final_text = "error", None, ""
    journal.emit("run.started", slot="mini-swe-agent", slot_version="2.4.6", model=task.model,
                 task_prompt=task.prompt, cwd=task.cwd)
    journal.emit("session.ref", native_session_id=session_id, transcript_path=str(history.path.resolve()))
    if inherited:
        # Make standalone branch indexing aware of the inherited call ids too.
        journal.emit("mini.restored", inherited_native={
            **inherited, "slot": "mini-swe-agent",
            "byte_length": inherited.get("byte_length", Path(inherited["path"]).stat().st_size)})
    journal.emit("branch.boundary.policy", policy=TURN_BRANCH_POLICY,
                 storage="per-tool", closure="mini-response-and-observations")
    timer = threading.Timer(task.timeout_s, lambda: control.request_stop(
        "mini wall-time budget exhausted", stop_reason="timeout"))
    timer.daemon = True
    timer.start()
    try:
        model = ChatModel(task, journal, deadline, control, **model_config)
        journal.emit("rollout.model_tools", shape="chat/completions", tools=model.tools)
        env = McpEnvironment(mcp, journal, deadline, control, env_config)
        env.repository_dir = workspace["repository_dir"]
        agent = CheckpointAgent(model, env, history, journal, control,
                                assistant_turn=assistant_turn, resume_without_hint=resume_without_hint,
                                **agent_config)
        info = agent.run(task.prompt)
        final_text = info.get("submission", "")
        status = "completed" if info.get("exit_status") == "Submitted" else "error"
        error = None if status == "completed" else info.get("exit_status", "mini did not submit")
    except (TimeoutError, KeyboardInterrupt) as exc:
        control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
        status, error = "timeout", str(exc) or control.reason
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
        history.close()
    usage = model.usage.as_dict() if model else {}
    journal.emit("run.result", text=final_text)
    journal.emit("run.finished", status=status, error=error, usage=usage)
    return SlotResult(status=status, final_text=final_text, usage=usage,
                      native_session_id=session_id, error=error)
