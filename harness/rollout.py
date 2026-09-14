"""Opt-in execution controls for externally scheduled rollouts.

The orchestrator consumes this contract; a queue/driver never runs the agent.
It contains no trainer or driver imports.
"""

from copy import deepcopy
import math
import threading
import time
from urllib.parse import quote, urlsplit

import httpx

from harness.execution.pipeline import Continue, Reject, ToolInterceptor


def endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("Rollout endpoint must be HTTP(S), without credentials/query/fragment")
    value = value.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def sampling_parameters(value: dict, shape: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("sampling_params must be an object")
    allowed = {"temperature", "top_p", "max_tokens", "max_new_tokens", "max_output_tokens"}
    if set(value) - allowed:
        raise ValueError("Unsupported native sampling controls: " + ", ".join(sorted(set(value) - allowed)))
    result = {}
    for key in ("temperature", "top_p"):
        if key in value:
            number = value[key]
            if (type(number) not in (int, float) or not math.isfinite(number) or number < 0
                    or (key == "top_p" and not 0 < number <= 1)):
                raise ValueError(f"Invalid sampling parameter {key}")
            result[key] = number
    lengths = [value[key] for key in ("max_tokens", "max_new_tokens", "max_output_tokens") if key in value]
    if lengths:
        if any(type(n) is not int or n <= 0 or n != lengths[0] for n in lengths):
            raise ValueError("Output token limits must be positive and agree")
        result["max_output_tokens" if shape == "responses" else "max_tokens"] = lengths[0]
    return result


def remaining_timeout(contract: dict, timeout_s: float) -> float:
    deadline = contract.get("deadline_at")
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("Rollout deadline must be finite")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ValueError("Rollout group wall-time budget expired before execution")
    return min(timeout_s, remaining)


class RolloutControls(ToolInterceptor):
    fail_mode = "closed"

    def __init__(self, contract: dict, journal, control):
        self.contract = deepcopy(contract)
        self.journal = journal
        self.control = control
        self.base_url = endpoint(contract["model_endpoint"])
        self.session_endpoint = (endpoint(contract["session_server_endpoint"])
                                 if contract.get("session_server_endpoint") else None)
        self.session_id = None
        self.model_calls = self.tool_calls = 0
        self.lock = threading.Lock()
        self.timer = None
        self.deadline = contract["deadline_at"]
        if type(self.deadline) not in (float, int) or not math.isfinite(self.deadline):
            raise ValueError("Rollout deadline must be finite")
        if contract.get("message_export"):
            from runstore.message_sampling import validate

            if {"max_model_calls", "max_tool_calls"} & contract.keys():
                raise ValueError("Message rollouts use per-trajectory max_turns, not model/tool call budgets")
            max_turns = contract.get("max_turns")
            if type(max_turns) is not int or max_turns <= 0:
                raise ValueError("Invalid max_turns")
            self._call_limits = {"model": max_turns, "tool": None}
            validate(contract.get("sampling_params", {}))
        else:
            for key, minimum in (("max_model_calls", 1), ("max_tool_calls", 0)):
                if type(contract[key]) is not int or contract[key] < minimum:
                    raise ValueError(f"Invalid {key}")
            if "max_turns" in contract:
                raise ValueError("max_turns requires the message rollout contract")
            self._call_limits = {"model": contract["max_model_calls"], "tool": contract["max_tool_calls"]}
            sampling_parameters(contract.get("sampling_params", {}), "responses")

    def start(self):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise ValueError("Rollout group wall-time budget expired before execution")
        self.timer = threading.Timer(remaining, lambda: self.control.request_stop(
            "rollout wall-time budget exhausted", stop_reason="timeout"))
        self.timer.daemon = True
        self.timer.start()
        if self.session_endpoint:
            response = httpx.post(self.session_endpoint + "/sessions", json={}, timeout=min(30, remaining))
            response.raise_for_status()
            self.session_id = response.json().get("session_id")
            if not isinstance(self.session_id, str) or not self.session_id:
                raise ValueError("Miles session server returned no session_id")
            self.base_url = self.session_endpoint + "/sessions/" + quote(self.session_id, safe="")
            self.journal.emit("rollout.session", session_id=self.session_id, endpoint=self.session_endpoint)

    def _reserve(self, kind: str):
        with self.lock:
            used = getattr(self, kind + "_calls")
            limit = self._call_limits[kind]
            reason = None
            stop_reason = None
            if self.control.reason or time.time() >= self.deadline:
                reason = "rollout wall-time budget exhausted or execution stopped"
                if not self.control.reason:
                    stop_reason = "timeout"
            elif limit is not None and used >= limit:
                reason = ("rollout turn budget exhausted" if self.contract.get("message_export")
                          else f"rollout {kind}-call budget exhausted")
                if self.contract.get("message_export"):
                    stop_reason = "max_turns_reached"
            if reason is None:
                setattr(self, kind + "_calls", used + 1)
        if reason:
            self.control.request_stop(reason, stop_reason=stop_reason)
            raise ValueError(reason)

    def prepare_model_request(self, payload: dict, shape: str) -> dict:
        self._reserve("model")
        if self.contract.get("message_export"):
            from runstore.message_sampling import parameters

            self.journal.emit("rollout.model_tools", shape=shape, tools=payload.get("tools", []))
            sampling = parameters(self.contract.get("sampling_params", {}), shape)
        else:
            sampling = sampling_parameters(self.contract.get("sampling_params", {}), shape)
        payload = {**payload, **sampling}
        if self.contract.get("model"):
            payload["model"] = self.contract["model"]
        return payload

    def before(self, ctx):
        try:
            self._reserve("tool")
            return Continue()
        except ValueError as error:
            return Reject(str(error))

    def finish(self):
        if self.timer:
            self.timer.cancel()
        self.journal.emit("rollout.usage", model_calls=self.model_calls, tool_calls=self.tool_calls)
        if not self.session_id:
            return
        url = self.session_endpoint + "/sessions/" + quote(self.session_id, safe="")
        try:
            response = httpx.get(url, timeout=30)
            response.raise_for_status()
            state = response.json()
            if not isinstance(state, dict):
                raise ValueError("Miles session state is not an object")
            self.journal.emit("rollout.session_state", session_id=self.session_id, state=state)
        except Exception as error:
            self.journal.emit("rollout.export_unavailable", error=str(error))
        finally:
            try:
                response = httpx.delete(url, timeout=30)
                response.raise_for_status()
            except Exception as error:
                self.journal.emit("rollout.session_release_failed", session_id=self.session_id, error=str(error))
