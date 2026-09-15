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

    _TOKEN_PROGRESS_FIELDS = (
        "trajectory_tokens",
        "assistant_generated_tokens",
        "current_context_tokens",
        "peak_context_tokens",
        "last_model_output_tokens",
    )

    def __init__(self, contract: dict, journal, control, on_progress=None):
        self.contract = deepcopy(contract)
        self.journal = journal
        self.control = control
        self.on_progress = on_progress or (lambda payload: None)
        self.base_url = endpoint(contract["model_endpoint"])
        self.session_endpoint = (endpoint(contract["session_server_endpoint"])
                                 if contract.get("session_server_endpoint") else None)
        self.session_id = None
        self.model_calls = self.tool_calls = 0
        self._observed_output_tokens = 0
        self._peak_context_tokens = 0
        self._session_server_instance_id = None
        self._parent_anchor_pending = contract.get("model_parent_position") is not None
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
            if max_turns is not None and (type(max_turns) is not int or max_turns <= 0):
                raise ValueError("Invalid max_turns")
            self._call_limits = {"model": max_turns, "tool": None}
            validate(contract.get("sampling_params", {}))
        else:
            for key, minimum in (("max_model_calls", 1), ("max_tool_calls", 0)):
                value = contract[key]
                if value is not None and (type(value) is not int or value < minimum):
                    raise ValueError(f"Invalid {key}")
            if "max_turns" in contract:
                raise ValueError("max_turns requires the message rollout contract")
            self._call_limits = {"model": contract["max_model_calls"], "tool": contract["max_tool_calls"]}
            sampling_parameters(contract.get("sampling_params", {}), "responses")
        configured_session = contract.get("session_id")
        if configured_session is not None and (
            not isinstance(configured_session, str) or not configured_session
        ):
            raise ValueError("rollout session_id must be a nonempty string")
        if contract.get("retain_session") not in {None, True, False}:
            raise ValueError("retain_session must be boolean")
        parent = contract.get("model_parent_position")
        if parent is not None and (
            not isinstance(parent, dict)
            or parent.get("session_id") != configured_session
            or not isinstance(parent.get("session_server_instance_id"), str)
            or not isinstance(parent.get("response_id"), str)
            or not isinstance(parent.get("token_sha256"), str)
        ):
            raise ValueError("model_parent_position does not match rollout session_id")

    def _publish_progress(self, phase: str, **payload) -> None:
        try:
            self.on_progress({
                "phase": phase,
                "model_calls": self.model_calls,
                "tool_calls": self.tool_calls,
                **payload,
            })
        except Exception:
            # Progress is observational. A broken monitor must not alter the
            # rollout or its training trajectory.
            pass

    def _session_progress(self) -> dict:
        if not self.session_id or not self.session_endpoint:
            return {}
        url = self.session_endpoint + "/sessions/" + quote(self.session_id, safe="") + "/progress"
        try:
            remaining = max(0.1, self.deadline - time.time())
            response = httpx.get(url, timeout=min(30, remaining))
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                return {}
            progress = {name: value.get(name) for name in self._TOKEN_PROGRESS_FIELDS}
            if any(type(item) is not int or item < 0 for item in progress.values()):
                return {}
            return progress
        except Exception:
            return {}

    def _model_position(self, response_id: str) -> dict:
        if not self.session_id or not self.session_endpoint or not response_id:
            return {}
        url = (
            self.session_endpoint
            + "/sessions/"
            + quote(self.session_id, safe="")
            + "/positions/"
            + quote(response_id, safe="")
        )
        remaining = max(0.1, self.deadline - time.time())
        response = httpx.get(url, timeout=min(30, remaining))
        response.raise_for_status()
        value = response.json()
        required = {
            "session_id",
            "session_server_instance_id",
            "node_id",
            "response_id",
            "path_node_ids",
            "token_count",
            "token_sha256",
        }
        if (
            not isinstance(value, dict)
            or not required <= set(value)
            or value["session_id"] != self.session_id
            or value["session_server_instance_id"]
            != self._session_server_instance_id
            or value["response_id"] != response_id
            or type(value["node_id"]) is not int
            or type(value["token_count"]) is not int
            or not isinstance(value["path_node_ids"], list)
            or not isinstance(value["token_sha256"], str)
        ):
            raise ValueError("Miles returned an invalid model position")
        return value

    def upstream_headers(self) -> dict[str, str]:
        with self.lock:
            if not self._parent_anchor_pending:
                return {}
            position = self.contract["model_parent_position"]
            return {
                "X-Miles-Expected-Session-Server-Instance-ID": position[
                    "session_server_instance_id"
                ],
                "X-Miles-Expected-Parent-Response-ID": position["response_id"],
                "X-Miles-Expected-Parent-Token-SHA256": position["token_sha256"],
            }

    def start(self):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise ValueError("Rollout group wall-time budget expired before execution")
        self.timer = threading.Timer(remaining, lambda: self.control.request_stop(
            "rollout wall-time budget exhausted", stop_reason="timeout"))
        self.timer.daemon = True
        self.timer.start()
        if self.session_endpoint:
            if self.contract.get("retain_session"):
                response = httpx.get(
                    self.session_endpoint + "/health", timeout=min(30, remaining)
                )
                response.raise_for_status()
                health = response.json()
                capabilities = health.get("capabilities") or []
                required = {"session-tree-v2", "session-position-v1"}
                if not required <= set(capabilities):
                    raise ValueError(
                        "Miles shared session is missing capabilities: "
                        + ", ".join(sorted(required - set(capabilities)))
                    )
                self._session_server_instance_id = health.get(
                    "session_server_instance_id"
                )
                if not isinstance(self._session_server_instance_id, str) or not self._session_server_instance_id:
                    raise ValueError("Miles shared session has no stable server instance ID")
                parent = self.contract.get("model_parent_position")
                if (
                    parent is not None
                    and parent["session_server_instance_id"]
                    != self._session_server_instance_id
                ):
                    raise ValueError(
                        "Miles session server restarted after the recovery point was recorded"
                    )
            requested_session_id = self.contract.get("session_id")
            response = httpx.post(
                self.session_endpoint + "/sessions",
                json=(
                    {"session_id": requested_session_id}
                    if requested_session_id is not None
                    else {}
                ),
                timeout=min(30, remaining),
            )
            response.raise_for_status()
            self.session_id = response.json().get("session_id")
            if not isinstance(self.session_id, str) or not self.session_id:
                raise ValueError("Miles session server returned no session_id")
            self.base_url = self.session_endpoint + "/sessions/" + quote(self.session_id, safe="")
            self.journal.emit("rollout.session", session_id=self.session_id, endpoint=self.session_endpoint)
        self._publish_progress("model_ready")

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
        self._publish_progress("model_generation" if kind == "model" else "tool_execution")

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

    def model_response_completed(self, usage=None, *, response_id=None) -> None:
        """Publish one post-response snapshot from the authoritative session.

        The gateway invokes this only after it has relayed a complete response.
        Querying Miles here observes a committed SessionTree node and therefore
        avoids retokenizing text in Ash.
        """
        with self.lock:
            self._parent_anchor_pending = False
        progress = self._session_progress()
        if not progress and usage is not None:
            # Message-only endpoints do not expose SessionTree state. Provider
            # usage is still useful for coarse monitoring, but it is never used
            # as training-token metadata.
            output = int(getattr(usage, "output_tokens", 0) or 0)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            self._observed_output_tokens += output
            self._peak_context_tokens = max(self._peak_context_tokens, input_tokens)
            progress = {
                "trajectory_tokens": input_tokens + output,
                "assistant_generated_tokens": self._observed_output_tokens,
                "current_context_tokens": input_tokens,
                "peak_context_tokens": self._peak_context_tokens,
                "last_model_output_tokens": output,
            }
        self._publish_progress("model_response", **progress)
        if response_id and self.session_endpoint:
            try:
                position = self._model_position(response_id)
                self.journal.emit(
                    "rollout.model_response",
                    response_id=response_id,
                    model_position=position,
                )
            except Exception as error:
                self.journal.emit(
                    "rollout.model_position_unavailable",
                    response_id=response_id,
                    error=str(error),
                )
        elif self.session_endpoint:
            self.journal.emit(
                "rollout.model_position_unavailable",
                response_id=response_id,
                error="completed Miles response has no response_id",
            )

    def finish(self):
        if self.timer:
            self.timer.cancel()
        self.journal.emit("rollout.usage", model_calls=self.model_calls, tool_calls=self.tool_calls)
        self._publish_progress("finalizing")
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
            if not self.contract.get("retain_session"):
                try:
                    response = httpx.delete(url, timeout=30)
                    response.raise_for_status()
                except Exception as error:
                    self.journal.emit(
                        "rollout.session_release_failed",
                        session_id=self.session_id,
                        error=str(error),
                    )
