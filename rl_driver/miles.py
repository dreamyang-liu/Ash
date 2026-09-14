"""Original Miles wire contract translated to the existing execution queue."""

from copy import deepcopy
import hashlib
import math
import time
from urllib.parse import urlsplit

from rl_driver.environment_catalog import EnvironmentCatalog
from rl_driver.ledger import Conflict, canonical
from rl_driver.protocol import PROTOCOL_VERSION, RolloutGroupRequest, RolloutGroupResult, Trajectory


def internal_id(value: str) -> str:
    return "miles-" + hashlib.sha256(value.encode()).hexdigest()


def validate_endpoint(value: str) -> None:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("Model/session endpoint must be HTTP(S) without credentials/query/fragment")


def validate_sampling(value: dict) -> None:
    allowed = {"temperature", "top_p", "max_new_tokens", "max_tokens", "max_output_tokens"}
    if set(value) - allowed:
        raise ValueError("Unsupported native sampling controls: " + ", ".join(sorted(set(value) - allowed)))
    for key in ("temperature", "top_p"):
        if key in value:
            number = value[key]
            if (type(number) not in (int, float) or not math.isfinite(number) or number < 0
                    or (key == "top_p" and not 0 < number <= 1)):
                raise ValueError(f"Invalid sampling parameter {key}")
    lengths = [value[key] for key in ("max_tokens", "max_new_tokens", "max_output_tokens") if key in value]
    if lengths and any(type(n) is not int or n <= 0 or n != lengths[0] for n in lengths):
        raise ValueError("Output token limits must be positive and agree")


def session_trajectory(request: RolloutGroupRequest, sample, state: dict, *, status: str) -> Trajectory:
    """Export observed tokens only. Never substitute an expected weight version."""
    if not isinstance(state, dict) or not isinstance(state.get("metadata"), dict):
        raise ValueError("Malformed Miles session state/metadata")
    records = state.get("records")
    tokens = state["metadata"].get("accumulated_token_ids")
    if not isinstance(records, list) or not records or not isinstance(tokens, list) or not tokens:
        raise ValueError("Missing Miles session records/accumulated_token_ids")
    spans = []
    messages = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed Miles session record")
        query = record.get("request") or {}
        envelope = record.get("response") or {}
        if not isinstance(query, dict) or not isinstance(envelope, dict):
            raise ValueError("Malformed session request/response")
        choices = envelope.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Session record has no response choice")
        response = choices[0]
        info = response.get("meta_info") or {}
        if not isinstance(info, dict):
            raise ValueError("Malformed generation metadata")
        pairs = info.get("output_token_logprobs")
        inputs = query.get("input_ids")
        version = info.get("weight_version")
        if type(version) not in (str, int) or str(version) in {"", "unknown"}:
            raise ValueError("Missing observed weight_version in session record")
        if request.expected_weight_version is not None and str(version) != request.expected_weight_version:
            raise ValueError("Observed weight_version differs from expected_weight_version")
        if not isinstance(inputs, list) or not inputs or not isinstance(pairs, list) or not pairs:
            raise ValueError("Session record lacks exact input/output token data")
        if any(not isinstance(pair, list) or len(pair) < 2 for pair in pairs):
            raise ValueError("Invalid output_token_logprobs record")
        output = [pair[1] for pair in pairs]
        spans.append({"response_id": envelope.get("id"), "start": len(inputs),
                      "end": len(inputs) + len(output), "input_token_ids": inputs,
                      "output_token_ids": output, "weight_version": str(version),
                      "finish_reason": response.get("finish_reason") or "stop",
                      "output_token_log_probs": [pair[0] for pair in pairs] if request.return_rollout_logprobs else None})
        messages = list(query.get("messages") or []) + [response.get("message") or {}]
    return Trajectory.from_dict({
        "sample_slot_id": sample.sample_slot_id,
        "branch_id": f"{request.rollout_job_id}:root:{sample.sample_index}",
        "messages": messages, "token_ids": tokens, "prompt_length": spans[0]["start"],
        "generated_spans": spans, "response_text": str(messages[-1].get("content") or ""),
        "status": status,
    })


class MilesAdapter:
    def __init__(self, driver, config: dict):
        self.driver = driver
        self.config = deepcopy(config)
        if not isinstance(config, dict) or set(config) - {
            "environment_catalog", "profile", "run_defaults", "resources", "tasks", "api_key_env", "max_samples",
            "image_resources"}:
            raise ValueError("Invalid Miles adapter config")
        self.catalog = (EnvironmentCatalog.from_dict(config["environment_catalog"])
                        if config.get("environment_catalog") else None)
        self.defaults = deepcopy(config.get("run_defaults", {}))
        if self.defaults.get("slot", "codex") not in {"codex", "claude-code"}:
            raise ValueError("Miles RunSpec adapter supports codex and claude-code")
        if "rollout_contract" in self.defaults.get("extra", {}):
            raise ValueError("rollout_contract is derived from each Miles request")

    def environments(self):
        return {"protocol_version": PROTOCOL_VERSION,
                "environments": [ref.to_dict() for ref in self.catalog.list_refs()] if self.catalog else []}

    def submit(self, body: dict) -> dict:
        if body.get("protocol_version") == "ash-rollout-v3":
            from rl_driver.messages import MessageAdapter

            return MessageAdapter(self.driver, self.config).submit(body)
        request = RolloutGroupRequest.from_dict(body)
        normalized = request.to_dict()
        identifier = internal_id(request.rollout_job_id)
        try:
            previous = self.driver.ledger.get(identifier)
        except KeyError:
            previous = None
        if previous is not None:
            if canonical(previous["request"]) != canonical(normalized):
                raise Conflict("rollout_job_id already names a different Miles request")
        else:
            plan = self._plan(request)
            self.driver.submit(plan, source_request=normalized,
                               extra_document={"miles_request": normalized,
                                               "deadline_at": plan["context"]["deadline_at"]})
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("protocol_version", "rollout_job_id", "status")}

    def _plan(self, request: RolloutGroupRequest) -> dict:
        if self.catalog is None:
            raise ValueError("v2 requires an environment catalog; v3 delegates images to Ash")
        entry = self.catalog.resolve(request.environment_ref)
        if request.max_samples > self.config.get("max_samples", 1000):
            raise ValueError("Requested group exceeds deployment max_samples")
        if request.budgets.max_model_calls < request.max_samples:
            raise ValueError("This strategy needs at least one model call per allocated sample")
        validate_endpoint(request.model_endpoint)
        if request.session_server_endpoint:
            validate_endpoint(request.session_server_endpoint)
        sampling = deepcopy(request.sampling_params)
        sampling_model = sampling.pop("model", None)
        if request.model and sampling_model and request.model != sampling_model:
            raise ValueError("model and sampling_params.model disagree")
        model = request.model or sampling_model or self.defaults.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Provide model or configure run_defaults.model")
        validate_sampling(sampling)
        prompt = request.prompt
        if isinstance(prompt, list):
            if (len(prompt) != 1 or set(prompt[0]) - {"role", "content"}
                    or prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str)):
                raise ValueError("Native RunSpec currently accepts text or one user message; arbitrary history needs a native-prefix adapter")
            prompt = prompt[0]["content"]
        resources = self.config.get("resources", {}).get(request.environment_ref.resource_profile)
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory_mb"}
                or any(type(n) is not int or n <= 0 for n in resources.values())):
            raise ValueError("Configure positive cpu/memory_mb for this resource_profile")
        deadline = time.time() + request.budgets.max_wall_time_seconds
        model_calls, extra_models = divmod(request.budgets.max_model_calls, request.max_samples)
        tool_calls, extra_tools = divmod(request.budgets.max_tool_calls, request.max_samples)
        samples = []
        for index, slot in enumerate(request.sample_slots[:request.max_samples]):
            spec = deepcopy(self.defaults)
            spec.update(prompt=prompt, slot=spec.get("slot", "codex"), model=model,
                        sandbox_image=entry.spawn_ref, sandbox_resources=resources, transport="http",
                        use_gateway=True, timeout_s=min(spec.get("timeout_s", float("inf")), request.budgets.max_wall_time_seconds))
            contract = {"model_endpoint": request.model_endpoint,
                        "session_server_endpoint": request.session_server_endpoint, "model": model,
                        "sampling_params": sampling, "deadline_at": deadline,
                        "max_model_calls": model_calls + (index < extra_models),
                        "max_tool_calls": tool_calls + (index < extra_tools),
                        "api_key_env": self.config.get("api_key_env")}
            spec["extra"] = {**spec.get("extra", {}), "rollout_contract": contract}
            context = {"rollout_id": request.rollout_id, "prompt_group_id": request.prompt_group_id,
                       "task_id": request.task_id, "sample_slot_id": slot.sample_slot_id,
                       "sample_index": slot.sample_index, "prompt_token_ids": list(request.prompt_token_ids),
                       "expected_weight_version": request.expected_weight_version,
                       "return_rollout_logprobs": request.return_rollout_logprobs,
                       "environment_ref": request.environment_ref.to_dict()}
            sample = {"sample_slot_id": internal_id(slot.sample_slot_id), "run": {
                "kind": "rollout", "profile": self.config.get("profile", "codex"),
                "spec": spec, "context": context,
                # Fresh worker attempts would reset per-sample admission caps.
                "max_infra_retries": 0,
            }}
            task = self.config.get("tasks", {}).get(request.task_id, {})
            if task.get("grade"):
                sample["grade"] = deepcopy(task["grade"])
            samples.append(sample)
        return {"rollout_job_id": internal_id(request.rollout_job_id),
                "prompt_group_id": internal_id(request.prompt_group_id),
                "context": {"deadline_at": deadline}, "samples": samples}

    def execution(self, group_id: str) -> dict:
        return self.driver.get(internal_id(group_id))

    def get(self, group_id: str) -> dict:
        row = self.driver.ledger.get(internal_id(group_id))
        document = row["document"]
        if "message_request" in document:
            from rl_driver.messages import MessageAdapter

            return MessageAdapter(self.driver, self.config).get(group_id)
        request = RolloutGroupRequest.from_dict(document["miles_request"])
        if request.rollout_job_id != group_id:
            raise KeyError(group_id)
        if "miles_result" in document:
            return document["miles_result"]
        if row["cancel_requested"]:
            return RolloutGroupResult(group_id, request.prompt_group_id, "cancelled", request.max_samples,
                                      stop_reason="Group cancelled; execution cleanup remains tracked by Run Store").to_dict()
        if not document["ready"]:
            return RolloutGroupResult(group_id, request.prompt_group_id,
                                      "queued" if document["status"] == "queued" else "running", request.max_samples,
                                      stop_reason="Run Store execution quarantined" if document["status"] == "quarantined" else None).to_dict()
        result = self._export(request, document)
        document["miles_result"] = result
        self.driver.ledger.save(internal_id(group_id), document, terminal=True)
        return result

    def _export(self, request: RolloutGroupRequest, document: dict) -> dict:
        trajectories, errors = [], []
        consumed = {"model_calls": 0, "tool_calls": 0}
        for slot, sample in zip(request.sample_slots[:request.max_samples], document["samples"]):
            actor = sample["actor"]
            if not actor.get("attempt_id"):
                errors.append(f"{slot.sample_slot_id}: no execution attempt")
                continue
            events = self.driver.client.all_events(actor["job_id"], attempt_id=actor["attempt_id"])
            usage = [event for event in events if event.get("type") == "rollout.usage"]
            if usage:
                for name in consumed:
                    consumed[name] += usage[-1].get(name, 0)
            try:
                records = [event for event in events if event.get("type") == "rollout.session_state"]
                if not records:
                    raise ValueError("Execution has no recorded training tokens; see /miles-executions for ordinary trajectory")
                error = str(actor.get("error") or (actor.get("result") or {}).get("error") or "")
                status = "completed" if actor["state"] == "succeeded" else "truncated" if "budget" in error else "failed"
                trajectory = session_trajectory(request, slot, records[-1]["state"], status=status)
                trajectory.metadata.update(job_id=actor["job_id"], attempt_id=actor["attempt_id"])
                grade = sample.get("grade")
                if grade is not None:
                    if grade["state"] != "succeeded" or type((grade.get("result") or {}).get("resolved")) is not bool:
                        raise ValueError("Requested grading did not produce an official resolved verdict")
                    trajectory.reward = float(grade["result"]["resolved"])
                trajectories.append(trajectory)
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"{slot.sample_slot_id}: {error}")
        enough = len(trajectories) >= request.minimum_returned_samples
        return RolloutGroupResult(
            request.rollout_job_id, request.prompt_group_id,
            "failed" if not enough else "early_stopped" if errors else "completed",
            request.max_samples, trajectories=trajectories, consumed_budget=consumed,
            stop_reason="; ".join(errors) if errors else None,
        ).to_dict()

    def release(self, group_id: str) -> dict:
        result = self.get(group_id)
        self.driver.release(internal_id(group_id))
        return {"protocol_version": result["protocol_version"], "rollout_job_id": group_id,
                "status": result["status"] if result["status"] not in {"queued", "running"} else "cancelled"}
