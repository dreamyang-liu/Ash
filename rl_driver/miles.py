"""Original Miles wire contract translated to the existing execution queue."""

from copy import deepcopy
import hashlib
import math
import time
from urllib.parse import urlsplit

import httpx

from rl_driver.environment_catalog import EnvironmentCatalog, EnvironmentResolver
from rl_driver.ledger import Conflict, canonical
from rl_driver.profiling import build_profile_records
from rl_driver.protocol import (
    PROTOCOL_VERSION,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutProgress,
    Trajectory,
)
from rl_driver.tasks import resolve_task


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


def _session_path(state: dict, response_ids: list[str]) -> dict:
    """Select one immutable SessionTree path by the responses this job observed."""
    metadata = state.get("metadata") or {}
    tree = metadata.get("tree") or {}
    records = metadata.get("tree_records") or {}
    if not response_ids or not records:
        return state
    nodes = {node.get("id"): node for node in tree.get("nodes") or []}
    by_response = {}
    for node in nodes.values():
        response_id = node.get("response_id")
        if response_id in by_response:
            raise ValueError("Miles SessionTree contains duplicate response_id")
        by_response[response_id] = node
    final = by_response.get(response_ids[-1])
    if final is None:
        raise ValueError("Miles SessionTree lacks this job's final response")
    path = []
    node = final
    while node is not None:
        path.append(node["id"])
        parent = node.get("parent")
        node = nodes.get(parent) if parent is not None else None
    path.reverse()
    path_responses = [nodes[node_id].get("response_id") for node_id in path]
    cursor = iter(path_responses)
    if any(response_id not in cursor for response_id in response_ids):
        raise ValueError("This job's model responses do not form one SessionTree path")
    if any(str(node_id) not in records for node_id in path):
        raise ValueError("Miles SessionTree path is missing token records")
    return {
        "records": [records[str(node_id)]["record"] for node_id in path],
        "metadata": {
            **metadata,
            "accumulated_token_ids": records[str(path[-1])]["token_ids"],
            "selected_node_id": path[-1],
            "selected_path_node_ids": path,
        },
    }


def session_trajectory(
    request: RolloutGroupRequest,
    sample,
    state: dict,
    *,
    status: str,
    prompt_token_alignment: str = "request_exact",
    response_ids: list[str] | None = None,
    branch_id: str | None = None,
    parent_branch_id: str | None = None,
    branch_point_token_count: int | None = None,
) -> Trajectory:
    """Export observed tokens only. Never substitute an expected weight version."""
    state = _session_path(state, response_ids or [])
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
        "branch_id": branch_id or f"{request.rollout_job_id}:root:{sample.sample_index}",
        "parent_branch_id": parent_branch_id,
        "branch_point_token_count": branch_point_token_count,
        "messages": messages, "token_ids": tokens, "prompt_length": spans[0]["start"],
        "generated_spans": spans, "response_text": str(messages[-1].get("content") or ""),
        "status": status, "prompt_token_alignment": prompt_token_alignment,
    })


class MilesAdapter:
    def __init__(self, driver, config: dict, *, branch_policy=None):
        self.driver = driver
        self.config = deepcopy(config)
        if not isinstance(config, dict) or set(config) - {
            "environment_catalog", "profile", "run_defaults", "resources", "tasks", "api_key_env", "max_samples",
            "allowed_oci_registries", "branch_policy"}:
            raise ValueError("Invalid Miles adapter config")
        self.branch_policy = branch_policy
        if config.get("branch_policy") is not None and branch_policy is None:
            raise ValueError("Configured branch_policy was not loaded")
        self.catalog = (EnvironmentCatalog.from_dict(config["environment_catalog"])
                        if config.get("environment_catalog") else None)
        self.environment_resolver = EnvironmentResolver(
            self.catalog,
            allowed_oci_registries=config.get("allowed_oci_registries", ()),
        )
        self.defaults = deepcopy(config.get("run_defaults", {}))
        if self.defaults.get("slot", "codex") not in {"codex", "claude-code"}:
            raise ValueError("Miles RunSpec adapter supports codex and claude-code")
        if "rollout_contract" in self.defaults.get("extra", {}):
            raise ValueError("rollout_contract is derived from each Miles request")

    def environments(self):
        return {"protocol_version": PROTOCOL_VERSION,
                "environments": [ref.to_dict() for ref in self.environment_resolver.list_refs()]}

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
            deferred = (
                {sample["sample_slot_id"] for sample in plan["samples"][1:]}
                if self.branch_policy is not None
                else set()
            )
            self.driver.submit(
                plan,
                source_request=normalized,
                extra_document={
                    "miles_request": normalized,
                    "deadline_at": plan["context"]["deadline_at"],
                    "branch_policy": self.config.get("branch_policy"),
                },
                deferred_sample_ids=deferred,
            )
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("protocol_version", "rollout_job_id", "status")}

    def _plan(self, request: RolloutGroupRequest) -> dict:
        entry = self.environment_resolver.resolve(request.environment_ref)
        task = resolve_task(
            self.config,
            request.task_id,
            request.environment_ref,
            require_grade=False,
        )
        if request.max_samples > self.config.get("max_samples", 1000):
            raise ValueError("Requested group exceeds deployment max_samples")
        if (request.budgets.max_model_calls is not None
                and request.budgets.max_model_calls < request.max_samples):
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
        slot_name = self.defaults.get("slot", "codex")
        prompt, system_prompt = _native_prompt(request.prompt, slot_name)
        resources = self.config.get("resources", {}).get(request.environment_ref.resource_profile)
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory_mb"}
                or any(type(n) is not int or n <= 0 for n in resources.values())):
            raise ValueError("Configure positive cpu/memory_mb for this resource_profile")
        deadline = time.time() + request.budgets.max_wall_time_seconds
        shared_session_id = (
            internal_id(request.rollout_job_id)
            if request.session_server_endpoint
            else None
        )
        if request.budgets.max_model_calls is None:
            model_calls, extra_models = None, 0
        else:
            model_calls, extra_models = divmod(request.budgets.max_model_calls, request.max_samples)
        if request.budgets.max_tool_calls is None:
            tool_calls, extra_tools = None, 0
        else:
            tool_calls, extra_tools = divmod(request.budgets.max_tool_calls, request.max_samples)
        samples = []
        for index, slot in enumerate(request.sample_slots[:request.max_samples]):
            spec = deepcopy(self.defaults)
            spec.update(prompt=prompt, slot=slot_name, model=model,
                        sandbox_image=entry.spawn_ref, sandbox_resources=resources, transport="http",
                        use_gateway=True, timeout_s=min(spec.get("timeout_s", float("inf")), request.budgets.max_wall_time_seconds))
            if system_prompt is not None:
                spec["extra"] = {**spec.get("extra", {}), "system_prompt": system_prompt}
            contract = {"model_endpoint": request.model_endpoint,
                        "session_server_endpoint": request.session_server_endpoint, "model": model,
                        "sampling_params": sampling, "deadline_at": deadline,
                        "max_model_calls": (None if model_calls is None
                                            else model_calls + (index < extra_models)),
                        "max_tool_calls": (None if tool_calls is None
                                           else tool_calls + (index < extra_tools)),
                        "api_key_env": self.config.get("api_key_env")}
            if shared_session_id is not None:
                contract.update(
                    session_id=shared_session_id,
                    retain_session=True,
                )
            spec["extra"] = {**spec.get("extra", {}), "rollout_contract": contract}
            if task.get("repository"):
                spec["extra"]["repository_preflight"] = deepcopy(
                    task["repository"]
                )
            context = {"rollout_id": request.rollout_id, "prompt_group_id": request.prompt_group_id,
                       "task_id": request.task_id, "sample_slot_id": slot.sample_slot_id,
                       "sample_index": slot.sample_index, "prompt_token_ids": list(request.prompt_token_ids),
                       "expected_weight_version": request.expected_weight_version,
                       "return_rollout_logprobs": request.return_rollout_logprobs,
                       "prompt_token_alignment": (
                           "harness_rendered" if slot_name == "claude-code" else "request_exact"
                       ),
                       "environment_ref": request.environment_ref.to_dict()}
            if shared_session_id is not None:
                context["miles_session_id"] = shared_session_id
            run = {
                "kind": "rollout", "profile": self.config.get("profile", "codex"),
                "spec": spec, "context": context,
                # Fresh worker attempts would reset per-sample admission caps.
                "max_infra_retries": 0,
            }
            sample = {"sample_slot_id": internal_id(slot.sample_slot_id), "run": run}
            if task.get("grade"):
                sample["grade"] = deepcopy(task["grade"])
            samples.append(sample)
        return {"rollout_job_id": internal_id(request.rollout_job_id),
                "prompt_group_id": internal_id(request.prompt_group_id),
                "context": {"deadline_at": deadline}, "samples": samples}

    def execution(self, group_id: str) -> dict:
        return self.driver.get(internal_id(group_id))

    def reconcile_policy(self) -> None:
        """Let one trusted policy consume v2 and v3 deferred sample slots."""
        from rl_driver.message_protocol import MessageRequest
        from rl_driver.policy import reconcile_branch_policy

        reconcile_branch_policy(
            self.driver,
            self.branch_policy,
            {"miles_request": RolloutGroupRequest, "message_request": MessageRequest},
            internal_id,
        )

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
            if "profiling_records" not in document:
                document["profiling_records"] = build_profile_records(
                    request.to_dict(), document["miles_result"], document,
                    created_at=row["created_at"],
                )
                self.driver.ledger.save(internal_id(group_id), document, terminal=True)
            return document["miles_result"]
        if not document["ready"]:
            progress = rollout_progress(
                row, document,
                [slot.sample_slot_id for slot in request.sample_slots[:request.max_samples]],
            )
            if row["cancel_requested"]:
                progress = RolloutProgress(
                    **{**progress.to_dict(), "phase": "cancelling"}
                )
            return RolloutGroupResult(group_id, request.prompt_group_id,
                                      "running" if row["cancel_requested"]
                                      else "queued" if document["status"] == "queued"
                                      else "running", request.max_samples,
                                      stop_reason=(
                                          "Cancellation requested; execution cleanup is still in progress"
                                          if row["cancel_requested"]
                                          else "Run Store execution quarantined"
                                          if document["status"] == "quarantined"
                                          else None
                                      ),
                                      progress=progress).to_dict()
        if row["cancel_requested"]:
            return RolloutGroupResult(
                group_id,
                request.prompt_group_id,
                "cancelled",
                request.max_samples,
                stop_reason="Group cancelled after Run Store execution cleanup",
            ).to_dict()
        result = self._export(request, document)
        document["miles_result"] = result
        document["profiling_records"] = build_profile_records(
            request.to_dict(), result, document, created_at=row["created_at"]
        )
        self.driver.ledger.save(internal_id(group_id), document, terminal=True)
        return result

    def _export(self, request: RolloutGroupRequest, document: dict) -> dict:
        trajectories, errors = [], []
        search_branches = 0
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
                execution_context = actor.get("submission", {}).get("context", {})
                response_ids = [
                    event.get("response_id")
                    for event in events
                    if event.get("type") == "rollout.model_response"
                    and event.get("response_id")
                ]
                origin = actor.get("origin") or {}
                parent_branch_id = origin.get("job_id")
                branch_point = None
                if parent_branch_id is not None:
                    selected = _session_path(records[-1]["state"], response_ids)
                    first_local = response_ids[0] if response_ids else None
                    for node in ((selected.get("metadata") or {}).get("tree") or {}).get("nodes", []):
                        if node.get("response_id") == first_local:
                            span = node.get("completion_span") or []
                            if len(span) == 2:
                                branch_point = int(span[0])
                            break
                    if branch_point is None:
                        raise ValueError("Child trajectory has no verified branch-point token count")
                    search_branches += 1
                trajectory = session_trajectory(
                    request,
                    slot,
                    records[-1]["state"],
                    status=status,
                    prompt_token_alignment=execution_context.get(
                        "prompt_token_alignment", "request_exact"
                    ),
                    response_ids=response_ids,
                    branch_id=actor["job_id"],
                    parent_branch_id=parent_branch_id,
                    branch_point_token_count=branch_point,
                )
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
            search_branches=search_branches,
            stop_reason="; ".join(errors) if errors else None,
        ).to_dict()

    def release(self, group_id: str) -> dict:
        document = self.driver.ledger.get(internal_id(group_id))["document"]
        if "message_request" in document:
            from rl_driver.messages import MessageAdapter

            result = MessageAdapter(self.driver, self.config).release(group_id)
            self._release_session_if_eligible(internal_id(group_id))
            return result
        result = self.get(group_id)
        self.driver.release(internal_id(group_id))
        self._release_session_if_eligible(internal_id(group_id))
        return {"protocol_version": result["protocol_version"], "rollout_job_id": group_id,
                "status": result["status"] if result["status"] not in {"queued", "running"} else "cancelled"}

    def reconcile_sessions(self) -> None:
        """Release shared sessions only after their whole group is quiescent.

        Successful groups retain the SessionTree until the caller acknowledges
        consumption with DELETE, so a branch planner can still inspect their
        recovery points.  Cancelled groups retain it until every Run Store job
        has actually stopped.  Session DELETE is idempotent and the durable
        marker prevents ordinary polling from repeating it.
        """
        for row in self.driver.ledger.rows():
            document = row["document"]
            if not ({"miles_request", "message_request"} & document.keys()):
                continue
            if row["terminal"] and (row["cancel_requested"] or row["acknowledged_at"] is not None):
                self._release_session_if_eligible(row["id"])

    def _release_session_if_eligible(self, identifier: str) -> bool:
        row = self.driver.ledger.get(identifier)
        document = row["document"]
        if document.get("miles_session_released_at") is not None:
            return False
        if not row["terminal"] or not (
            row["cancel_requested"] or row["acknowledged_at"] is not None
        ):
            return False
        if "message_request" in document:
            from rl_driver.message_protocol import MessageRequest

            request = MessageRequest.from_dict(document["message_request"])
        else:
            request = RolloutGroupRequest.from_dict(document["miles_request"])
        if request.session_server_endpoint:
            url = (
                request.session_server_endpoint.rstrip("/")
                + "/sessions/"
                + internal_id(request.rollout_job_id)
            )
            response = httpx.delete(url, timeout=30)
            if response.status_code not in {204, 404}:
                response.raise_for_status()
        # Save only after the idempotent remote release succeeds.  A crash in
        # between can repeat DELETE safely; it cannot mark a live session dead.
        row = self.driver.ledger.get(identifier)
        document = row["document"]
        document["miles_session_released_at"] = time.time()
        self.driver.ledger.save(identifier, document, terminal=row["terminal"])
        return True


_TOKEN_PROGRESS_FIELDS = (
    "trajectory_tokens",
    "assistant_generated_tokens",
    "current_context_tokens",
    "peak_context_tokens",
    "last_model_output_tokens",
)


def rollout_progress(
    row: dict, document: dict, sample_slot_ids: list[str] | None = None
) -> RolloutProgress:
    actors = [sample["actor"] for sample in document["samples"]]
    active = [actor for actor in actors if actor.get("state") == "running"]
    selected = max(
        active,
        key=lambda actor: (actor.get("progress") or {}).get(
            "updated_at_unix_seconds", 0
        ),
        default=None,
    )
    selected_progress = (selected or {}).get("progress") or {}
    now = time.time()
    deadline = document.get("deadline_at")
    active_slot = None
    if selected is not None:
        selected_index = next(
            (index for index, sample in enumerate(document["samples"])
             if sample["actor"] is selected),
            None,
        )
        if selected_index is not None:
            active_slot = (
                sample_slot_ids[selected_index]
                if sample_slot_ids is not None
                else document["samples"][selected_index]["sample_slot_id"]
            )
    return RolloutProgress(
        phase=(
            selected_progress.get("phase")
            or (selected or {}).get("phase")
            or document.get("status", "queued")
        ),
        model_calls=sum(
            int((actor.get("progress") or {}).get("model_calls", 0))
            for actor in actors
        ),
        tool_calls=sum(
            int((actor.get("progress") or {}).get("tool_calls", 0))
            for actor in actors
        ),
        completed_samples=sum(actor.get("state") == "succeeded" for actor in actors),
        active_sample_slot_id=active_slot,
        **{
            name: int(selected_progress.get(name, 0))
            for name in _TOKEN_PROGRESS_FIELDS
        },
        elapsed_seconds=round(max(0.0, now - float(row["created_at"])), 3),
        remaining_wall_time_seconds=(
            round(max(0.0, float(deadline) - now), 3)
            if deadline is not None
            else None
        ),
        updated_at_unix_seconds=float(
            selected_progress.get("updated_at_unix_seconds", row["created_at"])
        ),
    )


def _native_prompt(prompt: str | list[dict], slot: str) -> tuple[str, object | None]:
    """Map a fresh structured task onto a native SDK without flattening roles.

    Run Store branches use native prefix restoration and are handled elsewhere.
    A fresh rollout therefore accepts only system/developer preamble followed by
    one user task. Claude's stable built-in harness prompt is retained while the
    caller's system text is appended through the SDK's preset form.
    """
    if isinstance(prompt, str):
        return prompt, None
    if not prompt:
        raise ValueError("prompt messages must not be empty")
    system_parts = []
    user = None
    for index, message in enumerate(prompt):
        if set(message) != {"role", "content"} or not isinstance(
            message.get("content"), str
        ):
            raise ValueError(
                "Native prompt messages require only string role/content fields"
            )
        role = message.get("role")
        if role in {"system", "developer"} and user is None:
            system_parts.append(message["content"])
        elif role == "user" and user is None and index == len(prompt) - 1:
            user = message["content"]
        else:
            raise ValueError(
                "Fresh native rollouts accept a system/developer preamble and "
                "one final user message; assistant/tool history requires a "
                "validated native prefix"
            )
    if user is None or not user.strip():
        raise ValueError(
            "Native prompt messages require one nonempty final user message"
        )
    if not system_parts:
        return user, None
    if slot != "claude-code":
        raise ValueError(
            "Structured system prompts currently require the claude-code slot"
        )
    return user, {
        "type": "preset",
        "preset": "claude_code",
        "append": "\n\n".join(system_parts),
    }
