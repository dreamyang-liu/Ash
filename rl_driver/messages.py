"""Message-only Miles adapter; execution and grading stay in Run Store."""

from copy import deepcopy
import time

from runstore.message_sampling import validate as validate_sampling
from rl_driver.environment_catalog import EnvironmentCatalog, EnvironmentResolver
from rl_driver.ledger import Conflict, canonical
from rl_driver.message_protocol import MESSAGE_VERSION, MessageRequest
from rl_driver.miles import internal_id, native_prompt, rollout_progress, validate_endpoint
from rl_driver.profiling import build_profile_records
from rl_driver.tasks import resolve_task
from runstore.message_export import clean_messages
from runstore.message_completion import is_truncated_result


class MessageAdapter:
    def __init__(self, driver, config):
        self.driver, self.config = driver, deepcopy(config)
        catalog = (EnvironmentCatalog.from_dict(config["environment_catalog"])
                   if config.get("environment_catalog") else None)
        self.environments = EnvironmentResolver(
            catalog,
            allowed_oci_registries=config.get("allowed_oci_registries", ()),
        )

    def submit(self, body):
        request = MessageRequest.from_dict(body)
        normalized = request.to_dict()
        identifier = internal_id(request.rollout_job_id)
        try:
            previous = self.driver.ledger.get(identifier)
        except KeyError:
            previous = None
        if previous is not None:
            if canonical(previous["request"]) != canonical(normalized):
                raise Conflict("rollout_job_id already names a different request")
        else:
            plan = self._plan(request)
            deferred = (
                {sample["sample_slot_id"] for sample in plan["samples"][1:]}
                if self.config.get("branch_policy") is not None
                else set()
            )
            self.driver.submit(
                plan,
                source_request=normalized,
                extra_document={
                    "message_request": normalized,
                    "execution_deadline_at": plan["context"]["deadline_at"],
                    "deadline_at": plan["context"]["deadline_at"]
                    + request.finalization_timeout_seconds,
                    "branch_policy": self.config.get("branch_policy"),
                },
                deferred_sample_ids=deferred,
            )
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("protocol_version", "rollout_job_id", "status")}

    def _plan(self, request):
        validate_endpoint(request.model_endpoint)
        if request.session_server_endpoint:
            validate_endpoint(request.session_server_endpoint)
        validate_sampling(request.sampling_params)
        if request.max_samples > self.config.get("max_samples", 1000):
            raise ValueError("Group exceeds deployment max_samples")
        defaults = self.config.get("run_defaults", {})
        model = request.model or defaults.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Provide model or run_defaults.model")
        entry = self.environments.resolve(request.environment_ref)
        resources = self.config.get("resources", {}).get(request.environment_ref.resource_profile)
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory_mb"}
                or any(type(n) is not int or n <= 0 for n in resources.values())):
            raise ValueError("Ash must configure positive cpu/memory_mb for this resource_profile")
        if request.task_id not in self.config.get("tasks", {}):
            raise ValueError(
                f"Configure Ash tasks[{request.task_id!r}] with environment_ref"
            )
        task = resolve_task(
            self.config,
            request.task_id,
            request.environment_ref,
            require_grade=False,
        )
        deadline = time.time() + request.budgets.max_wall_time_seconds
        shared_session_id = (
            internal_id(request.rollout_job_id)
            if request.session_server_endpoint
            else None
        )
        slot_name = defaults.get("slot", "codex")
        prompt, system_prompt = native_prompt(request.prompt, slot_name)
        samples = []
        for slot in request.sample_slots[:request.max_samples]:
            spec = deepcopy(defaults)
            spec.update(prompt=prompt, model=model, slot=slot_name,
                        sandbox_image=entry.spawn_ref, sandbox_resources=resources, use_gateway=True, transport="http",
                        timeout_s=min(spec.get("timeout_s", float("inf")), request.budgets.max_wall_time_seconds))
            contract = {
                "message_export": True, "model_endpoint": request.model_endpoint, "model": model,
                "sampling_params": request.sampling_params, "deadline_at": deadline,
                "max_turns": request.max_turns,
                "capture_recovery_points": self.config.get("branch_policy") is not None,
                "capture_final_snapshot": True,
                "api_key_env": self.config.get("api_key_env"),
            }
            if shared_session_id is not None:
                contract.update(
                    session_server_endpoint=request.session_server_endpoint,
                    session_id=shared_session_id,
                    retain_session=True,
                )
            spec["extra"] = {**spec.get("extra", {}), "rollout_contract": contract}
            if system_prompt is not None:
                spec["extra"]["system_prompt"] = system_prompt
            if task.get("repository") is not None:
                spec["extra"]["repository_preflight"] = deepcopy(task["repository"])
            sample = {
                "sample_slot_id": internal_id(slot.sample_slot_id),
                "run": {"kind": "rollout", "profile": self.config.get("profile", "codex"),
                        "spec": spec, "max_infra_retries": 0,
                        "context": {"task_id": request.task_id,
                                    "prompt_token_alignment": (
                                        "harness_rendered"
                                        if slot_name == "claude-code"
                                        else "request_exact"
                                    ),
                                    "environment_ref": request.environment_ref.to_dict(),
                                    "sample_slot_id": slot.sample_slot_id,
                                    **({"miles_session_id": shared_session_id}
                                       if shared_session_id is not None else {})}},
            }
            if task.get("grade") is not None:
                sample["grade"] = deepcopy(task["grade"])
            samples.append(sample)
        return {"rollout_job_id": internal_id(request.rollout_job_id),
                "prompt_group_id": internal_id(request.prompt_group_id),
                "context": {"deadline_at": deadline}, "samples": samples}

    def get(self, group_id):
        row = self.driver.ledger.get(internal_id(group_id))
        document = row["document"]
        request = MessageRequest.from_dict(document["message_request"])
        if request.rollout_job_id != group_id:
            raise KeyError(group_id)
        if "message_result" in document:
            if "profiling_records" not in document:
                document["profiling_records"] = build_profile_records(
                    request.to_dict(), document["message_result"], document,
                    created_at=row["created_at"],
                )
                self.driver.ledger.save(internal_id(group_id), document, terminal=True)
            return document["message_result"]
        result = {
            "protocol_version": MESSAGE_VERSION, "rollout_job_id": group_id,
            "prompt_group_id": request.prompt_group_id, "max_samples": request.max_samples,
            "actual_samples": 0, "trajectories": [], "search_branches": 0, "consumed_budget": {},
            "status": "queued" if document["status"] == "queued" else "running", "stop_reason": None,
        }
        if not document["ready"]:
            progress = rollout_progress(
                row, document,
                [slot.sample_slot_id for slot in request.sample_slots[:request.max_samples]],
            ).to_dict()
            if row["cancel_requested"]:
                result["status"] = "running"
                result["stop_reason"] = (
                    "Cancellation requested; execution cleanup is still in progress"
                )
                progress["phase"] = "cancelling"
            result["progress"] = progress
            return result
        if row["cancel_requested"]:
            result["status"] = "cancelled"
            result["stop_reason"] = "Group cancelled after Run Store execution cleanup"
            return result
        self._export(request, document, result)
        document["message_result"] = result
        document["profiling_records"] = build_profile_records(
            request.to_dict(), result, document, created_at=row["created_at"]
        )
        self.driver.ledger.save(internal_id(group_id), document, terminal=True)
        return result

    def _export(self, request, document, result):
        errors = []
        for slot, sample in zip(request.sample_slots[:request.max_samples], document["samples"], strict=False):
            actor, grade = sample["actor"], sample["grade"]
            try:
                if actor["state"] != "succeeded":
                    raise ValueError("Actor did not complete")
                output = actor.get("result") or {}
                truncated = is_truncated_result(output)
                if output.get("status") == "truncated" and not truncated:
                    raise ValueError("Truncated actor lacks a verified cutoff result")
                if output.get("training_export_error"):
                    raise ValueError(output["training_export_error"])
                messages = output.get("training_messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("Execution has no exported native messages")
                verdict = (grade or {}).get("result") or {}
                if grade is not None and (
                    grade["state"] != "succeeded"
                    or type(verdict.get("resolved")) is not bool
                ):
                    raise ValueError("Ash grading did not produce a resolved verdict")
                origin = actor.get("origin") or output.get("training_origin") or {}
                parent = origin.get("job_id")
                result["trajectories"].append({
                    "sample_slot_id": slot.sample_slot_id, "branch_id": actor["job_id"],
                    "parent_branch_id": parent, "messages": clean_messages(messages),
                    "tools": output.get("training_tools", []),
                    "reward": (
                        float(verdict["resolved"])
                        if grade is not None
                        else None
                    ),
                    "status": "truncated" if truncated else "completed",
                    "stop_reason": output.get("stop_reason") if truncated else None,
                    "hints_removed": True,
                    "metadata": {"job_id": actor["job_id"], "attempt_id": actor["attempt_id"],
                                 "environment_ref": request.environment_ref.to_dict(),
                                 "graded_snapshot_id": (
                                     grade.get("snapshot_id") if grade is not None else None
                                 ),
                                 "origin": origin, "logprob_context": "hint_free_messages"},
                })
                result["search_branches"] += int(parent is not None)
                for name, count in output.get("rollout_usage", {}).items():
                    result["consumed_budget"][name] = result["consumed_budget"].get(name, 0) + count
            except (ValueError, KeyError, TypeError) as error:
                errors.append(f"{slot.sample_slot_id}: {error}")
        result["actual_samples"] = len(result["trajectories"])
        result["status"] = ("failed" if result["actual_samples"] < request.minimum_returned_samples
                            else "early_stopped" if errors else "completed")
        result["stop_reason"] = "; ".join(errors) or None

    def release(self, group_id):
        result = self.get(group_id)
        self.driver.release(internal_id(group_id))
        return {"protocol_version": MESSAGE_VERSION, "rollout_job_id": group_id,
                "status": result["status"] if result["status"] not in {"queued", "running"} else "cancelled"}
