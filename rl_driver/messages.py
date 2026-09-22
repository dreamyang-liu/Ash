"""Message-only Miles adapter; execution and grading stay in Run Store."""

from copy import deepcopy
import time

from runstore.message_sampling import validate as validate_sampling
from rl_driver.ledger import Conflict, canonical
from rl_driver.message_protocol import MESSAGE_VERSION, MessageRequest
from rl_driver.miles import internal_id, validate_endpoint
from runstore.message_export import clean_messages
from runstore.message_completion import is_truncated_result
from runstore.branch_guidance import resolve_guidance


class MessageAdapter:
    def __init__(self, driver, config):
        self.driver, self.config = driver, deepcopy(config)

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
            self.driver.submit(plan, source_request=normalized, extra_document={
                "message_request": normalized,
                "execution_deadline_at": plan["context"]["deadline_at"],
                "deadline_at": plan["context"]["deadline_at"] + request.finalization_timeout_seconds,
            })
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("protocol_version", "rollout_job_id", "status")}

    def _plan(self, request):
        validate_endpoint(request.model_endpoint)
        validate_sampling(request.sampling_params)
        if request.max_samples > self.config.get("max_samples", 1000):
            raise ValueError("Group exceeds deployment max_samples")
        defaults = self.config.get("run_defaults", {})
        model = request.model or defaults.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Provide model or run_defaults.model")
        resources = self.config.get("image_resources") or self.config.get("resources", {}).get("standard")
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory_mb"}
                or any(type(n) is not int or n <= 0 for n in resources.values())):
            raise ValueError("Ash must configure positive image_resources cpu/memory_mb")
        task = self.config.get("tasks", {}).get(request.task_id, {})
        if not task.get("grade"):
            raise ValueError(f"Configure Ash tasks[{request.task_id!r}].grade for message rollout rewards")
        image = request.image
        if image.startswith("prime/primeintellect/"):
            # Prime's dataset documents this public-registry namespace rewrite.
            image = "docker.io/swerebenchv2/" + image.removeprefix("prime/primeintellect/")
        deadline = time.time() + request.budgets.max_wall_time_seconds
        prompt = request.prompt
        if isinstance(prompt, list):
            if (len(prompt) != 1 or prompt[0].get("role") != "user"
                    or set(prompt[0]) != {"role", "content"} or not isinstance(prompt[0]["content"], str)):
                raise ValueError("Native task input must be text or one user message")
            prompt = prompt[0]["content"]
        samples = []
        for slot in request.sample_slots[:request.max_samples]:
            spec = deepcopy(defaults)
            spec.update(prompt=prompt, model=model, slot=spec.get("slot", "mini-swe-agent"),
                        sandbox_image=image, sandbox_resources=resources, use_gateway=True, transport="http",
                        timeout_s=min(spec.get("timeout_s", float("inf")), request.budgets.max_wall_time_seconds))
            spec["extra"] = {**spec.get("extra", {}), "rollout_contract": {
                "message_export": True, "model_endpoint": request.model_endpoint, "model": model,
                "sampling_params": request.sampling_params, "deadline_at": deadline,
                "max_turns": request.max_turns,
                "api_key_env": self.config.get("api_key_env"),
            }}
            spec["extra"]["branch_guidance"] = resolve_guidance(
                spec["extra"].get("branch_guidance"), spec["slot"])
            samples.append({
                "sample_slot_id": internal_id(slot.sample_slot_id),
                "run": {"kind": "rollout", "profile": self.config.get(
                    "profile", "mini-swe-agent" if spec["slot"] == "mini-swe-agent" else "codex"),
                        "spec": spec, "max_infra_retries": 0,
                        "context": {"task_id": request.task_id, "image": request.image,
                                    "sample_slot_id": slot.sample_slot_id}},
                "grade": deepcopy(task["grade"]),
            })
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
            return document["message_result"]
        result = {
            "protocol_version": MESSAGE_VERSION, "rollout_job_id": group_id,
            "prompt_group_id": request.prompt_group_id, "max_samples": request.max_samples,
            "actual_samples": 0, "trajectories": [], "search_branches": 0, "consumed_budget": {},
            "status": "queued" if document["status"] == "queued" else "running", "stop_reason": None,
        }
        if row["cancel_requested"]:
            result["status"] = "cancelled"
            return result
        if not document["ready"]:
            return result
        self._export(request, document, result)
        document["message_result"] = result
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
                if not grade or grade["state"] != "succeeded" or type(verdict.get("resolved")) is not bool:
                    raise ValueError("Ash grading did not produce a resolved verdict")
                origin = {**(output.get("training_origin") or {}), **(actor.get("origin") or {})}
                parent = origin.get("job_id")
                if origin.get("recovery_kind") == "retry" and parent == actor["job_id"]:
                    parent = None
                result["trajectories"].append({
                    "sample_slot_id": slot.sample_slot_id, "branch_id": actor["job_id"],
                    "parent_branch_id": parent, "messages": clean_messages(messages),
                    "tools": output.get("training_tools", []), "reward": float(verdict["resolved"]),
                    "status": "truncated" if truncated else "completed",
                    "stop_reason": output.get("stop_reason") if truncated else None,
                    "hints_removed": True,
                    "metadata": {"job_id": actor["job_id"], "attempt_id": actor["attempt_id"],
                                 "source_image": request.image,
                                 "graded_snapshot_id": grade.get("snapshot_id"),
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
