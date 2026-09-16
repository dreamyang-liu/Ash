"""One restartable coordinator; all agent and grader jobs execute elsewhere."""

from copy import deepcopy
import threading
import time

import httpx

from runstore.specs import JobSpec
from rl_driver.ledger import Conflict, Ledger, canonical
from rl_driver.specs import initial_document, submission_key, validate_request

# Run Store quarantine is terminal from the RL caller's perspective.  It needs
# operator reconciliation before the execution can make progress, so keeping a
# rollout group active would make clients poll until their unrelated wall-time
# deadline expires.  Export it as a failed sample while preserving the Run
# Store job and its reconciliation evidence.
DONE = {"succeeded", "failed", "quarantined", "cancelled", "skipped"}


class Driver:
    def __init__(self, client, ledger: Ledger):
        self.client = client
        self.ledger = ledger
        self._tick_lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stopping = threading.Event()

    def submit(
        self,
        body: dict,
        *,
        source_request: dict | None = None,
        extra_document: dict | None = None,
        deferred_sample_ids: set[str] | None = None,
    ) -> dict:
        """Persist a group, optionally deferring already-validated sample slots.

        Deferral is an RL-driver implementation detail.  It deliberately is
        not part of the public execution-group schema: every entry in ``body``
        remains an ordinary validated root run, while a policy may later turn
        a deferred slot into a root, a Run Store branch, or a skipped slot.
        """
        request = validate_request(body)
        document = {**initial_document(request), **(extra_document or {})}
        deferred = deferred_sample_ids or set()
        known = {sample["sample_slot_id"] for sample in document["samples"]}
        if deferred - known:
            raise ValueError("Deferred sample IDs must belong to this execution group")
        for sample in document["samples"]:
            actor = sample["actor"]
            if sample["sample_slot_id"] not in deferred:
                continue
            if actor["operation"] != "run":
                raise ValueError("Only fresh root templates can be deferred")
            actor.update(
                state="deferred",
                operation="deferred",
                template=actor.pop("submission"),
            )
        row = self.ledger.create(request["rollout_job_id"], source_request or request, document)
        self.wakeup.set()
        return self._view(row)

    @staticmethod
    def _view(row: dict) -> dict:
        document = deepcopy(row["document"])
        document["cancel_requested"] = row["cancel_requested"]
        document["acknowledged_at"] = row["acknowledged_at"]
        if row["cancel_requested"] and not row["terminal"]:
            document["status"] = "cancelling"
        # Submission bodies may contain long prompts and are already retained
        # in the request ledger. Status/results expose the execution references.
        for sample in document["samples"]:
            for phase in ("actor", "grade"):
                stage = sample[phase]
                if stage:
                    stage.pop("submission", None)
                    stage.pop("template", None)
        return document

    def get(self, group_id: str) -> dict:
        return self._view(self.ledger.get(group_id))

    def release(self, group_id: str) -> dict:
        self.ledger.release(group_id)
        self.wakeup.set()
        view = self.get(group_id)
        return {key: view[key] for key in ("protocol_version", "rollout_job_id", "status", "acknowledged_at")}

    def _save(self, document: dict) -> None:
        self.ledger.save(document["rollout_job_id"], document, terminal=document["ready"])

    def _cancelled(self, document: dict) -> bool:
        return self.ledger.get(document["rollout_job_id"])["cancel_requested"]

    def _dispatch(self, document: dict, sample: dict, phase: str) -> None:
        stage = sample[phase]
        if stage["state"] == "planned":
            if self._cancelled(document):
                stage["state"] = "cancelled"
                self._save(document)
                return
            stage["state"] = "submitting"
            self._save(document)  # intent is durable BEFORE a remote side effect
        key = submission_key(document["rollout_job_id"], sample["sample_slot_id"], phase)
        try:
            if stage["operation"] == "branch":
                branch = stage["submission"]
                job_id = self.client.branch(branch["job_id"], branch["point_id"],
                                            idempotency_key=key,
                                            context=branch.get("context"),
                                            **branch["overrides"])
            else:
                job_id = self.client.submit(JobSpec.from_dict(stage["submission"]), key)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {400, 404, 409, 422} and not stage.get("submission_uncertain"):
                stage.update(state="failed", error=error.response.text)
                self._save(document)
                return
            stage["submission_uncertain"] = True
            raise
        except httpx.RequestError:
            stage["submission_uncertain"] = True
            raise
        stage.update(job_id=job_id, state="queued")
        stage.pop("submission_uncertain", None)
        stage.pop("last_error", None)
        self._save(document)

    def _poll(self, document: dict, stage: dict) -> None:
        job = self.client.get(stage["job_id"])
        if job["state"] in {"queued", "running"} and self._cancelled(document):
            self.client.request_cancel(stage["job_id"])
            job = self.client.get(stage["job_id"])
        stage.update(state=job["state"], attempt_id=job.get("active_attempt"),
                     phase=job.get("phase"), error=job.get("error"))
        if isinstance(job.get("progress"), dict):
            stage["progress"] = deepcopy(job["progress"])
        if job["state"] in DONE:
            stage["result"] = job.get("result")
        stage.pop("last_error", None)
        self._save(document)

    def _prepare_grade(self, document: dict, sample: dict) -> None:
        actor, grade = sample["actor"], sample["grade"]
        attempt = actor["attempt_id"]
        if not attempt:
            raise ValueError("Completed actor has no attempt identity")
        output = actor.get("result") or {}
        if output.get("training_snapshot_error"):
            raise ValueError(output["training_snapshot_error"])
        if output.get("final_snapshot_id"):
            # A worker-captured final snapshot includes text-only last turns and
            # also permits scoring a run with no tool calls.
            point = {"id": None, "snapshot_id": output["final_snapshot_id"]}
        else:
            tools = self.client.all_tools(actor["job_id"], attempt_id=attempt)
            if not tools or any(tool.get("response") is None for tool in tools):
                raise ValueError("Cannot grade without a complete final tool boundary")
            final_depth = tools[-1]["depth"]
            if [tool["depth"] for tool in tools] != list(range(1, final_depth + 1)):
                raise ValueError("Final tool prefix has a gap")
            points = self.client.points(actor["job_id"], attempt_id=attempt)
            candidates = [point for point in points if point.get("available") is True
                          and point["tool_depth"] == final_depth]
            if not candidates:
                raise ValueError("Final message has no available recovery point; refusing an older snapshot")
            point = max(candidates, key=lambda item: item["message_step"])
        submission = deepcopy(grade["template"])
        submission["spec"]["snapshot_id"] = point["snapshot_id"]
        if submission["spec"].get("benchmark") == "swe-rebench-v2":
            prepared = self.client.events_of_type(
                actor["job_id"],
                attempt_id=attempt,
                event_types=("environment.prepared",),
                limit=2,
            )
            if len(prepared) != 1:
                raise ValueError(
                    "SWE-rebench grading requires exactly one durable repository baseline"
                )
            baseline = prepared[0].get("baseline_untracked")
            if (
                not isinstance(baseline, list)
                or any(not isinstance(path, str) or not path for path in baseline)
                or baseline != sorted(set(baseline))
            ):
                raise ValueError("SWE-rebench repository baseline is invalid")
            submission["spec"]["baseline_untracked"] = baseline
            submission["spec"]["repository_workdir"] = prepared[0].get("workdir")
            submission["spec"]["repository_base_commit"] = prepared[0].get(
                "base_commit"
            )
        submission["context"]["rl_driver"] = {
            "rollout_job_id": document["rollout_job_id"], "sample_slot_id": sample["sample_slot_id"],
            "actor_job_id": actor["job_id"], "actor_attempt_id": attempt, "point_id": point["id"],
        }
        grade["submission"] = JobSpec.from_dict(submission).validate()
        grade["snapshot_id"] = point["snapshot_id"]
        self._save(document)

    def _advance(self, document: dict, sample: dict, phase: str) -> None:
        stage = sample[phase]
        if stage is None or stage["state"] in DONE:
            return
        if stage["state"] == "deferred":
            if self._cancelled(document):
                stage["state"] = "cancelled"
                self._save(document)
            return
        if not stage["job_id"]:
            if self._cancelled(document):
                if stage["state"] == "planned":
                    stage["state"] = "cancelled"
                else:
                    # A lost acknowledgement cannot be reconciled by creating
                    # a possibly NEW job after cancellation was requested.
                    stage["last_error"] = "Submission acknowledgement unknown; inspect Run Store using the idempotency key"
                    stage["idempotency_key"] = submission_key(
                        document["rollout_job_id"], sample["sample_slot_id"], phase)
                self._save(document)
                return
            if phase == "grade" and "submission" not in stage:
                try:
                    self._prepare_grade(document, sample)
                except ValueError as error:
                    stage.update(state="failed", error=str(error))
                    self._save(document)
                    return
            self._dispatch(document, sample, phase)
        if stage["job_id"]:
            self._poll(document, stage)

    def tick(self) -> None:
        """Poll all owned groups once. A disconnected group does not block others."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            for group_id in self.ledger.active():
                if self.stopping.is_set():
                    break
                document = self.ledger.get(group_id)["document"]
                if document.get("deadline_at") is not None and time.time() >= document["deadline_at"]:
                    self.ledger.release(group_id)
                for sample in document["samples"]:
                    for phase in ("actor", "grade"):
                        if self.stopping.is_set():
                            return
                        stage = sample[phase]
                        if stage is None:
                            continue
                        if phase == "grade" and sample["actor"]["state"] != "succeeded":
                            if sample["actor"]["state"] in DONE:
                                stage.update(state="skipped", error="Actor did not succeed")
                            continue
                        try:
                            self._advance(document, sample, phase)
                        except (httpx.HTTPError, ValueError, KeyError) as error:
                            # Do not turn a polling failure into a new execution.
                            stage["last_error"] = str(error)
                            self._save(document)
                stages = [sample[phase] for sample in document["samples"]
                          for phase in ("actor", "grade") if sample[phase] is not None]
                document["ready"] = all(stage["state"] in DONE for stage in stages)
                if document["ready"]:
                    document.setdefault("completed_at", time.time())
                    document["status"] = ("cancelled" if self._cancelled(document) else
                                          "failed" if any(stage["state"] != "succeeded" for stage in stages)
                                          else "completed")
                else:
                    document["status"] = ("quarantined" if any(stage["state"] == "quarantined" for stage in stages)
                                          else "running")
                self._save(document)
        finally:
            self._tick_lock.release()

    def sample_actor(self, group_id: str, sample_id: str) -> dict:
        document = self.ledger.get(group_id)["document"]
        for sample in document["samples"]:
            if sample["sample_slot_id"] == sample_id:
                if not sample["actor"]["job_id"]:
                    raise ValueError("Sample has not received a Run Store job ID")
                return sample["actor"]
        raise KeyError(sample_id)

    def decide_deferred(self, group_id: str, sample_id: str, decision: dict) -> dict:
        """Atomically consume one driver-internal deferred training slot.

        The policy chooses a recovery point; the driver enforces group quota,
        source ownership and durable idempotency.  It never chooses a point on
        the policy's behalf.
        """
        expected = (
            {"kind", "source_sample_slot_id", "point_id", "overrides"}
            if isinstance(decision, dict) and decision.get("kind") == "branch"
            else {"kind"}
        )
        if not isinstance(decision, dict) or set(decision) != expected:
            raise ValueError(
                "Policy decision must be root/skip or a branch with source_sample_slot_id, point_id and overrides"
            )
        kind = decision.get("kind")
        if kind not in {"root", "branch", "skip"}:
            raise ValueError("Policy decision kind must be root, branch or skip")
        normalized = deepcopy(decision)
        if kind == "branch":
            source_id = normalized.get("source_sample_slot_id")
            point_id = normalized.get("point_id")
            if not isinstance(source_id, str) or not source_id or not isinstance(point_id, str) or not point_id:
                raise ValueError("Branch admission requires source sample and recovery point IDs")
            overrides = normalized.get("overrides")
            if not isinstance(overrides, dict) or set(overrides) - {
                "prompt", "model", "timeout_s", "budget_usd"
            }:
                raise ValueError("Invalid branch admission overrides")
        with self._tick_lock:
            row = self.ledger.get(group_id)
            document = row["document"]
            if row["cancel_requested"] or row["terminal"]:
                raise Conflict("Cannot admit a sample into a cancelled or terminal group")
            sample = next((item for item in document["samples"]
                           if item["sample_slot_id"] == sample_id), None)
            if sample is None:
                raise KeyError(sample_id)
            actor = sample["actor"]
            if actor.get("policy_decision") is not None:
                if canonical(actor["policy_decision"]) != canonical(normalized):
                    raise Conflict("Sample slot already has a different policy decision")
                return self._view(self.ledger.get(group_id))
            if actor["state"] != "deferred" or actor.get("template") is None:
                raise Conflict("Sample slot is not deferred for policy scheduling")
            template = actor["template"]
            if kind == "skip":
                actor.update(
                    state="skipped",
                    operation="skip",
                    policy_decision=normalized,
                    error="Training policy released this sample slot",
                )
            elif kind == "root":
                actor.update(state="planned", operation="run",
                             submission=template, policy_decision=normalized)
            else:
                source = next((item for item in document["samples"]
                               if item["sample_slot_id"] == normalized["source_sample_slot_id"]), None)
                if source is None or source is sample:
                    raise ValueError("Branch source must be another sample in this group")
                parent = source["actor"]
                if not parent.get("job_id") or not parent.get("attempt_id"):
                    raise ValueError("Branch source has no durable execution attempt")
                points = self.client.points(parent["job_id"], attempt_id=parent["attempt_id"])
                point = next((item for item in points if item.get("id") == normalized["point_id"]), None)
                if point is None or point.get("available") is not True:
                    raise ValueError("Branch recovery point is absent or unavailable")
                contract = deepcopy(
                    template["spec"].get("extra", {}).get("rollout_contract") or {}
                )
                if contract.get("session_id"):
                    model_position = point.get("model_position")
                    if (
                        not isinstance(model_position, dict)
                        or model_position.get("session_id") != contract["session_id"]
                        or not isinstance(model_position.get("response_id"), str)
                        or not model_position["response_id"]
                    ):
                        raise ValueError(
                            "Miles-backed branch point has no matching model position"
                        )
                parent_submission = parent.get("submission") or {}
                inherited = deepcopy(
                    parent_submission.get("spec", {}).get("extra", {}).get(
                        "rollout_contract"
                    )
                    or contract
                )
                contract_overrides = {
                    name: contract.get(name)
                    for name in ("max_model_calls", "max_tool_calls")
                    if contract.get(name) != inherited.get(name)
                }
                context = deepcopy(template["context"])
                if contract_overrides:
                    context.setdefault("rl_driver", {})[
                        "rollout_contract_overrides"
                    ] = contract_overrides
                actor.update(
                    state="planned",
                    operation="branch",
                    submission={
                        "job_id": parent["job_id"],
                        "point_id": point["id"],
                        "overrides": normalized["overrides"],
                        "context": context,
                    },
                    origin={"job_id": parent["job_id"], "point_id": point["id"]},
                    policy_decision=normalized,
                )
            actor.pop("template", None)
            self._save(document)
        self.wakeup.set()
        return self.get(group_id)

    def reconcile_submission(self, group_id: str, sample_id: str, phase: str, job_id: str) -> dict:
        """Attach a known accepted job after cancellation plus a lost HTTP reply.

        The upstream idempotency key proves it belongs to this exact submission.
        This operation never submits work or changes an execution's state.
        """
        if phase not in {"actor", "grade"}:
            raise ValueError("phase must be actor or grade")
        with self._tick_lock:
            document = self.ledger.get(group_id)["document"]
            sample = next((s for s in document["samples"] if s["sample_slot_id"] == sample_id), None)
            if sample is None:
                raise KeyError(sample_id)
            stage = sample[phase]
            if stage is None or stage["state"] != "submitting" or stage["job_id"]:
                raise ValueError("Only an unacknowledged submission can be reconciled")
            job = self.client.get(job_id)
            expected = submission_key(group_id, sample_id, phase)
            if job.get("idempotency_key") != expected:
                raise ValueError("Job does not match this submission's idempotency key")
            stage.update(job_id=job["id"], state=job["state"], attempt_id=job.get("active_attempt"),
                         phase=job.get("phase"), error=job.get("error"))
            if job["state"] in DONE:
                stage["result"] = job.get("result")
            stage.pop("last_error", None)
            stage.pop("submission_uncertain", None)
            self._save(document)
        self.wakeup.set()
        return self.get(group_id)
