"""One restartable coordinator; all agent and grader jobs execute elsewhere."""

from copy import deepcopy
import threading
import time

import httpx

from runstore.specs import JobSpec
from rl_driver.ledger import Ledger
from rl_driver.specs import initial_document, submission_key, validate_request

DONE = {"succeeded", "failed", "cancelled", "skipped"}


class Driver:
    def __init__(self, client, ledger: Ledger):
        self.client = client
        self.ledger = ledger
        self._tick_lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stopping = threading.Event()

    def submit(self, body: dict, *, source_request: dict | None = None, extra_document: dict | None = None) -> dict:
        request = validate_request(body)
        document = {**initial_document(request), **(extra_document or {})}
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
                                            idempotency_key=key, **branch["overrides"])
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
        if job["state"] == "queued" and self._cancelled(document):
            self.client.cancel_queued(stage["job_id"])
            job = self.client.get(stage["job_id"])
        stage.update(state=job["state"], attempt_id=job.get("active_attempt"),
                     phase=job.get("phase"), error=job.get("error"))
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
        if document.get("message_request") and output.get("training_snapshot_error"):
            raise ValueError(output["training_snapshot_error"])
        if document.get("message_request") and output.get("final_snapshot_id"):
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
