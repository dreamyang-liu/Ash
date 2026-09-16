"""Durable root → grade → review → branch scheduling for message rollouts."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import time

from rl_driver.branch_review import BranchingConfig, ask_reviewer, validate_plan, validate_reviewer_environment
from rl_driver.specs import initial_document, validate_request


DONE = {"succeeded", "failed", "cancelled", "skipped"}


def initial_state(config):
    return {
        "config": config.to_dict(), "phase": "root", "target_resolved": None,
        "reviews": [], "stop_reason": None, "error": None,
    }


def verdict(sample):
    grade = sample["grade"] or {}
    result = grade.get("result") or {}
    value = result.get("resolved")
    if sample["actor"]["state"] != "succeeded" or grade.get("state") != "succeeded":
        raise ValueError("Branching requires a successful actor and real grader result")
    output = sample["actor"].get("result") or {}
    if (result.get("status") != "completed" or output.get("status") not in {"completed", "truncated"}
            or output.get("training_export_error")):
        raise ValueError("Execution/export/grader errors are not negative rewards")
    if type(value) is not bool:
        raise ValueError("Branching requires grader resolved=true or resolved=false")
    return value


def selected_samples(document):
    samples = document["samples"]
    state = document.get("branching")
    if not state or state["config"]["return_mode"] == "all":
        return samples
    completed = []
    for sample in samples[1:]:
        try:
            value = verdict(sample)
        except ValueError:
            continue
        completed.append((sample, value))
    match = next((sample for sample, value in completed if value == state["target_resolved"]), None)
    child = match or (completed[-1][0] if completed else None)
    return [samples[0], child] if child is not None else [samples[0]]


def _bounded_messages(messages, limit=40000):
    """Keep both ends of long messages, then retain a bounded recent history."""
    result, remaining = [], limit
    for original in reversed(messages):
        item = deepcopy(original)
        for key in ("content", "reasoning_content"):
            text = item.get(key)
            if isinstance(text, str) and len(text) > 6000:
                item[key] = text[:3000] + "\n[review excerpt truncated]\n" + text[-3000:]
        size = len(json.dumps(item, ensure_ascii=False))
        if size > remaining:
            break
        result.append(item)
        remaining -= size
    return list(reversed(result))


def _bounded_steps(tools, limit=40000):
    result, remaining = [], limit
    for original in reversed(tools):
        row = {"depth": original["depth"]}
        for key in ("call", "response"):
            text = json.dumps(original.get(key), ensure_ascii=False)
            row[key] = text if len(text) <= 6000 else text[:3000] + "\n[truncated]\n" + text[-3000:]
        size = len(json.dumps(row, ensure_ascii=False))
        if size > remaining:
            break
        result.append(row)
        remaining -= size
    return list(reversed(result))


class BranchController:
    def __init__(self, client, ledger, *, reviewer=ask_reviewer):
        self.client, self.ledger, self.reviewer = client, ledger, reviewer
        self._executor = None
        self._futures = {}

    def close(self):
        for future in self._futures.values():
            future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def validate_config(self, config):
        config.require_reviewer()
        if self.reviewer is ask_reviewer:
            validate_reviewer_environment()

    def _save(self, document):
        self.ledger.save(document["rollout_job_id"], document, terminal=False)

    def _finish(self, document, reason, error=None):
        state = document["branching"]
        state.update(phase="finished", stop_reason=reason, error=error)
        future = self._futures.pop(document["rollout_job_id"], None)
        if future is not None:
            future.cancel()
        self._save(document)

    def _evidence(self, document):
        attempts = []
        for sample in document["samples"]:
            actor = sample["actor"]
            grade = verdict(sample)
            points = self.client.points(actor["job_id"], attempt_id=actor["attempt_id"])
            available = [
                {key: point[key] for key in ("id", "message_step", "tool_depth", "snapshot_id") if key in point}
                for point in points if point.get("available") is True
            ]
            output = actor.get("result") or {}
            tools = self.client.all_tools(actor["job_id"], attempt_id=actor["attempt_id"])
            attempts.append({
                "job_id": actor["job_id"], "resolved": grade,
                "messages": _bounded_messages(output.get("training_messages") or []),
                "tool_steps": _bounded_steps(tools),
                "grader": deepcopy(sample["grade"].get("result")),
                "available_points": available,
                "prior_hint": actor.get("submission", {}).get("overrides", {}).get("prompt"),
            })
        return {
            "task": document["message_request"]["prompt"],
            "target_resolved": document["branching"]["target_resolved"],
            "round": len(document["branching"]["reviews"]) + 1,
            "attempts": attempts,
        }

    def _start_review(self, document, config):
        evidence = self._evidence(document)
        if not any(attempt["available_points"] for attempt in evidence["attempts"]):
            self._finish(document, "no_available_recovery_point")
            return
        state = document["branching"]
        state["reviews"].append({
            "round": evidence["round"], "state": "pending", "evidence": evidence,
            "started_at": time.time(),
        })
        state["phase"] = "reviewing"
        self._save(document)
        self._launch_review(document, config)

    def _launch_review(self, document, config):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=config.reviewer_workers, thread_name_prefix="ash-branch-review",
            )
        review = document["branching"]["reviews"][-1]
        self._futures[document["rollout_job_id"]] = self._executor.submit(
            self.reviewer, config, deepcopy(review["evidence"]),
        )

    def _append_branch(self, document, review):
        plan = review["plan"]
        # Revalidate availability after review; never substitute another point.
        parent = next(s["actor"] for s in document["samples"] if s["actor"]["job_id"] == plan["job_id"])
        points = self.client.points(plan["job_id"], attempt_id=parent["attempt_id"])
        if not any(p["id"] == plan["point_id"] and p.get("available") is True for p in points):
            raise ValueError("Selected recovery point became unavailable after review")
        sample_id = f'{document["rollout_job_id"]}:branch:{review["round"]}'
        if not any(sample["sample_slot_id"] == sample_id for sample in document["samples"]):
            template = deepcopy(document["samples"][0]["grade"]["template"])
            body = validate_request({
                "rollout_job_id": document["rollout_job_id"],
                "prompt_group_id": document["prompt_group_id"],
                "samples": [{
                    "sample_slot_id": sample_id,
                    "branch": {
                        "job_id": plan["job_id"], "point_id": plan["point_id"],
                        # Worker marks the entire continuation once it restores
                        # the validated native prefix.
                        "overrides": {"prompt": plan["hint"]},
                    },
                    "grade": template,
                }],
            })
            child = initial_document(body)["samples"][0]
            child["branch_round"] = review["round"]
            document["samples"].append(child)
        review["state"] = "submitted"
        document["branching"]["phase"] = "branching"
        self._save(document)

    def _poll_review(self, document, config):
        review = document["branching"]["reviews"][-1]
        if review["state"] == "accepted":
            self._append_branch(document, review)
            return
        group = document["rollout_job_id"]
        future = self._futures.get(group)
        if future is None:
            # A restart may repeat an unacknowledged review, but an accepted
            # plan and every actor submission retain durable identities.
            self._launch_review(document, config)
            return
        if not future.done():
            return
        del self._futures[group]
        try:
            output = future.result()
            review["output"] = deepcopy(output)
            review["plan"] = validate_plan(output, review["evidence"])
        except Exception as error:
            review.update(state="failed", error=str(error))
            self._finish(document, "review_failed", str(error))
            return
        review["state"] = "accepted"
        self._save(document)
        self._append_branch(document, review)

    def advance(self, document, *, cancelled=False):
        state = document["branching"]
        if state["phase"] == "finished":
            return
        if cancelled:
            self._finish(document, "cancelled")
            return
        if time.time() >= document["execution_deadline_at"]:
            self._finish(document, "execution_deadline")
            return
        config = BranchingConfig.from_dict(state["config"])
        if state["phase"] == "reviewing":
            self._poll_review(document, config)
            return
        if any(
            sample[phase] is not None and sample[phase]["state"] not in DONE
            for sample in document["samples"] for phase in ("actor", "grade")
        ):
            return
        try:
            values = [verdict(sample) for sample in document["samples"]]
        except ValueError as error:
            self._finish(document, "execution_or_grading_failed", str(error))
            return
        state["target_resolved"] = not values[0]
        found = state["target_resolved"] in values[1:]
        if found and (state["target_resolved"] or config.stop_on_negative):
            self._finish(document, "target_found")
        elif len(state["reviews"]) >= config.max_rounds:
            self._finish(document, "round_limit")
        else:
            self._start_review(document, config)
