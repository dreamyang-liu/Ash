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
            validate_reviewer_environment(config)

    def _save(self, document):
        self.ledger.save(document["rollout_job_id"], document, terminal=False)

    def _finish(self, document, reason, error=None):
        state = document["branching"]
        future = self._futures.get(document["rollout_job_id"])
        draining = future is not None and not future.done() and not future.cancel()
        state.update(phase="draining_review" if draining else "finished", stop_reason=reason, error=error)
        if future is not None and future.cancelled() and state["reviews"]:
            state["reviews"][-1]["state"] = "cancelled"
        if not draining:
            self._futures.pop(document["rollout_job_id"], None)
        self._save(document)

    def _evidence(self, document, config):
        attempts = []
        per_attempt_limit = max(4000, 40000 // len(document["samples"]))
        for sample in document["samples"]:
            actor = sample["actor"]
            grade = verdict(sample)
            points = self.client.points(actor["job_id"], attempt_id=actor["attempt_id"])
            output = actor.get("result") or {}
            cap = output.get("max_sequence_tokens")
            lengths = output.get("training_point_tokens") or {}
            cut_depth = (output.get("sequence_truncation") or {}).get("tool_depth")
            available = [
                {
                    **{key: point[key] for key in ("id", "message_step", "tool_depth", "snapshot_id") if key in point},
                    **({"sequence_tokens": lengths[str(point["tool_depth"])]}
                       if str(point["tool_depth"]) in lengths else {}),
                    **({"remaining_sequence_tokens": cap - lengths[str(point["tool_depth"])] - 1024}
                       if cap is not None and str(point["tool_depth"]) in lengths else {}),
                }
                for point in points
                if point.get("available") is True
                and (cut_depth is None or point["tool_depth"] <= cut_depth)
                and (cap is None or lengths.get(str(point["tool_depth"]), cap) < cap - 1024)
            ]
            tools = self.client.all_tools(actor["job_id"], attempt_id=actor["attempt_id"])
            attempts.append({
                "job_id": actor["job_id"], "resolved": grade,
                "messages": _bounded_messages(output.get("training_messages") or [], limit=per_attempt_limit),
                "tool_steps": _bounded_steps(tools, limit=per_attempt_limit),
                "grader": deepcopy(sample["grade"].get("result")),
                "available_points": available,
                "prior_hint": actor.get("submission", {}).get("overrides", {}).get("prompt"),
            })
        round_index = len(document["branching"]["reviews"])
        root_resolved = verdict(document["samples"][0])
        return {
            "task": document["message_request"]["prompt"],
            "target_resolved": document["branching"]["target_resolved"],
            "root_resolved": root_resolved,
            "round": round_index + 1,
            "branch_limit": config.limits_for(root_resolved)[round_index],
            "max_sequence_tokens": document["message_request"].get("max_sequence_tokens"),
            "attempts": attempts,
        }

    def _start_review(self, document, config):
        evidence = self._evidence(document, config)
        if not any(attempt["available_points"] for attempt in evidence["attempts"]):
            self._finish(document, "no_available_recovery_point")
            return
        state = document["branching"]
        state["reviews"].append({
            "round": evidence["round"], "branch_limit": evidence["branch_limit"],
            "state": "pending", "evidence": evidence,
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

    def _append_branches(self, document, review):
        branches = review["plan"]["branches"]
        if not branches:
            review["state"] = "declined"
            self._finish(document, "reviewer_no_branches")
            return
        # Validate the whole round before appending any child. Several children
        # may share a point; each still has its own durable submission identity.
        available = {}
        for branch in branches:
            parent_id = branch["job_id"]
            if parent_id not in available:
                parent = next(s["actor"] for s in document["samples"] if s["actor"]["job_id"] == parent_id)
                points = self.client.points(parent_id, attempt_id=parent["attempt_id"])
                available[parent_id] = {p["id"] for p in points if p.get("available") is True}
            if branch["point_id"] not in available[parent_id]:
                raise ValueError("Selected recovery point became unavailable after review")
        specs = [
            {
                "sample_slot_id": f'{document["rollout_job_id"]}:branch:{review["round"]}:{index}',
                "branch": {
                    "job_id": branch["job_id"], "point_id": branch["point_id"],
                    "overrides": {"prompt": branch["hint"]},
                },
                "grade": deepcopy(document["samples"][0]["grade"]["template"]),
            }
            for index, branch in enumerate(branches, 1)
        ]
        body = validate_request({
            "rollout_job_id": document["rollout_job_id"],
            "prompt_group_id": document["prompt_group_id"], "samples": specs,
        })
        existing = {sample["sample_slot_id"] for sample in document["samples"]}
        for index, child in enumerate(initial_document(body)["samples"], 1):
            if child["sample_slot_id"] not in existing:
                child.update(branch_round=review["round"], branch_index=index)
                document["samples"].append(child)
        review["branch_count"] = len(branches)
        review["point_count"] = len({(b["job_id"], b["point_id"]) for b in branches})
        review["state"] = "submitted"
        document["branching"]["phase"] = "branching"
        self._save(document)

    def _poll_review(self, document, config):
        review = document["branching"]["reviews"][-1]
        if review["state"] == "accepted":
            self._append_branches(document, review)
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
        self._append_branches(document, review)

    def advance(self, document, *, cancelled=False):
        state = document["branching"]
        if state["phase"] == "finished":
            return
        if state["phase"] == "draining_review":
            future = self._futures.get(document["rollout_job_id"])
            if future is None or future.done():
                review = state["reviews"][-1]
                review["state"] = "discarded"
                if future is not None and not future.cancelled():
                    try:
                        review["discarded_output"] = future.result()
                    except Exception as error:
                        review["discarded_error"] = str(error)
                self._futures.pop(document["rollout_job_id"], None)
                state["phase"] = "finished"
                self._save(document)
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
        # All selected actors AND graders above are terminal before evaluating
        # the signal mix. A positive arriving early never cancels its siblings.
        if len(set(values)) == 2:
            self._finish(document, "mixed_rewards")
        elif len(state["reviews"]) >= len(config.limits_for(values[0])):
            self._finish(document, "round_limit")
        else:
            self._start_review(document, config)
