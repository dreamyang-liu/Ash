from copy import deepcopy
import json
from threading import Event
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from rl_driver.branch_review import BranchingConfig, review_prompt
from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.miles import MilesAdapter, internal_id
from rl_driver.server import create_app
from rl_driver.tests.test_driver import peer as peer
from rl_driver.tests.test_message_rollout import config, messages, request
from runstore.message_export import mark_hint


HINT = "Inspect the boundary conversion before changing the parser."


def pair_request(group="branch-test"):
    body = request()
    body.update(rollout_job_id=group, branching=True, max_samples=2, minimum_returned_samples=2)
    body["sample_slots"] = [
        {"sample_slot_id": f"{group}:slot:{i}", "sample_index": i} for i in range(2)
    ]
    return body


def plan(evidence, hint=HINT):
    attempt = next(a for a in evidence["attempts"] if a["available_points"])
    return {
        "job_id": attempt["job_id"], "point_id": attempt["available_points"][-1]["id"],
        "reason": "Follow the retained code path", "hint": hint,
    }


def spin(driver, condition):
    deadline = time.monotonic() + 3
    while not condition():
        assert time.monotonic() < deadline, driver.ledger.active()
        driver.tick()
        time.sleep(0.001)


class Scenario:
    def __init__(self, tmp_path, peer, *, reviewer=None, settings=None, body=None):
        self.queue, self.client = peer
        self.reviews = []
        self.body = body or pair_request()
        self.path = tmp_path / "ledger"
        self.settings = config(self.body)
        self.settings["branching"] = {
            "reviewer_model": "test-reviewer", **(settings or {}),
        }

        def review(configuration, evidence):
            self.reviews.append(deepcopy(evidence))
            return reviewer(configuration, evidence) if reviewer else plan(evidence)

        self.driver = Driver(self.client, Ledger(self.path), branch_reviewer=review)
        self.adapter = MilesAdapter(self.driver, self.settings)

    def start(self):
        self.adapter.submit(self.body)
        self.driver.tick()
        assert len(self.actors()) == 1
        assert not self.reviews

    def actors(self):
        return [key for key, job in self.queue.jobs.items() if job["kind"] == "rollout"]

    def finish(self, actor_id, resolved, *, points=True):
        self.queue.finish(actor_id)
        output = self.queue.jobs[actor_id]["result"]
        output.update(
            training_messages=messages(), final_snapshot_id=f"snapshot-{actor_id}",
            rollout_usage={"model_calls": 3, "tool_calls": 1},
        )
        if actor_id != self.actors()[0]:
            output["training_messages"].insert(-1, {"role": "user", "content": mark_hint(HINT)})
            output["training_origin"] = {"job_id": self.actors()[0], "hint": HINT, "prompt": HINT}
        if points:
            self.queue.final_point(actor_id)
        self.driver.tick()
        grade = next(key for key, job in self.queue.jobs.items()
                     if job["kind"] == "grade" and job["state"] == "queued")
        self.queue.finish(grade, resolved=resolved)
        self.driver.tick()

    def wait_branch(self, count):
        spin(self.driver, lambda: len(self.actors()) == count)
        return self.actors()[-1]

    def result(self):
        spin(self.driver, lambda: self.driver.get(internal_id(self.body["rollout_job_id"]))["ready"])
        return self.adapter.get(self.body["rollout_job_id"])


@pytest.mark.parametrize("root_reward", [False, True])
def test_root_grade_controls_review_direction_and_first_opposite_stops(tmp_path, peer, root_reward):
    scenario = Scenario(tmp_path, peer)
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], root_reward)
        child = scenario.wait_branch(2)
        assert scenario.reviews[0]["target_resolved"] is not root_reward
        assert scenario.reviews[0]["attempts"][0]["resolved"] is root_reward
        assert scenario.reviews[0]["attempts"][0]["tool_steps"][-1]["depth"] == 3
        scenario.finish(child, not root_reward)
        result = scenario.result()
        assert result["status"] == "completed", result
        assert [t["reward"] for t in result["trajectories"]] == [float(root_reward), float(not root_reward)]
        assert result["actual_samples"] == 2 and result["search_branches"] == 1
        assert len(scenario.reviews) == 1 and len(scenario.actors()) == 2
        assert result["trajectories"][1]["parent_branch_id"] == scenario.actors()[0]
        assert result["trajectories"][1]["metadata"]["branching"]["stop_reason"] == "target_found"
        assert HINT not in json.dumps(result)
        assert "<ash_training_hint>" not in json.dumps(result)
        branch = next(row for row in scenario.queue.requests if row[1].endswith("/branch"))
        assert branch[2]["overrides"] == {"prompt": HINT}
        assert branch[2]["point_id"] == "last"
    finally:
        scenario.driver.close()


@pytest.mark.parametrize("last_reward", [False, True])
def test_two_round_bound_and_pair_selection_keep_intermediate_attempts(tmp_path, peer, last_reward):
    scenario = Scenario(tmp_path, peer)
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False)
        first_child = scenario.wait_branch(2)
        scenario.finish(first_child, False)
        second_child = scenario.wait_branch(3)
        assert len(scenario.reviews[1]["attempts"]) == 2
        assert scenario.reviews[1]["attempts"][1]["prior_hint"] == HINT
        scenario.finish(second_child, last_reward)
        result = scenario.result()
        assert result["status"] == "completed", result
        assert result["search_branches"] == 2 and result["actual_samples"] == 2
        assert [t["branch_id"] for t in result["trajectories"]] == [scenario.actors()[0], second_child]
        assert result["consumed_budget"] == {"model_calls": 9, "tool_calls": 3}
        state = scenario.driver.ledger.get(internal_id(scenario.body["rollout_job_id"]))["document"]
        assert len(state["samples"]) == 3
        assert state["branching"]["stop_reason"] == ("target_found" if last_reward else "round_limit")
    finally:
        scenario.driver.close()


def test_success_can_be_configured_to_run_both_rounds(tmp_path, peer):
    scenario = Scenario(tmp_path, peer, settings={"stop_on_negative": False})
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], True)
        first_child = scenario.wait_branch(2)
        scenario.finish(first_child, False)
        second_child = scenario.wait_branch(3)
        scenario.finish(second_child, False)
        assert scenario.result()["search_branches"] == 2
    finally:
        scenario.driver.close()


def test_all_mode_returns_only_real_trajectories_with_no_padding(tmp_path, peer):
    body = pair_request()
    body.update(max_samples=3, minimum_returned_samples=1)
    body["sample_slots"].append({"sample_slot_id": "third", "sample_index": 2})
    scenario = Scenario(tmp_path, peer, body=body, settings={"return_mode": "all"})
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False)
        scenario.finish(scenario.wait_branch(2), True)
        result = scenario.result()
        assert result["status"] == "completed"
        assert result["actual_samples"] == 2 and result["max_samples"] == 3
    finally:
        scenario.driver.close()


def test_no_point_returns_explicit_failure_without_blind_retry(tmp_path, peer):
    scenario = Scenario(tmp_path, peer)
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False, points=False)
        result = scenario.result()
        assert result["status"] == "failed" and result["actual_samples"] == 1
        assert "no_available_recovery_point" in result["stop_reason"]
        assert len(scenario.actors()) == 1 and not scenario.reviews
    finally:
        scenario.driver.close()


@pytest.mark.parametrize("change", [
    {"point_id": "missing"},
    {"job_id": "another-parent"},
    {"hint": "<ash_training_hint>hidden</ash_training_hint>"},
    {"hint": ""},
])
def test_invalid_review_never_launches_a_branch(tmp_path, peer, change):
    scenario = Scenario(tmp_path, peer, reviewer=lambda _config, e: {**plan(e), **change})
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False)
        result = scenario.result()
        assert result["status"] == "failed" and "review_failed" in result["stop_reason"]
        assert len(scenario.actors()) == 1
    finally:
        scenario.driver.close()


def test_slow_review_does_not_block_other_roots_and_cancel_discards_it(tmp_path, peer):
    release = Event()

    def reviewer(_config, evidence):
        assert release.wait(3)
        return plan(evidence)

    scenario = Scenario(tmp_path, peer, reviewer=reviewer)
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False)
        spin(scenario.driver, lambda: bool(scenario.reviews))
        second = pair_request("second")
        scenario.adapter.submit(second)
        scenario.driver.tick()
        assert len(scenario.actors()) == 2
        scenario.adapter.release(scenario.body["rollout_job_id"])
        scenario.driver.tick()
        release.set()
        scenario.driver.tick()
        first = scenario.driver.get(internal_id(scenario.body["rollout_job_id"]))
        assert first["branching"]["stop_reason"] == "cancelled"
        assert not any(row[1].endswith("/branch") for row in scenario.queue.requests)
    finally:
        release.set()
        scenario.driver.close()


def test_accepted_review_and_lost_branch_reply_survive_restart(tmp_path, peer, monkeypatch):
    scenario = Scenario(tmp_path, peer)
    restored = None
    try:
        scenario.start()
        scenario.finish(scenario.actors()[0], False)

        def disconnected(*_args):
            raise httpx.ReadError("temporarily disconnected")

        monkeypatch.setattr(scenario.driver.branching, "_append_branch", disconnected)
        group = internal_id(scenario.body["rollout_job_id"])
        spin(scenario.driver, lambda: scenario.driver.ledger.get(group)["document"]["branching"]["reviews"][-1]["state"] == "accepted")
        scenario.driver.close()

        def unexpected_review(*_args):
            raise AssertionError("Accepted review must not be repeated")

        restored = Driver(scenario.client, Ledger(scenario.path), branch_reviewer=unexpected_review)
        restored.tick()
        scenario.queue.lose_ack = True
        restored.tick()
        assert len(scenario.actors()) == 2
        restored.close()
        restored = Driver(scenario.client, Ledger(scenario.path), branch_reviewer=unexpected_review)
        restored.tick()
        assert len(scenario.actors()) == 2
        assert restored.get(group)["samples"][-1]["actor"]["job_id"] == scenario.actors()[-1]
    finally:
        scenario.driver.close()
        if restored:
            restored.close()


def test_expired_execution_deadline_prevents_review_and_branch(tmp_path, peer, monkeypatch):
    scenario = Scenario(tmp_path, peer)
    try:
        scenario.start()
        group = internal_id(scenario.body["rollout_job_id"])
        deadline = scenario.driver.ledger.get(group)["document"]["execution_deadline_at"]
        monkeypatch.setattr("rl_driver.branching.time.time", lambda: deadline + 1)
        scenario.finish(scenario.actors()[0], False)
        result = scenario.result()
        assert "execution_deadline" in result["stop_reason"]
        assert not scenario.reviews and len(scenario.actors()) == 1
    finally:
        scenario.driver.close()


def test_missing_reviewer_and_slot_mismatch_fail_before_any_execution(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    body = pair_request()
    adapter = MilesAdapter(driver, config(body))
    with pytest.raises(ValueError, match="reviewer_model"):
        adapter.submit(body)
    body["max_samples"] = body["minimum_returned_samples"] = 1
    settings = config(body)
    settings["branching"] = {"enabled": True, "reviewer_model": "test-reviewer"}
    with pytest.raises(ValueError, match="exactly two"):
        MilesAdapter(driver, settings).submit(body)
    assert not queue.jobs


def test_missing_reviewer_credentials_fail_before_root_submission(tmp_path, peer, monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    queue, client = peer
    body = pair_request()
    settings = config(body)
    settings["branching"] = {"reviewer_model": "test-model"}
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    with pytest.raises(ValueError, match="driver environment"):
        MilesAdapter(driver, settings).submit(body)
    assert not queue.jobs


def test_invalid_grader_result_is_not_a_negative_training_signal(tmp_path, peer):
    scenario = Scenario(tmp_path, peer)
    try:
        scenario.start()
        actor = scenario.actors()[0]
        scenario.queue.finish(actor)
        scenario.queue.jobs[actor]["result"].update(
            training_messages=messages(), final_snapshot_id="root-snapshot",
        )
        scenario.queue.final_point(actor)
        scenario.driver.tick()
        grade = next(key for key, job in scenario.queue.jobs.items() if job["kind"] == "grade")
        scenario.queue.finish(grade, resolved=False)
        scenario.queue.jobs[grade]["result"].update(status="error", error="verifier unavailable")
        scenario.driver.tick()
        state = scenario.driver.get(internal_id(scenario.body["rollout_job_id"]))
        assert state["branching"]["stop_reason"] == "execution_or_grading_failed"
        assert not scenario.reviews and len(scenario.actors()) == 1
    finally:
        scenario.driver.close()


def test_point_revoked_while_reviewing_is_not_replaced(tmp_path, peer):
    release = Event()

    def reviewer(_config, evidence):
        assert release.wait(3)
        return plan(evidence)

    scenario = Scenario(tmp_path, peer, reviewer=reviewer)
    try:
        scenario.start()
        root = scenario.actors()[0]
        scenario.finish(root, False)
        spin(scenario.driver, lambda: bool(scenario.reviews))
        scenario.queue.points[root][-1]["available"] = False
        release.set()
        result = scenario.result()
        assert result["status"] == "failed" and "branch_planning_failed" in result["stop_reason"]
        assert len(scenario.actors()) == 1
    finally:
        release.set()
        scenario.driver.close()


def test_success_and_failure_review_prompts_have_distinct_goals():
    evidence = {"target_resolved": True, "attempts": []}
    assert "concrete repair direction" in review_prompt(evidence)
    evidence["target_resolved"] = False
    prompt = review_prompt(evidence)
    assert "plausible mistaken reasoning" in prompt and "not arbitrary\nsabotage" in prompt


@pytest.mark.parametrize("body", [
    {"enabled": "yes"}, {"max_rounds": True}, {"max_rounds": 0},
    {"reviewer_timeout_s": float("inf")}, {"reviewer_workers": 0},
    {"return_mode": "duplicate"}, {"unknown": 1},
])
def test_invalid_branching_configuration_is_rejected(body):
    with pytest.raises(ValueError):
        BranchingConfig.from_dict(body)
