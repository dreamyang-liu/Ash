from copy import deepcopy

import pytest

from harness.tests.test_assistant_turn import assistant_turn
from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.specs import validate_request
from rl_driver.tests.test_driver import peer


def branch_request():
    return {"rollout_job_id": "guided", "prompt_group_id": "prompt",
            "samples": [{"sample_slot_id": "branch-0", "branch": {
                "job_id": "parent", "point_id": "closed-point",
                "overrides": {"assistant_turn": assistant_turn()}}}]}


def test_assistant_turn_survives_driver_http_dispatch(tmp_path, peer):
    queue, client = peer
    request = branch_request()
    original = deepcopy(request)
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    driver.submit(request)
    driver.tick()
    calls = [row for row in queue.requests if row[0] == "POST" and row[1].endswith("/branch")]
    assert len(calls) == 1
    assert calls[0][2] == {
        "point_id": "closed-point", "overrides": original["samples"][0]["branch"]["overrides"]}
    assert request == original


@pytest.mark.parametrize("invalid", ["tool", "missing", "mixed"])
def test_bad_guidance_rejected_before_driver_submission(invalid):
    request = branch_request()
    overrides = request["samples"][0]["branch"]["overrides"]
    if invalid == "tool":
        overrides["assistant_turn"]["tool_calls"][0]["function"]["name"] = "python"
    elif invalid == "missing":
        overrides.clear()
        overrides["branch_guidance"] = "assistant-turn"
    else:
        overrides["prompt"] = "private user hint"
    with pytest.raises(ValueError):
        validate_request(request)
