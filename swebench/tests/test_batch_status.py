from types import SimpleNamespace

import pytest

from swebench.batch_status import classify_worker_exit, grading_failure_scope


def finished(**changes):
    return {"status": "completed", "finished_at": "recorded", "remaining_owned_sandboxes": [], **changes}


def grade(**changes):
    return SimpleNamespace(**{"resolved": False, "error": None, "detail": "assertion failed",
                              "verifier_artifacts": "/logs", "verifier_artifact_error": None, **changes})


@pytest.mark.parametrize("code", [0, 1])
def test_valid_completion_counts_regardless_of_resolution(code):
    assert classify_worker_exit(code, finished(resolved_any=code == 0)) == "completed"


def test_clean_task_hold_does_not_stop_batch():
    assert classify_worker_exit(3, finished(status="held", failure_scope="task")) == "task_hold"


@pytest.mark.parametrize("code,worker", [
    (1, None), (0, {}), (2, finished()),
    (3, finished(status="held", failure_scope="global")),
    (0, finished(remaining_owned_sandboxes=["leaked"])),
    (3, finished(status="held", failure_scope="task", cleanup_error="API unavailable")),
    (0, finished(finished_at=None)),
])
def test_unproven_completion_or_shared_failure_stops_batch(code, worker):
    assert classify_worker_exit(code, worker) == "global_hold"


def test_timeout_is_task_local_even_without_verifier_archive():
    scope, reason = grading_failure_scope("error", "timed out after 1800.0s", grade(verifier_artifacts=None))
    assert scope == "task" and "1800" in reason


def test_missing_final_snapshot_is_task_local_not_relaxed_grading():
    scope, reason = grading_failure_scope("completed", None, grade(error="final step has no exact snapshot", verifier_artifacts=None))
    assert scope == "task" and "no exact snapshot" in reason


def test_assertion_failure_continues_to_branching():
    assert grading_failure_scope("completed", None, grade()) == (None, "")


def test_lost_grader_logs_remain_a_global_safety_stop():
    assert grading_failure_scope("completed", None, grade(verifier_artifact_error="disk full"))[0] == "global"


def test_known_date_dependent_environment_stays_task_local():
    assert grading_failure_scope("completed", None, grade(detail="leap-second file is expired"))[0] == "task"
