"""Separate benchmark outcomes, task failures and batch safety failures."""


def classify_worker_exit(exit_code: int, worker: dict | None) -> str:
    if not worker or not worker.get("finished_at"):
        return "global_hold"
    if worker.get("cleanup_error") or worker.get("remaining_owned_sandboxes") != []:
        return "global_hold"
    if worker.get("status") == "completed" and exit_code in (0, 1):
        return "completed"
    if worker.get("status") == "held" and worker.get("failure_scope") == "task" and exit_code == 3:
        return "task_hold"
    return "global_hold"


def grading_failure_scope(actor_status: str, actor_error: str | None, grade) -> tuple[str | None, str]:
    if actor_status != "completed":
        return "task", "actor did not complete cleanly: " + str(actor_error or actor_status)
    if grade.error:
        return "task", "attempt is ungradable: " + grade.error
    if not grade.resolved and "leap-second file is expired" in grade.detail:
        return "task", "known expired leap-second test environment"
    if grade.verifier_artifact_error or not grade.verifier_artifacts:
        return "global", "verifier evidence export incomplete: " + str(grade.verifier_artifact_error)
    return None, ""
