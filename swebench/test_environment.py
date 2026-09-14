"""Use the prepared SWE-bench conda environment for verifier commands."""

import json
import shlex


CONDA_INIT = ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
FAILURE_MARKER = "ASH_SWE_TEST_ENVIRONMENT_ERROR"


def test_command(command: str, needs_pytest: bool = True) -> str:
    prerequisite = "python -m pytest --version" if needs_pytest else "python -c 'import sys'"
    script = (f"if ! {{ {CONDA_INIT}; }}; then echo {FAILURE_MARKER} >&2; exit 86; fi; "
              f"if ! {prerequisite} >/dev/null; then echo {FAILURE_MARKER} >&2; exit 86; fi; "
              "exec sh -c " + shlex.quote(command))
    return "bash -c " + shlex.quote(script)


def validate_test_execution(result) -> None:
    text = result.output or ""
    try:
        outcome = json.loads(text)
    except (ValueError, TypeError):
        outcome = {}
    if isinstance(outcome, dict) and "exit_code" in outcome:
        text = str(outcome.get("stdout") or "") + str(outcome.get("stderr") or "")
        if outcome.get("timed_out") or outcome.get("running") or outcome["exit_code"] is None:
            raise RuntimeError("test command did not finish within its budget")
        if outcome["exit_code"] in (3, 4, 5, 86, 126, 127):
            raise RuntimeError("test runner could not execute requested tests: " + text[-2000:])
    elif not result.success:
        raise RuntimeError("test tool failed: " + str(result.error or result.output))
    if FAILURE_MARKER in text or "No module named pytest" in text:
        raise RuntimeError("test environment unavailable: " + text[-2000:])
