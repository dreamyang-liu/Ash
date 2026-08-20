"""ToolResult parsing (ash_sandbox/result.py).

`from_response` is the one parser for every transport's tools/call result — four
backends hand-rolled it and drifted the moment the shape grew fields. It also
unpacks the runtime's CommandOutcome schema, which `shell` and `process read`
both answer in.

Covered:
- envelope: text slot, isError, notifications
- a CommandOutcome in the text slot becomes first-class fields
- detection is by shape: a JSON document a *command* printed is not an outcome
- exit_code None vs 0 stay distinguishable
- a plain success carries no outcome fields
"""

from __future__ import annotations

import json

from ash_sandbox.result import ToolResult


def _payload(text: str = "hello", **overrides) -> dict:
    base = {
        "content": [{"type": "text", "text": text}],
        "isError": False,
        "notifications": [],
    }
    base.update(overrides)
    return base


def _outcome(**fields) -> str:
    base = {"stdout": "", "stderr": "", "exit_code": 0,
            "stdout_bytes": 0, "stderr_bytes": 0,
            "stdout_truncated": False, "stderr_truncated": False}
    base.update(fields)
    return json.dumps(base)


# --------------------------------------------------------------------------- #
#  Envelope
# --------------------------------------------------------------------------- #

def test_parses_text_and_error_flag():
    result = ToolResult.from_response(_payload(isError=True))
    assert result.output == "hello" and result.is_error


def test_missing_content_is_empty_output():
    result = ToolResult.from_response({"isError": True})
    assert result.output == "" and result.is_error


def test_notifications_ride_through():
    payload = _payload(notifications=[{"kind": "process_exited", "data": {}}])
    assert ToolResult.from_response(payload).notifications == [
        {"kind": "process_exited", "data": {}}]


# --------------------------------------------------------------------------- #
#  CommandOutcome unpacking
# --------------------------------------------------------------------------- #

def test_an_outcome_becomes_first_class_fields():
    text = _outcome(stdout="collected 42\n", stderr="AssertionError\n",
                    exit_code=1, stdout_bytes=13, stderr_bytes=15)
    result = ToolResult.from_response(_payload(text, isError=True))
    assert result.exit_code == 1
    assert result.stdout == "collected 42\n"
    assert result.stderr == "AssertionError\n"          # streams stay separate
    assert result.stderr_bytes == 15
    assert not result.truncated


def test_truncation_and_timeout_are_flags_not_prose():
    text = _outcome(stdout="partial\n", exit_code=137, timed_out=True,
                    stdout_bytes=419430400, stdout_truncated=True)
    result = ToolResult.from_response(_payload(text, isError=True))
    assert result.timed_out and result.truncated
    assert result.stdout_bytes == 419430400


def test_a_running_process_has_no_exit_code_yet():
    """None and 0 are different answers, so `is None` is the right test."""
    text = _outcome(stdout="so far\n", exit_code=None, running=True)
    result = ToolResult.from_response(_payload(text))
    assert result.running and result.exit_code is None

    done = ToolResult.from_response(_payload(_outcome(exit_code=0)))
    assert done.exit_code == 0 and not done.running


def test_a_plain_success_carries_no_outcome_fields():
    """`echo hello` has nothing to report beyond its stdout, so the runtime
    sends bare text and the outcome fields stay unset."""
    result = ToolResult.from_response(_payload("hello_ci\n"))
    assert result.output == "hello_ci\n"
    assert result.stdout is None and result.exit_code is None


# --------------------------------------------------------------------------- #
#  Detection is by shape, not by tool name
# --------------------------------------------------------------------------- #

def test_json_a_command_printed_is_not_mistaken_for_an_outcome():
    """`cat package.json` returns JSON as its output. Treating that as the
    runtime's own report would silently replace the model's view of it."""
    printed = json.dumps({"stdout": "x", "stderr": "y", "name": "my-package"})
    result = ToolResult.from_response(_payload(printed))
    assert result.stdout is None                        # not unpacked
    assert result.output == printed                     # passed through


def test_the_background_pid_payload_is_not_an_outcome():
    """`shell --background` returns {"pid": ...} — a different shape."""
    result = ToolResult.from_response(_payload('{"pid":"a1b2c3"}'))
    assert result.stdout is None
    assert json.loads(result.output)["pid"] == "a1b2c3"


def test_both_streams_must_be_present_to_be_an_outcome():
    """An outcome always reports both streams. A payload borrowing one field name
    is not one — accepting it would fabricate empty streams for a command whose
    real output was that JSON."""
    partial = json.dumps({"exit_code": 0})           # a valid field, no streams
    result = ToolResult.from_response(_payload(partial))
    assert result.stdout is None and result.exit_code is None
    assert result.output == partial

    one_stream = json.dumps({"stdout": "x", "exit_code": 0})
    assert ToolResult.from_response(_payload(one_stream)).stdout is None


def test_non_json_and_malformed_json_pass_through():
    for text in ("plain output", "{not json", "[1,2,3]", ""):
        result = ToolResult.from_response(_payload(text))
        assert result.stdout is None and result.output == text


def test_every_backend_parses_through_from_response():
    """Four backends once hand-rolled this parse and drifted when the shape grew
    fields — the outcome silently stopped arriving while every unit test passed,
    because they construct ToolResult directly. Grep is the cheap guard."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] /
              "ash_sandbox" / "backends.py").read_text()
    assert source.count("ToolResult.from_response") == 4
    assert 'result["content"][0]["text"]' not in source, \
        "a backend is parsing the envelope by hand again"
