import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from runstore.message_export import export_messages, mark_hint
from runstore.swerebench import grade_snapshot


@pytest.mark.parametrize("fix,expected", [(False, False), (True, True)])
def test_grader_runs_hidden_tests_after_actor_and_distinguishes_fix(tmp_path, monkeypatch, fix, expected):
    """Real git/test execution; only the VM filesystem boundary is substituted."""
    repo = tmp_path / "repo"
    repo.mkdir()
    def command(*argv):
        return subprocess.check_output(argv, cwd=repo, text=True).strip()
    command("git", "init", "-q")
    (repo / "answer.py").write_text("VALUE = 0\n")
    command("git", "add", ".")
    command("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    base = command("git", "rev-parse", "HEAD")
    patch = (
        "diff --git a/test_answer.py b/test_answer.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/test_answer.py\n@@ -0,0 +1,3 @@\n"
        "+from answer import VALUE\n+\n+def test_answer(): assert VALUE == 1\n"
    )
    # A solver-created fake test must be removed before installing hidden tests.
    (repo / "test_answer.py").write_text("def test_answer(): assert True\n")
    if fix:
        (repo / "answer.py").write_text("VALUE = 1\n")
    parser = tmp_path / "parser.py"
    parser.write_text(
        "import re\n"
        "def parse(text):\n"
        " return {name:status for name,status in re.findall(r'(test_answer.py::test_answer) (PASSED|FAILED)', text)}\n"
        "NAME_TO_PARSER={'fixture':parse}\n"
    )
    lifecycle = []
    class Session:
        def create(self, snapshot, resources):
            lifecycle.append(("create", snapshot))
            return True
        def destroy(self):
            lifecycle.append(("destroy",))
    def shell(session, text, timeout=120):
        text = text.replace("cd /repo", "cd " + str(repo))
        text = text.replace("/tmp/ash-test.patch", str(tmp_path / "test.patch"))
        text = text.replace("/tmp/ash-eval.sh", str(tmp_path / "eval.sh"))
        result = subprocess.run(["bash", "-c", text], text=True, capture_output=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    def checked(session, text, timeout=120):
        stdout, stderr, code = shell(session, text, timeout)
        assert code == 0, stderr
        return stdout
    monkeypatch.setattr("swebench_pro.grade.shell", shell)
    monkeypatch.setattr("swebench_pro.grade.checked", checked)
    spec = SimpleNamespace(
        parser_path=str(parser), grader_revision="sha256:" + hashlib.sha256(parser.read_bytes()).hexdigest(),
        snapshot_id="actor-final-snapshot", resources={"cpu": 2, "memory_mb": 4096}, timeout_s=60,
    )
    row = {
        "repo": "owner/repo", "base_commit": base, "test_patch": patch,
        "install_config": {"test_cmd": "python3.12 -m pytest -vv -rA test_answer.py", "log_parser": "fixture"},
        "FAIL_TO_PASS": ["test_answer.py::test_answer"], "PASS_TO_PASS": [],
    }
    result = grade_snapshot(spec, row, {}, tmp_path, session_factory=lambda **_: Session())
    assert result["resolved"] is expected, result
    assert lifecycle == [("create", "actor-final-snapshot"), ("destroy",)]
    assert (repo / "answer.py").read_text() == ("VALUE = 1\n" if fix else "VALUE = 0\n")


def test_export_native_prefix_and_marked_branch_hint(tmp_path):
    import json

    home = tmp_path / "native-home"
    home.mkdir()
    history = [
        {"type": "response_item", "payload": {"type": "message", "role": role,
         "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}]}}
        for role, text in [
            ("user", "original task"), ("assistant", "parent attempt"),
            ("user", mark_hint("private guidance")), ("assistant", "branch patch"),
        ]
    ]
    (home / "session-id.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    messages = export_messages(tmp_path, "session-id", "codex", [])
    assert [m["content"] for m in messages] == ["original task", "parent attempt", "branch patch"]


def test_plaintext_responses_reasoning_is_exported():
    from runstore.message_export import codex_messages

    entries = [
        {"type": "response_item", "payload": {
            "type": "reasoning", "content": [{"type": "reasoning_text", "text": "Inspect the parser."}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}],
        }},
    ]
    messages = codex_messages(entries)
    assert len(messages) == 1
    assert messages[0]["reasoning_content"] == "Inspect the parser."
    assert messages[0]["content"] == "Done."


def test_native_fragments_merge_within_a_response_but_not_across_boundaries():
    from runstore.message_export import codex_messages

    def item(payload):
        return {"type": "response_item", "payload": payload}
    call = {"type": "function_call", "name": "shell", "arguments": '{"command":"pwd"}'}
    entries = [
        item({"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Inspect."}]}),
        item({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "\n\n"}]}),
        item({**call, "call_id": "one"}),
        item({**call, "call_id": "two"}),
        item({"type": "function_call_output", "call_id": "one", "output": "/repo"}),
        item({"type": "function_call_output", "call_id": "two", "output": "/repo"}),
        {"type": "event_msg", "payload": {"type": "token_count"}},
        item({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "First."}]}),
        {"type": "event_msg", "payload": {"type": "token_count"}},
        item({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Second."}]}),
    ]
    messages = codex_messages(entries)
    assert len(messages) == 5
    assert messages[0]["reasoning_content"] == "Inspect."
    assert len(messages[0]["tool_calls"]) == 2
    assert [m["content"] for m in messages[-2:]] == ["First.", "Second."]
