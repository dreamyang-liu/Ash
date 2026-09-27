"""Real upstream mini loop over real HTTP model/MCP boundaries.

The model replies are controlled and the VM filesystem is a temporary directory.
Shell commands really execute; snapshot contents are copied at actual boundaries.
"""

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ash_sandbox.result import ToolResult
from harness.core.journal import read_journal
from harness.core.result import ToolResult as CoreToolResult
from harness.execution.interceptors import MutationTracker
from harness.execution.pipeline import ToolPipeline
from harness.execution.server import HttpMcpServer
from harness.execution.wiring import http_wiring
from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec
from harness.rollback import branch_checkpoints, turn_branch_checkpoints
from harness.slots.mini_history import read_entries, training_messages
from runstore.message_export import clean_messages

pytestmark = pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("minisweagent") is None,
    reason="install harness/requirements-mini-swe-agent.txt")


class FilesystemSession:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = {}
        self.destroyed = False

    def supports_snapshot(self):
        return True

    def execute(self, name, args, timeout=30):
        assert name == "shell"
        directory = args.get("working_dir", "/testbed")
        if directory != "/testbed":
            return CoreToolResult(False, "", f"failed to start shell: chdir {directory}: no such file or directory")
        result = subprocess.run(
            ["sh", "-c", args["command"]], cwd=self.root,
            text=True, capture_output=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).replace(str(self.root), "/testbed")
        return CoreToolResult(result.returncode == 0, output)

    def snapshot(self, **kwargs):
        sid = "snapshot-" + uuid4().hex
        self.snapshots[sid] = {str(p.relative_to(self.root)): p.read_bytes()
                               for p in self.root.rglob("*") if p.is_file()}
        return SimpleNamespace(id=sid, rootfs_layers=None, memory_layers=None, chain_size_mb=None)

    async def call(self, name, **args):
        assert name == "shell"
        assert args["working_dir"] == "/testbed"
        result = await asyncio.to_thread(
            subprocess.run, ["bash", "-c", args["command"]], cwd=self.root,
            text=True, capture_output=True, timeout=args["timeout"])
        if result.returncode == 0 and not result.stderr:
            # Real runtime CommandOutcome.Result uses bare stdout for this
            # case; always returning structured JSON hid an adapter defect.
            return ToolResult(result.stdout, False)
        return ToolResult.from_response({"content": [{"type": "text", "text": json.dumps({
            "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
            "running": False, "timed_out": False,
        })}]})

    def destroy(self):
        self.destroyed = True


def owned_filesystem(session):
    entry = SimpleNamespace(id="fixture", sandbox=session, visible_to=lambda _: True)
    async def destroy_all():
        pass
    pool = SimpleNamespace(get=lambda identifier: entry if identifier == "fixture" else None,
                           destroy_all=destroy_all)
    tracker = MutationTracker()
    server = HttpMcpServer(pool, host="127.0.0.1", port=0, pipeline=ToolPipeline([tracker])).start()
    owned = OwnedSandbox(session=session, server=server, tracker=tracker, sandbox_id="fixture")
    return owned, http_wiring(server.base_url, agent_id=uuid4().hex, sandbox_id="fixture")


@contextmanager
def model_server(replies):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    iterator = iter(replies)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert self.path == "/v1/chat/completions"
            reply = next(iterator)
            reply = reply(body) if callable(reply) else reply
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def reply(*commands, content="Thinking.", reasoning="Inspect the actual file.", ids=None):
    return {"id": "chat-" + uuid4().hex, "object": "chat.completion", "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": reasoning,
                                     "tool_calls": [
                                         {"id": ids[i] if ids else uuid4().hex, "type": "function",
                                          "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
                                         for i, command in enumerate(commands)]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 3},
                      "completion_tokens_details": {"reasoning_tokens": 5}}}


def spec(tmp_path, url, **changes):
    values = {
        "prompt": "Write the answer.", "slot": "mini-swe-agent", "model": "fixture",
        "transport": "http", "tools": "shell_only", "timeout_s": 30,
        "journal_path": tmp_path / "trajectory.jsonl",
        "extra": {"native_home": str(tmp_path / "native-home"), "rollout_contract": {
            "message_export": True, "model_endpoint": url, "model": "fixture",
            "deadline_at": time.time() + 30, "max_turns": 5,
            "sampling_params": {"temperature": .8, "top_k": 20, "max_new_tokens": 300},
        }},
    }
    values.update(changes)
    return RunSpec(**values)


def test_upstream_loop_executes_then_captures_and_records_submission(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("printf first > answer", "cat answer"),
                       reply("printf second > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert memory.destroyed and (memory.root / "answer").read_text() == "second"
    all_points = branch_checkpoints(outcome.journal_path)
    points = turn_branch_checkpoints(outcome.journal_path)
    assert set(all_points) == {1, 2, 3, 4} and set(points) == {2, 3, 4}
    assert memory.snapshots[points[2].snapshot_id]["answer"] == b"first"
    assert memory.snapshots[points[3].snapshot_id]["answer"] == b"second"
    entries = read_entries((tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl").read_bytes())
    messages = clean_messages(training_messages(entries))
    assert len([m for m in messages if m["role"] == "tool"]) == 4
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in messages[-1]["content"]
    assert all(m["reasoning_content"] == "Inspect the actual file."
               for m in messages if m["role"] == "assistant")
    assert all(request["tools"][0]["function"]["name"] == "bash" for request in requests)
    system = requests[0]["messages"][0]
    assert "/testbed" in system["content"]
    assert all(request["messages"][0] == system for request in requests)
    workspace = next(entry["workspace"] for entry in entries if entry["type"] == "mini.session")
    assert workspace == {"cwd": "/testbed", "repository_dir": None}
    assert all(request["top_k"] == 20 and request["max_tokens"] == 300 for request in requests)
    assert outcome.usage["input_tokens"] == 33 and outcome.usage["reasoning_output_tokens"] == 15


def test_real_exit_code_is_observed_before_next_query(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("printf diagnostic; exit 42"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    observed = json.loads(requests[1]["messages"][-1]["content"])
    assert observed["returncode"] == 42 and "diagnostic" in observed["output"]


def test_model_turn_limit_keeps_closed_history_without_an_extra_request(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo fix > answer")]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["rollout_contract"]["max_turns"] = 1
        outcome = Orchestrator(out_dir=tmp_path).run(run_spec)
    assert outcome.stop_reason == "max_turns_reached" and len(requests) == 1
    entries = read_entries((tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl").read_bytes())
    messages = clean_messages(training_messages(entries))
    assert messages[-1]["role"] == "tool"
    assert len(turn_branch_checkpoints(outcome.journal_path)) == 1


def test_invalid_configured_directory_fails_before_model_or_agent_tool(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["mini"] = {"environment": {"cwd": "/missing-workspace"}}
        outcome = Orchestrator().run(run_spec)
    assert "chdir /missing-workspace" in outcome.error
    assert requests == []
    assert not any(event["type"] == "tool.started" for event in read_journal(outcome.journal_path))
    assert memory.destroyed
