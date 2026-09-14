"""Explicit no-model gate: real AgentENV tools, snapshots, restore and Harbor grading."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from harness.core.journal import read_journal
from harness.core.slot import SlotResult
from harness.execution.session import SandboxSession
from harness.slots.claude_code import ClaudeCodeSlot
from terminalbench.eval import main as eval_main


class ScriptedSlot(ClaudeCodeSlot):
    def version(self):
        return "scripted-no-model-gate"

    def run(self, task, journal, mcp=None):
        self._journal = journal
        self._checkpoint_server = "ash"
        self._control = task.control
        journal.emit("run.started", slot="scripted-no-model", task_prompt=task.prompt)
        journal.emit("session.ref", native_session_id="scripted-no-model")
        commands = [
            "mkdir -p /app; printf 'first\\n' > /app/answer.txt; "
            "nohup sleep 600 </dev/null >/tmp/gate-sleep.log 2>&1 & echo $! > /app/sleep.pid",
            "printf 'ready\\n' > /app/answer.txt",
        ]
        with httpx.Client(headers=mcp.headers, timeout=180) as client:
            tools = client.post(mcp.url, json={"jsonrpc": "2.0", "id": "tools", "method": "tools/list"}).json()
            assert [tool["name"] for tool in tools["result"]["tools"]] == ["shell"]
            for index, command in enumerate(commands):
                call_id = f"gate-{index}"
                verdict = asyncio.run(self._pre_tool_use(
                    {"tool_name": "mcp__ash__shell", "tool_input": {"command": command}}, call_id))
                arguments = verdict["hookSpecificOutput"]["updatedInput"]
                reply = client.post(mcp.url, json={"jsonrpc": "2.0", "id": call_id,
                    "method": "tools/call", "params": {"name": "shell", "arguments": arguments}}).json()
                result = reply["result"]
                assert not result.get("isError"), result
                journal.emit("tool.finished", call_id=call_id, status="ok", output=json.dumps(result["content"]))
        journal.emit("run.finished", status="completed", usage={})
        return SlotResult(status="completed", final_text="scripted gate complete", native_session_id="scripted-no-model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-bin", required=True)
    parser.add_argument("--tasks-dir", default=str(Path(__file__).parent / "agentenv_fixtures"))
    parser.add_argument("--image-registry")
    args = parser.parse_args()
    import harness.slots

    original = harness.slots.load_slot
    harness.slots.load_slot = lambda name: ScriptedSlot
    try:
        command = ["--phase", "run", "--model", "scripted-no-model", "--workers", "1",
                   "--output", str(args.output), "--runtime-bin", args.runtime_bin,
                   "--tasks-dir", args.tasks_dir, "--env", "agentenv", "--checkpoint-mode", "full"]
        if args.image_registry:
            command += ["--image-registry", args.image_registry]
        code = eval_main(command)
    finally:
        harness.slots.load_slot = original
    if code:
        return code
    summary = json.loads((args.output / "ash-summary.json").read_text())
    assert summary["resolved_trials"] == summary["expected_trials"] == 1
    journal = next(args.output.glob("*/agent/trajectory.jsonl"))
    points = [row for row in read_journal(journal)
              if row["type"] == "checkpoint.captured" and row.get("snapshot_id") and row.get("step") == 1]
    assert points
    snapshot_id = points[-1]["snapshot_id"]
    session = SandboxSession(quiet=True, backend={"backend": "microvm", "microvm": {
        "server_url": "http://127.0.0.1:18000", "runtime_bin": args.runtime_bin,
        "image_env": True, "sandbox_ttl": 600}})
    try:
        assert session.create(snapshot_id), session.create_error
        result = session.execute("shell", {"command": "cat /app/answer.txt; kill -0 $(cat /app/sleep.pid)", "timeout": 10})
        assert result.success and "first" in result.output, result
        evidence = {"snapshot_id": snapshot_id, "restored_first_step": True,
                    "background_process_alive": True, "output": result.output,
                    "resolved_trials": summary["resolved_trials"]}
        (args.output / "checkpoint-restore.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2))
    finally:
        session.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
