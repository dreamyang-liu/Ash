from copy import deepcopy
import json
import re
import os
import subprocess
import sys

import pytest

from deepswe.bench import DeepSWE
from harness.core.assistant_turn import validate_assistant_turn
from harness.core.journal import read_journal
from harness.core.mini_tools import mini_tool_schema
from harness.orchestrator.run import Orchestrator
from harness.tests.test_assistant_turn import assistant_turn
from harness.tests.test_mini_swe import FilesystemSession, model_server, owned_filesystem, pytestmark, reply, spec
from runstore.tests.test_mini_native import parent_run
from swebench import fork_eval
from swebench.assistant_branch import actor_tools_at, reviewer_context


def fenced_json(text):
    return [json.loads(body) for body in re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)]


def test_actor_primer_deepswe_prompt_and_actual_request_share_one_tool_contract(tmp_path, monkeypatch):
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    original = deepcopy(BASH_TOOL)
    schema = mini_tool_schema(BASH_TOOL)
    assert BASH_TOOL == original  # never mutate upstream globals
    assert schema[0]["function"]["parameters"]["additionalProperties"] is False
    task = {"repo": "repo", "problem": "Fix the public task.", "slot": "mini-swe-agent", "agent_network": "deny"}
    deep_prompt = DeepSWE(tmp_path).prompt(task)
    schema_block, example = fenced_json(deep_prompt)
    assert schema_block == schema and set(example) == {"command"}
    assert "cd /app" in example["command"]
    assert "`text_editor`" not in deep_prompt and "working_dir" not in deep_prompt
    assert "timeout (seconds)" not in deep_prompt
    assert "Only COMMITTED work counts" in deep_prompt

    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("pwd"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        prompt = fork_eval.SweBench().prompt(task)
        outcome = Orchestrator().run(spec(tmp_path, url, prompt=prompt))
    assert outcome.status == "completed", outcome.error
    first = requests[0]
    assert first["tools"] == schema
    task_prompt = next(m["content"] for m in first["messages"] if m["role"] == "user")
    assert fenced_json(task_prompt)[0] == first["tools"]
    records = [e["tools"] for e in read_journal(outcome.journal_path) if e["type"] == "rollout.model_tools"]
    assert records and all(tools == schema for tools in records)
    # This is the shape actually returned by the failed first Luna parent.
    invalid = assistant_turn()
    invalid["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "command": "git status --short --branch", "working_dir": "/app", "timeout": 120})
    with pytest.raises(ValueError, match="exactly one"):
        validate_assistant_turn(invalid, tools=first["tools"])


def test_reviewer_gets_recorded_tools_and_workspace_at_its_cut(tmp_path, monkeypatch):
    _, parent, points = parent_run(tmp_path, monkeypatch)
    checkpoint = fork_eval.available_branch_points(parent.journal_path)[2]
    context = reviewer_context(parent.journal_path, {2: checkpoint})
    assert context["tools"] == mini_tool_schema()
    assert context["workspace"]["cwd"] == "/testbed"
    rows = read_journal(parent.journal_path)
    # Later tool changes cannot change an earlier checkpoint's contract.
    rows.append({"type": "rollout.model_tools", "shape": "chat/completions", "tools": []})
    parent.journal_path.write_text("".join(json.dumps(e) + "\n" for e in rows))
    assert actor_tools_at(parent.journal_path, 2) == context["tools"]
    # Do not guess from local defaults if the actual declaration is absent.
    rows = [e for e in rows if e["type"] != "rollout.model_tools"]
    parent.journal_path.write_text("".join(json.dumps(e) + "\n" for e in rows))
    with pytest.raises(ValueError, match="No recorded actor tool schema"):
        reviewer_context(parent.journal_path, {2: checkpoint})


def test_generated_call_is_checked_against_recorded_parameter_constraints():
    tools = mini_tool_schema()
    tools[0]["function"]["parameters"]["properties"]["command"]["pattern"] = "^git "
    with pytest.raises(ValueError, match="recorded actor tool schema"):
        validate_assistant_turn(assistant_turn("printf wrong"), tools=tools)
    assert validate_assistant_turn(assistant_turn("git status"), tools=tools)
    tools[0]["function"]["name"] = "different_tool"
    with pytest.raises(ValueError, match="must be the bash"):
        validate_assistant_turn(assistant_turn(), tools=tools)


def test_rendering_actor_primer_does_not_import_mini_or_load_host_config(tmp_path):
    config = tmp_path / "not-created"
    code = (
        "import sys,os\n"
        "from swebench.fork_eval import tool_primer\n"
        "before=dict(os.environ)\n"
        "assert 'bash' in tool_primer('mini-swe-agent','/app')\n"
        "assert 'minisweagent' not in sys.modules\n"
        "assert dict(os.environ)==before\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(config)})
    assert not config.exists()
