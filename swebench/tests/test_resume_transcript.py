"""Resume with memory: the environment from the snapshot, the conversation
from the transcript, cut at the same step."""

import json

from swebench.agent import AshAgent
from swebench.agent.conversation import Conversation
from swebench.models import AgentConfig, CostTracker, Trajectory

HISTORY = [
    {"role": "system", "content": "original system prompt, possibly from an "
                                  "older prompt builder"},
    {"role": "user", "content": "Build the decoder."},
    {"role": "assistant", "content": "Starting.", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "shell", "arguments": '{"command": "make"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "built"},
]


def test_seed_is_verbatim():
    """No regenerated system prompt, no appended note, tool_calls and
    tool_call_id untouched: the model must see exactly the conversation the
    original run held, or the resumed run is a different experiment."""
    conv = Conversation(Trajectory())
    conv.seed(HISTORY)
    assert conv.messages == HISTORY
    assert conv.trajectory.messages == HISTORY, (
        "the saved trajectory carries the seeded history too, so a second "
        "resume can cut a prefix from this run's own file")


def test_run_with_history_sends_it_untouched(monkeypatch):
    """The first model call of a resumed run receives the seeded transcript
    and nothing else -- task and prompt builders stay out of it."""
    seen = {}

    class Msg:
        content = "done"
        tool_calls = None
        thinking_blocks = None

    class Resp:
        choices = [type("C", (), {"message": Msg(), "finish_reason": "stop"})()]
        usage = None

    def fake_completion(**kwargs):
        seen.setdefault("messages", [dict(m) for m in kwargs["messages"]])
        return Resp()

    import swebench.agent.llm as llm
    monkeypatch.setattr(llm, "_get_litellm",
                        lambda: (fake_completion, lambda **k: None))
    agent = AshAgent(AgentConfig(model="m", prompt_cache=False),
                     executor=lambda *a, **k: None, agent_id="a")
    agent.stream = False
    agent.run("IGNORED TASK TEXT", instance_id="x", history=HISTORY)

    assert seen["messages"][:len(HISTORY)] == HISTORY
    assert not any("IGNORED TASK TEXT" in str(m) for m in seen["messages"])
    assert agent.turn_base == 1, "one assistant message in the seeded history"


def test_checkpoint_turns_continue_the_transcript_numbering():
    """Turn N must always mean 'the transcript's first N assistant messages',
    whichever run wrote them. Without the offset, a resumed run's
    step->snapshot map would disagree with the very transcript it saves, and
    a second resume would replay the wrong prefix."""
    from swebench.agent.checkpoints import install

    class FakeSnapshot:
        id = "snap-1"
        rootfs_layers = 3
        memory_layers = 0
        chain_size_mb = 1

    class FakeSession:
        def supports_snapshot(self): return True
        def snapshot(self, name=None, disk_only=True): return FakeSnapshot()
        def swap_sandbox(self, s): return True

    class FakeAgent:
        pipeline = None
        before_query_hooks = []
        cost = CostTracker()
        trajectory = Trajectory()
        turn_base = 102          # resumed after a 102-step segment

    agent = FakeAgent()
    checkpointer = install(agent, FakeSession(), always=True)
    agent.cost.api_calls = 1     # first model call of the resumed segment
    agent.before_query_hooks[-1](agent, None)
    assert checkpointer.records[-1].turn == 103


def test_harness_derives_snapshot_and_prefix_from_one_file(tmp_path):
    """Alignment by construction: both come from the same trajectory at the
    same step, and an explicitly given snapshot that is not in the map is
    refused rather than paired with a mismatched history."""
    import inspect
    from swebench.harnesses import marathon

    source = inspect.getsource(marathon)
    assert "resume_transcript" in source
    assert "messages_through_step(resume_transcript, step)" in source
    assert "refusing a mismatched" in source
    # The no-transcript note stays out of the with-memory path.
    assert 'config.get("resume_from") and not history' in source


def test_cli_exposes_resume_with_transcript():
    import inspect
    from swebench import __main__ as cli
    source = inspect.getsource(cli)
    assert '"--resume-with-transcript"' in source
    assert '"resume_transcript"' in source


def test_trajectory_rows_are_translated_back_to_wire_format(tmp_path):
    """The trajectory is a record, not a transcript: tool results are stored
    as tool_result rows with evaluation metadata, and error rows never went in
    front of the model at all. Seeding rows verbatim died on the first model
    call -- "Invalid Message ... role 'tool_result'" -- with the environment
    already restored, so the run graded 0 having done nothing."""
    from swebench.replay import messages_through_step

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "shell", "arguments": "{}"}}]},
        {"role": "tool_result", "tool_call_id": "c1", "content": "out",
         "tool_name": "shell", "tool_args": {"command": "ls"}, "success": True},
        {"role": "error", "content": "litellm.SomethingTransient: ..."},
        {"role": "assistant", "content": "done"},
    ]}))
    prefix = messages_through_step(path, 2)
    assert prefix == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "shell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
        {"role": "assistant", "content": "done"},
    ], ("tool_result becomes role=tool with only wire keys; the error row "
        "is omitted entirely")


def test_prefix_cut_matches_the_map_on_a_resumed_trajectory(tmp_path):
    """End to end on files: a seg-2 trajectory carries seeded history plus new
    turns, its map numbers from the seed, and cutting at a map key returns
    the transcript through exactly that assistant message."""
    from swebench.replay import load_step_snapshots, messages_through_step

    messages = list(HISTORY) + [
        {"role": "assistant", "content": "resumed work", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "shell", "arguments": "{}"}}]},
        {"role": "tool_result", "tool_call_id": "c2", "content": "ran"},
        {"role": "assistant", "content": "more"},
    ]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "messages": messages,
        "info": {"checkpoints": {"step_snapshots": {"1": "snap-a", "2": "snap-b"}}},
    }))
    step_map = load_step_snapshots(path)
    assert step_map[2] == "snap-b"
    prefix = messages_through_step(path, 2)
    assert len(prefix) == 6, "through assistant #2 and its tool result"
    assert prefix[-1]["role"] == "tool", "translated on the way out"
    assert prefix[-1]["tool_call_id"] == "c2"
