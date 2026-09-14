from types import SimpleNamespace

from harness.core.journal import JournalWriter, read_journal
from harness.normalize.claude_code import normalize
from harness.normalize.claude_turns import ModelTurnTracker, completed_turn_steps


def typed(class_name, **kwargs):
    return type(class_name, (SimpleNamespace,), {})(**kwargs)


def assistant(turn, *calls):
    return typed("AssistantMessage", message_id=turn, content=[
        typed("ToolUseBlock", id=cid, name="shell", input={}) for cid in calls])


def result(*calls, error=False):
    return typed("UserMessage", content=[typed("ToolResultBlock", tool_use_id=cid,
                                               content="ok", is_error=error) for cid in calls])


def feed(journal, tracker, message):
    for kind, payload in normalize(message):
        journal.emit(kind, **payload)
    tracker.observe(message)


def test_same_response_split_by_result_is_not_prematurely_complete(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path) as journal:
        tracker = ModelTurnTracker(journal)
        feed(journal, tracker, assistant("turn1", "a"))
        feed(journal, tracker, result("a"))
        assert completed_turn_steps(read_journal(path)) == {}
        feed(journal, tracker, assistant("turn1", "b"))
        feed(journal, tracker, result("b"))
        assert completed_turn_steps(read_journal(path)) == {}
        feed(journal, tracker, assistant("turn2"))
        points = completed_turn_steps(read_journal(path))
        assert list(points) == [2]
        assert points[2]["call_ids"] == ["a", "b"]


def test_closed_output_waits_for_all_results_and_allows_ordinary_failure(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path) as journal:
        tracker = ModelTurnTracker(journal)
        feed(journal, tracker, assistant("t1", "a", "b"))
        feed(journal, tracker, result("b"))
        feed(journal, tracker, assistant("t2"))
        assert completed_turn_steps(read_journal(path)) == {}
        feed(journal, tracker, result("a", error=True))
        assert list(completed_turn_steps(read_journal(path))) == [2]


def test_query_result_closes_single_tool_turn_without_changing_step(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path) as journal:
        tracker = ModelTurnTracker(journal)
        feed(journal, tracker, assistant("t1", "a"))
        feed(journal, tracker, result("a"))
        feed(journal, tracker, typed("ResultMessage", is_error=False))
        assert list(completed_turn_steps(read_journal(path))) == [1]
        assert len(journal.tool_calls()) == 1


def test_missing_identity_and_interrupted_output_fail_closed(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path) as journal:
        tracker = ModelTurnTracker(journal)
        feed(journal, tracker, assistant(None, "a"))
        feed(journal, tracker, result("a"))
        feed(journal, tracker, assistant("t2", "b"))
        feed(journal, tracker, result("b"))
        feed(journal, tracker, typed("ResultMessage", is_error=True))
        assert completed_turn_steps(read_journal(path)) == {}


def test_reopened_response_invalidates_its_earlier_boundary(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path) as journal:
        tracker = ModelTurnTracker(journal)
        for msg in (assistant("t1", "a"), result("a"), assistant("t2")):
            feed(journal, tracker, msg)
        assert list(completed_turn_steps(read_journal(path))) == [1]
        feed(journal, tracker, assistant("t1", "b"))
        assert completed_turn_steps(read_journal(path)) == {}
