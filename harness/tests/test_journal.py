"""Journal invariants: monotonic seq, append-only, crash-survivable, thread-safe."""

from __future__ import annotations

import json
import threading

from harness.core.events import AGENT_MESSAGE, JOURNAL_SCHEMA_VERSION, Usage
from harness.core.journal import JournalWriter, read_journal


def test_envelope_and_monotonic_seq(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, run_id="r1", agent_id="a1", sandbox_id="s1") as journal:
        journal.emit(AGENT_MESSAGE, text="one")
        journal.emit(AGENT_MESSAGE, text="two")

    records = read_journal(path)
    assert [r["seq"] for r in records] == [1, 2]
    assert all(r["v"] == JOURNAL_SCHEMA_VERSION for r in records)
    first = records[0]
    assert first["run_id"] == "r1" and first["agent_id"] == "a1" and first["sandbox_id"] == "s1"
    assert first["ts"].endswith("Z")
    assert first["text"] == "one"


def test_flushed_per_line_so_killed_runs_keep_trajectory(tmp_path):
    """No context manager, no close: the file must already be readable."""
    path = tmp_path / "killed.jsonl"
    journal = JournalWriter(path, run_id="r2")
    journal.emit(AGENT_MESSAGE, text="before the kill")
    assert len(read_journal(path)) == 1


def test_appends_across_writers(tmp_path):
    path = tmp_path / "resumed.jsonl"
    with JournalWriter(path, run_id="r3") as journal:
        journal.emit(AGENT_MESSAGE, text="first session")
    with JournalWriter(path, run_id="r3") as journal:
        journal.emit(AGENT_MESSAGE, text="second session")
    assert [r["text"] for r in read_journal(path)] == ["first session", "second session"]


def test_concurrent_emitters_produce_unique_seqs(tmp_path):
    """CLI slots emit from a reader thread plus a stderr drain thread."""
    path = tmp_path / "threads.jsonl"
    journal = JournalWriter(path, run_id="r4")

    def emit_many(tag):
        for i in range(50):
            journal.emit(AGENT_MESSAGE, text="%s-%d" % (tag, i))

    threads = [threading.Thread(target=emit_many, args=(t,)) for t in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    journal.close()

    records = read_journal(path)
    seqs = [r["seq"] for r in records]
    assert len(records) == 150
    assert sorted(seqs) == list(range(1, 151))  # no duplicates, no gaps


def test_emit_returns_record_with_seq(tmp_path):
    with JournalWriter(tmp_path / "x.jsonl") as journal:
        record = journal.emit(AGENT_MESSAGE, text="hi")
    assert record["seq"] == 1 and record["text"] == "hi"


def test_non_json_values_do_not_break_the_line(tmp_path):
    class Weird:
        def __repr__(self):
            return "<weird>"

    path = tmp_path / "weird.jsonl"
    with JournalWriter(path) as journal:
        journal.emit(AGENT_MESSAGE, text="x", obj=Weird())
    line = path.read_text().strip()
    assert json.loads(line)["obj"] == "<weird>"


def test_usage_accumulates_dimensions_separately():
    total = Usage()
    total.add_dict({"input_tokens": 10, "cached_input_tokens": 5, "cost_usd": 0.5})
    total.add_dict({"input_tokens": 3, "reasoning_output_tokens": 7, "cost_usd": 0.25})
    assert total.input_tokens == 13
    assert total.cached_input_tokens == 5
    assert total.reasoning_output_tokens == 7
    assert total.cost_usd == 0.75
    # partial payloads must not raise or zero out other fields
    total.add_dict({})
    assert total.input_tokens == 13
