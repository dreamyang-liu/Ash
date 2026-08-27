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


def test_events_emitted_from_inside_a_subscriber_reach_every_subscriber(tmp_path):
    """Regression: nested emissions used to be suppressed entirely.

    A subscriber that emits is normal -- the checkpoint bridge turns a
    turn.completed into a checkpoint.captured. Suppressing that nested event made
    it invisible to *other* subscribers, so the resource ledger never recorded the
    snapshots the bridge had just claimed and a killed process left them
    unreclaimable. Recursing instead would let a subscriber re-enter the fan-out
    it is already in, so nested events are queued and drained after it.
    """
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r")
    seen_by_observer = []

    def emitter(record):
        if record["type"] == "trigger":
            journal.emit("derived", note="from inside a subscriber")

    journal.subscribe(emitter)
    journal.subscribe(lambda r: seen_by_observer.append(r["type"]))

    journal.emit("trigger")
    journal.close()

    assert seen_by_observer == ["trigger", "derived"], seen_by_observer
    assert [r["type"] for r in read_journal(tmp_path / "j.jsonl")] == ["trigger", "derived"]


def test_a_subscriber_that_emits_its_own_trigger_does_not_recurse_forever(tmp_path):
    """The queue is drained iteratively, so depth stays flat -- but an emitter with
    no stop condition is still the caller's bug, bounded here only by the guard
    that it is not re-entered."""
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r")
    count = {"n": 0}

    def echo(record):
        count["n"] += 1
        if count["n"] < 5:                 # a stop condition, as any emitter needs
            journal.emit("echo", depth=count["n"])

    journal.subscribe(echo)
    journal.emit("start")
    journal.close()

    assert count["n"] == 5
    assert len(read_journal(tmp_path / "j.jsonl")) == 5
