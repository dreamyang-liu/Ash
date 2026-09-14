"""Complete model-response/tool-result boundaries, independent of snapshots.

The SDK may split ONE response into several AssistantMessages with the same
message_id, even interleaving tool results. Neither an AssistantMessage nor an
empty pending set closes a response. Without changing SDK streaming behavior,
the next distinct response (or successful query result) proves output closure.
This deliberately publishes the preceding boundary one response late.
"""
from __future__ import annotations

TURN_BRANCH_POLICY = "completed-model-turn-v1"


class ModelTurnTracker:
    def __init__(self, journal):
        self.journal = journal
        self.current = None
        self.turns = {}
        self.results = set()
        journal.emit("branch.boundary.policy", policy=TURN_BRANCH_POLICY,
                     storage="per-tool", closure="next-response-or-query-result")

    def observe(self, message):
        kind = type(message).__name__
        if kind == "AssistantMessage":
            turn_id = getattr(message, "message_id", None)
            if not isinstance(turn_id, str) or not turn_id:
                # No guessed grouping by SDK chunk, timestamp or pending count.
                self.journal.emit("model.turn.unavailable", reason="missing_message_id")
                return
            if self.current and self.current != turn_id:
                self._close(self.current, "next_response")
            self.current = turn_id
            turn = self.turns.setdefault(turn_id, {"calls": [], "closed": False,
                                                   "published": False, "invalid": False})
            if turn["closed"]:
                turn["invalid"] = True
                self.journal.emit("model.turn.invalid", turn_id=turn_id,
                                  reason="response_reopened")
                return
            for block in getattr(message, "content", None) or []:
                if type(block).__name__ == "ToolUseBlock":
                    call_id = getattr(block, "id", None)
                    if call_id and call_id not in turn["calls"]:
                        turn["calls"].append(call_id)
                        self.journal.emit("model.turn.tool", turn_id=turn_id, call_id=call_id)
        elif kind == "UserMessage":
            for block in getattr(message, "content", None) or []:
                if type(block).__name__ == "ToolResultBlock":
                    self.results.add(getattr(block, "tool_use_id", None))
        elif kind == "ResultMessage" and not getattr(message, "is_error", False):
            if self.current:
                self._close(self.current, "query_result")
        self._publish_ready()

    def _close(self, turn_id, evidence):
        turn = self.turns[turn_id]
        if not turn["closed"]:
            turn["closed"] = True
            self.journal.emit("model.turn.output_completed", turn_id=turn_id,
                              call_ids=list(turn["calls"]), closure=evidence)

    def _publish_ready(self):
        positions = {r["call_id"]: r["step"] for r in self.journal.tool_calls()}
        for turn_id, turn in self.turns.items():
            calls = turn["calls"]
            if (turn["closed"] and not turn["invalid"] and not turn["published"]
                    and calls and set(calls) <= self.results and set(calls) <= positions.keys()):
                self.journal.emit("model.turn.completed", turn_id=turn_id,
                                  call_ids=list(calls), step=max(positions[c] for c in calls))
                turn["published"] = True


def completed_turn_steps(events):
    """None for legacy journals; validated step -> turn metadata for new ones.

This is conversation grouping, NOT proof of snapshot availability or native
resume. Consumers must intersect it with the exact snapshot ledger and cut.
"""
    if not any(e.get("type") == "branch.boundary.policy" for e in events):
        return None
    if any(e.get("type") == "branch.boundary.policy" and
           e.get("policy") != TURN_BRANCH_POLICY for e in events):
        return {}
    calls = [e["call_id"] for e in events
             if e.get("type") == "tool.started" and e.get("call_id")]
    positions = {cid: i for i, cid in enumerate(calls, 1)}
    invalid = {e.get("turn_id") for e in events if e.get("type") == "model.turn.invalid"}
    closed, finished, covered, boundaries = {}, set(), set(), {}
    for event in events:
        kind = event.get("type")
        tid = event.get("turn_id")
        if kind == "tool.finished":
            finished.add(event.get("call_id"))
        elif kind == "model.turn.output_completed":
            closed[tid] = event.get("call_ids", [])
        elif kind == "model.turn.completed" and tid not in invalid:
            group = event.get("call_ids", [])
            step = event.get("step")
            if (not group or closed.get(tid) != group or not set(group) <= finished
                    or not set(group) <= positions.keys()
                    or type(step) is not int or step != max(positions[c] for c in group)):
                continue
            covered.update(group)
            if set(calls[:step]) <= covered:
                boundaries[step] = event
    return boundaries
